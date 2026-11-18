"""KVConnector extension: engine-facing transfer endpoint, no vLLM import.

Mirrors the *shape* of vLLM's ``KVConnectorBase_V1`` hooks (lookup /
save / get / load) as a standalone extension point, so an engine
adapter (vLLM, SGLang, or custom) only binds methods — this module
never imports vLLM.

Layering: the connector is thin orchestration over the rest of the
package — :class:`kv_compression.BlockStore` identity, ``wire`` frames,
and the :func:`kv_compression.transport.choose_plan` operating point.

Transport is the :class:`KVLink` port — reliable, ordered message
delivery (TCP / QUIC stream / RDMA-with-ordering shaped).
:class:`MemoryLink` is the in-process simulation used by tests and
offline planning; swap it for a socket implementation without touching
the connector.

Message protocol (one pump-driven exchange)::

    sender                          receiver
      save(request) xN ; flush()
      MANIFEST ---------------------->  accept_manifest
      HAVE(keys)   <------------------  have_keys()        # reconcile
      FRAME*       ------------------>  accept_frame       # missing only
      READY[idx]   <------------------  ready_requests()

Dropped mid-transfer: re-``save`` the same requests and ``flush``
again — the sender store dedups and the receiver's HAVE reply makes
the retry incremental.
"""

from __future__ import annotations

import json
import select
import socket
from collections import deque
from typing import Optional, Protocol

import numpy as np

from kv_compression.codec import BEST_METHOD
from kv_compression.dedup import (
    DEFAULT_BLOCK_TOKENS,
    BlockStore,
    Request,
    _ingest_batch,
    block_keys,
    request_manifest,
)
from kv_compression.transport import CANDIDATES, PROFILES, choose_plan
from kv_compression.wire import WireReceiver, stream_frames

_MSG_MANIFEST, _MSG_HAVE, _MSG_FRAME, _MSG_READY = 1, 2, 3, 4
_EXACT_CANDIDATES = tuple(c for c in CANDIDATES if c["exact"])
_PLAN_CODEC = {"raw": ("raw", 1), "lossless/l3": (BEST_METHOD, 3),
               "lossless/l19": (BEST_METHOD, 19)}


def _pack(msg_type: int, payload: bytes) -> bytes:
    return bytes([msg_type]) + len(payload).to_bytes(4, "little") + payload


def _unpack(buf: bytes) -> tuple[int, bytes]:
    if len(buf) < 5:
        raise ValueError("short message")
    msg_type = buf[0]
    n = int.from_bytes(buf[1:5], "little")
    if len(buf) != 5 + n:
        raise ValueError("truncated message")
    return msg_type, buf[5:]


# --------------------------------------------------------------------------
# transport port
# --------------------------------------------------------------------------


class KVLink(Protocol):
    """Reliable ordered byte-message transport (TCP/QUIC-stream shaped)."""

    def send(self, buf: bytes) -> None: ...
    def recv(self) -> Optional[bytes]: ...
    def pending(self) -> bool: ...
    def flush(self) -> None: ...


class MemoryLink:
    """In-process lossless duplex link (tests, offline planning, sim)."""

    def __init__(self) -> None:
        self._inbox: deque[bytes] = deque()
        self._peer: Optional["MemoryLink"] = None

    @classmethod
    def pair(cls) -> tuple["MemoryLink", "MemoryLink"]:
        a, b = cls(), cls()
        a._peer, b._peer = b, a
        return a, b

    def send(self, buf: bytes) -> None:
        if self._peer is None:
            raise RuntimeError("unpaired link")
        self._peer._inbox.append(buf)

    def recv(self) -> Optional[bytes]:
        return self._inbox.popleft() if self._inbox else None

    def pending(self) -> bool:
        return bool(self._inbox)

    def flush(self) -> None:
        pass  # delivery is immediate


class TCPLink:
    """KVLink over a TCP stream socket — both ends pump-driven.

    ``send`` only queues (never blocks); ``pump``'s trailing ``flush``
    pushes bytes out; ``recv`` parses complete length-prefixed messages
    from a non-blocking read buffer.  ``pending`` is true while output
    is unflushed, a message is buffered, or the socket is readable
    (bytes in flight) — so a pump loop never stops mid-exchange.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        sock.setblocking(False)
        try:  # request/response protocol: disable Nagle (else small frames
              # wait on delayed ACKs and pump loops spin empty for ~40-200 ms)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._in = bytearray()
        self._out = bytearray()

    @classmethod
    def pair(cls, host: str = "127.0.0.1", port: int = 0,
             timeout: float = 10.0) -> tuple["TCPLink", "TCPLink"]:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.settimeout(timeout)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.settimeout(timeout)
        c.connect(srv.getsockname())
        conn, _ = srv.accept()
        srv.close()
        c.settimeout(None)
        conn.settimeout(None)
        return cls(c), cls(conn)

    def send(self, buf: bytes) -> None:
        self._out += len(buf).to_bytes(4, "little")
        self._out += buf

    def flush(self) -> None:
        while self._out:
            try:
                n = self._sock.send(self._out[:1 << 18])
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            del self._out[:n]

    def _drain(self) -> None:
        while True:
            try:
                data = self._sock.recv(1 << 20)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return
            if not data:
                return
            self._in += data

    def _complete(self) -> bool:
        return len(self._in) >= 4 and len(self._in) >= 4 + int.from_bytes(self._in[:4], "little")

    def recv(self) -> Optional[bytes]:
        self._drain()
        if not self._complete():
            return None
        n = int.from_bytes(self._in[:4], "little")
        msg = bytes(self._in[4 : 4 + n])
        del self._in[: 4 + n]
        return msg

    def pending(self) -> bool:
        if self._out or self._in:
            return True
        try:
            r, _, _ = select.select([self._sock], [], [], 0)
        except (OSError, ValueError):
            return False
        return bool(r)


def pump_until(*connectors: "KVConnector", max_iters: int = 10_000) -> int:
    """Drive every connector's pump until all are idle; returns iterations."""
    for i in range(max_iters):
        if not any(c.busy() for c in connectors):
            return i
        for c in connectors:
            c.pump()
    raise TimeoutError("connector exchange did not settle")


# --------------------------------------------------------------------------
# connector
# --------------------------------------------------------------------------


class KVConnector:
    """One endpoint of a compressed, deduplicated KV transfer link.

    *Sender (prefill side)*: ``save`` each finished request, ``flush``
    to start the exchange (the codec operating point is chosen then,
    from bandwidth/deadline/profile via :func:`choose_plan`), ``pump``
    until ``completed()`` lists every request.

    *Receiver (decode side)*: ``lookup`` reports external hits before
    the engine commits (recompute-or-wait decision), ``poll`` lists
    whole requests, ``load`` materializes bit-exact tensors.

    Both ends keep their block stores across transfers, so a second
    flush sharing a prefix transmits only the missing blocks.
    """

    def __init__(self, link: KVLink, *, block_tokens: Optional[int] = None,
                 profile: Optional[str] = None, bandwidth_bits_per_s: Optional[float] = None,
                 deadline_s: Optional[float] = None, rtt_s: float = 0.0,
                 frame_order: Optional[str] = None, method: Optional[str] = None,
                 level: Optional[int] = None):
        prof = PROFILES[profile] if profile is not None else {}
        self.link = link
        self.block_tokens = block_tokens if block_tokens is not None else prof.get("block_tokens", DEFAULT_BLOCK_TOKENS)
        self.frame_order = frame_order if frame_order is not None else prof.get("frame_order", "block-major")
        self._pipelined = prof.get("pipelined", True)
        self.bandwidth_bits_per_s = bandwidth_bits_per_s
        self.deadline_s = deadline_s
        self.rtt_s = rtt_s
        self._method = method
        self._level = level
        # sender state
        self._pending: list[Request] = []
        self._store: Optional[BlockStore] = None
        self._needed: list[bytes] = []      # blocks referenced by the in-flight flush
        self._stores: dict[tuple[str, int], BlockStore] = {}  # persistent, per codec point
        self._await: Optional[str] = None          # "have" | "ready"
        self._completed: list[int] = []
        self.wire_bytes = 0
        self.last_plan: Optional[dict] = None
        # receiver state
        self.rx = WireReceiver()
        self._announced: set[int] = set()

    # -- shared -----------------------------------------------------------

    def busy(self) -> bool:
        """True while inbound messages exist or a flush is in flight."""
        return self.link.pending() or self._await is not None

    def pump(self) -> None:
        """Process every inbound message (non-blocking; call from a loop)."""
        self.link.flush()
        while (buf := self.link.recv()) is not None:
            msg_type, payload = _unpack(buf)
            if msg_type == _MSG_MANIFEST:
                self._on_manifest(payload)
            elif msg_type == _MSG_HAVE:
                self._on_have(payload)
            elif msg_type == _MSG_FRAME:
                self._on_frame(payload)
            elif msg_type == _MSG_READY:
                self._on_ready(payload)
            else:
                raise ValueError(f"unknown message type {msg_type}")
        self.link.flush()

    # -- sender -----------------------------------------------------------

    def save(self, request: Request) -> int:
        """Queue a finished request's cache for transfer -> its index."""
        if self._await is not None:
            raise RuntimeError("flush in flight — pump until completed first")
        self._pending.append(request)
        return len(self._pending) - 1

    def flush(self) -> dict:
        """Start transferring queued requests (chooses the codec point).

        Returns a summary; the detailed operating point lands in
        ``last_plan`` once the receiver's HAVE reply is processed.
        """
        if self._await is not None:
            raise RuntimeError("flush in flight — pump until completed first")
        if not self._pending:
            return {"requests": 0, "plan": None}
        n_requests = len(self._pending)
        method, level, plan = self._choose_codec(self._pending)
        # persistent sender store per codec point: repeated flushes re-encode
        # nothing for blocks already compressed earlier (warm transfers)
        store = self._stores.get((method, level))
        if store is None:
            store = BlockStore(self.block_tokens, method, level)
            self._stores[(method, level)] = store
        refs = _ingest_batch(store, self._pending)
        manifest, needed = request_manifest(store, refs, method, level)
        self._store, self._needed = store, needed
        self.last_plan = plan
        self._pending, self._completed = [], []
        self.wire_bytes += len(manifest)
        self._await = "have"
        self.link.send(_pack(_MSG_MANIFEST, manifest))
        return {"requests": n_requests, "plan": plan}

    def _choose_codec(self, requests: list[Request]) -> tuple[str, int, dict]:
        if self._method is not None:
            return self._method, self._level or 3, {"name": f"fixed/{self._method}", "clamped": False}
        if self.bandwidth_bits_per_s is None or self.deadline_s is None:
            return BEST_METHOD, 3, {"name": "default/lossless-l3", "clamped": False}
        n_tokens = sum(len(r.token_ids) for r in requests)
        eps = sum(r.tensors[n].shape[0] * r.tensors[n].shape[2]
                  for r in requests[:1] for n in r.tensors)
        kw = dict(elems_per_token=eps, block_tokens=self.block_tokens,
                  rtt_s=self.rtt_s, pipelined=self._pipelined)
        full = choose_plan(n_tokens, self.bandwidth_bits_per_s, self.deadline_s, **kw)
        exact = choose_plan(n_tokens, self.bandwidth_bits_per_s, self.deadline_s,
                            candidates=_EXACT_CANDIDATES, **kw)
        method, level = _PLAN_CODEC[exact["name"]]
        return method, level, {"full": full, "exact": exact, "codec_method": method,
                               "codec_level": level, "clamped": not full["exact"]}

    def completed(self) -> list[int]:
        """Request indices the receiver confirmed whole (this flush)."""
        return list(self._completed)

    def _on_have(self, payload: bytes) -> None:
        keys = {bytes.fromhex(k) for k in json.loads(payload)}
        n = 0
        for frame in stream_frames(self._store, receiver_keys=keys,
                                   order=self.frame_order, keys=self._needed):
            self.link.send(_pack(_MSG_FRAME, frame))
            n += len(frame) + 5
            self.link.flush()
        if self.last_plan is not None:
            self.last_plan["frame_bytes"] = n
        self.wire_bytes += n
        self._await = "ready"

    def _on_ready(self, payload: bytes) -> None:
        newly = list(json.loads(payload))
        self._completed.extend(i for i in newly if i not in self._completed)
        self._await = None

    # -- receiver -----------------------------------------------------------

    def lookup(self, token_ids) -> dict:
        """External store hits for a request, before committing it."""
        keys = block_keys(token_ids, self.block_tokens)
        names = self.rx.store.names
        hits = sum(1 for k in keys
                   if (e := self.rx.store._blocks.get(k)) is not None
                   and names and all(n in e["t"] for n in names))
        return {"blocks": len(keys), "hits": hits, "missing": len(keys) - hits}

    def poll(self) -> list[int]:
        """Request indices fully arrived in the current manifest."""
        return self.rx.ready_requests()

    def load(self, i: int, token_ids) -> dict[str, np.ndarray]:
        """Bit-exact tensors for request ``i`` (receiver side)."""
        return self.rx.take_request(i, token_ids)

    def _on_manifest(self, payload: bytes) -> None:
        self.rx.accept_manifest(payload)
        self._announced = set()
        have = [k.hex() for k in self.rx.have_keys()]
        self.link.send(_pack(_MSG_HAVE, json.dumps(have).encode()))
        # fully-warm transfer: the receiver already holds every block, so no
        # FRAME will ever arrive to trigger the READY below — announce now.
        newly = [i for i in self.rx.ready_requests() if i not in self._announced]
        if newly:
            self._announced.update(newly)
            self.link.send(_pack(_MSG_READY, json.dumps(newly).encode()))

    def _on_frame(self, payload: bytes) -> None:
        self.rx.accept_frame(payload)
        newly = [i for i in self.rx.ready_requests() if i not in self._announced]
        if newly:
            self._announced.update(newly)
            self.link.send(_pack(_MSG_READY, json.dumps(newly).encode()))
