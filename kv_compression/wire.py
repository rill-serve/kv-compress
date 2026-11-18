"""Streamable block frames + incremental receiver for KV transmission.

The blob archive (:func:`kv_compression.encode_batch`) is a *storage*
format; transmission wants three things it cannot give:

* **self-contained frames** — one (block, tensor) per frame, so blocks
  stream out as they are produced (layer-major order pipelines with
  prefill), arrive out of order, and resume at block granularity after
  a dropped connection;
* **receiver state** — a :class:`kv_compression.BlockStore` on the far
  side, so shared prefixes are transferred once per *receiver*, not
  once per request: the sender asks for ``receiver_keys`` and skips
  everything the receiver already holds (KV rsync);
* **completion tracking** — the manifest declares each request's block
  references, so the receiver knows exactly when a request is whole.

Wire exchange (both regimes; see :mod:`kv_compression.transport` for
which codec point to pick)::

    manifest, store = build_manifest(requests, block_tokens=64)
    rx = WireReceiver(); rx.accept_manifest(manifest)   # -> receiver reply:
    for frame in stream_frames(store, receiver_keys=rx.have_keys()):
    ...                                                 # send frames
    rx.accept_frame(frame)
    tensors = rx.take_request(0, token_ids)             # bit-exact
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from kv_compression.codec import BEST_METHOD, decode
from kv_compression.dedup import (
    DEFAULT_BLOCK_TOKENS,
    BlockStore,
    Request,
    _header_dict,
    _ingest_batch,
)

_FRAME_BYTEORDER = "little"  # u32 header-length prefix


# --------------------------------------------------------------------------
# manifest (sender side)
# --------------------------------------------------------------------------


def build_manifest(requests: Sequence[Request], block_tokens: int = DEFAULT_BLOCK_TOKENS,
                   method: str = BEST_METHOD, level: int = 3, threads: int = 0,
                   on_collision: str = "ignore") -> tuple[bytes, BlockStore]:
    """Requests -> ``(manifest_bytes, sender_store)``.

    The manifest is the archive header alone (requests, refs, block
    keys, names, shapes); the store feeds :func:`stream_frames`.
    """
    store = BlockStore(block_tokens, method, level, threads, on_collision)
    refs = _ingest_batch(store, requests)
    return json.dumps(_header_dict(store, refs, block_tokens, method, level)).encode(), store


# --------------------------------------------------------------------------
# frames: one (block, tensor) per self-contained record
# --------------------------------------------------------------------------


def encode_frame(key: bytes, name: str, n: int, sum_hex: str, streams: list[bytes],
                 method: str) -> bytes:
    """``[u32 header_len][json header][streams...]`` — resumable at frame edges."""
    head = json.dumps({"key": key.hex(), "name": name, "n": n, "sum": sum_hex,
                       "method": method, "lens": [len(s) for s in streams]}).encode()
    return len(head).to_bytes(4, _FRAME_BYTEORDER) + head + b"".join(streams)


def decode_frame(buf: bytes) -> tuple[dict, list[bytes]]:
    """Inverse of :func:`encode_frame` -> ``(header, streams)``."""
    hlen = int.from_bytes(buf[:4], _FRAME_BYTEORDER)
    head = json.loads(buf[4 : 4 + hlen])
    off = 4 + hlen
    streams = []
    for l in head["lens"]:
        streams.append(bytes(buf[off : off + l]))
        off += l
    if off != len(buf):
        raise ValueError(f"frame consumed {off} of {len(buf)} bytes")
    return head, streams


def stream_frames(store: BlockStore, receiver_keys: Optional[Iterable[bytes]] = None,
                  order: str = "block-major", keys: Optional[Iterable[bytes]] = None) -> Iterator[bytes]:
    """Yield one frame per (unique block, tensor) the receiver is missing.

    ``order``: ``"block-major"`` sends a block's every tensor before the
    next block (dedup- and resume-friendly, natural for the internet);
    ``"name-major"`` sends a tensor's every block first (layer-pipelined
    with prefill, natural for the datacenter).  Blocks the receiver
    already has (``receiver_keys``, from ``WireReceiver.have_keys()``)
    are skipped entirely.  ``keys`` restricts to a subset of the store
    (persistent sender stores spanning many transfers).
    """
    if order not in ("block-major", "name-major"):
        raise ValueError(order)
    skip = set(receiver_keys) if receiver_keys is not None else set()
    names = store.names

    def emit(key: bytes, name: str) -> bytes:
        e = store._blocks[key]
        method = e.get("m", {}).get(name, store.method)
        return encode_frame(key, name, e["n"], e["sums"][name].hex(), e["t"][name], method)

    src = list(store._order) if keys is None else list(keys)
    keys = [k for k in src if k not in skip]
    if order == "block-major":
        for k in keys:
            for name in names:
                yield emit(k, name)
    else:
        for name in names:
            for k in keys:
                yield emit(k, name)


# --------------------------------------------------------------------------
# receiver
# --------------------------------------------------------------------------


class WireReceiver:
    """Incremental far-end of the frame protocol.

    Feed it a manifest, then frames in any order; it fills an internal
    :class:`BlockStore` and reports which requests are whole.  With
    ``verify=True`` every frame is decompressed and re-hashed against
    its declared checksum before being accepted (paranoid path; frame
    transport integrity is otherwise the transport's job).
    """

    def __init__(self, verify: bool = False):
        self.verify = verify
        self._head: Optional[dict] = None
        self._keys: list[bytes] = []          # manifest block order
        self._store = BlockStore()
        self._frames = 0

    @property
    def store(self) -> BlockStore:
        """The receiver-side block store (persists across transfers)."""
        return self._store

    def accept_manifest(self, buf: bytes) -> dict:
        """Parse a manifest; returns ``{"n_requests", "n_blocks", "names"}``.

        Merges into the existing store: blocks already held (from a
        previous transfer) keep their content — that state is what
        makes the *next* transfer incremental.
        """
        head = json.loads(buf)
        if head.get("codec") != "dedup-batch":
            raise ValueError(f"not a dedup manifest: {head.get('codec')}")
        self._head = head
        self._keys = [bytes.fromhex(b["key"]) for b in head["blocks"]]
        st = self._store
        st.block_tokens, st.method, st.level = head["block_tokens"], head["method"], head["level"]
        for n, sd in head["shapes"].items():
            st._shapes.setdefault(n, tuple(sd))
        for k, bmeta in zip(self._keys, head["blocks"]):
            if k in st._blocks:
                if st._blocks[k]["n"] != int(bmeta["n"]):
                    raise ValueError(f"block {k.hex()[:8]} identity mismatch across manifests")
                continue
            st._blocks[k] = {"n": int(bmeta["n"]),
                             "sums": {n: bytes.fromhex(s) for n, s in bmeta["sums"].items()},
                             "m": {}, "t": {}}
            st._index[k] = len(st._order)
            st._order.append(k)
        return {"n_requests": len(head["requests"]), "n_blocks": len(self._keys),
                "names": list(head["names"])}

    def have_keys(self) -> set[bytes]:
        """Keys of *complete* blocks (every tensor present) — the reconcile
        reply for :func:`stream_frames`."""
        names = self._head["names"]
        return {k for k, e in self._store._blocks.items() if all(n in e["t"] for n in names)}

    def accept_frame(self, buf: bytes) -> dict:
        """Ingest one frame -> ``{"key", "name", "dup"}`` (first-writer-wins)."""
        if self._head is None:
            raise RuntimeError("accept_manifest first")
        head, streams = decode_frame(buf)
        key, name = bytes.fromhex(head["key"]), head["name"]
        entry = self._store._blocks.get(key)
        if entry is None:
            raise ValueError(f"frame for unknown block {key.hex()[:8]} (manifest mismatch?)")
        if entry["n"] != head["n"]:
            raise ValueError(f"block {key.hex()[:8]} token count mismatch: "
                             f"{entry['n']} manifest vs {head['n']} frame")
        if name in entry["t"]:
            return {"key": key, "name": name, "dup": True}
        if self.verify:
            H, D = self._store._shapes[name]
            bits = decode(streams, (H, entry["n"], D), head["method"])
            if BlockStore._name_sum(name, bits).hex() != head["sum"]:
                raise ValueError(f"frame checksum mismatch: block {key.hex()[:8]} tensor {name!r}")
        entry["t"][name] = streams
        entry.setdefault("m", {})[name] = head["method"]
        self._frames += 1
        return {"key": key, "name": name, "dup": False}

    def request_ready(self, i: int) -> bool:
        """True when every block/tensor frame of request ``i`` has arrived."""
        if self._head is None:
            raise RuntimeError("accept_manifest first")
        names = self._head["names"]
        refs = self._head["requests"][i]["refs"]
        for idx in refs:
            e = self._store._blocks.get(self._keys[idx])
            if e is None or any(n not in e["t"] for n in names):
                return False
        return True

    def ready_requests(self) -> list[int]:
        return [i for i in range(len(self._head["requests"])) if self.request_ready(i)]

    def take_request(self, i: int, token_ids) -> dict[str, np.ndarray]:
        """Bit-exact tensors of request ``i`` (raises if not ready)."""
        if not self.request_ready(i):
            raise RuntimeError(f"request {i} not ready: missing frames")
        return self._store.get_request(token_ids, names=self._head["names"])
