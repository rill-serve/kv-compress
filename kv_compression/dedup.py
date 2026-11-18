"""Cross-request KV deduplication: a prefix block store over the codecs.

Why prefix-only.  Layer >= 1 K/V are projections of *contextualized*
hidden states (and K carries RoPE position), so the KV columns of two
requests are bit-equal iff their token prefixes from position 0 are
equal.  Arbitrary repeated spans — the same document at different
offsets, common suffixes — do NOT produce equal KV and cannot be
deduplicated; shared prefixes do.  This module is the storage-side
analogue of prefix caching (vLLM APC / SGLang RadixAttention):

* each request's cache is split into fixed ``block_tokens`` blocks;
* a block is keyed by a **chained hash of token ids** — its *logical*
  identity, not a hash of the KV bytes.  Recomputing the same prefix in
  a different batch shape can yield non-bit-exact KV (GEMM reduction
  order); token-keying still hits, and first-writer-wins keeps every
  decoded request bit-exact against the first stored copy.  A stored
  64-bit content checksum is kept only to catch true key collisions
  (``on_collision``);
* unique blocks are compressed independently (default ``chan+planes``)
  so one block fetches/decompresses in O(block) — the deliberate
  trade-off vs concatenating all unique blocks into one stream is a
  little ratio for random access;
* per request only block references are stored (~4 bytes per block).

Partial tails (< ``block_tokens``) are keyed by their own token count:
a 36-token tail and a 64-token block covering the same first 36 tokens
are different identities and do not cross-dedup.

``encode_batch`` / ``decode_batch`` serialize a whole batch of requests
as ``unique blocks + per-request references`` with the same
``list[bytes]`` + JSON-header convention as the lossy codecs.  Note the
lossy interaction: tiered importance is per-request (each suffix
attends to the shared prefix differently), so a shared compressed block
forces a shared tier decision — dedup here is lossless-layer only.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from kv_compression.codec import BEST_METHOD, _unzstd, _zstd, decode, encode

DEFAULT_BLOCK_TOKENS = 64
KEY_BYTES = 16  # 128-bit block keys (vLLM-prefix-cache sized)
_ARCHIVE_CODEC = "dedup-batch"


# --------------------------------------------------------------------------
# block identity: chained token-id hash
# --------------------------------------------------------------------------


def block_keys(token_ids, block_tokens: int = DEFAULT_BLOCK_TOKENS,
               key_bytes: int = KEY_BYTES) -> list[bytes]:
    """Token ids -> per-block chained hash keys (RadixAttention-style).

    ``key_i = sha256(key_{i-1} || tokens[iB:(i+1)B])[:key_bytes]`` with
    ``key_{-1} = b""``.  Consequences: identical prefixes share identical
    leading keys; after the first differing block every later key differs
    (chaining), even if the tokens re-converge; tails hash only their own
    (shorter) byte length.  This is the *logical* identity of the block —
    independent of the KV bytes, so batch-shape kernel nondeterminism
    cannot break a hit.
    """
    if block_tokens < 1:
        raise ValueError(block_tokens)
    ids = np.ascontiguousarray(token_ids, dtype="<i4").reshape(-1)
    keys: list[bytes] = []
    h = b""
    for i in range(0, ids.shape[0], block_tokens):
        h = hashlib.sha256(h + ids[i : i + block_tokens].tobytes()).digest()[:key_bytes]
        keys.append(h)
    return keys


def split_blocks(bits: np.ndarray, block_tokens: int = DEFAULT_BLOCK_TOKENS) -> list[np.ndarray]:
    """uint16 ``[H, S, D]`` -> list of ``[H, b, D]`` blocks along S (copies)."""
    return [
        np.ascontiguousarray(bits[:, i : i + block_tokens, :])
        for i in range(0, bits.shape[1], block_tokens)
    ]


def join_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Inverse of :func:`split_blocks` -> uint16 ``[H, S, D]``."""
    if not blocks:
        raise ValueError("empty block list")
    return np.concatenate(list(blocks), axis=1)


@dataclass
class Request:
    """One request's cache: token ids + per-(layer, tensor) uint16 bits.

    All tensors share ``S = len(token_ids)``; ``tensors`` maps a stable
    name (``"3.k"``, ``"3.v"``, ...) to uint16 ``[H, S, D]`` bits as
    produced by :func:`kv_compression.bf16_bits`.
    """

    token_ids: np.ndarray
    tensors: dict[str, np.ndarray]


# --------------------------------------------------------------------------
# block store
# --------------------------------------------------------------------------


class BlockStore:
    """Content(logic)-addressed compressed KV block store.

    Blocks live one ``block_tokens``-token span per key, holding the
    compressed streams of every tensor name of that span.  Keys come from
    :func:`block_keys` (token-id hash chain); on a hit the stored copy is
    reused as-is — first-writer-wins — so decode is always bit-exact
    against the first copy ever stored for a prefix.
    """

    def __init__(self, block_tokens: int = DEFAULT_BLOCK_TOKENS, method: str = BEST_METHOD,
                 level: int = 3, threads: int = 0, on_collision: str = "ignore"):
        if block_tokens < 1:
            raise ValueError(block_tokens)
        if on_collision not in ("ignore", "warn", "error"):
            raise ValueError(on_collision)
        self.block_tokens = block_tokens
        self.method = method
        self.level = level
        self.threads = threads
        self.on_collision = on_collision
        self._blocks: dict[bytes, dict] = {}   # key -> {"n", "sum", "t": {name: streams}}
        self._shapes: dict[str, tuple[int, int]] = {}  # name -> (H, D)
        self._index: dict[bytes, int] = {}
        self._order: list[bytes] = []
        self.counters = {"puts": 0, "new": 0, "dup": 0}

    # -- introspection ---------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Sorted tensor names seen by the store."""
        return sorted(self._shapes)

    def order(self) -> list[bytes]:
        """Block keys in insertion (first-seen) order."""
        return list(self._order)

    def index(self, key: bytes) -> Optional[int]:
        """Insertion-order index of a block key, or None."""
        return self._index.get(key)

    def __len__(self) -> int:
        return len(self._blocks)

    def nbytes(self) -> int:
        """Total compressed stream bytes stored."""
        return sum(len(s) for e in self._blocks.values() for s in e["t"].values())

    def stats(self) -> dict:
        return {"blocks": len(self._blocks), "nbytes": self.nbytes(), **self.counters}

    # -- put --------------------------------------------------------------

    @staticmethod
    def _name_sum(name: str, bits: np.ndarray) -> bytes:
        """64-bit content checksum of one (block, tensor) — collision guard."""
        h = hashlib.sha256()
        h.update(name.encode())
        h.update(np.ascontiguousarray(bits).tobytes())
        return h.digest()[:8]

    def _handle_collision(self, key: bytes, blocks: dict[str, np.ndarray]) -> None:
        if self.on_collision == "ignore":
            return
        stored = self._blocks[key]["sums"]
        for name, bits in blocks.items():
            if name in stored and stored[name] != self._name_sum(name, bits):
                msg = (f"block key collision: same token prefix, different KV content "
                       f"(key {key.hex()[:8]}, tensor {name!r}) — keeping the first-stored copy")
                if self.on_collision == "error":
                    raise ValueError(msg)
                warnings.warn(msg)
                return

    def put_request(self, token_ids, tensors: dict[str, np.ndarray]) -> dict:
        """Split a request into blocks, compress and store the missing ones.

        Returns per-call stats ``{"blocks", "new", "dup"}``.  Hits reuse
        the stored copy (first-writer-wins); see ``on_collision`` for the
        (astronomically rare, or bug-indicating) content-mismatch case.
        """
        ids = np.asarray(token_ids).reshape(-1)
        if ids.size == 0:
            raise ValueError("empty token_ids")
        names = sorted(tensors)
        if not names:
            raise ValueError("no tensors")
        for name in names:
            t = tensors[name]
            if t.ndim != 3 or t.dtype != np.uint16:
                raise ValueError(f"{name}: uint16 [H, S, D] expected, got {t.dtype} {t.shape}")
            if t.shape[1] != ids.size:
                raise ValueError(f"{name}: S={t.shape[1]} != len(token_ids)={ids.size}")
            H, D = t.shape[0], t.shape[2]
            if name in self._shapes and self._shapes[name] != (H, D):
                raise ValueError(f"{name}: shape {(H, D)} != stored {self._shapes[name]}")
            self._shapes.setdefault(name, (H, D))

        keys = block_keys(ids, self.block_tokens)
        splits = {name: split_blocks(tensors[name], self.block_tokens) for name in names}
        new = dup = 0
        for i, key in enumerate(keys):
            blocks = {name: splits[name][i] for name in names}
            n = int(min(ids.size - i * self.block_tokens, self.block_tokens))
            if key in self._blocks:
                dup += 1
                self._handle_collision(key, blocks)
            else:
                self._blocks[key] = {
                    "n": n,
                    "sums": {name: self._name_sum(name, blocks[name]) for name in names},
                    "m": {name: self.method for name in names},
                    "t": {name: encode(blocks[name], self.method, self.level, self.threads)
                          for name in names},
                }
                self._index[key] = len(self._order)
                self._order.append(key)
                new += 1
        self.counters["puts"] += len(keys)
        self.counters["new"] += new
        self.counters["dup"] += dup
        return {"blocks": len(keys), "new": new, "dup": dup}

    # -- get ---------------------------------------------------------------

    def get_block(self, key: bytes, name: str) -> Optional[np.ndarray]:
        """One block's bits ``[H, b, D]``, or None if the key is unknown.

        The codec point is per (block, tensor): different transfers may
        legitimately choose different operating points for the same
        block identity."""
        e = self._blocks.get(key)
        if e is None:
            return None
        if name not in e["t"]:
            raise KeyError(f"block {key.hex()[:8]} has no tensor {name!r}")
        H, D = self._shapes[name]
        return decode(e["t"][name], (H, e["n"], D), e.get("m", {}).get(name, self.method))

    def missing_keys(self, token_ids) -> list[bytes]:
        """Block keys of a request not present in the store."""
        return [k for k in block_keys(token_ids, self.block_tokens) if k not in self._blocks]

    def get_request(self, token_ids, names: Optional[Sequence[str]] = None
                    ) -> Optional[dict[str, np.ndarray]]:
        """Full reconstruct of a request's tensors, or None if any block misses.

        ``names=None`` uses every name the store has seen.  Shared-prefix
        blocks return the first-stored copy (bit-exact for the first
        writer by construction).
        """
        keys = block_keys(token_ids, self.block_tokens)
        if any(k not in self._blocks for k in keys):
            return None
        names = self.names if names is None else list(names)
        return {name: join_blocks([self.get_block(k, name) for k in keys]) for name in names}

    @classmethod
    def from_archive(cls, streams: list[bytes]) -> tuple["BlockStore", list[list[int]]]:
        """Rebuild a serving-side store from an :func:`encode_batch` archive.

        Returns ``(store, refs)``; the store can then serve
        :meth:`get_request` for any request whose blocks the archive holds.
        """
        return _archive_to_store(streams)


# --------------------------------------------------------------------------
# batch archive: unique blocks + per-request references
# --------------------------------------------------------------------------


def _ingest_batch(store: BlockStore, requests: Sequence[Request]) -> list[list[int]]:
    """Ingest requests into a (possibly persistent) store; return per-request
    block-reference lists as indices into the store's first-seen order."""
    refs: list[list[int]] = []
    ref_names: Optional[set[str]] = None
    for r in requests:
        if ref_names is None:
            ref_names = set(r.tensors)
        elif set(r.tensors) != ref_names:
            raise ValueError(
                f"all requests in a batch must share tensor names: {sorted(ref_names)} vs {sorted(r.tensors)}")
        keys = block_keys(r.token_ids, store.block_tokens)
        store.put_request(r.token_ids, r.tensors)
        refs.append([store.index(k) for k in keys])
    return refs


def _spn(store: BlockStore) -> int:
    """Compressed streams per (block, tensor) for the store's method."""
    if not store._order:
        return 0
    first = store._blocks[store._order[0]]["t"]
    return len(next(iter(first.values())))


def _header_dict(store: BlockStore, refs: list[list[int]], block_tokens: int,
                 method: str, level: int, keys: Optional[list[bytes]] = None) -> dict:
    keys = store._order if keys is None else keys
    return {
        "codec": _ARCHIVE_CODEC, "block_tokens": block_tokens, "method": method,
        "level": level, "spn": _spn(store),
        "names": store.names,
        "shapes": {n: list(sd) for n, sd in store._shapes.items()},
        "blocks": [{"key": k.hex(), "n": store._blocks[k]["n"],
                    "sums": {n: s.hex() for n, s in store._blocks[k]["sums"].items()}}
                   for k in keys],
        "requests": [{"refs": rr} for rr in refs],
    }


def request_manifest(store: BlockStore, refs: list[list[int]], method: Optional[str] = None,
                     level: Optional[int] = None) -> tuple[bytes, list[bytes]]:
    """Manifest covering exactly the blocks referenced by ``refs`` — for
    persistent sender stores whose first-seen order spans many transfers.

    Returns ``(manifest_bytes, needed_keys)``; refs are remapped to indices
    within the manifest's own block list."""
    needed: list[bytes] = []
    seen: set[bytes] = set()
    for rr in refs:
        for idx in rr:
            k = store._order[idx]
            if k not in seen:
                seen.add(k)
                needed.append(k)
    sub = {k: i for i, k in enumerate(needed)}
    refs2 = [[sub[store._order[i]] for i in rr] for rr in refs]
    head = _header_dict(store, refs2, store.block_tokens,
                        method or store.method, level if level is not None else store.level,
                        keys=needed)
    return json.dumps(head).encode(), needed


def encode_batch(requests: Sequence[Request], block_tokens: int = DEFAULT_BLOCK_TOKENS,
                 method: str = BEST_METHOD, level: int = 3, threads: int = 0,
                 on_collision: str = "ignore") -> list[bytes]:
    """Many requests -> one archive ``[header, lens_zstd, body]``.

    Every request must carry the same set of tensor names (one batch =
    one model config).  The body holds each *unique* block's compressed
    streams once, in first-seen order; per request only the block
    reference list is stored.
    """
    store = BlockStore(block_tokens, method, level, threads, on_collision)
    refs = _ingest_batch(store, requests)
    lens: list[int] = []
    body = bytearray()
    for key in store._order:
        e = store._blocks[key]
        for name in store.names:
            for s in e["t"][name]:
                lens.append(len(s))
                body += s
    header = json.dumps(_header_dict(store, refs, block_tokens, method, level)).encode()
    lens_z = _zstd(np.asarray(lens, dtype=np.int32).tobytes(), 1)
    return [header, lens_z, bytes(body)]


def _archive_to_store(streams: list[bytes]) -> tuple[BlockStore, list[list[int]]]:
    head = json.loads(streams[0])
    if head["codec"] != _ARCHIVE_CODEC:
        raise ValueError(f"not a {_ARCHIVE_CODEC} archive: {head['codec']}")
    store = BlockStore(head["block_tokens"], head["method"], head["level"])
    store._shapes = {n: tuple(sd) for n, sd in head["shapes"].items()}
    names: list[str] = head["names"]
    lens = np.frombuffer(_unzstd(streams[1]), dtype=np.int32)
    body, off = streams[2], 0
    it = iter(lens.tolist())
    for bmeta in head["blocks"]:
        key = bytes.fromhex(bmeta["key"])
        t: dict[str, list[bytes]] = {}
        for name in names:
            ss = []
            for _ in range(head["spn"]):
                n_ = int(next(it))
                ss.append(bytes(body[off : off + n_]))
                off += n_
            t[name] = ss
        store._blocks[key] = {"n": int(bmeta["n"]),
                              "sums": {n: bytes.fromhex(s) for n, s in bmeta["sums"].items()},
                              "m": {n: head["method"] for n in names},
                              "t": t}
        store._index[key] = len(store._order)
        store._order.append(key)
    if off != len(body):
        raise ValueError(f"archive body consumed {off} of {len(body)} bytes")
    refs = [list(rq["refs"]) for rq in head["requests"]]
    return store, refs


def decode_batch(streams: list[bytes]) -> list[dict[str, np.ndarray]]:
    """Inverse of :func:`encode_batch` -> per-request ``{name: bits [H, S, D]}``.

    Token ids are not part of the archive (they live with the caller);
    shared-prefix regions return the first-stored copy, bit-exact.
    """
    store, refs = _archive_to_store(streams)
    out: list[dict[str, np.ndarray]] = []
    for ref in refs:
        out.append({
            name: join_blocks([store.get_block(store._order[i], name) for i in ref])
            for name in store.names
        })
    return out


def batch_report(streams: list[bytes]) -> dict:
    """Dedup + rate stats for an :func:`encode_batch` archive."""
    head = json.loads(streams[0])
    if head["codec"] != _ARCHIVE_CODEC:
        raise ValueError(head["codec"])
    hd = sum(H * D for H, D in (tuple(sd) for sd in head["shapes"].values()))
    elems_logical = sum(sum(head["blocks"][i]["n"] for i in rq["refs"]) * hd
                        for rq in head["requests"])
    elems_stored = sum(b["n"] for b in head["blocks"]) * hd
    archive_bytes = sum(len(s) for s in streams)
    body_bytes = len(streams[2]) if len(streams) > 2 else 0
    return {
        "n_requests": len(head["requests"]),
        "block_tokens": head["block_tokens"],
        "blocks_logical": sum(len(rq["refs"]) for rq in head["requests"]),
        "blocks_unique": len(head["blocks"]),
        "elems_logical": elems_logical,
        "elems_stored": elems_stored,
        "dedup_frac": (1.0 - elems_stored / elems_logical) if elems_logical else 0.0,
        "bytes_raw_bf16": 2 * elems_logical,
        "bytes_body": body_bytes,
        "bytes_archive": archive_bytes,
        "bps_vs_raw": 8 * archive_bytes / elems_logical if elems_logical else 0.0,
        "bps_unique": 8 * body_bytes / elems_stored if elems_stored else 0.0,
    }
