"""Per-method encode/decode for lossless KV-cache compression.

Methods (all verified bit-exact on decode):

* ``raw``               — token-major bytes + zstd (naive baseline);
* ``chan``              — channel-major ``(H, D, S)`` layout only;
* ``chan+planes``       — + low/high byte-plane split   <- BEST (measured);
* ``chan+delta``        — + monotone int16 delta + zigzag (worse: reference);
* ``chan+xor``          — + bitwise xor delta (reference);
* ``chan+delta+planes`` — delta then plane split (reference);
* ``chan+xor+planes``   — xor then plane split (reference).

``encode`` takes uint16 bits ``[H, S, D]`` (head, sequence, head-dim), as
produced by :func:`kv_compression.bf16_bits`. ``decode`` is its exact
inverse; both are pure numpy + zstd.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from kv_compression.bits import delta_last, undelta_last, unxor_last, xor_last

BEST_METHOD = "chan+planes"
METHODS = ("raw", "chan", "chan+delta", "chan+planes", "chan+delta+planes", "chan+xor+planes")


# --------------------------------------------------------------------------
# zstd wrapper
# --------------------------------------------------------------------------


def _zstd(data: bytes, level: int, threads: int = 0) -> bytes:
    import zstandard as zstd  # lazy

    if threads > 0:
        c = zstd.ZstdCompressor(level=level, threads=threads)
    else:
        c = zstd.ZstdCompressor(level=level)
    return c.compress(data)


def _unzstd(data: bytes) -> bytes:
    import zstandard as zstd  # lazy

    return zstd.ZstdDecompressor().decompress(data, max_output_size=1 << 34)


# --------------------------------------------------------------------------
# per-method encode / decode
# --------------------------------------------------------------------------


def encode(bits: np.ndarray, method: str, level: int = 3, threads: int = 0) -> list[bytes]:
    """bits: uint16 ``[H, S, D]``. Returns the compressed byte streams."""
    if method == "raw":
        return [_zstd(np.ascontiguousarray(bits).tobytes(), level, threads)]
    chan = np.ascontiguousarray(bits.transpose(0, 2, 1))  # (H, D, S)
    if method == "chan":
        return [_zstd(chan.tobytes(), level, threads)]
    if method == "chan+planes":
        lo = np.ascontiguousarray(chan & 0xFF).astype(np.uint8)
        hi = np.ascontiguousarray(chan >> 8).astype(np.uint8)
        return [_zstd(lo.tobytes(), level, threads), _zstd(hi.tobytes(), level, threads)]
    seed = np.ascontiguousarray(chan[..., 0])
    d = delta_last(chan) if method.startswith("chan+delta") else xor_last(chan)
    if method in ("chan+delta", "chan+xor"):
        return [_zstd(seed.tobytes(), level, threads), _zstd(d.tobytes(), level, threads)]
    if method in ("chan+delta+planes", "chan+xor+planes"):
        lo = np.ascontiguousarray(d & 0xFF).astype(np.uint8)
        hi = np.ascontiguousarray(d >> 8).astype(np.uint8)
        return [_zstd(seed.tobytes(), level, threads), _zstd(lo.tobytes(), level, threads), _zstd(hi.tobytes(), level, threads)]
    raise ValueError(method)


def decode(streams: list[bytes], shape_hsd: tuple[int, int, int], method: str) -> np.ndarray:
    """Inverse of :func:`encode`; returns uint16 bits ``[H, S, D]``."""
    H, S, D = shape_hsd
    if method == "raw":
        return np.frombuffer(_unzstd(streams[0]), dtype=np.uint16).reshape(H, S, D)
    if method == "chan":
        chan = np.frombuffer(_unzstd(streams[0]), dtype=np.uint16).reshape(H, D, S)
        return chan.transpose(0, 2, 1)
    if method == "chan+planes":
        lo = np.frombuffer(_unzstd(streams[0]), dtype=np.uint8).reshape(H, D, S)
        hi = np.frombuffer(_unzstd(streams[1]), dtype=np.uint8).reshape(H, D, S)
        chan = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
        return chan.transpose(0, 2, 1)
    seed = np.frombuffer(_unzstd(streams[0]), dtype=np.uint16).reshape(H, D)
    is_delta = method.startswith("chan+delta")
    if method in ("chan+delta", "chan+xor"):
        d = np.frombuffer(_unzstd(streams[1]), dtype=np.uint16).reshape(H, D, S - 1)
    else:
        lo = np.frombuffer(_unzstd(streams[1]), dtype=np.uint8).reshape(H, D, S - 1)
        hi = np.frombuffer(_unzstd(streams[2]), dtype=np.uint8).reshape(H, D, S - 1)
        d = (lo.astype(np.uint16) | (hi.astype(np.uint16) << 8))
    chan = undelta_last(seed, d) if is_delta else unxor_last(seed, d)
    return chan.transpose(0, 2, 1)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def delta_entropy_bits(bits: np.ndarray) -> float:
    """Shannon entropy (bits/elem) of the zigzag delta symbols (chan-major).

    An analytic floor for any entropy coder applied to the delta
    representation; the measured ``chan+planes`` bps sits essentially on it.
    """
    chan = np.ascontiguousarray(bits.transpose(0, 2, 1))
    d = delta_last(chan).reshape(-1)
    counts = np.bincount(d.astype(np.int64), minlength=1 << 16)
    p = counts.astype(np.float64) / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def evaluate_tensor(bits: np.ndarray, level: int = 3, verify: bool = True) -> dict:
    """Compress one (layer, tensor) uint16 ``[H, S, D]`` under all methods.

    Every method is decoded back and compared bit-exact before its ratio is
    reported; ``ok`` flags the verification.
    """
    n = bits.size
    out: dict[str, dict] = {}
    for m in METHODS:
        streams = encode(bits, m, level)
        rec = {"bytes": sum(len(s) for s in streams), "bps": 8 * sum(len(s) for s in streams) / n}
        if verify:
            back = decode(streams, bits.shape, m)
            rec["ok"] = bool(np.array_equal(back, bits))
        out[m] = rec
    out["delta_entropy"] = {"bps": delta_entropy_bits(bits)}
    return out
