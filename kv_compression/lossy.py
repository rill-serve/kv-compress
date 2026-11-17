"""Lossy (rate–distortion) KV-cache codecs, as optional companions to the
lossless ``chan+planes`` codec.

Extracted from ``rill-sim`` ``kv_rd.py`` (uniform quantization) and
``kv_sparse.py`` (importance-tiered quantization + eviction), after a
measured grid on Qwen3-8B (teacher-forced window KL/DNLL; see README).

Two codecs:

* **uniform quant** — asymmetric min–max uniform quantization with a
  granularity knob (``chan`` / ``head`` / ``tok`` / ``tensor``), codes
  zstd-compressed. ``chan`` (per (head, dim) along S) is dramatically
  better than coarser groups at low bits and is the default.
* **tiered** — per-token importance ranks map to bit tiers (0 = evict);
  kept tokens are quantized per tier with per-channel scales. This is the
  best measured rate–distortion point by a wide margin (attention mass is
  highly concentrated: importance entropy ~5.3 of 12.8 bits at P=7168).

Both operate on float32 values ``[H, S, D]`` (use :func:`bits_to_f32` on
bf16 bit patterns; bf16 ⊂ fp32 so the conversion is exact). Serialization
keeps the ``list[bytes]`` streams interface of the lossless codec:
``streams[0]`` is a compact JSON header describing the payloads.

Rate accounting matches the original measurements: zstd bytes of integer
codes + 8 bytes (fp32 min + fp32 scale) per quantization group.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

from kv_compression.codec import _zstd, _unzstd

GROUPS = ("chan", "head", "tok", "tensor")

#: tier maps: (cumulative fraction of tokens by importance rank, bits; 0 = drop)
TIER_MAPS: dict[str, list[tuple[float, int]]] = {
    "L-shape": [(0.01, 8), (0.10, 6), (0.25, 4), (0.50, 2), (1.0, 0)],
    "aggressive": [(0.01, 8), (0.10, 4), (1.0, 0)],
    "balanced": [(0.05, 8), (0.25, 6), (1.0, 0)],
    "quality": [(0.10, 8), (0.50, 6), (1.0, 4)],
}


# --------------------------------------------------------------------------
# bf16 bits <-> float32 bridge
# --------------------------------------------------------------------------


def bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 (exact; bf16 values are fp32)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bits(x_f32: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bf16 bit patterns (round-to-nearest-even, like torch)."""
    b = x_f32.astype(np.float32).view(np.uint32)
    rounded = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    # overflow past inf/nan payloads: keep exponent bits sane by clamping
    return (rounded >> 16).astype(np.uint16)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def group_view(x: np.ndarray, group: str) -> np.ndarray:
    """``[H, S, D]`` -> ``[n_groups, elems]`` view for scale computation."""
    H, S, D = x.shape
    if group == "tensor":
        return x.reshape(1, H * S * D)
    if group == "head":
        return x.reshape(H, S * D)
    if group == "chan":
        return np.ascontiguousarray(x.transpose(0, 2, 1)).reshape(H * D, S)
    if group == "tok":
        return x.reshape(H * S, D)
    raise ValueError(group)


def ungroup_view(g: np.ndarray, group: str, hsd: tuple[int, int, int]) -> np.ndarray:
    H, S, D = hsd
    if group == "tensor":
        return g.reshape(H, S, D)
    if group == "head":
        return g.reshape(H, S, D)
    if group == "chan":
        return g.reshape(H, D, S).transpose(0, 2, 1)
    if group == "tok":
        return g.reshape(H, S, D)
    raise ValueError(group)


# --------------------------------------------------------------------------
# quantization core
# --------------------------------------------------------------------------


def quantize(x_f32: np.ndarray, bits: int, group: str = "chan"):
    """Asymmetric uniform min–max quantization per group.

    Returns ``(codes [n_groups, elems] uint8/uint16, mn [G,1] f32,
    scale [G,1] f32)``. ``codes`` are raw integer codes in ``[0, 2**bits-1]``.
    """
    if not 1 <= bits <= 16:
        raise ValueError(bits)
    g = group_view(x_f32, group)
    mn = g.min(-1, keepdims=True).astype(np.float32)
    mx = g.max(-1, keepdims=True).astype(np.float32)
    n = 2 ** bits - 1
    scale = np.maximum((mx - mn) / n, 1e-12).astype(np.float32)
    q = np.rint((g - mn) / scale)
    np.clip(q, 0, n, out=q)
    return q.astype(np.uint8 if bits <= 8 else np.uint16), mn, scale


def dequantize(codes: np.ndarray, mn: np.ndarray, scale: np.ndarray,
               group: str, hsd: tuple[int, int, int]) -> np.ndarray:
    """Inverse of :func:`quantize` -> float32 ``[H, S, D]``."""
    g = codes.astype(np.float32) * scale + mn
    return np.ascontiguousarray(ungroup_view(g, group, hsd))


# --------------------------------------------------------------------------
# codec: uniform quantization
# --------------------------------------------------------------------------


def encode_quant(x_f32: np.ndarray, bits: int, group: str = "chan",
                 level: int = 3) -> list[bytes]:
    """float32 ``[H, S, D]`` -> [header, codes_zstd, scales] byte streams."""
    codes, mn, scale = quantize(x_f32, bits, group)
    codes_z = _zstd(np.ascontiguousarray(codes).tobytes(), level)
    scales = np.ascontiguousarray(np.concatenate([mn.reshape(-1), scale.reshape(-1)])).tobytes()
    head = json.dumps({
        "codec": "quant", "bits": bits, "group": group,
        "shape": list(x_f32.shape), "codes_len": len(codes_z), "scales_len": len(scales),
        "n_groups": int(mn.shape[0]),
    }).encode()
    return [head, codes_z, scales]


def decode_quant(streams: list[bytes]) -> np.ndarray:
    """Inverse of :func:`encode_quant` -> float32 ``[H, S, D]``."""
    head = json.loads(streams[0])
    assert head["codec"] == "quant"
    codes = np.frombuffer(_unzstd(streams[1]), dtype=np.uint8 if head["bits"] <= 8 else np.uint16)
    codes = codes.reshape(head["n_groups"], -1)
    sc = np.frombuffer(streams[2], dtype=np.float32)
    mn, scale = sc[: head["n_groups"], None], sc[head["n_groups"] :, None]
    return dequantize(codes, mn, scale, head["group"], tuple(head["shape"]))


# --------------------------------------------------------------------------
# codec: importance-tiered quantization + eviction
# --------------------------------------------------------------------------


def tier_assignment(token_imp: np.ndarray, tiers: Sequence[tuple[float, int]]) -> np.ndarray:
    """Importance ``[S]`` -> bits per token ``[S]`` (0 = dropped), by rank.

    ``tiers``: sorted list of ``(cumulative_fraction, bits)``.
    """
    P = token_imp.numel() if hasattr(token_imp, "numel") else len(token_imp)
    out = np.zeros(P, dtype=np.int64)
    order = np.argsort(-np.asarray(token_imp), kind="stable")
    prev = 0
    for cum_frac, bits in tiers:
        n_t = min(P, int(round(cum_frac * P)))
        out[order[prev:n_t]] = bits
        prev = n_t
        if prev >= P:
            break
    return out


def encode_tiered(x_f32: np.ndarray, token_imp: np.ndarray,
                  tiers: Optional[str | Sequence[tuple[float, int]]] = "balanced",
                  level: int = 3) -> list[bytes]:
    """float32 ``[H, S, D]`` + per-token importance ``[S]`` -> byte streams.

    Tokens ranked by importance are split into bit tiers (``0`` = evicted);
    kept tokens are quantized per tier with per-(head, dim) channel scales
    (min/max along the tier's token subset).
    """
    if isinstance(tiers, str):
        tiers = TIER_MAPS[tiers]
    bits_map = tier_assignment(token_imp, tiers)
    # kept tokens are stored grouped by tier (ascending bits), each tier a
    # contiguous block of columns; kept_idx preserves original positions.
    tier_bits = sorted({int(b) for _, b in tiers if b > 0})
    tier_pos = [np.nonzero(bits_map == b)[0] for b in tier_bits]
    kept_idx = np.concatenate(tier_pos) if tier_pos else np.empty(0, dtype=np.int64)
    xk = x_f32[:, kept_idx, :]

    streams_body: list[bytes] = []
    tiers_meta = []
    start = 0
    for b, pos in zip(tier_bits, tier_pos):
        n_tok = int(pos.size)
        if n_tok == 0:
            continue
        codes, mn, scale = quantize(xk[:, start : start + n_tok, :], b, "chan")
        codes_z = _zstd(np.ascontiguousarray(codes).tobytes(), level)
        scales = np.ascontiguousarray(np.concatenate([mn.reshape(-1), scale.reshape(-1)])).tobytes()
        tiers_meta.append({"bits": b, "start": start, "n_tok": n_tok,
                           "codes_len": len(codes_z), "scales_len": len(scales)})
        streams_body += [codes_z, scales]
        start += n_tok
    kept_z = _zstd(kept_idx.astype(np.int32).tobytes(), level)
    head = json.dumps({
        "codec": "tiered", "tiers": [list(t) for t in tiers], "tiers_meta": tiers_meta,
        "shape": list(x_f32.shape), "kept_len": len(kept_z), "n_kept": int(kept_idx.size),
    }).encode()
    return [head, kept_z, *streams_body]


def decode_tiered(streams: list[bytes]):
    """Inverse of :func:`encode_tiered`.

    Returns ``(x_kept [H, S_kept, D] float32, kept_idx [S_kept] int32)`` —
    evicted positions are gone; the caller materializes cache holes from
    ``kept_idx`` (original token positions, sorted ascending).
    """
    head = json.loads(streams[0])
    assert head["codec"] == "tiered"
    H, S, D = head["shape"]
    kept_idx = np.frombuffer(_unzstd(streams[1]), dtype=np.int32)
    n_kept = head["n_kept"]
    x = np.empty((H, n_kept, D), dtype=np.float32)
    off = 2
    for tm in head["tiers_meta"]:
        codes_raw = np.frombuffer(_unzstd(streams[off]), dtype=np.uint8 if tm["bits"] <= 8 else np.uint16)
        sc = np.frombuffer(streams[off + 1], dtype=np.float32)
        n_groups = H * D
        codes = codes_raw.reshape(n_groups, tm["n_tok"])
        mn, scale = sc[:n_groups, None], sc[n_groups:, None]
        x[:, tm["start"] : tm["start"] + tm["n_tok"], :] = dequantize(
            codes, mn, scale, "chan", (H, tm["n_tok"], D))
        off += 2
    return x, kept_idx


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def rel_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Relative RMSE: ||x - y|| / ||x||."""
    d = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt((d * d).sum() / max((x.astype(np.float64) ** 2).sum(), 1e-30)))


def evaluate_tensor_lossy(x_f32: np.ndarray, token_imp: Optional[np.ndarray] = None,
                          bits: Sequence[int] = (2, 3, 4, 6, 8)) -> dict:
    """Rate + relative-RMSE report for the lossy codecs on one tensor.

    ``bps`` is over the ORIGINAL element count (evicted elements are free),
    matching the rill-sim rate accounting.
    """
    n = x_f32.size
    out: dict[str, dict] = {}
    for b in bits:
        s = encode_quant(x_f32, b, "chan")
        xq = decode_quant(s)
        out[f"quant{b}b-chan"] = {"bps": 8 * sum(len(t) for t in s) / n, "rel_rmse": rel_rmse(x_f32, xq)}
    if token_imp is not None:
        for name in TIER_MAPS:
            s = encode_tiered(x_f32, token_imp, name)
            xk, kept = decode_tiered(s)
            out[f"tiered/{name}"] = {
                "bps": 8 * sum(len(t) for t in s) / n,
                "keep": float(len(kept) / x_f32.shape[1]),
                "rel_rmse": rel_rmse(x_f32[:, kept, :], xk),
            }
    return out
