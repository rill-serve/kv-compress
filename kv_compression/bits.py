"""bf16 <-> uint16 bit plumbing (bijective, NaN-safe).

All transforms are at the *bit-pattern* level: values are never converted
through float arithmetic, so NaNs, infinities, signed zeros and denormals
survive round trips exactly.

The optional torch helpers (:func:`bf16_bits` / :func:`bits_bf16`) import
torch lazily; the rest of the package is numpy-only.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# torch <-> bits (lazy torch import)
# --------------------------------------------------------------------------


def bf16_bits(t) -> np.ndarray:
    """``[...]`` bf16 torch tensor (any device) -> ``[...]`` uint16 numpy bits."""
    import torch  # lazy

    if t.dtype != torch.bfloat16:
        raise ValueError(f"bf16 expected, got {t.dtype} (view-reinterpret would corrupt layout)")
    return t.detach().view(torch.int16).cpu().numpy().view(np.uint16)


def bits_bf16(u: np.ndarray, like) -> np.ndarray:
    """uint16 numpy bits -> bf16 torch tensor on ``like``'s device."""
    import torch  # lazy

    it = torch.from_numpy(np.ascontiguousarray(u).view(np.int16))
    return it.view(torch.bfloat16).to(
        like.device, like.dtype if like.dtype == torch.bfloat16 else torch.bfloat16
    )


# --------------------------------------------------------------------------
# monotone key (IEEE ordering trick)
# --------------------------------------------------------------------------


def mono(u: np.ndarray) -> np.ndarray:
    """uint16 bit pattern -> monotone uint16 key.

    Maps the IEEE-754 bit layout onto the unsigned integer order:
    ``k = u ^ 0xFFFF`` for negatives (reversed), ``u ^ 0x8000`` otherwise,
    so that ``k(a) < k(b)`` iff ``a < b`` under total order (+0.0 > -0.0).
    """
    sign = (u & 0x8000) != 0
    return np.where(sign, u ^ 0xFFFF, u ^ 0x8000).astype(np.uint16)


def unmono(k: np.ndarray) -> np.ndarray:
    sign = k < 0x8000
    return np.where(sign, k ^ 0xFFFF, k ^ 0x8000).astype(np.uint16)


# --------------------------------------------------------------------------
# sequence-axis delta / xor (channel-major (H, D, S) layout)
# --------------------------------------------------------------------------


def zigzag16(d_i16: np.ndarray) -> np.ndarray:
    d = d_i16.astype(np.int32)
    return ((d << 1) ^ (d >> 15)).astype(np.uint16)


def unzigzag16(u: np.ndarray) -> np.ndarray:
    d = u.astype(np.int32)
    return ((d >> 1) ^ -(d & 1)).astype(np.int16)


def delta_last(k_u16: np.ndarray) -> np.ndarray:
    """Monotone-key int16 wraparound delta along the last axis -> zigzag uint16.

    Input is channel-major ``(H, D, S)``; the first S element (seed) is
    handled by the caller.
    """
    k = k_u16.view(np.int16)
    d = (k[..., 1:] - k[..., :-1]).astype(np.int16)  # wraps, C semantics
    return zigzag16(d)


def undelta_last(seed_u16: np.ndarray, d_u16: np.ndarray) -> np.ndarray:
    k0 = seed_u16.view(np.int16).astype(np.int32)
    d = unzigzag16(d_u16).astype(np.int32)
    ks = np.concatenate(
        [k0[..., None], k0[..., None] + np.cumsum(d, axis=-1)], axis=-1
    )
    return (ks & 0xFFFF).astype(np.uint16)


def xor_last(u16: np.ndarray) -> np.ndarray:
    """Bitwise XOR of consecutive raw bit patterns along the last axis.

    Unlike arithmetic delta, XOR carries no borrows: equal exponents cancel
    in the high byte while mantissa differences stay in the low byte."""
    return u16[..., 1:] ^ u16[..., :-1]


def unxor_last(seed_u16: np.ndarray, x_u16: np.ndarray) -> np.ndarray:
    out = np.empty(x_u16.shape[:-1] + (x_u16.shape[-1] + 1,), dtype=np.uint16)
    out[..., 0] = seed_u16
    np.bitwise_xor.accumulate(x_u16, axis=-1, out=out[..., 1:])
    out[..., 1:] ^= seed_u16[..., None]
    return out
