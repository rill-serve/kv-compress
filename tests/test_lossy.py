"""Lossy codec tests: quantization round trips, tiered layout, bf16 bridge."""

from __future__ import annotations

import numpy as np
import pytest

from kv_compression.lossy import (
    GROUPS,
    TIER_MAPS,
    bits_to_f32,
    decode_quant,
    decode_tiered,
    dequantize,
    encode_quant,
    encode_tiered,
    evaluate_tensor_lossy,
    f32_to_bits,
    group_view,
    quantize,
    rel_rmse,
    tier_assignment,
    ungroup_view,
)


def _cache_like(h=4, s=128, d=16, seed=7):
    """Cache-like float32: channel scales vary, per-token outliers (sinks)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(h, s, d)).astype(np.float32) * rng.uniform(0.01, 3.0, size=(1, 1, d)).astype(np.float32)
    x[:, :4] *= 20.0
    imp = np.zeros(s, dtype=np.float64)
    imp[:4] = 10.0  # sink tokens dominate importance
    imp[4:] = rng.exponential(size=s - 4)
    return x, imp


# ---- bf16 <-> f32 bridge ------------------------------------------------------


def test_bits_f32_bridge_exact():
    """bf16 bits -> f32 -> bf16 bits is identity for all 2^16 patterns."""
    u = np.arange(1 << 16, dtype=np.uint16)
    # NaN/inf rounding through f32->bf16 keeps payload; finite values exact
    back = f32_to_bits(bits_to_f32(u))
    finite = ~np.isnan(bits_to_f32(u)) & ~np.isinf(bits_to_f32(u))
    assert np.array_equal(back[finite], u[finite])


def test_bits_f32_bridge_matches_torch_bf16_cast():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 64, 8)).astype(np.float32)
    ours = bits_to_f32(f32_to_bits(x))
    ref = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    assert np.array_equal(ours, ref)


# ---- grouping ----------------------------------------------------------------


@pytest.mark.parametrize("group", GROUPS)
def test_group_ungroup_roundtrip(group):
    x, _ = _cache_like()
    g = group_view(x, group)
    assert g.shape[0] * g.shape[1] == x.size
    assert np.array_equal(ungroup_view(g, group, x.shape), x)


# ---- quantization --------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 6, 8, 12])
def test_quant_codes_in_range_and_dequant_close(bits):
    x, _ = _cache_like()
    codes, mn, scale = quantize(x, bits, "chan")
    assert codes.max() <= 2 ** bits - 1 and codes.min() >= 0
    xq = dequantize(codes, mn, scale, "chan", x.shape)
    # error bounded by half a quantization step of the widest channel
    max_step = float(scale.max()) / 2 + 1e-12
    assert np.abs(x - xq).max() <= max_step


@pytest.mark.parametrize("group", GROUPS)
def test_quant_codec_streams_roundtrip_structure(group):
    x, _ = _cache_like(h=2, s=32, d=4)
    streams = encode_quant(x, 4, group)
    assert len(streams) == 3
    back = decode_quant(streams)
    assert back.shape == x.shape and back.dtype == np.float32
    assert rel_rmse(x, back) < 0.3  # 4-bit, sink-inflated ranges; structure round-trip


def test_quant_8bit_chan_is_near_lossless_on_bf16_source():
    """8-bit per-channel quantization of a bf16 tensor: rel RMSE ~ 1e-3."""
    x, _ = _cache_like()
    xb = bits_to_f32(f32_to_bits(x))  # round source to bf16 first
    back = decode_quant(encode_quant(xb, 8, "chan"))
    assert rel_rmse(xb, back) < 0.02


def test_quant_rate_decreases_with_bits():
    x, _ = _cache_like()
    bps = {b: sum(len(t) for t in encode_quant(x, b, "chan")) for b in (2, 4, 8)}
    assert bps[2] < bps[4] < bps[8]


def test_chan_granularity_beats_tensor_on_skewed_channels():
    """The measured finding: per-channel scales ≫ per-tensor at low bits."""
    x, _ = _cache_like()
    r = {g: rel_rmse(x, decode_quant(encode_quant(x, 3, g))) for g in ("chan", "tensor")}
    assert r["chan"] < r["tensor"] * 0.5


# ---- tiered -------------------------------------------------------------------


def test_tier_assignment_ranks_and_drops():
    imp = np.array([0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    tiers = [(0.25, 8), (0.50, 4), (1.0, 0)]  # top-2 @8b, next-2 @4b, rest dropped
    m = tier_assignment(imp, tiers)
    assert m[1] == 8 and m[2] == 8       # ranks 1,2
    assert m[3] == 4 and m[4] == 4       # ranks 3,4
    assert m[0] == 0 and m[5] == 0       # dropped
    assert set(m) == {0, 4, 8}


def test_tiered_codec_roundtrip_and_layout():
    x, imp = _cache_like()
    for name, tiers in TIER_MAPS.items():
        streams = encode_tiered(x, imp, tiers)
        xk, kept = decode_tiered(streams)
        # kept positions match bits_map > 0 (grouped by tier, ascending bits)
        m = tier_assignment(imp, tiers)
        assert sorted(kept.tolist()) == sorted(np.nonzero(m > 0)[0].tolist()), name
        assert xk.shape == (x.shape[0], len(kept), x.shape[2]), name
        # per-tier RMSE bounded by half-step of that tier
        for b in sorted({t[1] for t in tiers if t[1] > 0}):
            sel = np.nonzero(m[kept] == b)[0]
            if sel.size == 0:
                continue
            codes, mn, scale = quantize(x[:, kept[sel], :], b, "chan")
            ref = dequantize(codes, mn, scale, "chan", (x.shape[0], sel.size, x.shape[2]))
            assert np.allclose(xk[:, sel, :], ref, atol=1e-6), (name, b)


def test_tiered_drops_low_importance_tokens():
    x, imp = _cache_like()
    xk, kept = decode_tiered(encode_tiered(x, imp, "aggressive"))
    assert kept.size <= int(round(0.10 * x.shape[1])) + 1
    assert 0 in kept  # sink token kept
    # kept tokens are exactly the top-importance ones
    top = set(np.argsort(-imp)[: kept.size].tolist())
    assert set(kept.tolist()) == top


def test_tiered_rate_accounts_over_original_size():
    x, imp = _cache_like(h=2, s=64, d=8)
    rep = evaluate_tensor_lossy(x, imp, bits=(4, 8))
    for k, r in rep.items():
        assert 0 < r["bps"] < 16 and r["rel_rmse"] >= 0, k
    # aggressive keeps ~10%: rate must be far below uniform 8-bit chan
    assert rep["tiered/aggressive"]["bps"] < rep["quant8b-chan"]["bps"]
    assert rep["tiered/aggressive"]["keep"] <= 0.15


def test_tiered_quality_beats_aggressive_rmse():
    x, imp = _cache_like()
    rep = evaluate_tensor_lossy(x, imp)
    assert rep["tiered/quality"]["rel_rmse"] < rep["tiered/aggressive"]["rel_rmse"]
