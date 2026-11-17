"""Round-trip and behavioural tests for the lossless KV codec."""

from __future__ import annotations

import numpy as np
import pytest

from kv_compression import (
    BEST_METHOD,
    METHODS,
    bf16_bits,
    bits_bf16,
    decode,
    delta_entropy_bits,
    encode,
    mono,
    undelta_last,
    unmono,
    unxor_last,
    unzigzag16,
    xor_last,
    zigzag16,
    delta_last,
)


def _correlated_bits(h=4, s=64, d=8, noise=0.01, seed=3):
    """bf16 cache-like signal: smooth along S + small noise, as uint16 bits."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(h, 1, d)).astype(np.float32)
    walk = np.cumsum(rng.normal(scale=noise, size=(h, s, d)), axis=1).astype(np.float32)
    x = (base + walk).astype(np.float32)
    # emulate bf16 rounding (round-to-nearest-even, like torch's cast)
    bits32 = x.view(np.uint32)
    rounded = ((bits32 + 0x7FFF + ((bits32 >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


# ---- bit plumbing bijectivity ----------------------------------------------


def test_mono_roundtrip_on_all_edge_patterns():
    edges = np.array([0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF], dtype=np.uint16)
    rng = np.random.default_rng(0)
    u = np.concatenate([edges, rng.integers(0, 1 << 16, size=100_000, dtype=np.uint16)])
    assert np.array_equal(unmono(mono(u)), u)
    # ordering sanity: +0.0 (0x0000) maps above -0.0 (0x8000)
    assert mono(np.array([0x8000], dtype=np.uint16))[0] < mono(np.array([0x0000], dtype=np.uint16))[0]


def test_zigzag_roundtrip_random_int16():
    rng = np.random.default_rng(1)
    d = rng.integers(-(1 << 15), 1 << 15, size=100_000).astype(np.int16)
    assert np.array_equal(unzigzag16(zigzag16(d)), d)


def test_delta_roundtrip_channel_major():
    rng = np.random.default_rng(2)
    k = rng.integers(0, 1 << 16, size=(4, 8, 33)).astype(np.uint16)  # (H, D, S)
    seed = np.ascontiguousarray(k[..., 0])
    back = undelta_last(seed, delta_last(k))
    assert np.array_equal(back, k)


def test_xor_roundtrip_channel_major():
    rng = np.random.default_rng(5)
    k = rng.integers(0, 1 << 16, size=(4, 8, 33)).astype(np.uint16)
    seed = np.ascontiguousarray(k[..., 0])
    assert np.array_equal(unxor_last(seed, xor_last(k)), k)


def test_bf16_bits_roundtrip():
    torch = pytest.importorskip("torch")
    x = torch.randn(3, 5, 7, dtype=torch.bfloat16)
    assert torch.equal(bits_bf16(bf16_bits(x), x), x)


def test_special_values_survive_roundtrip():
    """NaN, +-inf, +-0, denormals: every method must be bit-exact."""
    torch = pytest.importorskip("torch")
    special = np.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F,  # +-0, smallest denormals
         0x3F80, 0xBF80,  # +-1.0
         0x7F80, 0xFF80,  # +-inf
         0x7FC0, 0xFFC0,  # NaNs
         0x7F7F, 0xFF7F],  # +-max normal
        dtype=np.uint16,
    )
    bits = np.broadcast_to(special.reshape(1, 14, 1), (3, 14, 1)).copy()  # (H, S, D)
    for m in METHODS:
        streams = encode(bits, m, 3)
        assert np.array_equal(decode(streams, bits.shape, m), bits), m


# ---- method encode/decode: bit-exact and behaviour ---------------------------


def test_all_methods_bit_exact():
    bits = _correlated_bits()
    for m in METHODS:
        streams = encode(bits, m, 3)
        back = decode(streams, bits.shape, m)
        assert np.array_equal(back, bits), m


def test_incompressible_random_data_roundtrip():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 1 << 16, size=(2, 16, 4)).astype(np.uint16)
    for m in METHODS:
        streams = encode(bits, m, 3)
        assert np.array_equal(decode(streams, bits.shape, m), bits), m


def test_best_method_beats_raw_on_correlated_signal():
    """The headline claim: chan+planes < raw bytes on cache-like bf16 data."""
    bits = _correlated_bits(s=256, noise=0.005)
    sizes = {m: sum(len(s) for s in encode(bits, m, 3)) for m in ("raw", BEST_METHOD)}
    assert sizes[BEST_METHOD] < sizes["raw"] * 0.9


def test_plane_split_beats_interleaved_bytes():
    """chan+planes < chan: the byte-plane split itself is the win."""
    bits = _correlated_bits(s=256, noise=0.005)
    sizes = {m: sum(len(s) for s in encode(bits, m, 3)) for m in ("chan", BEST_METHOD)}
    assert sizes[BEST_METHOD] < sizes["chan"]


def test_delta_beats_layout_only_on_smooth_signal():
    """Sanity of the delta machinery on a synthetic smooth signal (on real
    KV caches the delta representation loses; see README)."""
    bits = _correlated_bits()
    sizes = {m: sum(len(s) for s in encode(bits, m, 3)) for m in METHODS}
    assert sizes["chan+delta"] < sizes["chan"] * 0.8
    assert sizes["chan+delta"] < sizes["raw"]
    assert delta_entropy_bits(bits) * bits.size / 8 < sizes["chan"]  # floor below layout-only bytes


def test_evaluate_tensor_reports_all_methods_verified():
    from kv_compression import evaluate_tensor

    bits = _correlated_bits(h=2, s=32, d=4)
    report = evaluate_tensor(bits, level=3)
    assert set(report) == set(METHODS) | {"delta_entropy"}
    for m in METHODS:
        assert report[m]["ok"] is True
        assert 0 < report[m]["bps"] <= 16 + 16 * 0.06  # incompressible may expand slightly
    assert 0 < report["delta_entropy"]["bps"] <= 16.0


# ---- throughput smoke --------------------------------------------------------


def test_bench_tensors_smoke():
    from kv_compression.bench import bench_tensors

    bits = _correlated_bits(h=2, s=64, d=4)
    tensors = [("0.k", bits), ("0.v", bits)]
    rows = bench_tensors(tensors, [("raw", 1, 0), (BEST_METHOD, 1, 2)], repeats=1)
    assert len(rows) == 2 and all(r["ok"] for r in rows)
    for r in rows:
        assert r["enc_gbps"] > 0 and r["dec_gbps"] > 0 and 0 < r["bps"] < 20
