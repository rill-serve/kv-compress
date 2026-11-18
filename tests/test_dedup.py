"""Cross-request prefix dedup: block keys, block store, batch archive."""

from __future__ import annotations

import numpy as np
import pytest

from kv_compression.codec import BEST_METHOD
from kv_compression.dedup import (
    BlockStore,
    Request,
    batch_report,
    block_keys,
    decode_batch,
    encode_batch,
    join_blocks,
    split_blocks,
)

B = 16  # small block size keeps many-block cases cheap


def _correlated_bits(h=2, s=64, d=8, noise=0.01, seed=3):
    """bf16 cache-like signal as uint16 bits (same generator as test_codec)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(h, 1, d)).astype(np.float32)
    walk = np.cumsum(rng.normal(scale=noise, size=(h, s, d)), axis=1).astype(np.float32)
    x = base + walk
    bits32 = x.view(np.uint32)
    rounded = ((bits32 + 0x7FFF + ((bits32 >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def _request(seed, ids, names=("0.k", "0.v"), h=2, d=8):
    ids = np.asarray(ids)
    tensors = {n: _correlated_bits(h=h, s=len(ids), d=d, seed=seed + i) for i, n in enumerate(names)}
    return Request(ids, tensors)


def _shared_request(seed, base: Request, n_shared_blocks: int, suffix_len: int, names,
                    block_tokens: int = B, h=2, d=8):
    """A request sharing ``base``'s first ``n_shared_blocks`` blocks (same KV
    content for the shared span, as a real prefix cache would produce),
    with its own suffix."""
    ids = np.concatenate([base.token_ids[: n_shared_blocks * block_tokens],
                          np.random.default_rng(seed).integers(0, 100_000, size=suffix_len)])
    tensors = {}
    for i, n in enumerate(sorted(names)):
        shared = split_blocks(base.tensors[n], block_tokens)[:n_shared_blocks]
        own = _correlated_bits(h=h, s=suffix_len, d=d, seed=seed + 100 + i)
        tensors[n] = join_blocks(shared + [own])
    return Request(ids, tensors)


# ---- block keys: chained-hash identity ---------------------------------------


def test_block_keys_prefix_stability():
    ids = np.arange(1, 201)  # 3 full 64-blocks + 8-token tail
    k = block_keys(ids, 64)
    assert len(k) == 4
    assert block_keys(ids[:64], 64) == [k[0]]          # one-block request hits block 0
    assert block_keys(ids[:128], 64) == k[:2]          # extension preserves all full-block keys
    assert block_keys(list(map(int, ids)), 64) == k    # python ints hash identically


def test_block_keys_diverge_and_never_reconverge():
    head = np.arange(B)
    a = np.concatenate([head, [999], np.arange(100)])
    b = np.concatenate([head, [1000], np.arange(100)])
    ka, kb = block_keys(a, B), block_keys(b, B)
    assert ka[0] == kb[0]        # shared first block
    assert ka[1] != kb[1]        # first differing block
    assert ka[2] != kb[2]        # chained: stays different although tokens re-converge
    assert block_keys(np.concatenate([head, [999], np.arange(100)]), B) == ka


def test_block_keys_tail_is_not_a_full_block():
    a = np.arange(64 + 36)   # blocks: 64 full + 36-token tail
    b = np.arange(2 * 64)    # blocks: 2 full
    ka, kb = block_keys(a, 64), block_keys(b, 64)
    assert ka[0] == kb[0]
    assert ka[1] != kb[1]   # 36-token tail != 64-token block covering the same start


def test_split_join_roundtrip_nonmultiple():
    bits = _correlated_bits(s=2 * B + 7)
    blocks = split_blocks(bits, B)
    assert [b.shape[1] for b in blocks] == [B, B, 7]
    assert np.array_equal(join_blocks(blocks), bits)


# ---- block store --------------------------------------------------------------


def test_store_put_get_bit_exact_and_counters():
    r1 = _request(1, np.arange(2 * B + 6))            # blocks: 2 full + 6 tail
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=9, names=r1.tensors)
    store = BlockStore(block_tokens=B)
    assert store.put_request(r1.token_ids, r1.tensors) == {"blocks": 3, "new": 3, "dup": 0}
    assert store.put_request(r2.token_ids, r2.tensors) == {"blocks": 3, "new": 1, "dup": 2}
    assert len(store) == 4
    for r in (r1, r2):
        got = store.get_request(r.token_ids)
        assert got is not None and set(got) == set(r.tensors)
        for n in r.tensors:
            assert np.array_equal(got[n], r.tensors[n]), n


def test_store_first_writer_wins_silently():
    """Same tokens, different KV content (kernel nondeterminism stand-in):
    default policy ignores the second copy and keeps decode bit-exact."""
    r1 = _request(1, np.arange(B))
    r2 = _request(2, np.arange(B))  # same ids, different bits
    store = BlockStore(block_tokens=B)
    store.put_request(r1.token_ids, r1.tensors)
    assert store.put_request(r2.token_ids, r2.tensors) == {"blocks": 1, "new": 0, "dup": 1}
    got = store.get_request(r1.token_ids)
    assert np.array_equal(got["0.k"], r1.tensors["0.k"])  # first writer's content


def test_store_collision_modes():
    r1 = _request(1, np.arange(B))
    r2 = _request(2, np.arange(B))
    store = BlockStore(block_tokens=B, on_collision="error")
    store.put_request(r1.token_ids, r1.tensors)
    with pytest.raises(ValueError, match="collision"):
        store.put_request(r2.token_ids, r2.tensors)
    store_warn = BlockStore(block_tokens=B, on_collision="warn")
    store_warn.put_request(r1.token_ids, r1.tensors)
    with pytest.warns(UserWarning, match="collision"):
        store_warn.put_request(r2.token_ids, r2.tensors)
    # identical re-put stays silent even in warn mode
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store_warn.put_request(r1.token_ids, r1.tensors)


def test_store_identical_request_full_dup():
    r = _request(1, np.arange(3 * B))
    store = BlockStore(block_tokens=B)
    store.put_request(r.token_ids, r.tensors)
    nb = store.nbytes()
    assert store.put_request(r.token_ids, r.tensors) == {"blocks": 3, "new": 0, "dup": 3}
    assert store.nbytes() == nb and len(store) == 3


def test_store_partial_prefix_no_cross_dedup():
    """A short request that is a strict prefix of a longer one hits all its
    FULL blocks but its tail (36 < 64 tokens) is a separate identity."""
    r_long = _request(1, np.arange(2 * 64 + 36))   # blocks: 64, 64, 36-tail
    r_short = _shared_request(2, r_long, n_shared_blocks=1, suffix_len=36,
                              names=r_long.tensors, block_tokens=64)  # 64, 36-tail
    store = BlockStore(block_tokens=64)
    store.put_request(r_long.token_ids, r_long.tensors)
    assert store.put_request(r_short.token_ids, r_short.tensors) == {"blocks": 2, "new": 1, "dup": 1}
    got = store.get_request(r_short.token_ids)
    for n in r_short.tensors:
        assert np.array_equal(got[n], r_short.tensors[n]), n


def test_store_missing_keys_and_none():
    store = BlockStore(block_tokens=B)
    r = _request(1, np.arange(2 * B))
    assert store.get_request(r.token_ids) is None
    assert len(store.missing_keys(r.token_ids)) == 2
    store.put_request(r.token_ids, r.tensors)
    assert store.missing_keys(r.token_ids) == []
    assert store.get_request(np.arange(5 * B)) is None


def test_store_shape_mismatch_raises():
    store = BlockStore(block_tokens=B)
    r = _request(1, np.arange(B))
    store.put_request(r.token_ids, r.tensors)
    bad = Request(r.token_ids, {"0.k": _correlated_bits(h=4, s=B), "0.v": _correlated_bits(s=B)})
    with pytest.raises(ValueError, match="shape"):
        store.put_request(bad.token_ids, bad.tensors)


def test_store_wrong_s_raises():
    r = _request(1, np.arange(B))
    store = BlockStore(block_tokens=B)
    bad = Request(r.token_ids, {"0.k": _correlated_bits(s=B + 1)})
    with pytest.raises(ValueError, match="len\\(token_ids\\)"):
        store.put_request(bad.token_ids, bad.tensors)


# ---- batch archive -------------------------------------------------------------


def test_encode_decode_batch_bit_exact():
    r1 = _request(1, np.arange(2 * B + 5))
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=7, names=r1.tensors)
    r3 = _request(3, np.arange(1000, 1000 + 3 * B))  # unrelated
    r4 = r1  # exact duplicate
    streams = encode_batch([r1, r2, r3, r4], block_tokens=B)
    out = decode_batch(streams)
    assert len(out) == 4
    for got, r in zip(out, [r1, r2, r3, r4]):
        assert set(got) == set(r.tensors)
        for n in r.tensors:
            assert got[n].shape == r.tensors[n].shape
            assert np.array_equal(got[n], r.tensors[n]), n


def test_batch_report_dedup_stats():
    r1 = _request(1, np.arange(2 * B))                       # 2 blocks
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)  # 2 dup + 1
    r3 = _shared_request(3, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)
    rep = batch_report(encode_batch([r1, r2, r3], block_tokens=B))
    assert rep["n_requests"] == 3
    assert rep["blocks_logical"] == 8 and rep["blocks_unique"] == 4
    assert rep["dedup_frac"] == pytest.approx(0.5)
    assert 0 < rep["bps_vs_raw"] < 16
    assert 0 < rep["bps_unique"] < 16


def test_batch_dedup_beats_per_request_archives():
    prefix_len, suffix_len = 8 * B, B
    r1 = _request(1, np.arange(prefix_len + suffix_len))
    r2 = _shared_request(2, r1, n_shared_blocks=8, suffix_len=suffix_len, names=r1.tensors)
    r3 = _shared_request(3, r1, n_shared_blocks=8, suffix_len=suffix_len, names=r1.tensors)
    reqs = [r1, r2, r3]
    single = sum(len(s) for r in reqs for s in encode_batch([r], block_tokens=B))
    batch = sum(len(s) for s in encode_batch(reqs, block_tokens=B))
    assert batch < 0.6 * single  # shared 8/9 of every request


def test_batch_names_must_match():
    r1 = _request(1, np.arange(B))
    r2 = _request(2, np.arange(B), names=("0.k",))
    with pytest.raises(ValueError, match="tensor names"):
        encode_batch([r1, r2], block_tokens=B)


def test_batch_empty():
    streams = encode_batch([], block_tokens=B)
    assert decode_batch(streams) == []
    rep = batch_report(streams)
    assert rep["n_requests"] == 0 and rep["bytes_archive"] > 0


def test_store_from_archive_serves_requests():
    r1 = _request(1, np.arange(2 * B + 3))
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)
    streams = encode_batch([r1, r2], block_tokens=B, level=3)
    store, refs = BlockStore.from_archive(streams)
    assert [len(rr) for rr in refs] == [3, 3]
    assert store.method == BEST_METHOD and store.block_tokens == B
    for r in (r1, r2):
        got = store.get_request(r.token_ids)
        for n in r.tensors:
            assert np.array_equal(got[n], r.tensors[n]), n


def test_report_rejects_foreign_streams():
    from kv_compression.codec import encode as codec_encode

    bits = _correlated_bits(s=B)
    with pytest.raises(ValueError):
        batch_report(codec_encode(bits, BEST_METHOD, 3))
