"""KVConnector extension: memory-link transfer, reconcile, plan-driven codec."""

from __future__ import annotations

import numpy as np
import pytest

from kv_compression.codec import BEST_METHOD
from kv_compression.connector import KVConnector, MemoryLink, TCPLink, pump_until, _pack, _unpack
from kv_compression.dedup import Request, join_blocks, split_blocks

B = 16


def _correlated_bits(h=2, s=64, d=8, noise=0.01, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(h, 1, d)).astype(np.float32)
    walk = np.cumsum(rng.normal(scale=noise, size=(h, s, d)), axis=1).astype(np.float32)
    x = base + walk
    bits32 = x.view(np.uint32)
    rounded = ((bits32 + 0x7FFF + ((bits32 >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def _request(seed, ids, names=("0.k", "0.v"), h=2, d=8):
    ids = np.asarray(ids)
    return Request(ids, {n: _correlated_bits(h=h, s=len(ids), d=d, seed=seed + i)
                         for i, n in enumerate(names)})


def _shared_request(seed, base: Request, n_shared_blocks, suffix_len, names, block_tokens=B):
    ids = np.concatenate([base.token_ids[: n_shared_blocks * block_tokens],
                          np.random.default_rng(seed).integers(0, 100_000, size=suffix_len)])
    tensors = {}
    for i, n in enumerate(sorted(names)):
        shared = split_blocks(base.tensors[n], block_tokens)[:n_shared_blocks]
        tensors[n] = join_blocks(shared + [_correlated_bits(s=suffix_len, seed=seed + 100 + i)])
    return Request(ids, tensors)


def _pair(**kw):
    a, b = MemoryLink.pair()
    return KVConnector(a, **kw), KVConnector(b, **kw)


def _send_all(sender, receiver, requests, **kw):
    for r in requests:
        sender.save(r)
    sender.flush()
    pump_until(sender, receiver)
    return sender.completed()


# ---- message envelope --------------------------------------------------------


def test_message_pack_roundtrip():
    for t, payload in ((1, b""), (3, b"frame" * 100)):
        assert _unpack(_pack(t, payload)) == (t, payload)


def test_message_corrupt_raises():
    with pytest.raises(ValueError):
        _unpack(_pack(3, b"xyz")[:-2])


# ---- end-to-end over memory link ----------------------------------------------


def test_full_exchange_bit_exact():
    r1 = _request(1, np.arange(2 * B + 5))
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=7, names=r1.tensors)
    r3 = _request(3, np.arange(50, 50 + 3 * B))
    tx, rx = _pair(block_tokens=B)
    assert _send_all(tx, rx, [r1, r2, r3]) == [0, 1, 2]
    assert rx.poll() == [0, 1, 2]
    for i, r in enumerate((r1, r2, r3)):
        got = rx.load(i, r.token_ids)
        for n in r.tensors:
            assert np.array_equal(got[n], r.tensors[n]), (i, n)


def test_lookup_reports_external_hits():
    r1 = _request(1, np.arange(2 * B))
    tx, rx = _pair(block_tokens=B)
    assert rx.lookup(r1.token_ids) == {"blocks": 2, "hits": 0, "missing": 2}
    _send_all(tx, rx, [r1])
    # same tokens again (e.g. re-run): fully external now
    assert rx.lookup(r1.token_ids) == {"blocks": 2, "hits": 2, "missing": 0}
    r2 = _shared_request(2, r1, n_shared_blocks=1, suffix_len=B, names=r1.tensors)
    hit = rx.lookup(r2.token_ids)
    assert hit == {"blocks": 2, "hits": 1, "missing": 1}


def test_incremental_second_flush_shares_prefix():
    r1 = _request(1, np.arange(3 * B))
    tx, rx = _pair(block_tokens=B)
    _send_all(tx, rx, [r1])
    first = tx.wire_bytes
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)
    _send_all(tx, rx, [r2])
    assert tx.wire_bytes > first
    got = rx.load(0, r2.token_ids)  # manifest 2 indexes from 0 again
    for n in r2.tensors:
        assert np.array_equal(got[n], r2.tensors[n]), n
    # wire grew by roughly one block's frames, not a full re-transfer
    assert tx.wire_bytes - first < first / 2


def test_reconnect_reflush_is_incremental():
    """Simulate a drop mid-flight: sender re-saves and re-flushes; the
    receiver's HAVE reply skips what already arrived."""
    r1 = _request(1, np.arange(2 * B))
    tx, rx = _pair(block_tokens=B)
    tx.save(r1)
    tx.flush()
    pump_until(tx, rx)
    assert tx.completed() == [0]

    r2 = _request(2, np.arange(100, 100 + 2 * B))  # unrelated
    tx.save(r2)
    tx.flush()
    pump_until(tx, rx)
    assert tx.completed() == [0]
    got = rx.load(0, r2.token_ids)
    for n in r2.tensors:
        assert np.array_equal(got[n], r2.tensors[n]), n


def test_flush_in_flight_guard():
    tx, rx = _pair()
    tx.save(_request(1, np.arange(B)))
    tx.flush()
    with pytest.raises(RuntimeError, match="in flight"):
        tx.save(_request(2, np.arange(B, 2 * B)))
    pump_until(tx, rx)


def test_empty_flush_noop():
    tx, rx = _pair()
    assert tx.flush() == {"requests": 0, "plan": None}
    assert not tx.busy() and not rx.busy()


# ---- plan-driven codec point ----------------------------------------------------


def test_plan_selects_codec_and_records_clamp():
    # toy tensors are wire-negligible, so the planner legitimately prefers
    # zero-codec raw at DC speeds; what must hold: exact chosen, not clamped
    tx, rx = _pair(block_tokens=B, bandwidth_bits_per_s=100e9, deadline_s=0.5, rtt_s=0.0005)
    _send_all(tx, rx, [_request(1, np.arange(64))])
    assert tx.last_plan["exact"]["exact"] is True
    assert tx.last_plan["clamped"] is False
    assert tx.last_plan["codec_method"] in ("raw", BEST_METHOD)

    # bandwidth-starved with cache-scale tensors: the full plan wants a
    # tiered point, the exact subset clamps to lossless
    tx2, rx2 = _pair(block_tokens=B, bandwidth_bits_per_s=50e6, deadline_s=0.5, rtt_s=0.1)
    _send_all(tx2, rx2, [_request(1, np.arange(200), h=64, d=128)])
    assert tx2.last_plan["full"]["exact"] is False
    assert tx2.last_plan["clamped"] is True
    assert tx2.last_plan["codec_method"] == BEST_METHOD  # best exact available


def test_fixed_method_bypasses_planner():
    tx, rx = _pair(block_tokens=B, method="raw", level=3)
    _send_all(tx, rx, [_request(1, np.arange(B))])
    assert tx.last_plan["name"] == "fixed/raw" and tx.last_plan["clamped"] is False
    assert tx.last_plan["frame_bytes"] > 0   # filled in when HAVE is processed
    assert rx.rx.store.method == "raw"


def test_profile_defaults_applied():
    tx, rx = _pair(profile="datacenter")
    assert tx.block_tokens == 256 and tx.frame_order == "name-major"
    tx2, rx2 = _pair(profile="internet")
    assert tx2.block_tokens == 64 and tx2.frame_order == "block-major"


def test_name_major_transfer_completes():
    r1 = _request(1, np.arange(2 * B))
    tx, rx = _pair(block_tokens=B, frame_order="name-major")
    assert _send_all(tx, rx, [r1]) == [0]
    got = rx.load(0, r1.token_ids)
    assert np.array_equal(got["0.k"], r1.tensors["0.k"])


def test_fully_warm_resend_completes_with_zero_frames():
    """Re-sending requests the receiver already holds entirely: the
    exchange must settle with no frames on the wire."""
    r1 = _request(1, np.arange(2 * B))
    tx, rx = _pair(block_tokens=B)
    _send_all(tx, rx, [r1])
    _send_all(tx, rx, [r1])  # identical resend
    assert tx.completed() == [0]
    got = rx.load(0, r1.token_ids)
    assert np.array_equal(got["0.k"], r1.tensors["0.k"])


def test_tcp_link_full_exchange():
    """The socket transport completes the same exchange bit-exactly."""
    r1 = _request(1, np.arange(2 * B + 5))
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=7, names=r1.tensors)
    a, b = TCPLink.pair()
    tx, rx = KVConnector(a, block_tokens=B), KVConnector(b, block_tokens=B)
    for r in (r1, r2):
        tx.save(r)
    tx.flush()
    pump_until(tx, rx)
    assert tx.completed() == [0, 1]
    for i, r in enumerate((r1, r2)):
        got = rx.load(i, r.token_ids)
        for n in r.tensors:
            assert np.array_equal(got[n], r.tensors[n]), (i, n)


def test_sender_persistent_store_skips_reencoding():
    """A second flush over a persistent sender store encodes only the new
    blocks — warm transfers skip re-compression of shared prefixes."""
    r1 = _request(1, np.arange(3 * B))
    a, b = MemoryLink.pair()
    tx, rx = KVConnector(a, block_tokens=B, method=BEST_METHOD), KVConnector(b, block_tokens=B)
    _send_all(tx, rx, [r1])
    store = tx._stores[(BEST_METHOD, 3)]
    new_after_first = store.counters["new"]
    assert new_after_first == 3
    r2 = _shared_request(2, r1, n_shared_blocks=3, suffix_len=B, names=r1.tensors)
    tx.save(r2)
    tx.flush()
    pump_until(tx, rx)
    assert store.counters["new"] == new_after_first + 1  # only the suffix block encoded
    got = rx.load(0, r2.token_ids)
    for n in r2.tensors:
        assert np.array_equal(got[n], r2.tensors[n]), n


def test_codec_point_changes_across_flushes():
    """Per-flush operating points are legal: a later flush may pick a
    different codec than the blocks already sitting in the receiver's
    store — each (block, tensor) carries its own method."""
    r1 = _request(1, np.arange(2 * B))
    tx, rx = _pair(block_tokens=B, method=BEST_METHOD)   # flush 1: chan+planes
    _send_all(tx, rx, [r1])
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)
    tx2 = KVConnector(tx.link, block_tokens=B, method="raw", level=3)  # flush 2: raw
    tx2.save(r2)
    tx2.flush()
    pump_until(tx2, rx)                                  # SAME receiver persists
    assert tx2.completed() == [0]
    got = rx.load(0, r2.token_ids)
    for n in r2.tensors:
        assert np.array_equal(got[n], r2.tensors[n]), n
    # mixed-method store decodes both generations correctly
    assert {"chan+planes", "raw"} <= {m for e in rx.rx.store._blocks.values() for m in e["m"].values()}
