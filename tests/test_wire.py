"""Transmission: codec-point planner and streamable frame protocol."""

from __future__ import annotations

import numpy as np
import pytest

from kv_compression.codec import BEST_METHOD
from kv_compression.dedup import Request, split_blocks, join_blocks
from kv_compression.wire import WireReceiver, build_manifest, decode_frame, encode_frame, stream_frames
from kv_compression.transport import CANDIDATES, PROFILES, choose_plan

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
    tensors = {n: _correlated_bits(h=h, s=len(ids), d=d, seed=seed + i) for i, n in enumerate(names)}
    return Request(ids, tensors)


def _shared_request(seed, base: Request, n_shared_blocks, suffix_len, names, block_tokens=B):
    ids = np.concatenate([base.token_ids[: n_shared_blocks * block_tokens],
                          np.random.default_rng(seed).integers(0, 100_000, size=suffix_len)])
    tensors = {}
    for i, n in enumerate(sorted(names)):
        shared = split_blocks(base.tensors[n], block_tokens)[:n_shared_blocks]
        own = _correlated_bits(s=suffix_len, seed=seed + 100 + i)
        tensors[n] = join_blocks(shared + [own])
    return Request(ids, tensors)


def _family():
    r1 = _request(1, np.arange(2 * B + 5))
    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=7, names=r1.tensors)
    r3 = _request(3, np.arange(1000, 1000 + 3 * B))
    return [r1, r2, r3]


# ---- planner -----------------------------------------------------------------


def test_internet_chooses_tiered_aggressive():
    plan = choose_plan(8192, 50e6, 10.0, rtt_s=0.1)  # 50 Mbps, 10 s SLO
    assert plan["feasible"] and not plan["exact"]
    assert plan["name"] == "tiered/aggressive"


def test_dc_loose_deadline_chooses_exact():
    plan = choose_plan(8192, 100e9, 0.2, rtt_s=0.0005)  # 100 Gbps
    assert plan["feasible"] and plan["exact"] and plan["kl"] == 0.0


def test_dc_tight_deadline_chooses_quality():
    plan = choose_plan(8192, 100e9, 0.05, rtt_s=0.0005)
    assert plan["name"] == "tiered/quality" and plan["feasible"]


def test_infeasible_returns_fastest_with_flag():
    plan = choose_plan(8192, 10e6, 1.0)  # 10 Mbps, 1 s: nothing fits
    assert not plan["feasible"] and plan["slack_s"] < 0
    assert plan["name"] == "tiered/aggressive"  # least wire time


def test_more_bandwidth_never_increases_distortion():
    kls = [choose_plan(8192, bw, 0.2)["kl"] for bw in (10e6, 100e6, 1e9, 10e9, 100e9)]
    assert all(a >= b for a, b in zip(kls, kls[1:]))


def test_dedup_frac_scales_wire_bytes():
    p0 = choose_plan(8192, 100e9, 0.2)
    p5 = choose_plan(8192, 100e9, 0.2, dedup_frac=0.5)
    assert p0["name"] == p5["name"]
    assert p5["wire_bytes"] == pytest.approx(p0["wire_bytes"] / 2)


def test_candidates_all_evaluated():
    plan = choose_plan(1024, 1e9, 5.0)
    assert [r["name"] for r in plan["candidates"]] == [c["name"] for c in CANDIDATES]


def test_profiles_cover_both_regimes():
    assert set(PROFILES) == {"internet", "datacenter"}
    assert PROFILES["internet"]["pipelined"] is False
    assert PROFILES["datacenter"]["frame_order"] == "name-major"


# ---- frames: encode/decode -----------------------------------------------------


def test_frame_roundtrip_self_contained():
    key = b"\x01" * 16
    streams = [b"abc", b"defgh"]
    frame = encode_frame(key, "0.k", 16, "deadbeef", streams, BEST_METHOD)
    head, back = decode_frame(frame)
    assert head["key"] == key.hex() and head["name"] == "0.k" and head["n"] == 16
    assert back == streams


def test_frame_truncated_raises():
    frame = encode_frame(b"\x01" * 16, "0.k", 16, "00", [b"abc"], BEST_METHOD)
    with pytest.raises(ValueError):
        decode_frame(frame[:-1])


# ---- wire protocol: manifest -> frames -> receiver ------------------------------


def test_manifest_frames_roundtrip_bit_exact():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    info = rx.accept_manifest(manifest)
    assert info["n_requests"] == 3 and info["names"] == ["0.k", "0.v"]
    frames = list(stream_frames(store, order="block-major"))
    assert len(frames) == len(store) * 2  # unique blocks x names
    assert not rx.ready_requests()
    for f in frames:
        assert rx.accept_frame(f)["dup"] is False
    assert rx.ready_requests() == [0, 1, 2]
    for i, r in enumerate(reqs):
        got = rx.take_request(i, r.token_ids)
        for n in r.tensors:
            assert np.array_equal(got[n], r.tensors[n]), (i, n)


def test_name_major_order_same_result():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    rx.accept_manifest(manifest)
    for f in stream_frames(store, order="name-major"):
        rx.accept_frame(f)
    assert rx.ready_requests() == [0, 1, 2]
    assert np.array_equal(rx.take_request(1, reqs[1].token_ids)["0.v"], reqs[1].tensors["0.v"])


def test_reconcile_skips_receiver_blocks():
    """Second transfer sharing a prefix: receiver replies with its keys,
    sender streams only the missing blocks (KV rsync)."""
    r1 = _request(1, np.arange(3 * B))
    rx = WireReceiver()
    m1, s1 = build_manifest([r1], block_tokens=B)
    rx.accept_manifest(m1)
    for f in stream_frames(s1):
        rx.accept_frame(f)

    r2 = _shared_request(2, r1, n_shared_blocks=2, suffix_len=B, names=r1.tensors)
    m2, s2 = build_manifest([r2], block_tokens=B)
    rx.accept_manifest(m2)
    frames = list(stream_frames(s2, receiver_keys=rx.have_keys()))
    assert len(frames) == 1 * 2  # only the suffix block x 2 names
    for f in frames:
        rx.accept_frame(f)
    got = rx.take_request(0, r2.token_ids)  # index within manifest 2
    for n in r2.tensors:
        assert np.array_equal(got[n], r2.tensors[n]), n


def test_partial_arrival_and_resume():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    rx.accept_manifest(manifest)
    frames = list(stream_frames(store))
    half = frames[: len(frames) // 2]
    for f in half:
        rx.accept_frame(f)
    # 7 frames = 3 complete blocks (6 frames) + 1 partial block (1 frame);
    # have_keys reports only complete blocks, so the partial block's frames
    # are re-sent (idempotent, dup-ignored) and the 3 are skipped.
    assert len(rx.have_keys()) == 3
    # connection drops; reconnect: receiver reports what it has, sender resumes
    rest = list(stream_frames(store, receiver_keys=rx.have_keys()))
    assert len(rest) == len(frames) - 3 * 2
    for f in rest:
        rx.accept_frame(f)
    assert rx.ready_requests() == [0, 1, 2]


def test_dup_frame_first_writer_wins():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    rx.accept_manifest(manifest)
    frames = list(stream_frames(store))
    for f in frames:
        rx.accept_frame(f)
    nb = rx.store.nbytes()
    for f in frames:
        assert rx.accept_frame(f)["dup"] is True
    assert rx.store.nbytes() == nb


def test_verify_detects_checksum_mismatch():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver(verify=True)
    rx.accept_manifest(manifest)
    first = next(stream_frames(store))
    head, streams = decode_frame(first)
    bad = encode_frame(bytes.fromhex(head["key"]), head["name"], head["n"],
                       "00" * 8, streams, head["method"])
    with pytest.raises(ValueError, match="checksum"):
        rx.accept_frame(bad)


def test_frame_before_manifest_raises():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    with pytest.raises(RuntimeError, match="manifest"):
        rx.accept_frame(next(stream_frames(store)))


def test_unknown_block_frame_raises():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    rx.accept_manifest(manifest)
    frame = encode_frame(b"\xff" * 16, "0.k", B, "00", [b"x" * 8], store.method)
    with pytest.raises(ValueError, match="unknown block"):
        rx.accept_frame(frame)


def test_take_request_not_ready_raises():
    reqs = _family()
    manifest, store = build_manifest(reqs, block_tokens=B)
    rx = WireReceiver()
    rx.accept_manifest(manifest)
    with pytest.raises(RuntimeError, match="not ready"):
        rx.take_request(0, reqs[0].token_ids)
