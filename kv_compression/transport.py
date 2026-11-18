"""Transmission planning: pick the codec operating point for a link.

One stack, two regimes — the choice is not a deployment-time fork but a
per-transfer decision driven by measured bandwidth and deadline:

* **data center** (25–400 Gbps, sub-ms RTT): wire time dominates and the
  encoder (≈4.4 GB/s threaded) is slower than a 100 G line (12.5 GB/s),
  so serial compression is a net loss; pipelined lossless or raw wins,
  lossy only under tight SLO;
* **internet** (Mbps–Gbps, 50–200 ms RTT): bytes dominate — lossless
  ×1.47 barely moves the needle, so tiered lossy + dedup + incremental
  transfer are the only way to hit interactive deadlines.

:func:`choose_plan` evaluates every measured rate–distortion candidate
(the README tables, Qwen3-8B) against a time model —

* serial: ``encode + wire + decode + rtt``
* pipelined: ``wire + one block of encode/decode + rtt`` (blocks stream
  out as they are produced, e.g. layer-by-layer during prefill)

— and returns the least-distorting feasible candidate, or the fastest
one with ``feasible=False`` when the deadline cannot be met.  The
rate–distortion curve becomes the network rate controller.
"""

from __future__ import annotations

from typing import Optional, Sequence

#: Qwen3-8B: 36 layers x 2 (K,V) x 8 KV heads x 128 dim uint16 elems/token
QWEN3_8B_ELEMS_PER_TOKEN = 36 * 2 * 8 * 128

#: Measured operating points (README, Qwen3-8B book text).  ``bps`` is
#: bits per uint16 element over the bf16 cache; ``kl`` is the measured
#: teacher-forced window KL (nats); enc/dec throughputs in GB/s (None =
#: no codec work).  Tiered enc is quantize + zstd-on-small-codes; dec is
#: the measured 2.2 GB/s swap-in rate.  These are planning constants,
#: not laws — override via ``candidates``.
CANDIDATES: tuple[dict, ...] = (
    {"name": "raw",               "bps": 16.00, "kl": 0.000, "exact": True,  "enc_gbps": None, "dec_gbps": None},
    {"name": "lossless/l3",       "bps": 11.13, "kl": 0.000, "exact": True,  "enc_gbps": 4.4,  "dec_gbps": 10.0},
    {"name": "lossless/l19",      "bps": 10.63, "kl": 0.000, "exact": True,  "enc_gbps": 0.03, "dec_gbps": 10.0},
    {"name": "tiered/quality",    "bps": 4.10,  "kl": 0.002, "exact": False, "enc_gbps": 2.0,  "dec_gbps": 2.2},
    {"name": "tiered/balanced",   "bps": 1.28,  "kl": 0.030, "exact": False, "enc_gbps": 2.0,  "dec_gbps": 2.2},
    {"name": "tiered/aggressive", "bps": 0.38,  "kl": 0.044, "exact": False, "enc_gbps": 2.0,  "dec_gbps": 2.2},
)

#: Regime starting points (knobs, not requirements): internet favours
#: small resumable frames + strongest compression (encode CPU is cheap
#: next to seconds of wire time); datacenter favours large frames,
#: name-major order (layer-pipelined with prefill) and cheap levels.
PROFILES: dict[str, dict] = {
    "internet": {"block_tokens": 64, "level": 3, "frame_order": "block-major",
                 "reconcile": True, "pipelined": False},
    "datacenter": {"block_tokens": 256, "level": 3, "frame_order": "name-major",
                   "reconcile": True, "pipelined": True},
}


def _codec_s(n_elem: float, gbps: Optional[float]) -> float:
    """Time to codec ``n_elem`` uint16 elements at ``gbps`` GB/s (0 if None)."""
    if not gbps:
        return 0.0
    return n_elem * 2.0 / (gbps * 1e9)


def choose_plan(n_tokens: int, bandwidth_bits_per_s: float, deadline_s: float, *,
                elems_per_token: int = QWEN3_8B_ELEMS_PER_TOKEN, dedup_frac: float = 0.0,
                block_tokens: int = 64, rtt_s: float = 0.0, pipelined: bool = True,
                candidates: Sequence[dict] = CANDIDATES) -> dict:
    """Pick the least-distorting codec point that meets the deadline.

    ``dedup_frac`` is the fraction of elements already at the receiver
    (from :mod:`kv_compression.dedup` reconciliation) — only the missing
    elements ride the wire.  Returns a plan dict with the chosen
    candidate's fields plus ``time_s`` breakdown, ``feasible`` /
    ``slack_s``, and every evaluated ``candidates`` row for logging.
    """
    if not 0.0 <= dedup_frac < 1.0:
        raise ValueError(dedup_frac)
    if n_tokens <= 0 or bandwidth_bits_per_s <= 0 or deadline_s <= 0:
        raise ValueError("n_tokens, bandwidth and deadline must be positive")
    miss = 1.0 - dedup_frac
    n_elem = n_tokens * elems_per_token * miss
    block_elem = min(block_tokens, n_tokens) * elems_per_token * miss

    rows = []
    for c in candidates:
        wire_bytes = n_elem * c["bps"] / 8.0
        wire_s = n_elem * c["bps"] / bandwidth_bits_per_s
        if pipelined:
            t = wire_s + rtt_s + _codec_s(block_elem, c["enc_gbps"]) + _codec_s(block_elem, c["dec_gbps"])
        else:
            t = _codec_s(n_elem, c["enc_gbps"]) + wire_s + _codec_s(n_elem, c["dec_gbps"]) + rtt_s
        rows.append({"name": c["name"], "bps": c["bps"], "kl": c["kl"], "exact": c["exact"],
                     "wire_bytes": wire_bytes, "time_s": t, "feasible": t <= deadline_s})

    feasible = [r for r in rows if r["feasible"]]
    if feasible:
        best = min(feasible, key=lambda r: (r["kl"], r["time_s"]))
    else:
        best = min(rows, key=lambda r: r["time_s"])
    return {
        "name": best["name"], "bps": best["bps"], "kl": best["kl"], "exact": best["exact"],
        "wire_bytes": best["wire_bytes"], "time_s": best["time_s"], "deadline_s": deadline_s,
        "feasible": best["feasible"], "slack_s": deadline_s - best["time_s"],
        "model": "pipelined" if pipelined else "serial",
        "n_elem": n_elem, "block_tokens": block_tokens,
        "candidates": rows,
    }
