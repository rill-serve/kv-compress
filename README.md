# kv-compression

Lossless + lossy (rate–distortion) KV-cache compression. Extracted from
`rill-sim` (`rill_sim/research/kv_compress.py`, `kv_rd.py`, `kv_sparse.py`)
after measured method grids on Qwen3-8B (36L × 8 KV heads × d=128, book +
code text; `results/compress2_*.json`, `rd_*.json`, `tiered_*.json`,
`repl5_book_8b.json` in rill-sim).

Distortion metrics in the tables: teacher-forced window evaluation —
prefill P clean tokens, corrupt the cache, forward a W=1024 window; KL in
nats between clean/corrupt next-token distributions, ΔNLL on real text.
All end-to-end numbers validated with actual generation (swap-in decode).

## At a glance

| need | codec | rate (bps) | distortion |
|------|-------|-----------|------------|
| bit-exact restore | `encode(bits, "chan+planes")` | 10.9–11.1 (×1.4–1.5) | none |
| simple, no model signal | `lossy.encode_quant(x, bits∈{4,6,8}, "chan")` | 3.5–5.2 (×3–4.6) | KL 0.01–0.28 |
| best rate–distortion | `lossy.encode_tiered(x, imp, tiers∈{aggressive,balanced,quality})` | 0.38–4.1 (×3.9–42) | KL 0.002–0.04 |

## The optimal method: `chan+planes`

```text
bf16 K/V tensor [H, S, D]
  → uint16 bit patterns        (view-reinterpret; values, NaNs, denormals untouched)
  → transpose to (H, D, S)     (channel-major: consecutive sequence steps adjacent)
  → split into lo / hi bytes   (mantissa noise | exponent+sign structure)
  → zstd each plane (level 3)
```

Decode is the exact inverse; every method is verified bit-exact in tests and
was verified at every measurement checkpoint.

### Why it wins (measured, Qwen3-8B, zstd level 3)

Bits per element (bf16 reference = 16.000):

| method              | book @32k | code @32k | ratio |
|---------------------|-----------|-----------|-------|
| raw (zstd)          | 12.229    | 12.204    | ×1.31 |
| chan                | 11.951    | 11.927    | ×1.34 |
| chan+delta          | 12.389    | 12.374    | ×1.29 |
| chan+delta+planes   | 11.243    | 11.235    | ×1.42 |
| chan+xor+planes     | 11.066    | 11.049    | ×1.45 |
| **chan+planes**     | **10.868**| **10.848**| **×1.47** |
| *delta_entropy*     | *10.871*  | *10.899*  | *(floor)* |

Findings:

1. **Delta/XOR along the sequence axis hurt.** KV values are not smoothly
   varying along S; delta *expands* the data before zstd sees it
   (12.389 vs 11.951 bps for chan alone). The intuition that motivated delta
   encoding is wrong for real caches.
2. **The win is the byte-plane split.** bf16 high bytes (exponent+sign)
   cluster heavily → compress well; low bytes (mantissa) are near-random →
   incompressible. On interleaved bytes zstd must model both at once; split
   streams let it spend its window on the structured plane.
3. **`chan+planes` sits on the entropy floor.** 10.868 measured vs 10.871
   analytic Shannon entropy of the best delta symbols — no delta/xor
   representation can meaningfully beat it at any coder strength.
4. **Decode is as cheap as raw zstd** — plane merge + transpose vectorize
   for free (1.19 vs 1.22 GB/s serial; ~10 GB/s with a 16-thread pool,
   i.e. ~69k tok/s for an 8B model's 147 KB/token cache). Encode is
   0.18 GB/s serial, ~4.4 GB/s with a 32-thread pool.
5. **zstd level 19** reaches 10.63 bps (×1.51) but encodes ~100× slower
   (0.03 GB/s) — level 3 is the practical optimum. lz4 compresses worse
   than zstd here (13.03 bps) and offers no advantage worth the ratio.

Ratios are stationary in S (×1.43 at 1k → ×1.47 at 32k): the method keeps
working as the cache grows.

## Usage

### Lossless (default codec)

```python
import torch
from kv_compression import bf16_bits, bits_bf16, encode, decode, BEST_METHOD

# k: bf16 torch [B, H, S, D] (layer's K or V, one cache tensor)
bits = bf16_bits(k[0])                      # uint16 [H, S, D]
streams = encode(bits, BEST_METHOD, level=3)  # [lo_plane, hi_plane] compressed bytes

# ... store / transfer ...

back = decode(streams, bits.shape, BEST_METHOD)
k_restored = bits_bf16(back, k[0]).unsqueeze(0)   # bit-exact
```

Compare all lossless methods (with bit-exact verification):

```python
from kv_compression import evaluate_tensor
report = evaluate_tensor(bits)   # {"chan+planes": {"bps": ..., "ok": True}, ...}
```

### Cross-request dedup (shared-prefix block store)

`kv_compression.dedup` deduplicates whole batches of requests before
compression. KV columns of two requests are bit-equal iff their token
prefixes from position 0 are equal (layer >= 1 K/V are contextualized,
and K carries RoPE position) — so dedup is *prefix-only*: the
storage-side analogue of prefix caching (vLLM APC / RadixAttention),
not general content dedup.

```python
from kv_compression import Request, encode_batch, decode_batch, batch_report

reqs = [Request(token_ids, {f"{l}.{kv}": bf16_bits(cache[l][kv][0])
                             for l in layers for kv in "kv"})
        for ...]                     # same tensor names for every request
streams = encode_batch(reqs, block_tokens=64)   # unique blocks + per-request refs
back = decode_batch(streams)                    # bit-exact per request
print(batch_report(streams))  # {"dedup_frac": ..., "bps_vs_raw": ..., ...}
```

Design points (all measured-in-the-design, not tunables):

1. **Keys are a chained token-id hash** (`key_i = sha256(key_{i-1} ||
   tokens[iB:(i+1)B])`), *not* a hash of KV bytes: recomputing the same
   prefix in a different batch shape can yield non-bit-exact KV (GEMM
   reduction order); token-keying still hits, and first-writer-wins
   keeps decode bit-exact. A stored 64-bit content checksum only flags
   true collisions (`on_collision="ignore"|"warn"|"error"`).
2. **Unique blocks are compressed independently** (default
   `chan+planes`), so one block fetches in O(block) — trading a little
   ratio for random access vs one concatenated stream.
3. **Partial tails are their own identity**: a 36-token tail and a
   64-token block over the same start never cross-dedup.
4. `BlockStore` is the serving-side object: `put_request` compresses
   and stores only missing blocks, `get_request` reassembles (None if
   any block misses), `missing_keys` drives partial fetch, and
   `BlockStore.from_archive` reloads a persisted batch.
5. **Dedup is lossless-layer only**: tiered importance is per-request
   (each suffix attends the shared prefix differently), so a shared
   compressed block would force a shared tier decision.

Scale: 4k shared prefix + 1k private suffix, 5 requests, Qwen3-8B
(147 KB/token) is ~3.6 GB -> ~1.3 GB; a multi-turn conversation stores
only its new tokens per round.

### Transmission (internet + data center, one stack)

`kv_compression.transport` picks the codec operating point per transfer
— the measured rate–distortion table becomes a network rate controller:

```python
from kv_compression import choose_plan

plan = choose_plan(8192, bandwidth_bits_per_s=50e6, deadline_s=10.0, rtt_s=0.1)
# internet:  -> tiered/aggressive (0.38 bps), feasible, slack 5.3 s
# 100 Gbps   -> exact lossless pipelined; tight SLO -> tiered/quality
```

Serial compression is a net loss above ~35 Gbps (encoder 4.4 GB/s < line
rate), so the model also prices the pipelined regime (wire time + one
block of codec latency, blocks streaming out during prefill).

`kv_compression.wire` is the streamable side of the same block structure:

* **frames** — one self-contained record per (block, tensor); resumable
  at frame granularity, orderable block-major (internet: dedup/
  resume-friendly) or name-major (datacenter: layer-pipelined with
  prefill);
* **receiver state + reconcile (KV rsync)** — `WireReceiver` keeps a
  far-side `BlockStore`; the sender asks `have_keys()` and streams only
  missing blocks, so a second request sharing a prefix (or a multi-turn
  round) transfers only its new tokens;
* **manifest** declares per-request block refs, so the receiver knows
  exactly when each request is whole (`ready_requests()`).

```python
from kv_compression import Request, WireReceiver, build_manifest, stream_frames

manifest, store = build_manifest(requests, block_tokens=64)   # sender
rx = WireReceiver(); rx.accept_manifest(manifest)             # receiver
for frame in stream_frames(store, receiver_keys=rx.have_keys()):
    ...                                                        # wire
    rx.accept_frame(frame)
tensors = rx.take_request(0, token_ids)                       # bit-exact
```

`PROFILES` holds regime starting points (internet: 64-token blocks,
block-major, reconcile; datacenter: 256-token blocks, name-major,
pipelined). Dropped-mid-transfer frames re-arrive idempotently
(first-writer-wins, dup-ignored); `verify=True` decompresses and
re-hashes every frame against its checksum.

### KVConnector extension (no vLLM dependency)

`kv_compression.connector` mirrors the *shape* of vLLM's
`KVConnectorBase_V1` hooks as a standalone extension point — it never
imports vLLM; an engine adapter (vLLM, SGLang, custom) just binds
methods. Transport is the `KVLink` port (reliable ordered messages:
TCP/QUIC-stream shaped); `MemoryLink` is the in-process pair for tests
and simulation, and the exchange is pump-driven so it drops into any
event loop:

```python
from kv_compression import KVConnector, MemoryLink, Request, pump_until

a, b = MemoryLink.pair()
tx = KVConnector(a, profile="datacenter", bandwidth_bits_per_s=100e9, deadline_s=0.2)
rx = KVConnector(b, profile="datacenter")

tx.save(Request(token_ids, tensors))   # prefill side, after prefill
tx.flush()                              # chooses the codec point (choose_plan)
pump_until(tx, rx)                      # MANIFEST -> HAVE -> FRAME* -> READY

hit = rx.lookup(token_ids)              # external hits before committing
got = rx.load(0, token_ids)             # bit-exact tensors
```

Receiver and sender keep their block stores across transfers, so a
second flush sharing a prefix transmits only missing blocks (the
receiver's HAVE reply is the reconcile step); after a dropped
connection, re-`save` + `flush` resumes incrementally. The operating
point per flush comes from `choose_plan`; when the plan wants a lossy
tier not yet wire-implemented, `last_plan["clamped"]` records the
downgrade to the best exact codec. Senders keep a persistent store per
codec point, so warm flushes re-encode nothing; `TCPLink` is the socket
transport (Nagle off, pump-driven non-blocking I/O).

`benchmarks/bench_wire_link.py` compares raw page transfer
(vLLM-connector representation: bytes on wire = cache bytes) against
this wire on a real TCP loopback link — cold and dedup-warm — with
projected times at 50 Mbps / 25 Gbps from the measured wire bytes.

### Lossy (optional codecs)

Uniform per-channel quantization — no model signal needed:

```python
from kv_compression.lossy import bits_to_f32, f32_to_bits, encode_quant, decode_quant

x = bits_to_f32(bf16_bits(k[0]))            # [H, S, D] float32 (exact)
streams = encode_quant(x, bits=6, group="chan")
xq = decode_quant(streams)                  # float32, ~5.2 bps, KL 0.013
k_q = bits_bf16(f32_to_bits(xq), k[0])      # back to bf16 torch
```

Importance-tiered quantization + eviction — the best measured rate–distortion
(requires a per-token importance score, e.g. cumulative attention mass over
an observation window, SnapKV-style; the caller computes it from the model):

```python
from kv_compression.lossy import encode_tiered, decode_tiered

streams = encode_tiered(x, token_importance, tiers="balanced")
x_kept, kept_idx = decode_tiered(streams)   # [H, S_kept, D] + original positions
# column i of x_kept is original token kept_idx[i]; scatter back into a
# holed cache (kept_idx is grouped by tier, not globally sorted)
```

Tier maps (cumulative fraction by importance rank → bits, 0 = evict):

| name        | map                                   | measured (book) |
|-------------|---------------------------------------|-----------------|
| aggressive  | 1%→8b, 10%→4b, rest evicted           | 0.38 bps (×42), KL 0.044 |
| balanced    | 5%→8b, 25%→6b, rest evicted           | 1.28 bps (×12.5), KL 0.02 |
| quality     | 10%→8b, 50%→6b, rest 4b               | 4.10 bps (×3.9), KL 0.002 |
| L-shape     | 1%→8b, 10%→6b, 25%→4b, 50%→2b, rest evicted | 1.39 bps, KL 0.06 |

Both report rate+RMSE together:

```python
from kv_compression.lossy import evaluate_tensor_lossy
report = evaluate_tensor_lossy(x, token_importance)
# {"quant4b-chan": {"bps":..., "rel_rmse":...}, "tiered/balanced": {"bps":..., "keep":..., "rel_rmse":...}, ...}
```

### Throughput

```python
from kv_compression.bench import bench_tensors
rows = bench_tensors([("0.k", bits_k), ("0.v", bits_v)])
```

## Lossy: measured rate–distortion (Qwen3-8B, book text)

| method                      | rate bps | compression | KL    | ΔNLL   |
|-----------------------------|----------|-------------|-------|--------|
| lossless chan+planes        | 11.13    | ×1.4        | 0     | 0      |
| tiered/quality              | 4.10     | ×3.9        | 0.002 | ≈0     |
| mixed per-layer (greedy)    | ~5.5     | ×2.9        | 0.010 | +0.005 |
| tiered/balanced (online)    | 1.28     | ×12.5       | 0.030 | -0.008 (e2e) |
| tiered/aggressive           | 0.38     | ×42         | 0.044 | +0.039 |
| uniform 6b chan             | 5.19     | ×3.1        | 0.013 | +0.021 |
| uniform 4b chan             | 3.50     | ×4.6        | 0.279 | +0.257 |
| uniform 3b chan             | 2.36     | ×6.8        | 1.111 | +1.008 |
| evict keep-10% (bf16)       | 1.60     | ×10         | 0.044 | —      |
| recency / random evict      | —        | —           | >1.2  | collapse |

Findings:

1. **Attention mass is extremely concentrated** (importance entropy
   ~5.3 of 12.8 bits at P=7168): tiered eviction+quantization dominates
   everything else at low rate — 0.38 bps (×42) at KL 0.04.
2. **Per-channel (`chan`) granularity ≫ `head` ≫ `tensor`** at low bits
   (2-bit: KL 0.96 vs 5.8 vs 9.1). Channel scales along S are the right
   grouping for KV tensors.
3. **K is more sensitive than V** at low bits; mixed per-layer bit
   allocation gains ~2× rate at equal KL vs uniform.
4. **Importance must come from attention** — recency/random eviction
   collapse (KL > 1.2) even at keep-50%. Online importance (last W prompt
   tokens as queries) tracks oracle importance closely (0.085 vs 0.044 KL
   at 0.38 bps).
5. **VQ/PQ was a measured negative result** (see rill-sim `kv_vq.py`):
   codebook methods underperform simple per-channel quantization on KV.
6. End-to-end generation validation (`repl5`): tiered/balanced online at
   1.28 bps swaps in at 2.2 GB/s (≈14.7k tok/s) with ΔNLL ≈ 0.

## Install / test

```bash
uv sync --extra dev   # or: pip install -e '.[dev]'
pytest
```

Core codec needs only numpy + zstandard; torch is an optional extra used by
`bf16_bits`/`bits_bf16`.

## Layout

- `kv_compression/bits.py` — bf16 ↔ uint16 bit plumbing (bijective, NaN-safe),
  monotone key, zigzag, delta/xor transforms (reference methods).
- `kv_compression/codec.py` — per-method lossless `encode`/`decode`,
  `evaluate_tensor`, `delta_entropy_bits` (analytic floor).
- `kv_compression/lossy.py` — lossy codecs: uniform quantization
  (chan/head/tok/tensor granularity) and importance-tiered quantization +
  eviction, plus the bf16↔f32 bridge and rate/RMSE evaluation.
- `kv_compression/dedup.py` — cross-request prefix dedup: token-hash-chain
  block keys, `BlockStore`, batch archives (`encode_batch`/`decode_batch`/
  `batch_report`).
- `kv_compression/transport.py` — transmission planner: measured RD
  candidates vs bandwidth/deadline time model (serial + pipelined),
  `choose_plan`, internet/datacenter `PROFILES`.
- `kv_compression/wire.py` — streamable frames: `build_manifest`,
  `stream_frames` (block/name-major, reconcile), `WireReceiver`.
- `kv_compression/connector.py` — `KVConnector` extension endpoint
  (vLLM-shaped hooks, no vLLM import), `KVLink`/`MemoryLink`/`TCPLink`
  transport ports, pump-driven `MANIFEST→HAVE→FRAME*→READY` exchange,
  persistent sender stores.
- `benchmarks/bench_wire_link.py` — raw (vLLM-style) vs compressed wire
  on a real TCP link, cold + dedup-warm, with link-rate projections.
- `kv_compression/bench.py` — throughput benchmark.
- `tests/` — round-trip incl. special values, tier layout, bf16 bridge vs
  torch, behaviour, bench smoke, prefix dedup, planner + wire protocol,
  connector exchange (91 tests). The lossy core is verified bit-exact
  against the original rill-sim torch implementation.
