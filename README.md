# kv-compression

Lossless (bit-exact) KV-cache compression. Extracted from
`rill-sim/rill_sim/research/kv_compress.py` after a measured method grid on
Qwen3-8B (36L × 8 KV heads × d=128, book + code text, S = 1024..32768,
`results/compress2_{book,code}_8b.json`).

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
   (0.03 GB/s) — level 3 is the practical optimum. lz4 is *worse* than
   nothing interesting (13.03 bps).

Ratios are stationary in S (×1.43 at 1k → ×1.47 at 32k): the method keeps
working as the cache grows.

## Usage

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

Compare all methods (with bit-exact verification):

```python
from kv_compression import evaluate_tensor
report = evaluate_tensor(bits)   # {"chan+planes": {"bps": ..., "ok": True}, ...}
```

Throughput benchmark over a captured cache:

```python
from kv_compression.bench import bench_tensors
rows = bench_tensors([("0.k", bits_k), ("0.v", bits_v)])
```

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
- `kv_compression/codec.py` — per-method `encode`/`decode`, `evaluate_tensor`,
  `delta_entropy_bits` (analytic floor).
- `kv_compression/bench.py` — throughput benchmark.
- `tests/test_codec.py` — round-trip incl. special values, behaviour, bench smoke.
