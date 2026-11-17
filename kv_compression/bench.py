"""Throughput benchmark: encode/decode over a set of captured cache tensors.

``tensors``: ``(name, uint16 [H, S, D])`` per (layer, tensor).
``configs``: ``(method, zstd_level, threads)``; threads>0 enables
multithreaded zstd AND a thread-pool decode across tensors.
Returns one record per config: best-of-``repeats`` throughputs in GB/s
(uncompressed bf16 bytes basis), plus ratio and bit-exactness.

GPU->CPU copy is NOT included (arrays are pre-captured via
:func:`kv_compression.bf16_bits`).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from kv_compression.codec import decode, encode

# Measured reference points (Qwen3-8B cache, S=8192, 36 layers x (K, V)):
#   chan+planes zstd-3 serial:      enc 0.18 GB/s, dec 1.19 GB/s
#   chan+planes zstd-3 pool(16):    enc 0.18 GB/s, dec 10.2 GB/s
#   raw        zstd-3 serial:       enc 0.61 GB/s, dec 1.22 GB/s
#   zstd-19 (g16 pool32):           10.63 bps but enc 0.03 GB/s (~100x slower)
DEFAULT_CONFIGS = [
    (m, 3, 1) for m in
    ("raw", "chan", "chan+planes", "chan+delta", "chan+delta+planes", "chan+xor+planes")
] + [
    ("chan+planes", 3, 8),
    ("chan+planes", 3, 16),
    ("raw", 3, 8),
]


def bench_tensors(
    tensors: list[tuple[str, np.ndarray]],
    configs: Optional[list[tuple[str, int, int]]] = None,
    repeats: int = 3,
) -> list[dict]:
    if configs is None:
        configs = DEFAULT_CONFIGS
    total_bytes = sum(t.nbytes for _, t in tensors)
    out = []
    for method, level, threads in configs:
        best_enc = best_dec = float("inf")
        streams_all: list[list[bytes]] = []
        back: list[np.ndarray] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            streams_all = [encode(t, method, level, threads) for _, t in tensors]
            best_enc = min(best_enc, time.perf_counter() - t0)
        for _ in range(repeats):
            t0 = time.perf_counter()
            if threads > 1:
                with ThreadPoolExecutor(max_workers=threads) as ex:
                    back = list(ex.map(lambda p: decode(p[0], p[1], method),
                                       zip(streams_all, [t.shape for _, t in tensors])))
            else:
                back = [decode(s, t.shape, method) for s, (_, t) in zip(streams_all, tensors)]
            best_dec = min(best_dec, time.perf_counter() - t0)
        ok = all(np.array_equal(b, t) for b, (_, t) in zip(back, tensors))
        cbytes = sum(len(x) for s in streams_all for x in s)
        out.append({
            "method": method, "level": level, "threads": threads, "ok": ok,
            "enc_gbps": total_bytes / best_enc / 1e9,
            "dec_gbps": total_bytes / best_dec / 1e9,
            "bps": 8 * cbytes / (total_bytes // 2),
        })
    return out
