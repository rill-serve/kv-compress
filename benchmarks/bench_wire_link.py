#!/usr/bin/env python3
"""KV transfer on a real TCP link: raw pages (vLLM-connector style) vs
this package's wire (compressed frames + receiver-state dedup).

Baseline rationale: vLLM's built-in KV connectors (NIXL / SharedStorage)
move raw bf16 pages — bytes on wire = cache bytes, no compression, no
dedup at the transport.  ``raw`` below moves the identical tensors over
the identical loopback TCP link with length-prefixed sends, isolating
the representation-layer effect (compression + dedup + reconcile).

Scenarios (same tensors every time, 4 requests sharing a large prefix):
  raw      — raw bf16 bytes over TCP, both ends just move memory
  ours     — KVConnector over TCPLink, cold receiver (full transfer)
  ours+wdm — warm receiver (shared prefix already present: reconcile)
             and warm sender (persistent store: no re-encode)

Timeline split for ``ours``: encode (flush) | wire (pump to READY) |
decode (load).  Run from the repo root; numpy/zstandard only.
"""

from __future__ import annotations

import argparse
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from kv_compression import Request
from kv_compression.codec import BEST_METHOD
from kv_compression.connector import KVConnector, TCPLink, pump_until


def correlated_bits(h, s, d, seed):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(h, 1, d)).astype(np.float32)
         + np.cumsum(rng.normal(scale=0.01, size=(h, s, d)), axis=1).astype(np.float32))
    b = x.view(np.uint32)
    return (((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000) >> 16).astype(np.uint16)


def make_requests(n_req, prefix, suffix, h, d, seed0=0):
    rng = np.random.default_rng(seed0)
    prefix_ids = rng.integers(0, 150_000, size=prefix)
    pref = {n: correlated_bits(h, prefix, d, seed0 + 10 + i) for i, n in enumerate(("0.k", "0.v"))}
    reqs, suffixes = [], []
    for i in range(n_req):
        ids = np.concatenate([prefix_ids, rng.integers(0, 150_000, size=suffix)])
        t = {n: np.concatenate([pref[n], correlated_bits(h, suffix, d, seed0 + 100 * (i + 1) + j)],
                               axis=1) for j, n in enumerate(pref)}
        reqs.append(Request(ids, t))
        suffixes.append(t)
    return reqs


def bench_raw(tensors, repeats):
    """Raw bf16 over loopback TCP (vLLM-connector representation)."""
    blob = b"".join(np.ascontiguousarray(t).tobytes()
                    for r in tensors for t in r.tensors.values())
    times = []
    for _ in range(repeats):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        addr = srv.getsockname()
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect(addr)
        conn, _ = srv.accept()
        for s in (c, conn):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        got = threading.Event()
        state = {"n": 0}
        rx_thread = threading.Thread(target=lambda: _drain_sock(conn, len(blob), got, state))
        rx_thread.start()
        t0 = time.perf_counter()
        _send_sock(c, blob)
        got.wait()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        rx_thread.join()
        c.close(); conn.close(); srv.close()
    return len(blob), times


def _send_sock(s, blob):
    mv = memoryview(blob)
    while mv:
        n = s.send(mv[:1 << 20])
        mv = mv[n:]


def _drain_sock(s, total, done, state):
    while state["n"] < total:
        d = s.recv(1 << 20)
        if not d:
            break
        state["n"] += len(d)
    done.set()


def bench_ours(reqs, repeats, block_tokens, warm=False):
    """KVConnector over TCPLink.  warm=True keeps tx/rx across a first
    full transfer, then measures the second (dedup'd) transfer."""
    enc, wire, dec, nbytes = [], [], [], []
    for _ in range(repeats):
        a, b = TCPLink.pair()
        tx = KVConnector(a, block_tokens=block_tokens, method=BEST_METHOD, level=3)
        rx = KVConnector(b, block_tokens=block_tokens)
        if warm:
            _exchange(tx, rx, reqs)          # cold round warms both ends
        t0 = time.perf_counter()
        before = tx.wire_bytes
        for r in reqs:
            tx.save(r)
        tx.flush()
        t1 = time.perf_counter()             # encode + manifest
        pump_until(tx, rx)
        t2 = time.perf_counter()             # wire + READY
        for i, r in enumerate(reqs):
            rx.load(i, r.token_ids)
        t3 = time.perf_counter()             # decode
        enc.append(t1 - t0); wire.append(t2 - t1); dec.append(t3 - t2)
        nbytes.append(tx.wire_bytes - before)
        a._sock.close(); b._sock.close()
    return enc, wire, dec, nbytes


def _exchange(tx, rx, reqs):
    for r in reqs:
        tx.save(r)
    tx.flush()
    pump_until(tx, rx)
    for i, r in enumerate(reqs):
        rx.load(i, r.token_ids)


def med(xs):
    return statistics.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-req", type=int, default=4)
    ap.add_argument("--prefix", type=int, default=15360)
    ap.add_argument("--suffix", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    reqs = make_requests(args.n_req, args.prefix, args.suffix, args.heads, args.dim)
    logical = sum(t.nbytes for r in reqs for t in r.tensors.values())
    print(f"requests={args.n_req} prefix={args.prefix} suffix={args.suffix} "
          f"HxD={args.heads}x{args.dim} block={args.block_tokens}")
    print(f"logical cache: {logical/1e6:.1f} MB (bf16)\n")

    raw_n, raw_t = bench_raw(reqs, args.repeats)
    e, w, d, nb = bench_ours(reqs, args.repeats, args.block_tokens)
    e2, w2, d2, nb2 = bench_ours(reqs, args.repeats, args.block_tokens, warm=True)

    def row(name, ms, wire_mb, logical_mb):
        return (f"{name:22s} {ms:9.1f} {wire_mb:9.1f} {logical_mb/wire_mb:8.2f}x "
                f"{logical_mb/1e3/(ms/1e3):7.2f}")

    logical_mb = logical / 1e6
    print(f"{'scenario':22s} {'time ms':>9s} {'wire MB':>9s} {'shrink':>8s} {'eff GB/s':>8s}")
    print(row("raw (vLLM-style)", med(raw_t) * 1e3, raw_n / 1e6, logical_mb))
    total_ours = med(e) + med(w) + med(d)
    print(f"{'ours cold':22s} {total_ours*1e3:9.1f} {med(nb)/1e6:9.1f} "
          f"{logical/med(nb):8.2f}x {logical_mb/1e3/total_ours:7.2f}")
    total_warm = med(e2) + med(w2) + med(d2)
    print(f"{'ours warm (dedup)':22s} {total_warm*1e3:9.1f} {med(nb2)/1e6:9.1f} "
          f"{logical/med(nb2):8.2f}x {logical_mb/1e3/total_warm:7.2f}")

    print(f"\ncold breakdown (median): encode {med(e)*1e3:.0f} ms | wire {med(w)*1e3:.0f} ms | "
          f"decode {med(d)*1e3:.0f} ms")
    print(f"warm breakdown (median): encode {med(e2)*1e3:.0f} ms | wire {med(w2)*1e3:.0f} ms | "
          f"decode {med(d2)*1e3:.0f} ms")

    bw = raw_n / med(raw_t)
    print(f"\nloopback reference: {bw/1e9:.2f} GB/s")
    wire_cold, wire_warm = med(nb), med(nb2)
    for label, bps in (("50 Mbps (internet)", 50e6 / 8), ("25 Gbps (DC)", 25e9 / 8)):
        t_raw = logical / bps
        t_cold = wire_cold / bps + med(e) + med(d)
        t_warm = wire_warm / bps + med(e2) + med(d2)
        print(f"projected @{label:18s}: raw {t_raw:8.2f} s | ours cold {t_cold:7.2f} s | "
              f"warm {t_warm:6.2f} s")


if __name__ == "__main__":
    main()
