"""Lossless (bit-exact) KV-cache compression.

Extracted from `rill-sim/rill_sim/research/kv_compress.py` after a measured
method grid on Qwen3-8B (book + code text, S = 1024..32768). The winner is
``chan+planes``:

    1. reinterpret bf16 K/V as uint16 bit patterns (bijective, NaN-safe);
    2. transpose to channel-major (H, S, D) -> (H, D, S) so consecutive
       sequence positions of one channel are adjacent;
    3. split into low/high byte planes (mantissa noise vs exponent structure);
    4. compress each plane independently with zstd.

Measured bits/element (bf16 reference = 16.000), Qwen3-8B, zstd level 3:

    ==================  ===========  ===========  ========
    method              book @32k    code @32k    ratio
    ==================  ===========  ===========  ========
    raw (zstd)          12.229       12.204       x1.31
    chan                11.951       11.927       x1.34
    chan+delta          12.389       12.374       x1.29 (!)
    chan+delta+planes   11.243       11.235       x1.42
    chan+xor+planes     11.066       11.049       x1.45
    chan+planes         10.868       10.848       x1.47
    ------------------  -----------  -----------  --------
    delta_entropy       10.871       10.899       (floor)
    ==================  ===========  ===========  ========

``chan+planes`` sits at the analytic entropy floor of the delta symbols, so
no delta/xor representation can do meaningfully better with this bit layout.
Decode is as cheap as raw zstd (1.19 vs 1.22 GB/s serial; ~5-10 GB/s with a
thread pool), because plane merge + transpose vectorize for free.
"""

from kv_compression.bits import (
    bf16_bits,
    bits_bf16,
    delta_last,
    mono,
    undelta_last,
    unmono,
    unxor_last,
    unzigzag16,
    xor_last,
    zigzag16,
)
from kv_compression.codec import (
    BEST_METHOD,
    METHODS,
    decode,
    delta_entropy_bits,
    encode,
    evaluate_tensor,
)

__all__ = [
    "BEST_METHOD",
    "METHODS",
    "bf16_bits",
    "bits_bf16",
    "decode",
    "delta_entropy_bits",
    "delta_last",
    "encode",
    "evaluate_tensor",
    "mono",
    "undelta_last",
    "unmono",
    "unxor_last",
    "unzigzag16",
    "xor_last",
    "zigzag16",
]
