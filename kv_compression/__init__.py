"""KV-cache compression: lossless (bit-exact) + optional lossy codecs.

The lossless codec (``chan+planes``, see README) was selected from a
measured method grid on Qwen3-8B (book + code text, S = 1024..32768). It
wins at ~×1.4–1.5 with decode as cheap as raw zstd.

Optional lossy codecs live in :mod:`kv_compression.lossy` (uniform
per-channel quantization; importance-tiered quantization + eviction,
the best measured rate–distortion: 0.38–4.1 bps at KL 0.002–0.04).

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

See README.md for the full measured tables, the lossy rate-distortion
results, and usage examples.
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
from kv_compression.lossy import (
    GROUPS,
    TIER_MAPS,
    bits_to_f32,
    decode_quant,
    decode_tiered,
    encode_quant,
    encode_tiered,
    evaluate_tensor_lossy,
    f32_to_bits,
    quantize,
    dequantize,
    rel_rmse,
    tier_assignment,
)
from kv_compression.dedup import (
    DEFAULT_BLOCK_TOKENS,
    BlockStore,
    Request,
    batch_report,
    block_keys,
    decode_batch,
    encode_batch,
    join_blocks,
    split_blocks,
)
from kv_compression.transport import (
    CANDIDATES,
    PROFILES,
    QWEN3_8B_ELEMS_PER_TOKEN,
    choose_plan,
)
from kv_compression.wire import (
    WireReceiver,
    build_manifest,
    decode_frame,
    encode_frame,
    stream_frames,
)
from kv_compression.connector import (
    KVConnector,
    KVLink,
    MemoryLink,
    pump_until,
)

__all__ = [
    "BEST_METHOD",
    "CANDIDATES",
    "DEFAULT_BLOCK_TOKENS",
    "GROUPS",
    "METHODS",
    "PROFILES",
    "QWEN3_8B_ELEMS_PER_TOKEN",
    "TIER_MAPS",
    "BlockStore",
    "KVConnector",
    "KVLink",
    "MemoryLink",
    "Request",
    "WireReceiver",
    "batch_report",
    "bf16_bits",
    "bits_bf16",
    "bits_to_f32",
    "block_keys",
    "build_manifest",
    "dequantize",
    "decode",
    "decode_batch",
    "decode_frame",
    "decode_quant",
    "decode_tiered",
    "delta_entropy_bits",
    "delta_last",
    "encode",
    "encode_quant",
    "encode_tiered",
    "encode_batch",
    "encode_frame",
    "choose_plan",
    "evaluate_tensor",
    "evaluate_tensor_lossy",
    "f32_to_bits",
    "join_blocks",
    "mono",
    "pump_until",
    "quantize",
    "rel_rmse",
    "split_blocks",
    "stream_frames",
    "tier_assignment",
    "undelta_last",
    "unmono",
    "unxor_last",
    "unzigzag16",
    "xor_last",
    "zigzag16",
]
