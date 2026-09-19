"""Typed model-execution configuration.

Bootstrap resolves these stable execution settings before model materialization
and passes the immutable value into every configured subsystem.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from uniserve.runtime.backends.attention.flashinfer import (
    Config as FlashInferConfig,
)
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.protocol.call import (
    CALL_KINDS,
    CallKind,
    ForwardMode,
    MediaCall,
    TransferMode,
)

__all__ = [
    "DEFAULT_DECODE_GRAPH_BATCH_SIZES",
    "DEFAULT_GRAPH_MEMORY_FRACTION",
    "DEFAULT_PREFILL_GRAPH_ROW_BUCKETS",
    "DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS",
    "graph_padding_block_count",
    "graph_memory_budget_bytes",
    "LaneConfig",
    "WorkerConfig",
    "worker_config_from_namespace",
]


# Dense small batches, then strides of 8 up to the largest captured batch.
DEFAULT_DECODE_GRAPH_BATCH_SIZES = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
)

# Token buckets grow geometrically so the captured-graph count stays bounded
# while padding waste stays proportional to the bucket size.
DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS = (
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    4608,
    5120,
    5632,
    6144,
    6656,
    7168,
    7680,
    8192,
    8704,
    9216,
    9728,
    10240,
    10752,
    11264,
    11776,
    12288,
    12800,
    13312,
    13824,
    14336,
    14848,
    15360,
    15872,
    16384,
)

DEFAULT_PREFILL_GRAPH_ROW_BUCKETS = (8, 16, 32)


def graph_padding_block_count(block_size: int) -> int:
    """Return the maximum additional KV pages.

    The additional pages are required to pad a decode graph batch.
    """
    block_size = max(1, int(block_size))
    decode_tokens = max(DEFAULT_DECODE_GRAPH_BATCH_SIZES) - 1

    # The largest gap between consecutive prefill buckets is the most padding a
    # rounded-up prefill graph ever appends to a real token count.
    prefill_tokens = max(
        current - previous
        for previous, current in zip(
            (0, *DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS[:-1]),
            DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
            strict=True,
        )
    )
    max_padding_tokens = max(decode_tokens, prefill_tokens)
    return (max_padding_tokens + block_size - 1) // block_size


# Captured graphs hold their memory for as long as they are retained, so the
# executable set owns a fixed share of the device rather than growing with the
# shape diversity a workload happens to present.
DEFAULT_GRAPH_MEMORY_FRACTION = 0.10


def graph_memory_budget_bytes(total_device_bytes: int) -> int:
    """Return the device-memory budget for retained graph executables."""
    return max(
        0,
        int(
            float(max(0, int(total_device_bytes)))
            * DEFAULT_GRAPH_MEMORY_FRACTION
        ),
    )


# JSON lane selectors resolve at startup. Execution binds concrete
# call kinds, so independent media calls never acquire a second
# scheduling classification.
LANE_COMPUTATION_GROUPS: dict[str, tuple[CallKind, ...]] = {
    "prefill": (
        ForwardMode.PREFILL,
        MediaCall.VISION_ENCODING,
        MediaCall.LATENT_ENCODING,
        MediaCall.TEXT_ENCODING,
        *TransferMode,
    ),
    "decode": (ForwardMode.DECODE, ForwardMode.VERIFY),
    "flow": (
        MediaCall.LATENT_PREPARATION,
        MediaCall.DENOISING,
        MediaCall.IMAGE_DECODING,
        MediaCall.VIDEO_DECODING,
        MediaCall.AUDIO_DECODING,
        MediaCall.VIDEO_ENCODING,
        MediaCall.AUDIO_ENCODING,
        MediaCall.MUXING,
    ),
}


@dataclass(frozen=True, slots=True)
class LaneConfig:
    """Assigns call_kinds and SM budget to one execution lane.

    Also assigns optional capacity overrides to the lane.
    """

    lane_id: str
    sm_budget: int
    call_kinds: tuple[CallKind, ...]
    kv_capacity_tokens: int | None = None
    latent_capacity_units: int | None = None
    max_batch_calls: int | None = None
    max_batch_tokens: int | None = None
    max_inflight: int | None = None

    def __post_init__(self) -> None:
        """Validate lane call kinds and validate SM and capacity overrides."""
        if not self.lane_id or any(
            character.isspace() for character in self.lane_id
        ):
            raise ValueError("lane id must be a non-empty token")
        if int(self.sm_budget) < 1:
            raise ValueError("lane SM budget must be positive")
        if not self.call_kinds or len(set(self.call_kinds)) != len(
            self.call_kinds
        ):
            raise ValueError("lane call kinds must be non-empty and unique")
        if any(kind not in CALL_KINDS for kind in self.call_kinds):
            raise ValueError("lane must bind concrete call kinds")
        for name in (
            "kv_capacity_tokens",
            "latent_capacity_units",
            "max_batch_calls",
            "max_batch_tokens",
            "max_inflight",
        ):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(
                    f"lane {name} must be positive when configured"
                )


@dataclass(frozen=True)
class WorkerConfig:
    """Canonical rank configuration for model execution.

    Also configures bounded runtime resources.
    """

    device: str = "cpu"
    rank: int = 0
    world_size: int = 1
    block_size: int = 64
    kv_token_capacity: int | None = None
    attention_backend: str | None = None
    max_batch_calls: int = 1024
    max_batch_tokens: int = 8192
    max_sequence_tokens: int = 16384
    max_video_seconds: float = 15.0
    max_request_pool_size: int = 128
    encoder_cache_entries: int = 256
    generation_device: str | None = None
    min_request_pool_size: int = 1
    pool_memory_bytes: int | None = None
    model_dtype: str = "bfloat16"
    kv_cache_dtype: str | None = None
    kv_memory_fraction: float = 0.70
    lanes: tuple[LaneConfig, ...] = ()
    graph_policy: str = "auto"
    decode_graph_batch_sizes: tuple[int, ...] = DEFAULT_DECODE_GRAPH_BATCH_SIZES
    prefill_cuda_graph: bool = False
    prefill_graph_token_sizes: tuple[int, ...] = (
        DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS
    )
    flow_graph_batch_sizes: tuple[int, ...] = (1, 2, 3, 4)
    flow_graph_shapes: tuple[tuple[int, int], ...] = (
        (1152, 2048),
        (2048, 1152),
    )
    # Video request shapes, as (duration in seconds, prompt tokens), whose
    # denoising ladders warmup makes resident on every request slot. A shape a
    # deployment does not declare still serves; its first request captures.
    video_graph_shapes: tuple[tuple[float, int], ...] = ()
    flashinfer: FlashInferConfig = FlashInferConfig()

    def __post_init__(self) -> None:
        """Validate topology axes and device identity.

        Also validate batch bounds and dtype policies.
        """
        if self.graph_policy not in {"off", "auto", "full"}:
            raise invalid_descriptor("graph policy must be off, auto, or full")
        if not self.device:
            raise invalid_descriptor("worker device must be named")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise invalid_descriptor(
                "worker configuration process rank is invalid"
            )
        if not 1 <= self.min_request_pool_size <= self.max_request_pool_size:
            raise invalid_descriptor("worker request slot bounds are invalid")
        if self.pool_memory_bytes is not None and self.pool_memory_bytes < 0:
            raise invalid_descriptor(
                "worker pool memory grant must not be negative"
            )
        if (
            self.block_size < 1
            or self.max_batch_calls < 1
            or self.max_batch_tokens < 1
            or self.max_sequence_tokens < 1
            or self.max_request_pool_size < 1
            or self.encoder_cache_entries < 1
        ):
            raise invalid_descriptor(
                "worker configuration capacities must be positive"
            )
        if (
            not math.isfinite(self.max_video_seconds)
            or self.max_video_seconds <= 0
        ):
            raise invalid_descriptor(
                "video duration capacity must be finite and positive"
            )
        if not 0 < self.kv_memory_fraction <= 1:
            raise invalid_descriptor(
                "worker configuration KV memory fraction must be in (0, 1]"
            )
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise invalid_descriptor("worker model dtype is unsupported")
        if self.kv_cache_dtype is not None and self.kv_cache_dtype not in {
            "float16",
            "bfloat16",
            "float32",
            "float8_e4m3fn",
        }:
            raise invalid_descriptor("worker KV dtype is unsupported")


def worker_config_from_namespace(
    namespace: Any, *, device: str, generation_device: str | None
) -> WorkerConfig:
    """Resolve the rank execution and resource configuration.

    The complete configuration is resolved from parsed CLI values.
    """
    return WorkerConfig(
        device=device,
        rank=int(namespace.rank),
        world_size=int(namespace.world_size),
        generation_device=generation_device,
        block_size=int(namespace.block_size),
        max_batch_calls=int(namespace.max_batch_calls),
        max_batch_tokens=int(namespace.max_batch_tokens),
        max_sequence_tokens=int(namespace.max_model_len),
        max_video_seconds=float(namespace.max_video_seconds),
        kv_token_capacity=_positive_optional_int(namespace.kv_token_capacity),
        attention_backend=str(namespace.attention_backend),
        model_dtype=str(namespace.model_dtype),
        kv_cache_dtype=_none_if_empty(namespace.kv_cache_dtype),
        kv_memory_fraction=_bounded_fraction(
            float(namespace.kv_memory_fraction),
            "kv-memory-fraction",
        ),
        lanes=_parse_lanes(getattr(namespace, "lane", ())),
        graph_policy=getattr(namespace, "graph_policy", "auto"),
        decode_graph_batch_sizes=_parse_positive_int_csv(
            namespace.decode_graph_batch_sizes,
            default=DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        ),
        prefill_cuda_graph=bool(namespace.prefill_cuda_graph),
        prefill_graph_token_sizes=_parse_positive_int_csv(
            namespace.prefill_graph_token_sizes,
            default=DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        ),
        flow_graph_batch_sizes=_parse_positive_int_csv(
            namespace.flow_graph_batch_sizes,
            default=(1, 2, 3, 4),
        ),
        flow_graph_shapes=_parse_image_shapes(namespace.flow_graph_shapes),
        video_graph_shapes=_parse_video_shapes(namespace.video_graph_shapes),
        flashinfer=FlashInferConfig(
            workspace_size=max(
                1,
                int(namespace.flashinfer_workspace_size),
            ),
            use_tensor_core=_parse_optional_bool(
                namespace.flashinfer_use_tensor_core
            ),
            decode_backend=str(namespace.flashinfer_decode_backend),
            prefill_backend=str(namespace.flashinfer_prefill_backend),
            decode_split_tile_size=_positive_optional_int(
                namespace.flashinfer_decode_split_tile_size
            ),
            prefill_split_tile_size=_positive_optional_int(
                namespace.flashinfer_prefill_split_tile_size
            ),
            disable_split_kv=bool(namespace.flashinfer_disable_split_kv),
        ),
    )


def _none_if_empty(value: object | None) -> str | None:
    """Normalize empty optional configuration values to ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(
            "an explicitly provided string setting must not be empty"
        )
    return text


def _positive_optional_int(value: object | None) -> int | None:
    """Parse an optional positive integer capacity."""
    if value is None:
        return None
    parsed = int(cast(Any, value))
    if parsed <= 0:
        raise ValueError(
            "optional integer tuning values must be positive when set"
        )
    return parsed


def _bounded_fraction(value: float, name: str) -> float:
    """Validate a floating-point fraction.

    The fraction must lie within the closed unit interval.
    """
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be greater than 0 and less than 1")
    return value


def _parse_positive_int_csv(
    raw: object | None, *, default: tuple[int, ...]
) -> tuple[int, ...]:
    """Parse a comma-separated sequence of positive integer bucket sizes."""
    if raw is None:
        return default
    parts = tuple(part.strip() for part in str(raw).split(","))
    if not parts or any(not part for part in parts):
        raise ValueError(
            "integer bucket lists must contain only non-empty values"
        )

    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError(
            "integer bucket lists must contain only positive values"
        )
    if tuple(sorted(set(values))) != values:
        raise ValueError("integer bucket lists must be strictly increasing")
    return values


def _parse_image_shapes(raw: object | None) -> tuple[tuple[int, int], ...]:
    """Parse and validate image-height and image-width buckets.

    The buckets must be unique and positive.
    """
    if raw is None:
        return ((1152, 2048), (2048, 1152))

    values: list[tuple[int, int]] = []
    for item in str(raw).split(","):
        height_text, separator, width_text = item.strip().lower().partition("x")
        if not separator:
            raise ValueError("flow graph shapes must use HEIGHTxWIDTH")
        shape = int(height_text), int(width_text)
        if min(shape) < 1 or shape in values:
            raise ValueError("flow graph shapes must be positive and unique")
        values.append(shape)

    if not values:
        raise ValueError("flow graph shapes must not be empty")
    return tuple(values)


def _parse_video_shapes(raw: object | None) -> tuple[tuple[float, int], ...]:
    """Parse the video shapes whose denoising ladders warmup makes resident.

    Each item is ``SECONDSxTOKENS``: the request duration and its prompt length
    in tokenizer tokens. Both determine the denoiser's numerical size, so a
    declared shape only serves requests that match it exactly.
    """
    if raw is None:
        return ()

    values: list[tuple[float, int]] = []
    for item in str(raw).split(","):
        text = item.strip().lower()
        if not text:
            continue
        seconds_text, separator, tokens_text = text.partition("x")
        if not separator:
            raise ValueError("video graph shapes must use SECONDSxTOKENS")
        shape = float(seconds_text), int(tokens_text)
        if not math.isfinite(shape[0]) or shape[0] <= 0 or shape[1] < 1:
            raise ValueError(
                "video graph shapes must have a positive duration and prompt "
                "length"
            )
        if shape in values:
            raise ValueError("video graph shapes must be unique")
        values.append(shape)
    return tuple(values)


def _parse_lanes(raw: object | None) -> tuple[LaneConfig, ...]:
    """Normalize lane declarations.

    Declarations become unique identifiers and positive capacities.
    """
    result: list[LaneConfig] = []
    values = () if raw is None else raw
    if not isinstance(values, (list, tuple)):
        raise ValueError("lanes must be a sequence of JSON objects")

    for index, value in enumerate(values):
        try:
            data = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError(f"lane {index} is not valid JSON") from error
        if not isinstance(data, dict):
            raise ValueError(f"lane {index} must be a JSON object")

        known = {
            "lane_id",
            "sm_budget",
            "domains",
            "kv_capacity_tokens",
            "latent_capacity_units",
            "max_batch_calls",
            "max_batch_tokens",
            "max_inflight",
        }
        unknown = set(data).difference(known)
        if unknown:
            raise ValueError(
                f"lane {index} has unknown fields {sorted(unknown)!r}"
            )

        domains = data.get("domains")
        if not isinstance(domains, list):
            raise ValueError(f"lane {index}.domains must be a JSON list")
        if any(
            not isinstance(name, str) or name not in LANE_COMPUTATION_GROUPS
            for name in domains
        ):
            raise ValueError(
                f"lane {index}.domains must name prefill, decode, or flow"
            )

        result.append(
            LaneConfig(
                lane_id=str(data.get("lane_id", "")),
                sm_budget=int(data.get("sm_budget", 0)),
                call_kinds=tuple(
                    kind
                    for name in domains
                    for kind in LANE_COMPUTATION_GROUPS[name]
                ),
                kv_capacity_tokens=_json_optional_int(
                    data, "kv_capacity_tokens"
                ),
                latent_capacity_units=_json_optional_int(
                    data, "latent_capacity_units"
                ),
                max_batch_calls=_json_optional_int(data, "max_batch_calls"),
                max_batch_tokens=_json_optional_int(data, "max_batch_tokens"),
                max_inflight=_json_optional_int(data, "max_inflight"),
            )
        )

    # Every computation binds to exactly one lane, and every lane id is
    # distinct, so scheduling classification stays unambiguous.
    if len({lane.lane_id for lane in result}) != len(result):
        raise ValueError("lane ids must be unique")
    call_kinds = tuple(kind for lane in result for kind in lane.call_kinds)
    if len(set(call_kinds)) != len(call_kinds):
        raise ValueError("call kinds must have one execution lane binding")
    return tuple(result)


def _json_optional_int(data: dict[str, object], name: str) -> int | None:
    """Read an optional integer field from a JSON lane descriptor."""
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"lane {name} must be an integer")
    return int(value)


def _parse_optional_bool(value: object | None) -> bool | None:
    """Parse an optional boolean from native or textual configuration."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected optional bool token, got {value!r}")
