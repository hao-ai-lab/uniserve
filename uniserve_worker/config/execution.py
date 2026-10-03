"""Typed model-execution configuration.

Bootstrap resolves these stable execution settings before model materialization
and passes the immutable value into every configured subsystem.

The module also owns the default CUDA graph capture buckets and the storage
reservations for captured graphs: ``graph_padding_block_count`` derives the KV
pages that graph padding may occupy from the default buckets, and
``graph_storage_budget_bytes`` sizes the device share for retained graph
executables. Capacity planning in ``uniserve_worker.bootstrap.report``
reserves both before it sizes the request pool, and
``uniserve_worker.model_executor.graph_storage`` uses the graph budget as the
default for a device whose budget has not been set.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from uniserve.runtime.backends.attention.flashinfer import (
    Config as FlashInferConfig,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.call import (
    CALL_KINDS,
    CallKind,
    ForwardMode,
    MediaCall,
    TransferMode,
)

__all__ = [
    "DEFAULT_DECODE_GRAPH_BATCH_SIZES",
    "DEFAULT_GRAPH_STORAGE_FRACTION",
    "DEFAULT_PREFILL_GRAPH_ROW_BUCKETS",
    "DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS",
    "graph_padding_block_count",
    "graph_storage_budget_bytes",
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

# Token buckets step linearly within each range, with a step that widens as the
# counts grow. A prefill rounded up to a bucket gains less padding than the gap
# below that bucket.
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

# Row (request) buckets for captured prefill graphs. They are not
# configurable: ``ModelExecutor`` and bootstrap capacity planning read them
# directly.
DEFAULT_PREFILL_GRAPH_ROW_BUCKETS = (8, 16, 32)


def graph_padding_block_count(block_size: int) -> int:
    """Return the most KV pages that padding a captured graph input can add.

    A decode batch padded up to the next captured batch size gains at most one
    token per padding row, and a prefill padded up to the next token bucket
    gains at most the widest gap between buckets. The count is derived from
    the default buckets, not from the sizes a ``WorkerConfig`` configures.
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


# Captured graphs hold their storage for as long as they are retained, so the
# executable set owns a fixed share of the device rather than growing with the
# shape diversity a workload happens to present.
DEFAULT_GRAPH_STORAGE_FRACTION = 0.10


def graph_storage_budget_bytes(total_device_bytes: int) -> int:
    """Return the device-storage budget for retained graph executables."""
    return max(
        0,
        int(
            float(max(0, int(total_device_bytes)))
            * DEFAULT_GRAPH_STORAGE_FRACTION
        ),
    )


# JSON lane selectors (the ``domains`` of a lane descriptor) resolve to call
# kinds at startup. Execution binds concrete call kinds, so independent media
# calls never acquire a second scheduling classification. The engine's lane
# parser accepts the same three selector names.
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
    """Assigns call kinds, an SM budget and batch limits to one lane.

    A lane is a partition of one device's streaming multiprocessors: the
    executor creates one stream per lane from the lanes' ``sm_budget`` values,
    so a lane configuration requires a single physical device. The batch
    overrides cap the worker-wide batch limits for calls bound to this lane,
    and ``max_inflight`` sizes the lane stream's event slots; ``None`` keeps
    the worker-wide value. A lane partitions compute only: its calls share the
    worker-wide KV and latent pools, whose placement the engine scheduler
    owns, so a lane has no storage capacity of its own.
    """

    lane_id: str
    sm_budget: int
    call_kinds: tuple[CallKind, ...]
    max_batch_calls: int | None = None
    max_batch_tokens: int | None = None
    max_inflight: int | None = None

    def __post_init__(self) -> None:
        """Validate lane call kinds, the SM budget and the batch limits."""
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

    Also configures bounded runtime resources. ``worker_config_from_namespace``
    sets the launch fields; bootstrap later derives others with
    ``dataclasses.replace``. For example, model loading resizes the batch and
    request-slot bounds of a model with a ``VideoDecoder`` and clears its KV
    capacity, attention backend and generation device; capacity fitting on a
    CUDA device sets ``pool_storage_bytes`` from the device storage grant and
    may shrink the request-slot and batch bounds; a ``kv_cache_dtype`` in
    the quantization config overrides the launch value; and an unset
    ``block_size`` is resolved from the model's cache layers and attention
    kernels (``bootstrap.cache.resolve_page_size``) before the unit pool is
    planned.
    """

    device: str = "cpu"
    rank: int = 0
    world_size: int = 1
    # Tokens per KV page of the cache group with the widest token rows; None
    # until resolved.
    block_size: int | None = None
    kv_token_capacity: int | None = None
    attention_backend: str | None = None
    max_batch_calls: int = 1024
    max_batch_tokens: int = 8192
    max_sequence_tokens: int = 16384
    max_video_seconds: float = 15.0
    # Most denoiser rows a video request's conditions may take. It bounds the
    # condition products a video worker provisions; zero provisions none, as
    # a deployment serving text-to-video alone does.
    max_condition_rows: int = 0
    # The ffmpeg executable the media reader decodes reference videos with.
    ffmpeg: str = "ffmpeg"
    # Shortest duration the server admits, which bounds the frame counts the
    # worker provisions from below; ``None`` provisions every frame count
    # the model generates up to the capacity.
    min_video_seconds: float | None = None
    # Text capacities, in prompt tokens, of the denoiser's layouts; empty
    # selects ``MediaBuilder``'s default spacing.
    video_text_capacities: tuple[int, ...] = ()
    # Canvases, as (height, width) pixels, a video deployment prepares and
    # admits; empty prepares the denoiser's own (``VideoDenoiser.canvases``).
    video_frame_sizes: tuple[tuple[int, int], ...] = ()
    # Names of every component any worker group of the deployment places;
    # it selects which of a checkpoint's video denoisers the deployment
    # serves.
    deployment_components: tuple[str, ...] = ()
    max_request_pool_size: int = 128
    encoder_cache_entries: int = 256
    generation_device: str | None = None
    min_request_pool_size: int = 1
    pool_storage_bytes: int | None = None
    model_dtype: str = "bfloat16"
    kv_cache_dtype: str | None = None
    kv_storage_fraction: float = 0.70
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
    flashinfer: FlashInferConfig = FlashInferConfig()

    def __post_init__(self) -> None:
        """Validate topology, device identity, batch bounds and dtypes.

        Raises:
            WorkerError: With the invalid-descriptor code, naming the first
                violated constraint.
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
        if self.pool_storage_bytes is not None and self.pool_storage_bytes < 0:
            raise invalid_descriptor(
                "worker pool storage grant must not be negative"
            )
        if (
            (self.block_size is not None and self.block_size < 1)
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
        if self.max_condition_rows < 0:
            raise invalid_descriptor(
                "video condition capacity must not be negative"
            )
        if not self.ffmpeg:
            raise invalid_descriptor("the media reader needs an ffmpeg path")
        if self.min_video_seconds is not None and not (
            math.isfinite(self.min_video_seconds)
            and 0 < self.min_video_seconds <= self.max_video_seconds
        ):
            raise invalid_descriptor(
                "shortest video duration must lie within the capacity"
            )
        if any(value < 1 for value in self.video_text_capacities):
            raise invalid_descriptor("video text capacities must be positive")
        if not 0 < self.kv_storage_fraction <= 1:
            raise invalid_descriptor(
                "worker configuration KV storage fraction must be in (0, 1]"
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

    Args:
        namespace: Parsed launch descriptor fields.
        device: The rank device, already normalized by the caller.
        generation_device: The separate generation-tower device, if any.

    Returns:
        The launch-time configuration; fields the descriptor does not carry
        keep their ``WorkerConfig`` defaults.

    Raises:
        ValueError: If a descriptor value fails to parse or is out of range.
        WorkerError: If the assembled configuration violates a
            ``WorkerConfig`` invariant.
    """
    return WorkerConfig(
        device=device,
        rank=int(namespace.rank),
        world_size=int(namespace.world_size),
        generation_device=generation_device,
        block_size=_positive_optional_int(namespace.block_size),
        max_batch_calls=int(namespace.max_batch_calls),
        max_batch_tokens=int(namespace.max_batch_tokens),
        max_sequence_tokens=int(namespace.max_model_len),
        max_video_seconds=float(namespace.max_video_seconds),
        max_condition_rows=int(namespace.max_condition_rows),
        ffmpeg=str(namespace.ffmpeg),
        min_video_seconds=_optional_float(
            getattr(namespace, "min_video_seconds", None)
        ),
        video_text_capacities=_parse_positive_int_csv(
            getattr(namespace, "video_text_capacities", None), default=()
        ),
        video_frame_sizes=_parse_image_shapes(
            getattr(namespace, "video_frame_sizes", None),
            default=(),
            name="video frame sizes",
        ),
        deployment_components=tuple(
            str(name) for name in namespace.deployment_components
        ),
        kv_token_capacity=_positive_optional_int(namespace.kv_token_capacity),
        attention_backend=str(namespace.attention_backend),
        model_dtype=str(namespace.model_dtype),
        kv_cache_dtype=_none_if_empty(namespace.kv_cache_dtype),
        kv_storage_fraction=_storage_fraction(
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
        flashinfer=FlashInferConfig(
            workspace_size=int(namespace.flashinfer_workspace_size),
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
    """Strip an optional string setting, keeping ``None`` as unset.

    Raises:
        ValueError: If the value is present but blank.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(
            "an explicitly provided string setting must not be empty"
        )
    return text


def _optional_float(value: object | None) -> float | None:
    """Parse an optional floating-point setting, keeping ``None`` as unset."""
    return None if value is None else float(cast(Any, value))


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


def _storage_fraction(value: float, name: str) -> float:
    """Validate a per-process device storage fraction.

    The fraction is a share of each device's total storage, which
    ``device_storage_budget`` also bounds by the storage free at startup, so
    any finite value in (0, 1] is meaningful; one grants the whole device.
    The engine checks the same range before it launches a rank.

    Raises:
        ValueError: If the value is not finite or lies outside (0, 1].
    """
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value


def _parse_positive_int_csv(
    raw: object | None, *, default: tuple[int, ...]
) -> tuple[int, ...]:
    """Parse a comma-separated sequence of positive integer bucket sizes.

    ``None`` selects ``default``. A present value must be strictly increasing.
    """
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


def _parse_image_shapes(
    raw: object | None,
    *,
    default: tuple[tuple[int, int], ...] = ((1152, 2048), (2048, 1152)),
    name: str = "flow graph shapes",
) -> tuple[tuple[int, int], ...]:
    """Parse comma-separated ``HEIGHTxWIDTH`` image shapes, in pixels.

    The shapes must be unique and positive; ``None`` selects ``default``.
    ``name`` labels the option in errors.
    """
    if raw is None:
        return default

    values: list[tuple[int, int]] = []
    for item in str(raw).split(","):
        height_text, separator, width_text = item.strip().lower().partition("x")
        if not separator:
            raise ValueError(f"{name} must use HEIGHTxWIDTH")
        shape = int(height_text), int(width_text)
        if min(shape) < 1 or shape in values:
            raise ValueError(f"{name} must be positive and unique")
        values.append(shape)

    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(values)


def _parse_lanes(raw: object | None) -> tuple[LaneConfig, ...]:
    """Resolve JSON lane descriptors into lane configurations.

    Each item is one JSON object string whose ``domains`` list names selectors
    from ``LANE_COMPUTATION_GROUPS``; the selectors expand to concrete call
    kinds.

    Raises:
        ValueError: If a descriptor is malformed, a lane id repeats, or a call
            kind would bind to more than one lane.
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
