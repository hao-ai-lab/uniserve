"""Typed model-execution configuration.

Bootstrap resolves stable execution settings before model materialization.
Deep execution modules read this immutable snapshot instead of ambient
environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from .env import DEFAULT_ATTENTION_BACKEND, DEFAULT_COMPILE_BACKEND

__all__ = [
    "DEFAULT_DECODE_GRAPH_BATCH_SIZES",
    "DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS",
    "decode_graph_padding_block_count",
    "FlashInferTuningConfig",
    "TorchCompileRuntimeConfig",
    "ExecutionConfig",
    "execution_config_from_namespace",
    "get_execution_config",
    "set_execution_config",
]


DEFAULT_DECODE_GRAPH_BATCH_SIZES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
)

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
    160,
    192,
    224,
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


def decode_graph_padding_block_count(block_size: int) -> int:
    block_size = max(1, int(block_size))
    max_padding_rows = max(DEFAULT_DECODE_GRAPH_BATCH_SIZES) - 1
    return (max_padding_rows + block_size - 1) // block_size


@dataclass(frozen=True)
class TorchCompileRuntimeConfig:
    enabled: bool = False
    backend: str = DEFAULT_COMPILE_BACKEND
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None


@dataclass(frozen=True)
class FlashInferTuningConfig:
    workspace_size: int = 512 * 1024 * 1024
    use_tensor_core: bool | None = None
    decode_backend: str = "fa2"
    prefill_backend: str = DEFAULT_ATTENTION_BACKEND
    decode_split_tile_size: int | None = None
    prefill_split_tile_size: int | None = None
    disable_split_kv: bool = False
    fast_decode_plan: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    model_dtype: str = "bfloat16"
    transformers_trust_remote_code: bool = False
    transformers_attn_implementation: str = "uniserve"
    allow_transformers_fallback: bool = False
    disabled_model_archs: tuple[str, ...] = ()
    strict_model_imports: bool = False
    kv_cache_dtype: str | None = None
    kv_memory_fraction: float = 0.70
    tp_backend: str | None = None
    tp_init_method: str | None = None
    mooncake_device: str = ""
    mooncake_protocol: str = "rdma"
    cuda_graph: bool = True
    cuda_graph_warmup: bool = True
    cuda_graph_warmup_batches: tuple[int, ...] = DEFAULT_DECODE_GRAPH_BATCH_SIZES
    prefill_cuda_graph: bool = False
    prefill_cuda_graph_warmup: bool = False
    prefill_cuda_graph_warmup_tokens: tuple[int, ...] = DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS
    mixed_text_max_tokens: int = 8192
    varlen_prefill: bool = True
    green_contexts: bool = False
    logits_processor_chunk_size: int = 0
    torch_compile: TorchCompileRuntimeConfig = TorchCompileRuntimeConfig()
    flashinfer: FlashInferTuningConfig = FlashInferTuningConfig()


_CURRENT_EXECUTION_CONFIG = ExecutionConfig()


def get_execution_config() -> ExecutionConfig:
    return _CURRENT_EXECUTION_CONFIG


def set_execution_config(config: ExecutionConfig) -> None:
    global _CURRENT_EXECUTION_CONFIG
    _CURRENT_EXECUTION_CONFIG = config


def execution_config_from_namespace(namespace: Any) -> ExecutionConfig:
    return ExecutionConfig(
        model_dtype=str(namespace.model_dtype),
        transformers_trust_remote_code=bool(namespace.transformers_trust_remote_code),
        transformers_attn_implementation=str(namespace.transformers_attn_implementation),
        allow_transformers_fallback=bool(namespace.allow_transformers_fallback),
        disabled_model_archs=tuple(str(value) for value in (namespace.disable_model_arch or ())),
        strict_model_imports=bool(namespace.strict_model_imports),
        kv_cache_dtype=_none_if_empty(namespace.kv_cache_dtype),
        kv_memory_fraction=_bounded_fraction(
            float(namespace.kv_memory_fraction),
            "kv-memory-fraction",
        ),
        tp_backend=_none_if_empty(namespace.tp_backend),
        tp_init_method=_none_if_empty(namespace.tp_init_method),
        mooncake_device=str(namespace.mooncake_device or ""),
        mooncake_protocol=str(namespace.mooncake_protocol or "rdma"),
        cuda_graph=bool(namespace.cuda_graph),
        cuda_graph_warmup=bool(namespace.cuda_graph_warmup),
        cuda_graph_warmup_batches=_parse_positive_int_csv(
            namespace.cuda_graph_warmup_batches,
            default=DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        ),
        prefill_cuda_graph=bool(namespace.prefill_cuda_graph),
        prefill_cuda_graph_warmup=bool(namespace.prefill_cuda_graph_warmup),
        prefill_cuda_graph_warmup_tokens=_parse_positive_int_csv(
            namespace.prefill_cuda_graph_warmup_tokens,
            default=DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        ),
        mixed_text_max_tokens=max(0, int(namespace.mixed_text_max_tokens)),
        varlen_prefill=bool(namespace.varlen_prefill),
        green_contexts=bool(namespace.green_contexts),
        logits_processor_chunk_size=max(
            0,
            int(namespace.logits_processor_chunk_size),
        ),
        torch_compile=TorchCompileRuntimeConfig(
            enabled=bool(namespace.torch_compile),
            backend=str(namespace.torch_compile_backend),
            mode=_none_if_empty(namespace.torch_compile_mode),
            fullgraph=bool(namespace.torch_compile_fullgraph),
            dynamic=_parse_optional_bool(namespace.torch_compile_dynamic),
        ),
        flashinfer=FlashInferTuningConfig(
            workspace_size=max(
                1,
                int(namespace.flashinfer_workspace_size),
            ),
            use_tensor_core=_parse_optional_bool(namespace.flashinfer_use_tensor_core),
            decode_backend=str(namespace.flashinfer_decode_backend),
            prefill_backend=str(namespace.flashinfer_prefill_backend),
            decode_split_tile_size=_positive_optional_int(
                namespace.flashinfer_decode_split_tile_size
            ),
            prefill_split_tile_size=_positive_optional_int(
                namespace.flashinfer_prefill_split_tile_size
            ),
            disable_split_kv=bool(namespace.flashinfer_disable_split_kv),
            fast_decode_plan=bool(namespace.flashinfer_fast_decode_plan),
        ),
    )


def _none_if_empty(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "auto"}:
        return None
    return text


def _positive_optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    parsed = int(cast(Any, value))
    if parsed <= 0:
        raise ValueError("optional integer tuning values must be positive when set")
    return parsed


def _bounded_fraction(value: float, name: str) -> float:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be greater than 0 and less than 1")
    return value


def _parse_positive_int_csv(raw: object | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value > 0:
            values.append(value)
    return tuple(sorted(set(values))) if values else default


def _parse_optional_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected optional bool token, got {value!r}")
