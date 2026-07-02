"""Typed worker runtime configuration.

The worker process resolves stable serving configuration at the composition root
(``main.py``) and deep subsystems read this immutable config instead of ambient
environment variables. Profiling/debug-only env toggles remain in their narrow
modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .env import DEFAULT_ATTENTION_BACKEND, DEFAULT_COMPILE_BACKEND

__all__ = [
    "DEFAULT_DECODE_GRAPH_BATCH_SIZES",
    "DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS",
    "FlashInferTuningConfig",
    "TorchCompileRuntimeConfig",
    "WorkerRuntimeConfig",
    "get_worker_config",
    "set_worker_config",
    "worker_config_from_args",
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
)


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
class WorkerRuntimeConfig:
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
    forward_max_memory_bound_tokens: int = 281
    green_contexts: bool = False
    logits_processor_chunk_size: int = 0
    torch_compile: TorchCompileRuntimeConfig = TorchCompileRuntimeConfig()
    flashinfer: FlashInferTuningConfig = FlashInferTuningConfig()


_CURRENT_CONFIG = WorkerRuntimeConfig()


def get_worker_config() -> WorkerRuntimeConfig:
    return _CURRENT_CONFIG


def set_worker_config(config: WorkerRuntimeConfig) -> None:
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = config


def worker_config_from_args(args: Any) -> WorkerRuntimeConfig:
    return WorkerRuntimeConfig(
        model_dtype=str(args.model_dtype),
        transformers_trust_remote_code=bool(args.transformers_trust_remote_code),
        transformers_attn_implementation=str(args.transformers_attn_implementation),
        allow_transformers_fallback=bool(getattr(args, "allow_transformers_fallback", False)),
        disabled_model_archs=tuple(str(v) for v in (args.disable_model_arch or ())),
        strict_model_imports=bool(args.strict_model_imports),
        kv_cache_dtype=_none_if_empty(args.kv_cache_dtype),
        kv_memory_fraction=_bounded_fraction(float(args.kv_memory_fraction), "kv-memory-fraction"),
        tp_backend=_none_if_empty(args.tp_backend),
        tp_init_method=_none_if_empty(args.tp_init_method),
        mooncake_device=str(args.mooncake_device or ""),
        mooncake_protocol=str(args.mooncake_protocol or "rdma"),
        cuda_graph=bool(args.cuda_graph),
        cuda_graph_warmup=bool(args.cuda_graph_warmup),
        cuda_graph_warmup_batches=_parse_positive_int_csv(
            args.cuda_graph_warmup_batches,
            default=DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        ),
        prefill_cuda_graph=bool(args.prefill_cuda_graph),
        prefill_cuda_graph_warmup=bool(args.prefill_cuda_graph_warmup),
        prefill_cuda_graph_warmup_tokens=_parse_positive_int_csv(
            args.prefill_cuda_graph_warmup_tokens,
            default=DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        ),
        mixed_text_max_tokens=max(0, int(args.mixed_text_max_tokens)),
        varlen_prefill=bool(args.varlen_prefill),
        forward_max_memory_bound_tokens=max(1, int(args.forward_max_memory_bound_tokens)),
        green_contexts=bool(args.green_contexts),
        logits_processor_chunk_size=max(0, int(args.logits_processor_chunk_size)),
        torch_compile=TorchCompileRuntimeConfig(
            enabled=bool(args.torch_compile),
            backend=str(args.torch_compile_backend),
            mode=_none_if_empty(args.torch_compile_mode),
            fullgraph=bool(args.torch_compile_fullgraph),
            dynamic=_parse_optional_bool(args.torch_compile_dynamic),
        ),
        flashinfer=FlashInferTuningConfig(
            workspace_size=max(1, int(args.flashinfer_workspace_size)),
            use_tensor_core=_parse_optional_bool(args.flashinfer_use_tensor_core),
            decode_backend=str(args.flashinfer_decode_backend),
            prefill_backend=str(args.flashinfer_prefill_backend),
            decode_split_tile_size=_positive_optional_int(args.flashinfer_decode_split_tile_size),
            prefill_split_tile_size=_positive_optional_int(args.flashinfer_prefill_split_tile_size),
            disable_split_kv=bool(args.flashinfer_disable_split_kv),
            fast_decode_plan=bool(args.flashinfer_fast_decode_plan),
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
    parsed = int(value)
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
