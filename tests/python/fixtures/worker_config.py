"""Resolved worker resources for deterministic model fixtures."""

from __future__ import annotations

from typing import Any

from uniserve.runtime.backends.attention.flashinfer import (
    Config as FlashInferConfig,
)
from uniserve_worker.bootstrap.cache import DEFAULT_BLOCK_SIZE
from uniserve_worker.bootstrap.capacity import DEFAULT_NUM_UNITS_FALLBACK
from uniserve_worker.config.execution import (
    DEFAULT_DECODE_GRAPH_BATCH_SIZES,
    DEFAULT_FLOW_GRAPH_BATCH_SIZES,
    DEFAULT_FLOW_GRAPH_SHAPES,
    DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
    WorkerConfig,
)


def cpu_worker_config(**overrides: Any) -> WorkerConfig:
    """Build a single-rank CPU worker configuration, overriding named fields.

    A launch field has no default in ``WorkerConfig``: the launch descriptor
    carries every value. Tests state the fields they exercise and take the
    rest from here.
    """
    fields: dict[str, Any] = {
        "device": "cpu",
        "rank": 0,
        "world_size": 1,
        "role": "model",
        "expert_exchange": None,
        "expert_microbatches": 1,
        "max_batch_calls": 1024,
        "max_batch_tokens": 8192,
        "max_model_len": 16384,
        "max_video_seconds": 15.0,
        "ffmpeg": "ffmpeg",
        "video_text_capacities": (),
        "video_resolutions": (),
        "video_aspect_ratios": (),
        "deployment_components": (),
        "max_request_pool_size": 128,
        "model_dtype": "bfloat16",
        "mem_fraction_static": 0.70,
        "lanes": (),
        "graph_policy": "auto",
        "decode_graph_batch_sizes": DEFAULT_DECODE_GRAPH_BATCH_SIZES,
        "prefill_cuda_graph": True,
        "prefill_outputs": True,
        "prefill_graph_token_sizes": DEFAULT_PREFILL_GRAPH_TOKEN_BUCKETS,
        "flow_cuda_graph": True,
        "flow_graph_batch_sizes": DEFAULT_FLOW_GRAPH_BATCH_SIZES,
        "flow_graph_shapes": DEFAULT_FLOW_GRAPH_SHAPES,
        "flashinfer": FlashInferConfig(),
    }
    return WorkerConfig(**{**fields, **overrides})


def stub_worker_config(
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    max_batch_calls: int = 1024,
    max_batch_tokens: int,
) -> WorkerConfig:
    """Build CPU execution capacity for the deterministic simulator."""
    return cpu_worker_config(
        block_size=int(block_size),
        max_total_tokens=int(block_size) * DEFAULT_NUM_UNITS_FALLBACK,
        attention_backend="torch",
        mem_fraction_static=1.0,
        max_batch_calls=int(max_batch_calls),
        max_batch_tokens=int(max_batch_tokens),
        max_request_pool_size=128,
        encoder_cache_entries=1024,
    )
