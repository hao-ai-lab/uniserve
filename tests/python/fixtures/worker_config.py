"""Resolved worker resources for deterministic model fixtures."""

from uniserve_worker.bootstrap.capacity import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_BATCH_OPS,
    DEFAULT_MAX_REQUEST_POOL_SIZE,
    DEFAULT_NUM_BLOCKS_FALLBACK,
)
from uniserve_worker.config import WorkerConfig


def stub_worker_config(
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    max_batch_calls: int = DEFAULT_MAX_BATCH_OPS,
    max_batch_tokens: int,
) -> WorkerConfig:
    """Build CPU execution capacity for the deterministic simulator."""
    return WorkerConfig(
        device="cpu",
        rank=0,
        world_size=1,
        block_size=int(block_size),
        kv_token_capacity=int(block_size) * DEFAULT_NUM_BLOCKS_FALLBACK,
        attention_backend="torch",
        model_dtype="bfloat16",
        kv_cache_dtype=None,
        kv_memory_fraction=1.0,
        max_batch_calls=int(max_batch_calls),
        max_batch_tokens=int(max_batch_tokens),
        max_request_pool_size=DEFAULT_MAX_REQUEST_POOL_SIZE,
        encoder_cache_entries=1024,
        generation_device=None,
    )
