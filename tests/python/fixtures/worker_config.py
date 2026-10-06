"""Resolved worker resources for deterministic model fixtures."""

from uniserve_worker.config.execution import WorkerConfig


def stub_worker_config(
    block_size: int = 64,
    *,
    max_batch_calls: int = 1024,
    max_batch_tokens: int,
) -> WorkerConfig:
    """Build CPU execution capacity for the deterministic simulator."""
    return WorkerConfig(
        device="cpu",
        rank=0,
        world_size=1,
        block_size=int(block_size),
        kv_token_capacity=int(block_size) * 4096,
        attention_backend="torch",
        model_dtype="bfloat16",
        kv_cache_dtype=None,
        kv_storage_fraction=1.0,
        max_batch_calls=int(max_batch_calls),
        max_batch_tokens=int(max_batch_tokens),
        max_request_pool_size=128,
        encoder_cache_entries=1024,
        generation_device=None,
    )
