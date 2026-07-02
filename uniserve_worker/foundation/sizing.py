"""Shared scalar sizing helpers for worker capacity declarations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    'DEFAULT_NUM_BLOCKS_FALLBACK',
    'DEFAULT_MAX_BATCH_OPS',
    'DEFAULT_BLOCK_SIZE',
    'CudaKVCapacity',
    'ceil_div',
    'derive_cuda_kv_capacity',
    'derive_num_blocks',
]

# Number of KV blocks assumed when a model declares no explicit token capacity
# (used as the fallback in :func:`derive_num_blocks`). Named for its use as a
# block-count fallback rather than a token multiplier.
DEFAULT_NUM_BLOCKS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024

DEFAULT_BLOCK_SIZE = 256


@dataclass(frozen=True)
class CudaKVCapacity:
    device: str
    free_bytes: int
    total_bytes: int
    memory_fraction: float
    bytes_per_token: int
    block_size: int
    token_capacity: int
    num_blocks: int


def ceil_div(value: int, divisor: int) -> int:
    """Ceiling division of two integers.

    The divisor is clamped to ``>= 1`` so a zero or negative ``block_size``
    cannot raise ``ZeroDivisionError``.
    """
    divisor = max(1, int(divisor))
    return (int(value) + divisor - 1) // divisor


def derive_num_blocks(
    block_size: int,
    kv_token_capacity: int | None,
    *,
    default_blocks: int | None = None,
    floor: int = 1,
) -> int:
    """Derive the KV block count from token capacity and block size.

    When ``kv_token_capacity`` is unset or non-positive, ``default_blocks`` (or
    :data:`DEFAULT_NUM_BLOCKS_FALLBACK`) is used. The result is at least
    ``floor`` (default 1).
    """

    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be positive")
    min_blocks = max(1, int(floor))
    if kv_token_capacity is None or int(kv_token_capacity) <= 0:
        blocks = DEFAULT_NUM_BLOCKS_FALLBACK if default_blocks is None else int(default_blocks)
    else:
        blocks = int(kv_token_capacity) // block
    return max(min_blocks, blocks)


def derive_cuda_kv_capacity(
    *,
    device: Any,
    block_size: int,
    bytes_per_token: int,
    memory_fraction: float,
    floor: int = 1,
) -> CudaKVCapacity | None:
    """Derive KV token/block capacity from currently free CUDA memory.

    CUDA visibility and free-memory accounting are deployment concerns, so model
    classes call this helper instead of reaching into ``torch.cuda`` directly.
    ``None`` means CUDA sizing is unavailable and callers should use their
    non-CUDA capacity policy.
    """

    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None
    try:
        cuda_device = torch.device(device)
    except Exception:
        return None
    if cuda_device.type != "cuda":
        return None
    block = int(block_size)
    token_bytes = max(1, int(bytes_per_token))
    if block <= 0:
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    except Exception:
        return None
    fraction = float(memory_fraction)
    usable_bytes = max(0, int(float(free_bytes) * fraction))
    raw_token_capacity = usable_bytes // token_bytes
    num_blocks = max(max(1, int(floor)), int(raw_token_capacity) // block)
    return CudaKVCapacity(
        device=str(cuda_device),
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=fraction,
        bytes_per_token=token_bytes,
        block_size=block,
        token_capacity=int(num_blocks * block),
        num_blocks=int(num_blocks),
    )
