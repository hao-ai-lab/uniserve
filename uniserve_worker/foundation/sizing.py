"""Shared scalar sizing helpers for worker capacity declarations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    'DEFAULT_NUM_BLOCKS_FALLBACK',
    'DEFAULT_MAX_BATCH_OPS',
    'DEFAULT_BLOCK_SIZE',
    'CudaKVCapacity',
    'RuntimeKVCapacity',
    'ceil_div',
    'device_total_bytes',
    'derive_cuda_kv_capacity',
    'derive_num_blocks',
    'derive_runtime_kv_capacity',
]

# Number of KV blocks assumed when a model declares no explicit token capacity
# (used as the fallback in :func:`derive_num_blocks`). Named for its use as a
# block-count fallback rather than a token multiplier.
DEFAULT_NUM_BLOCKS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024

DEFAULT_BLOCK_SIZE = 64


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


@dataclass(frozen=True)
class RuntimeKVCapacity:
    block_size: int
    bytes_per_token: int
    token_capacity: int
    num_blocks: int
    cuda: CudaKVCapacity | None = None


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


def device_total_bytes(device: Any) -> int:
    """Total memory of one device, or zero when it exposes no CUDA memory."""

    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    try:
        target = torch.device(device)
    except Exception:
        return 0
    if target.type != "cuda":
        return 0
    try:
        _free, total = torch.cuda.mem_get_info(target)
    except Exception:
        return 0
    return int(total)


def derive_cuda_kv_capacity(
    *,
    device: Any,
    block_size: int,
    bytes_per_token: int,
    memory_fraction: float,
    floor: int = 1,
    resident_copies: int = 1,
    co_resident_blocks: int = 0,
) -> CudaKVCapacity | None:
    """Derive KV token/block capacity from the device static-memory budget.

    CUDA visibility and free-memory accounting are deployment concerns, so model
    classes call this helper instead of reaching into ``torch.cuda`` directly.
    ``None`` means CUDA sizing is unavailable and callers should use their
    non-CUDA capacity policy.

    ``memory_fraction`` is the share of total device memory that static
    residency may hold: the memory already resident when sizing runs, such as
    model weights, plus every KV pool provisioned from the derived capacity.
    Whatever the fraction leaves stays available for activations, graph pools,
    and other transient allocations. A deployment that keeps ``resident_copies``
    copies of the derived pool resident at once, plus ``co_resident_blocks`` of
    fixed KV storage, receives a block count that satisfies
    ``resident_copies * num_blocks + co_resident_blocks`` within the budget.
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
    resident_bytes = max(0, int(total_bytes) - int(free_bytes))
    usable_bytes = max(0, int(float(total_bytes) * fraction) - resident_bytes)
    budget_blocks = (usable_bytes // token_bytes) // block
    copies = max(1, int(resident_copies))
    request_blocks = (budget_blocks - max(0, int(co_resident_blocks))) // copies
    num_blocks = max(max(1, int(floor)), request_blocks)
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


def derive_runtime_kv_capacity(
    *,
    block_size: int,
    kv_token_capacity: int | None,
    bytes_per_token: int,
    device: Any = None,
    memory_fraction: float = 1.0,
    floor: int = 1,
    default_blocks: int | None = None,
    resident_copies: int = 1,
    co_resident_blocks: int = 0,
) -> RuntimeKVCapacity:
    """Resolve the worker-facing KV capacity policy in one place.

    Explicit token capacity wins. Otherwise CUDA free-memory sizing is used
    when available, with ``resident_copies`` and ``co_resident_blocks``
    describing the KV storage that shares the memory-fraction budget with the
    request pool. CPU/unavailable-CUDA paths fall back to
    :func:`derive_num_blocks`' shared default block policy.
    """

    block = int(block_size)
    token_bytes = max(1, int(bytes_per_token))
    if kv_token_capacity is not None and int(kv_token_capacity) > 0:
        blocks = derive_num_blocks(block, kv_token_capacity, floor=floor)
        return RuntimeKVCapacity(
            block_size=block,
            bytes_per_token=token_bytes,
            token_capacity=int(blocks * block),
            num_blocks=int(blocks),
            cuda=None,
        )

    cuda = derive_cuda_kv_capacity(
        device=device,
        block_size=block,
        bytes_per_token=token_bytes,
        memory_fraction=memory_fraction,
        floor=floor,
        resident_copies=resident_copies,
        co_resident_blocks=co_resident_blocks,
    )
    if cuda is not None:
        return RuntimeKVCapacity(
            block_size=block,
            bytes_per_token=token_bytes,
            token_capacity=int(cuda.token_capacity),
            num_blocks=int(cuda.num_blocks),
            cuda=cuda,
        )

    blocks = derive_num_blocks(block, None, default_blocks=default_blocks, floor=floor)
    return RuntimeKVCapacity(
        block_size=block,
        bytes_per_token=token_bytes,
        token_capacity=int(blocks * block),
        num_blocks=int(blocks),
        cuda=None,
    )
