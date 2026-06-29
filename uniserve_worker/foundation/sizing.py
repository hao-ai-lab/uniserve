"""Shared scalar sizing helpers for worker capacity declarations."""
from __future__ import annotations

__all__ = [
    'DEFAULT_NUM_BLOCKS_FALLBACK',
    'DEFAULT_MAX_BATCH_OPS',
    'DEFAULT_BLOCK_SIZE',
    'ceil_div',
    'derive_num_blocks',
]

# Number of KV blocks assumed when a model declares no explicit token capacity
# (used as the fallback in :func:`derive_num_blocks`). Named for its use as a
# block-count fallback rather than a token multiplier.
DEFAULT_NUM_BLOCKS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024

DEFAULT_BLOCK_SIZE = 256


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
