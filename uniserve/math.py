"""Shared integer sizing helpers."""

from __future__ import annotations

__all__ = [
    "bucketed_length",
    "ceil_div",
]


def ceil_div(value: int, divisor: int) -> int:
    """Return the ceiling of ``value / divisor`` for a positive divisor.

    Capacity owners validate their own extents; a nonpositive divisor is a
    caller error rather than a unit-sized capacity.
    """
    if int(divisor) < 1:
        raise ValueError("ceiling division requires a positive divisor")
    return -(-int(value) // int(divisor))


def bucketed_length(value: int) -> int:
    """Round a shape length up to a power of two.

    Executable identity is exact-shape, so a length that tracks a sequence as it
    grows makes every length its own captured graph. Bucketing to a power of two
    keeps one shape serving a whole range of lengths. Every bucketed length must
    stay consistent with the tensors indexed by it: a bound and the tensor whose
    width the kernel checks against that bound have to be bucketed together.
    """
    count = int(value)
    if count <= 1:
        return max(0, count)
    return 1 << (count - 1).bit_length()
