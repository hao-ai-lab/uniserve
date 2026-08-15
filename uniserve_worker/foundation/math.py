"""Shared integer sizing helpers."""

from __future__ import annotations

__all__ = [
    "bucketed_length",
    "ceil_div",
]


def ceil_div(value: int, divisor: int) -> int:
    """Ceiling division of two integers.

    The divisor is clamped to ``>= 1`` so a zero or negative ``block_size``
    cannot raise ``ZeroDivisionError``.
    """
    divisor = max(1, int(divisor))
    return (int(value) + divisor - 1) // divisor


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
