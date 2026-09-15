"""Rectangular tensor indexing shared by loading, storage and reconstruction."""

from __future__ import annotations


def validate(region: tuple[slice, ...]) -> None:
    """Require explicit, nonnegative half-open bounds, including empty shards."""

    if not isinstance(region, tuple) or any(
        not isinstance(axis, slice)
        or type(axis.start) is not int
        or type(axis.stop) is not int
        or axis.start < 0
        or axis.stop < axis.start
        or (axis.step is not None and (type(axis.step) is not int or axis.step != 1))
        for axis in region
    ):
        raise ValueError("tensor slices require nonnegative explicit bounds and unit steps")


def shape(region: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple(axis.stop - axis.start for axis in region)


def offset(region: tuple[slice, ...]) -> tuple[int, ...]:
    return tuple(axis.start for axis in region)


def within(region: tuple[slice, ...], bounds: tuple[int, ...]) -> bool:
    validate(region)
    return len(region) == len(bounds) and all(
        axis.stop <= bound for axis, bound in zip(region, bounds, strict=True)
    )


def intersection(left: tuple[slice, ...], right: tuple[slice, ...]) -> tuple[slice, ...] | None:
    if len(left) != len(right):
        raise ValueError("tensor slices have different dimensions")
    result = tuple(
        slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(left, right, strict=True)
    )
    return None if any(axis.start >= axis.stop for axis in result) else result


def subtract(
    region: tuple[slice, ...], covered: tuple[slice, ...]
) -> tuple[tuple[slice, ...], ...]:
    """Partition uncovered elements into disjoint rectangles for physical reads."""

    if any(axis.start == axis.stop for axis in region):
        return ()
    overlap = intersection(region, covered)
    if overlap is None:
        return (region,)
    remainder = list(region)
    pieces = []
    for dim, axis in enumerate(overlap):
        current = remainder[dim]
        if current.start < axis.start:
            piece = remainder.copy()
            piece[dim] = slice(current.start, axis.start)
            pieces.append(tuple(piece))
        if axis.stop < current.stop:
            piece = remainder.copy()
            piece[dim] = slice(axis.stop, current.stop)
            pieces.append(tuple(piece))
        remainder[dim] = axis
    return tuple(pieces)


def relative(region: tuple[slice, ...], origin: tuple[int, ...]) -> tuple[slice, ...]:
    result = tuple(
        slice(axis.start - start, axis.stop - start)
        for axis, start in zip(region, origin, strict=True)
    )
    validate(result)
    return result
