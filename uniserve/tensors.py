"""Numerical tensor regions shared by computation and storage callers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod

import torch


@dataclass(frozen=True, slots=True)
class TensorRegion:
    """A nonempty rectangular region, measured in logical tensor elements."""

    offset: tuple[int, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.shape
            or len(self.offset) != len(self.shape)
            or any(start < 0 for start in self.offset)
            or any(extent < 1 for extent in self.shape)
        ):
            raise ValueError("tensor region has invalid bounds")

    def within(self, shape: tuple[int, ...]) -> bool:
        return len(shape) == len(self.shape) and all(
            start + extent <= bound
            for start, extent, bound in zip(self.offset, self.shape, shape, strict=True)
        )

    def intersection(self, other: TensorRegion) -> TensorRegion | None:
        if len(self.shape) != len(other.shape):
            raise ValueError("tensor regions have different dimensions")
        start = tuple(max(a, b) for a, b in zip(self.offset, other.offset, strict=True))
        end = tuple(
            min(a + n, b + m)
            for a, n, b, m in zip(self.offset, self.shape, other.offset, other.shape, strict=True)
        )
        if any(a >= b for a, b in zip(start, end, strict=True)):
            return None
        return TensorRegion(start, tuple(b - a for a, b in zip(start, end, strict=True)))

    def subtract(self, covered: TensorRegion) -> tuple[TensorRegion, ...]:
        """Partition the remainder without duplicating overlapping replica reads."""

        intersection = self.intersection(covered)
        if intersection is None:
            return (self,)
        start = list(self.offset)
        end = [a + n for a, n in zip(self.offset, self.shape, strict=True)]
        remaining = []
        for axis, (low, extent) in enumerate(
            zip(intersection.offset, intersection.shape, strict=True)
        ):
            high = low + extent
            if start[axis] < low:
                piece_end = end.copy()
                piece_end[axis] = low
                remaining.append(
                    TensorRegion(
                        tuple(start), tuple(b - a for a, b in zip(start, piece_end, strict=True))
                    )
                )
                start[axis] = low
            if high < end[axis]:
                piece_start = start.copy()
                piece_start[axis] = high
                remaining.append(
                    TensorRegion(
                        tuple(piece_start),
                        tuple(b - a for a, b in zip(piece_start, end, strict=True)),
                    )
                )
                end[axis] = high
        return tuple(remaining)

    def relative_to(self, origin: tuple[int, ...]) -> TensorRegion:
        return TensorRegion(
            tuple(a - b for a, b in zip(self.offset, origin, strict=True)), self.shape
        )

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, a + n) for a, n in zip(self.offset, self.shape, strict=True))


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Numerical view and backing capacity required by one allocator field.

    ``host`` requires CPU representation. The allocation caller chooses pinning,
    communication registration, initialization and the lifetime of that backing.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    capacity_shape: tuple[int, ...] | None = None
    host: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in self.shape
        ):
            raise ValueError("buffer extents must be nonnegative integers")
        if self.capacity_shape is not None and (
            not isinstance(self.capacity_shape, tuple)
            or len(self.capacity_shape) != len(self.shape)
            or any(
                not isinstance(bound, int) or isinstance(bound, bool) or bound < extent
                for extent, bound in zip(self.shape, self.capacity_shape, strict=True)
            )
        ):
            raise ValueError("buffer capacity must contain its numerical view")

    @property
    def nbytes(self) -> int:
        """Return backing bytes before allocator alignment or replication."""

        return (
            prod(self.capacity_shape if self.capacity_shape is not None else self.shape)
            * self.dtype.itemsize
        )


class ImageRange(StrEnum):
    """Whether image samples use signed-unit or unit numerical values."""

    SIGNED_UNIT = "signed_unit"
    UNIT = "unit"


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Global output representation and this rank's optional rectangular region.

    Variable axes describe bounded output extents for a numerical input size.
    A local region never reduces the global capacity needed to assemble shards.
    Returned tensors own or borrow their actual storage independently of this value.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    region: TensorRegion | None = None
    variable_axes: tuple[int, ...] = ()
    value_range: ImageRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in self.shape
        ):
            raise ValueError("output extents must be nonnegative integers")
        if (
            not isinstance(self.variable_axes, tuple)
            or len(set(self.variable_axes)) != len(self.variable_axes)
            or any(
                not isinstance(axis, int)
                or isinstance(axis, bool)
                or not 0 <= axis < len(self.shape)
                for axis in self.variable_axes
            )
        ):
            raise ValueError("variable output axes must be distinct axes of the global shape")
        if self.region is not None and not self.region.within(self.shape):
            raise ValueError("output region must lie within its global shape")
