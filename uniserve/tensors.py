"""Numerical buffer requirements and borrowed tensor outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, prod

import torch

from uniserve import _slices


def adjacent_view(values: Sequence[torch.Tensor]) -> torch.Tensor | None:
    """Return one flat view when tensors cover adjacent regions of one storage.

    The input order is significant. No allocation is performed, and ``None``
    indicates that concatenation is required.
    """

    if not values:
        return None

    flat = tuple(value.reshape(-1) for value in values)
    first = flat[0]
    if (
        not first.is_contiguous()
        or any(not value.is_contiguous() for value in flat)
        or any(value.dtype != first.dtype or value.device != first.device for value in flat)
    ):
        return None

    storage = first.untyped_storage().data_ptr()
    offset = int(first.storage_offset())

    # Each flattened view must share one storage and begin exactly where the
    # previous view ended; any gap or overlap forces the concatenation path.
    expected = offset
    for value in flat:
        if value.untyped_storage().data_ptr() != storage or int(value.storage_offset()) != expected:
            return None
        expected += int(value.numel())

    return first.as_strided((expected - offset,), (1,), storage_offset=offset)


def concatenate_views(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Borrow adjacent flattened views, or concatenate disjoint tensors."""

    tensors = tuple(value.reshape(-1) for value in values)
    if not tensors:
        raise ValueError("at least one tensor is required")
    view = adjacent_view(tensors)
    return torch.cat(tensors, dim=0) if view is None else view


def _join_channels(values, *, copy: bool = True):
    """Borrow adjacent channel views, or concatenate unrelated output storage.

    Channel views qualify when they share one storage, agree on every
    non-channel extent and stride, and tile the last axis back to back.
    """

    first = values[0]
    position = first.storage_offset()
    storage = first.untyped_storage().data_ptr()

    # A single mismatch on any view abandons the borrowed-view path entirely.
    for value in values:
        if (
            value.shape[:-1] != first.shape[:-1]
            or value.stride()[:-1] != first.stride()[:-1]
            or value.stride(-1) != 1
            or value.dtype != first.dtype
            or value.device != first.device
            or value.untyped_storage().data_ptr() != storage
            or value.storage_offset() != position
        ):
            break
        position += value.shape[-1]
    else:
        return first.as_strided(
            (*first.shape[:-1], sum(value.shape[-1] for value in values)), first.stride()
        )
    return torch.cat(values, dim=-1) if copy else None


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


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """Global output representation and this rank's rectangular slice.

    Variable axes describe bounded output extents for a numerical input size.
    A local slice never reduces the global capacity needed to assemble shards.
    Returned tensors own or borrow their actual storage independently of this value.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    local_slice: tuple[slice, ...]
    variable_axes: tuple[int, ...] = ()
    value_range: tuple[float, float] | None = None

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
        if not _slices.within(self.local_slice, self.shape):
            raise ValueError("output slice must lie within its global shape")
        if self.value_range is not None and (
            len(self.value_range) != 2
            or not all(isfinite(value) for value in self.value_range)
            or self.value_range[0] >= self.value_range[1]
        ):
            raise ValueError("output value range must be a finite increasing interval")


@dataclass(frozen=True, slots=True)
class TensorOutput:
    """Borrow a numerical tensor and its position within the complete result."""

    tensor: torch.Tensor
    layout: OutputLayout

    def __post_init__(self) -> None:
        if tuple(self.tensor.shape) != _slices.shape(self.layout.local_slice):
            raise ValueError("output tensor shape must match its local slice")
        if self.tensor.dtype != self.layout.dtype:
            raise ValueError("output tensor dtype must match its layout")
