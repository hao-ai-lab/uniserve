"""Match logical tensor coverage to bounded physical reads into reserved storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from ..foundation.errors import invalid_descriptor
from ..protocol.batch import (
    Locator,
    TensorTransfer,
    WorkerEndpoint,
)

if TYPE_CHECKING:
    import torch

    from .tickets import TransferTicket, Transport


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
            raise invalid_descriptor("tensor region has invalid bounds")

    def within(self, shape: tuple[int, ...]) -> bool:
        return len(shape) == len(self.shape) and all(
            start + extent <= bound
            for start, extent, bound in zip(self.offset, self.shape, shape, strict=True)
        )

    def intersection(self, other: TensorRegion) -> TensorRegion | None:
        if len(self.shape) != len(other.shape):
            raise invalid_descriptor("tensor regions have different dimensions")
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


def validate_destination(
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    shape: tuple[int, ...],
    dtype: str,
    device: torch.device,
) -> None:
    """Validate the complete logical destination before it is split into reads.

    Separate shard reads must not alias each other's destinations. Page views
    can interleave across a layer-major allocation, so overlapping envelopes
    are resolved into their actual contiguous physical intervals.
    """

    spans = destination if isinstance(destination, tuple) else (destination,)
    if not spans or any(
        span.device != device or str(span.dtype).removeprefix("torch.") != dtype for span in spans
    ):
        raise invalid_descriptor("transfer destination disagrees with the published representation")
    if isinstance(destination, tuple):
        if any(
            span.ndim != len(shape) or tuple(span.shape[1:]) != shape[1:] or int(span.shape[0]) < 1
            for span in spans
        ):
            raise invalid_descriptor(
                "transfer destination disagrees with the published representation"
            )
        actual_shape = (sum(int(span.shape[0]) for span in spans), *shape[1:])
    else:
        actual_shape = tuple(destination.shape)
    if actual_shape != shape:
        raise invalid_descriptor("transfer destination disagrees with the published representation")
    ranges = []
    for span in spans:
        extent = 1
        for stride, size in sorted(zip(span.stride(), span.shape, strict=True)):
            if size <= 1:
                continue
            if stride < extent:
                raise invalid_descriptor("transfer destination has overlapping physical elements")
            extent += (size - 1) * stride
        ranges.append((span.data_ptr(), span.data_ptr() + extent * span.element_size()))
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        intervals = []
        for span in spans:
            width = 1
            axes = []
            for stride, size in sorted(zip(span.stride(), span.shape, strict=True)):
                if size <= 1:
                    continue
                if stride == width:
                    width *= size
                else:
                    axes.append((stride, size))
            itemsize = span.element_size()
            for coordinates in product(*(range(size) for _, size in axes)):
                start = span.data_ptr() + itemsize * sum(
                    index * stride for index, (stride, _) in zip(coordinates, axes, strict=True)
                )
                intervals.append((start, start + width * itemsize))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise invalid_descriptor("transfer destination spans overlap")


def region_view(
    destination: torch.Tensor | tuple[torch.Tensor, ...], region: TensorRegion
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Select a region of one view or logical first-axis page spans without packing."""

    if not isinstance(destination, tuple):
        if not region.within(tuple(destination.shape)):
            raise invalid_descriptor("tensor region exceeds its destination")
        return destination[region.slices()]
    if not destination:
        raise invalid_descriptor("tensor destination has no spans")
    logical = (sum(int(span.shape[0]) for span in destination), *destination[0].shape[1:])
    if not region.within(logical):
        raise invalid_descriptor("tensor region exceeds its destination spans")
    pieces = []
    position = 0
    for span in destination:
        shape = tuple(span.shape)
        if shape[1:] != logical[1:]:
            raise invalid_descriptor("tensor destination spans disagree on trailing dimensions")
        covered = TensorRegion((position, *(0 for _ in shape[1:])), shape)
        overlap = region.intersection(covered)
        if overlap is not None:
            pieces.append(span[overlap.relative_to(covered.offset).slices()])
        position += int(span.shape[0])
    return tuple(pieces)


def fetch_tensor(
    tensor: TensorTransfer,
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    bindings: Mapping[tuple[WorkerEndpoint, str], Transport],
    region: TensorRegion | None = None,
    retain: Callable[[TransferTicket], None] | None = None,
) -> tuple[TransferTicket, ...]:
    """Deliver exactly the requested coverage using explicitly bound source edges.

    Replica choice is deterministic in publication order. Coverage is validated
    before any read starts; a failed selected source reports failure and does not
    trigger another backend or producer execution. Each ticket retains its own
    physical source and destination through cancellation and device completion.
    """

    from .tickets import _read_destination

    region = region or TensorRegion((0,) * len(tensor.shape), tensor.shape)
    if not region.within(tensor.shape):
        raise invalid_descriptor("consumer region exceeds the logical tensor")
    spans = destination if isinstance(destination, tuple) else (destination,)
    if not spans:
        raise invalid_descriptor("tensor destination has no spans")
    device = spans[0].device
    validate_destination(destination, shape=region.shape, dtype=tensor.dtype, device=device)
    missing = [region]
    reads: list[tuple[Transport, Locator, TensorRegion, TensorRegion]] = []
    for location in tensor.locations:
        transport = bindings.get((location.source, location.backend))
        if transport is None:
            continue
        if transport.name != location.backend:
            raise invalid_descriptor("bound transport disagrees with the physical edge")
        coverage = TensorRegion(location.offset, location.shape)
        remaining: list[TensorRegion] = []
        for required in missing:
            overlap = required.intersection(coverage)
            if overlap is not None:
                reads.append(
                    (
                        transport,
                        location,
                        overlap.relative_to(location.offset),
                        overlap.relative_to(region.offset),
                    )
                )
            remaining.extend(required.subtract(coverage))
        missing = remaining
        if not missing:
            break
    if missing:
        raise invalid_descriptor("bound product locations do not cover the consumer region")
    # Form and validate every physical destination before submitting any work.
    destinations = [region_view(destination, target_region) for _, _, _, target_region in reads]
    for (_, location, source_region, _), target in zip(reads, destinations, strict=True):
        _read_destination(location, device, target, source_region)
    tickets: list[TransferTicket] = []
    try:
        for (transport, location, source_region, _), target in zip(
            reads, destinations, strict=True
        ):
            ticket = transport.fetch(
                location, device=device, destination=target, region=source_region
            )
            tickets.append(ticket)
            if retain is not None:
                retain(ticket)
    except BaseException:
        for ticket in tickets:
            ticket.cancel()
        raise
    return tuple(tickets)
