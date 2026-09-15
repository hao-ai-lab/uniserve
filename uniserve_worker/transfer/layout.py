"""Match logical tensor coverage to bounded physical reads into reserved storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import product
from typing import TYPE_CHECKING

from uniserve import _slices

from ..foundation.errors import invalid_descriptor
from ..protocol.transfer import Locator, TensorTransfer, WorkerEndpoint

if TYPE_CHECKING:
    import torch

    from .tickets import TransferTicket, Transport


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
    destination: torch.Tensor | tuple[torch.Tensor, ...], region: tuple[slice, ...]
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Select a region of one view or logical first-axis page spans without packing."""

    if not isinstance(destination, tuple):
        if not _slices.within(region, tuple(destination.shape)):
            raise invalid_descriptor("tensor region exceeds its destination")
        return destination[region]
    if not destination:
        raise invalid_descriptor("tensor destination has no spans")
    logical = (sum(int(span.shape[0]) for span in destination), *destination[0].shape[1:])
    if not _slices.within(region, logical):
        raise invalid_descriptor("tensor region exceeds its destination spans")
    pieces = []
    position = 0
    for span in destination:
        shape = tuple(span.shape)
        if shape[1:] != logical[1:]:
            raise invalid_descriptor("tensor destination spans disagree on trailing dimensions")
        covered = tuple(
            slice(start, start + extent)
            for start, extent in zip((position, *(0 for _ in shape[1:])), shape, strict=True)
        )
        overlap = _slices.intersection(region, covered)
        if overlap is not None:
            pieces.append(span[_slices.relative(overlap, _slices.offset(covered))])
        position += int(span.shape[0])
    return tuple(pieces)


def fetch_tensor(
    tensor: TensorTransfer,
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    bindings: Mapping[tuple[WorkerEndpoint, str], Transport],
    region: tuple[slice, ...] | None = None,
    retain: Callable[[TransferTicket], None] | None = None,
) -> tuple[TransferTicket, ...]:
    """Deliver exactly the requested coverage using explicitly bound source edges.

    Replica choice is deterministic in publication order. Coverage is validated
    before any read starts; a failed selected source reports failure and does not
    trigger another backend or producer execution. Each ticket retains its own
    physical source and destination through cancellation and device completion.
    """

    from .tickets import _read_destination

    region = region or tuple(
        slice(start, start + extent)
        for start, extent in zip((0,) * len(tensor.shape), tensor.shape, strict=True)
    )
    if not _slices.within(region, tensor.shape):
        raise invalid_descriptor("consumer region exceeds the logical tensor")
    spans = destination if isinstance(destination, tuple) else (destination,)
    if not spans:
        raise invalid_descriptor("tensor destination has no spans")
    device = spans[0].device
    validate_destination(
        destination, shape=_slices.shape(region), dtype=tensor.dtype, device=device
    )
    missing = [region]
    reads: list[tuple[Transport, Locator, tuple[slice, ...], tuple[slice, ...]]] = []
    for location in tensor.locations:
        transport = bindings.get((location.source, location.backend))
        if transport is None:
            continue
        if transport.name != location.backend:
            raise invalid_descriptor("bound transport disagrees with the physical edge")
        coverage = tuple(
            slice(start, start + extent)
            for start, extent in zip(location.offset, location.shape, strict=True)
        )
        remaining: list[tuple[slice, ...]] = []
        for required in missing:
            overlap = _slices.intersection(required, coverage)
            if overlap is not None:
                reads.append(
                    (
                        transport,
                        location,
                        _slices.relative(overlap, location.offset),
                        _slices.relative(overlap, _slices.offset(region)),
                    )
                )
            remaining.extend(_slices.subtract(required, coverage))
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
