"""Tensor representation, source regions, and destination span validation."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, overload

from uniserve import _slices
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import Locator

if TYPE_CHECKING:
    import torch


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
        span.device != device or str(span.dtype).removeprefix("torch.") != dtype
        for span in spans
    ):
        raise invalid_descriptor(
            "transfer destination disagrees with the published representation"
        )

    # A tuple destination is a logical tensor partitioned along the first axis;
    # spans must agree on trailing dimensions and concatenate to the full shape.
    if isinstance(destination, tuple):
        if any(
            span.ndim != len(shape)
            or tuple(span.shape[1:]) != shape[1:]
            or int(span.shape[0]) < 1
            for span in spans
        ):
            raise invalid_descriptor(
                "transfer destination disagrees with the "
                "published representation"
            )
        actual_shape = (sum(int(span.shape[0]) for span in spans), *shape[1:])
    else:
        actual_shape = tuple(destination.shape)
    if actual_shape != shape:
        raise invalid_descriptor(
            "transfer destination disagrees with the published representation"
        )

    # First pass: a coarse per-span envelope [data_ptr, data_ptr + extent). A
    # stride smaller than the accumulated extent proves the view indexes the
    # same element twice, which is rejected outright.
    ranges = []
    for span in spans:
        extent = 1
        for stride, size in sorted(zip(span.stride(), span.shape, strict=True)):
            if size <= 1:
                continue
            if stride < extent:
                raise invalid_descriptor(
                    "transfer destination has overlapping physical elements"
                )
            extent += (size - 1) * stride
        ranges.append(
            (span.data_ptr(), span.data_ptr() + extent * span.element_size())
        )
    ranges.sort()

    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        # Envelopes overlap, but interleaved views may still be disjoint.
        # Second pass: split each span into its maximal contiguous runs
        # (trailing axes where stride == width fold into the run width) and
        # check the exact intervals for overlap.
        intervals = []
        for span in spans:
            width = 1
            axes = []
            for stride, size in sorted(
                zip(span.stride(), span.shape, strict=True)
            ):
                if size <= 1:
                    continue
                if stride == width:
                    width *= size
                else:
                    axes.append((stride, size))
            itemsize = span.element_size()
            for coordinates in product(*(range(size) for _, size in axes)):
                start = span.data_ptr() + itemsize * sum(
                    index * stride
                    for index, (stride, _) in zip(
                        coordinates, axes, strict=True
                    )
                )
                intervals.append((start, start + width * itemsize))
        intervals.sort()
        if any(
            left[1] > right[0] for left, right in zip(intervals, intervals[1:])
        ):
            raise invalid_descriptor("transfer destination spans overlap")


@overload
def region_view(
    destination: torch.Tensor, region: tuple[slice, ...]
) -> torch.Tensor: ...


@overload
def region_view(
    destination: tuple[torch.Tensor, ...], region: tuple[slice, ...]
) -> tuple[torch.Tensor, ...]: ...


def region_view(
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    region: tuple[slice, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Select a region of one view or logical first-axis page spans.

    No packing is performed.
    """
    if not isinstance(destination, tuple):
        if not _slices.within(region, tuple(destination.shape)):
            raise invalid_descriptor("tensor region exceeds its destination")
        return destination[region]

    if not destination:
        raise invalid_descriptor("tensor destination has no spans")
    logical = (
        sum(int(span.shape[0]) for span in destination),
        *destination[0].shape[1:],
    )
    if not _slices.within(region, logical):
        raise invalid_descriptor("tensor region exceeds its destination spans")

    # Walk the first-axis spans, emitting one view per span that the region
    # touches; each view is expressed in its own span's local coordinates.
    pieces = []
    position = 0
    for span in destination:
        shape = tuple(span.shape)
        if shape[1:] != logical[1:]:
            raise invalid_descriptor(
                "tensor destination spans disagree on trailing dimensions"
            )
        covered = tuple(
            slice(start, start + extent)
            for start, extent in zip(
                (position, *(0 for _ in shape[1:])), shape, strict=True
            )
        )
        overlap = _slices.intersection(region, covered)
        if overlap is not None:
            pieces.append(
                span[_slices.relative(overlap, _slices.offset(covered))]
            )
        position += int(span.shape[0])
    return tuple(pieces)


def dtype_name(dtype: torch.dtype) -> str:
    """Encode a torch dtype as its unqualified transport name."""
    return str(dtype).removeprefix("torch.")


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a transport dtype name to a torch dtype."""
    import torch

    return getattr(torch, name)


def tensor_nbytes(tensor: torch.Tensor | tuple[torch.Tensor, ...]) -> int:
    """Return the physical byte size of a tensor view."""
    spans = tensor if isinstance(tensor, tuple) else (tensor,)
    return sum(int(span.numel() * span.element_size()) for span in spans)


def read_destination(
    locator: Locator,
    device: torch.device,
    destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
    region: tuple[slice, ...] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Validate exact read bounds and writable, disjoint destination spans."""
    import torch

    region = region or tuple(
        slice(start, start + extent)
        for start, extent in zip(
            (0,) * len(locator.shape), locator.shape, strict=True
        )
    )
    if not _slices.within(region, locator.shape):
        raise invalid_descriptor("read region exceeds the published view")
    dtype = resolve_dtype(locator.dtype)
    if destination is None:
        return torch.empty(_slices.shape(region), dtype=dtype, device=device)
    validate_destination(
        destination,
        shape=_slices.shape(region),
        dtype=locator.dtype,
        device=device,
    )
    return destination


def publication_views(
    tensor: torch.Tensor | tuple[torch.Tensor, ...],
    offset: tuple[int, ...] | None,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, ...], tuple[int, ...], tuple[int, ...]
]:
    """Validate ordered first-axis spans and retain their immutable views."""
    spans = tensor if isinstance(tensor, tuple) else (tensor,)
    if not spans:
        raise invalid_descriptor("publication has no source spans")
    first = spans[0]
    if first.ndim < 1 or any(
        span.ndim != first.ndim
        or span.dtype != first.dtype
        or span.device != first.device
        or tuple(span.shape[1:]) != tuple(first.shape[1:])
        or any(size < 1 for size in span.shape)
        for span in spans
    ):
        raise invalid_descriptor(
            "publication spans disagree on their representation"
        )

    shape = (sum(int(span.shape[0]) for span in spans), *first.shape[1:])
    value = (0,) * len(shape) if offset is None else offset
    if len(value) != len(shape) or any(
        not isinstance(start, int) or start < 0 for start in value
    ):
        raise invalid_descriptor(
            "publication offset does not match its tensor shape"
        )

    # Detach so autograd metadata never reaches readers of the published view.
    source = tuple(span.detach() for span in spans)
    return (source if isinstance(tensor, tuple) else source[0]), shape, value


def copy_pairs(
    source: torch.Tensor | tuple[torch.Tensor, ...],
    destination: torch.Tensor | tuple[torch.Tensor, ...],
):
    """Walk two first-axis partitions together without packing into one tensor.

    Yields (target, value) view pairs whose first-axis lengths match, splitting
    at span boundaries on both sides. Both partitions must cover the same
    logical first-axis length.
    """
    sources = source if isinstance(source, tuple) else (source,)
    targets = destination if isinstance(destination, tuple) else (destination,)
    source_index = target_index = 0
    source_start = target_start = 0
    while source_index < len(sources) and target_index < len(targets):
        value, target = sources[source_index], targets[target_index]
        count = min(
            value.shape[0] - source_start, target.shape[0] - target_start
        )
        yield (
            target[target_start : target_start + count],
            value[source_start : source_start + count],
        )
        source_start += count
        target_start += count
        if source_start == value.shape[0]:
            source_index += 1
            source_start = 0
        if target_start == target.shape[0]:
            target_index += 1
            target_start = 0
    if source_index != len(sources) or target_index != len(targets):
        raise invalid_descriptor(
            "transfer partitions have different logical lengths"
        )


def row_span(
    locator: Locator, region: tuple[slice, ...] | None
) -> tuple[int, int]:
    """Return the byte offset and length of whole leading-axis rows.

    A borrowed span must be contiguous in the payload, so every axis but the
    first is taken whole; the payload's leading axis is indexed relative to
    the locator's own offset.
    """
    import torch

    shape = tuple(int(extent) for extent in locator.shape)
    itemsize = torch.empty(
        (), dtype=resolve_dtype(locator.dtype)
    ).element_size()
    row_bytes = itemsize
    for extent in shape[1:]:
        row_bytes *= extent
    if region is None:
        return 0, row_bytes * (shape[0] if shape else 1)
    if len(region) != len(shape) or any(
        (axis.start or 0) != 0
        or (axis.stop is not None and axis.stop != extent)
        for axis, extent in zip(region[1:], shape[1:], strict=True)
    ):
        raise invalid_descriptor(
            "a borrowed span covers whole rows of the leading axis"
        )
    leading = region[0]
    first = (leading.start or 0) - (locator.offset[0] if locator.offset else 0)
    last = (shape[0] if leading.stop is None else leading.stop) - (
        locator.offset[0] if locator.offset else 0
    )
    if not 0 <= first < last <= shape[0]:
        raise invalid_descriptor("a borrowed span lies outside its location")
    return first * row_bytes, (last - first) * row_bytes
