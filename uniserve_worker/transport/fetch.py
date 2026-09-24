"""Deliver logical tensor coverage through bound physical transports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from uniserve import _slices
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    Locator,
    TensorTransfer,
    WorkerEndpoint,
)

if TYPE_CHECKING:
    import torch

    from uniserve_worker.transport.interface import Transport
    from uniserve_worker.transport.ticket import TransferTicket

from uniserve_worker.transport.layout import (
    read_destination,
    region_view,
    validate_destination,
)


def fetch_tensor(
    tensor: TensorTransfer,
    destination: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    bindings: Mapping[tuple[WorkerEndpoint, str], Transport],
    region: tuple[slice, ...] | None = None,
    retain: Callable[[TransferTicket], None] | None = None,
) -> tuple[TransferTicket, ...]:
    """Deliver exactly the requested coverage using bound source edges.

    Replica choice is deterministic: locations published over `local` first,
    then the rest in publication order. Coverage is validated before any read
    starts; a failed selected source reports failure and does not trigger
    another backend or producer execution. Each ticket retains its own
    physical source and destination through cancellation and device
    completion.

    Args:
        tensor: The logical tensor and every location it was published at.
        destination: One tensor, or ordered first-axis spans, shaped like
            `region` and on the device the reads target.
        bindings: The transport bound for each (source endpoint, backend)
            edge; a location whose edge is unbound is skipped.
        region: The part of the logical tensor to read; the whole tensor when
            omitted.
        retain: Called with each ticket as soon as it is submitted.

    Returns:
        One ticket per physical read, in submission order.

    Raises:
        WorkerError: `invalid_descriptor` when the region, destination or
            bound locations are invalid or do not cover the region. An error
            raised while submitting, by a backend's `fetch` or by `retain`,
            propagates after the tickets already submitted are cancelled.
    """
    region = region or tuple(
        slice(start, start + extent)
        for start, extent in zip(
            (0,) * len(tensor.shape), tensor.shape, strict=True
        )
    )
    if not _slices.within(region, tensor.shape):
        raise invalid_descriptor("consumer region exceeds the logical tensor")
    spans = destination if isinstance(destination, tuple) else (destination,)
    if not spans:
        raise invalid_descriptor("tensor destination has no spans")
    device = spans[0].device
    validate_destination(
        destination,
        shape=_slices.shape(region),
        dtype=tensor.dtype,
        device=device,
    )

    # Greedily cover the requested region from bound source locations. Those
    # published over `local`, which only this address space can read, come
    # first and the rest keep publication order, so a rank reads its own copy
    # of a product before any other rank's; the head names a media product's
    # readers on the same basis. Each read records its region in both
    # source-local and region-local coordinates.
    missing = [region]
    reads: list[
        tuple[Transport, Locator, tuple[slice, ...], tuple[slice, ...]]
    ] = []
    ordered = sorted(
        tensor.locations, key=lambda location: location.backend != "local"
    )
    for location in ordered:
        transport = bindings.get((location.source, location.backend))
        if transport is None:
            continue
        if transport.name != location.backend:
            raise invalid_descriptor(
                "bound transport disagrees with the physical edge"
            )
        coverage = tuple(
            slice(start, start + extent)
            for start, extent in zip(
                location.offset, location.shape, strict=True
            )
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
        raise invalid_descriptor(
            "bound product locations do not cover the consumer region"
        )
    # Form and validate every physical destination before submitting any work.
    destinations = [
        region_view(destination, target_region)
        for _, _, _, target_region in reads
    ]
    for (_, location, source_region, _), target in zip(
        reads, destinations, strict=True
    ):
        read_destination(location, device, target, source_region)

    tickets: list[TransferTicket] = []
    try:
        for (transport, location, source_region, _), target in zip(
            reads, destinations, strict=True
        ):
            ticket = transport.fetch(
                location,
                device=device,
                destination=target,
                region=source_region,
            )
            tickets.append(ticket)
            if retain is not None:
                retain(ticket)
    except BaseException:
        # A partially submitted fan-out must not leave orphan reads running.
        for ticket in tickets:
            ticket.cancel()
        raise
    return tuple(tickets)
