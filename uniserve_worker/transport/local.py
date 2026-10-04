"""In-process tensor publication and borrowed consumer views.

A `local` publication is read only within the producer's own address space
and on the source device. Its locator carries the publishing instance's
endpoint name and an integer key into that instance's table; a reader finds
the owning instance through the process-wide `_endpoints` registry, so any
`LocalTransport` in the process can read another's publications.
"""

from __future__ import annotations

import concurrent.futures
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any

from uniserve import _slices
from uniserve.runtime import EventPool
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    LocalTransfer,
    Locator,
    WorkerEndpoint,
)
from uniserve_worker.transport.endpoint import (
    BufferRegistry,
    _endpoint_lock,
    _endpoints,
)
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    dtype_name,
    publication_views,
    read_destination,
    region_view,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import (
    ReadReservation,
    TransferCapacity,
    TransferPool,
)
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class _LocalSource:
    """Detached tensor views and their producer fence, if on a device."""

    tensor: torch.Tensor | tuple[torch.Tensor, ...]
    event: torch.cuda.Event | None


class LocalTransport(Transport):
    """Publications read in the producer's process through a local table.

    A read without a destination borrows the published views themselves and
    copies nothing; a read with one copies into it on the read pool.
    """

    name = "local"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
    ) -> None:
        """Create a process-local tensor table with bounded retained bytes."""
        self.source = source or WorkerEndpoint.local()
        self._borrowed: weakref.WeakSet[TransferTicket] = weakref.WeakSet()
        self._events = event_pool
        self._next = count()
        self._completion_wake: Any = None
        self._buffers = BufferRegistry(
            # Each buffer carries at least one byte, so the byte budget also
            # bounds how many registrations can be retained.
            capacity=capacity.capacity,
            reclaim=self._reclaim,
            drain=self._drain,
            settled=lambda source: True,
        )
        with _endpoint_lock:
            _endpoints[self.endpoint()] = self
        self.capacity = capacity
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-local-read",
            event_pool=event_pool,
        )

    def endpoint(self) -> str:
        """Expose the process-unique endpoint encoded into local locators."""
        return self._buffers.name

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        """Retain the allocation ownership shared by local calls.

        Local copies and borrowed views use the same ownership.
        """
        return self._buffers.retirement(locator)

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        # Storage read in its own process retires with its readers here;
        # the consumers the head names never read it through this mechanism.
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Register a detached tensor in the in-process endpoint table.

        Return its locator.
        """
        t, shape, offset = publication_views(tensor, offset)
        first = t[0] if isinstance(t, tuple) else t
        self._events.reap()
        nbytes = tensor_nbytes(t)
        self.capacity.acquire(nbytes)

        # A CUDA source carries a producer fence recorded on the caller's
        # current stream; readers order their copies behind it.
        event = None
        if first.is_cuda:
            event = self._events.acquire(first.device)
            self._events.record(event, first.device)
            self._events.retain(event, first.device)

        source = _LocalSource(t, event)
        try:
            locator = Locator(
                source=self.source,
                transport=LocalTransfer(
                    endpoint=self.endpoint(), key=next(self._next)
                ),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            self._buffers.register(locator, source)
        except BaseException:
            self._reclaim(source, concurrent.futures.Future())
            raise
        return locator

    def set_completion_wake(self, wake: Any) -> None:
        self._completion_wake = wake
        self._reads.set_completion_wake(wake)

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        """Borrow or copy from a verified publisher in this address space.

        The producer owns publication bytes and their event. The destination
        owns copy capacity; a borrowed view retains its producer's event pool
        until consumer completion returns the source grant.
        """
        handle = locator.transport
        if not isinstance(handle, LocalTransfer):
            raise invalid_descriptor("local read requires a local locator")

        with _endpoint_lock:
            owner = _endpoints.get(handle.endpoint)
        if not isinstance(owner, LocalTransport):
            raise invalid_descriptor("local buffer has no live owner")

        source = owner._buffers.acquire(locator)
        borrowed_ticket = None
        try:
            tensor, event = source.tensor, source.event
            first = tensor[0] if isinstance(tensor, tuple) else tensor
            if device != first.device:
                raise invalid_descriptor(
                    "local binding requires the source device"
                )
            if region is not None:
                if not _slices.within(region, locator.shape):
                    raise invalid_descriptor(
                        "read region exceeds the published view"
                    )
                tensor = region_view(tensor, region)
            target = (
                None
                if destination is None
                else read_destination(locator, device, destination, region)
            )

            if target is not None:
                # Copy path: the pool ticket retires when the copy's physical
                # access ends, then the source's reader count drops.
                ticket = self._reads.submit(
                    self._reads.copy,
                    tensor,
                    target,
                    event,
                    nbytes=locator.nbytes,
                    destination=target,
                    reservation=reservation,
                )
                ticket.add_retirement_callback(
                    lambda: owner._buffers.release_reader(locator)
                )
                return ticket

            # Borrow path: no copy. The source views themselves are handed out
            # and the reader count drops when every consumer stream completes.
            # A borrowed view holds a read ticket, like the copies
            # `TransferPool` submits, until then.
            if reservation is None:
                self.capacity.take_reads(
                    message="local borrowed-view ticket capacity is exhausted"
                )
            else:
                reservation.use()
            try:
                ticket = TransferTicket(
                    owner._events,
                    release=lambda: self._finish_borrow(owner, locator),
                )
                borrowed_ticket = ticket
                if self._completion_wake is not None:
                    ticket.add_done_callback(self._completion_wake)
                    ticket.add_retirement_callback(self._completion_wake)
                ticket._complete(tensor, event)
                self._borrowed.add(ticket)
            except BaseException:
                if borrowed_ticket is None:
                    self.capacity.return_reads()
                raise
            return ticket
        except BaseException:
            # Once a borrowed ticket owns the grant, its close path returns
            # both source ownership and credit, including setup failure.
            if borrowed_ticket is None:
                owner._buffers.release_reader(locator)
            else:
                borrowed_ticket.close()
            raise

    def _finish_borrow(self, owner: LocalTransport, locator: Locator) -> None:
        try:
            owner._buffers.release_reader(locator)
        finally:
            self.capacity.return_reads()

    def _reclaim(
        self, source: _LocalSource, retirement: concurrent.futures.Future[None]
    ) -> None:
        def completed() -> None:
            self.capacity.release(tensor_nbytes(source.tensor))
            retirement.set_result(None)

        if source.event is None:
            completed()
        else:
            self._events.defer_release(
                (source.event,), source, completed=completed
            )

    def _drain(self, source: _LocalSource) -> None:
        if source.event is not None:
            source.event.synchronize()
            self._events.reap()

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        """Revoke new local reads.

        Existing copies and borrowed views are retained.
        """
        retirement = self._buffers.release(locator)
        if retirement is not None and not retirement.done():
            source = self._buffers.source(locator)
            # Wake the controller when a pending producer fence can be reaped.
            if source.event is not None:
                self._events.schedule_completion_wake(
                    (
                        source.tensor[0]
                        if isinstance(source.tensor, tuple)
                        else source.tensor
                    ).device,
                    source.event,
                )
        return retirement

    def close(self) -> None:
        """Release every tensor registered under this local endpoint."""
        try:
            self._reads.close()
        finally:
            for ticket in tuple(self._borrowed):
                ticket._drain_consumers()
            self._buffers.close()
            with _endpoint_lock:
                _endpoints.pop(self.endpoint(), None)
