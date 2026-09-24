"""In-process tensor publication and borrowed consumer views.

A `local` publication is read only within the producer's own address space
and on the source device. Its locator carries the publishing instance's
endpoint name and an integer key into that instance's table; a reader finds
the owning instance through the process-wide `_endpoints` registry, so any
`LocalTransport` in the process can read another's publications.
"""

from __future__ import annotations

import concurrent.futures
import threading
import uuid
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from uniserve import _slices
from uniserve.runtime import EventPool
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.protocol.transfer import (
    LocalTransfer,
    Locator,
    WorkerEndpoint,
)
from uniserve_worker.transport.endpoint import _endpoint_lock, _endpoints
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    dtype_name,
    publication_views,
    read_destination,
    region_view,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import TransferCapacity, TransferPool
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class _LocalSource:
    """One registered publication and its reclamation state.

    Attributes:
        tensor: The detached published views.
        event: Producer fence recorded at publication; None for a CPU source.
        capacity: Budget the publication's bytes are reserved against.
        locator: The exact locator the publication was registered under.
        readers: Copies and borrowed views taken and not yet finished.
        released: New reads are revoked, by `release` or `close`.
        reclaiming: Hand-back has started; `retirement` completes once the
            producer fence, if any, drains.
        retirement: Completes when the bytes are returned to `capacity`.
    """

    tensor: torch.Tensor | tuple[torch.Tensor, ...]
    event: torch.cuda.Event | None
    capacity: TransferCapacity
    locator: Locator
    readers: int = 0
    released: bool = False
    reclaiming: bool = False
    retirement: concurrent.futures.Future[None] = field(
        default_factory=concurrent.futures.Future
    )

    def events_released(self) -> None:
        self.capacity.release(tensor_nbytes(self.tensor))
        self.retirement.set_result(None)


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
        self._table: dict[int, _LocalSource] = {}
        self._borrowed: weakref.WeakSet[TransferTicket] = weakref.WeakSet()
        # Borrowed views draw on the same read-ticket semaphore as the copies
        # `TransferPool` submits.
        self._borrow_slots = capacity.read_slots
        self._events = event_pool
        self._next = 0
        self._completion_wake: Any = None
        self._lock = threading.RLock()
        self._endpoint = f"local:{uuid.uuid4().hex}"
        with _endpoint_lock:
            _endpoints[self._endpoint] = self
        self._bytes = capacity
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-local-read",
            event_pool=event_pool,
        )

    def endpoint(self) -> str:
        """Expose the process-unique endpoint encoded into local locators."""
        return self._endpoint

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        """Retain the allocation ownership shared by local calls.

        Local copies and borrowed views use the same ownership.
        """
        if locator.source != self.source:
            raise invalid_descriptor(
                "local publication belongs to another rank incarnation"
            )
        handle = locator.transport
        if (
            not isinstance(handle, LocalTransfer)
            or handle.endpoint != self._endpoint
        ):
            raise invalid_descriptor(
                "local publication belongs to another endpoint"
            )
        with self._lock:
            source = self._table.get(handle.key)
            if source is None or locator != source.locator:
                raise invalid_descriptor(
                    "local publication changed its registered view"
                )
            return source.retirement

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
        self._bytes.acquire(nbytes)

        # A CUDA source carries a producer fence recorded on the caller's
        # current stream; readers order their copies behind it.
        event = None
        if first.is_cuda:
            event = self._events.acquire(first.device)
            self._events.record(event, first.device)
            self._events.retain(event, first.device)

        with self._lock:
            # Drop table entries whose physical ownership already completed.
            self._table = {
                key: source
                for key, source in self._table.items()
                if not source.retirement.done()
            }
            key = self._next
            self._next += 1
            locator = Locator(
                source=self.source,
                transport=LocalTransfer(endpoint=self._endpoint, key=key),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            self._table[key] = _LocalSource(t, event, self._bytes, locator)
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
    ) -> TransferTicket:
        """Borrow or copy from a verified publisher in this address space.

        The producer owns publication bytes and their event. The destination
        owns copy capacity; a borrowed view retains its producer's event pool
        until consumer completion returns the source grant.
        """
        handle = locator.transport
        if (
            not isinstance(handle, LocalTransfer)
            or locator.source.node != self.source.node
            or locator.source.address_space != self.source.address_space
        ):
            raise invalid_descriptor(
                "local locator belongs to another address space"
            )

        with _endpoint_lock:
            owner = _endpoints.get(handle.endpoint)
        if (
            not isinstance(owner, LocalTransport)
            or owner.source != locator.source
        ):
            raise invalid_descriptor(
                "local publication belongs to another rank incarnation"
            )

        with owner._lock:
            source = owner._table.get(handle.key)
            if source is None or source.released:
                raise invalid_descriptor(
                    "local publication is no longer registered"
                )
            tensor, event = source.tensor, source.event
            first = tensor[0] if isinstance(tensor, tuple) else tensor
            if device != first.device:
                raise invalid_descriptor(
                    "local binding requires the source device"
                )
            if locator != source.locator:
                raise invalid_descriptor(
                    "local locator changed its registered view"
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
            source.readers += 1

        try:
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
                )
                ticket.add_retirement_callback(
                    lambda: owner._release_reader(source)
                )
                return ticket

            # Borrow path: no copy. The source views themselves are handed out
            # and the reader count drops when every consumer stream completes.
            if not self._borrow_slots.acquire(blocking=False):
                raise resource_error(
                    "local borrowed-view ticket capacity is exhausted"
                )
            try:
                ticket = TransferTicket(owner._events)
                if self._completion_wake is not None:
                    ticket.add_done_callback(self._completion_wake)
                    ticket.add_retirement_callback(self._completion_wake)
                ticket._complete(tensor, event)
                ticket._consumer_release = lambda: self._finish_borrow(
                    owner, source
                )
                self._borrowed.add(ticket)
            except BaseException:
                self._borrow_slots.release()
                raise
            return ticket
        except BaseException:
            owner._release_reader(source)
            raise

    def _release_reader(self, source: _LocalSource) -> None:
        with self._lock:
            source.readers -= 1
            self._reclaim(source)

    def _finish_borrow(
        self, owner: LocalTransport, source: _LocalSource
    ) -> None:
        owner._release_reader(source)
        self._borrow_slots.release()

    def _reclaim(self, source: _LocalSource) -> None:
        # A source is handed back once, after new reads were revoked and its
        # last reader finished; a CPU source has no fence to wait on.
        if not source.released or source.readers != 0 or source.reclaiming:
            return
        source.reclaiming = True
        if source.event is None:
            source.events_released()
        else:
            self._events.defer_release(
                (source.event,), source, completed=source.events_released
            )

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        """Revoke new local reads.

        Existing copies and borrowed views are retained.
        """
        if locator.source != self.source:
            raise invalid_descriptor(
                "local publication belongs to another rank incarnation"
            )
        handle = locator.transport
        if (
            not isinstance(handle, LocalTransfer)
            or handle.endpoint != self._endpoint
        ):
            raise invalid_descriptor(
                "local release belongs to another endpoint"
            )
        with self._lock:
            source = self._table.get(handle.key)
            if source is None:
                return None
            source.released = True
            self._reclaim(source)
            # A retirement waiting on the producer fence completes only when an
            # `EventPool.reap` observes it, so the controller is woken once the
            # fence completes.
            if not source.retirement.done() and source.event is not None:
                self._events.schedule_completion_wake(
                    (
                        source.tensor[0]
                        if isinstance(source.tensor, tuple)
                        else source.tensor
                    ).device,
                    source.event,
                )
            return source.retirement

    def close(self) -> None:
        """Release every tensor registered under this local endpoint."""
        try:
            self._reads.close()
        finally:
            for ticket in tuple(self._borrowed):
                ticket._drain_consumers()
            with self._lock:
                values = tuple(self._table.values())
                for source in values:
                    source.released = True
                    self._reclaim(source)
            # Drain each remaining fence. A source still unretired after that,
            # such as one with a reader outstanding, raises `resource_error`.
            for source in values:
                if not source.retirement.done() and source.event is not None:
                    source.event.synchronize()
                    self._events.reap()
                if not source.retirement.done():
                    raise resource_error(
                        "local source retains unfinished physical readers"
                    )
                source.retirement.result()
            self._table.clear()
            with _endpoint_lock:
                _endpoints.pop(self._endpoint, None)
