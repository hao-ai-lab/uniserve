"""In-process tensor export and borrowed consumer views.

A `local` export is read only within the producer's own address space
and on the source device. Its locator carries the exporting instance's
endpoint name and an integer key into that instance's table. Native readers
resolve the producer through a weak registry shared by transports in the
process, then retain it through the consumer's final access.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import count
from typing import TYPE_CHECKING, Any

from uniserve import _slices
from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import (
    BufferRegistry,
    Completion,
    TransportBuffer,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    LocalTransfer,
    Locator,
    WorkerEndpoint,
)
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    dtype_name,
    export_views,
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


class LocalTransport(Transport):
    """Exports read in the producer's process through a local table.

    A read without a destination borrows the exported views themselves and
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
        self._events = event_pool
        self._next = count()
        self._buffers = BufferRegistry(
            # Each buffer carries at least one byte, so the byte budget also
            # bounds how many registrations can be retained.
            capacity=capacity.capacity,
            event_pool=event_pool,
        )
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

    def retirement(self, locator: Locator) -> Completion:
        """Retain the allocation ownership shared by local calls.

        Local copies and borrowed views use the same ownership.
        """
        return self._buffers.retirement(locator)

    def export(
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
        t, shape, offset = export_views(tensor, offset)
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

        source = TransportBuffer.local(t, event, nbytes, self.capacity)
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
            source.retire(self._events)
            raise
        return locator

    def set_completion_wake(self, wake: Any) -> None:
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
        """Borrow or copy from a verified producer in this address space.

        The producer owns export bytes and their event. The destination
        owns copy capacity; a borrowed view retains its producer's event pool
        until consumer completion returns the source grant.
        """
        return self._reads.fetch_local(
            locator,
            device=device,
            destination=destination,
            region=region,
            reservation=reservation,
        )

    def release(self, locator: Locator) -> Completion | None:
        """Revoke new local reads.

        Existing copies and borrowed views are retained.
        """
        return self._buffers.release(locator)

    def close(self) -> None:
        """Release every tensor registered under this local endpoint."""
        try:
            self._reads.close()
        finally:
            self._buffers.close()


def _read_views(
    tensor: torch.Tensor | tuple[torch.Tensor, ...],
    locator: Locator,
    device: torch.device,
    destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
    region: tuple[slice, ...] | None,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, ...],
    torch.Tensor | tuple[torch.Tensor, ...] | None,
]:
    """Borrow source spans and validate an optional copy destination."""
    first = tensor[0] if isinstance(tensor, tuple) else tensor
    if device != first.device:
        raise invalid_descriptor("local binding requires the source device")
    if region is not None:
        if not _slices.within(region, locator.shape):
            raise invalid_descriptor("read region exceeds the exported view")
        tensor = region_view(tensor, region)

    target = (
        None
        if destination is None
        else read_destination(locator, device, destination, region)
    )
    return tensor, target
