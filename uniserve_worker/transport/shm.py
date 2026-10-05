"""POSIX segment export, direct host borrowing, and acknowledgment.

`ShmTransport` exports a host product, or a device product copied to the
host, into a fresh POSIX shared-memory segment per export, laid out as
`segment` describes. Readers on the producer's host open the segment by name:
`fetch` copies the payload out on a `TransferPool` thread, and `borrow`
exposes it in place to a reader such as a media unit encode task. Neither
needs a connection to the producing rank: the producer sweeps the
acknowledgment words when it reaps, and unlinks a retired segment once no
named consumer is still reading it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import (
    SHM_HEADER_BYTES,
    Completion,
    HostLane,
    HostTask,
    SharedBuffer,
    SharedRead,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from uniserve_worker.transport.endpoint import BufferRegistry
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    copy_pairs,
    dtype_name,
    export_views,
    read_destination,
    resolve_dtype,
    row_span,
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
class _ShmSource:
    storage: SharedBuffer
    retirement: HostTask[None] | None = None


class ShmTransport(Transport):
    """Shared storage export whose segment carries its own readiness.

    A consumer opens the segment by name, waits on readiness and writes its own
    acknowledgment word once it has copied the payload out. The producer
    unlinks the segment once every named consumer has acknowledged, so a
    read needs no connection to the producing rank.
    """

    name = "shm"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
        acknowledgment_slot: int = 0,
        host_slots: Sequence[int] = (),
    ) -> None:
        self.capacity = capacity
        # Slots of the ranks on this host: the only ones a segment named in
        # this host's namespace can reach.
        self._host_slots = frozenset(int(slot) for slot in host_slots)
        # CUDA host unregistration may wait on unrelated device work. Keep it
        # off the execution loop, using the shared native host executor.
        self._retirements = HostLane(
            max_inflight=256, workers=1, name="uniserve-shm-retire"
        )
        self._buffers = BufferRegistry(
            capacity=256,
            reclaim=self._reclaim,
            drain=self._drain,
            settled=lambda source: source.storage.settled(),
        )
        # This rank's own word in the header of every segment it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        self.source = source or WorkerEndpoint.local()
        self._events = event_pool
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-shm-read",
            event_pool=event_pool,
        )

    def endpoint(self) -> str:
        return self._buffers.name

    def serves(self, consumers: Sequence[int]) -> bool:
        return not consumers or any(
            int(slot) in self._host_slots for slot in consumers
        )

    def retirement(self, locator: Locator) -> Completion:
        return self._buffers.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)
        self._retirements.set_completion_wake(wake)

    def _reclaim(self, source: _ShmSource, retirement: Completion) -> None:
        if source.storage.is_cuda:
            source.retirement = self._retirements.reserve()
            source.retirement.submit(self._retire, source.storage, retirement)
        else:
            self._retire(source.storage, retirement)

    def _retire(self, storage: SharedBuffer, retirement: Completion) -> None:
        try:
            storage.close()
            self.capacity.release(storage.nbytes)
        except BaseException as error:
            retirement.set_exception(error)
            raise
        else:
            retirement.set_result(None)

    def _drain(self, source: _ShmSource) -> None:
        source.storage.synchronize()
        self._buffers.reap()
        if source.retirement is not None:
            source.retirement.result()

    def reap(self) -> None:
        self._buffers.reap()

    def awaiting_acknowledgment(self) -> bool:
        return self._buffers.awaiting_acknowledgment()

    def export(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Copy tensor spans into native shared storage and return its locator.

        CUDA copies land directly in the registered mapping. A native callback
        marks the bytes readable after DMA, without entering Python. Reaping
        returns capacity after producer completion and reader acknowledgments.
        """
        import torch

        source, shape, offset = export_views(tensor, offset)
        first = source[0] if isinstance(source, tuple) else source
        nbytes = tensor_nbytes(source)
        self._buffers.reap()
        self.capacity.acquire(nbytes)

        storage = None
        registered = False
        try:
            stream = (
                torch.cuda.current_stream(first.device)
                if first.is_cuda
                else None
            )
            device = (
                None
                if stream is None
                else (stream.device.index, int(stream.cuda_stream))
            )
            storage = SharedBuffer(nbytes, tuple(consumers), device)
            locator = Locator(
                source=self.source,
                transport=PosixShmTransfer(
                    endpoint=self.endpoint(), name=storage.name
                ),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            packed = torch.frombuffer(
                memoryview(storage)[SHM_HEADER_BYTES:], dtype=first.dtype
            ).reshape(shape)
            self._buffers.register(locator, _ShmSource(storage))
            registered = True

            if stream is None:
                for target, value in copy_pairs(source, packed):
                    target.copy_(value)
            else:
                from uniserve_kernels.peer_storage import copy_host_device

                storage.begin_copy()
                for target, value in copy_pairs(source, packed):
                    copy_host_device(target, value, stream)
                    value.record_stream(stream)

            storage.mark_ready()
            if stream is not None:
                self._events.notify_stream(int(stream.cuda_stream))
            return locator
        except BaseException as error:
            # A failed copy can leave earlier spans in flight. Native close
            # drains those accesses before unregistering and unlinking storage;
            # a failed drain retains the mapping and its byte reservation.
            if storage is not None:
                try:
                    storage.close()
                except BaseException as cleanup_error:
                    error.add_note(
                        f"shared buffer cleanup failed: {cleanup_error}"
                    )
                    raise error from cleanup_error
            if registered:
                self._buffers.release(locator)
            else:
                self.capacity.release(nbytes)
            raise

    def _read_tensor(
        self,
        ticket: TransferTicket,
        locator: Locator,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
        region: tuple[slice, ...] | None,
    ) -> None:
        """Copy the payload out of the segment, then acknowledge it.

        The segment is mapped and claimed only until the payload has been
        copied into a private buffer, and it is acknowledged as soon as that
        copy ends. The private buffer is pinned for a CUDA destination and
        feeds the possibly asynchronous destination copy.
        """
        import torch

        handle = locator.transport
        if not isinstance(handle, PosixShmTransfer):
            raise invalid_descriptor(
                "shared storage read requires a SHM locator"
            )
        try:
            with SharedRead(
                handle.name,
                locator.nbytes,
                self._acknowledgment_slot,
                ticket=ticket,
            ) as read:
                view = torch.frombuffer(
                    read, dtype=resolve_dtype(locator.dtype)
                ).reshape(locator.shape)
                # The read ticket retains this private buffer through DMA.
                # Copy directly from the mapped bytes; no bytearray or second
                # host mapping is needed between the producer and this owner.
                source = torch.empty(
                    view.shape,
                    dtype=view.dtype,
                    pin_memory=device.type == "cuda",
                )
                source.copy_(view)
        except BaseException as error:
            ticket._fail(error)
            raise

        target = read_destination(locator, device, destination, region)
        if region is not None:
            source = source[region]
        self._reads.copy(ticket, source, target)

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        """Submit a read of a segment exported on this node.

        Raises:
            WorkerError: `invalid_descriptor` when the export lies on
                another node or, with a `destination`, when `region` exceeds
                the exported view or `destination` does not match it;
                without one, those errors fail the ticket instead. Errors
                from `TransferPool.submit` propagate.
        """
        if locator.source.node != self.source.node:
            raise invalid_descriptor(
                "shared storage transport requires the source node"
            )
        target = (
            None
            if destination is None
            else read_destination(locator, device, destination, region)
        )
        return self._reads.submit(
            self._read_tensor,
            locator,
            device,
            target,
            region,
            nbytes=locator.nbytes,
            destination=target,
            reservation=reservation,
        )

    def borrow(
        self, locator: Locator, region: tuple[slice, ...] | None = None
    ) -> SharedRead:
        """Borrow a mapped payload range for a host numerical consumer.

        Native readiness and acknowledgment retain the source through the
        consumer's last read. The returned object exposes only the selected
        rows through the buffer protocol; release ends its read grant.
        """
        handle = locator.transport
        if not isinstance(handle, PosixShmTransfer):
            raise invalid_descriptor(
                "shared storage borrow requires a shared storage locator"
            )
        if locator.source.node != self.source.node:
            raise invalid_descriptor(
                "shared storage transport requires the source node"
            )
        start, nbytes = row_span(locator, region)
        return SharedRead(
            handle.name, nbytes, self._acknowledgment_slot, offset=start
        )

    def release(self, locator: Locator) -> Completion | None:
        if not isinstance(locator.transport, PosixShmTransfer):
            raise invalid_descriptor(
                "shared storage release requires a shared storage locator"
            )
        return self._buffers.release(locator)

    def close(self) -> None:
        # Drain reads first. Registry close finishes producer copies, then
        # reaps buffers whose granted readers have all acknowledged.
        try:
            self._reads.close()
        finally:
            try:
                self._buffers.close()
            finally:
                self._retirements.close()
