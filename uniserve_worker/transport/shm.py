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

from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker._uniserve_ipc import (
    Completion,
    HostLane,
    HostTask,
    SharedBuffer,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.transfer import (
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from uniserve_worker.transport import segment
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
from uniserve_worker.transport.shared_storage import open_shared_storage
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class _ShmSource:
    storage: SharedBuffer
    retirement: HostTask[None] | None = None


@dataclass(slots=True)
class HostBorrow:
    """A consumer's direct view of an exported segment's bytes.

    The bytes stay in the producer's segment: a reader on this host, such as
    a media unit encode task, maps them by name and offset. ``release``
    writes this rank's acknowledgment word and closes this borrow's own
    mapping, which lets the producer retire the segment; the caller calls it
    once every read of the bytes is done. Later calls do nothing.
    """

    segment: str
    offset: int
    #: Byte length of the borrowed span. `execution.host_media` narrows it to
    #: one media unit's frames before handing the borrow to its encode task.
    nbytes: int
    _release: Callable[[], None]

    def release(self) -> None:
        release, self._release = self._release, lambda: None
        release()


@contextmanager
def _shared_read(locator: Locator, slot: int, *, check=None):
    """Own a mapping and claim through its last read, including failure.

    Opens the segment named by ``locator``, claims this rank's ``slot`` and
    waits for readiness before yielding the mapping.
    Leaving the context, normally or by an exception, acknowledges a claimed
    slot and closes the mapping. ``check`` is forwarded to
    `segment.await_ready`.

    Raises:
        WorkerError: `invalid_descriptor` when ``locator`` is not a shared
            storage locator, or its segment is missing; `resource_error`
            when the producer failed or readiness
            timed out. Whatever ``check`` raises, and any other error opening
            the segment, propagates.
    """
    handle = locator.transport
    if not isinstance(handle, PosixShmTransfer):
        raise invalid_descriptor("shared storage read requires a SHM locator")
    try:
        shm = open_shared_storage(
            handle.name, segment.HEADER_BYTES + locator.nbytes
        )
    except FileNotFoundError:
        raise invalid_descriptor(
            "shared buffer is no longer available"
        ) from None

    header = memoryview(shm)
    claimed = False
    try:
        segment.claim(header, slot)
        claimed = True
        segment.await_ready(header, check=check)
        yield shm
    finally:
        # Failed readiness and cancellation end this reader's access too. The
        # producer separately retains its own writes until they have completed.
        try:
            if claimed:
                segment.acknowledge(header, slot)
        finally:
            header.release()
            shm.close()


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
                memoryview(storage)[segment.HEADER_BYTES :], dtype=first.dtype
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
        copy ends. The private buffer, staged in pinned memory for a CUDA
        destination, then feeds the possibly asynchronous destination copy.
        """
        import torch

        try:
            ticket._require_active()
            with _shared_read(
                locator,
                self._acknowledgment_slot,
                check=ticket._require_active,
            ) as shm:
                payload = segment.HEADER_BYTES
                buf = bytearray(shm[payload : payload + locator.nbytes])
        except BaseException as error:
            ticket._fail(error)
            raise
        source = torch.frombuffer(
            buf, dtype=resolve_dtype(locator.dtype)
        ).reshape(locator.shape)
        if device.type == "cuda":
            # The read ticket retains this bounded pinned buffer until DMA
            # retires.
            pinned = torch.empty(
                source.shape, dtype=source.dtype, pin_memory=True
            )
            pinned.copy_(source)
            source = pinned

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
    ) -> HostBorrow:
        """Expose a exported payload in place for a reader on this host.

        The reader is told the segment's name and the byte span of ``region``,
        a span of whole leading-axis rows, and reads it through its own
        mapping; the bytes are neither copied nor retained here. Readiness
        is awaited before returning, and releasing the borrow writes this
        rank's acknowledgment word.

        Raises:
            WorkerError: `invalid_descriptor` when ``locator`` is not a
                shared storage locator on this node or `row_span` refuses
                ``region``; otherwise as `_shared_read` raises when opening,
                identifying or awaiting the segment.
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
        # On success the mapping's exit moves into the borrow's release; if
        # the borrow cannot be built, the stack acknowledges and closes now.
        with ExitStack() as ownership:
            ownership.enter_context(
                _shared_read(locator, self._acknowledgment_slot)
            )
            return HostBorrow(
                handle.name,
                segment.HEADER_BYTES + start,
                nbytes,
                ownership.pop_all().close,
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
