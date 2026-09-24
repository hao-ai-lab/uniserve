"""POSIX segment publication, direct host borrowing, and acknowledgment.

`ShmTransport` publishes a host product, or a device product copied to the
host, into a fresh POSIX shared-memory segment per publication, laid out as
`segment` describes. Readers on the producer's host open the segment by name:
`fetch` copies the payload out on a `TransferPool` thread, and `borrow`
exposes it in place to a reader such as a media unit encode task. Neither
needs a connection to the producing rank: the producer sweeps the
acknowledgment words when it reaps, and unlinks a retired segment once no
named consumer is still reading it.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import queue
import selectors
import socket
import threading
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker.errors import invalid_descriptor, resource_error
from uniserve_worker.protocol.transfer import (
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from uniserve_worker.transport import segment
from uniserve_worker.transport.endpoint import Publications, locator_digest
from uniserve_worker.transport.interface import Transport
from uniserve_worker.transport.layout import (
    copy_pairs,
    dtype_name,
    publication_views,
    read_destination,
    resolve_dtype,
    row_span,
    tensor_nbytes,
)
from uniserve_worker.transport.pool import TransferCapacity, TransferPool
from uniserve_worker.transport.shared_storage import (
    allocate_shared_storage,
    open_shared_storage,
)
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class _ShmSource:
    """Own a shared segment and any unfinished device-to-host publication.

    The segment begins with the header of ``segment``: the publication's
    digest, its readiness word and one acknowledgment word per instance rank.
    ``consumers`` are the slots whose words return the segment.
    """

    shm: Any
    nbytes: int
    consumers: tuple[int, ...]
    #: `StreamSignal` of an unfinished device-to-host copy into the segment,
    #: cleared by the publication thread once the signal fires.
    signal: Any = None
    #: Address of the segment's mapping while it is registered with the
    #: CUDA driver, so a device-to-host copy lands in it directly.
    registered: int | None = None


def _register_segment(buffer: memoryview) -> int:
    """Page-lock a segment's mapping so device copies land in it directly.

    Returns the mapping's address, which the unregistration needs. The
    registered range is the whole mapping, which starts on a page boundary
    as every mmap does.

    Raises:
        WorkerError: `resource_error` when the driver refuses the range.
    """
    import torch

    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    status = torch.cuda.cudart().cudaHostRegister(address, len(buffer), 0)
    if int(status) != 0:
        raise resource_error(
            f"registering a shared storage segment with the CUDA driver "
            f"failed with status {int(status)}"
        )
    return address


def _unregister_segment(address: int) -> None:
    """Undo `_register_segment`; the driver's status is not checked."""
    import torch

    torch.cuda.cudart().cudaHostUnregister(address)


@dataclass(slots=True)
class HostBorrow:
    """A consumer's direct view of a published segment's bytes.

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

    Opens the segment named by ``locator``, checks its digest, claims this
    rank's ``slot`` and waits for readiness before yielding the mapping.
    Leaving the context, normally or by an exception, acknowledges a claimed
    slot and closes the mapping. ``check`` is forwarded to
    `segment.await_ready`.

    Raises:
        WorkerError: `invalid_descriptor` when ``locator`` is not a shared
            storage locator, or its segment is missing or holds another
            view; `resource_error` when the producer failed or readiness
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
            "publication is retired, invalid, or belongs to another view"
        ) from None

    header = memoryview(shm)
    claimed = False
    try:
        if segment.digest(header) != locator_digest(locator):
            raise invalid_descriptor(
                "publication is retired, invalid, or belongs to another view"
            )
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
    """Shared storage publication whose segment carries its own readiness.

    A consumer opens the segment by name, checks the digest in its header
    against the locator, waits on the readiness word and writes its own
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
        self._bytes = capacity
        # Slots of the ranks on this host: the only ones a segment named in
        # this host's namespace can reach.
        self._host_slots = frozenset(int(slot) for slot in host_slots)
        # `_reclaim` completes a retirement before returning, so `close` has
        # no in-flight reclamation to drain.
        self._publications = Publications[_ShmSource](
            capacity=256,
            reclaim=self._reclaim,
            drain=lambda source: None,
            settled=self._settled,
        )
        # This rank's own word in the header of every segment it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        self.source = source or WorkerEndpoint.local()
        # Device publications reach the publication thread through this
        # queue; a byte on the socket pair wakes that thread's selector, and
        # a None item asks it to exit once every queued copy has signaled.
        self._publication_queue: queue.Queue[
            tuple[Locator, _ShmSource] | None
        ] = queue.Queue()
        self._publication_control_rx, self._publication_control_tx = (
            socket.socketpair()
        )
        self._publication_control_rx.setblocking(False)
        self._publication_control_tx.setblocking(False)
        self._completion_wake: Any = None
        self._closed = False
        self._publication_worker = threading.Thread(
            target=self._complete_publications,
            name="uniserve-shm-publication",
            daemon=True,
        )
        self._publication_worker.start()
        self._reads = TransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-shm-read",
            event_pool=event_pool,
        )

    def endpoint(self) -> str:
        return self._publications.name

    def serves(self, consumers: Sequence[int]) -> bool:
        return not consumers or any(
            int(slot) in self._host_slots for slot in consumers
        )

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        return self._publications.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._completion_wake = wake
        self._reads.set_completion_wake(wake)

    def _reclaim(
        self, source: _ShmSource, retirement: concurrent.futures.Future[None]
    ) -> None:
        # Called by `Publications` under its lock, once the publication is
        # retired, its producer has completed and it is settled. The CUDA
        # registration covers the mapping, so it is removed first.
        if source.registered is not None:
            _unregister_segment(source.registered)
            source.registered = None
        source.shm.close()
        try:
            source.shm.unlink()
        except FileNotFoundError:
            pass
        self._bytes.release(source.nbytes)
        retirement.set_result(None)

    @staticmethod
    def _settled(source: _ShmSource) -> bool:
        """Report whether no named consumer is still reading the segment.

        A retired segment returns once every consumer that began reading has
        acknowledged; a consumer that never began, such as the reader of a
        request cancelled before its call was submitted, holds nothing.
        """
        if not source.consumers:
            return True
        return segment.settled(source.shm.buf, source.consumers)

    def reap(self) -> None:
        self._publications.reap()

    def awaiting_acknowledgment(self) -> bool:
        return self._publications.awaiting_acknowledgment()

    def _queue_publication(
        self, item: tuple[Locator, _ShmSource] | None
    ) -> None:
        self._publication_queue.put(item)
        try:
            self._publication_control_tx.send(b"P")
        except BlockingIOError:
            pass

    def _complete_publications(self) -> None:
        """Publish completed host bytes without waiting in the Worker thread.

        Runs on the publication thread. Each queued device publication's
        stream signal is watched until its device-to-host copy completes;
        the segment's readiness word is then set and the publication marked
        complete. After the exit request, the loop keeps running until every
        watched signal has fired, so no segment is left pending.
        """
        selector = selectors.DefaultSelector()
        selector.register(self._publication_control_rx, selectors.EVENT_READ)
        closing = False
        try:
            while not closing or len(selector.get_map()) > 1:
                for key, _events in selector.select():
                    if key.fileobj is self._publication_control_rx:
                        # Drain the wakeup, then register each queued
                        # publication's stream signal for readiness.
                        while True:
                            try:
                                if not self._publication_control_rx.recv(4096):
                                    closing = True
                                    break
                            except BlockingIOError:
                                break
                        while True:
                            try:
                                item = self._publication_queue.get_nowait()
                            except queue.Empty:
                                break
                            if item is None:
                                closing = True
                            else:
                                selector.register(
                                    item[1].signal, selectors.EVENT_READ, item
                                )
                        continue

                    # A stream signal fired: the device-to-host DMA into the
                    # segment is done, so expose (or fail) the publication.
                    locator, source = key.data
                    selector.unregister(source.signal)
                    failure = None
                    try:
                        source.signal.consume()
                    except BaseException as error:
                        device_completed = False
                        failure = error
                    else:
                        device_completed = True
                    source.signal = None
                    # The readiness word is written after the payload, with
                    # release ordering, so a consumer that sees it sees the
                    # bytes it announces.
                    segment.set_state(
                        source.shm.buf,
                        segment.READY if failure is None else segment.FAILED,
                    )
                    self._publications.complete(
                        locator,
                        error=failure,
                        producer_completed=device_completed,
                    )
                    if self._completion_wake is not None:
                        self._completion_wake()
        finally:
            selector.close()

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Publish into a segment whose header carries its readiness.

        A host source is copied synchronously and the segment is ready on
        return. A device source is copied into the page-locked segment on the
        current stream; the publication thread marks the segment ready once
        that copy completes, and the source is recorded on the stream so its
        storage outlives the copy. ``consumers`` names the acknowledgment
        slots whose words must settle before the segment is unlinked.

        Raises:
            WorkerError: `invalid_descriptor` when `publication_views`
                refuses the source or the locator's identity is already
                registered; `resource_error` when transfer bytes or
                the publication table are exhausted, the table is closing,
                or the CUDA driver refuses to register the segment.
                Allocation and copy errors propagate.
        """
        import torch

        source, shape, offset = publication_views(tensor, offset)
        first = source[0] if isinstance(source, tuple) else source
        nbytes = tensor_nbytes(source)
        # Capacity acknowledged since the last sweep is reclaimed first.
        self._publications.reap()
        self._bytes.acquire(nbytes)

        shm = None
        registered = False
        submitted = False
        try:
            shm = allocate_shared_storage(segment.HEADER_BYTES + max(1, nbytes))
            buffer = shm.buf
            if buffer is None:
                raise RuntimeError(
                    "shared storage publication has no writable buffer"
                )
            locator = Locator(
                source=self.source,
                transport=PosixShmTransfer(
                    endpoint=self.endpoint(),
                    name=shm.name,
                ),
                nbytes=nbytes,
                dtype=dtype_name(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            # The header names the exact view this segment holds, so a
            # consumer holding a locator for another view refuses it without
            # asking this rank.
            segment.initialize(buffer, locator_digest(locator))
            payload = buffer[segment.HEADER_BYTES :]

            packed = torch.frombuffer(payload, dtype=first.dtype).reshape(shape)
            address = None
            if first.is_cuda:
                # Device bytes land in the segment itself: its mapping is
                # page-locked for the copy, and the stream signal marks the
                # DMA complete on the publication thread.
                from uniserve_worker._uniserve_ipc import StreamSignal

                address = _register_segment(buffer)
                signal = StreamSignal()
            else:
                signal = None
                for target, value in copy_pairs(source, packed):
                    target.copy_(value)
                # `target` views the segment's mapping. No such view may
                # outlive the mapping: `SharedMemory.close` raises
                # `BufferError` while any view of it is alive.
                del target, value
                segment.set_state(buffer, segment.READY)

            publication = _ShmSource(
                shm,
                nbytes,
                tuple(int(slot) for slot in consumers),
                signal,
                address,
            )
            self._publications.publish(
                locator, publication, pending=first.is_cuda
            )
            registered = True

            if first.is_cuda:
                assert signal is not None
                submitted = True
                from uniserve_kernels.peer_storage import copy_host_device

                stream = torch.cuda.current_stream(first.device)
                for target, value in copy_pairs(source, packed):
                    copy_host_device(target, value, stream)
                    value.record_stream(stream)
                signal.schedule(int(stream.cuda_stream))
                self._queue_publication((locator, publication))
            del packed, payload
            return locator
        except BaseException:
            # Once registered, the publication table owns the segment: the
            # publication is retired, and a device publication is marked
            # failed and complete so `_reclaim` can unlink it. Before that,
            # the segment and bytes are returned here.
            if registered:
                self._publications.release(locator)
                if submitted:
                    # The segment stays mapped and registered until the copy
                    # into it has retired.
                    torch.cuda.current_stream(first.device).synchronize()
                if first.is_cuda:
                    assert buffer is not None
                    segment.set_state(buffer, segment.FAILED)
                    self._publications.complete(locator)
            else:
                if shm is not None:
                    shm.close()
                    shm.unlink()
                self._bytes.release(nbytes)
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
    ) -> TransferTicket:
        """Submit a read of a segment published on this node.

        Raises:
            WorkerError: `invalid_descriptor` when the publication lies on
                another node or, with a `destination`, when `region` exceeds
                the published view or `destination` does not match it;
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
        )

    def borrow(
        self, locator: Locator, region: tuple[slice, ...] | None = None
    ) -> HostBorrow:
        """Expose a published payload in place for a reader on this host.

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

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        if not isinstance(locator.transport, PosixShmTransfer):
            raise invalid_descriptor(
                "shared storage release requires a shared storage locator"
            )
        return self._publications.release(locator)

    def close(self) -> None:
        # Reads drain first, then the publication thread exits once every
        # pending device copy has signaled, and only then are the
        # publications retired; `Publications.close` raises `resource_error`
        # when any source is still retained.
        try:
            self._reads.close()
        finally:
            if not self._closed:
                self._queue_publication(None)
                self._publication_worker.join()
                self._publication_control_rx.close()
                self._publication_control_tx.close()
                self._closed = True
            self._publications.close()
