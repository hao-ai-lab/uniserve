"""Bounded local, shared-memory and CUDA VMM product transport.

Backends publish canonical typed locators. A descriptor carries the physical
handle and readiness fence; asynchronous reads establish access to its bytes.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import mmap
import os
import queue
import selectors
import socket
import sys
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from itertools import groupby, repeat
from typing import TYPE_CHECKING, Any, ClassVar

from uniserve import _slices
from uniserve.profiling import profile_range
from uniserve.runtime import EventPool

from ..foundation.errors import (
    invalid_descriptor,
    resource_error,
    unsupported_setup,
)
from ..foundation.shared_memory import allocate_shared_memory
from ..protocol.transfer import (
    DESCRIPTOR_HANDLE_BYTES,
    FABRIC_HANDLE_BYTES,
    ChannelTransfer,
    CudaVmmTransfer,
    LocalTransfer,
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from . import descriptor_grants, segment, vmm_pool
from .descriptor_grants import DescriptorGrants
from .endpoint import Publications, locator_digest
from .layout import region_view, validate_destination
from .vmm_pool import ACK_WORD_BYTES, PoolChunk, PoolExhaustedError, VmmPool

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch

__all__ = [
    "Transport",
    "TransferTicket",
    "LocalTransport",
    "ShmTransport",
    "CudaVmmTransport",
    "TransportKind",
    "TRANSPORTS",
]


class TransportKind(StrEnum):
    """Selects in-process, shared-memory, device, or rank-channel transport."""

    LOCAL = "local"
    SHM = "shm"
    CUDA_VMM = "cuda_vmm"
    CHANNEL = "channel"


TRANSPORTS = tuple(kind.value for kind in TransportKind)


@cache
def _can_access_peer(device: str, peer: str) -> bool:
    """Return the process-stable CUDA peer relation for two visible devices."""
    import torch

    return torch.cuda.can_device_access_peer(
        torch.device(device), torch.device(peer)
    )


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Encode a torch dtype as its unqualified transport name."""
    return str(dtype).removeprefix("torch.")


def _dtype_from_str(name: str) -> torch.dtype:
    """Resolve a transport dtype name to a torch dtype."""
    import torch

    return getattr(torch, name)


def _nbytes(tensor: torch.Tensor | tuple[torch.Tensor, ...]) -> int:
    """Return the physical byte size of a tensor view."""
    spans = tensor if isinstance(tensor, tuple) else (tensor,)
    return sum(int(span.numel() * span.element_size()) for span in spans)


def _read_destination(
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
    dtype = _dtype_from_str(locator.dtype)
    if destination is None:
        return torch.empty(_slices.shape(region), dtype=dtype, device=device)
    validate_destination(
        destination,
        shape=_slices.shape(region),
        dtype=locator.dtype,
        device=device,
    )
    return destination


def _publication_views(
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


def _copy_pairs(
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


class Transport(ABC):
    """Bounded physical publications and asynchronous reads per endpoint."""

    name: ClassVar[str]
    source: WorkerEndpoint

    @abstractmethod
    def endpoint(self) -> str:
        """Return the publishing address-space incarnation."""

    @abstractmethod
    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Expose a descriptor and producer fence for an immutable version.

        The allocation owner must retain the published range, without writes,
        until publication_retirement() completes after release(). Keeping a
        tensor reference does not authorize reuse of an arena or page range.

        `consumers` are the acknowledgment slots of the ranks that read this
        publication, as the head stated them on the producing call. A
        mechanism that holds storage another process reads returns it once
        each has acknowledged; a mechanism whose consumers are in this process
        or hold their own copy has nothing to wait for and ignores them.
        """

    def serves(self, consumers: Sequence[int]) -> bool:
        """Whether this mechanism reaches the named consumers.

        A publication is made over each mechanism that reaches one of the
        acknowledgment slots the producing call names; a call that names none
        is published over every mechanism the rank binds. A mechanism that
        reaches every consumer wherever it runs serves any call.
        """
        del consumers
        return True

    @abstractmethod
    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
    ) -> TransferTicket:
        """Read into one tensor or ordered first-axis spans on the device.

        Spans must be writable, disjoint and cover the exact requested region
        without dtype conversion. Backend layout restrictions are checked before
        submission. One ticket and one completion fence cover the whole read.
        An omitted destination lets the backend allocate or borrow.
        """

    @abstractmethod
    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        """Retire a publication after its physical readers release ownership."""

    @abstractmethod
    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        """Observe physical ownership completion.

        The publication is not revoked.
        """

    def reap(self) -> None:
        """Release publications their consumers have finished acknowledging.

        A consumer acknowledges a product by writing into the storage it read,
        which reaches the producer with no local notification. A transport
        whose publications retire with their own producer has nothing to sweep.
        """

    def awaiting_acknowledgment(self) -> bool:
        """Report whether a retired publication still waits on a consumer.

        An acknowledgment arrives with no notification, so a producer with one
        outstanding sweeps on a short period rather than on its next event.
        """
        return False

    @abstractmethod
    def close(self) -> None:
        """Drain reads and release endpoint resources."""

    @abstractmethod
    def set_completion_wake(self, wake: Any) -> None:
        """Connect asynchronous readiness to the worker controller."""


# Address-space lookup shares existing owners; it does not own their lifetimes.
_endpoints: weakref.WeakValueDictionary[str, Transport] = (
    weakref.WeakValueDictionary()
)
_endpoint_lock = threading.Lock()


@cache
def _chunk_word(state: int) -> torch.Tensor:
    """Return the pinned host word a consumer writes into a chunk's header.

    A claim precedes the consumer's first read of the chunk and an
    acknowledgment follows its last, so a producing rank sweeping a retired
    publication can tell a consumer that is still reading from one that never
    began.
    """
    import torch

    return torch.full((1,), state, dtype=torch.int32).pin_memory()


class TransferTicket:
    """One read, available once its consuming stream can wait on a fence.

    Descriptor readiness does not imply device completion. The transport keeps
    physical source and mapping leases until its copy has actually completed.
    """

    def __init__(self, event_pool: EventPool) -> None:
        self._events = event_pool
        self._event: torch.cuda.Event | None = None
        self._error: BaseException | None = None
        self._cancelled = False
        self._state_lock = threading.RLock()

        # Physical retirement: the backend has stopped all access to source
        # and destination storage. Resources whose completion could not be
        # drained stay listed in _unretired and block retirement forever.
        self._unretired: tuple[object, ...] = ()
        self._retirement: concurrent.futures.Future[None] = (
            concurrent.futures.Future()
        )
        self._work: concurrent.futures.Future[None] | None = None

        # Borrowed-view consumption: streams that received the views and the
        # release that returns the source grant once they all complete.
        self._consumer_release: Any = None
        self._consumer_streams: dict[int, torch.cuda.Stream] = {}
        self._destination_stream: torch.cuda.Stream | None = None
        self._consumer_events: tuple[torch.cuda.Event, ...] = ()
        self._closed = False

        # Stream-readiness result: destination views plus the device fence a
        # consumer must wait on before touching them.
        self._future: concurrent.futures.Future[
            tuple[
                torch.Tensor | tuple[torch.Tensor, ...], torch.cuda.Event | None
            ]
        ] = concurrent.futures.Future()

    def ready(self) -> bool:
        """Query whether result() can establish stream access.

        No host wait is needed.
        """
        return self._future.done()

    def retired(self) -> bool:
        """Query whether the backend stopped access to the read's storage."""
        return self._retirement.done()

    def retirement_ready(self) -> bool:
        """Require known physical completion first.

        Only then can an allocation be acknowledged free.
        """
        if self._unretired:
            raise resource_error(
                "transfer physical completion is unknown"
            ) from self._error
        return self.retired()

    def add_retirement_callback(self, callback: Any) -> None:
        """Notify allocation owners after access finishes.

        Both physical access and acknowledgement must finish first.
        """

        def notify(_future: concurrent.futures.Future[None]) -> None:
            nonlocal callback
            try:
                callback()
            finally:
                callback = None

        self._retirement.add_done_callback(notify)

    def cancel(self) -> None:
        """Revoke consumption while retaining storage until the read retires."""
        with self._state_lock:
            self._cancelled = True
            if self._error is None:
                self._error = resource_error("transfer read was cancelled")
            if not self._future.done():
                self._future.set_exception(self._error)
        work = self._work
        if work is not None:
            work.cancel()

    def _require_active(self) -> None:
        """Stop a cancelled read before it opens or copies source storage."""
        with self._state_lock:
            if self._cancelled:
                assert self._error is not None
                raise self._error

    def _retire(self) -> None:
        self._retirement.set_result(None)

    def result(
        self, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Return destination views and order reads on the consumer stream."""
        if not self.ready():
            raise RuntimeError("transfer ticket was observed before readiness")
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("transfer consumption has already closed")

        value, event = self._future.result()
        if event is not None:
            import torch

            spans = value if isinstance(value, tuple) else (value,)
            device = spans[0].device
            consumer = (
                torch.cuda.current_stream(device) if stream is None else stream
            )
            if consumer.device != device:
                raise invalid_descriptor(
                    "transfer consumer stream is on another device"
                )
            consumer.wait_event(event)
            for span in spans:
                span.record_stream(consumer)
            if self._consumer_release is not None:
                # Borrowed views: track every consuming stream so close() can
                # fence each one before returning the source grant.
                self._consumer_streams[int(consumer.cuda_stream)] = consumer
        return value

    def close(self) -> None:
        """End a borrowed-view read after work submitted by its consumers.

        Records one fence on every consumer stream observed by result(); the
        source grant returns only after all of those fences complete.
        """
        if self._consumer_release is None or self._closed:
            return
        self._closed = True
        import torch

        events = []
        for stream in self._consumer_streams.values():
            with torch.cuda.device(stream.device), torch.cuda.stream(stream):
                event = self._events.acquire(stream.device)
                self._events.retain(event, stream.device)
                self._events.record(event, stream.device)
                self._events.schedule_completion_wake(stream.device, event)
                events.append(event)
        self._consumer_events = tuple(events)

        if events:
            self._events.defer_release(
                events, self, completed=self.events_released
            )
        else:
            self.events_released()

    def events_released(self) -> None:
        """Return source grant after every borrowed-view consumer completed."""
        release = self._consumer_release
        self._consumer_release = None
        self._consumer_streams.clear()
        self._consumer_events = ()
        if release is not None:
            release()
            self._retire()

    def _drain_consumers(self) -> None:
        """Drain borrowed-view fences during transport shutdown."""
        self.close()
        for event in self._consumer_events:
            event.synchronize()
        self._events.reap()

    def add_done_callback(self, callback: Any) -> None:
        """Notify the owner when stream access or an error is observable."""

        def notify(_future: object) -> None:
            nonlocal callback
            try:
                callback()
            finally:
                callback = None

        self._future.add_done_callback(notify)

    def _complete(
        self,
        value: torch.Tensor | tuple[torch.Tensor, ...],
        event: torch.cuda.Event | None = None,
    ) -> None:
        with self._state_lock:
            if event is not None:
                device = (
                    value[0].device
                    if isinstance(value, tuple)
                    else value.device
                )
                self._events.retain(event, device)
                self._event = event
            # Cancellation may already have exposed an error. The backend still
            # owns a started copy and its fence through physical retirement.
            if not self._future.done():
                self._future.set_result((value, event))

    def _fail(self, error: BaseException) -> bool:
        """Preserve failures after stream readiness.

        Submission failures are preserved as well.
        """
        with self._state_lock:
            late = self._future.done() and self._future.exception() is None
            self._error = error
            if not self._future.done():
                self._future.set_exception(error)
            return late

    def _retain_failed_read(self, *resources: object) -> None:
        """Keep allocations whose device access could not be drained."""
        self._unretired = resources

    def __del__(self) -> None:
        self.close()
        if self._event is not None:
            self._events.defer_release((self._event,), self._future)


class TransferCapacity:
    """Share a Worker rank's byte and read-ticket budget across its backends."""

    def __init__(self, byte_capacity: int, ticket_capacity: int) -> None:
        """Initialize reservations against a fixed positive capacity."""
        self.capacity = int(byte_capacity)
        self.ticket_capacity = int(ticket_capacity)
        if min(self.capacity, self.ticket_capacity) < 1:
            raise ValueError("transfer byte capacity must be positive")
        self.read_slots = threading.BoundedSemaphore(self.ticket_capacity)
        self.used = 0
        self._lock = threading.Lock()

    def acquire(self, amount: int) -> None:
        """Reserve bytes if capacity is available, else report backpressure."""
        value = int(amount)
        if value < 0:
            raise ValueError("transfer byte reservation must not be negative")
        with self._lock:
            projected = self.used + value
            if projected > self.capacity:
                raise resource_error(
                    f"transfer byte capacity is exhausted "
                    f"({projected}>{self.capacity})"
                )
            self.used = projected

    def release(self, amount: int) -> None:
        """Return bytes after the physical owner releases its allocation."""
        value = int(amount)
        with self._lock:
            if value < 0 or value > self.used:
                raise RuntimeError(
                    "transfer byte release exceeds the live reservation"
                )
            self.used -= value


def _row_span(
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
        (), dtype=_dtype_from_str(locator.dtype)
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


_SHM_LIBC = ctypes.CDLL(None, use_errno=True)
_SHM_LIBC.shm_open.restype = ctypes.c_int


def _open_shared_memory(name: str, size: int) -> mmap.mmap:
    """Open an existing shared-memory segment.

    The mapping is writable, because a consumer writes its own acknowledgment
    word in the segment's header once it has copied the payload out.
    """
    canonical_name = name if name.startswith("/") else f"/{name}"
    descriptor = _SHM_LIBC.shm_open(canonical_name.encode(), os.O_RDWR)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), canonical_name)
    try:
        return mmap.mmap(
            descriptor,
            int(size),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    finally:
        os.close(descriptor)


class _BoundedTransferPool:
    """Bounds asynchronous transfers for one transport backend.

    Both transfer count and aggregate bytes are bounded.
    """

    def __init__(
        self,
        *,
        workers: int,
        capacity: TransferCapacity,
        name: str,
        event_pool: EventPool,
    ) -> None:
        """Create a worker pool governed by byte and entry reservations."""
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=name,
        )
        self._entries = capacity.read_slots
        self._bytes = capacity
        self._events = event_pool
        self._completion_wake: Any = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._unretired: list[TransferTicket] = []
        self._read_streams: dict[tuple[int, str], torch.cuda.Stream] = {}

    def set_completion_wake(self, wake: Any) -> None:
        """Install the controller callback for transfer completion.

        The callback is invoked after an asynchronous transfer finishes.
        """
        self._completion_wake = wake

    def submit(
        self,
        call: Any,
        *args: Any,
        nbytes: int,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    ) -> TransferTicket:
        """Reserve a read and retain the caller's destination stream."""
        import torch

        with self._lock:
            if self._error is not None:
                raise self._error

        if not self._entries.acquire(blocking=False):
            raise resource_error(
                "asynchronous transfer ticket capacity is exhausted"
            )
        try:
            self._bytes.acquire(nbytes)
        except BaseException:
            self._entries.release()
            raise

        ticket = TransferTicket(self._events)
        if destination is not None:
            first = (
                destination[0]
                if isinstance(destination, tuple)
                else destination
            )
            if first.is_cuda:
                ticket._destination_stream = torch.cuda.current_stream(
                    first.device
                )

        def run() -> None:
            import torch

            try:
                ticket._require_active()
                # Inference mode is thread-local. Destinations reserved by an
                # inference caller retain that behavior on transport threads.
                with torch.inference_mode():
                    call(ticket, *args)
            except BaseException as error:
                late = ticket._fail(error)
                if late or ticket._unretired:
                    with self._lock:
                        if self._error is None:
                            self._error = error
                        if ticket._unretired:
                            self._unretired.append(ticket)
                    if late and self._completion_wake is not None:
                        self._completion_wake()

        def finished(work: concurrent.futures.Future[None]) -> None:
            ticket._work = None
            # A cancelled executor task never enters run(), so credits and
            # destination lifetime must be settled by its terminal callback.
            if not ticket._unretired:
                self._bytes.release(nbytes)
                self._entries.release()
                ticket._retire()
            if work.cancelled():
                ticket._fail(
                    resource_error(
                        "transfer read was cancelled before submission"
                    )
                )

        if self._completion_wake is not None:
            ticket.add_done_callback(self._completion_wake)
            ticket.add_retirement_callback(self._completion_wake)

        try:
            work = self._executor.submit(run)
        except BaseException:
            self._bytes.release(nbytes)
            self._entries.release()
            raise
        ticket._work = work
        work.add_done_callback(finished)
        return ticket

    def copy(
        self,
        ticket: TransferTicket,
        source: torch.Tensor | tuple[torch.Tensor, ...],
        destination: torch.Tensor | tuple[torch.Tensor, ...],
        producer: torch.cuda.Event | None = None,
        acknowledgment: torch.Tensor | None = None,
    ) -> None:
        """Copy into a reserved view.

        All storage is retained through device completion.

        `acknowledgment` is this rank's word in the source chunk's header. It
        is written after the copies on the same stream, so the producer sees it
        only once every read of that chunk has completed.
        """
        import torch

        ticket._require_active()
        spans = (
            destination if isinstance(destination, tuple) else (destination,)
        )
        pairs = tuple(_copy_pairs(source, destination))
        device = spans[0].device

        if device.type != "cuda":
            if acknowledgment is not None:
                acknowledgment.copy_(_chunk_word(vmm_pool.CLAIMED))
            for target, value in pairs:
                target.copy_(value)
            if acknowledgment is not None:
                acknowledgment.copy_(_chunk_word(vmm_pool.ACKNOWLEDGED))
            ticket._complete(destination)
            return

        # Each transport thread reuses one dedicated copy stream per device.
        key = (threading.get_ident(), str(device))
        stream = self._read_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._read_streams[key] = stream

        completed = None
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                if acknowledgment is not None:
                    # The claim lands before any copy is submitted, so a
                    # producer sweeping a retired publication cannot hand the
                    # chunk out again while this read is in flight. The read
                    # stream is idle here, this thread having synchronized it
                    # at the end of its previous read, so the four-byte
                    # blocking copy waits on nothing.
                    acknowledgment.copy_(_chunk_word(vmm_pool.CLAIMED))
                # The caller may still be initializing or consuming this
                # backing. Establish its handoff before exposing copy
                # readiness, so a later caller wait on our completion cannot
                # create a dependency cycle.
                if ticket._destination_stream is not None:
                    stream.wait_stream(ticket._destination_stream)
                    ticket._destination_stream = None
                if producer is not None:
                    stream.wait_event(producer)
                for target, value in pairs:
                    if value.device.type == "cpu":
                        from uniserve_kernel.peer_memory import copy_host_device

                        copy_host_device(target, value, stream)
                    else:
                        target.copy_(value, non_blocking=True)
                if acknowledgment is not None:
                    # A pinned host word makes this a memcpy on the read
                    # stream. Filling the word would launch a kernel, and the
                    # first launch of one in a process pays CUDA module
                    # loading, which a one-word acknowledgment should not.
                    acknowledgment.copy_(
                        _chunk_word(vmm_pool.ACKNOWLEDGED), non_blocking=True
                    )
                completed = self._events.acquire(device)
                self._events.record(completed, device)

            ticket._complete(destination, completed)
            completed.synchronize()
        except BaseException as error:
            ticket._fail(error)
            raise
        finally:
            # A stream that cannot be drained keeps every allocation it may
            # still be touching; the ticket never reports physical retirement.
            try:
                stream.synchronize()
            except BaseException:
                ticket._retain_failed_read(
                    destination, source, producer, completed, stream
                )
                raise

    def close(self) -> None:
        """Drain reads and report failure following consumable completion."""
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._read_streams.clear()
        if self._error is not None:
            raise self._error


@dataclass(slots=True)
class _LocalSource:
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
        self.capacity.release(_nbytes(self.tensor))
        self.retirement.set_result(None)


class LocalTransport(Transport):
    """Same process, zero copy. The locator is a counter into a local table."""

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
        self._borrow_slots = capacity.read_slots
        self._events = event_pool
        self._next = 0
        self._completion_wake: Any = None
        self._lock = threading.RLock()
        self._endpoint = f"local:{uuid.uuid4().hex}"
        with _endpoint_lock:
            _endpoints[self._endpoint] = self
        self._bytes = capacity
        self._reads = _BoundedTransferPool(
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
        t, shape, offset = _publication_views(tensor, offset)
        first = t[0] if isinstance(t, tuple) else t
        self._events.reap()
        nbytes = _nbytes(t)
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
                dtype=_dtype_to_str(first.dtype),
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
                else _read_destination(locator, device, destination, region)
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
    signal: Any = None
    #: Address of the segment's mapping while it is registered with the
    #: CUDA driver, so a device-to-host copy lands in it directly.
    registered: int | None = None


def _register_segment(buffer: memoryview) -> int:
    """Page-lock a segment's mapping so device copies land in it directly.

    Returns the mapping's address, which the unregistration needs. The
    mapping is page-aligned and page-sized, as every shared-memory mapping
    is, which the driver requires of a registered range.
    """
    import torch

    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    status = torch.cuda.cudart().cudaHostRegister(address, len(buffer), 0)
    if int(status) != 0:
        raise resource_error(
            f"registering a shared-memory segment with the CUDA driver "
            f"failed with status {int(status)}"
        )
    return address


def _unregister_segment(address: int) -> None:
    import torch

    torch.cuda.cudart().cudaHostUnregister(address)


@dataclass(slots=True)
class HostBorrow:
    """A consumer's direct view of a published segment's bytes.

    The bytes stay in the producer's segment: a codec process maps them by
    name and offset. ``release`` writes this rank's acknowledgment word once
    every read of them is done, which lets the producer retire the segment.
    """

    segment: str
    offset: int
    nbytes: int
    _release: Callable[[], None]

    def release(self) -> None:
        release, self._release = self._release, lambda: None
        release()


class ShmTransport(Transport):
    """Shared-memory publication whose segment carries its own readiness.

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
        self._publications = Publications[_ShmSource](
            capacity=256,
            reclaim=self._reclaim,
            drain=lambda source: None,
            settled=self._settled,
        )
        # This rank's own word in the header of every segment it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        self.source = source or WorkerEndpoint.local()
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
        self._reads = _BoundedTransferPool(
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
        """Publish completed host bytes without waiting in the Worker thread."""
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
        """Publish into a segment whose header carries its readiness."""
        import torch

        source, shape, offset = _publication_views(tensor, offset)
        first = source[0] if isinstance(source, tuple) else source
        nbytes = _nbytes(source)
        # Capacity acknowledged since the last sweep is reclaimed first.
        self._publications.reap()
        self._bytes.acquire(nbytes)

        shm = None
        registered = False
        submitted = False
        try:
            shm = allocate_shared_memory(segment.HEADER_BYTES + max(1, nbytes))
            buffer = shm.buf
            if buffer is None:
                raise RuntimeError(
                    "shared-memory publication has no writable buffer"
                )
            locator = Locator(
                source=self.source,
                transport=PosixShmTransfer(
                    endpoint=self.endpoint(),
                    name=shm.name,
                ),
                nbytes=nbytes,
                dtype=_dtype_to_str(first.dtype),
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
                from .._uniserve_ipc import StreamSignal

                address = _register_segment(buffer)
                signal = StreamSignal()
            else:
                signal = None
                for target, value in _copy_pairs(source, packed):
                    target.copy_(value)
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
                from uniserve_kernel.peer_memory import copy_host_device

                stream = torch.cuda.current_stream(first.device)
                for target, value in _copy_pairs(source, packed):
                    copy_host_device(target, value, stream)
                    value.record_stream(stream)
                signal.schedule(int(stream.cuda_stream))
                self._queue_publication((locator, publication))
            del packed, payload
            return locator
        except BaseException:
            if registered:
                self._publications.release(locator)
                if submitted:
                    # The segment stays mapped and registered until the copy
                    # into it has retired.
                    torch.cuda.current_stream(first.device).synchronize()
                if first.is_cuda:
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

        The copy holds the segment only while it runs; the bytes are held
        through the (possibly asynchronous) destination copy in this process.
        """
        import torch

        handle = locator.transport
        if not isinstance(handle, PosixShmTransfer):
            raise invalid_descriptor(
                "shared-memory read requires a shared-memory locator"
            )
        try:
            ticket._require_active()
            try:
                shm = _open_shared_memory(
                    handle.name, segment.HEADER_BYTES + locator.nbytes
                )
            except FileNotFoundError:
                raise invalid_descriptor(
                    "publication is retired, invalid, or belongs to another "
                    "view"
                ) from None
            try:
                header = memoryview(shm)
                if segment.digest(header) != locator_digest(locator):
                    raise invalid_descriptor(
                        "publication is retired, invalid, or belongs to "
                        "another view"
                    )
                # The claim precedes the first read and the acknowledgment
                # follows the copy, both with release ordering, so the
                # producer reclaims nothing this rank still reads and waits
                # for no rank that never began.
                segment.claim(header, self._acknowledgment_slot)
                segment.await_ready(header, check=ticket._require_active)
                payload = segment.HEADER_BYTES
                buf = bytearray(shm[payload : payload + locator.nbytes])
                segment.acknowledge(header, self._acknowledgment_slot)
            finally:
                header.release()
                shm.close()
        except BaseException as error:
            ticket._fail(error)
            raise
        source = torch.frombuffer(
            buf, dtype=_dtype_from_str(locator.dtype)
        ).reshape(locator.shape)
        if device.type == "cuda":
            # The read ticket retains this bounded pinned buffer until DMA
            # retires.
            pinned = torch.empty(
                source.shape, dtype=source.dtype, pin_memory=True
            )
            pinned.copy_(source)
            source = pinned

        target = _read_destination(locator, device, destination, region)
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
        if locator.source.node != self.source.node:
            raise invalid_descriptor(
                "shared-memory transport requires the source node"
            )
        target = (
            None
            if destination is None
            else _read_destination(locator, device, destination, region)
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
        """
        handle = locator.transport
        if not isinstance(handle, PosixShmTransfer):
            raise invalid_descriptor(
                "shared-memory borrow requires a shared-memory locator"
            )
        if locator.source.node != self.source.node:
            raise invalid_descriptor(
                "shared-memory transport requires the source node"
            )
        start, nbytes = _row_span(locator, region)
        try:
            shm = _open_shared_memory(
                handle.name, segment.HEADER_BYTES + locator.nbytes
            )
        except FileNotFoundError:
            raise invalid_descriptor(
                "publication is retired, invalid, or belongs to another view"
            ) from None
        header = memoryview(shm)
        try:
            if segment.digest(header) != locator_digest(locator):
                raise invalid_descriptor(
                    "publication is retired, invalid, or belongs to another "
                    "view"
                )
            # The claim precedes the reader's first use of the payload.
            segment.claim(header, self._acknowledgment_slot)
            segment.await_ready(header)
        except BaseException:
            header.release()
            shm.close()
            raise

        def release() -> None:
            # The word is written after the reader's use of the payload, with
            # release ordering, so the producer reclaims nothing still read.
            try:
                segment.acknowledge(header, self._acknowledgment_slot)
            finally:
                header.release()
                shm.close()

        return HostBorrow(
            handle.name, segment.HEADER_BYTES + start, nbytes, release
        )

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        if not isinstance(locator.transport, PosixShmTransfer):
            raise invalid_descriptor(
                "shared-memory release requires a shared-memory locator"
            )
        return self._publications.release(locator)

    def close(self) -> None:
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


@dataclass(slots=True)
class _CudaSource:
    """Retain publication bytes through the producer's final device access."""

    tensor: torch.Tensor | tuple[torch.Tensor, ...]
    event: torch.cuda.Event
    nbytes: int
    capacity: TransferCapacity
    handle: bytes
    copied_source: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    retirement: concurrent.futures.Future[None] | None = None
    #: Pool and chunk this publication occupies, when it came from a pool.
    pool: VmmPool | None = None
    chunk: PoolChunk | None = None
    #: Acknowledgment slots of the ranks that read this publication.
    consumers: tuple[int, ...] = ()
    #: Grant table this publication's descriptor is lent from, on a device
    #: that exports descriptors rather than fabric handles.
    grants: DescriptorGrants | None = None
    publication_id: str = ""

    def events_released(self) -> None:
        # A fabric handle is bytes the publication carried and this rank owns
        # nothing. A descriptor is an open file of this process, and a direct
        # export's belongs to the publication: the engine retires a
        # publication only once no named reader is still reading it, so the
        # grant and the descriptor end here together.
        #
        # A chunk's descriptor belongs to its pool, and a chunk outlives its
        # source by design -- the source was copied into it and is released
        # while consumers are still reading. Withdrawing that grant here would
        # refuse a reader that has not imported yet, so it is withdrawn where
        # the chunk returns to the pool instead.
        if (
            self.grants is not None
            and self.publication_id
            and self.pool is None
        ):
            self.grants.release(self.publication_id)
            os.close(int.from_bytes(self.handle, sys.byteorder))

        # This ends the source's lifetime, not the chunk's. A pool publication
        # was copied into its chunk, so the source is the producer's to reuse
        # as soon as its own fence drains, however long consumers keep reading
        # the chunk. The chunk returns separately, once they acknowledge it.
        self.capacity.release(self.nbytes)
        if self.retirement is not None:
            self.retirement.set_result(None)


class ChannelTransport(Transport):
    """Host products carried on the rank channel's data path.

    Shared memory names a segment in one host's namespace, so it cannot serve
    a consumer on another host. This transport puts the product's bytes in its
    locator instead: they travel in the producing rank's result, into the
    head's custody, and out in the consuming rank's batch, reaching wherever
    the rank channel does.

    The producing rank owns nothing after publication. The bytes are copied out
    of its storage while it publishes, so the source is its own again as soon
    as the locator exists, and the head releases its copy when the buffer is
    freed. That is what the acknowledgment discipline reduces to when the head
    is the intermediary: it already holds the product exactly as long as some
    consumer may still be given it.
    """

    name = "channel"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
        host_slots: Sequence[int] = (),
    ) -> None:
        self._bytes = capacity
        self._events = event_pool
        self.source = source or WorkerEndpoint.local()
        # Slots of the ranks on this host, which shared memory reaches; the
        # channel carries a product only for a consumer elsewhere.
        self._host_slots = frozenset(int(slot) for slot in host_slots)
        self._endpoint = f"uniserve-channel-{uuid.uuid4().hex}"
        self._reads = _BoundedTransferPool(
            workers=1,
            capacity=capacity,
            name="uniserve-channel-read",
            event_pool=event_pool,
        )
        # What carrying a product on the channel costs this rank. A publication
        # blocks on its own stream and then copies the bytes out, and both are
        # on the batch's critical path, so each is counted separately from the
        # payload they move.
        self._published = 0
        self._payload_bytes = 0
        self._largest_payload = 0
        self._synchronize_seconds = 0.0
        self._longest_synchronize = 0.0
        self._copy_seconds = 0.0
        self._fetched = 0
        self._fetch_seconds = 0.0

    def endpoint(self) -> str:
        return self._endpoint

    def serves(self, consumers: Sequence[int]) -> bool:
        return not consumers or any(
            int(slot) not in self._host_slots for slot in consumers
        )

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        # The head holds the bytes for its consumers and releases them with
        # the buffer, so nothing here waits for an acknowledgment.
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Copy the product into a locator that carries it."""
        import torch

        source, shape, offset = _publication_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        nbytes = _nbytes(source)
        self._bytes.acquire(nbytes)
        try:
            # One contiguous host buffer in physical tensor order. A device
            # product is staged through it, which is the same crossing a host
            # product would make to reach any consumer off this device.
            packed = torch.empty(shape, dtype=first.dtype, device="cpu")
            for target, value in _copy_pairs(source, packed):
                target.copy_(value)
            if first.is_cuda:
                # The producer waits here for its own writes: the bytes leave
                # with the result, so nothing downstream can fence them. This
                # is the publication's synchronize cost.
                started = time.perf_counter()
                with profile_range("channel_publish_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()
                waited = time.perf_counter() - started
                self._synchronize_seconds += waited
                self._longest_synchronize = max(
                    self._longest_synchronize, waited
                )

            started = time.perf_counter()
            with profile_range("channel_publish_payload"):
                payload = bytes(packed.flatten().view(torch.uint8).numpy())
            self._copy_seconds += time.perf_counter() - started
            self._published += 1
            self._payload_bytes += nbytes
            self._largest_payload = max(self._largest_payload, nbytes)

            return Locator(
                source=self.source,
                transport=ChannelTransfer(
                    endpoint=self._endpoint,
                    # A byte view, so a dtype NumPy does not model travels as
                    # readily as one it does.
                    payload=payload,
                ),
                nbytes=nbytes,
                dtype=_dtype_to_str(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
        finally:
            # The staging buffer is the only thing this rank held: the bytes
            # are in the locator by now, and the source is its own again.
            self._bytes.release(nbytes)

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
    ) -> TransferTicket:
        """Copy the locator's own bytes into a reserved destination."""
        import torch

        handle = locator.transport
        if not isinstance(handle, ChannelTransfer):
            raise invalid_descriptor("channel read requires a channel locator")
        target = _read_destination(locator, device, destination, region)
        started = time.perf_counter()
        with profile_range("channel_fetch_payload"):
            payload = torch.frombuffer(
                bytearray(handle.payload), dtype=_dtype_from_str(locator.dtype)
            ).reshape(locator.shape)
            # A device destination is filled by an asynchronous copy, which
            # reads pinned host storage; the bytes arrived pageable.
            if device.type == "cuda":
                carried = torch.empty_like(payload, pin_memory=True)
                carried.copy_(payload)
            else:
                carried = payload
        self._fetch_seconds += time.perf_counter() - started
        self._fetched += 1
        if region is not None:
            if not _slices.within(region, locator.shape):
                raise invalid_descriptor(
                    "read region exceeds the published view"
                )
            carried = region_view(carried, region)
        return self._reads.submit(
            self._reads.copy,
            carried,
            target,
            None,
            nbytes=locator.nbytes,
            destination=target,
        )

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        """Revoke a publication the rank no longer owns anything of."""
        self._require_own(locator)
        return None

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        """Expose completion, which publication itself established.

        The product was copied out of the rank's storage while it published,
        so there is nothing left to wait for and the source is reusable at
        once. The head holds the bytes from here, until the buffer is freed.
        """
        self._require_own(locator)
        settled: concurrent.futures.Future[None] = concurrent.futures.Future()
        settled.set_result(None)
        return settled

    def _require_own(self, locator: Locator) -> ChannelTransfer:
        """Return this endpoint's handle, or refuse another's."""
        handle = locator.transport
        if (
            not isinstance(handle, ChannelTransfer)
            or handle.endpoint != self._endpoint
        ):
            raise invalid_descriptor(
                "channel publication belongs to another endpoint"
            )
        return handle

    def close(self) -> None:
        """Report what the channel cost this rank, then release its reads."""
        if self._published or self._fetched:
            mean = (
                self._payload_bytes // self._published if self._published else 0
            )
            _LOG.info(
                "channel transport retired: published=%d payload_total=%d "
                "payload_mean=%d payload_max=%d synchronize_ms_total=%.3f "
                "synchronize_ms_max=%.3f copy_ms_total=%.3f fetched=%d "
                "fetch_ms_total=%.3f",
                self._published,
                self._payload_bytes,
                mean,
                self._largest_payload,
                self._synchronize_seconds * 1e3,
                self._longest_synchronize * 1e3,
                self._copy_seconds * 1e3,
                self._fetched,
                self._fetch_seconds * 1e3,
            )
        self._reads.close()


class CudaVmmTransport(Transport):
    """Device publications a consumer imports by their shareable handle.

    A consumer maps the producer's allocation from the handle the locator
    carries and, for a pool chunk, writes its acknowledgment word in the
    chunk's header once its copies retire. Where the device exports fabric
    handles nothing connects back to the producing rank, which is what lets a
    device product cross hosts.

    Where it exports POSIX descriptors instead, the published handle names an
    open file of the producing process and carries no meaning elsewhere, so a
    consumer asks this rank for it over the grant socket in
    `descriptor_grants`. Such a device cannot place an edge across hosts in
    any case, so that connection costs nothing the fabric case has.
    """

    name = "cuda_vmm"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
        acknowledgment_slot: int = 0,
        cross_host_consumers: bool = False,
    ) -> None:
        from uniserve_kernel.peer_memory import _extension

        _extension()
        self.source = source or WorkerEndpoint.local()
        self._events = event_pool
        self._failed_publication: (
            tuple[
                BaseException,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
                torch.cuda.Event | None,
                torch.Tensor | tuple[torch.Tensor, ...] | None,
            ]
            | None
        ) = None
        self._bytes = capacity
        # Grants exist only where the device exports descriptors, so the
        # socket is bound when the first such publication needs it.
        self._grants: DescriptorGrants | None = None
        # One bounded pool per device, reserved when that device first
        # publishes, so a rank reserves nothing on a device it never
        # publishes from. The pool is bounded by this rank's transfer byte
        # budget, which is what that budget already governs.
        self._pools: dict[str, VmmPool] = {}
        # This rank's own word in every chunk header it reads.
        self._acknowledgment_slot = int(acknowledgment_slot)
        # Whether a publication has to state readiness without an interprocess
        # event, which only a consumer on another host requires.
        self._cross_host_consumers = bool(cross_host_consumers)
        # What publishing costs this rank. A crossing publication drains the
        # producer's stream, because a consumer on another host can wait on no
        # fence this rank records, and that wait is on the batch's critical
        # path. The payloads are counted with it, since the cost of a crossing
        # is only interpretable against what it carries.
        self._published = 0
        self._payload_bytes = 0
        self._largest_payload = 0
        self._synchronize_seconds = 0.0
        self._longest_synchronize = 0.0
        # Publications whose fence has drained but whose consumers have not all
        # acknowledged. Their chunks are held until reap() finds them settled.
        self._unacknowledged: list[_CudaSource] = []
        # A source is released when its own fence drains: a pool publication
        # was copied into its chunk, and the chunk is what waits for the
        # consumers, swept separately in reap().
        self._publications = Publications[_CudaSource](
            capacity=256,
            reclaim=self._reclaim,
            drain=self._drain,
            # A device publication's chunk is held by the pool until its
            # readers finish, which the transport's own sweep decides; the
            # publication itself owes nothing once the producer's work is
            # complete.
            settled=lambda source: True,
        )
        self._reads = _BoundedTransferPool(
            workers=2,
            capacity=capacity,
            name="uniserve-cuda-read",
            event_pool=event_pool,
        )

        with _endpoint_lock:
            _endpoints[self.endpoint()] = self

    def endpoint(self) -> str:
        return self._publications.name

    def _descriptor_grants(self) -> DescriptorGrants:
        """Bind this address space's grant socket on first use."""
        if self._grants is None:
            self._grants = DescriptorGrants(self.endpoint())
        return self._grants

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        return self._publications.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def _release_chunk(self, source: _CudaSource) -> None:
        """Return one chunk to its pool and withdraw the grant that named it.

        The grant lends the pool's descriptor, which the pool keeps; what ends
        here is a consumer's right to ask for this chunk, which ends when the
        chunk can be handed out again.
        """
        assert source.pool is not None and source.chunk is not None
        if source.grants is not None and source.publication_id:
            source.grants.release(source.publication_id)
        source.pool.release(source.chunk)

    def _reclaim(
        self,
        source: _CudaSource,
        retirement: concurrent.futures.Future[None] | None = None,
    ) -> None:
        source.retirement = retirement
        if source.chunk is not None and source.pool is not None:
            if source.consumers:
                # A consumer may still be reading this chunk, and it will say
                # so by writing its word rather than by closing a connection.
                # Holding the chunk until then does not hold the source: that
                # was copied into the chunk and is released below.
                self._unacknowledged.append(source)
            else:
                # No other rank reads this product, so nothing can be waiting.
                self._release_chunk(source)
        self._events.defer_release(
            (source.event,), source, completed=source.events_released
        )

    def reap(self) -> None:
        """Return chunks whose consumers have finished acknowledging.

        The producing rank sweeps here rather than waiting on a per-reader
        connection, so a consumer on another host retires a product the same
        way one on this host does: by writing its slot's word in the chunk it
        mapped.
        """
        import torch

        held = [
            source
            for source in self._unacknowledged
            if source.pool is not None and source.chunk is not None
        ]
        if not held:
            return
        # Every held chunk's words are read in one transfer. Asking each chunk
        # separately would put one device-to-host synchronize per held
        # publication into every retirement pass.
        watched = [
            source.chunk.acknowledgments[list(source.consumers)]
            for source in held
            if source.chunk is not None
        ]
        observed = (
            torch.cat(watched).cpu().split([len(words) for words in watched])
        )

        waiting = []
        for source, words in zip(held, observed, strict=True):
            # A chunk returns once no named consumer is still reading it: one
            # that never claimed its word holds nothing, which is how a
            # product whose consuming call was never submitted retires.
            if bool((words != vmm_pool.CLAIMED).all()):
                self._release_chunk(source)
            else:
                waiting.append(source)
        self._unacknowledged = waiting

    def awaiting_acknowledgment(self) -> bool:
        return bool(self._unacknowledged)

    def _drain(self, source: _CudaSource) -> None:
        source.event.synchronize()
        self._events.reap()

    def _pool(self, device: torch.device) -> VmmPool:
        """Return this device's pool, reserving it on first publication."""
        key = str(device)
        pool = self._pools.get(key)
        if pool is None:
            pool = VmmPool(device, capacity_bytes=self._bytes.capacity)
            self._pools[key] = pool
        return pool

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
        consumers: Sequence[int] = (),
    ) -> Locator:
        """Export an immutable source and retain capacity.

        Capacity is retained until its producer fence retires. A product that
        can neither be exported where it lies nor fit its device's pool raises
        `PoolExhaustedError`, and the caller publishes it as bytes over the
        host mechanism instead.
        """
        import torch
        from uniserve_kernel.peer_memory import export_handle

        source, shape, offset = _publication_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        if not first.is_cuda:
            raise invalid_descriptor(
                "cuda_vmm transport requires a CUDA tensor"
            )
        if any(
            span.untyped_storage().data_ptr()
            != first.untyped_storage().data_ptr()
            or span.stride() != first.stride()
            for span in spans
        ):
            raise invalid_descriptor(
                "CUDA VMM publication spans require one allocation and stride"
            )
        if self._failed_publication is not None:
            raise self._failed_publication[0]

        self._events.reap()
        nbytes = _nbytes(tensor)
        self._bytes.acquire(nbytes)

        event = None
        publication = None
        descriptor = None
        copied_source = None
        try:
            # Storage that can be exported where it lies is published where
            # it lies, wherever it is read. That copies nothing, and it is
            # what lets a publication name a row whose bytes arrive later: an
            # encoded media unit is published with the batch that reserves its
            # row and filled when the encode completes, so a snapshot taken
            # now would carry whatever the row held before the encoder wrote
            # it. Where the device exports a fabric handle, the handle this
            # produces reaches another host too; what does not reach is the
            # fence, and that is settled below.
            exported = export_handle(first)
            pool = None
            chunk = None
            if exported is None:
                # Otherwise the product is materialized in this device's pool,
                # whose one handle a consumer imports once however many
                # products it reads from that device. Only the publication's
                # logical spans are materialized, never an enclosing allocator
                # segment. A product the pool cannot hold is the caller's to
                # publish over the host mechanism; the pool reports the
                # exhaustion once.
                pool = self._pool(first.device)
                try:
                    chunk = pool.reserve(_nbytes(tensor))
                except PoolExhaustedError:
                    self._bytes.release(nbytes)
                    raise
            if exported is not None:
                descriptor, storage_size, storage_offset = exported
            else:
                assert chunk is not None
                shared = chunk.storage.view(first.dtype).view(shape)
                for target, value in _copy_pairs(source, shared):
                    target.copy_(value, non_blocking=True)
                copied_source = source
                source = shared
                spans = (shared,)
                first = shared
                descriptor = pool.handle
                storage_size = pool.capacity
                # A consumer reads the payload, which follows the chunk's
                # acknowledgment words, so the offset names the payload.
                storage_offset = chunk.payload_offset
            # A publication hands its consumers an event wherever one can
            # reach them, which is every consumer on this host. A consumer
            # elsewhere can wait on nothing this rank records: an event handle
            # is host-local, and imported VMM memory admits no device-side
            # wait on current drivers. So the producer drains its stream
            # instead, and the publication carries no fence at all.
            interprocess = not self._cross_host_consumers
            if not interprocess:
                started = time.perf_counter()
                with profile_range("vmm_publish_synchronize"):
                    torch.cuda.current_stream(first.device).synchronize()
                waited = time.perf_counter() - started
                self._synchronize_seconds += waited
                self._longest_synchronize = max(
                    self._longest_synchronize, waited
                )
            self._published += 1
            self._payload_bytes += nbytes
            self._largest_payload = max(self._largest_payload, nbytes)
            event = self._events.acquire(
                first.device, interprocess=interprocess
            )
            self._events.retain(event, first.device)
            self._events.record(event, first.device)
            # The chunk returns to its pool once every rank the head named as
            # a reader of this call's products has written its acknowledgment,
            # which a consumer on another host can do as well as one here.
            readers = tuple(int(slot) for slot in consumers)
            publication_id = uuid.uuid4().hex
            grants = None
            if len(descriptor) == DESCRIPTOR_HANDLE_BYTES:
                # The handle is an open file of this process. A consumer can
                # only receive it over the grant socket, so it is registered
                # before the locator naming it leaves this rank.
                grants = self._descriptor_grants()
                grants.register(
                    publication_id, int.from_bytes(descriptor, sys.byteorder)
                )
            publication = _CudaSource(
                source,
                event,
                nbytes,
                self._bytes,
                descriptor,
                copied_source,
                pool=pool if chunk is not None else None,
                chunk=chunk,
                consumers=readers if chunk is not None else (),
                grants=grants,
                publication_id=publication_id,
            )

            # Run-length encode first-axis span lengths so the importer can
            # rebuild every span view without a per-span locator entry.
            length_runs = tuple(
                (length, sum(1 for _ in values))
                for length, values in groupby(
                    int(span.shape[0]) for span in spans
                )
            )
            locator = Locator(
                source=self.source,
                transport=CudaVmmTransfer(
                    endpoint=self.endpoint(),
                    publication_id=publication_id,
                    storage_size_bytes=storage_size,
                    storage_offsets_bytes=tuple(
                        storage_offset + span.data_ptr() - first.data_ptr()
                        for span in spans
                    ),
                    span_lengths=tuple(length for length, _ in length_runs),
                    span_counts=tuple(count for _, count in length_runs),
                    tensor_stride=tuple(first.stride()),
                    ready_event_handle=(
                        bytes(event.ipc_handle()) if interprocess else b""
                    ),
                    # A fabric handle travels with the publication and a
                    # consumer imports it directly, anywhere in the fabric
                    # domain. A descriptor travels only so a consumer can tell
                    # which kind it is; the usable one comes from the grant.
                    allocation_handle=descriptor,
                    # A consumer writes its own slot's word here once its reads
                    # retire. A publication outside the pool carries no header.
                    acknowledgment_offset=(
                        chunk.offset if chunk is not None else -1
                    ),
                ),
                nbytes=nbytes,
                dtype=_dtype_to_str(first.dtype),
                shape=shape,
                offset=offset,
                device=str(first.device),
            )
            self._publications.publish(locator, publication)
            return locator
        except BaseException:
            if publication is not None:
                self._reclaim(publication)
            else:
                try:
                    # No usable producer fence exists on this failure path.
                    # Keep its allocation and quota if draining also fails.
                    torch.cuda.current_stream(first.device).synchronize()
                    if event is not None:
                        self._events.defer_release((event,), source)
                except BaseException as error:
                    self._failed_publication = (
                        error,
                        source,
                        event,
                        copied_source,
                    )
                    raise
                self._bytes.release(nbytes)
            raise

    def fetch(
        self,
        locator: Locator,
        *,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        region: tuple[slice, ...] | None = None,
    ) -> TransferTicket:
        if not isinstance(locator.transport, CudaVmmTransfer):
            raise invalid_descriptor(
                "CUDA VMM read requires a CUDA VMM locator"
            )
        # A device product reaches another host as a fabric handle and not as
        # a process descriptor, which names an allocation only within the host
        # that exported it. The head refuses an edge whose devices cannot
        # export one, so this is the rank restating what it was given.
        if (
            locator.source.node != self.source.node
            and len(locator.transport.allocation_handle) != FABRIC_HANDLE_BYTES
        ):
            raise invalid_descriptor(
                "a device product crosses hosts only as a fabric handle"
            )
        if device.type != "cuda":
            raise invalid_descriptor(
                "CUDA VMM destination must be a CUDA device"
            )
        target = (
            None
            if destination is None
            else _read_destination(locator, device, destination, region)
        )
        return self._reads.submit(
            self._read,
            locator,
            device,
            target,
            region,
            nbytes=locator.nbytes,
            destination=target,
        )

    def _read(
        self,
        ticket: TransferTicket,
        locator: Locator,
        device: torch.device,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None,
        region: tuple[slice, ...] | None,
    ) -> None:
        import torch
        from uniserve_kernel.peer_memory import import_handle

        handle = locator.transport
        assert isinstance(handle, CudaVmmTransfer)
        # No consumer can check that a locator still names a publication the
        # producer holds; the locator is the engine's word. The engine binds
        # only locators the producing rank reported to it, and frees a
        # product's buffer only once the batch consuming it has completed.
        mapped = None
        event = None
        import_device = device
        # A read in the producer's own address space needs no acknowledgment:
        # the publication's own owner reclaims it.
        acknowledgment = None
        try:
            ticket._require_active()
            destination = _read_destination(
                locator, device, destination, region
            )
            with torch.cuda.device(device):
                if locator.source.address_space == self.source.address_space:
                    # Same address space: borrow the owner's registered tensor
                    # and producer fence directly; no IPC mapping is needed.
                    with _endpoint_lock:
                        owner = _endpoints.get(handle.endpoint)
                    if (
                        not isinstance(owner, CudaVmmTransport)
                        or owner.source != locator.source
                    ):
                        raise invalid_descriptor(
                            "CUDA publication has no live local owner"
                        )
                    publication = owner._publications.source(
                        locator, reading=True
                    )
                    mapped = publication.tensor
                    event = publication.event
                else:
                    destination_prototype = (
                        destination[0]
                        if isinstance(destination, tuple)
                        else destination
                    )
                    itemsize = destination_prototype.element_size()
                    if any(
                        offset % itemsize
                        for offset in handle.storage_offsets_bytes
                    ):
                        raise invalid_descriptor(
                            "CUDA VMM span offset is not element aligned"
                        )
                    # A VMM mapping can be granted only to a device that can
                    # access the producing allocation. Some hosts expose all
                    # GPUs to one process but grant peer access only inside
                    # smaller peer islands. Map such an allocation on its
                    # owning GPU, then let CUDA perform the cross-device copy
                    # into the destination. CUDA stages through host memory
                    # when no peer path exists. This decision follows the
                    # discovered CUDA topology, not a device model or SKU.
                    source_device = torch.device(locator.device)
                    if (
                        locator.source.node == self.source.node
                        and source_device.type == "cuda"
                        and source_device != device
                        and not _can_access_peer(
                            str(device), str(source_device)
                        )
                    ):
                        import_device = source_device
                    prototype = (
                        destination_prototype
                        if import_device == device
                        else torch.empty(
                            0,
                            dtype=destination_prototype.dtype,
                            device=import_device,
                        )
                    )
                    # One mapping owns every span; tensor views share its
                    # deleter.
                    # A fabric handle is importable as published. A descriptor
                    # names an open file of the producing process, so the
                    # usable one is received from that rank over its grant
                    # socket and closed once the allocation, which holds its
                    # own reference, has been imported.
                    granted = None
                    if len(handle.allocation_handle) == DESCRIPTOR_HANDLE_BYTES:
                        granted = descriptor_grants.fetch(
                            handle.endpoint, handle.publication_id
                        )
                        exported = descriptor_grants.descriptor_bytes(granted)
                    else:
                        exported = handle.allocation_handle
                    try:
                        allocation = import_handle(
                            prototype,
                            exported,
                            handle.storage_size_bytes,
                        )
                    finally:
                        if granted is not None:
                            os.close(granted)
                    lengths = (
                        length
                        for length, count in zip(
                            handle.span_lengths, handle.span_counts, strict=True
                        )
                        for length in repeat(length, count)
                    )
                    mapped = tuple(
                        allocation.as_strided(
                            (length, *locator.shape[1:]),
                            handle.tensor_stride,
                            byte_offset // itemsize,
                        )
                        for byte_offset, length in zip(
                            handle.storage_offsets_bytes, lengths, strict=True
                        )
                    )
                    # This rank's acknowledgment word inside the source
                    # chunk's header, written once the copies below complete.
                    # It shares the mapping's deleter, like the span views.
                    if handle.acknowledgment_offset >= 0:
                        acknowledgment = allocation.view(torch.uint8)[
                            handle.acknowledgment_offset
                            + self._acknowledgment_slot * ACK_WORD_BYTES :
                        ][:ACK_WORD_BYTES].view(torch.int32)
                    del allocation
                    # A pool publication was made readable by the producer's
                    # own synchronize; only an in-place one carries a fence,
                    # and that fence reaches this rank only on its own host.
                    if handle.ready_event_handle:
                        event = torch.cuda.Event.from_ipc_handle(
                            import_device, handle.ready_event_handle
                        )
                if region is not None:
                    mapped = region_view(mapped, region)
                if event is not None and import_device != device:
                    # CUDA does not permit the destination device's stream to
                    # wait on this imported source-device IPC event on every
                    # topology. Drain the producer on its owning device before
                    # submitting the host-staged cross-device copy.
                    event.synchronize()
                    event = None
                if acknowledgment is not None and import_device != device:
                    # The acknowledgment word belongs to the source-device
                    # mapping. Claim it before the staged copy, then publish
                    # completion only after the destination copy has drained.
                    with torch.cuda.device(import_device):
                        acknowledgment.copy_(_chunk_word(vmm_pool.CLAIMED))
                        torch.cuda.current_stream(import_device).synchronize()
                    self._reads.copy(ticket, mapped, destination, event, None)
                    with torch.cuda.device(import_device):
                        acknowledgment.copy_(_chunk_word(vmm_pool.ACKNOWLEDGED))
                        torch.cuda.current_stream(import_device).synchronize()
                else:
                    self._reads.copy(
                        ticket, mapped, destination, event, acknowledgment
                    )
        except BaseException as error:
            # Failure visibility must not wait for the source's retirement
            # acknowledgement. Physical ownership remains with the backend.
            ticket._fail(error)
            raise
        finally:
            # An undrained read keeps its mapping and fence through the
            # ticket; a physically settled read drops them here.
            if not ticket._unretired:
                mapped = None
                event = None

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        if not isinstance(locator.transport, CudaVmmTransfer):
            raise invalid_descriptor(
                "CUDA VMM release requires a CUDA VMM locator"
            )
        retirement = self._publications.release(locator)
        if retirement is not None and not retirement.done():
            source = self._publications.source(locator)
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
        if self._published:
            _LOG.info(
                "cuda_vmm transport retired: published=%d crossing=%s "
                "payload_total=%d payload_mean=%d payload_max=%d "
                "synchronize_ms_total=%.3f synchronize_ms_max=%.3f "
                "synchronize_ms_mean=%.3f",
                self._published,
                self._cross_host_consumers,
                self._payload_bytes,
                self._payload_bytes // self._published,
                self._largest_payload,
                self._synchronize_seconds * 1e3,
                self._longest_synchronize * 1e3,
                self._synchronize_seconds * 1e3 / self._published,
            )
        try:
            self._reads.close()
        finally:
            self._publications.close()
            # Consumers of this rank's remaining chunks are gone with it, so
            # their acknowledgments will never arrive. The pools are released
            # whole, which is what closing the transport means for them.
            self._unacknowledged.clear()
            self._pools.clear()
            if self._grants is not None:
                self._grants.close()
                self._grants = None
            with _endpoint_lock:
                _endpoints.pop(self.endpoint(), None)
        if self._failed_publication is not None:
            raise self._failed_publication[0]


def make_transports(
    names: Sequence[str],
    *,
    byte_capacity: int,
    ticket_capacity: int,
    event_pool: EventPool,
    source: WorkerEndpoint | None = None,
    acknowledgment_slot: int = 0,
    host_slots: Sequence[int] = (),
    cross_host_consumers: bool = False,
) -> dict[str, Transport]:
    """Construct configured backends against one rank resource budget.

    `acknowledgment_slot` is this rank's own word, which it writes in the
    header of every chunk or segment it reads. `host_slots` are the words of
    the ranks on this host, which decide whether a host product's consumers
    are reached over shared memory or over the rank channel.
    `cross_host_consumers` says whether a rank on another host reads this
    rank's products, which decides how a device publication states
    readiness. Which ranks read a given product is stated on the call that
    produces it.
    """
    if not names or len(set(names)) != len(names):
        raise invalid_descriptor(
            "transport bindings must be nonempty and unique"
        )
    if any(name not in TRANSPORTS for name in names):
        raise invalid_descriptor(
            f"unknown transport binding; expected names from {TRANSPORTS}"
        )
    if min(byte_capacity, ticket_capacity) < 1:
        raise unsupported_setup(
            "transport byte and ticket capacities must be positive"
        )
    capacity = TransferCapacity(byte_capacity, ticket_capacity)
    endpoint = source or WorkerEndpoint.local()
    constructors: Mapping[str, Callable[..., Transport]] = {
        "local": LocalTransport,
        "shm": ShmTransport,
        "cuda_vmm": CudaVmmTransport,
        "channel": ChannelTransport,
    }
    transports: dict[str, Transport] = {}
    try:
        for name in names:
            arguments = {
                "capacity": capacity,
                "event_pool": event_pool,
                "source": endpoint,
            }
            if name == "cuda_vmm":
                arguments["acknowledgment_slot"] = acknowledgment_slot
                arguments["cross_host_consumers"] = cross_host_consumers
            if name == "shm":
                arguments["acknowledgment_slot"] = acknowledgment_slot
                arguments["host_slots"] = host_slots
            if name == "channel":
                arguments["host_slots"] = host_slots
            transports[name] = constructors[name](**arguments)
    except BaseException:
        for transport in transports.values():
            transport.close()
        raise
    return transports


#: Mechanisms that carry a product where it lies on a device.
DEVICE_MECHANISMS = ("local", "cuda_vmm")
#: Mechanisms that carry a product as host bytes.
HOST_MECHANISMS = ("local", "shm", "channel")


def publish_tensor(
    transports: Mapping[str, Transport],
    source: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    retain: Callable[[concurrent.futures.Future[None]], None],
    offset: tuple[int, ...] | None = None,
    consumers: Sequence[int] = (),
    host: bool = False,
) -> tuple[Locator, ...]:
    """Publish one representation through the mechanisms its location needs.

    A device product is published where it lies, over the device mechanism
    of the rank's edges; a host product, or a device product the caller marks
    `host`, is published as host bytes over each host mechanism that reaches
    one of its named consumers: shared memory for a consumer on this host,
    the rank channel for one elsewhere. A device product that neither exports
    in place nor fits its device's pool falls back to host bytes for that
    product. A partial failure revokes all preceding locations. Each backend
    continues to retain the source until its submitted device work and
    readers retire.
    """
    if not transports:
        raise unsupported_setup(
            "tensor publication requires a configured transport"
        )
    first = source[0] if isinstance(source, tuple) else source
    device_product = first.is_cuda and not host
    names = DEVICE_MECHANISMS if device_product else HOST_MECHANISMS
    # A rank whose edges carry no device mechanism sends its device products
    # as bytes, the crossing any consumer off this device makes anyway.
    if device_product and "cuda_vmm" not in transports:
        names = HOST_MECHANISMS
    selected = [
        transports[name]
        for name in names
        if name in transports and transports[name].serves(consumers)
    ]
    locations: list[Locator] = []
    try:
        for transport in selected:
            try:
                location = transport.publish(
                    source, offset=offset, consumers=consumers
                )
            except PoolExhaustedError:
                # The product does not fit its device's pool: it travels as
                # host bytes instead, over every host mechanism this rank
                # publishes on.
                for fallback in HOST_MECHANISMS:
                    if (
                        fallback in transports
                        and fallback != "local"
                        and transports[fallback].serves(consumers)
                    ):
                        location = transports[fallback].publish(
                            source, offset=offset, consumers=consumers
                        )
                        locations.append(location)
                        retain(
                            transports[fallback].publication_retirement(
                                location
                            )
                        )
                continue
            locations.append(location)
            retain(transport.publication_retirement(location))
    except BaseException:
        for location in locations:
            transports[location.backend].release(location)
        raise
    return tuple(locations)
