"""Bounded local, shared-memory and CUDA IPC product transport.

Backends publish canonical typed locators. A descriptor carries the physical
handle and readiness fence; asynchronous reads establish access to its bytes.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import mmap
import os
import queue
import selectors
import socket
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import groupby, repeat
from typing import TYPE_CHECKING, Any, ClassVar

from uniserve import _slices
from uniserve.runtime import EventPool

from ..foundation.errors import (
    invalid_descriptor,
    resource_error,
    unsupported_setup,
)
from ..foundation.shared_memory import allocate_shared_memory
from ..protocol.transfer import (
    CudaIpcTransfer,
    LocalTransfer,
    Locator,
    PosixShmTransfer,
    WorkerEndpoint,
)
from .endpoint import PublicationEndpoint, finish_reader, open_reader
from .layout import region_view, validate_destination

if TYPE_CHECKING:
    import torch

__all__ = [
    "Transport",
    "TransferTicket",
    "LocalTransport",
    "ShmTransport",
    "CudaIpcTransport",
    "TransportKind",
    "TRANSPORTS",
]


class TransportKind(StrEnum):
    """Selects in-process, POSIX shared-memory, or CUDA IPC byte transport."""

    LOCAL = "local"
    SHM = "shm"
    CUDA_IPC = "cuda_ipc"


TRANSPORTS = tuple(kind.value for kind in TransportKind)


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
    ) -> Locator:
        """Expose a descriptor and producer fence for an immutable version.

        The allocation owner must retain the published range, without writes,
        until publication_retirement() completes after release(). Keeping a
        tensor reference does not authorize reuse of an arena or page range.
        """

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


_SHM_LIBC = ctypes.CDLL(None, use_errno=True)
_SHM_LIBC.shm_open.restype = ctypes.c_int


def _open_shared_memory(name: str, size: int) -> mmap.mmap:
    """Open an existing shared-memory segment and validate its declared size."""
    canonical_name = name if name.startswith("/") else f"/{name}"
    descriptor = _SHM_LIBC.shm_open(canonical_name.encode(), os.O_RDONLY)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), canonical_name)
    try:
        return mmap.mmap(
            descriptor,
            int(size),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
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
    ) -> None:
        """Copy into a reserved view.

        All storage is retained through device completion.
        """
        import torch

        ticket._require_active()
        spans = (
            destination if isinstance(destination, tuple) else (destination,)
        )
        pairs = tuple(_copy_pairs(source, destination))
        device = spans[0].device

        if device.type != "cuda":
            for target, value in pairs:
                target.copy_(value)
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
    """Own a shared allocation and any unfinished device-to-host publication."""

    shm: Any
    nbytes: int
    host: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    signal: Any = None


class ShmTransport(Transport):
    """Shared-memory publication with producer readiness and reader grants."""

    name = "shm"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
    ) -> None:
        self._bytes = capacity
        self._publications = PublicationEndpoint[_ShmSource](
            reader_capacity=capacity.ticket_capacity,
            publication_capacity=256,
            reclaim=self._reclaim,
            drain=lambda source: None,
        )
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
        source.shm.close()
        try:
            source.shm.unlink()
        except FileNotFoundError:
            pass
        self._bytes.release(source.nbytes)
        retirement.set_result(None)

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
        import torch

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

                    # A stream signal fired: the device-to-host DMA is done,
                    # so move the staged bytes into the shared segment and
                    # expose (or fail) the publication.
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
                        try:
                            assert source.host is not None
                            raw = source.host.view(torch.uint8).reshape(-1)
                            source.shm.buf[: source.nbytes] = bytes(raw.numpy())
                        except BaseException as error:
                            failure = error
                        source.host = None
                        source.signal = None
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
    ) -> Locator:
        """Register source storage before exposing its readiness descriptor."""
        import torch

        source, shape, offset = _publication_views(tensor, offset)
        first = source[0] if isinstance(source, tuple) else source
        nbytes = _nbytes(source)
        self._bytes.acquire(nbytes)

        shm = None
        registered = False
        submitted = False
        try:
            shm = allocate_shared_memory(max(1, nbytes))
            buffer = shm.buf
            if buffer is None:
                raise RuntimeError(
                    "shared-memory publication has no writable buffer"
                )

            if first.is_cuda:
                # Device bytes move through a pinned staging buffer; the
                # stream signal marks the DMA complete on the worker thread.
                from .._uniserve_ipc import StreamSignal

                host = torch.empty(
                    shape, dtype=first.dtype, device="cpu", pin_memory=True
                )
                signal = StreamSignal()
            else:
                host = None
                signal = None
                packed = torch.frombuffer(buffer, dtype=first.dtype).reshape(
                    shape
                )
                for target, value in _copy_pairs(source, packed):
                    target.copy_(value)
                del packed, target, value

            publication = _ShmSource(shm, nbytes, host, signal)
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
            self._publications.publish(
                locator, publication, pending=first.is_cuda
            )
            registered = True

            if first.is_cuda:
                assert host is not None
                assert signal is not None
                submitted = True
                from uniserve_kernel.peer_memory import copy_host_device

                stream = torch.cuda.current_stream(first.device)
                for target, value in _copy_pairs(source, host):
                    copy_host_device(target, value, stream)
                    value.record_stream(stream)
                signal.schedule(int(stream.cuda_stream))
                self._queue_publication((locator, publication))
            return locator
        except BaseException:
            if registered:
                self._publications.release(locator)
                if submitted:
                    # Preserve the registered pinned destination if CUDA cannot
                    # establish completion on this exceptional publication path.
                    torch.cuda.current_stream(first.device).synchronize()
                if first.is_cuda:
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
        """Hold host bytes through source retirement.

        The bytes are held through asynchronous destination copying as well.
        """
        import torch

        handle = locator.transport
        if not isinstance(handle, PosixShmTransfer):
            raise invalid_descriptor(
                "shared-memory read requires a shared-memory locator"
            )
        connection, _descriptor = open_reader(locator)
        failure: BaseException | None = None
        try:
            ticket._require_active()
            # Copy the bytes out of the segment up front so the reader grant
            # can be returned before the (possibly asynchronous) destination
            # copy; the grant only needs to cover access to the segment.
            shm = _open_shared_memory(handle.name, locator.nbytes)
            try:
                buf = bytearray(shm[: locator.nbytes])
            finally:
                shm.close()
        except BaseException as error:
            failure = error
            ticket._fail(error)
            raise
        finally:
            try:
                try:
                    finish_reader(connection)
                except BaseException as cleanup_error:
                    if failure is not None:
                        raise failure from cleanup_error
                    raise
            finally:
                connection.close()
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
    descriptor: int
    copied_source: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    retirement: concurrent.futures.Future[None] | None = None

    def events_released(self) -> None:
        os.close(self.descriptor)
        self.capacity.release(self.nbytes)
        if self.retirement is not None:
            self.retirement.set_result(None)


class CudaIpcTransport(Transport):
    """CUDA mapping and asynchronous copies protected by reader grants."""

    name = "cuda_ipc"

    def __init__(
        self,
        *,
        capacity: TransferCapacity,
        event_pool: EventPool,
        source: WorkerEndpoint | None = None,
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
        self._publications = PublicationEndpoint[_CudaSource](
            reader_capacity=capacity.ticket_capacity,
            publication_capacity=256,
            reclaim=self._reclaim,
            drain=self._drain,
            descriptor=lambda source: source.descriptor,
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

    def publication_retirement(
        self, locator: Locator
    ) -> concurrent.futures.Future[None]:
        return self._publications.retirement(locator)

    def set_completion_wake(self, wake: Any) -> None:
        self._reads.set_completion_wake(wake)

    def _reclaim(
        self,
        source: _CudaSource,
        retirement: concurrent.futures.Future[None] | None = None,
    ) -> None:
        source.retirement = retirement
        self._events.defer_release(
            (source.event,), source, completed=source.events_released
        )

    def _drain(self, source: _CudaSource) -> None:
        source.event.synchronize()
        self._events.reap()

    def publish(
        self,
        tensor: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        offset: tuple[int, ...] | None = None,
    ) -> Locator:
        """Export an immutable source and retain capacity.

        Capacity is retained until its producer fence retires.
        """
        import torch
        from uniserve_kernel.peer_memory import empty, export_fd

        source, shape, offset = _publication_views(tensor, offset)
        spans = source if isinstance(source, tuple) else (source,)
        first = spans[0]
        if not first.is_cuda:
            raise invalid_descriptor(
                "cuda_ipc transport requires a CUDA tensor"
            )
        if any(
            span.untyped_storage().data_ptr()
            != first.untyped_storage().data_ptr()
            or span.stride() != first.stride()
            for span in spans
        ):
            raise invalid_descriptor(
                "CUDA IPC publication spans require one allocation and stride"
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
            exported = export_fd(first)
            if exported is None:
                # Arbitrary CUDA tensors retain the same publication behavior.
                # Materialize only their logical spans, never their enclosing
                # allocator segment. Shared worker arenas export directly.
                shared = empty(shape, dtype=first.dtype, device=first.device)
                for target, value in _copy_pairs(source, shared):
                    target.copy_(value, non_blocking=True)
                copied_source = source
                source = shared
                spans = (shared,)
                first = shared
                exported = export_fd(first)
                if exported is None:
                    raise RuntimeError("shared allocation cannot be exported")
            descriptor, storage_size, storage_offset = exported
            event = self._events.acquire(first.device, interprocess=True)
            self._events.retain(event, first.device)
            self._events.record(event, first.device)
            publication = _CudaSource(
                source, event, nbytes, self._bytes, descriptor, copied_source
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
                transport=CudaIpcTransfer(
                    endpoint=self.endpoint(),
                    publication_id=uuid.uuid4().hex,
                    storage_size_bytes=storage_size,
                    storage_offsets_bytes=tuple(
                        storage_offset + span.data_ptr() - first.data_ptr()
                        for span in spans
                    ),
                    span_lengths=tuple(length for length, _ in length_runs),
                    span_counts=tuple(count for _, count in length_runs),
                    tensor_stride=tuple(first.stride()),
                    ready_event_handle=bytes(event.ipc_handle()),
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
                # No reader can own an unregistered descriptor. Its allocation
                # remains retained separately if producer draining fails.
                if descriptor is not None:
                    os.close(descriptor)
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
        if locator.source.node != self.source.node:
            raise invalid_descriptor(
                "CUDA IPC transport requires the source node"
            )
        if not isinstance(locator.transport, CudaIpcTransfer):
            raise invalid_descriptor(
                "CUDA IPC read requires a CUDA IPC locator"
            )
        if device.type != "cuda":
            raise invalid_descriptor(
                "CUDA IPC destination must be a CUDA device"
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
        from uniserve_kernel.peer_memory import import_fd

        handle = locator.transport
        assert isinstance(handle, CudaIpcTransfer)
        connection, descriptor = open_reader(locator)
        assert descriptor is not None
        mapped = None
        event = None
        failure: BaseException | None = None
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
                        not isinstance(owner, CudaIpcTransport)
                        or owner.source != locator.source
                    ):
                        raise invalid_descriptor(
                            "CUDA publication has no live local owner"
                        )
                    publication = owner._publications.source(locator)
                    mapped = publication.tensor
                    event = publication.event
                else:
                    prototype = (
                        destination[0]
                        if isinstance(destination, tuple)
                        else destination
                    )
                    itemsize = prototype.element_size()
                    if any(
                        offset % itemsize
                        for offset in handle.storage_offsets_bytes
                    ):
                        raise invalid_descriptor(
                            "CUDA IPC span offset is not element aligned"
                        )
                    # One mapping owns every span; tensor views share its
                    # deleter.
                    allocation = import_fd(
                        prototype,
                        descriptor,
                        handle.storage_size_bytes,
                    )
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
                    del allocation
                    event = torch.cuda.Event.from_ipc_handle(
                        device, handle.ready_event_handle
                    )
                if region is not None:
                    mapped = region_view(mapped, region)
                self._reads.copy(ticket, mapped, destination, event)
        except BaseException as error:
            failure = error
            # Failure visibility must not wait for the source's retirement
            # acknowledgement. Physical ownership remains with the backend.
            ticket._fail(error)
            raise
        finally:
            try:
                # An undrained read keeps its mapping and fence through the
                # ticket; only a physically settled read returns its grant.
                if not ticket._unretired:
                    mapped = None
                    event = None
                    try:
                        finish_reader(connection)
                    except BaseException as cleanup_error:
                        if failure is not None:
                            raise failure from cleanup_error
                        raise
            finally:
                os.close(descriptor)
                connection.close()

    def release(
        self, locator: Locator
    ) -> concurrent.futures.Future[None] | None:
        if not isinstance(locator.transport, CudaIpcTransfer):
            raise invalid_descriptor(
                "CUDA IPC release requires a CUDA IPC locator"
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
        try:
            self._reads.close()
        finally:
            self._publications.close()
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
) -> dict[str, Transport]:
    """Construct configured backends against one rank resource budget."""
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
        "cuda_ipc": CudaIpcTransport,
    }
    transports: dict[str, Transport] = {}
    try:
        for name in names:
            transports[name] = constructors[name](
                capacity=capacity,
                event_pool=event_pool,
                source=endpoint,
            )
    except BaseException:
        for transport in transports.values():
            transport.close()
        raise
    return transports


def publish_tensor(
    transports: Mapping[str, Transport],
    source: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    retain: Callable[[concurrent.futures.Future[None]], None],
    offset: tuple[int, ...] | None = None,
) -> tuple[Locator, ...]:
    """Publish one representation through each explicitly required backend.

    A partial failure revokes all preceding locations. Each backend continues
    to retain the source until its submitted device work and readers retire.
    """
    if not transports:
        raise unsupported_setup(
            "tensor publication requires a configured transport"
        )
    locations: list[Locator] = []
    try:
        for transport in transports.values():
            location = transport.publish(source, offset=offset)
            locations.append(location)
            retain(transport.publication_retirement(location))
    except BaseException:
        for location in locations:
            transports[location.backend].release(location)
        raise
    return tuple(locations)
