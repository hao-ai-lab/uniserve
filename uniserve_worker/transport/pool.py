"""Shared byte reservations and bounded asynchronous transport reads."""

from __future__ import annotations

import concurrent.futures
import threading
from functools import cache
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker.errors import resource_error
from uniserve_worker.transport import vmm_pool
from uniserve_worker.transport.layout import copy_pairs
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


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


@cache
def chunk_word(state: int) -> torch.Tensor:
    """Return the pinned host word a consumer writes into a chunk's header.

    A claim precedes the consumer's first read of the chunk and an
    acknowledgment follows its last, so a producing rank sweeping a retired
    publication can tell a consumer that is still reading from one that never
    began.
    """
    import torch

    return torch.full((1,), state, dtype=torch.int32).pin_memory()


class TransferPool:
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
        pairs = tuple(copy_pairs(source, destination))
        device = spans[0].device

        if device.type != "cuda":
            if acknowledgment is not None:
                acknowledgment.copy_(chunk_word(vmm_pool.CLAIMED))
            for target, value in pairs:
                target.copy_(value)
            if acknowledgment is not None:
                acknowledgment.copy_(chunk_word(vmm_pool.ACKNOWLEDGED))
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
                    acknowledgment.copy_(chunk_word(vmm_pool.CLAIMED))
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
                        from uniserve_kernel.peer_storage import (
                            copy_host_device,
                        )

                        copy_host_device(target, value, stream)
                    else:
                        target.copy_(value, non_blocking=True)
                if acknowledgment is not None:
                    # A pinned host word makes this a memcpy on the read
                    # stream. Filling the word would launch a kernel, and the
                    # first launch of one in a process pays CUDA module
                    # loading, which a one-word acknowledgment should not.
                    acknowledgment.copy_(
                        chunk_word(vmm_pool.ACKNOWLEDGED), non_blocking=True
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
