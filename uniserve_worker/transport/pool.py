"""Shared byte reservations and bounded asynchronous transport reads.

`TransferCapacity` is one Worker rank's budget of reserved bytes and read
tickets, shared by every backend `make_transports` builds. `TransferPool`
runs one backend's reads on its own threads against that budget: a read holds
its ticket and bytes from submission until it physically retires, and a CUDA
read copies on a per-thread read stream that it drains before the read
retires. A fetch takes the tickets of all its reads together
(`ReadReservation`). `chunk_word` supplies the host words a read writes into
a `vmm_pool` chunk header to claim and acknowledge that chunk.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker.errors import (
    ResourceError,
    resource_error,
    unsupported_setup,
)
from uniserve_worker.transport import vmm_pool
from uniserve_worker.transport.layout import copy_pairs
from uniserve_worker.transport.ticket import TransferTicket

if TYPE_CHECKING:
    import torch


class ReadBackpressureError(ResourceError):
    """Too few of the rank's read tickets are free for a fetch right now.

    A ticket returns when its read physically retires, which happens once the
    read's copy drains, whatever its submitter does next. A caller therefore
    retries after a return of ``capacity``'s tickets: ``returns`` is its
    return count when the fetch was refused, which
    `TransferCapacity.notify_reads_returned` compares against so that no
    return between the refusal and the request for a notification is missed.
    """

    def __init__(
        self,
        message: str,
        *,
        capacity: TransferCapacity,
        returns: int,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.capacity = capacity
        self.returns = returns


class TransferCapacity:
    """Share a Worker rank's byte and read-ticket budget across its backends.

    Neither budget blocks. `acquire` raises `resource_error` when the bytes
    are exhausted. A read takes one ticket when it is submitted and returns
    it when it physically retires, and a local borrowed view holds one until
    its consumers finish; `take_reads` raises `ReadBackpressureError` while too
    few are free, and `notify_reads_returned` tells the caller when to try
    again.
    """

    def __init__(self, byte_capacity: int, ticket_capacity: int) -> None:
        """Initialize reservations against fixed capacities.

        Raises:
            ValueError: When either capacity is less than one.
        """
        self.capacity = int(byte_capacity)
        self.ticket_capacity = int(ticket_capacity)
        if min(self.capacity, self.ticket_capacity) < 1:
            raise ValueError("transfer byte capacity must be positive")
        self.used = 0
        self._lock = threading.Lock()
        self._reads_free = self.ticket_capacity
        # Ticket returns so far, and the callbacks waiting for the next one.
        self._returns = 0
        self._return_waiters: list[Callable[[], None]] = []

    def take_reads(
        self,
        count: int = 1,
        *,
        message: str = "asynchronous transfer ticket capacity is exhausted",
    ) -> None:
        """Take ``count`` read tickets together, without blocking.

        Raises:
            ValueError: ``count`` is less than one.
            WorkerError: `unsupported_setup` when ``count`` exceeds the
                rank's tickets, which no wait can grant, and
                `ReadBackpressureError` (with ``message``) when fewer are free
                now.
        """
        if type(count) is not int or count < 1:
            raise ValueError("a read ticket reservation takes at least one")
        if count > self.ticket_capacity:
            raise unsupported_setup(
                f"a fetch of {count} reads exceeds the rank's "
                f"{self.ticket_capacity} read tickets"
            )
        with self._lock:
            if count > self._reads_free:
                raise ReadBackpressureError(
                    message, capacity=self, returns=self._returns
                )
            self._reads_free -= count

    def return_reads(self, count: int = 1) -> None:
        """Return read tickets and wake the callers waiting for a return.

        Raises:
            RuntimeError: More tickets would be free than the rank has.
        """
        with self._lock:
            if count < 0 or self._reads_free + count > self.ticket_capacity:
                raise RuntimeError(
                    "read ticket return exceeds the tickets taken"
                )
            self._reads_free += count
            self._returns += 1
            waiters, self._return_waiters = self._return_waiters, []
        for waiter in waiters:
            waiter()

    def notify_reads_returned(
        self, callback: Callable[[], None], *, after: int
    ) -> None:
        """Call ``callback`` once a read ticket returns after ``after``.

        ``after`` is the return count a `ReadBackpressureError` carried. When a
        ticket has returned since, ``callback`` runs at once on this thread;
        otherwise it runs on the thread that returns the next one.
        """
        with self._lock:
            if self._returns == after:
                self._return_waiters.append(callback)
                return
        callback()

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
        """Return bytes after the physical owner releases its allocation.

        Raises:
            RuntimeError: When `amount` is negative or exceeds the bytes
                currently reserved.
        """
        value = int(amount)
        with self._lock:
            if value < 0 or value > self.used:
                raise RuntimeError(
                    "transfer byte release exceeds the live reservation"
                )
            self.used -= value


class ReadReservation:
    """Read tickets one fetch takes together for all of its reads.

    Each read submitted with the reservation uses one of its tickets, which
    returns to the capacity when that read retires. `close` returns the
    tickets no read used, so a fetch that fails part way holds none it did
    not submit.
    """

    def __init__(self, capacity: TransferCapacity, count: int) -> None:
        """Take ``count`` tickets; see `TransferCapacity.take_reads`."""
        capacity.take_reads(count)
        self._capacity = capacity
        self._unused = count
        self._lock = threading.Lock()

    def use(self) -> None:
        """Hand one ticket to a read being submitted.

        Raises:
            RuntimeError: Every ticket is already in use.
        """
        with self._lock:
            if self._unused < 1:
                raise RuntimeError("read reservation has no ticket left")
            self._unused -= 1

    def close(self) -> None:
        """Return the tickets no read used; later calls do nothing."""
        with self._lock:
            unused, self._unused = self._unused, 0
        if unused:
            self._capacity.return_reads(unused)

    def __enter__(self) -> ReadReservation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@cache
def chunk_word(state: int) -> torch.Tensor:
    """Return the pinned host word a consumer writes into a chunk's header.

    A claim precedes the consumer's first read of the chunk and an
    acknowledgment follows its last, so a producing rank sweeping a retired
    publication can tell a consumer that is still reading from one that never
    began.

    One tensor is cached per state value and shared by every thread and read
    in the process, so it is only ever a copy source and must never be
    written.
    """
    import torch

    return torch.full((1,), state, dtype=torch.int32).pin_memory()


class TransferPool:
    """Bounds asynchronous transfers for one transport backend.

    Both transfer count and aggregate bytes are bounded, by the
    `TransferCapacity` shared with the rank's other backends.

    A failure that arrives after a ticket already exposed its views, or a
    read whose device access could not be drained, is recorded on the pool:
    every later `submit` and `close` raise it.
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
        self._capacity = capacity
        self._events = event_pool
        self._completion_wake: Any = None
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        # Tickets of reads that could not be drained, referenced for the
        # pool's lifetime so the resources they retain are never freed.
        self._unretired: list[TransferTicket] = []
        # One read stream per (transport thread ident, device), created on
        # first use by `copy`.
        self._read_streams: dict[tuple[int, str], torch.cuda.Stream] = {}

    def set_completion_wake(self, wake: Any) -> None:
        """Install the controller callback for transfer completion.

        Tickets submitted afterwards invoke the callback when they become
        ready or fail and when they physically retire. A failure that arrives
        after a ticket's readiness also invokes it.
        """
        self._completion_wake = wake

    def submit(
        self,
        call: Any,
        *args: Any,
        nbytes: int,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket:
        """Reserve a read and run `call(ticket, *args)` on a transport thread.

        `call` is `copy` itself or a backend read routine that ends by
        calling it. When `destination` is on a CUDA device, the caller's
        current stream on that device is recorded here, on the caller's
        thread, so the read stream waits for work the caller already queued
        on the destination. The read uses a ticket of `reservation`, or takes
        one of its own without one; the ticket and `nbytes` stay reserved
        until the read physically retires.

        Raises:
            BaseException: The pool's recorded failure, if any.
            WorkerError: `ReadBackpressureError` when no read ticket is free and
                `resource_error` when the bytes are exhausted. An error
                submitting to the executor propagates after both
                reservations are returned.
        """
        import torch

        with self._lock:
            if self._error is not None:
                raise self._error

        if reservation is None:
            self._capacity.take_reads(1)
        else:
            reservation.use()
        try:
            self._capacity.acquire(nbytes)
        except BaseException:
            self._capacity.return_reads(1)
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
                # A failure before readiness reaches the consumer through the
                # ticket alone. A late failure cannot retract views already
                # exposed, and an undrained read never retires, so both are
                # also recorded on the pool. The ticket's done callback fired
                # at readiness, so a late failure wakes the controller here.
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
            # A task that ran has drained any read stream it used, since
            # `copy` synchronizes it before returning, so unless the ticket
            # lists undrained resources no device access to its storage
            # remains.
            if not ticket._unretired:
                self._capacity.release(nbytes)
                self._capacity.return_reads(1)
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
            self._capacity.release(nbytes)
            self._capacity.return_reads(1)
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
        """Copy `source` into the reserved `destination` and complete `ticket`.

        Runs on a transport thread, inside the `call` given to `submit`. A
        CPU destination is copied synchronously and completes the ticket with
        no fence. A CUDA destination is copied on this thread's read stream,
        ordered after the caller's destination stream recorded by `submit`
        and after `producer`; the ticket completes with a fence recorded on
        the read stream, and this thread then waits for the fence and drains
        the stream, so all storage is retained through device completion.

        `acknowledgment` is this rank's word in the source chunk's header. It
        is claimed before the copies and acknowledged after them on the same
        stream, so the producer sees the acknowledgment only once every read
        of that chunk has completed.

        When the read stream cannot be drained, the ticket retains every
        resource of the read and never retires.

        Raises:
            WorkerError: The ticket's cancellation error when it was cancelled
                before the copy began. Copy and drain errors propagate; the
                task `submit` runs records them as the ticket's failure.
        """
        import torch

        ticket._require_active()
        spans = (
            destination if isinstance(destination, tuple) else (destination,)
        )
        pairs = tuple(copy_pairs(source, destination))
        device = spans[0].device

        # A host destination: every copy below is synchronous, so the claim
        # and acknowledgment bracket them directly.
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
                        from uniserve_kernels.peer_storage import (
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

            # Readiness is exposed before this thread waits, so a consumer can
            # queue its work behind the fence while the copies still run.
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
        """Wait for every submitted read, then raise any recorded failure.

        Queued reads are not cancelled; they run to completion first.
        """
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._read_streams.clear()
        if self._error is not None:
            raise self._error
