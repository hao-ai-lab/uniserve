"""Stream readiness and physical retirement of one transport read.

Every backend's `fetch` returns a `TransferTicket`. Copy reads are driven by
`TransferPool`; borrowed views of an in-process publication are driven by
`LocalTransport`. Both complete the same two stages, which consumers and
storage owners observe separately.
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import TYPE_CHECKING, Any

from uniserve.runtime import EventPool
from uniserve_worker.errors import invalid_descriptor, resource_error

if TYPE_CHECKING:
    import torch


class TransferTicket:
    """One read, available once its consuming stream can wait on a fence.

    A ticket completes in two stages:

    - Readiness (`ready`, `result`, `add_done_callback`): the destination
      views and, for a CUDA read, a fence the consumer's stream waits on are
      available, or the read has failed. Readiness does not imply device
      completion.
    - Physical retirement (`retired`, `add_retirement_callback`): the
      backend has stopped all access to source and destination storage.
      `TransferPool` retires a copy read once its read stream has drained;
      a borrowed view retires in `events_released`, after the fences `close`
      records on its consumer streams complete.

    The transport keeps physical source and mapping leases until retirement.
    A read whose device work could not be drained never retires, and
    `retirement_ready` then raises. Methods and attributes with a leading
    underscore are the backend's side of the contract.
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
        # The executor task running a pool read, cleared when it finishes;
        # `cancel` cancels it while it is still queued.
        self._work: concurrent.futures.Future[None] | None = None

        # Borrowed-view consumption: streams that received the views and the
        # release that returns the source grant once they all complete.
        self._consumer_release: Any = None
        self._consumer_streams: dict[int, torch.cuda.Stream] = {}
        # Copy reads only: the caller's stream on a CUDA destination, recorded
        # by `TransferPool.submit` and waited on by the read stream in
        # `TransferPool.copy`.
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
        """Report whether the read reached readiness or failed.

        Once true, `result` returns the views or raises the failure without
        a host wait; before that it raises.
        """
        return self._future.done()

    def retired(self) -> bool:
        """Query whether the backend stopped access to the read's storage."""
        return self._retirement.done()

    def retirement_ready(self) -> bool:
        """Report physical retirement, raising when it can never be known.

        An allocation owner calls this before treating a destination as free.

        Raises:
            WorkerError: `resource_error`, chained from the read's failure,
                when the read's device access could not be drained and its
                resources are retained indefinitely.
        """
        if self._unretired:
            raise resource_error(
                "transfer physical completion is unknown"
            ) from self._error
        return self.retired()

    def add_retirement_callback(self, callback: Any) -> None:
        """Invoke `callback` once, after the read physically retires.

        The callback reference is dropped after its call.
        """

        def notify(_future: concurrent.futures.Future[None]) -> None:
            nonlocal callback
            try:
                callback()
            finally:
                callback = None

        self._retirement.add_done_callback(notify)

    def cancel(self) -> None:
        """Revoke consumption while retaining storage until the read retires.

        A read that has not reached readiness fails with a cancellation error,
        and after readiness `result` raises that error instead of returning
        the views. A queued read that has not started is cancelled in its
        executor. A running read stops at its next `_require_active` check; a
        copy already submitted to a device still runs to completion before
        the ticket retires.
        """
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
        """Return destination views and order reads on the consumer stream.

        When the read carries a fence, `stream` (default: the current stream
        on the views' device) waits on it, and every view is recorded on the
        stream so the caching allocator does not reuse destination storage
        before the consumer's queued work completes. For a borrowed view, the
        stream is also tracked so `close` can fence it.

        Raises:
            RuntimeError: When called before readiness, or after `close`
                ended a borrowed-view read.
            WorkerError: `invalid_descriptor` when `stream` is on another
                device than the views. The read's failure, if any, is raised
                as is.
        """
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
        source grant returns only after all of those fences complete. Does
        nothing for a copy read or a ticket already closed.
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
        """Return the source grant and retire the borrowed-view read.

        `EventPool.reap` calls this once every fence `close` recorded has
        completed; `close` calls it directly when no consumer stream was
        observed.
        """
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
        """Invoke `callback` once, when readiness or a failure is observable.

        The callback reference is dropped after its call.
        """

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
        """Publish readiness with the destination views and their fence.

        The ticket takes its own event-pool reference on `event`, released by
        `__del__` once the fence has completed. A ticket that already failed,
        for example by cancellation, keeps its failure.
        """
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
        """Record a failure, before or after readiness.

        A failure before readiness is what `result` raises. After readiness
        the views may already be in use, so the failure is kept for later
        `result` calls and the backend must also surface it elsewhere.

        Returns:
            Whether the failure arrived after a successful readiness.
        """
        with self._state_lock:
            late = self._future.done() and self._future.exception() is None
            self._error = error
            if not self._future.done():
                self._future.set_exception(error)
            return late

    def _retain_failed_read(self, *resources: object) -> None:
        """Keep allocations whose device access could not be drained.

        The ticket then never retires, and `retirement_ready` raises.
        """
        self._unretired = resources

    def __del__(self) -> None:
        # A borrowed view its consumer never closed is closed on collection.
        self.close()
        if self._event is not None:
            self._events.defer_release((self._event,), self._future)
