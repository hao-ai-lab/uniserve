"""Reference-counted CUDA events shared by asynchronous runtime stores."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock

import torch

from ..foundation.errors import WorkerError, WorkerErrorCode
from .device import canonical_device


def _invariant(message: str) -> WorkerError:
    """Construct a classified invariant error for event-pool misuse."""

    return WorkerError(
        code=WorkerErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(slots=True)
class _EventState:
    """Tracks one CUDA event’s device, producer stream, timing mode, recording state, and reference count."""

    event: torch.cuda.Event
    device_name: str
    timing: bool
    interprocess: bool
    references: int = 0
    stream: torch.cuda.Stream | None = None
    stream_id: int | None = None
    recorded: bool = False


@dataclass(slots=True)
class _DeferredRelease:
    """Retains event-pool references until stream completion permits safe release."""

    events: tuple[torch.cuda.Event, ...]
    owner: object
    completed: Callable[[], None] | None


class EventPool:
    """Own CUDA events until every store reference is query-ready and released."""

    def __init__(self) -> None:
        """Initialize reusable CUDA-event pools and generation-tagged active ownership."""

        self._available: dict[tuple[str, bool, bool], deque[torch.cuda.Event]] = {}
        self._active: dict[int, _EventState] = {}
        self._deferred: list[_DeferredRelease] = []
        self._wake_on_stream: Callable[[int], None] | None = None
        self._wake_streams: dict[tuple[str, int], torch.cuda.Stream] = {}
        self._lock = RLock()
        self._closed = False

    def set_completion_wake(self, wake_on_stream: Callable[[int], None] | None) -> None:
        """Install the callback used to wake a device-specific completion stream."""

        self._wake_on_stream = wake_on_stream

    def schedule_completion_wake(
        self,
        device: torch.device | str,
        event: torch.cuda.Event,
    ) -> None:
        """Schedule the registered wake callback after a producer event on the target device."""

        wake_on_stream = self._wake_on_stream
        if wake_on_stream is None:
            return
        target = canonical_device(device)
        device_name = str(target)
        with self._lock:
            state = self._require_locked(event, target)
            if not state.recorded or state.stream_id is None:
                raise _invariant("completion notification requires a recorded producer event")
            # Independent producers must not queue behind each other's waits.
            key = (device_name, state.stream_id)
            stream = self._wake_streams.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=target)
                self._wake_streams[key] = stream
            stream.wait_event(event)
            stream_id = int(stream.cuda_stream)
        wake_on_stream(stream_id)

    def acquire(
        self,
        device: torch.device | str,
        *,
        timing: bool = False,
        interprocess: bool = False,
    ) -> torch.cuda.Event:
        """Lease an unrecorded CUDA event for one device and timing mode."""

        target = canonical_device(device)
        if target.type != "cuda":
            raise _invariant("device event requires a CUDA device")
        device_name = str(target)
        key = (device_name, bool(timing), bool(interprocess))
        self.reap()
        with self._lock:
            if self._closed:
                raise _invariant("device event pool is closed")
            available = self._available.get(key)
            event = (
                available.pop()
                if available
                else torch.cuda.Event(
                    blocking=False, enable_timing=bool(timing), interprocess=bool(interprocess)
                )
            )
            if id(event) in self._active:
                raise _invariant("device event was reused while still referenced")
            self._active[id(event)] = _EventState(
                event=event,
                device_name=device_name,
                timing=bool(timing),
                interprocess=bool(interprocess),
            )
            return event

    def declare_stream(
        self,
        event: torch.cuda.Event,
        device: torch.device | str,
    ) -> int:
        """Bind an event to the current producer stream without recording it."""

        target = canonical_device(device)
        stream = torch.cuda.current_stream(target)
        stream_id = int(stream.cuda_stream)
        with self._lock:
            state = self._require_locked(event, target)
            if state.stream_id is not None and state.stream_id != stream_id:
                raise _invariant("device event spans incompatible producer streams")
            state.stream = stream
            state.stream_id = stream_id
        return stream_id

    def record(
        self,
        event: torch.cuda.Event,
        device: torch.device | str,
    ) -> int:
        """Record a leased event exactly once on its declared or current stream."""

        target = canonical_device(device)
        with self._lock:
            state = self._require_locked(event, target)
            if state.recorded:
                raise _invariant("device event was recorded more than once")
            stream = state.stream
            if stream is None:
                stream = torch.cuda.current_stream(target)
                state.stream = stream
            stream_id = int(stream.cuda_stream)
            state.stream_id = stream_id
            event.record(stream)
            state.recorded = True
        return stream_id

    def retain(
        self,
        event: torch.cuda.Event,
        device: torch.device | str,
        count: int = 1,
    ) -> None:
        """Add references that keep a leased event active across asynchronous owners."""

        target = canonical_device(device)
        references = int(count)
        if references < 1:
            raise _invariant("device event retain count must be positive")
        with self._lock:
            state = self._require_locked(event, target)
            state.references += references

    def release(self, event: torch.cuda.Event, count: int = 1) -> None:
        """Drop event references and recycle the event after its producer work becomes query-ready."""

        references = int(count)
        if references < 1:
            raise _invariant("device event release count must be positive")
        with self._lock:
            state = self._active.get(id(event))
            if state is None or state.event is not event or state.references < references:
                raise _invariant("device event reference accounting is invalid")
            state.references -= references
            if state.references != 0:
                return
            if not state.recorded or not bool(event.query()):
                raise _invariant("device event was released before it became query-ready")
            self._recycle_locked(state)

    def defer_release(
        self,
        events: Sequence[torch.cuda.Event],
        owner: object,
        *,
        completed: Callable[[], None] | None = None,
    ) -> None:
        """Retain the owner, then release event references and invoke its completion callback."""

        retained = tuple(events)
        if not retained:
            return
        with self._lock:
            # Shutdown has already drained all recorded events. Owners retained
            # by completed public tickets can outlive the pool's active scope.
            if self._closed:
                return
            for event in retained:
                state = self._active.get(id(event))
                if state is None or state.event is not event or state.references < 1:
                    raise _invariant("deferred device event has invalid ownership")
            self._deferred.append(_DeferredRelease(retained, owner, completed))
        self.reap()

    def reap(self) -> None:
        """Recycle deferred events whose recorded CUDA work has completed."""

        with self._lock:
            callbacks = self._reap_locked()
        # Owners have their own locks and may release another event. Calling
        # them outside this lock preserves that ownership order across threads.
        for callback in callbacks:
            callback()

    def close(self) -> None:
        """Drain recorded device work and release pooled and deferred references."""

        for stream in self._wake_streams.values():
            stream.synchronize()
        with self._lock:
            for state in self._active.values():
                if state.recorded:
                    state.event.synchronize()
        self.reap()
        with self._lock:
            self._closed = True
            self._wake_streams.clear()
            self._deferred.clear()
            self._active.clear()
            self._available.clear()

    def _reap_locked(self) -> tuple[Callable[[], None], ...]:
        """Return deferred CUDA events to their reusable pools once query-ready."""

        callbacks = []
        deferred_releases, self._deferred = self._deferred, []
        for deferred in deferred_releases:
            if not all(bool(event.query()) for event in deferred.events):
                self._deferred.append(deferred)
                continue
            for event in deferred.events:
                state = self._active.get(id(event))
                if state is None or state.event is not event or state.references < 1:
                    raise _invariant("deferred device event lost its ownership")
                state.references -= 1
                if state.references == 0:
                    self._recycle_locked(state)
            if deferred.completed is not None:
                callbacks.append(deferred.completed)
        return tuple(callbacks)

    def _recycle_locked(self, state: _EventState) -> None:
        """Remove an active event state and return its event to the reusable pool."""

        self._active.pop(id(state.event))
        self._available.setdefault(
            (state.device_name, state.timing, state.interprocess), deque()
        ).append(state.event)

    def _require_locked(
        self,
        event: torch.cuda.Event,
        device: torch.device,
    ) -> _EventState:
        """Require active ownership of an event on the specified device."""

        state = self._active.get(id(event))
        if state is None or state.event is not event or state.device_name != str(device):
            raise _invariant("device event is not owned by its declared device")
        return state


__all__ = ["EventPool"]
