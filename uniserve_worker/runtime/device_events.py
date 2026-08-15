"""Reference-counted CUDA events shared by asynchronous runtime stores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock

import torch

from .device import canonical_device
from ..foundation.errors import ErrorCode, WorkerError


def _resolved_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device) and (device.type != "cuda" or device.index is not None):
        return device
    return canonical_device(device)


def _invariant(message: str) -> WorkerError:
    return WorkerError(
        code=ErrorCode.INVARIANT_VIOLATION,
        message=message,
        fatal=True,
    )


@dataclass(slots=True)
class _EventState:
    event: torch.cuda.Event
    device_name: str
    timing: bool
    references: int = 0
    stream: torch.cuda.Stream | None = None
    stream_id: int | None = None
    recorded: bool = False


class DeviceEventPool:
    """Own CUDA events until every store reference is query-ready and released."""

    def __init__(self) -> None:
        self._available: dict[tuple[str, bool], deque[torch.cuda.Event]] = {}
        self._active: dict[int, _EventState] = {}
        self._lock = RLock()

    def acquire(
        self,
        device: torch.device | str,
        *,
        timing: bool = False,
    ) -> torch.cuda.Event:
        target = _resolved_device(device)
        if target.type != "cuda":
            raise _invariant("device event requires a CUDA device")
        device_name = str(target)
        key = (device_name, bool(timing))
        with self._lock:
            available = self._available.get(key)
            event = (
                available.pop()
                if available
                else torch.cuda.Event(blocking=False, enable_timing=bool(timing))
            )
            if id(event) in self._active:
                raise _invariant("device event was reused while still referenced")
            self._active[id(event)] = _EventState(
                event=event,
                device_name=device_name,
                timing=bool(timing),
            )
            return event

    def declare_stream(
        self,
        event: torch.cuda.Event,
        device: torch.device | str,
    ) -> int:
        target = _resolved_device(device)
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
        target = _resolved_device(device)
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
        target = _resolved_device(device)
        references = int(count)
        if references < 1:
            raise _invariant("device event retain count must be positive")
        with self._lock:
            state = self._require_locked(event, target)
            state.references += references

    def release(self, event: torch.cuda.Event, count: int = 1) -> None:
        references = int(count)
        if references < 1:
            raise _invariant("device event release count must be positive")
        with self._lock:
            state = self._active.get(id(event))
            if (
                state is None
                or state.event is not event
                or state.references < references
            ):
                raise _invariant("device event reference accounting is invalid")
            state.references -= references
            if state.references != 0:
                return
            if not state.recorded or not bool(event.query()):
                raise _invariant("device event was released before it became query-ready")
            self._active.pop(id(event))
            self._available.setdefault((state.device_name, state.timing), deque()).append(event)

    def close(self) -> None:
        with self._lock:
            self._active.clear()
            self._available.clear()

    def _require_locked(
        self,
        event: torch.cuda.Event,
        device: torch.device,
    ) -> _EventState:
        state = self._active.get(id(event))
        if (
            state is None
            or state.event is not event
            or state.device_name != str(device)
        ):
            raise _invariant("device event is not owned by its declared device")
        return state


__all__ = ["DeviceEventPool"]
