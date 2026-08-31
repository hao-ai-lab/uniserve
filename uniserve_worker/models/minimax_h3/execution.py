"""MiniMax H3 output-ring declarations and request mux coordination."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np
import torch

from ...execution.batch import RequestKey
from ...foundation.errors import resource_error
from ...server.completion import EventGatedDeferredTask, PinnedByteCapture
from ...server.cpu_tasks import CpuTaskReservation
from ...server.media_output import AvMuxConfig, AvMuxSession, require_media_codecs
from .state import PROFILE_AUDIO_RATE, PROFILE_FPS, PROFILE_HEIGHT, PROFILE_WIDTH

__all__ = [
    "H3MuxCoordinator",
    "H3OutputRing",
    "H3OutputRingLease",
    "require_h3_codecs",
]


def require_h3_codecs() -> None:
    require_media_codecs("libx264", "aac")


class H3OutputRing:
    """Bounded ownership for capacity-sized pinned video and audio captures."""

    def __init__(
        self,
        *,
        state_slots: int,
        unresolved_window: int,
        max_video_frames_per_round: int,
        max_frame_count: int,
    ) -> None:
        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("H3 output-ring capacities must be positive")
        video_bytes = int(max_video_frames_per_round) * PROFILE_HEIGHT * PROFILE_WIDTH * 3
        audio_bytes = round(int(max_frame_count) * PROFILE_AUDIO_RATE / PROFILE_FPS) * 2 * 2
        if min(video_bytes, audio_bytes) < 1:
            raise ValueError("H3 output-ring media capacities must be positive")
        self._video_storage = tuple(
            torch.empty(video_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            torch.empty(audio_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.audio_capacity)
        )
        self._video_free = list(range(self.video_capacity - 1, -1, -1))
        self._audio_free = list(range(self.audio_capacity - 1, -1, -1))
        self._lock = RLock()

    def reserve(self, kind: str) -> "H3OutputRingLease":
        if kind not in {"video", "audio"}:
            raise ValueError(f"unknown H3 output-ring kind {kind!r}")
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            if not free:
                raise resource_error(f"H3 {kind} output ring is exhausted")
            index = free.pop()
        return H3OutputRingLease(self, kind, index)

    def _storage(self, kind: str, index: int) -> torch.Tensor:
        values = self._video_storage if kind == "video" else self._audio_storage
        return values[int(index)]

    def _release(self, kind: str, index: int) -> None:
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            capacity = self.video_capacity if kind == "video" else self.audio_capacity
            if int(index) in free or not 0 <= int(index) < capacity:
                raise RuntimeError("H3 output-ring ownership is invalid")
            free.append(int(index))

    @property
    def used(self) -> tuple[int, int]:
        with self._lock:
            return (
                self.video_capacity - len(self._video_free),
                self.audio_capacity - len(self._audio_free),
            )


class H3OutputRingLease:
    __slots__ = ("_ring", "kind", "index", "_released")

    def __init__(self, ring: H3OutputRing, kind: str, index: int) -> None:
        self._ring = ring
        self.kind = kind
        self.index = int(index)
        self._released = False

    @property
    def storage(self) -> torch.Tensor:
        if self._released:
            raise RuntimeError("H3 output-ring storage was accessed after release")
        return self._ring._storage(self.kind, self.index)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._ring._release(self.kind, self.index)

    def defer_until_capture_ready(self, capture: PinnedByteCapture) -> None:
        if self._released:
            return
        if capture.ready():
            self.release()
            return
        self._released = True
        capture.buffer.retain_until_ready(_DeferredRingRelease(self._ring, self.kind, self.index))

    def __del__(self) -> None:
        self.release()


class _DeferredRingRelease:
    __slots__ = ("_ring", "_kind", "_index")

    def __init__(self, ring: H3OutputRing, kind: str, index: int) -> None:
        self._ring = ring
        self._kind = kind
        self._index = int(index)

    def __del__(self) -> None:
        self._ring._release(self._kind, self._index)


class H3MuxCoordinator:
    """Request-indexed mux sessions with explicit per-request dependency tails."""

    def __init__(self) -> None:
        self._sessions: dict[RequestKey, AvMuxSession] = {}
        self._video_tails: dict[RequestKey, concurrent.futures.Future[None] | None] = {}
        self._audio_tails: dict[RequestKey, concurrent.futures.Future[None] | None] = {}

    def open(
        self,
        request_key: RequestKey,
        path: Path,
        *,
        frame_count: int,
        video_unit_frames: tuple[int, ...],
    ) -> None:
        if request_key in self._sessions:
            raise RuntimeError("H3 mux session is already active")
        config = AvMuxConfig(
            width=PROFILE_WIDTH,
            height=PROFILE_HEIGHT,
            frame_count=int(frame_count),
            frame_rate=PROFILE_FPS,
            audio_rate=PROFILE_AUDIO_RATE,
            video_unit_frames=video_unit_frames,
        )
        self._sessions[request_key] = AvMuxSession(path, config)
        self._video_tails[request_key] = None
        self._audio_tails[request_key] = None

    def _task(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        action: Callable[[AvMuxSession], None],
        capture: PinnedByteCapture | None,
        dependencies: tuple[concurrent.futures.Future[None], ...],
        ring_lease: H3OutputRingLease | None = None,
        *,
        profile_name: str,
    ) -> EventGatedDeferredTask:
        session = self._sessions.get(request_key)
        if session is None:
            raise RuntimeError("H3 mux session is not active")
        return EventGatedDeferredTask(
            reservation,
            lambda: action(session),
            capture=capture,
            dependencies=dependencies,
            release=None if ring_lease is None else ring_lease.release,
            defer_release=None if ring_lease is None else ring_lease.defer_until_capture_ready,
            profile_name=profile_name,
        )

    def video(
        self,
        request_key: RequestKey,
        start_unit: int,
        unit_count: int,
        capture: PinnedByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: H3OutputRingLease,
        operation_id: int,
    ) -> EventGatedDeferredTask:
        dependency = self._video_tails[request_key]
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_video(start_unit, unit_count, capture.numpy()),
            capture,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=video start_unit={start_unit} "
                f"unit_count={unit_count} rank=0"
            ),
        )
        self._video_tails[request_key] = task.promise
        return task

    def audio(
        self,
        request_key: RequestKey,
        capture: PinnedByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: H3OutputRingLease,
        operation_id: int,
    ) -> EventGatedDeferredTask:
        dependency = self._audio_tails[request_key]
        task = self._task(
            request_key,
            reservation,
            lambda session: session.write_audio(
                capture.numpy().reshape(-1).view(np.int16).reshape(-1, 2)
            ),
            capture,
            () if dependency is None else (dependency,),
            ring_lease,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=audio rank=0"
            ),
        )
        self._audio_tails[request_key] = task.promise
        return task

    def materialize(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        operation_id: int,
    ) -> EventGatedDeferredTask:
        dependencies = tuple(
            tail
            for tail in (self._video_tails[request_key], self._audio_tails[request_key])
            if tail is not None
        )
        return self._task(
            request_key,
            reservation,
            lambda session: session.close(),
            None,
            dependencies,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=materialize rank=0"
            ),
        )

    def drop(self, session_id: int) -> None:
        selected = [key for key in self._sessions if key.session_id == int(session_id)]
        for key in selected:
            session = self._sessions.pop(key)
            self._video_tails.pop(key, None)
            self._audio_tails.pop(key, None)
            session.abort()

    def close(self) -> None:
        for session in self._sessions.values():
            session.abort()
        self._sessions.clear()
        self._video_tails.clear()
        self._audio_tails.clear()


def _request_label(request_key: RequestKey) -> str:
    return f"{request_key.authority_id}:{request_key.session_id}:{request_key.epoch}"
