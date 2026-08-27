"""Deferred rank-zero H3 output capture and fixed-profile MP4 muxing."""

from __future__ import annotations

import concurrent.futures
from fractions import Fraction
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np
import torch

from ...batch import RequestKey
from ...foundation.errors import resource_error
from ...server.completion import DeferredCompletionTask, PinnedByteCapture
from ...server.cpu_tasks import CpuTaskReservation
from ...server.profiler import profile_range

__all__ = [
    "DeferredH3Task",
    "H3MuxCoordinator",
    "H3MuxSession",
    "H3OutputRing",
    "H3OutputRingLease",
    "require_h3_codecs",
]

VIDEO_WIDTH = 1344
VIDEO_HEIGHT = 768
VIDEO_FRAMES = 124
VIDEO_RATE = 24
AUDIO_RATE = 32_000
AUDIO_FRAME_SAMPLES = 1024
VIDEO_SLOT_BYTES = 22 * VIDEO_HEIGHT * VIDEO_WIDTH * 3
AUDIO_SLOT_BYTES = round(VIDEO_FRAMES * AUDIO_RATE / VIDEO_RATE) * 2 * 2


def require_h3_codecs() -> None:
    """Fail readiness unless the fixed encoder pair is installed."""

    try:
        import av
    except ImportError as error:
        raise RuntimeError("MiniMax H3 requires the PyAV runtime") from error
    available = set(av.codecs_available)
    missing = [name for name in ("libx264", "aac") if name not in available]
    if missing:
        raise RuntimeError(f"MiniMax H3 is missing required encoders {missing!r}")
    for name in ("libx264", "aac"):
        av.CodecContext.create(name, "w")


class H3MuxSession:
    """One request-owned, serial fixed-profile H.264/AAC container."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._container = None
        self._video = None
        self._audio = None
        self._next_unit = 0
        self._video_frames = 0
        self._audio_written = False
        self._closed = False
        self._lock = RLock()

    def _open(self) -> None:
        if self._container is not None:
            return
        import av

        self.path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(self.path), mode="w", format="mp4")
        video = container.add_stream("libx264", rate=VIDEO_RATE)
        video.width = VIDEO_WIDTH
        video.height = VIDEO_HEIGHT
        video.pix_fmt = "yuv420p"
        video.options = {"preset": "ultrafast", "tune": "zerolatency"}
        audio = container.add_stream("aac", rate=AUDIO_RATE)
        audio.layout = "stereo"
        audio.sample_rate = AUDIO_RATE
        self._container = container
        self._video = video
        self._audio = audio

    def write_video(self, unit: int, rgb24: np.ndarray) -> None:
        import av

        with self._lock:
            if self._closed or int(unit) != self._next_unit:
                raise RuntimeError("H3 video mux units are not request-ordered")
            if rgb24.ndim != 4 or rgb24.shape[1:] != (
                VIDEO_HEIGHT,
                VIDEO_WIDTH,
                3,
            ):
                raise RuntimeError("H3 video capture has invalid RGB24 geometry")
            self._open()
            container, stream = self._container, self._video
            if container is None or stream is None:
                raise RuntimeError("H3 video stream was not initialized")
            for pixels in rgb24:
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = self._video_frames
                frame.time_base = Fraction(1, VIDEO_RATE)
                for packet in stream.encode(frame):
                    container.mux(packet)
                self._video_frames += 1
            self._next_unit += 1

    def write_audio(self, pcm: np.ndarray) -> None:
        import av

        with self._lock:
            if self._closed or self._audio_written:
                raise RuntimeError("H3 audio was muxed more than once")
            if self._next_unit != 7 or pcm.ndim != 2 or pcm.shape[1] != 2:
                raise RuntimeError("H3 audio capture has invalid request state or stereo geometry")
            self._open()
            container, stream = self._container, self._audio
            if container is None or stream is None:
                raise RuntimeError("H3 audio stream was not initialized")
            target_samples = round(VIDEO_FRAMES * AUDIO_RATE / VIDEO_RATE)
            source = pcm[:target_samples]
            if source.shape[0] < target_samples:
                source = np.pad(source, ((0, target_samples - source.shape[0]), (0, 0)))
            pts = 0
            for start in range(0, target_samples, AUDIO_FRAME_SAMPLES):
                stop = min(start + AUDIO_FRAME_SAMPLES, target_samples)
                planar = np.zeros((2, AUDIO_FRAME_SAMPLES), dtype=np.int16)
                planar[:, : stop - start] = source[start:stop].T
                frame = av.AudioFrame.from_ndarray(planar, format="s16p", layout="stereo")
                frame.sample_rate = AUDIO_RATE
                frame.pts = pts
                frame.time_base = Fraction(1, AUDIO_RATE)
                for packet in stream.encode(frame):
                    container.mux(packet)
                pts += AUDIO_FRAME_SAMPLES
            self._audio_written = True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._video_frames != VIDEO_FRAMES or not self._audio_written:
                raise RuntimeError("H3 materialization is missing fixed-profile media")
            container, video, audio = self._container, self._video, self._audio
            if container is None or video is None or audio is None:
                raise RuntimeError("H3 mux session was never initialized")
            for stream in (video, audio):
                for packet in stream.encode(None):
                    container.mux(packet)
            container.close()
            self._closed = True

    def abort(self) -> None:
        with self._lock:
            completed = self._closed
            if not self._closed and self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
            self._closed = True
        if not completed:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


class H3OutputRing:
    """Bounded ownership for pinned video and audio captures."""

    def __init__(self, *, state_slots: int, unresolved_window: int) -> None:
        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("H3 output-ring capacities must be positive")
        self._video_storage = tuple(
            torch.empty(VIDEO_SLOT_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            torch.empty(AUDIO_SLOT_BYTES, dtype=torch.uint8, pin_memory=True)
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


class DeferredH3Task(DeferredCompletionTask):
    """One completion-owned CPU action submitted after its D2H capture lands."""

    __slots__ = (
        "capture",
        "reservation",
        "dependency",
        "action",
        "_future",
        "promise",
        "_submission_error",
        "ring_lease",
        "profile_name",
    )

    def __init__(
        self,
        reservation: CpuTaskReservation,
        action: Callable[[], None],
        *,
        capture: PinnedByteCapture | None = None,
        dependency: concurrent.futures.Future[None] | None = None,
        ring_lease: H3OutputRingLease | None = None,
        profile_name: str,
    ) -> None:
        self.capture = capture
        self.reservation = reservation
        self.dependency = dependency
        self.action = action
        self._future: concurrent.futures.Future[None] | None = None
        self.promise: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._submission_error: BaseException | None = None
        self.ring_lease = ring_lease
        self.profile_name = profile_name

    def _run(self) -> None:
        try:
            if self.dependency is not None:
                self.dependency.result()
            with profile_range(self.profile_name):
                self.action()
        except BaseException as error:
            self.promise.set_exception(error)
            raise
        else:
            self.promise.set_result(None)
        finally:
            if self.ring_lease is not None:
                self.ring_lease.release()

    def ready(self) -> bool:
        if self._submission_error is not None:
            return True
        if self._future is None:
            if self.capture is not None and not self.capture.ready():
                return False
            try:
                self._future = self.reservation.submit(self._run)
            except BaseException as error:
                self._submission_error = error
                if self.ring_lease is not None:
                    self.ring_lease.release()
                if not self.promise.done():
                    self.promise.set_exception(error)
                return True
        return bool(self._future.done())

    def finalize(self) -> None:
        if not self.ready():
            raise RuntimeError("H3 CPU completion was observed before it was ready")
        if self._submission_error is not None:
            raise self._submission_error
        if self._future is None:
            raise RuntimeError("H3 CPU completion lost its submitted future")
        self._future.result(timeout=0)

    def __del__(self) -> None:
        self.reservation.abandon()
        if self.ring_lease is not None:
            if self.capture is None:
                self.ring_lease.release()
            else:
                self.ring_lease.defer_until_capture_ready(self.capture)


class H3MuxCoordinator:
    """Request-indexed mux sessions with explicit per-request future chains."""

    def __init__(self) -> None:
        self._sessions: dict[RequestKey, H3MuxSession] = {}
        self._tails: dict[RequestKey, concurrent.futures.Future[None] | None] = {}

    def open(self, request_key: RequestKey, path: Path) -> None:
        if request_key in self._sessions:
            raise RuntimeError("H3 mux session is already active")
        self._sessions[request_key] = H3MuxSession(path)
        self._tails[request_key] = None

    def _task(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        action: Callable[[H3MuxSession], None],
        capture: PinnedByteCapture | None,
        ring_lease: H3OutputRingLease | None = None,
        *,
        profile_name: str,
    ) -> DeferredH3Task:
        session = self._sessions.get(request_key)
        if session is None:
            raise RuntimeError("H3 mux session is not active")
        task = DeferredH3Task(
            reservation,
            lambda: action(session),
            capture=capture,
            dependency=self._tails[request_key],
            ring_lease=ring_lease,
            profile_name=profile_name,
        )
        self._tails[request_key] = task.promise
        return task

    def video(
        self,
        request_key: RequestKey,
        unit: int,
        capture: PinnedByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: H3OutputRingLease,
        operation_id: int,
    ) -> DeferredH3Task:
        return self._task(
            request_key,
            reservation,
            lambda session: session.write_video(unit, capture.numpy()),
            capture,
            ring_lease,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=video unit={unit} rank=0"
            ),
        )

    def audio(
        self,
        request_key: RequestKey,
        capture: PinnedByteCapture,
        reservation: CpuTaskReservation,
        ring_lease: H3OutputRingLease,
        operation_id: int,
    ) -> DeferredH3Task:
        return self._task(
            request_key,
            reservation,
            lambda session: session.write_audio(
                capture.numpy().reshape(-1).view(np.int16).reshape(-1, 2)
            ),
            capture,
            ring_lease,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=audio rank=0"
            ),
        )

    def materialize(
        self,
        request_key: RequestKey,
        reservation: CpuTaskReservation,
        operation_id: int,
    ) -> DeferredH3Task:
        return self._task(
            request_key,
            reservation,
            lambda session: session.close(),
            None,
            profile_name=(
                f"uniserve.h3.mux request={_request_label(request_key)} "
                f"op={operation_id} kind=materialize rank=0"
            ),
        )

    def drop(self, session_id: int) -> None:
        selected = [key for key in self._sessions if key.session_id == int(session_id)]
        for key in selected:
            session = self._sessions.pop(key)
            self._tails.pop(key, None)
            session.abort()

    def close(self) -> None:
        for session in self._sessions.values():
            session.abort()
        self._sessions.clear()
        self._tails.clear()


def _request_label(request_key: RequestKey) -> str:
    return f"{request_key.authority_id}:{request_key.session_id}:{request_key.epoch}"
