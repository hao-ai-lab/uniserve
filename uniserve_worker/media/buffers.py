"""Bounded pinned storage retained by actual media encoders."""

from __future__ import annotations

import concurrent.futures
from threading import RLock

import torch

from uniserve.media.video import Config

from ..foundation.errors import resource_error


class MediaBuffers:
    """Bounded pinned captures shared by bounded video decode implementations."""

    def __init__(
        self,
        *,
        state_slots: int,
        unresolved_window: int,
        max_video_frames_per_round: int,
        video: Config,
        frame_rate: int,
        audio_rate: int,
    ) -> None:
        """Allocate bounded video and audio tensors with independent free-slot queues."""

        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("video output-ring capacities must be positive")

        # One slot holds one decode round: packed RGB24 pixels for video, or
        # the clip's stereo int16 samples (2 channels x 2 bytes) for audio.
        video_bytes = (
            int(max_video_frames_per_round) * int(video.frame.height) * int(video.frame.width) * 3
        )
        audio_bytes = round(int(video.num_frames) * int(audio_rate) / int(frame_rate)) * 2 * 2
        if min(video_bytes, audio_bytes) < 1:
            raise ValueError("video output-ring media capacities must be positive")

        self._video_storage = tuple(
            torch.empty(video_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            torch.empty(audio_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(self.audio_capacity)
        )

        # Free lists are reversed so pop() leases the lowest slot index first.
        self._video_free = list(range(self.video_capacity - 1, -1, -1))
        self._audio_free = list(range(self.audio_capacity - 1, -1, -1))
        self._lock = RLock()

    def reserve(self, kind: str) -> "MediaLease":
        """Lease an unused video or audio output slot from the fixed ring."""

        if kind not in {"video", "audio"}:
            raise ValueError(f"unknown video output-ring kind {kind!r}")
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            if not free:
                raise resource_error(f"video {kind} output ring is exhausted")
            index = free.pop()
        return MediaLease(self, kind, index)

    def _storage(self, kind: str, index: int) -> torch.Tensor:
        """Return backing storage for one typed output-ring slot."""

        values = self._video_storage if kind == "video" else self._audio_storage
        return values[int(index)]

    def _release(self, kind: str, index: int) -> None:
        """Return a typed output-ring slot to its free queue."""

        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            capacity = self.video_capacity if kind == "video" else self.audio_capacity
            if int(index) in free or not 0 <= int(index) < capacity:
                raise RuntimeError("video output-ring ownership is invalid")
            free.append(int(index))

    @property
    def used(self) -> tuple[int, int]:
        """Return the number of output-ring slots currently leased."""

        with self._lock:
            return (
                self.video_capacity - len(self._video_free),
                self.audio_capacity - len(self._audio_free),
            )


class MediaLease:
    """Grants exclusive access to one video output slot until immediate or deferred release."""

    __slots__ = ("_ring", "kind", "index", "_released")

    def __init__(self, ring: MediaBuffers, kind: str, index: int) -> None:
        """Take exclusive ownership of one typed output-ring slot."""

        self._ring = ring
        self.kind = kind
        self.index = int(index)
        self._released = False

    @property
    def storage(self) -> torch.Tensor:
        """Expose the leased output tensor while this ring slot remains owned."""

        if self._released:
            raise RuntimeError("video output-ring storage was accessed after release")
        return self._ring._storage(self.kind, self.index)

    def release(self) -> None:
        """Return this output slot to the ring exactly once."""

        if self._released:
            return
        self._released = True
        self._ring._release(self.kind, self.index)

    def defer_until_ready(self, completion: concurrent.futures.Future[None]) -> None:
        """Keep this output slot leased until its existing device fence completes."""

        if self._released:
            return
        if completion.done():
            self.release()
            return

        # Capture the slot identity locally: the callback fires after this
        # lease object is gone, so it must not close over ``self``.
        self._released = True
        ring, kind, index = self._ring, self.kind, self.index
        completion.add_done_callback(lambda _future: ring._release(kind, index))
