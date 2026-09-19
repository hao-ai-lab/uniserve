"""Bounded pinned storage retained by actual media encoders."""

from __future__ import annotations

import concurrent.futures
from threading import RLock

import torch

from uniserve.media.video import Config

from ..foundation.errors import resource_error
from .codec_process import SharedMapping, SharedSlice


class MediaBuffers:
    """Bounded pinned captures shared by bounded video decode implementations.

    Video and audio slots are leased from independent fixed free queues. The
    slots are one shared mapping, pinned for device-to-host copies when the
    rank decodes on a device, so a codec process reads a captured media unit
    where the copy left it.
    """

    def __init__(
        self,
        *,
        state_slots: int,
        unresolved_window: int,
        max_video_frames_per_round: int,
        video: Config,
        frame_rate: int,
        audio_rate: int,
        pin: bool,
    ) -> None:
        """Allocate bounded video and audio slots in one shared mapping.

        Each slot kind has an independent free-slot queue.
        """
        self.video_capacity = int(state_slots) * int(unresolved_window)
        self.audio_capacity = int(state_slots)
        if min(self.video_capacity, self.audio_capacity) < 1:
            raise ValueError("video output-ring capacities must be positive")

        # One slot holds one decode round: packed RGB24 pixels for video, or
        # the clip's stereo int16 samples (2 channels x 2 bytes) for audio.
        video_bytes = (
            int(max_video_frames_per_round)
            * int(video.frame.height)
            * int(video.frame.width)
            * 3
        )
        audio_bytes = (
            round(int(video.num_frames) * int(audio_rate) / int(frame_rate))
            * 2
            * 2
        )
        if min(video_bytes, audio_bytes) < 1:
            raise ValueError(
                "video output-ring media capacities must be positive"
            )

        # Video slots precede audio slots in the mapping; both are byte views
        # of it, so a slot's offset is its distance from the mapping's start.
        audio_start = self.video_capacity * video_bytes
        total = audio_start + self.audio_capacity * audio_bytes
        self.mapping = SharedMapping("uniserve-media-ring", total)
        self._pinned = False
        try:
            self._base = torch.frombuffer(
                self.mapping.buffer, dtype=torch.uint8
            )
            if pin:
                _register_pinned(self._base)
                self._pinned = True
        except BaseException:
            self.mapping.close()
            raise

        self._video_storage = tuple(
            self._base[index * video_bytes : (index + 1) * video_bytes]
            for index in range(self.video_capacity)
        )
        self._audio_storage = tuple(
            self._base[
                audio_start + index * audio_bytes : audio_start
                + (index + 1) * audio_bytes
            ]
            for index in range(self.audio_capacity)
        )

        # Free lists are reversed so pop() leases the lowest slot index first.
        self._video_free = list(range(self.video_capacity - 1, -1, -1))
        self._audio_free = list(range(self.audio_capacity - 1, -1, -1))
        self._lock = RLock()

    def reserve(self, kind: str) -> MediaLease:
        """Lease an unused video or audio output slot from the fixed ring."""
        if kind not in {"video", "audio"}:
            raise ValueError(f"unknown video output-ring kind {kind!r}")
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            if not free:
                raise resource_error(f"video {kind} output ring is exhausted")
            index = free.pop()
        return MediaLease(self, kind, index)

    def slice(self, capture: torch.Tensor) -> SharedSlice:
        """Name a captured view of a leased slot for a codec process."""
        if capture.device.type != "cpu" or capture.dtype is not torch.uint8:
            raise ValueError("a media capture is a CPU uint8 view")
        if not capture.is_contiguous():
            raise ValueError("a media capture is a contiguous view")
        offset = int(capture.data_ptr()) - int(self._base.data_ptr())
        nbytes = int(capture.numel())
        if offset < 0 or offset + nbytes > int(self._base.numel()):
            raise ValueError("a media capture lies outside the media ring")
        return SharedSlice(self.mapping.name, offset, nbytes)

    def _storage(self, kind: str, index: int) -> torch.Tensor:
        """Return backing storage for one typed output-ring slot."""
        values = self._video_storage if kind == "video" else self._audio_storage
        return values[int(index)]

    def _release(self, kind: str, index: int) -> None:
        """Return a typed output-ring slot to its free queue."""
        with self._lock:
            free = self._video_free if kind == "video" else self._audio_free
            capacity = (
                self.video_capacity if kind == "video" else self.audio_capacity
            )
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

    def close(self) -> None:
        """Unpin and unmap the ring once no encoder reads it."""
        if self._pinned:
            self._pinned = False
            _unregister_pinned(self._base)
        self._video_storage = ()
        self._audio_storage = ()
        del self._base
        self.mapping.close()


def _register_pinned(base: torch.Tensor) -> None:
    """Page-lock the mapping so device-to-host copies can target it."""
    cudart = torch.cuda.cudart()
    status = cudart.cudaHostRegister(int(base.data_ptr()), int(base.numel()), 0)
    if status != cudart.cudaError.success:
        raise RuntimeError(f"pinning the media ring failed: {status!r}")


def _unregister_pinned(base: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    status = cudart.cudaHostUnregister(int(base.data_ptr()))
    if status != cudart.cudaError.success:
        raise RuntimeError(f"unpinning the media ring failed: {status!r}")


class MediaLease:
    """Grants exclusive access to one video output slot.

    The access lasts until immediate or deferred release.
    """

    __slots__ = ("_ring", "kind", "index", "_released")

    def __init__(self, ring: MediaBuffers, kind: str, index: int) -> None:
        """Take exclusive ownership of one typed output-ring slot."""
        self._ring = ring
        self.kind = kind
        self.index = int(index)
        self._released = False

    @property
    def storage(self) -> torch.Tensor:
        """Expose the leased output tensor while this ring slot remains owned.

        Access after release raises an error.
        """
        if self._released:
            raise RuntimeError(
                "video output-ring storage was accessed after release"
            )
        return self._ring._storage(self.kind, self.index)

    def slice(self, capture: torch.Tensor) -> SharedSlice:
        """Name a captured view of this slot for a codec process."""
        if self._released:
            raise RuntimeError(
                "video output-ring storage was accessed after release"
            )
        return self._ring.slice(capture)

    def release(self) -> None:
        """Return this output slot to the ring exactly once."""
        if self._released:
            return
        self._released = True
        self._ring._release(self.kind, self.index)

    def defer_until_ready(
        self, completion: concurrent.futures.Future[None]
    ) -> None:
        """Keep this output slot leased.

        The lease ends when its existing device fence completes.
        """
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
