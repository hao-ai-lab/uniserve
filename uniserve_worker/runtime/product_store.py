"""System-owned storage for request-scoped materialized products."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["ProductStore"]


class ProductStore:
    """Own materialized frame products and their transactional snapshots."""

    def __init__(self) -> None:
        self._frames: dict[int, list[bytes]] = {}

    def append_frame(self, request_id: int, frame: bytes) -> int:
        frames = self._frames.setdefault(int(request_id), [])
        frames.append(bytes(frame))
        return len(frames)

    def release_frames(self, request_id: int) -> int:
        frames = self._frames.pop(int(request_id), None)
        return 0 if frames is None else len(frames)

    def snapshot_requests(self, request_ids: Iterable[int]) -> dict[int, tuple[bytes, ...]]:
        return {
            request_id: tuple(self._frames[request_id])
            for request_id in {int(value) for value in request_ids}
            if request_id in self._frames
        }

    def restore_requests(
        self,
        request_ids: Iterable[int],
        snapshot: dict[int, tuple[bytes, ...]],
    ) -> None:
        for request_id in {int(value) for value in request_ids}:
            self._frames.pop(request_id, None)
        for request_id, frames in snapshot.items():
            self._frames[int(request_id)] = [bytes(frame) for frame in frames]
