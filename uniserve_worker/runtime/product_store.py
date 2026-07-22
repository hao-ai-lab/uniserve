"""System-owned storage for intermediate and final request products."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..foundation.errors import invalid_descriptor

__all__ = [
    "EncoderCache",
    "ProductRecord",
    "ProductStore",
    "ProductView",
    "encoder_handle_from_mm_hash",
]


def encoder_handle_from_mm_hash(mm_hash: int | None) -> int:
    """Derive a stable, nonzero 64-bit encoder handle from an image content hash.

    The handle must be deterministic from ``mm_hash`` and nonzero (0 means
    no handle). A SplitMix64 finalizer spreads the hash across the u64 space.
    """
    x = int(mm_hash or 0) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF
    return x or 0x9E3779B97F4A7C15


class EncoderCache:
    """Intermediate encoder-output products (``encoder_output`` resource class).

    Handle->embedding residency, keyed by the encoder handle the host issues
    (derived from ``mm_hash`` for cross-request reuse). The worker owns the
    budget and eviction (the ``free_encoder`` control).
    """

    def __init__(self, budget: int = 0) -> None:
        self._store: dict[int, Any] = {}
        self.budget = int(budget)

    def put(self, handle: int, value: Any) -> None:
        self._store[int(handle)] = value

    def get(self, handle: int) -> Any:
        return self._store.get(int(handle))

    def pop(self, handle: int) -> Any:
        return self._store.pop(int(handle), None)

    def __contains__(self, handle: int) -> bool:
        return int(handle) in self._store


@dataclass
class ProductRecord:
    """Typed request metadata used while encoding or materializing products."""

    sampling: dict[str, Any] = field(default_factory=dict)
    image: dict[str, Any] = field(default_factory=dict)
    neg_token_ids: list[int] = field(default_factory=list)
    lora_id: Any = None
    dimensions: tuple[int, int] | None = None
    context_image_feedback: bool = False
    text_branch_kvlen: int | None = None
    text_branch_pos: int | None = None


class ProductStore:
    """Own intermediate and final products and their transactional snapshots.

    Final frame bytes and request product metadata are request-scoped and
    transactional (snapshotted per step). Intermediate encoder outputs are
    content-addressed for cross-request reuse and evicted only through the
    ``free_encoder`` control, so they sit outside the step transaction.
    """

    def __init__(
        self,
        *,
        encoder_cache_budget: int = 0,
        intermediates: EncoderCache | None = None,
    ) -> None:
        self._frames: dict[int, list[bytes]] = {}
        self._records: dict[int, ProductRecord] = {}
        self.intermediates = (
            intermediates if intermediates is not None else EncoderCache(encoder_cache_budget)
        )

    def append_frame(self, request_id: int, frame: bytes) -> int:
        frames = self._frames.setdefault(int(request_id), [])
        frames.append(bytes(frame))
        return len(frames)

    def frames(self, request_id: int) -> tuple[bytes, ...]:
        return tuple(self._frames.get(int(request_id), ()))

    def release_frames(self, request_id: int) -> int:
        frames = self._frames.pop(int(request_id), None)
        return 0 if frames is None else len(frames)

    def put_intermediate(self, handle: int, payload: Any) -> None:
        self.intermediates.put(int(handle), payload)

    def intermediate(self, handle: int) -> Any:
        return self.intermediates.get(int(handle))

    def pop_intermediate(self, handle: int) -> Any:
        return self.intermediates.pop(int(handle))

    def record(self, request_id: int, state: Any) -> ProductRecord:
        request_id = int(request_id)
        record = self._records.get(request_id)
        if record is None:
            record = ProductRecord(
                sampling=dict(state.sampling or {}),
                image=dict(state.image or {}),
                neg_token_ids=list(state.neg_token_ids or []),
                lora_id=state.lora_id,
            )
            self._records[request_id] = record
            return record
        if state.sampling:
            record.sampling = dict(state.sampling)
        if state.image:
            record.image = dict(state.image)
        if state.neg_token_ids:
            record.neg_token_ids = list(state.neg_token_ids)
        if state.lora_id is not None:
            record.lora_id = state.lora_id
        return record

    def drop(self, request_id: int) -> None:
        request_id = int(request_id)
        self._frames.pop(request_id, None)
        self._records.pop(request_id, None)

    def snapshot_requests(
        self,
        request_ids: Iterable[int],
    ) -> dict[int, tuple[tuple[bytes, ...] | None, ProductRecord | None]]:
        return {
            request_id: (
                tuple(self._frames[request_id]) if request_id in self._frames else None,
                copy.deepcopy(self._records.get(request_id)),
            )
            for request_id in {int(value) for value in request_ids}
        }

    def restore_requests(
        self,
        request_ids: Iterable[int],
        snapshot: dict[int, tuple[tuple[bytes, ...] | None, ProductRecord | None]],
    ) -> None:
        for request_id in {int(value) for value in request_ids}:
            self.drop(request_id)
        for request_id, (frames, record) in snapshot.items():
            if frames is not None:
                self._frames[int(request_id)] = [bytes(frame) for frame in frames]
            if record is not None:
                self._records[int(request_id)] = record

    def view(self, request_ids: Iterable[int], sessions: Any) -> "ProductView":
        return ProductView(self, frozenset(int(value) for value in request_ids), sessions)


class ProductView:
    """Batch-bounded access to product metadata."""

    def __init__(self, store: ProductStore, request_ids: frozenset[int], sessions: Any) -> None:
        self._store = store
        self._request_ids = request_ids
        self._sessions = sessions

    def record(self, request_id: int) -> ProductRecord:
        request_id = int(request_id)
        if request_id not in self._request_ids:
            raise invalid_descriptor(f"request {request_id} is outside the product view")
        return self._store.record(request_id, self._sessions.get(request_id))

    def append_frame(self, request_id: int, frame: bytes) -> int:
        request_id = int(request_id)
        if request_id not in self._request_ids:
            raise invalid_descriptor(f"request {request_id} is outside the product view")
        return self._store.append_frame(request_id, frame)
