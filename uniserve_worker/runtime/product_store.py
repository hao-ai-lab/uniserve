"""Transactional storage for encoded, transferred, and materialized products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from threading import RLock

import torch

from ..batch import SequenceMode


def encoder_handle_from_content_hash(content_hash: int) -> int:
    value = int(content_hash) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF
    return value or 0x9E3779B97F4A7C15


@dataclass(frozen=True, slots=True)
class VisionFeatureProduct:
    features: torch.Tensor
    grid: torch.Tensor | None
    height: int
    width: int
    source_base64: str

    def __post_init__(self) -> None:
        if not self.source_base64:
            raise ValueError("vision feature source image must not be empty")


@dataclass(frozen=True, slots=True)
class LatentFeatureProduct:
    latent: torch.Tensor
    height: int
    width: int
    source_base64: str

    def __post_init__(self) -> None:
        if not self.source_base64:
            raise ValueError("latent feature source image must not be empty")


@dataclass(frozen=True, slots=True)
class LogitsProduct:
    logits: torch.Tensor
    source_mode: SequenceMode
    draft_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.source_mode not in {
            SequenceMode.EXTEND,
            SequenceMode.DECODE,
            SequenceMode.VERIFY,
        }:
            raise ValueError("logits product source mode is invalid")
        if self.source_mode is not SequenceMode.VERIFY and self.draft_token_ids:
            raise ValueError("only verify logits may carry draft token ids")


class ImageRange(StrEnum):
    SIGNED_UNIT = "signed_unit"
    UNIT = "unit"


@dataclass(frozen=True, slots=True)
class ImageTensorProduct:
    image: torch.Tensor
    height: int
    width: int
    value_range: ImageRange


@dataclass(frozen=True, slots=True)
class EncodedImageProduct:
    base64: str

    def __post_init__(self) -> None:
        if not self.base64:
            raise ValueError("encoded image product must not be empty")


@dataclass(frozen=True, slots=True)
class FrameCollectionProduct:
    frames: tuple[EncodedImageProduct, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("frame collection must contain at least one frame")


ProductPayload = (
    VisionFeatureProduct
    | LatentFeatureProduct
    | LogitsProduct
    | ImageTensorProduct
    | EncodedImageProduct
    | FrameCollectionProduct
)


@dataclass(frozen=True, slots=True)
class ProductRecord:
    handle: int
    session_id: int
    payload: ProductPayload
    locator: str = ""
    content_hash: int | None = None

    def __post_init__(self) -> None:
        if self.handle < 1:
            raise ValueError("product handle must be positive")
        if not isinstance(
            self.payload,
            (
                VisionFeatureProduct,
                LatentFeatureProduct,
                LogitsProduct,
                ImageTensorProduct,
                EncodedImageProduct,
                FrameCollectionProduct,
            ),
        ):
            raise TypeError("product record payload is not a closed product variant")
        if self.content_hash is not None and self.content_hash < 1:
            raise ValueError("product content hash must be positive")
        dimensions = (
            (self.payload.height, self.payload.width)
            if isinstance(
                self.payload,
                (VisionFeatureProduct, LatentFeatureProduct, ImageTensorProduct),
            )
            else None
        )
        if dimensions is not None and min(dimensions) < 1:
            raise ValueError("product image geometry must be positive")


class ProductStore:
    def __init__(self, *, encoder_cache_budget: int = 0) -> None:
        self.encoder_cache_budget = int(encoder_cache_budget)
        self._records: dict[int, ProductRecord] = {}
        self._session_handles: dict[int, set[int]] = {}
        self._revisions: dict[int, int] = {}
        self._next_revision = 1
        self._lock = RLock()

    def get(self, handle: int) -> ProductRecord | None:
        with self._lock:
            return self._records.get(int(handle))

    def require(self, handle: int) -> ProductRecord:
        value = self.get(handle)
        if value is None:
            raise KeyError(f"unknown product handle {handle}")
        return value

    def encoder_output_count(self) -> int:
        """Return committed encoder products in the scheduler's handle unit."""

        with self._lock:
            return sum(
                isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
                for record in self._records.values()
            )

    def release(self, handles: tuple[int, ...]) -> None:
        with self._lock:
            for raw in handles:
                handle = int(raw)
                record = self._records.pop(handle, None)
                if record is not None:
                    self._session_handles.get(record.session_id, set()).discard(handle)
                    self._revisions[handle] = self._revision()

    def session_records(self, session_id: int) -> tuple[ProductRecord, ...]:
        with self._lock:
            return tuple(
                self._records[handle]
                for handle in self._session_handles.get(int(session_id), set())
                if handle in self._records
            )

    def rewrite_locators(self, session_ids: set[int], replacements: dict[str, str]) -> None:
        requested = {int(value) for value in session_ids}
        if not replacements:
            return
        with self._lock:
            for handle, record in tuple(self._records.items()):
                replacement = replacements.get(record.locator)
                if record.session_id in requested and replacement is not None:
                    self._records[handle] = replace(record, locator=replacement)

    def drop(self, session_id: int) -> None:
        with self._lock:
            handles = self._session_handles.pop(int(session_id), set())
            for handle in handles:
                self._records.pop(handle, None)
                self._revisions[handle] = self._revision()

    def snapshot_records(self, session_ids: set[int]) -> tuple[ProductRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                replace(record, payload=_snapshot_payload(record.payload))
                for record in self._records.values()
                if record.session_id in requested
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: tuple[ProductRecord, ...],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged = {int(record.handle): record for record in records}
        if len(staged) != len(records):
            raise ValueError("product snapshot repeats a handle")
        if any(record.session_id not in requested for record in staged.values()):
            raise ValueError("product snapshot contains an undeclared session")
        with self._lock:
            projected = {
                handle: record
                for handle, record in self._records.items()
                if record.session_id not in requested
            }
            if set(projected) & set(staged):
                raise ValueError("product snapshot handle conflicts with another session")
            projected.update(staged)
            used = sum(
                isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
                for record in projected.values()
            )
            if used > self.encoder_cache_budget:
                raise ValueError(
                    "product snapshot exceeds encoder-output capacity "
                    f"({used}>{self.encoder_cache_budget})"
                )
            replaced = [
                handle for handle, record in self._records.items() if record.session_id in requested
            ]
            self._records = projected
            self._session_handles = {}
            for handle, record in projected.items():
                self._session_handles.setdefault(record.session_id, set()).add(handle)
            for handle in (*replaced, *staged):
                self._revisions[handle] = self._revision()

    def begin_step(self, request_ids: set[int]) -> ProductTxn:
        return ProductTxn(self, frozenset(int(value) for value in request_ids))

    def _revision(self) -> int:
        value = self._next_revision
        self._next_revision += 1
        return value


class ProductTxn:
    def __init__(self, store: ProductStore, session_ids: frozenset[int]) -> None:
        self._store = store
        self._session_ids = session_ids
        self._staged: dict[int, ProductRecord] = {}
        self._bases: dict[int, int] = {}
        self._prior: dict[int, ProductRecord | None] = {}
        self._published: dict[int, int] = {}
        self._lock_held = False
        self._closed = False

    def view(self) -> ProductView:
        self._require_open()
        return ProductView(self)

    def stage(self, record: ProductRecord) -> None:
        self._require_open()
        if record.session_id not in self._session_ids:
            raise ValueError("product belongs to a session outside this step")
        if record.handle not in self._bases:
            with self._store._lock:
                self._bases[record.handle] = self._store._revisions.get(record.handle, 0)
        self._staged[record.handle] = record

    def read(self, handle: int) -> ProductRecord | None:
        self._require_open()
        if int(handle) in self._staged:
            return self._staged[int(handle)]
        return self._store.get(int(handle))

    def prepare(self) -> None:
        self._require_open()
        with self._store._lock:
            self._validate()

    def publish(self) -> None:
        self._require_open()
        self._store._lock.acquire()
        self._lock_held = True
        try:
            self._validate()
            for handle, record in self._staged.items():
                self._prior[handle] = self._store._records.get(handle)
                self._store._records[handle] = record
                self._store._session_handles.setdefault(record.session_id, set()).add(handle)
                revision = self._store._revision()
                self._store._revisions[handle] = revision
                self._published[handle] = revision
        except BaseException:
            self._lock_held = False
            self._store._lock.release()
            raise

    def rollback(self) -> None:
        if self._closed:
            return
        try:
            if self._published:
                with self._store._lock:
                    for handle, revision in self._published.items():
                        if self._store._revisions.get(handle) != revision:
                            raise RuntimeError("published product changed before rollback")
                        record = self._store._records.get(handle)
                        if record is not None:
                            self._store._session_handles.get(record.session_id, set()).discard(
                                handle
                            )
                        prior = self._prior[handle]
                        if prior is None:
                            self._store._records.pop(handle, None)
                        else:
                            self._store._records[handle] = prior
                            self._store._session_handles.setdefault(prior.session_id, set()).add(
                                handle
                            )
                        self._store._revisions[handle] = self._store._revision()
        finally:
            self._release()

    def finalize(self) -> None:
        self._require_open()
        self._release()

    def _validate(self) -> None:
        for handle, revision in self._bases.items():
            if self._store._revisions.get(handle, 0) != revision:
                raise RuntimeError("product changed during step execution")
        projected = dict(self._store._records)
        projected.update(self._staged)
        used = sum(
            isinstance(record.payload, (VisionFeatureProduct, LatentFeatureProduct))
            for record in projected.values()
        )
        if used > self._store.encoder_cache_budget:
            raise RuntimeError(
                "encoder-output residency exceeds capacity "
                f"({used}>{self._store.encoder_cache_budget})"
            )

    def _release(self) -> None:
        if self._lock_held:
            self._lock_held = False
            self._store._lock.release()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("product transaction is closed")


class ProductView:
    def __init__(self, transaction: ProductTxn) -> None:
        self._transaction = transaction

    def put(self, record: ProductRecord) -> None:
        self._transaction.stage(record)

    def get(self, handle: int) -> ProductRecord | None:
        return self._transaction.read(handle)

    def require(self, handle: int) -> ProductRecord:
        record = self.get(handle)
        if record is None:
            raise KeyError(f"unknown product handle {handle}")
        return record


def _snapshot_payload(payload: ProductPayload) -> ProductPayload:
    if isinstance(payload, VisionFeatureProduct):
        return replace(
            payload,
            features=payload.features.detach().cpu().contiguous(),
            grid=(None if payload.grid is None else payload.grid.detach().cpu().contiguous()),
        )
    if isinstance(payload, LatentFeatureProduct):
        return replace(payload, latent=payload.latent.detach().cpu().contiguous())
    if isinstance(payload, LogitsProduct):
        return replace(payload, logits=payload.logits.detach().cpu().contiguous())
    if isinstance(payload, ImageTensorProduct):
        return replace(payload, image=payload.image.detach().cpu().contiguous())
    return payload


__all__ = [
    "EncodedImageProduct",
    "FrameCollectionProduct",
    "ImageRange",
    "ImageTensorProduct",
    "LatentFeatureProduct",
    "LogitsProduct",
    "ProductRecord",
    "ProductPayload",
    "ProductStore",
    "ProductTxn",
    "ProductView",
    "VisionFeatureProduct",
    "encoder_handle_from_content_hash",
]
