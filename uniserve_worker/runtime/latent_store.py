"""Copy-on-write latent trajectories keyed by exact product identity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from threading import RLock

import torch

from ..batch import ProductKind, ProductRef, StorageClass
from ..foundation.errors import invalid_descriptor

_LatentKey = tuple[int, int, int, int, int]


def _latent_key(reference: ProductRef) -> _LatentKey:
    request_key = reference.request_key
    return (
        int(request_key.authority_id),
        int(request_key.session_id),
        int(request_key.epoch),
        int(reference.producer_op_id),
        int(reference.output_index),
    )


@dataclass(frozen=True, slots=True)
class LatentRecord:
    reference: ProductRef
    producer_plan_digest: str
    value: torch.Tensor
    step: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if (
            self.reference.kind is not ProductKind.LATENT
            or self.reference.storage_class is not StorageClass.LATENT_ARENA
            or self.reference.generation < 1
        ):
            raise ValueError("latent record requires an exact latent-arena product reference")
        if (
            len(self.producer_plan_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.producer_plan_digest)
        ):
            raise ValueError("latent record producer plan digest is invalid")
        if self.step < 0 or self.height < 1 or self.width < 1:
            raise ValueError("latent record geometry is invalid")
        if not self.value.is_floating_point():
            raise ValueError("latent tensors must use a floating dtype")


class LatentStore:
    def __init__(self, *, capacity_bytes: int = 0) -> None:
        self.capacity_bytes = int(capacity_bytes)
        if self.capacity_bytes < 0:
            raise ValueError("latent store byte capacity is invalid")
        self._records: dict[_LatentKey, LatentRecord] = {}
        self._revisions: dict[_LatentKey, int] = {}
        self._next_revision = 1
        self._lock = RLock()

    def get(self, reference: ProductRef) -> LatentRecord | None:
        with self._lock:
            record = self._records.get(_latent_key(reference))
            if record is not None and record.reference != reference:
                raise invalid_descriptor("stale latent product generation")
            return record

    def require(self, reference: ProductRef) -> LatentRecord:
        value = self.get(reference)
        if value is None:
            raise KeyError("unknown latent product reference")
        return value

    def resident_byte_count(self) -> int:
        """Return exact tensor storage bytes held by committed trajectories."""

        with self._lock:
            return sum(self._byte_count(record) for record in self._records.values())

    @staticmethod
    def _byte_count(record: LatentRecord) -> int:
        return int(record.value.numel()) * int(record.value.element_size())

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            keys = [
                key
                for key, record in self._records.items()
                if record.reference.request_key.session_id == int(session_id)
            ]
            for key in keys:
                del self._records[key]
                self._revisions[key] = self._revision()

    def release_operations(self, releases: Iterable[tuple[object, int]]) -> None:
        identities = {(request_key, int(op_id)) for request_key, op_id in releases}
        if not identities:
            return
        with self._lock:
            keys = [
                key
                for key, record in self._records.items()
                if (
                    record.reference.request_key,
                    int(record.reference.producer_op_id),
                )
                in identities
            ]
            for key in keys:
                del self._records[key]
                self._revisions[key] = self._revision()

    def snapshot_records(self, session_ids: set[int]) -> tuple[LatentRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                replace(record, value=record.value.detach().cpu().contiguous())
                for record in self._records.values()
                if record.reference.request_key.session_id in requested
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: tuple[LatentRecord, ...],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged = {_latent_key(record.reference): record for record in records}
        if len(staged) != len(records):
            raise ValueError("latent snapshot repeats a product identity")
        if any(
            record.reference.request_key.session_id not in requested
            for record in staged.values()
        ):
            raise ValueError("latent snapshot contains an undeclared session")
        with self._lock:
            projected = {
                key: record
                for key, record in self._records.items()
                if record.reference.request_key.session_id not in requested
            }
            if set(projected) & set(staged):
                raise ValueError("latent snapshot identity conflicts with another session")
            projected.update(staged)
            used = sum(self._byte_count(record) for record in projected.values())
            if used > self.capacity_bytes:
                raise ValueError(
                    f"latent snapshot exceeds byte capacity ({used}>{self.capacity_bytes})"
                )
            replaced = [
                key
                for key, record in self._records.items()
                if record.reference.request_key.session_id in requested
            ]
            self._records = projected
            for key in (*replaced, *staged):
                self._revisions[key] = self._revision()

    def begin_step(self, request_ids: set[int]) -> LatentTxn:
        return LatentTxn(self, frozenset(int(value) for value in request_ids))

    def _revision(self) -> int:
        value = self._next_revision
        self._next_revision += 1
        return value


class LatentTxn:
    def __init__(self, store: LatentStore, session_ids: frozenset[int]) -> None:
        self._store = store
        self._session_ids = session_ids
        self._staged: dict[_LatentKey, LatentRecord] = {}
        self._deleted: dict[_LatentKey, ProductRef] = {}
        self._bases: dict[_LatentKey, int] = {}
        self._prior: dict[_LatentKey, LatentRecord | None] = {}
        self._published: dict[_LatentKey, int] = {}
        self._lock_held = False
        self._closed = False

    def view(self) -> LatentTxnView:
        self._require_open()
        return LatentTxnView(self)

    def read(self, reference: ProductRef) -> LatentRecord | None:
        self._require_open()
        key = _latent_key(reference)
        if key in self._deleted:
            return None
        record = self._staged.get(key)
        if record is not None:
            if record.reference != reference:
                raise invalid_descriptor("stale latent product generation")
            return record
        return self._store.get(reference)

    def write(self, record: LatentRecord) -> None:
        self._require_open()
        if record.reference.request_key.session_id not in self._session_ids:
            raise ValueError("latent belongs to a session outside this step")
        key = _latent_key(record.reference)
        if key not in self._bases:
            with self._store._lock:
                current = self._store._records.get(key)
                if current is not None and current.reference != record.reference:
                    raise invalid_descriptor("stale latent product generation")
                self._bases[key] = self._store._revisions.get(key, 0)
        self._deleted.pop(key, None)
        self._staged[key] = record

    def delete(self, reference: ProductRef) -> None:
        self._require_open()
        key = _latent_key(reference)
        if key not in self._bases:
            with self._store._lock:
                current = self._store._records.get(key)
                if current is not None and current.reference != reference:
                    raise invalid_descriptor("stale latent product generation")
                self._bases[key] = self._store._revisions.get(key, 0)
        self._staged.pop(key, None)
        self._deleted[key] = reference

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
            for key in (*self._staged, *self._deleted):
                self._prior[key] = self._store._records.get(key)
            for key, record in self._staged.items():
                self._store._records[key] = record
                revision = self._store._revision()
                self._store._revisions[key] = revision
                self._published[key] = revision
            for key in self._deleted:
                self._store._records.pop(key, None)
                revision = self._store._revision()
                self._store._revisions[key] = revision
                self._published[key] = revision
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
                    for key, revision in self._published.items():
                        if self._store._revisions.get(key) != revision:
                            raise RuntimeError("published latent changed before rollback")
                        prior = self._prior[key]
                        if prior is None:
                            self._store._records.pop(key, None)
                        else:
                            self._store._records[key] = prior
                        self._store._revisions[key] = self._store._revision()
        finally:
            self._release()

    def finalize(self) -> None:
        self._require_open()
        self._release()

    def _validate(self) -> None:
        for key, revision in self._bases.items():
            if self._store._revisions.get(key, 0) != revision:
                raise RuntimeError("latent changed during step execution")
        projected = dict(self._store._records)
        projected.update(self._staged)
        for key in self._deleted:
            projected.pop(key, None)
        used = sum(self._store._byte_count(record) for record in projected.values())
        if used > self._store.capacity_bytes:
            raise RuntimeError(
                f"latent residency exceeds byte capacity ({used}>{self._store.capacity_bytes})"
            )

    def _release(self) -> None:
        if self._lock_held:
            self._lock_held = False
            self._store._lock.release()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("latent transaction is closed")


class LatentTxnView:
    def __init__(self, transaction: LatentTxn) -> None:
        self._transaction = transaction

    def record(self, reference: ProductRef) -> LatentRecord:
        record = self._transaction.read(reference)
        if record is None:
            raise KeyError("unknown latent product reference")
        return record

    def read(self, reference: ProductRef) -> torch.Tensor:
        return self.record(reference).value

    def write(self, reference: ProductRef, value: torch.Tensor) -> None:
        record = self.record(reference)
        self._transaction.write(replace(record, value=value))

    def put(self, record: LatentRecord) -> None:
        self._transaction.write(record)

    def delete(self, reference: ProductRef) -> None:
        self._transaction.delete(reference)


__all__ = ["LatentRecord", "LatentStore", "LatentTxn", "LatentTxnView"]
