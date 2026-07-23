"""Copy-on-write latent trajectories owned by the worker runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock

import torch


@dataclass(frozen=True, slots=True)
class LatentRecord:
    handle: int
    session_id: int
    value: torch.Tensor
    step: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.handle < 1 or self.step < 0 or self.height < 1 or self.width < 1:
            raise ValueError("latent record geometry is invalid")
        if not self.value.is_floating_point():
            raise ValueError("latent tensors must use a floating dtype")


class LatentStore:
    def __init__(self, *, capacity_tokens: int = 0, downsample: int = 1) -> None:
        self.capacity_tokens = int(capacity_tokens)
        self.downsample = int(downsample)
        if self.capacity_tokens < 0 or self.downsample < 1:
            raise ValueError("latent store capacity and downsample are invalid")
        self._records: dict[int, LatentRecord] = {}
        self._revisions: dict[int, int] = {}
        self._next_revision = 1
        self._lock = RLock()

    def get(self, handle: int) -> LatentRecord | None:
        with self._lock:
            return self._records.get(int(handle))

    def require(self, handle: int) -> LatentRecord:
        value = self.get(handle)
        if value is None:
            raise KeyError(f"unknown latent handle {handle}")
        return value

    def resident_token_count(self) -> int:
        """Return logical image-latent tokens held by committed trajectories."""

        with self._lock:
            return sum(self._token_count(record) for record in self._records.values())

    def _token_count(self, record: LatentRecord) -> int:
        return (record.height // self.downsample) * (record.width // self.downsample)

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            handles = [
                handle
                for handle, record in self._records.items()
                if record.session_id == int(session_id)
            ]
            for handle in handles:
                del self._records[handle]
                self._revisions[handle] = self._revision()

    def snapshot_records(self, session_ids: set[int]) -> tuple[LatentRecord, ...]:
        requested = {int(value) for value in session_ids}
        with self._lock:
            return tuple(
                replace(record, value=record.value.detach().cpu().contiguous())
                for record in self._records.values()
                if record.session_id in requested
            )

    def restore_records(
        self,
        session_ids: set[int],
        records: tuple[LatentRecord, ...],
    ) -> None:
        requested = {int(value) for value in session_ids}
        staged = {int(record.handle): record for record in records}
        if len(staged) != len(records):
            raise ValueError("latent snapshot repeats a handle")
        if any(record.session_id not in requested for record in staged.values()):
            raise ValueError("latent snapshot contains an undeclared session")
        with self._lock:
            projected = {
                handle: record
                for handle, record in self._records.items()
                if record.session_id not in requested
            }
            if set(projected) & set(staged):
                raise ValueError("latent snapshot handle conflicts with another session")
            projected.update(staged)
            used = sum(self._token_count(record) for record in projected.values())
            if used > self.capacity_tokens:
                raise ValueError(
                    f"latent snapshot exceeds capacity ({used}>{self.capacity_tokens})"
                )
            replaced = [
                handle for handle, record in self._records.items() if record.session_id in requested
            ]
            self._records = projected
            for handle in (*replaced, *staged):
                self._revisions[handle] = self._revision()

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
        self._staged: dict[int, LatentRecord] = {}
        self._deleted: set[int] = set()
        self._bases: dict[int, int] = {}
        self._prior: dict[int, LatentRecord | None] = {}
        self._published: dict[int, int] = {}
        self._lock_held = False
        self._closed = False

    def view(self) -> LatentTxnView:
        self._require_open()
        return LatentTxnView(self)

    def read(self, handle: int) -> LatentRecord | None:
        self._require_open()
        handle = int(handle)
        if handle in self._deleted:
            return None
        if handle in self._staged:
            return self._staged[handle]
        return self._store.get(handle)

    def write(self, record: LatentRecord) -> None:
        self._require_open()
        if record.session_id not in self._session_ids:
            raise ValueError("latent belongs to a session outside this step")
        if record.handle not in self._bases:
            with self._store._lock:
                self._bases[record.handle] = self._store._revisions.get(record.handle, 0)
        self._deleted.discard(record.handle)
        self._staged[record.handle] = record

    def delete(self, handle: int) -> None:
        self._require_open()
        handle = int(handle)
        if handle not in self._bases:
            with self._store._lock:
                self._bases[handle] = self._store._revisions.get(handle, 0)
        self._staged.pop(handle, None)
        self._deleted.add(handle)

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
            for handle in (*self._staged, *self._deleted):
                self._prior[handle] = self._store._records.get(handle)
            for handle, record in self._staged.items():
                self._store._records[handle] = record
                revision = self._store._revision()
                self._store._revisions[handle] = revision
                self._published[handle] = revision
            for handle in self._deleted:
                self._store._records.pop(handle, None)
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
                            raise RuntimeError("published latent changed before rollback")
                        prior = self._prior[handle]
                        if prior is None:
                            self._store._records.pop(handle, None)
                        else:
                            self._store._records[handle] = prior
                        self._store._revisions[handle] = self._store._revision()
        finally:
            self._release()

    def finalize(self) -> None:
        self._require_open()
        self._release()

    def _validate(self) -> None:
        for handle, revision in self._bases.items():
            if self._store._revisions.get(handle, 0) != revision:
                raise RuntimeError("latent changed during step execution")
        projected = dict(self._store._records)
        projected.update(self._staged)
        for handle in self._deleted:
            projected.pop(handle, None)
        used = sum(self._store._token_count(record) for record in projected.values())
        if used > self._store.capacity_tokens:
            raise RuntimeError(
                f"latent residency exceeds capacity ({used}>{self._store.capacity_tokens})"
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

    def record(self, handle: int) -> LatentRecord:
        record = self._transaction.read(handle)
        if record is None:
            raise KeyError(f"unknown latent handle {handle}")
        return record

    def read(self, handle: int) -> torch.Tensor:
        return self.record(handle).value

    def write(self, handle: int, value: torch.Tensor) -> None:
        record = self.record(handle)
        self._transaction.write(replace(record, value=value))

    def put(self, record: LatentRecord) -> None:
        self._transaction.write(record)

    def delete(self, handle: int) -> None:
        self._transaction.delete(handle)


__all__ = ["LatentRecord", "LatentStore", "LatentTxn", "LatentTxnView"]
