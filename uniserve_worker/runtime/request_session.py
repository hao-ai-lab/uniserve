"""Single-authority request sessions and atomic register-before-submit steps."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Protocol, TypeAlias, cast

from ..batch import (
    Admission,
    Batch,
    FixedPoint,
    ImageParams,
    Operation,
    RequestKey,
    SamplingParams,
    VersionRef,
)
from ..foundation.errors import invalid_descriptor


class SnapshotStore(Protocol):
    def snapshot_requests(self, request_ids: set[int]) -> object: ...

    def restore_requests(self, request_ids: set[int], snapshot: object) -> None: ...


class StoreTxn(Protocol):
    def prepare(self) -> None: ...

    def publish(self) -> None: ...

    def rollback(self) -> None: ...

    def finalize(self) -> None: ...


class ScratchStore(Protocol):
    def begin_step(self, request_ids: set[int]) -> StoreTxn: ...


TransactionalStore: TypeAlias = SnapshotStore | ScratchStore


@dataclass(slots=True)
class RequestSession:
    """All worker-owned scalar state and logical handles for one request.

    Committed request state is a fixed :class:`VersionRef`. ``version`` is the
    accounting point index of that committed version, ``committed_op_id`` its
    producer operation, and ``committed_digest`` its semantic digest; together
    they reconstruct the committed :class:`VersionRef` a successor's ``parent``
    must name exactly. ``logical_position`` is independent model accounting: it
    is the absolute semantic position consumed by token and multimodal RoPE and
    is never used as lineage identity.
    """

    request_key: RequestKey
    admission_digest: str
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    adapter_id: int | None
    version: int = 0
    committed_op_id: int = 0
    committed_digest: str = ""
    latent_handle: int | None = None
    product_handles: set[int] = field(default_factory=set)
    prompt_logits_handle: int | None = None
    logical_position: int = 0
    flow_step: int = 0
    rng_counter: int = 0
    last_op_id: int | None = None
    last_step_id: int | None = None

    @property
    def session_id(self) -> int:
        return self.request_key.session_id

    @property
    def epoch(self) -> int:
        return self.request_key.epoch

    def committed_version(self) -> VersionRef:
        """The fixed committed version a successor's ``parent`` must equal."""

        # ``committed_digest`` may be a deferred digest while a decode response is
        # still in flight; ``str`` finalizes it. A fixed-parent successor only
        # reaches here after a host round-trip, by which point the producing
        # response has serialized and the digest has landed, so this never stalls.
        return VersionRef(
            request_key=self.request_key,
            producer_op_id=self.committed_op_id,
            point=FixedPoint(self.version, str(self.committed_digest)),
        )

    def rollback_snapshot(self) -> RequestSession:
        """Copy mutable session-owned state while sharing immutable declarations."""

        return replace(self, product_handles=set(self.product_handles))


class SessionStore:
    """Own live request sessions and their single-writer leases."""

    def __init__(self) -> None:
        self._sessions: dict[int, RequestSession] = {}
        self._locks: dict[int, RLock] = {}
        self._index_lock = RLock()

    def get(self, session_id: int) -> RequestSession:
        try:
            return self._sessions[int(session_id)]
        except KeyError:
            raise invalid_descriptor(f"unknown session {session_id}") from None

    def peek(self, session_id: int) -> RequestSession | None:
        return self._sessions.get(int(session_id))

    def session_ids(self) -> tuple[int, ...]:
        with self._index_lock:
            return tuple(sorted(self._sessions))

    def __contains__(self, session_id: object) -> bool:
        return isinstance(session_id, int) and session_id in self._sessions

    def prepare(self, batch: Batch) -> None:
        """Admit new sessions and validate that every operation names its parent."""

        operations = {operation.request_key.session_id: operation for operation in batch.operations}
        session_ids = sorted(operations)
        locks = [self._lock(session_id) for session_id in session_ids]
        for lock in locks:
            lock.acquire()
        try:
            for admission in batch.admissions:
                self.admit(admission)
            self.validate_operations(batch.operations, batch.admissions)
        finally:
            for lock in reversed(locks):
                lock.release()

    def validate_operations(
        self,
        operations: Sequence[Operation],
        admissions: Sequence[Admission],
    ) -> None:
        admitted = {value.request_key.session_id: value for value in admissions}
        for operation in operations:
            session_id = operation.request_key.session_id
            session = self.peek(session_id)
            admission = admitted.get(session_id)
            if session is None:
                if admission is None:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} references an unknown session"
                    )
                continue
            if admission is not None and admission.digest != session.admission_digest:
                raise invalid_descriptor(
                    f"session {session_id} admission conflicts with committed state"
                )
            if operation.request_key.epoch != session.epoch:
                raise invalid_descriptor(
                    f"operation {operation.op_id} has stale epoch {operation.request_key.epoch}; "
                    f"session epoch is {session.epoch}"
                )
            parent = operation.parent
            if parent.is_fixed():
                expected = session.committed_version()
                if parent != expected:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} expects parent version {session.version}; "
                        f"its declared parent does not match committed state"
                    )
            else:
                # A device-relay successor roots on its predecessor's
                # not-yet-observed selected point. By the time the worker
                # registers it, the predecessor has committed and advanced this
                # session, so the device reference must name the session's
                # current committed producer.
                if parent.producer_op_id != session.committed_op_id:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} names a device parent whose "
                        f"producer {parent.producer_op_id} does not match the "
                        f"session's committed op {session.committed_op_id}"
                    )

    def admit(self, admission: Admission) -> RequestSession:
        session_id = admission.request_key.session_id
        existing = self.peek(session_id)
        if existing is not None:
            if existing.admission_digest != admission.digest:
                raise invalid_descriptor(
                    f"session {session_id} admission conflicts with committed state"
                )
            return existing
        session = RequestSession(
            request_key=admission.request_key,
            admission_digest=admission.digest,
            sampling=None if admission.und is None else admission.und.sampling,
            image=None if admission.gen_admission is None else admission.gen_admission.image,
            negative_token_ids=(
                () if admission.und is None else admission.und.negative_token_ids
            ),
            adapter_id=admission.adapter_id,
            committed_digest=admission.digest,
        )
        self._sessions[session_id] = session
        return session

    def begin_step(
        self,
        step_id: int,
        operations: tuple[Operation, ...],
        stores: Sequence[TransactionalStore],
    ) -> StepTxn:
        return StepTxn(
            sessions=self,
            step_id=step_id,
            operations=operations,
            stores=stores,
        )

    def drop(self, session_id: int) -> None:
        session_id = int(session_id)
        with self._index_lock:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)

    def snapshot_committed(self, session_ids: set[int]) -> tuple[RequestSession, ...]:
        requested = sorted(int(value) for value in session_ids)
        locks = [self._lock(session_id) for session_id in requested]
        for lock in locks:
            lock.acquire()
        try:
            return tuple(copy.deepcopy(self.get(session_id)) for session_id in requested)
        finally:
            for lock in reversed(locks):
                lock.release()

    def restore_committed(
        self,
        sessions: Sequence[RequestSession],
        session_ids: set[int] | None = None,
    ) -> None:
        staged = {int(session.session_id): copy.deepcopy(session) for session in sessions}
        if len(staged) != len(sessions):
            raise invalid_descriptor("session snapshot repeats a session identity")
        requested = set(staged) if session_ids is None else {int(value) for value in session_ids}
        if not set(staged) <= requested:
            raise invalid_descriptor("session snapshot contains an undeclared session")
        ordered = sorted(requested)
        locks = [self._lock(session_id) for session_id in ordered]
        for lock in locks:
            lock.acquire()
        try:
            for session_id in requested - set(staged):
                self._sessions.pop(session_id, None)
            for session_id, session in staged.items():
                if session.epoch < 0 or session.version < 0:
                    raise invalid_descriptor("session snapshot version is invalid")
                if not session.admission_digest:
                    raise invalid_descriptor("session snapshot admission identity is missing")
                self._sessions[session_id] = session
        finally:
            for lock in reversed(locks):
                lock.release()

    def discard_product_handles(self, handles: set[int]) -> set[int]:
        requested = {int(value) for value in handles}
        affected: set[int] = set()
        for session_id in self.session_ids():
            lock = self._lock(session_id)
            with lock:
                session = self._sessions.get(session_id)
                if session is None:
                    continue
                removed = session.product_handles & requested
                if removed:
                    session.product_handles.difference_update(removed)
                    if session.prompt_logits_handle in removed:
                        session.prompt_logits_handle = None
                    affected.add(session_id)
        return affected

    def _lock(self, session_id: int) -> RLock:
        with self._index_lock:
            return self._locks.setdefault(int(session_id), RLock())


@dataclass(frozen=True, slots=True)
class _SessionSnapshot:
    existed: bool
    value: RequestSession | None


class StepTxn:
    """Atomic register-before-submit scope across every touched authority.

    Registration binds and validates before the operation is runnable: the
    operations' declared parents are checked against committed state and every
    touched store opens its transaction. Only after device work resolves does
    :meth:`commit` advance each request to its selected fixed version. A
    rejection at any point leaves no registered product, storage entry, or
    committed version.
    """

    def __init__(
        self,
        *,
        sessions: SessionStore,
        step_id: int,
        operations: tuple[Operation, ...],
        stores: Sequence[TransactionalStore],
    ) -> None:
        self.sessions = sessions
        self.step_id = int(step_id)
        self.operations = operations
        self.request_ids = {value.request_key.session_id for value in operations}
        self._locks = [sessions._lock(value) for value in sorted(self.request_ids)]
        for lock in self._locks:
            lock.acquire()
        self._snapshots = {
            session_id: self._snapshot(session_id) for session_id in self.request_ids
        }
        self._store_snapshots: list[tuple[SnapshotStore, object]] = []
        self._store_transactions: list[tuple[ScratchStore, StoreTxn]] = []
        for store in stores:
            begin = getattr(store, "begin_step", None)
            if callable(begin):
                scratch = cast(ScratchStore, store)
                self._store_transactions.append((scratch, begin(self.request_ids)))
            else:
                typed = cast_snapshot_store(store)
                self._store_snapshots.append((typed, typed.snapshot_requests(self.request_ids)))
        self._closed = False

    def store_transaction(self, store: ScratchStore) -> StoreTxn:
        self._require_open()
        for candidate, transaction in self._store_transactions:
            if candidate is store:
                return transaction
        raise RuntimeError("store is not part of this step transaction")

    def commit(
        self,
        committed: Mapping[int, VersionRef],
        publish: Callable[[], None] | None = None,
    ) -> None:
        """Advance each request to its selected fixed version and publish stores.

        ``committed`` maps each session id to the fixed :class:`VersionRef`
        selected for its operation. A non-state-advancing operation names its
        own committed parent, so the version does not move.
        """

        self._require_open()
        for operation in self.operations:
            session = self.sessions.get(operation.request_key.session_id)
            parent = operation.parent
            # A fixed parent must equal the session's committed version. A device
            # parent names the predecessor that already committed and advanced
            # this session, so it is checked against that committed producer.
            parent_matches = (
                session.committed_version() == parent
                if parent.is_fixed()
                else parent.producer_op_id == session.committed_op_id
            )
            if session.request_key != operation.request_key or not parent_matches:
                raise RuntimeError(
                    f"session {operation.request_key.session_id} changed outside its transaction"
                )
        try:
            for _store, transaction in self._store_transactions:
                transaction.prepare()
            for operation in self.operations:
                session = self.sessions.get(operation.request_key.session_id)
                selected = committed[operation.request_key.session_id]
                point = selected.point
                if not isinstance(point, FixedPoint):
                    raise RuntimeError("committed version must be fixed")
                session.version = point.point_index
                session.committed_op_id = selected.producer_op_id
                session.committed_digest = point.semantic_digest
                session.last_op_id = operation.op_id
                session.last_step_id = self.step_id
            for _store, transaction in self._store_transactions:
                transaction.publish()
            if publish is not None:
                publish()
            for _store, transaction in self._store_transactions:
                transaction.finalize()
        except BaseException:
            self.rollback()
            raise
        self._close()

    def rollback(self) -> None:
        if self._closed:
            return
        try:
            for _store, transaction in reversed(self._store_transactions):
                transaction.rollback()
            for session_id, session_snapshot in self._snapshots.items():
                if not session_snapshot.existed:
                    self.sessions._sessions.pop(session_id, None)
                elif session_snapshot.value is not None:
                    self.sessions._sessions[session_id] = session_snapshot.value
            for store, store_snapshot in reversed(self._store_snapshots):
                store.restore_requests(self.request_ids, store_snapshot)
        finally:
            self._close()

    def _snapshot(self, session_id: int) -> _SessionSnapshot:
        session = self.sessions.peek(session_id)
        return _SessionSnapshot(
            existed=session is not None,
            value=None if session is None else session.rollback_snapshot(),
        )

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lock in reversed(self._locks):
            lock.release()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("step transaction is closed")


def cast_snapshot_store(value: object) -> SnapshotStore:
    return cast(SnapshotStore, value)


__all__ = [
    "RequestSession",
    "SessionStore",
    "StepTxn",
    "TransactionalStore",
]
