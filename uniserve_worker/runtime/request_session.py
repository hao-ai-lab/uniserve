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
    Close,
    Commit,
    Control,
    FixedPoint,
    ImageParams,
    Operation,
    RequestKey,
    SamplingParams,
    VersionRef,
    control_content_digest,
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
MAX_SESSION_HISTORY_POINTS = 262_144


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeState:
    logical_position: int
    rng_counter: int
    kv_length: int


@dataclass(slots=True)
class RequestSession:
    """All worker-owned scalar state and logical handles for one request.

    ``version``, ``resolved_op_id``, and ``resolved_digest`` name the latest
    device-resolved point. ``committed_point``, ``committed_op_id``, and
    ``committed_digest`` name the latest scheduler-selected semantic point.
    ``logical_position`` is independent model accounting: it is the absolute
    semantic position consumed by token and multimodal RoPE and is never used as
    lineage identity.
    """

    request_key: RequestKey
    admission_digest: str
    sampling: SamplingParams | None
    image: ImageParams | None
    negative_token_ids: tuple[int, ...]
    finish_token_ids: tuple[int, ...]
    adapter_id: int | None
    version: int = 0
    resolved_op_id: int = 0
    resolved_digest: str = ""
    committed_point: int = 0
    committed_op_id: int = 0
    committed_digest: str = ""
    public_event_limit: int = 0
    applied_control_seq: int = 0
    control_digests: dict[tuple[int, str], str] = field(default_factory=dict)
    resolved_versions: dict[int, VersionRef] = field(default_factory=dict)
    resolved_runtime: dict[int, ResolvedRuntimeState] = field(default_factory=dict)
    terminal_cutoff: VersionRef | None = None
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
            point=FixedPoint(self.committed_point, str(self.committed_digest)),
        )

    def resolved_version(self) -> VersionRef:
        """The latest worker-resolved point, including uncommitted descendants."""

        return VersionRef(
            request_key=self.request_key,
            producer_op_id=self.resolved_op_id,
            point=FixedPoint(self.version, str(self.resolved_digest)),
        )

    def rollback_snapshot(self) -> RequestSession:
        """Capture scalar transaction state without copying the lineage ledger.

        A step appends at most its own selected versions. ``StepTxn`` journals
        those exact dictionary writes, so the historical cutoff ledger remains
        shared and rollback cost is independent of generated sequence length.
        """

        return replace(
            self,
            product_handles=set(self.product_handles),
        )


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
            if session.terminal_cutoff is not None:
                raise invalid_descriptor(
                    f"operation {operation.op_id} targets a closed request lineage"
                )
            if operation.control_seq != session.applied_control_seq:
                raise invalid_descriptor(
                    f"operation {operation.op_id} requires control sequence "
                    f"{operation.control_seq}; worker applied {session.applied_control_seq}"
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
                # A device-relay successor roots on the latest worker-resolved
                # product or on a predicated alias that selected the same fixed
                # ancestor without advancing semantic state.
                if parent.producer_op_id not in session.resolved_versions:
                    raise invalid_descriptor(
                        f"operation {operation.op_id} names a device parent whose "
                        f"producer {parent.producer_op_id} is not in the "
                        "session's resolved execution window"
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
            negative_token_ids=(() if admission.und is None else admission.und.negative_token_ids),
            finish_token_ids=(() if admission.und is None else admission.und.finish_token_ids),
            adapter_id=admission.adapter_id,
            resolved_digest=admission.digest,
            committed_digest=admission.digest,
        )
        root = session.committed_version()
        session.resolved_versions[0] = root
        session.resolved_runtime[0] = ResolvedRuntimeState(
            logical_position=session.logical_position,
            rng_counter=session.rng_counter,
            kv_length=(0 if admission.und is None else int(admission.und.kv.prefix_len)),
        )
        self._sessions[session_id] = session
        return session

    def apply_controls(self, controls: Sequence[Control]) -> tuple[tuple[int, int], ...]:
        """Apply ordered semantic controls before registering new operations."""

        rewinds: list[tuple[int, int]] = []
        for control in controls:
            if isinstance(control, Commit):
                self._apply_commit(control)
            elif isinstance(control, Close):
                rewind = self._apply_close(control)
                if rewind is not None:
                    rewinds.append(rewind)
        return tuple(rewinds)

    def finalize_predicated(
        self,
        session_id: int,
        op_id: int,
        parent: VersionRef,
    ) -> tuple[VersionRef, ResolvedRuntimeState, bool]:
        """Resolve a device-predicated no-op to its predecessor's selected point."""

        session = self.get(session_id)
        with self._lock(session_id):
            selected: VersionRef | None = (
                parent
                if parent.is_fixed()
                else session.resolved_versions.get(parent.producer_op_id)
            )
            runtime = session.resolved_runtime.get(parent.producer_op_id)
            if selected is None or runtime is None:
                raise invalid_descriptor(
                    f"predicated operation {op_id} lost its resolved parent runtime state"
                )
            point = selected.point
            if not isinstance(point, FixedPoint):
                raise invalid_descriptor(
                    f"predicated operation {op_id} resolved to a non-fixed parent"
                )
            selected = VersionRef(
                request_key=selected.request_key,
                producer_op_id=selected.producer_op_id,
                point=FixedPoint(point.point_index, str(point.semantic_digest)),
            )
            session.resolved_versions[int(op_id)] = selected
            session.resolved_runtime[int(op_id)] = runtime
            latest = session.resolved_op_id == int(op_id)
            if latest:
                session.version = int(point.point_index)
                session.resolved_op_id = int(selected.producer_op_id)
                session.resolved_digest = point.semantic_digest
                session.logical_position = runtime.logical_position
                session.rng_counter = runtime.rng_counter
            return selected, runtime, latest

    def _validate_control_identity(
        self,
        session: RequestSession,
        control: Commit | Close,
    ) -> tuple[bool, tuple[int, str], str]:
        kind = "commit" if isinstance(control, Commit) else "close"
        identity = (int(control.control_seq), kind)
        digest = control_content_digest(control)
        existing = session.control_digests.get(identity)
        if existing is not None:
            if existing != digest:
                raise invalid_descriptor(
                    f"control identity {identity} conflicts with its committed content"
                )
            return True, identity, digest
        if int(control.control_seq) != session.applied_control_seq + 1:
            raise invalid_descriptor(
                f"control sequence {control.control_seq} does not follow "
                f"{session.applied_control_seq}"
            )
        return False, identity, digest

    def _apply_commit(self, control: Commit) -> None:
        session = self.get(control.request_key.session_id)
        if control.request_key != session.request_key:
            raise invalid_descriptor("commit control has a stale request key")
        duplicate, identity, digest = self._validate_control_identity(session, control)
        if duplicate:
            return
        if session.terminal_cutoff is not None:
            raise invalid_descriptor("commit control targets a closed request lineage")
        if control.expected_parent != session.committed_version():
            raise invalid_descriptor("commit control expected parent is not current")
        selected = control.selected
        if not selected.is_fixed():
            raise invalid_descriptor("commit control selected point is not fixed")
        resolved = session.resolved_versions.get(selected.producer_op_id)
        if resolved != selected:
            raise invalid_descriptor("commit control selected point was not resolved")
        point = selected.point
        assert isinstance(point, FixedPoint)
        if int(control.public_event_limit) < session.public_event_limit:
            raise invalid_descriptor("commit control regresses the public event limit")
        session.committed_point = int(point.point_index)
        session.committed_op_id = int(selected.producer_op_id)
        session.committed_digest = point.semantic_digest
        session.public_event_limit = int(control.public_event_limit)
        session.applied_control_seq = int(control.control_seq)
        session.control_digests[identity] = digest

    def _apply_close(self, control: Close) -> tuple[int, int] | None:
        session = self.get(control.request_key.session_id)
        if control.request_key != session.request_key:
            raise invalid_descriptor("close control has a stale request key")
        duplicate, identity, digest = self._validate_control_identity(session, control)
        if duplicate:
            return None
        cutoff = control.cutoff
        if not cutoff.is_fixed():
            raise invalid_descriptor("close control cutoff is not fixed")
        point = cutoff.point
        assert isinstance(point, FixedPoint)
        reachable = (
            cutoff == session.committed_version()
            or session.resolved_versions.get(cutoff.producer_op_id) == cutoff
        )
        if not reachable:
            raise invalid_descriptor("close control cutoff is not on the resolved lineage")
        if int(point.point_index) > session.version:
            raise invalid_descriptor("close control cutoff is beyond the resolved lineage")
        runtime = session.resolved_runtime.get(cutoff.producer_op_id)
        if runtime is None:
            raise invalid_descriptor("close control cutoff lost its runtime state")
        session.committed_point = int(point.point_index)
        session.committed_op_id = int(cutoff.producer_op_id)
        session.committed_digest = point.semantic_digest
        session.version = int(point.point_index)
        session.resolved_op_id = int(cutoff.producer_op_id)
        session.resolved_digest = point.semantic_digest
        session.logical_position = runtime.logical_position
        session.rng_counter = runtime.rng_counter
        session.terminal_cutoff = cutoff
        session.applied_control_seq = int(control.control_seq)
        session.control_digests[identity] = digest
        session.resolved_versions = {int(cutoff.producer_op_id): cutoff}
        session.resolved_runtime = {int(cutoff.producer_op_id): runtime}
        return session.session_id, runtime.kv_length

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
            snapshots: list[RequestSession] = []
            for session_id in requested:
                snapshot = copy.deepcopy(self.get(session_id))
                snapshot.version = snapshot.committed_point
                snapshot.resolved_op_id = snapshot.committed_op_id
                snapshot.resolved_digest = str(snapshot.committed_digest)
                snapshot.resolved_versions = {
                    snapshot.committed_op_id: snapshot.committed_version()
                }
                runtime = snapshot.resolved_runtime.get(snapshot.committed_op_id)
                if runtime is None:
                    raise invalid_descriptor("committed session runtime state is missing")
                snapshot.logical_position = runtime.logical_position
                snapshot.rng_counter = runtime.rng_counter
                snapshot.resolved_runtime = {snapshot.committed_op_id: runtime}
                snapshots.append(snapshot)
            return tuple(snapshots)
        finally:
            for lock in reversed(locks):
                lock.release()

    def snapshot_live(self, session_ids: set[int]) -> tuple[RequestSession, ...]:
        """Capture exact live session state for an atomic administrative rollback."""

        requested = sorted(int(value) for value in session_ids)
        locks = [self._lock(session_id) for session_id in requested]
        for lock in locks:
            lock.acquire()
        try:
            return tuple(copy.deepcopy(self.get(session_id)) for session_id in requested)
        finally:
            for lock in reversed(locks):
                lock.release()

    def restore_sessions(
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
                if session.epoch < 0 or session.version < 0 or session.committed_point < 0:
                    raise invalid_descriptor("session snapshot version is invalid")
                if not session.admission_digest:
                    raise invalid_descriptor("session snapshot admission identity is missing")
                if (
                    session.resolved_versions.get(session.resolved_op_id)
                    != session.resolved_version()
                    or session.resolved_runtime.get(session.resolved_op_id) is None
                    or session.resolved_versions.get(session.committed_op_id)
                    != session.committed_version()
                    or session.resolved_runtime.get(session.committed_op_id) is None
                ):
                    raise invalid_descriptor("session snapshot lineage state is incomplete")
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
    operations' declared parents are checked against the applicable committed
    or device-resolved state and every touched store opens its transaction. Only
    after device work resolves does :meth:`commit` advance each request's
    resolved point. A rejection at any point leaves no registered product,
    storage entry, or resolved version.
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
        self._history_undo: dict[
            tuple[int, str, int], tuple[dict[int, object], int, bool, object | None]
        ] = {}
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
        resolved_runtime: Mapping[int, ResolvedRuntimeState],
        publish: Callable[[], None] | None = None,
    ) -> None:
        """Advance each request to its selected resolved version and publish stores.

        ``committed`` maps each session id to the resolved fixed :class:`VersionRef`
        selected for its operation. A non-state-advancing operation names its
        own parent, so the version does not move.
        """

        self._require_open()
        for session_id in self.request_ids:
            session = self.sessions.get(session_id)
            additions = sum(
                1
                for operation in self.operations
                if operation.request_key.session_id == session_id
                and operation.advances_state
                and operation.op_id not in session.resolved_versions
            )
            if len(session.resolved_versions) + additions > MAX_SESSION_HISTORY_POINTS:
                raise invalid_descriptor("request session history capacity is exhausted")
        for operation in self.operations:
            session = self.sessions.get(operation.request_key.session_id)
            parent = operation.parent
            # A fixed parent must equal the semantic commit cursor. A device
            # parent names the latest worker-resolved predecessor, which may be
            # ahead of host observation.
            parent_matches = (
                session.committed_version() == parent
                if parent.is_fixed()
                else parent.producer_op_id in session.resolved_versions
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
                session.resolved_op_id = selected.producer_op_id
                session.resolved_digest = point.semantic_digest
                self._record_history_write(
                    operation.request_key.session_id,
                    "version",
                    cast(dict[int, object], session.resolved_versions),
                    int(selected.producer_op_id),
                )
                session.resolved_versions[selected.producer_op_id] = selected
                runtime = (
                    resolved_runtime.get(operation.request_key.session_id)
                    if operation.advances_state
                    else session.resolved_runtime.get(selected.producer_op_id)
                )
                if runtime is None:
                    raise RuntimeError("resolved operation runtime state is missing")
                self._record_history_write(
                    operation.request_key.session_id,
                    "runtime",
                    cast(dict[int, object], session.resolved_runtime),
                    int(selected.producer_op_id),
                )
                session.resolved_runtime[selected.producer_op_id] = runtime
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
            for mapping, key, existed, value in reversed(tuple(self._history_undo.values())):
                if existed:
                    mapping[key] = value
                else:
                    mapping.pop(key, None)
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

    def _record_history_write(
        self,
        session_id: int,
        ledger: str,
        mapping: dict[int, object],
        key: int,
    ) -> None:
        identity = (int(session_id), ledger, int(key))
        if identity in self._history_undo:
            return
        self._history_undo[identity] = (mapping, key, key in mapping, mapping.get(key))

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
    "MAX_SESSION_HISTORY_POINTS",
    "RequestSession",
    "ResolvedRuntimeState",
    "SessionStore",
    "StepTxn",
    "TransactionalStore",
]
