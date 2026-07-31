from __future__ import annotations

import pytest

from tests.python.fixtures.depth_one import root_parent, token_operation, und_admission
from uniserve_worker.batch import (
    Admission,
    Close,
    CloseReason,
    Commit,
    Disposition,
    FixedPoint,
    RequestKey,
    TokenMode,
    UndAdmission,
    VersionRef,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.request_session import ResolvedRuntimeState, SessionStore
from uniserve_worker.runtime.snapshot_store import SnapshotProvider


def test_step_rollback_restores_scalar_and_lineage_state() -> None:
    sessions = SessionStore()
    admission = und_admission(7, block_ids=(0,))
    session = sessions.admit(admission)
    session.product_handles.update({11, 13})
    operation, _token_input = token_operation(
        admission.request_key,
        op_id=11,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    selected = VersionRef(admission.request_key, 11, FixedPoint(1, "a" * 64))
    transaction = sessions.begin_step(1, (operation,), ())

    def fail_publish() -> None:
        raise RuntimeError("publication failed")

    with pytest.raises(RuntimeError, match="publication failed"):
        transaction.commit(
            {7: selected},
            {7: ResolvedRuntimeState(logical_position=1, rng_counter=1, kv_length=1)},
            fail_publish,
        )

    restored = sessions.get(7)
    assert restored.resolved_version() == root_parent(admission)
    assert restored.product_handles == {11, 13}
    assert tuple(restored.resolved_versions) == (0,)


def _resolved_version(
    sessions: SessionStore,
    session_id: int,
    *,
    op_id: int,
    point: int,
    digest: str,
) -> VersionRef:
    session = sessions.get(session_id)
    selected = VersionRef(session.request_key, op_id, FixedPoint(point, digest))
    session.version = point
    session.resolved_op_id = op_id
    session.resolved_digest = digest
    session.resolved_versions[op_id] = selected
    session.logical_position = point
    session.rng_counter = point
    session.resolved_runtime[op_id] = ResolvedRuntimeState(
        logical_position=point,
        rng_counter=point,
        kv_length=point,
    )
    return selected


def test_commit_advances_only_the_ordered_semantic_cursor() -> None:
    sessions = SessionStore()
    session = sessions.admit(Admission.create(RequestKey(0, 7, 3), und=UndAdmission()))
    root = session.committed_version()
    selected = _resolved_version(sessions, 7, op_id=11, point=1, digest="a" * 64)

    control = Commit(
        request_key=session.request_key,
        control_seq=1,
        expected_parent=root,
        selected=selected,
        public_event_limit=4,
        disposition=Disposition.PUBLISH,
    )
    sessions.apply_controls((control,))
    sessions.apply_controls((control,))

    assert session.committed_version() == selected
    assert session.resolved_version() == selected
    assert session.applied_control_seq == 1
    assert session.public_event_limit == 4


def test_close_rewinds_resolved_descendants_to_the_exact_cutoff() -> None:
    sessions = SessionStore()
    session = sessions.admit(Admission.create(RequestKey(0, 7, 3), und=UndAdmission()))
    root = session.committed_version()
    cutoff = _resolved_version(sessions, 7, op_id=11, point=1, digest="a" * 64)
    sessions.apply_controls(
        (
            Commit(
                request_key=session.request_key,
                control_seq=1,
                expected_parent=root,
                selected=cutoff,
                public_event_limit=4,
                disposition=Disposition.PUBLISH,
            ),
        )
    )
    committed = _resolved_version(sessions, 7, op_id=12, point=2, digest="b" * 64)
    sessions.apply_controls(
        (
            Commit(
                request_key=session.request_key,
                control_seq=2,
                expected_parent=cutoff,
                selected=committed,
                public_event_limit=8,
                disposition=Disposition.PUBLISH,
            ),
        )
    )
    _resolved_version(sessions, 7, op_id=13, point=3, digest="c" * 64)
    assert session.resolved_runtime[committed.producer_op_id].kv_length == 2
    live = sessions.snapshot_live({7})[0]
    assert live.committed_version() == committed
    assert live.resolved_version().point == FixedPoint(3, "c" * 64)

    sessions.apply_controls(
        (
            Close(
                request_key=session.request_key,
                control_seq=3,
                cutoff=cutoff,
                reason=CloseReason.CANCELLED,
            ),
        )
    )

    assert session.terminal_cutoff == cutoff
    assert session.committed_version() == cutoff
    assert session.resolved_version() == cutoff
    assert session.logical_position == 1
    assert session.rng_counter == 1
    assert session.resolved_runtime[cutoff.producer_op_id].kv_length == 1
    assert session.applied_control_seq == 3
    assert tuple(session.resolved_versions.values()) == (cutoff,)

    restored = SnapshotProvider._session_from_json(SnapshotProvider._session_to_json(session))
    assert restored.committed_version() == cutoff
    assert restored.resolved_version() == cutoff
    assert restored.terminal_cutoff == cutoff
    assert restored.control_digests == session.control_digests


def test_rejected_control_identity_does_not_consume_its_sequence() -> None:
    sessions = SessionStore()
    session = sessions.admit(Admission.create(RequestKey(0, 7, 3), und=UndAdmission()))
    root = session.committed_version()
    selected = _resolved_version(sessions, 7, op_id=11, point=1, digest="a" * 64)
    invalid = Commit(
        request_key=session.request_key,
        control_seq=1,
        expected_parent=selected,
        selected=selected,
        public_event_limit=4,
        disposition=Disposition.PUBLISH,
    )
    with pytest.raises(WorkerError):
        sessions.apply_controls((invalid,))

    sessions.apply_controls(
        (
            Commit(
                request_key=session.request_key,
                control_seq=1,
                expected_parent=root,
                selected=selected,
                public_event_limit=4,
                disposition=Disposition.PUBLISH,
            ),
        )
    )
    assert session.committed_version() == selected
