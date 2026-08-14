from __future__ import annotations

import pytest

from uniserve_worker.batch import (
    Admission,
    Bounds,
    Close,
    CloseReason,
    Commit,
    Disposition,
    Domain,
    FixedPoint,
    Operation,
    RequestKey,
    TokenMode,
    UndAdmission,
    VersionRef,
    Work,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.request_session import ResolvedRuntimeState, SessionStore
from uniserve_worker.runtime.snapshot_store import SnapshotProvider


def _admit(sessions: SessionStore, session_id: int = 7) -> tuple[Admission, object]:
    admission = Admission.create(
        RequestKey(0, session_id, 3),
        request_pool_idx=session_id + 1,
        und=UndAdmission(),
    )
    return admission, sessions.admit(admission)


def _runtime(point: int, *, initialized: int = 4, committed: int = 0) -> ResolvedRuntimeState:
    return ResolvedRuntimeState(
        logical_position=point,
        rng_counter=point,
        latent_product=None,
        flow_step=0,
        kv_reserved_len=4,
        kv_initialized_len=initialized,
        kv_visible_len=point,
        kv_committed_len=committed,
        kv_published_len=0,
    )


def _version(request_key: RequestKey, op_id: int, point: int, byte: str) -> VersionRef:
    return VersionRef(request_key, op_id, FixedPoint(point, byte * 64))


def _verify_operation(request_key: RequestKey, parent: VersionRef, op_id: int = 11) -> Operation:
    return Operation.registered(
        request_key=request_key,
        op_id=op_id,
        parent=parent,
        work=Work.token(TokenMode.VERIFY),
        route=0,
        domain=Domain.DECODE,
        bounds=Bounds(max_points=4, max_tokens=4),
    )


def test_step_rollback_restores_scalar_and_prefix_ledgers() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    session.product_handles.update({11, 13})
    operation = _verify_operation(admission.request_key, session.committed_version())
    selected = _version(admission.request_key, 11, 2, "b")
    prefixes = (
        (_version(admission.request_key, 11, 1, "a"), _runtime(1)),
        (selected, _runtime(2)),
    )
    transaction = sessions.begin_step(1, (operation,), ())

    def fail_publish() -> None:
        raise RuntimeError("publication failed")

    with pytest.raises(RuntimeError, match="publication failed"):
        transaction.commit(
            {7: selected},
            {7: _runtime(2)},
            resolved_prefixes={7: prefixes},
            publish=fail_publish,
        )

    restored = sessions.get(7)
    assert restored.resolved_version() == session.committed_version()
    assert restored.product_handles == {11, 13}


def test_commit_selects_any_contiguous_prefix_with_its_exact_kv_extent() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    root = session.committed_version()
    operation = _verify_operation(admission.request_key, root)
    prefixes = tuple(
        (_version(admission.request_key, 11, point, byte), _runtime(point))
        for point, byte in ((1, "a"), (2, "b"), (3, "c"))
    )
    selected = prefixes[-1][0]
    transaction = sessions.begin_step(1, (operation,), ())
    transaction.commit(
        {7: selected},
        {7: prefixes[-1][1]},
        resolved_prefixes={7: prefixes},
    )

    cutoff = prefixes[1][0]
    updates = sessions.apply_controls(
        (
            Commit(
                request_key=admission.request_key,
                control_seq=1,
                expected_parent=root,
                selected=cutoff,
                public_event_limit=2,
                disposition=Disposition.PUBLISH,
            ),
        )
    )

    assert session.committed_version() == cutoff
    assert session.resolved_version() == selected
    assert session.runtime_for(cutoff) == _runtime(2)
    assert updates[0].visible_len == 2
    assert updates[0].committed_len == 2


def test_close_retracts_to_one_exact_resolved_prefix() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    root = session.committed_version()
    operation = _verify_operation(admission.request_key, root)
    prefixes = tuple(
        (_version(admission.request_key, 11, point, byte), _runtime(point))
        for point, byte in ((1, "a"), (2, "b"), (3, "c"))
    )
    transaction = sessions.begin_step(1, (operation,), ())
    transaction.commit(
        {7: prefixes[-1][0]},
        {7: prefixes[-1][1]},
        resolved_prefixes={7: prefixes},
    )
    sessions.apply_controls(
        (
            Close(
                request_key=admission.request_key,
                control_seq=1,
                cutoff=prefixes[0][0],
                reason=CloseReason.CANCELLED,
            ),
        )
    )

    cutoff = prefixes[0][0]
    assert session.terminal_cutoff == cutoff
    assert session.committed_version() == cutoff
    assert session.resolved_version() == cutoff
    assert session.logical_position == 1
    assert session.rng_counter == 1
    assert tuple(session.resolved_versions.values()) == (cutoff,)
    assert tuple(session.resolved_runtime.values()) == (_runtime(1),)

    restored = SnapshotProvider._session_from_json(SnapshotProvider._session_to_json(session))
    assert restored.committed_version() == cutoff
    assert restored.resolved_version() == cutoff
    assert restored.terminal_cutoff == cutoff
    assert restored.runtime_for(cutoff) == _runtime(1)


def test_commit_requires_the_selected_operation_to_name_the_current_parent() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    root = session.committed_version()
    operation = _verify_operation(admission.request_key, root)
    selected = _version(admission.request_key, 11, 1, "a")
    transaction = sessions.begin_step(1, (operation,), ())
    transaction.commit(
        {7: selected},
        {7: _runtime(1)},
        resolved_prefixes={7: ((selected, _runtime(1)),)},
    )
    different_parent = VersionRef(admission.request_key, 0, FixedPoint(0, "d" * 64))

    with pytest.raises(WorkerError, match="expected parent is not current"):
        sessions.apply_controls(
            (
                Commit(
                    request_key=admission.request_key,
                    control_seq=1,
                    expected_parent=different_parent,
                    selected=selected,
                    public_event_limit=1,
                    disposition=Disposition.PUBLISH,
                ),
            )
        )

    assert session.committed_version() == root
    assert session.applied_control_seq == 0


def test_operation_with_stale_epoch_is_rejected_before_registration() -> None:
    sessions = SessionStore()
    admission, _session = _admit(sessions)
    stale_key = RequestKey(
        admission.request_key.authority_id,
        admission.request_key.session_id,
        admission.request_key.epoch - 1,
    )
    stale = _verify_operation(
        stale_key,
        VersionRef(stale_key, 0, FixedPoint(0, admission.digest)),
    )

    with pytest.raises(WorkerError, match="stale epoch"):
        sessions.validate_operations((stale,), ())


def test_request_pool_binding_rejects_slot_drift_and_reuses_released_capacity() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    operation = _verify_operation(admission.request_key, session.committed_version())

    with pytest.raises(WorkerError, match="session index"):
        sessions.validate_operations(
            (operation,),
            (),
            (admission.request_pool_idx + 1,),
        )

    colliding = Admission.create(
        RequestKey(0, 8, 3),
        request_pool_idx=admission.request_pool_idx,
        und=UndAdmission(),
    )
    with pytest.raises(WorkerError, match="is occupied"):
        sessions.admit(colliding)

    sessions.drop(admission.request_key.session_id)
    assert sessions.admit(colliding).request_pool_idx == admission.request_pool_idx


def test_control_sequence_gap_cannot_advance_or_close_the_lineage() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)

    with pytest.raises(WorkerError, match="does not follow"):
        sessions.apply_controls(
            (
                Close(
                    request_key=admission.request_key,
                    control_seq=2,
                    cutoff=session.committed_version(),
                    reason=CloseReason.CANCELLED,
                ),
            )
        )

    assert session.applied_control_seq == 0
    assert session.terminal_cutoff is None


def test_close_cutoff_rejects_every_later_operation() -> None:
    sessions = SessionStore()
    admission, session = _admit(sessions)
    cutoff = session.committed_version()
    sessions.apply_controls(
        (
            Close(
                request_key=admission.request_key,
                control_seq=1,
                cutoff=cutoff,
                reason=CloseReason.CANCELLED,
            ),
        )
    )
    descendant = _verify_operation(admission.request_key, cutoff)

    with pytest.raises(WorkerError, match="closed request lineage"):
        sessions.validate_operations((descendant,), ())

    assert session.committed_version() == cutoff
    assert session.resolved_version() == cutoff


def test_runtime_state_enforces_the_five_extent_order() -> None:
    with pytest.raises(WorkerError, match="not monotonically contained"):
        ResolvedRuntimeState(
            logical_position=1,
            rng_counter=1,
            latent_product=None,
            flow_step=0,
            kv_reserved_len=4,
            kv_initialized_len=2,
            kv_visible_len=3,
            kv_committed_len=1,
            kv_published_len=0,
        )
