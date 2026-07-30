from __future__ import annotations

from uniserve_worker.batch import Admission, RequestKey, UndAdmission
from uniserve_worker.runtime.request_session import SessionStore


def test_rollback_snapshot_shares_declarations_and_owns_mutable_handles() -> None:
    sessions = SessionStore()
    admission = Admission.create(RequestKey(0, 7, 3), und=UndAdmission())
    session = sessions.admit(admission)
    session.product_handles.update({11, 13})

    snapshot = session.rollback_snapshot()

    assert snapshot.sampling is session.sampling
    assert snapshot.negative_token_ids is session.negative_token_ids
    assert snapshot.product_handles == session.product_handles
    assert snapshot.product_handles is not session.product_handles
