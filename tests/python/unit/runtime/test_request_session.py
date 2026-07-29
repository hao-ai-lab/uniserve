from __future__ import annotations

import copy

import torch

from uniserve_worker.batch import Admission, RequestKey, UndAdmission
from uniserve_worker.runtime.request_session import SampledTokenRelay, SessionStore


def test_transaction_snapshot_shares_immutable_sampled_token_relay() -> None:
    sessions = SessionStore()
    admission = Admission.create(RequestKey(0, 7, 3), und=UndAdmission())
    session = sessions.admit(admission)
    relay = SampledTokenRelay(torch.tensor([19]))
    session.last_sampled_token = relay

    snapshot = sessions.snapshot_committed({7})[0]

    assert snapshot.last_sampled_token is relay
    assert copy.deepcopy(relay) is relay


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
