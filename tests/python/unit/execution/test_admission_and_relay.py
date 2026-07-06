"""Admission routing + deferred decode-relay finalize behavior.

Covers ``ForwardAdmissionRouter.decide`` (the worker-side narrowing gate that
classifies a co-batched op group into PER_MODE / FORWARD / MODEL_CHECKED_FORWARD)
and ``DeferredTextSeqResult.finalize`` (the deferred-sampling seq-result whose
CPU token is materialized at response time without clobbering a newer relay).
"""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.contracts.op_kinds import (
    COMMIT_GEN,
    DECODE_UND,
    DENOISE_GEN,
    OP_KIND_TABLE,
    PREFILL_UND,
)
from uniserve_worker.execution.forward_admission import (
    ForwardAdmissionRouter,
    Route,
)
from uniserve_worker.execution.text_driver import DeferredTextSeqResult
from uniserve_worker.nn.sampler import DeferredBatchedSamplingResult
from uniserve_worker.runtime.request_state import RequestState, RequestStateTable

pytestmark = pytest.mark.unit


# --- ForwardAdmissionRouter.decide ----------------------------------------


def test_decide_empty_group_routes_per_mode():
    router = ForwardAdmissionRouter()

    decision = router.decide([])

    assert decision.route is Route.PER_MODE
    assert decision.use_forward is False
    assert decision.modes == ()


def test_decide_text_extend_plus_decode_routes_model_checked_forward():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": PREFILL_UND, "token_ids": [1, 2, 3]},
        {"kind": DECODE_UND, "token_ids": [4]},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.MODEL_CHECKED_FORWARD
    assert decision.requires_model_acceptance is True
    assert decision.use_forward is True


def test_decide_text_extend_plus_decode_with_spec_tokens_routes_per_mode():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": PREFILL_UND, "token_ids": [1, 2, 3]},
        {"kind": DECODE_UND, "token_ids": [4], "spec_token_ids": [9]},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.PER_MODE
    assert decision.use_forward is False


def test_decide_decode_plus_denoise_routes_forward():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": DECODE_UND, "token_ids": list(range(10))},
        {"kind": DENOISE_GEN},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.FORWARD
    assert decision.use_forward is True
    assert decision.requires_model_acceptance is False


def test_decide_prefill_plus_denoise_routes_forward():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": PREFILL_UND, "token_ids": [1, 2, 3]},
        {"kind": DENOISE_GEN},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.FORWARD
    assert decision.use_forward is True


def test_decide_decode_plus_commit_routes_forward():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": DECODE_UND, "token_ids": [1]},
        {"kind": COMMIT_GEN},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.FORWARD
    assert decision.use_forward is True


def test_decide_denoise_without_decode_routes_per_mode():
    router = ForwardAdmissionRouter()
    ops = [{"kind": DENOISE_GEN}]

    decision = router.decide(ops)

    assert decision.route is Route.PER_MODE
    assert decision.use_forward is False


@pytest.mark.parametrize("kind", sorted(OP_KIND_TABLE))
def test_decide_terminates_for_every_op_kind(kind):
    # Dispatch must always terminate at a terminal route: every known op kind has
    # an always-runnable path, so a single-op group of any kind resolves to a
    # valid Route without raising or falling off the cascade.
    router = ForwardAdmissionRouter()

    decision = router.decide([{"kind": kind, "token_ids": [1]}])

    assert decision.route in {
        Route.PER_MODE,
        Route.FORWARD,
        Route.MODEL_CHECKED_FORWARD,
    }
    assert len(decision.modes) == 1


# --- RequestStateTable block registration ----------------------------------


def test_duplicate_new_request_with_empty_blocks_does_not_clear_live_chain():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11]})

    same = states.create_or_update(6, {"req_id": 6, "block_ids": []})

    assert same is state
    assert same.block_ids == [10, 11]


def test_duplicate_new_request_merges_longer_registered_chain():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11]})

    same = states.create_or_update(6, {"req_id": 6, "block_ids": [10, 11, 12]})

    assert same is state
    assert same.block_ids == [10, 11, 12]


# --- DeferredTextSeqResult.finalize ---------------------------------------


def _deferred_sampling_result(token_id: int) -> DeferredBatchedSamplingResult:
    """A single-row deferred sampling result whose CPU copy is already resident.

    ``copy_event=None`` keeps ``finalize()`` synchronization-free, so this is
    valid and deterministic on CPU.
    """
    return DeferredBatchedSamplingResult(
        tokens_cpu=torch.tensor([token_id], dtype=torch.long),
        device_tokens=torch.tensor([token_id], dtype=torch.long),
        copy_event=None,
    )


def _store_relay(state: RequestState, token_id: int | None, tensor: torch.Tensor) -> None:
    """Mirror the driver's relay store at the public state boundary."""
    state.decode_relay.token_id = token_id
    state.decode_relay.token_tensor = tensor.detach().reshape(1)


def test_finalize_returns_its_own_sampled_token():
    torch.manual_seed(0)
    state = RequestState()
    sampling_result = _deferred_sampling_result(7)
    relay_tensor = sampling_result.device_tokens[0:1].detach().reshape(1)
    _store_relay(state, token_id=None, tensor=relay_tensor)
    deferred = DeferredTextSeqResult(
        req_id=1,
        row=0,
        state=state,
        sampling_result=sampling_result,
        relay_token_tensor=state.decode_relay.token_tensor,
    )

    result = deferred.finalize()

    assert result == {"req_id": 1, "sampled_token_id": 7}


def test_finalize_publishes_token_id_when_relay_unchanged():
    # When no later step replaced the relay tensor, the deferred finalize is the
    # authority that materializes this step's host-visible token id on the relay.
    torch.manual_seed(0)
    state = RequestState()
    sampling_result = _deferred_sampling_result(42)
    relay_tensor = sampling_result.device_tokens[0:1].detach().reshape(1)
    _store_relay(state, token_id=None, tensor=relay_tensor)
    deferred = DeferredTextSeqResult(
        req_id=5,
        row=0,
        state=state,
        sampling_result=sampling_result,
        relay_token_tensor=state.decode_relay.token_tensor,
    )

    deferred.finalize()

    assert state.decode_relay.token_id == 42


def test_finalize_does_not_clobber_newer_decode_relay_token():
    # Two-step ordering: step 1's sampling result is deferred; step 2 stores a
    # fresher token on the relay before step 1's finalize runs. The earlier
    # finalize must return its own token but leave the newer relay intact.
    torch.manual_seed(0)
    state = RequestState()

    # Step 1 (earlier): deferred result, relay points at its device token (==7).
    step1_result = _deferred_sampling_result(7)
    step1_relay = step1_result.device_tokens[0:1].detach().reshape(1)
    _store_relay(state, token_id=None, tensor=step1_relay)
    deferred_step1 = DeferredTextSeqResult(
        req_id=1,
        row=0,
        state=state,
        sampling_result=step1_result,
        relay_token_tensor=state.decode_relay.token_tensor,
    )

    # Step 2 (later) stores a fresher token (==99) on the same request's relay.
    _store_relay(state, token_id=99, tensor=torch.tensor([99], dtype=torch.long))

    # Step 1's deferred finalize now runs out of order.
    result = deferred_step1.finalize()

    assert result["sampled_token_id"] == 7
    assert state.decode_relay.token_id == 99
    assert state.decode_relay.token_tensor.tolist() == [99]


def test_finalize_is_idempotent():
    torch.manual_seed(0)
    state = RequestState()
    sampling_result = _deferred_sampling_result(13)
    relay_tensor = sampling_result.device_tokens[0:1].detach().reshape(1)
    _store_relay(state, token_id=None, tensor=relay_tensor)
    deferred = DeferredTextSeqResult(
        req_id=3,
        row=0,
        state=state,
        sampling_result=sampling_result,
        relay_token_tensor=state.decode_relay.token_tensor,
    )

    first = deferred.finalize()
    second = deferred.finalize()

    assert first == second == {"req_id": 3, "sampled_token_id": 13}
