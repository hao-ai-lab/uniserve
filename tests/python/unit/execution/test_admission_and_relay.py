"""Admission routing + deferred decode-relay finalize behavior."""

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
from uniserve_worker.execution.engine import (
    DeferredDecodeBurstSeqResult,
    DeferredTerminalDecodeBurstSeqResult,
    DeferredTextSeqResult,
    ForwardAdmissionRouter,
    Route,
    TextDecodeRelay,
)
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


def test_decide_text_extend_plus_decode_routes_forward():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": PREFILL_UND, "token_ids": [1, 2, 3]},
        {"kind": DECODE_UND, "token_ids": [4]},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.FORWARD
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


def test_decide_encode_with_text_and_denoise_does_not_admit_whole_batch():
    router = ForwardAdmissionRouter()
    ops = [
        {"kind": DECODE_UND, "token_ids": [1]},
        {"kind": "vit_encode"},
        {"kind": DENOISE_GEN},
    ]

    decision = router.decide(ops)

    assert decision.route is Route.PER_MODE
    assert decision.use_forward is False


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


def test_request_random_streams_are_persistent_and_domain_separated():
    states = RequestStateTable()
    state = states.create_or_update(6, {"req_id": 6, "sampling": {"seed": 0}})

    text = state.device_rng("cpu", stream="text_sampling")
    image = state.device_rng("cpu", stream="model")

    assert state.device_rng("cpu", stream="text_sampling") is text
    assert state.device_rng("cpu", stream="model") is image
    assert text is not image
    expected = torch.randint(
        0,
        1000,
        (4,),
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).tolist()
    assert torch.randint(0, 1000, (4,), generator=text).tolist() == expected
    assert torch.randint(0, 1000, (4,), generator=image).tolist() == expected


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


def test_decode_relay_accepts_canonical_device_alias(monkeypatch):
    from uniserve_worker.execution import engine as text_decode_relay

    state = RequestState()
    relay_tensor = torch.tensor([7], dtype=torch.long)
    TextDecodeRelay().publish_sample(state, token_id=7, token_tensor=relay_tensor)
    monkeypatch.setattr(
        text_decode_relay,
        "canonical_device",
        lambda device: torch.device("cpu")
        if torch.device(device).type in {"cpu", "cuda"}
        else torch.device(device),
    )

    consumed = TextDecodeRelay().consume_token(
        state,
        expected_token_id=None,
        device=torch.device("cuda"),
        token_source="last_sampled",
        require=True,
    )

    assert consumed is not None
    assert consumed.data_ptr() == relay_tensor.data_ptr()


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


def test_deferred_decode_burst_result_finalizes_final_token():
    torch.manual_seed(0)
    state = RequestState()
    sampling_result = _deferred_sampling_result(17)
    relay_tensor = sampling_result.device_tokens[0:1].detach().reshape(1)
    _store_relay(state, token_id=None, tensor=relay_tensor)
    pending = DeferredTextSeqResult(
        req_id=3,
        row=0,
        state=state,
        sampling_result=sampling_result,
        relay_token_tensor=state.decode_relay.token_tensor,
    )
    burst = DeferredDecodeBurstSeqResult(
        req_id=3,
        prefix_token_ids=[11, 13],
        pending=pending,
    )

    first = burst.finalize()
    second = burst.finalize()

    assert (
        first == second == {"req_id": 3, "sampled_token_id": 17, "sampled_token_ids": [11, 13, 17]}
    )
    assert state.decode_relay.token_id == 17


def test_deferred_terminal_decode_burst_result_truncates_at_stop_token():
    first = DeferredTextSeqResult(
        req_id=3,
        row=0,
        state=RequestState(),
        sampling_result=_deferred_sampling_result(11),
        relay_token_tensor=torch.tensor([11], dtype=torch.long),
    )
    stop = DeferredTextSeqResult(
        req_id=3,
        row=0,
        state=RequestState(),
        sampling_result=_deferred_sampling_result(13),
        relay_token_tensor=torch.tensor([13], dtype=torch.long),
    )
    after = DeferredTextSeqResult(
        req_id=3,
        row=0,
        state=RequestState(),
        sampling_result=_deferred_sampling_result(17),
        relay_token_tensor=torch.tensor([17], dtype=torch.long),
    )
    burst = DeferredTerminalDecodeBurstSeqResult(
        req_id=3,
        pending_tokens=[first, stop, after],
        stop_token_ids=[13],
    )

    assert burst.finalize() == {"req_id": 3, "sampled_token_id": 13, "sampled_token_ids": [11, 13]}
