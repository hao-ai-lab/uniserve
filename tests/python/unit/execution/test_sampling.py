"""Sampling observations preserve the selected token and filtering semantics."""

from contextlib import contextmanager

import pytest
import torch

from uniserve.runtime import EventPool
from uniserve.sampling import SamplingParams
from uniserve_worker.execution.output import PendingOutput
from uniserve_worker.execution.request import RequestPool
from uniserve_worker.protocol.batch import GenerationParams, NewRequest
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    DrawLayout,
    ForwardMode,
    Rng,
    SamplingState,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.sampling.metadata import SamplingMetadata
from uniserve_worker.sampling.sampler import sample
from uniserve_worker.storage.decode_state import DecodeState
from uniserve_worker.storage.output import OutputPool

pytestmark = pytest.mark.unit

_CUDA = pytest.param(
    "cuda:0",
    marks=(
        pytest.mark.gpu,
        pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is required"
        ),
    ),
)


@pytest.mark.parametrize(
    "logits,parameters",
    (
        ([2.0, 1.9, 1.8, -20.0], SamplingParams(typical_p=0.1)),
        ([1.0, 1.0, 1.0, 1.0], SamplingParams(top_k=1)),
        ([1.0, 1.0, 1.0, 1.0], SamplingParams(top_k=3, top_p=0.5)),
        ([1.0, 1.0, 1.0, 1.0], SamplingParams(top_p=0.5)),
        ([1.0, 1.0, 1.0, 1.0], SamplingParams(temperature=0.8, top_k=1)),
        (
            [1.0, 1.0, 1.0, 1.0],
            SamplingParams(temperature=0.8, top_k=3, top_p=0.5),
        ),
    ),
)
@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda:0", marks=pytest.mark.gpu))
)
@pytest.mark.parametrize("draw", (0.0, 0.25, 0.73))
def test_logprob_observation_preserves_filtered_selection(
    logits, parameters, device, draw
):
    results = []
    for observations in (
        {},
        {"return_logprobs": True},
        {"n_logprobs": 2},
        {"logprob_token_ids": (0, 3)},
    ):
        params = parameters.replace(**observations)
        greedy = params.device_greedy()
        task = SamplingMetadata(
            logits=torch.tensor([logits], device=device),
            parameters=params,
            penalty_counts=(None,),
            allowed=(None,),
            suppress=(),
            finish_token_ids=(),
            transition_token_ids=(),
            force_finish=False,
            draws=None if greedy else torch.full((1,), draw, device=device),
            parameter_values=None
            if greedy
            else torch.tensor(
                [[params.temperature, params.top_p, params.min_p]],
                device=device,
            ),
        )
        result = sample((task,))[0]
        assert result.valid.item()
        results.append(result.tokens.item())
    assert len(set(results)) == 1
    if parameters.typical_p < 1:
        # The middle logit has surprise closest to this distribution's entropy.
        assert results[0] == 1


@pytest.mark.parametrize(
    "logits,parameters,expected",
    (
        # Only token 2 is allowed, so it holds all of the mass.
        (
            [1.0, 2.0, 3.0],
            SamplingParams(temperature=1.0, allowed_token_ids=(2,)),
            2,
        ),
        # Tokens 1 and 2 hold about .512 and .463 of the mass, so top-p .9
        # keeps them and drops tokens 0 and 3.
        (
            [0.5, 4.0, 3.9, 0.0],
            SamplingParams(temperature=1.0, top_p=0.9),
            1,
        ),
        # The top-3 candidates are 1, 2, and 0 with about .517, .468, and
        # .016; top-p .9 drops token 0, the lowest-ID candidate. CUDA routes
        # this row to the fused top-k sampler.
        (
            [0.5, 4.0, 3.9, 0.0],
            SamplingParams(temperature=1.0, top_k=3, top_p=0.9),
            1,
        ),
    ),
)
@pytest.mark.parametrize("device", ("cpu", _CUDA))
def test_zero_draw_selects_the_first_retained_token(
    logits, parameters, expected, device
):
    # A draw of exactly 0.0 is a canonical uniform value. Filtered tokens own
    # no part of the draw range, so it selects the lowest-ID retained token.
    task = SamplingMetadata(
        logits=torch.tensor([logits], device=device),
        parameters=parameters,
        penalty_counts=(None,),
        allowed=(parameters.allowed_token_ids,),
        suppress=(),
        finish_token_ids=(),
        transition_token_ids=(),
        force_finish=False,
        draws=torch.zeros((1,), device=device),
        parameter_values=torch.tensor(
            [[parameters.temperature, parameters.top_p, parameters.min_p]],
            device=device,
        ),
    )

    result = sample((task,))[0]

    assert result.valid.item()
    assert result.tokens.item() == expected


@contextmanager
def _pending_sample(call, parameters, finish=()):
    events = EventPool()
    buffers = OutputPool(capacity=1, max_words=4, event_pool=events)
    requests = RequestPool(1)
    requests.start(
        NewRequest(
            call.request_key,
            1,
            generation=GenerationParams(
                sampling=parameters, finish_token_ids=finish
            ),
        )
    )
    output = PendingOutput(
        call,
        requests.get(call.request_key.request_id),
        buffers.acquire(1, token_capacity=1),
        0,
    )
    try:
        yield output
    finally:
        output.abandon()
        buffers.close()
        requests.close()
        events.close()


@pytest.mark.parametrize("device", ("cpu", _CUDA))
@pytest.mark.parametrize(
    "seed,expected",
    ((0x0123456789ABCDEF, (4, 4, 0)), (0xFFFFFFFFFFFFFFFF, (3, 4, 3))),
)
def test_request_sampling_keeps_semantic_draws_when_rows_are_reordered(
    device, seed, expected
):
    # Dyadic probabilities and known selections are from the shared Philox
    # fixture sampling_rng_parity.json, at positions 40, 0 and 1 respectively.
    logits = torch.tensor(
        [0.125, 0.1875, 0.0625, 0.25, 0.3125, 0.0625], device=device
    ).log()
    parameters = SamplingParams(temperature=1.0, seed=seed)
    call = Call(
        RequestKey(7, 11, 3),
        CallId(1, 0),
        CallCoordinates(),
        ForwardMode.DECODE,
        Bounds(),
    )
    with _pending_sample(call, parameters) as output:
        tasks = tuple(
            SamplingMetadata.for_call(
                call.replace(
                    rng=Rng(seed, position, DrawLayout.TARGET_SAMPLING)
                ),
                logits,
                output,
                positions=(position,),
                request_pool_index=torch.tensor([1], device=device),
                decode_state=None,
            )
            for position in (40, 0, 1)
        )
        selected = sample(tasks)
        assert tuple(row.tokens.item() for row in selected) == expected
        assert all(row.valid.item() for row in selected)


@pytest.mark.parametrize("device", ("cpu", _CUDA))
def test_verification_counts_draft_prefixes_and_stops_at_terminal_draft(device):
    parameters = SamplingParams(frequency_penalty=1.0)
    call = Call(
        RequestKey(1, 9, 1),
        CallId(1, 0),
        CallCoordinates(),
        ForwardMode.VERIFY,
        Bounds(),
        sampling_state=SamplingState(finish_token_ids=(0,)),
    )
    states = DecodeState(
        request_pool_size=1,
        vocab_size=3,
        continuation_width=3,
        device=device,
    )
    counts = torch.tensor([2, 0, 0], dtype=torch.int32, device=device)
    states.penalty_counts[1].copy_(counts)
    with _pending_sample(call, parameters, finish=(2,)) as output:
        metadata = SamplingMetadata.for_call(
            call,
            torch.tensor([[3.0, 2.0, 1.0]] * 3, device=device),
            output,
            positions=(1, 2, 3),
            draft_token_ids=(1, 0),
            request_pool_index=torch.tensor([1], device=device),
            decode_state=states,
        )
        selected = sample((metadata,))[0]
        # Penalized argmax follows 1, 0, 1. The second accepted draft is a
        # branch stop token, so the bonus selection must not advance the call.
        assert selected.valid.item()
        assert selected.accepted_draft_count.item() == 2
        assert selected.accepted_token_count.item() == 2
        assert selected.tokens.item() == 0
        assert not selected.continuation.item()
        torch.testing.assert_close(states.penalty_counts[1], counts)
