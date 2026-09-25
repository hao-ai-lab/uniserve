"""Sampling observations preserve the selected token and filtering semantics."""

from dataclasses import replace

import pytest
import torch

from uniserve.sampling import SamplingParams
from uniserve_worker.sampling.metadata import SamplingMetadata
from uniserve_worker.sampling.sampler import device_greedy_parameters, sample

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
        params = replace(parameters, **observations)
        greedy = device_greedy_parameters(params)
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
