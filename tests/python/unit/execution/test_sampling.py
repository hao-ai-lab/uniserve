"""Sampling observations preserve the selected token and filtering semantics."""

from dataclasses import replace

import pytest
import torch

from uniserve.sampling import SamplingParams
from uniserve_worker.sampling.metadata import SamplingMetadata
from uniserve_worker.sampling.sampler import device_greedy_parameters, sample


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
