from __future__ import annotations

import time
from typing import cast

import pytest
import torch

from uniserve_worker.batch import Operation, SamplingParams
from uniserve_worker.execution.executor import (
    _sample_task_batch,
    _SampleTask,
    _sampling_task_tensors,
    _SamplingRow,
    _semantic_sampling_draws,
)
from uniserve_worker.runtime.completion_store import CompletionArena
from uniserve_worker.runtime.rng import sampling_draw_seed

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_ENVELOPE = cast(Operation, None)


def _task(logits: torch.Tensor, parameters: SamplingParams, position: int) -> _SampleTask:
    row = _SamplingRow(
        parameters=parameters,
        recent_counts=(),
        allowed=None,
        suppress=(),
        draw_seed=sampling_draw_seed(int(parameters.seed or 0), position),
        n_logprobs=int(parameters.n_logprobs),
        finish_token_ids=(3,),
    )
    if (
        parameters.temperature <= 0
        and not parameters.return_logprobs
        and int(parameters.n_logprobs) == 0
        and not parameters.logprob_token_ids
    ):
        return _SampleTask(_ENVELOPE, logits.reshape(1, -1), (row,), None, None, None, None)
    rows = logits.reshape(1, -1)
    draws = _semantic_sampling_draws((row,), device=rows.device)
    return _SampleTask(
        _ENVELOPE,
        rows,
        (row,),
        draws,
        *_sampling_task_tensors((row,), vocab=int(rows.shape[1]), device=rows.device),
    )


def test_supported_device_sampling_returns_before_any_host_scalar_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    tasks = (
        _task(torch.tensor([0.2, 0.7, 2.1, 0.4], device=device), SamplingParams(), 3),
        _task(
            torch.tensor([1.2, 0.5, 0.8, 1.7], device=device),
            SamplingParams(temperature=0.8, top_p=0.9, seed=17),
            7,
        ),
        _task(
            torch.tensor([0.1, 1.3, 0.6, 2.0], device=device),
            SamplingParams(
                temperature=0.7,
                top_k=3,
                seed=29,
                return_logprobs=True,
                n_logprobs=2,
            ),
            11,
        ),
        _task(
            torch.tensor([0.3, 1.1, 0.8, 1.9], device=device),
            SamplingParams(temperature=0.9, top_k=3, seed=43),
            13,
        ),
    )
    warm_arena = CompletionArena(depth=1, token_capacity=3, devices=(device,))
    warm_lease = warm_arena.reserve(1, devices=(device,))
    warm_sample = _sample_task_batch((tasks[3],), warm_lease)[0]
    warm_lease.seal()
    torch.cuda.synchronize()
    int(warm_sample.token_id)
    warm_lease.observe(0, warm_lease.generation)

    arena = CompletionArena(depth=1, token_capacity=64, devices=(device,))
    lease = arena.reserve(len(tasks), devices=(device,))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("sampling observed a live device scalar on the host")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "tolist", forbidden)
        sampled = _sample_task_batch(tasks, lease)

    assert all(value.device_token is not None for value in sampled)
    assert all(value.device_finish is not None for value in sampled)
    assert not lease.ready()
    lease.seal()
    deadline = time.monotonic() + 5.0
    while not lease.ready():
        if time.monotonic() >= deadline:
            raise TimeoutError("sampling completion did not become query-ready")

    observation_order = (2, 0, 3, 1)
    observed = {index: int(sampled[index].token_id) for index in observation_order}
    tokens = tuple(observed[index] for index in range(len(sampled)))
    assert tokens[0] == 2
    assert float(cast(object, sampled[2].logprob)) <= 0.0
    top = sampled[2].top_logprobs
    assert top is not None
    entries = top.finalize() if hasattr(top, "finalize") else top
    assert entries[0][0] == tokens[2]
    for row in range(len(tasks)):
        lease.observe(row, lease.generation)


def test_top_k_fast_path_and_logprob_path_use_the_same_inverse_cdf_order() -> None:
    device = torch.device("cuda:0")
    logits = torch.tensor([0.2, 2.4, 0.7, 1.5, -0.1, 1.9], device=device)
    fast = _task(
        logits,
        SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=71),
        17,
    )
    with_logprobs = _task(
        logits,
        SamplingParams(
            temperature=0.8,
            top_k=4,
            top_p=0.9,
            seed=71,
            return_logprobs=True,
            n_logprobs=2,
        ),
        17,
    )

    fast_result = _sample_task_batch((fast,))[0]
    logprob_result = _sample_task_batch((with_logprobs,))[0]

    assert int(fast_result.token_id) == int(logprob_result.token_id)
