"""Specification-derived checks for round-level token sampling."""

from __future__ import annotations

from itertools import permutations
from typing import cast

import pytest
import torch

from uniserve_worker.batch import OperationEnvelope, SamplingParams
from uniserve_worker.execution.executor import (
    _sample_task_batch,
    _SampleTask,
    _sampling_task_tensors,
    _SamplingRow,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.rng import sampling_draw_seed, uniform_samples

pytestmark = pytest.mark.unit

_ENVELOPE = cast(OperationEnvelope, None)


def _row(
    parameters: SamplingParams,
    *,
    session_seed: int,
    position: int,
    recent: tuple[int, ...] = (),
    allowed: tuple[int, ...] | None = None,
    suppress: tuple[int, ...] = (),
) -> _SamplingRow:
    counts: dict[int, int] = {}
    for token_id in recent:
        counts[token_id] = counts.get(token_id, 0) + 1
    return _SamplingRow(
        parameters=parameters,
        recent_counts=tuple(counts.items()),
        allowed=allowed,
        suppress=suppress,
        draw_seed=sampling_draw_seed(session_seed, position),
        n_logprobs=parameters.n_logprobs,
    )


def _task(logits: torch.Tensor, row: _SamplingRow) -> _SampleTask:
    values = logits.reshape(1, -1)
    noise = (
        uniform_samples(
            (values.shape[1],),
            seed=row.draw_seed,
            device=values.device,
        ).reshape_as(values)
        if row.parameters.temperature > 0
        else torch.zeros_like(values, dtype=torch.float32)
    )
    penalty_token_ids, penalty_counts, parameter_values = _sampling_task_tensors(
        (row,),
        vocab=int(values.shape[1]),
        device=values.device,
    )
    return _SampleTask(
        _ENVELOPE,
        values,
        (row,),
        noise,
        penalty_token_ids,
        penalty_counts,
        parameter_values,
    )


def _reference_workspace(logits: torch.Tensor, row: _SamplingRow) -> torch.Tensor:
    work = logits.float().clone()
    vocab = int(work.numel())
    if row.allowed is not None:
        allowed = tuple(dict.fromkeys(value for value in row.allowed if 0 <= value < vocab))
        selected = torch.tensor(allowed, dtype=torch.long, device=work.device)
        masked = torch.full_like(work, float("-inf"))
        masked[selected] = work[selected]
        work = masked
    suppressed = tuple(dict.fromkeys(value for value in row.suppress if 0 <= value < vocab))
    if suppressed:
        work[torch.tensor(suppressed, dtype=torch.long, device=work.device)] = float("-inf")
    for token_id, bias in row.parameters.logit_bias:
        if 0 <= token_id < vocab and not torch.isneginf(work[token_id]):
            work[token_id] += bias
    for token_id, count in row.recent_counts:
        value = work[token_id]
        if torch.isneginf(value):
            continue
        if row.parameters.repetition_penalty != 1.0:
            value = torch.where(
                value > 0,
                value / row.parameters.repetition_penalty,
                value * row.parameters.repetition_penalty,
            )
        work[token_id] = (
            value - row.parameters.frequency_penalty * count - row.parameters.presence_penalty
        )
    if row.parameters.temperature > 0:
        work /= row.parameters.temperature
    if row.parameters.min_p > 0:
        work[work < work.max() + torch.log(torch.tensor(row.parameters.min_p))] = float("-inf")
    if 0 < row.parameters.top_k < vocab:
        values, indexes = torch.topk(work, row.parameters.top_k, sorted=False)
        masked = torch.full_like(work, float("-inf"))
        masked[indexes] = values
        work = masked
    if 0 < row.parameters.top_p < 1:
        ordered, indexes = torch.sort(work, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        drop = cumulative > row.parameters.top_p
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        work[indexes[drop]] = float("-inf")
    return work


def _reference_token(logits: torch.Tensor, row: _SamplingRow) -> tuple[int, torch.Tensor]:
    work = _reference_workspace(logits, row)
    if row.parameters.temperature > 0:
        uniform = uniform_samples(
            (work.numel(),),
            seed=row.draw_seed,
            device=work.device,
        )
        work_for_draw = work - torch.log(-torch.log(uniform))
    else:
        work_for_draw = work
    return int(torch.argmax(work_for_draw)), work


def _reference_logprobs(
    work: torch.Tensor,
    token_id: int,
    parameters: SamplingParams,
) -> tuple[float, tuple[tuple[int, float, int], ...]]:
    scores = torch.log_softmax(work, dim=-1)
    selected = float(scores[token_id])
    entries = [(token_id, selected, int((scores > scores[token_id]).sum()) + 1)]
    seen = {token_id}
    count = min(parameters.n_logprobs, int(scores.numel()))
    if count:
        values, indexes = torch.topk(scores, count)
        for candidate, value in zip(indexes.tolist(), values.tolist(), strict=True):
            if candidate not in seen:
                entries.append((candidate, value, int((scores > scores[candidate]).sum()) + 1))
                seen.add(candidate)
    for candidate in dict.fromkeys(parameters.logprob_token_ids):
        if 0 <= candidate < scores.numel() and candidate not in seen:
            entries.append(
                (candidate, float(scores[candidate]), int((scores > scores[candidate]).sum()) + 1)
            )
            seen.add(candidate)
    return selected, tuple(entries)


def test_heterogeneous_rows_match_shaped_logits_plus_semantic_gumbel_noise() -> None:
    logits = (
        torch.tensor([0.2, 1.6, -0.4, 2.1, 0.7, 1.3]),
        torch.tensor([1.7, -0.2, 0.4, 0.8, 2.2, 1.1]),
        torch.tensor([-0.3, 0.2, 1.4, 1.0, 0.6, 1.8]),
    )
    rows = (
        _row(SamplingParams(), session_seed=11, position=5, allowed=(0, 1, 3, 5), suppress=(3,)),
        _row(
            SamplingParams(
                temperature=0.7,
                top_k=4,
                top_p=0.82,
                min_p=0.1,
                repetition_penalty=1.2,
                frequency_penalty=0.15,
                presence_penalty=0.05,
                logit_bias=((2, 0.8), (4, -0.2)),
            ),
            session_seed=19,
            position=8,
            recent=(3, 3, 1),
            suppress=(4,),
        ),
        _row(
            SamplingParams(temperature=1.1, top_k=2, top_p=0.95, seed=31),
            session_seed=31,
            position=13,
            allowed=(1, 2, 4, 5),
        ),
    )

    actual = _sample_task_batch(
        tuple(_task(value, row) for value, row in zip(logits, rows, strict=True))
    )
    expected = tuple(
        _reference_token(value, row)[0] for value, row in zip(logits, rows, strict=True)
    )

    assert tuple(value.token_id for value in actual) == expected
    assert actual[0].token_id == int(torch.argmax(_reference_workspace(logits[0], rows[0])))


def test_semantic_draw_is_batch_invariant_order_invariant_and_replay_stable() -> None:
    logits = (
        torch.tensor([0.2, 1.1, 0.7, -0.4, 1.8]),
        torch.tensor([1.3, 0.4, -0.2, 1.0, 0.8]),
        torch.tensor([-0.1, 1.2, 0.5, 1.6, 0.3]),
    )
    rows = tuple(
        _row(
            SamplingParams(temperature=0.9, top_k=4, top_p=0.88),
            session_seed=seed,
            position=position,
        )
        for seed, position in ((7, 3), (17, 9), (29, 15))
    )
    tasks = tuple(_task(value, row) for value, row in zip(logits, rows, strict=True))
    individual = tuple(_sample_task_batch((task,))[0].token_id for task in tasks)

    for order in permutations(range(len(tasks))):
        ordered = tuple(tasks[index] for index in order)
        sampled = _sample_task_batch(ordered)
        by_original_index = {
            original: value.token_id for original, value in zip(order, sampled, strict=True)
        }
        assert tuple(by_original_index[index] for index in range(len(tasks))) == individual
    assert tuple(value.token_id for value in _sample_task_batch(tasks)) == individual
    assert tuple(value.token_id for value in _sample_task_batch(tasks)) == individual


def test_host_visible_greedy_rows_share_the_batched_argmax_result() -> None:
    parameters = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    rows = (
        _row(parameters, session_seed=7, position=3),
        _row(parameters, session_seed=17, position=9),
    )
    logits = (
        torch.tensor([[0.2, 1.8, 0.7, -0.4]]),
        torch.tensor([[1.3, 0.4, 2.2, 1.0]]),
    )
    tasks = tuple(
        _SampleTask(
            _ENVELOPE,
            values,
            (row,),
            None,
            None,
            None,
            None,
        )
        for values, row in zip(logits, rows, strict=True)
    )

    sampled = _sample_task_batch(tasks)

    assert tuple(value.token_id for value in sampled) == (1, 2)
    assert tuple(int(value.device_token.item()) for value in sampled) == (1, 2)


def test_logprob_values_ranks_and_entry_sets_match_full_vocab_reference() -> None:
    parameters = SamplingParams(
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        min_p=0.03,
        return_logprobs=True,
        n_logprobs=3,
        logprob_token_ids=(0, 5, 2, 5),
    )
    row = _row(parameters, session_seed=41, position=12, recent=(1, 1, 4))
    logits = torch.tensor([0.1, 1.4, 0.7, 2.0, -0.2, 1.1])
    actual = _sample_task_batch((_task(logits, row),))[0]
    expected_token, work = _reference_token(logits, row)
    expected_logprob, expected_entries = _reference_logprobs(work, expected_token, parameters)

    assert actual.token_id == expected_token
    assert actual.logprob == pytest.approx(expected_logprob, abs=1e-6)
    assert actual.top_logprobs is not None
    assert tuple((token, rank) for token, _value, rank in actual.top_logprobs) == tuple(
        (token, rank) for token, _value, rank in expected_entries
    )
    assert tuple(value for _token, value, _rank in actual.top_logprobs) == pytest.approx(
        tuple(value for _token, value, _rank in expected_entries),
        abs=1e-6,
    )


def test_verify_uses_prefix_acceptance_and_the_residual_position_draw() -> None:
    parameters = SamplingParams(temperature=0.75, return_logprobs=True, n_logprobs=2)
    rows = tuple(_row(parameters, session_seed=53, position=position) for position in (21, 22, 23))
    logits = torch.tensor(
        [
            [0.2, 2.4, 0.5, 0.1],
            [1.7, 0.2, 1.1, 0.6],
            [0.3, 0.8, 1.9, 0.4],
        ]
    )
    draft = (1, 2)
    task = _SampleTask(
        _ENVELOPE,
        logits,
        rows,
        torch.stack(
            tuple(
                uniform_samples(
                    (logits.shape[1],),
                    seed=row.draw_seed,
                    device=logits.device,
                )
                for row in rows
            )
        ),
        *_sampling_task_tensors(
            rows,
            vocab=int(logits.shape[1]),
            device=logits.device,
        ),
        draft_token_ids=draft,
        acceptance_uniforms=torch.tensor([0.0, 0.99]),
    )

    actual = _sample_task_batch((task,))[0]
    residual = _reference_workspace(logits[1], rows[1])
    residual[draft[1]] = float("-inf")
    uniform = uniform_samples((residual.numel(),), seed=rows[1].draw_seed, device=residual.device)
    expected = int(torch.argmax(residual - torch.log(-torch.log(uniform))))

    assert actual.num_accepted_tokens == 1
    assert actual.token_id == expected


def test_a_policy_must_leave_a_finite_vocabulary_entry() -> None:
    parameters = SamplingParams(temperature=0.7)
    row = _row(parameters, session_seed=61, position=4, allowed=(1,), suppress=(1,))

    with pytest.raises(WorkerError, match="masked every vocabulary entry"):
        _sample_task_batch((_task(torch.tensor([0.2, 0.8, 0.1]), row),))
