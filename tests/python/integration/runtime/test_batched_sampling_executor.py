"""Executor integration for round-level sampling tasks through the real forward.

Sampling is device postprocessing inside the token modes, so inline rows from
one submission round share a single batched sampling task, and a verifier
submits its position rows as one task. The committed tokens are the serial
oracle the stub model defines through its deterministic next-token map.
"""

from __future__ import annotations

import pytest

from tests.python.fixtures.depth_one import root_parent, token_operation, und_admission
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import Batch, SamplingParams, TokenMode
from uniserve_worker.execution import executor as executor_module
from uniserve_worker.server.stub import STUB_IMG_START_TOKEN_ID, _next_token

pytestmark = pytest.mark.integration


def _observe_sample_batches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    observed: list[tuple[int, int]] = []
    implementation = executor_module._sample_task_batch

    def wrapped(tasks, token_mirrors=None):
        observed.append((len(tasks), sum(len(task.rows) for task in tasks)))
        return implementation(tasks, token_mirrors)

    monkeypatch.setattr(executor_module, "_sample_task_batch", wrapped)
    return observed


def test_inline_rows_from_one_round_share_one_sampling_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker()
    first = und_admission(
        1, block_ids=(0,), sampling=SamplingParams(temperature=0.7, top_k=4, top_p=0.9, seed=11)
    )
    second = und_admission(
        2, block_ids=(1,), sampling=SamplingParams(temperature=0.9, top_k=3, top_p=0.85, seed=17)
    )
    first_op, first_input = token_operation(
        first.request_key, op_id=1, parent=root_parent(first), mode=TokenMode.EXTEND, tokens=(8, 9)
    )
    second_op, second_input = token_operation(
        second.request_key, op_id=2, parent=root_parent(second), mode=TokenMode.EXTEND, tokens=(12,)
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(
            step_id=1,
            admissions=(first, second),
            operations=(first_op, second_op),
            input_products=(first_input, second_input),
        )
    )

    assert observed == [(2, 2)]
    assert result.completions[0].committed_tokens == (_next_token(9),)
    assert result.completions[1].committed_tokens == (_next_token(12),)


def test_batched_decode_shares_one_sampling_task(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = execution_worker()
    admissions = (und_admission(21, block_ids=(0,)), und_admission(22, block_ids=(1,)))
    for index, admission in enumerate(admissions):
        extend, extend_input = token_operation(
            admission.request_key,
            op_id=1,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(3, 4),
        )
        worker.execute(
            Batch(
                step_id=1 + index,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )

    decode_ops = []
    decode_inputs = []
    for admission in admissions:
        session_id = admission.request_key.session_id
        operation, payload = token_operation(
            admission.request_key,
            op_id=2,
            parent=worker.sessions.get(session_id).committed_version(),
            mode=TokenMode.DECODE,
            tokens=(worker.sessions.get(session_id).version,),
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(step_id=9, admissions=(), operations=tuple(decode_ops), input_products=tuple(decode_inputs))
    )

    assert observed == [(2, 2)]
    expected = _next_token(_next_token(4))
    assert result.completions[0].committed_tokens == (expected,)
    assert result.completions[1].committed_tokens == (expected,)


def test_verify_submits_its_position_rows_as_one_sampling_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker()
    admission = und_admission(
        4, block_ids=(3,), sampling=SamplingParams(return_logprobs=True, n_logprobs=2, seed=31)
    )
    # Prime the request so the verifier's current token is the last sampled one.
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        Batch(step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,))
    )
    current = worker.sessions.get(4).last_sampled_token
    assert int(current) == _next_token(4) == 1000

    verify, verify_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=worker.sessions.get(4).committed_version(),
        mode=TokenMode.VERIFY,
        tokens=(1001, STUB_IMG_START_TOKEN_ID),
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(step_id=2, admissions=(), operations=(verify,), input_products=(verify_input,))
    )
    committed = result.completions[0].committed_tokens

    assert observed == [(1, 3)]
    assert committed == (1001, STUB_IMG_START_TOKEN_ID, 1002)
