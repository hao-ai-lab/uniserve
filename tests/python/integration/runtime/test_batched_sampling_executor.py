"""Executor integration for round-level sampling tasks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
    KvAllocation,
    KvLeaseDelta,
    OperationEnvelope,
    SamplingParams,
    SequenceAdmission,
    SequenceMode,
    SequenceOperation,
    TokenInput,
    TokenPolicy,
)
from uniserve_worker.execution import executor as executor_module
from uniserve_worker.execution.executor import ModelExecutor
from uniserve_worker.server.stub import STUB_IMG_START_TOKEN_ID
from uniserve_worker.spec import OperationType

pytestmark = pytest.mark.integration


def _admission(
    session_id: int,
    block_id: int,
    sampling: SamplingParams,
) -> Admission:
    return Admission.create(
        session_id,
        sequence=SequenceAdmission(
            sampling=sampling,
            kv=KvAllocation(block_ids=(block_id,)),
        ),
    )


def _envelope(
    worker,
    admission: Admission,
    operation: SequenceOperation,
    *,
    op_id: int,
    base_version: int = 0,
) -> OperationEnvelope:
    return OperationEnvelope.create(
        session_id=admission.session_id,
        epoch=1,
        op_id=op_id,
        base_version=base_version,
        admission_digest=admission.digest,
        model_spec_digest=worker.model_spec_digest,
        weight_digest=worker.weight_digest,
        operation=operation,
    )


def _observe_sample_batches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    observed: list[tuple[int, int]] = []
    implementation = executor_module._sample_task_batch

    def wrapped(tasks, token_mirrors=None):
        observed.append((len(tasks), sum(len(task.rows) for task in tasks)))
        return implementation(tasks, token_mirrors)

    monkeypatch.setattr(executor_module, "_sample_task_batch", wrapped)
    return observed


def test_inline_rows_from_one_resume_round_share_one_sampling_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker()
    first_admission = _admission(
        1,
        0,
        SamplingParams(temperature=0.7, top_k=4, top_p=0.9, seed=11),
    )
    second_admission = _admission(
        2,
        1,
        SamplingParams(temperature=0.9, top_k=3, top_p=0.85, seed=17),
    )
    first = _envelope(
        worker,
        first_admission,
        SequenceOperation(
            SequenceMode.EXTEND,
            KvLeaseDelta(),
            (0, 2),
            TokenPolicy(recent_tokens=(8, 8)),
            TokenInput((8, 9)),
        ),
        op_id=1,
    )
    second = _envelope(
        worker,
        second_admission,
        SequenceOperation(
            SequenceMode.EXTEND,
            KvLeaseDelta(),
            (0, 1),
            TokenPolicy(suppress_tokens=(7,)),
            TokenInput((12,)),
        ),
        op_id=2,
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(
        Batch(
            1,
            (first_admission, second_admission),
            (),
            (first, second),
        )
    )

    assert observed == [(2, 2)]
    assert result.operations[0].delta.effect.sampled_token_ids == (1000,)
    assert result.operations[1].delta.effect.sampled_token_ids == (1000,)


def test_deferred_sample_operations_use_the_same_sampling_facility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker(defer_sampling=True)
    admission = _admission(
        3,
        2,
        SamplingParams(temperature=0.8, top_k=4, seed=23),
    )
    decode = _envelope(
        worker,
        admission,
        SequenceOperation(
            SequenceMode.DECODE,
            KvLeaseDelta(),
            (0, 1),
            TokenPolicy(),
            TokenInput((1000,)),
        ),
        op_id=3,
    )
    forwarded = worker.execute(Batch(1, (admission,), (), (decode,)))
    published = forwarded.operations[0].delta.effect.published_logits
    assert published is not None
    sample = _envelope(
        worker,
        admission,
        SequenceOperation(
            SequenceMode.SAMPLE,
            KvLeaseDelta(),
            (1, 1),
            TokenPolicy(),
            published,
        ),
        op_id=4,
        base_version=1,
    )
    observed = _observe_sample_batches(monkeypatch)

    sampler = ModelExecutor(
        spec=None,
        deployment=None,
        runner=None,
        attention=None,
        sessions=worker.sessions,
        kv=worker.kv,
        latents=worker.latents,
        products=worker.products,
        replay=worker.replay,
        adapters=None,
        mesh=None,
        transport=worker.mover.transport,
        tokenizer=None,
        model_spec_digest=None,
        weight_digest=None,
        allowed_operation_types=frozenset({OperationType.SEQUENCE_SAMPLE}),
        trace=worker.trace,
    )
    result = sampler.execute(Batch(2, (), (), (sample,)))

    assert observed == [(1, 1)]
    assert result.operations[0].delta.effect.sampled_token_ids == (1001,)


def test_verify_submits_its_position_rows_as_one_sampling_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker()
    admission = _admission(
        4,
        3,
        SamplingParams(return_logprobs=True, n_logprobs=2, seed=31),
    )
    verify = _envelope(
        worker,
        admission,
        SequenceOperation(
            SequenceMode.VERIFY,
            KvLeaseDelta(),
            (0, 1),
            TokenPolicy(),
            TokenInput(
                (1000,),
                draft_token_ids=(1001, STUB_IMG_START_TOKEN_ID),
            ),
        ),
        op_id=5,
    )
    observed = _observe_sample_batches(monkeypatch)

    result = worker.execute(Batch(1, (admission,), (), (verify,)))
    effect = result.operations[0].delta.effect

    assert observed == [(1, 3)]
    assert effect.accepted_draft_tokens == 2
    assert effect.sampled_token_ids == (1001, STUB_IMG_START_TOKEN_ID, 1002)
    assert effect.sampled_logprob is not None


def test_inactive_kv_publication_accepts_opaque_deferred_tokens() -> None:
    class OpaqueDeferredToken:
        def __hash__(self) -> int:
            raise AssertionError("inactive publication must accept an unresolved token")

    executor = object.__new__(ModelExecutor)
    operation = SimpleNamespace(policy=TokenPolicy())

    result = executor._publish_kv_if_requested(
        SimpleNamespace(),
        operation,
        (OpaqueDeferredToken(),),
        SimpleNamespace(),
    )

    assert result is None
