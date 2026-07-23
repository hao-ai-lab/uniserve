"""Behavioral conformance at the canonical ModelExecutor boundary."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
    FlowAdmission,
    FlowDelta,
    FlowOperation,
    Guidance,
    ImageParams,
    KvAllocation,
    KvLeaseDelta,
    Operation,
    OperationEnvelope,
    SequenceAdmission,
    SequenceDelta,
    SequenceMode,
    SequenceOperation,
    TokenInput,
    TokenPolicy,
)
from uniserve_worker.forward import FlowRow, ForwardBatch, ForwardOutput
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.server.stub import StubModel

pytestmark = pytest.mark.integration


class _ObservedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural = StubModel()
        self.spec = self.neural.spec
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.flow_inputs: list[torch.Tensor] = []
        self.fault: str | None = None

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append((str(batch.route), tuple(type(row).__name__ for row in batch.rows)))
        self.flow_inputs.extend(
            row.latent.detach().clone() for row in batch.rows if isinstance(row, FlowRow)
        )
        output = self.neural(batch)
        if self.fault == "raise":
            raise RuntimeError("injected neural failure")
        if self.fault == "misaligned":
            return ForwardOutput(output.rows[:-1])
        return output


def _sequence_admission(session_id: int, block: int) -> Admission:
    return Admission.create(
        session_id,
        sequence=SequenceAdmission(kv=KvAllocation(block_ids=(block,))),
    )


def _flow_admission(session_id: int) -> Admission:
    return Admission.create(
        session_id,
        flow=FlowAdmission(ImageParams(steps=2, height=16, width=16, seed=29)),
    )


def _sequence(tokens: tuple[int, ...]) -> SequenceOperation:
    return SequenceOperation(
        SequenceMode.EXTEND,
        KvLeaseDelta(),
        (0, len(tokens)),
        TokenPolicy(),
        TokenInput(tokens),
    )


def _decode(token: int, position: int) -> SequenceOperation:
    return SequenceOperation(
        SequenceMode.DECODE,
        KvLeaseDelta(),
        (position, position + 1),
        TokenPolicy(),
        TokenInput((token,)),
    )


def _flow(handle: int) -> FlowOperation:
    return FlowOperation(
        latent_handle=handle,
        position=0,
        start_step=0,
        step_count=1,
        conditioning_position=0,
        conditioning=None,
        guidance=Guidance(1, 1.0, 1.0, "none", 0.0, (0.0, 1.0)),
        image_prompt="",
    )


def _envelope(
    worker,
    admission: Admission,
    operation: Operation,
    *,
    op_id: int,
    base_version: int = 0,
    epoch: int = 1,
) -> OperationEnvelope:
    return OperationEnvelope.create(
        session_id=admission.session_id,
        epoch=epoch,
        op_id=op_id,
        base_version=base_version,
        admission_digest=admission.digest,
        model_spec_digest=worker.model_spec_digest,
        weight_digest=worker.weight_digest,
        operation=operation,
    )


def test_mixed_token_and_flow_match_homogeneous_projection_in_one_forward():
    mixed_model = _ObservedModel()
    mixed = execution_worker(mixed_model)
    sequence_admission = _sequence_admission(1, 0)
    flow_admission = _flow_admission(2)
    sequence = _envelope(mixed, sequence_admission, _sequence((3, 4)), op_id=11)
    flow = _envelope(mixed, flow_admission, _flow(22), op_id=12)

    mixed_result = mixed.execute(
        Batch(1, (sequence_admission, flow_admission), (), (sequence, flow))
    )

    split_model = _ObservedModel()
    split = execution_worker(split_model)
    split_sequence = _envelope(split, sequence_admission, _sequence((3, 4)), op_id=11)
    split_flow = _envelope(split, flow_admission, _flow(22), op_id=12)
    sequence_result = split.execute(Batch(1, (sequence_admission,), (), (split_sequence,)))
    flow_result = split.execute(Batch(2, (flow_admission,), (), (split_flow,)))

    assert mixed_result.operations[0].delta == sequence_result.operations[0].delta
    assert mixed_result.operations[1].delta == flow_result.operations[0].delta
    torch.testing.assert_close(
        mixed.latents.require(22).value,
        split.latents.require(22).value,
        rtol=0,
        atol=0,
    )
    assert len(mixed_model.calls) == 1
    assert set(mixed_model.calls[0][1]) == {"TokenRow", "FlowRow"}
    assert len(split_model.calls) == 2
    observation = mixed.executor.runner.last_observation
    assert observation is not None
    assert observation.model_forward_calls == 1
    assert observation.row_kind_counts == (("flow", 1), ("token", 1))


def test_replay_is_one_effect_and_conflicts_or_stale_work_do_not_mutate_state():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = _sequence_admission(3, 3)
    operation = _envelope(worker, admission, _sequence((8, 9)), op_id=21)
    batch = Batch(7, (admission,), (), (operation,))

    first = worker.execute(batch)
    replayed = worker.execute(batch)
    committed = deepcopy(worker.sessions.get(3))
    kv = worker.kv.get(3)
    committed_kv = (tuple(kv.block_ids), kv.prefix_len, kv.length, kv.group_id)

    assert replayed.operations == first.operations
    assert len(model.calls) == 1
    assert worker.sessions.get(3).version == 1

    conflicting = _envelope(worker, admission, _sequence((8, 10)), op_id=21)
    with pytest.raises(WorkerError, match="conflicts with its committed digest"):
        worker.execute(Batch(8, (), (), (conflicting,)))

    stale_version = _envelope(
        worker,
        admission,
        _decode(1000, 2),
        op_id=22,
        base_version=0,
    )
    with pytest.raises(WorkerError, match="expects version 0"):
        worker.execute(Batch(9, (), (), (stale_version,)))

    stale_epoch = _envelope(
        worker,
        admission,
        _decode(1000, 2),
        op_id=23,
        base_version=1,
        epoch=2,
    )
    with pytest.raises(WorkerError, match="stale epoch"):
        worker.execute(Batch(10, (), (), (stale_epoch,)))

    current = worker.sessions.get(3)
    assert current == committed
    current_kv = worker.kv.get(3)
    assert (tuple(current_kv.block_ids), current_kv.prefix_len, current_kv.length, current_kv.group_id) == committed_kv
    assert len(model.calls) == 1


def test_output_validation_failure_rolls_back_and_exact_retry_commits_once():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = _sequence_admission(4, 4)
    initial = _envelope(worker, admission, _sequence((12, 13)), op_id=31)
    worker.execute(Batch(11, (admission,), (), (initial,)))
    committed = deepcopy(worker.sessions.get(4))
    committed_length = worker.kv.get(4).length

    retry_operation = _envelope(
        worker,
        admission,
        _decode(1000, 2),
        op_id=32,
        base_version=1,
    )
    retry_batch = Batch(12, (), (), (retry_operation,))
    model.fault = "misaligned"
    with pytest.raises(WorkerError, match="output count"):
        worker.execute(retry_batch)

    assert worker.sessions.get(4) == committed
    assert worker.kv.get(4).length == committed_length
    assert len(worker.replay.snapshot_records({4})) == 1

    model.fault = None
    result = worker.execute(retry_batch)
    assert isinstance(result.operations[0].delta, SequenceDelta)
    assert worker.sessions.get(4).version == 2
    assert worker.kv.get(4).length == committed_length + 1
    assert len(worker.replay.snapshot_records({4})) == 2


def test_failed_first_flow_attempt_reclaims_all_state_and_retries_deterministically():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = _flow_admission(5)
    operation = _envelope(worker, admission, _flow(55), op_id=41)
    batch = Batch(13, (admission,), (), (operation,))
    model.fault = "raise"

    with pytest.raises(WorkerError, match="injected neural failure"):
        worker.execute(batch)

    first_input = model.flow_inputs[-1]
    assert worker.sessions.peek(5) is None
    assert worker.latents.get(55) is None
    assert worker.kv.scratch_token_count() == 0
    assert worker.replay.snapshot_records({5}) == ()

    model.fault = None
    result = worker.execute(batch)

    assert isinstance(result.operations[0].delta, FlowDelta)
    torch.testing.assert_close(model.flow_inputs[-1], first_input, rtol=0, atol=0)
    assert worker.sessions.get(5).version == 1
    assert worker.latents.require(55).step == 1


def test_commit_publication_failure_restores_every_authority(monkeypatch):
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = _sequence_admission(6, 6)
    operation = _envelope(worker, admission, _sequence((14, 15)), op_id=51)
    batch = Batch(14, (admission,), (), (operation,))

    def fail_commit(_operations, _result, commit_state):
        def fail_publish() -> None:
            raise RuntimeError("injected commit publication failure")

        commit_state(fail_publish)

    monkeypatch.setattr(worker.replay, "commit_atomic", fail_commit)
    with pytest.raises(RuntimeError, match="commit publication"):
        worker.execute(batch)

    assert worker.sessions.peek(6) is None
    with pytest.raises(WorkerError, match="no KV state"):
        worker.kv.get(6)
    assert worker.kv.scratch_token_count() == 0
    assert worker.replay.snapshot_records({6}) == ()
