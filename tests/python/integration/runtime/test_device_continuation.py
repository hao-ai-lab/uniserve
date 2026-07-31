from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.depth_one import (
    commit_resolved,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Batch,
    DevicePoint,
    DrawLayout,
    Operation,
    ProductKind,
    Release,
    Rng,
    SamplingParams,
    TokenMode,
    VersionRef,
)
from uniserve_worker.execution.executor import (
    completion_report_ready,
    finalize_completion_report,
)
from uniserve_worker.runtime.completion_store import CompletionLease
from uniserve_worker.server.stub import _next_token

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_token_and_finish_outputs_share_the_sampling_submission_fence() -> None:
    worker = execution_worker(device="cuda:0", pipeline_depth=2)
    admission = und_admission(29, block_ids=(0,))
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )

    worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(operation,),
            input_products=(token_input,),
        )
    )
    token = next(output for output in operation.outputs if output.kind is ProductKind.TOKEN)
    finish = next(output for output in operation.outputs if output.kind is ProductKind.FINISH)
    token_read, finish_read = worker.products.device_products.consume_batch(
        (
            (token, 2, operation.plan_digest, "cuda:0"),
            (finish, 2, operation.plan_digest, "cuda:0"),
        ),
        device="cuda:0",
    )

    assert token_read._write.producer_event is not None
    assert token_read._write.producer_event is finish_read._write.producer_event
    worker.products.device_products.record_readers(
        (token_read, finish_read),
        device="cuda:0",
    )


def test_same_request_continues_from_device_products_before_parent_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = "cuda:0"
    worker = execution_worker(device=device, pipeline_depth=2)
    warm_admission = und_admission(30, block_ids=(1,))
    warm_operation, warm_input = token_operation(
        warm_admission.request_key,
        op_id=100,
        parent=root_parent(warm_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        Batch(
            step_id=100,
            admissions=(warm_admission,),
            operations=(warm_operation,),
            input_products=(warm_input,),
        )
    )
    torch.cuda.synchronize()
    worker.drop_session(30)

    admission = und_admission(31, block_ids=(0,))
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )

    observe = {"enabled": False}
    ready = CompletionLease.ready

    def gated_ready(self: CompletionLease) -> bool:
        return observe["enabled"] and ready(self)

    monkeypatch.setattr(CompletionLease, "ready", gated_ready)
    parent_report = worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    assert not completion_report_ready(parent_report)

    device_parent = VersionRef(
        admission.request_key,
        parent.op_id,
        DevicePoint(parent.outputs[0], parent.plan_digest),
    )
    successor_template, _ = token_operation(
        admission.request_key,
        op_id=2,
        parent=device_parent,
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(
            output for output in parent.outputs if output.kind is ProductKind.COMPLETION
        ),
    )
    successor = Operation.registered(
        request_key=successor_template.request_key,
        op_id=successor_template.op_id,
        parent=device_parent,
        work=successor_template.work,
        route=successor_template.route,
        domain=successor_template.domain,
        bounds=successor_template.bounds,
        outputs=successor_template.outputs,
        predicate=successor_template.predicate,
        rng=successor_template.rng,
    )
    successor_report = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            controls=(Release(admission.request_key, parent.op_id),),
        )
    )

    observe["enabled"] = True
    torch.cuda.synchronize()
    parent_report = finalize_completion_report(parent_report)
    successor_report = finalize_completion_report(successor_report)

    first = _next_token(4)
    assert parent_report.completions[0].committed_tokens == (first,)
    assert successor_report.completions[0].committed_tokens == (_next_token(first),)


def test_stochastic_device_continuation_matches_depth_one_serial_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = execution_worker(device="cuda:0", pipeline_depth=2)
    sampling = SamplingParams(temperature=0.8, top_k=32, top_p=0.93, seed=917)
    pipelined = und_admission(41, block_ids=(2,), sampling=sampling)
    parent, parent_input = token_operation(
        pipelined.request_key,
        op_id=1,
        parent=root_parent(pipelined),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        rng=Rng(seed=917, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    observe = {"enabled": False}
    ready = CompletionLease.ready

    def gated_ready(self: CompletionLease) -> bool:
        return observe["enabled"] and ready(self)

    monkeypatch.setattr(CompletionLease, "ready", gated_ready)
    parent_report = worker.execute(
        Batch(
            step_id=1,
            admissions=(pipelined,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    assert not completion_report_ready(parent_report)
    successor, successor_input = token_operation(
        pipelined.request_key,
        op_id=2,
        parent=VersionRef(
            pipelined.request_key,
            parent.op_id,
            DevicePoint(parent.outputs[0], parent.plan_digest),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(
            output for output in parent.outputs if output.kind is ProductKind.COMPLETION
        ),
        rng=Rng(seed=917, semantic_index_base=3, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    successor_report = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            input_products=(successor_input,),
        )
    )

    observe["enabled"] = True
    torch.cuda.synchronize()
    parent_tokens = finalize_completion_report(parent_report).completions[0].committed_tokens
    successor_tokens = finalize_completion_report(successor_report).completions[0].committed_tokens

    serial_worker = execution_worker(device="cuda:0", pipeline_depth=1)
    serial = und_admission(41, block_ids=(2,), sampling=sampling)
    serial_parent, serial_parent_input = token_operation(
        serial.request_key,
        op_id=11,
        parent=root_parent(serial),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        rng=Rng(seed=917, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    serial_parent_report = serial_worker.execute(
        Batch(
            step_id=11,
            admissions=(serial,),
            operations=(serial_parent,),
            input_products=(serial_parent_input,),
        )
    )
    torch.cuda.synchronize()
    serial_parent_report = finalize_completion_report(serial_parent_report)
    serial_first = serial_parent_report.completions[0].committed_tokens[0]
    commit = commit_resolved(serial_worker.sessions.get(41))
    serial_successor, serial_successor_input = token_operation(
        serial.request_key,
        op_id=12,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(serial_first,),
        rng=Rng(seed=917, semantic_index_base=3, draw_layout=DrawLayout.TARGET_SAMPLING),
        control_seq=commit.control_seq,
    )
    serial_successor_report = serial_worker.execute(
        Batch(
            step_id=12,
            admissions=(),
            operations=(serial_successor,),
            controls=(commit,),
            input_products=(serial_successor_input,),
        )
    )
    torch.cuda.synchronize()
    serial_successor_report = finalize_completion_report(serial_successor_report)

    assert parent_tokens == serial_parent_report.completions[0].committed_tokens
    assert successor_tokens == serial_successor_report.completions[0].committed_tokens
