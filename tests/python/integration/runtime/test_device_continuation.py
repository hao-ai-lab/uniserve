from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.depth_one import (
    commit_for_completion,
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    Commit,
    DevicePoint,
    Disposition,
    DrawLayout,
    FixedPoint,
    Operation,
    ProductKind,
    Release,
    Rng,
    SamplingParams,
    TokenMode,
    VersionRef,
)
from uniserve_worker.execution.output import finalize_completion_report
from uniserve_worker.models.stub import _next_token

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def test_same_request_continues_before_parent_report_materialization() -> None:
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
        execution_batch(
            step_id=0,
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

    parent_report = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    device_parent = VersionRef(
        admission.request_key,
        parent.op_id,
        DevicePoint(1, None),
    )
    successor_template, _ = token_operation(
        admission.request_key,
        op_id=2,
        parent=device_parent,
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
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
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            controls=(Release(admission.request_key, parent.op_id),),
        )
    )

    torch.cuda.synchronize()
    parent_report = finalize_completion_report(parent_report)
    successor_report = finalize_completion_report(successor_report)

    first = _next_token(4)
    assert parent_report.completions[0].committed_tokens == (first,)
    assert successor_report.completions[0].committed_tokens == (_next_token(first),)
    parent_record = parent_report.completions[0]
    parent_selected = VersionRef(
        admission.request_key,
        parent.op_id,
        FixedPoint(parent_record.selected_point),
    )
    worker.execute(
        execution_batch(
            step_id=3,
            admissions=(),
            operations=(),
            controls=(
                Commit(
                    request_key=admission.request_key,
                    control_seq=1,
                    expected_parent=root_parent(admission),
                    selected=parent_selected,
                    public_event_limit=1,
                    disposition=Disposition.PUBLISH,
                ),
            ),
        )
    )


def test_device_continuation_chain_matches_serial_token_sequence() -> None:
    worker = execution_worker(device="cuda:0", pipeline_depth=4)
    admission = und_admission(32, block_ids=(0,))
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    reports = [
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input,),
            )
        )
    ]
    operations = [operation]

    for step_id in range(2, 5):
        parent = operations[-1]
        operation, token_input = token_operation(
            admission.request_key,
            op_id=step_id,
            parent=VersionRef(
                admission.request_key,
                parent.op_id,
                DevicePoint(1, None),
            ),
            mode=TokenMode.DECODE,
            tokens=(0,),
            predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
        )
        reports.append(
            worker.execute(
                execution_batch(
                    step_id=step_id,
                    admissions=(),
                    operations=(operation,),
                    input_products=(token_input,),
                )
            )
        )
        operations.append(operation)

    torch.cuda.synchronize()
    tokens = tuple(
        finalize_completion_report(report).completions[0].committed_tokens[0] for report in reports
    )
    expected = []
    current = 4
    for _ in range(4):
        current = _next_token(current)
        expected.append(current)
    assert tokens == tuple(expected)


def test_relay_window_retains_a_consumer_fenced_predecessor() -> None:
    worker = execution_worker(device="cuda:0", pipeline_depth=3)
    admission = und_admission(33, block_ids=(0,))
    operation, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    reports = [
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input,),
            )
        )
    ]
    operations = [operation]
    for step_id in range(2, 5):
        parent = operations[-1]
        operation, token_input = token_operation(
            admission.request_key,
            op_id=step_id,
            parent=VersionRef(
                admission.request_key,
                parent.op_id,
                DevicePoint(1, None),
            ),
            mode=TokenMode.DECODE,
            tokens=(0,),
            predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
        )
        reports.append(
            worker.execute(
                execution_batch(
                    step_id=step_id,
                    operations=(operation,),
                    controls=(Release(admission.request_key, parent.op_id),),
                    input_products=(token_input,),
                )
            )
        )
        operations.append(operation)

    torch.cuda.synchronize()
    tokens = tuple(
        finalize_completion_report(report).completions[0].committed_tokens[0]
        for report in reports
    )
    expected = []
    current = 4
    for _ in range(4):
        current = _next_token(current)
        expected.append(current)
    assert tokens == tuple(expected)


def test_stochastic_device_continuation_matches_depth_one_serial_execution() -> None:
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
    parent_report = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(pipelined,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    successor, successor_input = token_operation(
        pipelined.request_key,
        op_id=2,
        parent=VersionRef(
            pipelined.request_key,
            parent.op_id,
            DevicePoint(1, None),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
        rng=Rng(seed=917, semantic_index_base=3, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    successor_report = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            input_products=(successor_input,),
        )
    )

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
        execution_batch(
            step_id=11,
            admissions=(serial,),
            operations=(serial_parent,),
            input_products=(serial_parent_input,),
        )
    )
    torch.cuda.synchronize()
    serial_parent_report = finalize_completion_report(serial_parent_report)
    serial_first = serial_parent_report.completions[0].committed_tokens[0]
    commit = commit_for_completion(serial_parent, serial_parent_report)
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
        execution_batch(
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


def test_penalty_device_continuation_matches_depth_one_serial_execution() -> None:
    # A penalty-bearing pipelined successor must match the equivalent serial lineage.
    sampling = SamplingParams(
        temperature=0.8,
        top_k=48,
        seed=613,
        repetition_penalty=1.4,
        frequency_penalty=0.6,
        presence_penalty=0.3,
    )
    worker = execution_worker(device="cuda:0", pipeline_depth=2)
    pipelined = und_admission(57, block_ids=(3,), sampling=sampling)
    parent, parent_input = token_operation(
        pipelined.request_key,
        op_id=1,
        parent=root_parent(pipelined),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        rng=Rng(seed=613, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    parent_report = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(pipelined,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    successor, successor_input = token_operation(
        pipelined.request_key,
        op_id=2,
        parent=VersionRef(
            pipelined.request_key,
            parent.op_id,
            DevicePoint(1, None),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
        rng=Rng(seed=613, semantic_index_base=3, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    successor_report = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            input_products=(successor_input,),
        )
    )

    torch.cuda.synchronize()
    parent_tokens = finalize_completion_report(parent_report).completions[0].committed_tokens
    successor_tokens = finalize_completion_report(successor_report).completions[0].committed_tokens

    serial_worker = execution_worker(device="cuda:0", pipeline_depth=1)
    serial = und_admission(57, block_ids=(3,), sampling=sampling)
    serial_parent, serial_parent_input = token_operation(
        serial.request_key,
        op_id=11,
        parent=root_parent(serial),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
        rng=Rng(seed=613, semantic_index_base=2, draw_layout=DrawLayout.TARGET_SAMPLING),
    )
    serial_parent_report = serial_worker.execute(
        execution_batch(
            step_id=11,
            admissions=(serial,),
            operations=(serial_parent,),
            input_products=(serial_parent_input,),
        )
    )
    torch.cuda.synchronize()
    serial_parent_report = finalize_completion_report(serial_parent_report)
    serial_first = serial_parent_report.completions[0].committed_tokens[0]
    commit = commit_for_completion(serial_parent, serial_parent_report)
    serial_successor, serial_successor_input = token_operation(
        serial.request_key,
        op_id=12,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(serial_first,),
        rng=Rng(seed=613, semantic_index_base=3, draw_layout=DrawLayout.TARGET_SAMPLING),
        control_seq=commit.control_seq,
    )
    serial_successor_report = serial_worker.execute(
        execution_batch(
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
