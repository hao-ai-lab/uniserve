from __future__ import annotations

from dataclasses import replace

from tests.python.fixtures.depth_one import (
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Close,
    CloseReason,
    Commit,
    DevicePoint,
    Disposition,
    FixedPoint,
    OpStatus,
    ProductKind,
    TokenMode,
    VersionRef,
)
from uniserve_worker.execution.executor import finalize_completion_report
from uniserve_worker.server.stub import _next_token


def test_false_device_predicate_selects_the_parent_cutoff() -> None:
    worker = execution_worker(device="cpu", pipeline_depth=2)
    base = und_admission(51, block_ids=(0,))
    admission = Admission.create(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        und=replace(base.und, finish_token_ids=(_next_token(4),)),
    )
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
    continuation = next(output for output in parent.outputs if output.kind is ProductKind.TOKEN)
    successor, successor_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=VersionRef(
            admission.request_key,
            parent.op_id,
            DevicePoint(1, None, parent.plan_digest),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=continuation,
    )
    successor_report = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(successor,),
            input_products=(successor_input,),
        )
    )
    successor_continuation = next(
        output for output in successor.outputs if output.kind is ProductKind.TOKEN
    )
    descendant, descendant_input = token_operation(
        admission.request_key,
        op_id=3,
        parent=VersionRef(
            admission.request_key,
            successor.op_id,
            DevicePoint(1, None, successor.plan_digest),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=successor_continuation,
    )
    descendant_report = finalize_completion_report(
        worker.execute(
            execution_batch(
                step_id=3,
                admissions=(),
                operations=(descendant,),
                input_products=(descendant_input,),
            )
        )
    )
    successor_report = finalize_completion_report(successor_report)

    completion = successor_report.completions[0]
    completion.validate()
    assert completion.status is OpStatus.PREDICATED
    assert completion.selected_point == 1
    descendant_completion = descendant_report.completions[0]
    descendant_completion.validate()
    assert descendant_completion.status is OpStatus.PREDICATED
    assert descendant_completion.selected_point == 1
    parent_completion = finalize_completion_report(parent_report).completions[0]
    resolved_parent = VersionRef(
        admission.request_key,
        parent.op_id,
        FixedPoint(parent_completion.selected_point, parent_completion.semantic_digest),
    )
    session = worker.sessions.get(51)
    parent_runtime = session.runtime_for(resolved_parent)
    assert parent_runtime is not None
    assert session.resolved_version() == resolved_parent
    assert completion.logical_lengths.token_len == parent_runtime.logical_position
    assert completion.logical_lengths.kv_visible_len == parent_runtime.kv_visible_len
    assert descendant_completion.logical_lengths == completion.logical_lengths
    assert session.logical_position == parent_runtime.logical_position
    assert session.rng_counter == parent_runtime.rng_counter
    assert worker.kv.get(51).length == parent_runtime.kv_visible_len

    selected = VersionRef(
        admission.request_key,
        parent.op_id,
        FixedPoint(parent_completion.selected_point, parent_completion.semantic_digest),
    )
    commit = Commit(
        request_key=admission.request_key,
        control_seq=1,
        expected_parent=root_parent(admission),
        selected=selected,
        public_event_limit=2,
        disposition=Disposition.PUBLISH,
    )
    worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(),
            controls=(commit,),
        )
    )
    later, later_input = token_operation(
        admission.request_key,
        op_id=4,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(7,),
        control_seq=commit.control_seq,
    )
    later_completion = finalize_completion_report(
        worker.execute(
            execution_batch(
                step_id=5,
                admissions=(),
                operations=(later,),
                input_products=(later_input,),
            )
        )
    ).completions[0]
    later_selected = VersionRef(
        admission.request_key,
        later.op_id,
        FixedPoint(later_completion.selected_point, later_completion.semantic_digest),
    )
    later_commit = Commit(
        request_key=admission.request_key,
        control_seq=commit.control_seq + 1,
        expected_parent=commit.selected,
        selected=later_selected,
        public_event_limit=3,
        disposition=Disposition.PUBLISH,
    )
    worker.execute(
        execution_batch(
            step_id=6,
            admissions=(),
            operations=(),
            controls=(later_commit,),
        )
    )
    worker.execute(
        execution_batch(
            step_id=7,
            admissions=(),
            operations=(),
            controls=(
                Close(
                    request_key=admission.request_key,
                    control_seq=later_commit.control_seq + 1,
                    cutoff=commit.selected,
                    reason=CloseReason.COMPLETED,
                ),
            ),
        )
    )
    session = worker.sessions.get(51)
    assert session.committed_version() == commit.selected
    assert session.resolved_version() == commit.selected
    assert session.logical_position == parent_runtime.logical_position
    assert session.rng_counter == parent_runtime.rng_counter
    assert worker.kv.get(51).length == parent_runtime.kv_length
