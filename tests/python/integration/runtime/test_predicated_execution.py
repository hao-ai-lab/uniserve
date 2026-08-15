from __future__ import annotations

import time
from dataclasses import replace

from tests.python.fixtures.depth_one import (
    commit_for_completion,
    execution_batch,
    gen_admission,
    gen_transition_operation,
    kv_publication_operation,
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
    DType,
    FixedPoint,
    ImageParams,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TokenMode,
    VersionRef,
    encode_sampling_state_bytes,
)
from uniserve_worker.server.completion import finalize_completion_report
from uniserve_worker.server.stub import _next_token


def _with_transition_predicate(
    operation: Operation,
    token_id: int,
) -> tuple[Operation, ProductPayload]:
    selected = operation
    payload = encode_sampling_state_bytes(SamplingState(transition_token_ids=(token_id,)))
    state = ProductRef(
        request_key=selected.request_key,
        producer_op_id=selected.op_id,
        output_index=(1 << 16) - 2,
        generation=selected.op_id * 8 + 7,
        kind=ProductKind.SAMPLING_STATE,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U8,
        shape_bound=ShapeBound((StaticDim(len(payload)),)),
        point_range=PointRange(),
    )
    transition = ProductRef(
        request_key=selected.request_key,
        producer_op_id=selected.op_id,
        output_index=6,
        generation=selected.op_id * 8 + 8,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    return (
        Operation.registered(
            request_key=selected.request_key,
            op_id=selected.op_id,
            parent=selected.parent,
            work=selected.work,
            route=selected.route,
            domain=selected.domain,
            bounds=selected.bounds,
            inputs=(*selected.inputs, state),
            outputs=(*selected.outputs, transition),
            kv_capacity_pages=selected.kv_capacity_pages,
            predicate=selected.predicate,
            rng=selected.rng,
            control_seq=selected.control_seq,
        ),
        ProductPayload(product=state, payload=payload),
    )


def test_false_device_predicate_preserves_parent_cutoff_across_registered_descendants() -> None:
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
    successor_report = finalize_completion_report(successor_report)
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
    completion = successor_report.completions[0]
    completion.validate()
    assert completion.status is OpStatus.PREDICATED
    assert completion.selected_point == 1
    descendant_completion = descendant_report.completions[0]
    descendant_completion.validate()
    assert descendant_completion.status is OpStatus.PREDICATED
    assert descendant_completion.selected_point == 1
    parent_completion = finalize_completion_report(parent_report).completions[0]
    assert completion.logical_lengths == parent_completion.logical_lengths
    assert descendant_completion.logical_lengths == completion.logical_lengths

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
    close_report = worker.execute(
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
    assert close_report.completions == ()


def test_false_generation_predicate_preserves_the_selected_text_state_and_latent_capacity() -> None:
    worker = execution_worker(device="cpu", pipeline_depth=2)
    generation = gen_admission(52, ImageParams(steps=2, height=16, width=16, seed=29))
    understanding = und_admission(52, block_ids=(0,))
    admission = Admission.create(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        und=understanding.und,
        gen_admission=generation.gen_admission,
    )
    initial, initial_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    initial_report = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(initial,),
            input_products=(initial_input,),
        )
    )
    initial_commit = commit_for_completion(initial, initial_report)
    publication, conditioning = kv_publication_operation(
        admission.request_key,
        op_id=2,
        parent=initial_commit.selected,
        control_seq=initial_commit.control_seq,
    )
    worker.execute(
        execution_batch(
            step_id=2,
            operations=(publication,),
            controls=(initial_commit,),
        )
    )
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=3,
        parent=initial_commit.selected,
        mode=TokenMode.DECODE,
        tokens=(_next_token(4),),
        control_seq=initial_commit.control_seq,
    )
    parent, sampling_input = _with_transition_predicate(parent, 4_242)
    parent_report = worker.execute(
        execution_batch(
            step_id=3,
            operations=(parent,),
            input_products=(parent_input, sampling_input),
        )
    )
    transition_predicate = next(
        output for output in parent.outputs if output.kind is ProductKind.COMPLETION
    )
    candidate, _latent = gen_transition_operation(
        admission.request_key,
        op_id=4,
        parent=VersionRef(
            admission.request_key,
            parent.op_id,
            DevicePoint(1, None, parent.plan_digest),
        ),
        conditioning=conditioning,
        control_seq=initial_commit.control_seq,
    )
    candidate = Operation.registered(
        request_key=candidate.request_key,
        op_id=candidate.op_id,
        parent=candidate.parent,
        work=candidate.work,
        route=candidate.route,
        domain=candidate.domain,
        bounds=candidate.bounds,
        inputs=candidate.inputs,
        outputs=candidate.outputs,
        kv_capacity_pages=candidate.kv_capacity_pages,
        predicate=transition_predicate,
        rng=candidate.rng,
        control_seq=candidate.control_seq,
    )
    candidate_batch = execution_batch(step_id=4, operations=(candidate,))
    prepared = worker.prepare_execute(candidate_batch)
    assert prepared is not None
    deadline = time.monotonic() + 1.0
    while not prepared.ready() and time.monotonic() < deadline:
        time.sleep(0.0001)
    assert prepared.ready()
    candidate_report = finalize_completion_report(worker.execute_prepared(prepared))
    parent_completion = finalize_completion_report(parent_report).completions[0]
    candidate_completion = candidate_report.completions[0]
    assert candidate_completion.status is OpStatus.PREDICATED
    assert candidate_completion.semantic_digest == parent_completion.semantic_digest
    assert candidate_completion.logical_lengths == parent_completion.logical_lengths
    assert candidate_completion.product_generations == ()

    parent_commit = commit_for_completion(parent, parent_report)
    selected, _selected_latent = gen_transition_operation(
        admission.request_key,
        op_id=5,
        parent=parent_commit.selected,
        conditioning=conditioning,
        control_seq=parent_commit.control_seq,
    )
    selected_report = finalize_completion_report(
        worker.execute(
            execution_batch(
                step_id=5,
                operations=(selected,),
                controls=(parent_commit,),
            )
        )
    )
    assert selected_report.completions[0].status is OpStatus.OK
    assert selected_report.completions[0].product_generations
