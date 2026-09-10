from __future__ import annotations

import time
from dataclasses import replace

from tests.python.fixtures.depth_one import (
    ar_params,
    commit_for_completion,
    diffusion_prepare_operation,
    execution_run,
    kv_publication_operation,
    root_parent,
    token_operation,
    umm_params,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    Checkpoint,
    CloseReason,
    Commit,
    DeviceSelected,
    Disposition,
    DType,
    Finish,
    FixedCheckpoint,
    Free,
    ImageParams,
    NewRequest,
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
    encode_sampling_state_bytes,
)
from uniserve_worker.execution.output import finalize_run_result
from uniserve_worker.models.stub import _next_token


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
        output_index=3,
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
            kind=selected.kind,
            bounds=selected.bounds,
            inputs=(*selected.inputs, state),
            outputs=(*selected.outputs, transition),
            predicate=selected.predicate,
            rng=selected.rng,
            control_seq=selected.control_seq,
        ),
        ProductPayload(product=state, payload=payload),
    )


def _release_relay_outputs(worker, *operations: Operation) -> None:
    finalize_run_result(
        worker.execute(
            execution_run(
                run_id=max(operation.op_id for operation in operations) + 1,
                commands=tuple(
                    Free(output.buffer_id)
                    for operation in operations
                    for output in operation.outputs
                    if output.storage_class is StorageClass.REQUEST_RELAY
                ),
            )
        )
    )


def test_feedback_operation_publishes_distinct_completion_relay_outputs() -> None:
    worker = execution_worker(device="cpu", pipeline_depth=2)
    admission = ar_params(50, block_ids=(0,))
    base, token_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    with_transition, sampling_input = _with_transition_predicate(base, _next_token(4))
    token = replace(
        next(output for output in with_transition.outputs if output.kind is ProductKind.TOKEN),
        output_index=1,
    )
    transition = replace(
        next(output for output in with_transition.outputs if output.kind is ProductKind.COMPLETION),
        storage_class=StorageClass.REQUEST_RELAY,
    )
    completion = ProductRef(
        request_key=base.request_key,
        producer_op_id=base.op_id,
        output_index=0,
        generation=base.op_id * 8 + 6,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.REQUEST_RELAY,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    operation = Operation.registered(
        request_key=base.request_key,
        op_id=base.op_id,
        parent=base.parent,
        kind=base.kind,
        bounds=base.bounds,
        inputs=with_transition.inputs,
        outputs=(completion, token, transition),
        predicate=base.predicate,
        rng=base.rng,
        control_seq=base.control_seq,
    )

    report = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(token_input, sampling_input),
            )
        )
    )

    assert report.completions[0].status is OpStatus.OK
    assert set(report.completions[0].product_generations) == {
        completion.generation,
        token.generation,
        transition.generation,
    }
    _release_relay_outputs(worker, operation)


def test_false_device_predicate_preserves_parent_cutoff_across_registered_descendants() -> None:
    worker = execution_worker(device="cpu", pipeline_depth=2)
    base = ar_params(51, block_ids=(0,))
    admission = NewRequest.create(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        ar=replace(base.ar, finish_token_ids=(_next_token(4),)),
    )
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    parent_report = worker.execute(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    continuation = next(output for output in parent.outputs if output.kind is ProductKind.TOKEN)
    successor, successor_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=Checkpoint(
            parent.op_id,
            DeviceSelected(),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=continuation,
    )
    successor_report = worker.execute(
        execution_run(
            run_id=2,
            admissions=(),
            operations=(successor,),
            input_products=(successor_input,),
        )
    )
    successor_report = finalize_run_result(successor_report)
    successor_continuation = next(
        output for output in successor.outputs if output.kind is ProductKind.TOKEN
    )
    descendant, descendant_input = token_operation(
        admission.request_key,
        op_id=3,
        parent=Checkpoint(
            successor.op_id,
            DeviceSelected(),
        ),
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=successor_continuation,
    )
    descendant_report = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=3,
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
    parent_completion = finalize_run_result(parent_report).completions[0]
    assert completion.logical_lengths == parent_completion.logical_lengths
    assert descendant_completion.logical_lengths == completion.logical_lengths

    selected = Checkpoint(
        parent.op_id,
        FixedCheckpoint(parent_completion.selected_point),
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
        execution_run(
            run_id=4,
            admissions=(),
            operations=(),
            commands=(commit,),
        )
    )
    _release_relay_outputs(worker, parent, successor, descendant)
    later, later_input = token_operation(
        admission.request_key,
        op_id=4,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(7,),
        control_seq=commit.control_seq,
    )
    later_completion = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=5,
                admissions=(),
                operations=(later,),
                input_products=(later_input,),
            )
        )
    ).completions[0]
    later_selected = Checkpoint(
        later.op_id,
        FixedCheckpoint(later_completion.selected_point),
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
        execution_run(
            run_id=6,
            admissions=(),
            operations=(),
            commands=(later_commit,),
        )
    )
    close_report = worker.execute(
        execution_run(
            run_id=7,
            admissions=(),
            operations=(),
            commands=(
                Finish(
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
    generation = umm_params(52, ImageParams(steps=2, height=16, width=16, seed=29))
    understanding = ar_params(52, block_ids=(0,))
    admission = NewRequest.create(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        ar=understanding.ar,
        umm=generation.umm,
    )
    initial, initial_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    initial_report = worker.execute(
        execution_run(
            run_id=1,
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
        execution_run(
            run_id=2,
            operations=(publication,),
            commands=(initial_commit,),
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
        execution_run(
            run_id=3,
            operations=(parent,),
            input_products=(parent_input, sampling_input),
        )
    )
    transition_predicate = next(
        output for output in parent.outputs if output.kind is ProductKind.COMPLETION
    )
    candidate, _latent = diffusion_prepare_operation(
        admission.request_key,
        op_id=4,
        parent=Checkpoint(
            parent.op_id,
            DeviceSelected(),
        ),
        conditioning=conditioning,
        control_seq=initial_commit.control_seq,
    )
    candidate = Operation.registered(
        request_key=candidate.request_key,
        op_id=candidate.op_id,
        parent=candidate.parent,
        kind=candidate.kind,
        bounds=candidate.bounds,
        inputs=candidate.inputs,
        outputs=candidate.outputs,
        predicate=transition_predicate,
        rng=candidate.rng,
        control_seq=candidate.control_seq,
    )
    candidate_batch = execution_run(run_id=4, operations=(candidate,))
    prepared = worker.prepare_execute(candidate_batch)
    assert prepared is not None
    deadline = time.monotonic() + 1.0
    while not prepared.ready() and time.monotonic() < deadline:
        time.sleep(0.0001)
    assert prepared.ready()
    candidate_report = finalize_run_result(worker.execute_prepared(prepared))
    parent_completion = finalize_run_result(parent_report).completions[0]
    candidate_completion = candidate_report.completions[0]
    assert candidate_completion.status is OpStatus.PREDICATED
    assert candidate_completion.logical_lengths == parent_completion.logical_lengths
    assert candidate_completion.product_generations == ()

    parent_commit = commit_for_completion(parent, parent_report)
    _release_relay_outputs(worker, initial, parent, candidate)
    selected, _selected_latent = diffusion_prepare_operation(
        admission.request_key,
        op_id=5,
        parent=parent_commit.selected,
        conditioning=conditioning,
        control_seq=parent_commit.control_seq,
    )
    selected_report = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=5,
                operations=(selected,),
                commands=(parent_commit,),
            )
        )
    )
    assert selected_report.completions[0].status is OpStatus.OK
    assert selected_report.completions[0].product_generations
