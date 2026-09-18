from __future__ import annotations

import time
from dataclasses import replace

from tests.python.fixtures.depth_one import (
    ar_params,
    diffusion_prepare_operation,
    execution_batch,
    finalized_report,
    kv_publication_operation,
    record_completion,
    root_parent,
    token_operation,
    umm_params,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from uniserve_worker.protocol.batch import Finish, Free, NewRequest
from uniserve_worker.protocol.identity import ComputationId
from uniserve_worker.protocol.operation import (
    ForwardMode,
    ImageParams,
    OpStatus,
    SamplingState,
    ScheduledRequest,
)
from uniserve_worker.protocol.tensor import DType, ShapeBound, TensorRef


def _with_transition_predicate(
    operation: ScheduledRequest,
    token_id: int,
) -> ScheduledRequest:
    selected = operation
    transition = TensorRef(
        request_key=selected.request_key,
        producer_op_id=selected.op_id,
        output_index=3,
        generation=selected.op_id.batch_id * 8 + 8,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    return replace(
        selected,
        transition_output=transition,
        sampling_state=SamplingState(transition_token_ids=(token_id,)),
    )


def _release_relay_outputs(worker, *operations: ScheduledRequest) -> None:
    finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=max(
                    operation.op_id.batch_id for operation in operations
                )
                + 1,
                commands=tuple(
                    Free(output.buffer_id)
                    for operation in operations
                    for output in (
                        operation.token_output,
                        operation.completion_output,
                        operation.transition_output,
                    )
                    if output is not None
                ),
            )
        ),
    )


def test_feedback_operation_publishes_distinct_completion_relay_outputs() -> (
    None
):
    worker = execution_worker(device="cpu", queue_depth=2)
    admission = ar_params(50, block_ids=(0,))
    base = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    with_transition = _with_transition_predicate(base, expected_successor(4))
    token = replace(
        with_transition.token_output,
        output_index=1,
    )
    transition = with_transition.transition_output
    completion = TensorRef(
        request_key=base.request_key,
        producer_op_id=base.op_id,
        output_index=0,
        generation=base.op_id.batch_id * 8 + 6,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    operation = replace(
        with_transition,
        completion_output=completion,
        token_output=token,
        transition_output=transition,
    )

    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                operations=(operation,),
            )
        ),
    )

    assert report.completions[0].status is OpStatus.OK
    assert set(report.completions[0].product_generations) == {
        completion.generation,
        token.generation,
        transition.generation,
    }
    _release_relay_outputs(worker, operation)


def test_false_device_predicate_preserves_parent_cutoff_across_registered_descendants(  # noqa: E501
) -> None:
    worker = execution_worker(device="cpu", queue_depth=2)
    base = ar_params(51, block_ids=(0,))
    admission = NewRequest(
        base.request_key,
        request_pool_idx=base.request_pool_idx,
        generation=replace(
            base.generation, finish_token_ids=(expected_successor(4),)
        ),
    )
    predecessor = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    parent_report = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            operations=(predecessor,),
        )
    )
    continuation = predecessor.token_output
    successor = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=predecessor.op_id,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=continuation,
    )
    successor_report = worker.submit(
        execution_batch(
            batch_id=2,
            admissions=(),
            operations=(successor,),
        )
    )
    successor_report = finalized_report(worker, successor_report)
    successor_continuation = successor.token_output
    descendant = token_operation(
        admission.request_key,
        op_id=ComputationId(3, 0),
        predecessor=successor.op_id,
        mode=ForwardMode.DECODE,
        tokens=(0,),
        predicate=successor_continuation,
    )
    descendant_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(),
                operations=(descendant,),
            )
        ),
    )
    completion = successor_report.completions[0]
    completion.validate()
    assert completion.status is OpStatus.PREDICATED
    descendant_completion = descendant_report.completions[0]
    descendant_completion.validate()
    assert descendant_completion.status is OpStatus.PREDICATED
    parent_report = finalized_report(worker, parent_report)
    parent_completion = parent_report.completions[0]
    assert completion.position == parent_completion.position
    assert completion.kv_visible_len == parent_completion.kv_visible_len
    assert completion.kv_computed_len == parent_completion.kv_computed_len
    assert (
        completion.num_completed_steps == parent_completion.num_completed_steps
    )
    assert descendant_completion.position == completion.position
    assert descendant_completion.kv_visible_len == completion.kv_visible_len
    assert descendant_completion.kv_computed_len == completion.kv_computed_len
    assert (
        descendant_completion.num_completed_steps
        == completion.num_completed_steps
    )

    selected = predecessor.op_id

    _release_relay_outputs(worker, predecessor, successor, descendant)
    later = token_operation(
        admission.request_key,
        op_id=ComputationId(4, 0),
        predecessor=selected,
        mode=ForwardMode.DECODE,
        tokens=(7,),
    )
    later_completion = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=5,
                admissions=(),
                operations=(later,),
            )
        ),
    ).completions[0]
    assert later_completion.status is OpStatus.OK
    assert later_completion.position == parent_completion.position + 1

    close_report = worker.submit(
        execution_batch(
            batch_id=7,
            admissions=(),
            operations=(),
            commands=(
                Finish(
                    request_key=admission.request_key,
                ),
            ),
        )
    )
    close_report = finalized_report(worker, close_report)
    assert close_report.completions == ()


def test_false_generation_predicate_preserves_the_selected_text_state_and_latent_capacity(  # noqa: E501
) -> None:
    worker = execution_worker(device="cpu", queue_depth=2)
    generation = umm_params(
        52, ImageParams(steps=2, height=16, width=16, seed=29)
    )
    understanding = ar_params(52, block_ids=(0,))
    admission = NewRequest(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        generation=understanding.generation,
        image=generation.image,
    )
    initial = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    initial_report = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            operations=(initial,),
        )
    )
    initial_report = finalized_report(worker, initial_report)
    initial_observation = record_completion(initial, initial_report)
    publication, conditioning = kv_publication_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=initial_observation.op_id,
    )
    worker.submit(
        execution_batch(
            batch_id=2,
            operations=(publication,),
            commands=(),
        )
    )
    predecessor = token_operation(
        admission.request_key,
        op_id=ComputationId(3, 0),
        predecessor=initial_observation.op_id,
        mode=ForwardMode.DECODE,
        tokens=(expected_successor(4),),
    )
    predecessor = _with_transition_predicate(predecessor, 4_242)
    parent_report = worker.submit(
        execution_batch(
            batch_id=3,
            operations=(predecessor,),
        )
    )
    transition_predicate = predecessor.transition_output
    candidate, _latent = diffusion_prepare_operation(
        admission.request_key,
        op_id=ComputationId(4, 0),
        predecessor=predecessor.op_id,
        conditioning=conditioning,
    )
    candidate = replace(candidate, predicate=transition_predicate)

    candidate_batch = execution_batch(batch_id=4, operations=(candidate,))
    prepared = worker.submit(candidate_batch)
    assert prepared is not None
    deadline = time.monotonic() + 1.0
    while not prepared.inputs_ready() and time.monotonic() < deadline:
        worker.advance_inputs(prepared)
        time.sleep(0.0001)
    assert prepared.inputs_ready()
    prepared = finalized_report(worker, prepared)
    candidate_report = prepared
    parent_report = finalized_report(worker, parent_report)
    parent_completion = parent_report.completions[0]
    candidate_completion = candidate_report.completions[0]
    assert candidate_completion.status is OpStatus.PREDICATED
    assert candidate_completion.position == parent_completion.position
    assert (
        candidate_completion.kv_visible_len == parent_completion.kv_visible_len
    )
    assert (
        candidate_completion.kv_computed_len
        == parent_completion.kv_computed_len
    )
    assert (
        candidate_completion.num_completed_steps
        == parent_completion.num_completed_steps
    )
    assert candidate_completion.product_generations == ()

    parent_observation = record_completion(predecessor, parent_report)
    _release_relay_outputs(worker, initial, predecessor, candidate)
    selected, _selected_latent = diffusion_prepare_operation(
        admission.request_key,
        op_id=ComputationId(5, 0),
        predecessor=parent_observation.op_id,
        conditioning=conditioning,
    )
    selected_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=5,
                operations=(selected,),
                commands=(),
            )
        ),
    )
    assert selected_report.completions[0].status is OpStatus.OK
    assert selected_report.completions[0].product_generations
