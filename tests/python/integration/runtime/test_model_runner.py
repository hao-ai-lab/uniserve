"""Serial-oracle behavior at the canonical ModelRunner boundary."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import replace

import pytest
import torch
from PIL import Image

from tests.python.fixtures.depth_one import (
    ar_params,
    bind_request_placement,
    commit_for_completion,
    diffusion_finalize_operation,
    diffusion_prepare_operation,
    diffusion_step_operation,
    encode_operation,
    execution_run,
    finalized_report,
    kv_publication_operation,
    root_parent,
    token_operation,
    umm_params,
    visual_state_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    ArRequestParams,
    BlockTable,
    Bounds,
    Checkpoint,
    Commit,
    DeviceDim,
    DeviceSelected,
    DType,
    ErrorCode,
    Free,
    ImageParams,
    NewRequest,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RunKind,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TokenMode,
    TransferHandle,
    UmmRequestParams,
    encode_sampling_state_bytes,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
)
from uniserve_worker.execution.output import finalize_run_result, run_result_ready
from uniserve_worker.foundation.errors import WorkerError, WorkerErrorCode
from uniserve_worker.models.stub import StubModel, _next_token
from uniserve_worker.transfer.tickets import decode_transfer_handle

pytestmark = pytest.mark.integration

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _MisalignedOutputModel(StubModel):
    def __init__(self) -> None:
        super().__init__()
        self.misaligned = False

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        output = super().project(hidden, batch)
        if self.misaligned:
            return ForwardOutput(output.values[:-1])
        return output


class _SeparatePhaseModel(StubModel):
    def __init__(self) -> None:
        super().__init__()
        self.tensorized_mixed = False


def _publish_conditioning(worker: object, admission: NewRequest, *, op_id: int, run_id: int):
    publication, product = kv_publication_operation(
        admission.request_key,
        op_id=op_id,
        parent=root_parent(admission),
    )
    worker.execute(
        execution_run(run_id=run_id, admissions=(admission,), operations=(publication,))
    )
    return product


def _prepare_media(
    worker: object,
    admission: NewRequest,
    conditioning: object,
    *,
    op_id: int,
    parent: object,
    run_id: int,
    control_seq: int = 0,
    seed: int = 29,
    image_index: int = 1,
):
    preparation, latent = diffusion_prepare_operation(
        admission.request_key,
        op_id=op_id,
        parent=parent,
        conditioning=conditioning,
        control_seq=control_seq,
        seed=seed,
        image_index=image_index,
    )
    report = worker.execute(execution_run(run_id=run_id, operations=(preparation,)))
    assert report.completions[0].status is OpStatus.OK
    return latent, commit_for_completion(preparation, report)


def _prepare_decode(
    worker: object,
    admission: NewRequest,
    *,
    op_id: int,
    run_id: int,
    tokens: tuple[int, ...],
):
    prefill, payload = token_operation(
        admission.request_key,
        op_id=op_id,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=tokens,
    )
    report = worker.execute(
        execution_run(
            run_id=run_id,
            admissions=(admission,),
            operations=(prefill,),
            input_products=(payload,),
        )
    )
    resolved = finalized_report(report)
    commit = commit_for_completion(prefill, resolved)
    decode, decode_input = token_operation(
        admission.request_key,
        op_id=op_id + 1,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(resolved.completions[0].committed_tokens[0],),
        control_seq=commit.control_seq,
    )
    return decode, decode_input, commit


def _finalized_artifact(
    worker: object,
    admission: NewRequest,
    latent: ProductRef,
    commit: Commit,
    *,
    op_id: int,
    run_id: int,
) -> bytes:
    operation = diffusion_finalize_operation(
        admission.request_key,
        op_id=op_id,
        parent=commit.selected,
        latent=latent,
        control_seq=commit.control_seq,
    )
    report = worker.execute(
        execution_run(run_id=run_id, operations=(operation,), commands=(commit,))
    )
    report = finalized_report(report)
    assert report.completions[0].status is OpStatus.OK
    artifacts = tuple(
        product.payload
        for product in report.products
        if product.product.kind is ProductKind.ARTIFACT
    )
    assert len(artifacts) == 1
    return artifacts[0]


def test_extend_then_decode_commit_the_serial_oracle_tokens():
    worker = execution_worker()
    admission = ar_params(1, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = finalized_report(
        worker.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
    )

    assert extended.completions[0].committed_tokens == (_next_token(4),)
    assert extended.completions[0].logical_lengths.kv_visible_len == 2
    assert extended.completions[0].logical_lengths.token_len == 2

    first_token = extended.completions[0].committed_tokens[0]
    commit = commit_for_completion(extend, extended)
    decode, decode_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(first_token,),
        control_seq=commit.control_seq,
    )
    decoded = finalized_report(
        worker.execute(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(decode,),
                commands=(commit,),
                input_products=(decode_input,),
            )
        )
    )

    assert decoded.completions[0].committed_tokens == (_next_token(first_token),)
    assert decoded.completions[0].logical_lengths.kv_visible_len == 3
    assert decoded.completions[0].logical_lengths.token_len == 3


def test_prefix_reuse_continues_from_the_admitted_logical_position():
    worker = execution_worker()
    admission = ar_params(8, block_ids=(0,), prefix_len=2)
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(4,),
    )

    report = finalized_report(
        worker.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
    )

    assert report.completions[0].logical_lengths.kv_visible_len == 3
    assert report.completions[0].logical_lengths.token_len == 3


def test_invalid_physical_placement_reports_error_behind_an_unobserved_parent() -> None:
    worker = execution_worker(pipeline_depth=2)
    admission = ar_params(9, block_ids=(0,))
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    device_parent = Checkpoint(
        parent.op_id,
        DeviceSelected(),
    )
    template, _ = token_operation(
        admission.request_key,
        op_id=2,
        parent=device_parent,
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
    )
    operation = Operation.registered(
        request_key=template.request_key,
        op_id=template.op_id,
        parent=template.parent,
        kind=template.kind,
        bounds=template.bounds,
        outputs=template.outputs,
        predicate=template.predicate,
    )
    invalid_table = BlockTable(
        request_pool_idx=admission.request_pool_idx,
        group_id=0,
        page_ids=(),
        allocated_tokens=0,
    )

    report = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=2,
                operations=(operation,),
                block_tables=(invalid_table,),
            )
        )
    )

    assert report.completions[0].status is OpStatus.ERROR
    assert report.completions[0].error_code is ErrorCode.INVALID_OPERATION


def test_decode_reuses_the_published_request_page_table() -> None:
    worker = execution_worker(pipeline_depth=2)
    admission = ar_params(10, block_ids=(0,))
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(parent,),
            input_products=(parent_input,),
        )
    )
    device_parent = Checkpoint(
        parent.op_id,
        DeviceSelected(),
    )
    template, _ = token_operation(
        admission.request_key,
        op_id=2,
        parent=device_parent,
        mode=TokenMode.DECODE,
        tokens=(0,),
        predicate=next(output for output in parent.outputs if output.kind is ProductKind.TOKEN),
    )
    operation = Operation.registered(
        request_key=template.request_key,
        op_id=template.op_id,
        parent=template.parent,
        kind=template.kind,
        bounds=template.bounds,
        outputs=template.outputs,
        predicate=template.predicate,
    )
    report = finalize_run_result(
        worker.execute(
            execution_run(
                run_id=2,
                operations=(operation,),
            )
        )
    )

    assert report.completions[0].status is OpStatus.OK
    assert report.completions[0].logical_lengths.kv_visible_len == 3


def test_mixed_token_and_flow_match_homogeneous_results():
    mixed = execution_worker()
    sequence_admission = ar_params(1, block_ids=(0,))
    flow_admission = umm_params(2, ImageParams(steps=1, height=16, width=16, seed=29))
    mixed_conditioning = _publish_conditioning(mixed, flow_admission, op_id=10, run_id=1)
    mixed_latent, mixed_preparation_commit = _prepare_media(
        mixed,
        flow_admission,
        mixed_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        run_id=2,
    )
    flow, mixed_output_latent = diffusion_step_operation(
        flow_admission.request_key,
        op_id=12,
        parent=mixed_preparation_commit.selected,
        conditioning=mixed_conditioning,
        latent=mixed_latent,
        steps=1,
        control_seq=mixed_preparation_commit.control_seq,
    )
    sequence, sequence_input, sequence_control = _prepare_decode(
        mixed,
        sequence_admission,
        op_id=10,
        run_id=3,
        tokens=(3, 4),
    )

    mixed_result = finalized_report(mixed.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(sequence, flow),
            commands=(mixed_preparation_commit, sequence_control),
            input_products=(sequence_input,),
        )
    ))

    split = execution_worker()
    split_conditioning = _publish_conditioning(split, flow_admission, op_id=10, run_id=1)
    split_latent, split_preparation_commit = _prepare_media(
        split,
        flow_admission,
        split_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        run_id=2,
    )
    split_flow, split_output_latent = diffusion_step_operation(
        flow_admission.request_key,
        op_id=12,
        parent=split_preparation_commit.selected,
        conditioning=split_conditioning,
        latent=split_latent,
        steps=1,
        control_seq=split_preparation_commit.control_seq,
    )
    split_sequence, split_sequence_input, split_sequence_control = _prepare_decode(
        split,
        sequence_admission,
        op_id=10,
        run_id=3,
        tokens=(3, 4),
    )
    sequence_result = finalized_report(split.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(split_sequence,),
            commands=(split_sequence_control,),
            input_products=(split_sequence_input,),
        )
    ))
    flow_result = finalized_report(split.execute(
        execution_run(
            run_id=5,
            admissions=(),
            operations=(split_flow,),
            commands=(split_preparation_commit,),
            input_products=(),
        )
    ))

    assert (
        mixed_result.completions[0].committed_tokens
        == sequence_result.completions[0].committed_tokens
    )
    assert (
        mixed_result.completions[0].logical_lengths
        == sequence_result.completions[0].logical_lengths
    )
    assert mixed_result.completions[1].logical_lengths == flow_result.completions[0].logical_lengths
    mixed_flow_commit = commit_for_completion(flow, mixed_result)
    split_flow_commit = commit_for_completion(split_flow, flow_result)
    assert _finalized_artifact(
        mixed,
        flow_admission,
        mixed_output_latent,
        mixed_flow_commit,
        op_id=13,
        run_id=5,
    ) == _finalized_artifact(
        split,
        flow_admission,
        split_output_latent,
        split_flow_commit,
        op_id=13,
        run_id=6,
    )


def test_mixed_submission_requires_tensorized_model_support():
    worker = execution_worker(_SeparatePhaseModel())
    token_admission = ar_params(1, block_ids=(0,))
    flow_admission = umm_params(2, ImageParams(steps=1, height=16, width=16, seed=29))
    token, token_input = token_operation(
        token_admission.request_key,
        op_id=11,
        parent=root_parent(token_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    conditioning = ProductRef(
        request_key=flow_admission.request_key,
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
        dtype=DType.U8,
        shape_bound=ShapeBound((DeviceDim(1 << 20),)),
        point_range=PointRange(),
    )
    preparation, _latent = diffusion_prepare_operation(
        flow_admission.request_key,
        op_id=11,
        parent=root_parent(flow_admission),
        conditioning=conditioning,
    )

    with pytest.raises(WorkerError) as rejected:
        worker.execute(
            execution_run(
                run_id=2,
                admissions=(token_admission,),
                operations=(token, preparation),
                input_products=(token_input,),
            )
        )

    assert rejected.value.code is WorkerErrorCode.INVALID_DESCRIPTOR


def test_request_scoped_operation_identity_preserves_homogeneous_decode():
    worker = execution_worker()
    admissions = (ar_params(41, block_ids=(0,)), ar_params(42, block_ids=(1,)))
    prefill_ops = []
    prefill_inputs = []
    last_tokens = []
    for index, admission in enumerate(admissions):
        tokens = (3 + 4 * index, 4 + 4 * index)
        operation, payload = token_operation(
            admission.request_key,
            op_id=50,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=tokens,
        )
        prefill_ops.append(operation)
        prefill_inputs.append(payload)
        last_tokens.append(tokens[-1])
    prefilled = worker.execute(
        execution_run(
            run_id=1,
            admissions=admissions,
            operations=tuple(prefill_ops),
            input_products=tuple(prefill_inputs),
        )
    )

    decode_ops = []
    decode_inputs = []
    commits = []
    for index, admission in enumerate(admissions):
        commit = commit_for_completion(prefill_ops[index], prefilled)
        operation, payload = token_operation(
            admission.request_key,
            op_id=60,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(_next_token(last_tokens[index]),),
            control_seq=commit.control_seq,
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
        commits.append(commit)
    decoded = finalized_report(worker.execute(
        execution_run(
            run_id=2,
            admissions=(),
            operations=tuple(decode_ops),
            commands=tuple(commits),
            input_products=tuple(decode_inputs),
        )
    ))

    assert tuple(record.committed_tokens for record in decoded.completions) == tuple(
        (_next_token(_next_token(token)),) for token in last_tokens
    )


def test_output_validation_failure_discards_all_candidate_state():
    model = _MisalignedOutputModel()
    worker = execution_worker(model)
    admission = ar_params(4, block_ids=(4,))
    initial, initial_input = token_operation(
        admission.request_key,
        op_id=31,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(12, 13),
    )
    initial_report = worker.execute(
        execution_run(
            run_id=11,
            admissions=(admission,),
            operations=(initial,),
            input_products=(initial_input,),
        )
    )
    commit = commit_for_completion(initial, initial_report)
    worker.execute(execution_run(run_id=12, admissions=(), operations=(), commands=(commit,)))
    retry, retry_input = token_operation(
        admission.request_key,
        op_id=32,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(_next_token(13),),
        control_seq=commit.control_seq,
    )
    retry_batch = execution_run(
        run_id=13, admissions=(), operations=(retry,), input_products=(retry_input,)
    )
    model.misaligned = True
    failed = worker.execute(retry_batch)

    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.COMPUTE_ERROR

    model.misaligned = False
    replacement, replacement_input = token_operation(
        admission.request_key,
        op_id=33,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(_next_token(13),),
        control_seq=commit.control_seq,
    )
    result = finalized_report(worker.execute(
        execution_run(
            run_id=14,
            operations=(replacement,),
            input_products=(replacement_input,),
        )
    ))
    assert result.completions[0].committed_tokens == (_next_token(_next_token(13)),)
    assert result.completions[0].logical_lengths.kv_visible_len == 3


def test_failed_flow_preserves_the_next_accepted_trajectory_and_final_artifact():
    worker = execution_worker(block_size=4)
    admission = umm_params(5, ImageParams(steps=2, height=64, width=64, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=40, run_id=12)
    latent, preparation_commit = _prepare_media(
        worker,
        admission,
        conditioning,
        op_id=41,
        parent=root_parent(admission),
        run_id=13,
    )
    flow, _output_latent = diffusion_step_operation(
        admission.request_key,
        op_id=42,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=2,
        control_seq=preparation_commit.control_seq,
    )
    worker.execute(execution_run(run_id=14, commands=(preparation_commit,)))
    valid_batch = execution_run(
        run_id=15,
        admissions=(),
        operations=(flow,),
        input_products=(),
    )
    batch = replace(
        valid_batch,
        lanes=tuple(
            replace(
                lane,
                forward_rows=tuple(
                    replace(
                        row,
                        seq_len=1,
                    )
                    if row.request_pool_index != admission.request_pool_idx
                    else row
                    for row in lane.forward_rows
                ),
            )
            for lane in valid_batch.lanes
        ),
    )

    failed = finalized_report(worker.execute(batch))

    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_OPERATION
    assert failed.completions[0].logical_lengths.latent_len == 0

    replacement, _replacement_latent = diffusion_step_operation(
        admission.request_key,
        op_id=43,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=2,
        control_seq=preparation_commit.control_seq,
    )
    recovered = finalized_report(
        worker.execute(execution_run(run_id=16, operations=(replacement,)))
    )
    assert recovered.completions[0].status is OpStatus.OK
    assert recovered.completions[0].logical_lengths.latent_len == 2
    recovered_commit = commit_for_completion(replacement, recovered)
    recovered_artifact = _finalized_artifact(
        worker,
        admission,
        _replacement_latent,
        recovered_commit,
        op_id=44,
        run_id=17,
    )

    reference = execution_worker(block_size=4)
    reference_conditioning = _publish_conditioning(reference, admission, op_id=40, run_id=12)
    reference_latent, reference_preparation_commit = _prepare_media(
        reference,
        admission,
        reference_conditioning,
        op_id=41,
        parent=root_parent(admission),
        run_id=13,
    )
    reference_flow, reference_output = diffusion_step_operation(
        admission.request_key,
        op_id=43,
        parent=reference_preparation_commit.selected,
        conditioning=reference_conditioning,
        latent=reference_latent,
        steps=2,
        control_seq=reference_preparation_commit.control_seq,
    )
    reference_result = reference.execute(
        execution_run(
            run_id=14,
            operations=(reference_flow,),
            commands=(reference_preparation_commit,),
        )
    )
    assert reference_result.completions[0].status is OpStatus.OK
    reference_commit = commit_for_completion(reference_flow, reference_result)
    reference_artifact = _finalized_artifact(
        reference,
        admission,
        reference_output,
        reference_commit,
        op_id=44,
        run_id=15,
    )
    assert recovered_artifact == reference_artifact
    reference.close()


def test_mixed_lane_descriptor_failure_preserves_the_other_domain_candidate():
    worker = execution_worker()
    sequence_admission = ar_params(61, block_ids=(0,))
    generation_admission = umm_params(
        62,
        ImageParams(steps=1, height=16, width=16, seed=29),
    )
    conditioning = _publish_conditioning(worker, generation_admission, op_id=1, run_id=1)
    latent, preparation_commit = _prepare_media(
        worker,
        generation_admission,
        conditioning,
        op_id=2,
        parent=root_parent(generation_admission),
        run_id=2,
    )
    sequence, sequence_input, sequence_control = _prepare_decode(
        worker,
        sequence_admission,
        op_id=2,
        run_id=3,
        tokens=(7, 8),
    )
    missing_latent = replace(latent, generation=latent.generation + 1000)
    flow, _output_latent = diffusion_step_operation(
        generation_admission.request_key,
        op_id=3,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=missing_latent,
        steps=1,
        control_seq=preparation_commit.control_seq,
    )
    report = finalized_report(worker.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(sequence, flow),
            commands=(preparation_commit, sequence_control),
            input_products=(sequence_input,),
        )
    ))

    by_request = {record.request_key.request_id: record for record in report.completions}
    assert by_request[61].status is OpStatus.OK
    assert by_request[62].status is OpStatus.ERROR
    assert by_request[62].error_code is ErrorCode.INVALID_OPERATION
    sequence_commit = commit_for_completion(sequence, report)
    worker.execute(execution_run(run_id=5, commands=(sequence_commit,)))


def test_initial_flow_noise_is_stable_across_operation_schedules():
    admission = umm_params(5, ImageParams(steps=1, height=16, width=16, seed=29))
    artifacts: list[bytes] = []
    for op_id in (41, 109):
        worker = execution_worker()
        conditioning = _publish_conditioning(worker, admission, op_id=1, run_id=1)
        latent, preparation_commit = _prepare_media(
            worker,
            admission,
            conditioning,
            op_id=op_id,
            parent=root_parent(admission),
            run_id=2,
            seed=29,
            image_index=3,
        )
        flow, output_latent = diffusion_step_operation(
            admission.request_key,
            op_id=op_id + 1,
            parent=preparation_commit.selected,
            conditioning=conditioning,
            latent=latent,
            steps=1,
            control_seq=preparation_commit.control_seq,
        )
        flow_report = worker.execute(
            execution_run(
                run_id=3,
                admissions=(),
                operations=(flow,),
                commands=(preparation_commit,),
                input_products=(),
            )
        )
        flow_commit = commit_for_completion(flow, flow_report)
        artifacts.append(
            _finalized_artifact(
                worker,
                admission,
                output_latent,
                flow_commit,
                op_id=op_id + 2,
                run_id=4,
            )
        )
        worker.close()

    assert artifacts[1] == artifacts[0]


@pytest.mark.parametrize(
    ("height", "width", "cfg_text_scale", "cfg_img_scale"),
    (
        (16, 16, 1.0, 1.0),
        (16, 32, 4.0, 1.0),
        (32, 32, 4.0, 2.0),
    ),
)
def test_multi_step_quantum_matches_the_serial_model_artifact(
    height: int,
    width: int,
    cfg_text_scale: float,
    cfg_img_scale: float,
) -> None:
    def run(step_quantum: int) -> bytes:
        worker = execution_worker()
        admission = umm_params(
            76,
            ImageParams(
                steps=4,
                height=height,
                width=width,
                seed=31,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
            ),
        )
        conditioning = _publish_conditioning(worker, admission, op_id=1, run_id=1)
        latent, commit = _prepare_media(
            worker,
            admission,
            conditioning,
            op_id=2,
            parent=root_parent(admission),
            run_id=2,
            seed=31,
        )
        step = 0
        op_id = 3
        while step < 4:
            count = min(step_quantum, 4 - step)
            operation, successor = diffusion_step_operation(
                admission.request_key,
                op_id=op_id,
                parent=commit.selected,
                conditioning=conditioning,
                latent=latent,
                steps=count,
                control_seq=commit.control_seq,
            )
            report = finalized_report(worker.execute(
                execution_run(
                    run_id=op_id,
                    operations=(operation,),
                    commands=(commit,),
                )
            ))
            assert report.completions[0].status is OpStatus.OK
            step += count
            assert report.completions[0].logical_lengths.latent_len == step
            latent = successor
            commit = commit_for_completion(operation, report)
            op_id += 1
        artifact = _finalized_artifact(
            worker,
            admission,
            latent,
            commit,
            op_id=op_id,
            run_id=op_id,
        )
        worker.close()
        return artifact

    assert run(4) == run(1)


def test_decode_grows_logical_capacity_across_a_kv_page_boundary():
    # A small page forces the decode chain to cross a registration boundary.
    block_size = 4
    worker = execution_worker(block_size=block_size)
    admission = ar_params(1, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = finalized_report(worker.execute(
        execution_run(
            run_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    ))
    committed = list(extended.completions[0].committed_tokens)
    commit = commit_for_completion(extend, extended)
    next_block = 1
    block_count = 1
    crossed = False
    for step in range(4):
        length = 2 + step
        logical_delta: tuple[int, ...] = ()
        if length // block_size >= block_count:
            logical_delta = (next_block,)
            next_block += 1
            block_count += 1
            crossed = True
        decode, decode_input = token_operation(
            admission.request_key,
            op_id=2 + step,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(committed[-1],),
            block_table_delta=logical_delta,
            control_seq=commit.control_seq,
        )
        report = finalized_report(worker.execute(
            execution_run(
                run_id=2 + step,
                admissions=(),
                operations=(decode,),
                commands=(commit,),
                input_products=(decode_input,),
            )
        ))
        assert report.completions[0].logical_lengths.kv_visible_len == length + 1
        committed.extend(report.completions[0].committed_tokens)
        commit = commit_for_completion(decode, report)

    assert crossed  # the chain actually crossed a page boundary
    assert block_count == 2
    # Committed tokens follow the stub oracle unbroken across the boundary.
    chain = [_next_token(4)]
    for _ in range(4):
        chain.append(_next_token(chain[-1]))
    assert committed == chain


def test_flow_run_results_cumulative_denoise_step_in_latent_len():
    # The ordered-commit validator matches latent_len against the cumulative
    # denoise step (start_step + step_count), not a constant token count, so two
    # single-step quanta must report 1 then 2.
    worker = execution_worker()
    admission = umm_params(2, ImageParams(steps=2, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, run_id=1)
    initial_latent, preparation_commit = _prepare_media(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        run_id=2,
    )
    first, first_latent = diffusion_step_operation(
        admission.request_key,
        op_id=3,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
        control_seq=preparation_commit.control_seq,
    )
    first_report = finalized_report(worker.execute(
        execution_run(
            run_id=3,
            admissions=(),
            operations=(first,),
            commands=(preparation_commit,),
            input_products=(),
        )
    ))
    commit = commit_for_completion(first, first_report)
    second, _second_latent = diffusion_step_operation(
        admission.request_key,
        op_id=4,
        parent=commit.selected,
        conditioning=conditioning,
        latent=first_latent,
        steps=1,
        control_seq=commit.control_seq,
    )
    second_report = finalized_report(worker.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(second,),
            commands=(commit,),
            input_products=(),
        )
    ))

    assert first_report.completions[0].logical_lengths.latent_len == 1
    assert second_report.completions[0].logical_lengths.latent_len == 2


def test_trajectory_advances_across_many_generations_and_rejects_a_stale_reference():
    worker = execution_worker()
    admission = umm_params(71, ImageParams(steps=50, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, run_id=1)
    current, commit = _prepare_media(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        run_id=2,
    )
    stale = current
    releasable = None

    for index in range(50):
        operation, successor = diffusion_step_operation(
            admission.request_key,
            op_id=3 + index,
            parent=commit.selected,
            conditioning=conditioning,
            latent=current,
            steps=1,
            control_seq=commit.control_seq,
        )
        report = finalized_report(worker.execute(
            execution_run(
                run_id=3 + index,
                operations=(operation,),
                commands=(commit,),
            )
        ))
        assert report.completions[0].status is OpStatus.OK
        assert report.completions[0].logical_lengths.latent_len == index + 1
        releasable = current
        current = successor
        commit = commit_for_completion(operation, report)

    worker.execute(
        execution_run(
            run_id=53,
            commands=(
                commit,
                Free(releasable.buffer_id),
            ),
        )
    )
    stale_operation, _unused = diffusion_step_operation(
        admission.request_key,
        op_id=54,
        parent=commit.selected,
        conditioning=conditioning,
        latent=stale,
        steps=1,
        control_seq=commit.control_seq,
    )
    stale_report = worker.execute(execution_run(run_id=54, operations=(stale_operation,)))
    assert stale_report.completions[0].status is OpStatus.ERROR


def test_cross_stage_feature_transfer_rebinds_exact_product_without_request_thread_wait() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = ar_params(73, block_ids=(0,))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (64, 96, 128)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    operation, source = encode_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        image_base64=encoded,
        encoder_handle=11,
    )
    assert source is not None
    try:
        produced = producer.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(source,),
            )
        )
        deadline = time.monotonic() + 5.0
        while not run_result_ready(produced) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert run_result_ready(produced)
        produced = finalize_run_result(produced)
        assert len(produced.products) == 1
        transferred = produced.products[0]
        assert transferred.product == operation.outputs[0]
        assert isinstance(transferred.payload, TransferHandle)
        visual = visual_state_operation(
            admission.request_key,
            op_id=2,
            parent=root_parent(admission),
            feature=operation.outputs[0],
            sample_continuation=False,
            max_tokens=2,
        )
        batch = execution_run(
            run_id=2,
            admissions=(admission,),
            operations=(visual,),
            input_products=(transferred,),
        )
        prepared = consumer.prepare_execute(batch)
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()
        consumed = consumer.execute_prepared(prepared)
        assert consumed.completions[0].status is OpStatus.OK
        producer.execute(
            execution_run(
                run_id=3,
                commands=(Free(operation.outputs[0].buffer_id),),
            )
        )
    finally:
        producer.close()
        consumer.close()


def test_cross_stage_device_product_transfer_preserves_generation_and_value() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = ar_params(77, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    try:
        extended = finalized_report(
            producer.execute(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(extend,),
                    input_products=(extend_input,),
                )
            )
        )
        commit = commit_for_completion(extend, extended)
        source = next(output for output in extend.outputs if output.kind is ProductKind.TOKEN)
        transferred = replace(source, producer_op_id=2, generation=901)
        transfer = Operation.registered(
            request_key=admission.request_key,
            op_id=2,
            parent=commit.selected,
            kind=RunKind.TRANSFER_PRODUCT,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(transferred,),
            control_seq=commit.control_seq,
        )
        transfer_report = finalized_report(
            producer.execute(
                execution_run(
                    run_id=2,
                    operations=(transfer,),
                    commands=(commit,),
                )
            )
        )
        assert len(transfer_report.products) == 1
        payload = transfer_report.products[0]
        assert payload.product == transferred
        kind, descriptor = decode_transfer_handle(payload.payload)
        assert kind == "device_product"
        assert descriptor["generation"] == transferred.generation

        consume, consume_input = token_operation(
            admission.request_key,
            op_id=3,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(9,),
            predicate=transferred,
        )
        consume = Operation.registered(
            request_key=consume.request_key,
            op_id=consume.op_id,
            parent=consume.parent,
            kind=consume.kind,
            bounds=consume.bounds,
            inputs=(*consume.inputs, transferred),
            outputs=consume.outputs,
            predicate=transferred,
            rng=consume.rng,
            control_seq=consume.control_seq,
        )
        prepared = consumer.prepare_execute(
            execution_run(
                run_id=3,
                admissions=(admission,),
                operations=(consume,),
                input_products=(consume_input, payload),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()
        consumed = finalized_report(consumer.execute_prepared(prepared))
        assert consumed.completions[0].status is OpStatus.OK
        assert consumed.completions[0].logical_lengths.kv_visible_len == 1
    finally:
        producer.close()
        consumer.close()


def test_cross_stage_completion_predicate_preserves_device_continuation() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    generation = umm_params(79, ImageParams(steps=1, height=16, width=16, seed=31))
    admission = NewRequest.create(
        generation.request_key,
        request_pool_idx=generation.request_pool_idx,
        ar=ArRequestParams(),
        umm=generation.umm,
    )
    bind_request_placement(
        admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        page_ids=(1,),
    )
    try:
        conditioning = _publish_conditioning(producer, admission, op_id=1, run_id=1)
        preparation, _latent = diffusion_prepare_operation(
            admission.request_key,
            op_id=2,
            parent=root_parent(admission),
            conditioning=conditioning,
            seed=31,
        )
        transitioned = finalized_report(
            producer.execute(execution_run(run_id=2, operations=(preparation,)))
        )
        preparation_commit = commit_for_completion(preparation, transitioned)
        source = next(
            output for output in preparation.outputs if output.kind is ProductKind.COMPLETION
        )
        transferred = replace(source, producer_op_id=3, generation=903)
        transfer = Operation.registered(
            request_key=admission.request_key,
            op_id=3,
            parent=preparation_commit.selected,
            kind=RunKind.TRANSFER_PRODUCT,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(transferred,),
            control_seq=preparation_commit.control_seq,
        )
        transfer_report = finalized_report(
            producer.execute(
                execution_run(
                    run_id=3,
                    operations=(transfer,),
                    commands=(preparation_commit,),
                )
            )
        )
        payload = next(
            product for product in transfer_report.products if product.product == transferred
        )
        consume, consume_input = token_operation(
            admission.request_key,
            op_id=4,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(9,),
            predicate=transferred,
        )
        prepared = consumer.prepare_execute(
            execution_run(
                run_id=4,
                admissions=(admission,),
                operations=(consume,),
                input_products=(consume_input, payload),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()
        consumed = finalized_report(consumer.execute_prepared(prepared))
        assert consumed.completions[0].status is OpStatus.OK
        assert consumed.completions[0].logical_lengths.kv_visible_len == 1
    finally:
        producer.close()
        consumer.close()


def test_cross_stage_latent_transfer_preserves_generation_step_and_artifact() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = umm_params(75, ImageParams(steps=1, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(producer, admission, op_id=1, run_id=1)
    initial_latent, preparation_commit = _prepare_media(
        producer,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        run_id=2,
    )
    flow, final_latent = diffusion_step_operation(
        admission.request_key,
        op_id=3,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
        control_seq=preparation_commit.control_seq,
    )
    produced = producer.execute(
        execution_run(
            run_id=3,
            operations=(flow,),
            commands=(preparation_commit,),
        )
    )
    deadline = time.monotonic() + 5.0
    while not run_result_ready(produced) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert run_result_ready(produced)
    produced = finalize_run_result(produced)
    transferred = tuple(product for product in produced.products if product.product == final_latent)
    assert len(transferred) == 1
    assert isinstance(transferred[0].payload, TransferHandle)

    flow_commit = commit_for_completion(flow, produced)
    source_artifact = _finalized_artifact(
        producer,
        admission,
        final_latent,
        flow_commit,
        op_id=4,
        run_id=4,
    )
    diffusion_finalize = diffusion_finalize_operation(
        admission.request_key,
        op_id=4,
        parent=root_parent(admission),
        latent=final_latent,
    )
    batch = execution_run(
        run_id=4,
        admissions=(admission,),
        operations=(diffusion_finalize,),
        input_products=transferred,
    )
    prepared = consumer.prepare_execute(batch)
    assert prepared is not None
    deadline = time.monotonic() + 5.0
    while not prepared.ready() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert prepared.ready()
    received = consumer.execute_prepared(prepared)
    deadline = time.monotonic() + 5.0
    while not run_result_ready(received) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert run_result_ready(received)
    received = finalize_run_result(received)
    assert received.completions[0].status is OpStatus.OK
    assert received.completions[0].logical_lengths.latent_len == 0
    received_artifacts = tuple(
        product.payload
        for product in received.products
        if product.product.kind is ProductKind.ARTIFACT
    )
    assert received_artifacts == (source_artifact,)
    producer.close()
    consumer.close()


def test_encode_publishes_an_immutable_feature_without_advancing_state():
    worker = execution_worker()
    admission = ar_params(3, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = finalized_report(worker.execute(
        execution_run(
            run_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    ))
    session_kv_before = 2

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    handle = 0xABCDEF
    commit = commit_for_completion(extend, extended)
    encode, encode_input = encode_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        image_base64=image_base64,
        encoder_handle=handle,
        control_seq=commit.control_seq,
    )
    report = finalized_report(worker.execute(
        execution_run(
            run_id=2,
            admissions=(),
            operations=(encode,),
            commands=(commit,),
            input_products=(encode_input,),
        )
    ))
    completion = report.completions[0]
    assert completion.logical_lengths.kv_visible_len == session_kv_before
    assert completion.logical_lengths.token_len == 2
    assert completion.selected_point == 0
    assert completion.product_generations
    assert completion.product_generations[0] == handle
    assert completion.product_generations[0] != 0


def test_generated_feedback_commits_absolute_visual_token_state():
    worker = execution_worker()
    understanding = ar_params(6, block_ids=(0,))
    admission = NewRequest.create(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        ar=understanding.ar,
        umm=UmmRequestParams(
            ImageParams(steps=2, height=16, width=16, seed=29, retain_images=True)
        ),
    )
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = worker.execute(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )
    first_commit = commit_for_completion(extend, extended)
    publication, conditioning = kv_publication_operation(
        admission.request_key,
        op_id=2,
        parent=first_commit.selected,
        control_seq=first_commit.control_seq,
    )
    worker.execute(
        execution_run(
            run_id=2,
            admissions=(),
            operations=(publication,),
            commands=(first_commit,),
        )
    )
    initial_latent, preparation_commit = _prepare_media(
        worker,
        admission,
        conditioning,
        op_id=3,
        parent=first_commit.selected,
        run_id=3,
        control_seq=first_commit.control_seq,
    )
    flow, completed_latent = diffusion_step_operation(
        admission.request_key,
        op_id=4,
        parent=preparation_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=2,
        control_seq=preparation_commit.control_seq,
    )
    flow_report = worker.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(flow,),
            commands=(preparation_commit,),
            input_products=(),
        )
    )
    commit = commit_for_completion(flow, flow_report)
    diffusion_finalize = diffusion_finalize_operation(
        admission.request_key,
        op_id=5,
        parent=commit.selected,
        latent=completed_latent,
        feedback_source=True,
        control_seq=commit.control_seq,
    )
    diffusion_finalize_report = worker.execute(
        execution_run(
            run_id=5,
            admissions=(),
            operations=(diffusion_finalize,),
            commands=(commit,),
            input_products=(),
        )
    )
    deadline = time.monotonic() + 5.0
    while not run_result_ready(diffusion_finalize_report) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert run_result_ready(diffusion_finalize_report)
    diffusion_finalize_report = finalize_run_result(diffusion_finalize_report)
    assert diffusion_finalize_report.completions[0].logical_lengths.kv_visible_len == 2
    assert diffusion_finalize_report.completions[0].selected_point == 0

    encode, _ = encode_operation(
        admission.request_key,
        op_id=6,
        parent=commit.selected,
        image_base64=None,
        encoder_handle=10,
        source_product=diffusion_finalize.outputs[1],
        control_seq=commit.control_seq,
    )
    encode_report = finalized_report(
        worker.execute(
            execution_run(run_id=6, admissions=(), operations=(encode,), input_products=())
        )
    )
    assert encode_report.completions[0].logical_lengths.kv_visible_len == 2
    assert encode_report.completions[0].selected_point == 0

    state = visual_state_operation(
        admission.request_key,
        op_id=7,
        parent=commit.selected,
        feature=encode.outputs[0],
        sample_continuation=True,
        max_tokens=2,
        control_seq=commit.control_seq,
    )
    next_token = _next_token(1007)
    sampling_bytes = encode_sampling_state_bytes(
        SamplingState(
            finish_token_ids=(next_token + 1,),
            transition_token_ids=(next_token,),
        )
    )
    sampling_product = ProductRef(
        request_key=admission.request_key,
        producer_op_id=state.op_id,
        output_index=0xFFFF,
        generation=state.op_id * 8 + 7,
        kind=ProductKind.SAMPLING_STATE,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U8,
        shape_bound=ShapeBound((StaticDim(len(sampling_bytes)),)),
        point_range=PointRange(),
    )
    transition_product = ProductRef(
        request_key=admission.request_key,
        producer_op_id=state.op_id,
        output_index=3,
        generation=state.op_id * 8 + 9,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.REQUEST_RELAY,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    state = Operation.registered(
        request_key=state.request_key,
        op_id=state.op_id,
        parent=state.parent,
        kind=state.kind,
        bounds=state.bounds,
        inputs=(*state.inputs, sampling_product),
        outputs=(*state.outputs, transition_product),
        predicate=state.predicate,
        rng=state.rng,
        control_seq=state.control_seq,
    )
    report = finalized_report(worker.execute(
        execution_run(
            run_id=7,
            admissions=(),
            operations=(state,),
            input_products=(ProductPayload(sampling_product, sampling_bytes),),
        )
    ))

    completion = report.completions[0]
    assert completion.committed_tokens == (next_token,)
    assert completion.token_span.base == 2
    assert completion.token_span.len == 1
    assert completion.logical_lengths.token_len == 4
    assert completion.logical_lengths.kv_visible_len == 4
    assert completion.selected_point == 1

    feedback_commit = commit_for_completion(state, report)
    publication, next_conditioning = kv_publication_operation(
        admission.request_key,
        op_id=8,
        parent=feedback_commit.selected,
        control_seq=feedback_commit.control_seq,
    )
    worker.execute(
        execution_run(
            run_id=8,
            operations=(publication,),
            commands=(feedback_commit,),
        )
    )
    next_latent, _next_preparation_commit = _prepare_media(
        worker,
        admission,
        next_conditioning,
        op_id=9,
        parent=feedback_commit.selected,
        run_id=9,
        control_seq=feedback_commit.control_seq,
        image_index=2,
    )
    next_flow, next_completed_latent = diffusion_step_operation(
        admission.request_key,
        op_id=10,
        parent=_next_preparation_commit.selected,
        conditioning=next_conditioning,
        latent=next_latent,
        steps=2,
        control_seq=_next_preparation_commit.control_seq,
    )
    next_report = finalized_report(worker.execute(
        execution_run(
            run_id=10,
            operations=(next_flow,),
            commands=(_next_preparation_commit,),
        )
    ))
    assert next_report.completions[0].status is OpStatus.OK
    assert next_report.completions[0].logical_lengths.latent_len == 2
    next_commit = commit_for_completion(next_flow, next_report)
    assert base64.b64decode(
        _finalized_artifact(
            worker,
            admission,
            next_completed_latent,
            next_commit,
            op_id=11,
            run_id=11,
        ).decode("ascii"),
        validate=True,
    ).startswith(_PNG_MAGIC)

    artifacts = [p for p in diffusion_finalize_report.products if p.product.kind is ProductKind.ARTIFACT]
    assert len(artifacts) == 1
    # The Artifact product carries the base64 PNG string as bytes: the scheduler
    # recovers it with String::from_utf8 and hands it to validate_png_artifact,
    # which base64-decodes it and checks the PNG dimensions. Mirror that response.
    png_b64 = artifacts[0].payload
    png_bytes = base64.b64decode(png_b64.decode("ascii"), validate=True)
    assert png_bytes[:8] == _PNG_MAGIC
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (16, 16)
