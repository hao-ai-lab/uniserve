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
    bind_request_placement,
    commit_for_completion,
    encode_operation,
    execution_batch,
    finalized_report,
    flow_operation,
    gen_admission,
    gen_transition_operation,
    kv_publication_operation,
    materialize_operation,
    root_parent,
    token_operation,
    und_admission,
    visual_state_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Bounds,
    CacheGroupPlacement,
    Commit,
    DeviceDim,
    DevicePoint,
    Domain,
    DType,
    ErrorCode,
    GenAdmission,
    ImageParams,
    KvPlacement,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RecoveryPlacement,
    Release,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TokenMode,
    TransferMode,
    UndAdmission,
    VersionRef,
    Work,
    encode_sampling_state_bytes,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
)
from uniserve_worker.foundation.errors import ErrorCode as HostErrorCode
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.server.completion import completion_report_ready, finalize_completion_report
from uniserve_worker.server.stub import StubModel, _next_token
from uniserve_worker.transfer.tickets import (
    TRANSFER_DESCRIPTOR_PREFIX,
    decode_transfer_descriptor,
)

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


def _publish_conditioning(worker: object, admission: Admission, *, op_id: int, step_id: int):
    publication, product = kv_publication_operation(
        admission.request_key,
        op_id=op_id,
        parent=root_parent(admission),
    )
    worker.execute(
        execution_batch(step_id=step_id, admissions=(admission,), operations=(publication,))
    )
    return product


def _transition_generation(
    worker: object,
    admission: Admission,
    conditioning: object,
    *,
    op_id: int,
    parent: object,
    step_id: int,
    control_seq: int = 0,
    seed: int = 29,
    image_index: int = 1,
):
    transition, latent = gen_transition_operation(
        admission.request_key,
        op_id=op_id,
        parent=parent,
        conditioning=conditioning,
        control_seq=control_seq,
        seed=seed,
        image_index=image_index,
    )
    report = worker.execute(execution_batch(step_id=step_id, operations=(transition,)))
    assert report.completions[0].status is OpStatus.OK
    return latent, commit_for_completion(transition, report)


def _prepare_decode(
    worker: object,
    admission: Admission,
    *,
    op_id: int,
    step_id: int,
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
        execution_batch(
            step_id=step_id,
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


def _materialized_artifact(
    worker: object,
    admission: Admission,
    latent: ProductRef,
    commit: Commit,
    *,
    op_id: int,
    step_id: int,
) -> bytes:
    operation = materialize_operation(
        admission.request_key,
        op_id=op_id,
        parent=commit.selected,
        latent=latent,
        control_seq=commit.control_seq,
    )
    report = worker.execute(
        execution_batch(step_id=step_id, operations=(operation,), controls=(commit,))
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
    admission = und_admission(1, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = finalized_report(worker.execute(
        execution_batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    ))

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
    decoded = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(decode,),
            controls=(commit,),
            input_products=(decode_input,),
        )
    )

    assert decoded.completions[0].committed_tokens == (_next_token(first_token),)
    assert decoded.completions[0].logical_lengths.kv_visible_len == 3
    assert decoded.completions[0].logical_lengths.token_len == 3


def test_prefix_reuse_continues_from_the_admitted_logical_position():
    worker = execution_worker()
    admission = und_admission(8, block_ids=(0,), prefix_len=2)
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(4,),
    )

    report = finalized_report(worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    ))

    assert report.completions[0].logical_lengths.kv_visible_len == 3
    assert report.completions[0].logical_lengths.token_len == 3


def test_invalid_physical_placement_reports_error_behind_an_unobserved_parent() -> None:
    worker = execution_worker(pipeline_depth=2)
    admission = und_admission(9, block_ids=(0,))
    parent, parent_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
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
        DevicePoint(1, None, parent.plan_digest),
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
        work=template.work,
        route=template.route,
        domain=template.domain,
        bounds=template.bounds,
        outputs=template.outputs,
        predicate=template.predicate,
    )
    invalid_placement = KvPlacement(
        request_key=operation.request_key,
        op_id=operation.op_id,
        group_id=0,
        block_table=(),
        pages_to_zero=(),
        prefix_length=2,
        input_length=1,
        visible_length=2,
        resulting_length=3,
    )

    report = finalize_completion_report(
        worker.execute(
            execution_batch(
                step_id=2,
                operations=(operation,),
                kv_placements=(invalid_placement,),
            )
        )
    )

    assert report.completions[0].status is OpStatus.ERROR
    assert report.completions[0].error_code is ErrorCode.INVALID_OPERATION


def test_mixed_token_and_flow_match_homogeneous_results():
    mixed = execution_worker()
    sequence_admission = und_admission(1, block_ids=(0,))
    flow_admission = gen_admission(2, ImageParams(steps=1, height=16, width=16, seed=29))
    mixed_conditioning = _publish_conditioning(mixed, flow_admission, op_id=10, step_id=1)
    mixed_latent, mixed_transition_commit = _transition_generation(
        mixed,
        flow_admission,
        mixed_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        step_id=2,
    )
    flow, mixed_output_latent = flow_operation(
        flow_admission.request_key,
        op_id=12,
        parent=mixed_transition_commit.selected,
        conditioning=mixed_conditioning,
        latent=mixed_latent,
        steps=1,
        control_seq=mixed_transition_commit.control_seq,
    )
    sequence, sequence_input, sequence_control = _prepare_decode(
        mixed,
        sequence_admission,
        op_id=10,
        step_id=3,
        tokens=(3, 4),
    )

    mixed_result = mixed.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(sequence, flow),
            controls=(mixed_transition_commit, sequence_control),
            input_products=(sequence_input,),
        )
    )

    split = execution_worker()
    split_conditioning = _publish_conditioning(split, flow_admission, op_id=10, step_id=1)
    split_latent, split_transition_commit = _transition_generation(
        split,
        flow_admission,
        split_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        step_id=2,
    )
    split_flow, split_output_latent = flow_operation(
        flow_admission.request_key,
        op_id=12,
        parent=split_transition_commit.selected,
        conditioning=split_conditioning,
        latent=split_latent,
        steps=1,
        control_seq=split_transition_commit.control_seq,
    )
    split_sequence, split_sequence_input, split_sequence_control = _prepare_decode(
        split,
        sequence_admission,
        op_id=10,
        step_id=3,
        tokens=(3, 4),
    )
    sequence_result = split.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(split_sequence,),
            controls=(split_sequence_control,),
            input_products=(split_sequence_input,),
        )
    )
    flow_result = split.execute(
        execution_batch(
            step_id=5,
            admissions=(),
            operations=(split_flow,),
            controls=(split_transition_commit,),
            input_products=(),
        )
    )

    assert (
        mixed_result.completions[0].committed_tokens
        == sequence_result.completions[0].committed_tokens
    )
    assert (
        mixed_result.completions[0].semantic_digest
        == sequence_result.completions[0].semantic_digest
    )
    assert mixed_result.completions[1].semantic_digest == flow_result.completions[0].semantic_digest
    mixed_flow_commit = commit_for_completion(flow, mixed_result)
    split_flow_commit = commit_for_completion(split_flow, flow_result)
    assert _materialized_artifact(
        mixed,
        flow_admission,
        mixed_output_latent,
        mixed_flow_commit,
        op_id=13,
        step_id=5,
    ) == _materialized_artifact(
        split,
        flow_admission,
        split_output_latent,
        split_flow_commit,
        op_id=13,
        step_id=6,
    )


def test_mixed_submission_requires_tensorized_model_capability():
    worker = execution_worker(_SeparatePhaseModel())
    token_admission = und_admission(1, block_ids=(0,))
    flow_admission = gen_admission(2, ImageParams(steps=1, height=16, width=16, seed=29))
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
    transition, _latent = gen_transition_operation(
        flow_admission.request_key,
        op_id=11,
        parent=root_parent(flow_admission),
        conditioning=conditioning,
    )

    with pytest.raises(WorkerError) as rejected:
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(token_admission,),
                operations=(token, transition),
                input_products=(token_input,),
            )
        )

    assert rejected.value.code is HostErrorCode.INVALID_DESCRIPTOR


def test_request_scoped_operation_identity_preserves_homogeneous_decode():
    worker = execution_worker()
    admissions = (und_admission(41, block_ids=(0,)), und_admission(42, block_ids=(1,)))
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
        execution_batch(
            step_id=1,
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
    decoded = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=tuple(decode_ops),
            controls=tuple(commits),
            input_products=tuple(decode_inputs),
        )
    )

    assert tuple(record.committed_tokens for record in decoded.completions) == tuple(
        (_next_token(_next_token(token)),) for token in last_tokens
    )


def test_output_validation_failure_discards_all_candidate_state():
    model = _MisalignedOutputModel()
    worker = execution_worker(model)
    admission = und_admission(4, block_ids=(4,))
    initial, initial_input = token_operation(
        admission.request_key,
        op_id=31,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(12, 13),
    )
    initial_report = worker.execute(
        execution_batch(
            step_id=11,
            admissions=(admission,),
            operations=(initial,),
            input_products=(initial_input,),
        )
    )
    commit = commit_for_completion(initial, initial_report)
    worker.execute(execution_batch(step_id=12, admissions=(), operations=(), controls=(commit,)))
    retry, retry_input = token_operation(
        admission.request_key,
        op_id=32,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(_next_token(13),),
        control_seq=commit.control_seq,
    )
    retry_batch = execution_batch(
        step_id=13, admissions=(), operations=(retry,), input_products=(retry_input,)
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
    result = worker.execute(
        execution_batch(
            step_id=14,
            operations=(replacement,),
            input_products=(replacement_input,),
        )
    )
    assert result.completions[0].committed_tokens == (_next_token(_next_token(13)),)
    assert result.completions[0].logical_lengths.kv_visible_len == 3


def test_failed_flow_preserves_the_next_accepted_trajectory_and_final_artifact():
    worker = execution_worker(block_size=4)
    admission = gen_admission(5, ImageParams(steps=2, height=64, width=64, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=40, step_id=12)
    latent, transition_commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=41,
        parent=root_parent(admission),
        step_id=13,
    )
    flow, _output_latent = flow_operation(
        admission.request_key,
        op_id=42,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=2,
        control_seq=transition_commit.control_seq,
    )
    worker.execute(execution_batch(step_id=14, controls=(transition_commit,)))
    valid_batch = execution_batch(
        step_id=15,
        admissions=(),
        operations=(flow,),
        input_products=(),
    )
    batch = replace(
        valid_batch,
        partitions=tuple(
            replace(
                partition,
                kv_branch_placements=tuple(
                    replace(
                        placement,
                        block_table=placement.block_table[:1],
                        pages_to_zero=placement.pages_to_zero[:1],
                    )
                    for placement in partition.kv_branch_placements
                ),
            )
            for partition in valid_batch.partitions
        ),
    )

    failed = worker.execute(batch)

    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.INVALID_OPERATION
    assert failed.completions[0].logical_lengths.latent_len == 0

    replacement, _replacement_latent = flow_operation(
        admission.request_key,
        op_id=43,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=2,
        control_seq=transition_commit.control_seq,
    )
    recovered = worker.execute(execution_batch(step_id=16, operations=(replacement,)))
    assert recovered.completions[0].status is OpStatus.OK
    assert recovered.completions[0].logical_lengths.latent_len == 2
    recovered_commit = commit_for_completion(replacement, recovered)
    recovered_artifact = _materialized_artifact(
        worker,
        admission,
        _replacement_latent,
        recovered_commit,
        op_id=44,
        step_id=17,
    )

    reference = execution_worker(block_size=4)
    reference_conditioning = _publish_conditioning(reference, admission, op_id=40, step_id=12)
    reference_latent, reference_transition_commit = _transition_generation(
        reference,
        admission,
        reference_conditioning,
        op_id=41,
        parent=root_parent(admission),
        step_id=13,
    )
    reference_flow, reference_output = flow_operation(
        admission.request_key,
        op_id=43,
        parent=reference_transition_commit.selected,
        conditioning=reference_conditioning,
        latent=reference_latent,
        steps=2,
        control_seq=reference_transition_commit.control_seq,
    )
    reference_result = reference.execute(
        execution_batch(
            step_id=14,
            operations=(reference_flow,),
            controls=(reference_transition_commit,),
        )
    )
    assert reference_result.completions[0].status is OpStatus.OK
    reference_commit = commit_for_completion(reference_flow, reference_result)
    reference_artifact = _materialized_artifact(
        reference,
        admission,
        reference_output,
        reference_commit,
        op_id=44,
        step_id=15,
    )
    assert recovered_artifact == reference_artifact
    reference.close()


def test_mixed_partition_descriptor_failure_preserves_the_other_domain_candidate():
    worker = execution_worker()
    sequence_admission = und_admission(61, block_ids=(0,))
    generation_admission = gen_admission(
        62,
        ImageParams(steps=1, height=16, width=16, seed=29),
    )
    conditioning = _publish_conditioning(worker, generation_admission, op_id=1, step_id=1)
    latent, transition_commit = _transition_generation(
        worker,
        generation_admission,
        conditioning,
        op_id=2,
        parent=root_parent(generation_admission),
        step_id=2,
    )
    sequence, sequence_input, sequence_control = _prepare_decode(
        worker,
        sequence_admission,
        op_id=2,
        step_id=3,
        tokens=(7, 8),
    )
    missing_latent = replace(latent, generation=latent.generation + 1000)
    flow, _output_latent = flow_operation(
        generation_admission.request_key,
        op_id=3,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=missing_latent,
        steps=1,
        control_seq=transition_commit.control_seq,
    )
    report = worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(sequence, flow),
            controls=(transition_commit, sequence_control),
            input_products=(sequence_input,),
        )
    )

    by_request = {record.request_key.session_id: record for record in report.completions}
    assert by_request[61].status is OpStatus.OK
    assert by_request[62].status is OpStatus.ERROR
    assert by_request[62].error_code is ErrorCode.INVALID_OPERATION
    sequence_commit = commit_for_completion(sequence, report)
    worker.execute(execution_batch(step_id=5, controls=(sequence_commit,)))


def test_initial_flow_noise_is_stable_across_operation_schedules():
    admission = gen_admission(5, ImageParams(steps=1, height=16, width=16, seed=29))
    artifacts: list[bytes] = []
    for op_id in (41, 109):
        worker = execution_worker()
        conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
        latent, transition_commit = _transition_generation(
            worker,
            admission,
            conditioning,
            op_id=op_id,
            parent=root_parent(admission),
            step_id=2,
            seed=29,
            image_index=3,
        )
        flow, output_latent = flow_operation(
            admission.request_key,
            op_id=op_id + 1,
            parent=transition_commit.selected,
            conditioning=conditioning,
            latent=latent,
            steps=1,
            control_seq=transition_commit.control_seq,
        )
        flow_report = worker.execute(
            execution_batch(
                step_id=3,
                admissions=(),
                operations=(flow,),
                controls=(transition_commit,),
                input_products=(),
            )
        )
        flow_commit = commit_for_completion(flow, flow_report)
        artifacts.append(
            _materialized_artifact(
                worker,
                admission,
                output_latent,
                flow_commit,
                op_id=op_id + 2,
                step_id=4,
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
        admission = gen_admission(
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
        conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
        latent, commit = _transition_generation(
            worker,
            admission,
            conditioning,
            op_id=2,
            parent=root_parent(admission),
            step_id=2,
            seed=31,
        )
        step = 0
        op_id = 3
        while step < 4:
            count = min(step_quantum, 4 - step)
            operation, successor = flow_operation(
                admission.request_key,
                op_id=op_id,
                parent=commit.selected,
                conditioning=conditioning,
                latent=latent,
                steps=count,
                control_seq=commit.control_seq,
            )
            report = worker.execute(
                execution_batch(
                    step_id=op_id,
                    operations=(operation,),
                    controls=(commit,),
                )
            )
            assert report.completions[0].status is OpStatus.OK
            step += count
            assert report.completions[0].logical_lengths.latent_len == step
            latent = successor
            commit = commit_for_completion(operation, report)
            op_id += 1
        artifact = _materialized_artifact(
            worker,
            admission,
            latent,
            commit,
            op_id=op_id,
            step_id=op_id,
        )
        worker.close()
        return artifact

    assert run(4) == run(1)


def test_decode_grows_logical_capacity_across_a_kv_page_boundary():
    # A small page forces the decode chain to cross a registration boundary.
    block_size = 4
    worker = execution_worker(block_size=block_size)
    admission = und_admission(1, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = worker.execute(
        execution_batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    )
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
        report = worker.execute(
            execution_batch(
                step_id=2 + step,
                admissions=(),
                operations=(decode,),
                controls=(commit,),
                input_products=(decode_input,),
            )
        )
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


def test_flow_completion_reports_cumulative_denoise_step_in_latent_len():
    # The ordered-commit validator matches latent_len against the cumulative
    # denoise step (start_step + step_count), not a constant token count, so two
    # single-step quanta must report 1 then 2.
    worker = execution_worker()
    admission = gen_admission(2, ImageParams(steps=2, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
    initial_latent, transition_commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        step_id=2,
    )
    first, first_latent = flow_operation(
        admission.request_key,
        op_id=3,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
        control_seq=transition_commit.control_seq,
    )
    first_report = worker.execute(
        execution_batch(
            step_id=3,
            admissions=(),
            operations=(first,),
            controls=(transition_commit,),
            input_products=(),
        )
    )
    commit = commit_for_completion(first, first_report)
    second, _second_latent = flow_operation(
        admission.request_key,
        op_id=4,
        parent=commit.selected,
        conditioning=conditioning,
        latent=first_latent,
        steps=1,
        control_seq=commit.control_seq,
    )
    second_report = worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(second,),
            controls=(commit,),
            input_products=(),
        )
    )

    assert first_report.completions[0].logical_lengths.latent_len == 1
    assert second_report.completions[0].logical_lengths.latent_len == 2


def test_trajectory_advances_across_many_generations_and_rejects_a_stale_reference():
    worker = execution_worker()
    admission = gen_admission(71, ImageParams(steps=50, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
    current, commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        step_id=2,
    )
    stale = current
    releasable = None

    for index in range(50):
        operation, successor = flow_operation(
            admission.request_key,
            op_id=3 + index,
            parent=commit.selected,
            conditioning=conditioning,
            latent=current,
            steps=1,
            control_seq=commit.control_seq,
        )
        controls = (commit,) + (
            ()
            if releasable is None
            else (Release(admission.request_key, releasable.producer_op_id),)
        )
        report = worker.execute(
            execution_batch(
                step_id=3 + index,
                operations=(operation,),
                controls=controls,
            )
        )
        assert report.completions[0].status is OpStatus.OK
        assert report.completions[0].logical_lengths.latent_len == index + 1
        releasable = current
        current = successor
        commit = commit_for_completion(operation, report)

    worker.execute(
        execution_batch(
            step_id=53,
            controls=(
                commit,
                Release(admission.request_key, releasable.producer_op_id),
            ),
        )
    )
    stale_operation, _unused = flow_operation(
        admission.request_key,
        op_id=54,
        parent=commit.selected,
        conditioning=conditioning,
        latent=stale,
        steps=1,
        control_seq=commit.control_seq,
    )
    stale_report = worker.execute(execution_batch(step_id=54, operations=(stale_operation,)))
    assert stale_report.completions[0].status is OpStatus.ERROR


def test_snapshot_restore_preserves_an_active_trajectory_and_final_artifact(tmp_path) -> None:
    snapshot_dir = str(tmp_path / "worker-state")
    worker = execution_worker(snapshot_dir=snapshot_dir)
    admission = gen_admission(72, ImageParams(steps=2, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
    latent, transition_commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        step_id=2,
    )
    first_flow, partial_latent = flow_operation(
        admission.request_key,
        op_id=3,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=1,
        control_seq=transition_commit.control_seq,
    )
    first = worker.execute(
        execution_batch(
            step_id=3,
            operations=(first_flow,),
            controls=(transition_commit,),
        )
    )
    assert first.completions[0].status is OpStatus.OK
    assert first.completions[0].logical_lengths.latent_len == 1
    partial_commit = commit_for_completion(first_flow, first)
    worker.execute(execution_batch(step_id=4, controls=(partial_commit,)))
    placement = RecoveryPlacement(
        request_key=admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        cache_groups=(CacheGroupPlacement(group_id=0, page_ids=(), length=0),),
        latent_page_table=(1,),
    )
    reference = worker.snapshot_session(placement)

    source_continuation, source_final_latent = flow_operation(
        admission.request_key,
        op_id=4,
        parent=reference.version,
        conditioning=conditioning,
        latent=partial_latent,
        steps=1,
        control_seq=partial_commit.control_seq,
    )
    source_report = worker.execute(execution_batch(step_id=5, operations=(source_continuation,)))
    assert source_report.completions[0].status is OpStatus.OK
    assert source_report.completions[0].logical_lengths.latent_len == 2
    source_commit = commit_for_completion(source_continuation, source_report)
    source_artifact = _materialized_artifact(
        worker,
        admission,
        source_final_latent,
        source_commit,
        op_id=5,
        step_id=6,
    )

    restored = execution_worker(
        snapshot_dir=snapshot_dir,
    )
    restored.restore_session(reference, placement)
    continuation, successor = flow_operation(
        admission.request_key,
        op_id=4,
        parent=reference.version,
        conditioning=conditioning,
        latent=partial_latent,
        steps=1,
        control_seq=partial_commit.control_seq,
    )

    report = restored.execute(execution_batch(step_id=5, operations=(continuation,)))

    assert report.completions[0].status is OpStatus.OK
    assert report.completions[0].logical_lengths.latent_len == 2
    restored_commit = commit_for_completion(continuation, report)
    restored_artifact = _materialized_artifact(
        restored,
        admission,
        successor,
        restored_commit,
        op_id=5,
        step_id=6,
    )
    assert restored_artifact == source_artifact
    worker.close()
    restored.close()


def test_cross_stage_feature_transfer_rebinds_exact_product_without_request_thread_wait() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = und_admission(73, block_ids=(0,))
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
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(source,),
            )
        )
        deadline = time.monotonic() + 5.0
        while not completion_report_ready(produced) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert completion_report_ready(produced)
        produced = finalize_completion_report(produced)
        assert len(produced.products) == 1
        transferred = produced.products[0]
        assert transferred.product == operation.outputs[0]
        assert transferred.payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)
        visual = visual_state_operation(
            admission.request_key,
            op_id=2,
            parent=root_parent(admission),
            feature=operation.outputs[0],
            sample_continuation=False,
            max_tokens=2,
        )
        batch = execution_batch(
            step_id=2,
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
            execution_batch(
                step_id=3,
                controls=(Release(admission.request_key, operation.op_id),),
            )
        )
    finally:
        producer.close()
        consumer.close()


def test_cross_stage_device_product_transfer_preserves_generation_and_value() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = und_admission(77, block_ids=(0,))
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
                execution_batch(
                    step_id=1,
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
            work=Work("transfer", TransferMode.PRODUCT.value),
            route=0,
            domain=Domain.PREFILL,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(transferred,),
            control_seq=commit.control_seq,
        )
        transfer_report = finalized_report(
            producer.execute(
                execution_batch(
                    step_id=2,
                    operations=(transfer,),
                    controls=(commit,),
                )
            )
        )
        assert len(transfer_report.products) == 1
        payload = transfer_report.products[0]
        assert payload.product == transferred
        kind, descriptor, producer_digest = decode_transfer_descriptor(payload.payload)
        assert kind == "device_product"
        assert descriptor["generation"] == transferred.generation
        assert producer_digest == transfer.plan_digest

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
            work=consume.work,
            route=consume.route,
            domain=consume.domain,
            bounds=consume.bounds,
            inputs=(*consume.inputs, transferred),
            outputs=consume.outputs,
            kv_capacity_pages=consume.kv_capacity_pages,
            predicate=transferred,
            rng=consume.rng,
            control_seq=consume.control_seq,
        )
        prepared = consumer.prepare_execute(
            execution_batch(
                step_id=3,
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
    generation = gen_admission(79, ImageParams(steps=1, height=16, width=16, seed=31))
    admission = Admission.create(
        generation.request_key,
        request_pool_idx=generation.request_pool_idx,
        und=UndAdmission(),
        gen_admission=generation.gen_admission,
    )
    bind_request_placement(
        admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        page_ids=(1,),
    )
    try:
        conditioning = _publish_conditioning(producer, admission, op_id=1, step_id=1)
        transition, _latent = gen_transition_operation(
            admission.request_key,
            op_id=2,
            parent=root_parent(admission),
            conditioning=conditioning,
            seed=31,
        )
        transitioned = finalized_report(
            producer.execute(execution_batch(step_id=2, operations=(transition,)))
        )
        transition_commit = commit_for_completion(transition, transitioned)
        source = next(
            output for output in transition.outputs if output.kind is ProductKind.COMPLETION
        )
        transferred = replace(source, producer_op_id=3, generation=903)
        transfer = Operation.registered(
            request_key=admission.request_key,
            op_id=3,
            parent=transition_commit.selected,
            work=Work("transfer", TransferMode.PRODUCT.value),
            route=0,
            domain=Domain.FLOW,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(transferred,),
            control_seq=transition_commit.control_seq,
        )
        transfer_report = finalized_report(
            producer.execute(
                execution_batch(
                    step_id=3,
                    operations=(transfer,),
                    controls=(transition_commit,),
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
            execution_batch(
                step_id=4,
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
    admission = gen_admission(75, ImageParams(steps=1, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(producer, admission, op_id=1, step_id=1)
    initial_latent, transition_commit = _transition_generation(
        producer,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        step_id=2,
    )
    flow, final_latent = flow_operation(
        admission.request_key,
        op_id=3,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=1,
        control_seq=transition_commit.control_seq,
    )
    produced = producer.execute(
        execution_batch(
            step_id=3,
            operations=(flow,),
            controls=(transition_commit,),
        )
    )
    deadline = time.monotonic() + 5.0
    while not completion_report_ready(produced) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert completion_report_ready(produced)
    produced = finalize_completion_report(produced)
    transferred = tuple(product for product in produced.products if product.product == final_latent)
    assert len(transferred) == 1
    assert transferred[0].payload.startswith(TRANSFER_DESCRIPTOR_PREFIX)

    flow_commit = commit_for_completion(flow, produced)
    source_artifact = _materialized_artifact(
        producer,
        admission,
        final_latent,
        flow_commit,
        op_id=4,
        step_id=4,
    )
    materialize = materialize_operation(
        admission.request_key,
        op_id=4,
        parent=root_parent(admission),
        latent=final_latent,
    )
    batch = execution_batch(
        step_id=4,
        admissions=(admission,),
        operations=(materialize,),
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
    while not completion_report_ready(received) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert completion_report_ready(received)
    received = finalize_completion_report(received)
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
    admission = und_admission(3, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = worker.execute(
        execution_batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    )
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
    report = worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(encode,),
            controls=(commit,),
            input_products=(encode_input,),
        )
    )
    completion = report.completions[0]
    assert completion.logical_lengths.kv_visible_len == session_kv_before
    assert completion.logical_lengths.token_len == 2
    assert completion.selected_point == 0
    assert completion.product_generations
    assert completion.product_generations[0] == handle
    assert completion.product_generations[0] != 0


def test_generated_feedback_commits_absolute_visual_token_state():
    worker = execution_worker()
    understanding = und_admission(6, block_ids=(0,))
    admission = Admission.create(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        und=understanding.und,
        gen_admission=GenAdmission(
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
        execution_batch(
            step_id=1,
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
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(publication,),
            controls=(first_commit,),
        )
    )
    initial_latent, transition_commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=3,
        parent=first_commit.selected,
        step_id=3,
        control_seq=first_commit.control_seq,
    )
    flow, completed_latent = flow_operation(
        admission.request_key,
        op_id=4,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=initial_latent,
        steps=2,
        control_seq=transition_commit.control_seq,
    )
    flow_report = worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(flow,),
            controls=(transition_commit,),
            input_products=(),
        )
    )
    commit = commit_for_completion(flow, flow_report)
    materialize = materialize_operation(
        admission.request_key,
        op_id=5,
        parent=commit.selected,
        latent=completed_latent,
        feedback_source=True,
        control_seq=commit.control_seq,
    )
    materialize_report = worker.execute(
        execution_batch(
            step_id=5,
            admissions=(),
            operations=(materialize,),
            controls=(commit,),
            input_products=(),
        )
    )
    deadline = time.monotonic() + 5.0
    while not completion_report_ready(materialize_report) and time.monotonic() < deadline:
        time.sleep(0.001)
    assert completion_report_ready(materialize_report)
    materialize_report = finalize_completion_report(materialize_report)
    assert materialize_report.completions[0].logical_lengths.kv_visible_len == 2
    assert materialize_report.completions[0].selected_point == 0

    encode, _ = encode_operation(
        admission.request_key,
        op_id=6,
        parent=commit.selected,
        image_base64=None,
        encoder_handle=10,
        source_product=materialize.outputs[1],
        control_seq=commit.control_seq,
    )
    encode_report = worker.execute(
        execution_batch(step_id=6, admissions=(), operations=(encode,), input_products=())
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
    finish_product = ProductRef(
        request_key=admission.request_key,
        producer_op_id=state.op_id,
        output_index=2,
        generation=state.op_id * 8 + 8,
        kind=ProductKind.FINISH,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    transition_product = ProductRef(
        request_key=admission.request_key,
        producer_op_id=state.op_id,
        output_index=4,
        generation=state.op_id * 8 + 9,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    state = Operation.registered(
        request_key=state.request_key,
        op_id=state.op_id,
        parent=state.parent,
        work=state.work,
        route=state.route,
        domain=state.domain,
        bounds=state.bounds,
        inputs=(*state.inputs, sampling_product),
        outputs=(*state.outputs, finish_product, transition_product),
        kv_capacity_pages=state.kv_capacity_pages,
        predicate=state.predicate,
        rng=state.rng,
        control_seq=state.control_seq,
    )
    report = worker.execute(
        execution_batch(
            step_id=7,
            admissions=(),
            operations=(state,),
            input_products=(ProductPayload(sampling_product, sampling_bytes),),
        )
    )

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
        execution_batch(
            step_id=8,
            operations=(publication,),
            controls=(feedback_commit,),
        )
    )
    next_latent, _next_transition_commit = _transition_generation(
        worker,
        admission,
        next_conditioning,
        op_id=9,
        parent=feedback_commit.selected,
        step_id=9,
        control_seq=feedback_commit.control_seq,
        image_index=2,
    )
    next_flow, next_completed_latent = flow_operation(
        admission.request_key,
        op_id=10,
        parent=_next_transition_commit.selected,
        conditioning=next_conditioning,
        latent=next_latent,
        steps=2,
        control_seq=_next_transition_commit.control_seq,
    )
    next_report = worker.execute(
        execution_batch(
            step_id=10,
            operations=(next_flow,),
            controls=(_next_transition_commit,),
        )
    )
    assert next_report.completions[0].status is OpStatus.OK
    assert next_report.completions[0].logical_lengths.latent_len == 2
    next_commit = commit_for_completion(next_flow, next_report)
    assert base64.b64decode(
        _materialized_artifact(
            worker,
            admission,
            next_completed_latent,
            next_commit,
            op_id=11,
            step_id=11,
        ).decode("ascii"),
        validate=True,
    ).startswith(_PNG_MAGIC)

    artifacts = [p for p in materialize_report.products if p.product.kind is ProductKind.ARTIFACT]
    assert len(artifacts) == 1
    # The Artifact product carries the base64 PNG string as bytes: the scheduler
    # recovers it with String::from_utf8 and hands it to validate_png_artifact,
    # which base64-decodes it and checks the PNG dimensions. Mirror that contract.
    png_b64 = artifacts[0].payload
    png_bytes = base64.b64decode(png_b64.decode("ascii"), validate=True)
    assert png_bytes[:8] == _PNG_MAGIC
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (16, 16)
