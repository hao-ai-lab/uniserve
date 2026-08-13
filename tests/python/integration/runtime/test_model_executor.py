"""Serial-oracle behavior at the canonical ModelExecutor boundary."""

from __future__ import annotations

import base64
import io
import time
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from PIL import Image

from tests.python.fixtures.depth_one import (
    bind_request_placement,
    commit_resolved,
    encode_operation,
    execution_batch,
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
    CacheGroupPlacement,
    DeviceDim,
    DevicePoint,
    DType,
    ErrorCode,
    GenAdmission,
    ImageParams,
    KvPlacement,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductRef,
    RecoveryPlacement,
    Release,
    ShapeBound,
    StorageClass,
    TokenMode,
    VersionRef,
)
from uniserve_worker.execution.executor import (
    completion_report_ready,
    finalize_completion_report,
)
from uniserve_worker.execution.forward_batch import (
    ForwardBatch,
    ForwardOutput,
    PackedAttentionPlan,
    PagedDecodePlan,
)
from uniserve_worker.foundation.errors import ErrorCode as HostErrorCode
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.completion_store import CompletionArena
from uniserve_worker.runtime.transfer import TRANSFER_DESCRIPTOR_PREFIX
from uniserve_worker.server.stub import StubModel, _next_token

pytestmark = pytest.mark.integration

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _ObservedModel(StubModel):
    def __init__(self) -> None:
        super().__init__()
        self.flow_inputs: list[torch.Tensor] = []
        self.token_positions: list[tuple[int, ...]] = []
        self.attention_plans: list[object] = []
        self.fault: str | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        self.attention_plans.append(batch.attention)
        self.flow_inputs.extend(value.detach().clone() for value in batch.flow_latents)
        offset = 0
        for count in batch.query_lens:
            row_positions = positions[..., offset : offset + count]
            self.token_positions.append(
                tuple(int(value) for value in row_positions.reshape(-1).tolist())
            )
            offset += count
        hidden = super().forward(input_ids, positions, batch)
        if self.fault == "raise":
            raise RuntimeError("injected neural failure")
        return hidden

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        output = super().project(hidden, batch)
        if self.fault == "misaligned":
            return ForwardOutput(output.values[:-1])
        return output


class _KvRecoveryModel(_ObservedModel):
    def __init__(self) -> None:
        super().__init__()
        self.observed_prefixes: list[tuple[float, ...]] = []
        self.observed_pages: list[tuple[int, ...]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        token_rows = batch.token_row_indices
        plan = batch.attention
        if token_rows and batch.kv.base_lens[0] > 0:
            if not isinstance(plan, PagedDecodePlan):
                raise TypeError("KV recovery probe requires paged decode attention")
            keys, _values = batch.kv.layer_kv(0)
            pages = tuple(int(value) for value in plan.block_table[0].tolist())
            prefix = tuple(
                float(
                    keys[
                        pages[position // batch.kv.block_size],
                        position % batch.kv.block_size,
                        0,
                        0,
                    ].item()
                )
                for position in range(int(batch.kv.base_lens[0]))
            )
            self.observed_pages.append(pages)
            self.observed_prefixes.append(prefix)

        output = super().forward(input_ids, positions, batch)
        if not token_rows:
            return output
        token_count = sum(batch.query_lens)
        values = torch.arange(
            1,
            token_count + 1,
            device=positions.device,
            dtype=torch.bfloat16,
        ).view(token_count, 1, 1)
        if isinstance(plan, PagedDecodePlan):
            batch.kv.append(0, values.unsqueeze(1), values.unsqueeze(1))
        elif isinstance(plan, PackedAttentionPlan):
            batch.kv.append_packed(
                0,
                values,
                values,
                page_ids=plan.write_page_ids,
                page_offsets=plan.write_page_offsets,
                token_indices=plan.write_token_indices,
            )
        else:
            raise TypeError("KV recovery probe requires paged attention")
        return output


class _SeparatePhaseModel(_ObservedModel):
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
    worker.execute(execution_batch(step_id=step_id, operations=(transition,)))
    return latent, commit_resolved(worker.sessions.get(admission.request_key.session_id))


def test_extend_then_decode_commit_the_serial_oracle_tokens():
    model = _ObservedModel()
    worker = execution_worker(model)
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

    assert extended.completions[0].committed_tokens == (_next_token(4),)
    assert extended.completions[0].logical_lengths.kv_visible_len == 2
    assert extended.completions[0].logical_lengths.token_len == 2

    first_token = extended.completions[0].committed_tokens[0]
    commit = commit_resolved(worker.sessions.get(1))
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
    assert model.token_positions == [(0, 1), (2,)]


def test_prefix_reuse_continues_from_the_admitted_logical_position():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = und_admission(8, block_ids=(0,), prefix_len=2)
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(4,),
    )

    worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )

    session = worker.sessions.get(8)
    assert model.token_positions == [(2,)]
    assert session.logical_position == 3
    root_runtime = session.runtime_for(root_parent(admission))
    assert root_runtime is not None
    assert root_runtime.logical_position == 2


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
    mixed_model = _ObservedModel()
    mixed = execution_worker(mixed_model)
    sequence_admission = und_admission(1, block_ids=(0,))
    flow_admission = gen_admission(2, ImageParams(steps=1, height=16, width=16, seed=29))
    sequence, sequence_input = token_operation(
        sequence_admission.request_key,
        op_id=11,
        parent=root_parent(sequence_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    mixed_conditioning = _publish_conditioning(mixed, flow_admission, op_id=10, step_id=1)
    mixed_latent, mixed_transition_commit = _transition_generation(
        mixed,
        flow_admission,
        mixed_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        step_id=2,
    )
    flow, _mixed_output_latent = flow_operation(
        flow_admission.request_key,
        op_id=12,
        parent=mixed_transition_commit.selected,
        conditioning=mixed_conditioning,
        latent=mixed_latent,
        steps=1,
        control_seq=mixed_transition_commit.control_seq,
    )

    mixed_result = mixed.execute(
        execution_batch(
            step_id=3,
            admissions=(sequence_admission,),
            operations=(sequence, flow),
            controls=(mixed_transition_commit,),
            input_products=(sequence_input,),
        )
    )

    split_model = _ObservedModel()
    split = execution_worker(split_model)
    split_sequence, split_sequence_input = token_operation(
        sequence_admission.request_key,
        op_id=11,
        parent=root_parent(sequence_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    split_conditioning = _publish_conditioning(split, flow_admission, op_id=10, step_id=1)
    split_latent, split_transition_commit = _transition_generation(
        split,
        flow_admission,
        split_conditioning,
        op_id=11,
        parent=root_parent(flow_admission),
        step_id=2,
    )
    split_flow, _split_output_latent = flow_operation(
        flow_admission.request_key,
        op_id=12,
        parent=split_transition_commit.selected,
        conditioning=split_conditioning,
        latent=split_latent,
        steps=1,
        control_seq=split_transition_commit.control_seq,
    )
    sequence_result = split.execute(
        execution_batch(
            step_id=3,
            admissions=(sequence_admission,),
            operations=(split_sequence,),
            input_products=(split_sequence_input,),
        )
    )
    flow_result = split.execute(
        execution_batch(
            step_id=4,
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
    torch.testing.assert_close(
        mixed.latents.require(mixed.sessions.get(2).latent_product).value,
        split.latents.require(split.sessions.get(2).latent_product).value,
        rtol=0,
        atol=0,
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
    model = _ObservedModel()
    worker = execution_worker(model)
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
    worker.execute(
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
        session_id = admission.request_key.session_id
        commit = commit_resolved(worker.sessions.get(session_id))
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

    plan = model.attention_plans[-1]
    assert isinstance(plan, PagedDecodePlan)
    assert tuple(plan.query_lens.tolist()) == (1, 1)
    assert tuple(record.committed_tokens for record in decoded.completions) == tuple(
        (_next_token(_next_token(token)),) for token in last_tokens
    )


def test_image_capability_preserves_the_pure_token_decode_plan():
    model = _ObservedModel()
    worker = execution_worker(model)
    understanding = und_admission(43, block_ids=(0,))
    admission = Admission.create(
        understanding.request_key,
        request_pool_idx=understanding.request_pool_idx,
        und=understanding.und,
        gen_admission=GenAdmission(ImageParams(steps=1, height=16, width=16, seed=29)),
    )
    prefill, prefill_input = token_operation(
        admission.request_key,
        op_id=70,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    extended = worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(prefill,),
            input_products=(prefill_input,),
        )
    )
    commit = commit_resolved(worker.sessions.get(43))
    decode, decode_input = token_operation(
        admission.request_key,
        op_id=71,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(extended.completions[0].committed_tokens[0],),
        control_seq=commit.control_seq,
    )

    worker.execute(
        execution_batch(
            step_id=2,
            admissions=(),
            operations=(decode,),
            controls=(commit,),
            input_products=(decode_input,),
        )
    )

    assert isinstance(model.attention_plans[-1], PagedDecodePlan)


def test_replay_identity_is_idempotent_and_conflicts_are_atomic():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = und_admission(3, block_ids=(3,))
    operation, payload = token_operation(
        admission.request_key,
        op_id=21,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(8, 9),
    )
    batch = execution_batch(
        step_id=7, admissions=(admission,), operations=(operation,), input_products=(payload,)
    )

    first = worker.execute(batch)
    replayed = worker.execute(batch)
    committed = deepcopy(worker.sessions.get(3))

    assert replayed.completions == first.completions
    assert worker.sessions.get(3).version == 1

    # Token values ride the input payload, not the plan identity, so a genuine
    # op-id conflict must differ in the plan itself: here a wider bounded span.
    conflicting, conflicting_input = token_operation(
        admission.request_key,
        op_id=21,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(8, 9, 10),
    )
    with pytest.raises(Exception, match="conflicts with its committed digest"):
        worker.execute(
            execution_batch(
                step_id=8,
                admissions=(),
                operations=(conflicting,),
                input_products=(conflicting_input,),
            )
        )

    assert worker.sessions.get(3) == committed


def test_output_validation_failure_is_terminal_and_rolls_back_every_authority():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = und_admission(4, block_ids=(4,))
    initial, initial_input = token_operation(
        admission.request_key,
        op_id=31,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(12, 13),
    )
    worker.execute(
        execution_batch(
            step_id=11,
            admissions=(admission,),
            operations=(initial,),
            input_products=(initial_input,),
        )
    )
    commit = commit_resolved(worker.sessions.get(4))
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
    model.fault = "misaligned"
    failed = worker.execute(retry_batch)

    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.COMPUTE_ERROR

    model.fault = None
    replayed = worker.execute(retry_batch)
    assert replayed.completions == failed.completions
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


def test_failed_flow_reclaims_state_and_a_new_operation_repeats_the_same_input():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = gen_admission(5, ImageParams(steps=2, height=16, width=16, seed=29))
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
    batch = execution_batch(
        step_id=15,
        admissions=(),
        operations=(flow,),
        input_products=(),
    )
    model.fault = "raise"

    failed = worker.execute(batch)

    first_input = model.flow_inputs[-1]
    assert failed.completions[0].status is OpStatus.ERROR
    assert failed.completions[0].error_code is ErrorCode.COMPUTE_ERROR

    model.fault = None
    replacement, _replacement_latent = flow_operation(
        admission.request_key,
        op_id=43,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=2,
        control_seq=transition_commit.control_seq,
    )
    worker.execute(execution_batch(step_id=16, operations=(replacement,)))

    torch.testing.assert_close(model.flow_inputs[-1], first_input, rtol=0, atol=0)
    assert worker.sessions.get(5).version == 1
    assert worker.latents.require(worker.sessions.get(5).latent_product).step == 2


def test_mixed_partition_descriptor_failure_does_not_rollback_the_other_domain():
    model = _ObservedModel()
    worker = execution_worker(model)
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
    sequence, sequence_input = token_operation(
        sequence_admission.request_key,
        op_id=3,
        parent=root_parent(sequence_admission),
        mode=TokenMode.EXTEND,
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
            step_id=3,
            admissions=(sequence_admission,),
            operations=(sequence, flow),
            controls=(transition_commit,),
            input_products=(sequence_input,),
        )
    )

    by_request = {record.request_key.session_id: record for record in report.completions}
    assert by_request[61].status is OpStatus.OK
    assert by_request[62].status is OpStatus.ERROR
    assert by_request[62].error_code is ErrorCode.INVALID_OPERATION
    assert worker.sessions.get(61).version == 1
    assert worker.sessions.get(62).version == transition_commit.selected.point.point_index


def test_mixed_partition_completion_pressure_is_contained_to_one_domain():
    model = _ObservedModel()
    worker = execution_worker(model)
    sequence_admission = und_admission(63, block_ids=(0,))
    generation_admission = gen_admission(
        64,
        ImageParams(steps=1, height=16, width=16, seed=31),
    )
    conditioning = _publish_conditioning(worker, generation_admission, op_id=1, step_id=1)
    latent, transition_commit = _transition_generation(
        worker,
        generation_admission,
        conditioning,
        op_id=2,
        parent=root_parent(generation_admission),
        step_id=2,
        seed=31,
    )
    sequence, sequence_input = token_operation(
        sequence_admission.request_key,
        op_id=3,
        parent=root_parent(sequence_admission),
        mode=TokenMode.EXTEND,
        tokens=(9, 10),
    )
    flow, _output_latent = flow_operation(
        generation_admission.request_key,
        op_id=3,
        parent=transition_commit.selected,
        conditioning=conditioning,
        latent=latent,
        steps=1,
        control_seq=transition_commit.control_seq,
    )
    worker.executor._completions = CompletionArena(
        depth=2,
        token_capacity=4,
        total_token_capacity=4,
    )

    report = worker.execute(
        execution_batch(
            step_id=3,
            admissions=(sequence_admission,),
            operations=(sequence, flow),
            controls=(transition_commit,),
            input_products=(sequence_input,),
        )
    )

    by_request = {record.request_key.session_id: record for record in report.completions}
    assert by_request[63].status is OpStatus.OK
    assert by_request[64].status is OpStatus.ERROR
    assert by_request[64].error_code is ErrorCode.RESOURCE_EXHAUSTED
    assert worker.sessions.get(63).version == 1
    assert worker.sessions.get(64).latent_product == latent


def test_initial_flow_noise_is_stable_across_operation_schedules():
    admission = gen_admission(5, ImageParams(steps=1, height=16, width=16, seed=29))
    observed: list[torch.Tensor] = []
    for op_id in (41, 109):
        model = _ObservedModel()
        worker = execution_worker(model)
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
        flow, _output_latent = flow_operation(
            admission.request_key,
            op_id=op_id + 1,
            parent=transition_commit.selected,
            conditioning=conditioning,
            latent=latent,
            steps=1,
            control_seq=transition_commit.control_seq,
        )
        worker.execute(
            execution_batch(
                step_id=3,
                admissions=(),
                operations=(flow,),
                controls=(transition_commit,),
                input_products=(),
            )
        )
        observed.append(model.flow_inputs[0])

    torch.testing.assert_close(observed[1], observed[0], rtol=0, atol=0)


def test_decode_grows_logical_capacity_across_a_kv_page_boundary():
    # A small page forces the decode chain to cross a registration boundary.
    block_size = 4
    worker = execution_worker(_ObservedModel(), block_size=block_size)
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
        commit = commit_resolved(worker.sessions.get(1))
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
    worker = execution_worker(_ObservedModel())
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
    commit = commit_resolved(worker.sessions.get(2))
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


def test_exact_latent_chain_reclaims_committed_ancestors_and_rejects_a_stale_reference():
    worker = execution_worker(_ObservedModel())
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
        commit = commit_resolved(worker.sessions.get(admission.request_key.session_id))

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


def test_snapshot_restore_rebinds_the_exact_committed_latent_product(tmp_path) -> None:
    snapshot_dir = str(tmp_path / "worker-state")
    worker = execution_worker(_ObservedModel(), snapshot_dir=snapshot_dir)
    admission = gen_admission(72, ImageParams(steps=2, height=16, width=16, seed=29))
    conditioning = _publish_conditioning(worker, admission, op_id=1, step_id=1)
    latent, commit = _transition_generation(
        worker,
        admission,
        conditioning,
        op_id=2,
        parent=root_parent(admission),
        step_id=2,
    )
    worker.execute(execution_batch(step_id=3, controls=(commit,)))
    placement = RecoveryPlacement(
        request_key=admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        cache_groups=(CacheGroupPlacement(group_id=0, page_ids=(), length=0),),
    )
    reference = worker.snapshot_session(placement)
    worker.close()

    restored = execution_worker(
        _ObservedModel(),
        snapshot_dir=snapshot_dir,
    )
    restored.restore_session(reference, placement)
    continuation, _successor = flow_operation(
        admission.request_key,
        op_id=3,
        parent=reference.version,
        conditioning=conditioning,
        latent=latent,
        steps=1,
        control_seq=commit.control_seq,
    )

    report = restored.execute(execution_batch(step_id=4, operations=(continuation,)))

    assert report.completions[0].status is OpStatus.OK
    assert report.completions[0].logical_lengths.latent_len == 1
    restored.close()


def test_snapshot_restore_rebinds_committed_kv_to_scheduler_placement(tmp_path) -> None:
    snapshot_dir = str(tmp_path / "worker-state")
    source_model = _KvRecoveryModel()
    worker = execution_worker(source_model, snapshot_dir=snapshot_dir)
    admission = und_admission(74, block_ids=(0,))
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
    commit = commit_resolved(worker.sessions.get(admission.request_key.session_id))
    worker.execute(execution_batch(step_id=2, controls=(commit,)))
    source_placement = RecoveryPlacement(
        request_key=admission.request_key,
        request_pool_idx=admission.request_pool_idx,
        cache_groups=(CacheGroupPlacement(group_id=0, page_ids=(1,), length=2),),
    )
    reference = worker.snapshot_session(source_placement)
    worker.close()

    destination = RecoveryPlacement(
        request_key=admission.request_key,
        request_pool_idx=admission.request_pool_idx + 1,
        cache_groups=(CacheGroupPlacement(group_id=0, page_ids=(2,), length=2),),
    )
    restored_model = _KvRecoveryModel()
    restored = execution_worker(restored_model, snapshot_dir=snapshot_dir)
    restored.restore_session(reference, destination)
    bind_request_placement(
        admission.request_key,
        request_pool_idx=destination.request_pool_idx,
        page_ids=(2,),
    )
    decode, decode_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=reference.version,
        mode=TokenMode.DECODE,
        tokens=(extended.completions[0].committed_tokens[0],),
        control_seq=commit.control_seq,
    )

    report = restored.execute(
        execution_batch(
            step_id=3,
            operations=(decode,),
            input_products=(decode_input,),
        )
    )

    assert report.completions[0].status is OpStatus.OK
    assert report.completions[0].logical_lengths.kv_visible_len == 3
    assert restored_model.observed_pages == [(2,)]
    assert restored_model.observed_prefixes == [(1.0, 2.0)]
    restored.close()


def test_cross_stage_feature_transfer_rebinds_exact_product_without_request_thread_wait() -> None:
    producer = execution_worker(_ObservedModel(), transfer_backend="shm")
    consumer = execution_worker(_ObservedModel(), transfer_backend="shm")
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


def test_encode_publishes_an_immutable_feature_without_advancing_state():
    worker = execution_worker(_ObservedModel())
    admission = und_admission(3, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        execution_batch(
            step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,)
        )
    )
    session_kv_before = 2

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    handle = 0xABCDEF
    commit = commit_resolved(worker.sessions.get(3))
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
    worker = execution_worker(_ObservedModel())
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
    worker.execute(
        execution_batch(
            step_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )
    first_commit = commit_resolved(worker.sessions.get(6))
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
    worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(flow,),
            controls=(transition_commit,),
            input_products=(),
        )
    )
    assert worker.sessions.get(6).version == 1

    commit = commit_resolved(worker.sessions.get(6))
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
    assert worker.sessions.get(6).version == 1
    assert worker.sessions.get(6).logical_position == 2

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
    assert worker.sessions.get(6).version == 1
    assert worker.sessions.get(6).logical_position == 2

    state = visual_state_operation(
        admission.request_key,
        op_id=7,
        parent=commit.selected,
        feature=encode.outputs[0],
        sample_continuation=True,
        max_tokens=2,
        control_seq=commit.control_seq,
    )
    report = worker.execute(
        execution_batch(step_id=7, admissions=(), operations=(state,), input_products=())
    )

    completion = report.completions[0]
    assert completion.committed_tokens == (_next_token(1007),)
    assert completion.token_span.base == 2
    assert completion.token_span.len == 1
    assert completion.logical_lengths.token_len == 4
    assert completion.logical_lengths.kv_visible_len == 4
    assert completion.selected_point == 1

    feedback_commit = commit_resolved(worker.sessions.get(6))
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
    assert worker.sessions.get(6).latent_product == next_latent
    assert worker.latents.require(next_latent).step == 0

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
