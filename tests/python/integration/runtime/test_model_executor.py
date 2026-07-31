"""Serial-oracle behavior at the canonical ModelExecutor boundary."""

from __future__ import annotations

import base64
import io
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.python.fixtures.depth_one import (
    commit_resolved,
    encode_operation,
    flow_operation,
    gen_admission,
    materialize_operation,
    root_parent,
    token_operation,
    und_admission,
    visual_state_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
    GenAdmission,
    ImageParams,
    ProductKind,
    TokenMode,
)
from uniserve_worker.forward import (
    FlowRow,
    ForwardBatch,
    ForwardOutput,
    PagedDecodePlan,
    TokenRow,
)
from uniserve_worker.server.stub import _next_token
from uniserve_worker.spec import (
    FeatureInjectionSpec,
    FeatureLayout,
    OperationSpec,
    OperationStageCondition,
    OperationStagePurpose,
    OperationStageSpec,
    OperationType,
    PositionLayout,
    RouteRowKind,
)

pytestmark = pytest.mark.integration

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _ObservedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from uniserve_worker.server.stub import StubModel

        self.neural = StubModel()
        self.spec = self.neural.spec
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.flow_inputs: list[torch.Tensor] = []
        self.token_positions: list[tuple[int, ...]] = []
        self.attention_plans: list[object] = []
        self.fault: str | None = None

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append((str(batch.route), tuple(type(row).__name__ for row in batch.rows)))
        self.attention_plans.append(batch.context.attention)
        self.flow_inputs.extend(
            row.latent.detach().clone() for row in batch.rows if isinstance(row, FlowRow)
        )
        self.token_positions.extend(
            tuple(int(value) for value in row.positions.reshape(-1).tolist())
            for row in batch.rows
            if isinstance(row, TokenRow)
        )
        output = self.neural(batch)
        if self.fault == "raise":
            raise RuntimeError("injected neural failure")
        if self.fault == "misaligned":
            return ForwardOutput(output.rows[:-1])
        return output


class _RetainedImageStateModel(_ObservedModel):
    def __init__(self) -> None:
        super().__init__()
        images = self.spec.inputs.images
        assert images is not None
        inputs = replace(
            self.spec.inputs,
            images=replace(
                images,
                feature_injection=FeatureInjectionSpec(
                    layout=FeatureLayout.DIRECT,
                    positions=PositionLayout.TEMPORAL_SPATIAL,
                    end_token_id=1007,
                ),
            ),
        )
        operations = tuple(
            OperationSpec(
                OperationType.ENCODE_VISION,
                (
                    OperationStageSpec(
                        "encode",
                        RouteRowKind.ENCODE,
                        OperationStagePurpose.PRIMARY,
                    ),
                    OperationStageSpec(
                        "stub",
                        RouteRowKind.TOKEN,
                        OperationStagePurpose.STATE,
                        OperationStageCondition.RETAIN_IMAGE,
                    ),
                ),
            )
            if operation.kind is OperationType.ENCODE_VISION
            else operation
            for operation in self.spec.operations
        )
        self.spec = replace(self.spec, operations=operations, inputs=inputs)


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
        Batch(step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,))
    )

    assert extended.completions[0].committed_tokens == (_next_token(4),)
    assert worker.kv.get(1).length == 2
    assert worker.sessions.get(1).version == 1
    assert worker.sessions.get(1).logical_position == 2

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
        Batch(
            step_id=2,
            admissions=(),
            operations=(decode,),
            controls=(commit,),
            input_products=(decode_input,),
        )
    )

    assert decoded.completions[0].committed_tokens == (_next_token(first_token),)
    assert worker.kv.get(1).length == 3
    assert worker.sessions.get(1).version == 2
    assert worker.sessions.get(1).logical_position == 3
    assert model.token_positions == [(0, 1), (2,)]


def test_mixed_token_and_flow_match_homogeneous_projection_in_one_forward():
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
    flow = flow_operation(flow_admission.request_key, op_id=12, parent=root_parent(flow_admission), steps=1)

    mixed_result = mixed.execute(
        Batch(
            step_id=1,
            admissions=(sequence_admission, flow_admission),
            operations=(sequence, flow),
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
    split_flow = flow_operation(
        flow_admission.request_key, op_id=12, parent=root_parent(flow_admission), steps=1
    )
    sequence_result = split.execute(
        Batch(
            step_id=1,
            admissions=(sequence_admission,),
            operations=(split_sequence,),
            input_products=(split_sequence_input,),
        )
    )
    flow_result = split.execute(
        Batch(step_id=2, admissions=(flow_admission,), operations=(split_flow,), input_products=())
    )

    assert (
        mixed_result.completions[0].committed_tokens
        == sequence_result.completions[0].committed_tokens
    )
    assert mixed_result.completions[0].semantic_digest == sequence_result.completions[0].semantic_digest
    assert mixed_result.completions[1].semantic_digest == flow_result.completions[0].semantic_digest
    torch.testing.assert_close(
        mixed.latents.require(mixed.sessions.get(2).latent_handle).value,
        split.latents.require(split.sessions.get(2).latent_handle).value,
        rtol=0,
        atol=0,
    )
    assert len(mixed_model.calls) == 1
    assert set(mixed_model.calls[0][1]) == {"TokenRow", "FlowRow"}
    assert len(split_model.calls) == 2


def test_token_decode_uses_the_homogeneous_paged_plan():
    model = _ObservedModel()
    worker = execution_worker(model)
    admissions = (und_admission(41, block_ids=(0,)), und_admission(42, block_ids=(1,)))
    prefill_ops = []
    prefill_inputs = []
    for index, admission in enumerate(admissions):
        operation, payload = token_operation(
            admission.request_key,
            op_id=50 + index,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(3, 4),
        )
        prefill_ops.append(operation)
        prefill_inputs.append(payload)
    worker.execute(
        Batch(
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
            op_id=60 + index,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(_next_token(4),),
            control_seq=commit.control_seq,
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
        commits.append(commit)
    decoded = worker.execute(
        Batch(
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
    assert decoded.completions[0].committed_tokens == (_next_token(_next_token(4)),)
    assert decoded.completions[1].committed_tokens == (_next_token(_next_token(4)),)


def test_image_capable_pure_token_decode_uses_paged_attention():
    model = _ObservedModel()
    worker = execution_worker(model)
    understanding = und_admission(43, block_ids=(0,))
    admission = Admission.create(
        understanding.request_key,
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
        Batch(
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
        Batch(
            step_id=2,
            admissions=(),
            operations=(decode,),
            controls=(commit,),
            input_products=(decode_input,),
        )
    )

    plan = model.attention_plans[-1]
    assert isinstance(plan, PagedDecodePlan)
    assert plan.query_lens_cpu == (1,)


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
    batch = Batch(step_id=7, admissions=(admission,), operations=(operation,), input_products=(payload,))

    first = worker.execute(batch)
    replayed = worker.execute(batch)
    committed = deepcopy(worker.sessions.get(3))

    assert replayed.completions == first.completions
    assert len(model.calls) == 1
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
            Batch(step_id=8, admissions=(), operations=(conflicting,), input_products=(conflicting_input,))
        )

    assert worker.sessions.get(3) == committed
    assert len(model.calls) == 1


def test_output_validation_failure_rolls_back_every_authority():
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
        Batch(step_id=11, admissions=(admission,), operations=(initial,), input_products=(initial_input,))
    )
    commit = commit_resolved(worker.sessions.get(4))
    worker.execute(
        Batch(step_id=12, admissions=(), operations=(), controls=(commit,))
    )
    committed = deepcopy(worker.sessions.get(4))
    committed_length = worker.kv.get(4).length

    retry, retry_input = token_operation(
        admission.request_key,
        op_id=32,
        parent=commit.selected,
        mode=TokenMode.DECODE,
        tokens=(_next_token(13),),
        control_seq=commit.control_seq,
    )
    retry_batch = Batch(step_id=13, admissions=(), operations=(retry,), input_products=(retry_input,))
    model.fault = "misaligned"
    with pytest.raises(Exception):
        worker.execute(retry_batch)

    assert worker.sessions.get(4) == committed
    assert worker.kv.get(4).length == committed_length

    model.fault = None
    result = worker.execute(retry_batch)
    assert result.completions[0].committed_tokens == (_next_token(_next_token(13)),)
    assert worker.sessions.get(4).version == 2
    assert worker.kv.get(4).length == committed_length + 1


def test_failed_first_flow_attempt_reclaims_state_and_retries_deterministically():
    model = _ObservedModel()
    worker = execution_worker(model)
    admission = gen_admission(5, ImageParams(steps=2, height=16, width=16, seed=29))
    flow = flow_operation(admission.request_key, op_id=41, parent=root_parent(admission), steps=2)
    batch = Batch(step_id=13, admissions=(admission,), operations=(flow,), input_products=())
    model.fault = "raise"

    with pytest.raises(Exception, match="injected neural failure"):
        worker.execute(batch)

    first_input = model.flow_inputs[-1]
    assert worker.sessions.peek(5) is None

    model.fault = None
    worker.execute(batch)

    torch.testing.assert_close(model.flow_inputs[-1], first_input, rtol=0, atol=0)
    assert worker.sessions.get(5).version == 1
    assert worker.latents.require(worker.sessions.get(5).latent_handle).step == 2


def test_decode_grows_the_block_lease_across_a_kv_page_boundary():
    # A small page (4 tokens/block) forces the decode chain to cross a block
    # boundary. Each step's `new_kv_blocks` grows the session lease before the
    # forward computes its page index; without that growth the attention plan
    # indexes past block_ids and IndexErrors at the boundary.
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
        Batch(step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,))
    )
    committed = list(extended.completions[0].committed_tokens)
    assert worker.kv.get(1).block_ids == [0]

    next_block = 1
    crossed = False
    for step in range(4):
        length = worker.kv.get(1).length
        blocks = tuple(worker.kv.get(1).block_ids)
        new_kv_blocks: tuple[int, ...] = ()
        if length // block_size >= len(blocks):
            new_kv_blocks = (next_block,)
            next_block += 1
            crossed = True
        commit = commit_resolved(worker.sessions.get(1))
        decode, decode_input = token_operation(
            admission.request_key,
            op_id=2 + step,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(committed[-1],),
            new_kv_blocks=new_kv_blocks,
            control_seq=commit.control_seq,
        )
        report = worker.execute(
            Batch(
                step_id=2 + step,
                admissions=(),
                operations=(decode,),
                controls=(commit,),
                input_products=(decode_input,),
            )
        )
        if new_kv_blocks:
            assert tuple(worker.kv.get(1).block_ids) == blocks + new_kv_blocks
        else:
            assert tuple(worker.kv.get(1).block_ids) == blocks
        committed.extend(report.completions[0].committed_tokens)

    assert crossed  # the chain actually crossed a page boundary
    assert worker.kv.get(1).block_ids == [0, 1]
    assert worker.kv.get(1).length == 6
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
    first = flow_operation(admission.request_key, op_id=1, parent=root_parent(admission), steps=1)
    first_report = worker.execute(
        Batch(step_id=1, admissions=(admission,), operations=(first,), input_products=())
    )
    commit = commit_resolved(worker.sessions.get(2))
    second = flow_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        steps=1,
        control_seq=commit.control_seq,
    )
    second_report = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(second,),
            controls=(commit,),
            input_products=(),
        )
    )

    assert first_report.completions[0].logical_lengths.latent_len == 1
    assert second_report.completions[0].logical_lengths.latent_len == 2


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
        Batch(step_id=1, admissions=(admission,), operations=(extend,), input_products=(extend_input,))
    )
    session_kv_before = worker.kv.get(3).length
    assert session_kv_before == 2

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
        Batch(
            step_id=2,
            admissions=(),
            operations=(encode,),
            controls=(commit,),
            input_products=(encode_input,),
        )
    )
    completion = report.completions[0]
    assert completion.logical_lengths.kv_visible_len == 0
    assert worker.kv.get(3).length == session_kv_before
    assert worker.sessions.get(3).logical_position == 2
    assert completion.selected_point == 1
    assert completion.product_generations
    assert completion.product_generations[0] == handle
    assert completion.product_generations[0] != 0


def test_generated_feedback_advances_state_only_in_visual_token_extend():
    worker = execution_worker(_RetainedImageStateModel())
    understanding = und_admission(6, block_ids=(0,))
    admission = Admission.create(
        understanding.request_key,
        und=understanding.und,
        gen_admission=GenAdmission(
            ImageParams(steps=2, height=16, width=16, seed=29, retain_images=True)
        ),
    )
    flow = flow_operation(admission.request_key, op_id=1, parent=root_parent(admission), steps=2)
    worker.execute(Batch(step_id=1, admissions=(admission,), operations=(flow,), input_products=()))
    assert worker.sessions.get(6).version == 1

    commit = commit_resolved(worker.sessions.get(6))
    materialize = materialize_operation(
        admission.request_key,
        op_id=2,
        parent=commit.selected,
        feedback_source=True,
        control_seq=commit.control_seq,
    )
    materialize_report = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(materialize,),
            controls=(commit,),
            input_products=(),
        )
    )
    assert materialize_report.completions[0].logical_lengths.kv_visible_len == 0
    assert materialize_report.completions[0].selected_point == 1
    assert worker.sessions.get(6).version == 1
    assert worker.sessions.get(6).logical_position == 0

    encode, _ = encode_operation(
        admission.request_key,
        op_id=3,
        parent=commit.selected,
        image_base64=None,
        encoder_handle=10,
        source_product=materialize.outputs[1],
        control_seq=commit.control_seq,
    )
    encode_report = worker.execute(
        Batch(step_id=3, admissions=(), operations=(encode,), input_products=())
    )
    assert encode_report.completions[0].logical_lengths.kv_visible_len == 0
    assert encode_report.completions[0].selected_point == 1
    assert worker.sessions.get(6).version == 1
    assert worker.sessions.get(6).logical_position == 0

    state = visual_state_operation(
        admission.request_key,
        op_id=4,
        parent=commit.selected,
        feature=encode.outputs[0],
        sample_continuation=True,
        control_seq=commit.control_seq,
    )
    report = worker.execute(Batch(step_id=4, admissions=(), operations=(state,), input_products=()))

    completion = report.completions[0]
    assert completion.committed_tokens == (_next_token(1007),)
    assert completion.token_span.base == 0
    assert completion.token_span.len == 1
    assert completion.logical_lengths.token_len == 2
    assert completion.selected_point == 2
    assert worker.sessions.get(6).version == 2
    assert worker.sessions.get(6).logical_position == 2

    artifacts = [
        p for p in materialize_report.products if p.product.kind is ProductKind.ARTIFACT
    ]
    assert len(artifacts) == 1
    # The Artifact product carries the base64 PNG string as bytes: the scheduler
    # recovers it with String::from_utf8 and hands it to validate_png_artifact,
    # which base64-decodes it and checks the PNG dimensions. Mirror that contract.
    png_b64 = artifacts[0].payload
    png_bytes = base64.b64decode(png_b64.decode("ascii"), validate=True)
    assert png_bytes[:8] == _PNG_MAGIC
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (16, 16)
