"""Depth-one forward behavior at the canonical ModelExecutor boundary.

These tests run the real forward through the new four-record protocol and assert
that committed tokens, KV growth, latent trajectory, image artifacts, replay, and
rollback reproduce the serial depth-one oracle.
"""

from __future__ import annotations

import base64
import io
from copy import deepcopy

import pytest
import torch
from PIL import Image
from torch import nn

from tests.python.fixtures.depth_one import (
    encode_operation,
    flow_operation,
    gen_admission,
    materialize_operation,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import Batch, ImageParams, ProductKind, TokenMode
from uniserve_worker.forward import FlowRow, ForwardBatch, ForwardOutput, PagedDecodePlan
from uniserve_worker.server.stub import _next_token

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
        self.attention_plans: list[object] = []
        self.fault: str | None = None

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append((str(batch.route), tuple(type(row).__name__ for row in batch.rows)))
        self.attention_plans.append(batch.context.attention)
        self.flow_inputs.extend(
            row.latent.detach().clone() for row in batch.rows if isinstance(row, FlowRow)
        )
        output = self.neural(batch)
        if self.fault == "raise":
            raise RuntimeError("injected neural failure")
        if self.fault == "misaligned":
            return ForwardOutput(output.rows[:-1])
        return output


def test_extend_then_decode_commit_the_serial_oracle_tokens():
    worker = execution_worker(_ObservedModel())
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

    first_token = extended.completions[0].committed_tokens[0]
    decode, decode_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=worker.sessions.get(1).committed_version(),
        mode=TokenMode.DECODE,
        tokens=(first_token,),
    )
    decoded = worker.execute(
        Batch(step_id=2, admissions=(), operations=(decode,), input_products=(decode_input,))
    )

    assert decoded.completions[0].committed_tokens == (_next_token(first_token),)
    assert worker.kv.get(1).length == 3
    assert worker.sessions.get(1).version == 2


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
    for index, admission in enumerate(admissions):
        session_id = admission.request_key.session_id
        operation, payload = token_operation(
            admission.request_key,
            op_id=60 + index,
            parent=worker.sessions.get(session_id).committed_version(),
            mode=TokenMode.DECODE,
            tokens=(worker.sessions.get(session_id).version,),
        )
        decode_ops.append(operation)
        decode_inputs.append(payload)
    decoded = worker.execute(
        Batch(step_id=2, admissions=(), operations=tuple(decode_ops), input_products=tuple(decode_inputs))
    )

    plan = model.attention_plans[-1]
    assert isinstance(plan, PagedDecodePlan)
    assert tuple(plan.query_lens.tolist()) == (1, 1)
    assert decoded.completions[0].committed_tokens == (_next_token(_next_token(4)),)
    assert decoded.completions[1].committed_tokens == (_next_token(_next_token(4)),)


def test_replay_is_idempotent_and_conflicts_or_stale_work_do_not_mutate_state():
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

    stale, stale_input = token_operation(
        admission.request_key,
        op_id=22,
        parent=root_parent(admission),
        mode=TokenMode.DECODE,
        tokens=(0,),
    )
    with pytest.raises(Exception, match="parent"):
        worker.execute(
            Batch(step_id=9, admissions=(), operations=(stale,), input_products=(stale_input,))
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
    committed = deepcopy(worker.sessions.get(4))
    committed_length = worker.kv.get(4).length

    retry, retry_input = token_operation(
        admission.request_key,
        op_id=32,
        parent=worker.sessions.get(4).committed_version(),
        mode=TokenMode.DECODE,
        tokens=(worker.sessions.get(4).version,),
    )
    retry_batch = Batch(step_id=12, admissions=(), operations=(retry,), input_products=(retry_input,))
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
        decode, decode_input = token_operation(
            admission.request_key,
            op_id=2 + step,
            parent=worker.sessions.get(1).committed_version(),
            mode=TokenMode.DECODE,
            tokens=(0,),
            new_kv_blocks=new_kv_blocks,
        )
        report = worker.execute(
            Batch(step_id=2 + step, admissions=(), operations=(decode,), input_products=(decode_input,))
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
    second = flow_operation(
        admission.request_key, op_id=2, parent=worker.sessions.get(2).committed_version(), steps=1
    )
    second_report = worker.execute(
        Batch(step_id=2, admissions=(), operations=(second,), input_products=())
    )

    assert first_report.completions[0].logical_lengths.latent_len == 1
    assert second_report.completions[0].logical_lengths.latent_len == 2


def test_encode_reports_per_op_image_kv_and_echoes_the_encoder_handle():
    # kv_visible_len is the image KV this encode wrote (its state-driver forward
    # span), not the session's committed total, and product_generations echoes the
    # scheduler-assigned nonzero encoder handle from the encode output reference.
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
    encode, encode_input = encode_operation(
        admission.request_key,
        op_id=2,
        parent=worker.sessions.get(3).committed_version(),
        image_base64=image_base64,
        encoder_handle=handle,
    )
    report = worker.execute(
        Batch(step_id=2, admissions=(), operations=(encode,), input_products=(encode_input,))
    )
    completion = report.completions[0]
    image_kv_written = worker.kv.get(3).length - session_kv_before

    # The real per-op image KV the forward wrote — not the session total (2).
    assert completion.logical_lengths.kv_visible_len == image_kv_written
    assert completion.logical_lengths.kv_visible_len != session_kv_before
    # The nonzero encoder handle echoed from the encode output reference.
    assert completion.product_generations
    assert completion.product_generations[0] == handle
    assert completion.product_generations[0] != 0


def test_flow_then_materialize_emits_a_png_image_artifact():
    worker = execution_worker(_ObservedModel())
    admission = gen_admission(6, ImageParams(steps=2, height=16, width=16, seed=29))
    flow = flow_operation(admission.request_key, op_id=1, parent=root_parent(admission), steps=2)
    worker.execute(Batch(step_id=1, admissions=(admission,), operations=(flow,), input_products=()))

    materialize = materialize_operation(
        admission.request_key, op_id=2, parent=worker.sessions.get(6).committed_version()
    )
    report = worker.execute(
        Batch(step_id=2, admissions=(), operations=(materialize,), input_products=())
    )

    artifacts = [p for p in report.products if p.product.kind is ProductKind.ARTIFACT]
    assert len(artifacts) == 1
    # The Artifact product carries the base64 PNG string as bytes: the scheduler
    # recovers it with String::from_utf8 and hands it to validate_png_artifact,
    # which base64-decodes it and checks the PNG dimensions. Mirror that contract.
    png_b64 = artifacts[0].payload
    png_bytes = base64.b64decode(png_b64.decode("ascii"), validate=True)
    assert png_bytes[:8] == _PNG_MAGIC
    with Image.open(io.BytesIO(png_bytes)) as image:
        assert image.size == (16, 16)
