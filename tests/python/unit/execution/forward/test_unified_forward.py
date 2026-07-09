from __future__ import annotations

from typing import Any

import pytest
import torch

from uniserve_worker.contracts.forward_mode import ForwardMode
from uniserve_worker.execution.forward import (
    EagerFallbackRecorder,
    ForwardBatchBuilder,
    ForwardExecutor,
    ForwardGraphPolicy,
    ForwardModelDescriptor,
    ForwardModelModules,
    ForwardOutputKind,
    ForwardPlanBuilder,
    ForwardPostprocessor,
    ForwardResult,
    ForwardRuntimeHandles,
    StrictForwardGraphError,
)
from uniserve_worker.execution.forward.graph import (
    ForwardGraphBufferRegistry,
    PaddingPolicy,
    SlotAxis,
    graph_shape_key,
)

pytestmark = pytest.mark.unit


def test_plan_builder_represents_text_generation_and_output_order():
    plan = ForwardPlanBuilder().build(
        [
            (3, {"req_id": 10, "kind": "decode_und", "token_ids": [7], "pos_range": [4, 5]}),
            (
                1,
                {
                    "req_id": 11,
                    "kind": "denoise_gen",
                    "latent_shape": [2, 2],
                    "cfg": {"branch_count": 3},
                },
            ),
            (2, {"req_id": 12, "kind": "commit_gen", "latent_shape": [2, 2]}),
        ]
    )

    assert plan.forward_mode is ForwardMode.MIXED
    assert plan.op_modes == (ForwardMode.DECODE, ForwardMode.DENOISE, ForwardMode.COMMIT)
    assert [row.original_index for row in plan.rows] == [3, 1, 2]
    assert [slot.req_id for slot in plan.output_slots] == [10, 11, 12]
    assert [slot.kind for slot in plan.output_slots] == [
        ForwardOutputKind.TEXT_TOKEN,
        ForwardOutputKind.DENOISE_STEP,
        ForwardOutputKind.COMMIT,
    ]
    assert plan.shape.branch_count == 3
    assert len(plan.segments) == 5


def test_batch_builder_flattens_mixed_text_tokens_and_segment_tables():
    plan = ForwardPlanBuilder().build(
        [
            {"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [3, 4]},
            {"req_id": 2, "kind": "prefill_und", "token_ids": [11, 12], "pos_range": [0, 2]},
        ]
    )

    batch = ForwardBatchBuilder().build(plan, device="cpu")

    assert batch.forward_mode is ForwardMode.MIXED
    assert batch.op_modes == (ForwardMode.DECODE, ForwardMode.EXTEND)
    torch.testing.assert_close(batch.input_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    torch.testing.assert_close(batch.positions, torch.tensor([3, 0, 1], dtype=torch.long))
    assert [segment.length for segment in batch.segments] == [1, 2]


def test_graph_shape_key_excludes_refreshable_runtime_values():
    builder = ForwardPlanBuilder()
    batch_builder = ForwardBatchBuilder()
    plan_a = builder.build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}]
    )
    plan_b = builder.build(
        [{"req_id": 9, "kind": "decode_und", "token_ids": [99], "pos_range": [128, 129]}]
    )

    key_a = graph_shape_key(program="decode", batch=batch_builder.build(plan_a), plan=plan_a)
    key_b = graph_shape_key(program="decode", batch=batch_builder.build(plan_b), plan=plan_b)

    assert key_a == key_b


def test_graph_buffer_registry_preserves_identity_and_applies_padding():
    registry = ForwardGraphBufferRegistry()
    slot = registry.register_slot(
        "tokens",
        axis=SlotAxis.TOKENS,
        shape=(4,),
        dtype=torch.long,
        device="cpu",
        padding=PaddingPolicy.SENTINEL,
        sentinel=-1,
    )
    ptr = slot.tensor.data_ptr()

    registry.refresh_slot("tokens", torch.tensor([3, 4], dtype=torch.long))

    assert registry.tensor("tokens").data_ptr() == ptr
    torch.testing.assert_close(registry.tensor("tokens"), torch.tensor([3, 4, -1, -1]))


def test_executor_strict_graph_policy_rejects_eager_fallback():
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(graph_policy=ForwardGraphPolicy(prefer_graph=True, strict=True))

    with pytest.raises(StrictForwardGraphError):
        executor.execute(batch, plan, forward_fn=lambda _batch: ForwardResult(runtime_outputs=({"req_id": 1},)))


def test_executor_delegated_graph_policy_does_not_record_fallback():
    recorder = EagerFallbackRecorder()
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        graph_policy=ForwardGraphPolicy(
            prefer_graph=True,
            strict=True,
            graph_selection_delegated=True,
        ),
    )
    batch = ForwardBatchBuilder().build(plan)
    executor = ForwardExecutor(
        graph_policy=ForwardGraphPolicy(graph_selection_delegated=True),
        fallback_recorder=recorder,
    )

    result = executor.execute(
        batch,
        plan,
        forward_fn=lambda _batch: ForwardResult(runtime_outputs=({"req_id": 1},)),
    )

    assert result.runtime_outputs == ({"req_id": 1},)
    assert recorder.counts == {}


def test_postprocessor_validates_before_runtime_side_effects():
    touched: list[Any] = []
    handles = ForwardRuntimeHandles(
        values={
            "postprocess_side_effects": lambda outputs: touched.extend(outputs),
        }
    )
    plan = ForwardPlanBuilder().build(
        [{"req_id": 1, "kind": "decode_und", "token_ids": [10], "pos_range": [0, 1]}],
        runtime_handles=handles,
    )

    with pytest.raises(Exception, match="output count"):
        ForwardPostprocessor().apply(plan, ForwardResult(runtime_outputs=()))

    assert touched == []


def test_descriptor_rejects_missing_declared_text_surfaces():
    descriptor = ForwardModelDescriptor(
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=4,
        vocab_size=16,
        num_layers=1,
        num_q_heads=1,
        num_kv_heads=1,
        head_dim=4,
        supports_text=True,
        modules=ForwardModelModules(),
    )

    with pytest.raises(Exception, match="text neural surfaces"):
        descriptor.validate()
