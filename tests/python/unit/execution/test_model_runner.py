"""Device staging and validation through the canonical ModelRunner seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.python.fixtures.model_execution import TEST_MODEL_SPEC
from uniserve_worker.execution._forward_plan import (
    ForwardPlan,
    GraphKey,
    OutputKind,
    OutputSlot,
    TransactionId,
)
from uniserve_worker.execution.model_runner import ModelRunner, RunPath
from uniserve_worker.forward import (
    AttentionSelection,
    EmptyKvView,
    EmptyLatentView,
    EmptyMeshView,
    EmptyOutputView,
    FlowOutput,
    FlowRow,
    ForwardBatch,
    ForwardContext,
    ForwardOutput,
    NoAttention,
    NoFlowConditioning,
    RouteId,
    TokenHidden,
    TokenIds,
    TokenOutput,
    TokenRow,
    TokenSelection,
)
from uniserve_worker.foundation.errors import ComputeError, InputError
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.execution_trace import ExecutionTrace
from uniserve_worker.runtime.graph_store import GraphStore

pytestmark = pytest.mark.unit


class _NoAttentionBackend:
    name = "none"

    @staticmethod
    def capabilities() -> object:
        return object()


class _MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ForwardBatch] = []
        self.output_dtype = torch.float32
        self.flow_output_dtype: torch.dtype | None = None

    def forward(self, batch: ForwardBatch) -> ForwardOutput:
        self.calls.append(batch)
        outputs: list[TokenOutput | FlowOutput] = []
        for row in batch.rows:
            if isinstance(row, TokenRow):
                value = row.positions.to(self.output_dtype).reshape(-1, 1)
                outputs.append(TokenOutput(row.row_id, row.output_slot, TokenHidden(value)))
            else:
                assert isinstance(row, FlowRow)
                outputs.append(
                    FlowOutput(
                        row.row_id,
                        row.output_slot,
                        torch.zeros_like(
                            row.latent,
                            dtype=self.flow_output_dtype or self.output_dtype,
                        ),
                    )
                )
        return ForwardOutput(tuple(outputs))


def _context() -> ForwardContext:
    selection = AttentionSelection("none", (_NoAttentionBackend(),))
    return ForwardContext(
        kv=EmptyKvView(),
        latent=EmptyLatentView(),
        attention=NoAttention(selection),
        mesh=EmptyMeshView(),
        output=EmptyOutputView(),
    )


def _plan(
    model: nn.Module,
    *,
    device: str = "cpu",
    flow_output_dtype: str = "float32",
) -> ForwardPlan:
    rows = (
        TokenRow(
            row_id=0,
            inputs=TokenIds(torch.tensor([3, 4], dtype=torch.long)),
            positions=torch.tensor([0, 1], dtype=torch.long),
            output_slot=0,
            selection=TokenSelection.HIDDEN,
        ),
        FlowRow(
            row_id=1,
            conditioning=NoFlowConditioning(),
            positions=torch.zeros((3, 1), dtype=torch.long),
            timestep=torch.tensor([0.5], dtype=torch.float32),
            latent=torch.ones((1, 4), dtype=torch.float32),
            image_tokens=1,
            image_height=16,
            image_width=16,
            output_slot=1,
        ),
    )
    return ForwardPlan(
        route=RouteId("mixed"),
        rows=rows,
        context=_context(),
        outputs=(
            OutputSlot(0, 0, OutputKind.TOKEN, "float32"),
            OutputSlot(1, 1, OutputKind.FLOW, flow_output_dtype),
        ),
        transaction=TransactionId(((1, 1, 1), (2, 1, 2)), (0, 0)),
        graph_key=GraphKey(
            model_revision="revision",
            spec_digest="d" * 64,
            route=RouteId("mixed"),
            shape=(1, 1, 2, 1),
            dtype="float32",
            backend="none",
            topology="tp:0/1",
        ),
        graph_eligible=True,
        device=device,
        weights=WeightSet.from_module(model),
    )


def _graph_store(*, enabled: bool) -> GraphStore:
    return GraphStore(
        enabled=enabled,
        prefill_enabled=False,
        cache=TEST_MODEL_SPEC.cache,
        block_size=16,
        spec_digest="d" * 64,
    )


def test_runner_stages_one_mixed_batch_and_returns_an_observation():
    model = _MixedModel()
    runner = ModelRunner(model, _graph_store(enabled=False), ExecutionTrace("d" * 64))

    output = runner.run(_plan(model))

    assert len(output.rows) == 2
    assert len(model.calls) == 1
    assert tuple(type(row).__name__ for row in model.calls[0].rows) == ("TokenRow", "FlowRow")
    observation = runner.last_observation
    assert observation is not None
    assert observation.path is RunPath.EAGER
    assert observation.model_forward_calls == 1
    assert observation.row_kind_counts == (("flow", 1), ("token", 1))


def test_graph_capture_failure_falls_back_to_the_same_mixed_eager_call(monkeypatch):
    model = _MixedModel()
    graph = _graph_store(enabled=True)
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)

    def fail_capture(_batch, _forward):
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(graph, "_capture", fail_capture)
    runner = ModelRunner(model, graph, ExecutionTrace("d" * 64))

    warmed = runner.run(_plan(model))
    output = runner.run(_plan(model))

    assert len(warmed.rows) == 2
    assert len(output.rows) == 2
    assert len(model.calls) == 2
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_FALLBACK
    assert runner.last_observation.model_forward_calls == 1


def test_graph_capture_follows_one_exact_shape_warmup(monkeypatch):
    model = _MixedModel()
    graph = _graph_store(enabled=True)
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)

    def capture(batch, forward):
        return SimpleNamespace(
            graph=SimpleNamespace(replay=lambda: None),
            batch=batch,
            output=forward(batch),
            releases=(),
        )

    monkeypatch.setattr(graph, "_capture", capture)
    runner = ModelRunner(model, graph, ExecutionTrace("d" * 64))

    runner.run(_plan(model))
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_FALLBACK

    runner.run(_plan(model))
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_CAPTURE
    assert len(model.calls) == 2


def test_runner_normalizes_staging_and_output_failures():
    model = _MixedModel()
    runner = ModelRunner(model, _graph_store(enabled=False), ExecutionTrace("d" * 64))

    with pytest.raises(InputError) as staging:
        runner.run(_plan(model, device="cuda:99"))
    assert staging.value.phase == "input_staging"
    assert staging.value.route == "mixed"

    model.output_dtype = torch.float64
    with pytest.raises(ComputeError, match="expected torch.float32") as output:
        runner.run(_plan(model))
    assert output.value.phase == "output_validation"
    assert output.value.route == "mixed"


def test_runner_validates_each_declared_output_dtype():
    model = _MixedModel()
    model.flow_output_dtype = torch.bfloat16
    runner = ModelRunner(model, _graph_store(enabled=False), ExecutionTrace("d" * 64))

    output = runner.run(_plan(model, flow_output_dtype="bfloat16"))

    assert isinstance(output.rows[1], FlowOutput)
    assert output.rows[1].prediction.dtype is torch.bfloat16


def test_equivalent_fresh_forward_inputs_produce_equivalent_raw_outputs():
    model = _MixedModel()
    runner = ModelRunner(model, _graph_store(enabled=False), ExecutionTrace("d" * 64))

    first = runner.run(_plan(model))
    second = runner.run(_plan(model))

    for left, right in zip(first.rows, second.rows, strict=True):
        if isinstance(left, TokenOutput) and isinstance(right, TokenOutput):
            left_tensor = left.value.value
            right_tensor = right.value.value
        else:
            assert isinstance(left, FlowOutput) and isinstance(right, FlowOutput)
            left_tensor = left.prediction
            right_tensor = right.prediction
        torch.testing.assert_close(left_tensor, right_tensor, rtol=0, atol=0)
