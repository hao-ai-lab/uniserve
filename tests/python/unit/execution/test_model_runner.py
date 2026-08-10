"""Device staging and validation through the canonical ModelRunner seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.python.fixtures.model_execution import TEST_MODEL
from uniserve_worker.execution._forward_plan import (
    ForwardBinding,
    ForwardPlan,
    GraphKey,
)
from uniserve_worker.execution.model_runner import ModelRunner, RunPath, _stage_rows
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
    packed_token_ids,
    packed_token_positions,
)
from uniserve_worker.foundation.errors import ComputeError, InputError, ResourceError
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.execution_trace import ExecutionTrace
from uniserve_worker.runtime.graph_store import GraphStore
from uniserve_worker.runtime.host_staging import TensorStager

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
    tokens: int = 2,
) -> ForwardPlan:
    rows = (
        TokenRow(
            row_id=0,
            inputs=TokenIds(torch.arange(3, 3 + tokens, dtype=torch.long)),
            positions=torch.arange(tokens, dtype=torch.long),
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
        bindings=(
            ForwardBinding(
                row_id=0,
                slot=0,
                output_dtype="float32",
                session_id=1,
                epoch=1,
                op_id=1,
                base_version=0,
            ),
            ForwardBinding(
                row_id=1,
                slot=1,
                output_dtype=flow_output_dtype,
                session_id=2,
                epoch=1,
                op_id=2,
                base_version=0,
            ),
        ),
        graph_key=GraphKey(
            architecture_digest="a" * 64,
            weight_digest="d" * 64,
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


def _graph_store(*, enabled: bool, memory_budget_bytes: int = 1 << 34) -> GraphStore:
    return GraphStore(
        enabled=enabled,
        prefill_enabled=False,
        cache=TEST_MODEL.cache_geometry,
        block_size=16,
        weight_digest="d" * 64,
        memory_budget_bytes=memory_budget_bytes,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_staging_packs_mixed_host_and_device_rows():
    device = torch.device("cuda", torch.cuda.current_device())
    rows = (
        TokenRow(
            row_id=0,
            inputs=TokenIds(torch.tensor([11], dtype=torch.long)),
            positions=torch.tensor([3], dtype=torch.long, device=device),
            output_slot=0,
            selection=TokenSelection.HIDDEN,
        ),
        TokenRow(
            row_id=1,
            inputs=TokenIds(torch.tensor([17], dtype=torch.long, device=device)),
            positions=torch.tensor([9], dtype=torch.long),
            output_slot=1,
            selection=TokenSelection.HIDDEN,
        ),
    )
    stager = TensorStager(capacity=2, byte_capacity=1 << 20)

    staged = _stage_rows(rows, device, stager.acquire(device))
    token_rows = tuple(row for row in staged if isinstance(row, TokenRow))

    ids = packed_token_ids(token_rows)
    positions = packed_token_positions(token_rows)
    assert ids is not None
    assert positions is not None
    torch.testing.assert_close(ids.cpu(), torch.tensor([11, 17]), rtol=0, atol=0)
    torch.testing.assert_close(positions.cpu(), torch.tensor([3, 9]), rtol=0, atol=0)


def test_staging_capacity_becomes_available_after_event_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[SimpleNamespace] = []

    def event_factory(*, blocking: bool) -> SimpleNamespace:
        event = SimpleNamespace(
            blocking=blocking,
            ready=False,
            record=lambda _stream: None,
        )
        event.query = lambda: event.ready
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", event_factory)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    device = torch.device("cuda:0")
    stager = TensorStager(capacity=1, byte_capacity=1 << 20)
    first = stager.acquire(device)
    stager.mark_submitted(first, device)

    with pytest.raises(ResourceError):
        stager.acquire(device)

    events[0].ready = True
    successor = stager.acquire(device)
    assert successor.generation != first.generation


def test_staging_byte_capacity_rejects_growth_without_mutating_usage() -> None:
    stager = TensorStager(capacity=1, byte_capacity=16)
    slot = stager.acquire("cpu")
    assert slot.int_buffer("tokens", 4, pin=False).numel() == 4
    assert stager.allocated_bytes == 16
    with pytest.raises(ResourceError, match="byte capacity"):
        slot.int_buffer("positions", 1, pin=False)
    assert stager.allocated_bytes == 16


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
