"""Device staging and validation through the canonical ModelRunner seam."""

from __future__ import annotations

from dataclasses import replace
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
    GraphBinding,
    NoAttention,
    NoFlowConditioning,
    PagedDecodePlan,
    PagedVarlenPlan,
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
from uniserve_worker.runtime.graph_store import (
    GraphStore,
    _graph_selection,
    _GraphState,
    _prefill_padding,
)
from uniserve_worker.runtime.host_staging import TensorStager
from uniserve_worker.runtime.kv_pool import PagedKVPool
from uniserve_worker.runtime.kv_store import KvBatchView

pytestmark = pytest.mark.unit


class _NoAttentionBackend:
    name = "none"

    @staticmethod
    def capabilities() -> object:
        return object()


class _GraphAttentionBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def capabilities() -> object:
        return SimpleNamespace(paged_varlen_cuda_graph=True)


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


def _fake_capture():
    """Stand in for a real capture, without touching the device."""

    def capture(batch, forward):
        return _GraphState(
            graph=SimpleNamespace(replay=lambda: None, reset=lambda: None),
            batch=batch,
            output=forward(batch),
            releases=(),
            prefill_padding=_prefill_padding(batch),
        )

    return capture


def _graph_store(*, enabled: bool, memory_budget_bytes: int = 1 << 34) -> GraphStore:
    return GraphStore(
        enabled=enabled,
        prefill_enabled=False,
        cache=TEST_MODEL_SPEC.cache,
        block_size=16,
        spec_digest="d" * 64,
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

    monkeypatch.setattr(graph, "_capture", _fake_capture())
    runner = ModelRunner(model, graph, ExecutionTrace("d" * 64))

    runner.run(_plan(model))
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_FALLBACK

    runner.run(_plan(model))
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_CAPTURE
    assert len(model.calls) == 2

    runner.run(_plan(model))
    assert runner.last_observation is not None
    assert runner.last_observation.path is RunPath.GRAPH_REPLAY


def test_graph_store_executes_decode_in_smallest_reserved_bucket(monkeypatch):
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=8,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        reserved_tail_blocks=2,
        branch_blocks=1,
    )
    rows = tuple(
        TokenRow(
            row_id=row,
            inputs=TokenIds(torch.tensor([row], dtype=torch.long)),
            positions=torch.tensor([1], dtype=torch.long),
            output_slot=row,
            selection=TokenSelection.HIDDEN,
        )
        for row in range(3)
    )
    selection = AttentionSelection("paged", (_NoAttentionBackend(),))
    attention = PagedDecodePlan(
        backends=selection,
        block_table=torch.tensor([[0], [1], [2]], dtype=torch.int32),
        cache_seqlens=torch.ones(3, dtype=torch.int32),
        kv_seqlens=torch.full((3,), 2, dtype=torch.int32),
        query_lens=torch.ones(3, dtype=torch.int32),
        cache_seqlens_cpu=(1, 1, 1),
        kv_seqlens_cpu=(2, 2, 2),
        query_lens_cpu=(1, 1, 1),
        decode_page_ids=torch.tensor([0, 1, 2], dtype=torch.int32),
        decode_page_offsets=torch.ones(3, dtype=torch.int32),
        max_context_len=4,
        causal=True,
        binding=GraphBinding(1),
    )
    batch = ForwardBatch(
        RouteId("decode"),
        rows,
        ForwardContext(
            kv=KvBatchView(pool, ((0,), (1,), (2,)), (1, 1, 1), (1, 1, 1)),
            latent=EmptyLatentView(),
            attention=attention,
            mesh=EmptyMeshView(),
            output=EmptyOutputView(),
        ),
    )
    model = _MixedModel()
    graph = GraphStore(
        enabled=True,
        prefill_enabled=False,
        cache=TEST_MODEL_SPEC.cache,
        block_size=4,
        spec_digest="d" * 64,
        memory_budget_bytes=1 << 30,
        decode_batch_sizes=(1, 2, 4, 8),
        decode_context_blocks=6,
    )
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)

    run = graph.execute("decode", batch, model, eligible=True)

    assert run.row_count == 3
    assert run.padded_row_count == 4
    assert len(run.output.rows) == 3
    executed = model.calls[0]
    assert len(executed.rows) == 4
    token_rows = tuple(row for row in executed.rows if isinstance(row, TokenRow))
    assert len(token_rows) == 4
    assert packed_token_ids(token_rows) is not None
    assert packed_token_positions(token_rows) is not None
    padded = executed.context.attention
    assert isinstance(padded, PagedDecodePlan)
    assert tuple(padded.block_table.shape) == (4, 6)
    assert tuple(padded.block_table[-1, :2].tolist()) == pool.reserved_block_ids
    assert tuple(padded.block_table[-1, 2:].tolist()) == (0, 0, 0, 0)
    assert int(padded.decode_page_ids[-1]) == pool.reserved_block_ids[0]
    assert int(padded.decode_page_offsets[-1]) == 0


def _prefill_batch(
    pool: PagedKVPool,
    query_lens: tuple[int, ...],
    *,
    selection: TokenSelection = TokenSelection.LAST_LOGITS,
) -> ForwardBatch:
    source_rows = tuple(
        TokenRow(
            row_id=row,
            inputs=TokenIds(torch.arange(length, dtype=torch.long) + row * 100),
            positions=torch.arange(length, dtype=torch.long),
            output_slot=row,
            selection=selection,
        )
        for row, length in enumerate(query_lens)
    )
    rows = tuple(
        row
        for row in _stage_rows(source_rows, torch.device("cpu"), None)
        if isinstance(row, TokenRow)
    )
    block_ids = tuple((row * 2, row * 2 + 1) for row in range(len(rows)))
    view = KvBatchView(pool, block_ids, (0,) * len(rows), query_lens)
    cumulative = [0]
    for length in query_lens:
        cumulative.append(cumulative[-1] + length)
    selection = AttentionSelection("paged", (_NoAttentionBackend(),))
    attention = PagedVarlenPlan(
        backends=selection,
        block_table=view.block_table(torch.device("cpu")),
        cache_seqlens=view.cache_seqlens(torch.device("cpu")),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        kv_seqlens=torch.tensor(query_lens, dtype=torch.int32),
        cu_seqlens_q=torch.tensor(cumulative, dtype=torch.int32),
        cu_seqlens_k=torch.tensor(cumulative, dtype=torch.int32),
        output_indices=torch.tensor(
            tuple(value - 1 for value in cumulative[1:]),
            dtype=torch.int64,
        ),
        cache_seqlens_cpu=(0,) * len(rows),
        query_lens_cpu=query_lens,
        kv_seqlens_cpu=query_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(query_lens),
        max_context_len=8,
        causal=True,
        binding=GraphBinding(1),
    )
    return ForwardBatch(
        RouteId("prefill"),
        rows,
        ForwardContext(
            kv=view,
            latent=EmptyLatentView(),
            attention=attention,
            mesh=EmptyMeshView(),
            output=EmptyOutputView(),
        ),
    )


def test_graph_store_executes_unqualified_full_logits_prefill_eagerly(monkeypatch):
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=12,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        reserved_tail_blocks=2,
    )
    model = _MixedModel()
    graph = GraphStore(
        enabled=True,
        prefill_enabled=True,
        cache=TEST_MODEL_SPEC.cache,
        block_size=4,
        spec_digest="d" * 64,
        memory_budget_bytes=1 << 30,
        prefill_token_sizes=(8, 16),
    )
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)

    first = graph.execute(
        "verify",
        _prefill_batch(pool, (2, 3), selection=TokenSelection.ALL_LOGITS),
        model,
        eligible=True,
    )
    second = graph.execute(
        "verify",
        _prefill_batch(pool, (3, 4), selection=TokenSelection.ALL_LOGITS),
        model,
        eligible=True,
    )

    assert first.path == "eager"
    assert second.path == "eager"
    assert tuple(row.value.value.shape for row in first.output.rows) == ((2, 1), (3, 1))
    assert tuple(row.value.value.shape for row in second.output.rows) == ((3, 1), (4, 1))


def test_graph_store_reuses_prefill_token_bucket_across_ragged_shapes(monkeypatch):
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=12,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        reserved_tail_blocks=2,
    )
    model = _MixedModel()
    graph = GraphStore(
        enabled=True,
        prefill_enabled=True,
        cache=TEST_MODEL_SPEC.cache,
        block_size=4,
        spec_digest="d" * 64,
        memory_budget_bytes=1 << 30,
        prefill_token_sizes=(8, 16),
    )
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)
    monkeypatch.setattr(graph, "_capture", _fake_capture())

    warmed = graph.execute("prefill", _prefill_batch(pool, (2, 3)), model, eligible=True)
    captured = graph.execute("prefill", _prefill_batch(pool, (3, 3)), model, eligible=True)
    replayed = graph.execute("prefill", _prefill_batch(pool, (3, 4)), model, eligible=True)

    assert warmed.path == "graph_fallback"
    assert captured.path == "graph_capture"
    assert replayed.path == "graph_replay"
    assert (warmed.row_count, warmed.padded_row_count) == (2, 8)
    executed_rows = tuple(row for row in model.calls[0].rows if isinstance(row, TokenRow))
    packed = packed_token_ids(executed_rows)
    assert packed is not None
    assert int(packed.numel()) == 8
    assert tuple(int(row.inputs.values.numel()) for row in executed_rows) == (
        2,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
    )
    replay_attention = captured.output.rows
    assert len(replay_attention) == 2
    static = next(iter(graph._states.values())).batch.context.attention
    assert isinstance(static, PagedVarlenPlan)
    assert tuple(static.output_indices.tolist()) == (2, 6, 7, 0, 0, 0, 0, 0)


def test_graph_execution_resolves_one_attention_provider():
    pool = PagedKVPool(
        num_layers=1,
        num_blocks=12,
        block_size=4,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        reserved_tail_blocks=2,
    )
    batch = _prefill_batch(pool, (2, 3))
    attention = batch.context.attention
    assert isinstance(attention, PagedVarlenPlan)
    first = _GraphAttentionBackend("first")
    second = _GraphAttentionBackend("second")
    plan = replace(
        attention,
        backends=AttentionSelection("ordered", (first, second)),
    )

    selection = _graph_selection(plan)

    assert selection.providers == (first,)
    assert selection.identity == "ordered:graph:first"


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
    with pytest.raises(ResourceError, match="byte credit"):
        slot.int_buffer("positions", 1, pin=False)
    assert stager.allocated_bytes == 16


def test_graph_residency_stops_growing_with_batch_shape_diversity(monkeypatch):
    capture_bytes = 1 << 20
    model = _MixedModel()
    graph = _graph_store(enabled=True, memory_budget_bytes=2 * capture_bytes)
    monkeypatch.setattr(graph, "_cuda_batch", lambda _batch: True)
    monkeypatch.setattr(graph, "_capture", _fake_capture())
    # Stand in for the allocator: every retained executable holds one pool.
    monkeypatch.setattr(
        GraphStore,
        "resident_bytes",
        property(lambda store: len(store._states) * capture_bytes),
    )
    runner = ModelRunner(model, graph, ExecutionTrace("d" * 64))

    def run_widths(widths):
        for width in widths:
            for _ in range(2):
                runner.run(_plan(model, tokens=width))

    run_widths(range(1, 5))
    retained = len(graph._states)
    run_widths(range(5, 13))

    # The budget holds two executables; a third may be captured before the
    # next capture attempt frees room, and nothing beyond that accumulates.
    assert retained <= 3
    assert len(graph._states) == retained
    assert graph.evictions > 0


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
