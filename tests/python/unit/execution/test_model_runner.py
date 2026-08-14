"""Fixed-address staging and direct execution through the model runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.python.fixtures.model_execution import TEST_MODEL
from uniserve_worker.backends.attention.torch_sdpa import TorchSDPAAttentionBackend
from uniserve_worker.execution.cuda_graph import CudaGraphRunner, GraphExecutionError
from uniserve_worker.execution.forward_batch import (
    AttentionSelection,
    EmptyKvView,
    EmptyMeshView,
    ForwardBatch,
    ForwardOutput,
    ModelPhase,
    NoAttention,
    PagedDecodePlan,
    TokenSelection,
    packed_tensor_views,
)
from uniserve_worker.execution.model_runner import ModelRunner, RunPath
from uniserve_worker.foundation.errors import ComputeError, InputError, ResourceError
from uniserve_worker.loader.weight_set import WeightSet
from uniserve_worker.runtime.cache_pool import CacheBatchView, CachePool, CacheRow
from uniserve_worker.runtime.execution_trace import ExecutionTrace

pytestmark = pytest.mark.unit


class _NoAttentionBackend:
    name = "none"

    @staticmethod
    def capabilities() -> object:
        return object()


def _attention() -> NoAttention:
    return NoAttention(AttentionSelection("none", (_NoAttentionBackend(),)))


class _MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.invalid_flow_shape = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        del input_ids
        chunks: list[torch.Tensor | None] = [None] * batch.row_count
        offset = 0
        for row_index, count in zip(batch.token_row_indices, batch.query_lens, strict=True):
            chunks[row_index] = (
                positions[offset : offset + count].float().reshape(-1, 1) * self.scale
            )
            offset += count
        for flow_index, row_index in enumerate(batch.flow_row_indices):
            count = int(batch.flow_image_tokens[flow_index])
            chunks[row_index] = torch.zeros((count, 1), device=positions.device) * self.scale
        return torch.cat(tuple(chunk for chunk in chunks if chunk is not None), dim=0)

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        lengths = [0] * batch.row_count
        for row_index, count in zip(batch.token_row_indices, batch.query_lens, strict=True):
            lengths[row_index] = count
        for row_index, count in zip(batch.flow_row_indices, batch.flow_image_tokens, strict=True):
            lengths[row_index] = count
        rows: list[torch.Tensor] = []
        offset = 0
        flow_by_row = dict(
            zip(batch.flow_row_indices, range(len(batch.flow_row_indices)), strict=True)
        )
        for row_index, count in enumerate(lengths):
            value = hidden[offset : offset + count]
            offset += count
            if row_index in flow_by_row:
                latent = batch.flow_latents[flow_by_row[row_index]]
                value = latent * self.scale
                if self.invalid_flow_shape:
                    value = value.reshape(-1)[:1]
            rows.append(value)
        return ForwardOutput(tuple(rows))


class _StagingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        axes = positions.unsqueeze(0) if positions.ndim == 1 else positions
        tokens = input_ids.to(dtype=torch.float32)
        if batch.input_embeddings is not None:
            if batch.embedding_mask is None:
                raise RuntimeError("staged embeddings have no selection mask")
            tokens = torch.where(
                batch.embedding_mask,
                batch.input_embeddings[:, 0].to(dtype=torch.float32),
                tokens,
            )
        return torch.cat((tokens.unsqueeze(1), axes.T.to(dtype=torch.float32)), dim=1) * self.scale

    def project(self, hidden: torch.Tensor, batch: ForwardBatch) -> ForwardOutput:
        rows: list[torch.Tensor] = []
        offset = 0
        for count in batch.query_lens:
            rows.append(hidden[offset : offset + count])
            offset += count
        return ForwardOutput(tuple(rows))


def _graph_runner(*, enabled: bool = False) -> CudaGraphRunner:
    return CudaGraphRunner(
        enabled=enabled,
        prefill_enabled=False,
        cache=TEST_MODEL.cache_geometry,
        block_size=16,
        weight_digest="d" * 64,
        memory_budget_bytes=1 << 30,
    )


def _runner(model: nn.Module) -> ModelRunner:
    return ModelRunner(
        model,
        _graph_runner(),
        ExecutionTrace("d" * 64),
        max_rows=4,
        max_tokens=16,
        max_blocks_per_row=4,
        hidden_size=4,
        devices=("cpu",),
    )


def _task(
    model: nn.Module,
    *,
    row: int,
    phase: ModelPhase,
    weights: WeightSet | None = None,
) -> SimpleNamespace:
    token = phase is ModelPhase.TEXT
    return SimpleNamespace(
        operation=SimpleNamespace(
            request_key=SimpleNamespace(session_id=row + 1, epoch=1),
            op_id=row + 1,
            parent=SimpleNamespace(point=SimpleNamespace(point_index=0)),
        ),
        session=SimpleNamespace(request_pool_idx=row + 1),
        weights=WeightSet.from_module(model) if weights is None else weights,
        phase=phase,
        token_ids=torch.tensor([3 + row, 4 + row]) if token else None,
        token_embeddings=None,
        token_embedding_mask=None,
        positions=torch.tensor([0, 1]) if token else torch.tensor([[0], [0], [0]]),
        selection=TokenSelection.HIDDEN if token else None,
        flow_conditioning=None,
        timestep=None if token else torch.tensor([0.5]),
        latent=None if token else torch.ones((1, 4)),
        image_tokens=0 if token else 1,
        image_height=0 if token else 16,
        image_width=0 if token else 16,
        encode_pixels=None,
        encode_grid=None,
        encode_grid_shape=None,
        kind="token" if token else "flow",
    )


def _run(runner: ModelRunner, tasks: tuple[SimpleNamespace, ...]) -> tuple[torch.Tensor, ...]:
    return runner.run(
        tasks,
        device="cpu",
        kv=EmptyKvView(),
        attention=_attention(),
        mesh=EmptyMeshView(),
        graph_shape=(len(tasks),),
        graph_eligible=False,
    )


def test_runner_stages_mixed_position_axes_and_embeddings_independently() -> None:
    model = _StagingModel()
    runner = _runner(model)
    priming = _task(model, row=0, phase=ModelPhase.TEXT)
    priming.positions = torch.tensor(((10, 11), (12, 13), (14, 15)))
    priming.token_embeddings = torch.tensor(((31.0, 0.0, 0.0, 0.0),) * 2)
    _run(runner, (priming,))

    text = _task(model, row=0, phase=ModelPhase.TEXT)
    text.positions = torch.tensor((2, 3))
    embedded = _task(model, row=1, phase=ModelPhase.TEXT)
    embedded.positions = torch.tensor(((4, 5), (6, 7), (8, 9)))
    embedded.token_embeddings = torch.tensor(((21.0, 0.0, 0.0, 0.0), (22.0, 0.0, 0.0, 0.0)))

    outputs = _run(runner, (text, embedded))

    torch.testing.assert_close(
        outputs[0],
        torch.tensor(((3.0, 2.0, 0.0, 0.0), (4.0, 3.0, 0.0, 0.0))),
    )
    torch.testing.assert_close(
        outputs[1],
        torch.tensor(((21.0, 4.0, 6.0, 8.0), (22.0, 5.0, 7.0, 9.0))),
    )


def test_runner_executes_one_mixed_batch_and_reports_the_eager_path():
    model = _MixedModel()
    runner = _runner(model)

    output = _run(
        runner,
        (_task(model, row=0, phase=ModelPhase.TEXT), _task(model, row=1, phase=ModelPhase.DENOISE)),
    )

    assert len(output) == 2
    torch.testing.assert_close(output[0], torch.tensor([[0.0], [1.0]]))
    torch.testing.assert_close(output[1], torch.ones((1, 4)))
    observation = runner.last_observation
    assert observation is not None
    assert observation.path is RunPath.EAGER
    assert observation.row_kind_counts == (("flow", 1), ("token", 1))


def test_runner_applies_one_immutable_weight_set_to_forward_and_projection():
    model = _MixedModel()
    runner = _runner(model)
    weights = WeightSet(
        digest="e" * 64,
        version=1,
        tensors={"scale": torch.tensor(3.0)},
    )

    (output,) = _run(runner, (_task(model, row=0, phase=ModelPhase.TEXT, weights=weights),))

    torch.testing.assert_close(output, torch.tensor([[0.0], [3.0]]))
    assert model.scale.item() == 1.0


def test_runner_reports_staging_and_model_output_failures():
    model = _MixedModel()
    runner = _runner(model)
    task = _task(model, row=0, phase=ModelPhase.TEXT)

    with pytest.raises(InputError, match="no 'prefill' execution partition"):
        runner.run(
            (task,),
            device="cuda:99",
            kv=EmptyKvView(),
            attention=_attention(),
            mesh=EmptyMeshView(),
            graph_shape=(1,),
            graph_eligible=False,
        )

    model.invalid_flow_shape = True
    with pytest.raises(ComputeError, match="flow prediction shape"):
        _run(runner, (_task(model, row=0, phase=ModelPhase.DENOISE),))


def test_equivalent_fresh_tasks_produce_equivalent_raw_outputs():
    model = _MixedModel()
    runner = _runner(model)

    first = _run(runner, (_task(model, row=0, phase=ModelPhase.TEXT),))
    second = _run(runner, (_task(model, row=0, phase=ModelPhase.TEXT),))

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_decode_graph_bucket_replays_smaller_batches_and_faults_on_a_covered_miss():
    device = torch.device("cuda", torch.cuda.current_device())
    model = _MixedModel().to(device)
    graphs = CudaGraphRunner(
        enabled=True,
        prefill_enabled=False,
        cache=TEST_MODEL.cache_geometry,
        block_size=4,
        weight_digest="d" * 64,
        memory_budget_bytes=1 << 40,
        decode_batch_sizes=(4,),
        decode_context_blocks=1,
    )
    runner = ModelRunner(
        model,
        graphs,
        ExecutionTrace("d" * 64),
        max_rows=4,
        max_tokens=4,
        max_blocks_per_row=1,
        hidden_size=4,
        devices=(device,),
    )
    pool = CachePool(
        num_layers=1,
        request_pages=8,
        scratch_pages=0,
        page_size=4,
        num_kv_heads=1,
        head_dim=2,
        device=device,
        dtype=torch.float32,
    )
    selection = AttentionSelection("torch_sdpa", (TorchSDPAAttentionBackend(),))

    def execute(count: int, *, position_base: int = 7) -> tuple[torch.Tensor, ...]:
        tasks = tuple(_task(model, row=index, phase=ModelPhase.TEXT) for index in range(count))
        for index, task in enumerate(tasks):
            task.token_ids = torch.tensor([3 + index])
            task.positions = torch.tensor([position_base + index])
            task.selection = TokenSelection.LAST_LOGITS
        rows = tuple(CacheRow((index + 1,), 0, 4) for index in range(count))
        kv = CacheBatchView(pool, rows, query_lengths=(1,) * count)
        attention = PagedDecodePlan(
            backends=selection,
            block_table=torch.tensor(
                tuple((index + 1,) for index in range(count)), dtype=torch.int32
            ),
            cache_seqlens=torch.zeros(count, dtype=torch.int32),
            kv_seqlens=torch.ones(count, dtype=torch.int32),
            query_lens=torch.ones(count, dtype=torch.int32),
            cache_seqlens_cpu=(0,) * count,
            kv_seqlens_cpu=(1,) * count,
            query_lens_cpu=(1,) * count,
            decode_page_ids=torch.arange(1, count + 1, dtype=torch.int32),
            decode_page_offsets=torch.zeros(count, dtype=torch.int32),
            max_context_len=4,
            causal=True,
            binding=1,
        )
        return runner.run(
            tasks,
            device=device,
            kv=kv,
            attention=attention,
            mesh=EmptyMeshView(),
            graph_shape=(count, count, (1,) * count, ()),
            graph_eligible=True,
        )

    execute(4)
    execute(4)
    graphs.complete_startup()
    replayed = execute(2)

    torch.testing.assert_close(replayed[0], torch.tensor([[7.0]], device=device))
    torch.testing.assert_close(replayed[1], torch.tensor([[8.0]], device=device))
    packed = packed_tensor_views(replayed)
    assert packed is not None
    preserved = packed.clone()
    updated = execute(2, position_base=17)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(preserved, torch.tensor([7.0, 8.0], device=device))
    torch.testing.assert_close(updated[0], torch.tensor([[17.0]], device=device))
    torch.testing.assert_close(updated[1], torch.tensor([[18.0]], device=device))
    observation = runner.last_observation
    assert observation is not None
    assert observation.path is RunPath.GRAPH_REPLAY
    assert observation.graph_padded_tokens == 2

    missed_tasks = tuple(_task(model, row=index, phase=ModelPhase.TEXT) for index in range(2))
    for task in missed_tasks:
        task.token_ids = torch.tensor([3])
        task.positions = torch.tensor([7])
        task.selection = TokenSelection.ALL_LOGITS
    missed_kv = CacheBatchView(
        pool,
        (CacheRow((1,), 0, 4), CacheRow((2,), 0, 4)),
        query_lengths=(1, 1),
    )
    missed_attention = PagedDecodePlan(
        backends=selection,
        block_table=torch.tensor(((1,), (2,)), dtype=torch.int32),
        cache_seqlens=torch.zeros(2, dtype=torch.int32),
        kv_seqlens=torch.ones(2, dtype=torch.int32),
        query_lens=torch.ones(2, dtype=torch.int32),
        cache_seqlens_cpu=(0, 0),
        kv_seqlens_cpu=(1, 1),
        query_lens_cpu=(1, 1),
        decode_page_ids=torch.tensor((1, 2), dtype=torch.int32),
        decode_page_offsets=torch.zeros(2, dtype=torch.int32),
        max_context_len=4,
        causal=True,
        binding=2,
    )
    with pytest.raises(ResourceError, match="not resident"):
        runner.run(
            missed_tasks,
            device=device,
            kv=missed_kv,
            attention=missed_attention,
            mesh=EmptyMeshView(),
            graph_shape=(2, 2, (1, 1), ()),
            graph_eligible=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_exact_flow_graph_replays_fresh_operation_inputs_after_startup():
    device = torch.device("cuda", torch.cuda.current_device())
    model = _MixedModel().to(device)
    graphs = _graph_runner(enabled=True)
    runner = ModelRunner(
        model,
        graphs,
        ExecutionTrace("d" * 64),
        max_rows=1,
        max_tokens=4,
        max_blocks_per_row=1,
        hidden_size=4,
        devices=(device,),
    )

    def execute(latent_value: float) -> tuple[torch.Tensor, RunPath]:
        task = _task(model, row=0, phase=ModelPhase.DENOISE)
        task.latent = torch.full((1, 4), latent_value, device=device)
        (output,) = runner.run(
            (task,),
            device=device,
            kv=EmptyKvView(),
            attention=_attention(),
            mesh=EmptyMeshView(),
            graph_shape=(1, 1, (), ((16, 16),)),
            graph_eligible=True,
        )
        observation = runner.last_observation
        assert observation is not None
        return output, observation.path

    first, first_path = execute(2.0)
    captured, captured_path = execute(3.0)
    graphs.complete_startup()
    replayed, replayed_path = execute(4.0)

    torch.testing.assert_close(first, torch.full((1, 4), 2.0, device=device))
    torch.testing.assert_close(captured, torch.full((1, 4), 3.0, device=device))
    torch.testing.assert_close(replayed, torch.full((1, 4), 4.0, device=device))
    assert first_path is RunPath.GRAPH_FALLBACK
    assert captured_path is RunPath.GRAPH_CAPTURE
    assert replayed_path is RunPath.GRAPH_REPLAY

    captured_pointer = captured.data_ptr()
    replayed_pointer = replayed.data_ptr()
    recycled, recycled_path = execute(5.0)
    assert captured_pointer != replayed_pointer
    assert recycled.data_ptr() == captured_pointer
    torch.testing.assert_close(recycled, torch.full((1, 4), 5.0, device=device))
    assert recycled_path is RunPath.GRAPH_REPLAY


def test_graph_startup_rejects_an_incomplete_advertised_catalog():
    graphs = CudaGraphRunner(
        enabled=True,
        prefill_enabled=False,
        cache=TEST_MODEL.cache_geometry,
        block_size=16,
        weight_digest="d" * 64,
        memory_budget_bytes=1 << 30,
        expected_captures=1,
    )

    with pytest.raises(GraphExecutionError, match="advertised bucket catalog"):
        graphs.complete_startup()
