"""Global sparse selection and compression across ordered context owners."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from uniserve_worker.backends.attention.video_sparse import (
    VideoSparseAttentionBackend,
    VideoSparseAttentionWorkspace,
    build_video_sparse_metadata,
    video_sparse_selected_tiles,
)
from uniserve_worker.bootstrap.distributed import (
    initialize_model_parallel,
    initialize_process_groups,
)
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.nn.parallel_attention import (
    AttentionContextGeometry,
    AttentionRowExchange,
    ParallelAttention,
)
from uniserve_worker.runtime.attention_storage import allocate_attention_context
from uniserve_worker.runtime.peer_memory import allocate_symmetric_memory

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_context(rank: int, rendezvous: str, world_size: int, kind: str) -> None:
    device = torch.device("cuda", rank)
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    ranks = tuple(reversed(range(world_size)))
    if kind == "hybrid":
        sequence = SequenceParallel("hybrid", (2, world_size // 2))
    elif kind == "attention2d":
        sequence = SequenceParallel("attention2d", (2, world_size // 2, 1))
    else:
        sequence = (
            SequenceParallel("allgather", (world_size,))
            if kind == "allgather"
            else SequenceParallel("ring", (world_size,))
        )
    mesh = initialize_model_parallel(
        environment,
        {"denoiser": (ranks, ParallelConfig(sequence_parallel=sequence))},
    )["denoiser"]
    torch.manual_seed(904)
    rows, heads, width, prefix, video_tiles = 1024, 7, 128, 2, 12
    tiles = rows // 64
    local_rows = rows // world_size
    owner = mesh.coord("sp")
    context_rows = local_rows * mesh.size("ulysses")
    context_start = mesh.coord("cp") * context_rows
    begin = owner * local_rows
    end = begin + local_rows
    local_video_tiles = max(
        0,
        min((context_start + context_rows) // 64, prefix + video_tiles)
        - max(context_start // 64, prefix),
    )
    global_heads = heads * mesh.size("ulysses")
    projected = torch.randn(rows, global_heads, 4, width, device=device, dtype=torch.bfloat16)
    query, key, value, gate = projected.unbind(2)
    valid = torch.tensor([64, 19, *([64] * video_tiles), 0, 0], device=device, dtype=torch.int32)
    metadata = build_video_sparse_metadata(
        padded_rows=rows,
        prefix_tiles=prefix,
        video_tiles=video_tiles,
        valid_sizes=valid,
        device=device,
    )
    backend = VideoSparseAttentionBackend(metadata)
    attention = ParallelAttention(mesh=mesh)
    fine_rows = context_rows
    fine_tiles = fine_rows // 64
    workspace = VideoSparseAttentionWorkspace(
        attention_output=torch.empty(
            fine_rows,
            heads,
            width,
            device=device,
            dtype=torch.bfloat16,
        ),
        tile_scores=torch.empty(heads, fine_tiles, tiles, device=device),
        block_counts=torch.empty(heads, fine_tiles, device=device, dtype=torch.int32),
        block_indices=torch.empty(
            heads, fine_tiles, prefix + video_tiles, device=device, dtype=torch.int32
        ),
        pooled_query=torch.empty(fine_tiles, heads, width, device=device),
        pooled_key=torch.empty(tiles, heads, width, device=device),
        pooled_value=torch.empty(tiles, heads, width, device=device),
        compressed_tiles=torch.empty(heads, fine_tiles, width, device=device),
        topk_indices_i32=torch.empty(
            heads,
            local_video_tiles,
            video_sparse_selected_tiles(video_tiles),
            device=device,
            dtype=torch.int32,
        ),
    )
    exchange = allocate_symmetric_memory(
        mesh.get_group("ulysses"),
        (local_rows, global_heads, width),
        dtype=torch.bfloat16,
    )
    outputs = exchange.peers
    output = exchange.local
    sync_input = torch.zeros(1, device=device, dtype=torch.int32)
    sync_output = torch.empty(mesh.size("ulysses"), device=device, dtype=torch.int32)
    key_group = mesh.get_group("cp_row" if kind == "attention2d" else "cp")
    context_workspace = allocate_attention_context(
        AttentionContextGeometry(
            group=key_group,
            rows=context_rows * (mesh.size("cp_col") if kind == "attention2d" else 1),
            heads=heads,
            mapped=kind != "allgather",
            head_dim=width,
            dtype=torch.bfloat16,
            block_size=64,
        ),
    )
    prefix_indices = torch.arange(prefix, device=device, dtype=torch.int32)
    dense_indices = torch.arange(prefix + video_tiles, device=device, dtype=torch.int32)
    prefix_count = torch.tensor(prefix, device=device, dtype=torch.int32)

    def execute():
        local_query, local_key, local_value, local_gate = attention.exchange_heads(
            projected[begin:end]
        ).unbind(2)
        result = backend.forward_parallel(
            attention,
            local_query,
            local_key,
            local_value,
            local_gate,
            valid,
            prefix_indices,
            dense_indices,
            prefix_count,
            workspace,
            outputs=outputs,
            sync_input=sync_input,
            sync_output=sync_output,
            context_workspace=context_workspace,
        )
        return result.materialize() if isinstance(result, AttentionRowExchange) else result

    with torch.inference_mode():
        actual = execute()
        # Explicit dense reference: valid-row tile means, globally selected
        # video blocks, exempt prefix queries, and a separate compression softmax.
        valid_rows = torch.arange(rows, device=device) % 64 < valid.repeat_interleave(64)
        means = []
        for tensor in (query, key, value):
            masked = tensor.double() * valid_rows.view(rows, 1, 1)
            pooled = masked.reshape(tiles, 64, global_heads, width).sum(1)
            means.append((pooled / valid.clamp_min(1).view(tiles, 1, 1)).transpose(0, 1))
        pooled_q, pooled_k, pooled_v = means
        tile_scores = (pooled_q @ pooled_k.transpose(-1, -2)) * width**-0.5
        block_mask = torch.zeros(global_heads, tiles, tiles, device=device, dtype=torch.bool)
        block_mask[:, :prefix, : prefix + video_tiles] = True
        block_mask[:, prefix : prefix + video_tiles, :prefix] = True
        selected = (
            tile_scores[:, prefix : prefix + video_tiles, prefix : prefix + video_tiles]
            .topk(
                video_sparse_selected_tiles(video_tiles),
                dim=-1,
            )
            .indices
            + prefix
        )
        block_mask[:, prefix : prefix + video_tiles].scatter_(2, selected, True)
        row_mask = block_mask.repeat_interleave(64, 1).repeat_interleave(64, 2)
        row_mask &= valid_rows.view(1, 1, rows)
        logits = query.transpose(0, 1).double() @ key.transpose(0, 1).double().transpose(-1, -2)
        logits.mul_(width**-0.5).masked_fill_(~row_mask, -torch.inf)
        fine = logits.softmax(-1).nan_to_num() @ value.transpose(0, 1).double()
        tile_scores.masked_fill_(valid.view(1, 1, -1) == 0, -torch.inf)
        compressed = tile_scores.softmax(-1) @ pooled_v
        reference = fine.transpose(0, 1) + gate.double() * compressed.transpose(
            0, 1
        ).repeat_interleave(64, 0)
        live = valid_rows[begin:end]
        torch.testing.assert_close(
            actual[live].double(), reference[begin:end][live], rtol=2e-2, atol=2e-2
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        graph.replay()
        torch.testing.assert_close(
            output[live].double(), reference[begin:end][live], rtol=2e-2, atol=2e-2
        )
        graph.reset()
    environment.close()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kind", ["allgather", "ring", "attention2d", "hybrid"])
def test_sparse_context_preserves_global_selection_and_compression(
    tmp_path: Path, world_size: int, kind: str
):
    mp.spawn(
        _run_context,
        ((tmp_path / "rendezvous").as_uri(), world_size, kind),
        nprocs=world_size,
        join=True,
    )
