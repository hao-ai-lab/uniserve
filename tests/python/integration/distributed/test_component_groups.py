"""Real collective and checkpoint-to-Linear behavior on overlapping component groups."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from uniserve_worker.loader.handles import TensorWeightHandle
from uniserve_worker.loader.weight_loaders import attach_parameter_loaders, load_parameter_weight
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.linear import (
    ColumnParallelLinear,
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from uniserve_worker.nn.mesh import DeviceMesh
from uniserve_worker.nn.parallel import ComponentConfig, ParallelConfig, SequenceParallel
from uniserve_worker.nn.vocab_parallel_embedding import VocabParallelEmbedding
from uniserve_worker.runtime.distributed import init_distributed_environment

pytestmark = pytest.mark.integration


@torch.inference_mode()
def _run_groups(rank: int, rendezvous: str, backend: str):
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=device,
        backend=backend,
        init_method=rendezvous,
    )
    bindings = environment.initialize_entries(
        {
            "denoiser": ComponentConfig(
                (0, 1, 2, 3),
                ParallelConfig(2, sequence_parallel=SequenceParallel("ulysses", (2,))),
            ),
            "encoder": ComponentConfig((3, 1), ParallelConfig(2)),
            "output": ComponentConfig((2,)),
            "decoder": ComponentConfig((3, 1), distribution="temporal_units", units_per_rank=2),
        },
    )
    meshes = bindings.meshes
    assert bindings.input_ranks("decoder") == (3, 1)
    assert bindings.output_ranks("decoder") == (3, 1)
    assert bindings.owns("decoder") == (rank in (3, 1))
    if bindings.owns("decoder"):
        value = torch.tensor([rank + 1.0], device=device)
        result = meshes["decoder"].get_group("tp").all_reduce(value.clone())
        torch.testing.assert_close(result, value, rtol=0, atol=0)

    for mesh in meshes.values():
        for dimension in ("tp", "ulysses", "sp", "pp"):
            group = mesh.get_group(dimension)
            value = torch.tensor([[rank + 1.0]], device=device)
            expected = torch.tensor([[sum(member + 1.0 for member in group.ranks)]], device=device)
            torch.testing.assert_close(group.all_reduce(value.clone()), expected, rtol=0, atol=0)
            gathered = torch.empty((group.world_size, 1), device=device)
            group.all_gather_into_tensor(gathered, value)
            reference = torch.tensor([[member + 1.0] for member in group.ranks], device=device)
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
            torch.testing.assert_close(group.all_gather(value), reference, rtol=0, atol=0)
            root = group.world_size - 1
            broadcast = group.broadcast(value.clone(), src=root)
            torch.testing.assert_close(broadcast, reference[root : root + 1], rtol=0, atol=0)
            gathered_root = (
                torch.empty((group.world_size, 1, 1), device=device)
                if group.rank_in_group == root
                else None
            )
            group.gather_into_tensor(gathered_root, value, dst=root)
            if gathered_root is not None:
                torch.testing.assert_close(gathered_root.reshape(-1, 1), reference, rtol=0, atol=0)
            send = torch.tensor([[rank * 10 + member] for member in group.ranks], device=device)
            received = torch.empty_like(send)
            group.all_to_all_single_into(
                received, send, [1] * group.world_size, [1] * group.world_size
            )
            torch.testing.assert_close(
                received,
                torch.tensor([[member * 10 + rank] for member in group.ranks], device=device),
                rtol=0,
                atol=0,
            )
            reduced = group.reduce_scatter(
                torch.ones(group.world_size, 2, device=device) * (rank + 1)
            )
            torch.testing.assert_close(
                reduced,
                torch.full((1, 2), sum(member + 1.0 for member in group.ranks), device=device),
                rtol=0,
                atol=0,
            )

    if backend == "nccl":
        from uniserve_worker.nn.parallel_attention import ParallelAttention
        from uniserve_worker.ops.video_sparse import compose_to_head_shards

        for component, dimension in (("denoiser", "ulysses"), ("encoder", "tp"), ("output", "tp")):
            if component not in meshes:
                continue
            group = meshes[component].get_group(dimension)
            rows, local_heads, width = 64, 7, 128
            count = group.world_size
            # Projection transport does not invoke the backend. The public
            # exchange contract can be checked with exact rank/row/head values.
            projection_mesh = DeviceMesh(
                group.ranks,
                group.rank,
                ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (count,))),
                torch.device(device),
                {"ulysses": group},
            )
            parallel_attention = ParallelAttention(
                mesh=projection_mesh,
            )
            projected = torch.arange(
                rows * count * local_heads * 4 * width, device=device, dtype=torch.float32
            )
            projected = projected.view(rows, count * local_heads, 4, width) + rank * 1_000_000
            exchanged = parallel_attention.exchange_heads(projected)
            reference = torch.cat(
                [
                    (projected - rank * 1_000_000 + member * 1_000_000).chunk(count, dim=1)[
                        group.rank_in_group
                    ]
                    for member in group.ranks
                ]
            )
            torch.testing.assert_close(exchanged, reference, rtol=0, atol=0)
            workspace = environment.symmetric_memory(
                group,
                (rows, count * local_heads, width),
                dtype=torch.bfloat16,
                name="attention_output",
                layout=(),
            )
            global_rows = rows * count
            attended = torch.full(
                (1, local_heads, global_rows, width),
                float(rank),
                device=device,
                dtype=torch.bfloat16,
            )
            gate = torch.ones(global_rows, local_heads, width, device=device, dtype=torch.bfloat16)
            compressed = (
                torch.arange(count, device=device, dtype=torch.float32)
                .view(1, count, 1)
                .expand(local_heads, count, width)
                .contiguous()
            )
            sync_input = torch.ones(1, device=device, dtype=torch.int32)
            sync_output = torch.empty(count, device=device, dtype=torch.int32)

            def compose():
                compose_to_head_shards(
                    attended, gate, compressed, workspace.peers, group.rank_in_group
                )
                group.all_gather_into_tensor(sync_output, sync_input)

            compose()
            expected = torch.cat(
                [
                    torch.full(
                        (rows, local_heads, width),
                        float(member + group.rank_in_group),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    for member in group.ranks
                ],
                dim=1,
            )
            torch.testing.assert_close(workspace.local, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compose()
            attended.add_(1)
            graph.replay()
            torch.testing.assert_close(workspace.local, expected + 1, rtol=0, atol=0)
            # Captured NCCL operations retain their communicator until graph
            # retirement. Release graphs before destroying runtime groups.
            graph.reset()

    tp = meshes["denoiser"].get_group("tp")
    config = LayerConfig(tp, None)
    weight = torch.arange(32, dtype=torch.float32, device=device).reshape(8, 4) / 32
    down_weight = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 8) / 16
    bias = torch.tensor([2.0, -1.0, 0.5], device=device)
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4) / 8
    with torch.device(device):
        column = ColumnParallelLinear(4, 8, layer_config=config, bias=False)
        row = RowParallelLinear(8, 3, layer_config=config, bias=True)
    load_parameter_weight(column.weight, TensorWeightHandle("weight", weight))
    load_parameter_weight(row.weight, TensorWeightHandle("weight", down_weight))
    load_parameter_weight(row.bias, TensorWeightHandle("bias", bias))
    actual = row(column(x))
    torch.testing.assert_close(
        actual, F.linear(F.linear(x, weight), down_weight, bias), rtol=1e-6, atol=1e-6
    )

    with torch.device(device):
        vocab = VocabParallelEmbedding(65, 4, layer_config=config, init_weights=False)
        qkv = QKVParallelLinear(4, 2, 2, 2, layer_config=config, bias=False)
    attach_parameter_loaders(vocab, device=device, dtype=torch.float32)
    vocab_weight = torch.arange(260, dtype=torch.float32, device=device).reshape(65, 4) / 32
    load_parameter_weight(vocab.weight, TensorWeightHandle("weight", vocab_weight))
    ids = torch.tensor([0, 31, 63, 64], device=device)
    torch.testing.assert_close(vocab(ids), F.embedding(ids, vocab_weight), rtol=0, atol=0)
    projections = [weight[:4] + offset for offset in (0, 1, 2)]
    for name, projection_weight in zip(("q", "k", "v"), projections):
        load_parameter_weight(qkv.weight, TensorWeightHandle(name, projection_weight), name)
    local_projection = qkv(x)
    expected_projection = torch.cat(
        [
            F.linear(x, projection_weight).chunk(tp.world_size, dim=-1)[tp.rank_in_group]
            for projection_weight in projections
        ],
        dim=-1,
    )
    torch.testing.assert_close(local_projection, expected_projection, rtol=1e-6, atol=1e-6)

    sequence = meshes["denoiser"].get_group("sp")
    with torch.device(device):
        projection = LinearBase(4, 8, layer_config=config, bias=False, sequence_group=sequence)
    load_parameter_weight(projection.weight, TensorWeightHandle("weight", weight))
    local = x + rank
    workspace = torch.empty(
        local.numel() * sequence.world_size * local.element_size(), dtype=torch.uint8, device=device
    )
    result = projection.forward_sequence_parallel(local, workspace)
    global_x = torch.cat([x + member for member in sequence.ranks])
    torch.testing.assert_close(result, F.linear(global_x, weight), rtol=1e-6, atol=1e-6)
    environment.close()


@pytest.mark.parametrize("backend", ["gloo", pytest.param("nccl", marks=pytest.mark.gpu)])
def test_component_collectives_and_loaded_linears(tmp_path: Path, backend: str):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_run_groups, (rendezvous, backend), nprocs=4, join=True)
