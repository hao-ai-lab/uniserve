"""Communicator sums preserve values across mutable graph bucket reuse."""

from uniserve.distributed import DeviceMesh
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from uniserve.runtime._collectives import allocate_peer_reductions
from uniserve.runtime.process_groups import initialize_process_groups
from uniserve.runtime._communication import collective_scope
from uniserve_worker.bootstrap.config import ParallelConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_sum_reduction(rank: int, rendezvous: str, world_size: int) -> None:
    device = torch.device("cuda", rank)
    environment = initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    bound_meshes = {}
    for mesh_name, (mesh_ranks, mesh_parallel) in sorted(({'model': (tuple(reversed(range(world_size))), ParallelConfig(world_size))}).items()):
        topology = DeviceMesh(
            ranks=mesh_ranks,
            shape=tuple(size for _, size in mesh_parallel.dimensions),
            axes=tuple(axis for axis, _ in mesh_parallel.dimensions),
            rank=environment.rank,
        )
        bound_mesh = environment.bind(topology, device=environment.device)
        if environment.rank in mesh_ranks:
            bound_meshes[mesh_name] = bound_mesh
    mesh = bound_meshes['model']
    group = mesh.get_group("tp")
    reductions = allocate_peer_reductions((group,))
    stream = torch.cuda.Stream(device=device)
    graphs = []
    try:
        torch.manual_seed(618)
        for dtype in (torch.bfloat16, torch.float16):
            tolerance = 2 * torch.finfo(dtype).eps
            cases = []
            for rows, width in ((7, 5120), (128, 8192)):
                base = torch.randn((rows, width), dtype=dtype, device=device)
                value = base + rank

                def execute():
                    with collective_scope(reductions):
                        return group.all_reduce(value)

                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    result = execute()
                stream.synchronize()
                expected = sum((base + peer).double() for peer in range(world_size))
                torch.testing.assert_close(
                    result.double(), expected, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(value.double(), expected, rtol=tolerance, atol=tolerance)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    execute()
                graphs.append(graph)
                cases.append((base, value, graph))

            # Alternating row counts exercise reuse of the same bounded peer
            # storage across captures and successive protocol cycles.
            for iteration in range(6):
                base, value, graph = cases[iteration % 2]
                with torch.cuda.stream(stream):
                    value.copy_(base + iteration * 0.25 + rank)
                    graph.replay()
                stream.synchronize()
                reference = sum(
                    (base + iteration * 0.25 + peer).double() for peer in range(world_size)
                )
                torch.testing.assert_close(
                    value.double(), reference, rtol=tolerance, atol=tolerance
                )
            for graph in graphs:
                graph.reset()
            graphs.clear()

        # The same public communicator retains its full FP32 sum contract.
        value = torch.arange(91, dtype=torch.float32, device=device).view(7, 13) + rank
        with collective_scope(reductions):
            result = group.all_reduce(value)
        expected = torch.arange(91, dtype=torch.float32, device=device).view(7, 13) * world_size
        expected += sum(range(world_size))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()
        for reduction in reductions.values():
            reduction.close()
        environment.close()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sum_reduction_preserves_values_in_place_and_under_graph_replay(tmp_path: Path, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices are required")
    mp.spawn(
        _run_sum_reduction,
        ((tmp_path / "rendezvous").as_uri(), world_size),
        nprocs=world_size,
        join=True,
    )
