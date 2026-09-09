"""Communicator sums preserve values across mutable graph bucket reuse."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve_worker.nn.collective import collective_scope
from uniserve_worker.nn.parallel import ParallelConfig
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_sum_reduction(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    mesh = initialize_model_parallel(environment, {"model": ((1, 0), ParallelConfig(2))})["model"]
    group = mesh.get_group("tp")
    reductions = environment.sum_reductions()
    stream = torch.cuda.Stream(device=device)
    graphs = []
    try:
        torch.manual_seed(618)
        for dtype in (torch.bfloat16, torch.float16):
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
                expected = (base.float() + (base + 1).float()).to(dtype)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
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
                reference = (
                    (base + iteration * 0.25).float() + (base + iteration * 0.25 + 1).float()
                ).to(dtype)
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
            for graph in graphs:
                graph.reset()
            graphs.clear()

        # The same public communicator retains its full FP32 sum contract.
        value = torch.arange(91, dtype=torch.float32, device=device).view(7, 13) + rank
        with collective_scope(reductions):
            result = group.all_reduce(value)
        expected = torch.arange(91, dtype=torch.float32, device=device).view(7, 13) * 2 + 1
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize(device)
        for graph in graphs:
            graph.reset()
        for reduction in reductions.values():
            reduction.close()
        environment.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_sum_reduction_preserves_values_in_place_and_under_graph_replay(tmp_path: Path):
    mp.spawn(
        _run_sum_reduction,
        ((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
