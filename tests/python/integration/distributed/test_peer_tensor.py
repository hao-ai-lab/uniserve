"""Runtime-owned peer tensors preserve ordered values and captured reuse."""

from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_peer_tensor(rank: int, rendezvous: str, world_size: int) -> None:
    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=device,
        backend="nccl",
        init_method=rendezvous,
    )
    ranks = tuple(reversed(range(world_size)))
    mesh = initialize_model_parallel(
        environment,
        {
            "model": (
                ranks,
                ParallelConfig(sequence_parallel=SequenceParallel("ring", (world_size,))),
            )
        },
    )["model"]
    group = mesh.get_group("cp")
    workspace = environment.peer_tensor(
        group, (123, 7, 128), dtype=torch.bfloat16, name="rows", row_multiple=64
    )
    assert workspace.local.shape[0] >= 123 and workspace.local.shape[0] % 64 == 0
    sync_input = torch.zeros(1, dtype=torch.int32, device=device)
    sync_output = torch.empty(world_size, dtype=torch.int32, device=device)
    output = torch.empty_like(workspace.global_tensor)
    workspace.local.fill_(rank)

    def execute():
        workspace.local.add_(1)
        workspace.fence(sync_input, sync_output)
        output.copy_(workspace.global_tensor)
        workspace.fence(sync_input, sync_output)

    def check_result(increments):
        expected = torch.tensor(ranks, dtype=torch.bfloat16, device=device) + increments
        expected = expected.repeat_interleave(workspace.local.shape[0])
        expected = expected[:, None, None].expand_as(output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    with torch.inference_mode():
        execute()
        check_result(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            execute()
        for increments in (2, 3):
            graph.replay()
            check_result(increments)
        graph.reset()
    torch.cuda.synchronize()
    environment.close()


@pytest.mark.parametrize("world_size", [2, 4])
def test_peer_tensor_preserves_ordered_rows_and_graph_reuse(tmp_path: Path, world_size: int):
    mp.spawn(
        _run_peer_tensor,
        ((tmp_path / "rendezvous").as_uri(), world_size),
        nprocs=world_size,
        join=True,
    )
