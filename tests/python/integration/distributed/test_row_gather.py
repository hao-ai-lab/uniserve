"""Row gathering preserves values and ownership through CUDA Graph replay."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve_worker.execution.bounded_storage import BoundedTensorStorage, TensorSchema
from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _run_gather(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=2,
        device=str(device),
        backend="nccl",
        init_method=rendezvous,
    )
    parallel = ParallelConfig(sequence_parallel=SequenceParallel("ulysses", (2,)))
    meshes = initialize_model_parallel(
        environment,
        {"ordered": ((0, 1), parallel), "reversed": ((1, 0), parallel)},
    )
    try:
        for mesh in meshes.values():
            group = mesh.get_group("sp")
            for dtype in (torch.bfloat16, torch.uint8):
                storage = BoundedTensorStorage.allocate(
                    {
                        "rows": TensorSchema(
                            (8192,), dtype, memory="symmetric", group=group
                        )
                    },
                    device,
                    environment=environment,
                )
                gathered = storage.capacity["rows"]
                local = (torch.arange(4096, device=device) % 31 + rank * 64).to(dtype)
                expected = torch.cat(
                    [
                        (torch.arange(4096, device=device) % 31 + member * 64).to(dtype)
                        for member in group.ranks
                    ]
                )
                group.all_gather_into_tensor(gathered, local)
                torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    group.all_gather_into_tensor(gathered, local)
                local.add_(1)
                graph.replay()
                torch.testing.assert_close(gathered, expected + 1, rtol=0, atol=0)
                graph.reset()
                del gathered, storage
    finally:
        environment.close()
        dist.destroy_process_group()


def test_row_gather_replays_updated_values_in_logical_rank_order(tmp_path):
    mp.spawn(_run_gather, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
