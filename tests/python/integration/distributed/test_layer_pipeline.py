"""Iterative layer pipelines preserve row ownership and independent request state."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve_worker.nn.parallel import ParallelConfig, SequenceParallel
from uniserve_worker.nn.parallel_pipeline import LayerPipeline
from uniserve_worker.runtime.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

pytestmark = pytest.mark.integration


def _run_pipeline(rank: int, rendezvous: str, stages: int) -> None:
    environment = init_distributed_environment(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device=torch.device("cpu"),
        backend="gloo",
        init_method=rendezvous,
    )
    mesh = initialize_model_parallel(
        environment,
        {
            "denoiser": (
                (3, 2, 1, 0),
                ParallelConfig(
                    pipeline_parallel_size=stages,
                    sequence_parallel=SequenceParallel("ulysses", (4 // stages,)),
                ),
            )
        },
    )["denoiser"]
    pipeline = LayerPipeline(mesh.get_group("pp"), 7)
    torch.manual_seed(191)
    # Seven unequal nonlinear layers expose missing/repeated layer ranges.
    weights = torch.randn(7, 8, 8, dtype=torch.float64) / 8
    initial = torch.randn(2, 16, 8, dtype=torch.float64)
    local_rows = 16 // mesh.size("sp")
    offset = mesh.coord("sp") * local_rows
    states = initial[:, offset : offset + local_rows].clone()
    hidden = torch.empty_like(states[0])
    for step in range(4):
        # Two interleaved requests share stage scratch and keep separate latents.
        for request in range(2):
            if pipeline.first:
                hidden.copy_(states[request])
            pipeline.receive_activation(hidden)
            for layer in pipeline.layers:
                hidden = torch.tanh(hidden @ weights[layer] + layer / 16)
            pipeline.send_activation(hidden)
            if pipeline.last:
                states[request].add_(hidden, alpha=(step + 1) / 64)
            pipeline.feedback((states[request],))

    reference = initial.clone()
    for step in range(4):
        value = reference.clone()
        for layer in range(7):
            value = torch.tanh(value @ weights[layer] + layer / 16)
        reference.add_(value, alpha=(step + 1) / 64)
    if pipeline.first or pipeline.last:
        torch.testing.assert_close(
            states, reference[:, offset : offset + local_rows], rtol=0, atol=0
        )
    environment.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("stages", [2, 4])
def test_layer_pipeline_recurrence_preserves_interleaved_requests(tmp_path: Path, stages: int):
    mp.spawn(_run_pipeline, ((tmp_path / "rendezvous").as_uri(), stages), nprocs=4, join=True)
