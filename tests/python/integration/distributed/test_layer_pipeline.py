"""Pipeline decoder calls preserve nonlinear recurrence.

They also preserve independent inputs.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.model import TransformerDecoder
from uniserve.nn.attention import (
    AttentionParallelConfig,
    SequenceLengths,
    Ulysses,
    VarlenInput,
)
from uniserve.runtime import initialize_process_groups

pytestmark = pytest.mark.integration


class NonlinearLayer(nn.Module):
    def __init__(self, weight, offset):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.offset = offset

    def forward(self, hidden, residual, positions, attention):
        hidden = hidden if residual is None else hidden + residual
        value = torch.tanh(hidden @ self.weight + self.offset)
        return value, torch.zeros_like(value)


@torch.inference_mode()
def _run(rank, rendezvous, stages):
    with initialize_process_groups(
        rank=rank,
        local_rank=rank,
        world_size=4,
        device="cpu",
        init_method=rendezvous,
    ) as owner:
        mesh = owner.bind(
            DeviceMesh(
                ranks=(3, 2, 1, 0),
                shape=(stages, 4 // stages),
                axes=("pp", "tokens"),
                rank=rank,
            ),
            device="cpu",
        )
        generator = torch.Generator().manual_seed(191)
        weights = (
            torch.randn(7, 8, 8, dtype=torch.float64, generator=generator) / 8
        )
        initial = torch.randn(
            2, 16, 8, dtype=torch.float64, generator=generator
        )
        model = TransformerDecoder(
            nn.Embedding(32, 8, dtype=torch.float64),
            nn.ModuleDict(
                {
                    str(index): NonlinearLayer(weight, index / 16)
                    for index, weight in enumerate(weights)
                }
            ),
            nn.Identity(),
        )
        parallelize_(
            model,
            mesh,
            attention=AttentionParallelConfig(heads=Ulysses("tokens")),
        )
        pipeline = mesh.get_group("pp")
        lengths = SequenceLengths.from_lengths((16,), device="cpu")
        attention = VarlenInput(lengths, lengths, (False,))
        positions = torch.arange(16)
        states = initial.clone()
        for step in range(4):
            for request in range(2):
                value = model(
                    states[request] if pipeline.rank == 0 else None,
                    positions,
                    attention,
                )
                if pipeline.rank + 1 == pipeline.size:
                    states[request].add_(value, alpha=(step + 1) / 64)
                # Solver feedback is ordinary numerical communication; each
                # trajectory retains its own state outside the shared module.
                pipeline.broadcast(
                    states[request], src=pipeline.size - 1, out=states[request]
                )
        reference = initial.clone()
        for step in range(4):
            value = reference.clone()
            for index, weight in enumerate(weights):
                value = torch.tanh(value @ weight + index / 16)
            reference.add_(value, alpha=(step + 1) / 64)
        torch.testing.assert_close(states, reference, rtol=0, atol=0)


@pytest.mark.parametrize("stages", (2, 4))
def test_pipeline_recurrence_preserves_independent_trajectories(
    tmp_path, stages
):
    mp.spawn(
        _run,
        args=((tmp_path / "pipeline").as_uri(), stages),
        nprocs=4,
        join=True,
    )
