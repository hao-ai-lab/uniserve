"""Small numerical diffusion modules for public runner contracts."""

from typing import Literal

import torch
from torch import nn

from uniserve.model.batch import DiffusionBatch, TensorOutput
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.tensors import TensorViews
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver, EulerSolver


class LinearDenoiser(DiffusionMixin, nn.Module):
    """Apply a learned scalar projection to each named latent independently."""

    def __init__(
        self,
        modalities: tuple[str, ...] = ("image",),
        *,
        solver: Literal["euler", "clean_sample_euler"] = "clean_sample_euler",
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.solver = CleanSampleEulerSolver() if solver == "clean_sample_euler" else EulerSolver()
        self.projection = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(0.25)

    def forward_diffusion(
        self,
        batch: DiffusionBatch,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
    ) -> TensorOutput:
        return TensorOutput(
            {
                name: tuple(
                    self.projection(sample.unsqueeze(-1)).squeeze(-1) + constants["offset"]
                    for sample in batch.latents[name]
                )
                for name in self.modalities
            }
        )
