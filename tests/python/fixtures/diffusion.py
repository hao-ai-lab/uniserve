"""Small numerical diffusion modules for public runner contracts."""

from typing import Literal

import torch
from torch import nn

from uniserve_worker.modeling.batch import DiffusionBatch, TensorOutput
from uniserve_worker.modeling.diffusion import DiffusionMixin
from uniserve_worker.modeling.geometry import MediaShape
from uniserve_worker.modeling.tensors import TensorViews
from uniserve_worker.nn.diffusion.schedule import ScheduleDirection, ScheduleShiftDomain
from uniserve_worker.nn.diffusion.spec import DiffusionSpec, ModalitySpec, ScheduleRule


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
        self.solver = solver
        self.projection = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(0.25)

    def diffusion_spec(self, shape: MediaShape, steps: int) -> DiffusionSpec:
        return DiffusionSpec(
            tuple(
                ModalitySpec(
                    name,
                    (shape.width,),
                    (shape.width,),
                    ScheduleRule(
                        ScheduleDirection.ASCENDING,
                        ScheduleShiftDomain.SIGMA,
                        1.0,
                        timestep="one_minus_sigma",
                    ),
                    "velocity",
                    torch.float32,
                )
                for name in self.modalities
            ),
            steps,
            None,
            1,
            self.solver,
            "input",
            "identity",
        )

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
