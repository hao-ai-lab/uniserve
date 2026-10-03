"""A scalar numerical denoiser for schedule and execution-lifetime contracts."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from uniserve.diffusion import (
    CleanSampleEulerSolver,
    EulerSolver,
    LinearGrid,
)
from uniserve.model import Denoiser, DenoiserInput
from uniserve.nn import Linear
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput


@dataclass(frozen=True)
class Size:
    width: int
    offset: float = 2.0


class LinearDenoiser(Denoiser[DenoiserInput[Size], Size]):
    """Predict an affine velocity independently for every latent modality."""

    def __init__(
        self,
        modalities=("image",),
        *,
        solver: Literal["euler", "clean_sample_euler"] = "clean_sample_euler",
    ):
        super().__init__(
            modalities=modalities,
            prediction_dtype=torch.float32,
            solver=CleanSampleEulerSolver()
            if solver == "clean_sample_euler"
            else EulerSolver(),
            # Distinct default shifts give each modality its own schedule.
            grids={
                name: LinearGrid(
                    1 + index * 2, direction="ascending", shift_domain="time"
                )
                for index, name in enumerate(modalities)
            },
        )
        self.offset_scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.projection = Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(0.25)

    def latent_shape(self, modality, size):
        if modality not in self.modalities:
            raise ValueError("unknown latent modality")
        return (size.width,)

    def noise_shape(self, modality, size):
        return self.latent_shape(modality, size)

    def prepare_latents(self, sizes, *, noise, state, constants, workspace):
        for name in self.modalities:
            state[name].copy_(noise[name])

    def constant_buffers(self, size):
        return {"offset": BufferConfig((), torch.float32)}

    def prepare_constants(self, size, *, out):
        out["offset"].fill_(size.offset)

    def forward(
        self, inputs: DenoiserInput[Size], *, state, constants, workspace
    ):
        result = {}
        for name in self.modalities:
            outputs = []
            for latent in inputs.latents[name]:
                prediction = self.projection(
                    latent.tensor.unsqueeze(-1)
                ).squeeze(-1)
                prediction = (
                    prediction.to(latent.tensor.device)
                    + constants["offset"] * self.offset_scale
                )
                shape = tuple(prediction.shape)
                outputs.append(
                    TensorOutput(
                        prediction,
                        OutputLayout(
                            shape,
                            prediction.dtype,
                            tuple(slice(0, n) for n in shape),
                        ),
                    )
                )
            result[name] = tuple(outputs)
        return result
