"""Compose one numerical denoiser invocation with solver update and PP feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from .schedule import Schedule

if TYPE_CHECKING:
    from uniserve.model import Denoiser

InputT = TypeVar("InputT")
SizeT = TypeVar("SizeT")


@dataclass(frozen=True, slots=True)
class DenoisingStep(Generic[InputT, SizeT]):
    """Borrow one trajectory's inputs, schedules and numerical storage.

    Prediction storage may be consumed by the solver. The returned samples are
    the same supplied tensors; callers retain every backing through replay and
    decide whether to commit request progress after this computation.
    """

    denoiser: Denoiser[InputT, SizeT]
    inputs: InputT
    schedules: Mapping[str, Schedule]
    state: Mapping[str, torch.Tensor]
    constants: Mapping[str, torch.Tensor]
    workspace: Mapping[str, torch.Tensor]

    @torch.inference_mode()
    def __call__(self) -> Mapping[str, tuple[torch.Tensor, ...]]:
        index = self.inputs.step_index
        names = self.denoiser.modalities
        if set(self.inputs.latents) != set(names) or set(self.schedules) != set(names):
            raise ValueError("denoising inputs and schedules must cover the declared modalities")
        if any(not 0 <= index < schedule.num_steps for schedule in self.schedules.values()):
            raise ValueError("denoising requires an evaluation index within every schedule")

        predictions = self.denoiser(
            self.inputs, state=self.state, constants=self.constants, workspace=self.workspace
        )
        if set(predictions) != set(names):
            raise ValueError("denoising predictions must cover the declared modalities")

        samples = {}
        for name in names:
            schedule = self.schedules[name]
            values = self.inputs.latents[name]
            if len(predictions[name]) != len(values):
                raise ValueError("denoising predictions must align with the input samples")
            samples[name] = tuple(value.tensor for value in values)
            for latent, prediction in zip(values, predictions[name], strict=True):
                if prediction is None:
                    continue
                sample = latent.tensor
                # A caller may provide the complete sample or borrow precisely
                # its local numerical shard. The output layout distinguishes it.
                if tuple(sample.shape) == prediction.layout.shape:
                    sample = sample[prediction.layout.local_slice]
                if sample.shape != prediction.tensor.shape:
                    raise ValueError("prediction shard and sample storage have incompatible shapes")
                self.denoiser.solver.step_(
                    prediction.tensor,
                    sample,
                    latent.timestep,
                    schedule.timesteps[index + 1],
                    sigma=schedule.sigmas[index],
                    next_sigma=schedule.sigmas[index + 1],
                )
        # With pipeline parallelism the last stage holds the updated samples;
        # it sends them back to rank 0 so every stage agrees on the result.
        mesh = self.denoiser.mesh
        pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        if pipeline.size > 1:
            for name in names:
                for sample in samples[name]:
                    if pipeline.rank == pipeline.size - 1:
                        pipeline.send(sample, dst=0)
                    elif pipeline.rank == 0:
                        pipeline.recv(src=pipeline.size - 1, out=sample)
        return samples
