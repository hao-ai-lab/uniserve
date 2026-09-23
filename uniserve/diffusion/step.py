"""Compose one numerical denoiser invocation.

Includes the solver update and PP feedback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from uniserve.tensors import TensorOutput

from .schedule import Schedule

if TYPE_CHECKING:
    from uniserve.model import Denoiser, LatentInput

InputT = TypeVar("InputT")
SizeT = TypeVar("SizeT")


def advance_(
    denoiser: Denoiser,
    latents: Mapping[str, tuple[LatentInput, ...]],
    predictions: Mapping[str, tuple[TensorOutput | torch.Tensor | None, ...]],
    schedules: Mapping[str, Schedule],
    index: int,
) -> Mapping[str, tuple[torch.Tensor, ...]]:
    """Apply one solver update to every sample from its prediction.

    ``latents`` holds each modality's samples at their current timesteps and
    ``predictions`` the aligned predictions for evaluation ``index``: a
    ``TensorOutput`` whose layout locates a sample shard, a tensor covering
    the whole sample, or ``None`` where this rank holds no prediction (a
    pipeline stage before the last). Samples are updated in place toward
    ``schedule.timesteps[index + 1]``, and prediction storage may be consumed
    by the solver. Returns the updated sample tensors.
    """
    names = denoiser.modalities
    if set(latents) != set(names) or set(schedules) != set(names):
        raise ValueError(
            "denoising inputs and schedules must cover the declared modalities"
        )
    if any(
        not 0 <= index < schedule.num_steps for schedule in schedules.values()
    ):
        raise ValueError(
            "denoising requires an evaluation index within every schedule"
        )
    if set(predictions) != set(names):
        raise ValueError(
            "denoising predictions must cover the declared modalities"
        )

    samples = {}
    for name in names:
        schedule = schedules[name]
        values = latents[name]
        if len(predictions[name]) != len(values):
            raise ValueError(
                "denoising predictions must align with the input samples"
            )
        samples[name] = tuple(value.tensor for value in values)
        for latent, prediction in zip(values, predictions[name], strict=True):
            if prediction is None:
                continue
            sample = latent.tensor
            if isinstance(prediction, TensorOutput):
                # A caller may provide the complete sample or borrow precisely
                # its local numerical shard. The output layout distinguishes
                # it.
                if tuple(sample.shape) == prediction.layout.shape:
                    sample = sample[prediction.layout.local_slice]
                prediction = prediction.tensor
            if sample.shape != prediction.shape:
                raise ValueError(
                    "prediction shard and sample storage have incompatible "
                    "shapes"
                )
            denoiser.solver.step_(
                prediction,
                sample,
                latent.timestep,
                schedule.timesteps[index + 1],
                sigma=schedule.sigmas[index],
                next_sigma=schedule.sigmas[index + 1],
            )
    return samples


@dataclass(frozen=True, slots=True)
class DenoisingStep(Generic[InputT, SizeT]):
    """Borrow one trajectory's inputs, schedules and numerical storage.

    One call evaluates the denoiser, applies the solver update with
    :func:`advance_` and, under pipeline parallelism, returns the updated
    samples from the last stage to the first. Prediction storage may be
    consumed by the solver. The returned samples are the same supplied
    tensors; callers retain every backing through replay and decide whether
    to commit request progress after this computation.
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
        if any(
            not 0 <= index < schedule.num_steps
            for schedule in self.schedules.values()
        ):
            raise ValueError(
                "denoising requires an evaluation index within every schedule"
            )
        predictions = self.denoiser(
            self.inputs,
            state=self.state,
            constants=self.constants,
            workspace=self.workspace,
        )
        samples = advance_(
            self.denoiser,
            self.inputs.latents,
            predictions,
            self.schedules,
            index,
        )

        # With pipeline parallelism the last stage holds the updated samples;
        # it sends them back to rank 0 so every stage agrees on the result.
        mesh = self.denoiser.mesh
        pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
        if pipeline.size > 1:
            for name in self.denoiser.modalities:
                for sample in samples[name]:
                    if pipeline.rank == pipeline.size - 1:
                        pipeline.send(sample, dst=0)
                    elif pipeline.rank == 0:
                        pipeline.recv(src=pipeline.size - 1, out=sample)
        return samples
