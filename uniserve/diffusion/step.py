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
    from uniserve.model import Denoiser, DenoiserInput, LatentInput

# Matches the input bound of ``uniserve.model.Denoiser``.
InputT = TypeVar("InputT", bound="DenoiserInput")
SizeT = TypeVar("SizeT")


def advance_(
    denoiser: Denoiser,
    latents: Mapping[str, tuple[LatentInput, ...]],
    predictions: Mapping[str, tuple[TensorOutput | torch.Tensor | None, ...]],
    schedules: Mapping[str, Schedule],
    step: torch.Tensor,
) -> Mapping[str, tuple[torch.Tensor, ...]]:
    """Apply one solver update to every sample from its prediction.

    ``latents`` holds each modality's samples at their current timesteps and
    ``predictions`` the aligned predictions for the evaluation ``step`` names:
    a ``TensorOutput`` whose layout locates a sample shard, a tensor covering
    the whole sample, or ``None`` where this rank holds no prediction (a
    pipeline stage before the last). ``step`` is the evaluation's [1] int64
    device index (``Schedule.step``); samples are updated in place toward
    ``schedule.timesteps[step + 1]``, gathered on the device, so the same
    computation serves every step. Prediction storage may be consumed by the
    solver. Returns the updated sample tensors.
    """
    names = denoiser.modalities
    if set(latents) != set(names) or set(schedules) != set(names):
        raise ValueError(
            "denoising inputs and schedules must cover the declared modalities"
        )
    if set(predictions) != set(names):
        raise ValueError(
            "denoising predictions must cover the declared modalities"
        )
    if step.dtype != torch.int64 or tuple(step.shape) != (1,):
        raise ValueError("denoising requires a [1] int64 step index")

    samples = {}
    for name in names:
        schedule = schedules[name]
        values = latents[name]
        # Endpoints of this evaluation, gathered by the device step as 0-d
        # FP32 values: the same shapes and dtypes the host-indexed entries
        # had, so the solver's type promotion and arithmetic are unchanged.
        # ``timesteps[1:]`` offsets the gather by one to reach step + 1.
        next_timestep = schedule.timesteps[1:].index_select(0, step)[0]
        sigma = schedule.sigmas.index_select(0, step)[0]
        next_sigma = schedule.sigmas[1:].index_select(0, step)[0]
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
                next_timestep,
                sigma=sigma,
                next_sigma=next_sigma,
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
        # The step is device data; ``Schedule.step`` validated it on the host
        # when the caller named the evaluation.
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
            self.inputs.step,
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
