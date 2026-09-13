"""One declared diffusion evaluation, solver update, and numerical feedback."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, fields, is_dataclass

import torch

from ..modeling.batch import DiffusionBatch
from ..modeling.diffusion import DiffusionMixin
from ..modeling.geometry import MediaShape
from ..modeling.tensors import TensorViews
from ..nn.diffusion.integrator import clean_sample_euler_step_, euler_step
from ..nn.diffusion.schedule import (
    DiffusionSchedule,
    ScheduleDirection,
    x_pred_to_velocity,
)


def denoising_batch(
    model: DiffusionMixin,
    shape: MediaShape,
    tensors: TensorViews,
    schedule: DiffusionSchedule,
    index: int,
) -> DiffusionBatch:
    """Bind a single sequence's named latent views to a declared schedule row."""

    steps = schedule.sigmas[0].numel() - 1
    spec = model.diffusion_spec(shape, steps)
    if not 0 <= index < steps:
        raise ValueError("denoising index is outside the supplied schedule")
    if len(schedule.timesteps) != len(spec.modalities):
        raise ValueError("denoising schedule must follow the declared modality order")
    return DiffusionBatch(
        {item.name: (tensors[item.name],) for item in spec.modalities},
        (shape,),
        timesteps={
            item.name: (schedule.timesteps[modality][index],)
            for modality, item in enumerate(spec.modalities)
        },
        conditioning={"text": (tensors["text_condition"],)} if "text_condition" in tensors else {},
        ladder_index=index,
    )


def numerical_signature(value: object) -> Hashable:
    """Identify graph-visible geometry and fixed tensor backing without contents.

    Numerical dataclasses and mappings may be rebuilt between calls. Their
    values and borrowed tensor addresses determine reuse, not Python identity.
    Mutable tensor contents are read by replay and never copied to the host.
    """

    if isinstance(value, torch.Tensor):
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            value.data_ptr(),
        )
    if isinstance(value, Mapping):
        return tuple((key, numerical_signature(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(numerical_signature(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value), tuple(
            (field.name, numerical_signature(getattr(value, field.name))) for field in fields(value)
        )
    if isinstance(value, Hashable):
        return value
    raise TypeError(f"unsupported numerical graph value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class DenoisingStep:
    """Compose the public diffusion call with its declared solver arithmetic.

    Predictions may borrow scratch and are disposable after integration. Latent,
    constant, scratch, and schedule backing must survive graph replay. A stage
    without output heads returns None and participates in pipeline feedback.
    The batch's modulation index and timesteps must match the supplied schedule.
    """

    model: DiffusionMixin
    batch: DiffusionBatch
    state: TensorViews
    constants: TensorViews
    scratch: TensorViews
    schedule: DiffusionSchedule

    @property
    def samples(self) -> tuple[torch.Tensor, ...]:
        """Borrow latent rows in the declared modality and sequence order."""

        return tuple(sample for rows in self.batch.latents.values() for sample in rows)

    @torch.inference_mode()
    def __call__(self) -> tuple[torch.Tensor, ...]:
        index = self.batch.ladder_index
        steps = self.schedule.sigmas[0].numel() - 1
        if index is None or not 0 <= index < steps:
            raise ValueError("denoising requires a valid supplied schedule index")
        spec = self.model.diffusion_spec(self.batch.shapes[0], steps)
        if tuple(item.name for item in spec.modalities) != tuple(self.batch.latents):
            raise ValueError("denoising inputs must follow the declared modality order")
        predictions = self.model.forward_diffusion(
            self.batch, state=self.state, constants=self.constants, scratch=self.scratch
        )
        if tuple(predictions.values) != tuple(self.batch.latents):
            raise ValueError("denoising predictions must follow the input modality order")
        for modality, item in enumerate(spec.modalities):
            for sample, prediction in zip(
                self.batch.latents[item.name], predictions.values[item.name], strict=True
            ):
                if prediction is None:
                    continue
                timestep = self.schedule.timesteps[modality][index]
                if spec.solver == "clean_sample_euler":
                    if item.prediction != "velocity":
                        raise ValueError("clean-sample Euler requires velocity predictions")
                    clean_sample_euler_step_(
                        sample,
                        prediction,
                        timestep,
                        self.schedule.sigmas[modality][index],
                        self.schedule.sigmas[modality][index + 1],
                    )
                elif spec.solver == "euler":
                    if item.prediction == "sample":
                        prediction = x_pred_to_velocity(prediction, sample, timestep)
                    if item.schedule.timestep == "one_minus_sigma":
                        next_timestep = 1.0 - self.schedule.sigmas[modality][index + 1]
                    elif index + 1 < steps:
                        next_timestep = self.schedule.timesteps[modality][index + 1]
                    else:
                        terminal = float(item.schedule.direction is ScheduleDirection.ASCENDING)
                        next_timestep = timestep.new_tensor(terminal)
                    sample.copy_(euler_step(sample, prediction, timestep, next_timestep))
                else:
                    raise ValueError(f"unsupported diffusion solver {spec.solver!r}")
        samples = self.samples
        pipeline = self.model.diffusion_pipeline
        if pipeline is not None:
            pipeline.feedback(samples)
        return samples
