"""Numerical diffusion evaluation and solver advancement over borrowed tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.model.batch import DiffusionBatch
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.tensors import TensorViews
from uniserve.nn.diffusion.schedule import DiffusionSchedule


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

    def __post_init__(self) -> None:
        self.model.validate_schedule(self.schedule)

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
        if self.model.modalities != tuple(self.batch.latents):
            raise ValueError("denoising inputs must follow the declared modality order")
        predictions = self.model.forward_diffusion(
            self.batch, state=self.state, constants=self.constants, scratch=self.scratch
        )
        if tuple(predictions.values) != tuple(self.batch.latents):
            raise ValueError("denoising predictions must follow the input modality order")
        for modality, name in enumerate(self.model.modalities):
            for sample, prediction in zip(
                self.batch.latents[name], predictions.values[name], strict=True
            ):
                if prediction is None:
                    continue
                self.model.solver.step(
                    prediction,
                    sample,
                    self.schedule.timesteps[modality][index],
                    self.schedule.timesteps[modality][index + 1],
                    sigma=self.schedule.sigmas[modality][index],
                    next_sigma=self.schedule.sigmas[modality][index + 1],
                )
        samples = self.samples
        pipeline = self.model.diffusion_pipeline
        if pipeline is not None:
            pipeline.feedback(samples)
        return samples
