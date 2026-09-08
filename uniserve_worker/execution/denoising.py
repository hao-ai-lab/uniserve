"""One numerical diffusion step over model-bound predictions and solver state."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..nn.diffusion.integrator import clean_sample_euler_step_
from ..nn.diffusion.schedule import DiffusionSchedule


@dataclass(frozen=True, slots=True)
class DenoisingStep:
    """Bind one prediction, FP32 clean-sample update, and distributed feedback.

    Prediction buffers are disposable scratch. State and schedule tensors must
    retain their addresses through execution and any graph replay. A pipeline
    stage without output heads returns None and participates only in feedback.
    """

    predict: Callable[[], tuple[torch.Tensor, ...] | None]
    samples: tuple[torch.Tensor, ...]
    feedback: Callable[[tuple[torch.Tensor, ...]], None]
    schedule: DiffusionSchedule
    index: int

    @torch.inference_mode()
    def __call__(self) -> tuple[torch.Tensor, ...]:
        predictions = self.predict()
        if predictions is not None:
            for modality, (sample, prediction) in enumerate(
                zip(self.samples, predictions, strict=True)
            ):
                clean_sample_euler_step_(
                    sample,
                    prediction,
                    self.schedule.timesteps[modality][self.index],
                    self.schedule.sigmas[modality][self.index],
                    self.schedule.sigmas[modality][self.index + 1],
                )
        self.feedback(self.samples)
        return self.samples
