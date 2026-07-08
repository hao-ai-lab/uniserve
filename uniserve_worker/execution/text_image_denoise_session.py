"""Session object for one text-image denoise step."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..contracts.outputs import DenoiseOutput
from ..foundation.errors import invalid_descriptor
from ..nn.diffusion import euler_step

__all__ = [
    "TextImageDenoiseSession",
]


class TextImageDenoiseSession:
    """Owns branch prediction and latent update semantics for one denoise step."""

    def __init__(
        self,
        model: Any,
        step: Any,
        *,
        combine_velocity: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
        accept_update: Callable[[Any, Any, torch.Tensor], None],
    ) -> None:
        self.model = model
        self.step = step
        self._combine_velocity = combine_velocity
        self._accept_update = accept_update

    def prepare_step(self, op: Mapping[str, Any] | None = None) -> Any:
        del op
        return self.step

    def predict_velocity(self, branch: str) -> torch.Tensor:
        velocity = self.model.predict_velocity(
            self.step,
            self.step.t,
            self.step.latent,
            branch,
        )
        if not isinstance(velocity, torch.Tensor) or velocity.shape != self.step.latent.shape:
            raise invalid_descriptor(f"{branch} velocity must be a tensor matching the denoise latent")
        return velocity

    def apply_update(self, velocities: Mapping[str, torch.Tensor]) -> DenoiseOutput:
        velocity = self._combine_velocity(self.step, velocities)
        updated = euler_step(self.step.latent, velocity, self.step.t, self.step.t_next)
        self._accept_update(self.model, self.step, updated)
        done = self.step.step_index + 1 >= self.step.total_steps
        return DenoiseOutput(
            req_id=self.step.req_id,
            denoise_done=done,
            num_steps_done=self.step.step_index + 1,
        )

    def release(self) -> None:
        release = getattr(self.step, "release", None)
        if callable(release):
            release()
