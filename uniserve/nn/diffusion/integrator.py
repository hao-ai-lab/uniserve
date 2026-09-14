"""Diffusion integration primitives."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from uniserve.nn.diffusion.schedule import x_pred_to_velocity

__all__ = [
    "euler_step",
    "EulerSolver",
    "CleanSampleEulerSolver",
]


def euler_step(
    z: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
) -> torch.Tensor:
    """Advance a latent from ``t`` to ``t_next`` under a predicted flow velocity."""

    return z + (t_next - t).to(dtype=z.dtype, device=z.device) * velocity


def clean_sample_euler_step_(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> None:
    """Update the sample via a clean prediction, reusing velocity as scratch.

    The clean-time coordinate is cast to sample precision before subtraction;
    the sigma ratio remains FP32. Both mutation order and cast location are
    part of this solver's floating-point contract.
    """

    sigma_from_timestep = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
    velocity.mul_(sigma_from_timestep).add_(sample)
    ratio = sigma_next.float() / sigma.float()
    sample.mul_(ratio)
    sample.addcmul_(velocity, 1.0 - ratio)


class EulerSolver(nn.Module):
    """Integrate velocity or clean-sample predictions using the supplied times."""

    def __init__(self, prediction_type: Literal["velocity", "sample"] = "velocity") -> None:
        super().__init__()
        if prediction_type not in {"velocity", "sample"}:
            raise ValueError("Euler prediction must be velocity or sample")
        self.prediction_type = prediction_type

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
        next_sigma: torch.Tensor | None = None,
    ) -> None:
        """Update sample in place, retaining the established time-difference cast."""

        velocity = (
            x_pred_to_velocity(model_output, sample, timestep)
            if self.prediction_type == "sample"
            else model_output
        )
        sample.copy_(euler_step(sample, velocity, timestep, next_timestep))


class CleanSampleEulerSolver(nn.Module):
    """Integrate through a clean prediction, consuming prediction as scratch."""

    prediction_type = "velocity"

    def step(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
        *,
        sigma: torch.Tensor | None = None,
        next_sigma: torch.Tensor | None = None,
    ) -> None:
        """Preserve state-precision clean time and FP32 sigma-ratio arithmetic."""

        if sigma is None or next_sigma is None:
            raise ValueError("clean-sample Euler requires both sigma endpoints")
        clean_sample_euler_step_(sample, model_output, timestep, sigma, next_sigma)
