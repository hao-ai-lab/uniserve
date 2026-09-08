"""Diffusion integration primitives."""

from __future__ import annotations

import torch

__all__ = [
    "euler_step",
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
