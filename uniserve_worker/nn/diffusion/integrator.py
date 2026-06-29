"""Diffusion integration primitives."""
from __future__ import annotations

import torch

__all__ = [
    'euler_step',
]


def euler_step(z: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
    return z + (t_next - t).to(dtype=z.dtype, device=z.device) * velocity
