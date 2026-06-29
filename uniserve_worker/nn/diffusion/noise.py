"""Latent noise initialization."""
from __future__ import annotations

import torch

__all__ = [
    'init_latent',
]


def init_latent(
    shape: tuple[int, ...] | list[int],
    *,
    rng: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    latent = torch.randn(tuple(shape), generator=rng, device=device, dtype=dtype)
    if isinstance(scale, torch.Tensor):
        return latent * scale.to(device=latent.device, dtype=latent.dtype)
    s = float(scale)
    return latent if s == 1.0 else latent * s
