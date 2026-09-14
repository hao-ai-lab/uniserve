"""Latent noise initialization."""

from __future__ import annotations

import torch

__all__ = [
    "init_latent",
]


def init_latent(
    shape: tuple[int, ...] | list[int],
    *,
    rng: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    source_device: torch.device | str | None = None,
    source_dtype: torch.dtype | None = None,
    scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Sample latent noise, optionally in a distinct source format.

    The generator must belong to the source device. By default the source is
    the requested output device and dtype, preserving direct-sampling behavior.
    Conversion happens before scaling so scaling retains the output format's
    existing arithmetic semantics.
    """
    sample_device = device if source_device is None else source_device
    sample_dtype = dtype if source_dtype is None else source_dtype
    latent = torch.randn(
        tuple(shape),
        generator=rng,
        device=sample_device,
        dtype=sample_dtype,
    ).to(device=device, dtype=dtype)
    if isinstance(scale, torch.Tensor):
        return latent * scale.to(device=latent.device, dtype=latent.dtype)
    s = float(scale)
    return latent if s == 1.0 else latent * s
