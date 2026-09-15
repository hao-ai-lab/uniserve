"""Shared FP32 latent normalization before an ordinary decoder call."""

from __future__ import annotations

import torch
from torch import nn


class LatentDecoder(nn.Module):
    """Invert channel statistics and call the composed numerical decoder.

    Mean and standard deviation must broadcast to ``latent_shape``. A None
    extent permits that axis to vary. Derived constants retain real values
    during meta construction; loading moves their backing with the decoder.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        latent_shape: tuple[int | None, ...],
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        super().__init__()
        if not latent_shape or any(
            size is not None and (type(size) is not int or size < 1)
            for size in latent_shape
        ):
            raise ValueError(
                "latent dimensions must be positive integers or None"
            )
        if mean.is_meta or std.is_meta or mean.shape != std.shape:
            raise ValueError("latent statistics require real matching tensors")
        if (
            not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0).all())
        ):
            raise ValueError(
                "latent statistics must be finite with positive standard "
                "deviations"
            )
        self.decoder, self.latent_shape = decoder, latent_shape
        self.register_buffer("mean", mean.float(), persistent=False)
        self.register_buffer("std", std.float(), persistent=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != len(self.latent_shape) or any(
            size is not None and size != actual
            for size, actual in zip(
                self.latent_shape, latents.shape, strict=True
            )
        ):
            raise ValueError(
                f"decoder latent shape must match {self.latent_shape}"
            )
        if (
            latents.device != self.mean.device
            or self.std.device != latents.device
        ):
            raise ValueError(
                "latent values and normalization statistics must share "
                "their device"
            )

        from .spatial import SpatialDecoder

        values = latents.float() * self.std + self.mean
        if isinstance(self.decoder, SpatialDecoder):
            return self.decoder.decode(values, tiled=True)
        return self.decoder(values)
