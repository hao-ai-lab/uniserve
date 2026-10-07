"""Latent decoding through a codec's normalization and decoder."""

from __future__ import annotations

import torch
from torch import nn

from .normalization import LatentNormalization


class LatentDecoder(nn.Module):
    """Denormalize latents and call the composed numerical decoder.

    ``normalization`` maps the normalized latent back to the codec's native
    latent; ``decoder`` maps that to its output, through overlapping tiles
    when it is a ``SpatialDecoder``. Latents must have ``latent_shape``; a
    None extent permits that axis to vary.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        normalization: LatentNormalization,
        latent_shape: tuple[int | None, ...],
    ):
        super().__init__()
        if not latent_shape or any(
            size is not None and (type(size) is not int or size < 1)
            for size in latent_shape
        ):
            raise ValueError(
                "latent dimensions must be positive integers or None"
            )
        self.decoder, self.latent_shape = decoder, latent_shape
        self.normalization = normalization

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
        return self.decoder(self.normalization.denormalize(latents))
