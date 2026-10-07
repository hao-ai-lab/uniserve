"""Latent encoding through a codec's posterior and normalization."""

from __future__ import annotations

import torch
from torch import nn

from .layers import DiagonalGaussian
from .normalization import LatentNormalization


class LatentEncoder(nn.Module):
    """Call the composed numerical encoder, sample, then normalize.

    The inverse of ``LatentDecoder``. ``encoder`` maps its native input to
    posterior moments, or directly to latents when ``posterior`` is None; a
    ``SpatialEncoder`` does so through overlapping tiles. The posterior turns
    the moments into native latents and ``normalization`` maps them to the
    normalized latent the codec's reference produces.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        normalization: LatentNormalization,
        posterior: DiagonalGaussian | None = None,
    ):
        super().__init__()
        self.encoder, self.posterior = encoder, posterior
        self.normalization = normalization

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        window: tuple[slice, ...] | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return the normalized latent of ``inputs``.

        ``window`` indexes the encoder output in native latent coordinates
        and keeps only that region, such as the latent frames of an input
        that was padded to the encoder's temporal extent; the posterior then
        samples the kept region alone. ``noise`` is the standard normal draw
        a sampling posterior adds, with the kept mean's shape; without it the
        posterior draws its own from ``generator``.
        """
        values = self.encoder(inputs)
        if window is not None:
            values = values[window]
        if self.posterior is not None:
            values = self.posterior(values, generator=generator, noise=noise)
        elif noise is not None:
            raise ValueError("a deterministic encoder takes no posterior noise")
        return self.normalization.normalize(values)
