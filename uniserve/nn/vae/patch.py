"""Canonical patch serialization of image latents."""

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.functional import patchify, unpatchify

from .decoder import LatentDecoder
from .encoder import LatentEncoder


class PatchAutoencoder(nn.Module):
    """Serialize a latent codec's image latents as canonical patch rows.

    ``encoder`` samples and normalizes the latent of NCHW pixels and
    ``decoder`` restores pixels from it; their normalization defines the
    latent convention. ``downsample`` is the output-pixel stride of one
    latent patch token, and patch rows hold ``latent_dtype`` values. Inputs
    reach each network in the dtype of its first parameter. The posterior
    generator and every input tensor belong to the caller.
    """

    value_range = (0.0, 1.0)

    def __init__(
        self,
        encoder: LatentEncoder,
        decoder: LatentDecoder,
        *,
        patch_size: int,
        latent_channels: int,
        latent_dtype: torch.dtype,
        downsample: int,
    ):
        super().__init__()
        if any(
            type(value) is not int or value < 1
            for value in (patch_size, latent_channels, downsample)
        ):
            raise ValueError("latent patch dimensions must be positive")
        self.encoder, self.decoder = encoder, decoder
        self.patch_size, self.latent_channels = patch_size, latent_channels
        self.latent_dtype, self.downsample = latent_dtype, downsample

    def encode(
        self, pixels: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample NCHW pixels into normalized latent patch rows."""
        if pixels.ndim != 4:
            raise ValueError("latent encoding requires NCHW pixels")

        dtype = next(self.encoder.parameters()).dtype
        latents = self.encoder(pixels.to(dtype), generator=generator)

        # Trim to complete latent patches before serialization. Trimming
        # follows sampling, so the posterior draws the reference's full
        # latent from ``generator``.
        height = pixels.shape[-2] // self.downsample * self.patch_size
        width = pixels.shape[-1] // self.downsample * self.patch_size
        return self.patchify(latents[:, :, :height, :width]).to(
            self.latent_dtype
        )

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return patchify(latents, patch_size=self.patch_size)

    def unpatchify(
        self, patches: torch.Tensor, size: image.Config
    ) -> torch.Tensor:
        latent_size = image.Config(
            size.height // self.downsample * self.patch_size,
            size.width // self.downsample * self.patch_size,
        )
        return unpatchify(
            patches,
            latent_size,
            patch_size=self.patch_size,
            channels=self.latent_channels,
        )

    def decode(self, patches: torch.Tensor, size: image.Config) -> torch.Tensor:
        """Restore patch rows to clamped [0, 1] pixels of the given image
        size.
        """  # noqa: D205
        latents = self.unpatchify(patches, size)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)

        dtype = next(self.decoder.parameters()).dtype
        pixels = self.decoder(latents.to(dtype))
        return (pixels * 0.5 + 0.5).clamp(0, 1)


class RGBDecoder(nn.Module):
    """Restore signed RGB pixels from canonical spatial patch rows."""

    value_range = (-1.0, 1.0)

    def __init__(self, patch_size: int):
        super().__init__()
        if type(patch_size) is not int or patch_size < 1:
            raise ValueError("RGB patch size must be positive")
        self.patch_size = patch_size

    def decode(self, patches: torch.Tensor, size: image.Config) -> torch.Tensor:
        pixels = unpatchify(
            patches, size, patch_size=self.patch_size, channels=3
        )
        return pixels.unsqueeze(0) if pixels.ndim == 3 else pixels
