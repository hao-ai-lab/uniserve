"""Image latent posterior sampling and canonical patch serialization."""

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.functional import patchify, unpatchify

from .layers import DiagonalGaussian


class PatchAutoencoder(nn.Module):
    """Compose numerical encoder/decoder modules with latent patch coordinates.

    downsample is the output-pixel stride of one latent patch token. Scale and
    shift normalize posterior samples before their canonical patch conversion.
    The posterior generator and every input tensor belong to the caller.
    """

    value_range = (0.0, 1.0)

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        posterior: DiagonalGaussian,
        *,
        patch_size: int,
        latent_channels: int,
        latent_dtype: torch.dtype,
        downsample: int,
        scale: float,
        shift: float,
    ):
        super().__init__()
        self.encoder, self.decoder, self.posterior = encoder, decoder, posterior
        if (
            any(
                type(value) is not int or value < 1
                for value in (patch_size, latent_channels, downsample)
            )
            or scale <= 0
        ):
            raise ValueError(
                "latent patch dimensions and scale must be positive"
            )
        self.patch_size, self.latent_channels = patch_size, latent_channels
        self.latent_dtype, self.downsample = latent_dtype, downsample
        self.scale, self.shift = scale, shift

    def encode(
        self, pixels: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample NCHW pixels into normalized latent patch rows."""
        if pixels.ndim != 4:
            raise ValueError("latent encoding requires NCHW pixels")

        dtype = next(self.encoder.parameters()).dtype
        moments = self.encoder(pixels.to(dtype))
        latents = self.scale * (
            self.posterior(moments, generator=generator) - self.shift
        )

        # Trim to complete latent patches before serialization.
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
        latents = latents.to(dtype) / self.scale + self.shift
        pixels = self.decoder(latents)
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
