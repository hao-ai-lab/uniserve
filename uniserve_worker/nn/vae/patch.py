"""Image autoencoders over spatial patch sequences."""

from __future__ import annotations

import torch
from torch import nn

from ...modeling.components import Call
from ...modeling.geometry import MediaShape
from ...modeling.resources import TensorNeeds, TensorSchema
from ...modeling.tensors import ImageRange
from ..vision.patching import patchify_batch, unpatchify_batch
from .autoencoder import AutoEncoder


class PatchAutoencoder(nn.Module):
    """VAE posterior sampling and reconstruction in latent patch coordinates.

    The registered autoencoder retains its native posterior and decoder. The
    patch codec supplies the geometry and dtype of the diffusion representation.
    """

    value_range = ImageRange.UNIT

    def __init__(
        self,
        autoencoder: AutoEncoder,
        *,
        patch_size: int,
        downsample: int,
        channels: int,
        latent_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.patch_size = patch_size
        self.downsample = downsample
        self.channels = channels
        self.latent_dtype = latent_dtype

    def tensor_specs(self, call: Call, shape: MediaShape) -> TensorNeeds:
        """Declare latent patch rows or the reconstructed pixel representation."""

        if call is Call.ENCODE_LATENT:
            rows = shape.height // self.downsample * (shape.width // self.downsample)
            width = self.patch_size**2 * self.channels
            return TensorNeeds(outputs={"latents": TensorSchema((rows, width), self.latent_dtype)})
        if call is not Call.DECODE_IMAGE:
            raise ValueError("patch autoencoding requires latent encoding or image decoding")
        return TensorNeeds(
            outputs={
                "image": TensorSchema((3, shape.height, shape.width), next(self.parameters()).dtype)
            }
        )

    def encode(
        self, pixels: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Sample the native posterior, crop to complete patches, and flatten."""

        dtype = next(self.parameters()).dtype
        latents = self.autoencoder.encode(pixels.to(dtype), generator)
        height = pixels.shape[-2] // self.downsample * self.patch_size
        width = pixels.shape[-1] // self.downsample * self.patch_size
        return patchify_batch(latents[:, :, :height, :width], self.patch_size).to(self.latent_dtype)

    def decode(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Restore spatial VAE latents and reconstruct unit-range pixels."""

        rows = height // self.downsample * (width // self.downsample)
        patches = latents.reshape(-1, rows, self.patch_size**2 * self.channels)
        images = unpatchify_batch(
            patches,
            self.patch_size,
            height=height // self.downsample * self.patch_size,
            width=width // self.downsample * self.patch_size,
            channels=self.channels,
        )
        decoded = self.autoencoder.decode(images.to(next(self.parameters()).dtype))
        # Preserve the VAE's arithmetic dtype and order before pixel quantization.
        return (decoded * 0.5 + 0.5).clamp(0, 1)


class RgbDecoder(nn.Module):
    """Restore signed RGB patch values without a learned autoencoder."""

    value_range = ImageRange.SIGNED_UNIT

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def tensor_specs(self, call: Call, shape: MediaShape) -> TensorNeeds:
        """Declare signed RGB pixels in the supplied latent representation."""

        if call is not Call.DECODE_IMAGE:
            raise ValueError("RGB patch reconstruction requires image decoding")
        if shape.dtype is None:
            raise ValueError("RGB decoding requires the input dtype in its numerical shape")
        return TensorNeeds(
            outputs={"image": TensorSchema((3, shape.height, shape.width), shape.dtype)}
        )

    def decode(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        rows = height // self.patch_size * (width // self.patch_size)
        patches = latents.reshape(-1, rows, self.patch_size**2 * 3)
        return unpatchify_batch(patches, self.patch_size, height=height, width=width, channels=3)
