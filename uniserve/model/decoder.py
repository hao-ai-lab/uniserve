"""Homogeneous image reconstruction over shared numerical decoders."""

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.vae.patch import PatchAutoencoder, RGBDecoder


class ImageDecoder(nn.Module):
    """Batch matching raster sizes.

    While preserving the caller's sample order.
    """

    def __init__(self, decoder: PatchAutoencoder | RGBDecoder):
        super().__init__()
        self.decoder = decoder

    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        sizes: tuple[image.Config, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(latents) != len(sizes):
            raise ValueError("image latents and raster sizes must align")
        if not latents:
            return ()

        results = [None] * len(latents)
        groups = {}
        for index, (latent, size) in enumerate(
            zip(latents, sizes, strict=True)
        ):
            groups.setdefault((size, latent.dtype, latent.device), []).append(
                index
            )

        for (size, _, _), indices in groups.items():
            pixels = self.decoder.decode(
                torch.stack(tuple(latents[index] for index in indices)), size
            )
            if pixels.shape != (len(indices), 3, size.height, size.width):
                raise ValueError(
                    "image decoder output must match the declared NCHW raster"
                )
            for index, value in zip(indices, pixels.unbind(0), strict=True):
                results[index] = value
        return tuple(results)
