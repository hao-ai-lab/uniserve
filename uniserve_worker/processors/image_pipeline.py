"""System image-input stage: decode + declared transforms ahead of model encode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..contracts.model_spec import ImageInputSpec, ImagePatchSpec, ImageTowerSpec
from ..contracts.op_kinds import VAE_ENCODE, VIT_ENCODE
from ..foundation.errors import invalid_descriptor

__all__ = [
    "ImageInputPipeline",
    "PreparedImageInput",
]


@dataclass(frozen=True)
class PreparedImageInput:
    """Typed transform products for one encode op's image payload."""

    pixels: torch.Tensor
    grid: torch.Tensor | None
    image_hw: tuple[int, int]


class ImageInputPipeline:
    """Run one model's declared image transforms ahead of its encode entry points.

    The pipeline owns media decode, the ``InputSpec.images`` transform chains
    (executed through the family processor's math), and device/dtype staging,
    so models receive typed tensors and bounded views only.
    """

    def __init__(
        self,
        images: ImageInputSpec,
        processor: Any,
        *,
        device: Any = None,
    ) -> None:
        self.images = images
        self.processor = processor
        self.device = device
        self.staging_dtype = (
            getattr(torch, images.staging_dtype) if images.staging_dtype else None
        )

    def prepare(self, kind: str, image_b64: str) -> PreparedImageInput:
        """Decode and transform one base64 image payload for encode kind ``kind``."""
        transform = self._transform_for(kind)
        if isinstance(transform, ImagePatchSpec):
            image = self.processor.decode_image_b64(image_b64)
            image_hw = (int(image.height), int(image.width))
            pixels, grid = self.processor.understanding_patches(image)
            return PreparedImageInput(pixels=self._stage(pixels), grid=grid, image_hw=image_hw)
        # Tower chain: decode plus the declared canvas resize, then the
        # per-kind tower transform over the canvas image.
        image = self.processor.prepare_from_b64(image_b64)
        image_hw = (int(image.size[1]), int(image.size[0]))
        pixels = (
            self.processor.vae_tensor(image)
            if kind == VAE_ENCODE
            else self.processor.vit_tensor(image)
        )
        return PreparedImageInput(pixels=self._stage(pixels), grid=None, image_hw=image_hw)

    def _transform_for(self, kind: str) -> ImagePatchSpec | ImageTowerSpec:
        transform = (
            self.images.vit
            if kind == VIT_ENCODE
            else self.images.vae if kind == VAE_ENCODE else None
        )
        if transform is None:
            raise invalid_descriptor(f"model declares no image transform for {kind!r}")
        return transform

    def _stage(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.device is None and self.staging_dtype is None:
            return pixels
        if self.staging_dtype is None:
            return pixels.to(device=self.device)
        return pixels.to(device=self.device, dtype=self.staging_dtype)
