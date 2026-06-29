"""BAGEL image preprocessing."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image

from .base import MultimodalProcessor
from .registry import register_processor

__all__ = [
    'BagelGeometry',
    'BAGEL_GEOMETRY',
    'ImageTransform',
    'BagelImageProcessor',
]


@dataclass(frozen=True)
class BagelGeometry:
    """BAGEL's fixed ViT/VAE preprocessing geometry (architecture constants).

    `(max_size, min_size, stride)` for each tower. `BagelConfig` models only the
    ViT image-size/patch, so the min-size and the VAE geometry are not
    config-derivable and are pinned here as the single BAGEL-specific source
    rather than scattered positional literals.
    """

    vit_max_size: int = 980
    vit_min_size: int = 224
    vit_stride: int = 14
    vae_max_size: int = 1024
    vae_min_size: int = 512
    vae_stride: int = 16


BAGEL_GEOMETRY = BagelGeometry()


class _Resize:
    def __init__(self, max_size, min_size, stride, max_pixels):
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels

    def _div(self, value):
        return max(self.stride, int(round(value / self.stride) * self.stride))

    def _scale(self, width, height, scale):
        return self._div(round(width * scale)), self._div(round(height * scale))

    def __call__(self, image: Image.Image) -> Image.Image:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        width, height = image.size
        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._scale(width, height, scale)
        if new_width * new_height > self.max_pixels:
            scale = self.max_pixels / (new_width * new_height)
            new_width, new_height = self._scale(new_width, new_height, scale)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._scale(new_width, new_height, scale)
        return TF.resize(image, (new_height, new_width), InterpolationMode.BICUBIC, antialias=True)


class ImageTransform:
    """Faithful BAGEL resize + normalize transform used by the processor layer."""

    def __init__(self, max_image_size, min_image_size, image_stride, max_pixels=14 * 14 * 9 * 1024):
        self.stride = image_stride
        self.resize_transform = _Resize(max_image_size, min_image_size, image_stride, max_pixels)

    def __call__(self, image: Image.Image):
        from torchvision.transforms import functional as TF

        image = self.resize_transform(image)
        tensor = TF.to_tensor(image)
        return (tensor - 0.5) / 0.5


@register_processor
class BagelImageProcessor(MultimodalProcessor):
    """Faithful BAGEL ViT/VAE preprocessing configuration."""

    model_architectures = ("BagelForUnifiedGeneration", "BAGEL", "bagel")

    def __init__(self, geometry: BagelGeometry = BAGEL_GEOMETRY) -> None:
        self.vit_transform = ImageTransform(
            geometry.vit_max_size, geometry.vit_min_size, geometry.vit_stride
        )
        self.vae_transform = ImageTransform(
            geometry.vae_max_size, geometry.vae_min_size, geometry.vae_stride
        )

    @staticmethod
    def decode_image_b64(image_b64: str) -> Image.Image:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
            image = image.convert("RGBA")
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            return rgb
        return image.convert("RGB")

    def prepare_from_b64(self, image_b64: str) -> Image.Image:
        return self.resize_for_vae(self.decode_image_b64(image_b64))

    def resize_for_vae(self, image: Image.Image) -> Image.Image:
        return self.vae_transform.resize_transform(image)

    def vae_tensor(self, image: Image.Image):
        return self.vae_transform(image)

    def vit_tensor(self, image: Image.Image):
        return self.vit_transform(image)
