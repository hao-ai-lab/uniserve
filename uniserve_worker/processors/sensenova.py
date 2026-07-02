"""SenseNova-U1 understanding-image preprocessing.

Faithful port of the reference preprocessing used by the SenseNova-U1
pipeline (vLLM-Omni ``_load_image_native``): RGBA→white-composite→RGB,
smart-resize to a grid multiple with pixel-count bounds, ImageNet
normalization, and patchification into flattened per-patch rows the
conv patch-embedder consumes.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import torch
from PIL import Image

from .base import MultimodalProcessor
from .registry import register_processor

__all__ = [
    'SenseNovaImageGeometry',
    'SENSENOVA_IMAGE_GEOMETRY',
    'smart_resize',
    'SenseNovaImageProcessor',
]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SenseNovaImageGeometry:
    """Understanding-image preprocessing geometry.

    ``patch_size``/``downsample_ratio`` mirror the checkpoint vision config;
    the pixel bounds are the reference pipeline's understanding-input policy
    (min 512², max 2048² per image — tighter than the raw config bounds),
    with the per-image budget shrinking when multiple images share a prompt.
    """

    patch_size: int = 16
    downsample_ratio: float = 0.5
    min_pixels: int = 512 * 512
    max_pixels: int = 2048 * 2048
    multi_image_pixel_budget: int = 4096 * 4096

    @property
    def size_factor(self) -> int:
        # H/W must be divisible by one downsampled patch (16 / 0.5 = 32 px).
        return int(self.patch_size // self.downsample_ratio)

    def max_pixels_for(self, num_images: int) -> int:
        return min(self.max_pixels, self.multi_image_pixel_budget // max(1, int(num_images)))


SENSENOVA_IMAGE_GEOMETRY = SenseNovaImageGeometry()


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Rescale so H/W divide ``factor`` and the pixel count lands in bounds."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


@register_processor
class SenseNovaImageProcessor(MultimodalProcessor):
    """Understanding-input preprocessing for the SenseNova-U1 conv embedder."""

    model_architectures = ("SenseNovaU1ForUnifiedGeneration", "NEOChatModel", "neo_chat")

    def __init__(self, geometry: SenseNovaImageGeometry = SENSENOVA_IMAGE_GEOMETRY) -> None:
        self.geometry = geometry

    @staticmethod
    def decode_image_b64(image_b64: str) -> Image.Image:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
            image = image.convert("RGBA")
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            return rgb
        return image.convert("RGB")

    def resize_for_understanding(self, image: Image.Image, *, num_images: int = 1) -> Image.Image:
        geometry = self.geometry
        resized_h, resized_w = smart_resize(
            image.height,
            image.width,
            factor=geometry.size_factor,
            min_pixels=geometry.min_pixels,
            max_pixels=geometry.max_pixels_for(num_images),
        )
        return image.resize((resized_w, resized_h))

    def vit_tensor(self, image: Image.Image) -> torch.Tensor:
        """Normalized CHW float32 tensor of an already-resized image."""
        from torchvision.transforms import functional as TF

        tensor = TF.to_tensor(image.convert("RGB")).to(torch.float32)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def vae_tensor(self, image: Image.Image) -> torch.Tensor:
        raise NotImplementedError("SenseNova-U1 has no VAE; understanding uses vit_tensor only")

    def understanding_patches(
        self, image: Image.Image, *, num_images: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(flattened_patches [N, 3·p²], grid_hw [1, 2])`` for one image."""
        resized = self.resize_for_understanding(image, num_images=num_images)
        pixel_values = self.vit_tensor(resized)
        patch = self.geometry.patch_size
        channels, height, width = pixel_values.shape
        grid_h = height // patch
        grid_w = width // patch
        flattened = (
            pixel_values.view(channels, grid_h, patch, grid_w, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch**2)
        )
        return flattened, torch.tensor([[grid_h, grid_w]], dtype=torch.long)

    def understanding_patches_from_b64(
        self, image_b64: str, *, num_images: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.understanding_patches(self.decode_image_b64(image_b64), num_images=num_images)
