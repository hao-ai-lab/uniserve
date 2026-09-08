"""Image decoding and staging for model-owned processing policy."""

from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision

from uniserve_worker.execution.batch import EncodeMode
from uniserve_worker.foundation.errors import invalid_descriptor
from uniserve_worker.models.inputs import (
    ImageProcessor,
    PatchTransform,
    StrideResize,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Holds normalized image pixels, patch-grid coordinates, and original media dimensions."""

    pixels: torch.Tensor
    grid: torch.Tensor | None
    grid_shape: tuple[int, int] | None
    height: int
    width: int


def prepare_image(
    processor: ImageProcessor,
    kind: EncodeMode,
    encoded: str,
    *,
    device: torch.device,
) -> PreparedImage:
    """Decode, resize, normalize, and stage one encoded image for the selected model tower."""

    image = _decode_rgb(encoded)
    transform = processor.vit if kind is EncodeMode.VISION else processor.vae
    if transform is None:
        raise invalid_descriptor(f"model declares no {kind.value} image transform")

    if isinstance(transform, PatchTransform):
        height, width = image.height, image.width
        resized = _resize_patch_image(image, transform)
        normalized = _normalize(resized, transform.normalization)
        patch = int(transform.patch_size)
        channels, resized_height, resized_width = normalized.shape
        grid_height = resized_height // patch
        grid_width = resized_width // patch
        pixels = (
            normalized.view(channels, grid_height, patch, grid_width, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_height * grid_width, channels * patch * patch)
        )
        grid = torch.tensor([[grid_height, grid_width]], dtype=torch.long)
        return PreparedImage(
            _stage(pixels, processor, device),
            grid.to(device=device, non_blocking=True),
            (grid_height, grid_width),
            height,
            width,
        )

    canvas = image
    if processor.vae is not None:
        canvas = _resize_stride(canvas, processor.vae.resize)
    height, width = canvas.height, canvas.width
    tower_image = _resize_stride(canvas, transform.resize)
    pixels = _normalize(tower_image, transform.normalization)
    return PreparedImage(_stage(pixels, processor, device), None, None, height, width)


def prepare_tensor_image(
    processor: ImageProcessor,
    kind: EncodeMode,
    image: torch.Tensor,
    *,
    device: torch.device,
    signed_unit: bool,
) -> PreparedImage:
    """Apply the declared image transform to an already decoded RGB tensor."""

    value = image.detach().to(dtype=torch.float32)
    if value.ndim == 4:
        if int(value.shape[0]) != 1:
            raise invalid_descriptor("generated image staging accepts one image")
        value = value[0]
    if value.ndim != 3 or int(value.shape[0]) != 3:
        raise invalid_descriptor("generated image tensor must have shape [3, height, width]")
    if signed_unit:
        value = (value + 1.0) * 0.5
    value = value.clamp(0.0, 1.0)
    source_height, source_width = int(value.shape[1]), int(value.shape[2])
    transform = processor.vit if kind is EncodeMode.VISION else processor.vae
    if transform is None:
        raise invalid_descriptor(f"model declares no {kind.value} image transform")

    if isinstance(transform, PatchTransform):
        resized_height, resized_width = _patch_image_shape(
            transform,
            source_height,
            source_width,
        )
        value = _resize_tensor(value, resized_height, resized_width)
        normalized = _normalize_tensor(value, transform.normalization)
        patch = int(transform.patch_size)
        channels = int(normalized.shape[0])
        grid_height = resized_height // patch
        grid_width = resized_width // patch
        pixels = (
            normalized.view(channels, grid_height, patch, grid_width, patch)
            .permute(1, 3, 0, 2, 4)
            .reshape(grid_height * grid_width, channels * patch * patch)
        )
        grid = torch.tensor([[grid_height, grid_width]], dtype=torch.long)
        return PreparedImage(
            _stage(pixels, processor, device),
            grid.to(device=device, non_blocking=True),
            (grid_height, grid_width),
            source_height,
            source_width,
        )

    resize = transform.resize
    scale = min(int(resize.max_size) / max(source_width, source_height), 1.0)
    scale = max(scale, int(resize.min_size) / min(source_width, source_height))
    target_width, target_height = _stride_shape(
        source_width,
        source_height,
        scale,
        int(resize.stride),
    )
    if target_width * target_height > int(resize.max_pixels):
        scale = int(resize.max_pixels) / (target_width * target_height)
        target_width, target_height = _stride_shape(
            target_width,
            target_height,
            scale,
            int(resize.stride),
        )
    if max(target_width, target_height) > int(resize.max_size):
        scale = int(resize.max_size) / max(target_width, target_height)
        target_width, target_height = _stride_shape(
            target_width,
            target_height,
            scale,
            int(resize.stride),
        )
    value = _resize_tensor(value, target_height, target_width)
    normalized = _normalize_tensor(value, transform.normalization)
    return PreparedImage(
        _stage(normalized, processor, device),
        None,
        None,
        source_height,
        source_width,
    )


def _decode_rgb(encoded: str) -> Image.Image:
    """Decode a base64 image payload and normalize it to RGB."""

    if not isinstance(encoded, str) or not encoded:
        raise invalid_descriptor("inline image payload must be non-empty base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, OSError, ValueError) as error:
        raise invalid_descriptor("inline image payload is not a valid encoded image") from error
    if image.mode == "RGBA" or image.info.get("transparency") is not None:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return rgb
    return image.convert("RGB")


def _resize_patch_image(image: Image.Image, processor: PatchTransform) -> Image.Image:
    """Resize an image to the processor's bounded aspect-preserving patch grid."""

    height, width = _patch_image_shape(processor, image.height, image.width)
    return vision.resize(
        image,
        (height, width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def patch_grid_shape(
    processor: PatchTransform,
    source_height: int,
    source_width: int,
) -> tuple[int, int]:
    """Return the host-known patch grid for one declared image geometry."""

    height, width = _patch_image_shape(processor, source_height, source_width)
    patch = int(processor.patch_size)
    return height // patch, width // patch


def _patch_image_shape(
    processor: PatchTransform,
    source_height: int,
    source_width: int,
) -> tuple[int, int]:
    """Resolve patch-grid dimensions after applying processor pixel bounds."""

    factor = int(round(int(processor.patch_size) / float(processor.downsample_ratio)))
    return _bounded_grid_shape(
        source_height,
        source_width,
        factor=factor,
        minimum=int(processor.min_pixels),
        maximum=int(processor.max_pixels),
    )


def _bounded_grid_shape(
    height: int,
    width: int,
    *,
    factor: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    """Fit an aspect-preserving patch grid within minimum and maximum token bounds."""

    if min(height, width, factor) < 1:
        raise invalid_descriptor("image geometry must be positive")
    if max(height, width) / min(height, width) > 200:
        raise invalid_descriptor("image aspect ratio must be at most 200")
    result_height = max(factor, round(height / factor) * factor)
    result_width = max(factor, round(width / factor) * factor)
    if result_height * result_width > maximum:
        scale = math.sqrt((height * width) / maximum)
        result_height = max(factor, math.floor(height / scale / factor) * factor)
        result_width = max(factor, math.floor(width / scale / factor) * factor)
    elif result_height * result_width < minimum:
        scale = math.sqrt(minimum / (height * width))
        result_height = math.ceil(height * scale / factor) * factor
        result_width = math.ceil(width * scale / factor) * factor
    return result_height, result_width


def _resize_stride(image: Image.Image, processor: StrideResize) -> Image.Image:
    """Resize an image so both dimensions align with the configured spatial stride."""

    width, height = image.size
    scale = min(int(processor.max_size) / max(width, height), 1.0)
    scale = max(scale, int(processor.min_size) / min(width, height))
    new_width, new_height = _stride_shape(width, height, scale, int(processor.stride))
    if new_width * new_height > int(processor.max_pixels):
        scale = int(processor.max_pixels) / (new_width * new_height)
        new_width, new_height = _stride_shape(new_width, new_height, scale, int(processor.stride))
    if max(new_width, new_height) > int(processor.max_size):
        scale = int(processor.max_size) / max(new_width, new_height)
        new_width, new_height = _stride_shape(new_width, new_height, scale, int(processor.stride))
    return vision.resize(
        image,
        (new_height, new_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def _stride_shape(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
    """Round scaled dimensions to positive multiples of the required stride."""

    def align(value: float) -> int:
        """Round a scaled edge to the nearest positive model stride."""

        return max(stride, round(value / stride) * stride)

    return align(width * scale), align(height * scale)


def _normalize(image: Image.Image, name: str) -> torch.Tensor:
    """Apply channel-wise normalization to a PIL image converted to a tensor."""

    tensor = vision.to_tensor(image).to(torch.float32)
    return _normalize_tensor(tensor, name)


def _normalize_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Apply the named channel normalization policy to an image tensor."""

    if name == "signed_unit":
        return (tensor - 0.5) / 0.5
    if name == "imagenet":
        mean = tensor.new_tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(_IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std
    raise invalid_descriptor(f"unknown image normalization {name!r}")


def _resize_tensor(value: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a batched tensor image with bicubic interpolation."""

    return F.interpolate(
        value.unsqueeze(0),
        size=(int(height), int(width)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0]


def _stage(value: torch.Tensor, processor: ImageProcessor, device: torch.device) -> torch.Tensor:
    """Convert preprocessing output to the processor staging dtype and device."""

    dtype = (
        None if processor.staging_dtype is None else getattr(torch, processor.staging_dtype, None)
    )
    if processor.staging_dtype is not None and not isinstance(dtype, torch.dtype):
        raise invalid_descriptor(f"unknown image staging dtype {processor.staging_dtype!r}")
    return value.to(device=device, dtype=dtype, non_blocking=True)


__all__ = [
    "PreparedImage",
    "patch_grid_shape",
    "prepare_image",
    "prepare_tensor_image",
]
