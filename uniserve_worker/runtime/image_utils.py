"""Shared image output helpers."""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

__all__ = [
    "pil_image_to_png_bytes",
    "pil_image_to_png_b64",
    "png_bytes_to_b64",
    "quantize_image_hwc",
    "uint8_image_to_png_base64_bytes",
]


def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG container bytes.

    Centralizes the BytesIO -> save(PNG) tail so the PNG container format is
    chosen in exactly one place.
    """
    buffer = io.BytesIO()
    # Lossless PNG: lower compression keeps pixels identical while avoiding
    # spending hundreds of milliseconds per 2K generated image on CPU deflate.
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def png_bytes_to_b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def pil_image_to_png_b64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    return png_bytes_to_b64(pil_image_to_png_bytes(image))


def quantize_image_hwc(
    tensor: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Quantize one CHW/NCHW image to contiguous HWC uint8 on its source device.

    ``value_range`` is the ``(lo, hi)`` span the tensor's values occupy; it is
    rescaled to ``[0, 1]``, clamped, and quantized to ``[0, 255]``. The single
    place the value-range assumption is made explicit — the default ``(-1, 1)``
    is the diffusion latent-decode convention; pass ``(0, 1)`` for tensors that
    are already in normalized image space.
    """
    image = tensor.detach()
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError("image quantization requires a single-item batch")
        image = image[0]
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("image quantization requires RGB CHW pixels")
    image = image.float()
    lo, hi = float(value_range[0]), float(value_range[1])
    span = hi - lo
    if span != 0:
        image = (image - lo) / span
    image = image.clamp(0, 1)
    return (image.permute(1, 2, 0) * 255.0).round().to(dtype=torch.uint8).contiguous()


def uint8_image_to_png_base64_bytes(image: torch.Tensor) -> bytes:
    """Encode a query-ready CPU HWC uint8 tensor without observing a device value."""

    if image.device.type != "cpu" or image.dtype is not torch.uint8 or image.ndim != 3:
        raise ValueError("PNG encoding requires a CPU HWC uint8 tensor")
    png = pil_image_to_png_bytes(Image.fromarray(image.numpy()))
    return base64.b64encode(png)
