"""Shared image output helpers."""
from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

__all__ = [
    'pil_image_to_png_b64',
    'to_uint8_image',
    'tensor_to_png_b64',
]


def pil_image_to_png_b64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string.

    Centralizes the BytesIO -> save(PNG) -> b64encode tail so the PNG container
    format is chosen in exactly one place.
    """
    buffer = io.BytesIO()
    # Lossless PNG: lower compression keeps pixels identical while avoiding
    # spending hundreds of milliseconds per 2K generated image on CPU deflate.
    image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def to_uint8_image(
    tensor: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Convert a CHW (or NCHW, first item taken) float image tensor to HWC uint8.

    ``value_range`` is the ``(lo, hi)`` span the tensor's values occupy; it is
    rescaled to ``[0, 1]``, clamped, and quantized to ``[0, 255]``. The single
    place the value-range assumption is made explicit — the default ``(-1, 1)``
    is the diffusion latent-decode convention; pass ``(0, 1)`` for tensors that
    are already in normalized image space.
    """
    image = tensor.detach().float()
    if image.ndim == 4:
        image = image[0]
    lo, hi = float(value_range[0]), float(value_range[1])
    span = hi - lo
    if span != 0:
        image = (image - lo) / span
    image = image.clamp(0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)


def tensor_to_png_b64(batch: torch.Tensor, *, value_range: tuple[float, float] = (-1.0, 1.0)) -> str:
    return pil_image_to_png_b64(Image.fromarray(to_uint8_image(batch, value_range=value_range)))
