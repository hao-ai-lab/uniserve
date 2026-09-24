"""Image pixel normalization and container encoding."""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

__all__ = [
    "pil_image_to_png_bytes",
    "quantize_image_hwc",
    "uint8_image_to_png_base64_bytes",
]


def pil_image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG container bytes.

    PNG is lossless at every compression level; level 1 trades output size
    for lower CPU encoding time on high-resolution generated images.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def quantize_image_hwc(
    tensor: torch.Tensor,
    *,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Quantize one CHW/NCHW image to contiguous HWC uint8 on its source device.

    ``value_range`` declares the source interval mapped onto ``[0, 1]`` before
    clamping and conversion to ``[0, 255]``. The default ``(-1, 1)`` matches
    diffusion decoder output; callers with normalized pixels pass ``(0, 1)``.

    Raises:
        ValueError: When a 4-D input's batch is not one, or the image is not
            3-channel CHW.
    """
    # Collapse the supported singleton batch form into the canonical CHW layout.
    image = tensor.detach()
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError("image quantization requires a single-item batch")
        image = image[0]
    if image.ndim != 3 or int(image.shape[0]) != 3:
        raise ValueError("image quantization requires RGB CHW pixels")

    # Normalize on-device, then transpose into the packed HWC encoder layout.
    image = image.float()
    lo, hi = float(value_range[0]), float(value_range[1])
    span = hi - lo
    # A degenerate range skips the affine map; the pixels are only clamped.
    if span != 0:
        image = (image - lo) / span
    image = image.clamp(0, 1)
    return (
        (image.permute(1, 2, 0) * 255.0)
        .round()
        .to(dtype=torch.uint8)
        .contiguous()
    )


def uint8_image_to_png_base64_bytes(image: torch.Tensor) -> bytes:
    """Encode a CPU HWC uint8 image as base64 PNG bytes.

    The function never touches a device and performs no synchronization: when
    ``image`` is the destination of a device-to-host copy, the caller must
    have observed that copy's completion first (`defer_image_encoding` in
    `uniserve_worker.execution.image` runs it as a host task gated on the
    output buffer's readiness).

    Raises:
        ValueError: When ``image`` is not a 3-D ``uint8`` CPU tensor.
    """
    if (
        image.device.type != "cpu"
        or image.dtype is not torch.uint8
        or image.ndim != 3
    ):
        raise ValueError("PNG encoding requires a CPU HWC uint8 tensor")
    png = pil_image_to_png_bytes(Image.fromarray(image.numpy()))
    return base64.b64encode(png)
