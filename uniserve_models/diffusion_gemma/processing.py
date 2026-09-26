"""Gemma-4 image preprocessing for the DiffusionGemma vision tower.

An RGB image is resized, preserving its aspect ratio, to the largest size
whose sides are multiples of ``patch_size * pooling_kernel_size`` (48
pixels) and whose patch count fits the tower's budget of
``soft_tokens_per_image * pooling_kernel_size**2`` patches (2520). Pixels
then scale from ``[0, 255]`` to ``[0, 1]``; the tower maps them to
``[-1, 1]`` itself, so no mean or deviation normalization applies. The
serving front end reproduces the same size arithmetic to count an image's
soft tokens before encoding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torchvision.transforms.v2 import functional as transforms

from uniserve.model import VisionInput
from uniserve.nn.functional import patchify

from .config import VisionConfig

# The reference processor's rescale factor, 1/255 as stored in its config.
RESCALE_FACTOR = 0.00392156862745098


def resized_size(
    height: int, width: int, config: VisionConfig
) -> tuple[int, int]:
    """Return the ``(height, width)`` an image of the given size resizes to.

    Both sides become multiples of ``patch_size * pooling_kernel_size``.
    The area approaches the patch budget from below; a side that would
    round to zero takes one multiple, and the other side then follows the
    integer aspect ratio up to the longest side the budget allows.

    Raises:
        ValueError: If a side is not positive or both sides round to zero.
    """
    if min(height, width) < 1:
        raise ValueError("image sides must be positive")
    patch, kernel = config.patch_size, config.pooling_kernel_size
    budget = config.max_patches * patch**2
    factor = math.sqrt(budget / (height * width))
    side = patch * kernel

    target_height = int(math.floor(factor * height / side)) * side
    target_width = int(math.floor(factor * width / side)) * side
    if target_height == 0 and target_width == 0:
        raise ValueError(
            f"a {height}x{width} image cannot hold one {side}-pixel square"
        )

    longest = (config.max_patches // kernel**2) * side
    if target_height == 0:
        target_height = side
        target_width = min(int(math.floor(width / height)) * side, longest)
    elif target_width == 0:
        target_width = side
        target_height = min(int(math.floor(height / width)) * side, longest)
    return target_height, target_width


def soft_tokens(height: int, width: int, config: VisionConfig) -> int:
    """Count the soft tokens an image of the given size encodes to."""
    resized = resized_size(height, width, config)
    patch, kernel = config.patch_size, config.pooling_kernel_size
    return (resized[0] // patch) * (resized[1] // patch) // kernel**2


def preprocess(image: torch.Tensor, config: VisionConfig) -> torch.Tensor:
    """Resize and rescale one uint8 RGB ``[3, height, width]`` image.

    Resizing is bicubic with antialiasing on the uint8 pixels and is skipped
    when the size already fits. Returns the FP32 ``[3, height', width']``
    image with values in ``[0, 1]``, ready for
    ``Model.vision_encoder.encode``.
    """
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
        raise ValueError("Gemma-4 preprocessing takes uint8 RGB CHW images")
    size = resized_size(int(image.shape[1]), int(image.shape[2]), config)
    if size != tuple(image.shape[1:]):
        image = transforms.resize(
            image,
            list(size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return image * RESCALE_FACTOR


def patches(
    image: torch.Tensor, config: VisionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a preprocessed ``[3, height, width]`` image into patch rows.

    Returns ``[patches, 3 * patch_size**2]`` rows in row-major patch order,
    each in pixel row, pixel column, then channel order, and their
    ``[patches, 2]`` int64 ``(x, y)`` = (column, row) positions.
    """
    rows = patchify(image, patch_size=config.patch_size)
    height = image.shape[-2] // config.patch_size
    width = image.shape[-1] // config.patch_size
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device),
        torch.arange(width, device=image.device),
        indexing="ij",
    )
    return rows, torch.stack((x, y), dim=-1).reshape(-1, 2)


def vision_input(
    images: Sequence[torch.Tensor], config: VisionConfig
) -> VisionInput:
    """Preprocess uint8 RGB images into the vision encoder's input."""
    processed = tuple(preprocess(image, config) for image in images)
    return VisionInput(
        processed, (None,) * len(processed), (None,) * len(processed)
    )
