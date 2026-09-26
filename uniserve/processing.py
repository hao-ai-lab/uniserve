"""Caller-owned image transforms and diffusion prompt framing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import torch


class BranchSource(StrEnum):
    """Choose conditioning content for one guidance branch."""

    CONDITIONING = "conditioning"
    NEGATIVE_OR_START = "negative_or_start"
    START = "start"


class PositionLayout(StrEnum):
    """Choose temporal or temporal/spatial coordinates for inserted features."""

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


class FeatureLayout(StrEnum):
    """Selects direct feature insertion or start/end-token framing."""

    DIRECT = "direct"
    FRAMED = "framed"


@dataclass(frozen=True, slots=True)
class FeatureInjection:
    """Defines how encoder features replace or frame tokens in the language sequence."""  # noqa: E501

    layout: FeatureLayout
    positions: PositionLayout
    start_token: str | None = None
    end_token: str | None = None
    start_token_id: int | None = None
    end_token_id: int | None = None


def _positive(value: object, name: str) -> None:
    """Reject a transform dimension that is not a positive integer."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PixelBounds:
    """Round an image to whole pooled patches within a pixel-area range.

    Each side rounds to the nearest multiple of the tower's pooled patch
    side, at least one multiple. When the rounded area exceeds the upper
    bound, both sides rescale to that bound and floor to the multiple; when
    it falls below ``min_pixels`` they rescale to it and ceil. Aspect
    ratios above 200 are rejected.
    """

    min_pixels: int
    max_pixels: int
    # Pixel budget a request's input images share, or None when each input
    # image is bounded by ``max_pixels`` alone. With ``n`` input images each
    # one is bounded by ``min(max_pixels, max_total_pixels // n)``; images
    # that are not request inputs, such as generated images, keep
    # ``max_pixels``.
    max_total_pixels: int | None = None

    def __post_init__(self) -> None:
        """Require positive pixel bounds."""
        _positive(self.min_pixels, "minimum image pixels")
        _positive(self.max_pixels, "maximum image pixels")
        if self.max_total_pixels is not None:
            _positive(self.max_total_pixels, "shared image pixel budget")

    def pixel_bound(
        self, *, patch_size: int, input_images: int | None = None
    ) -> int:
        """Return the upper pixel bound for one image.

        Args:
            patch_size: The tower's patch side; the bound does not depend
                on it.
            input_images: Number of input images in the image's request, or
                None for an image that is not a request input.

        Raises:
            ValueError: If ``input_images`` is not positive.
        """
        del patch_size
        if input_images is None or self.max_total_pixels is None:
            return self.max_pixels
        if input_images < 1:
            raise ValueError("input image count must be positive")
        return min(self.max_pixels, self.max_total_pixels // input_images)

    def fit(
        self,
        height: int,
        width: int,
        *,
        patch_size: int,
        downsample: int,
        input_images: int | None = None,
    ) -> tuple[int, int]:
        """Return the resized ``(height, width)`` in pixels.

        Raises:
            ValueError: For a nonpositive side, an aspect ratio above 200,
                or a nonpositive ``input_images``.
        """
        if min(height, width) < 1:
            raise ValueError("image dimensions must be positive")
        if max(height, width) / min(height, width) > 200:
            raise ValueError("image aspect ratio must be at most 200")

        unit = patch_size * downsample
        maximum = self.pixel_bound(
            patch_size=patch_size, input_images=input_images
        )
        result_height = max(unit, round(height / unit) * unit)
        result_width = max(unit, round(width / unit) * unit)
        if result_height * result_width > maximum:
            scale = math.sqrt((height * width) / maximum)
            result_height = max(unit, math.floor(height / scale / unit) * unit)
            result_width = max(unit, math.floor(width / scale / unit) * unit)
        elif result_height * result_width < self.min_pixels:
            scale = math.sqrt(self.min_pixels / (height * width))
            result_height = math.ceil(height * scale / unit) * unit
            result_width = math.ceil(width * scale / unit) * unit
        return result_height, result_width


@dataclass(frozen=True, slots=True)
class PatchTransform:
    """Resize and normalization for a patch-sequence tower.

    The tower cuts an image into ``patch_size`` squares and merges each
    ``downsample x downsample`` block of patches into one output token, so
    resized sides are multiples of ``patch_size * downsample`` pixels;
    ``resize`` chooses the size. Packed patch rows hold channel, pixel row,
    pixel column values and follow the patch grid in raster order.
    """

    patch_size: int
    downsample: int
    resize: PixelBounds
    normalization: Literal["imagenet", "signed_unit"] = "imagenet"

    def __post_init__(self) -> None:
        """Require positive patch dimensions and a known resize policy."""
        _positive(self.patch_size, "patch size")
        _positive(self.downsample, "patch downsample")
        if not isinstance(self.resize, PixelBounds):
            raise ValueError("patch towers resize by pixel bounds")

    def pixel_bound(self, input_images: int | None = None) -> int:
        """Return the upper bound on one resized image's pixel area.

        Args:
            input_images: Number of input images in the image's request, or
                None for an image that is not a request input.

        Raises:
            ValueError: If ``input_images`` is not positive.
        """
        return self.resize.pixel_bound(
            patch_size=self.patch_size, input_images=input_images
        )

    def resized_size(
        self, height: int, width: int, input_images: int | None = None
    ) -> tuple[int, int]:
        """Return the ``(height, width)`` in pixels an image resizes to.

        ``input_images`` is the number of input images in the request of a
        request input image, or None for a generated image.

        Raises:
            ValueError: When ``resize`` rejects the source dimensions.
        """
        return self.resize.fit(
            height,
            width,
            patch_size=self.patch_size,
            downsample=self.downsample,
            input_images=input_images,
        )

    def grid_shape(
        self, height: int, width: int, input_images: int | None = None
    ) -> tuple[int, int]:
        """Return the ``(rows, columns)`` patch grid of a resized image.

        Raises:
            ValueError: When ``resize`` rejects the source dimensions.
        """
        resized = self.resized_size(height, width, input_images)
        return resized[0] // self.patch_size, resized[1] // self.patch_size

    def tokens(
        self, height: int, width: int, input_images: int | None = None
    ) -> int:
        """Count the output tokens the tower produces for an image.

        Raises:
            ValueError: When ``resize`` rejects the source dimensions.
        """
        rows, columns = self.grid_shape(height, width, input_images)
        return rows * columns // self.downsample**2


@dataclass(frozen=True, slots=True)
class StrideResize:
    """Defines bounded aspect-preserving image resizing aligned to a spatial stride."""  # noqa: E501

    max_size: int
    min_size: int
    stride: int
    max_pixels: int


@dataclass(frozen=True, slots=True)
class TowerTransform:
    """Combines resize and normalization policy for one image tower."""

    resize: StrideResize
    normalization: Literal["imagenet", "signed_unit"] = "signed_unit"


@dataclass(frozen=True, slots=True)
class ImageProcessor:
    """Defines ViT and VAE transforms, staging dtype, and language-sequence feature injection."""  # noqa: E501

    vit: PatchTransform | TowerTransform | None = None
    vae: TowerTransform | None = None
    staging_dtype: torch.dtype | None = None
    feature_injection: FeatureInjection | None = None

    def __post_init__(self) -> None:
        """Require at least one image transform for the caller."""
        if self.vit is None and self.vae is None:
            raise ValueError(
                "image processor must implement at least one transform"
            )


__all__ = [
    "FlowPrompt",
    "BranchSource",
    "PositionLayout",
    "load_tokenizer",
    "FeatureInjection",
    "FeatureLayout",
    "ImageProcessor",
    "PatchTransform",
    "PixelBounds",
    "StrideResize",
    "TowerTransform",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowPrompt:
    """Immutable caller-owned framing for one classifier-free-guidance prefix."""  # noqa: E501

    user_prefix: str
    user_suffix: str
    assistant_suffix: str
    conditioned_append: str
    unconditional_append: str
    system_prefix: str = ""
    system_message: str = ""
    system_suffix: str = ""
    add_special_tokens: bool = True

    def encode(
        self, tokenizer: Any, *, text: str, conditioned: bool
    ) -> tuple[int, ...]:
        """Frame and tokenize either the conditioned or unconditional diffusion prompt."""  # noqa: E501
        if tokenizer is None:
            raise ValueError(
                "the configured generation prompt requires a tokenizer"
            )

        append = (
            self.conditioned_append
            if conditioned
            else self.unconditional_append
        )
        framed = (
            self.system_prefix
            + self.system_message
            + self.system_suffix
            + self.user_prefix
            + text
            + self.user_suffix
            + self.assistant_suffix
            + append
        )
        return tuple(
            int(value)
            for value in tokenizer.encode(
                framed, add_special_tokens=self.add_special_tokens
            )
        )


def load_tokenizer(path: Path):
    """Load a caller-owned tokenizer from a resolved local checkpoint directory."""  # noqa: E501
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, use_fast=False, trust_remote_code=False, local_files_only=True
    )
