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
    """Choose the position coordinates of inserted features.

    ``TEMPORAL`` places every feature at one temporal position and
    ``TEMPORAL_SPATIAL`` adds each feature's grid row and column to it;
    both advance the sequence by one position. ``SEQUENTIAL`` gives the
    features consecutive positions, as text tokens take, and advances the
    sequence past all of them.
    """

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"
    SEQUENTIAL = "sequential"


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
class PatchBudget:
    """Scale an image to fill a fixed patch budget at its aspect ratio.

    The image scales so that its area reaches ``max_patches`` patches, and
    each side floors to a multiple of the pooled patch side. A side that
    floors to zero, which only a very elongated image produces, takes one
    multiple; the other side then becomes the integer aspect ratio times
    one multiple, capped at the longest side the budget's pooled tokens
    allow. Every image therefore scales, up or down, to at most the budget.
    """

    max_patches: int

    def __post_init__(self) -> None:
        """Require a positive patch budget."""
        _positive(self.max_patches, "image patch budget")

    def pixel_bound(
        self, *, patch_size: int, input_images: int | None = None
    ) -> int:
        """Return the pixel area of the budget, the same for every image.

        ``input_images`` does not affect the bound: each image receives the
        whole budget.
        """
        del input_images
        return self.max_patches * patch_size**2

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

        The float arithmetic follows the Transformers Gemma-4
        ``get_aspect_ratio_preserving_size`` operation for operation, and
        the server's soft-token planner performs the same IEEE-754 double
        operations, so all three agree on every size whose pixel area is
        exactly representable as a double.

        Raises:
            ValueError: For a nonpositive side.
        """
        del input_images
        if min(height, width) < 1:
            raise ValueError("image dimensions must be positive")

        unit = patch_size * downsample
        budget = self.pixel_bound(patch_size=patch_size)
        factor = math.sqrt(budget / (height * width))
        target_height = math.floor(factor * height / unit) * unit
        target_width = math.floor(factor * width / unit) * unit
        if target_height == 0 and target_width == 0:
            raise ValueError(
                f"a {height}x{width} image cannot hold one {unit}-pixel square"
            )

        longest = (self.max_patches // downsample**2) * unit
        if target_height == 0:
            target_height = unit
            target_width = min(math.floor(width / height) * unit, longest)
        elif target_width == 0:
            target_width = unit
            target_height = min(math.floor(height / width) * unit, longest)

        # Positive sides never exceed the budget; the reference keeps the
        # same guard.
        if target_height * target_width > budget:
            raise ValueError(
                f"a {height}x{width} image resizes beyond "
                f"{self.max_patches} patches"
            )
        return target_height, target_width


@dataclass(frozen=True, slots=True)
class PatchTransform:
    """Resize, pixel scaling and patch packing for a patch-sequence tower.

    The tower cuts an image into ``patch_size`` squares and merges each
    ``downsample x downsample`` block of patches into one output token, so
    resized sides are multiples of ``patch_size * downsample`` pixels;
    ``resize`` chooses the size. ``resampling`` names the implementation of
    the antialiased bicubic filter applied to decoded 8-bit images:
    ``"pil"`` is PIL's filter, ``"torchvision"`` is torchvision's on the
    ``uint8`` CHW tensor, rounded back to 8 bits. The two differ by one
    level on a fraction of pixels, so a tower names the one its reference
    processor uses.

    ``normalization`` maps 8-bit values to the tower's input range:
    ``"imagenet"`` and ``"signed_unit"`` divide by 255 and then standardize
    or map to ``[-1, 1]``; ``"unit"`` multiplies by the float32 rescale
    factor ``1/255`` as Transformers image processors do, which differs
    from dividing by 255 in the last bit for about half of the byte values.

    ``patch_layout`` orders the values within each packed patch row:
    ``"channels_first"`` is channel, pixel row, pixel column, as a patch
    convolution reads them, and ``"channels_last"`` is pixel row, pixel
    column, channel (``uniserve.nn.functional.patchify``), as a patch
    linear projection reads them. Rows follow the patch grid in raster
    order either way.
    """

    patch_size: int
    downsample: int
    resize: PixelBounds | PatchBudget
    normalization: Literal["imagenet", "signed_unit", "unit"] = "imagenet"
    patch_layout: Literal["channels_first", "channels_last"] = "channels_first"
    resampling: Literal["pil", "torchvision"] = "pil"

    def __post_init__(self) -> None:
        """Require positive patch dimensions and known policies."""
        _positive(self.patch_size, "patch size")
        _positive(self.downsample, "patch downsample")
        if not isinstance(self.resize, (PixelBounds, PatchBudget)):
            raise ValueError("patch towers resize by pixel bounds or budget")
        if self.normalization not in ("imagenet", "signed_unit", "unit"):
            raise ValueError(
                f"unknown image normalization {self.normalization!r}"
            )
        if self.patch_layout not in ("channels_first", "channels_last"):
            raise ValueError(f"unknown patch layout {self.patch_layout!r}")
        if self.resampling not in ("pil", "torchvision"):
            raise ValueError(f"unknown image resampling {self.resampling!r}")

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
    """Defines ViT and VAE transforms, staging dtype, and language-sequence feature injection.

    ``alpha`` sets how a decoded image with transparency becomes RGB before
    either transform: ``"white"`` composites it over an opaque white
    background, and ``"drop"`` discards the alpha channel and keeps every
    pixel's stored color, as PIL's ``convert("RGB")`` does. A
    ``staging_dtype`` of None stages the FP32 transform output.
    """  # noqa: E501

    vit: PatchTransform | TowerTransform | None = None
    vae: TowerTransform | None = None
    staging_dtype: torch.dtype | None = None
    feature_injection: FeatureInjection | None = None
    alpha: Literal["white", "drop"] = "white"

    def __post_init__(self) -> None:
        """Require at least one image transform and a known alpha policy."""
        if self.vit is None and self.vae is None:
            raise ValueError(
                "image processor must implement at least one transform"
            )
        if self.alpha not in ("white", "drop"):
            raise ValueError(f"unknown image alpha policy {self.alpha!r}")


__all__ = [
    "FlowPrompt",
    "BranchSource",
    "PositionLayout",
    "load_tokenizer",
    "FeatureInjection",
    "FeatureLayout",
    "ImageProcessor",
    "PatchBudget",
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
