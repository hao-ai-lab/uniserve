"""Caller-owned image transforms and diffusion prompt framing."""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class PatchTransform:
    """Defines patch sizing, pixel bounds, downsampling, and normalization for a vision tower."""  # noqa: E501

    patch_size: int
    downsample_ratio: float
    min_pixels: int
    max_pixels: int
    normalization: Literal["imagenet", "signed_unit"] = "imagenet"
    # Pixel budget a request's input images share, or None when each input
    # image is bounded by ``max_pixels`` alone. With ``n`` input images each
    # one is bounded by ``min(max_pixels, max_total_pixels // n)``; images
    # that are not request inputs, such as generated images, keep
    # ``max_pixels``.
    max_total_pixels: int | None = None

    def pixel_bound(self, input_images: int | None = None) -> int:
        """Return the upper pixel bound for one image.

        Args:
            input_images: Number of input images in the image's request, or
                None for an image that is not a request input.

        Raises:
            ValueError: If ``input_images`` is not positive.
        """
        if input_images is None or self.max_total_pixels is None:
            return self.max_pixels
        if input_images < 1:
            raise ValueError("input image count must be positive")
        return min(self.max_pixels, self.max_total_pixels // input_images)


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
