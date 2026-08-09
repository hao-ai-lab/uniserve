"""Image processing policy owned by multimodal execution models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..foundation.errors import invalid_descriptor
from .runtime import PositionLayout


class FeatureLayout(StrEnum):
    DIRECT = "direct"
    FRAMED = "framed"


@dataclass(frozen=True, slots=True)
class FeatureInjection:
    layout: FeatureLayout
    positions: PositionLayout
    start_token: str | None = None
    end_token: str | None = None
    start_token_id: int | None = None
    end_token_id: int | None = None


@dataclass(frozen=True, slots=True)
class PatchTransform:
    patch_size: int
    downsample_ratio: float
    min_pixels: int
    max_pixels: int
    multi_image_pixel_budget: int
    normalization: str = "imagenet"


@dataclass(frozen=True, slots=True)
class StrideResize:
    max_size: int
    min_size: int
    stride: int
    max_pixels: int


@dataclass(frozen=True, slots=True)
class TowerTransform:
    resize: StrideResize
    normalization: str = "signed_unit"


@dataclass(frozen=True, slots=True)
class ImageProcessor:
    vit: PatchTransform | TowerTransform | None = None
    vae: TowerTransform | None = None
    staging_dtype: str | None = None
    feature_injection: FeatureInjection | None = None

    def __post_init__(self) -> None:
        if self.vit is None and self.vae is None:
            raise invalid_descriptor("image processor must implement at least one transform")


__all__ = [
    "FeatureInjection",
    "FeatureLayout",
    "ImageProcessor",
    "PatchTransform",
    "StrideResize",
    "TowerTransform",
]
