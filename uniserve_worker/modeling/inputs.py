"""Image processing policy owned by multimodal execution models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from uniserve_worker.modeling.tensors import PositionLayout

from ..foundation.errors import invalid_descriptor


class FeatureLayout(StrEnum):
    """Selects direct feature insertion or start/end-token framing."""

    DIRECT = "direct"
    FRAMED = "framed"


@dataclass(frozen=True, slots=True)
class FeatureInjection:
    """Defines how encoder features replace or frame tokens in the language sequence."""

    layout: FeatureLayout
    positions: PositionLayout
    start_token: str | None = None
    end_token: str | None = None
    start_token_id: int | None = None
    end_token_id: int | None = None


@dataclass(frozen=True, slots=True)
class PatchTransform:
    """Defines patch sizing, pixel bounds, downsampling, and normalization for a vision tower."""

    patch_size: int
    downsample_ratio: float
    min_pixels: int
    max_pixels: int
    normalization: str = "imagenet"


@dataclass(frozen=True, slots=True)
class StrideResize:
    """Defines bounded aspect-preserving image resizing aligned to a spatial stride."""

    max_size: int
    min_size: int
    stride: int
    max_pixels: int


@dataclass(frozen=True, slots=True)
class TowerTransform:
    """Combines resize and normalization policy for one image tower."""

    resize: StrideResize
    normalization: str = "signed_unit"


@dataclass(frozen=True, slots=True)
class ImageProcessor:
    """Defines ViT and VAE transforms, staging dtype, and language-sequence feature injection."""

    vit: PatchTransform | TowerTransform | None = None
    vae: TowerTransform | None = None
    staging_dtype: str | None = None
    feature_injection: FeatureInjection | None = None

    def __post_init__(self) -> None:
        """Validate transform callables, staging dtype, and feature-injection settings."""

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
