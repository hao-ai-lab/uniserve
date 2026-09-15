"""Image transforms for the deterministic serving simulator."""

from __future__ import annotations

import torch

from uniserve_models.processing import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
    PositionLayout,
    StrideResize,
    TowerTransform,
)


def image_processor() -> ImageProcessor:
    """Build the checkpoint architecture's caller-owned image transforms."""
    return ImageProcessor(
        vit=PatchTransform(
            patch_size=16,
            downsample_ratio=1.0,
            min_pixels=16 * 16,
            max_pixels=512 * 512,
            normalization="signed_unit",
        ),
        vae=TowerTransform(
            StrideResize(
                max_size=512, min_size=16, stride=16, max_pixels=512 * 512
            )
        ),
        staging_dtype=torch.bfloat16,
        feature_injection=FeatureInjection(
            layout=FeatureLayout.DIRECT,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            end_token_id=1007,
        ),
    )
