"""Image transforms of the deterministic stub model."""

from __future__ import annotations

import torch

from uniserve.processing import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
    PixelBounds,
    PositionLayout,
    StrideResize,
    TowerTransform,
)


def image_processor() -> ImageProcessor:
    """Build the stub model's caller-owned image transforms.

    Patch sizes and resize strides are fixed at 16 pixels, matching the
    default ``Config.patch_size`` the worker builds ``Model`` with.
    """
    return ImageProcessor(
        vit=PatchTransform(
            patch_size=16,
            downsample=1,
            resize=PixelBounds(min_pixels=16 * 16, max_pixels=512 * 512),
            normalization="signed_unit",
        ),
        vae=TowerTransform(
            StrideResize(
                max_size=512, min_size=16, stride=16, max_pixels=512 * 512
            )
        ),
        staging_dtype=torch.bfloat16,
        # The direct layout has no start marker; a row that closes the image
        # appends the end marker. ``model._Head`` maps 1007 to EOS, so logits
        # read at that marker select EOS.
        feature_injection=FeatureInjection(
            layout=FeatureLayout.DIRECT,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            end_token_id=1007,
        ),
    )
