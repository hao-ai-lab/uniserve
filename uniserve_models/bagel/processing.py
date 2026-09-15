"""BAGEL image transforms and prompt framing."""

from __future__ import annotations

from uniserve_models.processing import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PositionLayout,
    StrideResize,
    TowerTransform,
)

from .config import Config

_BAGEL_VIT_MIN_SIZE = 224


_BAGEL_VAE_MIN_SIZE = 512


_BAGEL_VAE_MAX_SIZE = 1024


_BAGEL_VAE_STRIDE = 16


_BAGEL_MAX_IMAGE_PIXELS = 14 * 14 * 9 * 1024


def image_processor(config: Config) -> ImageProcessor:
    """Build the checkpoint architecture's caller-owned image transforms."""

    return ImageProcessor(
        vit=TowerTransform(
            resize=StrideResize(
                max_size=int(config.vision.image_size),
                min_size=_BAGEL_VIT_MIN_SIZE,
                stride=int(config.vision.patch_size),
                max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
            ),
        ),
        vae=TowerTransform(
            resize=StrideResize(
                max_size=_BAGEL_VAE_MAX_SIZE,
                min_size=_BAGEL_VAE_MIN_SIZE,
                stride=_BAGEL_VAE_STRIDE,
                max_pixels=_BAGEL_MAX_IMAGE_PIXELS,
            ),
        ),
        feature_injection=FeatureInjection(
            layout=FeatureLayout.FRAMED,
            positions=PositionLayout.TEMPORAL,
            start_token_id=int(config.start_of_image_id),
            end_token_id=int(config.end_of_image_id),
        ),
    )


flow_prompt = None
