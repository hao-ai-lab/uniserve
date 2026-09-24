"""BAGEL image transforms and prompt framing."""

from __future__ import annotations

from uniserve.processing import (
    FeatureInjection,
    FeatureLayout,
    ImageProcessor,
    PositionLayout,
    StrideResize,
    TowerTransform,
)

from .config import Config

# Side bounds and strides in pixels. The ViT tower's maximum side and stride
# come from the SigLIP config, so its inputs hold whole patches.
_BAGEL_VIT_MIN_SIZE = 224

# The VAE stride is one flow token in pixels (autoencoder downsample times
# latent patch size) for BAGEL's default config; these constants do not follow
# ``Config``.
_BAGEL_VAE_MIN_SIZE = 512
_BAGEL_VAE_MAX_SIZE = 1024
_BAGEL_VAE_STRIDE = 16

# Both towers share one per-image pixel budget.
_BAGEL_MAX_IMAGE_PIXELS = 14 * 14 * 9 * 1024


def image_processor(config: Config) -> ImageProcessor:
    """Build the checkpoint architecture's caller-owned image transforms.

    Encoded image features enter the language sequence framed by the
    start/end-of-image tokens and take temporal positions only.
    """
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


# BAGEL has no guidance prompt framing beyond its chat template. Without one,
# the worker's ``resolve_prefix`` rejects a generation prompt override.
flow_prompt = None
