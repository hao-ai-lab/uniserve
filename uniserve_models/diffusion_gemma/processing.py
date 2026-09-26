"""The Gemma-4 image processor as a shared image transform descriptor."""

from __future__ import annotations

from uniserve.processing import ImageProcessor, PatchBudget, PatchTransform

from .config import Config


def image_processor(config: Config) -> ImageProcessor:
    """Describe the Gemma-4 preprocessing of ``Model.vision_encoder`` inputs.

    This is the Transformers ``Gemma4ImageProcessor`` with the checkpoint's
    settings. An image converts to RGB by dropping any alpha channel, then
    resizes, preserving its aspect ratio, to fill the budget of
    ``soft_tokens_per_image * pooling_kernel_size**2`` patches with sides
    that are multiples of ``patch_size * pooling_kernel_size`` pixels (2520
    patches and 48 pixels for the released checkpoints). The resize is
    torchvision's antialiased bicubic filter on the 8-bit image, skipped
    when the size already fits. Pixels then rescale to ``[0, 1]``; the tower
    maps them to ``[-1, 1]`` itself, so no mean or deviation normalization
    applies. Patch rows are in pixel row, pixel column, channel order, as
    the tower's patch projection reads them, and each
    ``pooling_kernel_size`` square of patches pools into one soft token.

    The rows stage in FP32, the reference processor's output dtype, so the
    tower's ``[-1, 1]`` mapping rounds exactly as the reference does before
    the BF16 projection. The serving front end counts an image's soft
    tokens with the same budget arithmetic before the image is decoded.

    Image features take the place of the prompt's image placeholder tokens
    inside the text prefill, so the processor declares no separate feature
    injection.
    """
    vision = config.vision
    return ImageProcessor(
        vit=PatchTransform(
            patch_size=vision.patch_size,
            downsample=vision.pooling_kernel_size,
            resize=PatchBudget(vision.max_patches),
            normalization="unit",
            patch_layout="channels_last",
            resampling="torchvision",
        ),
        alpha="drop",
    )
