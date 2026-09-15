"""SenseNova U1 image transforms and prompt framing."""

from __future__ import annotations

import torch

from uniserve_models.processing import (
    FeatureInjection,
    FeatureLayout,
    FlowPrompt,
    ImageProcessor,
    PatchTransform,
    PositionLayout,
)

from .config import Config


def image_processor(config: Config) -> ImageProcessor:
    """Build the checkpoint architecture's caller-owned image transforms."""
    return ImageProcessor(
        vit=PatchTransform(
            patch_size=int(config.vision.patch_size),
            downsample_ratio=float(config.vision.downsample_ratio),
            min_pixels=512 * 512,
            max_pixels=2048 * 2048,
        ),
        staging_dtype=torch.bfloat16,
        feature_injection=FeatureInjection(
            layout=FeatureLayout.DIRECT,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            start_token="<img>",
            end_token="</img>",
        ),
    )


_FLOW_SYSTEM_MESSAGE = (
    "You are an image generation and editing assistant that accurately "
    "understands and executes user intent.\n\n"
    "You support two modes:\n\n1. Think Mode:\n"
    "If the task requires reasoning, you MUST start with a <think></think> "
    "block. Put all reasoning inside the block using plain text. DO NOT "
    "include any image tags. Keep it reasonable and directly useful for "
    "producing the final image.\n\n"
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the "
    "final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's "
    "intent.\n\n"
    "B. Image Editing:\n- Use the provided image(s) as input or reference "
    "for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the "
    "reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n- For any visible text in the image, follow the "
    "language specified for the rendered text in the user's description, "
    "not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


flow_prompt = FlowPrompt(
    user_prefix="<|im_start|>user\n",
    user_suffix="<|im_end|>\n",
    assistant_suffix="<|im_start|>assistant\n",
    conditioned_append="<think>\n\n</think>\n\n<img>",
    unconditional_append="<img>",
    system_prefix="<|im_start|>system\n",
    system_message=_FLOW_SYSTEM_MESSAGE,
    system_suffix="<|im_end|>\n",
)
