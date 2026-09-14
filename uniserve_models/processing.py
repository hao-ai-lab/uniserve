"""Caller-owned image transforms and diffusion prompt framing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uniserve_models.bagel import BagelConfig
    from uniserve_models.sensenova.config import NeoChatConfig

from uniserve.model.tensors import PositionLayout


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
        """Require at least one image transform for the caller."""

        if self.vit is None and self.vae is None:
            raise ValueError("image processor must implement at least one transform")


__all__ = [
    "FlowPrompt",
    "bagel_processor",
    "sensenova_processor",
    "stub_processor",
    "resolve_input_tokens",
    "SENSENOVA_PROMPT",
    "FeatureInjection",
    "FeatureLayout",
    "ImageProcessor",
    "PatchTransform",
    "StrideResize",
    "TowerTransform",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowPrompt:
    """Immutable caller-owned framing for one classifier-free-guidance prefix."""

    user_prefix: str
    user_suffix: str
    assistant_suffix: str
    conditioned_append: str
    unconditional_append: str
    system_prefix: str = ""
    system_message: str = ""
    system_suffix: str = ""
    add_special_tokens: bool = True

    def encode(self, tokenizer: Any, *, text: str, conditioned: bool) -> tuple[int, ...]:
        """Frame and tokenize either the conditioned or unconditional diffusion prompt."""

        if tokenizer is None:
            raise ValueError("the configured generation prompt requires a tokenizer")
        append = self.conditioned_append if conditioned else self.unconditional_append
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
            for value in tokenizer.encode(framed, add_special_tokens=self.add_special_tokens)
        )


_BAGEL_VIT_MIN_SIZE = 224
_BAGEL_VAE_MIN_SIZE = 512
_BAGEL_VAE_MAX_SIZE = 1024
_BAGEL_VAE_STRIDE = 16
_BAGEL_MAX_IMAGE_PIXELS = 14 * 14 * 9 * 1024


def bagel_processor(config: BagelConfig) -> ImageProcessor:
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


def sensenova_processor(config: NeoChatConfig) -> ImageProcessor:
    """Build the checkpoint architecture's caller-owned image transforms."""

    return ImageProcessor(
        vit=PatchTransform(
            patch_size=int(config.vision.patch_size),
            downsample_ratio=float(config.vision.downsample_ratio),
            min_pixels=512 * 512,
            max_pixels=2048 * 2048,
        ),
        staging_dtype="bfloat16",
        feature_injection=FeatureInjection(
            layout=FeatureLayout.DIRECT,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            start_token="<img>",
            end_token="</img>",
        ),
    )


def stub_processor() -> ImageProcessor:
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
            StrideResize(max_size=512, min_size=16, stride=16, max_pixels=512 * 512)
        ),
        staging_dtype="bfloat16",
        feature_injection=FeatureInjection(
            layout=FeatureLayout.DIRECT,
            positions=PositionLayout.TEMPORAL_SPATIAL,
            end_token_id=1007,
        ),
    )


_FLOW_SYSTEM_MESSAGE = (
    "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
    "You support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n"
    "2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. If no language is specified, use the user's input language."
)

SENSENOVA_PROMPT = FlowPrompt(
    user_prefix="<|im_start|>user\n",
    user_suffix="<|im_end|>\n",
    assistant_suffix="<|im_start|>assistant\n",
    conditioned_append="<think>\n\n</think>\n\n<img>",
    unconditional_append="<img>",
    system_prefix="<|im_start|>system\n",
    system_message=_FLOW_SYSTEM_MESSAGE,
    system_suffix="<|im_end|>\n",
)


def resolve_input_tokens(
    processor: ImageProcessor | None, tokenizer: Any | None
) -> ImageProcessor | None:
    """Resolve model-specific input token identities from tokenizer metadata."""

    if processor is None or processor.feature_injection is None:
        return processor
    injection = processor.feature_injection
    updates: dict[str, int] = {}
    for token_field, id_field in (("start_token", "start_token_id"), ("end_token", "end_token_id")):
        token = getattr(injection, token_field)
        token_id = getattr(injection, id_field)
        if token_id is not None or token is None:
            continue
        if tokenizer is None:
            raise ValueError(f"model input declaration requires tokenizer resolution for {token!r}")
        resolved = tokenizer.convert_tokens_to_ids(token)
        if (
            resolved is None
            or int(resolved) < 0
            or (resolved == tokenizer.unk_token_id and token != tokenizer.unk_token)
        ):
            raise ValueError(f"tokenizer does not define declared token {token!r}")
        updates[id_field] = int(resolved)
    if not updates:
        return processor
    resolved_injection = replace(
        injection,
        start_token_id=updates.get("start_token_id", injection.start_token_id),
        end_token_id=updates.get("end_token_id", injection.end_token_id),
    )
    return replace(processor, feature_injection=resolved_injection)
