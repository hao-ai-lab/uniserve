"""Caller-owned image transforms and diffusion prompt framing."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    normalization: Literal["imagenet", "signed_unit"] = "imagenet"


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
    normalization: Literal["imagenet", "signed_unit"] = "signed_unit"


@dataclass(frozen=True, slots=True)
class ImageProcessor:
    """Defines ViT and VAE transforms, staging dtype, and language-sequence feature injection."""

    vit: PatchTransform | TowerTransform | None = None
    vae: TowerTransform | None = None
    staging_dtype: torch.dtype | None = None
    feature_injection: FeatureInjection | None = None

    def __post_init__(self) -> None:
        """Require at least one image transform for the caller."""

        if self.vit is None and self.vae is None:
            raise ValueError("image processor must implement at least one transform")


__all__ = [
    "FlowPrompt",
    "BranchSource",
    "PositionLayout",
    "load_tokenizer",
    "resolve_input_tokens",
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


def load_tokenizer(path: Path):
    """Load a caller-owned tokenizer from a resolved local checkpoint directory."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, use_fast=False, trust_remote_code=False, local_files_only=True
    )
