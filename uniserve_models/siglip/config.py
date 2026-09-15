"""Immutable SigLIP patch and transformer architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Widths, depth, and normalization epsilon of the SigLIP encoder stack."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    layer_norm_eps: float

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.hidden_size,
                    self.num_attention_heads,
                    self.intermediate_size,
                    self.num_hidden_layers,
                )
            )
            or self.hidden_size % self.num_attention_heads
        ):
            raise ValueError(
                "SigLIP widths, layers and heads must be positive "
                "and compatible"
            )
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError(
                "SigLIP layer norm epsilon must be finite and positive"
            )


@dataclass(frozen=True)
class Config:
    """Patch geometry plus the encoder stack configuration for one SigLIP tower."""  # noqa: E501

    patch_size: int
    image_size: int
    num_channels: int
    encoder: TransformerConfig

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value < 1
                for value in (
                    self.patch_size,
                    self.image_size,
                    self.num_channels,
                )
            )
            or self.image_size % self.patch_size
        ):
            raise ValueError(
                "SigLIP image and patch dimensions must be positive and aligned"
            )
