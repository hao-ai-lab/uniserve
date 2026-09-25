"""Immutable SigLIP patch and transformer architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Widths, depth, and normalization epsilon of the SigLIP encoder stack.

    ``__post_init__`` rejects non-integer or non-positive sizes, a
    ``hidden_size`` that ``num_attention_heads`` does not divide (the head
    width is their quotient), and a non-finite or non-positive epsilon.
    """

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
        eps = self.layer_norm_eps
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError(
                "SigLIP layer norm epsilon must be finite and positive"
            )


@dataclass(frozen=True)
class Config:
    """Patch geometry plus the encoder stack configuration for one SigLIP tower."""  # noqa: E501

    # Pixel side of one square patch.
    patch_size: int

    # Pixel side of the square resolution the learned position table covers:
    # the table has ``(image_size // patch_size) ** 2`` entries, and
    # ``Encoder.forward`` rejects grids with a larger side in patches.
    image_size: int

    # Pixel channels; one patch row holds ``num_channels * patch_size**2``
    # values.
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
