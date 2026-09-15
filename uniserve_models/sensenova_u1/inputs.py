"""Borrowed numerical inputs and conditioning for SenseNova U1 images."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionInput, DenseInput


@dataclass(frozen=True)
class ImageConditioning:
    """Borrow the complete noisy NCHW image, input patch grid and noise scale."""

    pixels: torch.Tensor
    grid: torch.Tensor
    noise_scale: torch.Tensor

    def __post_init__(self):
        if (
            self.pixels.ndim != 4
            or self.pixels.shape[:2] != (1, 3)
            or self.grid.shape != (1, 2)
            or self.noise_scale.numel() != 1
        ):
            raise ValueError(
                "SenseNova image conditioning requires one NCHW RGB image, grid and scalar noise scale"
            )


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Image tokens with temporal/height/width positions [3, tokens] per sample.

    Temporal coordinates apply to every token through axial RoPE; height and
    width index each token's patch within its image grid.
    """

    images: tuple[ImageConditioning, ...]
    positions: tuple[torch.Tensor, ...]
    sequence_lengths: tuple[int, ...]
    attention: AttentionInput

    def __post_init__(self):
        super().__post_init__()
        if any(
            len(values) != self.batch_size
            for values in (self.images, self.positions, self.sequence_lengths)
        ):
            raise ValueError("SenseNova image conditioning and positions must align with samples")
        if any(
            position.shape != (3, count)
            for position, count in zip(self.positions, self.sequence_lengths, strict=True)
        ):
            raise ValueError("SenseNova image positions must cover three axes per image token")
        if (
            not isinstance(self.attention, DenseInput)
            and self.attention.queries.host is not None
            and self.attention.queries.host != self.sequence_lengths
        ):
            raise ValueError("SenseNova attention lengths must match its image sequences")
