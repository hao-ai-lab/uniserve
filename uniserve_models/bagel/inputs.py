"""Borrowed numerical inputs for BAGEL image denoising."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionInput, DenseInput


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Framed image sequences with temporal/row/column positions [3, tokens].

    Each positions tensor includes both marker rows. Its spatial coordinates
    on interior rows index the learned latent grid; marker spatial coordinates
    are unused. Temporal coordinates apply to every row through shared RoPE.

    Attributes:
        positions: One [3, sequence length] coordinate tensor per image.
        sequence_lengths: Tokens per image: its latent patch rows plus the two
            markers.
        attention: Attention metadata over the images' sequences, packed in
            batch order.
    """

    positions: tuple[torch.Tensor, ...]
    sequence_lengths: tuple[int, ...]
    attention: AttentionInput

    def __post_init__(self):
        super().__post_init__()
        if (
            len(self.positions) != self.batch_size
            or len(self.sequence_lengths) != self.batch_size
        ):
            raise ValueError("BAGEL coordinates must align with image samples")
        if any(
            position.shape != (3, count) or count < 3
            for position, count in zip(
                self.positions, self.sequence_lengths, strict=True
            )
        ):
            raise ValueError(
                "BAGEL positions require three axes and two framing markers"
            )
        # Query lengths are compared through their optional host mirror only,
        # so validation never reads device lengths.
        if (
            not isinstance(self.attention, DenseInput)
            and self.attention.queries.host is not None
            and self.attention.queries.host != self.sequence_lengths
        ):
            raise ValueError(
                "BAGEL attention lengths must match its framed image sequences"
            )
