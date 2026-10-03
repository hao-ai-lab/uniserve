"""Borrowed numerical inputs and conditioning for SenseNova U1 images.

The worker's ``ImageBuilder`` supplies samples, positions and attention metadata
to ``Denoiser.bind_inputs``, which derives the per-step image conditioning and
returns a ``DenoiserInput``. These dataclasses hold borrowed tensors; they
neither copy nor own them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.media import image
from uniserve.model import DenoiserInput as NumericalDenoiserInput
from uniserve.nn.attention import AttentionBatch


@dataclass(frozen=True)
class ImageConditioning:
    """Borrow the complete noisy NCHW image, input patch grid and noise scale.

    ``__post_init__`` checks shapes only; dtypes and devices are the caller's
    responsibility.

    Attributes:
        pixels: ``[1, 3, height, width]`` image in pixel space.
        grid: ``[1, 2]`` integer vision patch grid ``(rows, columns)``, from
            which ``vision.Encoder`` derives its rotary coordinates.
        noise_scale: One-element scale of this image's initial noise, which
            the denoiser normalizes and embeds when the checkpoint enables
            the noise-scale embedding.
    """  # noqa: E501

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
                "SenseNova image conditioning requires one NCHW RGB image, "
                "grid and scalar noise scale"
            )


@dataclass(frozen=True)
class DenoiserInput(NumericalDenoiserInput[image.Config]):
    """Image tokens with temporal/height/width positions [3, tokens] per sample.

    Temporal coordinates apply to every token through axial RoPE; height and
    width index each token's patch within its image grid.

    Attributes:
        images: Per-sample conditioning, which ``Denoiser.bind_inputs``
            rebuilds from the current samples on every step.
        positions: Per-sample ``[3, tokens]`` rotary coordinates.
        sequence_lengths: Host token count of each sample; the denoiser
            requires it to equal the sample's latent rows.
        attention: Attention metadata for the packed image tokens. When it is
            not dense and carries host query lengths, ``__post_init__``
            requires them to equal ``sequence_lengths``.
    """

    images: tuple[ImageConditioning, ...]
    positions: tuple[torch.Tensor, ...]
    sequence_lengths: tuple[int, ...]
    attention: AttentionBatch

    def __post_init__(self):
        super().__post_init__()
        if any(
            len(values) != self.batch_size
            for values in (self.images, self.positions, self.sequence_lengths)
        ):
            raise ValueError(
                "SenseNova image conditioning and positions "
                "must align with samples"
            )
        if any(
            position.shape != (3, count)
            for position, count in zip(
                self.positions, self.sequence_lengths, strict=True
            )
        ):
            raise ValueError(
                "SenseNova image positions must cover three axes "
                "per image token"
            )
        queries = self.attention.queries
        if (
            queries is not None
            and queries.host is not None
            and queries.host != self.sequence_lengths
        ):
            raise ValueError(
                "SenseNova attention lengths must match its image sequences"
            )
