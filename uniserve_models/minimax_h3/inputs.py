"""Explicit H3 sample sizes, conditioning and borrowed attention indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.distributed import Communicator
from uniserve.media import image
from uniserve.model import DenoiserInput as BaseDenoiserInput
from uniserve.nn.attention import vsa

from .packing import FRAME_SIZES, Packing, video_latent_frames


@dataclass(frozen=True, slots=True)
class DenoiserSize:
    """Describe one sample's output timeline, raster and conditioning length.

    ``num_frames`` counts output video frames at 24 fps and must have the form
    ``17 * n + 5`` with ``n`` positive; ``video_latent_frames`` raises
    otherwise. ``frame`` is the output raster, one of ``FRAME_SIZES``.
    ``num_text_tokens`` is either a request's exact prompt length
    or, for a layout (see ``Denoiser.layout_size``), that length rounded up to
    whole 64-row tiles.
    """

    num_frames: int
    frame: image.Config
    num_text_tokens: int

    def __post_init__(self):
        # Raises ValueError for a frame count H3 does not generate.
        video_latent_frames(self.num_frames)
        if (
            not isinstance(self.frame, image.Config)
            or (self.frame.height, self.frame.width) not in FRAME_SIZES
        ):
            raise ValueError("H3 generates 1344x768 or 768x1344 video")
        if type(self.num_text_tokens) is not int or self.num_text_tokens < 1:
            raise ValueError(
                "H3 conditioning must contain a positive number of text tokens"
            )


@dataclass(frozen=True)
class DenoiserInput(BaseDenoiserInput[DenoiserSize]):
    """Carry ordered video/audio latents with one refined text tensor per sample."""  # noqa: E501

    # Refined text from ``Conditioner.encode``, one [text rows, hidden_size]
    # tensor per sample. On the first pipeline stage, ``Denoiser.forward``
    # requires its rows to match the layout's text rows; callers zero the
    # rows past the prompt.
    text_features: tuple[torch.Tensor, ...]

    def __post_init__(self):
        super().__post_init__()
        if (
            tuple(self.latents) != ("video", "audio")
            or len(self.text_features) != self.batch_size
        ):
            raise ValueError(
                "H3 inputs require video/audio latents "
                "and one condition per sample"
            )


@dataclass(frozen=True, slots=True)
class AttentionInput:
    """Address one token shard within the complete mathematical packing.

    ``__post_init__`` checks that ``token_slice`` is ``group``'s equal share of
    the padded rows, that ``vsa`` describes the same row count, and that the
    local index tables are 1-D int64.

    Attributes:
        packing: Complete packing of the layout being evaluated.
        token_slice: Global packed rows this sequence rank holds.
        group: Sequence-parallel group whose rank selects ``token_slice``.
        vsa: Tile domains and borrowed index views shared by every layer's
            sparse attention.
        local_text_indices: Int64 shard rows, relative to
            ``token_slice.start``, that hold text tokens.
        local_video_indices: Int64 shard rows that hold video tokens, in
            packed order.
        local_audio_indices: Int64 shard rows that hold audio tokens.
    """

    packing: Packing
    token_slice: slice
    group: Communicator
    vsa: vsa.Input
    local_text_indices: torch.Tensor
    local_video_indices: torch.Tensor
    local_audio_indices: torch.Tensor

    def __post_init__(self):
        tokens = self.packing.padded_tokens
        if tokens % self.group.size:
            raise ValueError(
                "H3 tokens must divide their numerical sequence group"
            )
        count = tokens // self.group.size
        if self.token_slice != slice(
            self.group.rank * count, (self.group.rank + 1) * count
        ):
            raise ValueError(
                "H3 token slice must match the logical sequence rank"
            )
        if self.vsa.padded_tokens != tokens:
            raise ValueError(
                "H3 attention and packing must describe the same token domain"
            )

        for indices in (
            self.local_text_indices,
            self.local_video_indices,
            self.local_audio_indices,
        ):
            if indices.ndim != 1 or indices.dtype != torch.int64:
                raise ValueError(
                    "H3 modality indices must be int64 token vectors"
                )
