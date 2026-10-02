"""Explicit H3 sample sizes, conditioning and borrowed attention indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.distributed import Communicator
from uniserve.media import image
from uniserve.model import DenoiserInput as BaseDenoiserInput
from uniserve.nn.attention import VisibleInput, vsa

from .packing import TilePacking, latent_raster, video_latent_frames


@dataclass(frozen=True, slots=True)
class DenoiserSize:
    """Describe one sample's output timeline, canvas and conditioning rows.

    ``num_frames`` counts output video frames at 24 fps and must have the form
    ``17 * n + 5`` with ``n`` positive; ``canvas`` is the output raster, sides
    multiples of 32. ``num_text_tokens`` and ``condition_rows`` are either a
    request's exact presented prompt length and condition rows or, for a
    layout (see ``Denoiser.layout_size``), the capacities of its text and
    condition regions.
    """

    num_frames: int
    canvas: image.Config
    num_text_tokens: int
    condition_rows: int

    def __post_init__(self):
        # Raise ValueError for a frame count or canvas H3 does not generate.
        video_latent_frames(self.num_frames)
        latent_raster(self.canvas)
        if type(self.num_text_tokens) is not int or self.num_text_tokens < 1:
            raise ValueError(
                "H3 conditioning must contain a positive number of text tokens"
            )
        if type(self.condition_rows) is not int or self.condition_rows < 0:
            raise ValueError("H3 condition rows must be a nonnegative count")


@dataclass(frozen=True)
class DenoiserInput(BaseDenoiserInput[DenoiserSize]):
    """Carry ordered video/audio latents with one prefix source per sample."""

    # The retained conditioning of each sample. Under sparse attention it is
    # the refined text over the layout's text rows, zero past the prompt;
    # under dense attention it is the prefix source the packed prefix rows
    # gather from: the text rows, then the projected condition rows, then
    # one zero row (see ``Denoiser.text_condition_rows``).
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


def _check_indices(*tables: torch.Tensor) -> None:
    for indices in tables:
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError("H3 modality indices must be int64 token vectors")


@dataclass(frozen=True, slots=True)
class AttentionInput:
    """Address one token shard within the complete tile packing.

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

    packing: TilePacking
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
        _check_indices(
            self.local_text_indices,
            self.local_video_indices,
            self.local_audio_indices,
        )


@dataclass(frozen=True, slots=True)
class SequenceInput:
    """Address one token shard of a dense layout.

    Attributes:
        token_slice: Global packed rows this sequence rank holds, its equal
            share of the padded rows.
        group: Sequence-parallel group whose rank selects ``token_slice``.
        visible: Attention over every padded row as one document, each query
            seeing the keys before the request's used row count.
        local_video_indices: Int64 shard rows that hold generated video
            rows, ascending.
        local_audio_indices: Int64 shard rows that hold generated audio
            rows, ascending.
    """

    token_slice: slice
    group: Communicator
    visible: VisibleInput
    local_video_indices: torch.Tensor
    local_audio_indices: torch.Tensor

    def __post_init__(self):
        tokens = self.visible.queries.num_tokens
        if tokens is None or tokens % self.group.size:
            raise ValueError(
                "H3 dense rows must divide their numerical sequence group"
            )
        count = tokens // self.group.size
        if self.token_slice != slice(
            self.group.rank * count, (self.group.rank + 1) * count
        ):
            raise ValueError(
                "H3 token slice must match the logical sequence rank"
            )
        _check_indices(self.local_video_indices, self.local_audio_indices)
