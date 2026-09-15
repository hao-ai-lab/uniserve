"""Explicit H3 sample sizes, conditioning and borrowed attention indices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.distributed import Communicator
from uniserve.model import DenoiserInput as BaseDenoiserInput
from uniserve.nn.attention import vsa

from .packing import Packing, video_latent_frames


@dataclass(frozen=True, slots=True)
class DenoiserSize:
    """Describe one sample's output timeline and conditioning length."""

    num_frames: int
    num_text_tokens: int

    def __post_init__(self):
        video_latent_frames(self.num_frames)
        if type(self.num_text_tokens) is not int or self.num_text_tokens < 1:
            raise ValueError(
                "H3 conditioning must contain a positive number of text tokens"
            )


@dataclass(frozen=True)
class DenoiserInput(BaseDenoiserInput[DenoiserSize]):
    """Carry ordered video/audio latents with one refined text tensor per sample."""  # noqa: E501

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
    """Address one token shard within the complete mathematical packing."""

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
