"""Numerical cache partitions and logical tensor result geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from ..foundation.errors import invalid_descriptor
from ..transfer.layout import TensorRegion
from .tensors import ImageRange, TokenSelection

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})


@dataclass(frozen=True, slots=True)
class TextShape:
    """Token extent, logical rows, and optional text-result selection.

    No selection describes backbone activations. A selection describes hidden
    or vocabulary rows after projection; inactive selected rows may have zero
    query tokens. Encoder calls require a positive unselected token extent.
    """

    tokens: int
    rows: int = 1
    selection: TokenSelection | None = None

    def __post_init__(self) -> None:
        if self.tokens < 0 or self.rows < 1 or (self.tokens == 0 and self.selection is None):
            raise ValueError("text shape requires nonnegative tokens and positive sequence extents")


@dataclass(frozen=True, slots=True)
class MediaShape:
    """Numerical raster, sequence lengths, and optional logical unit interval.

    ``dtype`` supplies the input representation for operations that preserve it,
    such as RGB patch decoding. Learned decoders declare their parameter dtype.
    """

    height: int
    width: int
    frames: int = 1
    prompt_tokens: int = 0
    audio_frames: int = 0
    unit_start: int = 0
    unit_count: int = 0
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if (
            min(self.height, self.width, self.frames) < 1
            or min(self.prompt_tokens, self.audio_frames, self.unit_start, self.unit_count) < 0
        ):
            raise ValueError("media shape has invalid spatial or temporal extents")


Shape: TypeAlias = TextShape | MediaShape


@dataclass(frozen=True, slots=True)
class VideoShape:
    """Logical raster, timing, and reconstruction extents of a complete video."""

    frame_count: int
    unit_frames: tuple[int, ...]
    width: int
    height: int
    frame_rate: int
    audio_rate: int

    def __post_init__(self) -> None:
        if min(self.frame_count, self.width, self.height, self.frame_rate, self.audio_rate) < 1:
            raise ValueError("video dimensions and rates must be positive")
        if (
            not self.unit_frames
            or min(self.unit_frames) < 1
            or sum(self.unit_frames) != self.frame_count
        ):
            raise ValueError("reconstruction units must cover the complete video")


@dataclass(frozen=True, slots=True)
class DecodeWindow:
    """One logical temporal reconstruction interval, independent of placement.

    Latent and output frame intervals are half-open. ``crop`` removes decoder
    boundary frames before postprocessing. A cropped segment contains a body,
    padding, then its successor overlap. Only the final window appends that
    overlap to its output; other windows retain it in caller-owned state.
    """

    latent_start: int
    latent_stop: int
    frame_start: int
    frame_stop: int
    body_frames: int
    overlap_frames: int
    padding_frames: int
    crop: tuple[int, int] = (0, 0)
    final: bool = False

    def __post_init__(self) -> None:
        if (
            min(self.latent_start, self.frame_start, self.padding_frames, *self.crop) < 0
            or self.latent_stop <= self.latent_start
            or min(self.body_frames, self.overlap_frames) < 1
        ):
            raise ValueError("decode window has invalid temporal extents")
        frames = self.body_frames + (self.overlap_frames if self.final else 0)
        if self.frame_stop - self.frame_start != frames:
            raise ValueError("decode window output must cover its body and final overlap")

    @property
    def segment_frames(self) -> int:
        """Length after decoder boundary cropping and before overlap processing."""

        return self.body_frames + self.padding_frames + self.overlap_frames


@dataclass(frozen=True, slots=True)
class TensorOutputLayout:
    """Logical result shape and the unique region produced by this rank.

    A missing shape uses the declared capacity. A missing region produces the
    complete tensor. Storage reservation and publication remain runtime-owned.
    """

    shape: tuple[int, ...] | None = None
    region: TensorRegion | None = None
    value_range: ImageRange | None = None


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Defines local KV storage and its layer/head region in the logical cache."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    total_kv_heads: int
    kv_head_offset: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None
    total_layers: int | None = None
    layer_offset: int = 0

    def __post_init__(self) -> None:
        """Validate physical dimensions, global head coverage, and numeric format."""

        for name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, name)) < 1:
                raise invalid_descriptor(f"cache geometry {name} must be positive")
        total_layers = self.num_layers if self.total_layers is None else self.total_layers
        object.__setattr__(self, "total_layers", total_layers)
        if self.layer_offset < 0 or self.layer_offset + self.num_layers > total_layers:
            raise invalid_descriptor("cache layer interval exceeds logical model geometry")
        if self.kv_head_offset < 0 or self.kv_head_offset + self.num_kv_heads > self.total_kv_heads:
            raise invalid_descriptor("cache head interval exceeds logical model geometry")
        if self.dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor("cache compute dtype is unsupported")
        if self.store_dtype is not None and self.store_dtype not in _KV_DTYPES:
            raise invalid_descriptor("cache storage dtype is unsupported")
