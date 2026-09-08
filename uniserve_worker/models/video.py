"""Common model boundary for bounded diffusion and video execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from ..execution.batch import MediaGeometry, MediaTrack
from ..execution.bounded_storage import BoundedTensorStorage
from ..nn.parallel_attention import AttentionContextWorkspace
from .runtime import ExecutionModel


@dataclass(frozen=True, slots=True)
class VideoOutputGeometry:
    """Defines raster size, frame rate, frame count, and reconstruction unit size for video output."""

    frame_count: int
    unit_frames: tuple[int, ...]
    width: int
    height: int
    frame_rate: int
    audio_rate: int


MetadataT = TypeVar("MetadataT")
TensorViewsT = TypeVar("TensorViewsT")


class VideoModel(ExecutionModel, Generic[MetadataT, TensorViewsT], ABC):
    """Numerical operations over explicitly supplied media tensor views."""

    owns_media_output: bool = False
    output_capacity: VideoOutputGeometry
    decode_frame_capacity: int
    text_encoder: torch.nn.Module | None
    denoiser: torch.nn.Module | None
    conditioner: torch.nn.Sequential | None

    @abstractmethod
    def execution_key(self, geometry: MediaGeometry) -> Hashable:
        """Validate geometry and identify reusable mathematical metadata."""

        raise NotImplementedError

    @abstractmethod
    def build_execution(
        self,
        geometry: MediaGeometry,
        storage: BoundedTensorStorage,
        context: AttentionContextWorkspace | None,
    ) -> MetadataT:
        """Materialize immutable mathematical metadata for the public shape cache."""

        raise NotImplementedError

    @abstractmethod
    def request_tensors(
        self, storage: BoundedTensorStorage, geometry: MediaGeometry, metadata: MetadataT
    ) -> TensorViewsT:
        """Borrow immutable views for the declared geometry from public storage."""

        raise NotImplementedError

    @abstractmethod
    def prepare_tensors(
        self,
        tensors: TensorViewsT,
        metadata: MetadataT,
        encoded: torch.Tensor | None,
        text_rows: int,
    ) -> None:
        """Install computed conditioning and mathematical metadata into request tensor views."""

        raise NotImplementedError

    @abstractmethod
    def initialize_tensors(
        self, tensors: TensorViewsT, seed: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Fill reserved initial values and return destination/source pairs to stage.

        Numerical initialization may use CPU RNG. The public executor owns
        copies and stream ordering; request storage retains every source until
        the preparation operation physically completes.
        """

        raise NotImplementedError

    @abstractmethod
    def decoder_input(
        self,
        metadata: MetadataT,
        latents: torch.Tensor,
        track: MediaTrack,
        cursor: int,
        max_units: int,
    ) -> torch.Tensor:
        """Pack immutable latent rows into the selected decoder's numerical layout."""

        raise NotImplementedError

    @abstractmethod
    def decoder_output(
        self, metadata: MetadataT, value: torch.Tensor, track: MediaTrack
    ) -> torch.Tensor:
        """Expose the logical decoded tensor, including the requested sample extent."""

        raise NotImplementedError

    @abstractmethod
    def assemble_video(
        self,
        tensors: TensorViewsT,
        metadata: MetadataT,
        segments: torch.Tensor,
        start_unit: int,
        unit_count: int,
    ) -> torch.Tensor:
        """Apply numerical overlap and pixel transforms to ordered video segments."""

        raise NotImplementedError

    @abstractmethod
    def output_geometry(self, geometry: MediaGeometry) -> VideoOutputGeometry:
        """Describe the raster, timing, and unit boundaries used by output muxing."""

        raise NotImplementedError


__all__ = ["VideoOutputGeometry", "VideoModel"]
