"""Common model boundary for bounded diffusion and video execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

import torch

from ..execution.batch import MediaGeometry, MediaTrack, OpCode
from ..execution.bounded_storage import BoundedTensorStorage
from ..execution.denoising import DenoisingStep
from ..nn.diffusion.schedule import DiffusionSchedule
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


class MediaPlanRepeat(StrEnum):
    """Select how a generation-plan stage expands for one request."""

    ONCE = "once"
    FIXED = "fixed"
    VIDEO_UNITS = "video_units"


@dataclass(frozen=True, slots=True)
class MediaPlanStage:
    """Declare one schedulable media stage and its model-level dependencies."""

    name: str
    operation: OpCode
    entry: str
    dependencies: tuple[str, ...] = ()
    input_from: str | None = None
    repeat: MediaPlanRepeat = MediaPlanRepeat.ONCE
    count: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.entry:
            raise ValueError("media plan stages require names and component entries")
        if self.repeat is MediaPlanRepeat.FIXED and self.count < 1:
            raise ValueError("a fixed media stage requires a positive repetition count")
        if self.repeat is not MediaPlanRepeat.FIXED and self.count != 1:
            raise ValueError("only fixed media stages declare a repetition count")

    def to_mapping(self) -> dict[str, object]:
        """Encode this model declaration for worker discovery."""

        return {
            "name": self.name,
            "operation": self.operation.value,
            "entry": self.entry,
            "dependencies": list(self.dependencies),
            "input_from": self.input_from,
            "repeat": self.repeat.value,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class MediaExecutionPlan:
    """Finite component graph consumed by the shared media scheduler."""

    stages: tuple[MediaPlanStage, ...]

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if not names or len(set(names)) != len(names):
            raise ValueError("a media plan requires unique ordered stage names")
        declared: set[str] = set()
        for stage in self.stages:
            references = stage.dependencies + ((stage.input_from,) if stage.input_from else ())
            if any(reference not in declared for reference in references):
                raise ValueError(f"media stage {stage.name!r} references a later or missing stage")
            declared.add(stage.name)

    @property
    def denoise_steps(self) -> int:
        """Return the fixed learned-prediction count declared by this plan."""

        stages = [stage for stage in self.stages if stage.operation is OpCode.DIFFUSION_STEP]
        if len(stages) != 1 or stages[0].repeat is not MediaPlanRepeat.FIXED:
            return 0
        return stages[0].count

    @property
    def operations(self) -> frozenset[OpCode]:
        """Return the worker operation families required by the declared stages."""

        return frozenset(stage.operation for stage in self.stages)

    def to_mapping(self) -> dict[str, object]:
        """Encode the complete finite graph for worker discovery."""

        return {"stages": [stage.to_mapping() for stage in self.stages]}


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
    media_plan: MediaExecutionPlan

    @property
    def denoise_steps(self) -> int:
        """Expose the scheduler repetition count from the declared media plan."""

        return self.media_plan.denoise_steps

    @abstractmethod
    def denoising_signature(self, tensors: TensorViewsT, metadata: MetadataT) -> Hashable:
        """Identify a complete denoising call's dynamic numerical shape."""

        raise NotImplementedError

    @abstractmethod
    def bind_denoising_step(
        self, tensors: TensorViewsT, metadata: MetadataT, step: int, schedule: DiffusionSchedule
    ) -> DenoisingStep:
        """Bind one learned prediction and its solver state without advancing it."""

        raise NotImplementedError

    def media_geometry(self, media) -> MediaGeometry:
        """Resolve worker geometry from admitted media before allocating entry products."""

        return media.geometry

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


__all__ = [
    "MediaExecutionPlan",
    "MediaPlanRepeat",
    "MediaPlanStage",
    "VideoOutputGeometry",
    "VideoModel",
]
