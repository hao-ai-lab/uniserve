"""Common model boundary for bounded diffusion and video execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .runtime import ExecutionModel


class DecodeKind(StrEnum):
    """Selects video reconstruction, audio reconstruction, or final artifact assembly."""

    VIDEO = "video"
    AUDIO = "audio"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    """Returns a reconstructed tensor, optional PCM audio, and completed media-unit count."""

    kind: DecodeKind
    value: Any | None
    unit_offset: int
    unit_count: int


@dataclass(frozen=True, slots=True)
class VideoOutputGeometry:
    """Defines raster size, frame rate, frame count, and reconstruction unit size for video output."""

    frame_count: int
    unit_frames: tuple[int, ...]
    width: int
    height: int
    frame_rate: int
    audio_rate: int


class VideoRunner(ExecutionModel):
    """Model-owned prepare, denoise, bounded decode, and finalize behavior."""

    def prepare(self, batch) -> None:
        """Initialize persistent request state from an admitted media descriptor."""

        raise NotImplementedError

    def denoise(self, batch) -> None:
        """Advance resident media state through the requested solver steps."""

        raise NotImplementedError

    def decode(self, batch, cursor: int, max_units: int) -> DecodeOutput:
        """Materialize at most ``max_units`` outputs beginning at a bounded cursor."""

        raise NotImplementedError

    def finalize(self, request):
        """Assemble any terminal artifact after all bounded decode units complete."""

        raise NotImplementedError

    def output_geometry(self, request) -> VideoOutputGeometry:
        """Describe the raster, timing, and unit boundaries used by output muxing."""

        raise NotImplementedError

    def validate_run(self, runtime, batch) -> None:
        """Validate media operations against runtime state before execution."""

        from ..execution.video import validate_batch

        validate_batch(runtime, batch)

    def run_operation(self, runtime, state) -> bool:
        """Advance one prepared media operation and report terminal completion."""

        from ..execution.video import run_action

        return run_action(runtime, state)


__all__ = ["DecodeKind", "DecodeOutput", "VideoOutputGeometry", "VideoRunner"]
