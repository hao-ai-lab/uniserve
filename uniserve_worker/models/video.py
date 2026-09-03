"""Common model boundary for bounded diffusion and video execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .runtime import ExecutionModel


class DecodeKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class DecodeOutput:
    kind: DecodeKind
    value: Any | None
    unit_offset: int
    unit_count: int


@dataclass(frozen=True, slots=True)
class VideoOutputGeometry:
    frame_count: int
    unit_frames: tuple[int, ...]
    width: int
    height: int
    frame_rate: int
    audio_rate: int


class VideoRunner(ExecutionModel):
    """Model-owned prepare, denoise, bounded decode, and finalize behavior."""

    def prepare(self, batch) -> None:
        raise NotImplementedError

    def denoise(self, batch) -> None:
        raise NotImplementedError

    def decode(self, batch, cursor: int, max_units: int) -> DecodeOutput:
        raise NotImplementedError

    def finalize(self, request):
        raise NotImplementedError

    def output_geometry(self, request) -> VideoOutputGeometry:
        raise NotImplementedError

    def validate_run(self, runtime, batch) -> None:
        from ..execution.video import validate_batch

        validate_batch(runtime, batch)

    def run_operation(self, runtime, state) -> bool:
        from ..execution.video import run_action

        return run_action(runtime, state)


__all__ = ["DecodeKind", "DecodeOutput", "VideoOutputGeometry", "VideoRunner"]
