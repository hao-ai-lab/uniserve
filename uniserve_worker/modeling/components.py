"""Logical model components and their numerical participation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Call(StrEnum):
    """Closed numerical capabilities, independent of worker operations."""

    TEXT = "text"
    ENCODE_TEXT = "encode:text"
    ENCODE_VISION = "encode:vision"
    ENCODE_LATENT = "encode:latent"
    ENCODE_CONDITIONING = "encode:conditioning"
    DIFFUSION = "diffusion"
    DECODE_IMAGE = "decode:image"
    DECODE_VIDEO = "decode:video"
    DECODE_AUDIO = "decode:audio"
    POSTPROCESS_VIDEO = "postprocess:video"


@dataclass(frozen=True, slots=True)
class CallSpec:
    """Pipeline stage participation and mathematical communication axes."""

    call: Call
    stage: Literal["all", "first", "last"] = "all"
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.call, Call) or self.stage not in {"all", "first", "last"}:
            raise ValueError("component calls require a numerical capability and pipeline stage")
        if len(set(self.groups)) != len(self.groups) or any(
            axis not in {"tp", "pp", "sp", "ulysses", "cp", "cp_row", "cp_col"}
            for axis in self.groups
        ):
            raise ValueError("component communication must name distinct mathematical axes")


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """One logical network role, its calls, and any shared parameter paths."""

    name: str
    calls: tuple[CallSpec, ...]
    shared_parameters: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.calls:
            raise ValueError("model components require a role and numerical calls")
        if len({call.call for call in self.calls}) != len(self.calls):
            raise ValueError("a component may declare each numerical call only once")
