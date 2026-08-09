"""Checkpoint mapping schema consumed only by the startup loaders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from ..foundation.errors import invalid_descriptor


class UnmatchedWeightPolicy(StrEnum):
    KEEP = "keep"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Rename:
    source: str
    target: str
    exact: bool = False
    substitutions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source or (self.exact and not self.target):
            raise invalid_descriptor("weight rename endpoints must not be empty")


@dataclass(frozen=True, slots=True)
class Slice:
    source: str
    target: str
    axis: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.start < 0 or self.stop <= self.start:
            raise invalid_descriptor("weight slice declaration is invalid")


@dataclass(frozen=True, slots=True)
class Split:
    source: str
    targets: tuple[str, ...]
    axis: int
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.targets or len(self.targets) != len(self.sizes) or any(size < 1 for size in self.sizes):
            raise invalid_descriptor("weight split targets and positive sizes must align")


@dataclass(frozen=True, slots=True)
class Stack:
    target: str
    source: str
    part: str | int

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise invalid_descriptor("weight stack endpoints must not be empty")


@dataclass(frozen=True, slots=True)
class Transpose:
    source: str
    target: str
    axes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.axes:
            raise invalid_descriptor("weight transpose declaration is invalid")


@dataclass(frozen=True, slots=True)
class Reshape:
    source: str
    target: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.shape or any(dimension == 0 or dimension < -1 for dimension in self.shape):
            raise invalid_descriptor("weight reshape declaration is invalid")


@dataclass(frozen=True, slots=True)
class Shard:
    source: str
    target: str
    axis: int
    topology_axis: str = "tp"

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.topology_axis:
            raise invalid_descriptor("weight shard declaration is invalid")


@dataclass(frozen=True, slots=True)
class Tie:
    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise invalid_descriptor("weight tie declaration is invalid")


@dataclass(frozen=True, slots=True)
class Cast:
    source: str
    target: str
    dtype: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.dtype:
            raise invalid_descriptor("weight cast declaration is invalid")


@dataclass(frozen=True, slots=True)
class Quantize:
    source: str
    target: str
    scheme: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.scheme:
            raise invalid_descriptor("weight quantization declaration is invalid")


WeightTransform: TypeAlias = Rename | Slice | Split | Stack | Transpose | Reshape | Shard | Tie | Cast | Quantize


@dataclass(frozen=True, slots=True)
class Sidecar:
    file: str
    module: str
    optional_substrings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.file or not self.module:
            raise invalid_descriptor("sidecar file and module must not be empty")


@dataclass(frozen=True, slots=True)
class TowerSplit:
    generation_prefixes: tuple[str, ...] = ()
    generation_infixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WeightTarget:
    name: str
    shape: tuple[int, ...]
    dtype: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or any(dimension < 0 for dimension in self.shape) or not self.dtype:
            raise invalid_descriptor("weight target declaration is invalid")


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Closed checkpoint-to-parameter mapping consumed at startup."""

    files: tuple[str, ...] = ()
    sidecars: tuple[Sidecar, ...] = ()
    transforms: tuple[WeightTransform, ...] = ()
    targets: tuple[WeightTarget, ...] = ()
    unmatched: UnmatchedWeightPolicy = UnmatchedWeightPolicy.KEEP
    tower: TowerSplit | None = None

    def __post_init__(self) -> None:
        names = tuple(target.name for target in self.targets)
        if len(set(names)) != len(names):
            raise invalid_descriptor("weight target names must be unique")
        if any(not file for file in self.files):
            raise invalid_descriptor("checkpoint file names must not be empty")
        stack_sources = tuple(transform.source for transform in self.transforms if isinstance(transform, Stack))
        if len(set(stack_sources)) != len(stack_sources):
            raise invalid_descriptor("weight stack sources must be unique")


class ModelLoadScope(StrEnum):
    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"

    @property
    def tower_role(self) -> str | None:
        if self is ModelLoadScope.UNDERSTANDING:
            return "und"
        if self is ModelLoadScope.GENERATION:
            return "gen"
        return None


__all__ = [
    "Cast",
    "ModelLoadScope",
    "Quantize",
    "Rename",
    "Reshape",
    "Shard",
    "Sidecar",
    "Slice",
    "Split",
    "Stack",
    "Tie",
    "TowerSplit",
    "Transpose",
    "UnmatchedWeightPolicy",
    "WeightSpec",
    "WeightTarget",
    "WeightTransform",
]
