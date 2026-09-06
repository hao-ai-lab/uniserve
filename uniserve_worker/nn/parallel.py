"""Immutable model-parallel degrees, independent of process placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import prod
from typing import ClassVar, Mapping, TypeAlias


def _degree(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class _SequenceConfig:
    kind: ClassVar[str]

    def __post_init__(self) -> None:
        for field in fields(self):
            _degree(field.name, getattr(self, field.name))

    @property
    def size(self) -> int:
        return prod(getattr(self, field.name) for field in fields(self))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, **asdict(self)}


@dataclass(frozen=True, slots=True)
class LocalSequence(_SequenceConfig):
    kind: ClassVar[str] = "local"


@dataclass(frozen=True, slots=True)
class UlyssesSequence(_SequenceConfig):
    ulysses_degree: int = 1
    kind: ClassVar[str] = "ulysses"


@dataclass(frozen=True, slots=True)
class RingSequence(_SequenceConfig):
    ring_degree: int = 1
    kind: ClassVar[str] = "ring"


@dataclass(frozen=True, slots=True)
class HybridSequence(_SequenceConfig):
    ulysses_degree: int = 1
    ring_degree: int = 1
    kind: ClassVar[str] = "hybrid"


@dataclass(frozen=True, slots=True)
class GatherSequence(_SequenceConfig):
    allgather_degree: int = 1
    kind: ClassVar[str] = "allgather"


@dataclass(frozen=True, slots=True)
class Attention2DSequence(_SequenceConfig):
    attn2d_row_size: int = 1
    attn2d_col_size: int = 1
    ulysses_degree: int = 1
    kind: ClassVar[str] = "attention2d"


SequenceParallel: TypeAlias = (
    LocalSequence
    | UlyssesSequence
    | RingSequence
    | HybridSequence
    | GatherSequence
    | Attention2DSequence
)


def sequence_from_dict(value: Mapping[str, object]) -> SequenceParallel:
    """Parse one tagged strategy; degrees from other strategies are errors."""

    strategies = (
        LocalSequence,
        UlyssesSequence,
        RingSequence,
        HybridSequence,
        GatherSequence,
        Attention2DSequence,
    )
    kind = value.get("kind")
    for strategy in strategies:
        if kind == strategy.kind:
            arguments = {name: item for name, item in value.items() if name != "kind"}
            unexpected = arguments.keys() - {field.name for field in fields(strategy)}
            if unexpected:
                raise ValueError(
                    f"sequence strategy {kind!r} has unknown fields: {sorted(unexpected)}"
                )
            return strategy(**{name: _degree(name, value) for name, value in arguments.items()})
    raise ValueError(f"unknown sequence parallel strategy {kind!r}")


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Logical degrees for a component; sequence degree is always derived.

    Parsing establishes geometry only. The model and backend must validate an
    executing binding before loading any parameter storage.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: SequenceParallel = LocalSequence()

    def __post_init__(self) -> None:
        _degree("tensor_parallel_size", self.tensor_parallel_size)
        _degree("pipeline_parallel_size", self.pipeline_parallel_size)
        if not isinstance(self.sequence_parallel, _SequenceConfig):
            raise ValueError("sequence_parallel must be a tagged sequence configuration")

    @property
    def sequence_parallel_size(self) -> int:
        return self.sequence_parallel.size

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.sequence_parallel_size

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        """Independent dimensions in rank order, with Ulysses varying fastest."""

        sequence = self.sequence_parallel
        context: tuple[tuple[str, int], ...]
        if isinstance(sequence, Attention2DSequence):
            context = (("cp_row", sequence.attn2d_row_size), ("cp_col", sequence.attn2d_col_size))
        elif isinstance(sequence, (RingSequence, HybridSequence)):
            context = (("cp", sequence.ring_degree),)
        elif isinstance(sequence, GatherSequence):
            context = (("cp", sequence.allgather_degree),)
        else:
            context = (("cp", 1),)
        ulysses = (
            sequence.ulysses_degree
            if isinstance(sequence, (UlyssesSequence, HybridSequence, Attention2DSequence))
            else 1
        )
        return (
            ("pp", self.pipeline_parallel_size),
            *context,
            ("tp", self.tensor_parallel_size),
            ("ulysses", ulysses),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ParallelConfig:
        unexpected = value.keys() - {field.name for field in fields(cls)}
        if unexpected:
            raise ValueError(f"unknown parallel_config fields: {sorted(unexpected)}")
        sequence = value.get("sequence_parallel", {"kind": "local"})
        if not isinstance(sequence, Mapping):
            raise ValueError("sequence_parallel must be a tagged object")
        return cls(
            tensor_parallel_size=_degree(
                "tensor_parallel_size", value.get("tensor_parallel_size", 1)
            ),
            pipeline_parallel_size=_degree(
                "pipeline_parallel_size", value.get("pipeline_parallel_size", 1)
            ),
            sequence_parallel=sequence_from_dict(sequence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel": self.sequence_parallel.to_dict(),
        }
