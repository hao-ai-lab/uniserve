"""Immutable model-parallel degrees, independent of process params."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import prod
from typing import Mapping


def _degree(name: str, value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


# Each tag admits exactly these degree fields. Inactive axes are absent from
# both the immutable value and its serialized configuration.
_SEQUENCE_FIELDS = {
    "local": (),
    "ulysses": ("ulysses_degree",),
    "ring": ("ring_degree",),
    "hybrid": ("ulysses_degree", "ring_degree"),
    "allgather": ("allgather_degree",),
    "attention2d": ("attn2d_row_size", "attn2d_col_size", "ulysses_degree"),
}


@dataclass(frozen=True, slots=True)
class SequenceParallel:
    """A closed attention strategy with only its active parallel degrees.

    Degree order follows the serialized fields for the selected strategy. Peer
    K/V mapping is the algorithm currently selected by the ring/hybrid tags;
    these tags do not imply partial-softmax ring reduction.
    """

    kind: str = "local"
    degrees: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        names = _SEQUENCE_FIELDS.get(self.kind)
        if names is None:
            raise ValueError(f"unknown sequence parallel strategy {self.kind!r}")
        if not isinstance(self.degrees, tuple) or len(self.degrees) != len(names):
            raise ValueError(f"sequence strategy {self.kind!r} requires degrees {names!r}")
        for name, degree in zip(names, self.degrees):
            _degree(name, degree)

    @property
    def size(self) -> int:
        return prod(self.degrees)

    @property
    def dimensions(self) -> tuple[tuple[str, int], ...]:
        """Context axes followed by Ulysses, in logical member order."""

        match self.kind:
            case "local":
                return (("cp", 1), ("ulysses", 1))
            case "ulysses":
                return (("cp", 1), ("ulysses", self.degrees[0]))
            case "ring" | "allgather":
                return (("cp", self.degrees[0]), ("ulysses", 1))
            case "hybrid":
                return (("cp", self.degrees[1]), ("ulysses", self.degrees[0]))
            case "attention2d":
                row, col, ulysses = self.degrees
                return (("cp_row", row), ("cp_col", col), ("ulysses", ulysses))
        raise ValueError(f"unknown sequence parallel strategy {self.kind!r}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SequenceParallel:
        """Parse a tagged strategy and reject degrees belonging to other tags."""

        kind = value.get("kind")
        if not isinstance(kind, str) or kind not in _SEQUENCE_FIELDS:
            raise ValueError(f"unknown sequence parallel strategy {kind!r}")
        names = _SEQUENCE_FIELDS[kind]
        unexpected = value.keys() - {"kind", *names}
        if unexpected:
            raise ValueError(f"sequence strategy {kind!r} has unknown fields: {sorted(unexpected)}")
        return cls(kind, tuple(_degree(name, value.get(name, 1)) for name in names))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, **dict(zip(_SEQUENCE_FIELDS[self.kind], self.degrees))}


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Logical degrees for a component; sequence degree is always derived.

    Parsing establishes geometry only. The model and backend must validate an
    executing binding before loading any parameter storage.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel: SequenceParallel = SequenceParallel()

    def __post_init__(self) -> None:
        _degree("tensor_parallel_size", self.tensor_parallel_size)
        _degree("pipeline_parallel_size", self.pipeline_parallel_size)
        if not isinstance(self.sequence_parallel, SequenceParallel):
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

        *context, ulysses = self.sequence_parallel.dimensions
        return (
            ("pp", self.pipeline_parallel_size),
            *context,
            ("tp", self.tensor_parallel_size),
            ulysses,
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
            sequence_parallel=SequenceParallel.from_dict(sequence),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel": self.sequence_parallel.to_dict(),
        }
