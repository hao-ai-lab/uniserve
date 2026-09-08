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


@dataclass(frozen=True, slots=True)
class EntryConfig:
    """Ordered process membership and logical parallelism for one model component."""

    ranks: tuple[int, ...]
    parallel_config: ParallelConfig = ParallelConfig()
    distribution: str | None = None
    units_per_rank: int = 1

    def __post_init__(self) -> None:
        if not self.ranks or len(set(self.ranks)) != len(self.ranks):
            raise ValueError("component ranks must be unique and non-empty")
        if any(type(rank) is not int or rank < 0 for rank in self.ranks):
            raise ValueError("component ranks must be nonnegative integers")
        if self.distribution not in (None, "temporal_units"):
            raise ValueError(f"unsupported component distribution {self.distribution!r}")
        if type(self.units_per_rank) is not int or self.units_per_rank < 1:
            raise ValueError("units_per_rank must be a positive integer")
        if self.distribution is None and len(self.ranks) != self.parallel_config.world_size:
            raise ValueError("component membership must equal TP × sequence × pipeline degrees")
        if self.distribution is not None and self.parallel_config.world_size != 1:
            raise ValueError("temporal unit distribution requires a local decoder parallel_config")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EntryConfig:
        unknown = value.keys() - {"ranks", "parallel_config", "distribution", "units_per_rank"}
        if unknown:
            raise ValueError(f"unknown component fields: {sorted(unknown)}")
        ranks = value.get("ranks")
        parallel = value.get("parallel_config", {})
        if not isinstance(ranks, (list, tuple)) or not isinstance(parallel, dict):
            raise ValueError("component requires ranks and a parallel_config object")
        distribution = value.get("distribution")
        units = value.get("units_per_rank", 1)
        if distribution is not None and not isinstance(distribution, str):
            raise ValueError("component distribution must be a string")
        if type(units) is not int:
            raise ValueError("units_per_rank must be a positive integer")
        return cls(tuple(ranks), ParallelConfig.from_dict(parallel), distribution, units)

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "ranks": list(self.ranks),
            "parallel_config": self.parallel_config.to_dict(),
        }
        if self.distribution is not None:
            value.update(distribution=self.distribution, units_per_rank=self.units_per_rank)
        return value


def parse_entries(value: dict[str, object], world_size: int) -> tuple[tuple[str, EntryConfig], ...]:
    """Validate the authoritative expanded component assignments from the host."""

    components = []
    if not value:
        raise ValueError("entry configuration must not be empty")
    for name, component in sorted(value.items()):
        if not name or not isinstance(component, dict):
            raise ValueError("entry configuration requires named component objects")
        resolved = EntryConfig.from_dict(component)
        if any(rank >= world_size for rank in resolved.ranks):
            raise ValueError(f"component {name!r} contains ranks outside the process world")
        components.append((name, resolved))
    return tuple(components)
