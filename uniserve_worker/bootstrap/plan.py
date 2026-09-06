"""Pure compilation of configured operation support into one worker plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..execution.batch import RunKind
from ..nn.parallel import ParallelConfig


class ModelLoadScope(StrEnum):
    """Enumerates the checkpoint scope materialized by a worker role."""

    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """Defines the model scope and operation variants assigned to one worker."""

    model_scope: ModelLoadScope
    allowed_work_variants: frozenset[RunKind]
    components: tuple[tuple[str, ComponentDeployConfig], ...] = ()


def resolve_worker_plan(
    supported_ops: frozenset[RunKind],
    components: tuple[tuple[str, ComponentDeployConfig], ...] = (),
    rank: int = 0,
) -> WorkerPlan:
    """Map supported operation kinds to the model scope and executable variants for one worker."""

    if not supported_ops:
        raise ValueError("worker pool must support at least one operation")
    return WorkerPlan(
        model_scope=ModelLoadScope.WHOLE,
        allowed_work_variants=supported_ops,
        components=tuple((name, config) for name, config in components if rank in config.ranks),
    )


__all__ = [
    "ComponentDeployConfig",
    "ModelLoadScope",
    "WorkerPlan",
    "parse_component_deployment",
    "resolve_worker_plan",
]


@dataclass(frozen=True, slots=True)
class ComponentDeployConfig:
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
    def from_dict(cls, value: dict[str, object]) -> ComponentDeployConfig:
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


def parse_component_deployment(
    value: dict[str, object], world_size: int
) -> tuple[tuple[str, ComponentDeployConfig], ...]:
    """Validate the authoritative expanded component assignments from the host."""

    components = []
    if not value:
        raise ValueError("component deployment must not be empty")
    for name, component in sorted(value.items()):
        if not name or not isinstance(component, dict):
            raise ValueError("component deployment requires named component objects")
        resolved = ComponentDeployConfig.from_dict(component)
        if any(rank >= world_size for rank in resolved.ranks):
            raise ValueError(f"component {name!r} contains ranks outside the process world")
        components.append((name, resolved))
    return tuple(components)
