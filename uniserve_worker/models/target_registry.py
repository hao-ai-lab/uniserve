"""Strict target model registry: explicit registrations, no probing.

Dormant Stage 6 slice from ``specs/unified_forward_execution.md``. The
target `ModelRegistration` is a frozen record binding architecture names to
one family, one closed advertised operation set, and the family's static
cache registration factory. Resolution is explicit: an unknown or ambiguous
architecture fails typed, no class is asked which methods it has, and no
generic fallback exists.

Registration validation proves the advertised operation set is exactly what
the family's closed cache schema can lower — the registry cannot advertise
an operation the lowerer has no regions for, and cannot silently drop one
the schema supports.

The concrete adapter factories (resident model construction) arrive with
the family ports at cutover; the current production registry in
``models/registry.py`` remains authoritative until then.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..contracts.cache_schema import FamilyCacheRegistration
from ..contracts.execution import OperationTag
from .cache_registrations import (
    bagel_cache_registration,
    qwen3_cache_registration,
    sensenova_cache_registration,
)

__all__ = [
    "ModelRegistration",
    "RegistryError",
    "TargetModelRegistry",
    "target_registry",
]


class RegistryError(ValueError):
    """A registration or resolution violates the strict registry contract."""


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """One family's explicit static registration."""

    family: str
    architectures: tuple[str, ...]
    operations: frozenset[OperationTag]
    cache_registration: Callable[..., FamilyCacheRegistration]

    def validate(self, **geometry: int) -> None:
        """Prove the advertised operations equal the schema's lowerable set."""

        registration = self.cache_registration(**geometry)
        lowerable = frozenset(
            region.operation_tag for region in registration.schema.regions
        )
        if self.operations != lowerable:
            raise RegistryError(
                f"family {self.family} advertises "
                f"{sorted(tag.name for tag in self.operations)} but its cache "
                f"schema lowers {sorted(tag.name for tag in lowerable)}"
            )


class TargetModelRegistry:
    """Explicit architecture-to-family resolution; unknown fails closed."""

    def __init__(self) -> None:
        self._by_architecture: dict[str, ModelRegistration] = {}

    def register(self, registration: ModelRegistration) -> None:
        if not registration.architectures:
            raise RegistryError(
                f"family {registration.family} declares no architectures"
            )
        for name in registration.architectures:
            if name in self._by_architecture:
                raise RegistryError(
                    f"architecture {name!r} is already registered"
                )
            self._by_architecture[name] = registration

    def resolve(self, architectures: tuple[str, ...]) -> ModelRegistration:
        matches = {
            self._by_architecture[name].family: self._by_architecture[name]
            for name in architectures
            if name in self._by_architecture
        }
        if not matches:
            known = ", ".join(sorted(self._by_architecture))
            raise RegistryError(
                f"no explicit registration for architectures {architectures!r}; "
                f"known: {known}"
            )
        if len(matches) > 1:
            raise RegistryError(
                f"architectures {architectures!r} match several families: "
                f"{sorted(matches)}"
            )
        return next(iter(matches.values()))


def target_registry() -> TargetModelRegistry:
    """The initial explicit registrations for the three target families."""

    registry = TargetModelRegistry()
    registry.register(
        ModelRegistration(
            family="qwen3",
            architectures=("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"),
            operations=frozenset({OperationTag.SEQUENCE_STEP}),
            cache_registration=qwen3_cache_registration,
        )
    )
    registry.register(
        ModelRegistration(
            family="bagel",
            architectures=("BagelForUnifiedGeneration", "BAGEL", "bagel"),
            operations=frozenset(
                {OperationTag.SEQUENCE_STEP, OperationTag.FLOW_STEP}
            ),
            cache_registration=bagel_cache_registration,
        )
    )
    registry.register(
        ModelRegistration(
            family="sensenova",
            architectures=("NEOChatModel", "neo_chat", "neo-unify", "neo_unify"),
            operations=frozenset(
                {OperationTag.SEQUENCE_STEP, OperationTag.FLOW_STEP}
            ),
            cache_registration=sensenova_cache_registration,
        )
    )
    return registry
