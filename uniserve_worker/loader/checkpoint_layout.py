"""Checkpoint tensor naming and loading layout adapters."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .weight_utils import StackedParamMapping

__all__ = [
    "CheckpointLayout",
]


@dataclass(frozen=True)
class CheckpointLayout:
    """Family-owned checkpoint mapping and tensor policy."""

    name_map: Callable[[str], str] | None = None
    stacked: tuple[StackedParamMapping | tuple[str, str, str | int], ...] = ()
    tower_predicate: Callable[[str, str | None], bool] | None = None
    optional_names: frozenset[str] = field(default_factory=frozenset)
    ignored_names: frozenset[str] = field(default_factory=frozenset)

    def map_name(self, name: str) -> str:
        return self.name_map(name) if self.name_map is not None else name

    def stacked_params(self) -> tuple[StackedParamMapping | tuple[str, str, str | int], ...]:
        return self.stacked

    def tower_filter(self, tower_role: str | None = None) -> Callable[[str], bool] | None:
        tower_predicate = self.tower_predicate
        if tower_predicate is None:
            return None

        def predicate(name: str) -> bool:
            return bool(tower_predicate(name, tower_role))

        return predicate

    def optional_tensor(self, name: str) -> bool:
        return name in self.optional_names

    def ignored_tensor(self, name: str) -> bool:
        return name in self.ignored_names
