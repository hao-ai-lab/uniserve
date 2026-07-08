"""Operator provider catalog and registration seam."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "OperatorProviderCatalog",
]


DispatcherFactory = Callable[[], Any]


@dataclass
class OperatorProviderCatalog:
    """Owns provider-pack registration and dispatcher lookup."""

    _factories: dict[str, DispatcherFactory] = field(default_factory=dict)

    def register_pack(self, operator: str, factory: DispatcherFactory) -> None:
        key = str(operator)
        if key in self._factories:
            raise ValueError(f"operator provider pack {key!r} is already registered")
        self._factories[key] = factory

    def dispatcher_for(self, operator: str) -> Any:
        try:
            return self._factories[str(operator)]()
        except KeyError as exc:
            raise ValueError(f"unknown operator provider pack {operator!r}") from exc

    def list_providers(self, operator: str) -> tuple[str, ...]:
        dispatcher = self.dispatcher_for(operator)
        names = getattr(dispatcher, "provider_names", None)
        if callable(names):
            return tuple(names())
        providers = getattr(dispatcher, "providers", ())
        return tuple(str(getattr(provider, "name", type(provider).__name__)) for provider in providers)
