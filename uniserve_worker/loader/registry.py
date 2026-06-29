"""Model loader registry."""
from __future__ import annotations

from .base import BaseModelLoader

__all__ = [
    'register_loader',
    'get_loader',
]

_LOADERS: dict[str, BaseModelLoader] = {}


def register_loader(name: str, loader: BaseModelLoader) -> BaseModelLoader:
    key = name.lower()
    if key in _LOADERS:
        raise ValueError(f"model loader {name!r} already registered")
    _LOADERS[key] = loader
    return loader


def get_loader(name: str = "default") -> BaseModelLoader:
    try:
        return _LOADERS[name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(_LOADERS)) or "<none>"
        raise ValueError(f"unknown model loader {name!r}; known loaders: {known}") from exc
