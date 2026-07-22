"""Shared plugin auto-discovery for package-scoped registries.

Worker subpackages that expose a registry of plugin classes (multimodal
processors) let each plugin register itself at import time. This module
implements the shared package scan: walk immediate submodules with
``pkgutil.iter_modules``, skip bookkeeping modules, and import the rest.

``loader`` and ``backends.attention`` do not use this helper; they register
plugins via eager imports in their package ``__init__`` so optional,
dependency-gated backends can be guarded with ``try``/``except`` individually.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Iterable

__all__ = [
    'DEFAULT_SKIP',
    'discover_package_plugins',
]

# Bookkeeping modules that live in every plugin package but never hold plugins.
DEFAULT_SKIP: frozenset[str] = frozenset({"base", "registry"})


def discover_package_plugins(
    package: str | ModuleType,
    *,
    skip: Iterable[str] = DEFAULT_SKIP,
) -> None:
    """Import every plugin module in ``package`` to trigger self-registration.

    ``package`` is the package whose immediate submodules are scanned (a module
    object or its dotted import path).  Submodules whose base name is in
    ``skip`` are ignored; the default skips the ``base`` and ``registry``
    bookkeeping modules common to every plugin package.

    A plugin module that fails to import propagates its error: a broken plugin
    is a startup defect, not a condition to serve around.

    Callers are expected to wrap this in ``functools.lru_cache(maxsize=1)`` so
    the scan runs at most once per process.
    """

    pkg = importlib.import_module(package) if isinstance(package, str) else package
    skip_set = set(skip)
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in skip_set:
            continue
        importlib.import_module(f"{pkg.__name__}.{mod_info.name}")
