"""Shared plugin auto-discovery for package-scoped registries.

Several worker subpackages expose a registry of plugin classes (model
architectures, multimodal processors, ...) that register themselves on import.
This module implements the shared package scan: walk immediate submodules with
``pkgutil.iter_modules``, skip bookkeeping modules, import the rest, and invoke
an optional ``on_module`` callback for registry-specific work.

``loader`` and ``backends.attention`` do not use this helper; they register
plugins via eager imports in their package ``__init__`` so optional,
dependency-gated backends can be guarded with ``try``/``except`` individually.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from types import ModuleType
from typing import Callable, Iterable

__all__ = [
    'DEFAULT_SKIP',
    'discover_package_plugins',
]

logger = logging.getLogger(__name__)

# Bookkeeping modules that live in every plugin package but never hold plugins.
DEFAULT_SKIP: frozenset[str] = frozenset({"base", "registry"})


def discover_package_plugins(
    package: str | ModuleType,
    *,
    skip: Iterable[str] = DEFAULT_SKIP,
    strict: bool = False,
    on_module: Callable[[ModuleType], None] | None = None,
) -> None:
    """Import every plugin module in ``package`` to trigger self-registration.

    ``package`` is the package whose immediate submodules are scanned (a module
    object or its dotted import path).  Submodules whose base name is in
    ``skip`` are ignored; the default skips the ``base`` and ``registry``
    bookkeeping modules common to every plugin package.

    For each imported module ``on_module`` (when given) is invoked so callers
    can perform plugin-specific post-import work (e.g. pulling an
    ``EntryClass`` and registering it).

    When ``strict`` is false (the default) a module that fails to import -- or
    whose ``on_module`` callback raises -- is logged and skipped so one broken
    plugin cannot take down the whole registry.  When ``strict`` is true the
    failure propagates.

    Callers are expected to wrap this in ``functools.lru_cache(maxsize=1)`` so
    the scan runs at most once per process.
    """

    pkg = importlib.import_module(package) if isinstance(package, str) else package
    skip_set = set(skip)
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in skip_set:
            continue
        module_name = f"{pkg.__name__}.{mod_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            if strict:
                raise
            logger.warning(
                "skipping plugin module that failed to import",
                extra={"module_name": module_name},
                exc_info=True,
            )
            continue
        if on_module is None:
            continue
        try:
            on_module(module)
        except Exception:
            if strict:
                raise
            logger.warning(
                "skipping plugin module whose registration callback failed",
                extra={"module_name": module_name},
                exc_info=True,
            )
