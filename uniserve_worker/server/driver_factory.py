"""Worker-driver factory registry."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ..foundation.errors import invalid_descriptor
from .worker_kind import ENCODER, POSTPROCESS, SAMPLER

DriverFactory = Callable[[Any], Any]


class _DriverFactoryRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, DriverFactory] = {}
        self._lock = threading.Lock()
        self._frozen = False

    def register(self, kind: str, factory: DriverFactory) -> None:
        with self._lock:
            if self._frozen:
                raise invalid_descriptor("driver factory registry is frozen")
            if kind in self._factories:
                raise invalid_descriptor(f"driver factory already registered for {kind!r}")
            self._factories[kind] = factory

    def freeze(self) -> None:
        with self._lock:
            self._frozen = True

    def build(self, kind: str, args: Any) -> Any | None:
        with self._lock:
            factory = self._factories.get(kind)
        return None if factory is None else factory(args)


_DRIVER_FACTORY_REGISTRY = _DriverFactoryRegistry()

__all__ = ["build_registered_driver", "register_driver_factory"]


def register_driver_factory(kind: str, factory: DriverFactory) -> None:
    _DRIVER_FACTORY_REGISTRY.register(kind, factory)


def build_registered_driver(kind: str, args: Any) -> Any | None:
    return _DRIVER_FACTORY_REGISTRY.build(kind, args)


def _register_builtin_factories() -> None:
    # Deferred imports keep peeled drivers from importing optional model/torch
    # stacks until their worker kind is selected.
    from .encode_only_driver import build_encode_only_driver
    from .postprocess_driver import build_postprocess_driver
    from .sampler_driver import build_sampler_driver

    register_driver_factory(ENCODER, build_encode_only_driver)
    register_driver_factory(SAMPLER, build_sampler_driver)
    register_driver_factory(POSTPROCESS, build_postprocess_driver)
    _DRIVER_FACTORY_REGISTRY.freeze()


_register_builtin_factories()
