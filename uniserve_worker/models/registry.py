"""Auto-discovery registry for new-style UniModel classes."""
from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Type

from ..contracts.forward_mode import ForwardMode, mode_for_op
from ..contracts.model_protocols import UniModel
from ..foundation.errors import WorkerError, capability_mismatch, invalid_descriptor
from ..foundation.plugins import discover_package_plugins
from ..foundation.runtime_config import get_worker_config

__all__ = [
    'ModelRegistry',
    'MODEL_REGISTRY',
    'import_model_classes',
    'resolve_model_cls',
    'detect_model_architectures',
]

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Maps architecture names to :class:`UniModel` implementations."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[UniModel]] = {}
        self._fallback_cls: Type[UniModel] | None = None

    def register(self, model_cls: Type[UniModel], *, names: list[str] | tuple[str, ...] | None = None) -> None:
        _validate_model_contract(model_cls)
        keys = tuple(names or (model_cls.__name__,))
        if bool(getattr(model_cls, "fallback", False)):
            if self._fallback_cls is not None and self._fallback_cls is not model_cls:
                raise invalid_descriptor("only one fallback model class can be registered")
            self._fallback_cls = model_cls
        for key in keys:
            if key in self._classes:
                if self._classes[key] is model_cls:
                    continue
                raise invalid_descriptor(f"model architecture {key!r} already registered")
            self._classes[key] = model_cls

    def resolve(self, architectures: list[str] | tuple[str, ...]) -> Type[UniModel]:
        disabled = set(get_worker_config().disabled_model_archs)
        for arch in architectures:
            if arch in disabled:
                continue
            if arch in self._classes:
                return self._classes[arch]
        config = get_worker_config()
        if self._fallback_cls is not None and bool(config.allow_transformers_fallback):
            fallback_names = tuple(getattr(self._fallback_cls, "architectures", (self._fallback_cls.__name__,)))
            if any(name not in disabled for name in fallback_names):
                logger.warning(
                    "no UniModel registered for requested architectures; explicit fallback is enabled",
                    extra={
                        "architectures": list(architectures),
                        "fallback_model_class": getattr(self._fallback_cls, "__name__", repr(self._fallback_cls)),
                    },
                )
                return self._fallback_cls
        known = ", ".join(sorted(self._classes)) or "<none>"
        message = f"no UniModel registered for architectures {architectures!r}; known architectures: {known}"
        if self._fallback_cls is not None and not bool(config.allow_transformers_fallback):
            message += "; generic Transformers fallback is disabled by default, pass --allow-transformers-fallback to opt in"
        raise capability_mismatch(message)

    def registered_classes(self) -> tuple[Type[UniModel], ...]:
        """Distinct registered model classes (a name may map several aliases)."""
        return tuple(dict.fromkeys(self._classes.values()))


MODEL_REGISTRY = ModelRegistry()


def _register_module_models(module: ModuleType, *, strict: bool) -> None:
    entry = getattr(module, "EntryClass", None)
    if entry is None:
        return
    entries = entry if isinstance(entry, list) else [entry]
    for cls in entries:
        names = getattr(cls, "architectures", None)
        try:
            MODEL_REGISTRY.register(cls, names=names)
        except (WorkerError, ValueError):
            # Duplicate registration or contract mismatch: skip in non-strict mode.
            if strict:
                raise
            logger.error(
                "skipping model class that failed contract validation",
                extra={
                    "module_name": module.__name__,
                    "model_class": getattr(cls, "__name__", repr(cls)),
                },
                exc_info=True,
            )


@lru_cache(maxsize=1)
def import_model_classes(strict: bool | None = None) -> None:
    if strict is None:
        strict = get_worker_config().strict_model_imports
    discover_package_plugins(
        importlib.import_module(__package__ or "uniserve_worker.models"),
        strict=bool(strict),
        on_module=lambda module: _register_module_models(module, strict=bool(strict)),
    )


def resolve_model_cls(architectures: list[str] | tuple[str, ...]) -> Type[UniModel]:
    import_model_classes()
    return MODEL_REGISTRY.resolve(tuple(architectures))


def detect_model_architectures(model_path: str | Path) -> list[str]:
    """Ask registered model classes whether they recognize a checkpoint path."""

    import_model_classes()
    path = Path(model_path)
    detected: list[str] = []
    for cls in MODEL_REGISTRY.registered_classes():
        recognizes = getattr(cls, "recognizes", None)
        if callable(recognizes) and bool(recognizes(path)):
            detected.extend(str(name) for name in getattr(cls, "architectures", (cls.__name__,)))
    return detected


def _validate_model_contract(model_cls: Type[UniModel]) -> None:
    supported_ops = tuple(getattr(model_cls, "supported_ops", ()) or ())
    if not supported_ops:
        return
    if bool(getattr(model_cls, "whole_batch_forward", False)):
        if not callable(getattr(model_cls, "forward", None)):
            raise capability_mismatch(
                f"{model_cls.__name__} declares a whole-batch forward but has no forward()"
            )
        return
    for op in supported_ops:
        missing = _missing_capability(model_cls, str(op))
        if missing is not None:
            raise capability_mismatch(
                f"{model_cls.__name__} declares op {op!r} but does not implement {missing}"
            )


def _missing_capability(model_cls: Type[UniModel], op: str) -> str | None:
    try:
        mode = mode_for_op(op)
    except WorkerError as exc:
        raise capability_mismatch(
            f"{model_cls.__name__} declares unknown op {op!r}"
        ) from exc
    required: tuple[str, ...]
    if mode in {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY}:
        required = ("forward",)
    elif mode == ForwardMode.DENOISE:
        required = ("predict_velocity",)
    elif mode == ForwardMode.COMMIT:
        required = ("decode_image",)
    elif mode == ForwardMode.ENCODE and op == "vit_encode":
        required = ("encode_image",)
    elif mode == ForwardMode.ENCODE and op == "vae_encode":
        required = ("encode_latents",)
    else:
        required = ("forward",)
    for method in required:
        if callable(getattr(model_cls, method, None)):
            return None
    return " or ".join(f"{method}()" for method in required)
