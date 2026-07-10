"""Scoped quantization configuration for shared layer construction."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping

import torch

from .base import QuantizeMethodBase, UnquantizedLinearMethod
from .fp8 import W8A8Fp8LinearMethod
from .kv_cache import KV_CACHE_NO_OVERRIDE_SENTINELS, resolve_kv_store_dtype

__all__ = [
    'QuantizationConfig',
    'get_current_quantization_config',
    'is_quant_context_active',
    'warn_if_no_quant_context',
    'get_current_kv_cache_dtype',
    'kv_cache_dtype_from_model_config',
    'use_quantization_config',
]

logger = logging.getLogger(__name__)

_KV_CACHE_DTYPE_KEYS = ("kv_cache_dtype", "kv_dtype", "cache_dtype")


@dataclass(frozen=True)
class _QuantMethodSpec:
    """Single source of truth for one canonical quantization method.

    Owns the method's accepted aliases and its ``QuantizeMethodBase`` factory, so
    the alias normalizer, the supported-method check, and the per-layer dispatch
    all resolve through one table.
    """

    canonical: str
    aliases: tuple[str, ...]
    factory: Callable[[], QuantizeMethodBase]


_QUANT_METHOD_SPECS: tuple[_QuantMethodSpec, ...] = (
    _QuantMethodSpec(
        canonical="unquantized",
        aliases=("", "no_quant", "none", "unquantized", "unquant"),
        factory=UnquantizedLinearMethod,
    ),
    _QuantMethodSpec(
        canonical="fp8",
        aliases=("fp8",),
        factory=W8A8Fp8LinearMethod,
    ),
    _QuantMethodSpec(
        canonical="w8a8_fp8",
        aliases=("w8a8", "w8a8_fp8", "w8a8fp8"),
        factory=W8A8Fp8LinearMethod,
    ),
)

# name -> canonical method, derived from each spec's aliases (canonical included).
_QUANT_METHOD_ALIASES: dict[str, str] = {
    alias: spec.canonical
    for spec in _QUANT_METHOD_SPECS
    for alias in (*spec.aliases, spec.canonical)
}

# canonical method -> factory.
_QUANT_METHOD_FACTORIES: dict[str, Callable[[], QuantizeMethodBase]] = {
    spec.canonical: spec.factory for spec in _QUANT_METHOD_SPECS
}


# Sentinel distinguishing "no quantization context has ever been entered" from
# "a context is active and resolves to None (unquantized)". A model entry or
# loader always enters a context (``use_quantization_config``), so the sentinel
# is what a layer sees only when it is constructed entirely outside any context.
_NO_CONTEXT = object()

_CURRENT_QUANT_CONFIG: ContextVar[Any] = ContextVar(
    "uniserve_quant_config",
    default=_NO_CONTEXT,
)

_warned_no_quant_context = False


@dataclass(frozen=True)
class QuantizationConfig:
    """Checkpoint-level quantization policy.

    This is intentionally conservative: unsupported checkpoint quantization
    fails during construction instead of silently running with the wrong layout.
    """

    method: str = "unquantized"
    ignored_layers: tuple[str, ...] = ()
    kv_cache_dtype: str | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_model_config(cls, config: Any | None) -> "QuantizationConfig | None":
        raw = _extract_quantization_config(config)
        if raw is None:
            return None
        if isinstance(raw, str):
            raw_map: Mapping[str, Any] = {"quant_method": raw}
        elif isinstance(raw, Mapping):
            raw_map = raw
        else:
            raise TypeError(
                "quantization_config must be a mapping or string, "
                f"got {type(raw).__name__}"
            )
        method = _normalize_method(
            raw_map.get("quant_method")
            or raw_map.get("quantization_method")
            or raw_map.get("method")
            or raw_map.get("type")
            or raw_map.get("name")
            or "unquantized"
        )
        ignored = raw_map.get("ignored_layers") or raw_map.get("modules_to_not_convert") or ()
        ignored_layers: tuple[str, ...]
        if isinstance(ignored, str):
            ignored_layers = (ignored,)
        else:
            ignored_layers = tuple(str(item) for item in ignored)
        if method not in _QUANT_METHOD_FACTORIES:
            raise NotImplementedError(
                f"checkpoint quantization method {method!r} is not supported yet; "
                "UniServe currently supports unquantized and W8A8 FP8 linear "
                "quantization"
            )
        kv_cache_dtype = _kv_cache_dtype(raw_map)
        return cls(
            method=method,
            ignored_layers=ignored_layers,
            kv_cache_dtype=kv_cache_dtype,
            raw=MappingProxyType(dict(raw_map)),
        )

    def get_quant_method(self, prefix: str = "") -> QuantizeMethodBase:
        if self._is_ignored(prefix):
            return UnquantizedLinearMethod()
        factory = _QUANT_METHOD_FACTORIES.get(self.method)
        if factory is None:
            return UnquantizedLinearMethod()
        return factory()

    def _is_ignored(self, prefix: str) -> bool:
        if not prefix:
            return False
        return any(prefix == name or prefix.startswith(f"{name}.") for name in self.ignored_layers)


def get_current_quantization_config() -> QuantizationConfig | None:
    current = _CURRENT_QUANT_CONFIG.get()
    return None if current is _NO_CONTEXT else current


def is_quant_context_active() -> bool:
    """Whether a quantization context has been entered in this execution scope.

    ``True`` even when the active config is ``None`` (an explicit unquantized
    context); ``False`` only when no ``use_quantization_config`` scope is open,
    which is the silent-unquantized case worth surfacing.
    """
    return _CURRENT_QUANT_CONFIG.get() is not _NO_CONTEXT


def warn_if_no_quant_context() -> None:
    """Emit a once-per-process warning when a layer is built with no active
    quantization context, so silently-unquantized construction is observable."""
    global _warned_no_quant_context
    if is_quant_context_active() or _warned_no_quant_context:
        return
    _warned_no_quant_context = True
    logger.warning(
        "building a quantizable layer with no active quantization context; "
        "defaulting to unquantized. A checkpoint with quantized weights built "
        "outside use_quantization_config(...) would be served unquantized. "
        "Loaders and model entries enter this context automatically; this "
        "warning indicates a layer constructed outside that path."
    )


def get_current_kv_cache_dtype(config: Any | None = None) -> str | None:
    current = get_current_quantization_config()
    if current is not None and current.kv_cache_dtype is not None:
        return current.kv_cache_dtype
    return kv_cache_dtype_from_model_config(config)


def kv_cache_dtype_from_model_config(config: Any | None) -> str | None:
    top = _kv_cache_dtype(_read_quant_section(config))
    if top is not None:
        return top
    raw = _extract_quantization_config(config)
    if isinstance(raw, Mapping):
        return _kv_cache_dtype(raw)
    return None


@contextmanager
def use_quantization_config(config: QuantizationConfig | None) -> Iterator[None]:
    token = _CURRENT_QUANT_CONFIG.set(config)
    try:
        yield
    finally:
        _CURRENT_QUANT_CONFIG.reset(token)


def _read_quant_section(config: Any | None) -> Mapping[str, Any]:
    """Normalize any config object into its top-level mapping.

    This is the single boundary that turns a Hugging Face config object (or a
    raw dict) into a ``Mapping`` so the rest of the module reads keys instead of
    re-probing ``.raw``/``.to_dict()``/attribute shapes.
    """

    if config is None:
        return {}
    if isinstance(config, Mapping):
        return config
    raw = getattr(config, "raw", None)
    if isinstance(raw, Mapping):
        return raw
    if hasattr(config, "to_dict"):
        maybe = config.to_dict()
        if isinstance(maybe, Mapping):
            return maybe
    values: dict[str, Any] = {}
    for key in _KV_CACHE_DTYPE_KEYS:
        if hasattr(config, key):
            values[key] = getattr(config, key)
    return values


def _extract_quantization_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    top = _read_quant_section(config)
    if "quantization_config" in top:
        return top.get("quantization_config")
    return getattr(config, "quantization_config", None)


def _kv_cache_dtype(raw_map: Mapping[str, Any]) -> str | None:
    value: Any = None
    for key in _KV_CACHE_DTYPE_KEYS:
        value = value or raw_map.get(key)
    if value is None:
        return None
    name = str(value).strip()
    if not name or name.lower() in KV_CACHE_NO_OVERRIDE_SENTINELS | {"null"}:
        return None
    # This single extraction point validates the name: ``resolve_kv_store_dtype``
    # raises ValueError for store dtypes it cannot map, surfacing bad config at
    # parse time.  ``compute_dtype`` is irrelevant here because only the string
    # name is being validated, not resolved to a runtime dtype.
    resolve_kv_store_dtype(torch.bfloat16, name)
    return name


def _normalize_method(value: object) -> str:
    method = str(value).strip().lower().replace("-", "_")
    return _QUANT_METHOD_ALIASES.get(method, method)
