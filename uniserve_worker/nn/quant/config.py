"""Immutable checkpoint quantization declarations."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from .base import QuantizeMethodBase, UnquantizedLinearMethod
from .fp8 import W8A8Fp8LinearMethod
from .kv_cache import resolve_kv_store_dtype

__all__ = [
    'QuantizationConfig',
]

_QUANT_METHOD_FACTORIES: dict[str, Callable[[], QuantizeMethodBase]] = {
    "unquantized": UnquantizedLinearMethod,
    "fp8": W8A8Fp8LinearMethod,
}


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
        if isinstance(raw, Mapping):
            raw_map = raw
        else:
            raise TypeError(
                "quantization_config must be a mapping, "
                f"got {type(raw).__name__}"
            )
        method = str(raw_map.get("quant_method", "unquantized"))
        ignored = raw_map.get("ignored_layers", ())
        if isinstance(ignored, (str, bytes)) or not isinstance(ignored, (list, tuple)):
            raise TypeError("quantization_config.ignored_layers must be a list")
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
            raise NotImplementedError(f"quantization method {self.method!r} is not supported")
        return factory()

    def _is_ignored(self, prefix: str) -> bool:
        if not prefix:
            return False
        return any(prefix == name or prefix.startswith(f"{name}.") for name in self.ignored_layers)


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
    if hasattr(config, "kv_cache_dtype"):
        values["kv_cache_dtype"] = getattr(config, "kv_cache_dtype")
    return values


def _extract_quantization_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    top = _read_quant_section(config)
    if "quantization_config" in top:
        return top.get("quantization_config")
    return getattr(config, "quantization_config", None)


def _kv_cache_dtype(raw_map: Mapping[str, Any]) -> str | None:
    value = raw_map.get("kv_cache_dtype")
    if value is None:
        return None
    name = str(value)
    if not name:
        raise ValueError("quantization_config.kv_cache_dtype must not be empty")
    # This single extraction point validates the name: ``resolve_kv_store_dtype``
    # raises ValueError for store dtypes it cannot map, surfacing bad config at
    # parse time.  ``compute_dtype`` is irrelevant here because only the string
    # name is being validated, not resolved to a runtime dtype.
    resolve_kv_store_dtype(torch.bfloat16, name)
    return name
