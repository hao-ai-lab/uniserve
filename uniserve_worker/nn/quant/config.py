"""Immutable checkpoint quantization declarations and configuration normalization."""

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
    """Defines checkpoint linear precision, excluded layer prefixes, and KV storage dtype."""

    method: str = "unquantized"
    ignored_layers: tuple[str, ...] = ()
    kv_cache_dtype: str | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_model_config(cls, config: Any | None) -> "QuantizationConfig | None":
        """Validate and normalize quantization metadata from a model configuration."""

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
        """Construct the linear method selected for a parameter prefix."""

        if self._is_ignored(prefix):
            return UnquantizedLinearMethod()
        factory = _QUANT_METHOD_FACTORIES.get(self.method)
        if factory is None:
            raise NotImplementedError(f"quantization method {self.method!r} is not supported")
        return factory()

    def _is_ignored(self, prefix: str) -> bool:
        """Return whether a parameter prefix is excluded from checkpoint quantization."""

        if not prefix:
            return False
        return any(prefix == name or prefix.startswith(f"{name}.") for name in self.ignored_layers)


def _read_quant_section(config: Any | None) -> Mapping[str, Any]:
    """Normalize mapping, wrapper, and Hugging Face configuration surfaces.

    This boundary keeps quantization parsing independent of the model loader's
    concrete config type while preserving serialized keys without reinterpretation.
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
    """Extract nested quantization metadata from a normalized model mapping."""

    if config is None:
        return None
    top = _read_quant_section(config)
    if "quantization_config" in top:
        return top.get("quantization_config")
    return getattr(config, "quantization_config", None)


def _kv_cache_dtype(raw_map: Mapping[str, Any]) -> str | None:
    """Validate and return the optional serialized KV storage dtype."""

    value = raw_map.get("kv_cache_dtype")
    if value is None:
        return None
    name = str(value)
    if not name:
        raise ValueError("quantization_config.kv_cache_dtype must not be empty")
    # Resolve only to validate the serialized storage name; runtime compute
    # precision does not affect this configuration boundary.
    resolve_kv_store_dtype(torch.bfloat16, name)
    return name
