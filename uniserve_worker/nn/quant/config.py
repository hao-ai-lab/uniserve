"""Immutable checkpoint quantization declarations and configuration normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, TypeAlias

import torch

from .base import LinearMethod, UnquantizedLinearMethod
from .fp8 import DynamicW8A8Fp8LinearMethod, W8A8Fp8LinearMethod
from .kv_cache import resolve_kv_store_dtype
from .mxfp8 import DynamicW8A8MxFp8LinearMethod
from .nvfp4 import DynamicW4A4NvFp4LinearMethod

__all__ = [
    "QuantizationConfig",
    "create_linear_method",
]

LinearPrecision: TypeAlias = Literal["fp32", "fp16", "bf16", "fp8", "mxfp8", "nvfp4"]


def resolve_component_precisions(
    value: Mapping[str, object],
    *,
    supported: Mapping[str, tuple[LinearPrecision, ...]],
    presets: Mapping[str, Mapping[str, LinearPrecision]],
    shorthands: Mapping[str, Mapping[str, LinearPrecision]],
    default_mode: str,
) -> Mapping[str, LinearPrecision]:
    """Resolve a model's declared formats through the shared quantization boundary.

    Models supply supported formats and fixed preset values; this function owns
    shorthand expansion, component overrides and validation. The resulting
    immutable mapping is the sole precision selection passed to model loading.
    """

    unknown = set(value) - {"mode", "quant_method", "components"}
    if unknown:
        raise ValueError(f"quantization_config has unknown fields {sorted(unknown)!r}")
    if "mode" in value and "quant_method" in value:
        raise ValueError("quantization_config cannot specify both mode and quant_method")
    if value.get("quant_method") is not None:
        field = "quant_method"
        selected = value[field]
        choices = shorthands
    else:
        field = "mode"
        selected = value.get(field)
        if selected is None:
            selected = default_mode
        choices = presets
    if not isinstance(selected, str) or selected not in choices:
        raise ValueError(f"quantization_config.{field} must be one of {tuple(choices)!r}")
    resolved = dict(choices[selected])
    overrides = value.get("components", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("quantization_config.components must be an object")
    unknown_components = set(overrides) - set(supported)
    if unknown_components:
        raise ValueError(
            f"quantization_config.components has unknown entries {sorted(unknown_components)!r}"
        )
    for name, precision in overrides.items():
        if precision is not None:
            resolved[name] = precision
    if set(resolved) != set(supported):
        raise ValueError("precision preset must define every supported component")
    for name, precision in resolved.items():
        if precision not in supported[name]:
            raise ValueError(
                f"quantization_config.components.{name} must be one of {supported[name]!r}"
            )
    return MappingProxyType(resolved)


_QUANT_METHOD_FACTORIES: dict[str, Callable[[], LinearMethod]] = {
    "unquantized": UnquantizedLinearMethod,
    "fp8": W8A8Fp8LinearMethod,
    "mxfp8": DynamicW8A8MxFp8LinearMethod,
    "nvfp4": DynamicW4A4NvFp4LinearMethod,
}


def create_linear_method(precision: str, *, tensorwise: bool = False) -> LinearMethod:
    """Select a shared projection backend and its logical activation scale domain.

    FP8 supports row or tensor scales. MXFP8 scales independent 32-column
    blocks; NVFP4 combines a tensor scale with block scales. Dense formats
    retain the construction dtype. Checkpoint-scale FP8 uses the row domain.
    """

    if precision in {"fp32", "fp16", "bf16"}:
        precision = "unquantized"
    if tensorwise and precision == "fp8":
        return DynamicW8A8Fp8LinearMethod(tensorwise=True)
    if tensorwise and precision == "mxfp8":
        raise ValueError("MXFP8 requires block-local activation scales")
    try:
        factory = _QUANT_METHOD_FACTORIES[precision]
    except KeyError as error:
        raise ValueError(f"unsupported linear precision {precision!r}") from error
    return factory()


@dataclass(frozen=True)
class QuantizationConfig:
    """Defines checkpoint linear precision, excluded layer prefixes, and KV storage dtype."""

    method: str = "unquantized"
    ignored_layers: tuple[str, ...] = ()
    kv_cache_dtype: str | None = None
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_model_config(
        cls, config: Any | None, *, overrides: Mapping[str, object] | None = None
    ) -> "QuantizationConfig | None":
        """Validate and normalize quantization metadata from a model configuration."""

        raw = _extract_quantization_config(config)
        if raw is None and not overrides:
            return None
        if raw is None:
            raw_map: dict[str, Any] = {}
        elif isinstance(raw, Mapping):
            raw_map = dict(raw)
        else:
            raise TypeError(f"quantization_config must be a mapping, got {type(raw).__name__}")
        if overrides:
            unknown = set(overrides) - {"quant_method", "ignored_layers", "kv_cache_dtype"}
            if unknown:
                raise ValueError(f"linear quantization has unknown fields {sorted(unknown)!r}")
            checkpoint_method = str(raw_map.get("quant_method", "unquantized"))
            requested_method = str(overrides.get("quant_method", checkpoint_method))
            if checkpoint_method != "unquantized" and checkpoint_method != requested_method:
                raise ValueError(
                    f"checkpoint declares {checkpoint_method!r}; loading it as "
                    f"{requested_method!r} requires checkpoint format conversion"
                )
            raw_map.update(overrides)
        method = str(raw_map.get("quant_method", "unquantized"))
        ignored = raw_map.get("ignored_layers", ())
        if isinstance(ignored, (str, bytes)) or not isinstance(ignored, (list, tuple)):
            raise TypeError("quantization_config.ignored_layers must be a list")
        ignored_layers = tuple(str(item) for item in ignored)
        if method not in _QUANT_METHOD_FACTORIES:
            raise NotImplementedError(
                f"checkpoint quantization method {method!r} is not supported yet; "
                f"supported methods are {tuple(_QUANT_METHOD_FACTORIES)!r}"
            )
        kv_cache_dtype = _kv_cache_dtype(raw_map)
        return cls(
            method=method,
            ignored_layers=ignored_layers,
            kv_cache_dtype=kv_cache_dtype,
            raw=MappingProxyType(dict(raw_map)),
        )

    def get_quant_method(
        self, prefix: str = "", *, packed_prefixes: tuple[str, ...] = ()
    ) -> LinearMethod:
        """Construct the linear method selected for a parameter prefix."""

        if self.method == "unquantized" or self._is_ignored(prefix):
            return UnquantizedLinearMethod()
        if packed_prefixes:
            ignored = tuple(self._is_ignored(name) for name in packed_prefixes)
            if any(ignored) != all(ignored):
                raise ValueError(
                    f"packed projection {prefix!r} requires the same precision for all parts"
                )
            if all(ignored):
                return UnquantizedLinearMethod()
        return create_linear_method(self.method)

    def validate_device(self, device: str | torch.device, dtype: torch.dtype) -> None:
        """Reject unsupported compute formats before allocating checkpoint storage."""

        if self.method == "unquantized":
            return
        target = torch.device(device)
        if self.method in {"mxfp8", "nvfp4"}:
            if dtype != torch.bfloat16:
                raise ValueError(f"{self.method.upper()} requires bfloat16 activations")
            if target.type != "cuda" or torch.cuda.get_device_capability(target) < (10, 0):
                raise ValueError(f"{self.method.upper()} requires an SM100-class CUDA device")
        elif self.method == "fp8":
            if dtype not in {torch.float16, torch.bfloat16}:
                raise ValueError("W8A8 FP8 requires float16 or bfloat16 activations")
            if target.type == "cuda" and torch.cuda.get_device_capability(target) < (8, 9):
                raise ValueError("W8A8 FP8 requires CUDA compute capability 8.9 or newer")
        else:
            raise ValueError(f"unsupported linear precision {self.method!r}")

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
