"""The single startup loader for every catalog checkpoint format."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ..foundation.errors import capability_mismatch
from ..foundation.runtime_config import ExecutionConfig
from ..nn.layer import LayerSpec
from ..nn.mesh import TensorParallelSpec
from ..nn.quant import QuantizationConfig
from ..nn.quant.base import process_quantized_modules
from ..nn.quant.load_state import is_optional_checkpoint
from ..spec import (
    Sidecar,
    WeightSpec,
    WeightTarget,
)
from .paths import read_config
from .transformers import dtype_from_name, load_native_transformers_checkpoint
from .weight_utils import (
    apply_weight_ties,
    iter_weights,
    load_declared_weights,
    resolve_weight_files,
    tensor_shape,
)

logger = logging.getLogger(__name__)


def _declare_targets(spec: WeightSpec, model: nn.Module) -> WeightSpec:
    targets = tuple(
        WeightTarget(
            name=name,
            shape=tuple(int(dimension) for dimension in parameter.shape),
            dtype=str(parameter.dtype).removeprefix("torch."),
        )
        for name, parameter in model.named_parameters()
    )
    return replace(spec, targets=targets)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: nn.Module
    tokenizer: Any | None
    device: str


class Loader:
    """Construct, validate, and inject one immutable ready model."""

    def load(
        self,
        entry: Any,
        config: Any,
        *,
        model_path: str,
        device: str,
        attention_backend: str | None,
        model_scope: str,
        execution: ExecutionConfig,
        parallel: TensorParallelSpec,
    ) -> LoadedModel:
        kind = str(entry.checkpoint)
        if kind == "stream":
            loaded = self._stream(
                entry,
                config,
                model_path=model_path,
                device=device,
                parallel=parallel,
            )
        elif kind == "composite":
            loaded = self._composite(
                entry,
                model_path=model_path,
                device=device,
                parallel=parallel,
            )
        elif kind == "native":
            loaded = self._native(
                entry,
                model_path=model_path,
                device=device,
                attention_backend=attention_backend,
                model_scope=model_scope,
                execution=execution,
                parallel=parallel,
            )
        else:
            raise capability_mismatch(f"unknown catalog checkpoint format {kind!r}")
        spec = getattr(loaded.model, "spec", None)
        if spec is None or not isinstance(getattr(spec, "weights", None), WeightSpec):
            raise capability_mismatch("a loaded model must declare a WeightSpec before ready")
        loaded.model.spec = replace(spec, weights=_declare_targets(spec.weights, loaded.model))
        return loaded

    def _stream(
        self,
        entry: Any,
        config: Any,
        *,
        model_path: str,
        device: str,
        parallel: TensorParallelSpec,
    ) -> LoadedModel:
        quantization = QuantizationConfig.from_model_config(config)
        model = cast(Any, entry.model_class)(
            config=config,
            layer_spec=LayerSpec(parallel=parallel, quantization=quantization),
        )
        spec = _weight_spec(model)
        loaded, ignored = load_declared_weights(
            model,
            iter_weights(resolve_weight_files(model_path)),
            spec=spec,
        )
        apply_weight_ties(model, spec)
        _require_loaded_parameters(model, loaded)
        process_quantized_modules(model.modules())
        _prepare_serving_dtype(model)
        model.to(device)
        model.eval()
        if ignored:
            logger.warning("ignored %d checkpoint tensors during model load", len(ignored))
        return LoadedModel(model, None, _model_device(model, device))

    def _composite(
        self,
        entry: Any,
        *,
        model_path: str,
        device: str,
        parallel: TensorParallelSpec,
    ) -> LoadedModel:
        config_class = entry.config_class
        graph_class = entry.graph_class
        if config_class is None or graph_class is None:
            raise capability_mismatch("composite catalog entry is incomplete")
        from_mapping = getattr(config_class, "from_mapping", None)
        if not callable(from_mapping):
            raise capability_mismatch("composite config classes must implement from_mapping")
        config = from_mapping(_declared_config(entry, model_path))
        spec = _class_weight_spec(entry.model_class)
        config = _resolve_config_dimensions(
            config,
            _root_weight_file(model_path, spec.files),
            tuple(entry.config_dimensions),
        )
        layer_spec = LayerSpec(
            parallel=parallel,
            quantization=QuantizationConfig.from_model_config(config),
        )
        graph = graph_class(config, layer_spec=layer_spec).eval()
        serving_dtype = dtype_from_name(entry.serving_dtype)
        loaded, ignored = load_declared_weights(
            graph,
            iter_weights([_root_weight_file(model_path, spec.files)]),
            spec=spec,
            dtype=serving_dtype,
        )
        apply_weight_ties(graph, spec)
        sidecar_prefixes = tuple(f"{sidecar.module}." for sidecar in spec.sidecars)
        expected = {
            name
            for name, _ in graph.named_parameters()
            if not (sidecar_prefixes and name.startswith(sidecar_prefixes))
        }
        missing = sorted(expected - loaded)
        if missing:
            raise capability_mismatch(
                f"composite checkpoint load mismatch: missing={missing[:8]} ({len(missing)}) ignored={ignored[:8]}"
            )
        for sidecar in spec.sidecars:
            _load_sidecar(graph, model_path, sidecar)
        graph.to(device=device, dtype=serving_dtype)
        process_quantized_modules(graph.modules())
        model = entry.model_class(config, layer_spec=layer_spec, graph=graph)
        model.eval()
        return LoadedModel(model, None, device)

    def _native(
        self,
        entry: Any,
        *,
        model_path: str,
        device: str,
        attention_backend: str | None,
        model_scope: str,
        execution: ExecutionConfig,
        parallel: TensorParallelSpec,
    ) -> LoadedModel:
        config_class = entry.config_class
        if config_class is None:
            raise capability_mismatch("native catalog entry is incomplete")
        spec = _class_weight_spec(entry.model_class)
        role = None if model_scope == "whole" else model_scope
        model, tokenizer, real_device = load_native_transformers_checkpoint(
            model_path,
            device,
            config_cls=config_class,
            model_cls=entry.model_class,
            attention_backend=attention_backend,
            min_version_key=entry.minimum_code_version_key,
            code_version=entry.code_version,
            weight_spec=spec,
            model_scope=role,
            execution=execution,
            parallel=parallel,
        )
        return LoadedModel(model, tokenizer, real_device)


def _class_weight_spec(model_class: type[nn.Module]) -> WeightSpec:
    value = getattr(model_class, "weight_spec", None)
    if not isinstance(value, WeightSpec):
        raise capability_mismatch(f"{model_class.__name__} must declare a WeightSpec")
    return value


def _declared_config(entry: Any, model_path: str) -> dict[str, Any]:
    raw = dict(read_config(model_path))
    root = Path(model_path)
    for field, filename in tuple(entry.config_files):
        if field in raw:
            continue
        path = root / filename
        if not path.is_file():
            raise capability_mismatch(
                f"checkpoint config is missing declared file {filename!r} for {field!r}"
            )
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise capability_mismatch(f"cannot read checkpoint config {filename!r}: {exc}") from exc
        if not isinstance(value, dict):
            raise capability_mismatch(f"checkpoint config {filename!r} must contain an object")
        raw[field] = value
    return raw


def _weight_spec(model: nn.Module) -> WeightSpec:
    spec = getattr(getattr(model, "spec", None), "weights", None)
    if not isinstance(spec, WeightSpec):
        raise capability_mismatch(f"{type(model).__name__} must declare a WeightSpec")
    return spec


def _require_loaded_parameters(model: nn.Module, loaded: set[str]) -> None:
    missing = sorted(
        name
        for name, parameter in model.named_parameters()
        if name not in loaded and not is_optional_checkpoint(parameter)
    )
    if missing:
        raise capability_mismatch(
            f"checkpoint load mismatch: missing={len(missing)} {missing[:20]!r}"
        )


def _prepare_serving_dtype(model: nn.Module) -> None:
    routes = tuple(getattr(getattr(model, "spec", None), "routes", ()))
    names = {str(route.dtype).removeprefix("torch.") for route in routes}
    if len(names) != 1:
        return
    dtype = getattr(torch, names.pop(), None)
    if not isinstance(dtype, torch.dtype):
        return
    quantized = any(
        bool(getattr(getattr(module, "quant_method", None), "is_quantized", False))
        for module in model.modules()
    )
    if not quantized:
        model.to(dtype=dtype)
        return
    for parameter in model.parameters():
        if parameter.dtype == torch.float32 and not bool(
            getattr(parameter, "_uniserve_skip_serving_cast", False)
        ):
            parameter.data = parameter.data.to(dtype=dtype)


def _root_weight_file(model_path: str, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = Path(model_path) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no root weight file among {candidates!r} under {model_path}")


def _resolve_config_dimensions(config: Any, weights: Path, bindings: tuple[Any, ...]) -> Any:
    if not bindings:
        return config
    updates: dict[str, int] = {}
    for binding in bindings:
        shape = tensor_shape(weights, str(binding.tensor))
        axis = int(binding.axis)
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            raise capability_mismatch(
                f"checkpoint tensor {binding.tensor!r} has no declared axis {binding.axis}"
            )
        value = int(shape[axis])
        if str(binding.transform) == "square_root":
            root = math.isqrt(value)
            if root * root != value:
                raise capability_mismatch(
                    f"checkpoint tensor {binding.tensor!r} dimension {value} is not square"
                )
            value = root
        elif str(binding.transform) != "identity":
            raise capability_mismatch(
                f"unknown configuration dimension transform {binding.transform!r}"
            )
        updates[str(binding.field)] = value
    return replace(config, **updates)


def _load_sidecar(graph: nn.Module, model_path: str, sidecar: Sidecar) -> None:
    state = dict(iter_weights([Path(model_path) / sidecar.file]))
    module = graph.get_submodule(sidecar.module)
    missing, unexpected = module.load_state_dict(state, strict=False)
    required_missing = [
        name
        for name in missing
        if not any(substring in name for substring in sidecar.optional_substrings)
    ]
    if required_missing or unexpected:
        raise capability_mismatch(
            f"sidecar {sidecar.file!r} load mismatch: missing={required_missing[:8]} unexpected={unexpected[:8]}"
        )


def _model_device(model: nn.Module, fallback: str) -> str:
    return next((str(parameter.device) for parameter in model.parameters()), fallback)


__all__ = ["LoadedModel", "Loader"]
