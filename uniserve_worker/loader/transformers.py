"""Hugging Face native checkpoint loader: meta-init plus per-tensor streaming."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from transformers import AutoTokenizer

from ..foundation.env import DEFAULT_ATTENTION_BACKEND
from ..foundation.runtime_config import ExecutionConfig
from ..nn.layer import LayerSpec
from ..nn.mesh import TensorParallelSpec
from ..nn.quant import QuantizationConfig
from ..nn.quant.base import process_quantized_modules
from ..nn.quant.load_state import (
    capture_tensor_policy,
    is_optional_checkpoint,
    restore_tensor_policy,
    skip_serving_cast,
)
from .paths import read_config
from .schema import Tie, TowerSplit, WeightSpec
from .weight_utils import (
    apply_weight_ties,
    iter_weights,
    load_parameter,
    map_weight_name,
    maybe_remap_kv_scale_name,
    resolve_weight_files,
    transform_weight,
)

__all__ = [
    "dtype_from_name",
    "infer_input_device",
    "load_native_transformers_checkpoint",
]


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown transformer dtype {name!r}; expected one of bfloat16, float16, or float32"
        ) from exc


def _load_tokenizer(
    model_dir: str,
) -> Any:
    """Load a tokenizer and attach ``model_dir`` context to failures."""

    try:
        return AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=False,
            trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - error-context wrapper.
        raise RuntimeError(
            f"failed to load the configured SenseNova tokenizer from {model_dir!r}: {exc}"
        ) from exc


def infer_input_device(
    model: nn.Module, fallback: str | torch.device | None = None
) -> torch.device:
    for param in model.parameters():
        if param.device.type not in {"cpu", "meta"}:
            return param.device
    if fallback is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(fallback) if isinstance(fallback, str) else fallback


def _check_min_code_version(config: Any, key: str | None, code_version: str | None) -> None:
    """Reject a checkpoint that declares a newer minimum model-code version."""
    if key is None or code_version is None:
        return
    try:
        from packaging.version import Version
    except ImportError:  # pragma: no cover
        return
    cfg = config.to_dict() if hasattr(config, "to_dict") else config
    if not isinstance(cfg, dict):
        return
    required = cfg.get(key)
    if required and Version(code_version) < Version(str(required)):
        raise RuntimeError(f"checkpoint requires UniServe model code >= {required}")


def load_native_transformers_checkpoint(
    model_dir: str,
    device: str,
    *,
    config_cls: Any,
    model_cls: Any,
    attention_backend: str | None = None,
    min_version_key: str | None = None,
    code_version: str | None = None,
    weight_spec: WeightSpec,
    model_scope: str | None,
    execution: ExecutionConfig,
    parallel: TensorParallelSpec,
) -> tuple[nn.Module, Any, str]:
    """Instantiate a native ``nn.Module`` and materialize its declared checkpoint."""

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific.
        raise RuntimeError(
            "native checkpoint loading requires accelerate; install it in the worker environment"
        ) from exc

    attn_backend = attention_backend or DEFAULT_ATTENTION_BACKEND
    dtype = dtype_from_name(execution.model_dtype)

    from_dict = getattr(config_cls, "from_dict", None)
    if not callable(from_dict):
        raise TypeError("native checkpoint config classes must implement from_dict")
    config = from_dict(read_config(model_dir))
    config.uniserve_attention_backend = attn_backend
    _check_min_code_version(config, min_version_key, code_version)

    tokenizer = _load_tokenizer(model_dir)
    quant_config = QuantizationConfig.from_model_config(config)
    with init_empty_weights():
        model = model_cls(
            config,
            layer_spec=LayerSpec(parallel=parallel, quantization=quant_config),
        )
    _stream_checkpoint_weights(
        model,
        model_dir,
        device=device,
        dtype=dtype,
        set_module_tensor_to_device=set_module_tensor_to_device,
        weight_spec=weight_spec,
        model_scope=model_scope,
    )
    apply_weight_ties(model, weight_spec)
    process_quantized_modules(model.modules())
    model.to(device=device)
    model.eval()
    return model, tokenizer, str(infer_input_device(model, fallback=device))


def _materialize_tensor_if_needed(
    model: nn.Module,
    name: str,
    reference: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    set_module_tensor_to_device: Callable[..., Any],
) -> None:
    """Provision target storage while retaining immutable placement policy."""

    parent, leaf, old_tensor = _resolve_module_tensor(model, name)
    if not getattr(old_tensor, "is_meta", False):
        return
    attrs = capture_tensor_policy(old_tensor)
    target_dtype = _target_dtype_for_loaded_tensor(
        reference,
        old_tensor,
        dtype,
        _should_keep_checkpoint_dtype(reference, old_tensor),
    )
    target_value = torch.empty(
        tuple(int(dimension) for dimension in old_tensor.shape),
        dtype=target_dtype,
    )
    set_module_tensor_to_device(
        model,
        name,
        device,
        value=target_value,
        dtype=target_dtype if target_value.is_floating_point() else None,
        clear_cache=False,
    )
    restore_tensor_policy(getattr(parent, leaf), attrs)


def _stream_checkpoint_weights(
    model: nn.Module,
    model_dir: str,
    *,
    device: str,
    dtype: torch.dtype,
    set_module_tensor_to_device: Callable[..., Any],
    weight_spec: WeightSpec,
    model_scope: str | None,
) -> None:
    """Stream declared weights into final device storage and reject drift."""

    expected = set(model.state_dict().keys())
    in_scope = {name for name in expected if _tower_includes(weight_spec.tower, name, model_scope)}
    optional = {name for name, param in model.named_parameters() if is_optional_checkpoint(param)}
    loaded: set[str] = set()
    unexpected: list[str] = []
    params = dict(model.named_parameters())
    for source_name, tensor in iter_weights(resolve_weight_files(model_dir)):
        fragments = transform_weight(weight_spec, source_name, tensor)
        for fragment in fragments:
            target_name = maybe_remap_kv_scale_name(fragment.name, params)
            shard_id = fragment.shard_id
            if target_name not in params and fragment.stacked:
                fallback = map_weight_name(weight_spec, source_name)
                if fallback is not None and fallback in params:
                    target_name = fallback
                    shard_id = None
            if target_name not in expected:
                unexpected.append(source_name)
                continue
            if target_name not in in_scope:
                continue
            _materialize_tensor_if_needed(
                model,
                target_name,
                fragment.tensor,
                device=device,
                dtype=dtype,
                set_module_tensor_to_device=set_module_tensor_to_device,
            )
            load_parameter(
                model,
                target_name,
                fragment.tensor,
                shard_id=shard_id,
                declared_shard=fragment.sharded,
                quantization=fragment.quantization,
                dtype=dtype,
            )
            loaded.add(target_name)
    tied = {transform.target for transform in weight_spec.transforms if isinstance(transform, Tie)}
    missing = sorted(in_scope - loaded - optional - tied)
    if missing or unexpected:
        raise RuntimeError(
            "native checkpoint load mismatch: "
            f"missing={len(missing)} {_preview_names(missing)} "
            f"unexpected={len(unexpected)} {_preview_names(unexpected)}"
        )


def _preview_names(names: list[str], *, limit: int = 50) -> str:
    """Render a name list for error messages without silently dropping entries.

    Shows up to ``limit`` names; if more remain, an explicit
    ``(+N more)`` marker reports the discarded count so the truncation is never
    silent.
    """

    if len(names) <= limit:
        return repr(names)
    shown = names[:limit]
    return f"{shown!r} (+{len(names) - limit} more)"


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }


def _resolve_module_tensor(
    module: nn.Module, tensor_name: str
) -> tuple[nn.Module, str, torch.Tensor]:
    parent = module
    leaf = tensor_name
    if "." in tensor_name:
        parts = tensor_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf = parts[-1]
    tensor = getattr(parent, leaf)
    return parent, leaf, tensor


def _should_keep_checkpoint_dtype(tensor: torch.Tensor, target: torch.Tensor) -> bool:
    return _is_float8_dtype(tensor.dtype) or skip_serving_cast(target)


def _tower_includes(split: TowerSplit | None, name: str, role: str | None) -> bool:
    if role is None or split is None:
        return True
    generation = name.startswith(split.generation_prefixes) or any(
        infix in name for infix in split.generation_infixes
    )
    if role == "generation":
        return generation
    if role == "understanding":
        return not generation
    raise ValueError(f"unknown model scope {role!r}")


def _target_dtype_for_loaded_tensor(
    tensor: torch.Tensor,
    target: torch.Tensor,
    dtype: torch.dtype,
    keep_checkpoint_dtype: bool,
) -> torch.dtype:
    if keep_checkpoint_dtype and tensor.is_floating_point():
        return tensor.dtype
    if tensor.is_floating_point():
        return dtype
    return target.dtype
