"""Hugging Face native checkpoint loader: meta-init plus per-tensor streaming."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from ..foundation.env import DEFAULT_ATTENTION_BACKEND
from ..foundation.runtime_config import get_execution_config
from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..nn.placement import get_shard_plan
from ..nn.quant import QuantizationConfig, use_quantization_config
from ..nn.quant.base import process_quantized_modules
from ..nn.quant.load_state import (
    allow_shape_mismatch,
    capture_tensor_policy,
    get_weight_loader,
    has_weight_loader,
    is_optional_checkpoint,
    restore_tensor_policy,
    set_fp8_scale_loaded,
    set_fp8_weight_loaded_offline,
    skip_serving_cast,
)
from .base import BaseModelLoader, LoadResult
from .registry import register_loader
from .weight_spec import WeightSpec, weight_spec_of
from .weight_utils import StackedParamMapping, iter_weights, load_parameter, resolve_weight_files

__all__ = [
    "dtype_from_name",
    "infer_input_device",
    "load_native_transformers_checkpoint",
    "NativeTransformersLoader",
]


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    try:
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown transformer dtype {name!r}; expected one of bfloat16, float16, or float32"
        ) from exc


def _load_tokenizer(
    tokenizer_cls: Any,
    model_dir: str,
    *,
    use_fast: bool = False,
    extra_special_tokens: dict[str, Any] | None = None,
) -> Any:
    """Load a tokenizer and attach ``model_dir`` context to failures."""

    try:
        return tokenizer_cls.from_pretrained(
            model_dir,
            use_fast=use_fast,
            extra_special_tokens=dict(extra_special_tokens or {}),
            trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - error-context wrapper.
        raise RuntimeError(
            f"failed to load tokenizer from {model_dir!r} (use_fast={use_fast}): {exc}"
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
    tokenizer_cls: Any,
    attention_backend: str | None = None,
    use_fast: bool = False,
    extra_special_tokens: dict[str, Any] | None = None,
    min_version_key: str | None = None,
    code_version: str | None = None,
    param_filter: Callable[[str], bool] | None = None,
    stacked_params_mapping: tuple[StackedParamMapping | tuple[str, str, str | int], ...] = (),
) -> tuple[nn.Module, Any, str]:
    """Instantiate a native ``nn.Module`` model and stream HF-format weights.

    This keeps Hugging Face config/tokenizer compatibility while avoiding the
    ``PreTrainedModel.from_pretrained`` model base.  Parameters are created on
    ``meta`` and materialized one checkpoint tensor at a time on ``device``.
    ``param_filter`` restricts materialization to a tower role (partial load).
    """

    try:
        from accelerate import init_empty_weights
        from accelerate.utils import set_module_tensor_to_device
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific.
        raise RuntimeError(
            "native checkpoint loading requires accelerate; install it in the worker environment"
        ) from exc

    runtime = get_execution_config()
    attn_backend = attention_backend or DEFAULT_ATTENTION_BACKEND
    dtype = dtype_from_name(runtime.model_dtype)

    config = config_cls.from_pretrained(model_dir)
    config.uniserve_attention_backend = attn_backend
    _check_min_code_version(config, min_version_key, code_version)

    tokenizer = _load_tokenizer(
        tokenizer_cls,
        model_dir,
        use_fast=use_fast,
        extra_special_tokens=extra_special_tokens,
    )
    quant_config = QuantizationConfig.from_model_config(config)
    with use_quantization_config(quant_config):
        with init_empty_weights():
            model = model_cls(config)
    _stream_checkpoint_weights(
        model,
        model_dir,
        device=device,
        dtype=dtype,
        set_module_tensor_to_device=set_module_tensor_to_device,
        param_filter=param_filter,
        stacked_params_mapping=stacked_params_mapping,
    )
    process_quantized_modules(model.modules())
    model.eval()
    return model, tokenizer, str(infer_input_device(model, fallback=device))


def _materialize_one_tensor(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    set_module_tensor_to_device: Callable[..., Any],
) -> None:
    """Materialize one checkpoint tensor onto a meta-initialized parameter.

    Routes through the destination param's ``weight_loader`` when present
    (honoring merged-QKV/quant/kv-scale logic) and otherwise writes directly;
    in both cases the param's tensor policy and quant-loaded markers are
    preserved/updated identically to the inline path.
    """
    parent, leaf, old_tensor = _resolve_module_tensor(model, name)
    attrs = capture_tensor_policy(old_tensor)
    keep_checkpoint_dtype = _should_keep_checkpoint_dtype(tensor, old_tensor)
    if _should_load_with_weight_loader(tensor, old_tensor):
        target_dtype = _target_dtype_for_loaded_tensor(
            tensor, old_tensor, dtype, keep_checkpoint_dtype
        )
        target_value = torch.empty(
            tuple(int(dim) for dim in old_tensor.shape),
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
        materialized = getattr(parent, leaf)
        restore_tensor_policy(materialized, attrs)
        loaded_value = (
            tensor
            if keep_checkpoint_dtype
            else (tensor.to(dtype=target_dtype) if tensor.is_floating_point() else tensor)
        )
        get_weight_loader(materialized)(materialized, loaded_value)
        _mark_quant_tensor_loaded(parent, leaf, tensor)
        return
    value = (
        tensor
        if keep_checkpoint_dtype
        else (tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor)
    )
    set_module_tensor_to_device(
        model,
        name,
        device,
        value=value,
        dtype=tensor.dtype
        if keep_checkpoint_dtype and tensor.is_floating_point()
        else (dtype if tensor.is_floating_point() else None),
        clear_cache=False,
    )
    restore_tensor_policy(getattr(parent, leaf), attrs)
    _mark_quant_tensor_loaded(parent, leaf, tensor)


def _materialize_stacked_tensor_if_needed(
    model: nn.Module,
    name: str,
    reference: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    set_module_tensor_to_device: Callable[..., Any],
) -> None:
    parent, leaf, old_tensor = _resolve_module_tensor(model, name)
    if not getattr(old_tensor, "is_meta", False):
        return
    attrs = capture_tensor_policy(old_tensor)
    keep_checkpoint_dtype = _should_keep_checkpoint_dtype(reference, old_tensor)
    target_dtype = _target_dtype_for_loaded_tensor(
        reference, old_tensor, dtype, keep_checkpoint_dtype
    )
    target_value = torch.empty(
        tuple(int(dim) for dim in old_tensor.shape),
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
    param_filter: Callable[[str], bool] | None = None,
    stacked_params_mapping: tuple[StackedParamMapping | tuple[str, str, str | int], ...] = (),
) -> None:
    """Stream HF-format weights into ``model`` one tensor at a time.

    Materializes each expected meta parameter from its checkpoint tensor and
    raises if any required parameter is missing or any checkpoint tensor is
    unexpected (optional params are exempt from the missing check).

    When ``param_filter`` is given (tower partial load), only params it accepts
    are materialized; the rest stay on ``meta`` (they cost no memory and the
    worker's role never reads them) and are exempt from the missing check.
    A checkpoint tensor for an out-of-scope param is skipped, not flagged.
    """
    expected = set(model.state_dict().keys())
    in_scope = {n for n in expected if param_filter(n)} if param_filter is not None else expected
    optional = {name for name, param in model.named_parameters() if is_optional_checkpoint(param)}
    loaded: set[str] = set()
    unexpected: list[str] = []
    params = dict(model.named_parameters())
    stacked = [
        item if isinstance(item, StackedParamMapping) else StackedParamMapping(*item)
        for item in stacked_params_mapping
    ]
    for name, tensor in iter_weights(resolve_weight_files(model_dir)):
        target_name = name
        shard_id: str | int | None = None
        matched_stacked = False
        for item in stacked:
            if item.source in name:
                candidate = name.replace(item.source, item.target, 1)
                if candidate in params:
                    target_name = candidate
                    shard_id = item.shard_id
                    matched_stacked = True
                break
        if target_name not in expected:
            unexpected.append(name)
            continue
        if target_name not in in_scope:
            continue
        if matched_stacked:
            _materialize_stacked_tensor_if_needed(
                model,
                target_name,
                tensor,
                device=device,
                dtype=dtype,
                set_module_tensor_to_device=set_module_tensor_to_device,
            )
            load_parameter(
                model, target_name, tensor.to(device=device), shard_id=shard_id, dtype=dtype
            )
        else:
            _materialize_one_tensor(
                model,
                target_name,
                tensor,
                device=device,
                dtype=dtype,
                set_module_tensor_to_device=set_module_tensor_to_device,
            )
        loaded.add(target_name)
    missing = sorted(in_scope - loaded - optional)
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


def _should_load_with_weight_loader(tensor: torch.Tensor, target: torch.Tensor) -> bool:
    """Decide whether to route a checkpoint tensor through ``param.weight_loader``.

    Any destination that carries a callable ``weight_loader`` (every
    ``LinearBase`` weight/bias param, plus merged/QKV/quant overrides) is
    materialized and then populated via the hook, so merged-QKV sharding,
    kv-scale remap, and quant create/copy logic stay honored on the native
    loader path instead of being silently bypassed by a direct
    ``set_module_tensor_to_device`` copy. Genuinely unhookable params
    (embeddings/norms carry no ``weight_loader``) fall through to the direct
    write.

    Shape-mismatched tensors additionally require the param to opt in via
    ``_uniserve_allow_shape_mismatch`` (e.g. a merged QKV destination); a
    mismatch without that opt-in is left to the direct-write path so its shape
    check surfaces the error rather than masking it.
    """

    if not has_weight_loader(target):
        return False
    if tuple(tensor.shape) != tuple(target.shape):
        return allow_shape_mismatch(target) or _shard_plan_expects_global_shape(tensor, target)
    return True


def _shard_plan_expects_global_shape(tensor: torch.Tensor, target: torch.Tensor) -> bool:
    """True when a tensor-parallel shard plan explains the shape mismatch.

    Under tp>1 every sharded ``LinearBase`` parameter is materialized at its
    local (narrowed) shape while the checkpoint streams the global tensor; the
    param's ``weight_loader`` performs the narrowing. Accept the mismatch only
    when narrowing the checkpoint shape by the plan's spec reproduces the
    target shape exactly, so genuinely wrong checkpoints still fail loudly on
    the direct-write path.
    """
    if not isinstance(target, torch.nn.Parameter):
        return False
    plan = get_shard_plan(target)
    if plan is None:
        return False
    spec = plan.spec
    if spec.replicated or int(spec.size) <= 1:
        return False
    if tensor.ndim != target.ndim or int(spec.axis) >= tensor.ndim:
        return False
    expected = list(tensor.shape)
    dim = int(expected[int(spec.axis)])
    if dim % int(spec.size) != 0:
        return False
    expected[int(spec.axis)] = dim // int(spec.size)
    return tuple(expected) == tuple(target.shape)


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


def _mark_quant_tensor_loaded(module: nn.Module, leaf: str, tensor: torch.Tensor) -> None:
    if leaf == "weight_scale" and hasattr(module, "_fp8_scale_loaded"):
        set_fp8_scale_loaded(module, True)
    if (
        leaf == "weight"
        and _is_float8_dtype(tensor.dtype)
        and hasattr(module, "_fp8_weight_loaded_offline")
    ):
        set_fp8_weight_loaded_offline(module, True)


class NativeTransformersLoader(BaseModelLoader):
    """The native HF weight-streaming materialization, governed by the loader ABC.

    Registered under the ``native`` load_format. ``model_cls`` is the serving
    wrapper class; its declared ``WeightSpec`` names the native config,
    tokenizer, and inner-module bindings plus the stacked-parameter and
    tower-split rules. The loader meta-initializes the inner module, streams
    the checkpoint into it, and constructs the serving wrapper around the
    ready module.
    """

    def load_model(
        self,
        model_cls: Any,
        config: Any,
        *,
        device: str = "cpu",
        model_path: str | None = None,
        **kwargs: Any,
    ) -> LoadResult:
        del config  # the native path resolves its own HF config from model_path
        if model_path is None:
            raise ValueError("NativeTransformersLoader requires model_path")
        spec: WeightSpec = weight_spec_of(model_cls)
        native = spec.native
        if native is None:
            raise ValueError(
                f"{model_cls.__name__} declares no native checkpoint source in its weight_spec"
            )
        # Tower partial load: the spec's tower split names the checkpoint params
        # belonging to a ``tower_role`` so an und/gen worker materializes only
        # its tower. Whole-model kinds pass ``tower_role=None`` (no filter).
        tower_role = kwargs.get("tower_role")
        param_filter = spec.tower.role_filter(tower_role) if spec.tower is not None else None
        inner, tokenizer, real_device = load_native_transformers_checkpoint(
            model_path,
            device,
            config_cls=native.config_cls,
            model_cls=native.module_cls,
            tokenizer_cls=native.tokenizer_cls,
            attention_backend=kwargs.get("attention_backend"),
            use_fast=native.use_fast,
            extra_special_tokens=native.extra_special_tokens,
            min_version_key=native.min_version_key,
            code_version=native.code_version,
            param_filter=param_filter,
            stacked_params_mapping=spec.stacked,
        )
        model = model_cls(
            inner.config,
            model=inner,
            tokenizer=tokenizer,
            device=real_device,
            gen_snapshot_kv_capacity=kwargs.get("gen_snapshot_kv_capacity"),
            block_size=kwargs.get("block_size", DEFAULT_BLOCK_SIZE),
            kv_token_capacity=kwargs.get("kv_token_capacity"),
            attention_backend=kwargs.get("attention_backend"),
            tower_role=tower_role,
        )
        return LoadResult(model=model, tokenizer=tokenizer, device=real_device)


register_loader("native", NativeTransformersLoader())
