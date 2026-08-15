"""Checkpoint tensor I/O and parameter placement primitives."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open
from torch import nn

from ..nn.placement import place_partitioned_tensor
from ..nn.quant.fp8 import W8A8Fp8LinearMethod
from ..nn.quant.load_state import (
    capture_tensor_policy,
    copy_tensor_policy,
    is_optional_checkpoint,
    restore_tensor_policy,
    set_fp8_scale_loaded,
    set_fp8_weight_loaded_offline,
    skip_serving_cast,
)
from ..nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    zero_vocab_padding,
)

WeightNameMap = tuple[tuple[str, str, str | int], ...]

__all__ = [
    "WeightNameMap",
    "dtype_from_name",
    "infer_input_device",
    "iter_weights",
    "load_parameter",
    "materialize_parameter",
    "missing_required_parameters",
    "prepare_serving_dtype",
    "preview_names",
    "resolve_weight_files",
    "root_weight_file",
    "stacked_weight_name",
    "tensor_shape",
]


def resolve_weight_files(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    if root.is_file():
        return [root]
    index = root / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        mapped = sorted({root / str(name) for name in data.get("weight_map", {}).values()})
        if mapped:
            missing = [path for path in mapped if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"checkpoint index references missing shard {missing[0]}")
            return mapped
    safetensors = sorted(root.glob("*.safetensors"))
    if safetensors:
        return safetensors
    binaries = [*sorted(root.glob("*.bin")), *sorted(root.glob("*.pt"))]
    if binaries:
        return binaries
    raise FileNotFoundError(f"no supported weight files found under {root}")


def root_weight_file(model_path: str | Path, candidates: tuple[str, ...]) -> Path:
    root = Path(model_path)
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"no model weight file among {candidates!r} under {root}")


def iter_weights(files: Iterable[Path]):
    for path in files:
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    yield name, checkpoint.get_tensor(name)
            continue
        if path.suffix not in {".bin", ".pt"}:
            raise ValueError(f"unsupported weight file {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        for name, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                yield name, tensor


def tensor_shape(path: str | Path, tensor_name: str) -> tuple[int, ...]:
    file_path = Path(path)
    if file_path.suffix == ".safetensors":
        with safe_open(file_path, framework="pt", device="cpu") as checkpoint:
            return tuple(int(value) for value in checkpoint.get_slice(tensor_name).get_shape())
    state = torch.load(file_path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    tensor = state[tensor_name]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name!r} in {file_path} is not a tensor")
    return tuple(int(value) for value in tensor.shape)


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown model dtype {name!r}; expected bfloat16, float16, or float32"
        ) from exc


def stacked_weight_name(name: str, mapping: WeightNameMap) -> tuple[str, str | int | None]:
    """Map one checkpoint projection onto the packed parameter that owns it."""

    for target, source, part in mapping:
        mapped = _replace_path_segment(name, source, target)
        if mapped is not None:
            return mapped, part
    return name, None


def _replace_path_segment(name: str, source: str, target: str) -> str | None:
    name_parts = name.split(".")
    source_parts = source.split(".")
    width = len(source_parts)
    for index in range(len(name_parts) - width + 1):
        if name_parts[index : index + width] == source_parts:
            return ".".join(
                (*name_parts[:index], *target.split("."), *name_parts[index + width :])
            )
    return None


def load_parameter(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    shard_id: str | int | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    owner, leaf, parameter = _parameter_owner(model, name)
    if isinstance(owner, (VocabParallelEmbedding, ParallelLMHead)) and leaf in {
        "weight",
        "bias",
    }:
        if shard_id is not None:
            raise ValueError("vocabulary parameters cannot be packed checkpoint targets")
        _load_vocab_parameter(owner, parameter, tensor)
        return
    if isinstance(getattr(owner, "quant_method", None), W8A8Fp8LinearMethod):
        if leaf == "weight":
            _load_fp8_weight(owner, parameter, tensor, shard_id=shard_id)
            return
        if leaf == "weight_scale":
            _load_fp8_scale(owner, parameter, tensor, shard_id=shard_id)
            return
    loaded = tensor
    if dtype is not None and loaded.is_floating_point():
        loaded = loaded.to(dtype=dtype)
    _load_partitioned(parameter, loaded, shard_id=shard_id)


def materialize_parameter(
    model: nn.Module,
    name: str,
    reference: torch.Tensor,
    *,
    device: str,
    dtype: torch.dtype,
    set_module_tensor_to_device: Callable[..., Any],
) -> None:
    parent, leaf, tensor = _resolve_module_tensor(model, name)
    if not tensor.is_meta:
        return
    policy = capture_tensor_policy(tensor)
    target_dtype = _materialized_dtype(reference, tensor, dtype)
    value = torch.empty(tuple(int(size) for size in tensor.shape), dtype=target_dtype)
    set_module_tensor_to_device(
        model,
        name,
        device,
        value=value,
        dtype=target_dtype if value.is_floating_point() else None,
        clear_cache=False,
    )
    restore_tensor_policy(getattr(parent, leaf), policy)


def missing_required_parameters(
    model: nn.Module,
    loaded: set[str],
    *,
    included: set[str] | None = None,
) -> list[str]:
    expected = included if included is not None else {name for name, _ in model.named_parameters()}
    optional = {
        name for name, parameter in model.named_parameters() if is_optional_checkpoint(parameter)
    }
    return sorted(expected - loaded - optional)


def prepare_serving_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    quantized = any(
        bool(getattr(getattr(module, "quant_method", None), "is_quantized", False))
        for module in model.modules()
    )
    if not quantized:
        model.to(dtype=dtype)
        return
    for parameter in model.parameters():
        if parameter.dtype == torch.float32 and not skip_serving_cast(parameter):
            parameter.data = parameter.data.to(dtype=dtype)


def infer_input_device(
    model: nn.Module, fallback: str | torch.device | None = None
) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type not in {"cpu", "meta"}:
            return parameter.device
    if fallback is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(fallback)


def preview_names(names: list[str], *, limit: int = 50) -> str:
    if len(names) <= limit:
        return repr(names)
    return f"{names[:limit]!r} (+{len(names) - limit} more)"


def _parameter_owner(model: nn.Module, name: str) -> tuple[nn.Module, str, nn.Parameter]:
    owner = model
    parts = name.split(".")
    for part in parts[:-1]:
        child = getattr(owner, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"weight target {name!r} traverses a non-module attribute")
        owner = child
    leaf = parts[-1]
    value = getattr(owner, leaf)
    if not isinstance(value, nn.Parameter):
        raise TypeError(f"weight target {name!r} is not a parameter")
    return owner, leaf, value


def _resolve_module_tensor(
    module: nn.Module, name: str
) -> tuple[nn.Module, str, torch.Tensor]:
    parent: nn.Module = module
    parts = name.split(".")
    for part in parts[:-1]:
        child = getattr(parent, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"tensor target {name!r} traverses a non-module attribute")
        parent = child
    leaf = parts[-1]
    tensor = getattr(parent, leaf)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor target {name!r} is not a tensor")
    return parent, leaf, tensor


def _load_partitioned(
    parameter: nn.Parameter,
    loaded: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    value = loaded.to(device=parameter.device, dtype=parameter.dtype)
    place_partitioned_tensor(parameter, parameter.data, value, shard_id=shard_id)


def _load_vocab_parameter(
    module: VocabParallelEmbedding | ParallelLMHead,
    parameter: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    target = parameter.data
    loaded = loaded_weight.to(device=target.device, dtype=target.dtype)
    raw_vocab_size = (
        module.num_embeddings
        if isinstance(module, VocabParallelEmbedding)
        else module.vocab_size
    )
    partition_size = int(target.shape[0])
    start = int(module.vocab_start_index)
    end = int(module.vocab_end_index)
    if loaded.shape == target.shape:
        target.copy_(loaded)
    else:
        if loaded.ndim != target.ndim or loaded.shape[1:] != target.shape[1:]:
            raise ValueError(
                f"loaded vocab tensor shape {tuple(loaded.shape)} does not match target {tuple(target.shape)}"
            )
        target.zero_()
        copy_start = max(0, start)
        copy_end = min(end, int(raw_vocab_size), int(loaded.shape[0]))
        if copy_end > copy_start:
            target[copy_start - start : copy_end - start].copy_(loaded[copy_start:copy_end])
    zero_vocab_padding(int(raw_vocab_size), start, partition_size, target)


def _load_fp8_weight(
    module: nn.Module,
    parameter: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    offline = loaded_weight.dtype == torch.float8_e4m3fn
    if offline and parameter.dtype != torch.float8_e4m3fn:
        replacement = nn.Parameter(
            torch.empty_like(parameter, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        copy_tensor_policy(parameter, replacement)
        module.weight = replacement
        parameter = replacement
    _load_partitioned(parameter, loaded_weight, shard_id=shard_id)
    set_fp8_weight_loaded_offline(module, offline)


def _load_fp8_scale(
    module: nn.Module,
    parameter: nn.Parameter,
    loaded_scale: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    canonical = loaded_scale.reshape(-1, 1) if loaded_scale.ndim == 1 else loaded_scale
    _load_partitioned(parameter, canonical, shard_id=shard_id)
    set_fp8_scale_loaded(module, True)


def _materialized_dtype(
    source: torch.Tensor,
    target: torch.Tensor,
    serving_dtype: torch.dtype,
) -> torch.dtype:
    float8_dtypes = {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }
    if source.dtype in float8_dtypes or skip_serving_cast(target):
        return source.dtype
    if source.is_floating_point():
        return serving_dtype
    return target.dtype
