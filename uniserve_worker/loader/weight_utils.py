"""Weight-file resolution and streaming helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import safe_open
from torch import nn

from ..nn.placement import get_shard_plan, place_partitioned_tensor
from ..nn.quant.fp8 import W8A8Fp8LinearMethod
from ..nn.quant.load_state import (
    copy_tensor_policy,
    set_fp8_scale_loaded,
    set_fp8_weight_loaded_offline,
)
from ..nn.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    zero_vocab_padding,
)
from .schema import (
    Cast,
    Quantize,
    Rename,
    Reshape,
    Shard,
    Slice,
    Split,
    Stack,
    Tie,
    Transpose,
    UnmatchedWeightPolicy,
    WeightSpec,
)

__all__ = [
    'resolve_weight_files',
    'iter_weights',
    'tensor_shape',
    'parameter_by_name',
    'load_parameter',
    'map_weight_name',
    'maybe_remap_kv_scale_name',
    'load_declared_weights',
    'apply_weight_ties',
    'transform_weight',
    'WeightFragment',
]


@dataclass(frozen=True, slots=True)
class WeightFragment:
    """One tensor produced by the closed checkpoint transform algebra."""

    name: str
    tensor: torch.Tensor
    shard_id: str | int | None = None
    stacked: bool = False
    sharded: bool = False
    quantization: str | None = None


def map_weight_name(spec: WeightSpec, name: str) -> str | None:
    """Apply the declared checkpoint namespace transform exactly once."""

    for transform in spec.transforms:
        if not isinstance(transform, Rename):
            continue
        if transform.exact:
            if name != transform.source:
                continue
            mapped = transform.target
        else:
            if not name.startswith(transform.source):
                continue
            mapped = transform.target + name[len(transform.source) :]
        for source, target in transform.substitutions:
            mapped = mapped.replace(source, target)
        return mapped
    return name if spec.unmatched is UnmatchedWeightPolicy.KEEP else None


def resolve_weight_files(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    if root.is_file():
        return [root]
    index = root / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        mapped = {root / fname for fname in data.get("weight_map", {}).values()}
        if mapped:
            # Union the index-referenced shards with any other ``*.safetensors``
            # files in the directory. A partial/stale index must not silently drop
            # shards that exist on disk, and the union keeps shard discovery stable
            # regardless of glob ordering.
            return sorted(mapped | set(root.glob("*.safetensors")))
    safetensors = sorted(root.glob("*.safetensors"))
    if safetensors:
        return safetensors
    pt_files = [*sorted(root.glob("*.bin")), *sorted(root.glob("*.pt"))]
    if pt_files:
        return pt_files
    raise FileNotFoundError(f"no supported weight files found under {root}")


def iter_weights(files: Iterable[Path]):
    for path in files:
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)
        elif path.suffix in {".bin", ".pt"}:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            for name, tensor in state.items():
                if isinstance(tensor, torch.Tensor):
                    yield name, tensor
        else:
            raise ValueError(f"unsupported weight file {path}")


def tensor_shape(path: str | Path, tensor_name: str) -> tuple[int, ...]:
    file_path = Path(path)
    if file_path.suffix == ".safetensors":
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return tuple(int(v) for v in f.get_slice(tensor_name).get_shape())
    state = torch.load(file_path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    tensor = state[tensor_name]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name!r} in {file_path} is not a tensor")
    return tuple(int(v) for v in tensor.shape)


def parameter_by_name(model: nn.Module, name: str) -> nn.Parameter:
    params = dict(model.named_parameters())
    try:
        return params[name]
    except KeyError as exc:
        raise KeyError(f"model has no parameter named {name!r}") from exc


def load_parameter(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    shard_id: str | int | None = None,
    declared_shard: bool = False,
    quantization: str | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    owner, leaf, param = _parameter_owner(model, name)
    if declared_shard and get_shard_plan(param) is None:
        raise ValueError(f"declared shard target {name!r} has no resolved ShardPlan")
    if quantization is not None:
        if quantization not in {"fp8", "w8a8_fp8"}:
            raise ValueError(f"unsupported declared weight quantization {quantization!r}")
        if not isinstance(getattr(owner, "quant_method", None), W8A8Fp8LinearMethod):
            raise ValueError(
                f"declared quantization {quantization!r} does not match target {name!r}"
            )
    if isinstance(owner, (VocabParallelEmbedding, ParallelLMHead)) and leaf in {
        "weight",
        "bias",
    }:
        if shard_id is not None:
            raise ValueError("vocabulary parameters cannot be stacked checkpoint targets")
        _load_vocab_parameter(owner, param, tensor)
        return
    if isinstance(getattr(owner, "quant_method", None), W8A8Fp8LinearMethod):
        if leaf == "weight":
            _load_fp8_weight(owner, param, tensor, shard_id=shard_id)
            return
        if leaf == "weight_scale":
            _load_fp8_scale(owner, param, tensor, shard_id=shard_id)
            return
    loaded = tensor
    if dtype is not None and loaded.is_floating_point():
        loaded = loaded.to(dtype=dtype)
    _load_partitioned(param, loaded, shard_id=shard_id)


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


def _load_partitioned(
    param: nn.Parameter,
    loaded: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    value = loaded.to(device=param.device, dtype=param.dtype)
    place_partitioned_tensor(param, param.data, value, shard_id=shard_id)


def _load_vocab_parameter(
    module: VocabParallelEmbedding | ParallelLMHead,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    target = param.data
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
                f"loaded vocab tensor shape {tuple(loaded.shape)} != target {tuple(target.shape)}"
            )
        target.zero_()
        copy_start = max(0, start)
        copy_end = min(end, int(raw_vocab_size), int(loaded.shape[0]))
        if copy_end > copy_start:
            target[copy_start - start : copy_end - start].copy_(
                loaded[copy_start:copy_end]
            )
    zero_vocab_padding(int(raw_vocab_size), start, partition_size, target)


def _load_fp8_weight(
    module: nn.Module,
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    offline = loaded_weight.dtype == torch.float8_e4m3fn
    if offline and param.dtype != torch.float8_e4m3fn:
        replacement = nn.Parameter(
            torch.empty_like(param, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        copy_tensor_policy(param, replacement)
        module.weight = replacement
        param = replacement
    _load_partitioned(param, loaded_weight, shard_id=shard_id)
    set_fp8_weight_loaded_offline(module, offline)


def _load_fp8_scale(
    module: nn.Module,
    param: nn.Parameter,
    loaded_scale: torch.Tensor,
    *,
    shard_id: str | int | None,
) -> None:
    canonical = loaded_scale.reshape(-1, 1) if loaded_scale.ndim == 1 else loaded_scale
    _load_partitioned(param, canonical, shard_id=shard_id)
    set_fp8_scale_loaded(module, True)


def maybe_remap_kv_scale_name(name: str, params: dict[str, torch.nn.Parameter]) -> str:
    """Map future kv-scale checkpoint names when a matching param exists.

    UniServe does not have fp8 kv-scale params yet, so this is intentionally an
    honest no-op unless a model adds the expected destination parameter.
    """

    candidates = (
        name.replace(".kv_scale", ".attn.kv_scale"),
        name.replace(".k_scale", ".attn.k_scale"),
        name.replace(".v_scale", ".attn.v_scale"),
    )
    for candidate in candidates:
        if candidate in params:
            return candidate
    return name


def transform_weight(
    spec: WeightSpec,
    source_name: str,
    tensor: torch.Tensor,
) -> tuple[WeightFragment, ...]:
    """Apply the ordered, closed ``WeightSpec`` tensor algebra to one source."""

    fragments = [WeightFragment(source_name, tensor)]
    matched = False
    for transform in spec.transforms:
        if isinstance(transform, Tie):
            continue
        next_fragments: list[WeightFragment] = []
        for fragment in fragments:
            applied = _apply_transform(fragment, transform)
            if applied is None:
                next_fragments.append(fragment)
            else:
                matched = True
                next_fragments.extend(applied)
        fragments = next_fragments
    if not matched and spec.unmatched is UnmatchedWeightPolicy.SKIP:
        return ()
    return tuple(fragments)


def _apply_transform(
    fragment: WeightFragment,
    transform: object,
) -> tuple[WeightFragment, ...] | None:
    name = fragment.name
    if isinstance(transform, Rename):
        if transform.exact:
            if name != transform.source:
                return None
            target = transform.target
        else:
            if not name.startswith(transform.source):
                return None
            target = transform.target + name[len(transform.source) :]
        for source, replacement in transform.substitutions:
            target = target.replace(source, replacement)
        return (replace(fragment, name=target),)
    if isinstance(transform, Stack):
        if fragment.stacked:
            return None
        target_name = _replace_path_segments(name, transform.source, transform.target)
        if target_name is None:
            return None
        return (
            replace(
                fragment,
                name=target_name,
                shard_id=transform.part,
                stacked=True,
            ),
        )
    if not isinstance(
        transform,
        (Slice, Split, Transpose, Reshape, Shard, Cast, Quantize),
    ):
        raise TypeError(f"unknown weight transform {type(transform).__name__}")
    if transform.source != name:
        return None
    if isinstance(transform, Slice):
        axis = _axis(fragment.tensor, transform.axis, name)
        return (
            replace(
                fragment,
                name=transform.target,
                tensor=fragment.tensor.narrow(
                    axis,
                    int(transform.start),
                    int(transform.stop - transform.start),
                ),
            ),
        )
    if isinstance(transform, Split):
        axis = _axis(fragment.tensor, transform.axis, name)
        if sum(transform.sizes) != int(fragment.tensor.shape[axis]):
            raise ValueError(f"weight split sizes do not cover {name!r}")
        pieces = torch.split(fragment.tensor, list(transform.sizes), dim=axis)
        return tuple(
            replace(fragment, name=target, tensor=piece)
            for target, piece in zip(transform.targets, pieces, strict=True)
        )
    if isinstance(transform, Transpose):
        if sorted(transform.axes) != list(range(fragment.tensor.ndim)):
            raise ValueError(f"weight transpose axes are invalid for {name!r}")
        return (
            replace(
                fragment,
                name=transform.target,
                tensor=fragment.tensor.permute(transform.axes).contiguous(),
            ),
        )
    if isinstance(transform, Reshape):
        try:
            value = fragment.tensor.reshape(transform.shape)
        except RuntimeError as exc:
            raise ValueError(f"weight reshape is invalid for {name!r}") from exc
        return (replace(fragment, name=transform.target, tensor=value),)
    if isinstance(transform, Shard):
        if transform.topology_axis != "tp":
            raise ValueError(
                f"weight shard topology {transform.topology_axis!r} is not provisioned"
            )
        return (replace(fragment, name=transform.target, sharded=True),)
    if isinstance(transform, Cast):
        return (
            replace(
                fragment,
                name=transform.target,
                tensor=fragment.tensor.to(dtype=_declared_dtype(transform.dtype)),
            ),
        )
    if isinstance(transform, Quantize):
        return (
            replace(
                fragment,
                name=transform.target,
                quantization=transform.scheme,
            ),
        )
    raise AssertionError("closed weight transform dispatch is incomplete")


def _replace_path_segments(name: str, source: str, target: str) -> str | None:
    """Replace one exact dot-delimited parameter-path fragment."""

    name_parts = name.split(".")
    source_parts = source.split(".")
    target_parts = target.split(".")
    width = len(source_parts)
    for index in range(len(name_parts) - width + 1):
        if name_parts[index : index + width] == source_parts:
            return ".".join(
                (*name_parts[:index], *target_parts, *name_parts[index + width :])
            )
    return None


def _axis(tensor: torch.Tensor, axis: int, name: str) -> int:
    value = int(axis)
    if value < 0:
        value += tensor.ndim
    if value < 0 or value >= tensor.ndim:
        raise ValueError(f"weight transform axis is invalid for {name!r}")
    return value


def _declared_dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown declared weight dtype {name!r}")
    return value


def apply_weight_ties(model: nn.Module, spec: WeightSpec) -> None:
    """Publish declared parameter identity ties after checkpoint materialization."""

    for transform in spec.transforms:
        if not isinstance(transform, Tie):
            continue
        _source_owner, _source_leaf, source = _parameter_owner(model, transform.source)
        target_owner, target_leaf, target = _parameter_owner(model, transform.target)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f"weight tie {transform.source!r}->{transform.target!r} has incompatible geometry"
            )
        setattr(target_owner, target_leaf, source)


def load_declared_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    spec: WeightSpec,
    dtype: torch.dtype | None = None,
) -> tuple[set[str], list[str]]:
    """Load a checkpoint stream through the immutable ``WeightSpec`` declaration."""

    loaded: set[str] = set()
    ignored: list[str] = []
    params = dict(model.named_parameters())
    for source_name, tensor in weights:
        fragments = transform_weight(spec, source_name, tensor)
        if not fragments:
            ignored.append(source_name)
            continue
        for fragment in fragments:
            target_name = maybe_remap_kv_scale_name(fragment.name, params)
            shard_id = fragment.shard_id
            if target_name not in params and fragment.stacked:
                fallback = map_weight_name(spec, source_name)
                if fallback is not None and fallback in params:
                    target_name = fallback
                    shard_id = None
                else:
                    ignored.append(source_name)
                    continue
            if target_name not in params:
                ignored.append(source_name)
                continue
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
    return loaded, ignored
