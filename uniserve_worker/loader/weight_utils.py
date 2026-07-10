"""Weight-file resolution and streaming helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import torch
from safetensors.torch import safe_open

from ..nn.linear import default_weight_loader
from ..nn.quant.load_state import get_weight_loader

__all__ = [
    'resolve_weight_files',
    'iter_weights',
    'tensor_shape',
    'WeightLoadReport',
    'WeightLoadSummary',
    'strict_load_weights',
    'StackedParamMapping',
    'parameter_by_name',
    'load_parameter',
    'maybe_remap_kv_scale_name',
    'stacked_params_mapping_loop',
]


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


@dataclass(frozen=True)
class WeightLoadReport:
    loaded: set[str] = field(default_factory=set)
    ignored: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()


@dataclass(frozen=True)
class WeightLoadSummary:
    tensors_seen: int
    loaded_count: int | None
    ignored: tuple[str, ...] = ()


def strict_load_weights(model, weights) -> WeightLoadSummary:
    if not hasattr(model, "load_weights"):
        raise TypeError(f"{type(model).__name__} must implement load_weights(weights)")
    tensors_seen = 0

    def counted_weights():
        nonlocal tensors_seen
        for item in weights:
            tensors_seen += 1
            yield item

    result = model.load_weights(counted_weights())
    loaded_count: int | None = None
    ignored: tuple[str, ...] = ()
    if result is not None:
        if isinstance(result, set):
            loaded_count = len(result)
            missing = None
            unexpected = None
        else:
            # Consume the typed WeightLoadReport by attribute (its real
            # producer's contract), not by reflecting HF-style
            # ``.missing_keys``/``.unexpected_keys`` names no producer emits.
            loaded = getattr(result, "loaded", None)
            loaded_count = len(loaded) if loaded is not None else None
            ignored = tuple(getattr(result, "ignored", ()) or ())
            missing = getattr(result, "missing", None)
            unexpected = getattr(result, "unexpected", None)
        if missing or unexpected:
            raise RuntimeError(f"weight load mismatch: missing={missing} unexpected={unexpected}")
    return WeightLoadSummary(
        tensors_seen=tensors_seen,
        loaded_count=loaded_count,
        ignored=ignored,
    )


class StackedParamMapping(NamedTuple):
    target: str
    source: str
    shard_id: str | int


def parameter_by_name(model, name: str) -> torch.nn.Parameter:
    params = dict(model.named_parameters())
    try:
        return params[name]
    except KeyError as exc:
        raise KeyError(f"model has no parameter named {name!r}") from exc


def load_parameter(
    model,
    name: str,
    tensor: torch.Tensor,
    *,
    shard_id: str | int | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    param = parameter_by_name(model, name)
    loaded = tensor.to(dtype=dtype) if dtype is not None else tensor
    loader = get_weight_loader(param) or default_weight_loader
    loader(param, loaded, shard_id=shard_id)


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


def stacked_params_mapping_loop(
    model,
    weights: Iterable[tuple[str, torch.Tensor]],
    mapping: Iterable[StackedParamMapping | tuple[str, str, str | int]],
    *,
    name_mapper: Callable[[str], str | None] | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[set[str], list[str]]:
    """Load checkpoint tensors through per-parameter shard loaders.

    ``mapping`` is the SGLang/vLLM-style table of
    ``(target_substring, source_substring, shard_id)`` entries.  For each
    checkpoint tensor, the first matching source substring is replaced with the
    target substring and copied through the destination parameter's
    ``weight_loader`` hook.  ``name_mapper`` then normalizes the candidate
    target name (for example, HF prefix -> UniServe prefix); returning ``None``
    ignores that tensor.
    """

    loaded: set[str] = set()
    ignored: list[str] = []
    normalized = [
        item if isinstance(item, StackedParamMapping) else StackedParamMapping(*item)
        for item in mapping
    ]
    params = dict(model.named_parameters())
    for source_name, tensor in weights:
        target_name: str | None = source_name
        shard_id: str | int | None = None
        matched_item: StackedParamMapping | None = None
        for item in normalized:
            if item.source in source_name:
                # Replace only the first occurrence: the stacked-mapping table is a
                # SGLang/vLLM-style substring convention, so a source fragment such as
                # ``q_proj`` is expected to identify exactly one stacked weight in a
                # given checkpoint name. Bounding to a single substitution avoids
                # corrupting a name that happens to contain the fragment twice.
                target_name = source_name.replace(item.source, item.target, 1)
                shard_id = item.shard_id
                matched_item = item
                break
        matched_stacked = matched_item is not None
        if name_mapper is not None:
            target_name = name_mapper(source_name)
            if matched_item is not None and target_name is not None:
                target_name = name_mapper(
                    source_name.replace(matched_item.source, matched_item.target, 1)
                )
        elif not matched_stacked:
            target_name = None
        if target_name is None:
            ignored.append(source_name)
            continue
        target_name = maybe_remap_kv_scale_name(target_name, params)
        if target_name not in params:
            if matched_stacked and name_mapper is not None:
                fallback = name_mapper(source_name)
                if fallback is not None and fallback in params:
                    target_name = fallback
                    shard_id = None
                else:
                    ignored.append(source_name)
                    continue
            else:
                ignored.append(source_name)
                continue
        load_parameter(model, target_name, tensor, shard_id=shard_id, dtype=dtype)
        loaded.add(target_name)
    return loaded, ignored
