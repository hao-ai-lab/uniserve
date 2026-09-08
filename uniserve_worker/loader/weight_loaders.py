"""Parameter-owned checkpoint assignment, sharding, and materialization policies."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..nn.quant.load_state import (
    copy_tensor_policy,
    set_fp8_scale_loaded,
    set_fp8_weight_loaded_offline,
    set_skip_serving_cast,
    skip_serving_cast,
)
from ..nn.shard import ShardPlan, get_shard_plan
from .handles import WeightHandle

# A parameter owns the policy that maps one checkpoint handle and logical shard into storage.
WeightLoader = Callable[[nn.Parameter, WeightHandle, str | int | None], None]

__all__ = [
    "WeightLoader",
    "WeightAssignment",
    "attach_parameter_loaders",
    "attach_weight_loader",
    "copy_parameter_loader_state",
    "default_weight_loader",
    "defer_parameter_weights",
    "fp8_scale_loader",
    "fp8_weight_loader",
    "interleaved_packed_weight_loader",
    "load_parameter_weight",
    "packed_weight_loader",
    "set_vocab_layout",
    "sharded_weight_loader",
    "vocab_weight_loader",
]

_LOADER_ATTR = "weight_loader"
_OWNER_ATTR = "_uniserve_parameter_owner"
_NAME_ATTR = "_uniserve_parameter_name"
_DEVICE_ATTR = "_uniserve_serving_device"
_DTYPE_ATTR = "_uniserve_serving_dtype"
_LOADED_ATTR = "_uniserve_checkpoint_loaded"
_VOCAB_ATTR = "_uniserve_vocab_layout"
_SHARDS_ATTR = "_uniserve_checkpoint_shards"
_BINDINGS_ATTR = "_uniserve_parameter_bindings"


@dataclass(frozen=True, slots=True)
class WeightAssignment:
    """One model-resolved checkpoint assignment awaiting subtree materialization."""

    parameter: nn.Parameter
    handle: WeightHandle
    shard_id: str | int | None

    def apply(self) -> None:
        """Materialize and assign this checkpoint slice to its destination parameter."""

        load_parameter_weight(_current_parameter(self.parameter), self.handle, self.shard_id)


_PENDING_WEIGHTS: ContextVar[list[WeightAssignment] | None] = ContextVar(
    "uniserve_pending_weights",
    default=None,
)


def attach_weight_loader(parameter: nn.Parameter, loader: WeightLoader) -> None:
    """Attach a parameter-specific checkpoint assignment policy."""

    setattr(parameter, _LOADER_ATTR, loader)


def attach_parameter_loaders(
    module: nn.Module,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    recurse: bool = True,
) -> None:
    """Bind parameter ownership, serving assignment, and default assignment policies.

    Every alias of a tied parameter is retained so meta-device materialization can
    replace all owning module slots with one shared parameter object.
    """

    target = str(torch.device(device))

    # Record each direct owner; tied parameters may accumulate multiple bindings.
    for owner in module.modules() if recurse else (module,):
        for name, parameter in owner.named_parameters(recurse=False):
            bindings = list(getattr(parameter, _BINDINGS_ATTR, ()))
            if not any(
                owner_ref() is owner and binding_name == name
                for owner_ref, binding_name in bindings
            ):
                bindings.append((weakref.ref(owner), name))
            setattr(parameter, _BINDINGS_ATTR, tuple(bindings))
            setattr(parameter, _OWNER_ATTR, weakref.ref(owner))
            setattr(parameter, _NAME_ATTR, name)
            setattr(parameter, _DEVICE_ATTR, target)
            setattr(parameter, _DTYPE_ATTR, dtype)
            setattr(parameter, _LOADED_ATTR, False)

            # Architecture-specific loaders remain authoritative when already attached.
            if not callable(getattr(parameter, _LOADER_ATTR, None)):
                attach_weight_loader(parameter, default_weight_loader)


def set_vocab_layout(
    parameter: nn.Parameter,
    *,
    real_size: int,
    start: int,
    end: int,
) -> None:
    """Record the real and rank-local vocabulary interval on a padded parameter."""

    setattr(parameter, _VOCAB_ATTR, (int(real_size), int(start), int(end)))
    attach_weight_loader(parameter, vocab_weight_loader)


def load_parameter_weight(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
) -> None:
    """Dispatch checkpoint assignment through the parameter's attached loader."""

    # Layered construction records logical assignments before allocating their owners.
    deferred = _PENDING_WEIGHTS.get()
    if deferred is not None:
        deferred.append(WeightAssignment(parameter, handle, shard_id))
        return

    # Resolve a replacement created through another tied binding before assignment.
    parameter = _current_parameter(parameter)
    loader = getattr(parameter, _LOADER_ATTR, None)
    if not callable(loader):
        raise TypeError("loadable parameter has no weight_loader")
    loader(parameter, handle, shard_id)


@contextmanager
def defer_parameter_weights() -> Iterator[list[WeightAssignment]]:
    """Record architecture-resolved assignments without materializing parameters."""

    assignments: list[WeightAssignment] = []
    token = _PENDING_WEIGHTS.set(assignments)
    try:
        yield assignments
    finally:
        _PENDING_WEIGHTS.reset(token)


def copy_parameter_loader_state(source: nn.Parameter, target: nn.Parameter) -> None:
    """Preserve load ownership when post-load processing replaces a parameter."""

    copy_tensor_policy(source, target)
    for attribute in (
        _LOADER_ATTR,
        _OWNER_ATTR,
        _NAME_ATTR,
        _DEVICE_ATTR,
        _DTYPE_ATTR,
        _LOADED_ATTR,
        _VOCAB_ATTR,
        _SHARDS_ATTR,
        _BINDINGS_ATTR,
    ):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))


def default_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
) -> None:
    """Copy a shape-compatible checkpoint tensor into an unsharded parameter."""

    if shard_id is not None:
        raise ValueError("ordinary parameters cannot receive packed checkpoint shards")
    parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
    payload = handle.full()
    if parameter.numel() == 1 and payload.numel() == 1:
        parameter.data.fill_(payload.item())
    else:
        _copy(parameter.data, payload, parameter)
    _mark_loaded(parameter)


def sharded_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
) -> None:
    """Copy the rank-local slice selected by a parameter shard plan."""

    if shard_id is not None:
        raise ValueError("sharded parameters cannot receive packed checkpoint shards")
    plan = _required_plan(parameter)
    parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
    payload = _payload_for_plan(handle, plan, tuple(parameter.shape))
    _copy(parameter.data, payload, parameter)
    _mark_loaded(parameter)


def packed_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
) -> None:
    """Copy one logical packed shard into its assigned interval of a merged parameter."""

    plan = _required_plan(parameter)
    if plan.shard_axis is None:
        raise ValueError("packed parameter plan has no shard axis")
    if shard_id is None:
        parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
        _copy_packed_tensor(parameter, handle, plan)
        _mark_loaded(parameter)
        return
    slot = plan.slot_for(shard_id)
    if slot is None:
        raise ValueError(f"unknown packed checkpoint shard id {shard_id!r}")
    parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
    selection = [slice(None)] * parameter.ndim
    selection[plan.shard_axis] = slice(slot.offset, slot.offset + slot.size)
    target = parameter.data[tuple(selection)]
    payload = _payload_for_shard(handle, slot.shard, tuple(target.shape))
    _copy(target, payload, parameter)
    _mark_shard_loaded(parameter, plan, shard_id)


def interleaved_packed_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
    *,
    rank: int,
    size: int,
    branches: int,
    group_width: int,
) -> None:
    """Load one branch into a rank-sharded, group-interleaved merged projection."""

    # Validate source divisibility and the exact merged destination geometry.
    if not isinstance(shard_id, int) or not 0 <= shard_id < int(branches):
        raise ValueError("interleaved packed parameters require an integer branch id")
    if len(handle.shape) != 2 or int(handle.shape[0]) % int(size):
        raise ValueError("interleaved projection source cannot be column-sharded")
    local_rows = int(handle.shape[0]) // int(size)
    if local_rows % int(group_width):
        raise ValueError("interleaved projection branch does not divide into output groups")
    expected = (local_rows * int(branches), int(handle.shape[1]))
    if tuple(parameter.shape) != expected:
        raise ValueError(
            f"interleaved projection target shape {tuple(parameter.shape)} != {expected}"
        )

    # Extract this rank's rows and project the branch onto the interleaved target view.
    parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
    payload = handle.narrow(0, int(rank) * local_rows, local_rows)
    target = parameter.data.view(
        local_rows // int(group_width),
        int(branches),
        int(group_width),
        int(handle.shape[1]),
    )[:, shard_id]
    _copy(target, payload.reshape_as(target), parameter)

    # The merged parameter becomes complete only after every logical branch arrives.
    loaded = set(getattr(parameter, _SHARDS_ATTR, set()))
    loaded.add(shard_id)
    setattr(parameter, _SHARDS_ATTR, loaded)
    if len(loaded) == int(branches):
        _mark_loaded(parameter)


def vocab_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
) -> None:
    """Copy the rank-local real vocabulary interval and zero padded rows."""

    # Resolve the rank interval recorded when the vocabulary parameter was constructed.
    if shard_id is not None:
        raise ValueError("vocabulary parameters cannot receive packed checkpoint shards")
    layout = getattr(parameter, _VOCAB_ATTR, None)
    if not isinstance(layout, tuple) or len(layout) != 3:
        raise ValueError("vocabulary parameter has no partition layout")
    real_size, start, end = (int(value) for value in layout)
    parameter = _materialize(parameter, dtype=_target_dtype(parameter, handle))
    target = parameter.data

    # A rank-local checkpoint can be copied directly; a global checkpoint contributes
    # only the overlap between the rank interval and the real vocabulary.
    if handle.shape == tuple(target.shape):
        _copy(target, handle.full(), parameter)
    else:
        if len(handle.shape) != target.ndim or handle.shape[1:] != tuple(target.shape[1:]):
            raise ValueError(
                f"loaded vocabulary tensor shape {handle.shape} does not match target "
                f"{tuple(target.shape)}"
            )
        if handle.shape[0] < real_size:
            raise ValueError(
                f"checkpoint vocabulary tensor {handle.name!r} has {handle.shape[0]} rows; "
                f"expected at least {real_size}"
            )
        target.zero_()
        copy_start = max(0, start)
        copy_end = min(end, real_size, handle.shape[0])
        if copy_end > copy_start:
            payload = handle.narrow(0, copy_start, copy_end - copy_start)
            _copy(target[copy_start - start : copy_end - start], payload, parameter)

    # Padding must be deterministic because logits may include the padded local rows.
    padding_start = max(0, real_size - start)
    if padding_start < int(target.shape[0]):
        target[padding_start:].zero_()
    _mark_loaded(parameter)


def fp8_weight_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
    *,
    module: nn.Module,
) -> None:
    """Load an FP8 weight shard and record whether checkpoint quantization is authoritative."""

    # Native E4M3 values retain their dtype and bypass the serving-dtype cast policy.
    offline = handle.dtype == torch.float8_e4m3fn
    serving_dtype = getattr(parameter, _DTYPE_ATTR, parameter.dtype)
    target_dtype = (
        torch.float8_e4m3fn
        if offline
        else serving_dtype
        if isinstance(serving_dtype, torch.dtype)
        else parameter.dtype
    )
    parameter = _materialize(parameter, dtype=target_dtype)
    set_skip_serving_cast(parameter, offline)

    # Ordinary shard plans and packed logical shards share the same materialization path.
    plan = get_shard_plan(parameter)
    if shard_id is None:
        if plan is not None and plan.slots:
            _copy_packed_tensor(parameter, handle, plan, preserve_dtype=offline)
        else:
            payload = (
                handle.full()
                if plan is None
                else _payload_for_plan(handle, plan, tuple(parameter.shape))
            )
            _copy(parameter.data, payload, parameter, preserve_dtype=offline)
    else:
        _copy_packed(parameter, handle, shard_id, preserve_dtype=offline)
        _mark_shard_loaded(parameter, _required_plan(parameter), shard_id)
    set_fp8_weight_loaded_offline(module, offline)
    _mark_loaded(parameter)


def fp8_scale_loader(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int | None = None,
    *,
    module: nn.Module,
) -> None:
    """Load the float32 scale associated with an FP8 weight or packed shard."""

    parameter = _materialize(parameter, dtype=torch.float32)
    if shard_id is None:
        plan = get_shard_plan(parameter)
        if plan is not None and plan.slots:
            _copy_packed_tensor(parameter, handle, plan, preserve_dtype=True, reshape_scale=True)
        else:
            target_shape = tuple(parameter.shape)
            payload = (
                handle.full() if plan is None else _payload_for_plan(handle, plan, target_shape)
            )
            payload = payload.reshape(-1, 1) if payload.ndim == 1 else payload
            _copy(parameter.data, payload, parameter, preserve_dtype=True)
    else:
        _copy_packed(parameter, handle, shard_id, preserve_dtype=True, reshape_scale=True)
        _mark_shard_loaded(parameter, _required_plan(parameter), shard_id)
    set_fp8_scale_loaded(module, True)
    _mark_loaded(parameter)


def _copy_packed_tensor(
    parameter: nn.Parameter,
    handle: WeightHandle,
    plan: ShardPlan,
    *,
    preserve_dtype: bool = False,
    reshape_scale: bool = False,
) -> None:
    """Install a complete merged tensor using each branch's checkpoint coordinates.

    A rank's packed projection contains an interval from every logical branch.
    Slicing the concatenated source once would select the wrong branches. Scale
    columns use the same coordinates as the weights they reconstruct.
    """

    axis = plan.shard_axis
    if axis is None:
        raise ValueError("packed parameter plan has no shard axis")
    slots = sorted(plan.slots.values(), key=lambda slot: slot.offset)
    extents = [slot.size * (1 if slot.shard.replicated else slot.shard.size) for slot in slots]
    local = handle.shape[axis] == parameter.shape[axis]
    if (not local and handle.shape[axis] != sum(extents)) or any(
        slot.shard.axis != axis for slot in slots
    ):
        raise ValueError("packed checkpoint does not match its logical branch extents")
    offset = 0
    for slot, extent in zip(slots, extents):
        start = (
            slot.offset
            if local
            else offset + (0 if slot.shard.replicated else slot.shard.rank * slot.size)
        )
        target = parameter.data.narrow(axis, slot.offset, slot.size)
        payload = handle.narrow(axis, start, slot.size)
        if reshape_scale and payload.ndim == 1:
            payload = payload.reshape(-1, 1)
        _copy(target, payload, parameter, preserve_dtype=preserve_dtype)
        offset += extent
    setattr(parameter, _SHARDS_ATTR, set(plan.slots))


def _copy_packed(
    parameter: nn.Parameter,
    handle: WeightHandle,
    shard_id: str | int,
    *,
    preserve_dtype: bool,
    reshape_scale: bool = False,
) -> None:
    """Copy one packed logical shard into its assignment slot."""

    plan = _required_plan(parameter)
    if plan.shard_axis is None:
        raise ValueError("packed parameter plan has no shard axis")
    slot = plan.slot_for(shard_id)
    if slot is None:
        raise ValueError(f"unknown packed checkpoint shard id {shard_id!r}")
    selection = [slice(None)] * parameter.ndim
    selection[plan.shard_axis] = slice(slot.offset, slot.offset + slot.size)
    target = parameter.data[tuple(selection)]
    payload = _payload_for_shard(handle, slot.shard, tuple(target.shape))
    if reshape_scale and payload.ndim == 1:
        payload = payload.reshape(-1, 1)
    _copy(target, payload, parameter, preserve_dtype=preserve_dtype)


def _payload_for_plan(
    handle: WeightHandle,
    plan: ShardPlan,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """Select the rank-local payload described by a parameter's shard plan."""

    return _payload_for_shard(handle, plan.shard, target_shape)


def _payload_for_shard(
    handle: WeightHandle, shard: Any, target_shape: tuple[int, ...]
) -> torch.Tensor:
    """Materialize a replicated value or the rank-local interval along one shard axis."""

    axis = int(shard.axis)
    if shard.replicated or int(shard.size) <= 1 or handle.shape[axis] == target_shape[axis]:
        return handle.full()
    source_extent = int(handle.shape[axis])
    if source_extent % int(shard.size):
        raise ValueError(
            f"checkpoint tensor {handle.name!r} dimension {axis}={source_extent} is not divisible "
            f"by tensor-parallel size {int(shard.size)}"
        )
    local = source_extent // int(shard.size)
    return handle.narrow(axis, int(shard.rank) * local, local)


def _required_plan(parameter: nn.Parameter) -> ShardPlan:
    """Return a parameter's assignment plan or reject missing sharding metadata."""

    plan = get_shard_plan(parameter)
    if plan is None:
        raise ValueError("partitioned parameter has no ShardPlan")
    return plan


def _target_dtype(parameter: nn.Parameter, handle: WeightHandle) -> torch.dtype:
    """Resolve the materialized dtype from tensor kind and serving cast policy."""

    if not _is_float_dtype(handle.dtype) or skip_serving_cast(parameter):
        return handle.dtype if skip_serving_cast(parameter) else parameter.dtype
    value = getattr(parameter, _DTYPE_ATTR, parameter.dtype)
    return value if isinstance(value, torch.dtype) else parameter.dtype


def _materialize(parameter: nn.Parameter, *, dtype: torch.dtype) -> nn.Parameter:
    """Allocate or retype a parameter and rebind every recorded owner alias."""

    device = getattr(parameter, _DEVICE_ATTR, None)
    target = parameter.device if device is None else torch.device(device)
    # Both numerical format and physical destination belong to the assignment.
    if not parameter.is_meta and parameter.dtype == dtype and parameter.device == target:
        return parameter

    # Retype resident storage without replacing the Parameter identity.
    if not parameter.is_meta:
        parameter.data = torch.empty_like(parameter.data, device=target, dtype=dtype)
        return parameter
    bindings = getattr(parameter, _BINDINGS_ATTR, ())
    if not bindings or not isinstance(device, str):
        raise RuntimeError("meta parameter has no materialization owner")

    # Meta parameters need a concrete shared replacement installed into every owner.
    replacement = nn.Parameter(
        torch.empty(tuple(parameter.shape), device=device, dtype=dtype),
        requires_grad=parameter.requires_grad,
    )
    copy_parameter_loader_state(parameter, replacement)
    for owner_ref, name in bindings:
        owner = owner_ref() if callable(owner_ref) else None
        if not isinstance(owner, nn.Module) or not isinstance(name, str):
            raise RuntimeError("meta parameter has an invalid materialization binding")
        owner._parameters[name] = replacement
    return replacement


def _current_parameter(parameter: nn.Parameter) -> nn.Parameter:
    """Resolve the concrete replacement currently held by any recorded owner."""

    bindings = getattr(parameter, _BINDINGS_ATTR, ())
    for owner_ref, name in bindings:
        owner = owner_ref() if callable(owner_ref) else None
        if isinstance(owner, nn.Module) and isinstance(name, str):
            current = owner._parameters.get(name)
            if isinstance(current, nn.Parameter):
                return current
    return parameter


def _copy(
    target: torch.Tensor,
    payload: torch.Tensor,
    parameter: nn.Parameter,
    *,
    preserve_dtype: bool = False,
) -> None:
    """Validate shape, apply serving dtype policy, and copy a checkpoint payload."""

    value = payload
    if value.is_floating_point() and not preserve_dtype and not skip_serving_cast(parameter):
        value = value.to(dtype=target.dtype)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"loaded tensor shape {tuple(value.shape)} does not match target {tuple(target.shape)}"
        )
    target.copy_(value)


def _mark_loaded(parameter: nn.Parameter) -> None:
    """Mark a parameter as having received checkpoint data."""

    setattr(parameter, _LOADED_ATTR, True)


def _mark_shard_loaded(
    parameter: nn.Parameter,
    plan: ShardPlan,
    shard_id: str | int,
) -> None:
    """Record one packed shard and mark the destination parameter loaded."""

    key: str | int = shard_id
    if isinstance(shard_id, str):
        key = plan.mode.shard_key_to_index.get(shard_id, shard_id)
    loaded = set(getattr(parameter, _SHARDS_ATTR, set()))
    loaded.add(key)
    setattr(parameter, _SHARDS_ATTR, loaded)
    _mark_loaded(parameter)


def _is_float_dtype(dtype: torch.dtype) -> bool:
    """Return whether a dtype participates in floating-point serving casts."""

    return dtype in {
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }
