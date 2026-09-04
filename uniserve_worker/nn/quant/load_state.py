"""Typed tensor policies used by the system checkpoint loader and quantizer."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

__all__ = [
    'ParamLike',
    'Fp8LoadPhase',
    'set_optional_checkpoint',
    'is_optional_checkpoint',
    'set_skip_serving_cast',
    'skip_serving_cast',
    'init_fp8_phase',
    'set_fp8_weight_loaded_offline',
    'fp8_weight_loaded_offline',
    'set_fp8_scale_loaded',
    'fp8_scale_loaded',
    'fp8_load_phase',
    'copy_tensor_policy',
]

# Loader policy attributes may be attached to parameters, tensors, or modules.
ParamLike = Any

# Attribute names form the shared contract between materialization and execution.
_ATTR_OPTIONAL_CHECKPOINT = "_uniserve_optional_checkpoint"
_ATTR_SKIP_SERVING_CAST = "_uniserve_skip_serving_cast"
_ATTR_WEIGHT_LOADED_OFFLINE = "_fp8_weight_loaded_offline"
_ATTR_SCALE_LOADED = "_fp8_scale_loaded"

# Policy flags the native checkpoint loader re-attaches after materializing a tensor.
_TENSOR_POLICY_ATTRS = (
    _ATTR_OPTIONAL_CHECKPOINT,
    _ATTR_SKIP_SERVING_CAST,
)


class Fp8LoadPhase(Enum):
    """FP8 checkpoint materialization phase derived from recorded load state."""

    RAW = "raw"
    WEIGHT_LOADED = "weight_loaded"
    SCALE_LOADED = "scale_loaded"


def set_optional_checkpoint(param: ParamLike, value: bool = True) -> None:
    """Mark whether a parameter may be absent from an otherwise complete checkpoint."""

    setattr(param, _ATTR_OPTIONAL_CHECKPOINT, bool(value))


def is_optional_checkpoint(param: ParamLike) -> bool:
    """Read the optional-checkpoint policy attached to a parameter."""

    return bool(getattr(param, _ATTR_OPTIONAL_CHECKPOINT, False))


def set_skip_serving_cast(param: ParamLike, value: bool = True) -> None:
    """Preserve a parameter's loaded dtype across serving materialization."""

    setattr(param, _ATTR_SKIP_SERVING_CAST, bool(value))


def skip_serving_cast(param: ParamLike) -> bool:
    """Read whether serving materialization must preserve the loaded dtype."""

    return bool(getattr(param, _ATTR_SKIP_SERVING_CAST, False))


def init_fp8_phase(module: torch.nn.Module) -> None:
    """Initialize an FP8 module before checkpoint weight and scale loading."""

    setattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, False)
    setattr(module, _ATTR_SCALE_LOADED, False)


def set_fp8_weight_loaded_offline(module: torch.nn.Module, value: bool) -> None:
    """Record whether a checkpoint supplied an already-quantized FP8 weight."""

    setattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, bool(value))


def fp8_weight_loaded_offline(module: torch.nn.Module) -> bool:
    """Read whether an FP8 weight arrived in quantized checkpoint form."""

    return bool(getattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, False))


def set_fp8_scale_loaded(module: torch.nn.Module, value: bool = True) -> None:
    """Record whether the checkpoint supplied the FP8 weight scale."""

    setattr(module, _ATTR_SCALE_LOADED, bool(value))


def fp8_scale_loaded(module: torch.nn.Module) -> bool:
    """Read whether an FP8 weight scale has been materialized."""

    return bool(getattr(module, _ATTR_SCALE_LOADED, False))


def fp8_load_phase(module: torch.nn.Module) -> Fp8LoadPhase:
    """Derive the current checkpoint materialization phase."""
    if fp8_scale_loaded(module):
        return Fp8LoadPhase.SCALE_LOADED
    if fp8_weight_loaded_offline(module):
        return Fp8LoadPhase.WEIGHT_LOADED
    return Fp8LoadPhase.RAW


def copy_tensor_policy(src: ParamLike, dst: ParamLike) -> None:
    """Carry immutable checkpoint policy and sharding onto a retyped parameter."""
    from ..placement import get_shard_plan, set_shard_plan

    for name in _TENSOR_POLICY_ATTRS:
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))
    plan = get_shard_plan(src)
    if plan is not None:
        set_shard_plan(dst, plan)
