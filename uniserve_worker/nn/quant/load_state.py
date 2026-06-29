"""Typed loader/quant sidecar for parameters and modules.

The weight-loading and quantization seams carry a handful of loose flags on
``nn.Parameter`` / ``nn.Module`` objects across the offline-load and
``process_weights_after_loading`` steps.  This module is the single owner of
those attribute names and the only place that reads/writes them, so the
producers (linear/parallel/vocab/fp8) and consumers (native checkpoint loader,
serving-dtype fold) share one typed surface instead of scattered
``setattr``/``getattr`` calls.

``weight_loader`` deliberately stays a plain attribute on the parameter: it is
the public loader contract (callers invoke ``param.weight_loader(...)`` and
``copy_load_state`` carries it forward), so it is read/written here by name
rather than relocated off the object.
"""
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
    'set_allow_shape_mismatch',
    'allow_shape_mismatch',
    'get_weight_loader',
    'set_weight_loader',
    'has_weight_loader',
    'init_fp8_phase',
    'set_fp8_weight_loaded_offline',
    'fp8_weight_loaded_offline',
    'set_fp8_scale_loaded',
    'fp8_scale_loaded',
    'fp8_load_phase',
    'copy_load_state',
    'capture_tensor_policy',
    'restore_tensor_policy',
]

# Parameters are reached through ``nn.Module`` attribute access, which is typed
# ``Tensor | Module``; accept that union (aliased to ``Any``) for the attribute
# helpers so call sites need no ``# type: ignore`` casts.
ParamLike = Any

# Loader-state flags carried on params/modules.  These string names are the
# single source of truth for the loose attrs propagated by the shared layers and
# the native checkpoint loader (``loader/transformers.py``).
_ATTR_WEIGHT_LOADER = "weight_loader"
_ATTR_OPTIONAL_CHECKPOINT = "_uniserve_optional_checkpoint"
_ATTR_SKIP_SERVING_CAST = "_uniserve_skip_serving_cast"
_ATTR_ALLOW_SHAPE_MISMATCH = "_uniserve_allow_shape_mismatch"
_ATTR_WEIGHT_LOADED_OFFLINE = "_fp8_weight_loaded_offline"
_ATTR_SCALE_LOADED = "_fp8_scale_loaded"

# Attrs ``copy_load_state`` carries onto a retyped parameter: the loader hook and
# the typed sharding sidecar (``_uniserve_shard`` is copied directly from
# parallel.py's accessor).
_LOADER_COPY_ATTRS = (_ATTR_WEIGHT_LOADER,)

# Policy flags the native checkpoint loader re-attaches after materializing a
# tensor (the loader hook plus the per-tensor checkpoint-policy flags).
_TENSOR_POLICY_ATTRS = (
    _ATTR_WEIGHT_LOADER,
    _ATTR_OPTIONAL_CHECKPOINT,
    _ATTR_SKIP_SERVING_CAST,
    _ATTR_ALLOW_SHAPE_MISMATCH,
)


class Fp8LoadPhase(Enum):
    """FP8 module load lifecycle: each weight loader hook advances this so the
    "scale before finalize" precondition checked in
    ``process_weights_after_loading`` is structural rather than ad-hoc."""

    RAW = "raw"
    WEIGHT_LOADED = "weight_loaded"
    SCALE_LOADED = "scale_loaded"
    FINALIZED = "finalized"


# --- parameter policy flags -------------------------------------------------


def set_optional_checkpoint(param: ParamLike, value: bool = True) -> None:
    setattr(param, _ATTR_OPTIONAL_CHECKPOINT, bool(value))


def is_optional_checkpoint(param: ParamLike) -> bool:
    return bool(getattr(param, _ATTR_OPTIONAL_CHECKPOINT, False))


def set_skip_serving_cast(param: ParamLike, value: bool = True) -> None:
    setattr(param, _ATTR_SKIP_SERVING_CAST, bool(value))


def skip_serving_cast(param: ParamLike) -> bool:
    return bool(getattr(param, _ATTR_SKIP_SERVING_CAST, False))


def set_allow_shape_mismatch(param: ParamLike, value: bool = True) -> None:
    setattr(param, _ATTR_ALLOW_SHAPE_MISMATCH, bool(value))


def allow_shape_mismatch(param: ParamLike) -> bool:
    return bool(getattr(param, _ATTR_ALLOW_SHAPE_MISMATCH, False))


# --- weight-loader hook -----------------------------------------------------


def get_weight_loader(param: ParamLike):
    return getattr(param, _ATTR_WEIGHT_LOADER, None)


def set_weight_loader(param: ParamLike, loader) -> None:
    setattr(param, _ATTR_WEIGHT_LOADER, loader)


def has_weight_loader(param: ParamLike) -> bool:
    return callable(getattr(param, _ATTR_WEIGHT_LOADER, None))


# --- FP8 module lifecycle ---------------------------------------------------


def init_fp8_phase(module: torch.nn.Module) -> None:
    setattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, False)
    setattr(module, _ATTR_SCALE_LOADED, False)


def set_fp8_weight_loaded_offline(module: torch.nn.Module, value: bool) -> None:
    setattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, bool(value))


def fp8_weight_loaded_offline(module: torch.nn.Module) -> bool:
    return bool(getattr(module, _ATTR_WEIGHT_LOADED_OFFLINE, False))


def set_fp8_scale_loaded(module: torch.nn.Module, value: bool = True) -> None:
    setattr(module, _ATTR_SCALE_LOADED, bool(value))


def fp8_scale_loaded(module: torch.nn.Module) -> bool:
    return bool(getattr(module, _ATTR_SCALE_LOADED, False))


def fp8_load_phase(module: torch.nn.Module) -> Fp8LoadPhase:
    """Derive the load phase from the two recorded booleans.

    ``RAW`` -> nothing loaded; ``WEIGHT_LOADED`` -> an offline fp8 weight was
    placed but no scale yet; ``SCALE_LOADED`` -> the matching scale is present
    (the precondition for finalize).  ``FINALIZED`` is not separately tracked
    here because the finalize step is idempotent on an already-fp8 weight.
    """
    if fp8_scale_loaded(module):
        return Fp8LoadPhase.SCALE_LOADED
    if fp8_weight_loaded_offline(module):
        return Fp8LoadPhase.WEIGHT_LOADED
    return Fp8LoadPhase.RAW


# --- sidecar copy -----------------------------------------------------------


def copy_load_state(src: ParamLike, dst: ParamLike) -> None:
    """Carry the loader hook (and the typed shard sidecar) onto a retyped param.

    Used when a loader replaces a parameter object in place (e.g. retyping a
    dense weight to fp8): the new parameter must keep the same ``weight_loader``
    and ``_uniserve_shard`` plan so subsequent shard placement still works.
    """
    from ..placement import get_shard_plan, set_shard_plan

    for name in _LOADER_COPY_ATTRS:
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))
    plan = get_shard_plan(src)
    if plan is not None:
        set_shard_plan(dst, plan)


def capture_tensor_policy(tensor: ParamLike) -> dict[str, object]:
    """Snapshot the loader hook + checkpoint-policy flags before a tensor is
    re-materialized, so they can be restored onto the replacement tensor."""
    captured = {name: getattr(tensor, name) for name in _TENSOR_POLICY_ATTRS if hasattr(tensor, name)}
    from ..placement import get_shard_plan

    plan = get_shard_plan(tensor)
    if plan is not None:
        captured["_uniserve_shard"] = plan
    return captured


def restore_tensor_policy(tensor: ParamLike, captured: dict[str, object]) -> None:
    for name, value in captured.items():
        if name == "_uniserve_shard":
            from ..placement import set_shard_plan

            set_shard_plan(tensor, value)
        else:
            setattr(tensor, name, value)
