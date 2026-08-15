"""Load-time parameter sharding and modality-tower device placement."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn as nn

from .mesh import DeviceMesh, TensorParallelSpec

__all__ = [
    'WeightMode',
    'ShardSpec',
    'ShardSlot',
    'ShardPlan',
    'shard_spec',
    'get_shard_plan',
    'set_shard_plan',
    'place_partitioned_tensor',
    # tower-axis module placement
    'set_tower_coord',
    'get_tower_coord',
    'place_towers',
]


class WeightMode(Enum):
    """How a parameter packs (or does not pack) named shards along one axis."""

    VANILLA = auto()
    FUSED_QKV_LINEAR = auto()
    FUSED_GATE_UP_LINEAR = auto()

    @property
    def shard_keys(self) -> tuple[str, ...]:
        if self is WeightMode.FUSED_QKV_LINEAR:
            return ("q", "k", "v")
        if self is WeightMode.FUSED_GATE_UP_LINEAR:
            return ("gate", "up")
        return ()

    @property
    def shard_key_to_index(self) -> dict[str, int]:
        return {key: idx for idx, key in enumerate(self.shard_keys)}


@dataclass(frozen=True)
class ShardSpec:
    """Resolved split of one tensor across tensor-parallel coordinates."""

    axis: int
    rank: int
    size: int
    replicated: bool = False

    def narrow(self, loaded: torch.Tensor, target_dim: int) -> torch.Tensor:
        if self.replicated or self.size <= 1:
            return loaded
        dim = int(loaded.shape[self.axis])
        if dim == target_dim:
            return loaded
        if dim % self.size != 0:
            raise ValueError(f"loaded tensor dim {dim} is not divisible by size {self.size}")
        per_rank = dim // self.size
        return loaded.narrow(self.axis, self.rank * per_rank, per_rank)


@dataclass(frozen=True)
class ShardSlot:
    """One named shard packed within a fused/merged parameter axis."""

    offset: int
    size: int
    spec: ShardSpec


@dataclass
class ShardPlan:
    """Resolved per-parameter load layout, attached as ``_uniserve_shard``.

    ``spec`` is the whole-tensor placement. When the parameter packs several
    named shards along one axis (merged / fused QKV linears), ``shard_axis``
    names that axis and ``slots`` maps each shard id to its slice + per-shard
    placement; ``mode`` resolves string shard ids (``"q"``) to slot indices.
    """

    spec: ShardSpec
    mode: WeightMode = WeightMode.VANILLA
    shard_axis: int | None = None
    slots: dict[int | str, ShardSlot] = field(default_factory=dict)

    def slot_for(self, shard_id: int | str) -> ShardSlot | None:
        key: int | str = shard_id
        if isinstance(shard_id, str):
            key = self.mode.shard_key_to_index.get(shard_id, shard_id)
        return self.slots.get(key)


def shard_spec(axis: int, parallel: TensorParallelSpec, *, replicated: bool = False) -> ShardSpec:
    """Resolve a tensor-parallel :class:`ShardSpec` for one parameter dimension."""
    return ShardSpec(
        axis=int(axis),
        rank=int(parallel.rank),
        size=int(parallel.size),
        replicated=bool(replicated),
    )


_SHARD_PLAN_ATTR = "_uniserve_shard"


def get_shard_plan(param: nn.Parameter) -> ShardPlan | None:
    return getattr(param, _SHARD_PLAN_ATTR, None)


def set_shard_plan(param: nn.Parameter, plan: ShardPlan) -> None:
    setattr(param, _SHARD_PLAN_ATTR, plan)


def place_partitioned_tensor(
    param: nn.Parameter,
    target: torch.Tensor,
    loaded: torch.Tensor,
    *,
    shard_id: int | str | None = None,
) -> None:
    """Narrow ``loaded`` for this rank and copy it into ``target``.

    The single placement helper shared by the dense and FP8 weight loaders, so
    sharding semantics live in one place. ``target`` is the destination storage
    (usually ``param.data``); when ``shard_id`` selects a named shard the
    destination is the corresponding slice of ``target``.
    """
    plan = get_shard_plan(param)
    spec: ShardSpec | None = None
    if plan is not None and shard_id is not None and plan.shard_axis is not None:
        slot = plan.slot_for(shard_id)
        if slot is None:
            raise ValueError(f"unknown shard id {shard_id!r}")
        slices = [slice(None)] * target.ndim
        slices[plan.shard_axis] = slice(slot.offset, slot.offset + slot.size)
        target = target[tuple(slices)]
        spec = slot.spec
    elif plan is not None:
        spec = plan.spec
    if spec is not None:
        loaded = spec.narrow(loaded, int(target.shape[spec.axis]))
    if tuple(target.shape) != tuple(loaded.shape):
        raise ValueError(f"loaded tensor shape {tuple(loaded.shape)} != target {tuple(target.shape)}")
    target.copy_(loaded)


_TOWER_COORD_ATTR = "_uniserve_tower_coord"


def set_tower_coord(module: nn.Module, coord: int) -> nn.Module:
    """Assign a module subtree to one modality-tower coordinate."""
    setattr(module, _TOWER_COORD_ATTR, int(coord))
    return module


def get_tower_coord(module: nn.Module) -> int | None:
    """Return a module's tower coordinate, or ``None`` for shared modules."""
    coord = getattr(module, _TOWER_COORD_ATTR, None)
    return int(coord) if coord is not None else None


def place_towers(model: nn.Module, mesh: DeviceMesh) -> None:
    """Move each assigned modality-tower subtree to its coordinate's device."""
    axis = mesh.axis("tower") if mesh is not None else None
    if axis is None or int(axis.size) <= 1 or axis.transport is None:
        return
    device_for = getattr(axis.transport, "device", None)
    if not callable(device_for):
        raise RuntimeError("tower transport does not expose its local devices")
    primary = int(axis.coord)
    for module in model.modules():
        coord = get_tower_coord(module)
        if coord is None or coord == primary:
            continue
        module.to(device_for(coord))
