"""Load-time parameter sharding and modality-tower device placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch.nn as nn

from .mesh import DeviceMesh, TensorParallel

__all__ = [
    'WeightMode',
    'Shard',
    'ShardSlot',
    'ShardPlan',
    'shard_for',
    'get_shard_plan',
    'set_shard_plan',
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
        """List symbolic checkpoint shard ids in their packed parameter order."""

        if self is WeightMode.FUSED_QKV_LINEAR:
            return ("q", "k", "v")
        if self is WeightMode.FUSED_GATE_UP_LINEAR:
            return ("gate", "up")
        return ()

    @property
    def shard_key_to_index(self) -> dict[str, int]:
        """Map symbolic checkpoint shard ids to packed slot indices."""

        return {key: idx for idx, key in enumerate(self.shard_keys)}


@dataclass(frozen=True)
class Shard:
    """Resolved split of one tensor across tensor-parallel coordinates."""

    axis: int
    rank: int
    size: int
    replicated: bool = False


@dataclass(frozen=True)
class ShardSlot:
    """One named shard packed within a fused/merged parameter axis."""

    offset: int
    size: int
    shard: Shard


@dataclass
class ShardPlan:
    """Resolved per-parameter load layout, attached as ``_uniserve_shard``.

    ``shard`` is the whole-tensor placement. When the parameter packs several
    named shards along one axis (merged / fused QKV linears), ``shard_axis``
    names that axis and ``slots`` maps each shard id to its slice + per-shard
    placement; ``mode`` resolves string shard ids (``"q"``) to slot indices.
    """

    shard: Shard
    mode: WeightMode = WeightMode.VANILLA
    shard_axis: int | None = None
    slots: dict[int | str, ShardSlot] = field(default_factory=dict)

    def slot_for(self, shard_id: int | str) -> ShardSlot | None:
        """Resolve an integer or symbolic shard id to its destination parameter slice."""

        key: int | str = shard_id
        if isinstance(shard_id, str):
            key = self.mode.shard_key_to_index.get(shard_id, shard_id)
        return self.slots.get(key)


def shard_for(axis: int, parallel: TensorParallel, *, replicated: bool = False) -> Shard:
    """Resolve a tensor-parallel :class:`Shard` for one parameter dimension."""
    return Shard(
        axis=int(axis),
        rank=int(parallel.rank),
        size=int(parallel.size),
        replicated=bool(replicated),
    )


_SHARD_PLAN_ATTR = "_uniserve_shard"


def get_shard_plan(param: nn.Parameter) -> ShardPlan | None:
    """Read the load-time sharding plan attached to a parameter, if present."""

    return getattr(param, _SHARD_PLAN_ATTR, None)


def set_shard_plan(param: nn.Parameter, plan: ShardPlan) -> None:
    """Attach the load-time sharding plan consumed by checkpoint weight loaders."""

    setattr(param, _SHARD_PLAN_ATTR, plan)


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
