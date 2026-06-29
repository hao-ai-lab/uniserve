"""Placements and resharding — the unified communication vocabulary.

Two cooperating layers live here:

* **Activation placement + reshard.** :class:`Placement` (``Replicate`` /
  ``Shard`` / ``Partial`` / ``Pinned``) describes, per mesh axis, how an
  activation (or a whole tensor / the KV cache) is distributed. :func:`reshard`
  is the *single* site that turns one placement into another, choosing the
  mechanism from a closed per-axis transition table and dispatching it to the
  axis transport.

* **Load-time shard sidecar.** :class:`ShardPlan` (with :class:`ShardSpec` /
  :class:`ShardSlot` / :class:`WeightMode`) is the resolved per-parameter layout
  a checkpoint loader narrows into. It carries the resolved ``(tensor-dim, rank,
  size)`` so :func:`place_partitioned_tensor` needs no mesh, and is the proven
  "declarative shard layout, decoupled from communication" both reference
  systems keep. It is the load-time projection of a ``Shard`` placement.

Pure layer-2: depends only on torch and :mod:`uniserve_worker.nn.mesh`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .mesh import DeviceMesh, MeshAxis, ReduceOp

__all__ = [
    # activation placement + reshard
    'Placement',
    'Replicate',
    'Shard',
    'Partial',
    'Pinned',
    'Sharding',
    'reshard',
    'Region',
    'Router',
    # load-time shard sidecar
    'TensorParallelMode',
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


# ---------------------------------------------------------------------------
# Activation placement vocabulary
# ---------------------------------------------------------------------------


class Placement:
    """How a tensor is distributed along ONE mesh axis (topology-free intent)."""

    axis: str


@dataclass(frozen=True)
class Replicate(Placement):
    """Replicate the tensor along ``axis`` to every coordinate."""

    axis: str


@dataclass(frozen=True)
class Shard(Placement):
    """Shard ``dim`` across coordinates of ``axis``."""

    axis: str
    dim: int


@dataclass(frozen=True)
class Partial(Placement):
    """Partial reduction along ``axis`` before a subsequent reshard."""

    axis: str
    op: str = ReduceOp.SUM


@dataclass(frozen=True)
class Pinned(Placement):
    """The tensor exists only at one coordinate of ``axis`` (pipeline stage /
    modality tower / heterogeneous subtree)."""

    axis: str
    coord: int


@dataclass(frozen=True)
class Sharding:
    """A tensor's placement across the mesh: at most one :class:`Placement` per
    axis; an axis not named is implicitly ``Replicate`` on that axis."""

    placements: tuple[Placement, ...] = ()

    def on(self, axis: str) -> Placement | None:
        for p in self.placements:
            if p.axis == axis:
                return p
        return None

    def axes(self) -> tuple[str, ...]:
        return tuple(p.axis for p in self.placements)


def _reshard_axis(
    t: torch.Tensor,
    src: Placement,
    dst: Placement,
    axis: MeshAxis,
    *,
    owner: str,
) -> torch.Tensor:
    transport = axis.transport
    if transport is None:
        raise RuntimeError(
            f"reshard[{owner}] needs a transport to communicate on non-trivial axis {axis.name!r}"
        )
    if type(src) is type(dst) and src == dst:
        return t
    # Partial -> Replicate : all-reduce
    if isinstance(src, Partial) and isinstance(dst, Replicate):
        from ..ops import tp_all_reduce

        return tp_all_reduce(t, src.op, axis=axis)
    # Shard -> Replicate : all-gather
    if isinstance(src, Shard) and isinstance(dst, Replicate):
        return transport.all_gather(t, src.dim)
    # Partial -> Shard : reduce-scatter
    if isinstance(src, Partial) and isinstance(dst, Shard):
        return transport.reduce_scatter(t, dst.dim, src.op)
    # Shard(i) -> Shard(j) : all-to-all
    if isinstance(src, Shard) and isinstance(dst, Shard):
        if src.dim == dst.dim:
            return t
        return transport.all_to_all(t, in_dim=src.dim, out_dim=dst.dim)
    # Replicate -> Shard : local slice (no communication)
    if isinstance(src, Replicate) and isinstance(dst, Shard):
        pieces = torch.chunk(t, transport.size, dim=dst.dim)
        return pieces[transport.coord].contiguous()
    # Pinned(a) -> Pinned(b) : point-to-point / peer copy
    if isinstance(src, Pinned) and isinstance(dst, Pinned):
        if src.coord == dst.coord:
            return t
        return transport.copy_to(t, coord=dst.coord)
    # Pinned(c) -> Replicate : broadcast from c
    if isinstance(src, Pinned) and isinstance(dst, Replicate):
        return transport.broadcast(t, src=src.coord)
    raise NotImplementedError(
        f"reshard[{owner}] has no rule for {type(src).__name__}->{type(dst).__name__} "
        f"on axis {axis.name!r}"
    )


def reshard(
    t: torch.Tensor,
    src: Sharding,
    dst: Sharding,
    mesh: DeviceMesh,
    *,
    owner: str = "reshard",
) -> torch.Tensor:
    """Transform ``t`` from placement ``src`` to ``dst``, axis by axis.

    The only site that issues a collective or a peer copy. Axes are orthogonal,
    so a multi-axis transition composes from the per-axis rule; a trivial axis
    (size 1) is skipped, so the single-device default is a pure no-op.
    """
    out = t
    seen: set[str] = set()
    for p in (*src.placements, *dst.placements):
        if p.axis in seen:
            continue
        seen.add(p.axis)
        if mesh.is_trivial(p.axis):
            continue
        ax = mesh.axis(p.axis)
        if ax is None:
            continue
        s = src.on(p.axis) or Replicate(p.axis)
        d = dst.on(p.axis) or Replicate(p.axis)
        out = _reshard_axis(out, s, d, ax, owner=owner)
    return out


# ---------------------------------------------------------------------------
# Regions and routers (declared boundaries; routing axes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A span of the model that needs its activation at a declared placement.

    The boundary into a region is where ``reshard`` runs: ``reshard(activation,
    producer_out, region.in_layout, mesh)``. Generalizes a fixed scatter-mode
    enum to an arbitrary :class:`Sharding`.
    """

    name: str
    in_layout: Sharding
    out_layout: Sharding


@runtime_checkable
class Router(Protocol):
    """The activation-side dual of ``Pinned`` params for routing axes (``ep`` /
    ``tower``): map each token to the coordinate whose parameters apply, run it
    there, and bring the result back. ``ep`` realizes dispatch/combine as an
    all-to-all; ``tower`` as a deterministic modality split + peer copy."""

    def dispatch(self, tokens: torch.Tensor, mesh: DeviceMesh, axis: str) -> Any: ...
    def combine(self, results: Any, mesh: DeviceMesh, axis: str) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Load-time shard sidecar (resolved per-parameter layout)
# ---------------------------------------------------------------------------


class TensorParallelMode(Enum):
    """Which axis a tensor-parallel linear shards along."""

    COLUMN = auto()
    ROW = auto()

    def split_dim(self) -> int:
        return 0 if self is TensorParallelMode.COLUMN else 1

    def flip(self) -> "TensorParallelMode":
        return TensorParallelMode.ROW if self is TensorParallelMode.COLUMN else TensorParallelMode.COLUMN


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
    """Resolved split of one tensor across a mesh axis's coordinates.

    ``axis`` is the *tensor* split dimension; ``rank``/``size`` the resolved mesh
    coordinate/extent (baked at construction so the loader needs no mesh);
    ``replicated`` marks a tensor every coordinate holds whole. This is the
    load-time projection of a ``Shard`` placement.
    """

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


def shard_spec(axis: int, mesh: DeviceMesh, *, mesh_axis: str = "tp", replicated: bool = False) -> ShardSpec:
    """Resolve a :class:`ShardSpec` for tensor dimension ``axis`` from ``mesh``."""
    return ShardSpec(
        axis=int(axis),
        rank=int(mesh.coord(mesh_axis)),
        size=int(mesh.size(mesh_axis)),
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


# ---------------------------------------------------------------------------
# Tower-axis module placement (Pinned modules + the generic placement pass)
# ---------------------------------------------------------------------------

# A module subtree tagged with this attribute is ``Pinned(tower, coord)``: its
# parameters live only at tower coordinate ``coord``. The tag is the module-level
# analog of a ``Pinned`` activation/tensor placement (a whole stack of modules,
# not a single tensor), consumed by :func:`place_towers`.
_TOWER_COORD_ATTR = "_uniserve_tower_coord"


def set_tower_coord(module: nn.Module, coord: int) -> nn.Module:
    """Tag a module subtree as ``Pinned(tower, coord)`` and return it.

    This is how a model declares which modality tower a module belongs to,
    replacing per-model imperative ``.to(gen_device)`` placement: the model tags
    its generation-tower modules at construction (where it already knows its
    architecture), and :func:`place_towers` is the single, model-agnostic pass
    that realizes the placement against whatever ``tower`` axis the mesh carries.
    """
    setattr(module, _TOWER_COORD_ATTR, int(coord))
    return module


def get_tower_coord(module: nn.Module) -> int | None:
    """Return a module's tagged tower coordinate, or ``None`` if untagged.

    An untagged module is implicitly on the primary (shared) coordinate -- the
    ``Replicate``-on-the-tower-axis default for shared modules.
    """
    coord = getattr(module, _TOWER_COORD_ATTR, None)
    return int(coord) if coord is not None else None


def place_towers(model: nn.Module, mesh: DeviceMesh) -> None:
    """Place every ``Pinned(tower)`` module subtree on its coordinate's device.

    The single, model-agnostic placement pass for the ``tower`` axis. A module
    tagged ``set_tower_coord(m, c)`` is moved to the tower transport's device for
    coordinate ``c``; untagged modules stay on the model's primary device (the
    shared/understanding coordinate).

    A trivial or absent ``tower`` axis is a no-op. For a cross-process tower the
    transport exposes no in-process device map; placement is then a property of
    which params each worker loads (a coordinate not owned by this worker is simply
    not materialized), so this pass returns without moving anything.
    """
    axis = mesh.axis("tower") if mesh is not None else None
    if axis is None or int(axis.size) <= 1 or axis.transport is None:
        return
    device_for = getattr(axis.transport, "device", None)
    if not callable(device_for):
        # Cross-process tower: no in-process peer device to move to; the
        # per-worker load owns which coordinate's params exist here.
        return
    primary = int(axis.coord)
    for module in model.modules():
        coord = get_tower_coord(module)
        if coord is None or coord == primary:
            continue
        module.to(device_for(coord))
