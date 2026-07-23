"""Device mesh and per-axis transports — the unified parallelism topology.

This is the single topology object for every parallelism axis (``tp``, ``cp``,
``dp``, ``pp``, ``ep``, ``tower``). Each axis carries a *transport* describing how its coordinates communicate:

* :class:`CollectiveTransport` — a ``torch.distributed`` process group; coordinates
  are ranks in separate processes (tensor parallelism today).
* :class:`LocalP2PTransport` — coordinates are CUDA devices in *this* process;
  "communication" is a device-to-device copy plus CUDA-event barriers (the
  single-process modality/tower split).

A trivial axis (``size <= 1``) never communicates, so the default single-device
mesh makes every collective a no-op and the path is byte-identical to a
single-rank worker. ``reshard`` (see :mod:`uniserve_worker.nn.placement`) is the
only caller of the transport surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

__all__ = [
    'divide',
    'ReduceOp',
    'AxisTransport',
    'BroadcastTransport',
    'CollectiveAxisTransport',
    'PeerAxisTransport',
    'CollectiveTransport',
    'LocalP2PTransport',
    'DataPlaneTowerTransport',
    'MeshAxis',
    'DeviceMesh',
    'TensorParallelSpec',
]


def divide(numerator: int, denominator: int) -> int:
    """Exact integer division, raising on a non-divisible or non-positive denom."""
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


class ReduceOp:
    """Reduction op identifiers for ``Partial`` placements (transport-agnostic)."""

    SUM = "sum"
    MAX = "max"
    MIN = "min"


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


@runtime_checkable
class AxisTransport(Protocol):
    """Coordinate metadata common to every axis transport."""

    @property
    def size(self) -> int: ...
    @property
    def coord(self) -> int: ...


@runtime_checkable
class CollectiveAxisTransport(AxisTransport, Protocol):
    """Collective operations implemented by a distributed process group."""

    def all_reduce(self, t: torch.Tensor, op: str = ReduceOp.SUM) -> torch.Tensor: ...
    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor: ...
    def reduce_scatter(self, t: torch.Tensor, dim: int, op: str = ReduceOp.SUM) -> torch.Tensor: ...
    def all_to_all(self, t: torch.Tensor, *, in_dim: int, out_dim: int) -> torch.Tensor: ...


@runtime_checkable
class PeerAxisTransport(AxisTransport, Protocol):
    """Point-to-point movement implemented by an in-process or staged peer axis."""

    def copy_to(self, t: torch.Tensor, *, coord: int, non_blocking: bool = True) -> torch.Tensor: ...
    def record_ready(self, coord: int | None = None) -> Any | None: ...
    def wait_ready(self, event: Any | None, coord: int | None = None) -> None: ...


@runtime_checkable
class BroadcastTransport(AxisTransport, Protocol):
    """One-to-all movement for a transport that defines broadcast semantics."""

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor: ...


def _torch_reduce_op(op: str):
    table = {
        ReduceOp.SUM: torch.distributed.ReduceOp.SUM,
        ReduceOp.MAX: torch.distributed.ReduceOp.MAX,
        ReduceOp.MIN: torch.distributed.ReduceOp.MIN,
    }
    try:
        return table[op]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported reduce op {op!r}") from exc


@dataclass(frozen=True)
class CollectiveTransport:
    """A ``torch.distributed`` process group bound to one mesh axis.

    ``group`` is ``None`` for the default world group. ``size``/``coord`` are this
    axis's world size and this process's coordinate within it (== the rank inside
    ``group``). Ports the world-size validation the standalone TP collectives
    used so a mismatched default world still fails loudly.
    """

    axis: str
    _size: int
    _coord: int
    group: Any = None

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def coord(self) -> int:
        return int(self._coord)

    def _require(self) -> Any:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                f"mesh axis {self.axis!r} size>1 requires an initialized "
                "torch.distributed collective"
            )
        if self.group is None:
            world = int(torch.distributed.get_world_size())
            if world != self.size:
                raise RuntimeError(
                    f"mesh axis {self.axis!r} requires an explicit process group when "
                    f"the default world size ({world}) differs from axis size ({self.size})"
                )
        return self.group

    def all_reduce(self, t: torch.Tensor, op: str = ReduceOp.SUM) -> torch.Tensor:
        group = self._require()
        torch.distributed.all_reduce(t, op=_torch_reduce_op(op), group=group)
        return t

    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        group = self._require()
        chunks = [torch.empty_like(t) for _ in range(self.size)]
        torch.distributed.all_gather(chunks, t.contiguous(), group=group)
        return torch.cat(chunks, dim=dim)

    def reduce_scatter(self, t: torch.Tensor, dim: int, op: str = ReduceOp.SUM) -> torch.Tensor:
        group = self._require()
        pieces = list(torch.chunk(t.contiguous(), self.size, dim=dim))
        out = torch.empty_like(pieces[self.coord])
        torch.distributed.reduce_scatter(out, pieces, op=_torch_reduce_op(op), group=group)
        return out

    def all_to_all(self, t: torch.Tensor, *, in_dim: int, out_dim: int) -> torch.Tensor:
        group = self._require()
        send = list(torch.chunk(t.contiguous(), self.size, dim=in_dim))
        recv = [torch.empty_like(send[self.coord]) for _ in range(self.size)]
        torch.distributed.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=out_dim)

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor:
        group = self._require()
        torch.distributed.broadcast(t, src=src, group=group)
        return t

@dataclass(frozen=True)
class LocalP2PTransport:
    """Coordinates are CUDA devices inside one process.

    ``devices[i]`` is the device for coordinate ``i``; ``coord`` is this mesh
    view's home coordinate (the device shared/primary modules live on).
    Movement is a direct device-to-device copy over NVLink; ordering across the
    two device streams is enforced with CUDA events (``record_ready`` /
    ``wait_ready``), which is the readiness-barrier discipline a modality
    handoff needs. This transport intentionally exposes no collective surface:
    a tower axis routes values and cannot silently stand in for a reduction.
    """

    axis: str
    devices: tuple[torch.device, ...]
    _coord: int

    @property
    def size(self) -> int:
        return len(self.devices)

    @property
    def coord(self) -> int:
        return int(self._coord)

    def device(self, coord: int) -> torch.device:
        """The CUDA device backing ``coord`` (in-process tower routing seam)."""
        return self.devices[int(coord)]

    def copy_to(self, t: torch.Tensor, *, coord: int, non_blocking: bool = True) -> torch.Tensor:
        return t.to(self.devices[int(coord)], non_blocking=non_blocking)

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor:
        if int(src) < 0 or int(src) >= self.size:
            raise ValueError(f"broadcast source {src} is outside axis {self.axis!r}")
        return t.to(self.devices[self.coord]) if t.device != self.devices[self.coord] else t

    def record_ready(self, coord: int | None = None) -> Any | None:
        # Record on ``coord``'s stream (the primary for B1; the gen coordinate
        # for B2). Defaults to this view's home coordinate.
        dev = self.devices[self.coord if coord is None else int(coord)]
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        torch.cuda.current_stream(dev).record_event(event)
        return event

    def wait_ready(self, event: Any | None, coord: int | None = None) -> None:
        # Make ``coord``'s stream wait on ``event``; the gen coordinate waits on
        # the primary's B1 before the snapshot copy, and on B2 before its first read.
        if event is None:
            return
        dev = self.devices[self.coord if coord is None else int(coord)]
        if dev.type != "cuda" or not torch.cuda.is_available():
            return
        torch.cuda.current_stream(dev).wait_event(event)


@dataclass(frozen=True)
class DataPlaneTowerTransport:
    """Cross-process tower transport backed by the data plane.

    The und and gen towers are separate worker processes and the und→gen KV handoff crosses
    the process boundary over the data plane (``cuda_ipc`` / ``mooncake``) rather
    than an in-process NVLink peer copy. The model still expresses the handoff as
    ``reshard(Pinned(primary) -> Pinned(gen))``; only the bound transport differs.

    The actual publish/fetch and readiness gate are owned by the data-plane /
    ``StageRouter`` integration. This class fixes the seam:

    * ``publish(t)`` (producer/und side) registers a tensor with the data plane
      and returns the opaque wire ``locator``; ``receive(locator, like=t)``
      (consumer/gen side) materializes it on the gen device.
    * ``record_ready``/``wait_ready`` map to the transfer-readiness gate; the
      default is a no-op because the gate is enforced host-side.
    * It exposes **no** ``device(coord)``: a cross-process tower has no
      in-process peer device, so ``place_towers`` skips device moves and
      ``route_by_modality`` takes the single-modality in-place path.

    ``data_plane`` is the injected mover (a ``TensorStore``-like object exposing
    ``publish``/``fetch``).
    """

    axis: str
    _size: int
    _coord: int
    data_plane: Any = None
    gate: Any = None

    @property
    def size(self) -> int:
        return int(self._size)

    @property
    def coord(self) -> int:
        return int(self._coord)

    def publish(self, t: torch.Tensor, *, kind: str = "kv_pages") -> Any:
        """Producer side: register ``t`` with the data plane, return its wire locator."""
        if self.data_plane is None:
            raise RuntimeError("cross-process tower transport requires a data-plane mover")
        return self.data_plane.publish(t, kind)

    def receive(self, locator: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
        """Consumer side: materialize the tensor named by ``locator`` locally."""
        if self.data_plane is None:
            raise RuntimeError("cross-process tower transport requires a data-plane mover")
        del like
        return self.data_plane.fetch_locator(locator)

    def copy_to(self, t: torch.Tensor, *, coord: int, non_blocking: bool = True) -> torch.Tensor:
        del non_blocking
        if int(coord) != self.coord:
            raise RuntimeError(
                "cross-process tower values must be transferred before model execution"
            )
        return t

    def record_ready(self, coord: int | None = None) -> Any | None:
        # Readiness is the host-side StageRouter transfer gate, not a local stream
        # event. The gate (when wired) decides when the gen worker may read the
        # snapshot; locally there is nothing to record.
        if self.gate is not None:
            return self.gate.record_ready(coord)
        return None

    def wait_ready(self, event: Any | None, coord: int | None = None) -> None:
        if self.gate is not None:
            self.gate.wait_ready(event, coord)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshAxis:
    """One parallelism dimension of the mesh.

    ``parent`` marks a *sub-factorization* axis (e.g. ``attn_tp`` carved out of
    ``tp``) rather than an independent product dimension; this models the real
    topology of production systems where finer axes divide a coarser one.
    """

    name: str
    size: int
    coord: int
    transport: AxisTransport | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"mesh axis {self.name!r} size must be positive")
        if self.coord < 0 or self.coord >= self.size:
            raise ValueError(
                f"mesh axis {self.name!r} coord must satisfy 0 <= coord < size "
                f"(coord={self.coord}, size={self.size})"
            )
        # A transport is required only to *communicate*; an axis may describe a
        # sharding topology (sizes/coords for load-time narrowing) without a live
        # transport. ``reshard`` raises if it must move data on a transportless axis.


@dataclass(frozen=True)
class DeviceMesh:
    """The process-wide parallelism topology: named axes + this rank's local device.

    The degenerate mesh (no non-trivial axis) reproduces a single-rank,
    single-device worker exactly.
    """

    axes: Mapping[str, MeshAxis] = field(default_factory=dict)
    local_device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def axis(self, name: str) -> MeshAxis | None:
        return self.axes.get(name)

    def size(self, name: str) -> int:
        ax = self.axes.get(name)
        return int(ax.size) if ax is not None else 1

    def coord(self, name: str) -> int:
        ax = self.axes.get(name)
        return int(ax.coord) if ax is not None else 0

    def is_trivial(self, name: str) -> bool:
        return self.size(name) <= 1

    def transport(self, name: str) -> AxisTransport:
        ax = self.axes.get(name)
        if ax is None or ax.transport is None:
            raise RuntimeError(f"mesh axis {name!r} has no transport")
        return ax.transport

    @property
    def tp_size(self) -> int:
        return self.size("tp")

    @property
    def tp_rank(self) -> int:
        return self.coord("tp")

    @classmethod
    def trivial(cls, device: torch.device | str = "cpu") -> "DeviceMesh":
        return cls(axes={}, local_device=torch.device(device))

    @classmethod
    def tp(
        cls,
        rank: int,
        size: int,
        *,
        transport: AxisTransport | None = None,
        device: torch.device | str = "cpu",
    ) -> "DeviceMesh":
        """A mesh with only a tensor-parallel axis. ``transport=None`` describes
        the sharding topology for load-time narrowing without a live collective."""
        if int(size) <= 1:
            return cls.trivial(device)
        ax = MeshAxis(name="tp", size=int(size), coord=int(rank), transport=transport)
        return cls(axes={"tp": ax}, local_device=torch.device(device))

    @classmethod
    def of(cls, *axes: MeshAxis, device: torch.device | str = "cpu") -> "DeviceMesh":
        return cls(
            axes={ax.name: ax for ax in axes if ax.size > 1},
            local_device=torch.device(device),
        )

    def with_axis(self, ax: MeshAxis) -> "DeviceMesh":
        merged = dict(self.axes)
        if ax.size > 1:
            merged[ax.name] = ax
        else:
            merged.pop(ax.name, None)
        return DeviceMesh(axes=merged, local_device=self.local_device)


@dataclass(frozen=True, slots=True)
class TensorParallelSpec:
    """Transport-free tensor-parallel coordinates used during layer construction."""

    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("tensor-parallel size must be positive")
        if self.rank < 0 or self.rank >= self.size:
            raise ValueError("tensor-parallel rank must satisfy 0 <= rank < size")

    @classmethod
    def from_mesh(cls, mesh: DeviceMesh) -> "TensorParallelSpec":
        return cls(rank=mesh.tp_rank, size=mesh.tp_size)
