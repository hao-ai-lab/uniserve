"""Device mesh and transport boundaries for model parallelism."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

__all__ = [
    'divide',
    'AxisTransport',
    'BroadcastTransport',
    'CollectiveAxisTransport',
    'PeerAxisTransport',
    'CollectiveTransport',
    'LocalP2PTransport',
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

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor: ...
    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor: ...


@runtime_checkable
class PeerAxisTransport(AxisTransport, Protocol):
    """Point-to-point movement implemented by an in-process or staged peer axis."""

    def copy_to(self, t: torch.Tensor, *, coord: int, non_blocking: bool = True) -> torch.Tensor: ...


@runtime_checkable
class BroadcastTransport(AxisTransport, Protocol):
    """One-to-all movement for a transport that defines broadcast semantics."""

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class CollectiveTransport:
    """A ``torch.distributed`` process group bound to one mesh axis.

    ``group`` is ``None`` for the default world group. ``size`` and ``coord``
    identify this process within the axis.
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

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        group = self._require()
        torch.distributed.all_reduce(t, group=group)
        return t

    def all_gather(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        group = self._require()
        chunks = [torch.empty_like(t) for _ in range(self.size)]
        torch.distributed.all_gather(chunks, t.contiguous(), group=group)
        return torch.cat(chunks, dim=dim)

    def broadcast(self, t: torch.Tensor, *, src: int) -> torch.Tensor:
        group = self._require()
        torch.distributed.broadcast(t, src=src, group=group)
        return t

@dataclass(frozen=True)
class LocalP2PTransport:
    """Coordinates are CUDA devices inside one process.

    ``devices[i]`` is the device for coordinate ``i``; ``coord`` is this mesh
    view's home coordinate (the device shared/primary modules live on).
    Movement is a direct device-to-device copy. This transport exposes no
    collective surface, so routing axes cannot be used for reductions.
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
        # Load-time sharding can use axis coordinates without a live transport.


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
    def of(cls, *axes: MeshAxis, device: torch.device | str = "cpu") -> "DeviceMesh":
        return cls(
            axes={ax.name: ax for ax in axes if ax.size > 1},
            local_device=torch.device(device),
        )

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
