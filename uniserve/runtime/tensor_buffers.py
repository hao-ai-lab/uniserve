"""Capacity-owned tensors with validated, shape-keyed layout views."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from types import MappingProxyType

import torch

from uniserve.distributed.mesh import Communicator, SymmetricMemory
from uniserve.distributed.peer_memory import allocate_symmetric_memory
from uniserve.tensors import BufferConfig

__all__ = ["TensorBuffers"]


class TensorBuffers:
    """Own fixed-capacity tensors and cache prefix views for legal shape keys."""

    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Take ownership of named capacity tensors and initialize an empty view cache."""

        if not tensors:
            raise ValueError("bounded tensor storage requires at least one field")
        self._tensors = dict(tensors)
        if len(self._tensors) != len(tensors):
            raise ValueError("bounded tensor storage field names must be unique")
        if any(not tensor.is_contiguous() for tensor in self._tensors.values()):
            raise ValueError("bounded tensor storage fields must be contiguous")
        self._views: dict[Hashable, Mapping[str, torch.Tensor]] = {}
        self._peers: dict[str, tuple[torch.Tensor, ...]] = {}
        self._symmetric: list[SymmetricMemory] = []

    @classmethod
    def allocate(
        cls,
        configs: Mapping[str, BufferConfig],
        device: torch.device | str,
        *,
        pin_memory: bool = False,
        symmetric: Mapping[str, Communicator] | None = None,
        fill: Mapping[str, int | float] | None = None,
    ) -> TensorBuffers:
        """Allocate numerical capacities with caller-selected physical resources.

        ``pin_memory`` applies only to host fields. ``symmetric`` binds named
        device fields to communication groups; their local and peer views share
        the same owner. Initialization happens once, before views are borrowed.
        The caller retires graphs and asynchronous readers before releasing this owner.
        """

        symmetric = symmetric or {}
        fill = fill or {}
        unknown = (symmetric.keys() | fill.keys()) - configs.keys()
        if unknown:
            raise ValueError(f"allocation options refer to unknown fields: {sorted(unknown)}")
        for name, member in symmetric.items():
            if configs[name].host or member.device != torch.device(device):
                raise ValueError("symmetric storage requires the group's assigned device")
        tensors: dict[str, torch.Tensor] = {}
        peers: dict[str, tuple[torch.Tensor, ...]] = {}
        workspaces: list[SymmetricMemory] = []
        for name, field in configs.items():
            shape = field.capacity_shape if field.capacity_shape is not None else field.shape
            group = symmetric.get(name)
            if group is not None:
                workspace = allocate_symmetric_memory(group, shape, dtype=field.dtype)
                workspaces.append(workspace)
                tensors[name] = workspace.local
                peers[name] = workspace.peers
            else:
                tensors[name] = torch.empty(
                    shape,
                    dtype=field.dtype,
                    device="cpu" if field.host else device,
                    pin_memory=field.host and pin_memory,
                )
            if name in fill:
                tensors[name].fill_(fill[name])
        storage = cls(tensors)
        storage._peers = peers
        storage._symmetric = workspaces
        return storage

    def peers(self, name: str) -> tuple[torch.Tensor, ...]:
        """Borrow ordered member views of an explicitly shared allocation."""

        return self._peers[name]

    @property
    def capacity(self) -> Mapping[str, torch.Tensor]:
        """Expose read-only names for the fixed-capacity backing tensors."""

        return MappingProxyType(self._tensors)

    def bind(
        self,
        shape_key: Hashable,
        shapes: Mapping[str, tuple[int, ...]],
    ) -> Mapping[str, torch.Tensor]:
        """Borrow a call's named subset of the storage's fixed backing tensors.

        Different computations can share one request allocation while receiving
        only their declared fields. Each key retains one exact view geometry.
        """

        cached = self._views.get(shape_key)
        if cached is not None:
            requested = {
                name: tuple(int(value) for value in shape) for name, shape in shapes.items()
            }
            resident = {name: tuple(tensor.shape) for name, tensor in cached.items()}
            if requested != resident:
                raise ValueError("bounded shape key was rebound with different view geometry")
            return cached
        extra = shapes.keys() - self._tensors.keys()
        if extra:
            raise ValueError(f"bounded views have no backing for {sorted(extra)}")
        views: dict[str, torch.Tensor] = {}
        for name in shapes:
            storage = self._tensors[name]
            shape = tuple(int(value) for value in shapes[name])
            if len(shape) != storage.ndim or any(value < 0 for value in shape):
                raise ValueError(f"bounded view {name!r} has an invalid rank or extent")
            if any(value > capacity for value, capacity in zip(shape, storage.shape, strict=True)):
                raise ValueError(f"bounded view {name!r} exceeds resident capacity")
            elements = 1
            for extent in shape:
                elements *= extent
            views[name] = storage.reshape(-1)[:elements].view(shape)
        result = MappingProxyType(views)
        self._views[shape_key] = result
        return result
