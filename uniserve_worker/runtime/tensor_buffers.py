"""Capacity-owned tensors with validated, shape-keyed layout views."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Literal

import torch

from ..nn.mesh import Communicator, SymmetricMemory
from .peer_memory import allocate_symmetric_memory

__all__ = ["TensorBuffers", "TensorSchema"]


@dataclass(frozen=True, slots=True)
class TensorSchema:
    """Logical shape and representation required by a compute input or workspace."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    memory: Literal["device", "pinned", "symmetric"] = "device"
    group: Communicator | None = None
    fill: int | float | None = None

    def __post_init__(self) -> None:
        if any(extent < 0 for extent in self.shape):
            raise ValueError("tensor schema extents cannot be negative")
        if self.memory not in {"device", "pinned", "symmetric"}:
            raise ValueError("unknown tensor storage kind")
        if (self.memory == "symmetric") != (self.group is not None):
            raise ValueError("shared tensor storage requires explicit group membership")

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


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
        schema: Mapping[str, TensorSchema],
        device: torch.device | str,
    ) -> TensorBuffers:
        """Allocate the declared tensor capacities on the publicly assigned device."""

        tensors: dict[str, torch.Tensor] = {}
        peers: dict[str, tuple[torch.Tensor, ...]] = {}
        workspaces: list[SymmetricMemory] = []
        for name, field in schema.items():
            if field.memory == "symmetric":
                if field.group is None:
                    raise ValueError("shared tensor storage requires its process group")
                if field.group.device != torch.device(device):
                    raise ValueError("shared tensor storage must use the group's assigned device")
                symmetric = allocate_symmetric_memory(field.group, field.shape, dtype=field.dtype)
                workspaces.append(symmetric)
                tensors[name] = symmetric.local
                peers[name] = symmetric.peers
                if field.fill is not None:
                    peers[name][field.group.rank_in_group].fill_(field.fill)
            else:
                tensors[name] = torch.empty(
                    field.shape,
                    dtype=field.dtype,
                    device="cpu" if field.memory == "pinned" else device,
                    pin_memory=field.memory == "pinned",
                )
                if field.fill is not None:
                    tensors[name].fill_(field.fill)
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
