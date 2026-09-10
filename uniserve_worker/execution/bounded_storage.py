"""Capacity-owned tensors with validated, shape-keyed layout views."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import torch

from ..nn.mesh import Communicator

if TYPE_CHECKING:
    from ..runtime.distributed import DistributedEnvironment

__all__ = ["BoundedTensorStorage", "TensorSchema"]


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


class BoundedTensorStorage:
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

    @classmethod
    def allocate(
        cls,
        schema: Mapping[str, TensorSchema],
        device: torch.device | str,
        *,
        environment: DistributedEnvironment | None = None,
        layout: tuple[object, ...] = (),
    ) -> BoundedTensorStorage:
        """Allocate the declared tensor capacities on the publicly assigned device."""

        tensors: dict[str, torch.Tensor] = {}
        peers: dict[str, tuple[torch.Tensor, ...]] = {}
        for name, field in schema.items():
            if field.memory == "symmetric":
                if environment is None or field.group is None:
                    raise ValueError("shared tensor storage requires its distributed owner")
                if field.group.device != torch.device(device):
                    raise ValueError("shared tensor storage must use the group's assigned device")
                symmetric = environment.symmetric_memory(
                    field.group, field.shape, dtype=field.dtype, name=name, layout=layout
                )
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
        """Bind a shape key to validated views that share the storage’s fixed backing tensors."""

        cached = self._views.get(shape_key)
        if cached is not None:
            requested = {
                name: tuple(int(value) for value in shape) for name, shape in shapes.items()
            }
            resident = {name: tuple(tensor.shape) for name, tensor in cached.items()}
            if requested != resident:
                raise ValueError("bounded shape key was rebound with different view geometry")
            return cached
        if set(shapes) != set(self._tensors):
            missing = sorted(set(self._tensors) - set(shapes))
            extra = sorted(set(shapes) - set(self._tensors))
            raise ValueError(
                f"bounded view fields do not match storage (missing={missing}, extra={extra})"
            )
        views: dict[str, torch.Tensor] = {}
        for name, storage in self._tensors.items():
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
