"""Capacity-owned tensors with validated, shape-keyed layout views."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from types import MappingProxyType

import torch

__all__ = ["BoundedTensorStorage"]


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
                name: tuple(int(value) for value in shape)
                for name, shape in shapes.items()
            }
            resident = {name: tuple(tensor.shape) for name, tensor in cached.items()}
            if requested != resident:
                raise ValueError(
                    "bounded shape key was rebound with different view geometry"
                )
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
            if any(
                value > capacity
                for value, capacity in zip(shape, storage.shape, strict=True)
            ):
                raise ValueError(f"bounded view {name!r} exceeds resident capacity")
            elements = 1
            for extent in shape:
                elements *= extent
            views[name] = storage.reshape(-1)[:elements].view(shape)
        result = MappingProxyType(views)
        self._views[shape_key] = result
        return result
