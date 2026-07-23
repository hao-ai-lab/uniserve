"""Worker-owned device mesh and bounded per-forward mesh views."""

from __future__ import annotations

import torch

from ..nn.mesh import CollectiveAxisTransport, DeviceMesh, PeerAxisTransport


class MeshView:
    def __init__(self, mesh: DeviceMesh, allowed_axes: frozenset[str]) -> None:
        self._mesh = mesh
        self._allowed_axes = allowed_axes

    def all_reduce(self, value: torch.Tensor, axis: str) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, CollectiveAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support all-reduce")
        return transport.all_reduce(value)

    def all_gather(
        self, value: torch.Tensor, axis: str, dimension: int
    ) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, CollectiveAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support all-gather")
        return transport.all_gather(value, dimension)

    def dispatch(
        self, value: torch.Tensor, axis: str, coordinate: int
    ) -> torch.Tensor:
        transport = self._transport(axis)
        if transport is None:
            return value
        if not isinstance(transport, PeerAxisTransport):
            raise RuntimeError(f"mesh axis {axis!r} does not support peer dispatch")
        return transport.copy_to(value, coord=int(coordinate))

    def combine(
        self,
        value: torch.Tensor,
        axis: str,
        coordinate: int,
        target: torch.device,
    ) -> torch.Tensor:
        self._require_axis(axis)
        del coordinate
        return value if value.device == target else value.to(target, non_blocking=True)

    def _transport(self, axis: str):
        self._require_axis(axis)
        if self._mesh.is_trivial(axis):
            return None
        return self._mesh.transport(axis)

    def _require_axis(self, axis: str) -> None:
        if axis not in self._allowed_axes:
            raise RuntimeError(f"mesh axis {axis!r} is outside this forward route")


class MeshStore:
    def __init__(self, mesh: DeviceMesh) -> None:
        self.mesh = mesh

    def view(self, axes: tuple[str, ...]) -> MeshView:
        allowed = frozenset(axes)
        for axis in allowed:
            if not axis:
                raise ValueError("mesh axis names must not be empty")
        return MeshView(self.mesh, allowed)


__all__ = ["MeshStore", "MeshView"]
