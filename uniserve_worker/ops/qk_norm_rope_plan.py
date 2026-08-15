"""Shared planning for QK norm plus RoPE requests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch

from .requests import QKNormRopeReq

__all__ = [
    "QKAxisGroup",
    "QKNormRopePlan",
]


@dataclass(frozen=True)
class QKAxisGroup:
    start: int
    end: int
    dim: int

    @property
    def single_axis(self) -> bool:
        return self.end == self.start + 1


@dataclass(frozen=True)
class QKNormRopePlan:
    """Validated axis grouping and table layout for QK norm/RoPE."""

    req: QKNormRopeReq
    axis_dims: tuple[int, ...]
    q_weights: tuple[torch.Tensor, ...]
    k_weights: tuple[torch.Tensor, ...]
    cos_tables: tuple[torch.Tensor, ...]
    sin_tables: tuple[torch.Tensor, ...]
    groups: tuple[QKAxisGroup, ...]

    @classmethod
    def from_request(cls, req: QKNormRopeReq) -> "QKNormRopePlan":
        if req.axis_dims is None:
            return cls(
                req=req,
                axis_dims=(),
                q_weights=(),
                k_weights=(),
                cos_tables=(),
                sin_tables=(),
                groups=(),
            )
        cls.validate_multi_axis(req)
        axis_dims = tuple(int(dim) for dim in req.axis_dims)
        q_weights = cast(tuple[torch.Tensor, ...], req.q_weight)
        k_weights = cast(tuple[torch.Tensor, ...], req.k_weight)
        cos_tables = cast(tuple[torch.Tensor, ...], req.cos)
        sin_tables = cast(tuple[torch.Tensor, ...], req.sin)
        groups: list[QKAxisGroup] = []
        axis = 0
        while axis < len(axis_dims):
            end = cls.shared_norm_group_end(req, axis)
            groups.append(
                QKAxisGroup(
                    start=axis,
                    end=end,
                    dim=sum(axis_dims[axis:end]),
                )
            )
            axis = end
        return cls(
            req=req,
            axis_dims=axis_dims,
            q_weights=q_weights,
            k_weights=k_weights,
            cos_tables=cos_tables,
            sin_tables=sin_tables,
            groups=tuple(groups),
        )

    @staticmethod
    def validate_multi_axis(req: QKNormRopeReq) -> None:
        if req.axis_dims is None:
            return
        if not isinstance(req.q_weight, tuple) or not isinstance(req.k_weight, tuple):
            raise RuntimeError("multi-axis qk_norm_rope requires tuple weights")
        if not isinstance(req.cos, tuple) or not isinstance(req.sin, tuple):
            raise RuntimeError("multi-axis qk_norm_rope requires tuple cos/sin")
        if len(req.axis_dims) != len(req.cos) or len(req.cos) != len(req.sin):
            raise RuntimeError("multi-axis qk_norm_rope axis/cos/sin mismatch")
        if len(req.q_weight) != len(req.axis_dims) or len(req.k_weight) != len(req.axis_dims):
            raise RuntimeError("multi-axis qk_norm_rope weight/axis mismatch")

    @staticmethod
    def shared_norm_group_end(req: _QKNormAxes, start: int) -> int:
        if req.axis_dims is None:
            return int(start) + 1
        q_weights = cast(tuple[torch.Tensor, ...], req.q_weight)
        k_weights = cast(tuple[torch.Tensor, ...], req.k_weight)
        q_weight = q_weights[start]
        k_weight = k_weights[start]
        group_end = start + 1
        group_dim = int(req.axis_dims[start])
        while (
            group_end < len(req.axis_dims)
            and q_weights[group_end] is q_weight
            and k_weights[group_end] is k_weight
            and group_dim < int(q_weight.shape[-1])
        ):
            group_dim += int(req.axis_dims[group_end])
            group_end += 1
        return group_end if group_dim == int(q_weight.shape[-1]) else start + 1

class _QKNormAxes(Protocol):
    @property
    def axis_dims(self) -> tuple[int, ...] | None: ...

    @property
    def q_weight(self) -> torch.Tensor | tuple[torch.Tensor, ...]: ...

    @property
    def k_weight(self) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
