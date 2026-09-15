"""Normalization grouping and fused-launch planning for multi-axis QK RoPE.

Multi-axis requests may assign one normalization weight to several adjacent
rotary axes. This module validates the parallel axis metadata, recovers those
shared normalization groups from weight identity and width, and identifies the
head/tail layouts supported by specialized Triton kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from uniserve.ops.requests import MultiAxisQKNormReq, MultiAxisQKNormRopeReq

__all__ = [
    "FusedStrategy",
    "QKAxisGroup",
    "QKNormRopePlan",
]


class FusedStrategy(str, Enum):
    """Execution strategies for per-group and specialized two-group layouts."""

    PER_GROUP = "per_group"
    IDENTITY_TAIL = "identity_tail"
    ROTATED_SHARED_TAIL = "rotated_shared_tail"


@dataclass(frozen=True)
class QKAxisGroup:
    """Contiguous rotary axes normalized together by one weight tensor."""

    start: int
    end: int
    dim: int

    @property
    def single_axis(self) -> bool:
        """Report whether the normalization group contains exactly one axis."""

        return self.end == self.start + 1


@dataclass(frozen=True)
class QKNormRopePlan:
    """Validated axis metadata, normalization groups, and launch strategy."""

    axis_dims: tuple[int, ...]
    q_weights: tuple[torch.Tensor, ...]
    k_weights: tuple[torch.Tensor, ...]
    cos_tables: tuple[torch.Tensor, ...]
    sin_tables: tuple[torch.Tensor, ...]
    groups: tuple[QKAxisGroup, ...]
    strategy: FusedStrategy
    unsqueeze_dim: int
    eps: float

    @classmethod
    def from_request(cls, req: MultiAxisQKNormRopeReq) -> "QKNormRopePlan":
        """Validate a norm-plus-RoPE request and derive its execution plan."""

        _validate_multi_axis_rope(req)

        # Canonical integer dimensions feed both slicing and kernel launch sizes;
        # group discovery then determines which axes share an RMS reduction.
        axis_dims = tuple(int(dim) for dim in req.axis_dims)
        groups = _groups(axis_dims, req.q_weights, req.k_weights)

        return cls(
            axis_dims=axis_dims,
            q_weights=req.q_weights,
            k_weights=req.k_weights,
            cos_tables=req.cos_tables,
            sin_tables=req.sin_tables,
            groups=groups,
            strategy=_fused_strategy(groups, req.identity_axes, len(axis_dims)),
            unsqueeze_dim=int(req.unsqueeze_dim),
            eps=float(req.eps),
        )

    @classmethod
    def from_norm_request(cls, req: MultiAxisQKNormReq) -> tuple[QKAxisGroup, ...]:
        """Validate a normalization-only request and return its shared groups."""

        _validate_multi_axis_norm(req)
        return _groups(
            tuple(int(dim) for dim in req.axis_dims),
            req.q_weights,
            req.k_weights,
        )


def _validate_multi_axis_norm(req: MultiAxisQKNormReq) -> None:
    """Require one query and key weight reference per declared axis."""

    if len(req.q_weights) != len(req.axis_dims) or len(req.k_weights) != len(req.axis_dims):
        raise RuntimeError("multi-axis qk_norm weight/axis mismatch")


def _validate_multi_axis_rope(req: MultiAxisQKNormRopeReq) -> None:
    """Require parallel axis, rotary-table, and normalization-weight metadata."""

    if len(req.axis_dims) != len(req.cos_tables) or len(req.cos_tables) != len(req.sin_tables):
        raise RuntimeError("multi-axis qk_norm_rope axis/cos/sin mismatch")
    if len(req.q_weights) != len(req.axis_dims) or len(req.k_weights) != len(req.axis_dims):
        raise RuntimeError("multi-axis qk_norm_rope weight/axis mismatch")


def _groups(
    axis_dims: tuple[int, ...],
    q_weights: tuple[torch.Tensor, ...],
    k_weights: tuple[torch.Tensor, ...],
) -> tuple[QKAxisGroup, ...]:
    """Partition adjacent axes by their shared query/key normalization weights."""

    groups: list[QKAxisGroup] = []
    axis = 0

    # Weight object identity declares a shared normalization domain; each
    # accepted group must also consume the weight's exact feature width.
    while axis < len(axis_dims):
        end = _shared_norm_group_end(axis_dims, q_weights, k_weights, axis)
        groups.append(
            QKAxisGroup(
                start=axis,
                end=end,
                dim=sum(axis_dims[axis:end]),
            )
        )
        axis = end

    return tuple(groups)


def _shared_norm_group_end(
    axis_dims: tuple[int, ...],
    q_weights: tuple[torch.Tensor, ...],
    k_weights: tuple[torch.Tensor, ...],
    start: int,
) -> int:
    """Find the exclusive end of the shared normalization group at ``start``."""

    q_weight = q_weights[start]
    k_weight = k_weights[start]
    group_end = start + 1
    group_dim = int(axis_dims[start])

    while (
        group_end < len(axis_dims)
        and q_weights[group_end] is q_weight
        and k_weights[group_end] is k_weight
        and group_dim < int(q_weight.shape[-1])
    ):
        group_dim += int(axis_dims[group_end])
        group_end += 1

    # A partial width cannot share an RMS reduction: keep the starting axis
    # independent unless the accumulated axes cover the complete weight.
    return group_end if group_dim == int(q_weight.shape[-1]) else start + 1


def _fused_strategy(
    groups: tuple[QKAxisGroup, ...],
    identity_axes: tuple[int, ...],
    n_axes: int,
) -> FusedStrategy:
    """Select a specialized kernel for supported head-plus-tail group layouts."""

    # Both fused forms require axis zero as an independent head group and all
    # remaining axes as one shared tail normalization group.
    two_groups = (
        len(groups) == 2
        and groups[0].single_axis
        and groups[0].start == 0
        and groups[1].start == 1
        and groups[1].end == n_axes
    )

    if two_groups and identity_axes == tuple(range(1, n_axes)):
        return FusedStrategy.IDENTITY_TAIL
    if two_groups and n_axes == 3 and not identity_axes:
        return FusedStrategy.ROTATED_SHARED_TAIL

    return FusedStrategy.PER_GROUP
