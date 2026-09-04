"""Immutable analytical coordinates for flow-matching integration."""

from __future__ import annotations

from enum import Enum

import torch

__all__ = [
    "ScheduleDirection",
    "ScheduleShiftDomain",
    "flow_match_coordinate",
    "x_pred_to_velocity",
]


class ScheduleDirection(str, Enum):
    """Selects ascending or descending traversal of a flow schedule."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class ScheduleShiftDomain(str, Enum):
    """Selects whether schedule shifts operate in time or sigma coordinates."""

    TIME = "time"
    SIGMA = "sigma"


def flow_match_coordinate(
    num_steps: int,
    shift: float,
    direction: ScheduleDirection,
    shift_domain: ScheduleShiftDomain,
    index: int,
) -> float:
    """Return one analytical flow coordinate from immutable schedule parameters."""

    steps = int(num_steps)
    coordinate = int(index)
    shift = float(shift)
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    if coordinate < 0 or coordinate > steps:
        raise IndexError(coordinate)
    if shift <= 0:
        raise ValueError("shift must be positive")
    if direction is ScheduleDirection.ASCENDING:
        start, stop = 0.0, 1.0
    elif direction is ScheduleDirection.DESCENDING:
        start, stop = 1.0, 0.0
    else:
        raise ValueError(f"unknown direction {direction!r}")
    value = start + (stop - start) * (float(coordinate) / float(steps))
    if shift == 1.0:
        return value
    if shift_domain is ScheduleShiftDomain.TIME:
        return shift * value / (1.0 + (shift - 1.0) * value)
    if shift_domain is ScheduleShiftDomain.SIGMA:
        sigma = 1.0 - value
        shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        return 1.0 - shifted
    raise ValueError(f"unknown shift domain {shift_domain!r}")


def x_pred_to_velocity(
    x_pred: torch.Tensor,
    latent: torch.Tensor,
    t: torch.Tensor,
    *,
    t_eps: float = 1e-6,
) -> torch.Tensor:
    """Convert an x-prediction flow head output into velocity."""

    denom = (1 - t).clamp_min(float(t_eps))
    while denom.ndim < latent.ndim:
        denom = denom.unsqueeze(-1)
    return (x_pred - latent) / denom.to(dtype=latent.dtype, device=latent.device)
