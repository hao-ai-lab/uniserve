"""Immutable analytical coordinates for flow-matching integration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


def shifted_sigmas(ladder: tuple[int, ...], shift: float, *, scale: float) -> tuple[float, ...]:
    """Map a descending trained ladder to sigma coordinates with a terminal zero."""

    if shift <= 0 or scale <= 0:
        raise ValueError("schedule shift and scale must be positive")
    if not ladder or any(value <= 0 or value > scale for value in ladder):
        raise ValueError("schedule ladder must contain positive points within its scale")
    if any(left <= right for left, right in zip(ladder, ladder[1:])):
        raise ValueError("schedule ladder must be strictly descending")
    base = (*((value / scale) for value in ladder), 0.0)
    return tuple(shift * sigma / (1.0 + (shift - 1.0) * sigma) for sigma in base)


@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
    """Stable sigma and clean-time tensors in the declared modality order."""

    sigmas: tuple[torch.Tensor, ...]
    timesteps: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.sigmas or len(self.sigmas) != len(self.timesteps):
            raise ValueError("schedule modalities must have sigma and timestep tensors")
        for sigma, timestep in zip(self.sigmas, self.timesteps):
            if sigma.ndim != 1 or timestep.ndim != 1 or sigma.numel() != timestep.numel() + 1:
                raise ValueError("schedule requires one terminal sigma after its timesteps")
            if timestep.numel() == 0 or sigma.device != timestep.device:
                raise ValueError("schedule timesteps must be nonempty and share the sigma device")
            if sigma.dtype != torch.float32 or timestep.dtype != torch.float32:
                raise ValueError("schedule constants must use FP32")

    @classmethod
    def uniform_grid(
        cls,
        points: int,
        shifts: tuple[float, ...],
        *,
        device: torch.device | str,
    ) -> DiffusionSchedule:
        """Build a shifted FP32 grid with ``points - 1`` Euler intervals.

        Coordinates are evaluated on CPU before device transfer, matching the
        MiniMax-H3 scheduler's rounding and clean-time convention.
        """

        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise ValueError("a uniform diffusion grid requires at least two points")
        if not shifts or any(not shift > 0 for shift in shifts):
            raise ValueError("schedule shifts must be positive")
        base = torch.linspace(1.0, 0.0, points, dtype=torch.float32)
        sigmas = tuple(
            torch.unique_consecutive(shift * base / (1 + (shift - 1) * base)) for shift in shifts
        )
        return cls(
            tuple(value.to(device=device) for value in sigmas),
            tuple((1.0 - value[:-1]).to(device=device) for value in sigmas),
        )

    @classmethod
    def build(
        cls,
        ladder: tuple[int, ...],
        shifts: tuple[float, ...],
        *,
        scale: float,
        device: torch.device | str,
    ) -> DiffusionSchedule:
        """Materialize immutable FP32 schedule constants before graph capture."""

        if not shifts:
            raise ValueError("a diffusion schedule requires at least one modality")
        sigmas = tuple(
            torch.tensor(
                shifted_sigmas(ladder, shift, scale=scale), dtype=torch.float32, device=device
            )
            for shift in shifts
        )
        return cls(sigmas, tuple(1.0 - values[:-1] for values in sigmas))


__all__ = [
    "DiffusionSchedule",
    "shifted_sigmas",
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
