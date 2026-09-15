"""Analytical diffusion coordinates and their FP32 numerical endpoints."""

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class Schedule:
    """Trajectory endpoints: FP32 network times, sigmas, and host coordinates.

    All three hold ``num_steps + 1`` aligned entries. ``coordinates`` retains
    the unrounded analytical values that guidance interval comparisons use.
    """

    timesteps: torch.Tensor
    sigmas: torch.Tensor
    coordinates: tuple[float, ...]

    def __post_init__(self):
        if (
            self.timesteps.ndim != 1
            or self.sigmas.shape != self.timesteps.shape
            or self.timesteps.numel() < 2
            or len(self.coordinates) != self.timesteps.numel()
            or self.timesteps.device != self.sigmas.device
            or self.timesteps.dtype != torch.float32
            or self.sigmas.dtype != torch.float32
            or not isinstance(self.coordinates, tuple)
            or any(not math.isfinite(value) for value in self.coordinates)
        ):
            raise ValueError("schedules require aligned FP32 endpoints and analytical coordinates")

    @property
    def num_steps(self) -> int:
        return len(self.coordinates) - 1


def make_schedule(
    steps: int,
    *,
    shift: float,
    direction: Literal["ascending", "descending"],
    shift_domain: Literal["time", "sigma"],
    device: torch.device | str,
) -> Schedule:
    """Shift a complete linear trajectory, retaining unrounded host coordinates.

    The network time follows direction. Sigma decreases from one to zero for
    either direction, and is computed from the materialized FP32 network time.
    """
    if type(steps) is not int or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("schedule steps and shift must be positive")
    if direction not in {"ascending", "descending"} or shift_domain not in {"time", "sigma"}:
        raise ValueError("unknown schedule direction or shift domain")
    coordinates = []
    for index in range(steps + 1):
        value = index / steps if direction == "ascending" else 1.0 - index / steps
        if shift != 1:
            # The shift formula operates on the increasing coordinate; shifting
            # in the sigma domain mirrors the time coordinate into it.
            coordinate = value if shift_domain == "time" else 1.0 - value
            shifted = shift * coordinate / (1.0 + (shift - 1.0) * coordinate)
            value = shifted if shift_domain == "time" else 1.0 - shifted
        coordinates.append(value)

    times = torch.tensor(coordinates, dtype=torch.float32, device=device)
    return Schedule(times, times if direction == "descending" else 1.0 - times, tuple(coordinates))
