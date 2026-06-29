"""Flow-matching timestep schedules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import torch

__all__ = [
    'ScheduleDirection',
    'ScheduleShiftDomain',
    'FlowMatchSchedule',
    'x_pred_to_velocity',
]


class ScheduleDirection(str, Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


class ScheduleShiftDomain(str, Enum):
    TIME = "time"
    SIGMA = "sigma"


@dataclass
class FlowMatchSchedule:
    """A flow-matching timestep schedule (a stateful value object).

    The identity fields below are set once at construction; ``_timesteps_cache``
    memoizes the computed timesteps tensor keyed by (device, dtype) so repeated
    ``pair()``/``timesteps()`` calls in a denoise loop do not redo the
    linspace + shift transform per step. It is deliberately not ``frozen`` (the
    cache is real per-instance state) and is excluded from equality.
    """

    num_steps: int
    shift: float = 1.0
    direction: ScheduleDirection = ScheduleDirection.ASCENDING
    shift_domain: ScheduleShiftDomain = ScheduleShiftDomain.TIME
    _timesteps_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _base_timesteps(self, *, device, dtype) -> torch.Tensor:
        endpoints = _DIRECTION_ENDPOINTS.get(self.direction)
        if endpoints is None:
            raise ValueError(f"unknown direction {self.direction!r}")
        start, stop = endpoints
        return torch.linspace(start, stop, self.num_steps + 1, device=device, dtype=dtype)

    def _apply_shift_time(self, base: torch.Tensor) -> torch.Tensor:
        return self.shift_time(base)

    def _apply_shift_sigma(self, base: torch.Tensor) -> torch.Tensor:
        sigma = self.shift_time(1 - base)
        return 1 - sigma

    def _compute_timesteps(self, *, device, dtype) -> torch.Tensor:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        base = self._base_timesteps(device=device, dtype=dtype)
        if self.shift == 1.0:
            return base
        shift_method = _SHIFT_DOMAIN_METHODS.get(self.shift_domain)
        if shift_method is None:
            raise ValueError(f"unknown shift domain {self.shift_domain!r}")
        return shift_method(self, base)

    def timesteps(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        key = (device, dtype)
        cached = self._timesteps_cache.get(key)
        if cached is None:
            cached = self._compute_timesteps(device=device, dtype=dtype)
            self._timesteps_cache[key] = cached
        return cached

    def shift_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.shift <= 0:
            raise ValueError("shift must be positive")
        return self.shift * t / (1 + (self.shift - 1) * t)

    def pair(self, index: int, *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        steps = self.timesteps(device=device, dtype=dtype)
        if index < 0 or index >= self.num_steps:
            raise IndexError(index)
        return steps[index], steps[index + 1]


# Direction -> (linspace start, stop) for the base timesteps, and shift-domain ->
# the method that applies the shift transform. Splitting the two enum decisions
# into tables keeps each formula in one place and makes the direction x domain
# combinations explicit instead of nested if/elif. The arithmetic is identical to
# the prior branches.
_DIRECTION_ENDPOINTS = MappingProxyType({
    ScheduleDirection.ASCENDING: (0.0, 1.0),
    ScheduleDirection.DESCENDING: (1.0, 0.0),
})
_SHIFT_DOMAIN_METHODS = MappingProxyType({
    ScheduleShiftDomain.TIME: FlowMatchSchedule._apply_shift_time,
    ScheduleShiftDomain.SIGMA: FlowMatchSchedule._apply_shift_sigma,
})


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
