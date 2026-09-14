"""Immutable numerical options supplied for one diffusion trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass

from uniserve.nn.diffusion.cfg import RenormKind


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    """Evaluation count, guidance and seed without dimensions or runtime state.

    A missing timestep shift selects the model's mathematical default. The
    guidance interval uses analytical host coordinates before FP32 staging.
    """

    steps: int
    timestep_shift: float | None = None
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    cfg_renorm: RenormKind = RenormKind.GLOBAL
    cfg_renorm_min: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("diffusion evaluation count must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("diffusion seed must be an integer")
        if self.timestep_shift is not None and (
            not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0
        ):
            raise ValueError("diffusion timestep shift must be finite and positive")
        if not isinstance(self.cfg_interval, tuple) or len(self.cfg_interval) != 2:
            raise ValueError("guidance interval must contain two immutable endpoints")
        if not all(
            math.isfinite(value)
            for value in (
                self.cfg_text_scale,
                self.cfg_img_scale,
                *self.cfg_interval,
                self.cfg_renorm_min,
            )
        ):
            raise ValueError("guidance values must be finite")
        if min(self.cfg_text_scale, self.cfg_img_scale) < 0:
            raise ValueError("guidance scales must be nonnegative")
        if self.cfg_interval[0] > self.cfg_interval[1]:
            raise ValueError("guidance interval must be ordered")
        if not isinstance(self.cfg_renorm, RenormKind):
            raise TypeError("guidance renormalization must be a RenormKind")
