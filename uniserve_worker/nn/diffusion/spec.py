"""Numerical modality, schedule, noise, and integration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .cfg import CfgRecipe
from .schedule import ScheduleDirection, ScheduleShiftDomain


@dataclass(frozen=True, slots=True)
class ScheduleRule:
    """Analytical coordinates or a fixed trained ladder, before tensor casting.

    ``coordinate`` uses flow-match coordinates in the declared direction and
    shift domain. ``one_minus_sigma`` uses a descending, shifted trained sigma
    ladder and casts sigma to FP32 before subtracting it from one.
    """

    direction: ScheduleDirection
    shift_domain: ScheduleShiftDomain
    shift: float
    timestep: Literal["coordinate", "one_minus_sigma"] = "coordinate"
    ladder: tuple[int, ...] = ()
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class ModalitySpec:
    """Global latent representation and the ordered normal draw feeding it.

    ``input`` preserves the caller's bound state dtype; noise declared as
    ``state`` is drawn directly in that dtype, before scaling or patching.
    """

    name: str
    latent_shape: tuple[int, ...]
    noise_shape: tuple[int, ...]
    schedule: ScheduleRule
    prediction: Literal["velocity", "sample"]
    prediction_dtype: torch.dtype
    state_dtype: torch.dtype | Literal["input"] = torch.float32
    noise_dtype: torch.dtype | Literal["state"] = torch.float32
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.latent_shape or not self.noise_shape:
            raise ValueError("diffusion modalities require a name and complete tensor geometry")
        if min(*self.latent_shape, *self.noise_shape) < 1:
            raise ValueError("diffusion modality dimensions must be positive")


@dataclass(frozen=True, slots=True)
class DiffusionSpec:
    """Complete ordered numerical recipe for a homogeneous diffusion call.

    Normal draws use one freshly seeded PyTorch generator per logical row.
    Each modality draws its entire contiguous noise shape in declaration order
    from that same generator, before mathematical sharding or patch transforms.
    ``input`` selects the bound numerical input device; ``cpu`` requires the CPU
    generator independently of where subsequent computation executes.

    ``euler`` casts the time difference to state dtype, multiplies prediction,
    then adds state. ``clean_sample_euler`` casts timestep to state dtype before
    subtraction, forms a clean sample in prediction scratch, computes an FP32
    sigma ratio, multiplies state in place, then uses addcmul for the complement.
    These are the operations implemented by the public integrator primitives.
    """

    modalities: tuple[ModalitySpec, ...]
    steps: int
    cfg: CfgRecipe | None
    max_cfg_branches: int
    solver: Literal["euler", "clean_sample_euler"]
    noise_device: Literal["cpu", "input"]
    seed_transform: Literal["identity", "splitmix_coordinate"]
    noise_algorithm: Literal["torch_normal"] = "torch_normal"

    def __post_init__(self) -> None:
        names = tuple(modality.name for modality in self.modalities)
        if not names or len(set(names)) != len(names):
            raise ValueError("diffusion requires uniquely named ordered modalities")
        if self.steps < 1 or self.max_cfg_branches < 1:
            raise ValueError("diffusion steps and branch bounds must be positive")
        if self.cfg is None and self.max_cfg_branches != 1:
            raise ValueError("unguided diffusion has exactly one branch")
        for modality in self.modalities:
            rule = modality.schedule
            if rule.shift <= 0 or rule.scale <= 0:
                raise ValueError("diffusion schedule scales must be positive")
            if rule.ladder and len(rule.ladder) != self.steps:
                raise ValueError("diffusion step count must match its fixed numerical ladder")
