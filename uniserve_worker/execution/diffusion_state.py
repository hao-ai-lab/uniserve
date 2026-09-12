"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import torch

from ..models.generation import BranchSource
from ..nn.diffusion.cfg import Branch
from ..nn.diffusion.schedule import DiffusionSchedule


@dataclass(slots=True)
class DiffusionState:
    """Reuse schedules, conditioning prefixes, geometry, and request tensor views.

    Image latent workspaces belong to the current pending operation and are never
    retained here. Video tensor views borrow the request slot through retirement.
    Accepted step and latent generation remain in RequestProgress and LatentPool.
    Cache coordinates are refreshed from each operation's physical descriptors.
    """

    geometry: Hashable
    timesteps: tuple[tuple[float, float], ...] = ()
    schedule: DiffusionSchedule | None = None
    tensors: object | None = None
    metadata: object | None = None
    prefixes: dict[BranchSource, tuple[tuple[int, ...], bool]] = field(default_factory=dict)
    positions: dict[int, tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...]]] = field(
        default_factory=dict
    )
    cache: tuple[int, int, int, int] = (0, 0, 0, 0)
    entries: dict[Branch, tuple[int, int, int, int]] = field(default_factory=dict)
