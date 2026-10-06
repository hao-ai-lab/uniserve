"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.diffusion import Guidance, Schedule


@dataclass(slots=True)
class DiffusionState:
    """Numerical state of one admitted diffusion request.

    ``size`` is the admitted numerical size and ``schedules`` the fixed
    schedule of every sample modality. ``guidance`` selects and combines the
    branch predictions of a denoiser evaluated once per branch, and is
    ``None`` for a denoiser with one prediction per step. Request slots,
    bound execution and host preparation belong to the native request.
    Image denoisers cache numerical position tensors here.
    Accepted progress and product generations belong to the request, not
    this state.
    """

    size: Any
    schedules: Mapping[str, Schedule]
    guidance: Guidance | None = None
    positions: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        denoiser,
        size,
        *,
        steps: int,
        shift: float | None,
        device: torch.device | str,
        guidance: Guidance | None = None,
    ) -> DiffusionState:
        """Build schedules from admitted numerical sampling parameters."""
        return cls(
            size,
            dict(denoiser.make_schedules(steps, shift=shift, device=device)),
            guidance,
        )
