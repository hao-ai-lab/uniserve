"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.diffusion import Branch, Guidance, Schedule
from uniserve.media.image import Config
from uniserve.processing import BranchSource
from uniserve_worker.model_executor.diffusion_runner import Trajectory


@dataclass(slots=True)
class ImageState:
    """Retain numerical schedules and conditioning for one admitted image.

    Solver samples borrow the pending call's latent staging and are not
    retained here. Accepted progress and product generations belong to the
    request and latent pool; physical prefix coordinates refresh per call.
    """

    size: Config
    schedule: Schedule
    guidance: Guidance
    prefixes: dict[BranchSource, tuple[tuple[int, ...], bool]] = field(
        default_factory=dict
    )
    positions: dict[int, torch.Tensor] = field(default_factory=dict)
    cache: tuple[int, int, int, int] = (0, 0, 0, 0)
    entries: dict[Branch, tuple[int, int, int, int]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class VideoState:
    """Borrow request media tensors while their execution owners retain.

    backing.
    """

    size: Any
    schedules: dict[str, Schedule]
    tensors: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    denoising: Trajectory | None = None
    #: The seeded noise draw started when the request was admitted; latent
    #: preparation reads the noise, and retirement waits, once it completes.
    noise: Future[None] | None = None
