"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.diffusion import Branch, Guidance, Schedule
from uniserve.model import ImageDenoiser
from uniserve.processing import BranchSource
from uniserve_worker.model_executor.diffusion_runner import Ladder


@dataclass(slots=True)
class KVConditioning:
    """Where a KV-conditioned request's guidance branches read their prefix.

    Solver samples borrow the pending call's latent staging and are not
    retained here. Physical prefix coordinates follow each submission's
    descriptors, so ``cache`` and ``entries`` refresh per call; prefix tokens
    and rotary positions are retained for the request's lifetime.
    """

    # Prefix token ids per source, with a flag that is true when the branch
    # reads the request's own conditioning KV (``cache``) in place.
    prefixes: dict[BranchSource, tuple[tuple[int, ...], bool]] = field(
        default_factory=dict
    )
    # ``ImageBuilder.positions`` coordinates keyed by temporal position.
    positions: dict[int, torch.Tensor] = field(default_factory=dict)
    # The request's conditioning KV as (slot, visible length, token
    # capacity), from ``calls.cache_coordinates``.
    cache: tuple[int, int, int] = (0, 0, 0)
    # Each branch's prefix as (slot, materialized prefix length, token
    # capacity); the slot is the request's own or an alternative-prefix slot.
    entries: dict[Branch, tuple[int, int, int]] = field(default_factory=dict)


@dataclass(slots=True)
class SlotLadder:
    """A standalone request's views of its slot and its bound ladder.

    The views borrow request storage whose execution owners retain the
    backing. The ladder views the samples of its layout's runner and is bound
    again when that runner is replaced. ``staging`` is the host staging
    started at admission (the seeded draw and the request's own state
    tables); latent preparation reads what it filled, and retirement waits
    for it.
    """

    tensors: dict[str, Mapping[str, torch.Tensor]] = field(default_factory=dict)
    ladder: Ladder | None = None
    staging: Future[None] | None = None


@dataclass(slots=True)
class DiffusionState:
    """Numerical state of one admitted diffusion request.

    ``size`` is the admitted numerical size and ``schedules`` the fixed
    schedule of every sample modality. ``guidance`` selects and combines the
    branch predictions of a denoiser evaluated once per branch, and is
    ``None`` for a denoiser with one prediction per step. ``open`` sets
    exactly one of ``kv`` and ``slot``, by how the denoiser is conditioned.
    Accepted progress and product generations belong to the request, not
    this state.
    """

    size: Any
    schedules: Mapping[str, Schedule]
    guidance: Guidance | None = None
    kv: KVConditioning | None = None
    slot: SlotLadder | None = None

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
        """Build a request's state from its admitted sampling parameters.

        An ``ImageDenoiser`` attends to the request's KV prefixes; any other
        denoiser advances a ladder over its request slot.
        """
        attends = isinstance(denoiser, ImageDenoiser)
        return cls(
            size,
            dict(denoiser.make_schedules(steps, shift=shift, device=device)),
            guidance,
            kv=KVConditioning() if attends else None,
            slot=None if attends else SlotLadder(),
        )
