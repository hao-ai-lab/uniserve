"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import torch

from ..foundation.errors import invalid_descriptor
from ..modeling.components import Call
from ..modeling.image_diffusion import BranchSource, ImageDiffusion
from ..modeling.tensors import TensorViews
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
    tensors: dict[Call, TensorViews] = field(default_factory=dict)
    constants: dict[Call, TensorViews] = field(default_factory=dict)
    scratch: dict[Call, TensorViews] = field(default_factory=dict)
    prefixes: dict[BranchSource, tuple[tuple[int, ...], bool]] = field(default_factory=dict)
    positions: dict[int, tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...]]] = field(
        default_factory=dict
    )
    cache: tuple[int, int, int, int] = (0, 0, 0, 0)
    entries: dict[Branch, tuple[int, int, int, int]] = field(default_factory=dict)


def resolve_prefix(
    generation: ImageDiffusion,
    source: BranchSource,
    *,
    image_prompt: str,
    negative_prompt: str,
    negative_token_ids: tuple[int, ...],
    tokenizer: Any | None,
) -> tuple[tuple[int, ...], bool]:
    """Resolve a branch prefix and flag an empty positive prompt as start-state conditioning."""

    if source is BranchSource.CONDITIONING and not image_prompt.strip():
        return (), True
    if source is BranchSource.NEGATIVE_OR_START and negative_token_ids:
        return negative_token_ids, False
    if generation.prompt is None:
        if source is BranchSource.CONDITIONING:
            raise invalid_descriptor("this model does not accept a generation prompt override")
        return (), False
    if source is BranchSource.CONDITIONING:
        text = image_prompt.strip()
        conditioned = True
    elif source is BranchSource.NEGATIVE_OR_START:
        text = negative_prompt.strip()
        conditioned = False
    else:
        text = ""
        conditioned = False
    return generation.prompt.encode(tokenizer, text=text, conditioned=conditioned), False
