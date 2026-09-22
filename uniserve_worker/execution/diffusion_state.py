"""Numerical state retained by one admitted diffusion request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from uniserve.diffusion import Branch, Guidance, Schedule
from uniserve.media.image import Config
from uniserve.processing import BranchSource, FlowPrompt

from ..foundation.errors import invalid_descriptor
from .denoising_runner import Trajectory


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


def resolve_prefix(
    prompt: FlowPrompt | None,
    source: BranchSource,
    *,
    image_prompt: str,
    negative_prompt: str,
    negative_token_ids: tuple[int, ...],
    tokenizer: Any | None,
) -> tuple[tuple[int, ...], bool]:
    """Resolve a branch prefix and detect empty positive-prompt conditioning."""
    if source is BranchSource.CONDITIONING and not image_prompt.strip():
        return (), True
    if source is BranchSource.NEGATIVE_OR_START and negative_token_ids:
        return negative_token_ids, False

    if prompt is None:
        if source is BranchSource.CONDITIONING:
            raise invalid_descriptor(
                "this model does not accept a generation prompt override"
            )
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
    return prompt.encode(tokenizer, text=text, conditioned=conditioned), False
