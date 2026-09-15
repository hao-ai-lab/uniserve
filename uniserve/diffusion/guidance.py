"""Caller-held classifier-free guidance over analytical schedule coordinates."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch

from .schedule import Schedule


class Branch(Enum):
    CONDITIONED = "conditioned"
    TEXT_UNCONDITIONAL = "text_unconditional"
    IMAGE_UNCONDITIONAL = "image_unconditional"


class Renorm(Enum):
    NONE = "none"
    GLOBAL = "global"
    CHANNEL = "channel"
    TEXT_CHANNEL = "text_channel"
    RESCALE = "rescale"
    CFG_ZERO_STAR = "cfg_zero_star"


def _equal(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=0.0)


def _normalize(guided, reference, renorm, minimum):
    if renorm is Renorm.NONE:
        return guided
    if renorm is Renorm.CFG_ZERO_STAR:
        # Zero-center each sample instead of rescaling its norm.
        return guided - guided.mean(dim=tuple(range(1, guided.ndim)), keepdim=True)

    # GLOBAL reduces over every non-batch axis; the channel modes reduce over
    # the trailing channel axis only.
    dimensions = (
        (tuple(range(1, guided.ndim)) if guided.ndim >= 3 else tuple(range(guided.ndim)))
        if renorm is Renorm.GLOBAL
        else (guided.ndim - 1,)
    )
    epsilon = torch.finfo(guided.dtype).eps
    norm = guided.float().norm(dim=dimensions, keepdim=True).clamp_min(epsilon)
    target = reference.float().norm(dim=dimensions, keepdim=True).clamp_min(minimum)

    # The clamp only shrinks an over-normed guidance toward the reference;
    # RESCALE then interpolates back toward the raw guidance to bound drift.
    result = guided * (target / norm).clamp(max=1.0).to(guided.dtype)
    return 0.7 * result + 0.3 * guided if renorm is Renorm.RESCALE else result


@dataclass(frozen=True, slots=True)
class Guidance:
    """Select and combine the same branches for one supplied numerical step.

    Interval comparisons use the unrounded coordinate. Renormalization is
    bounded by the conditioned prediction; scales belong to this trajectory,
    and never modify the shared model or an execution context.
    """

    text_scale: float
    image_scale: float
    interval: tuple[float, float]
    renorm: Renorm
    renorm_min: float
    _nested: ClassVar[bool] = False

    def __post_init__(self):
        if not isinstance(self.renorm, Renorm):
            raise TypeError("guidance renormalization must use Renorm")
        if (
            not isinstance(self.interval, tuple)
            or len(self.interval) != 2
            or not 0 <= self.interval[0] <= self.interval[1] <= 1
        ):
            raise ValueError("guidance interval must lie in [0, 1]")
        if (
            any(
                not math.isfinite(value)
                for value in (self.text_scale, self.image_scale, self.renorm_min)
            )
            or self.renorm_min < 0
        ):
            raise ValueError("guidance scales must be finite and its norm floor nonnegative")

    def branches(self, schedule: Schedule, index: int) -> tuple[Branch, ...]:
        """Return the model branches this step must evaluate, conditioned first."""

        if type(index) is not int or not 0 <= index < schedule.num_steps:
            raise IndexError(index)

        conditioned = (Branch.CONDITIONED,)
        if not self.interval[0] <= schedule.coordinates[index] <= self.interval[1]:
            return conditioned

        # A scale of 1.0 (within tolerance) makes its unconditional branch
        # redundant; equal text/image scales collapse to one shared branch.
        text_off, image_off = _equal(self.text_scale, 1.0), _equal(self.image_scale, 1.0)
        if text_off and image_off:
            return conditioned
        if image_off:
            return (*conditioned, Branch.TEXT_UNCONDITIONAL)
        if self._nested and not text_off:
            return (*conditioned, Branch.TEXT_UNCONDITIONAL, Branch.IMAGE_UNCONDITIONAL)
        if text_off or _equal(self.text_scale, self.image_scale):
            return (*conditioned, Branch.IMAGE_UNCONDITIONAL)
        return (*conditioned, Branch.TEXT_UNCONDITIONAL, Branch.IMAGE_UNCONDITIONAL)

    def combine(self, outputs, schedule: Schedule, index: int, *, out=None):
        """Combine one step's branch predictions into the guided prediction.

        ``outputs`` maps every branch returned by :meth:`branches` to its
        model prediction. With ``out``, the result is copied into that tensor.
        """

        branches = self.branches(schedule, index)
        if any(branch not in outputs for branch in branches):
            raise ValueError("guidance requires every selected numerical prediction")
        conditioned = outputs[Branch.CONDITIONED]
        if any(
            outputs[branch].shape != conditioned.shape
            or outputs[branch].dtype != conditioned.dtype
            or outputs[branch].device != conditioned.device
            for branch in branches
        ):
            raise ValueError("guidance predictions must have identical representations")

        if len(branches) == 1:
            guided = conditioned
        elif len(branches) == 2:
            # Single-branch CFG: base + scale * (conditioned - base).
            base = outputs[branches[1]]
            scale = (
                self.image_scale
                if self._nested and branches[1] is Branch.IMAGE_UNCONDITIONAL
                else self.text_scale
            )
            guided = _normalize(
                base + scale * (conditioned - base), conditioned, self.renorm, self.renorm_min
            )
        elif self._nested and self.renorm is Renorm.TEXT_CHANNEL:
            # Text guidance with per-channel norms, then image guidance on top.
            base = outputs[Branch.TEXT_UNCONDITIONAL]
            guided = _normalize(
                base + self.text_scale * (conditioned - base),
                conditioned,
                self.renorm,
                self.renorm_min,
            )
            if self.image_scale > 1:
                image = outputs[Branch.IMAGE_UNCONDITIONAL]
                guided = image + self.image_scale * (guided - image)
        else:
            # Combine all three predictions in one weighted sum. The nested
            # coefficients apply image guidance to the text-guided prediction.
            text, image = self.text_scale, self.image_scale
            coefficients = (
                (1.0 - image, image * (1.0 - text), image * text)
                if self._nested
                else (1.0 - image, image - text, text)
            )
            values = torch.stack(
                (
                    outputs[Branch.IMAGE_UNCONDITIONAL],
                    outputs[Branch.TEXT_UNCONDITIONAL],
                    conditioned,
                )
            )
            factors = values.new_tensor(coefficients).reshape(3, *((1,) * conditioned.ndim))
            guided = _normalize(
                (values * factors).sum(0), conditioned, self.renorm, self.renorm_min
            )

        if out is None:
            return guided
        if out.shape != guided.shape or out.dtype != guided.dtype or out.device != guided.device:
            raise ValueError("guidance output must match the computed representation")
        return out.copy_(guided)


class AdditiveGuidance(Guidance):
    """Add image and text deltas around the image-unconditional prediction."""

    _nested = False


class NestedGuidance(Guidance):
    """Apply image guidance to the text-guided prediction."""

    _nested = True
