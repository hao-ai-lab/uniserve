"""Timestep-aware denoise residual reuse (TeaCache) — system-owned policy.

Flow-matching denoise steps change slowly between adjacent timesteps: when the
(normalized) step input barely moves, the transformer stack's *residual*
(output hidden minus input embeddings) barely moves either. This module owns
the reuse policy: it accumulates a polynomial-rescaled relative-L1 distance of
the model-supplied decision embedding across steps and, while the accumulator
stays under threshold, replays the previous step's residual instead of running
the backbone (`hidden ≈ input_embeds + prev_residual`).

Models participate only through a small adapter (:class:`DenoiseResidualCacheAdapter`)
that supplies the decision embedding (e.g. the layer-0 generation-branch input
norm of the step embeddings) and the rescale polynomial calibrated for that
model. No adapter → the cache never engages.

The policy is opt-in (`UNISERVE_DENOISE_RESIDUAL_CACHE=1`) because reuse is an
approximation: outputs differ slightly from the exact trajectory. The
threshold (`UNISERVE_DENOISE_RESIDUAL_CACHE_THRESHOLD`, default 0.2) trades
speed for fidelity; reference guidance for this family: 0.2 ≈ 1.5×, 0.4 ≈
1.8×, 0.6 ≈ 2.0× denoise speedup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from ..foundation.env import env_flag, env_str

__all__ = [
    'DenoiseResidualCacheAdapter',
    'DenoiseResidualCachePolicy',
    'ImageResidualCacheState',
    'resolve_denoise_residual_cache_policy',
]


@dataclass(frozen=True)
class DenoiseResidualCacheAdapter:
    """Model-supplied hooks: decision embedding + calibrated rescale polynomial.

    The cached residual lives in the *pre-final-norm* residual stream (where
    its magnitude dwarfs the input-embedding delta, so replaying
    ``embeds' + residual`` is a faithful approximation); ``finalize_hidden``
    re-applies whatever the model's backbone does after its block stack
    (typically the final RMSNorm of the generation branch) before the
    hidden→velocity head consumes the replayed stream.
    """

    # Maps the step's input embeddings [B, N, C] to the decision embedding the
    # distance metric runs on (typically a cheap norm of the first row).
    decision_embedding: Callable[[torch.Tensor], torch.Tensor]
    # Polynomial coefficients (highest degree first) rescaling the raw
    # relative-L1 distance into accumulated skip budget, calibrated per model.
    rescale_coefficients: tuple[float, ...]
    # Post-block-stack finalization applied to a replayed pre-norm stream.
    finalize_hidden: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DenoiseResidualCachePolicy:
    enabled: bool
    threshold: float

    def active(self, adapter: DenoiseResidualCacheAdapter | None) -> bool:
        return self.enabled and adapter is not None


def resolve_denoise_residual_cache_policy() -> DenoiseResidualCachePolicy:
    enabled = env_flag("UNISERVE_DENOISE_RESIDUAL_CACHE", default=False)
    raw = env_str("UNISERVE_DENOISE_RESIDUAL_CACHE_THRESHOLD", default="0.2")
    try:
        threshold = float(raw)
    except ValueError:
        threshold = 0.2
    return DenoiseResidualCachePolicy(enabled=enabled, threshold=max(0.0, threshold))


def _poly_eval(coefficients: tuple[float, ...], x: float) -> float:
    value = 0.0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


@dataclass
class ImageResidualCacheState:
    """Per-image reuse state: decision history + per-branch residuals.

    One instance rides on the :class:`ImageState` for the image being
    denoised, so its lifetime (and memory) ends with the image commit.
    """

    threshold: float
    coefficients: tuple[float, ...]
    accumulated: float = 0.0
    previous_decision: torch.Tensor | None = None
    residuals: dict[str, torch.Tensor] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def decide_reuse(self, decision: torch.Tensor, branches: tuple[str, ...]) -> bool:
        """Advance the decision stream; True when every branch can be replayed."""
        previous = self.previous_decision
        self.previous_decision = decision
        if previous is None or previous.shape != decision.shape:
            self.accumulated = 0.0
            self.misses += 1
            return False
        denom = previous.abs().mean()
        if float(denom) == 0.0:
            self.misses += 1
            return False
        rel_l1 = float((decision - previous).abs().mean() / denom)
        self.accumulated += _poly_eval(self.coefficients, rel_l1)
        if self.accumulated >= self.threshold:
            self.accumulated = 0.0
            self.misses += 1
            return False
        for branch in branches:
            if branch not in self.residuals:
                self.misses += 1
                return False
        self.hits += 1
        return True

    def replay(self, branch: str, input_embeds: torch.Tensor) -> torch.Tensor:
        return input_embeds + self.residuals[branch]

    def record(self, branch: str, input_embeds: torch.Tensor, hidden: torch.Tensor) -> None:
        self.residuals[branch] = (hidden - input_embeds).detach()

    def invalidate(self) -> None:
        """Drop replay state (e.g. the step ran outside this policy's view)."""
        self.previous_decision = None
        self.accumulated = 0.0
        self.residuals.clear()

    def stats(self) -> Mapping[str, int]:
        return {"hits": self.hits, "misses": self.misses}
