"""The fixed FastH3 v0.2 video/audio solver schedule."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["FASTH3_LADDER", "H3Schedule", "shifted_sigmas", "solver_step"]

FASTH3_LADDER = (1000, 750, 500, 250)


def shifted_sigmas(shift: float) -> tuple[float, ...]:
    """Map the published 1000-point ladder through one H3 scheduler shift."""

    if shift <= 0:
        raise ValueError("H3 scheduler shift must be positive")
    base = (*((value / 1000.0) for value in FASTH3_LADDER), 0.0)
    return tuple(shift * sigma / (1.0 + (shift - 1.0) * sigma) for sigma in base)


@dataclass(frozen=True, slots=True)
class H3Schedule:
    """Immutable host and device views of the checkpoint-trained ladder."""

    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    video_timesteps: torch.Tensor
    audio_timesteps: torch.Tensor

    @classmethod
    def build(cls, device: torch.device | str) -> "H3Schedule":
        """Materialize the four-step video and audio sigma ladders on one device."""

        video = torch.tensor(shifted_sigmas(12.0), dtype=torch.float32, device=device)
        audio = torch.tensor(shifted_sigmas(3.0), dtype=torch.float32, device=device)
        return cls(
            video_sigmas=video,
            audio_sigmas=audio,
            video_timesteps=1.0 - video[:-1],
            audio_timesteps=1.0 - audio[:-1],
        )

    def __post_init__(self) -> None:
        """Validate host and device schedule lengths against the fixed H3 step count."""

        if self.video_sigmas.shape != (5,) or self.audio_sigmas.shape != (5,):
            raise ValueError("FastH3 requires five sigma points")
        if self.video_timesteps.shape != (4,) or self.audio_timesteps.shape != (4,):
            raise ValueError("FastH3 requires exactly four denoiser evaluations")


def solver_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> None:
    """Apply the checkpoint Euler update in place with FP32 ratio arithmetic."""

    sigma_from_timestep = 1.0 - timestep.to(
        device=sample.device, dtype=sample.dtype
    )
    velocity.mul_(sigma_from_timestep).add_(sample)
    ratio = sigma_next.float() / sigma.float()
    sample.mul_(ratio)
    sample.addcmul_(velocity, 1.0 - ratio)
