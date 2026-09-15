"""Euler integration with explicit prediction and sample mutation behavior."""

from typing import Literal, TypeAlias


def _output(value, out):
    if out is None:
        return value
    if out.shape != value.shape or out.dtype != value.dtype or out.device != value.device:
        raise ValueError("solver output must match the computed shape, dtype and device")
    return out.copy_(value)


def euler_step(sample, velocity, timestep, next_timestep, *, out=None):
    """Cast the time difference to sample precision before the Euler update."""
    delta = (next_timestep - timestep).to(dtype=sample.dtype, device=sample.device)
    return _output(sample + delta * velocity, out)


def clean_sample_to_velocity(prediction, sample, timestep, *, out=None):
    """Convert an ascending clean-time prediction with a 1e-6 endpoint floor."""
    denominator = (1.0 - timestep).clamp_min(1e-6)
    while denominator.ndim < sample.ndim:
        denominator = denominator.unsqueeze(-1)
    result = (prediction - sample) / denominator.to(dtype=sample.dtype, device=sample.device)
    return _output(result, out)


class EulerSolver:
    def __init__(self, prediction_type: Literal["velocity", "sample"] = "velocity"):
        if prediction_type not in {"velocity", "sample"}:
            raise ValueError("Euler prediction must be velocity or sample")
        self.prediction_type = prediction_type

    def step_(self, prediction, sample, timestep, next_timestep, *, sigma, next_sigma) -> None:
        velocity = (
            clean_sample_to_velocity(prediction, sample, timestep)
            if self.prediction_type == "sample"
            else prediction
        )
        # Prediction precision can exceed the retained sample precision (for
        # example an FP32 image velocity with a BF16 trajectory). Preserve the
        # promoted Euler arithmetic, then round once when committing the sample.
        sample.copy_(euler_step(sample, velocity, timestep, next_timestep))


class CleanSampleEulerSolver:
    """Consume velocity as clean-prediction scratch and update sample in place."""

    prediction_type = "velocity"

    def step_(self, prediction, sample, timestep, next_timestep, *, sigma, next_sigma) -> None:
        # Clean time is rounded in sample precision before subtraction, whereas
        # the sigma ratio retains FP32. Their order is part of the solver math.
        clean_time = 1.0 - timestep.to(device=sample.device, dtype=sample.dtype)
        prediction.mul_(clean_time).add_(sample)
        ratio = next_sigma.float() / sigma.float()
        sample.mul_(ratio)
        sample.addcmul_(prediction, 1.0 - ratio)


Solver: TypeAlias = EulerSolver | CleanSampleEulerSolver
