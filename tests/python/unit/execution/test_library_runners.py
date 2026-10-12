"""Direct callers use the same numerical modules through public runners."""

import pytest
import torch
from torch import nn

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from uniserve.execution import DenoisingRunner, ModelRunner
from uniserve.model import DenoiserInput, LatentInput
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


def test_denoising_runner_advances_prepared_samples() -> None:
    model = LinearDenoiser()
    size = Size(4, 2.0)
    schedules = model.make_schedules(2, shift=1.0, device="cpu")
    state = {"image": torch.empty(4)}

    with ExecutionContext(model) as context:
        runner = DenoisingRunner(model, context=context)
        runner.warmup(model.layout_size(size))
        runner.prepare_latents(
            (size,),
            noise={"image": torch.full((1, 4), 7.0)},
            state={"image": state["image"].unsqueeze(0)},
        )
        # The linear denoiser's state is its samples alone.
        runner.prepare_state((size,), layouts=(size,), out={})
        for step in range(2):
            timestep = schedules["image"].timesteps[step]
            runner.step(
                DenoiserInput(
                    {"image": (LatentInput(state["image"], timestep),)},
                    (size,),
                    schedules["image"].step(step),
                ),
                schedules,
                state=state,
            )

    # Each unit-shift step moves half way along velocity 0.25 x + offset.
    expected = torch.full((4,), 7.0)
    for _ in range(2):
        expected.add_(0.5 * (expected * 0.25 + size.offset))
    torch.testing.assert_close(state["image"], expected)


def test_runner_rejects_mismatched_and_closed_contexts() -> None:
    model = nn.Identity()
    other = nn.Identity()
    with ExecutionContext(other) as context:
        with pytest.raises(ValueError, match="bound to the runner's model"):
            ModelRunner(model, context=context)

    with ExecutionContext(model) as context:
        runner = ModelRunner(model, context=context)
    with pytest.raises(RuntimeError, match="execution context is closed"):
        runner.warmup(1)
