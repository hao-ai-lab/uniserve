"""Direct callers use the same numerical modules through public runners."""

import pytest
import torch
from torch import nn

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from uniserve.execution import (
    DenoisingRunner,
    EncoderRunner,
    ImageRunner,
    ModelRunner,
)
from uniserve.media import image
from uniserve.model import DenoiserInput, Encoder, ImageDecoder, LatentInput
from uniserve.nn.vae import RGBDecoder
from uniserve.runtime import ExecutionContext

pytestmark = pytest.mark.unit


def test_encoder_runner_preserves_homogeneous_sample_order() -> None:
    model = Encoder(nn.Identity())
    inputs = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(3, dtype=torch.float32).reshape(1, 3),
        torch.arange(6, 12, dtype=torch.float32).reshape(2, 3),
    )

    with ExecutionContext(model) as context:
        runner = EncoderRunner(model, context=context)
        runner.warmup(3)
        output = runner.encode(inputs)

    assert output is not None
    for actual, expected in zip(output, inputs, strict=True):
        torch.testing.assert_close(actual, expected)


def test_image_runner_decodes_canonical_patch_rows() -> None:
    model = ImageDecoder(RGBDecoder(patch_size=1))
    size = image.Config(2, 2)
    latent = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    with ExecutionContext(model) as context:
        runner = ImageRunner(model, context=context)
        (pixels,) = runner.decode((latent,), sizes=(size,))

    expected = latent.reshape(2, 2, 3).permute(2, 0, 1)
    torch.testing.assert_close(pixels, expected)


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
        runner.prepare_state((size,), out={})
        for step in range(2):
            timestep = schedules["image"].timesteps[step]
            runner.step(
                DenoiserInput(
                    {"image": (LatentInput(state["image"], timestep),)},
                    (size,),
                    step,
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
