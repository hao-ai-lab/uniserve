"""Direct callers use the same numerical modules through public runners."""

import pytest
import torch
from torch import nn

from uniserve.execution import EncoderRunner, ImageRunner, ModelRunner
from uniserve.media import image
from uniserve.model import Encoder, ImageDecoder
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
        output = runner.encode(inputs, size=3)

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


def test_runner_rejects_mismatched_context_and_calls_after_close() -> None:
    model = nn.Identity()
    other = nn.Identity()
    with ExecutionContext(other) as context:
        with pytest.raises(ValueError, match="bound to the runner's model"):
            ModelRunner(model, context=context)

    with ExecutionContext(model) as context:
        runner = ModelRunner(model, context=context)
        runner.close()
        with pytest.raises(RuntimeError, match="runner is closed"):
            runner.warmup(1)
