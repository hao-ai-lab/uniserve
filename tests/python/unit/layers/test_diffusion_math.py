"""Diffusion trajectories retain analytical selection and draw order.

They also retain solver math.
"""

import pytest
import torch

from uniserve.diffusion import (
    AdditiveGuidance,
    Branch,
    CleanSampleEulerSolver,
    EulerSolver,
    NestedGuidance,
    NoiseScale,
    Renorm,
    make_schedule,
    normal_noise,
)

pytestmark = pytest.mark.unit


def test_guidance_interval_uses_unrounded_coordinate():
    schedule = make_schedule(
        3, shift=1.0, direction="ascending", shift_domain="time", device="cpu"
    )
    guidance = AdditiveGuidance(4.0, 1.0, (1 / 3, 1 / 3), Renorm.NONE, 0.0)
    assert guidance.branches(schedule, 1) == (
        Branch.CONDITIONED,
        Branch.TEXT_UNCONDITIONAL,
    )
    assert guidance.branches(schedule, 0) == (Branch.CONDITIONED,)
    assert guidance.branches(schedule, 2) == (Branch.CONDITIONED,)
    with pytest.raises(IndexError):
        guidance.branches(schedule, 3)


@pytest.mark.parametrize(
    "guidance,expected",
    [
        (AdditiveGuidance, [[15.0, -5.0], [5.0, 17.0]]),
        (NestedGuidance, [[27.0, -11.0], [11.0, 29.0]]),
    ],
)
def test_guidance_combines_text_and_image_equations(guidance, expected):
    schedule = make_schedule(
        1, shift=1.0, direction="ascending", shift_domain="time", device="cpu"
    )
    # C = [[5, -1], [3, 7]], T = [[1, 1], [1, 3]], I = [[-1, 1], [3, 1]].
    # Additive: I + 2(T-I) + 3(C-T); nested: I + 2(T + 3(C-T) - I).
    outputs = {
        Branch.CONDITIONED: torch.tensor([[5.0, -1.0], [3.0, 7.0]]),
        Branch.TEXT_UNCONDITIONAL: torch.tensor([[1.0, 1.0], [1.0, 3.0]]),
        Branch.IMAGE_UNCONDITIONAL: torch.tensor([[-1.0, 1.0], [3.0, 1.0]]),
    }
    options = guidance(3.0, 2.0, (0.0, 1.0), Renorm.NONE, 0.0)
    out = torch.empty(2, 4)[:, ::2]
    assert options.combine(outputs, schedule, 0, out=out) is out
    torch.testing.assert_close(out, torch.tensor(expected), rtol=0, atol=0)


def test_guidance_channel_norm_does_not_expand_predictions():
    schedule = make_schedule(
        1, shift=1, direction="ascending", shift_domain="time", device="cpu"
    )
    options = AdditiveGuidance(2, 1, (0, 1), Renorm.CHANNEL, 0)
    outputs = {
        Branch.CONDITIONED: torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        Branch.TEXT_UNCONDITIONAL: torch.zeros(2, 2),
    }
    torch.testing.assert_close(
        options.combine(outputs, schedule, 0),
        outputs[Branch.CONDITIONED],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "direction,domain,expected",
    [
        ("ascending", "time", (0.0, 0.75, 1.0)),
        ("ascending", "sigma", (0.0, 0.25, 1.0)),
        ("descending", "time", (1.0, 0.75, 0.0)),
        ("descending", "sigma", (1.0, 0.25, 0.0)),
    ],
)
def test_shifted_schedule_has_complete_endpoints(direction, domain, expected):
    schedule = make_schedule(
        2, shift=3, direction=direction, shift_domain=domain, device="cpu"
    )
    assert schedule.coordinates == expected
    assert schedule.num_steps == 2
    torch.testing.assert_close(
        schedule.sigmas[[0, -1]], torch.tensor([1.0, 0.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_euler_and_clean_sample_predictions_update_the_same_trajectory(dtype):
    sample = torch.tensor([2.0, -1.0], dtype=dtype)
    velocity = torch.tensor([1.0, 2.0], dtype=dtype)
    clean = torch.tensor([2.5, 0.0], dtype=dtype)
    for kind, prediction in (("velocity", velocity), ("sample", clean)):
        output = sample.clone()
        EulerSolver(kind).step_(
            prediction,
            output,
            torch.tensor(0.5),
            torch.tensor(0.75),
            sigma=torch.tensor(0.5),
            next_sigma=torch.tensor(0.25),
        )
        torch.testing.assert_close(
            output, torch.tensor([2.25, -0.5], dtype=dtype), rtol=0, atol=0
        )
    output = sample.clone()
    prediction = velocity.clone()
    CleanSampleEulerSolver().step_(
        prediction,
        output,
        torch.tensor(0.5),
        torch.tensor(0.75),
        sigma=torch.tensor(0.5),
        next_sigma=torch.tensor(0.25),
    )
    torch.testing.assert_close(
        output, torch.tensor([2.25, -0.5], dtype=dtype), rtol=0, atol=0
    )
    torch.testing.assert_close(prediction, clean, rtol=0, atol=0)


def test_noise_preserves_native_draw_order_and_global_rng():
    state = torch.random.get_rng_state().clone()
    first = torch.empty(2, 3, 8)[:, :, ::2]
    second = torch.empty(2, 5)
    normal_noise((83, 91), out=(first, second))
    for index, seed in enumerate((83, 91)):
        generator = torch.Generator().manual_seed(seed)
        torch.testing.assert_close(
            first[index], torch.randn(3, 4, generator=generator), rtol=0, atol=0
        )
        torch.testing.assert_close(
            second[index], torch.randn(5, generator=generator), rtol=0, atol=0
        )
    assert torch.equal(torch.random.get_rng_state(), state)
    assert NoiseScale(2, "resolution", 16, 9).scale(64) == 4
    assert NoiseScale(2, "dynamic_sqrt", 16, 9).scale(64) == 2
    assert NoiseScale(2, "constant", 16, 1).scale(64) == 1


def test_patch_order_preserves_pixels_then_channels():
    from uniserve.media import image
    from uniserve.nn.functional import patchify, unpatchify

    pixels = torch.arange(16).reshape(2, 2, 4)
    expected = torch.tensor(
        [[0, 8, 1, 9, 4, 12, 5, 13], [2, 10, 3, 11, 6, 14, 7, 15]]
    )
    patches = patchify(pixels, patch_size=2)
    torch.testing.assert_close(patches, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        unpatchify(patches, image.Config(2, 4), patch_size=2, channels=2),
        pixels,
        rtol=0,
        atol=0,
    )


def test_public_image_step_uses_borrowed_sample_and_input_time():
    from uniserve.diffusion import DenoisingStep
    from uniserve.media import image
    from uniserve.model import DenoiserInput, ImageDenoiser, LatentInput
    from uniserve.tensors import OutputLayout, TensorOutput

    class Expansion(ImageDenoiser):
        def __init__(self):
            super().__init__(
                patch_size=2,
                latent_channels=1,
                downsample=2,
                noise_scale=NoiseScale(2, "constant", 1, 3),
                prediction_dtype=torch.float32,
                solver=EulerSolver(),
            )

        def make_schedules(self, steps, *, shift, device):
            return {
                "image": make_schedule(
                    steps,
                    shift=1 if shift is None else shift,
                    direction="ascending",
                    shift_domain="time",
                    device=device,
                )
            }

        def forward(self, inputs, *, state, constants, workspace):
            outputs = []
            for latent in inputs.latents["image"]:
                prediction = 2 * latent.tensor
                layout = OutputLayout(
                    tuple(prediction.shape),
                    prediction.dtype,
                    local_slice=tuple(
                        slice(0, width) for width in prediction.shape
                    ),
                )
                outputs.append(TensorOutput(prediction, layout))
            return {"image": tuple(outputs)}

    network = Expansion()
    size = image.Config(2, 4)
    noise = torch.ones(1, *network.noise_shape("image", size))
    sample = torch.empty(1, *network.latent_shape("image", size))
    network.prepare_latents(
        (size,),
        noise={"image": noise},
        state={"image": sample},
        constants={},
        workspace={},
    )
    torch.testing.assert_close(
        sample, torch.full_like(sample, 2), rtol=0, atol=0
    )
    schedules = network.make_schedules(1, shift=None, device="cpu")
    inputs = DenoiserInput(
        {"image": (LatentInput(sample[0], torch.tensor(0.5)),)},
        (size,),
        schedules["image"].step(0),
    )
    step = DenoisingStep(
        network,
        inputs,
        schedules,
        {"image": sample},
        {},
        {},
    )
    result = step()
    assert result["image"][0] is inputs.latents["image"][0].tensor
    # dx/dt = 2x evaluated at x=2, from supplied t=1/2 to the terminal t=1.
    torch.testing.assert_close(
        sample, torch.full_like(sample, 4), rtol=0, atol=0
    )
    torch.testing.assert_close(noise, torch.ones_like(noise), rtol=0, atol=0)


def test_image_decoder_restores_different_rasters_in_input_order():
    from uniserve.media import image
    from uniserve.model import ImageDecoder
    from uniserve.nn import RGBDecoder
    from uniserve.nn.functional import patchify

    sizes = (image.Config(2, 4), image.Config(4, 2), image.Config(2, 4))
    pixels = tuple(
        torch.arange(3 * size.height * size.width)
        .reshape(3, size.height, size.width)
        .float()
        / 20
        + index
        for index, size in enumerate(sizes)
    )
    patches = tuple(patchify(value, patch_size=2) for value in pixels)
    decoded = ImageDecoder(RGBDecoder(2)).decode(patches, sizes=sizes)
    for actual, expected in zip(decoded, pixels, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
