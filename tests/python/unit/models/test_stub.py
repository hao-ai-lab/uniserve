"""The simulator preserves deterministic token, cache and image behavior."""

from dataclasses import replace

import pytest
import torch

from uniserve.media import image
from uniserve.model import LatentInput, TextInput, TextSize, VisionInput
from uniserve.nn.attention import PagedInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models.stub import DenoiserInput, Model

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_token_cycle_projects_selected_rows_and_writes_only_supplied_slots():
    model = Model()
    tokens = torch.tensor(
        [13, 1000, 1001, 151670, 1002, 1003, 1004, 1005, 1006, 1007]
    )
    batch = PagedInput.from_blocks(
        blocks=((0,),),
        query_lengths=(10,),
        prefix_lengths=(0,),
        block_size=16,
        causal=True,
        device="cpu",
    )
    inputs = TextInput(tokens, torch.arange(10), batch)
    with PrefixCache(
        model.cache_config, num_blocks=1, block_size=16, device="cpu"
    ) as cache:
        state = cache.state(next(iter(model.cache_config.layers)))
        state.key.fill_(7)
        state.value.fill_(7)
        with ExecutionContext(model, cache=cache, attention="torch") as context:
            context.prepare(TextSize(10, 1))
            hidden = model(inputs)
            logits = model.compute_logits(
                hidden, token_indices=torch.arange(10)
            ).gather()
            expected = torch.tensor(
                [1000, 1001, 151670, 1002, 1003, 1004, 1005, 1006, 1007, 151645]
            )
            torch.testing.assert_close(logits.argmax(-1), expected)
            selected = torch.tensor([9, 0, 3])
            torch.testing.assert_close(
                model.compute_logits(hidden, token_indices=selected).gather(),
                logits[selected],
            )
            assert model.compute_logits(
                hidden, token_indices=selected[:0]
            ).values.shape == (
                0,
                151671,
            )
            for value in (state.key, state.value):
                torch.testing.assert_close(
                    value[0, :10], torch.zeros_like(value[0, :10])
                )
                torch.testing.assert_close(
                    value[0, 10:], torch.full_like(value[0, 10:], 7)
                )
            state.key.fill_(3)
            model(replace(inputs, attention=replace(batch, write_indices=None)))
            torch.testing.assert_close(state.key, torch.full_like(state.key, 3))


@torch.inference_mode()
def test_zero_velocity_preserves_each_raster_through_solver_and_decoder():
    model = Model()
    sizes = (image.Config(16, 32), image.Config(32, 48))
    pixels = tuple(
        torch.linspace(
            -1, 1, 3 * size.height * size.width, dtype=torch.bfloat16
        ).reshape(1, 3, size.height, size.width)
        for size in sizes
    )
    patches = tuple(model.latent_encoder.encode(value)[0] for value in pixels)
    before = tuple(value.clone() for value in patches)
    schedule = model.denoiser.make_schedules(3, shift=1.0, device="cpu")[
        "image"
    ]
    batch = PagedInput.from_blocks(
        blocks=((0,), (1,)),
        query_lengths=(4, 8),
        prefix_lengths=(0, 0),
        block_size=16,
        causal=False,
        device="cpu",
    )
    batch = replace(batch, write_indices=None)
    for index in range(schedule.num_steps):
        inputs = DenoiserInput(
            {
                "image": tuple(
                    LatentInput(value, schedule.timesteps[index])
                    for value in patches
                )
            },
            sizes,
            schedule.step(index),
            batch,
        )
        predictions = model.denoiser(
            inputs, state={}, constants={}, workspace={}
        )["image"]
        for value, output in zip(patches, predictions, strict=True):
            torch.testing.assert_close(output.tensor, torch.zeros_like(value))
            model.denoiser.solver.step_(
                output.tensor,
                value,
                schedule.timesteps[index],
                schedule.timesteps[index + 1],
                sigma=schedule.sigmas[index],
                next_sigma=schedule.sigmas[index + 1],
            )
    for actual, expected in zip(patches, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        model.image_decoder.decode(patches, sizes=sizes), pixels, strict=True
    ):
        torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)


@torch.inference_mode()
def test_vision_features_are_independent_per_patch_and_sample():
    model = Model()
    first = torch.stack((torch.full((768,), -0.25), torch.full((768,), 0.75)))
    second = torch.full((1, 768), 0.125)
    inputs = VisionInput(
        (first, second),
        (torch.tensor([[1, 2]]), torch.tensor([[1, 1]])),
        ((1, 2), (1, 1)),
    )
    actual = model.vision_encoder.encode(inputs)
    for value, expected in zip(actual, ([-0.25, 0.75], [0.125]), strict=True):
        reference = (
            torch.tensor(expected, dtype=torch.bfloat16)
            .unsqueeze(1)
            .expand(-1, 4)
        )
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
