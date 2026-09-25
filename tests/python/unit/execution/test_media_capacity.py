"""Every admitted H3 size evaluates in a provisioned capacity layout."""

from __future__ import annotations

import pytest
import torch

from uniserve_models.minimax_h3 import Config, Model
from uniserve_worker.model_executor.media_inputs import MediaBuilder

pytestmark = pytest.mark.unit

# The 16 output lengths the video API admits: 4 to 15 seconds at 24 fps,
# each extended to a complete native temporal window.
ADMITTED_FRAMES = tuple(107 + 17 * index for index in range(16))


@pytest.fixture(scope="module")
def denoiser():
    with torch.device("meta"):
        return Model(Config()).denoiser


def test_capacity_layouts_cover_every_admitted_size(denoiser):
    builder = MediaBuilder(
        denoiser,
        max_frames=360,
        max_text_tokens=16384,
        min_frames=96,
        text_capacities=(1000, 4096, 16384),
    )
    assert builder.frame_counts == ADMITTED_FRAMES
    # Capacities are whole 64-row text tiles.
    assert builder.text_capacities == (1024, 4096, 16384)

    layouts = builder.layouts()
    assert len(layouts) == len(ADMITTED_FRAMES) * 3
    maximum = layouts[0]
    assert (maximum.num_frames, maximum.num_text_tokens) == (362, 16384)

    for frames in ADMITTED_FRAMES:
        for tokens in (1, 63, 64, 1000, 1024, 1025, 4096, 4097, 9999, 16384):
            size = builder.size(frames, tokens)
            layout = builder.layout(size)
            assert layout in layouts
            assert denoiser.holds(layout, size)
            # The smallest capacity that holds the prompt is chosen.
            assert layout.num_text_tokens == min(
                capacity
                for capacity in builder.text_capacities
                if capacity >= tokens
            )


def test_sizes_outside_the_admitted_range_have_no_layout(denoiser):
    builder = MediaBuilder(
        denoiser, max_frames=240, max_text_tokens=2048, min_frames=96
    )
    # 10 seconds is 240 frames, extended to 243.
    assert builder.frame_counts[-1] == 243
    for frames, tokens in ((90, 64), (260, 64), (243, 2049)):
        with pytest.raises(ValueError):
            builder.size(frames, tokens)


def test_default_capacities_step_to_the_prompt_capacity(denoiser):
    builder = MediaBuilder(denoiser, max_frames=124, max_text_tokens=5000)
    assert builder.text_capacities == (1024, 2048, 4096, 5056)
    # A prompt capacity below the first rung is the only capacity.
    small = MediaBuilder(denoiser, max_frames=124, max_text_tokens=500)
    assert small.text_capacities == (512,)
    # Without a floor, every native frame count up to the capacity is kept.
    assert builder.frame_counts == (22, 39, 56, 73, 90, 107, 124)


def test_a_layout_holds_only_prompts_that_fit_at_its_frame_count(denoiser):
    make = denoiser.make_size
    layout = make(124, 2048)
    assert denoiser.holds(layout, make(124, 1))
    assert denoiser.holds(layout, make(124, 2048))
    assert not denoiser.holds(layout, make(124, 2049))
    assert not denoiser.holds(layout, make(141, 64))
    # A text region that is not whole tiles is not a layout.
    assert not denoiser.holds(make(124, 100), make(124, 64))
