"""Every admitted H3 size evaluates in a provisioned capacity layout."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.h3 import WIDE, base_config, fasth3_config
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve_models.minimax_h3 import Model
from uniserve_worker.model_executor.media_inputs import MediaBuilder

pytestmark = pytest.mark.unit

# The 16 output lengths the video API admits: 4 to 15 seconds at 24 fps,
# each extended to a complete native temporal window.
ADMITTED_FRAMES = tuple(107 + 17 * index for index in range(16))


@pytest.fixture(scope="module")
def denoiser():
    with torch.device("meta"):
        return Model(fasth3_config()).denoiser


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
            size = builder.size(frames, tokens, WIDE)
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
            builder.size(frames, tokens, WIDE)


def test_default_capacities_step_to_the_prompt_capacity(denoiser):
    builder = MediaBuilder(denoiser, max_frames=124, max_text_tokens=5000)
    assert builder.text_capacities == (1024, 2048, 4096, 5056)
    # A prompt capacity below the first rung is the only capacity.
    small = MediaBuilder(denoiser, max_frames=124, max_text_tokens=500)
    assert small.text_capacities == (512,)
    # Without a floor, every native frame count up to the capacity is kept.
    assert builder.frame_counts == (22, 39, 56, 73, 90, 107, 124)


def test_a_layout_holds_only_prompts_that_fit_at_its_frame_count(denoiser):
    def make(frames, tokens):
        return denoiser.make_size(frames, tokens, canvas=WIDE)

    layout = make(124, 2048)
    assert denoiser.holds(layout, make(124, 1))
    assert denoiser.holds(layout, make(124, 2048))
    assert not denoiser.holds(layout, make(124, 2049))
    assert not denoiser.holds(layout, make(141, 64))
    # A text region that is not whole tiles is not a layout.
    assert not denoiser.holds(make(124, 100), make(124, 64))


def test_every_named_canvas_has_layouts_that_fit_the_slot_storage():
    """A dense denoiser prepares every named canvas within one slot.

    The maximum, prepared first, bounds every layout's row-shaped workspace,
    and the slot storage holds every layout's buffers dimension by dimension.
    """
    with torch.device("meta"):
        denoiser = Model(base_config()).denoiser
    builder = MediaBuilder(
        denoiser,
        max_frames=360,
        max_text_tokens=4096,
        min_frames=96,
        text_capacities=(1024, 4096),
    )
    canvases = {(canvas.width, canvas.height) for canvas in builder.canvases}
    assert canvases == {
        (1536, 672),
        (1344, 768),
        (1024, 768),
        (768, 768),
        (768, 1024),
        (768, 1344),
    }
    layouts = builder.layouts()
    assert layouts[0] == builder.maximum
    assert len(layouts) == len(canvases) * len(ADMITTED_FRAMES) * 2
    capacity = builder.capacity_buffers()
    maximum = denoiser.workspace_buffers(builder.maximum)
    for layout in layouts:
        size = builder.size(layout.num_frames, 1000, layout.canvas)
        assert builder.layout(size) in layouts
        for name, config in builder.buffers(size).items():
            assert all(
                extent <= bound
                for extent, bound in zip(
                    config.shape, capacity[name].shape, strict=True
                )
            ), name
        for name, config in denoiser.workspace_buffers(layout).items():
            assert all(
                extent <= bound
                for extent, bound in zip(
                    config.shape, maximum[name].shape, strict=True
                )
            ), name
    with pytest.raises(ValueError):
        builder.size(124, 64, image.Config(512, 2016))


def test_a_condition_capacity_bounds_each_conditioned_layout():
    """Requests with conditions evaluate in their own bounded layouts.

    A worker serving conditions sizes the slot storage and the layout its
    runner prepares first by its condition capacity; each conditioned
    request takes the smallest text capacity with its own condition tiles,
    and text-only requests keep the prepared ladder.
    """
    with torch.device("meta"):
        denoiser = Model(base_config()).denoiser
    first = Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE))
    plain = MediaBuilder(
        denoiser, max_frames=124, max_text_tokens=4096, min_frames=96
    )
    builder = MediaBuilder(
        denoiser,
        max_frames=124,
        max_text_tokens=4096,
        min_frames=96,
        condition_rows=2048,
    )
    # Without conditions the largest layout bounds every other.
    assert plain.maximum_layout == plain.layouts()[0]
    assert builder.layouts() == plain.layouts()
    assert builder.maximum_layout.condition_rows == 2048

    # One 16:9 keyframe is 1008 rows, 16 whole tiles, beside 100 tokens in
    # the 1024-token rung.
    size = builder.size(124, 100, WIDE, conditions=(first,))
    layout = builder.layout(size)
    assert layout not in builder.layouts()
    assert (layout.num_text_tokens, layout.condition_rows) == (1024, 1024)
    assert denoiser.holds(layout, size)
    assert builder.layout(builder.size(124, 100, WIDE)) in builder.layouts()

    # The runner's maximum bounds the conditioned layout's workspace, the
    # slot storage holds its buffers, and the retained conditioning holds
    # the widened maximum's rows.
    maximum = denoiser.workspace_buffers(builder.maximum_layout)
    for name, config in denoiser.workspace_buffers(layout).items():
        assert all(
            extent <= bound
            for extent, bound in zip(
                config.shape, maximum[name].shape, strict=True
            )
        ), name
    capacity = builder.capacity_buffers()
    for name, config in builder.buffers(size).items():
        assert all(
            extent <= bound
            for extent, bound in zip(
                config.shape, capacity[name].shape, strict=True
            )
        ), name
    assert capacity["text_condition"].shape[0] == denoiser.text_condition_rows(
        builder.maximum_layout
    )
    assert capacity["condition_noise"].shape[0] == (
        denoiser.condition_noise_capacity(builder.maximum_layout)
    )
    assert plain.capacity_buffers()["condition_noise"].shape == (0,)

    # Three keyframe rows sets exceed the 2048-row capacity.
    with pytest.raises(ValueError, match="condition capacity"):
        builder.size(124, 100, WIDE, conditions=(first, first, first))
    with pytest.raises(ValueError, match="condition capacity"):
        plain.size(124, 100, WIDE, conditions=(first,))
