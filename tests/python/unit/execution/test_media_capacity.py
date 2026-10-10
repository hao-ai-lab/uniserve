"""Every admitted H3 size evaluates in a provisioned capacity layout."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.h3 import (
    WIDE,
    base_config,
    fasth3_config,
    omniref_denoiser,
)
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole
from uniserve_models.minimax_h3 import Model
from uniserve_models.minimax_h3 import config as h3_config
from uniserve_models.minimax_h3.processing import reference_image_size
from uniserve_worker.model_executor.media_inputs import MediaBuilder

pytestmark = pytest.mark.unit

# The 16 output lengths the video API admits: 4 to 15 seconds at 24 fps,
# each extended to a complete native temporal window.
ADMITTED_FRAMES = tuple(107 + 17 * index for index in range(16))

# The default FastH3 deployment's canvases: 768p 16:9 and 9:16.
TALL = image.Config(1344, 768)
SERVED = (WIDE, TALL)

# FastH3's (height, width) training buckets: 21:9, 16:9, 4:3, 1:1, 3:4 and
# 9:16 at 768p, then at 480p.
BUCKETS = tuple(
    image.Config(height, width)
    for height, width in (
        (672, 1536),
        (768, 1344),
        (768, 1024),
        (768, 768),
        (1024, 768),
        (1344, 768),
        (416, 992),
        (480, 832),
        (480, 640),
        (480, 480),
        (640, 480),
        (832, 480),
    )
)


@pytest.fixture(scope="module")
def denoiser():
    with torch.device("meta"):
        return Model(fasth3_config()).transformer


def test_capacity_layouts_cover_every_admitted_size(denoiser):
    builder = MediaBuilder(
        denoiser,
        max_frames=360,
        max_text_tokens=16384,
        min_frames=96,
        text_capacities=(1000, 4096, 16384),
        canvases=SERVED,
    )
    assert builder.frame_counts == ADMITTED_FRAMES
    # Capacities are whole 64-row text tiles.
    assert builder.text_capacities == (1024, 4096, 16384)

    layouts = builder.layouts()
    assert len(layouts) == len(SERVED) * len(ADMITTED_FRAMES) * 3
    maximum = layouts[0]
    assert (maximum.num_frames, maximum.num_text_tokens) == (362, 16384)
    assert {layout.canvas for layout in layouts} == set(SERVED)

    for canvas in SERVED:
        for frames in ADMITTED_FRAMES:
            for tokens in (1, 63, 64, 1000, 1024, 1025, 4097, 16384):
                size = builder.size(frames, tokens, canvas)
                layout = builder.layout(size)
                assert layout in layouts
                assert layout.canvas == canvas
                assert denoiser.holds(layout, size)
                # The smallest capacity that holds the prompt is chosen.
                assert layout.num_text_tokens == min(
                    capacity
                    for capacity in builder.text_capacities
                    if capacity >= tokens
                )


def test_sizes_outside_the_admitted_range_have_no_layout(denoiser):
    builder = MediaBuilder(
        denoiser,
        max_frames=240,
        max_text_tokens=2048,
        min_frames=96,
        canvases=(WIDE,),
    )
    # 10 seconds is 240 frames, extended to 243.
    assert builder.frame_counts[-1] == 243
    for frames, tokens in ((90, 64), (260, 64), (243, 2049)):
        with pytest.raises(ValueError):
            builder.size(frames, tokens, WIDE)
    # A bucket the checkpoint generates but the deployment did not prepare
    # has no size, and neither does a canvas the checkpoint does not
    # generate.
    for canvas in (TALL, image.Config(1024, 1024)):
        with pytest.raises(ValueError):
            builder.size(243, 64, canvas)


def test_a_deployment_prepares_only_distinct_generated_canvases(denoiser):
    with pytest.raises(ValueError, match="distinct"):
        MediaBuilder(
            denoiser, max_frames=124, max_text_tokens=1024, canvases=SERVED * 2
        )
    with pytest.raises(ValueError, match="offers only"):
        MediaBuilder(
            denoiser,
            max_frames=124,
            max_text_tokens=1024,
            canvases=(WIDE, image.Config(1024, 1024)),
        )
    # Without a selection the deployment prepares every training bucket.
    builder = MediaBuilder(denoiser, max_frames=124, max_text_tokens=1024)
    assert set(builder.canvases) == set(BUCKETS)


def test_the_maximum_bounds_every_training_bucket(denoiser):
    """Serving every bucket, the maximum bounds every layout's storage.

    The runner prepares ``maximum_layout`` first and every other layout
    views its workspace, and every request's buffers view the slot storage,
    dimension by dimension. Given smallest first, the builder still chooses
    a maximum that bounds them all.
    """
    builder = MediaBuilder(
        denoiser,
        max_frames=240,
        max_text_tokens=1024,
        min_frames=96,
        canvases=tuple(reversed(BUCKETS)),
    )
    layouts = builder.layouts()
    assert {layout.canvas for layout in layouts} == set(BUCKETS)
    assert layouts[0] == builder.maximum_layout
    capacity = builder.capacity_buffers()
    maximum = denoiser.workspace_buffers(builder.maximum_layout)
    for layout in layouts:
        for name, config in denoiser.workspace_buffers(layout).items():
            assert all(
                extent <= bound
                for extent, bound in zip(
                    config.shape, maximum[name].shape, strict=True
                )
            ), (layout.canvas, name)
        size = builder.size(layout.num_frames, 1000, layout.canvas)
        for name, config in builder.buffers(size).items():
            assert all(
                extent <= bound
                for extent, bound in zip(
                    config.shape, capacity[name].shape, strict=True
                )
            ), (layout.canvas, name)


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
        denoiser = Model(base_config()).transformer
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
        denoiser = Model(base_config()).transformer
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

    # Three keyframes' 3024 rows exceed the 2048-row capacity, and the refusal
    # names both and the option that raises it.
    with pytest.raises(
        ValueError,
        match="3024 denoiser rows, more than the 2048 .*--max-condition-rows",
    ):
        builder.size(124, 100, WIDE, conditions=(first, first, first))
    with pytest.raises(ValueError, match="more than the 0 "):
        plain.size(124, 100, WIDE, conditions=(first,))


def test_reference_denoiser_has_no_text_only_layout():
    """A denoiser that serves only ``ref2va`` shares no layout at startup.

    Every request it admits brings a reference and evaluates in a layout of
    its own, which the widened maximum still bounds.
    """
    with torch.device("meta"):
        denoiser = Model(base_config()).transformer_ref
    builder = MediaBuilder(
        denoiser,
        max_frames=124,
        max_text_tokens=4096,
        min_frames=96,
        condition_rows=8192,
    )
    assert builder.layouts() == ()

    # One 16:9 reference image beside its 100 prompt tokens.
    reference = Condition(ConditionRole.REFERENCE, video.Config(1, WIDE))
    size = builder.size(124, 100, WIDE, conditions=(reference,))
    layout = builder.layout(size)
    assert denoiser.holds(layout, size)
    assert layout.num_text_tokens == 1024
    maximum = denoiser.workspace_buffers(builder.maximum_layout)
    for name, config in denoiser.workspace_buffers(layout).items():
        assert all(
            extent <= bound
            for extent, bound in zip(
                config.shape, maximum[name].shape, strict=True
            )
        ), name


def test_the_largest_condition_set_follows_the_served_tasks():
    """A denoiser declares the largest condition set its tasks admit.

    Text-to-video takes none and keyframe generation the two keyframes of
    the target canvas. A reference request carries up to nine images and
    three videos: images of the widest reference raster, 4:1 at the
    2048-pixel short edge, and videos of the generated 124 frames with their
    124 / 24 seconds of 32 kHz sound. A dense network packs the keyframes
    beside them; the region packing holds none.
    """
    with torch.device("meta"):
        base = Model(base_config())
        text_only = Model(fasth3_config()).transformer
        regional = Model(
            replace(
                base_config(),
                denoisers={"transformer_ref": omniref_denoiser()},
            )
        ).transformer_ref
    keyframes = (
        Condition(ConditionRole.FIRST_FRAME, video.Config(1, WIDE)),
        Condition(ConditionRole.LAST_FRAME, video.Config(1, WIDE)),
    )
    assert text_only.max_conditions(124, WIDE) == ()
    assert base.transformer.max_conditions(124, WIDE) == keyframes

    for denoiser, leading in (
        (base.transformer_ref, keyframes),
        (regional, ()),
    ):
        conditions = denoiser.max_conditions(124, WIDE)
        references = conditions[len(leading) :]
        assert conditions[: len(leading)] == leading
        stills = [item for item in references if item.video.num_frames == 1]
        clips = [item for item in references if item.video.num_frames > 1]
        assert len(stills) == 9 and len(clips) == 3
        assert {
            (item.video.num_frames, item.audio_samples) for item in clips
        } == {(124, 165_334)}

        def rows(condition, denoiser=denoiser):
            size = denoiser.make_size(
                124, 100, canvas=WIDE, conditions=(condition,)
            )
            return size.condition_rows

        # No reference a request may bring packs more rows than the
        # declared one of its kind, at any aspect from 1:4 to 4:1.
        for width in range(250, 4001, 7):
            still = Condition(
                ConditionRole.REFERENCE,
                video.Config(1, reference_image_size(width, 1000)),
            )
            clip = Condition(
                ConditionRole.REFERENCE,
                video.Config(124, h3_config.canvas(width, 1000)),
                165_334,
            )
            assert rows(still) <= rows(stills[0]), width
            assert rows(clip) <= rows(clips[0]), width
