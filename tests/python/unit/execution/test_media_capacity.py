"""Every admitted H3 size evaluates in a provisioned capacity layout."""

from __future__ import annotations

import math

import pytest
import torch

from uniserve.media import image
from uniserve.runtime import TensorBuffers
from uniserve_models.minimax_h3 import Config, Model
from uniserve_worker.model_executor.media_inputs import MediaBuilder

pytestmark = pytest.mark.unit

# The 16 output lengths the video API admits: 4 to 15 seconds at 24 fps,
# each extended to a complete native temporal window.
ADMITTED_FRAMES = tuple(107 + 17 * index for index in range(16))

# The served output rasters: 16:9 and 9:16.
LANDSCAPE = image.Config(768, 1344)
PORTRAIT = image.Config(1344, 768)
RASTERS = (LANDSCAPE, PORTRAIT)


@pytest.fixture(scope="module")
def denoiser():
    with torch.device("meta"):
        return Model(Config()).denoiser


def test_capacity_layouts_cover_every_admitted_size(denoiser):
    builder = MediaBuilder(
        denoiser,
        frame_sizes=RASTERS,
        max_frames=360,
        max_text_tokens=16384,
        min_frames=96,
        text_capacities=(1000, 4096, 16384),
    )
    assert builder.frame_counts == ADMITTED_FRAMES
    # Capacities are whole 64-row text tiles.
    assert builder.text_capacities == (1024, 4096, 16384)

    layouts = builder.layouts()
    assert len(layouts) == len(ADMITTED_FRAMES) * len(RASTERS) * 3
    maximum = layouts[0]
    assert (maximum.num_frames, maximum.num_text_tokens) == (362, 16384)
    assert {layout.frame for layout in layouts} == set(RASTERS)

    for frames in ADMITTED_FRAMES:
        for frame in RASTERS:
            for tokens in (1, 63, 1024, 1025, 4096, 4097, 16384):
                size = builder.size(frames, frame, tokens)
                layout = builder.layout(size)
                assert layout in layouts
                assert layout.frame == frame
                assert denoiser.holds(layout, size)
                # The smallest capacity that holds the prompt is chosen.
                assert layout.num_text_tokens == min(
                    capacity
                    for capacity in builder.text_capacities
                    if capacity >= tokens
                )


def test_both_orientations_pack_the_same_video_rows(denoiser):
    """A 9:16 request is the 16:9 request's token count, transposed."""
    for frames in (22, 124):
        landscape, portrait = (
            denoiser.make_size(frames, frame, 64) for frame in RASTERS
        )
        assert denoiser.latent_shape("video", landscape) == (
            denoiser.latent_shape("video", portrait)
        )
        assert denoiser.latent_shape("audio", landscape) == (
            denoiser.latent_shape("audio", portrait)
        )
        video, portrait_video = (
            denoiser.noise_shape("video", size)
            for size in (landscape, portrait)
        )
        assert video[-2:] == (48, 84)
        assert portrait_video[-2:] == (84, 48)


def test_request_storage_holds_every_admitted_raster(denoiser):
    """One request slot's storage holds a request at either raster."""
    builder = MediaBuilder(
        denoiser, frame_sizes=RASTERS, max_frames=124, max_text_tokens=1024
    )
    capacity = builder.capacity_buffers()
    with TensorBuffers.allocate(capacity, device="meta") as storage:
        for frame in RASTERS:
            for tokens in (1, 1024):
                size = builder.size(124, frame, tokens)
                views = storage.view(builder.buffers(size))
                assert set(views) == set(builder.buffers(size))
    # Both rasters pack the same rows, so the transposition costs no pages.
    landscape, portrait = (
        math.prod(denoiser.latent_shape("video", maximum))
        for maximum in builder.maxima
    )
    assert landscape == portrait


def test_sizes_outside_the_admitted_range_have_no_layout(denoiser):
    builder = MediaBuilder(
        denoiser,
        frame_sizes=RASTERS,
        max_frames=240,
        max_text_tokens=2048,
        min_frames=96,
    )
    # 10 seconds is 240 frames, extended to 243.
    assert builder.frame_counts[-1] == 243
    for frames, tokens in ((90, 64), (260, 64), (243, 2049)):
        with pytest.raises(ValueError):
            builder.size(frames, LANDSCAPE, tokens)
    # A raster the model does not generate has no size.
    with pytest.raises(ValueError):
        builder.size(243, image.Config(1024, 1024), 64)
    # A served raster the worker was not built for has no layout.
    landscape = MediaBuilder(
        denoiser,
        frame_sizes=(LANDSCAPE,),
        max_frames=240,
        max_text_tokens=2048,
        min_frames=96,
    )
    with pytest.raises(ValueError):
        landscape.size(243, PORTRAIT, 64)


def test_default_capacities_step_to_the_prompt_capacity(denoiser):
    builder = MediaBuilder(
        denoiser, frame_sizes=RASTERS, max_frames=124, max_text_tokens=5000
    )
    assert builder.text_capacities == (1024, 2048, 4096, 5056)
    # A prompt capacity below the first rung is the only capacity.
    small = MediaBuilder(
        denoiser, frame_sizes=RASTERS, max_frames=124, max_text_tokens=500
    )
    assert small.text_capacities == (512,)
    # Without a floor, every native frame count up to the capacity is kept.
    assert builder.frame_counts == (22, 39, 56, 73, 90, 107, 124)


def test_a_layout_holds_only_prompts_that_fit_at_its_frame_count(denoiser):
    def make(frames, tokens, frame=LANDSCAPE):
        return denoiser.make_size(frames, frame, tokens)

    layout = make(124, 2048)
    assert denoiser.holds(layout, make(124, 1))
    assert denoiser.holds(layout, make(124, 2048))
    assert not denoiser.holds(layout, make(124, 2049))
    assert not denoiser.holds(layout, make(141, 64))
    # A layout holds only prompts at its own raster.
    assert not denoiser.holds(layout, make(124, 64, PORTRAIT))
    # A text region that is not whole tiles is not a layout.
    assert not denoiser.holds(make(124, 100), make(124, 64))
