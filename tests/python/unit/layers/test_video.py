"""Numerical video reconstruction preserves window order and decoded precision."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from uniserve_worker.modeling.batch import TensorOutput
from uniserve_worker.modeling.geometry import DecodeWindow
from uniserve_worker.modeling.resources import TensorAlias, TensorNeeds, TensorSchema
from uniserve_worker.modeling.video import VideoMixin
from uniserve_worker.nn.vae.spatial import SpatialDecoder

pytestmark = pytest.mark.unit


def test_video_windows_preserve_pixels_across_separate_calls():
    model = VideoMixin()
    frames = (
        (1, 0, 0.5, 1, 0.25, 0.75),
        (1, 1, 0.5, 0, 1, 0),
        (0, 0, 0, 0, 0.5, 1),
    )
    segments = tuple(
        torch.tensor(values, dtype=torch.float16)
        .reshape(1, 1, 6, 1, 1)
        .expand(1, 3, 6, 2, 3)
        .contiguous()
        for values in frames
    )
    originals = tuple(segment.clone() for segment in segments)
    windows = tuple(
        DecodeWindow(
            index * 2,
            index * 2 + 3,
            index * 3,
            index * 3 + (5 if index == 2 else 3),
            3,
            2,
            1,
            final=index == 2,
        )
        for index in range(3)
    )
    constants = {
        "pixel_mean": torch.zeros((1, 3, 1, 1, 1)),
        "pixel_std": torch.ones((1, 3, 1, 1, 1)),
    }
    state = {"video_overlap": torch.full((1, 3, 2, 2, 3), 10, dtype=torch.float16)}
    scratch = {"rgb_frames": torch.empty((11, 2, 3, 3), dtype=torch.uint8)}
    needs = TensorNeeds(
        scratch={"rgb_frames": TensorSchema((11, 2, 3, 3), torch.uint8)},
        outputs={
            "video": TensorSchema(
                (11, 2, 3, 3),
                torch.uint8,
                variable_axes=(0,),
                alias=TensorAlias("scratch", "rgb_frames"),
            )
        },
    )
    # The first interval resets overlap numerically; later intervals consume
    # the exact successor state left by the previous independent call.
    parts = []
    for segment, window in zip(segments, windows, strict=True):
        result = model.postprocess_video(
            (segment,), (window,), state=state, constants=constants, scratch=scratch
        )
        result.validate(needs, state=state, scratch=scratch)
        value = result.values["video"][0]
        assert value is not None
        assert result.layouts["video"][0].shape == tuple(value.shape)
        parts.append(value.clone())
    expected = (
        torch.tensor([255, 0, 128, 64, 223, 128, 255, 0, 0, 128, 255], dtype=torch.uint8)
        .view(11, 1, 1, 1)
        .expand(11, 2, 3, 3)
    )
    torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
    together = model.postprocess_video(
        segments, windows, state=state, constants=constants, scratch=scratch
    )
    torch.testing.assert_close(together.values["video"][0], expected, rtol=0, atol=0)
    torch.testing.assert_close(state["video_overlap"], segments[-1][:, :, -2:], rtol=0, atol=0)
    for segment, original in zip(segments, originals, strict=True):
        torch.testing.assert_close(segment, original, rtol=0, atol=0)

    previous_state = state["video_overlap"].clone()
    previous_output = scratch["rgb_frames"].clone()
    with pytest.raises(ValueError, match="contiguous"):
        model.postprocess_video(
            segments[:2],
            (windows[0], replace(windows[1], frame_start=4, frame_stop=7)),
            state=state,
            constants=constants,
            scratch=scratch,
        )
    torch.testing.assert_close(state["video_overlap"], previous_state, rtol=0, atol=0)
    torch.testing.assert_close(scratch["rgb_frames"], previous_output, rtol=0, atol=0)

    # The numerical contract identifies a borrowed result. Replacing it with
    # independent storage or a different raster/representation violates that
    # contract even when the visible pixel values happen to agree.
    for invalid in (
        scratch["rgb_frames"].clone(),
        scratch["rgb_frames"][:, :, :2],
        scratch["rgb_frames"].view(torch.int8),
    ):
        with pytest.raises(ValueError):
            TensorOutput({"video": (invalid,)}).validate(needs, state=state, scratch=scratch)
    scratch["rgb_frames"][0].zero_()
    assert torch.count_nonzero(together.values["video"][0][0]) == 0


class RasterDecoder(SpatialDecoder):
    """A separable numerical decoder with exactly known spatial reconstruction."""

    use_tiling = True
    spatial_compression_ratio = 2
    tile_sample_min_height = 8
    tile_sample_min_width = 8
    tile_sample_min_overlap_height = 2
    tile_sample_min_overlap_width = 2

    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Identity()

    def forward(self, latents):
        return latents.repeat_interleave(2, -2).repeat_interleave(2, -1)


@pytest.mark.parametrize("height,width", [(3, 4), (5, 7), (9, 13)])
def test_spatial_decoder_preserves_batches_and_raster_alignment(height, width):
    decoder = RasterDecoder()
    latents = torch.arange(2 * 3 * 2 * height * width, dtype=torch.float32).reshape(
        2, 3, 2, height, width
    )
    expected = latents.repeat_interleave(2, -2).repeat_interleave(2, -1)
    if (height, width) == (5, 7):
        # This raster has two 8-pixel tiles per axis: a six-pixel vertical
        # overlap at rows [2, 8), then a two-pixel horizontal overlap at [6, 8).
        # Equal source pixels still undergo float32 cross-fade rounding.
        raster = expected.clone()
        weights = (torch.arange(6, dtype=torch.float32) / 6).view(1, 1, 1, 6, 1)
        expected[..., 2:8, :] = raster[..., 2:8, :] * (1 - weights) + raster[..., 2:8, :] * weights
        weights = (torch.arange(2, dtype=torch.float32) / 2).view(1, 1, 1, 1, 2)
        expected[..., 6:8] = raster[..., 6:8] * (1 - weights) + expected[..., 6:8] * weights
    original = latents.clone()
    torch.testing.assert_close(decoder.decode(latents), expected, rtol=0, atol=0)
    torch.testing.assert_close(latents, original, rtol=0, atol=0)
    decoder.use_tiling = False
    torch.testing.assert_close(
        decoder.decode(latents),
        latents.repeat_interleave(2, -2).repeat_interleave(2, -1),
        rtol=0,
        atol=0,
    )
