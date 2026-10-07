"""Numerical video reconstruction preserves window order.

It also preserves decoded precision.
"""

import pytest
import torch
from torch import nn

from uniserve.media import image, video
from uniserve.model import VideoPostprocessor
from uniserve.nn.vae.spatial import SpatialDecoder
from uniserve.tensors import OutputLayout, TensorOutput

pytestmark = pytest.mark.unit

# Eleven 2x3 frames.
SIZE = video.Config(11, image.Config(2, 3))


def test_video_windows_preserve_pixels_across_separate_calls():
    class ThreeFrameVideo(VideoPostprocessor):
        def reconstruction_slices(self, frames, num_frames):
            return slice(0, 3), slice(4, 6)

    model = ThreeFrameVideo(
        torch.tensor([0.0, 0.5], dtype=torch.float16),
        frame_rate=24,
    )
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
        slice(index * 3, index * 3 + (5 if index == 2 else 3))
        for index in range(3)
    )
    outputs = tuple(
        TensorOutput(
            value,
            OutputLayout(
                tuple(value.shape),
                value.dtype,
                tuple(slice(0, n) for n in value.shape),
            ),
        )
        for value in segments
    )
    constants = {
        "pixel_mean": torch.zeros((1, 3, 1, 1, 1)),
        "pixel_std": torch.ones((1, 3, 1, 1, 1)),
    }
    state = {
        "video_overlap": torch.full((1, 3, 2, 2, 3), 10, dtype=torch.float16)
    }
    scratch = {
        "rgb_frames": torch.empty((11, 2, 3, 3), dtype=torch.uint8),
        # One rank reconstructs every unit here, so the ring delivers each
        # window's own overlap to the call that follows it.
        "overlap_exchange": torch.empty((1, 3, 2, 2, 3), dtype=torch.float16),
    }
    # The first interval resets overlap numerically; later intervals consume
    # the exact successor state left by the previous independent call.
    parts = []
    for segment, window in zip(outputs, windows, strict=True):
        result = model(
            (segment,),
            frames=(window,),
            sizes=(SIZE,),
            state=state,
            constants=constants,
            workspace=scratch,
        )[0]
        value = result.tensor
        assert result.layout.shape == (11, 2, 3, 3)
        assert result.layout.local_slice[0] == window
        parts.append(value.clone())
    expected = (
        torch.tensor(
            [255, 0, 128, 64, 223, 128, 255, 0, 0, 128, 255], dtype=torch.uint8
        )
        .view(11, 1, 1, 1)
        .expand(11, 2, 3, 3)
    )
    torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
    together = model(
        outputs,
        frames=windows,
        sizes=(SIZE,) * 3,
        state=state,
        constants=constants,
        workspace=scratch,
    )
    torch.testing.assert_close(
        torch.cat(tuple(part.tensor for part in together)),
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state["video_overlap"], segments[-1][:, :, -2:], rtol=0, atol=0
    )
    for segment, original in zip(segments, originals, strict=True):
        torch.testing.assert_close(segment, original, rtol=0, atol=0)

    previous_state = state["video_overlap"].clone()
    previous_output = scratch["rgb_frames"].clone()
    with pytest.raises(ValueError, match="contiguous"):
        model(
            outputs[:2],
            frames=(windows[0], slice(4, 7)),
            sizes=(SIZE,) * 2,
            state=state,
            constants=constants,
            workspace=scratch,
        )
    torch.testing.assert_close(
        state["video_overlap"], previous_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        scratch["rgb_frames"], previous_output, rtol=0, atol=0
    )

    scratch["rgb_frames"][0].zero_()
    assert torch.count_nonzero(together[0].tensor[0]) == 0


@pytest.mark.parametrize("height,width", [(3, 4), (5, 7), (9, 13)])
def test_spatial_decoder_preserves_batches_and_raster_alignment(height, width):
    decoder = SpatialDecoder(
        nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        spatial_compression=2,
        tile_height=8,
        tile_width=8,
        overlap_height=2,
        overlap_width=2,
    )
    latents = torch.arange(
        2 * 3 * 2 * height * width, dtype=torch.float32
    ).reshape(2, 3, 2, height, width)
    expected = latents.repeat_interleave(2, -2).repeat_interleave(2, -1)
    if (height, width) == (5, 7):
        # This raster has two 8-pixel tiles per axis: a six-pixel vertical
        # overlap at rows [2, 8), then a two-pixel horizontal overlap at [6, 8).
        # Equal source pixels still undergo float32 cross-fade rounding.
        raster = expected.clone()
        weights = (torch.arange(6, dtype=torch.float32) / 6).view(1, 1, 1, 6, 1)
        expected[..., 2:8, :] = (
            raster[..., 2:8, :] * (1 - weights) + raster[..., 2:8, :] * weights
        )
        weights = (torch.arange(2, dtype=torch.float32) / 2).view(1, 1, 1, 1, 2)
        expected[..., 6:8] = (
            raster[..., 6:8] * (1 - weights) + expected[..., 6:8] * weights
        )
    original = latents.clone()
    torch.testing.assert_close(decoder(latents), expected, rtol=0, atol=0)
    torch.testing.assert_close(latents, original, rtol=0, atol=0)
    torch.testing.assert_close(
        decoder.decode_tile(latents),
        latents.repeat_interleave(2, -2).repeat_interleave(2, -1),
        rtol=0,
        atol=0,
    )
