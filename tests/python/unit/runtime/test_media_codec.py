"""Decoded temporal windows preserve frame order and normalization.

They also preserve input values.
"""

import pytest
import torch

from uniserve.nn.video import blend_decoded_overlap, video_segment_rgb


@pytest.mark.parametrize("final_unit", [False, True])
@pytest.mark.parametrize("has_overlap", [False, True])
def test_video_segment_normalizes_and_joins_the_declared_window(
    final_unit, has_overlap
):
    frames = torch.tensor([1.0, 1.0, 0.5, 0.75, 0.25, 1.0], dtype=torch.float16)
    segment = frames.reshape(1, 1, 6, 1, 1).expand(1, 3, 6, 1, 1).contiguous()
    original = segment.clone()
    previous = (
        torch.zeros((1, 3, 2, 1, 1), dtype=torch.float16)
        if has_overlap
        else None
    )
    rgb, overlap = video_segment_rgb(
        segment,
        previous,
        body_frames=3,
        overlap_frames=2,
        padding_frames=1,
        pixel_mean=torch.tensor([0.0, 0.25, -0.25]).reshape(1, 3, 1, 1, 1),
        pixel_std=torch.tensor([1.0, 0.5, 2.0]).reshape(1, 3, 1, 1, 1),
        final_unit=final_unit,
    )
    expected = (
        [[0, 64, 0], [128, 128, 191], [128, 128, 191]]
        if has_overlap
        else [[255, 191, 255], [255, 191, 255], [128, 128, 191]]
    )
    if final_unit:
        expected += [[64, 96, 64], [255, 191, 255]]
    assert torch.equal(
        rgb, torch.tensor(expected, dtype=torch.uint8).reshape(-1, 1, 1, 3)
    )
    assert torch.equal(
        overlap,
        torch.tensor([0.25, 1.0], dtype=torch.float16)
        .reshape(1, 1, 2, 1, 1)
        .expand(1, 3, 2, 1, 1),
    )
    assert torch.equal(segment, original)


def test_decoded_tiles_with_zero_overlap_preserve_the_current_tile():
    previous = torch.tensor([[1.0, 2.0]])
    current = torch.tensor([[3.0, 4.0]])
    assert torch.equal(blend_decoded_overlap(previous, current, 0, -1), current)
