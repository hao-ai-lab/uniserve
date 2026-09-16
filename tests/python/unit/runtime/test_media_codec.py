"""Decoded temporal windows preserve their current tile without overlap."""

import torch

from uniserve.nn.video import blend_decoded_overlap


def test_decoded_tiles_with_zero_overlap_preserve_the_current_tile():
    previous = torch.tensor([[1.0, 2.0]])
    current = torch.tensor([[3.0, 4.0]])
    assert torch.equal(blend_decoded_overlap(previous, current, 0, -1), current)
