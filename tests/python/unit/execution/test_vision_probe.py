"""A vision probe fills a prompt capacity with image blocks."""

from __future__ import annotations

from math import prod

import pytest

from uniserve_worker.execution.model_executor import vision_probe_layout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("capacity", (64, 4096, 8193, 8194, 30720, 84608))
@pytest.mark.parametrize("merge", (1, 2))
def test_the_probe_fills_the_capacity_with_whole_merged_columns(
    capacity, merge
):
    grids, padding = vision_probe_layout(capacity, merge)
    column = 128 // merge
    tokens = [prod(grid) // merge**2 for grid in grids]
    # One separator token before every image, the images' merged tokens
    # and the trailing text fill the capacity exactly.
    assert len(grids) + sum(tokens) + padding == capacity
    assert 0 <= padding < column + 1
    for time, height, width in grids:
        assert time == 1 and height == 128 and 0 < width <= 256
        assert width % merge == 0
    # Every image but the last is a whole 128 x 256-patch one.
    assert all(grid == (1, 128, 256) for grid in grids[:-1])
