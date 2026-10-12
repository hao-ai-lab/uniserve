"""MiniMax-H3 canvases and frame counts match the diffusers reference vectors.

The vectors in ``minimax_h3_plan.json``, generated from the diffusers
MiniMax-H3 pipeline, are shared with the Rust planner's tests
(``crates/server/src/serving/video``). The model package's canvas rule and
frame count are checked against the same reference values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve.media import image
from uniserve_models.minimax_h3 import config, processing

pytestmark = pytest.mark.unit

_FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_plan.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _FIXTURE["canvas_cases"])
def test_canvas(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            config.canvas(*case["aspect"])
        return
    width, height = case["canvas"]
    assert config.canvas(*case["aspect"]) == image.Config(
        height=height, width=width
    )


@pytest.mark.parametrize("case", _FIXTURE["duration_cases"])
def test_duration(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.frame_count(case["seconds"])
        return
    assert processing.frame_count(case["seconds"]) == case["num_frames"]
