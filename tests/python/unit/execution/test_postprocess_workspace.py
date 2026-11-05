"""Startup video post-processor contexts share their transient workspace."""

from __future__ import annotations

import pytest
from torch import nn

from uniserve.media import video
from uniserve_models.minimax_h3.config import FASTH3_CANVASES
from uniserve_models.minimax_h3.output import VideoPostprocessor
from uniserve_worker.execution.model_executor import ModelExecutor

pytestmark = pytest.mark.unit

KEY = ("video_postprocessor", "", "forward")


def _executor(*, startup_complete: bool = False) -> ModelExecutor:
    # Only the state ``_shared_workspace`` reads; no model or devices.
    executor = object.__new__(ModelExecutor)
    executor._startup_complete = startup_complete
    executor._module_workspaces = {}
    return executor


def _size(frames: int, name: tuple[int, str]) -> video.Config:
    return video.Config(frames, FASTH3_CANVASES[name])


def test_every_frame_count_of_a_canvas_borrows_one_workspace():
    executor = _executor()
    module = VideoPostprocessor(frame_rate=24)
    canvas = (480, "9:16")
    shared = {
        id(
            executor._shared_workspace(
                (*KEY, n), module, _size(n, canvas), "cpu"
            )
        )
        for n in (39, 124, 362)
    }
    assert len(shared) == 1
    # The longest clip is the same, so the views have one shape.
    views = [
        dict(
            executor._shared_workspace(
                (*KEY, n), module, _size(n, canvas), "cpu"
            ).view(module.workspace_buffers(_size(n, canvas)))
        )
        for n in (39, 362)
    ]
    assert all(
        a.data_ptr() == b.data_ptr()
        for a, b in zip(views[0].values(), views[1].values(), strict=True)
    )

    other = executor._shared_workspace(
        (*KEY, 39), module, _size(39, (768, "9:16")), "cpu"
    )
    assert id(other) not in shared
    assert len(executor._module_workspaces) == 2


def test_serving_contexts_and_other_modules_own_their_workspace():
    module = VideoPostprocessor(frame_rate=24)
    size = _size(124, (768, "16:9"))
    serving = _executor(startup_complete=True)
    assert serving._shared_workspace((*KEY, 124), module, size, "cpu") is None
    assert not serving._module_workspaces

    startup = _executor()
    assert (
        startup._shared_workspace((*KEY, 124), nn.Identity(), size, "cpu")
        is None
    )
    assert not startup._module_workspaces
