"""A deployment's ``--video-frame-sizes`` selects the canvases it serves."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.h3 import fasth3_config
from tests.python.fixtures.launch import worker_args
from uniserve.media import image
from uniserve_models.minimax_h3 import Model
from uniserve_worker.bootstrap.inputs import media_builder
from uniserve_worker.config.execution import WorkerConfig

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model():
    with torch.device("meta"):
        return Model(fasth3_config())


def _config(*sizes: tuple[int, int]) -> WorkerConfig:
    return WorkerConfig(
        device="cpu",
        max_sequence_tokens=1024,
        max_video_seconds=5.0,
        video_frame_sizes=sizes,
    )


def test_a_launch_carries_canvases_as_height_by_width(tmp_path):
    config = worker_args(tmp_path, video_frame_sizes="416x992, 768x768")
    assert config.execution.video_frame_sizes == ((416, 992), (768, 768))
    assert worker_args(tmp_path).execution.video_frame_sizes == ()


@pytest.mark.parametrize("sizes", ["768", "768x1344,768x1344", "0x768"])
def test_malformed_canvases_are_refused(sizes, tmp_path):
    with pytest.raises(SystemExit):
        worker_args(tmp_path, video_frame_sizes=sizes)


def test_the_worker_prepares_exactly_the_deployment_canvases(model):
    builder = media_builder(model, _config((416, 992), (768, 768)))
    served = {image.Config(416, 992), image.Config(768, 768)}
    assert set(builder.canvases) == served
    assert {layout.canvas for layout in builder.layouts()} == served
    # A training bucket the deployment did not prepare is not admitted.
    with pytest.raises(ValueError):
        builder.size(124, 64, image.Config(768, 1344))


def test_a_canvas_the_checkpoint_does_not_offer_is_refused(model):
    with pytest.raises(ValueError, match="offers only"):
        media_builder(model, _config((768, 1344), (1024, 1024)))
