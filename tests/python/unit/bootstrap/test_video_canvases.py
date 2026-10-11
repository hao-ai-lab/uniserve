"""A deployment's resolution classes and aspect ratios select its canvases."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.h3 import base_config, fasth3_config
from tests.python.fixtures.launch import worker_args
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.media import image
from uniserve_models.minimax_h3 import Model
from uniserve_worker.bootstrap.inputs import media_builder
from uniserve_worker.bootstrap.report import video_denoiser_info
from uniserve_worker.config.execution import WorkerConfig

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def model():
    with torch.device("meta"):
        return Model(fasth3_config())


def _config(
    resolutions: tuple[str, ...], aspect_ratios: tuple[str, ...], **overrides
) -> WorkerConfig:
    return cpu_worker_config(
        device="cpu",
        max_model_len=1024,
        max_video_seconds=5.0,
        video_resolutions=resolutions,
        video_aspect_ratios=aspect_ratios,
        **overrides,
    )


def test_a_launch_carries_the_canvas_names(tmp_path):
    config = worker_args(
        tmp_path,
        video_resolutions="768p, 480p",
        video_aspect_ratios="21:9,16:9",
    )
    assert config.execution.video_resolutions == ("768p", "480p")
    assert config.execution.video_aspect_ratios == ("21:9", "16:9")

    unselected = worker_args(tmp_path).execution
    assert unselected.video_resolutions == ()
    assert unselected.video_aspect_ratios == ()


@pytest.mark.parametrize("names", ["", "768p,,480p"])
def test_blank_names_are_refused(names, tmp_path):
    with pytest.raises(SystemExit):
        worker_args(tmp_path, video_resolutions=names)


def test_the_worker_prepares_the_named_canvases_resolution_major(model):
    builder = media_builder(model, _config(("480p", "768p"), ("21:9", "1:1")))
    # 480p's 21:9 bucket is the training table's, not a rescaled 768p one.
    served = (
        image.Config(416, 992),
        image.Config(480, 480),
        image.Config(672, 1536),
        image.Config(768, 768),
    )
    assert builder.canvases == served
    assert {layout.canvas for layout in builder.layouts()} == set(served)
    # A training bucket the deployment did not name is not admitted.
    with pytest.raises(ValueError):
        builder.size(124, 64, image.Config(768, 1344))


def test_a_deployment_that_names_nothing_prepares_every_canvas(model):
    builder = media_builder(model, _config((), ()))
    assert len(builder.canvases) == 12


def test_the_handshake_declares_each_canvas_with_its_names(model):
    config = _config(("768p", "480p"), ("16:9",))
    info = video_denoiser_info(
        media_builder(model, config).denoiser,
        canvases=media_builder(model, config).canvases,
    )
    assert [
        (c.short_edge, c.aspect_ratio, c.width, c.height) for c in info.canvases
    ] == [(768, "16:9", 1344, 768), (480, "16:9", 832, 480)]


@pytest.mark.parametrize(
    "resolutions, aspect_ratios",
    [
        # A class the checkpoint was not trained at.
        (("720p",), ("16:9",)),
        # An aspect ratio it has no bucket for.
        (("768p",), ("5:4",)),
        # Not a resolution class.
        (("768",), ("16:9",)),
        # A name twice.
        (("768p", "768p"), ("16:9",)),
    ],
)
def test_a_canvas_the_checkpoint_does_not_generate_is_refused(
    model, resolutions, aspect_ratios
):
    with pytest.raises(ValueError):
        media_builder(model, _config(resolutions, aspect_ratios))


def test_the_released_base_checkpoint_generates_768p_only():
    with torch.device("meta"):
        base = Model(base_config())
    placed = ("transformer",)
    served = media_builder(
        base, _config(("768p",), ("16:9",), deployment_components=placed)
    )
    assert served.canvases == (image.Config(768, 1344),)
    with pytest.raises(ValueError, match="generates only"):
        media_builder(
            base, _config(("480p",), ("16:9",), deployment_components=placed)
        )
