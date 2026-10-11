"""A deployment's resolution classes and aspect ratios select its canvases."""

from __future__ import annotations

import pytest
import torch

from tests.python.fixtures.h3 import base_config, fasth3_config
from tests.python.fixtures.launch import worker_args
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.media import image
from uniserve_models.minimax_h3 import Model
from uniserve_models.minimax_h3.config import DMD_CANVASES, NAMED_CANVASES
from uniserve_worker.bootstrap.inputs import media_builder
from uniserve_worker.bootstrap.report import (
    build_worker_layout,
    video_denoiser_info,
)
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
    served = tuple(
        DMD_CANVASES[name]
        for name in ((480, "21:9"), (480, "1:1"), (768, "21:9"), (768, "1:1"))
    )
    assert builder.canvases == served
    assert {layout.canvas for layout in builder.layouts()} == set(served)
    # A training bucket the deployment did not name is not admitted.
    with pytest.raises(ValueError):
        builder.size(124, 64, DMD_CANVASES[(768, "16:9")])


def test_a_deployment_that_names_nothing_prepares_every_canvas(model):
    builder = media_builder(model, _config((), ()))
    assert len(builder.canvases) == 12


def test_the_handshake_declares_each_canvas_with_its_names(model):
    config = _config(("768p", "480p"), ("16:9",))
    builder = media_builder(model, config)
    info = video_denoiser_info(
        builder.denoiser,
        frame_rate=model.video_postprocessor.frame_rate,
        audio_sample_rate=model.audio_decoder.sample_rate,
        canvases=builder.canvases,
    )
    names = [(768, "16:9"), (480, "16:9")]
    assert [
        ((c.short_edge, c.aspect_ratio), image.Config(c.height, c.width))
        for c in info.canvases
    ] == [(name, DMD_CANVASES[name]) for name in names]


def test_a_worker_reports_its_canvases_and_frame_rate_in_its_info(model):
    config = _config(
        ("480p",), ("21:9", "9:16"), max_request_pool_size=2, max_batch_calls=2
    )
    declared = build_worker_layout(
        model, config, queue_depth=6
    ).info.video_denoiser
    assert declared is not None
    assert declared.frame_rate == 24
    assert [
        (c.short_edge, c.aspect_ratio, c.width, c.height)
        for c in declared.canvases
    ] == [(480, "21:9", 992, 416), (480, "9:16", 480, 832)]


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
    assert served.canvases == (NAMED_CANVASES[(768, "16:9")],)
    with pytest.raises(ValueError, match="generates only"):
        media_builder(
            base, _config(("480p",), ("16:9",), deployment_components=placed)
        )
