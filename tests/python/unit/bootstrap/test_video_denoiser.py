"""A deployment serves exactly one of a checkpoint's video denoisers."""

import pytest
import torch

from tests.python.fixtures.h3 import (
    base_config,
    fasth3_config,
)
from uniserve.media import image
from uniserve_models.minimax_h3 import Model
from uniserve_worker.bootstrap.components import media_components
from uniserve_worker.bootstrap.inputs import media_builder, video_denoiser
from uniserve_worker.bootstrap.report import video_denoiser_info
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.call import MediaCall

pytestmark = pytest.mark.unit

_HOST = (
    "text_encoder",
    "video_decoder",
    "audio_decoder",
    "video_codec",
    "muxer",
)


def _config(*placed: str) -> WorkerConfig:
    return WorkerConfig(
        device="cpu",
        max_sequence_tokens=1024,
        max_video_seconds=5.0,
        deployment_components=(*_HOST, *placed),
    )


@pytest.fixture(scope="module")
def base():
    with torch.device("meta"):
        return Model(base_config())


@pytest.mark.parametrize("placed", ["transformer", "transformer_ref"])
def test_the_placed_denoiser_receives_the_video_calls(base, placed):
    config = _config(placed)
    denoiser = video_denoiser(base, config)
    assert denoiser is getattr(base, placed)
    assert media_builder(base, config).denoiser is denoiser
    routes = media_components(base, config.deployment_components)
    assert routes[MediaCall.DENOISING] == routes[MediaCall.LATENT_PREPARATION]
    assert routes[MediaCall.DENOISING] == placed


def test_the_handshake_reports_the_executed_tasks_and_schedule(base):
    info = video_denoiser_info(video_denoiser(base, _config("transformer")))
    # The worker executes both tasks of the t2va/fl2va DiT on the released
    # schedule: 50 sigma points, video shift 12, audio shift 3, the canvas
    # rule's canvas of every named aspect ratio.
    assert (info.tasks, info.schedule_points) == (("t2va", "fl2va"), 50)
    assert (info.video_shift, info.audio_shift) == (12.0, 3.0)
    assert info.canvases == (
        (1536, 672),
        (1344, 768),
        (1024, 768),
        (768, 768),
        (768, 1024),
        (768, 1344),
    )

    # The reference DiT serves ref2va alone.
    reference = video_denoiser(base, _config("transformer_ref"))
    assert video_denoiser_info(reference).tasks == ("ref2va",)


@pytest.mark.parametrize("placed", [(), ("transformer", "transformer_ref")])
def test_a_deployment_places_exactly_one_denoiser(base, placed):
    with pytest.raises(ValueError, match="exactly one"):
        video_denoiser(base, _config(*placed))


def test_a_text_only_export_declares_the_canvases_its_deployment_serves():
    with torch.device("meta"):
        model = Model(fasth3_config())
    # One candidate needs no deployment to name it.
    denoiser = video_denoiser(model, WorkerConfig(device="cpu"))
    info = video_denoiser_info(denoiser)
    assert info.tasks == ("t2va",)
    assert info.schedule_points == 5
    # Without a deployment's selection, every (width, height) training
    # bucket: 21:9, 16:9, 4:3, 1:1, 3:4 and 9:16 at 768p, then at 480p.
    assert info.canvases == (
        (1536, 672),
        (1344, 768),
        (1024, 768),
        (768, 768),
        (768, 1024),
        (768, 1344),
        (992, 416),
        (832, 480),
        (640, 480),
        (480, 480),
        (480, 640),
        (480, 832),
    )
    served = (image.Config(480, 832), image.Config(1344, 768))
    assert video_denoiser_info(denoiser, canvases=served).canvases == (
        (832, 480),
        (768, 1344),
    )
