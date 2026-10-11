"""A deployment serves exactly one of a checkpoint's video denoisers."""

import pytest
import torch

from tests.python.fixtures.h3 import (
    base_config,
    fasth3_config,
)
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.media import image
from uniserve_models.minimax_h3 import Model
from uniserve_worker.bootstrap.components import media_components
from uniserve_worker.bootstrap.inputs import media_builder, video_denoiser
from uniserve_worker.bootstrap.report import (
    build_worker_layout,
    video_denoiser_info,
)
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
    return cpu_worker_config(
        device="cpu",
        max_model_len=1024,
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
    info = video_denoiser_info(
        video_denoiser(base, _config("transformer")),
        audio_sample_rate=base.audio_decoder.sample_rate,
    )
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
    reference_info = video_denoiser_info(
        reference, audio_sample_rate=base.audio_decoder.sample_rate
    )
    assert reference_info.tasks == ("ref2va",)


@pytest.mark.parametrize("placed", [(), ("transformer", "transformer_ref")])
def test_a_deployment_places_exactly_one_denoiser(base, placed):
    with pytest.raises(ValueError, match="exactly one"):
        video_denoiser(base, _config(*placed))


def test_a_text_only_export_declares_the_canvases_its_deployment_serves():
    with torch.device("meta"):
        model = Model(fasth3_config())
    # One candidate needs no deployment to name it.
    denoiser = video_denoiser(model, cpu_worker_config(device="cpu"))
    rate = model.audio_decoder.sample_rate
    info = video_denoiser_info(denoiser, audio_sample_rate=rate)
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
    declared = video_denoiser_info(
        denoiser, audio_sample_rate=rate, canvases=served
    )
    assert declared.canvases == ((832, 480), (768, 1344))


def test_a_worker_reports_the_audio_rate_of_its_model(base):
    config = cpu_worker_config(
        device="cpu",
        max_model_len=1024,
        max_video_seconds=5.0,
        deployment_components=(*_HOST, "transformer"),
        max_request_pool_size=2,
        max_batch_calls=2,
    )
    declared = build_worker_layout(base, config, queue_depth=6).info
    assert declared.video_denoiser.audio_sample_rate == (
        base.audio_decoder.sample_rate
    )
