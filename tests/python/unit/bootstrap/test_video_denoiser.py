"""A deployment serves exactly one of a checkpoint's video denoisers."""

import pytest
import torch

from tests.python.fixtures.h3 import (
    base_config,
    fasth3_config,
)
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.model import ReferenceLimits
from uniserve_models.minimax_h3 import Model
from uniserve_models.minimax_h3.config import DMD_CANVASES, NAMED_CANVASES
from uniserve_worker.bootstrap.components import media_components
from uniserve_worker.bootstrap.inputs import media_builder, video_denoiser
from uniserve_worker.bootstrap.report import (
    build_worker_layout,
    frame_clips,
    video_denoiser_info,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.call import MediaCall
from uniserve_worker.protocol.worker_info import FrameClips

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


def _declared(info) -> tuple[tuple[int, str, int, int], ...]:
    """Each declared canvas as (short edge, aspect ratio, width, height)."""
    return tuple(
        (c.short_edge, c.aspect_ratio, c.width, c.height) for c in info.canvases
    )


def _trained(canvases) -> tuple[tuple[int, str, int, int], ...]:
    """A checkpoint's canvases as (short edge, aspect ratio, width, height)."""
    return tuple(
        (edge, ratio, canvas.width, canvas.height)
        for (edge, ratio), canvas in canvases.items()
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
    denoiser = video_denoiser(base, _config("transformer"))
    info = video_denoiser_info(
        denoiser,
        frame_rate=base.video_postprocessor.frame_rate,
        audio_sample_rate=base.audio_decoder.sample_rate,
        frames=frame_clips(denoiser, base.video_encoder),
    )
    # A video is `17n + 5` frames of `n + 1` encoder clips: five latent
    # frames per clip and two in the tail.
    assert info.frames == FrameClips(
        min_frames=22,
        clip_frames=17,
        clip_latent_frames=5,
        tail_latent_frames=2,
    )
    # The worker executes both tasks of the t2va/fl2va DiT on the released
    # schedule: 50 sigma points, video shift 12, audio shift 3, the canvas
    # rule's canvas of every named aspect ratio at 768p.
    assert (info.tasks, info.schedule_points) == (("t2va", "fl2va"), 50)
    assert (info.video_shift, info.audio_shift) == (12.0, 3.0)
    assert _declared(info) == _trained(NAMED_CANVASES)

    # It takes no references.
    assert info.references == ReferenceLimits()

    # The reference DiT serves ref2va alone, with the references the
    # checkpoint documents: nine images, three videos and three audio tracks,
    # twelve in all.
    reference_denoiser = video_denoiser(base, _config("transformer_ref"))
    reference = video_denoiser_info(
        reference_denoiser,
        frame_rate=base.video_postprocessor.frame_rate,
        audio_sample_rate=base.audio_decoder.sample_rate,
        frames=frame_clips(reference_denoiser, base.video_encoder),
    )
    assert reference.tasks == ("ref2va",)
    assert reference.references == ReferenceLimits(
        images=9, videos=3, audios=3, total=12
    )


@pytest.mark.parametrize("placed", [(), ("transformer", "transformer_ref")])
def test_a_deployment_places_exactly_one_denoiser(base, placed):
    with pytest.raises(ValueError, match="exactly one"):
        video_denoiser(base, _config(*placed))


def test_a_text_only_export_declares_the_canvases_its_deployment_serves():
    with torch.device("meta"):
        model = Model(fasth3_config())
    # One candidate needs no deployment to name it.
    denoiser = video_denoiser(model, cpu_worker_config(device="cpu"))
    frames = frame_clips(denoiser, model.video_encoder)
    info = video_denoiser_info(
        denoiser,
        frame_rate=model.video_postprocessor.frame_rate,
        audio_sample_rate=model.audio_decoder.sample_rate,
        frames=frames,
    )
    assert info.tasks == ("t2va",)
    assert info.schedule_points == 5
    # Without a deployment's selection, every training bucket: 21:9, 16:9,
    # 4:3, 1:1, 3:4 and 9:16 at 768p, then at 480p.
    assert _declared(info) == _trained(DMD_CANVASES)
    # A deployment's selection is declared in the order it prepares them.
    chosen = {
        name: DMD_CANVASES[name] for name in ((480, "16:9"), (768, "9:16"))
    }
    declared = video_denoiser_info(
        denoiser,
        frame_rate=model.video_postprocessor.frame_rate,
        audio_sample_rate=model.audio_decoder.sample_rate,
        frames=frames,
        canvases=tuple(chosen.values()),
    )
    assert _declared(declared) == _trained(chosen)


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
