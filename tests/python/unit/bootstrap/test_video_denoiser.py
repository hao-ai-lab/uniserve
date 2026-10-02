"""A deployment serves exactly one of a checkpoint's video denoisers."""

import math
from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.h3 import (
    WIDE,
    base_config,
    fasth3_config,
    omniref_denoiser,
)
from uniserve.media import image, video
from uniserve.model import Condition, ConditionRole, ConditionTiles
from uniserve_models.minimax_h3 import Model
from uniserve_worker.bootstrap.components import media_components
from uniserve_worker.bootstrap.inputs import media_builder, video_denoiser
from uniserve_worker.bootstrap.report import video_denoiser_info
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.call import MediaCall
from uniserve_worker.protocol.worker_info import VideoDenoiserInfo

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


@pytest.mark.parametrize("placed", ["denoiser", "reference_denoiser"])
def test_the_placed_denoiser_receives_the_video_calls(base, placed):
    config = _config(placed)
    denoiser = video_denoiser(base, config)
    assert denoiser is getattr(base, placed)
    assert media_builder(base, config).denoiser is denoiser
    routes = media_components(base, config.deployment_components)
    assert routes[MediaCall.DENOISING] == routes[MediaCall.LATENT_PREPARATION]
    assert routes[MediaCall.DENOISING] == placed


def test_the_handshake_reports_the_executed_tasks_and_schedule(base):
    info = video_denoiser_info(video_denoiser(base, _config("denoiser")))
    # The worker executes both tasks of the t2va/fl2va DiT on the released
    # schedule: 50 sigma points, video shift 12, audio shift 3, every canvas
    # of the canvas rule and no checkpoint sequence bound.
    assert (info.tasks, info.schedule_points) == (("t2va", "fl2va"), 50)
    assert (info.video_shift, info.audio_shift) == (12.0, 3.0)
    assert info.canvases == () and info.max_sequence_rows is None

    # The reference DiT serves ref2va alone.
    reference = video_denoiser(base, _config("reference_denoiser"))
    assert video_denoiser_info(reference).tasks == ("ref2va",)


@pytest.mark.parametrize("placed", [(), ("denoiser", "reference_denoiser")])
def test_a_deployment_places_exactly_one_denoiser(base, placed):
    with pytest.raises(ValueError, match="exactly one"):
        video_denoiser(base, _config(*placed))


def test_a_text_only_export_declares_its_only_canvas():
    with torch.device("meta"):
        model = Model(fasth3_config())
    # One candidate needs no deployment to name it.
    denoiser = video_denoiser(model, WorkerConfig(device="cpu"))
    info = video_denoiser_info(denoiser)
    assert info.tasks == ("t2va",)
    assert info.schedule_points == 5
    assert info.canvases == ((1344, 768),)


def test_the_handshake_declares_how_the_denoiser_counts_condition_rows(base):
    # Admission counts condition rows by the handshake's packing, so the
    # declared tiles must reproduce the rows the denoiser itself counts.
    dense = video_denoiser_info(
        video_denoiser(base, _config("reference_denoiser"))
    )
    assert dense.condition_tiles is None

    with torch.device("meta"):
        model = Model(
            replace(
                base_config(),
                denoisers={"reference_denoiser": omniref_denoiser()},
            )
        )
    denoiser = video_denoiser(model, _config("reference_denoiser"))
    info = video_denoiser_info(denoiser)
    assert info.condition_tiles == ConditionTiles(128, (4, 4, 8))
    assert VideoDenoiserInfo.from_mapping(info.to_mapping(), "info") == info

    # A 124-frame clip of a 1344x768 video with 5 s of soundtrack, a
    # 2048x1152 image and a voice: whole tiles of the clip's (37, 24, 42)
    # token grid and of every row group, each tile its 128 rows.
    clip = Condition(ConditionRole.REFERENCE, video.Config(124, WIDE), 160_000)
    picture = Condition(
        ConditionRole.REFERENCE, video.Config(1, image.Config(2048, 1152))
    )
    voice = Condition(ConditionRole.REFERENCE, None, 96_000)
    size = denoiser.make_size(
        124, 300, canvas=WIDE, conditions=(clip, picture, voice)
    )
    tiles = info.condition_tiles
    grid = (37, 768 // 32, 1344 // 32)
    clip_tiles = math.prod(
        math.ceil(extent / side) for extent, side in zip(grid, tiles.video)
    )
    # The soundtrack's 5 s and the voice's 3 s are 200 and 120 latents of
    # 800 samples at 32 kHz, two stereo rows each; the image's raster is
    # 64 x 36 rows.
    group_tiles = sum(
        math.ceil(rows / tiles.rows) for rows in (2 * 200, 64 * 36, 2 * 120)
    )
    assert size.condition_rows == tiles.rows * (clip_tiles + group_tiles)
