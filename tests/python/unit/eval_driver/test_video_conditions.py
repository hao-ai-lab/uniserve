"""Conditioned video requests name their media by ``file://`` URI.

A conditioned row names its media relative to the point's condition root,
and the server receives the canonical body with ``file://`` URIs; each
output is validated against the canvas its own request resolves to.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from PIL import Image

from tests.python.fixtures.http_stub import stub_server
from uniserve_eval.metrics import summarize
from uniserve_eval.tasks.video import VideoTask
from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    VIDEOS_SYNC,
    BenchmarkPoint,
    DecodedVideo,
    Example,
    MetricDefinition,
    TaskName,
    VideoConfig,
    VideoShape,
)

pytestmark = pytest.mark.unit

VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 reference video"
AUDIO_BYTES = b"ID3 reference voice"
# EXIF tag and value that display an image turned by 90 degrees.
ORIENTATION, ROTATED = 0x0112, 6


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """A condition root holding one file of each media type."""
    media = tmp_path / "inputs" / "media"
    media.mkdir(parents=True)
    Image.new("RGB", (64, 36), (200, 40, 40)).save(media / "keyframe.png")
    Image.new("RGB", (48, 64), (40, 200, 40)).save(media / "subject.png")
    # Stored landscape, displayed portrait.
    portrait = Image.new("RGB", (64, 36), (40, 40, 200))
    exif = portrait.getexif()
    exif[ORIENTATION] = ROTATED
    portrait.save(media / "portrait.jpg", exif=exif)
    (media / "clip.mp4").write_bytes(VIDEO_BYTES)
    (media / "voice.mp3").write_bytes(AUDIO_BYTES)
    return tmp_path / "inputs"


def point(inputs: Path, task: str, **video):
    return BenchmarkPoint(
        name="conditioned",
        server="video",
        task=TaskName.VIDEO,
        model="MiniMax-H3",
        dataset="jsonl",
        endpoint=VIDEOS_SYNC,
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        video=VideoConfig(
            task=task,
            aspect_ratio="auto",
            condition_root=str(inputs),
            **video,
        ),
    )


# The protocol's conditioned workloads in miniature: a first keyframe (W3),
# an image with a voice (W4), and a video with its soundtrack and a voice
# (W5).
FIRST_FRAME = [
    {
        "type": "image",
        "media": "media/keyframe.png",
        "role": "keyframe",
        "frame_index": 0,
    }
]
IMAGE_AUDIO = [
    {"type": "image", "media": "media/subject.png", "role": "reference"},
    {"type": "audio", "media": "media/voice.mp3", "role": "reference"},
]
VIDEO_AUDIO = [
    {"type": "video", "media": "media/clip.mp4", "role": "reference"},
    {"type": "audio", "media": "media/voice.mp3", "role": "reference"},
]


def send(request):
    """Send one request to a recording server and return what it received."""
    sent = []

    async def exercise():
        async def handler(received):
            sent.append((received.headers, await received.read()))
            return web.Response(
                body=b"mp4", headers={"content-type": "video/mp4"}
            )

        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            record = await send_request(
                client, base_url, request, "row", task="video"
            )
        assert record.success
        return record

    record = asyncio.run(exercise())
    return sent[0], record


@pytest.mark.parametrize(
    "task,conditions",
    [("fl2va", FIRST_FRAME), ("ref2va", IMAGE_AUDIO), ("ref2va", VIDEO_AUDIO)],
    ids=["first-frame", "image-audio", "video-audio"],
)
def test_the_canonical_body_names_condition_media_by_file_uri(
    inputs, task, conditions
):
    task_adapter = VideoTask(point(inputs, task))
    request = task_adapter.build_request(
        Example("row", "scene", seconds=5, seed=7, conditions=conditions)
    )
    received, record = send(request)

    body = json.loads(received[1])
    assert body["task"] == task
    assert body["target"] == {
        "short_edge": 768,
        "aspect_ratio": "auto",
        "duration_seconds": 5.0,
    }
    expected = []
    for row in conditions:
        condition = {
            "type": row["type"],
            "uri": (inputs / row["media"]).resolve().as_uri(),
            "role": row["role"],
        }
        if "frame_index" in row:
            condition["frame_index"] = row["frame_index"]
        expected.append(condition)
    assert body["conditions"] == expected
    # Both auto targets resolve to 16:9: the 16:9 keyframe, and the ref2va
    # default.
    assert record.video_shape == VideoShape(1344, 768, 124)


def test_a_480p_reference_target_is_its_fasth3_bucket(inputs):
    portrait = point(inputs, "ref2va", short_edge=480)
    portrait = replace(
        portrait, video=replace(portrait.video, aspect_ratio="9:16")
    )
    VideoTask.check_video(portrait.video, "point")
    request = VideoTask(portrait).build_request(
        Example("row", "scene", seconds=5, seed=7, conditions=IMAGE_AUDIO)
    )
    received, record = send(request)

    assert json.loads(received[1])["target"] == {
        "short_edge": 480,
        "aspect_ratio": "9:16",
        "duration_seconds": 5.0,
    }
    # The 480p 9:16 bucket is 480x832, from the FastH3 training table.
    assert record.video_shape == VideoShape(480, 832, 124)


def test_a_480p_target_names_its_aspect_ratio(inputs):
    with pytest.raises(ValueError, match="names its aspect ratio"):
        VideoTask.check_video(
            point(inputs, "ref2va", short_edge=480).video, "point"
        )


def test_an_fl2va_auto_canvas_follows_the_displayed_first_keyframe(inputs):
    """An EXIF-rotated portrait keyframe sets a portrait canvas."""
    task = VideoTask(point(inputs, "fl2va"))
    request = task.build_request(
        Example(
            "row",
            "scene",
            seconds=8,
            conditions=[
                {
                    "type": "image",
                    "media": "media/portrait.jpg",
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
        )
    )

    # 8 seconds are 192 frames; the displayed 36x64 keyframe is 9:16.
    assert request.video_shape == VideoShape(768, 1344, 192)


@pytest.mark.parametrize(
    "task,conditions,message",
    [
        ("fl2va", IMAGE_AUDIO, "fl2va takes image keyframes"),
        ("ref2va", None, "requires conditions"),
        (
            "ref2va",
            [{**FIRST_FRAME[0]}],
            "requires a reference",
        ),
        (
            "ref2va",
            [{"type": "audio", "media": "media/clip.mp4", "role": "reference"}],
            "is not audio media",
        ),
        (
            "ref2va",
            [{"type": "image", "media": "media/subject.png", "role": "anchor"}],
            "unknown role",
        ),
    ],
)
def test_a_row_whose_conditions_do_not_fit_the_task_is_not_sent(
    inputs, task, conditions, message
):
    with pytest.raises(ValueError, match=message):
        VideoTask(point(inputs, task)).build_request(
            Example("row", "scene", seconds=5, conditions=conditions)
        )


def test_a_t2va_row_with_conditions_is_not_sent(inputs):
    config = BenchmarkPoint(
        name="t2va",
        server="video",
        task=TaskName.VIDEO,
        model="MiniMax-H3",
        dataset="jsonl",
        endpoint=VIDEOS_SYNC,
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
    )
    with pytest.raises(ValueError, match="t2va takes no conditions"):
        VideoTask(config).build_request(
            Example("row", "scene", seconds=5, conditions=IMAGE_AUDIO)
        )


def test_a_row_naming_a_missing_file_is_not_sent(inputs):
    missing = [
        {"type": "audio", "media": "media/missing.wav", "role": "reference"}
    ]
    with pytest.raises(FileNotFoundError, match="missing.wav"):
        VideoTask(point(inputs, "ref2va")).build_request(
            Example("row", "scene", seconds=5, conditions=missing)
        )


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_server_reported_timings_are_recorded_and_summarized(inputs, endpoint):
    """Timings come from sync headers or from the completed job."""
    stages = {"denoising_stage": 30.5, "video_decoding_stage": 2.25}

    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return web.json_response({"id": "job", "status": "queued"})
            if request.method == "GET" and not request.path.endswith(
                "/content"
            ):
                return web.json_response(
                    {
                        "id": "job",
                        "status": "completed",
                        "inference_time_s": 33.0,
                        "stage_durations": stages,
                    },
                )
            headers = {"content-type": "video/mp4"}
            if endpoint == VIDEOS_SYNC:
                headers |= {
                    "x-inference-time-s": "33.0",
                    "x-stage-durations": json.dumps(stages),
                }
            return web.Response(body=b"mp4", headers=headers)

        config = replace(
            point(inputs, "ref2va", poll_interval_s=0.001), endpoint=endpoint
        )
        request = VideoTask(config).build_request(
            Example("row", "scene", seconds=5, conditions=IMAGE_AUDIO)
        )
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            return await send_request(
                client, base_url, request, "row", task="video"
            )

    record = asyncio.run(exercise())
    assert record.server_inference_s == 33.0
    assert record.server_stage_s == stages
    assert record.record_dict()["server_stage_s"] == stages

    # Summaries cover the requests whose output decoded.
    record.decoded_video = DecodedVideo(
        data=b"",
        sha256="",
        byte_size=3,
        mime="video/mp4",
        width=1344,
        height=768,
        frame_count=124,
        fps_numerator=24,
        fps_denominator=1,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=32000,
        audio_samples=round(124 / 24 * 32000),
        sample_filename="sample.mp4",
    )
    metrics = summarize([record], 40.0)
    assert metrics["server_inference_ms"]["p50"] == 33_000.0
    assert metrics["server_stage_ms"]["denoising_stage"]["p50"] == 30_500.0


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_stage_durations_alone_are_recorded(inputs, endpoint):
    """A server may report its stage durations without the other timings."""
    stages = {"encoding": 4.5, "denoising": 30.0, "decoding": 2.0}

    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return web.json_response({"id": "job", "status": "queued"})
            if request.method == "GET" and not request.path.endswith(
                "/content"
            ):
                return web.json_response(
                    {
                        "id": "job",
                        "status": "completed",
                        "stage_durations": stages,
                    },
                )
            headers = {"content-type": "video/mp4"}
            if endpoint == VIDEOS_SYNC:
                headers["x-stage-durations"] = json.dumps(stages)
            return web.Response(body=b"mp4", headers=headers)

        config = replace(
            point(inputs, "ref2va", poll_interval_s=0.001), endpoint=endpoint
        )
        request = VideoTask(config).build_request(
            Example("row", "scene", seconds=5, conditions=VIDEO_AUDIO)
        )
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            return await send_request(
                client, base_url, request, "row", task="video"
            )

    record = asyncio.run(exercise())
    assert record.server_stage_s == stages
    assert record.server_inference_s is None
