"""Conditioned video requests reach every backend as the same work.

A conditioned row names its media relative to the point's condition root.
UniServe and SGLang receive the canonical body with ``file://`` URIs,
vLLM-Omni the same files as uploaded parts, and FastVideo their local paths;
each output is validated against the canvas its own request resolves to.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import httpx
import pytest
from PIL import Image

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

BASE_SCHEDULE = {
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}
# FastH3 OmniRef: eight fused-block forwards over nine sigma points.
OMNIREF_POINTS = 9
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


def point(inputs: Path, task: str, backend: str = "uniserve", **video):
    settings = {"num_inference_steps": BASE_SCHEDULE["num_inference_steps"]}
    if backend in ("sglang", "vllm-omni"):
        settings = dict(BASE_SCHEDULE)
    elif backend == "uniserve":
        settings = {}
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
            backend=backend,
            condition_root=str(inputs),
            **{**settings, **video},
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
            sent.append(received)
            return httpx.Response(
                200, content=b"mp4", headers={"content-type": "video/mp4"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client, "http://backend", request, "row", task="video"
            )
        assert record.success
        return record

    record = asyncio.run(exercise())
    return sent[0], record


def multipart(request):
    """Decode a multipart request into its text fields and file parts."""
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + request.headers["content-type"].encode()
        + b"\r\n\r\n"
        + request.content
    )
    fields, files = {}, []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        if filename is None:
            fields[name] = part.get_payload(decode=True).decode()
        else:
            files.append(
                (
                    name,
                    filename,
                    part.get_content_type(),
                    part.get_payload(decode=True),
                )
            )
    return fields, files


@pytest.mark.parametrize("backend", ["uniserve", "sglang"])
@pytest.mark.parametrize(
    "task,conditions",
    [("fl2va", FIRST_FRAME), ("ref2va", IMAGE_AUDIO), ("ref2va", VIDEO_AUDIO)],
    ids=["first-frame", "image-audio", "video-audio"],
)
def test_the_canonical_body_names_condition_media_by_file_uri(
    inputs, backend, task, conditions
):
    task_adapter = VideoTask(point(inputs, task, backend))
    request = task_adapter.build_request(
        Example("row", "scene", seconds=5, seed=7, conditions=conditions)
    )
    received, record = send(request)

    body = json.loads(received.content)
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


@pytest.mark.parametrize(
    "task,conditions,kinds",
    [
        ("fl2va", FIRST_FRAME, ["image/png"]),
        ("ref2va", IMAGE_AUDIO, ["image/png", "audio/mpeg"]),
        ("ref2va", VIDEO_AUDIO, ["video/mp4", "audio/mpeg"]),
    ],
    ids=["first-frame", "image-audio", "video-audio"],
)
def test_vllm_omni_receives_the_same_files_as_uploaded_references(
    inputs, task, conditions, kinds
):
    request = VideoTask(point(inputs, task, "vllm-omni")).build_request(
        Example("row", "scene", seconds=5, seed=7, conditions=conditions)
    )
    received, _ = send(request)

    fields, files = multipart(received)
    assert [(name, kind) for name, _, kind, _ in files] == [
        ("input_references", kind) for kind in kinds
    ]
    assert [data for *_, data in files] == [
        (inputs / row["media"]).read_bytes() for row in conditions
    ]
    assert (int(fields["width"]), int(fields["height"])) == (1344, 768)
    assert int(fields["num_inference_steps"]) == 49
    extra = json.loads(fields["extra_params"])
    assert extra["task"] == task
    assert extra["duration"] == 5.0
    # Keyframes carry their frame indices; references need none.
    assert extra.get("frame_indices") == ([0] if task == "fl2va" else None)


@pytest.mark.parametrize(
    "conditions,references",
    [
        (
            IMAGE_AUDIO,
            {
                "image_reference": [{"image_url": "media/subject.png"}],
                "audio_reference": [{"audio_url": "media/voice.mp3"}],
            },
        ),
        (
            VIDEO_AUDIO,
            {
                "video_reference": [{"video_url": "media/clip.mp4"}],
                "audio_reference": [{"audio_url": "media/voice.mp3"}],
            },
        ),
    ],
    ids=["image-audio", "video-audio"],
)
def test_fastvideo_receives_reference_paths_and_its_block_count(
    inputs, conditions, references
):
    config = point(
        inputs,
        "ref2va",
        "fastvideo",
        num_inference_steps=OMNIREF_POINTS,
        parallel_decoding=True,
    )
    request = VideoTask(config).build_request(
        Example("row", "scene", seconds=5, seed=7, conditions=conditions)
    )
    received, _ = send(request)

    body = json.loads(received.content)
    for field, entries in references.items():
        assert body[field] == [
            {key: str((inputs / path).resolve())}
            for entry in entries
            for key, path in entry.items()
        ]
    assert body["task"] == "ref2va"
    assert body["size"] == "1344x768"
    assert body["num_frames"] == 124
    # A parallel decoding student is asked for its fused-block forwards.
    assert body["num_inference_steps"] == OMNIREF_POINTS - 1


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


@pytest.mark.parametrize("backend", ["vllm-omni", "fastvideo"])
def test_a_type_grouping_backend_refuses_conditions_it_would_reorder(
    inputs, backend
):
    """Audio before an image would be presented after it there.

    The request is refused when it is built, before any measurement.
    """
    schedule = (
        {"num_inference_steps": OMNIREF_POINTS, "parallel_decoding": True}
        if backend == "fastvideo"
        else {}
    )
    task = VideoTask(point(inputs, "ref2va", backend, **schedule))
    with pytest.raises(ValueError, match="images, then videos, then audio"):
        task.build_request(
            Example("row", "scene", seconds=5, conditions=IMAGE_AUDIO[::-1])
        )


def test_fastvideo_refuses_a_reference_start_offset_it_cannot_take(inputs):
    config = point(
        inputs,
        "ref2va",
        "fastvideo",
        num_inference_steps=OMNIREF_POINTS,
        parallel_decoding=True,
    )
    offset = [{**VIDEO_AUDIO[0], "start_time_seconds": 1.5}, VIDEO_AUDIO[1]]
    with pytest.raises(ValueError, match="start offset"):
        VideoTask(config).build_request(
            Example("row", "scene", seconds=5, conditions=offset)
        )


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_server_reported_timings_are_recorded_and_summarized(inputs, endpoint):
    """Timings come from sync headers or from the completed job."""
    stages = {"denoising_stage": 30.5, "video_decoding_stage": 2.25}

    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return httpx.Response(
                    200, json={"id": "job", "status": "queued"}
                )
            if request.method == "GET" and not request.url.path.endswith(
                "/content"
            ):
                return httpx.Response(
                    200,
                    json={
                        "id": "job",
                        "status": "completed",
                        "inference_time_s": 33.0,
                        "stage_durations": stages,
                        "peak_memory_mb": 91000.5,
                    },
                )
            headers = {"content-type": "video/mp4"}
            if endpoint == VIDEOS_SYNC:
                headers |= {
                    "x-inference-time-s": "33.0",
                    "x-stage-durations": json.dumps(stages),
                    "x-peak-memory-mb": "91000.5",
                }
            return httpx.Response(200, content=b"mp4", headers=headers)

        config = point(
            inputs,
            "ref2va",
            "fastvideo",
            num_inference_steps=OMNIREF_POINTS,
            parallel_decoding=True,
            poll_interval_s=0.001,
        )
        config = replace(config, endpoint=endpoint)
        request = VideoTask(config).build_request(
            Example("row", "scene", seconds=5, conditions=IMAGE_AUDIO)
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await send_request(
                client, "http://backend", request, "row", task="video"
            )

    record = asyncio.run(exercise())
    assert record.server_inference_s == 33.0
    assert record.server_stage_s == stages
    assert record.server_peak_memory_mib == 91000.5
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
    assert metrics["server_peak_memory_mib"] == 91000.5
