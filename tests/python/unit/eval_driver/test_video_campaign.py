"""Video requests, complete-media timing, excluded workloads and goodput."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import av
import pytest
from aiohttp import web

from tests.python.fixtures.http_stub import stub_server
from uniserve_eval.config import load_config
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.tasks.video import VideoTask
from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    VIDEOS_SYNC,
    BenchmarkPoint,
    DecodedVideo,
    Example,
    LoadConfig,
    MetricDefinition,
    RequestRecord,
    TaskName,
    VideoConfig,
    VideoShape,
)

pytestmark = pytest.mark.unit


def point(**kwargs):
    return BenchmarkPoint(
        name="video",
        server="video",
        task=TaskName.VIDEO,
        model="h3",
        dataset="jsonl",
        endpoint=VIDEOS_SYNC,
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        **kwargs,
    )


def decoded(width=1344, height=768, frames=124, audio_samples=None):
    """A decoded output with the H3 media contract at the given shape."""
    return DecodedVideo(
        data=b"",
        sha256="",
        byte_size=0,
        mime="video/mp4",
        width=width,
        height=height,
        frame_count=frames,
        fps_numerator=24,
        fps_denominator=1,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=32000,
        audio_samples=(
            round(frames / 24 * 32000)
            if audio_samples is None
            else audio_samples
        ),
        sample_filename="sample.mp4",
        video_variance=1,
        audio_rms=0.1,
    )


def answered(request, video):
    """The record of a request whose response decoded to ``video``."""
    return RequestRecord(
        request_id="row",
        task="video",
        requested_seconds=request.payload["target"]["duration_seconds"],
        video_shape=request.video_shape,
        decoded_video=video,
    )


def test_each_duration_uses_ties_to_even_and_native_alignment():
    task = VideoTask(point())
    records = []
    # 106.5 rounds to 106, giving 107; half-up would incorrectly yield 124.
    for seconds, frames in [
        (4, 107),
        (106.5 / 24, 107),
        (5, 124),
        (10, 243),
        (15, 362),
    ]:
        request = task.build_request(Example("row", "scene", seconds=seconds))
        assert request.video_shape == VideoShape(1344, 768, frames)
        records.append(answered(request, decoded(frames=frames)))
    assert task.validate_output(records).valid
    records[-1].decoded_video = replace(
        records[-1].decoded_video, frame_count=124
    )
    assert not task.validate_output(records).valid


@pytest.mark.parametrize(
    "frames,audio_samples,valid",
    [
        # One frame short or long with full-length audio.
        (361, 483328, True),
        (363, 483328, True),
        (360, 483328, False),
        # Audio truncated to the short video exceeds one frame period.
        (361, round(360 / 24 * 32000), False),
    ],
)
def test_media_length_tolerates_one_frame_around_the_aligned_duration(
    frames, audio_samples, valid
):
    task = VideoTask(point())
    request = task.build_request(Example("row", "scene", seconds=15))
    record = answered(
        request, decoded(frames=frames, audio_samples=audio_samples)
    )

    assert task.validate_output([record]).valid is valid


@pytest.mark.parametrize(
    "aspect_ratio,canvas",
    [
        ("auto", (1344, 768)),
        ("21:9", (1536, 672)),
        ("16:9", (1344, 768)),
        ("4:3", (1024, 768)),
        ("1:1", (768, 768)),
        ("3:4", (768, 1024)),
        ("9:16", (768, 1344)),
    ],
)
def test_media_must_have_the_canvas_the_target_resolves_to(
    aspect_ratio, canvas
):
    width, height = canvas
    task = VideoTask(point(video=VideoConfig(aspect_ratio=aspect_ratio)))
    request = task.build_request(Example("row", "scene", seconds=5))
    record = answered(request, decoded(width, height))

    assert task.validate_output([record]).valid
    record.decoded_video = decoded(width + 32, height)
    assert not task.validate_output([record]).checks["target_canvas"]


@pytest.mark.parametrize(
    "aspect_ratio,canvas", [("16:9", (1344, 768)), ("9:16", (768, 1344))]
)
@pytest.mark.parametrize("seconds,frames", [(5, 124), (7.3, 175)])
def test_the_server_receives_the_canonical_body_without_a_schedule(
    aspect_ratio, canvas, seconds, frames
):
    """The checkpoint fixes the schedule, so the request states none."""
    sent = []

    async def exercise():
        async def handler(request):
            sent.append(await request.read())
            return web.Response(
                body=b"mp4", headers={"content-type": "video/mp4"}
            )

        config = point(video=VideoConfig(aspect_ratio=aspect_ratio))
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=seconds, seed=11)
        )
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            record = await send_request(
                client, base_url, request, "row", task="video"
            )
        assert record.success
        assert record.requested_seconds == seconds
        assert record.video_shape == VideoShape(*canvas, frames)

    asyncio.run(exercise())
    assert json.loads(sent[0]) == {
        "model": "h3",
        "prompt": "precise prompt",
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": seconds,
        },
        "seed": 11,
    }


@pytest.mark.parametrize("seconds", [3.99, 15.01])
def test_a_duration_outside_the_api_range_is_not_sent(seconds):
    with pytest.raises(ValueError, match="duration"):
        VideoTask(point()).build_request(
            Example("row", "precise prompt", seconds=seconds)
        )


@pytest.mark.parametrize("endpoint", [VIDEOS_SYNC, "/v1/videos"])
def test_native_transport_waits_for_media_and_enforces_logical_deadline(
    endpoint,
):
    async def exercise():
        async def handler(request):
            if request.method == "POST" and endpoint == "/v1/videos":
                return web.json_response({"id": "job", "status": "queued"})
            if request.method == "GET" and not request.path.endswith(
                "/content"
            ):
                return web.json_response({"id": "job", "status": "completed"})
            await asyncio.sleep(0.03)
            return web.Response(
                body=b"original bytes",
                headers={"content-type": "video/mp4"},
            )

        config = replace(
            point(video=VideoConfig(poll_interval_s=0.001)),
            endpoint=endpoint,
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=5, seed=11)
        )
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            record = await send_request(
                client,
                base_url,
                request,
                "row",
                task="video",
                prompt_len=1000,
                timeout_s=1,
            )
            assert record.success and record.video_body == b"original bytes"
            assert record.prompt_len == 1000
            assert record.latency >= 0.03
            expired = await send_request(
                client,
                base_url,
                request,
                "row",
                task="video",
                timeout_s=0.01,
            )
            assert not expired.success
            assert expired.classifier == "request_deadline"
            assert expired.latency == pytest.approx(0.01)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "settings,message",
    [
        # A conditioned task reads its rows' media from a condition root,
        # which a t2va point has no use for.
        ({"task": "fl2va"}, "condition_root"),
        ({"task": "ref2va"}, "condition_root"),
        ({"condition_root": "inputs"}, "condition_root"),
        ({"poll_interval_s": 0.0}, "poll_interval_s"),
    ],
)
def test_video_config_refuses_settings_it_cannot_send(settings, message):
    with pytest.raises(ValueError, match=message):
        VideoConfig(**settings)


@pytest.mark.parametrize(
    "video,message",
    [
        ('aspect_ratio = "16:10"', "aspect_ratio must be auto or one of"),
        ('aspect_ratio = "016:9"', "aspect_ratio must be auto or one of"),
        # The checkpoint fixes the schedule, and UniServe is the only server.
        ("num_inference_steps = 9", "unknown fields: num_inference_steps"),
        ('backend = "uniserve"', "unknown fields: backend"),
    ],
)
def test_profiles_refuse_a_video_point_that_cannot_be_built(
    tmp_path, video, message
):
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
[servers.engine]
port = 8000
command = ["server"]

[benchmarks.point]
server = "engine"
task = "video"
model = "MiniMax-H3"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.point.video]
{video}

[benchmarks.point.metrics]
videos_per_second = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"benchmarks.point.video.*{message}"):
        load_config(profile)


@pytest.mark.parametrize("failed", [False, True])
def test_a_failed_job_is_terminal_and_a_completed_one_reads_its_content(
    failed,
):
    async def exercise():
        async def handler(request):
            if request.method == "POST":
                return web.json_response({"id": "job", "status": "queued"})
            if request.path == "/v1/videos/job":
                return web.json_response(
                    {
                        "id": "job",
                        "status": "failed" if failed else "completed",
                        "error": "decoder failed",
                    },
                )
            if request.path == "/v1/videos/job/content":
                await asyncio.sleep(0.02)
                return web.Response(
                    body=b"job media",
                    headers={"content-type": "video/mp4"},
                )
            return web.Response(status=404)

        task = VideoTask(
            replace(
                point(video=VideoConfig(poll_interval_s=0.001)),
                endpoint="/v1/videos",
            )
        )
        async with (
            stub_server(handler) as base_url,
            aiohttp.ClientSession() as client,
        ):
            record = await send_request(
                client,
                base_url,
                task.build_request(Example("row", "scene", seconds=5)),
                "row",
                task="video",
            )
        assert record.final_event_time >= record.start_time
        if failed:
            assert not record.success
            assert record.classifier == "video_job_failed"
            assert "decoder failed" in record.error
        else:
            assert record.success
            assert record.video_body == b"job media"
            assert record.latency >= 0.02

    asyncio.run(exercise())


def test_run_releases_slots_before_inspection_and_retains_invalid_media(
    tmp_path, media, monkeypatch
):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            request = json.loads(
                self.rfile.read(int(self.headers["content-length"]))
            )
            received.append(request["prompt"])
            body = b"broken mp4" if request["prompt"] == "invalid" else media
            self.send_response(200)
            self.send_header("content-type", "video/mp4")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            pass

    rows = [
        {"id": prompt, "prompt": prompt, "seconds": 4}
        for prompt in ("valid", "invalid")
    ]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    warmup = tmp_path / "warmup.jsonl"
    priming = tmp_path / "priming.jsonl"
    for path, identity in ((warmup, "warmup"), (priming, "prime")):
        path.write_text(
            json.dumps({"id": identity, "prompt": identity, "seconds": 4})
            + "\n"
        )
    # PyAV is an external decoder boundary. Decoding any response before both
    # requests arrive would occupy the first closed-loop slot and fail here.
    open_media = av.open

    def decode(*args, **kwargs):
        assert received in (["warmup"], ["warmup", "prime", "valid", "invalid"])
        return open_media(*args, **kwargs)

    monkeypatch.setattr(av, "open", decode)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = point(
        dataset_path=str(manifest),
        load=LoadConfig(
            num_prompts=2,
            max_concurrency=1,
            warmup_manifest=str(warmup),
            priming_manifest=str(priming),
        ),
    )
    try:
        result = asyncio.run(
            run_point(
                [f"http://127.0.0.1:{server.server_port}"],
                config,
                tmp_path / "result",
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    records = [
        json.loads(line)
        for line in (result.output_dir / "requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["success"] for record in records] == [True, False]
    assert (
        Path(records[1]["original_output"]["path"]).read_bytes()
        == b"broken mp4"
    )
    duration = max(record["final_event_time"] for record in records) - min(
        record["client_send_time"] for record in records
    )
    assert result.summary["elapsed_s"] == duration
    assert result.summary["metrics"]["videos_per_second"] == 1 / duration
    assert result.summary["failed_count"] == 1
    for filename, identity in (
        ("warmup_requests.jsonl", "warmup"),
        ("priming_requests.jsonl", "prime"),
    ):
        excluded = [
            json.loads(line)
            for line in (result.output_dir / filename).read_text().splitlines()
        ]
        assert [record["request_id"] for record in excluded] == [identity]
        assert excluded[0]["success"]


def test_excluded_manifests_preserve_order_and_do_not_enter_measurements(
    tmp_path,
):
    from uniserve_eval.load.arrival import run_load

    observed = []
    warmup = [
        Example(
            f"warm-{duration}-{tokens}",
            "warm",
            seconds=duration,
            prompt_len=tokens,
        )
        for duration in (5, 10, 15)
        for tokens in (1000, 10000)
    ]
    measured = [Example("measured", "measure")]

    async def submit(row, scheduled):
        observed.append((row.id, scheduled))
        record = RequestRecord(
            request_id=row.id, task="video", start_time=time.perf_counter()
        )
        record.close_now()
        record.mark_success()
        return record

    result = asyncio.run(
        run_load(
            measured,
            request_rate=float("inf"),
            max_concurrency=1,
            submit=submit,
            warmup_rows=warmup,
        )
    )
    assert [row for row, _ in observed] == [row.id for row in warmup] + [
        "measured"
    ]
    assert all(scheduled is None for _, scheduled in observed[:6])
    assert [record.request_id for record in result.outputs] == ["measured"]
