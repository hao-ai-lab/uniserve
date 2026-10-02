"""Video requests, complete-media timing, excluded workloads and goodput."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
import httpx
import pytest

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


# The released schedules in sigma points including the clean endpoint, with
# the video and audio shifts: the base checkpoint and FastH3 8-Step V2.
BASE_SCHEDULE = {
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}
FAST_H3_SCHEDULE = {
    "num_inference_steps": 9,
    "flow_shift": 10.0,
    "audio_flow_shift": 3.0,
}


def stated_schedule(backend, schedule=FAST_H3_SCHEDULE):
    """The schedule fields a point measuring ``backend`` states."""
    if backend == "fastvideo":
        # FastVideo takes its shifts from the checkpoint.
        return {"num_inference_steps": schedule["num_inference_steps"]}
    return dict(schedule)


def form_fields(request):
    """Decode a multipart request's text fields."""
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: "
        + request.headers["content-type"].encode()
        + b"\r\n\r\n"
        + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        ).decode()
        for part in message.iter_parts()
    }


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
        video = DecodedVideo(
            data=b"",
            sha256="",
            byte_size=0,
            mime="video/mp4",
            width=1344,
            height=768,
            frame_count=frames,
            fps_numerator=24,
            fps_denominator=1,
            video_codec="h264",
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=32000,
            audio_samples=round(frames / 24 * 32000),
            sample_filename="sample.mp4",
            video_variance=1,
            audio_rms=0.1,
        )
        records.append(
            RequestRecord(
                request_id=str(seconds),
                task="video",
                requested_seconds=seconds,
                decoded_video=video,
            )
        )
    assert task.validate_output(records).valid
    records[-1].decoded_video = replace(
        records[-1].decoded_video, frame_count=124
    )
    assert not task.validate_output(records).valid


@pytest.mark.parametrize(
    "frames,audio_samples,valid",
    [
        # One frame short with full-length audio, as FastVideo returns 15 s.
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
    video = DecodedVideo(
        data=b"",
        sha256="",
        byte_size=0,
        mime="video/mp4",
        width=1344,
        height=768,
        frame_count=frames,
        fps_numerator=24,
        fps_denominator=1,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=32000,
        audio_samples=audio_samples,
        sample_filename="sample.mp4",
        video_variance=1,
        audio_rms=0.1,
    )
    record = RequestRecord(
        request_id="15", task="video", requested_seconds=15, decoded_video=video
    )

    assert VideoTask(point()).validate_output([record]).valid is valid


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
    video = DecodedVideo(
        data=b"",
        sha256="",
        byte_size=0,
        mime="video/mp4",
        width=width,
        height=height,
        frame_count=124,
        fps_numerator=24,
        fps_denominator=1,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=32000,
        audio_samples=round(124 / 24 * 32000),
        sample_filename="sample.mp4",
        video_variance=1,
        audio_rms=0.1,
    )
    record = RequestRecord(
        request_id="5", task="video", requested_seconds=5, decoded_video=video
    )
    task = VideoTask(point(video=VideoConfig(aspect_ratio=aspect_ratio)))

    assert task.validate_output([record]).valid
    record.decoded_video = replace(video, width=width + 32)
    assert not task.validate_output([record]).checks["target_canvas"]


@pytest.mark.parametrize(
    "backend", ["uniserve", "sglang", "vllm-omni", "fastvideo"]
)
@pytest.mark.parametrize(
    "aspect_ratio,canvas", [("16:9", (1344, 768)), ("9:16", (768, 1344))]
)
@pytest.mark.parametrize("schedule", [BASE_SCHEDULE, FAST_H3_SCHEDULE])
@pytest.mark.parametrize("seconds,frames", [(5, 124), (7.3, 175)])
def test_each_backend_receives_the_target_and_schedule_in_its_own_fields(
    backend, aspect_ratio, canvas, schedule, seconds, frames
):
    """The canonical body reaches every backend as the same generated work."""
    sent = []

    async def exercise():
        async def handler(request):
            sent.append(request)
            return httpx.Response(
                200, content=b"mp4", headers={"content-type": "video/mp4"}
            )

        config = point(
            video=VideoConfig(
                backend=backend,
                aspect_ratio=aspect_ratio,
                **stated_schedule(backend, schedule),
            )
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=seconds, seed=11)
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client, "http://backend", request, "row", task="video"
            )
        assert record.success
        assert record.requested_seconds == seconds

    asyncio.run(exercise())
    width, height = canvas
    points = schedule["num_inference_steps"]
    if backend in ("uniserve", "sglang"):
        # Both take the official body; the stated schedule restates points.
        assert json.loads(sent[0].content) == {
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
            **schedule,
        }
    elif backend == "vllm-omni":
        fields = form_fields(sent[0])
        assert fields["prompt"] == "precise prompt"
        assert int(fields["seed"]) == 11
        assert (int(fields["width"]), int(fields["height"])) == canvas
        assert fields["aspect_ratio"] == aspect_ratio
        assert int(fields["fps"]) == 24
        # vLLM-Omni counts schedule intervals: one fewer than the points.
        assert int(fields["num_inference_steps"]) == points - 1
        assert float(fields["guidance_scale"]) == 1
        assert float(fields["flow_shift"]) == schedule["flow_shift"]
        assert json.loads(fields["extra_params"]) == {
            "task": "t2va",
            "duration": seconds,
            "audio_flow_shift": schedule["audio_flow_shift"],
        }
    else:
        payload = json.loads(sent[0].content)
        assert payload["prompt"] == "precise prompt"
        assert payload["seed"] == 11
        assert payload["size"] == f"{width}x{height}"
        assert payload["fps"] == 24
        assert payload["num_inference_steps"] == points
        assert payload["num_frames"] == frames
        # The generic seconds field takes integers; the frames carry the rest.
        assert payload.get("seconds") == (
            seconds if float(seconds).is_integer() else None
        )
        assert "flow_shift" not in payload
        assert "audio_flow_shift" not in payload


def test_uniserve_request_without_a_stated_schedule_omits_it():
    request = VideoTask(point()).build_request(
        Example("row", "precise prompt", seconds=15, seed=11)
    )

    assert request.payload == {
        "model": "h3",
        "prompt": "precise prompt",
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 15.0,
        },
        "seed": 11,
    }


@pytest.mark.parametrize("seconds", [3.99, 15.01])
def test_a_duration_outside_the_api_range_is_not_sent(seconds):
    with pytest.raises(ValueError, match="duration"):
        VideoTask(point()).build_request(
            Example("row", "precise prompt", seconds=seconds)
        )


@pytest.mark.parametrize(
    "backend,endpoint",
    [
        ("uniserve", VIDEOS_SYNC),
        ("vllm-omni", VIDEOS_SYNC),
        ("fastvideo", VIDEOS_SYNC),
        ("sglang", "/v1/videos"),
    ],
)
def test_native_transport_waits_for_media_and_enforces_logical_deadline(
    backend, endpoint
):
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
                    200, json={"id": "job", "status": "completed"}
                )
            await asyncio.sleep(0.03)
            return httpx.Response(
                200,
                content=b"original bytes",
                headers={"content-type": "video/mp4"},
            )

        config = replace(
            point(
                video=VideoConfig(
                    backend=backend,
                    poll_interval_s=0.001,
                    **(
                        stated_schedule(backend)
                        if backend != "uniserve"
                        else {}
                    ),
                )
            ),
            endpoint=endpoint,
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=5, seed=11)
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client,
                "http://backend",
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
                "http://backend",
                request,
                "row",
                task="video",
                timeout_s=0.01,
            )
            assert not expired.success
            assert expired.classifier == "request_deadline"
            assert expired.latency == pytest.approx(0.01)

    asyncio.run(exercise())


def test_vllm_omni_request_carries_configured_extra_params():
    async def exercise():
        sent = {}

        async def handler(request):
            sent.update(json.loads(form_fields(request)["extra_params"]))
            return httpx.Response(
                200,
                content=b"original bytes",
                headers={"content-type": "video/mp4"},
            )

        config = point(
            video=VideoConfig(
                backend="vllm-omni",
                extra_params={"preencode_mp4": True},
                **FAST_H3_SCHEDULE,
            )
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=10, seed=11)
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client, "http://backend", request, "row", task="video"
            )
        assert record.success
        # The configured option joins the fields that fix the work.
        assert sent == {
            "task": "t2va",
            "duration": 10.0,
            "audio_flow_shift": 3.0,
            "preencode_mp4": True,
        }

    asyncio.run(exercise())


def test_sglang_request_carries_configured_extra_params():
    async def exercise():
        sent = {}

        async def handler(request):
            if request.method == "POST":
                sent.update(json.loads(request.content))
                return httpx.Response(
                    200, json={"id": "job", "status": "completed"}
                )
            return httpx.Response(
                200,
                content=b"original bytes",
                headers={"content-type": "video/mp4"},
            )

        config = replace(
            point(
                video=VideoConfig(
                    backend="sglang",
                    extra_params={
                        "x264_preset": "ultrafast",
                        "output_compression": 53,
                    },
                    **FAST_H3_SCHEDULE,
                )
            ),
            endpoint="/v1/videos",
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=10, seed=11)
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client, "http://backend", request, "row", task="video"
            )
        assert record.success
        # The configured options join the native fields that fix the work.
        assert sent["x264_preset"] == "ultrafast"
        assert sent["output_compression"] == 53
        assert sent["task"] == "t2va"
        assert sent["target"]["duration_seconds"] == 10

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "backend,extra_params,message",
    [
        ("fastvideo", {"preencode_mp4": True}, "only to sglang and vllm-omni"),
        ("uniserve", {"preencode_mp4": True}, "only to sglang and vllm-omni"),
        ("vllm-omni", {"duration": 5}, "duration"),
        ("vllm-omni", {"task": "t2v"}, "task"),
        ("sglang", {"target": {}}, "target"),
        ("sglang", {"num_inference_steps": 4}, "num_inference_steps"),
    ],
)
def test_video_extra_params_cannot_change_the_work_or_go_unsent(
    backend, extra_params, message
):
    schedule = stated_schedule(backend) if backend != "uniserve" else {}
    with pytest.raises(ValueError, match=message):
        VideoConfig(backend=backend, extra_params=extra_params, **schedule)


@pytest.mark.parametrize(
    "settings,message",
    [
        # A baseline runs whatever schedule it is sent.
        ({"backend": "sglang"}, "requires"),
        (
            {
                "backend": "vllm-omni",
                "num_inference_steps": 9,
                "flow_shift": 10.0,
            },
            "audio_flow_shift",
        ),
        ({"backend": "fastvideo"}, "num_inference_steps"),
        # FastVideo takes its shifts from the checkpoint.
        (
            {
                "backend": "fastvideo",
                "num_inference_steps": 9,
                "flow_shift": 10.0,
            },
            "cannot receive",
        ),
        ({"num_inference_steps": 1}, "at least 2"),
        ({"flow_shift": 0.0}, "flow_shift"),
        ({"audio_flow_shift": float("nan")}, "audio_flow_shift"),
        # Rows carry no condition media, which fl2va and ref2va require.
        ({"task": "fl2va"}, "condition media"),
    ],
)
def test_video_config_refuses_work_a_backend_cannot_be_sent(settings, message):
    with pytest.raises(ValueError, match=message):
        VideoConfig(**settings)


@pytest.mark.parametrize(
    "video,message",
    [
        ('aspect_ratio = "16:10"', "aspect_ratio must be auto or one of"),
        ('aspect_ratio = "016:9"', "aspect_ratio must be auto or one of"),
        ('backend = "sglang"', "requires"),
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
def test_async_job_failure_or_published_media_url_is_terminal(failed):
    async def exercise():
        async def handler(request):
            if request.method == "POST":
                return httpx.Response(
                    200, json={"id": "job", "status": "queued"}
                )
            if request.url.path == "/v1/videos/job":
                return httpx.Response(
                    200,
                    json={
                        "id": "job",
                        "status": "failed" if failed else "completed",
                        "error": "decoder failed",
                        "url": "/published/output.mp4",
                    },
                )
            if request.url.path == "/published/output.mp4":
                await asyncio.sleep(0.02)
                return httpx.Response(
                    200,
                    content=b"published media",
                    headers={"content-type": "video/mp4"},
                )
            return httpx.Response(404)

        task = VideoTask(
            replace(
                point(
                    video=VideoConfig(
                        backend="sglang",
                        poll_interval_s=0.001,
                        **FAST_H3_SCHEDULE,
                    )
                ),
                endpoint="/v1/videos",
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            record = await send_request(
                client,
                "http://backend",
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
            assert record.video_body == b"published media"
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
            warmup_requests=0,
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
            warmup_requests=0,
            warmup_rows=warmup,
        )
    )
    assert [row for row, _ in observed] == [row.id for row in warmup] + [
        "measured"
    ]
    assert all(scheduled is None for _, scheduled in observed[:6])
    assert [record.request_id for record in result.outputs] == ["measured"]
