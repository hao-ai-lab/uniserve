"""Complete-media timing, excluded workloads and per-request goodput."""

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
    "backend,endpoint",
    [
        ("uniserve", VIDEOS_SYNC),
        ("vllm-omni", VIDEOS_SYNC),
        ("fastvideo", VIDEOS_SYNC),
        ("sglang", "/v1/videos"),
    ],
)
@pytest.mark.parametrize("seconds,frames", [(5, 124), (106.5 / 24, 107)])
def test_native_transport_waits_for_media_and_enforces_logical_deadline(
    backend, endpoint, seconds, frames
):
    async def exercise():
        async def handler(request):
            if request.method == "POST":
                if backend == "vllm-omni":
                    message = BytesParser(policy=default).parsebytes(
                        b"Content-Type: "
                        + request.headers["content-type"].encode()
                        + b"\r\n\r\n"
                        + request.content
                    )
                    fields = {
                        part.get_param(
                            "name", header="content-disposition"
                        ): part.get_payload(decode=True).decode()
                        for part in message.iter_parts()
                    }
                    assert fields["prompt"] == "precise prompt"
                    assert int(fields["num_inference_steps"]) == 8
                    assert int(fields["width"]) == 1344
                    assert int(fields["height"]) == 768
                    assert int(fields["fps"]) == 24
                    assert fields["aspect_ratio"] == "16:9"
                    assert float(fields["guidance_scale"]) == 1
                    assert float(fields["flow_shift"]) == 10
                    assert json.loads(fields["extra_params"]) == {
                        "task": "t2va",
                        "duration": seconds,
                        "audio_flow_shift": 3.0,
                    }
                else:
                    assert (
                        json.loads(request.content)["prompt"]
                        == "precise prompt"
                    )
                    if backend == "fastvideo":
                        payload = json.loads(request.content)
                        if float(seconds).is_integer():
                            assert payload["seconds"] == seconds
                        else:
                            assert "seconds" not in payload
                        assert payload["num_frames"] == frames
                if endpoint == "/v1/videos":
                    payload = json.loads(request.content)
                    assert payload["task"] == "t2va"
                    assert payload["conditions"] == []
                    assert payload["target"] == {
                        "short_edge": 768,
                        "aspect_ratio": "16:9",
                        "duration_seconds": seconds,
                    }
                    # Native H3 rejects generic transport timing fields.
                    assert "fps" not in payload and "num_frames" not in payload
                    return httpx.Response(
                        200, json={"id": "job", "status": "queued"}
                    )
            elif not request.url.path.endswith("/content"):
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
            point(video=VideoConfig(backend=backend, poll_interval_s=0.001)),
            endpoint=endpoint,
        )
        request = VideoTask(config).build_request(
            Example("row", "precise prompt", seconds=seconds, seed=11)
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
                    video=VideoConfig(backend="sglang", poll_interval_s=0.001)
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
