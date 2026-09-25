from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image

from uniserve_eval.datasets.jsonl import JsonlDataset
from uniserve_eval.load import WarmupFailure
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.transport.images import inspect_image_bytes
from uniserve_eval.types import (
    IMAGES_GENERATIONS,
    BenchmarkPoint,
    ImageConfig,
    LoadConfig,
    MetricDefinition,
    SamplingConfig,
    TaskName,
)

pytestmark = pytest.mark.unit


def test_jsonl_video_row_preserves_duration_override(tmp_path: Path) -> None:
    dataset = tmp_path / "video.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "heldout-video",
                "prompt": "A camera pans across a quiet harbor.",
                "seed": 73001,
                "seconds": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    point = BenchmarkPoint(
        name="video",
        server="server",
        task=TaskName.VIDEO,
        model="model",
        dataset="jsonl",
        dataset_path=str(dataset),
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        load=LoadConfig(num_prompts=1),
    )

    examples = JsonlDataset(point).load()

    assert examples[0].seconds == 10.0


class _EmptyStreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        payload = (
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_failed_warmup_writes_terminal_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(
        json.dumps({"id": "row", "prompt": "hello", "output_len": 2}) + "\n",
        encoding="utf-8",
    )
    point = BenchmarkPoint(
        name="text",
        server="server",
        task=TaskName.TEXT,
        model="model",
        dataset="jsonl",
        dataset_path=str(dataset),
        metrics=(MetricDefinition(("output_throughput",), "higher"),),
        load=LoadConfig(num_prompts=1, warmup_requests=1),
        sampling=SamplingConfig(ignore_eos=True, stream=True),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmptyStreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result"
    try:
        with pytest.raises(WarmupFailure, match="response_empty_output"):
            asyncio.run(
                run_point(
                    f"http://127.0.0.1:{server.server_port}",
                    point,
                    output,
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    state = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["valid"] is False
    assert state["selected_rows"]["count"] == 1
    assert state["error"]["type"] == "WarmupFailure"
    warmup = [
        json.loads(line)
        for line in (output / "warmup_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(warmup) == 1
    assert warmup[0]["classifier"] == "response_empty_output"
    assert (output / "requests.jsonl").read_text(encoding="utf-8") == ""
    assert not (output / "summary.json").exists()


class _CaptionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        payload = json.dumps(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "beans"}}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_interrupted_measurement_persists_the_finished_requests(
    tmp_path: Path,
) -> None:
    # The third row has no input image, so building its request raises
    # inside the measured window after the first two rows were served.
    dataset = tmp_path / "rows.jsonl"
    rows = [
        {"id": "row-0", "prompt": "describe", "input_image_b64": "aW1hZ2U="},
        {"id": "row-1", "prompt": "describe", "input_image_b64": "aW1hZ2U="},
        {"id": "row-2", "prompt": "describe"},
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    # One request at a time, so both served rows finish before the third
    # row's request is built.
    point = BenchmarkPoint(
        name="i2t",
        server="server",
        task=TaskName.I2T,
        model="model",
        dataset="jsonl",
        dataset_path=str(dataset),
        metrics=(MetricDefinition(("output_throughput",), "higher"),),
        load=LoadConfig(num_prompts=3, max_concurrency=1, warmup_requests=0),
        sampling=SamplingConfig(ignore_eos=False, stream=False, max_tokens=8),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result"
    try:
        with pytest.raises(ValueError, match="base64"):
            asyncio.run(
                run_point(
                    f"http://127.0.0.1:{server.server_port}",
                    point,
                    output,
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    state = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["valid"] is False
    assert state["error"]["type"] == "ValueError"
    measured = [
        json.loads(line)
        for line in (output / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["request_id"] for record in measured] == ["row-0", "row-1"]
    assert all(record["success"] for record in measured)
    assert not (output / "summary.json").exists()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_failed_sample_persistence_still_records_the_failed_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    png = _png()
    sample = output / "samples" / inspect_image_bytes(png).sample_filename

    class ImageHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
            length = int(self.headers.get("content-length", "0"))
            self.rfile.read(length)
            # A directory occupying the sample's path makes every attempt to
            # persist the returned image fail the same way.
            sample.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(
        json.dumps({"id": "row", "prompt": "draw"}) + "\n", encoding="utf-8"
    )
    point = BenchmarkPoint(
        name="t2i",
        server="server",
        task=TaskName.T2I,
        model="model",
        dataset="jsonl",
        dataset_path=str(dataset),
        endpoint=IMAGES_GENERATIONS,
        metrics=(MetricDefinition(("images_per_second",), "higher"),),
        load=LoadConfig(num_prompts=1, warmup_requests=0),
        sampling=SamplingConfig(stream=False),
        image=ImageConfig(image_count=1, width=2, height=3),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(IsADirectoryError):
            asyncio.run(
                run_point(
                    f"http://127.0.0.1:{server.server_port}",
                    point,
                    output,
                )
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    state = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["valid"] is False
    assert state["error"]["type"] == "IsADirectoryError"
    assert state["artifact_error"]["type"] == "IsADirectoryError"
    assert not (output / "summary.json").exists()
