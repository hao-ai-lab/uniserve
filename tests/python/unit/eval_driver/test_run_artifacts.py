from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from uniserve_eval.datasets.jsonl import JsonlDataset
from uniserve_eval.load import WarmupFailure
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
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


def _video_point(
    dataset: Path, prompts: list[str], load: LoadConfig
) -> BenchmarkPoint:
    """A t2va point whose four-second rows carry the given prompts."""
    dataset.write_text(
        "".join(
            json.dumps({"id": prompt, "prompt": prompt, "seconds": 4}) + "\n"
            for prompt in prompts
        ),
        encoding="utf-8",
    )
    return BenchmarkPoint(
        name="video",
        server="server",
        task=TaskName.VIDEO,
        model="model",
        dataset="jsonl",
        dataset_path=str(dataset),
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        load=load,
    )


def _serve(
    respond: Callable[[BaseHTTPRequestHandler, dict], None],
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Answer every POST with `respond(handler, request_body)`."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
            length = int(self.headers.get("content-length", "0"))
            respond(self, json.loads(self.rfile.read(length)))

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name.
            self.send_response(404)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _send_video(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    handler.send_response(200)
    handler.send_header("content-type", "video/mp4")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def test_failed_warmup_writes_terminal_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    # An empty body fails inspection, which runs on the warmup outputs
    # before the measured window opens.
    point = _video_point(
        tmp_path / "rows.jsonl",
        ["row"],
        LoadConfig(num_prompts=1, warmup_requests=1),
    )
    server, thread = _serve(lambda handler, _: _send_video(handler, b""))
    output = tmp_path / "result"
    try:
        with pytest.raises(WarmupFailure, match="response_empty_video"):
            asyncio.run(
                run_point(
                    [f"http://127.0.0.1:{server.server_port}"],
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
    assert warmup[0]["classifier"] == "response_empty_video"
    assert (output / "requests.jsonl").read_text(encoding="utf-8") == ""
    assert not (output / "summary.json").exists()


def test_interrupted_measurement_persists_the_finished_requests(
    tmp_path: Path, media: bytes
) -> None:
    # One request at a time: the first two rows are served, and the run is
    # cancelled while the server holds the third.
    point = _video_point(
        tmp_path / "rows.jsonl",
        ["row-0", "row-1", "row-2"],
        LoadConfig(num_prompts=3, max_concurrency=1, warmup_requests=0),
    )
    held = threading.Event()
    release = threading.Event()

    def respond(handler: BaseHTTPRequestHandler, request: dict) -> None:
        if request["prompt"] == "row-2":
            held.set()
            release.wait(timeout=10)
        _send_video(handler, media)

    server, thread = _serve(respond)
    output = tmp_path / "result"

    async def interrupt() -> None:
        run = asyncio.create_task(
            run_point(
                [f"http://127.0.0.1:{server.server_port}"],
                point,
                output,
            )
        )
        # A run that fails before the third request raises its own error.
        while not held.is_set() and not run.done():
            await asyncio.sleep(0.01)
        run.cancel()
        await run

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(interrupt())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    state = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["valid"] is False
    assert state["error"]["type"] == "CancelledError"
    measured = [
        json.loads(line)
        for line in (output / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["request_id"] for record in measured] == ["row-0", "row-1"]
    assert all(record["success"] for record in measured)
    assert not (output / "summary.json").exists()


def test_failed_sample_persistence_still_records_the_failed_run(
    tmp_path: Path, media: bytes
) -> None:
    output = tmp_path / "result"
    sample = output / "samples" / f"{hashlib.sha256(media).hexdigest()}.mp4"

    def respond(handler: BaseHTTPRequestHandler, _request: dict) -> None:
        # A directory occupying the sample's path makes every attempt to
        # persist the returned video fail the same way.
        sample.mkdir(parents=True, exist_ok=True)
        _send_video(handler, media)

    point = _video_point(
        tmp_path / "rows.jsonl",
        ["row"],
        LoadConfig(num_prompts=1, warmup_requests=0),
    )
    server, thread = _serve(respond)
    try:
        with pytest.raises(IsADirectoryError):
            asyncio.run(
                run_point(
                    [f"http://127.0.0.1:{server.server_port}"],
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
