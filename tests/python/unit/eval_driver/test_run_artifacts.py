from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from uniserve_eval.load import WarmupFailure
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    SamplingConfig,
    TaskName,
)

pytestmark = pytest.mark.unit


class _EmptyStreamHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        payload = b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_failed_warmup_writes_terminal_diagnostic_artifacts(tmp_path: Path) -> None:
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
        for line in (output / "warmup_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(warmup) == 1
    assert warmup[0]["classifier"] == "response_empty_output"
    assert (output / "requests.jsonl").read_text(encoding="utf-8") == ""
    assert not (output / "summary.json").exists()
