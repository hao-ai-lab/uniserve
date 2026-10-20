"""Measured-window telemetry: server counters and selected-GPU utilization."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from uniserve_eval.load.gpu import GpuStorageSampler
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.server_metrics import counter_deltas, window_summary
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    TaskName,
)

pytestmark = pytest.mark.unit

_BEFORE = """# TYPE uniserve:scheduler_domain_time_us counter
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="device"} 1000
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="queue"} 50
uniserve:request_prefill_time_seconds_sum{model_name="m"} 1.5
uniserve:request_prefill_time_seconds_count{model_name="m"} 3
uniserve:kv_cache_usage_perc{model_name="m"} 0.25
http_requests_total{path="/v1/videos/sync",reason="a b"} 4
# EOF
"""

_AFTER = """# TYPE uniserve:scheduler_domain_time_us counter
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="device"} 1600
uniserve:scheduler_domain_time_us_total{domain="prefill",phase="queue"} 50
uniserve:scheduler_domain_time_us_total{domain="denoise",phase="device"} 40 17
uniserve:request_prefill_time_seconds_sum{model_name="m"} 2.5
uniserve:request_prefill_time_seconds_count{model_name="m"} 5
uniserve:kv_cache_usage_perc{model_name="m"} 0.75
http_requests_total{path="/v1/videos/sync",reason="a b"} 9
# EOF
"""


def test_counter_deltas_cover_counters_that_advanced_in_the_window() -> None:
    # The gauge and the unchanged queue counter are left out; a series that
    # first appears after the window opened started from zero, and its
    # trailing timestamp is not its value.
    assert counter_deltas(_BEFORE, _AFTER) == {
        'http_requests_total{path="/v1/videos/sync",reason="a b"}': 5.0,
        'uniserve:request_prefill_time_seconds_count{model_name="m"}': 2.0,
        'uniserve:request_prefill_time_seconds_sum{model_name="m"}': 1.0,
        'uniserve:scheduler_domain_time_us_total{domain="denoise",'
        'phase="device"}': 40.0,
        'uniserve:scheduler_domain_time_us_total{domain="prefill",'
        'phase="device"}': 600.0,
    }


def test_window_without_both_scrapes_is_unavailable() -> None:
    assert window_summary(None, _AFTER) == {
        "available": False,
        "counter_deltas": {},
    }


def _sample(time: float, gpu0: int, gpu1: int) -> dict[str, Any]:
    return {
        "time": time,
        "gpus": [
            {"index": 0, "memory_used_mib": 1, "utilization_gpu_pct": gpu0},
            {"index": 1, "memory_used_mib": 1, "utilization_gpu_pct": gpu1},
        ],
    }


def test_window_utilization_reports_selected_gpus_inside_the_window() -> None:
    sampler = GpuStorageSampler()
    # A warmup sample before the window, three inside it, one after it; GPU
    # 1 belongs to another job.
    sampler.sample_records.extend(
        [
            _sample(9.0, 100, 100),
            _sample(10.0, 60, 100),
            _sample(10.5, 80, 100),
            _sample(11.0, 100, 100),
            _sample(12.0, 0, 100),
        ]
    )

    summary = sampler.window_utilization(10.0, 11.0, (0,))

    assert summary == {
        "interval_s": 0.5,
        "gpus": {"0": {"samples": 3, "mean_pct": 80.0, "p50_pct": 80.0}},
    }


def test_a_point_records_the_counters_its_window_advanced(
    tmp_path: Path, media: bytes
) -> None:
    served = [0]

    class Handler(BaseHTTPRequestHandler):
        """Serves videos and counts their denoising time in `/metrics`."""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name.
            body = (
                "# TYPE uniserve:scheduler_domain_time_us counter\n"
                "uniserve:scheduler_domain_time_us_total"
                f'{{domain="denoise",phase="device"}} {125 * served[0]}\n'
                f"uniserve:num_requests_running {served[0]}\n"
                "# EOF\n"
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/openmetrics-text")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
            self.rfile.read(int(self.headers.get("content-length", "0")))
            served[0] += 1
            self.send_response(200)
            self.send_header("content-type", "video/mp4")
            self.send_header("content-length", str(len(media)))
            self.end_headers()
            self.wfile.write(media)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"id": f"row-{index}", "prompt": "scene", "seconds": 4})
            + "\n"
            for index in range(3)
        ),
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
        load=LoadConfig(num_prompts=3, max_concurrency=1, warmup_requests=1),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result"
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        result = asyncio.run(run_point([base_url], point, output))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    summary = result.summary
    assert summary["validation"]["valid"] is True
    assert summary["metrics"]["completed_videos"] == 3
    # The warmup request precedes the window, so the counter advanced by
    # exactly the three measured requests; the gauge is not a counter.
    assert summary["server_metrics"][base_url] == {
        "available": True,
        "counter_deltas": {
            'uniserve:scheduler_domain_time_us_total{domain="denoise",'
            'phase="device"}': 375.0
        },
    }
    snapshots = json.loads((output / "server_metrics.json").read_text())
    assert base_url in snapshots["before"]
    assert base_url in snapshots["after"]
