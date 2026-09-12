from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from uniserve_eval.config import ROOT, load_config

pytestmark = pytest.mark.unit


def test_runtime_suites_cover_declared_workloads() -> None:
    config = load_config()
    points = config.selected_points("decode-runtime")
    assert tuple(point.name for point in points) == (
        "qwen-uniserve-sharegpt-r16",
        "sensenova-uniserve-i2t-c32",
        "sensenova-uniserve-t2i-c32",
        "sensenova-uniserve-interleave-c4",
        "bagel-uniserve-sharegpt-r16",
        "bagel-uniserve-i2t-c32",
        "bagel-uniserve-t2i-c32",
    )
    assert tuple((metric.name, metric.direction) for metric in points[3].metrics) == (
        ("mean_ttft_ms", "lower"),
        ("mean_tpot_ms", "lower"),
        ("image_latency_ms.mean", "lower"),
    )

    video_points = config.selected_points("fast_h3")
    assert tuple(point.name for point in video_points) == (
        "minimax-h3-5s-1k",
        "minimax-h3-5s-10k",
        "minimax-h3-15s-1k",
        "minimax-h3-15s-10k",
    )
    formal_points = points + video_points
    assert sum(point.load.num_prompts for point in formal_points) == 572
    assert all(point.load.warmup_requests == 1 for point in formal_points)


def test_serving_runtime_suite_exposes_stream_and_image_latency_metrics() -> None:
    config = load_config()
    points = config.selected_points("serving-runtime")
    assert tuple(point.name for point in points) == (
        "qwen-uniserve-sharegpt-r16",
        "sensenova-uniserve-i2t-stream",
        "sensenova-uniserve-t2i-image",
        "sensenova-uniserve-interleave-c4",
    )
    assert points[1].sampling.stream is True
    assert points[2].sampling.stream is False
    assert tuple(metric.name for metric in points[2].metrics) == (
        "images_per_second",
        "image_latency_ms.mean",
    )


def test_fast_h3_server_topology_accepts_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("UNISERVE_H3_WORKER_RANKS", "2")
    monkeypatch.setenv("UNISERVE_H3_CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("UNISERVE_H3_MEM_FRACTION", "0.99")

    server = load_config().servers["minimax-h3"]
    rank_value = server.command.index("--worker-ranks") + 1
    fraction_value = server.command.index("--mem-fraction-static") + 1

    assert server.command[rank_value] == "2"
    assert server.command[fraction_value] == "0.99"
    assert server.environment["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_toml_rejects_an_unknown_benchmark_field(tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
[servers.local]
port = 8000
command = ["server"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "model"
unexpected = true

[benchmarks.point.load]
num_prompts = 1

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_config(config)


def test_runtime_artifact_launch_uses_its_python_package(tmp_path: Path) -> None:
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        """
[servers.local]
port = 8000
command = ["target/release/uniserve", "serve", "model", "--worker-python", ".venv/bin/python"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "sharegpt"
tokenizer = "model"

[benchmarks.point.load]
num_prompts = 1

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    runtime_root = tmp_path / "runtime"
    executable = runtime_root / "bin" / "uniserve"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (runtime_root / "uniserve_worker").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uniserve_eval.cli",
            "--config",
            str(config_path),
            "plan",
            "point",
            "--executable",
            str(executable),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    plan = json.loads(result.stdout)[0]

    assert plan["server_command"][0] == str(executable.absolute())
    worker_python = plan["server_command"].index("--worker-python") + 1
    assert plan["server_command"][worker_python] == str((ROOT / ".venv/bin/python").absolute())
    assert plan["server_working_directory"] == str(runtime_root)
