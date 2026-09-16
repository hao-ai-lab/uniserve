from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from uniserve_eval.config import ROOT, load_config

pytestmark = pytest.mark.unit


def test_fast_h3_server_topology_accepts_environment_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("UNISERVE_H3_WORKER_RANKS", "2")
    monkeypatch.setenv("UNISERVE_H3_CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("UNISERVE_H3_MEM_FRACTION", "0.99")
    monkeypatch.setenv("UNISERVE_H3_QUANT_MODE", "performance")

    server = load_config().servers["minimax-h3"]
    rank_value = server.command.index("--worker-ranks") + 1
    fraction_value = server.command.index("--mem-fraction-static") + 1
    precision_value = server.command.index("--quantization-config") + 1

    assert server.command[rank_value] == "2"
    assert server.command[fraction_value] == "0.99"
    assert json.loads(server.command[precision_value]) == {
        "mode": "performance"
    }
    assert server.environment["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_fast_h3_server_defaults_to_balanced_precision(monkeypatch) -> None:
    monkeypatch.delenv("UNISERVE_H3_QUANT_MODE", raising=False)

    server = load_config().servers["minimax-h3"]
    precision_value = server.command.index("--quantization-config") + 1

    assert json.loads(server.command[precision_value]) == {"mode": "balanced"}


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


def test_runtime_artifact_launch_uses_its_python_package(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        """
[servers.local]
port = 8000
command = [
    "target/release/uniserve", "serve", "model", "--worker-python",
    ".venv/bin/python",
]

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
    assert plan["server_command"][worker_python] == str(
        (ROOT / ".venv/bin/python").absolute()
    )
    assert plan["server_working_directory"] == str(runtime_root)
