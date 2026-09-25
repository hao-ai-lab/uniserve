from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from uniserve_eval.config import load_config

pytestmark = pytest.mark.unit


def test_fast_h3_serves_the_deployment_configuration_it_is_given(
    monkeypatch,
) -> None:
    """A width is a deployment, so serving another one names another file."""
    monkeypatch.setenv(
        "UNISERVE_H3_DEPLOYMENT", "configs/minimax-h3-two-devices.json"
    )
    monkeypatch.setenv("UNISERVE_H3_CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("UNISERVE_H3_MEM_FRACTION", "0.99")
    monkeypatch.setenv("UNISERVE_H3_QUANT_MODE", "performance")

    server = load_config().servers["minimax-h3"]
    deployment = server.command.index("--workers") + 1
    fraction_value = server.command.index("--mem-fraction-static") + 1
    precision_value = server.command.index("--quantization-config") + 1

    assert server.command[deployment] == "configs/minimax-h3-two-devices.json"
    assert server.command[fraction_value] == "0.99"
    assert json.loads(server.command[precision_value]) == {
        "mode": "performance"
    }
    assert server.environment["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_fast_h3_defaults_to_the_four_device_deployment(monkeypatch) -> None:
    monkeypatch.delenv("UNISERVE_H3_DEPLOYMENT", raising=False)

    config = load_config()
    server = config.servers["minimax-h3"]
    deployment = server.command.index("--workers") + 1

    named = config.root / server.command[deployment]
    assert named.is_file(), f"{named} is the default deployment and must exist"
    # The numerical components run on the model worker's devices; the host
    # components encode and mux on a host rank.
    workers = {
        worker["id"]: sorted(worker["components"])
        for worker in json.loads(named.read_text())
    }
    assert workers == {
        "model": ["audio_decoder", "denoiser", "text_encoder", "video_decoder"],
        "host": ["muxer", "video_encoder"],
    }
    host = next(
        worker
        for worker in json.loads(named.read_text())
        if worker["id"] == "host"
    )
    assert all(rank["device"] == "cpu" for rank in host["ranks"])


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


def test_toml_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    config = tmp_path / "profiles.toml"
    config.write_text(
        """
artifct_root = "results"

[servers.local]
port = 8000
command = ["server"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifct_root"):
        load_config(config)


def test_a_profile_resolves_its_paths_against_the_root_it_states(
    tmp_path: Path,
) -> None:
    """A profile means the same thing wherever the driver is installed.

    Its executable, interpreter and working directory follow the tree the file
    names, so nothing depends on where this package's own source sits.
    """
    tree = tmp_path / "tree"
    (tree / "uniserve_eval").mkdir(parents=True)
    config_path = tree / "uniserve_eval" / "profiles.toml"
    config_path.write_text(
        """
root = ".."

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

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uniserve_eval.cli",
            "--config",
            str(config_path),
            "plan",
            "point",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    plan = json.loads(result.stdout)[0]

    assert plan["server_command"][0] == str(tree / "target/release/uniserve")
    worker_python = plan["server_command"].index("--worker-python") + 1
    assert plan["server_command"][worker_python] == str(
        tree / ".venv/bin/python"
    )
    assert plan["server_working_directory"] == str(tree)
