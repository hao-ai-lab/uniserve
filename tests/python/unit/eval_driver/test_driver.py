from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import uniserve_eval.backends
import uniserve_eval.verify
from uniserve_eval import cli
from uniserve_eval.profiles import DEFAULT_CONFIG, load_config, server_spec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


def _load_run_benchmarks():
    spec = importlib.util.spec_from_file_location("run_benchmarks", ROOT / "scripts" / "run_benchmarks.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_benchmarks"] = module
    spec.loader.exec_module(module)
    return module


def test_run_suite_can_manage_verify_workload_servers(tmp_path, monkeypatch):
    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "off-server": {
                        "model": "m",
                        "served_model_name": "m",
                        "host": "127.0.0.1",
                        "port": 18082,
                    },
                    "on-server": {
                        "model": "m",
                        "served_model_name": "m",
                        "host": "127.0.0.1",
                        "port": 18082,
                    },
                },
                "workloads": {
                    "off": {
                        "type": "verify",
                        "server": "off-server",
                        "manage_server": True,
                        "payload": {},
                    },
                    "on": {
                        "type": "verify",
                        "server": "on-server",
                        "manage_server": True,
                        "payload": {},
                    },
                },
                "suites": {"fusion": ["off", "on"]},
            }
        ),
        encoding="utf-8",
    )
    events: list[tuple[str, str]] = []

    def fake_clean(args):
        events.append(("clean", args.server))

    def fake_launch(args):
        events.append(("launch", args.server))

    def fake_verify(args):
        events.append(("verify", args.workload))

    monkeypatch.setattr(uniserve_eval.backends, "clean", fake_clean)
    monkeypatch.setattr(uniserve_eval.backends, "launch", fake_launch)
    monkeypatch.setattr(uniserve_eval.verify, "verify", fake_verify)

    cli.run_suite(
        argparse.Namespace(
            config=config_path,
            suite="fusion",
            server=None,
            manage_servers=False,
            launch_timeout_s=1.0,
            clean_grace_s=0.0,
        )
    )

    assert events == [
        ("clean", "off-server"),
        ("launch", "off-server"),
        ("verify", "off"),
        ("clean", "off-server"),
        ("clean", "on-server"),
        ("launch", "on-server"),
        ("verify", "on"),
        ("clean", "on-server"),
    ]


def test_sensenova_uniserve_profiles_use_tp2_graph_execution():
    config = load_config(DEFAULT_CONFIG)
    for server_name in ("benchmark/server/sensenova-uniserve", "gate/server/sensenova"):
        spec = server_spec(config, server_name)
        cmd = uniserve_eval.backends.build_serve_cmd(config, spec)
        assert spec["cuda_visible_devices"] == "0,1"
        assert spec["env"]["UNISERVE_DECODE_TOKEN_BURST"] == "8"
        assert cmd[cmd.index("--worker-ranks") + 1] == "2"
    # The denoise-step CUDA graph is on for both profiles (validated safe under
    # concurrency). The packed-mixed decoder graph was removed entirely (replay
    # against re-planned shared attention state segfaulted under concurrency),
    # so no profile may reference its env flag.
    benchmark = server_spec(config, "benchmark/server/sensenova-uniserve")
    gate = server_spec(config, "gate/server/sensenova")
    for spec in (benchmark, gate):
        assert spec["env"]["UNISERVE_DENOISE_STEP_GRAPH"] == "1"
        assert "UNISERVE_PACKED_MIXED_GRAPH" not in spec["env"]


def test_benchmark_runner_forwards_wire(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    cmd = run_benchmarks.harness_command(
        python=".venv/bin/python",
        base_url="http://127.0.0.1:18082",
        output_dir=tmp_path,
        defaults={},
        harness={
            "task": "interleave",
            "model": "SenseNova-U1",
            "dataset": "ueval",
            "num_prompts": 1,
            "wire": "openai_chat",
        },
        rate="1",
        datasets={},
    )
    assert cmd[cmd.index("--wire") + 1] == "openai_chat"
