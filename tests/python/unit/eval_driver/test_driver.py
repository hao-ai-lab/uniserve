from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import uniserve_eval.backends
import uniserve_eval.verify
from uniserve_eval import cli
from uniserve_eval.harness.artifacts import ArtifactWriter
from uniserve_eval.harness.provenance import execution_provenance, repository_state
from uniserve_eval.harness.report import (
    attach_execution_contract,
    benchmark_contract,
    benchmark_parity_contract,
    canonical_artifact_bundle_matches,
    record_collection_contract,
    write_summary_artifacts,
)
from uniserve_eval.harness.runner import BenchmarkRunner
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.profiles import DEFAULT_CONFIG, load_config, server_spec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


def test_benchmark_parity_excludes_backend_identity_but_pins_protocol_and_rows():
    rows = [{"id": "row-1", "prompt": "hello"}]
    candidate = BenchmarkSpec(
        task=TaskName.TEXT,
        model="candidate",
        runtime_profile_id="candidate-profile",
        plan_evidence_policy="runtime_inspection",
        num_prompts=1,
    )
    reference = BenchmarkSpec(
        task=TaskName.TEXT,
        model="reference",
        runtime_profile_id="reference-profile",
        plan_evidence_policy="reference_protocol",
        num_prompts=1,
    )

    candidate_parity = benchmark_parity_contract(benchmark_contract(candidate, rows))
    reference_parity = benchmark_parity_contract(benchmark_contract(reference, rows))
    assert candidate_parity == reference_parity

    different_wire = BenchmarkSpec(
        task=TaskName.I2T,
        model="reference",
        wire="openai_chat_json",
        num_prompts=1,
    )
    stream_wire = BenchmarkSpec(
        task=TaskName.I2T,
        model="candidate",
        wire="openai_chat",
        num_prompts=1,
    )
    assert benchmark_parity_contract(
        benchmark_contract(different_wire, rows)
    ) != benchmark_parity_contract(benchmark_contract(stream_wire, rows))
    assert benchmark_parity_contract(
        benchmark_contract(candidate, [{"id": "different"}])
    ) != candidate_parity


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
        assert cmd[cmd.index("--tp-size") + 1] == "2"
    # Both profiles use the validated denoise-step graph and the canonical
    # interleaved decoder configuration.
    benchmark = server_spec(config, "benchmark/server/sensenova-uniserve")
    gate = server_spec(config, "gate/server/sensenova")
    for spec in (benchmark, gate):
        assert spec["env"]["UNISERVE_DENOISE_STEP_GRAPH"] == "1"
        assert "UNISERVE_PACKED_MIXED_GRAPH" not in spec["env"]


def test_sensenova_default_gate_declares_complete_image_lifecycle():
    config = load_config(DEFAULT_CONFIG)
    workload = config["workloads"]["gate/sensenova/default-travel"]
    image_config = workload["payload"]["image_config"]

    assert workload["expect_images"] == image_config["num_images"]
    assert workload["expect_image_steps"] == image_config["steps"] * image_config["num_images"]
    assert uniserve_eval.verify.usage_image_steps({"usage": {"image_steps": 200}}) == 200


def test_benchmark_runner_forwards_wire(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    cmd = run_benchmarks.harness_command(
        python=".venv/bin/python",
        base_url="http://127.0.0.1:18082",
        output_dir=tmp_path,
        defaults={},
        harness={
            "task": "default",
            "model": "SenseNova-U1",
            "dataset": "ueval",
            "num_prompts": 1,
            "wire": "openai_chat",
        },
        rate="1",
        datasets={},
    )
    assert cmd[cmd.index("--wire") + 1] == "openai_chat"


def test_execution_provenance_covers_environment_executable_and_source(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    source = repository / "program.py"
    source.write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "program.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
    )
    executable = tmp_path / "runner"
    executable.write_bytes(b"version-one")
    executable.chmod(0o755)
    environment = {"PATH": os.environ.get("PATH", ""), "INHERITED_SETTING": "one"}

    baseline = execution_provenance(
        [str(executable), "serve"],
        environment,
        cwd=repository,
        workspace_root=repository,
    )
    inherited_changed = execution_provenance(
        [str(executable), "serve"],
        {**environment, "INHERITED_SETTING": "two"},
        cwd=repository,
        workspace_root=repository,
    )
    assert inherited_changed["fingerprint"] != baseline["fingerprint"]

    executable.write_bytes(b"version-two")
    executable_changed = execution_provenance(
        [str(executable), "serve"],
        environment,
        cwd=repository,
        workspace_root=repository,
    )
    assert executable_changed["fingerprint"] != baseline["fingerprint"]

    source.write_text("print('two')\n", encoding="utf-8")
    tracked_changed = repository_state(repository)
    untracked = repository / "new_module.py"
    untracked.write_text("VALUE = 1\n", encoding="utf-8")
    untracked_changed = repository_state(repository)
    assert tracked_changed["dirty"] is True
    assert untracked_changed["fingerprint"] != tracked_changed["fingerprint"]

    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"weights-one")
    model_baseline = execution_provenance(
        [str(executable), "serve", "--model", str(model)],
        environment,
        cwd=repository,
        workspace_root=repository,
    )
    assert model_baseline["command_inputs"][0]["kind"] == "directory"
    weights.write_bytes(b"weights-two")
    model_changed = execution_provenance(
        [str(executable), "serve", "--model", str(model)],
        environment,
        cwd=repository,
        workspace_root=repository,
    )
    assert model_changed["fingerprint"] != model_baseline["fingerprint"]


def test_python_execution_provenance_covers_runtime_and_target_module(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    module = repository / "benchmark_module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "benchmark_module.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=repository,
        check=True,
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository),
    }

    provenance = execution_provenance(
        [sys.executable, "-m", "benchmark_module"],
        environment,
        cwd=repository,
        workspace_root=repository,
    )

    assert provenance["schema_version"] == 2
    assert provenance["python_runtime"]["distribution_count"] > 0
    assert provenance["python_runtime"]["module"] == "benchmark_module"
    assert provenance["python_module_sources"][0]["resolved_name"] == "benchmark_module.py"


def test_execution_contract_attachment_writes_synchronized_artifacts(tmp_path):
    summary = {
        "artifact": {
            "schema_version": 2,
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "checks": {"harness_contract": True},
        }
    }
    contract = {"schema_version": 2, "fingerprint": "a" * 64}

    attach_execution_contract(summary, "matrix_contract", contract)
    write_summary_artifacts(tmp_path, summary)

    persisted_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    persisted_manifest = json.loads(
        (tmp_path / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_summary == summary
    assert persisted_manifest == summary["artifact"]
    assert persisted_summary["artifact"]["checks"]["matrix_contract"] is True

    attach_execution_contract(summary, "quality_contract", {"schema_version": 1}, valid=False)
    assert summary["artifact"]["valid"] is False
    assert summary["artifact"]["valid_marker"] is None


def test_canonical_artifact_bundle_binds_every_durable_record(tmp_path):
    expected_contract = {"fingerprint": "harness"}
    request_records = [{"request_id": "request-1", "success": True}]
    gpu_samples = [{"time": 1.0, "gpus": []}]
    summary = {
        "request_count": 1,
        "spec": {"task": "text"},
        "base_url": "http://127.0.0.1:8000",
        "artifact": {
            "schema_version": 2,
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "checks": {"contract": True},
            "contract": expected_contract,
        },
    }
    attach_execution_contract(
        summary,
        "request_records",
        record_collection_contract(request_records),
    )
    attach_execution_contract(
        summary,
        "gpu_samples",
        record_collection_contract(gpu_samples),
    )
    writer = ArtifactWriter(tmp_path)
    writer.write_jsonl("requests.jsonl", request_records)
    writer.write_jsonl("gpu_samples.jsonl", gpu_samples)
    writer.write_json(
        "run.json",
        {
            "harness_status": "completed",
            "artifact_valid": True,
            "items": 1,
            "spec": summary["spec"],
            "base_url": summary["base_url"],
        },
    )
    write_summary_artifacts(tmp_path, summary)

    assert canonical_artifact_bundle_matches(tmp_path, summary, expected_contract)

    writer.write_jsonl("requests.jsonl", [*request_records, {"request_id": "injected"}])
    assert not canonical_artifact_bundle_matches(tmp_path, summary, expected_contract)


def test_benchmark_runner_invalidates_commit_markers_before_loading_data(
    tmp_path,
    monkeypatch,
):
    for marker in ("summary.json", "artifact_manifest.json", "summary.md"):
        (tmp_path / marker).write_text("stale", encoding="utf-8")

    def fail_to_load(_spec):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr("uniserve_eval.harness.runner.load_benchmark_inputs", fail_to_load)
    runner = BenchmarkRunner(
        "http://127.0.0.1:8000",
        BenchmarkSpec(task=TaskName.TEXT, model="test", num_prompts=1),
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="dataset unavailable"):
        asyncio.run(runner.run())

    for marker in ("summary.json", "artifact_manifest.json", "summary.md"):
        assert not (tmp_path / marker).exists()
