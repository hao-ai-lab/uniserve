from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import uniserve_eval.backends
import uniserve_eval.harness.runner as harness_runner
import uniserve_eval.verify
from uniserve_eval import cli
from uniserve_eval.harness.artifacts import ArtifactWriter
from uniserve_eval.harness.datasets import BenchmarkInputs
from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.provenance import (
    execution_provenance,
    input_path_contract,
    performance_environment,
    repository_state,
)
from uniserve_eval.harness.report import (
    artifact_bundle_matches,
    attach_execution_contract,
    benchmark_contract,
    benchmark_parity_contract,
    canonical_artifact_bundle_matches,
    canonical_digest,
    record_collection_contract,
    write_summary_artifacts,
)
from uniserve_eval.harness.runner import BenchmarkRunner
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.profiles import (
    DEFAULT_CONFIG,
    benchmark_matrix_definition_contract,
    benchmark_matrix_definition_matches,
    command_template_matches,
    load_config,
    server_command_template,
    server_execution_matches_profile,
    server_profile_definition_contract,
    server_profile_definition_matches,
    server_spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


def test_benchmark_parity_excludes_backend_identity_but_pins_protocol_and_rows():
    rows = [{"id": "row-1", "prompt": "hello"}]
    candidate = BenchmarkSpec(
        task=TaskName.TEXT,
        model="candidate",
        runtime_profile_id="candidate-profile",
        num_prompts=1,
    )
    reference = BenchmarkSpec(
        task=TaskName.TEXT,
        model="reference",
        runtime_profile_id="reference-profile",
        request_schema="sglang",
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
    assert (
        benchmark_parity_contract(benchmark_contract(candidate, [{"id": "different"}]))
        != candidate_parity
    )


def test_benchmark_parity_normalizes_backend_schedule_points_to_denoise_updates():
    rows = [{"id": "row-1", "prompt": "hello"}]
    candidate = BenchmarkSpec(
        task=TaskName.T2I,
        model="candidate",
        runtime_profile_id="candidate-profile",
        num_prompts=1,
        steps=50,
        denoise_updates=50,
    )
    reference = BenchmarkSpec(
        task=TaskName.T2I,
        model="reference",
        runtime_profile_id="reference-profile",
        request_schema="vllm_omni",
        num_prompts=1,
        steps=51,
        denoise_updates=50,
    )

    candidate_contract = benchmark_contract(candidate, rows)
    reference_contract = benchmark_contract(reference, rows)
    assert candidate_contract != reference_contract
    assert benchmark_parity_contract(candidate_contract) == benchmark_parity_contract(
        reference_contract
    )

    different_work = dataclasses.replace(reference, denoise_updates=49)
    assert benchmark_parity_contract(
        benchmark_contract(different_work, rows)
    ) != benchmark_parity_contract(candidate_contract)


def test_denoise_updates_requires_positive_work_and_backend_steps():
    with pytest.raises(ValueError, match="denoise_updates must be positive"):
        BenchmarkSpec(task=TaskName.T2I, model="candidate", steps=1, denoise_updates=0)
    with pytest.raises(ValueError, match="requires a backend steps value"):
        BenchmarkSpec(task=TaskName.T2I, model="candidate", denoise_updates=50)


def test_benchmark_parity_uses_selected_content_not_relocated_dataset_path():
    rows = [{"id": "row-1", "image": "data:image/jpeg;base64,AA=="}]
    first = BenchmarkSpec(
        task=TaskName.I2T,
        model="candidate",
        dataset="image-dir",
        dataset_path="/formal/block-01/datasets/beans",
        num_prompts=1,
    )
    second = dataclasses.replace(
        first,
        model="reference",
        dataset_path="/formal/block-02/datasets/beans",
    )

    assert benchmark_parity_contract(benchmark_contract(first, rows)) == benchmark_parity_contract(
        benchmark_contract(second, rows)
    )
    assert benchmark_contract(first, rows) != benchmark_contract(second, rows)


def _load_run_benchmarks():
    spec = importlib.util.spec_from_file_location(
        "run_benchmarks", ROOT / "scripts" / "run_benchmarks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_benchmarks"] = module
    spec.loader.exec_module(module)
    return module


def _write_harness_bundle(output_dir: Path, contract: dict) -> None:
    request_records = [{"request_id": "request-1", "success": True}]
    gpu_samples: list[dict] = []
    summary = {
        "request_count": 1,
        "spec": {"task": "text"},
        "base_url": "http://127.0.0.1:8000",
        "artifact": {
            "schema_version": 4,
            "valid": True,
            "valid_marker": "artifact-valid-v4",
            "checks": {"contract": True},
            "contract": contract,
            "request_records": record_collection_contract(request_records),
            "gpu_samples": record_collection_contract(gpu_samples),
        },
    }
    writer = ArtifactWriter(output_dir)
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
    (output_dir / "summary.md").write_text("# benchmark\n", encoding="utf-8")
    write_summary_artifacts(output_dir, summary)


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


def test_sensenova_benchmark_uses_equal_single_gpu_graph_execution():
    config = load_config(DEFAULT_CONFIG)
    benchmark = server_spec(config, "benchmark/server/sensenova-uniserve")
    benchmark_cmd = uniserve_eval.backends.build_serve_cmd(config, benchmark)
    assert benchmark["cuda_visible_devices"] == "0"
    assert "--tp-size" not in benchmark_cmd
    assert benchmark_cmd[benchmark_cmd.index("--prefill-cuda-graph") + 1] == "true"

    # The correctness gate retains its separately validated TP2 topology.
    gate = server_spec(config, "gate/server/sensenova")
    gate_cmd = uniserve_eval.backends.build_serve_cmd(config, gate)
    assert gate["cuda_visible_devices"] == "0,1"
    assert gate_cmd[gate_cmd.index("--tp-size") + 1] == "2"


def test_multimodal_benchmark_profiles_pin_quality_relevant_generation_modes() -> None:
    config = load_config(DEFAULT_CONFIG)
    benchmark = config["benchmarks"]["main"]
    points = benchmark["points"]
    for name in ("sensenova_mjhq_t2i_uniserve", "sensenova_mjhq_t2i_omni"):
        harness = points[name]["harness"]
        assert harness["image_think"] is False
        assert harness["image_t_eps"] == 0.02
    interleave = points["sensenova_ueval_interleave_uniserve"]["harness"]
    assert interleave["image_think"] is False
    assert interleave["image_t_eps"] == 0.02
    assert interleave["max_tokens"] == 8192
    assert interleave["disable_ignore_eos"] is True
    # The answer decides how many images it needs, so the point states none and
    # measures the image work the model actually performs.
    assert "max_images" not in interleave
    interleave_loads = benchmark["load_cases"][
        points["sensenova_ueval_interleave_uniserve"]["load_case_set"]
    ]
    assert [case["max_concurrency"] for case in interleave_loads] == [1, 2, 4, 8, 16]
    for name in ("bagel_mjhq_t2i_uniserve", "bagel_mjhq_t2i_omni"):
        harness = points[name]["harness"]
        assert harness["image_think"] is False
        assert harness["image_guidance_scale"] == 1.5

    deploy = (ROOT / "uniserve_eval/configs/vllm_omni/bagel.yaml").read_text(encoding="utf-8")
    assert deploy.count("seed: 42") == 2

    omni = server_spec(config, "benchmark/server/bagel-omni")
    mot_dir = ROOT / omni["env"]["VLLM_TUNED_CONFIG_FOLDER"]
    mot_config = json.loads(
        (mot_dir / "device_name=GB200,dtype=w16a16.json").read_text(encoding="utf-8")
    )
    assert set(mot_config) == {"3584_3584", "3584_4608", "3584_37888", "18944_3584"}
    assert all({"4098", "12294"} <= set(shape_config) for shape_config in mot_config.values())


def test_main_benchmark_declares_every_runtime_comparison_pair() -> None:
    config = load_config(DEFAULT_CONFIG)
    benchmark = config["benchmarks"]["main"]
    run_benchmarks = _load_run_benchmarks()

    assert run_benchmarks.comparison_profile_roles(benchmark) == {
        "qwen3_sharegpt": {
            "candidate": "benchmark/server/qwen-uniserve",
            "reference": "benchmark/server/qwen-sglang",
        },
        "sensenova_mjhq_t2i": {
            "candidate": "benchmark/server/sensenova-uniserve",
            "reference": "benchmark/server/sensenova-omni",
        },
        "sensenova_beans_i2t": {
            "candidate": "benchmark/server/sensenova-uniserve",
            "reference": "benchmark/server/sensenova-omni",
        },
        "bagel_mjhq_t2i": {
            "candidate": "benchmark/server/bagel-uniserve",
            "reference": "benchmark/server/bagel-omni",
        },
        "bagel_beans_i2t": {
            "candidate": "benchmark/server/bagel-uniserve",
            "reference": "benchmark/server/bagel-omni",
        },
    }


def test_explicit_reference_servers_receive_the_same_numa_binding() -> None:
    config = load_config(DEFAULT_CONFIG)
    for server_name in (
        "benchmark/server/qwen-sglang",
        "benchmark/server/sensenova-omni",
        "benchmark/server/bagel-omni",
    ):
        cmd = uniserve_eval.backends.build_serve_cmd(config, server_spec(config, server_name))
        assert cmd[:3] == ["numactl", "--cpunodebind=0", "--membind=0"]


def test_sensenova_default_gate_declares_complete_image_lifecycle():
    config = load_config(DEFAULT_CONFIG)
    workload = config["workloads"]["gate/sensenova/default-travel"]
    image_config = workload["payload"]["image_config"]

    assert workload["warn_image_count"] == image_config["num_images"]
    assert workload["expect_image_steps_per_image"] == image_config["steps"]
    assert image_config["steps"] == 50
    assert uniserve_eval.verify.usage_image_steps_per_image(
        {"usage": {"image_steps_per_image": [50, 50, 50, 50]}}
    ) == [50, 50, 50, 50]
    checks, warnings, failures = uniserve_eval.verify.verification_checks(
        workload,
        images=[
            {"size": [workload["expect_image_width"], workload["expect_image_height"]]}
            for _ in range(workload["warn_image_count"])
        ],
        image_steps_per_image=[image_config["steps"]] * workload["warn_image_count"],
        errors=[],
        finished_count=1,
        text="A practical travel guide with a scenic visual.",
        output_modalities=["text", "image"],
    )
    assert all(checks.values())
    assert warnings == []
    assert failures == []


@pytest.mark.parametrize(
    ("image_count", "output_modalities", "expected_failed_checks"),
    [
        (0, ["text"], {"decoded_image_count", "text_image_transitions"}),
        (3, ["text", "image"], set()),
    ],
)
def test_default_travel_gate_rejects_missing_images_but_only_warns_on_variable_count(
    image_count: int,
    output_modalities: list[str],
    expected_failed_checks: set[str],
) -> None:
    config = load_config(DEFAULT_CONFIG)
    workload = config["workloads"]["gate/sensenova/default-travel"]
    checks, warnings, failures = uniserve_eval.verify.verification_checks(
        workload,
        images=[
            {"size": [workload["expect_image_width"], workload["expect_image_height"]]}
            for _ in range(image_count)
        ],
        image_steps_per_image=[50] * image_count or None,
        errors=[],
        finished_count=1,
        text="A practical travel guide with a scenic visual.",
        output_modalities=output_modalities,
    )

    assert {name for name, valid in checks.items() if not valid} == expected_failed_checks
    assert warnings == [f"expected 4 images, got {image_count}"]
    if image_count == 0:
        assert "expected at least 1 decoded image(s), got 0" in failures
    else:
        assert failures == []


@pytest.mark.parametrize(
    ("text", "image_count", "output_modalities", "expected_failed_checks"),
    [
        (" \t\n", 1, ["image"], {"visible_text", "text_image_transitions"}),
        (
            "A practical travel guide.",
            0,
            ["text"],
            {"decoded_image_count", "text_image_transitions"},
        ),
        (
            "A practical travel guide.",
            1,
            ["text", "text"],
            {"text_image_transitions"},
        ),
    ],
)
def test_default_travel_gate_requires_visible_interleaved_output(
    text: str,
    image_count: int,
    output_modalities: list[str],
    expected_failed_checks: set[str],
) -> None:
    config = load_config(DEFAULT_CONFIG)
    workload = config["workloads"]["gate/sensenova/default-travel"]
    checks, _warnings, _failures = uniserve_eval.verify.verification_checks(
        workload,
        images=[
            {"size": [workload["expect_image_width"], workload["expect_image_height"]]}
            for _ in range(image_count)
        ],
        image_steps_per_image=[50] * image_count or None,
        errors=[],
        finished_count=1,
        text=text,
        output_modalities=output_modalities,
    )

    assert {name for name, valid in checks.items() if not valid} == expected_failed_checks


def test_verify_cli_accepts_an_immutable_output_directory() -> None:
    from uniserve_eval.cli import build_parser

    args = build_parser().parse_args(
        [
            "verify",
            "gate/sensenova/default-travel",
            "--output-dir",
            "artifacts/eval/sensenova/default-travel",
        ]
    )

    assert args.output_dir == Path("artifacts/eval/sensenova/default-travel")


def test_verify_reference_evidence_uses_exact_transport_and_rgb_bytes(tmp_path):
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (2, 1), (10, 20, 30)).save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    images: list[dict] = []

    uniserve_eval.verify.save_image(image_url, tmp_path, images)
    event = uniserve_eval.verify.event_manifest_entry(
        {
            "type": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {"content": "trail", "images": [{"image_url": {"url": image_url}}]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1},
        }
    )

    assert images[0]["color_representation"] == "RGB uint8"
    assert images[0]["png_sha256"] == hashlib.sha256(buffer.getvalue()).hexdigest()
    assert images[0]["rgb_sha256"] == hashlib.sha256(bytes((10, 20, 30, 10, 20, 30))).hexdigest()
    assert event == {
        "type": "chat.completion.chunk",
        "visible_text_bytes": 5,
        "image_count": 1,
        "modalities": ["text", "image"],
        "finish_reasons": ["stop"],
        "has_usage": True,
        "has_error": False,
    }


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
            "disable_ignore_eos": True,
            "wire": "openai_chat",
        },
        load_case={"id": "r1", "request_rate": 1},
        datasets={},
    )
    assert cmd[cmd.index("--wire") + 1] == "openai_chat"
    assert "--disable-ignore-eos" in cmd
    assert run_benchmarks.spec_from_harness_command(cmd).ignore_eos is False


def test_benchmark_runner_serializes_explicit_chat_template_semantics(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    cmd = run_benchmarks.harness_command(
        python="python",
        base_url="http://127.0.0.1:1",
        output_dir=tmp_path,
        defaults={},
        harness={
            "task": "text",
            "model": "Qwen3-32B",
            "num_prompts": 1,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        load_case={"id": "r1", "request_rate": 1},
        datasets={},
    )

    assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"enable_thinking":true}'
    spec = run_benchmarks.spec_from_harness_command(cmd)
    assert spec.chat_template_kwargs == {"enable_thinking": True}


def test_benchmark_runner_applies_concurrency_load_case(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    cmd = run_benchmarks.harness_command(
        python="python",
        base_url="http://127.0.0.1:1",
        output_dir=tmp_path,
        defaults={},
        harness={"task": "t2i", "model": "image-model", "num_prompts": 32},
        load_case={"id": "c32", "request_rate": "inf", "max_concurrency": 32},
        datasets={},
    )

    assert cmd[cmd.index("--request-rate") + 1] == "inf"
    assert cmd[cmd.index("--max-concurrency") + 1] == "32"
    spec = run_benchmarks.spec_from_harness_command(cmd)
    assert spec.request_rate == float("inf")
    assert spec.max_concurrency == 32


def test_benchmark_runner_binds_load_case_harness_overrides(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    cmd = run_benchmarks.harness_command(
        python="python",
        base_url="http://127.0.0.1:1",
        output_dir=tmp_path,
        defaults={},
        harness={
            "task": "t2i",
            "model": "image-model",
            "num_prompts": 32,
        },
        load_case={
            "id": "c32",
            "request_rate": "inf",
            "max_concurrency": 32,
            "harness": {"acceptance_min_images_per_success": 0.5},
        },
        datasets={},
    )

    spec = run_benchmarks.spec_from_harness_command(cmd)
    assert spec.max_concurrency == 32
    assert spec.acceptance_min_images_per_success == 0.5


def test_formal_command_log_contains_only_the_current_invocation(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    log_path = tmp_path / "run.log"

    run_benchmarks.run_command(
        [sys.executable, "-c", "print('first invocation')"],
        log_path=log_path,
    )
    run_benchmarks.run_command(
        [sys.executable, "-c", "print('second invocation')"],
        log_path=log_path,
    )

    log = log_path.read_text(encoding="utf-8")
    assert "first invocation" not in log
    assert "second invocation" in log
    assert log.count("exit_code=0") == 1


def _write_formal_benchmark_bundle(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    point_dir = tmp_path / "point"
    server_dir = tmp_path / "server"
    harness_contract = {"fingerprint": "harness"}
    matrix_contract = {"schema_version": 2, "fingerprint": "a" * 64}
    _write_harness_bundle(point_dir, harness_contract)
    for name in ("command.txt", "preflight.txt", "postflight.txt", "run.log"):
        (point_dir / name).write_text(f"{name} contents\n", encoding="utf-8")
    server_dir.mkdir()
    for name in (
        "server_command.txt",
        "server.log",
        "server.exit",
        "pre_server_snapshot.txt",
        "post_server_snapshot.txt",
    ):
        (server_dir / name).write_text(f"{name} contents\n", encoding="utf-8")
    bench = run_benchmarks.BenchRunSpec(
        name="point-r1",
        group="group",
        command=("harness", "1"),
        output_dir=point_dir,
        server_output_dir=server_dir,
        process_environment={},
        harness_contract=harness_contract,
        parity_group=None,
        parity_contract=None,
        matrix_contract=matrix_contract,
    )

    run_benchmarks.canonicalize_benchmark_point(bench)
    assert run_benchmarks.summary_ok(bench)
    return run_benchmarks, bench


@pytest.mark.parametrize(
    ("directory_attribute", "name"),
    [
        ("output_dir", "command.txt"),
        ("output_dir", "preflight.txt"),
        ("output_dir", "postflight.txt"),
        ("output_dir", "run.json"),
        ("output_dir", "run.log"),
        ("server_output_dir", "server_command.txt"),
        ("server_output_dir", "server.log"),
        ("server_output_dir", "server.exit"),
        ("server_output_dir", "pre_server_snapshot.txt"),
        ("server_output_dir", "post_server_snapshot.txt"),
    ],
)
def test_summary_acceptance_binds_point_and_server_support_files(
    tmp_path,
    directory_attribute,
    name,
):
    run_benchmarks, bench = _write_formal_benchmark_bundle(tmp_path)
    artifact_path = getattr(bench, directory_attribute) / name

    with artifact_path.open("a", encoding="utf-8") as handle:
        handle.write("mutated\n")

    assert not run_benchmarks.summary_ok(bench)


def test_summary_acceptance_requires_the_exact_matrix_contract(tmp_path):
    run_benchmarks, bench = _write_formal_benchmark_bundle(tmp_path)

    summary = json.loads((bench.output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["artifact"]["matrix_contract"] = {
        "schema_version": 2,
        "fingerprint": "b" * 64,
    }
    write_summary_artifacts(bench.output_dir, summary)

    assert not run_benchmarks.summary_ok(bench)


def test_benchmark_group_restarts_server_for_every_operating_point(tmp_path, monkeypatch):
    run_benchmarks = _load_run_benchmarks()
    server = run_benchmarks.ServerRunSpec(
        name="group",
        profile="benchmark/server/test",
        host="127.0.0.1",
        port=18000,
        command=("server",),
        env={},
        process_environment={},
        model_contract=None,
    )
    benches = [
        run_benchmarks.BenchRunSpec(
            name=f"point-r{rate}",
            group="group",
            command=("harness", str(rate)),
            output_dir=tmp_path / f"point-r{rate}",
            server_output_dir=tmp_path / "servers" / "group" / f"point-r{rate}",
            process_environment={},
            harness_contract={},
            parity_group=None,
            parity_contract=None,
            matrix_contract={},
        )
        for rate in (1, 2)
    ]
    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        run_benchmarks,
        "write_snapshot",
        lambda path, **_kwargs: events.append(("snapshot", path.name)),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "launch_server",
        lambda _server, directory, _timeout: events.append(("launch", directory.name)) or object(),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "run_command",
        lambda command, **kwargs: events.append(("run", command[-1])) or 0,
    )
    monkeypatch.setattr(
        run_benchmarks,
        "canonicalize_benchmark_point",
        lambda bench: events.append(("accept", bench.name)),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "summary_ok",
        lambda bench: events.append(("verify", bench.name)) or True,
    )
    monkeypatch.setattr(
        run_benchmarks,
        "stop_server",
        lambda _proc, directory, _grace: events.append(("stop", directory.name)),
    )

    run_benchmarks.run_group(
        "group",
        server,
        benches,
        resume=False,
        server_timeout_s=1.0,
        server_grace_s=0.0,
        require_clean_gpu=False,
    )

    assert events == [
        ("snapshot", "pre_server_snapshot.txt"),
        ("launch", "point-r1"),
        ("snapshot", "preflight.txt"),
        ("run", "1"),
        ("snapshot", "postflight.txt"),
        ("stop", "point-r1"),
        ("snapshot", "post_server_snapshot.txt"),
        ("accept", "point-r1"),
        ("verify", "point-r1"),
        ("snapshot", "pre_server_snapshot.txt"),
        ("launch", "point-r2"),
        ("snapshot", "preflight.txt"),
        ("run", "2"),
        ("snapshot", "postflight.txt"),
        ("stop", "point-r2"),
        ("snapshot", "post_server_snapshot.txt"),
        ("accept", "point-r2"),
        ("verify", "point-r2"),
    ]


def test_benchmark_group_selection_preserves_declared_pair_order() -> None:
    run_benchmarks = _load_run_benchmarks()
    groups = ["candidate", "reference", "diagnostic"]

    selected = run_benchmarks.parse_only("reference,candidate", groups)
    filtered = run_benchmarks.filter_benchmark_groups(
        {"groups": {name: {"server": name} for name in groups}}, selected
    )

    assert selected == ["reference", "candidate"]
    assert list(filtered["groups"]) == ["reference", "candidate"]


def test_formal_selection_requires_complete_comparison_pairs() -> None:
    run_benchmarks = _load_run_benchmarks()
    benchmark = {
        "groups": {
            "candidate": {"points": ["candidate_point"]},
            "reference": {"points": ["reference_point"]},
        },
        "points": {
            "candidate_point": {"parity_group": "pair"},
            "reference_point": {"parity_group": "pair"},
        },
    }

    with pytest.raises(SystemExit, match="incomplete comparison pairs"):
        run_benchmarks.require_complete_parity_selection(benchmark, ["candidate"])
    run_benchmarks.require_complete_parity_selection(benchmark, ["reference", "candidate"])


def test_benchmark_comparison_profiles_come_from_matrix_groups() -> None:
    run_benchmarks = _load_run_benchmarks()
    benchmark = {
        "groups": {
            "candidate": {
                "server": "benchmark/server/candidate",
                "points": ["candidate_point"],
            },
            "reference": {
                "server": "benchmark/server/reference",
                "points": ["reference_point"],
            },
        },
        "points": {
            "candidate_point": {
                "parity_group": "pair",
                "comparison_role": "candidate",
            },
            "reference_point": {
                "parity_group": "pair",
                "comparison_role": "reference",
            },
        },
    }

    assert run_benchmarks.comparison_profile_roles(benchmark) == {
        "pair": {
            "candidate": "benchmark/server/candidate",
            "reference": "benchmark/server/reference",
        }
    }


def test_benchmark_comparison_requires_distinct_server_profiles() -> None:
    run_benchmarks = _load_run_benchmarks()
    benchmark = {
        "groups": {
            "candidate": {
                "server": "benchmark/server/shared",
                "points": ["candidate_point"],
            },
            "reference": {
                "server": "benchmark/server/shared",
                "points": ["reference_point"],
            },
        },
        "points": {
            "candidate_point": {
                "parity_group": "pair",
                "comparison_role": "candidate",
            },
            "reference_point": {
                "parity_group": "pair",
                "comparison_role": "reference",
            },
        },
    }

    with pytest.raises(SystemExit, match="distinct server profiles"):
        run_benchmarks.comparison_profile_roles(benchmark)


def test_benchmark_comparison_profiles_are_validated_before_group_filtering(
    tmp_path: Path, monkeypatch
) -> None:
    run_benchmarks = _load_run_benchmarks()
    benchmark = {
        "artifact_root": str(tmp_path / "benchmark"),
        "datasets": {},
        "groups": {
            "candidate": {
                "server": "benchmark/server/candidate",
                "points": ["candidate_point"],
            },
            "reference": {
                "server": "benchmark/server/reference",
                "points": ["reference_point"],
            },
        },
        "points": {
            "candidate_point": {
                "parity_group": "pair",
                "comparison_role": "candidate",
            },
            "reference_point": {
                "parity_group": "pair",
                "comparison_role": "reference",
            },
        },
    }
    observed_groups: list[list[str]] = []

    monkeypatch.setattr(run_benchmarks, "load_config", lambda _path: {})
    monkeypatch.setattr(run_benchmarks, "benchmark_spec", lambda _config, _name: benchmark)
    monkeypatch.setattr(
        run_benchmarks,
        "comparison_profile_roles",
        lambda value: observed_groups.append(list(value["groups"])) or {},
    )
    monkeypatch.setattr(run_benchmarks, "active_benchmark_processes", lambda: "(none)")
    monkeypatch.setattr(run_benchmarks, "build_servers", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        run_benchmarks,
        "build_benches",
        lambda _config, value, *_args, **_kwargs: {name: [] for name in value["groups"]},
    )
    monkeypatch.setattr(run_benchmarks, "write_runbook", lambda *_args, **_kwargs: None)

    assert run_benchmarks._main(["--dry-run", "--only", "candidate"]) == 0
    assert observed_groups == [["candidate", "reference"]]


def test_benchmark_repeat_wraps_complete_matrix_runs(tmp_path: Path, monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    output_root = tmp_path / "benchmark"
    observed: list[tuple[str, int, bool, bool]] = []

    monkeypatch.setattr(run_benchmarks, "load_config", lambda _path: {"benchmarks": {}})
    monkeypatch.setattr(
        run_benchmarks,
        "benchmark_spec",
        lambda _config, _name: {"artifact_root": str(output_root)},
    )

    def run_once(args: argparse.Namespace) -> int:
        child = Path(args.output_root)
        observed.append((child.name, args.repeat, args.text_canary, args.image_smoke))
        child.mkdir(parents=True)
        (child / "results.json").write_text(
            json.dumps({"comparisons": []}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(run_benchmarks, "_run_once", run_once)

    assert (
        run_benchmarks._main(
            [
                "--output-root",
                str(output_root),
                "--repeat",
                "3",
                "--text-canary",
                "--image-smoke",
            ]
        )
        == 0
    )

    assert observed == [
        ("run-001", 1, True, True),
        ("run-002", 1, True, True),
        ("run-003", 1, True, True),
    ]
    combined = json.loads((output_root / "results.json").read_text(encoding="utf-8"))
    assert combined == {"schema_version": 2, "run_count": 3, "comparisons": []}


def test_skipped_benchmarks_remain_in_the_aggregate_selection(tmp_path: Path) -> None:
    run_benchmarks = _load_run_benchmarks()

    def bench(name: str, group: str) -> Any:
        return run_benchmarks.BenchRunSpec(
            name=name,
            group=group,
            command=("harness",),
            output_dir=tmp_path / name,
            server_output_dir=tmp_path / "servers" / name,
            process_environment={},
            harness_contract={},
            parity_group=None,
            parity_contract=None,
            matrix_contract={},
        )

    first = bench("first", "group")
    second = bench("second", "group")
    execution, report = run_benchmarks.resolve_benchmark_selection(
        {"group": [first, second]},
        ["group"],
        only_benches=set(),
        skipped_benches={"first"},
    )

    assert [item.name for item in execution["group"]] == ["second"]
    assert [item.name for item in report] == ["first", "second"]


def test_build_manifest_identity_covers_runtime_binaries(tmp_path: Path, monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    build_log = tmp_path / "build.log"
    binary = tmp_path / "uniserve"
    python = tmp_path / "python"
    built_extension = tmp_path / "lib_uniserve_ipc.so"
    installed_extension = tmp_path / "_uniserve_ipc.so"
    binary.write_bytes(b"host")
    python.write_bytes(b"python")
    built_extension.write_bytes(b"worker")
    installed_extension.write_bytes(b"worker")
    build_log.write_text("started at 2026-01-01T00:00:00Z\n", encoding="utf-8")
    config = {"server_bin": str(binary)}
    monkeypatch.setattr(
        run_benchmarks,
        "worker_extension_paths",
        lambda _config: (python, built_extension, installed_extension),
    )

    first = run_benchmarks.build_manifest_contract(config, build_log)
    build_log.write_text("started at 2027-01-01T00:00:00Z\n", encoding="utf-8")
    second = run_benchmarks.build_manifest_contract(config, build_log)

    assert first == second
    assert first["build_log_present"] is True
    assert first["binary"] == run_benchmarks._file_content_contract(binary)
    assert first["worker_python"] == run_benchmarks._file_content_contract(python)
    assert first["worker_extension"] == run_benchmarks._file_content_contract(installed_extension)


def test_runtime_build_installs_release_worker_extension(tmp_path: Path, monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    python = tmp_path / "python"
    built_extension = tmp_path / "lib_uniserve_ipc.so"
    installed_extension = tmp_path / "_uniserve_ipc.so"
    build_log = tmp_path / "build.log"
    python.write_bytes(b"python")
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        run_benchmarks,
        "worker_extension_paths",
        lambda _config: (python, built_extension, installed_extension),
    )

    def build(command, *, log_path, env, check=True):
        observed.update(command=command, log_path=log_path, env=env, check=check)
        built_extension.write_bytes(b"release-worker")
        return 0

    monkeypatch.setattr(run_benchmarks, "run_command", build)

    run_benchmarks.build_uniserve_runtime({}, build_log)

    assert observed["command"] == [
        "cargo",
        "build",
        "--release",
        "--package",
        "uniserve-cli",
        "--package",
        "uniserve-ipc-py",
    ]
    assert observed["log_path"] == build_log
    assert observed["env"]["PYO3_PYTHON"] == str(python)
    assert installed_extension.read_bytes() == b"release-worker"


def test_clean_gpu_check_queries_only_the_selected_physical_gpu(monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    selectors: list[str | None] = []

    def selected_gpu(selector: str | None = None) -> str:
        selectors.append(selector)
        return "1, 0, 0, 189471"

    monkeypatch.setattr(run_benchmarks, "nvidia_smi", selected_gpu)

    run_benchmarks.wait_for_clean_gpu("1", timeout_s=0.1)

    assert selectors == ["1"]


def test_gpu_query_rejects_multiple_physical_selectors() -> None:
    run_benchmarks = _load_run_benchmarks()

    with pytest.raises(ValueError, match="one physical GPU selector"):
        run_benchmarks.nvidia_smi("0,1")


def test_server_profile_definition_contract_tracks_inheritance_and_launcher_inputs() -> None:
    config = {
        "python": ".venv/bin/python",
        "server_bin": "target/release/uniserve",
        "servers": {
            "benchmark/server/base": {"serve_args": ["--max-running-requests", "8"]},
            "benchmark/server/child": {
                "extends": "benchmark/server/base",
                "port": 8000,
            },
        },
    }
    contract = server_profile_definition_contract(config, "benchmark/server/child")
    inherited_change = deepcopy(config)
    inherited_change["servers"]["benchmark/server/base"]["serve_args"][-1] = "16"
    launcher_change = deepcopy(config)
    launcher_change["server_bin"] = "target/debug/uniserve"

    assert server_profile_definition_matches(contract, config)
    assert not server_profile_definition_matches(contract, inherited_change)
    assert not server_profile_definition_matches(contract, launcher_change)


def test_server_command_template_binds_repeated_environment_values() -> None:
    template = ["${PYTHON}", "--model", "${MODEL}", "--alias", "${MODEL}"]

    assert command_template_matches(
        template,
        ["/venv/python", "--model", "/models/qwen", "--alias", "/models/qwen"],
    )
    assert not command_template_matches(
        template,
        ["/venv/python", "--model", "/models/qwen", "--alias", "/models/other"],
    )


def test_server_execution_must_derive_from_profile_and_link_model_input(tmp_path) -> None:
    executable = tmp_path / "server"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "weights.bin").write_bytes(b"weights")
    config = {
        "servers": {
            "benchmark/server/test": {
                "numa_node": None,
                "model": str(model),
                "command": [str(executable), "--model-path", str(model)],
            }
        }
    }
    environment = {"PATH": os.environ.get("PATH", "")}
    execution = execution_provenance(
        server_command_template(config, "benchmark/server/test"),
        environment,
        cwd=ROOT,
        workspace_root=ROOT,
    )
    model_contract = input_path_contract(model, cwd=ROOT)

    assert server_execution_matches_profile(
        config,
        "benchmark/server/test",
        execution,
        model_contract=model_contract,
        model_revision_contract=None,
    )
    content_pinned_config = deepcopy(config)
    content_pinned_config["servers"]["benchmark/server/test"]["required_model_content"] = {
        key: model_contract[key]
        for key in ("kind", "file_count", "total_size_bytes", "tree_sha256")
    }
    assert server_execution_matches_profile(
        content_pinned_config,
        "benchmark/server/test",
        execution,
        model_contract=model_contract,
        model_revision_contract=None,
    )
    content_drift_config = deepcopy(content_pinned_config)
    content_drift_config["servers"]["benchmark/server/test"]["required_model_content"][
        "tree_sha256"
    ] = "0" * 64
    assert not server_execution_matches_profile(
        content_drift_config,
        "benchmark/server/test",
        execution,
        model_contract=model_contract,
        model_revision_contract=None,
    )
    performance_config = deepcopy(config)
    performance_config["servers"]["benchmark/server/test"]["env"] = {"OMP_NUM_THREADS": "8"}
    assert not server_execution_matches_profile(
        performance_config,
        "benchmark/server/test",
        execution,
        model_contract=model_contract,
        model_revision_contract=None,
    )
    declared_execution = execution_provenance(
        server_command_template(performance_config, "benchmark/server/test"),
        {**environment, "OMP_NUM_THREADS": "8"},
        cwd=ROOT,
        workspace_root=ROOT,
    )
    assert server_execution_matches_profile(
        performance_config,
        "benchmark/server/test",
        declared_execution,
        model_contract=model_contract,
        model_revision_contract=None,
    )

    other_model = tmp_path / "other-model"
    other_model.mkdir()
    (other_model / "config.json").write_text("{}\n", encoding="utf-8")
    (other_model / "weights.bin").write_bytes(b"other")
    assert not server_execution_matches_profile(
        config,
        "benchmark/server/test",
        execution,
        model_contract=input_path_contract(other_model, cwd=ROOT),
        model_revision_contract=None,
    )

    detached = execution_provenance(
        [str(executable), "--different-option", str(model)],
        environment,
        cwd=ROOT,
        workspace_root=ROOT,
    )
    assert not server_execution_matches_profile(
        config,
        "benchmark/server/test",
        detached,
        model_contract=model_contract,
        model_revision_contract=None,
    )


def _fingerprinted(payload: dict) -> dict:
    return {**payload, "fingerprint": canonical_digest(payload)}


def test_matrix_definition_binds_active_point_semantics_load_case_dataset_and_hardware() -> None:
    config = {
        "servers": {
            "benchmark/server/candidate": {
                "numa_node": None,
                "model": "/models/test",
                "command": ["server", "--model-path", "/models/test"],
            }
        },
        "benchmarks": {
            "main": {
                "hardware_requirements": {
                    "gpu_count_per_process": 1,
                    "gpu_model": "NVIDIA GB200",
                },
                "defaults": {"warmup_requests": 1, "seed": 42},
                "load_cases": {"text_arrival": [{"id": "r1", "request_rate": 1}]},
                "datasets": {},
                "groups": {
                    "candidate": {
                        "server": "benchmark/server/candidate",
                        "points": ["text_candidate"],
                    }
                },
                "points": {
                    "text_candidate": {
                        "load_case_set": "text_arrival",
                        "parity_group": "text_pair",
                        "comparison_role": "candidate",
                        "name": "text-candidate-{load_id}",
                        "output": "text/candidate-{load_id}",
                        "harness": {
                            "task": "text",
                            "model": "test",
                            "dataset": "sharegpt",
                            "dataset_revision": "a" * 40,
                            "num_prompts": 1,
                        },
                    }
                },
            }
        },
    }
    rows = [{"id": "row-1", "prompt": "hello"}]
    harness = benchmark_parity_contract(
        benchmark_contract(
            BenchmarkSpec(
                task=TaskName.TEXT,
                model="test",
                dataset="sharegpt",
                dataset_revision="a" * 40,
                num_prompts=1,
                request_rate=1.0,
            ),
            rows,
        )
    )
    parity_payload = {
        "schema_version": 2,
        "harness": harness,
        "model": {"kind": "model_directory"},
    }
    selected = _fingerprinted(
        {
            "schema_version": 1,
            "selector": "0",
            "gpu": {"name": "NVIDIA GB200"},
        }
    )
    hardware = _fingerprinted({"schema_version": 2, "host": {}, "selected_accelerator": selected})
    matrix = {
        "benchmark": "text-candidate-r1",
        "benchmark_profile": "main",
        "benchmark_definition": benchmark_matrix_definition_contract(
            config,
            "main",
            group_name="candidate",
            point_name="text_candidate",
            load_case_set="text_arrival",
            load_case_id="r1",
        ),
        "server_profile": "benchmark/server/candidate",
        "comparison_role": "candidate",
        "parity_group": "text_pair",
        "parity_contract": _fingerprinted(parity_payload),
        "server_execution": {
            "command": ["server", "--model-path", "/models/test"],
            "environment": _fingerprinted(
                {"schema_version": 1, "variable_count": 1, "variables_sha256": "1" * 64}
            ),
            "performance_environment": {"CUDA_VISIBLE_DEVICES": "0"},
        },
        "harness_execution": {
            "environment": _fingerprinted(
                {"schema_version": 1, "variable_count": 1, "variables_sha256": "1" * 64}
            ),
            "performance_environment": {"CUDA_VISIBLE_DEVICES": "0"},
        },
        "hardware": hardware,
    }
    assert benchmark_matrix_definition_matches(matrix, config)

    detached = deepcopy(matrix)
    detached_harness = dict(detached["parity_contract"]["harness"])
    detached_spec = dict(detached_harness["spec"])
    detached_spec["request_rate"] = 2.0
    detached_harness = _fingerprinted(
        {
            "schema_version": detached_harness["schema_version"],
            "spec": detached_spec,
            "selected_rows": detached_harness["selected_rows"],
        }
    )
    detached["parity_contract"] = _fingerprinted({**parity_payload, "harness": detached_harness})
    assert not benchmark_matrix_definition_matches(detached, config)

    wrong_hardware = deepcopy(matrix)
    wrong_selected = _fingerprinted(
        {"schema_version": 1, "selector": "0", "gpu": {"name": "Other GPU"}}
    )
    wrong_hardware["hardware"] = _fingerprinted(
        {"schema_version": 2, "host": {}, "selected_accelerator": wrong_selected}
    )
    assert not benchmark_matrix_definition_matches(wrong_hardware, config)

    wrong_process_gpu = deepcopy(matrix)
    wrong_process_gpu["harness_execution"]["performance_environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    assert not benchmark_matrix_definition_matches(wrong_process_gpu, config)

    changed_config = deepcopy(config)
    changed_config["benchmarks"]["main"]["points"]["text_candidate"]["harness"]["num_prompts"] = 2
    assert not benchmark_matrix_definition_matches(matrix, changed_config)


def test_matrix_definition_distinguishes_load_cases_of_one_point() -> None:
    config = load_config(DEFAULT_CONFIG)

    contracts = [
        benchmark_matrix_definition_contract(
            config,
            "main",
            group_name="sensenova-uniserve",
            point_name="sensenova_mjhq_t2i_uniserve",
            load_case_set="image_concurrency",
            load_case_id=load_case,
        )
        for load_case in ("c1", "c32")
    ]

    assert len({contract["declared_semantics_sha256"] for contract in contracts}) == 2


def test_formal_execution_rejects_noncanonical_profile_config(tmp_path: Path) -> None:
    run_benchmarks = _load_run_benchmarks()

    with pytest.raises(SystemExit, match="canonical uniserve_eval/profiles.json"):
        run_benchmarks._main(["--formal", "--config", str(tmp_path / "profiles.json")])


def test_formal_execution_rejects_dirty_source_state(tmp_path) -> None:
    run_benchmarks = _load_run_benchmarks()
    bench = run_benchmarks.BenchRunSpec(
        name="point",
        group="group",
        command=("harness",),
        output_dir=tmp_path / "point",
        server_output_dir=tmp_path / "server",
        process_environment={},
        harness_contract={},
        parity_group=None,
        parity_contract=None,
        matrix_contract={
            "server_execution": {
                "source_revisions": [
                    {
                        "roles": ["workspace"],
                        "state": {"dirty": True, "fingerprint": "dirty"},
                    }
                ]
            },
            "harness_execution": {"source_revisions": []},
        },
    )

    with pytest.raises(SystemExit, match="clean, reconstructable source"):
        run_benchmarks.require_clean_source({"group": [bench]})


def test_formal_execution_accepts_content_bound_input_from_dirty_repository(tmp_path) -> None:
    run_benchmarks = _load_run_benchmarks()
    bench = run_benchmarks.BenchRunSpec(
        name="point",
        group="group",
        command=("harness",),
        output_dir=tmp_path / "point",
        server_output_dir=tmp_path / "server",
        process_environment={},
        harness_contract={},
        parity_group=None,
        parity_contract=None,
        matrix_contract={
            "server_execution": {"source_revisions": []},
            "harness_execution": {
                "source_revisions": [
                    {
                        "roles": ["command_input:2", "command_input:3"],
                        "state": {"dirty": True, "fingerprint": "dirty-input-parent"},
                    }
                ]
            },
        },
    )

    run_benchmarks.require_clean_source({"group": [bench]})


def test_formal_execution_validates_gpu_numa_binding(monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    monkeypatch.setattr(run_benchmarks, "gpu_numa_nodes", lambda: {"0": 0, "2": 1})
    server = run_benchmarks.ServerRunSpec(
        name="group",
        profile="server",
        host="127.0.0.1",
        port=1,
        command=("numactl", "--cpunodebind=0", "--membind=0", "server"),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        process_environment={},
        model_contract=None,
    )
    run_benchmarks.validate_numa_binding({"group": server})

    mismatched = dataclasses.replace(
        server,
        command=("numactl", "--cpunodebind=0", "--membind=0", "server"),
        env={"CUDA_VISIBLE_DEVICES": "2"},
    )
    with pytest.raises(SystemExit, match="not bound"):
        run_benchmarks.validate_numa_binding({"group": mismatched})


def test_formal_execution_requires_pinned_reference_revision(tmp_path) -> None:
    run_benchmarks = _load_run_benchmarks()
    expected = "a" * 40
    server = run_benchmarks.ServerRunSpec(
        name="reference",
        profile="server",
        host="127.0.0.1",
        port=1,
        command=("server",),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        process_environment={},
        model_contract=None,
        required_source_revision=expected,
        required_source_role="pythonpath:0",
    )
    bench = run_benchmarks.BenchRunSpec(
        name="point",
        group="reference",
        command=("harness",),
        output_dir=tmp_path / "point",
        server_output_dir=tmp_path / "server",
        process_environment={},
        harness_contract={},
        parity_group="pair",
        parity_contract={},
        matrix_contract={
            "server_execution": {
                "source_revisions": [{"roles": ["pythonpath:0"], "state": {"head": "b" * 40}}]
            }
        },
    )

    with pytest.raises(SystemExit, match="reference source revision mismatch"):
        run_benchmarks.require_pinned_server_revisions(
            {"reference": [bench]}, {"reference": server}
        )

    matching = dataclasses.replace(
        bench,
        matrix_contract={
            "server_execution": {
                "source_revisions": [{"roles": ["pythonpath:0"], "state": {"head": expected}}]
            }
        },
    )
    run_benchmarks.require_pinned_server_revisions({"reference": [matching]}, {"reference": server})


def test_formal_execution_requires_one_declared_gpu_model(monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    server = run_benchmarks.ServerRunSpec(
        name="group",
        profile="server",
        host="127.0.0.1",
        port=1,
        command=("server",),
        env={"CUDA_VISIBLE_DEVICES": "0"},
        process_environment={"CUDA_VISIBLE_DEVICES": "0"},
        model_contract=None,
    )

    def selected(visible: str) -> dict:
        if "," in visible:
            raise ValueError("formal execution requires exactly one visible accelerator")
        return {"gpu": {"name": "NVIDIA GB200"}}

    monkeypatch.setattr(run_benchmarks, "selected_accelerator_contract", selected)
    requirements = {
        "hardware_requirements": {
            "gpu_count_per_process": 1,
            "gpu_model": "NVIDIA GB200",
        }
    }
    bench = run_benchmarks.BenchRunSpec(
        name="point",
        group="group",
        command=("harness",),
        output_dir=Path("point"),
        server_output_dir=Path("server"),
        process_environment={"CUDA_VISIBLE_DEVICES": "0"},
        harness_contract=None,
        parity_group=None,
        parity_contract=None,
        matrix_contract=None,
    )
    run_benchmarks.validate_hardware_requirements(
        {"group": server}, {"group": [bench]}, requirements
    )

    multi = dataclasses.replace(
        server,
        env={"CUDA_VISIBLE_DEVICES": "0,1"},
        process_environment={"CUDA_VISIBLE_DEVICES": "0,1"},
    )
    with pytest.raises(SystemExit, match="exactly one visible accelerator"):
        run_benchmarks.validate_hardware_requirements(
            {"group": multi}, {"group": [bench]}, requirements
        )

    mismatched_harness = dataclasses.replace(
        bench,
        process_environment={"CUDA_VISIBLE_DEVICES": "2"},
    )
    with pytest.raises(SystemExit, match="paired server GPU"):
        run_benchmarks.validate_hardware_requirements(
            {"group": server}, {"group": [mismatched_harness]}, requirements
        )


def test_harness_environment_inherits_only_the_paired_server_gpu(monkeypatch) -> None:
    run_benchmarks = _load_run_benchmarks()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("BACKEND_ONLY_SETTING", "ambient")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/ambient/cache")
    baseline_environment = dict(os.environ)
    server = run_benchmarks.ServerRunSpec(
        name="group",
        profile="benchmark/server/test",
        host="127.0.0.1",
        port=1,
        command=("server",),
        env={"CUDA_VISIBLE_DEVICES": "2", "BACKEND_ONLY_SETTING": "server"},
        process_environment={
            **baseline_environment,
            "CUDA_VISIBLE_DEVICES": "2",
            "BACKEND_ONLY_SETTING": "server",
        },
        model_contract=None,
        baseline_environment=baseline_environment,
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/late/runtime/mutation")

    environment = run_benchmarks.paired_harness_environment(server)

    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["BACKEND_ONLY_SETTING"] == "ambient"
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == "/ambient/cache"


def test_qwen_profiles_require_the_documented_model_revision() -> None:
    config = load_config(DEFAULT_CONFIG)
    expected = {
        "repository": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    }

    assert (
        server_spec(config, "benchmark/server/qwen-uniserve")["required_model_revision"] == expected
    )
    assert (
        server_spec(config, "benchmark/server/qwen-sglang")["required_model_revision"] == expected
    )


def test_multimodal_profiles_require_the_documented_model_content() -> None:
    config = load_config(DEFAULT_CONFIG)
    expected = {
        "benchmark/server/sensenova-uniserve": {
            "kind": "model_directory",
            "file_count": 20,
            "total_size_bytes": 35121626550,
            "tree_sha256": "17e3f80729ff5ec69207494387cc8d6e259073c45840f82adfb7172c9be168c0",
        },
        "benchmark/server/sensenova-omni": {
            "kind": "model_directory",
            "file_count": 20,
            "total_size_bytes": 35121626550,
            "tree_sha256": "17e3f80729ff5ec69207494387cc8d6e259073c45840f82adfb7172c9be168c0",
        },
        "benchmark/server/bagel-uniserve": {
            "kind": "model_directory",
            "file_count": 13,
            "total_size_bytes": 29561610186,
            "tree_sha256": "f1d61076bb5ce7f70d4b93b003576fabbb3f52681364c2de3848e9d531639c4f",
        },
        "benchmark/server/bagel-omni": {
            "kind": "model_directory",
            "file_count": 13,
            "total_size_bytes": 29561610186,
            "tree_sha256": "f1d61076bb5ce7f70d4b93b003576fabbb3f52681364c2de3848e9d531639c4f",
        },
    }

    for profile, contract in expected.items():
        assert server_spec(config, profile)["required_model_content"] == contract


def test_model_revision_contract_rejects_an_unproven_tree(tmp_path) -> None:
    run_benchmarks = _load_run_benchmarks()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    requirement = {
        "repository": "Qwen/Qwen3-32B",
        "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    }

    with pytest.raises(SystemExit, match="cannot prove the required revision"):
        run_benchmarks.model_revision_contract(
            str(model),
            requirement,
            input_path_contract(model, cwd=ROOT),
        )


def test_model_revision_contract_accepts_the_exact_huggingface_snapshot(tmp_path) -> None:
    run_benchmarks = _load_run_benchmarks()
    revision = "9216db5781bf21249d130ec9da846c4624c16137"
    repository_root = tmp_path / "models--Qwen--Qwen3-32B"
    blobs = repository_root / "blobs"
    snapshot = repository_root / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "config-blob"
    blob.write_text("{}\n", encoding="utf-8")
    (snapshot / "config.json").symlink_to(blob)
    model_contract = input_path_contract(snapshot, cwd=ROOT)

    contract = run_benchmarks.model_revision_contract(
        str(snapshot),
        {"repository": "Qwen/Qwen3-32B", "revision": revision},
        model_contract,
    )

    assert contract["repository"] == "Qwen/Qwen3-32B"
    assert contract["revision"] == revision
    assert contract["proof"]["kind"] == "huggingface_cache_snapshot"
    assert contract["model_contract_sha256"] == canonical_digest(model_contract)


def test_benchmark_host_lock_is_exclusive() -> None:
    run_benchmarks = _load_run_benchmarks()
    lock = run_benchmarks.acquire_host_benchmark_lock()
    try:
        with pytest.raises(SystemExit, match="holds the host lock"):
            run_benchmarks.acquire_host_benchmark_lock()
    finally:
        run_benchmarks.release_host_benchmark_lock(lock)


def test_materialized_dataset_binds_revision_selection_and_encoded_files(
    tmp_path, monkeypatch
) -> None:
    from PIL import Image

    run_benchmarks = _load_run_benchmarks()
    calls: list[dict] = []

    def load_dataset(name, *, split, revision):
        calls.append({"name": name, "split": split, "revision": revision})
        return [{"image": Image.new("RGB", (8, 8), (index, 0, 0))} for index in range(3)]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    benchmark = {
        "datasets": {
            "images": {
                "type": "hf_image_sample",
                "name": "example/images",
                "revision": "a" * 40,
                "split": "train",
                "count": 2,
                "seed": 42,
                "output_dir": "datasets/images",
                "prefix": "image",
            }
        }
    }

    resolved = run_benchmarks.materialize_datasets(tmp_path, benchmark)
    manifest_path = resolved["images"] / "SOURCE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["revision"] == "a" * 40
    assert len(manifest["files"]) == 2
    assert calls == [{"name": "example/images", "split": "train", "revision": "a" * 40}]

    run_benchmarks.materialize_datasets(tmp_path, benchmark)
    assert len(calls) == 1
    (resolved["images"] / manifest["files"][0]["name"]).write_bytes(b"corrupt")
    run_benchmarks.materialize_datasets(tmp_path, benchmark)
    assert len(calls) == 2


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

    wrapped = execution_provenance(
        ["/usr/bin/env", sys.executable, "-m", "benchmark_module"],
        environment,
        cwd=repository,
        workspace_root=repository,
    )
    assert wrapped["python_runtime"]["module"] == "benchmark_module"
    assert wrapped["python_module_sources"][0]["resolved_name"] == "benchmark_module.py"


def test_console_script_provenance_resolves_target_package(tmp_path):
    repository = tmp_path / "source"
    package = repository / "benchmark_module"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    launcher = tmp_path / "benchmark_module"
    launcher.write_text(
        f"#!{sys.executable}\nfrom benchmark_module import VALUE\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
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
    environment = {**os.environ, "PYTHONPATH": str(repository)}

    provenance = execution_provenance(
        [str(launcher)],
        environment,
        cwd=repository,
        workspace_root=repository,
    )

    assert provenance["python_runtime"]["module"] == "benchmark_module"
    assert "benchmark_module" in {
        source["resolved_name"] for source in provenance["python_module_sources"]
    }


def test_performance_environment_does_not_persist_asset_paths() -> None:
    persisted = performance_environment(
        {
            "UNISERVE_QWEN3_MODEL": "/private/model/path",
            "UNISERVE_SHAREGPT_PATH": "/private/dataset/path",
            "OMP_NUM_THREADS": "8",
            "VLLM_OMNI_USE_QUACK_FP8": "0",
        }
    )

    assert persisted == {
        "OMP_NUM_THREADS": "8",
        "VLLM_OMNI_USE_QUACK_FP8": "0",
    }


def test_execution_contract_attachment_writes_synchronized_artifacts(tmp_path):
    summary = {
        "artifact": {
            "schema_version": 4,
            "valid": True,
            "valid_marker": "canonical-valid-v4",
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
    assert persisted_summary["artifact"]["valid_marker"] == "canonical-valid-v4"

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
            "schema_version": 4,
            "valid": True,
            "valid_marker": "canonical-valid-v4",
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


def test_canonical_artifact_bundle_binds_reported_metrics(tmp_path):
    expected_contract = {"fingerprint": "harness"}
    request_records = [{"request_id": "request-1", "success": True}]
    gpu_samples: list[dict] = []
    summary = {
        "request_count": 1,
        "spec": {"task": "text"},
        "base_url": "http://127.0.0.1:8000",
        "metrics": {"output_throughput": 10.0},
        "artifact": {
            "schema_version": 4,
            "valid": True,
            "valid_marker": "canonical-valid-v4",
            "checks": {"contract": True},
            "contract": expected_contract,
            "request_records": record_collection_contract(request_records),
            "gpu_samples": record_collection_contract(gpu_samples),
        },
    }
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

    mutated = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    mutated["metrics"]["output_throughput"] = 1000.0
    (tmp_path / "summary.json").write_text(json.dumps(mutated), encoding="utf-8")
    assert not canonical_artifact_bundle_matches(tmp_path, mutated, expected_contract)


def test_direct_harness_bundle_is_not_a_matrix_canonical_point(tmp_path):
    run_benchmarks = _load_run_benchmarks()
    expected_contract = {"fingerprint": "harness"}
    expected_matrix_contract = {"schema_version": 2, "fingerprint": "a" * 64}
    request_records = [{"request_id": "request-1", "success": True}]
    gpu_samples: list[dict] = []
    summary = {
        "request_count": 1,
        "spec": {"task": "text"},
        "base_url": "http://127.0.0.1:8000",
        "artifact": {
            "schema_version": 4,
            "valid": True,
            "valid_marker": "artifact-valid-v4",
            "checks": {"contract": True},
            "contract": expected_contract,
            "request_records": record_collection_contract(request_records),
            "gpu_samples": record_collection_contract(gpu_samples),
        },
    }
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

    assert artifact_bundle_matches(tmp_path, summary, expected_contract)
    assert not canonical_artifact_bundle_matches(tmp_path, summary, expected_contract)
    bench = run_benchmarks.BenchRunSpec(
        name="point-r1",
        group="group",
        command=("harness", "1"),
        output_dir=tmp_path,
        server_output_dir=tmp_path / "server",
        process_environment={},
        harness_contract=expected_contract,
        parity_group=None,
        parity_contract=None,
        matrix_contract=expected_matrix_contract,
    )
    assert not run_benchmarks.summary_ok(bench)

    attach_execution_contract(summary, "matrix_contract", expected_matrix_contract)
    write_summary_artifacts(tmp_path, summary)
    assert canonical_artifact_bundle_matches(tmp_path, summary, expected_contract)
    assert not run_benchmarks.summary_ok(bench)


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


def test_benchmark_runner_warmup_uses_the_measured_request_shape(tmp_path, monkeypatch):
    row = {
        "id": "row-1",
        "task": "text",
        "prompt": "hello",
        "prompt_len": 1,
        "output_len": 97,
    }
    submitted: list[dict] = []

    monkeypatch.setattr(
        harness_runner,
        "load_benchmark_inputs",
        lambda _spec: BenchmarkInputs(measured=[row], warmup=[], tokenizer=None),
    )

    async def fake_submit(self, _client, submitted_row):
        del self
        submitted.append(dict(submitted_row))
        return RequestRecord(
            request_id="row-1",
            task="text",
            success=True,
            classifier="ok",
            latency=1.0,
            ttft=0.1,
            token_timing_available=True,
            prompt_len=1,
            output_len=97,
            generated_text="ok",
        )

    async def fake_run_load(rows, **kwargs):
        await kwargs["warmup_submit"](rows[0])
        measured = await kwargs["submit"](rows[0])
        return [measured], 1.0

    async def fake_server_info(self, _client):
        del self
        return None

    monkeypatch.setattr(BenchmarkRunner, "_submit", fake_submit)
    monkeypatch.setattr(BenchmarkRunner, "_fetch_server_info", fake_server_info)
    monkeypatch.setattr(harness_runner, "run_load", fake_run_load)

    runner = BenchmarkRunner(
        "http://127.0.0.1:8000",
        BenchmarkSpec(
            task=TaskName.TEXT,
            model="test",
            num_prompts=1,
            sample_gpu_memory=False,
        ),
        tmp_path,
    )
    asyncio.run(runner.run())

    assert submitted[0]["output_len"] == 97
    assert submitted[0].get("max_tokens") is None
