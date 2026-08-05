"""Paired generated-image smoke checks over canonical artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import uniserve_eval.harness.image_quality as image_quality
from uniserve_eval.harness.artifacts import ArtifactWriter
from uniserve_eval.harness.image_outputs import inspect_image_bytes
from uniserve_eval.harness.report import (
    attach_execution_contract,
    benchmark_contract,
    benchmark_parity_contract,
    canonical_digest,
    record_collection_contract,
    spec_to_dict,
    write_summary_artifacts,
)
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.profiles import (
    benchmark_matrix_definition_contract,
    server_command_template,
    server_profile_definition_contract,
    server_spec,
)

pytestmark = pytest.mark.unit

_REFERENCE_SERVER_PROFILE = "benchmark/server/image-reference"
_CANDIDATE_SERVER_PROFILE = "benchmark/server/image-candidate"
_REFERENCE_SOURCE_REVISION = "1" * 40
_REFERENCE_SOURCE_ROLE = "pythonpath:0"
_PROFILE_CONFIG_TEMPLATE = {
    "python": ".venv/bin/python",
    "server_bin": "target/release/uniserve",
    "servers": {
        _REFERENCE_SERVER_PROFILE: {
            "model": "fixture-model",
            "host": "127.0.0.1",
            "port": 18080,
            "command": ["fixture-image-reference", "--model", "fixture-model"],
            "required_source_revision": _REFERENCE_SOURCE_REVISION,
            "required_source_role": _REFERENCE_SOURCE_ROLE,
        },
        _CANDIDATE_SERVER_PROFILE: {
            "model": "fixture-model",
            "host": "127.0.0.1",
            "port": 18081,
            "command": ["fixture-image-candidate", "--model", "fixture-model"],
        },
    },
    "benchmarks": {
        "main": {
            "hardware_requirements": {
                "gpu_count_per_process": 1,
                "gpu_model": "fixture-gpu",
            },
            "defaults": {"warmup_requests": 0, "seed": 42},
            "load_cases": {
                "image_concurrency": [{"id": "c1", "request_rate": "inf", "max_concurrency": 1}]
            },
            "datasets": {},
            "groups": {
                "candidate": {
                    "server": _CANDIDATE_SERVER_PROFILE,
                    "points": ["candidate_point"],
                },
                "reference": {
                    "server": _REFERENCE_SERVER_PROFILE,
                    "points": ["reference_point"],
                },
            },
            "points": {
                "candidate_point": {
                    "load_case_set": "image_concurrency",
                    "parity_group": "image-generation",
                    "comparison_role": "candidate",
                    "name": "fixture-image-candidate-{load_id}",
                    "output": "candidate-{load_id}",
                    "harness": {
                        "task": "t2i",
                        "model": "fixture-model",
                        "num_prompts": 1,
                        "width": 16,
                        "height": 16,
                        "max_images": 1,
                        "runtime_profile_id": "fixture-image-candidate",
                    },
                },
                "reference_point": {
                    "load_case_set": "image_concurrency",
                    "parity_group": "image-generation",
                    "comparison_role": "reference",
                    "name": "fixture-image-reference-{load_id}",
                    "output": "reference-{load_id}",
                    "harness": {
                        "task": "t2i",
                        "model": "fixture-model",
                        "num_prompts": 1,
                        "width": 16,
                        "height": 16,
                        "max_images": 1,
                        "runtime_profile_id": "fixture-image-reference",
                        "request_schema": "vllm_omni",
                    },
                },
            },
        }
    },
}
_PROFILE_CONFIG = deepcopy(_PROFILE_CONFIG_TEMPLATE)


@pytest.fixture(autouse=True)
def _use_fixture_profile_config(monkeypatch: pytest.MonkeyPatch) -> None:
    global _PROFILE_CONFIG
    _PROFILE_CONFIG = deepcopy(_PROFILE_CONFIG_TEMPLATE)
    monkeypatch.setattr(image_quality, "load_config", lambda _path: _PROFILE_CONFIG)


class _FixedRuntime:
    numpy = np

    def __init__(self, metrics: dict[str, float] | None = None) -> None:
        self.provenance = {"implementation": "fixed-fixture"}
        self._metrics = metrics or {
            "lpips": 0.0,
            "ssim": 1.0,
            "psnr_db": math.inf,
            "uint8_mae": 0.0,
            "cosine_similarity": 1.0,
            "relative_l2": 0.0,
        }

    def compare(self, reference, candidate) -> dict[str, float]:
        assert reference.dtype == np.uint8
        assert candidate.dtype == np.uint8
        return dict(self._metrics)


def _image_bytes(*, width: int, height: int, color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="PNG")
    return output.getvalue()


def _record(
    request_id: str,
    images: list,
    *,
    width: int,
    height: int,
) -> dict:
    return {
        "request_id": request_id,
        "task": "t2i",
        "success": True,
        "classifier": "ok",
        "images": len(images),
        "image_output_mode": "required",
        "requested_image_count": len(images),
        "requested_image_width": width,
        "requested_image_height": height,
        "image_outputs": [image.metadata_dict() for image in images],
    }


def _contract(payload: dict) -> dict:
    return {**payload, "fingerprint": canonical_digest(payload)}


def _file_contract(path: Path) -> dict:
    contents = path.read_bytes()
    return {
        "size_bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }


def _server_profile_binding(
    profile_contract: dict,
    server_execution: dict,
    harness_environment_contract: dict,
    model_contract: dict | None,
    model_revision_contract: dict | None,
    required_source_revision: str | None,
    required_source_role: str | None,
    server_profile: str,
) -> dict:
    active_server = server_spec(_PROFILE_CONFIG, server_profile)
    overrides = {
        **{str(key): str(value) for key, value in dict(active_server.get("env") or {}).items()},
        "CUDA_VISIBLE_DEVICES": "0",
    }
    override_items = sorted(overrides.items())
    return _contract(
        {
            "schema_version": 3,
            "profile_definition_fingerprint": profile_contract["fingerprint"],
            "server_execution_fingerprint": server_execution["fingerprint"],
            "baseline_environment_fingerprint": harness_environment_contract["fingerprint"],
            "resolved_environment_fingerprint": server_execution["environment"]["fingerprint"],
            "environment_override_keys": [key for key, _value in override_items],
            "environment_overrides_sha256": canonical_digest(override_items),
            "command_template_sha256": canonical_digest(
                server_command_template(_PROFILE_CONFIG, server_profile)
            ),
            "model_contract_sha256": (
                canonical_digest(model_contract) if model_contract is not None else None
            ),
            "model_revision_contract_sha256": (
                canonical_digest(model_revision_contract)
                if model_revision_contract is not None
                else None
            ),
            "required_source_revision": required_source_revision,
            "required_source_role": required_source_role,
        }
    )


def _hardware_contract() -> dict:
    host = _contract({"schema_version": 1, "machine": "fixture"})
    selected = _contract(
        {
            "schema_version": 1,
            "selector": "0",
            "gpu": {"name": "fixture-gpu"},
        }
    )
    return _contract(
        {
            "schema_version": 2,
            "host": host,
            "selected_accelerator": selected,
        }
    )


def _write_canonical_artifact(
    directory: Path,
    *,
    model: str,
    request_ids: list[str],
    width: int = 16,
    height: int = 16,
    images_per_request: int = 1,
    selected_rows: list[dict] | None = None,
    seed: int = 42,
    matrix_canonical: bool = True,
    comparison_role: str | None = None,
    server_profile: str | None = None,
    server_execution_command: list[str] | None = None,
    force_generic_server_identity: bool = False,
    images_per_second: float = 100.0,
    started_at: float = 1.0,
) -> dict:
    for point in _PROFILE_CONFIG["benchmarks"]["main"]["points"].values():
        point["harness"]["num_prompts"] = len(request_ids)
    writer = ArtifactWriter(directory)
    records = []
    for request_index, request_id in enumerate(request_ids):
        images = []
        for image_index in range(images_per_request):
            data = _image_bytes(
                width=width,
                height=height,
                color=(31 + request_index, 47 + image_index, 71),
            )
            image = inspect_image_bytes(data)
            writer.write_image_sample(image)
            images.append(image)
        records.append(_record(request_id, images, width=width, height=height))

    spec = BenchmarkSpec(
        task=TaskName.T2I,
        model=model,
        num_prompts=len(request_ids),
        warmup_requests=0,
        seed=seed,
        request_rate=float("inf"),
        max_concurrency=1,
        width=16,
        height=16,
        max_images=1,
    )
    rows = selected_rows or [
        {"id": request_id, "prompt": "confidential fixture prompt"} for request_id in request_ids
    ]
    contract = benchmark_contract(spec, rows)
    gpu_samples: list[dict] = []
    summary = {
        "request_count": len(records),
        "spec": spec_to_dict(spec),
        "base_url": "http://127.0.0.1:8000",
        "metrics": {"images_per_second": images_per_second},
        "artifact": {
            "schema_version": 4,
            "valid": True,
            "valid_marker": "artifact-valid-v4",
            "checks": {"contract": True},
            "contract": contract,
            "request_records": record_collection_contract(records),
            "gpu_samples": record_collection_contract(gpu_samples),
        },
    }
    writer.write_jsonl("requests.jsonl", records)
    writer.write_jsonl("gpu_samples.jsonl", gpu_samples)
    writer.write_json(
        "run.json",
        {
            "harness_status": "completed",
            "artifact_valid": True,
            "items": len(records),
            "spec": summary["spec"],
            "base_url": summary["base_url"],
            "started_at": started_at,
        },
    )
    if matrix_canonical:
        role = comparison_role or ("reference" if directory.name == "reference" else "candidate")
        group_name = role
        point_name = f"{role}_point"
        benchmark = f"fixture-image-{role}-c1"
        resolved_server_profile = server_profile or (
            _REFERENCE_SERVER_PROFILE if role == "reference" else _CANDIDATE_SERVER_PROFILE
        )
        active_server = server_spec(_PROFILE_CONFIG, resolved_server_profile)
        required_source_revision = active_server.get("required_source_revision")
        required_source_role = active_server.get("required_source_role")
        model_revision_contract = None
        model_contract = {
            "kind": "model_directory",
            "resolved_name": "fixture-model",
            "file_count": 1,
            "total_size_bytes": 1,
            "tree_sha256": "1" * 64,
        }
        server_group = f"group-{directory.name}"
        for name in ("command.txt", "preflight.txt", "postflight.txt", "run.log"):
            (directory / name).write_text(f"{benchmark}:{name}\n", encoding="utf-8")
        point_files = {
            name: _file_contract(directory / name)
            for name in ("command.txt", "preflight.txt", "postflight.txt", "run.json", "run.log")
        }
        server_directory = directory.parent / "servers" / server_group / benchmark
        server_directory.mkdir(parents=True)
        server_names = (
            "server_command.txt",
            "server.log",
            "server.exit",
            "pre_server_snapshot.txt",
            "post_server_snapshot.txt",
        )
        for name in server_names:
            (server_directory / name).write_text(f"{benchmark}:{name}\n", encoding="utf-8")
        server_files = {name: _file_contract(server_directory / name) for name in server_names}
        build_log_path = directory.parent / "build.log"
        if not build_log_path.is_file():
            build_log_path.write_text("source build completed\n", encoding="utf-8")
        build_file = _file_contract(build_log_path)
        source_head = "1" * 40
        source_state = {
            "schema_version": 1,
            "head": source_head,
            "dirty": False,
            "fingerprint": "2" * 64,
        }
        default_server_command = server_command_template(
            _PROFILE_CONFIG,
            resolved_server_profile,
        )
        model_index = default_server_command.index(str(active_server["model"]))
        previous = default_server_command[model_index - 1]
        model_role = (
            f"option:{previous}" if previous.startswith("--") else f"command_argument:{model_index}"
        )
        server_execution = _contract(
            {
                "schema_version": 2,
                "role": "server",
                "command": server_execution_command or default_server_command,
                "environment": _contract(
                    {
                        "schema_version": 1,
                        "variable_count": 1,
                        "variables_sha256": "5" * 64,
                    }
                ),
                "performance_environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "command_inputs": [{"role": model_role, **model_contract}],
                "executables": (
                    [
                        {
                            "role": "command_argument:1",
                            "requested": "uniserve",
                            "resolved_name": "uniserve",
                            "size_bytes": build_file["size_bytes"],
                            "sha256": build_file["sha256"],
                        }
                    ]
                    if role == "candidate" and not force_generic_server_identity
                    else []
                ),
                "source_revisions": [
                    {
                        "roles": (
                            [str(required_source_role)]
                            if required_source_role is not None
                            and not force_generic_server_identity
                            else ["workspace"]
                        ),
                        "state": source_state,
                    }
                ],
            }
        )
        harness_execution = _contract(
            {
                "schema_version": 2,
                "role": "harness",
                "environment": _contract(
                    {
                        "schema_version": 1,
                        "variable_count": 1,
                        "variables_sha256": "5" * 64,
                    }
                ),
                "performance_environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "source_revisions": [{"roles": ["harness"], "state": source_state}],
            }
        )
        build_manifest = _contract(
            {
                "schema_version": 1,
                "command": ["cargo", "build", "--release", "--bin", "uniserve"],
                "cargo_version": "cargo test",
                "rustc_version": "rustc test",
                "source_state": source_state,
                "cargo_lock": build_file,
                "binary": build_file,
                "build_log": build_file,
            }
        )
        execution_policy = _contract(
            {
                "schema_version": 1,
                "formal": True,
                "selected_groups": ["reference", "candidate"],
                "checks": {
                    "host_lock": True,
                    "no_preexisting_benchmark_process": True,
                    "fresh_output_root": True,
                    "complete_parity_selection": True,
                    "clean_gpu_before_and_after_each_point": True,
                    "clean_source_before_and_after_each_point": True,
                    "pinned_reference_revisions": True,
                    "one_required_gpu_per_process": True,
                    "gpu_local_server_and_harness_numa_binding": True,
                    "fresh_server_per_point": True,
                    "source_build_for_this_root": True,
                },
                "build_manifest": build_manifest,
            }
        )
        parity_payload = {
            "schema_version": 2,
            "harness": benchmark_parity_contract(contract),
            "model": model_contract,
        }
        parity_contract = _contract(parity_payload)
        profile_contract = server_profile_definition_contract(
            _PROFILE_CONFIG,
            resolved_server_profile,
        )
        matrix_payload = {
            "schema_version": 2,
            "benchmark": benchmark,
            "benchmark_profile": "main",
            "benchmark_definition": benchmark_matrix_definition_contract(
                _PROFILE_CONFIG,
                "main",
                group_name=group_name,
                point_name=point_name,
                load_case_set="image_concurrency",
                load_case_id="c1",
            ),
            "server_profile": resolved_server_profile,
            "server_profile_contract": profile_contract,
            "server_profile_binding": _server_profile_binding(
                profile_contract,
                server_execution,
                harness_execution["environment"],
                model_contract,
                model_revision_contract,
                required_source_revision,
                required_source_role,
                resolved_server_profile,
            ),
            "comparison_role": role,
            "required_server_source_revision": required_source_revision,
            "required_server_source_role": required_source_role,
            "model_revision_contract": model_revision_contract,
            "execution_policy": execution_policy,
            "server_execution": server_execution,
            "harness_execution": harness_execution,
            "hardware": _hardware_contract(),
            "harness_contract_fingerprint": contract["fingerprint"],
            "parity_group": "image-generation",
            "parity_contract": parity_contract,
        }
        execution_payload = {
            "schema_version": 2,
            "benchmark": benchmark,
            "server_group": server_group,
            "point_files": point_files,
            "server_files": server_files,
        }
        attach_execution_contract(
            summary,
            "execution_bundle_contract",
            _contract(execution_payload),
        )
        attach_execution_contract(summary, "matrix_contract", _contract(matrix_payload))
    else:
        profile_payload = {"schema_version": 2, "name": "fixture-profile"}
        attach_execution_contract(summary, "profile_contract", _contract(profile_payload))
    write_summary_artifacts(directory, summary)
    return summary


def test_image_quality_passes_canonical_pairs_and_redacts_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    request_ids = ["private-request-b", "private-request-a"]
    selected_rows = [
        {"id": request_id, "prompt": "confidential fixture prompt"} for request_id in request_ids
    ]
    _write_canonical_artifact(
        reference,
        model="same-model-reference-runtime",
        request_ids=request_ids,
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="same-model-candidate-runtime",
        request_ids=list(reversed(request_ids)),
        selected_rows=selected_rows,
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert report["passed"] is True
    assert report["pair_count"] == 2
    assert all(report["gates"].values())
    assert report["thresholds"]["ssim"]["diagnostic_only"] is True
    assert "ssim" not in report["gates"]
    assert all(report["diagnostic_threshold_checks"].values())
    assert report["aggregate"]["lpips"]["finite_p95"] == 0.0
    assert report["aggregate"]["psnr_db"]["positive_infinity_count"] == 2
    serialized = json.dumps(report, sort_keys=True)
    markdown = image_quality.render_markdown(report)
    for private_value in (
        "private-request-a",
        "private-request-b",
        "confidential fixture prompt",
        str(reference),
        str(candidate),
    ):
        assert private_value not in serialized
        assert private_value not in markdown


def test_image_quality_gates_lpips_and_reports_every_diagnostic_threshold(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    failing_runtime = lambda: _FixedRuntime(  # noqa: E731 - compact injected fixture.
        {
            "lpips": 0.16,
            "ssim": 0.90,
            "psnr_db": 19.0,
            "uint8_mae": 13.0,
            "cosine_similarity": 0.97,
            "relative_l2": 0.21,
        }
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=failing_runtime,
    )

    assert report["passed"] is False
    assert report["gates"]["lpips_maximum"] is False
    assert report["diagnostic_threshold_checks"] == {
        "psnr_db_minimum": False,
        "uint8_mae_maximum": False,
        "cosine_similarity_minimum": False,
        "relative_l2_maximum": False,
    }
    assert report["evidence_valid"] is True
    assert report["regression_canary_passed"] is False
    assert report["failures"] == []


def test_secondary_image_thresholds_are_diagnostic_not_acceptance_gates(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    diagnostic_failure_runtime = lambda: _FixedRuntime(  # noqa: E731
        {
            "lpips": 0.10,
            "ssim": 0.50,
            "psnr_db": 19.0,
            "uint8_mae": 13.0,
            "cosine_similarity": 0.97,
            "relative_l2": 0.21,
        }
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=diagnostic_failure_runtime,
    )

    assert report["passed"] is True
    assert report["gates"]["lpips_maximum"] is True
    assert not any(report["diagnostic_threshold_checks"].values())
    assert report["failures"] == []


def test_image_quality_fails_closed_when_lpips_runtime_is_unavailable(
    tmp_path: Path,
) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )

    def unavailable():
        raise image_quality.QualityRuntimeError("unavailable")

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=unavailable,
    )

    assert report["gates"]["quality_dependencies"] is False
    assert report["gates"]["metrics_computed"] is False
    assert report["pairs"] == []
    assert {failure["code"] for failure in report["failures"]} >= {"quality_runtime_unavailable"}


def test_image_quality_rejects_request_count_and_dimension_mismatches(
    tmp_path: Path,
) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    count_candidate = tmp_path / "count-candidate"
    dimension_candidate = tmp_path / "dimension-candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        count_candidate,
        model="model-candidate",
        request_ids=["request-1"],
        images_per_request=2,
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        dimension_candidate,
        model="model-candidate",
        request_ids=["request-1"],
        width=17,
        selected_rows=selected_rows,
    )

    count_report = image_quality.evaluate_image_quality(
        reference,
        count_candidate,
        _runtime_factory=_FixedRuntime,
    )
    dimension_report = image_quality.evaluate_image_quality(
        reference,
        dimension_candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert count_report["gates"]["image_counts"] is False
    assert count_report["gates"]["metrics_computed"] is False
    assert dimension_report["gates"]["declared_dimensions"] is False
    assert dimension_report["gates"]["metrics_computed"] is False


def test_image_quality_revalidates_sample_bindings_and_semantic_contracts(
    tmp_path: Path,
) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    different_seed = tmp_path / "different-seed"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        different_seed,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
        seed=43,
    )

    sample = next((candidate / "samples").iterdir())
    sample.write_bytes(b"mutated")
    binding_report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )
    semantic_report = image_quality.evaluate_image_quality(
        reference,
        different_seed,
        _runtime_factory=_FixedRuntime,
    )

    assert binding_report["gates"]["canonical_artifacts"] is False
    assert {failure["code"] for failure in binding_report["failures"]} >= {
        "canonical_artifact_invalid"
    }
    assert semantic_report["gates"]["semantic_parity"] is False


def test_image_quality_rejects_profile_only_canonical_evidence(tmp_path: Path) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
        matrix_canonical=False,
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert report["gates"]["canonical_artifacts"] is False
    assert {failure["code"] for failure in report["failures"]} >= {
        "matrix_contract_invalid",
        "execution_bundle_invalid",
    }


def test_image_quality_rejects_reversed_reference_candidate_roles(tmp_path: Path) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )

    report = image_quality.evaluate_image_quality(
        candidate,
        reference,
        _runtime_factory=_FixedRuntime,
    )

    assert report["passed"] is False
    assert report["gates"]["comparison_roles"] is False


def test_image_quality_rejects_server_profile_outside_active_role(tmp_path: Path) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
        server_profile=_REFERENCE_SERVER_PROFILE,
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert report["passed"] is False
    assert report["gates"]["comparison_roles"] is True
    assert report["gates"]["active_server_profile_bindings"] is False
    assert "server_profile_binding_mismatch" in {failure["code"] for failure in report["failures"]}


def test_image_quality_requires_distinct_server_execution_identities(tmp_path: Path) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    shared_command = ["benchmark/server/shared-implementation"]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
        server_execution_command=shared_command,
        force_generic_server_identity=True,
    )
    _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
        server_execution_command=shared_command,
        force_generic_server_identity=True,
    )

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert report["passed"] is False
    assert report["gates"]["active_server_profile_bindings"] is False
    assert report["gates"]["distinct_server_implementation_identities"] is False
    assert "candidate_and_reference_server_identities_are_not_distinct" in {
        failure["code"] for failure in report["failures"]
    }


@pytest.mark.parametrize(
    "mutation",
    ["point_file_mutated", "run_file_missing", "server_file_missing"],
)
def test_image_quality_revalidates_every_execution_support_file(
    tmp_path: Path,
    mutation: str,
) -> None:
    selected_rows = [{"id": "request-1", "prompt": "x"}]
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_canonical_artifact(
        reference,
        model="model-reference",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    candidate_summary = _write_canonical_artifact(
        candidate,
        model="model-candidate",
        request_ids=["request-1"],
        selected_rows=selected_rows,
    )
    execution = candidate_summary["artifact"]["execution_bundle_contract"]
    assert set(execution["point_files"]) == {
        "command.txt",
        "preflight.txt",
        "postflight.txt",
        "run.json",
        "run.log",
    }
    if mutation == "point_file_mutated":
        (candidate / "command.txt").write_text("mutated\n", encoding="utf-8")
    elif mutation == "run_file_missing":
        (candidate / "run.json").unlink()
    else:
        server_directory = tmp_path / "servers" / execution["server_group"] / execution["benchmark"]
        (server_directory / "server.log").unlink()

    report = image_quality.evaluate_image_quality(
        reference,
        candidate,
        _runtime_factory=_FixedRuntime,
    )

    assert report["gates"]["canonical_artifacts"] is False
    assert {failure["code"] for failure in report["failures"]} >= {"execution_bundle_invalid"}


def test_image_quality_rejects_a_self_comparison(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_canonical_artifact(
        artifact,
        model="model",
        request_ids=["request-1"],
    )

    report = image_quality.evaluate_image_quality(
        artifact,
        artifact,
        _runtime_factory=_FixedRuntime,
    )

    assert report["passed"] is False
    assert report["gates"]["distinct_artifacts"] is False
    assert {failure["code"] for failure in report["failures"]} >= {
        "reference_and_candidate_are_same_artifact"
    }


def test_pixel_metric_definitions_use_full_uint8_declared_arrays() -> None:
    class _Model:
        def __call__(self, _reference, _candidate):
            return torch.tensor(0.0)

    runtime = image_quality._QualityRuntime(
        model=_Model(),
        transform=lambda _image: torch.zeros((3, 8, 8), dtype=torch.float32),
        torch=torch,
        numpy=np,
        structural_similarity=lambda *_args, **_kwargs: 0.75,
        provenance={},
    )
    reference = np.full((8, 8, 3), 100, dtype=np.uint8)
    candidate = np.full((8, 8, 3), 110, dtype=np.uint8)

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    metrics = runtime.compare(reference, candidate)

    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert metrics["lpips"] == 0.0
    assert metrics["uint8_mae"] == 10.0
    assert metrics["psnr_db"] == pytest.approx(28.1308036087)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == pytest.approx(0.1)
    assert metrics["ssim"] == 0.75
