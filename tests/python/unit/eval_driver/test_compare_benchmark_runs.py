from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_benchmark_runs",
        ROOT / "scripts" / "compare_benchmark_runs.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load_comparison_module()


def _git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return process.stdout.strip()


def _summary(root: Path, source_commit: str, throughput: float, *, source_clean: bool = True) -> None:
    root.mkdir(parents=True)
    value = {
        "task": "text",
        "request_count": 2,
        "ok_count": 2,
        "failed_count": 0,
        "metrics": {"output_throughput": throughput},
        "artifact": {
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "checks": {"matrix_contract": True, "request_work": True},
            "generation_conformance": {"valid": True},
            "contract": {
                "spec": {
                    "task": "text",
                    "model": "Qwen3-32B",
                    "dataset_revision": "1" * 40,
                    "dataset_path": str(root / "dataset"),
                    "temperature": 0.0,
                },
                "selected_rows": {"count": 2, "sha256": "2" * 64},
            },
            "matrix_contract": {
                "benchmark_definition": {
                    "fingerprint": "3" * 64,
                    "load_case_id": "r16",
                },
                "execution_policy": {
                    "build_manifest": {
                        "source_state": {"head": source_commit, "dirty": not source_clean}
                    }
                },
            },
        },
    }
    (root / "summary.json").write_text(json.dumps(value), encoding="utf-8")


def _manifest(tmp_path: Path, candidate_commit: str, comparator_commits: dict[str, str]) -> Path:
    value = {
        "candidate": {"commit": candidate_commit, "parent": "f" * 40},
        "comparators": {
            name: {
                "source_commit": commit,
                "artifact_root": f"references/{name}",
            }
            for name, commit in comparator_commits.items()
        },
        "checks": [
            {
                "id": "p1_retention",
                "measurement": {
                    "points": [
                        {
                            "id": "qwen3_sharegpt_r16",
                            "artifact_path": "{artifact_root}/qwen3",
                            "required_success_count": 2,
                            "comparator_paths": {
                                name: "qwen3" for name in comparator_commits
                            },
                            "metrics": [
                                {
                                    "name": "output_throughput",
                                    "objective": "maximize",
                                    "path": ["metrics", "output_throughput"],
                                }
                            ],
                        }
                    ]
                },
            }
        ],
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _environment(monkeypatch, tmp_path: Path, candidate_commit: str) -> Path:
    artifact_root = tmp_path / "qualification"
    monkeypatch.setenv("UNISERVE_QUALIFICATION_ROOT", str(artifact_root))
    monkeypatch.setenv("UNISERVE_CANDIDATE_COMMIT", candidate_commit)
    monkeypatch.setenv("UNISERVE_CANDIDATE_PARENT", "f" * 40)
    monkeypatch.setenv("UNISERVE_WORKSPACE", str(tmp_path))
    return artifact_root


def test_manifest_comparison_applies_every_comparator_at_the_fixed_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_commit = "a" * 40
    comparator_commits = {
        "construction_anchor": "b" * 40,
        "performance_reference": "c" * 40,
        "previous_candidate": "d" * 40,
    }
    artifact_root = _environment(monkeypatch, tmp_path, candidate_commit)
    _summary(artifact_root / "qwen3", candidate_commit, 80.0)
    for name, commit in comparator_commits.items():
        _summary(tmp_path / "references" / name / "qwen3", commit, 100.0)
    result = comparison.compare_manifest_check(
        _manifest(tmp_path, candidate_commit, comparator_commits),
        "p1_retention",
    )
    assert result["status"] == "passed"
    metric = result["points"][0]["metrics"][0]
    assert metric["effective_regression"] == pytest.approx(0.2)
    assert set(metric["regressions"]) == set(comparator_commits)


def test_manifest_comparison_propagates_nested_artifact_invariants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_commit = "a" * 40
    comparator_commits = {
        "construction_anchor": "b" * 40,
        "performance_reference": "c" * 40,
        "previous_candidate": "d" * 40,
    }
    artifact_root = _environment(monkeypatch, tmp_path, candidate_commit)
    _summary(artifact_root / "qwen3", candidate_commit, 100.0, source_clean=False)
    for name, commit in comparator_commits.items():
        _summary(tmp_path / "references" / name / "qwen3", commit, 100.0)
    result = comparison.compare_manifest_check(
        _manifest(tmp_path, candidate_commit, comparator_commits),
        "p1_retention",
    )
    assert result["status"] == "failed"
    assert result["points"][0]["status"] == "failed"
    assert "source_tree_not_clean" in result["points"][0]["hard_failures"]


def test_measurement_control_overlay_preserves_the_named_serving_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Runtime Test")
    _git(repo, "config", "user.email", "runtime@example.test")
    (repo / "serving.txt").write_text("immutable serving source\n", encoding="utf-8")
    _git(repo, "add", "serving.txt")
    _git(repo, "commit", "-m", "serving source")
    source_commit = _git(repo, "rev-parse", "HEAD")
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "run_benchmarks.py").write_text("PROFILE = 'frozen'\n", encoding="utf-8")
    _git(repo, "add", "scripts/run_benchmarks.py")
    _git(repo, "commit", "-m", "bind benchmark control")
    measurement_commit = _git(repo, "rev-parse", "HEAD")

    failures = comparison._control_overlay_failures(
        repo,
        {
            "source_commit": source_commit,
            "measurement_commit": measurement_commit,
        },
    )

    assert failures == []
    assert _git(repo, "show", f"{source_commit}:serving.txt") == _git(
        repo, "show", f"{measurement_commit}:serving.txt"
    )
