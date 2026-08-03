from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "generation_runtime_candidate",
        ROOT / "scripts" / "generation_runtime_candidate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controller = _load_controller()


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


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Runtime Test")
    _git(repo, "config", "user.email", "runtime@example.test")
    schemas = repo / "schemas"
    schemas.mkdir()
    for name in (
        "generation-runtime-candidate.schema.json",
        "generation-runtime-candidate-plan.schema.json",
    ):
        shutil.copy2(ROOT / "schemas" / name, schemas / name)
    profiles = repo / "uniserve_eval"
    profiles.mkdir()
    shutil.copy2(ROOT / "uniserve_eval" / "profiles.json", profiles / "profiles.json")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repo / "base.txt").write_text("accepted\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "base.txt", "schemas", "uniserve_eval/profiles.json")
    _git(repo, "commit", "-m", "accepted parent")
    return repo, _git(repo, "rev-parse", "HEAD")


def _check(
    check_id: str,
    order: int,
    code: str,
    *,
    gate: str = "C1",
    prerequisites: list[str] | None = None,
    measurement: dict | None = None,
) -> dict:
    value = {
        "id": check_id,
        "gate": gate,
        "order": order,
        "command": [sys.executable, "-c", code, "{artifact_root}"],
        "prerequisites": prerequisites or [],
        "timeout_s": 30,
        "artifact_role": "canonical",
        "artifact_path": f"artifacts/qualification/generation_runtime/S01/{{candidate}}/{check_id}.log",
    }
    if measurement is not None:
        value["measurement"] = measurement
    return value


def _comparators(parent: str) -> dict:
    return {
        "construction_anchor": {
            "source_commit": parent,
            "artifact_root": "artifacts/qualification/generation_runtime/reference/construction",
        },
        "performance_reference": {
            "source_commit": parent,
            "artifact_root": "artifacts/qualification/generation_runtime/reference/performance",
        },
        "previous_candidate": {
            "source_commit": parent,
            "artifact_root": "artifacts/qualification/generation_runtime/reference/previous",
        },
    }


def _plan(
    repo: Path,
    parent: str,
    checks: list[dict],
    *,
    comparators: dict | None = None,
) -> Path:
    plan = {
        "schema_version": 1,
        "boundary": "S01",
        "development_ref": "refs/heads/main",
        "affected_scope": {
            "paths": ["feature.txt"],
            "impact_classes": ["executable"],
        },
        "checks": checks,
        "comparators": comparators or {},
        "artifact_root": "artifacts/qualification/generation_runtime/S01/{candidate}",
        "acceptance": {"fast_forward": True, "update_ref": True},
    }
    path = repo / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _prepare(repo: Path, plan: Path) -> tuple[dict, Path]:
    output = Path("artifacts/qualification/generation_runtime/S01/{candidate}/candidate.json")
    manifest = controller.prepare_candidate(repo, plan, output, "candidate boundary")
    path = repo / manifest["artifact_root"] / "candidate.json"
    return manifest, path


def _stage_feature(repo: Path, content: str = "candidate\n") -> None:
    (repo / "feature.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", "feature.txt")


def test_prepare_materializes_an_immutable_candidate_without_advancing_the_ref(
    tmp_path: Path,
) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    plan = _plan(repo, parent, [_check("contract", 0, "print('ok')")])
    manifest, manifest_path = _prepare(repo, plan)
    candidate = manifest["candidate"]["commit"]
    assert _git(repo, "rev-parse", "refs/heads/main") == parent
    assert _git(repo, "rev-list", "--parents", "-n", "1", candidate).split() == [
        candidate,
        parent,
    ]
    assert _git(repo, "diff", "--name-only", parent, candidate) == "feature.txt"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_manifest_scope_is_bound_to_the_candidate_source_diff(tmp_path: Path) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    plan = _plan(repo, parent, [_check("contract", 0, "print('ok')")])
    manifest, _manifest_path = _prepare(repo, plan)
    manifest["affected_scope"]["paths"] = ["base.txt"]
    with pytest.raises(controller.ControllerError, match="affected scope"):
        controller._validate_manifest_semantics(controller.GitRepo(repo), manifest)


def test_candidate_lock_is_shared_by_worktrees_and_isolated_between_repositories(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, _parent = _repo(first_root)
    second, _parent = _repo(second_root)
    linked = tmp_path / "linked"
    _git(first, "worktree", "add", "--detach", str(linked), "HEAD")
    first_lock = controller.GitRepo(first).common_dir() / "generation-runtime-candidate.lock"
    linked_lock = controller.GitRepo(linked).common_dir() / "generation-runtime-candidate.lock"
    second_lock = controller.GitRepo(second).common_dir() / "generation-runtime-candidate.lock"
    assert first_lock == linked_lock
    assert first_lock != second_lock


def test_candidate_checks_execute_serially_and_accept_by_fast_forward(tmp_path: Path) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    first = "from pathlib import Path; import sys; Path(sys.argv[1], 'sequence.txt').write_text('ready')"
    second = "from pathlib import Path; import sys; p=Path(sys.argv[1], 'sequence.txt'); assert p.read_text() == 'ready'; p.write_text('complete')"
    checks = [
        _check("prepare_evidence", 0, first),
        _check("consume_evidence", 1, second, prerequisites=["prepare_evidence"]),
    ]
    plan = _plan(repo, parent, checks)
    manifest, manifest_path = _prepare(repo, plan)
    result = controller.run_candidate(repo, manifest_path, accept=True)
    candidate = manifest["candidate"]["commit"]
    assert result["status"] == "accepted"
    assert _git(repo, "rev-parse", "refs/heads/main") == candidate
    assert (repo / manifest["artifact_root"] / "sequence.txt").read_text() == "complete"
    assert json.loads(
        (repo / manifest["artifact_root"] / "acceptance.json").read_text(encoding="utf-8")
    )["candidate"] == candidate


def test_candidate_execution_stops_at_the_first_failed_check(tmp_path: Path) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    checks = [
        _check("first", 0, "print('first')"),
        _check("decisive_gate", 1, "raise SystemExit(7)", prerequisites=["first"]),
        _check("later", 2, "print('later')", prerequisites=["decisive_gate"]),
    ]
    plan = _plan(repo, parent, checks)
    _manifest, manifest_path = _prepare(repo, plan)
    result = controller.run_candidate(repo, manifest_path, accept=False)
    assert result["status"] == "failed"
    assert result["completed_check_count"] == 2
    assert result["stopped_after"] == "decisive_gate"
    assert [check["id"] for check in result["checks"]] == ["first", "decisive_gate"]
    assert _git(repo, "rev-parse", "refs/heads/main") == parent


def test_candidate_records_a_terminal_failure_when_a_check_cannot_launch(
    tmp_path: Path,
) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    check = _check("tooling", 0, "print('unreachable')")
    check["command"] = ["generation-runtime-missing-executable"]
    plan = _plan(repo, parent, [check])
    manifest, manifest_path = _prepare(repo, plan)
    result = controller.run_candidate(repo, manifest_path, accept=False)
    assert result["status"] == "failed"
    assert result["checks"][0]["returncode"] == 127
    assert result["checks"][0]["launch_error"].startswith("FileNotFoundError:")
    artifact_root = repo / manifest["artifact_root"]
    assert json.loads((artifact_root / "validation.json").read_text(encoding="utf-8")) == result


def test_performance_checks_bind_all_required_comparators(tmp_path: Path) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    plan = _plan(
        repo,
        parent,
        [_check("performance", 0, "print('p1')", gate="P1")],
        comparators={
            "construction_anchor": _comparators(parent)["construction_anchor"],
        },
    )
    with pytest.raises(controller.ControllerError, match="required comparators"):
        _prepare(repo, plan)


def test_dry_run_expands_the_declared_p1_through_p4_point_set(tmp_path: Path) -> None:
    repo, parent = _repo(tmp_path)
    _stage_feature(repo)
    point_sets = {
        "P1": [
            {"id": "qwen3_sharegpt_r16", "kind": "benchmark", "profile": "main", "group": "qwen-uniserve", "point": "qwen3_sharegpt_uniserve", "load_case_set": "text_arrival", "load_case_id": "r16", "role": "candidate"},
            {"id": "sensenova_t2i_c32", "kind": "benchmark", "profile": "main", "group": "sensenova-uniserve", "point": "sensenova_mjhq_t2i_uniserve", "load_case_set": "image_concurrency", "load_case_id": "c32", "role": "candidate"},
            {"id": "sensenova_i2t_c32", "kind": "benchmark", "profile": "main", "group": "sensenova-uniserve", "point": "sensenova_beans_i2t_uniserve", "load_case_set": "sensenova_i2t_load", "load_case_id": "c32", "role": "candidate"},
        ],
        "P2": [
            {"id": "sensenova_default_travel", "kind": "verification", "profile": "gate/sensenova/default-travel", "topology": "tp2", "role": "candidate"},
            {"id": "sensenova_greedy_interleave_c4", "kind": "benchmark", "profile": "main", "group": "sensenova-uniserve", "point": "sensenova_ueval_interleave_uniserve", "load_case_set": "interleave_concurrency", "load_case_id": "c4", "topology": "tp2", "role": "candidate"},
        ],
        "P3": [
            {"id": "sensenova_stochastic_interleave_c4_depth_one", "kind": "oracle", "profile": "main", "group": "sensenova-uniserve-stochastic", "point": "sensenova_ueval_stochastic_interleave_uniserve", "load_case_set": "interleave_qualification", "load_case_id": "c4", "execution_depth": 1, "topology": "tp2", "role": "oracle"},
            {"id": "sensenova_stochastic_interleave_c4_advertised_depth", "kind": "benchmark", "profile": "main", "group": "sensenova-uniserve-stochastic", "point": "sensenova_ueval_stochastic_interleave_uniserve", "load_case_set": "interleave_qualification", "load_case_id": "c4", "execution_depth": "advertised", "topology": "tp2", "role": "candidate"},
        ],
        "P4": [
            {"id": "qwen3_sharegpt_reference", "kind": "reference", "profile": "main", "group": "qwen-sglang", "point": "qwen3_sharegpt_sglang", "load_case_set": "text_arrival", "load_case_id": "r16", "role": "reference"},
            {"id": "sensenova_t2i_reference", "kind": "reference", "profile": "main", "group": "sensenova-omni", "point": "sensenova_mjhq_t2i_omni", "load_case_set": "image_concurrency", "load_case_id": "c32", "role": "reference"},
            {"id": "sensenova_i2t_reference", "kind": "reference", "profile": "main", "group": "sensenova-omni", "point": "sensenova_beans_i2t_omni", "load_case_set": "sensenova_i2t_load", "load_case_id": "c32", "role": "reference"},
            {"id": "tensor_parallel_single_rank", "kind": "oracle", "topology": "tp1", "role": "oracle"},
            {"id": "tensor_parallel_configured_multi_rank", "kind": "oracle", "topology": "tp2", "role": "oracle"},
            {"id": "tensorized_mixed", "kind": "oracle", "topology": "tp2", "role": "oracle"},
            {"id": "resource_failure_matrix", "kind": "stress", "topology": "configured", "role": "oracle"},
        ],
    }
    checks = []
    previous = None
    for order, (gate, points) in enumerate(point_sets.items()):
        check = _check(
            f"{gate.lower()}_points",
            order,
            f"print('{gate}')",
            gate=gate,
            prerequisites=[previous] if previous else [],
            measurement={"points": points},
        )
        checks.append(check)
        previous = check["id"]
    plan = _plan(repo, parent, checks, comparators=_comparators(parent))
    _manifest, manifest_path = _prepare(repo, plan)
    report = controller.dry_run(repo, manifest_path)
    expanded = {
        check["gate"]: [point["id"] for point in check["measurement_points"]]
        for check in report["checks"]
    }
    assert expanded == {
        gate: [point["id"] for point in points] for gate, points in point_sets.items()
    }
