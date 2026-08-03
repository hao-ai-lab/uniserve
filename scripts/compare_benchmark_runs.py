#!/usr/bin/env python3
"""Apply a candidate manifest's locked benchmark retention comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REGRESSION_LIMIT = 0.20
ROOT = Path(__file__).resolve().parents[1]


class ComparisonError(RuntimeError):
    """A declared artifact or comparison contract is invalid."""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"JSON document must be an object: {path}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _source_state(matrix: dict[str, Any]) -> dict[str, Any]:
    execution_policy = _mapping(matrix.get("execution_policy"))
    source_state = _mapping(_mapping(execution_policy.get("build_manifest")).get("source_state"))
    if source_state:
        return source_state
    for execution_name in ("server_execution", "harness_execution"):
        revisions = _mapping(matrix.get(execution_name)).get("source_revisions")
        if not isinstance(revisions, list):
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                continue
            if "workspace" in (revision.get("roles") or []):
                state = _mapping(revision.get("state"))
                if state:
                    return state
    return {}


def _workload_contract(artifact: dict[str, Any]) -> dict[str, Any]:
    contract = _mapping(artifact.get("contract"))
    request_spec = _mapping(contract.get("spec"))
    selected_rows = _mapping(contract.get("selected_rows"))
    definition = _mapping(_mapping(artifact.get("matrix_contract")).get("benchmark_definition"))
    if not request_spec:
        return {}
    normalized = dict(request_spec)
    normalized.pop("dataset_path", None)
    values = {
        "request_spec_sha256": hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "selected_row_count": selected_rows.get("count"),
        "selected_rows_sha256": selected_rows.get("sha256"),
        "benchmark_definition_fingerprint": definition.get("fingerprint"),
        "load_case_id": definition.get("load_case_id"),
    }
    return values if all(value is not None and value != "" for value in values.values()) else {}


@dataclass(frozen=True)
class BenchmarkPoint:
    root: Path
    summary: dict[str, Any]
    metrics: dict[str, Any]
    source_commit: str | None
    source_clean: bool
    workload_contract: dict[str, Any]
    failures: tuple[str, ...]

    def metric(self, path: Sequence[str]) -> float | None:
        value: Any = self.summary
        for part in path:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return _finite(value)


def load_benchmark_point(root: Path, *, required_success_count: int | None) -> BenchmarkPoint:
    summary = _read_json(root / "summary.json")
    artifact = _mapping(summary.get("artifact"))
    checks = _mapping(artifact.get("checks"))
    conformance = _mapping(artifact.get("generation_conformance"))
    matrix = _mapping(artifact.get("matrix_contract"))
    source = _source_state(matrix)
    failures: list[str] = []
    if artifact.get("valid") is not True or artifact.get("valid_marker") != "canonical-valid-v2":
        failures.append("artifact_invalid")
    if not checks:
        failures.append("artifact_checks_missing")
    else:
        failures.extend(f"artifact_check:{name}" for name, value in checks.items() if value is not True)
    if conformance.get("valid") is not True:
        failures.append("generation_conformance_invalid")
    request_count = summary.get("request_count")
    ok_count = summary.get("ok_count")
    failed_count = summary.get("failed_count")
    if failed_count != 0:
        failures.append(f"failed_requests={failed_count}")
    if request_count != ok_count:
        failures.append(f"incomplete_success={ok_count}/{request_count}")
    if required_success_count is not None and ok_count != required_success_count:
        failures.append(f"required_success={ok_count}/{required_success_count}")
    workload = _workload_contract(artifact)
    if not workload:
        failures.append("workload_contract_missing")
    if not source.get("head"):
        failures.append("source_commit_missing")
    if source.get("dirty") is not False:
        failures.append("source_tree_not_clean")
    return BenchmarkPoint(
        root=root,
        summary=summary,
        metrics=_mapping(summary.get("metrics")),
        source_commit=source.get("head"),
        source_clean=source.get("dirty") is False,
        workload_contract=workload,
        failures=tuple(failures),
    )


def _regression(candidate: float, baseline: float, objective: str) -> float:
    if baseline <= 0.0:
        raise ComparisonError("baseline metric must be positive")
    if objective == "maximize":
        return max(0.0, 1.0 - candidate / baseline)
    if objective == "minimize":
        return max(0.0, candidate / baseline - 1.0)
    raise ComparisonError(f"unsupported metric objective: {objective}")


def _runtime_bindings() -> dict[str, str]:
    required = {
        "artifact_root": os.environ.get("UNISERVE_QUALIFICATION_ROOT"),
        "candidate": os.environ.get("UNISERVE_CANDIDATE_COMMIT"),
        "parent": os.environ.get("UNISERVE_CANDIDATE_PARENT"),
        "workspace": os.environ.get("UNISERVE_WORKSPACE", str(ROOT)),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ComparisonError("missing runtime binding(s): " + ", ".join(sorted(missing)))
    return {name: str(value) for name, value in required.items()}


def _render_path(value: str, bindings: dict[str, str], *, base: Path) -> Path:
    rendered = value
    for name, replacement in bindings.items():
        rendered = rendered.replace("{" + name + "}", replacement)
    if "{" in rendered or "}" in rendered:
        raise ComparisonError(f"unresolved path placeholder: {value!r}")
    path = Path(rendered)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _control_overlay_failures(
    workspace: Path,
    declaration: dict[str, Any],
) -> list[str]:
    measurement_commit = declaration.get("measurement_commit")
    if measurement_commit is None:
        return []
    source_commit = declaration["source_commit"]

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    failures: list[str] = []
    parents = git("rev-list", "--parents", "-n", "1", str(measurement_commit))
    if parents.returncode != 0 or parents.stdout.split() != [measurement_commit, source_commit]:
        failures.append("measurement_commit_parent_mismatch")
        return failures
    changed = git("diff", "--name-only", source_commit, measurement_commit)
    paths = changed.stdout.splitlines() if changed.returncode == 0 else []
    if not paths:
        failures.append("measurement_control_overlay_missing")
    if any(
        path != "scripts/run_benchmarks.py" and not path.startswith("uniserve_eval/")
        for path in paths
    ):
        failures.append("measurement_control_overlay_changes_serving_source")
    return failures


def compare_manifest_check(manifest_path: Path, check_id: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    checks = [check for check in manifest.get("checks", []) if check.get("id") == check_id]
    if len(checks) != 1:
        raise ComparisonError(f"manifest must contain exactly one check named {check_id!r}")
    check = checks[0]
    measurement = _mapping(check.get("measurement"))
    points = measurement.get("points")
    if not isinstance(points, list) or not points:
        raise ComparisonError(f"check {check_id!r} has no measurement points")
    bindings = _runtime_bindings()
    workspace = Path(bindings["workspace"]).resolve()
    candidate_commit = manifest["candidate"]["commit"]
    outcomes: list[dict[str, Any]] = []
    any_failure = False
    for declaration in points:
        if not isinstance(declaration, dict) or not declaration.get("metrics"):
            continue
        point_id = declaration["id"]
        candidate_root = _render_path(declaration["artifact_path"], bindings, base=workspace)
        required_success = declaration.get("required_success_count")
        candidate = load_benchmark_point(
            candidate_root,
            required_success_count=required_success,
        )
        hard_failures = list(candidate.failures)
        if candidate.source_commit != candidate_commit:
            hard_failures.append("candidate_source_commit_mismatch")
        comparators: dict[str, BenchmarkPoint] = {}
        for name, relative in declaration.get("comparator_paths", {}).items():
            comparator_declaration = _mapping(manifest["comparators"].get(name))
            if not comparator_declaration:
                hard_failures.append(f"comparator_undeclared:{name}")
                continue
            comparator_base = _render_path(
                comparator_declaration["artifact_root"], bindings, base=workspace
            )
            comparator = load_benchmark_point(
                (comparator_base / relative).resolve(),
                required_success_count=required_success,
            )
            comparators[name] = comparator
            hard_failures.extend(f"{name}:{failure}" for failure in comparator.failures)
            expected_measurement = comparator_declaration.get(
                "measurement_commit", comparator_declaration["source_commit"]
            )
            if comparator.source_commit != expected_measurement:
                hard_failures.append(f"{name}:source_commit_mismatch")
            hard_failures.extend(
                f"{name}:{failure}"
                for failure in _control_overlay_failures(workspace, comparator_declaration)
            )
            if comparator.workload_contract != candidate.workload_contract:
                hard_failures.append(f"{name}:workload_contract_mismatch")
        metric_outcomes: list[dict[str, Any]] = []
        for metric in declaration["metrics"]:
            path = metric["path"]
            candidate_value = candidate.metric(path)
            values: dict[str, float | None] = {}
            regressions: dict[str, float | None] = {}
            if candidate_value is None:
                hard_failures.append(f"metric_unavailable:{metric['name']}:candidate")
            for name, comparator in comparators.items():
                baseline = comparator.metric(path)
                values[name] = baseline
                if candidate_value is None or baseline is None:
                    regressions[name] = None
                    hard_failures.append(f"metric_unavailable:{metric['name']}:{name}")
                else:
                    try:
                        regressions[name] = _regression(
                            candidate_value,
                            baseline,
                            metric["objective"],
                        )
                    except ComparisonError:
                        regressions[name] = None
                        hard_failures.append(f"metric_invalid:{metric['name']}:{name}")
            effective = (
                max(value for value in regressions.values() if value is not None)
                if regressions and all(value is not None for value in regressions.values())
                else None
            )
            status = (
                "passed"
                if effective is not None and effective <= REGRESSION_LIMIT
                else "failed"
            )
            if status == "failed":
                hard_failures.append(f"metric_regression:{metric['name']}")
            metric_outcomes.append(
                {
                    "name": metric["name"],
                    "objective": metric["objective"],
                    "candidate": candidate_value,
                    "comparators": values,
                    "regressions": regressions,
                    "effective_regression": effective,
                    "limit": REGRESSION_LIMIT,
                    "status": status,
                }
            )
        status = "passed" if not hard_failures else "failed"
        any_failure = any_failure or status == "failed"
        outcomes.append(
            {
                "id": point_id,
                "status": status,
                "candidate_root": str(candidate_root),
                "hard_failures": sorted(set(hard_failures)),
                "metrics": metric_outcomes,
            }
        )
    if not outcomes:
        raise ComparisonError(f"check {check_id!r} declares no comparable metrics")
    payload = {
        "schema_version": 1,
        "check": check_id,
        "candidate": candidate_commit,
        "status": "failed" if any_failure else "passed",
        "regression_limit": REGRESSION_LIMIT,
        "points": outcomes,
    }
    return {**payload, "fingerprint": hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("check_id")
    args = parser.parse_args(argv)
    try:
        result = compare_manifest_check(args.manifest.resolve(), args.check_id)
    except ComparisonError as error:
        print(f"benchmark comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
