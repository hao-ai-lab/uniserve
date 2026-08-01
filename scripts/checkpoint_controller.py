"""Decode-runtime checkpoint experiment controller.

This module applies the acceptance rules defined in
``specs/decode-runtime-construction.md`` and ``docs/benchmark-protocol.md`` to
ordinary ``uniserve_eval`` point artifacts. It consumes the ``summary.json`` and
``run.json`` bundles emitted by ``scripts/run_benchmarks.py`` and by
``uniserve-eval verify``, classifies each required performance metric against both
the immutable anchor and the immediately preceding accepted checkpoint, and emits an
acceptance verdict for a candidate source state.

The controller owns experiment-control policy only. It performs no measurement,
defines no metric, and never writes back into the evaluator. Measurement,
provenance capture, and artifact validity live in ``uniserve_eval``; the controller
reads those artifacts and decides whether a checkpoint may commit.

Classification (construction.md "Checkpoint performance protection"):

- ``regression`` for a maximize metric is ``max(0, 1 - candidate / baseline)``.
- ``regression`` for a minimize metric is ``max(0, candidate / baseline - 1)``.
- A metric's effective regression is the worse of its regression against the anchor
  and against the previous checkpoint.
- ``pass``: every required metric's effective regression is at most 20%.
- ``block``: any effective regression exceeds 20%, or any correctness, requested
  workload, output-validity, or provenance requirement fails.

The same performance limit applies to checkpoint triplets and UEval
major-boundary suites. Default travel is a correctness and provenance gate whose
elapsed time is retained for longitudinal diagnosis. Correctness, conformance,
requested-workload identity, and artifact-integrity requirements remain hard gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

REGRESSION_LIMIT = 0.20

MetricSpec = tuple[str, tuple[str, ...], str]

# Required performance metrics per point. Paths are keyed under ``summary.json``'s
# ``metrics`` object. Objective is "maximize" (larger is better) or "minimize".
TRIPLET_SPECS: dict[str, tuple[MetricSpec, ...]] = {
    "text": (
        ("output_throughput", ("output_throughput",), "maximize"),
        ("mean_ttft_ms", ("mean_ttft_ms",), "minimize"),
        ("mean_tpot_ms", ("mean_tpot_ms",), "minimize"),
    ),
    "t2i": (("images_per_second", ("images_per_second",), "maximize"),),
    "i2t": (("output_throughput", ("output_throughput",), "maximize"),),
}

# UEval interleave latency families (major boundary). Directional transition means
# are only classified when the baseline sample count is nonzero.
_TRANSITION = ("modality_interleave", "transition_timing")
UEVAL_SPECS: tuple[MetricSpec, ...] = (
    ("ttft_mean_ms", ("ttft_ms", "mean"), "minimize"),
    ("tpot_mean_ms", ("tpot_ms", "mean"), "minimize"),
    ("image_latency_mean_ms", ("images", "image_latency_ms", "mean"), "minimize"),
    ("transition_latency_mean_ms", (*_TRANSITION, "transition_latency_ms", "mean"), "minimize"),
)
UEVAL_DIRECTIONAL_SPECS: tuple[tuple[MetricSpec, tuple[str, ...]], ...] = (
    (
        (
            "text_to_image_transition_mean_ms",
            (*_TRANSITION, "text_to_image_transition_latency_ms", "mean"),
            "minimize",
        ),
        (*_TRANSITION, "text_to_image_transition_latency_ms", "count"),
    ),
    (
        (
            "image_to_text_transition_mean_ms",
            (*_TRANSITION, "image_to_text_transition_latency_ms", "mean"),
            "minimize",
        ),
        (*_TRANSITION, "image_to_text_transition_latency_ms", "count"),
    ),
)

MAJOR_INTERLEAVE_CHECKPOINTS = frozenset({"cp5", "cp7", "cp8"})
MAJOR_INTERLEAVE_PREVIOUS_REQUIRED = frozenset({"cp7", "cp8"})


@dataclass
class Point:
    """One loaded evaluation point (summary.json + run.json)."""

    directory: Path
    task: str | None
    load_case: str | None
    metrics: dict[str, Any]
    request_count: int | None
    ok_count: int | None
    failed_count: int | None
    elapsed_s: float | None
    artifact_valid: bool
    checks: dict[str, Any]
    generation_conformance_valid: bool
    interleave_latency: dict[str, Any] | None
    workload_contract: dict[str, Any]
    provenance: dict[str, Any]

    def metric(self, path: tuple[str, ...]) -> float | None:
        if path == ("__elapsed_s__",):
            return _finite(self.elapsed_s)
        value: Any = self.metrics
        for part in path:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return _finite(value)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mapping(value: Any) -> dict[str, Any]:
    """Narrow an untyped JSON value to the object shape used by this controller."""

    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def regression(candidate: float | None, baseline: float | None, objective: str) -> float | None:
    """Regression of a candidate metric against a baseline for the given objective.

    Returns ``None`` when either value is missing or the baseline is non-positive,
    which the caller treats as an unavailable required metric (a hard block).
    """

    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    if objective == "maximize":
        return max(0.0, 1.0 - candidate / baseline)
    if objective == "minimize":
        return max(0.0, candidate / baseline - 1.0)
    raise ValueError(f"unknown objective {objective!r}")


def _band(reg: float | None) -> str:
    if reg is None:
        return "unavailable"
    if reg <= REGRESSION_LIMIT:
        return "pass"
    return "block"


@dataclass
class MetricOutcome:
    name: str
    objective: str
    candidate: float | None
    anchor: float | None
    previous: float | None
    regression_anchor: float | None
    regression_previous: float | None
    effective_regression: float | None
    band: str


def evaluate_metric(
    spec: MetricSpec,
    candidate: Point,
    anchor: Point,
    previous: Point | None,
) -> MetricOutcome:
    name, path, objective = spec
    cand = candidate.metric(path)
    anc = anchor.metric(path)
    prev = previous.metric(path) if previous is not None else None
    reg_anchor = regression(cand, anc, objective)
    reg_prev = regression(cand, prev, objective) if previous is not None else None

    regressions = [r for r in (reg_anchor, reg_prev) if r is not None]
    # A required metric with no usable baseline value is unavailable -> block.
    required_baselines = 1 + (1 if previous is not None else 0)
    if len(regressions) < required_baselines or cand is None:
        effective: float | None = None
    else:
        effective = max(regressions)
    return MetricOutcome(
        name=name,
        objective=objective,
        candidate=cand,
        anchor=anc,
        previous=prev,
        regression_anchor=reg_anchor,
        regression_previous=reg_prev,
        effective_regression=effective,
        band=_band(effective),
    )


@dataclass
class PointOutcome:
    task: str | None
    directory: str
    metrics: list[MetricOutcome]
    hard_failures: list[str]


def hard_gate_failures(
    candidate: Point,
    anchor: Point | None,
    previous: Point | None = None,
    *,
    interleave: bool,
) -> list[str]:
    """Correctness, requested-workload, output-validity, and provenance gates."""

    failures: list[str] = []
    if candidate.failed_count not in (0, None) and candidate.failed_count != 0:
        failures.append(f"failed_requests={candidate.failed_count}")
    if (
        candidate.request_count is not None
        and candidate.ok_count is not None
        and candidate.ok_count != candidate.request_count
    ):
        failures.append(f"incomplete_success={candidate.ok_count}/{candidate.request_count}")
    if not candidate.artifact_valid:
        failures.append("artifact_invalid")
    for name, value in candidate.checks.items():
        if value is not True:
            failures.append(f"check:{name}")
    if not candidate.generation_conformance_valid:
        failures.append("generation_conformance_invalid")
    for key in ("source_revision", "model", "dataset_revision"):
        if not candidate.provenance.get(key):
            failures.append(f"provenance_missing:{key}")
    if candidate.provenance.get("source_dirty") is not False:
        failures.append("source_tree_dirty")

    if not candidate.workload_contract:
        failures.append("workload_contract_missing")
    for label, baseline in (("anchor", anchor), ("previous", previous)):
        if baseline is None:
            continue
        if not baseline.workload_contract:
            failures.append(f"{label}_workload_contract_missing")
        elif candidate.workload_contract != baseline.workload_contract:
            failures.append(f"workload_contract_mismatch:{label}")

    if interleave:
        if candidate.request_count != 32 or candidate.ok_count != 32:
            failures.append(f"interleave_success={candidate.ok_count}/{candidate.request_count}")
        il = candidate.interleave_latency or {}
        if il.get("valid") is not True:
            failures.append("interleave_latency_invalid")
        timing = (candidate.metrics.get("modality_interleave") or {}).get("transition_timing") or {}
        if timing.get("timestamp_coverage") != 1.0:
            failures.append(f"timestamp_coverage={timing.get('timestamp_coverage')}")
        if timing.get("transition_sample_coverage") != 1.0:
            failures.append(
                f"transition_sample_coverage={timing.get('transition_sample_coverage')}"
            )
    return failures


def _workload_contract(artifact: dict[str, Any]) -> dict[str, Any]:
    """Requested inputs and quality/load controls that make two points comparable."""

    contract = _mapping(artifact.get("contract"))
    request_spec = _mapping(contract.get("spec"))
    selected_rows = _mapping(contract.get("selected_rows"))
    matrix = _mapping(artifact.get("matrix_contract"))
    definition = _mapping(matrix.get("benchmark_definition"))
    if not request_spec:
        return {}
    normalized_spec = dict(request_spec)
    # The selected-row digest and dataset revision bind dataset content. The local
    # extraction directory is execution-environment provenance, not request work.
    normalized_spec.pop("dataset_path", None)
    values = {
        "request_spec_sha256": hashlib.sha256(
            json.dumps(
                normalized_spec,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "selected_row_count": selected_rows.get("count"),
        "selected_rows_sha256": selected_rows.get("sha256"),
        "benchmark_definition_fingerprint": definition.get("fingerprint"),
        "load_case_id": definition.get("load_case_id"),
    }
    return values if all(value is not None and value != "" for value in values.values()) else {}


def classify(metric_outcomes: Sequence[MetricOutcome]) -> str:
    bands = [m.band for m in metric_outcomes]
    if any(b in ("block", "unavailable") for b in bands):
        return "block"
    return "pass"


def specs_for(point: Point, *, major_boundary: bool) -> tuple[MetricSpec, ...]:
    if point.task in TRIPLET_SPECS:
        return TRIPLET_SPECS[point.task]
    if point.task == "interleave" and major_boundary:
        specs = list(UEVAL_SPECS)
        return tuple(specs)
    return ()


def load_point(directory: str | Path) -> Point:
    directory = Path(directory)
    summary = _mapping(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
    run: dict[str, Any] = {}
    run_path = directory / "run.json"
    if run_path.is_file():
        try:
            run = _mapping(json.loads(run_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            run = {}
    artifact = _mapping(summary.get("artifact"))
    checks = _mapping(artifact.get("checks"))
    gen = _mapping(artifact.get("generation_conformance"))
    il = _mapping(artifact.get("interleave_latency_conformance"))
    metrics = _mapping(summary.get("metrics"))
    matrix = _mapping(artifact.get("matrix_contract"))
    bench_def = _mapping(matrix.get("benchmark_definition"))
    return Point(
        directory=directory,
        task=summary.get("task"),
        load_case=bench_def.get("load_case_id"),
        metrics=metrics,
        request_count=summary.get("request_count"),
        ok_count=summary.get("ok_count"),
        failed_count=summary.get("failed_count"),
        elapsed_s=summary.get("elapsed_s"),
        artifact_valid=bool(
            artifact.get("valid") is True and artifact.get("valid_marker") == "canonical-valid-v2"
        ),
        checks=checks,
        generation_conformance_valid=bool(gen.get("valid") is True),
        interleave_latency=il or None,
        workload_contract=_workload_contract(artifact),
        provenance=_provenance(summary, run, artifact),
    )


def _source_state(matrix: dict[str, Any]) -> dict[str, Any]:
    """Workspace source-tree state, present as a formal build manifest or, for a
    non-formal run, as the workspace entry of an execution's source-revision list."""

    execution_policy = _mapping(matrix.get("execution_policy"))
    build_manifest = _mapping(execution_policy.get("build_manifest"))
    source_state = _mapping(build_manifest.get("source_state"))
    if source_state:
        return source_state
    for execution_key in ("harness_execution", "server_execution"):
        execution = matrix.get(execution_key)
        revisions = execution.get("source_revisions") if isinstance(execution, dict) else None
        if not isinstance(revisions, list):
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                continue
            state = _mapping(revision.get("state"))
            if state and "workspace" in (revision.get("roles") or []):
                return state
        for revision in revisions:
            if isinstance(revision, dict):
                state = _mapping(revision.get("state"))
                if state:
                    return state
    return {}


def _provenance(
    summary: dict[str, Any], run: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    matrix = _mapping(artifact.get("matrix_contract"))
    source_state = _source_state(matrix)
    model_contract = _mapping(matrix.get("model_revision_contract"))
    contract_spec = _mapping(_mapping(artifact.get("contract")).get("spec"))
    dataset_revision = (
        run.get("dataset_revision")
        or contract_spec.get("dataset_revision")
        or run.get("t2i_dataset_revision")
        or run.get("i2t_dataset_revision")
    )
    return {
        "source_revision": source_state.get("head"),
        "source_dirty": source_state.get("dirty"),
        "tracked_changes_sha256": source_state.get("tracked_changes_sha256"),
        "untracked_files_sha256": source_state.get("untracked_files_sha256"),
        "model": run.get("model") or summary.get("model"),
        "model_revision": model_contract.get("revision"),
        "dataset_revision": dataset_revision,
        "sampling_seed": run.get("sampling_seed"),
        "server_topology": run.get("server_topology"),
        "runtime_profile_id": run.get("runtime_profile_id"),
    }


def discover_points(root: str | Path) -> dict[str, Point]:
    """Index points under a benchmark output root by task."""

    root = Path(root)
    points: dict[str, Point] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            point = load_point(summary_path.parent)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if point.task:
            points[point.task] = point
    return points


@dataclass
class CheckpointReport:
    checkpoint: str
    major_boundary: bool
    source_revision: str | None = None
    points: list[PointOutcome] = field(default_factory=list)
    verdict: str = "block"


def load_default_travel(
    directory: str | Path,
) -> tuple[float | None, dict[str, Any], dict[str, Any]]:
    """Load the fixed default-travel verification artifact.

    This point is emitted by ``uniserve-eval verify`` rather than the serving
    benchmark harness, so its schema is intentionally handled at this boundary.
    """

    summary = _mapping(json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8")))
    artifact = _mapping(summary.get("artifact"))
    provenance = _mapping(artifact.get("provenance"))
    source_state = _mapping(provenance.get("source_state"))
    return _finite(summary.get("elapsed_s")), artifact, source_state


def default_travel_failures(
    directory: str | Path,
    *,
    expected_source_revision: str | None,
) -> tuple[float | None, list[str]]:
    failures: list[str] = []
    try:
        elapsed_s, artifact, source_state = load_default_travel(directory)
    except (OSError, json.JSONDecodeError):
        return None, ["point_missing"]
    if artifact.get("valid") is not True or artifact.get("valid_marker") != "verify-valid-v1":
        failures.append("artifact_invalid")
    checks = _mapping(artifact.get("checks"))
    if not checks:
        failures.append("checks_missing")
    else:
        failures.extend(f"check:{name}" for name, value in checks.items() if value is not True)
    provenance = _mapping(artifact.get("provenance"))
    profile = _mapping(provenance.get("profile_contract"))
    if profile.get("workload") != "gate/sensenova/default-travel":
        failures.append("profile_mismatch")
    source_revision = source_state.get("head")
    if not source_revision:
        failures.append("provenance_missing:source_revision")
    if source_state.get("dirty") is not False:
        failures.append("source_tree_dirty")
    if expected_source_revision is not None and source_revision != expected_source_revision:
        failures.append("source_revision_mismatch")
    if elapsed_s is None or elapsed_s <= 0.0:
        failures.append("elapsed_s_invalid")
    return elapsed_s, failures


def evaluate_checkpoint(
    candidate_root: str | Path,
    anchor_root: str | Path,
    previous_root: str | Path | None,
    *,
    checkpoint: str,
    major_boundary: bool,
    default_travel_dir: str | Path | None = None,
    candidate_major_root: str | Path | None = None,
    anchor_major_root: str | Path | None = None,
    previous_major_root: str | Path | None = None,
) -> CheckpointReport:
    candidate_points = discover_points(candidate_root)
    candidate_major_points = discover_points(candidate_major_root) if candidate_major_root else {}
    if "interleave" in candidate_major_points:
        candidate_points["interleave"] = candidate_major_points["interleave"]
    anchor_points = discover_points(anchor_root)
    anchor_major_points = discover_points(anchor_major_root) if anchor_major_root else {}
    if "interleave" in anchor_major_points:
        anchor_points["interleave"] = anchor_major_points["interleave"]
    previous_points = discover_points(previous_root) if previous_root else {}
    previous_major_points = discover_points(previous_major_root) if previous_major_root else {}

    report = CheckpointReport(checkpoint=checkpoint, major_boundary=major_boundary)
    require_interleave = major_boundary and checkpoint in MAJOR_INTERLEAVE_CHECKPOINTS
    required_tasks = ["text", "t2i", "i2t"] + (["interleave"] if require_interleave else [])

    all_metric_outcomes: list[MetricOutcome] = []
    any_hard_failure = False

    for task in required_tasks:
        cand = candidate_points.get(task)
        anc = anchor_points.get(task)
        directory = str(cand.directory) if cand else "<missing>"
        if cand is None or anc is None:
            report.points.append(
                PointOutcome(
                    task=task, directory=directory, metrics=[], hard_failures=["point_missing"]
                )
            )
            any_hard_failure = True
            continue
        prev = (
            previous_major_points.get(task) if task == "interleave" else previous_points.get(task)
        )
        specs = specs_for(cand, major_boundary=major_boundary)
        outcomes = [evaluate_metric(spec, cand, anc, prev) for spec in specs]
        if task == "interleave":
            for spec, count_path in UEVAL_DIRECTIONAL_SPECS:
                baseline_count = anc.metric(count_path)
                if baseline_count and baseline_count > 0:
                    outcomes.append(evaluate_metric(spec, cand, anc, prev))
        hard = hard_gate_failures(cand, anc, prev, interleave=(task == "interleave"))
        if (
            task == "interleave"
            and checkpoint in MAJOR_INTERLEAVE_PREVIOUS_REQUIRED
            and prev is None
        ):
            hard.append("previous_major_point_missing")
        if hard:
            any_hard_failure = True
        all_metric_outcomes.extend(outcomes)
        report.points.append(
            PointOutcome(task=task, directory=directory, metrics=outcomes, hard_failures=hard)
        )

    candidate_revisions = {
        point.provenance.get("source_revision")
        for task, point in candidate_points.items()
        if task in required_tasks and point.provenance.get("source_revision")
    }
    if len(candidate_revisions) == 1:
        report.source_revision = next(iter(candidate_revisions))
    else:
        any_hard_failure = True
        for point in report.points:
            point.hard_failures.append("candidate_source_revision_mismatch")

    if major_boundary:
        if default_travel_dir is None:
            elapsed_s, hard = None, ["point_missing"]
            directory = "<missing>"
        else:
            directory = str(default_travel_dir)
            elapsed_s, hard = default_travel_failures(
                default_travel_dir,
                expected_source_revision=report.source_revision,
            )
        if hard:
            any_hard_failure = True
        report.points.append(
            PointOutcome(
                task="default_travel",
                directory=directory,
                metrics=[
                    MetricOutcome(
                        name="elapsed_s",
                        objective="record",
                        candidate=elapsed_s,
                        anchor=None,
                        previous=None,
                        regression_anchor=None,
                        regression_previous=None,
                        effective_regression=0.0 if elapsed_s is not None else None,
                        band="pass" if elapsed_s is not None and not hard else "unavailable",
                    )
                ],
                hard_failures=hard,
            )
        )

    perf_verdict = classify(all_metric_outcomes)
    report.verdict = "block" if any_hard_failure else perf_verdict
    return report


def render_report(report: CheckpointReport) -> str:
    lines = [
        f"# Checkpoint {report.checkpoint} acceptance",
        "",
        f"- Boundary: {'major-integration' if report.major_boundary else 'checkpoint-triplet'}",
        f"- Maximum performance regression: {REGRESSION_LIMIT * 100:.0f}%",
        f"- **Verdict: {report.verdict.upper()}**",
        "",
        "| Point | Metric | Objective | Candidate | Anchor | Prev | Reg vs anchor | Reg vs prev | Effective | Band |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for point in report.points:
        if not point.metrics:
            lines.append(
                f"| {point.task} | — | — | — | — | — | — | — | — | {'; '.join(point.hard_failures) or '—'} |"
            )
        for m in point.metrics:
            lines.append(
                f"| {point.task} | {m.name} | {m.objective} | {_n(m.candidate)} | {_n(m.anchor)} | "
                f"{_n(m.previous)} | {_pct(m.regression_anchor)} | {_pct(m.regression_previous)} | "
                f"{_pct(m.effective_regression)} | {m.band} |"
            )
    hard = [f"{p.task}: {', '.join(p.hard_failures)}" for p in report.points if p.hard_failures]
    if hard:
        lines.extend(["", "## Hard-gate failures", ""])
        lines.extend(f"- {h}" for h in hard)
    return "\n".join(lines) + "\n"


def _n(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.6g}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _report_to_dict(report: CheckpointReport) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "checkpoint": report.checkpoint,
        "major_boundary": report.major_boundary,
        "source_revision": report.source_revision,
        "verdict": report.verdict,
        "regression_limit": REGRESSION_LIMIT,
        "points": [
            {
                "task": p.task,
                "directory": p.directory,
                "hard_failures": p.hard_failures,
                "metrics": [
                    {
                        "name": m.name,
                        "objective": m.objective,
                        "candidate": m.candidate,
                        "anchor": m.anchor,
                        "previous": m.previous,
                        "regression_anchor": m.regression_anchor,
                        "regression_previous": m.regression_previous,
                        "effective_regression": m.effective_regression,
                        "band": m.band,
                    }
                    for m in p.metrics
                ],
            }
            for p in report.points
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode-runtime checkpoint acceptance controller")
    parser.add_argument("--checkpoint", required=True, help="checkpoint identifier, e.g. cp1")
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--previous-root", default=None)
    parser.add_argument(
        "--candidate-major-root",
        default=None,
        help="candidate major-boundary artifact root when it is separate from the triplet root",
    )
    parser.add_argument(
        "--anchor-major-root",
        default=None,
        help="immutable major-boundary anchor root when it is separate from the triplet anchor",
    )
    parser.add_argument(
        "--previous-major-root",
        default=None,
        help="previous accepted major-boundary artifact root for UEval comparison",
    )
    parser.add_argument(
        "--major-boundary",
        action="store_true",
        help="require default travel and the checkpoint's declared integration-boundary points",
    )
    parser.add_argument(
        "--default-travel-dir",
        default=None,
        help="default-travel verification artifact required at a major boundary",
    )
    parser.add_argument(
        "--out", default=None, help="directory to write acceptance.json/acceptance.md"
    )
    args = parser.parse_args(argv)

    report = evaluate_checkpoint(
        args.candidate_root,
        args.anchor_root,
        args.previous_root,
        checkpoint=args.checkpoint,
        major_boundary=args.major_boundary,
        default_travel_dir=args.default_travel_dir,
        candidate_major_root=args.candidate_major_root,
        anchor_major_root=args.anchor_major_root,
        previous_major_root=args.previous_major_root,
    )
    markdown = render_report(report)
    print(markdown)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "acceptance.json").write_text(
            json.dumps(_report_to_dict(report), indent=2), encoding="utf-8"
        )
        (out / "acceptance.md").write_text(markdown, encoding="utf-8")
    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
