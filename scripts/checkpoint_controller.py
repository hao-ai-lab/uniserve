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
- ``block``: any effective regression exceeds 20%, or any correctness, work, output
  validity, or provenance requirement fails.

The same performance limit applies to checkpoint triplets and major-boundary suites.
Correctness, conformance, work, and artifact-integrity requirements remain hard gates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        ("text_to_image_transition_mean_ms", (*_TRANSITION, "text_to_image_transition_latency_ms", "mean"), "minimize"),
        (*_TRANSITION, "text_to_image_transition_latency_ms", "count"),
    ),
    (
        ("image_to_text_transition_mean_ms", (*_TRANSITION, "image_to_text_transition_latency_ms", "mean"), "minimize"),
        (*_TRANSITION, "image_to_text_transition_latency_ms", "count"),
    ),
)

# Default-travel gate: elapsed wall time (minimize). Step conformance is a hard gate.
DEFAULT_TRAVEL_SPECS: tuple[MetricSpec, ...] = (("elapsed_s", ("__elapsed_s__",), "minimize"),)


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
    request_signatures: dict[str, Any]
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
    *,
    interleave: bool,
) -> list[str]:
    """Correctness, work, output-validity, and provenance hard requirements."""

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
    if candidate.provenance.get("source_dirty") is True:
        failures.append("source_tree_dirty")

    if interleave:
        il = candidate.interleave_latency or {}
        if il.get("valid") is not True:
            failures.append("interleave_latency_invalid")
        timing = (candidate.metrics.get("modality_interleave") or {}).get("transition_timing") or {}
        if timing.get("timestamp_coverage") != 1.0:
            failures.append(f"timestamp_coverage={timing.get('timestamp_coverage')}")
        if timing.get("transition_sample_coverage") != 1.0:
            failures.append(f"transition_sample_coverage={timing.get('transition_sample_coverage')}")
        if anchor is not None:
            if candidate.request_signatures != anchor.request_signatures:
                failures.append("modality_signature_mismatch")
            if _image_step_work(candidate) != _image_step_work(anchor):
                failures.append("work_signature_mismatch")
    return failures


def _image_step_work(point: Point) -> dict[str, Any]:
    """Per-request decoded-image and image-step work, used for major-boundary equality."""

    timing = (point.metrics.get("modality_interleave") or {}).get("transition_timing") or {}
    return {"signatures": point.request_signatures, "expected_transitions": timing.get("expected_transition_count")}


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
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    run = {}
    run_path = directory / "run.json"
    if run_path.is_file():
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run = {}
    artifact = summary.get("artifact") if isinstance(summary.get("artifact"), dict) else {}
    checks = artifact.get("checks") if isinstance(artifact.get("checks"), dict) else {}
    gen = artifact.get("generation_conformance")
    il = artifact.get("interleave_latency_conformance")
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    timing = (metrics.get("modality_interleave") or {}).get("transition_timing") or {}
    matrix = artifact.get("matrix_contract") if isinstance(artifact.get("matrix_contract"), dict) else {}
    bench_def = matrix.get("benchmark_definition") if isinstance(matrix.get("benchmark_definition"), dict) else {}
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
        generation_conformance_valid=bool(isinstance(gen, dict) and gen.get("valid") is True),
        interleave_latency=il if isinstance(il, dict) else None,
        request_signatures=timing.get("request_signatures") if isinstance(timing.get("request_signatures"), dict) else {},
        provenance=_provenance(summary, run, artifact),
    )


def _source_state(matrix: dict[str, Any]) -> dict[str, Any]:
    """Workspace source-tree state, present as a formal build manifest or, for a
    non-formal run, as the workspace entry of an execution's source-revision list."""

    execution_policy = matrix.get("execution_policy") if isinstance(matrix.get("execution_policy"), dict) else {}
    build_manifest = execution_policy.get("build_manifest") if isinstance(execution_policy.get("build_manifest"), dict) else {}
    if isinstance(build_manifest.get("source_state"), dict):
        return build_manifest["source_state"]
    for execution_key in ("harness_execution", "server_execution"):
        execution = matrix.get(execution_key)
        revisions = execution.get("source_revisions") if isinstance(execution, dict) else None
        if not isinstance(revisions, list):
            continue
        for revision in revisions:
            if not isinstance(revision, dict):
                continue
            state = revision.get("state")
            if isinstance(state, dict) and "workspace" in (revision.get("roles") or []):
                return state
        for revision in revisions:
            if isinstance(revision, dict) and isinstance(revision.get("state"), dict):
                return revision["state"]
    return {}


def _provenance(summary: dict[str, Any], run: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    matrix = artifact.get("matrix_contract") if isinstance(artifact.get("matrix_contract"), dict) else {}
    source_state = _source_state(matrix)
    model_contract = matrix.get("model_revision_contract") if isinstance(matrix.get("model_revision_contract"), dict) else {}
    contract_spec = (artifact.get("contract") or {}).get("spec") if isinstance(artifact.get("contract"), dict) else {}
    dataset_revision = (
        run.get("dataset_revision")
        or (contract_spec.get("dataset_revision") if isinstance(contract_spec, dict) else None)
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
    points: list[PointOutcome] = field(default_factory=list)
    verdict: str = "block"


def evaluate_checkpoint(
    candidate_root: str | Path,
    anchor_root: str | Path,
    previous_root: str | Path | None,
    *,
    checkpoint: str,
    major_boundary: bool,
) -> CheckpointReport:
    candidate_points = discover_points(candidate_root)
    anchor_points = discover_points(anchor_root)
    previous_points = discover_points(previous_root) if previous_root else {}

    report = CheckpointReport(checkpoint=checkpoint, major_boundary=major_boundary)
    required_tasks = ["text", "t2i", "i2t"] + (["interleave"] if major_boundary else [])

    all_metric_outcomes: list[MetricOutcome] = []
    any_hard_failure = False

    for task in required_tasks:
        cand = candidate_points.get(task)
        anc = anchor_points.get(task)
        directory = str(cand.directory) if cand else "<missing>"
        if cand is None or anc is None:
            report.points.append(
                PointOutcome(task=task, directory=directory, metrics=[], hard_failures=["point_missing"])
            )
            any_hard_failure = True
            continue
        prev = previous_points.get(task)
        specs = specs_for(cand, major_boundary=major_boundary)
        outcomes = [evaluate_metric(spec, cand, anc, prev) for spec in specs]
        if task == "interleave":
            for spec, count_path in UEVAL_DIRECTIONAL_SPECS:
                baseline_count = anc.metric(count_path)
                if baseline_count and baseline_count > 0:
                    outcomes.append(evaluate_metric(spec, cand, anc, prev))
        hard = hard_gate_failures(cand, anc, interleave=(task == "interleave"))
        if hard:
            any_hard_failure = True
        all_metric_outcomes.extend(outcomes)
        report.points.append(
            PointOutcome(task=task, directory=directory, metrics=outcomes, hard_failures=hard)
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
            lines.append(f"| {point.task} | — | — | — | — | — | — | — | — | {'; '.join(point.hard_failures) or '—'} |")
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
        "schema_version": 2,
        "checkpoint": report.checkpoint,
        "major_boundary": report.major_boundary,
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
        "--major-boundary",
        action="store_true",
        help="also require the UEval interleave latency families (checkpoints 3, 5, 7, 8)",
    )
    parser.add_argument("--out", default=None, help="directory to write acceptance.json/acceptance.md")
    args = parser.parse_args(argv)

    report = evaluate_checkpoint(
        args.candidate_root,
        args.anchor_root,
        args.previous_root,
        checkpoint=args.checkpoint,
        major_boundary=args.major_boundary,
    )
    markdown = render_report(report)
    print(markdown)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "acceptance.json").write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
        (out / "acceptance.md").write_text(markdown, encoding="utf-8")
    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
