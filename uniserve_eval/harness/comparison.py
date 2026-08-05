"""Cross-runtime benchmark comparison over canonical point artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .text_parity import (
    evaluate_i2t_work_conformance,
    evaluate_text_canary,
    evaluate_text_work_conformance,
)

_IMAGE_TASKS = frozenset({"i2i", "t2i"})

_MetricSpec = tuple[str, tuple[str, ...], str]


def compare_pair(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    text_canary: bool = False,
    image_smoke: bool = False,
) -> dict[str, Any]:
    """Compare one reference/candidate pair through the benchmark artifact seam."""

    reference_path = Path(reference_directory)
    candidate_path = Path(candidate_directory)
    reference = _load_point(reference_path, expected_role="reference")
    candidate = _load_point(candidate_path, expected_role="candidate")
    failures: list[str] = []

    task = reference.get("task") if reference.get("task") == candidate.get("task") else None
    parity_group = (
        reference.get("parity_group")
        if reference.get("parity_group") == candidate.get("parity_group")
        else None
    )
    load_case = (
        reference.get("load_case")
        if reference.get("load_case") == candidate.get("load_case")
        else None
    )
    if task is None:
        failures.append("task_mismatch")
    if parity_group is None:
        failures.append("comparison_mismatch")
    if load_case is None:
        failures.append("load_case_mismatch")
    parity_fingerprint = (
        reference.get("parity_fingerprint")
        if reference.get("parity_fingerprint") == candidate.get("parity_fingerprint")
        and _is_fingerprint(reference.get("parity_fingerprint"))
        else None
    )
    if parity_fingerprint is None:
        failures.append("parity_contract_mismatch")
    if not reference.get("valid"):
        failures.append("reference_artifact_invalid")
    if not candidate.get("valid"):
        failures.append("candidate_artifact_invalid")

    work: dict[str, Any]
    canary: dict[str, Any] | None = None
    smoke: dict[str, Any] | None = None
    if task == "text":
        work_evidence = evaluate_text_work_conformance(
            reference_path / "requests.jsonl",
            candidate_path / "requests.jsonl",
        )
        work = {
            "passed": work_evidence.get("passed") is True,
            "checks": work_evidence.get("work_checks", {}),
            "mismatch_request_ids": work_evidence.get("mismatch_request_ids", []),
            "input_failures": work_evidence.get("input_failures", []),
        }
        if not work["passed"]:
            failures.append("work_mismatch")
        if text_canary:
            strict = evaluate_text_canary(
                reference_path / "requests.jsonl",
                candidate_path / "requests.jsonl",
            )
            canary = {
                "passed": strict.get("passed") is True,
                "checks": {
                    name: value
                    for name, value in strict.get("canary", {}).get("checks", {}).items()
                },
                "mismatch_request_ids": strict.get("canary", {}).get("mismatch_request_ids", []),
            }
    elif task == "i2t":
        ignore_eos = (
            reference.get("ignore_eos")
            if isinstance(reference.get("ignore_eos"), bool)
            and reference.get("ignore_eos") == candidate.get("ignore_eos")
            else None
        )
        if ignore_eos is None:
            failures.append("output_semantics_mismatch")
            work = {"passed": False, "checks": {}, "mismatch_request_ids": []}
        else:
            work_evidence = evaluate_i2t_work_conformance(
                reference_path / "requests.jsonl",
                candidate_path / "requests.jsonl",
                ignore_eos=ignore_eos,
            )
            work = {
                "passed": work_evidence.get("passed") is True,
                "checks": work_evidence.get("work_checks", {}),
                "mismatch_request_ids": work_evidence.get("mismatch_request_ids", []),
                "input_failures": work_evidence.get("input_failures", []),
            }
            if not work["passed"]:
                failures.append("work_mismatch")
    elif task in _IMAGE_TASKS:
        work = {
            "passed": bool(
                reference.get("generation_conformance") and candidate.get("generation_conformance")
            )
        }
        if not work["passed"]:
            failures.append("work_mismatch")
        if image_smoke:
            from .image_quality import evaluate_image_quality

            smoke_report = evaluate_image_quality(reference_path, candidate_path)
            smoke = {
                "evidence_valid": smoke_report.get("evidence_valid") is True,
                "canary_passed": smoke_report.get("regression_canary_passed") is True,
                "pair_count": smoke_report.get("pair_count"),
                "aggregate": smoke_report.get("aggregate"),
                "failures": smoke_report.get("failures", []),
            }
        if load_case not in {"c1", "c32"}:
            failures.append("unsupported_image_load_case")
    else:
        work = {"passed": False}
        failures.append("unsupported_task")

    metrics: list[dict[str, Any]] = []
    for metric_name, metric_path, objective in _comparison_metric_specs(task, load_case):
        reference_metric = _metric(reference, metric_path)
        candidate_metric = _metric(candidate, metric_path)
        ratio = None
        if reference_metric is None or candidate_metric is None or reference_metric <= 0.0:
            failures.append(f"metric_unavailable:{metric_name}")
        else:
            ratio = candidate_metric / reference_metric
        metrics.append(
            {
                "name": metric_name,
                "objective": objective,
                "reference": reference_metric,
                "candidate": candidate_metric,
                "candidate_over_reference": ratio,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 2,
        "comparison": parity_group,
        "task": task,
        "load_case": load_case,
        "valid": not failures,
        "work": work,
        "metrics": metrics,
        "reference": _point_result(reference),
        "candidate": _point_result(candidate),
        "failures": failures,
    }
    if canary is not None:
        result["text_canary"] = canary
    if smoke is not None:
        result["image_smoke"] = smoke
    return result


def summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine independently executed run summaries without imposing a repeat policy."""

    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for run in runs:
        for comparison in run.get("comparisons", []):
            name = comparison.get("comparison")
            load_case = comparison.get("load_case")
            metrics = comparison.get("metrics")
            if not (
                isinstance(name, str) and isinstance(load_case, str) and isinstance(metrics, list)
            ):
                continue
            for metric in metrics:
                metric_name = metric.get("name") if isinstance(metric, dict) else None
                if isinstance(metric_name, str):
                    grouped.setdefault((name, load_case, metric_name), []).append(
                        (comparison, metric)
                    )

    comparisons: list[dict[str, Any]] = []
    for (name, load_case, metric_name), values in sorted(grouped.items()):
        ratios = [
            float(metric["candidate_over_reference"])
            for comparison, metric in values
            if comparison.get("valid") is True
            and isinstance(metric.get("candidate_over_reference"), (int, float))
        ]
        comparisons.append(
            {
                "comparison": name,
                "load_case": load_case,
                "metric": metric_name,
                "objective": values[0][1].get("objective") if values else None,
                "run_count": len(values),
                "valid_run_count": len(ratios),
                "candidate_over_reference": ratios,
                "geometric_mean": (
                    math.exp(statistics.fmean(math.log(value) for value in ratios))
                    if ratios
                    else None
                ),
            }
        )
    return {
        "schema_version": 2,
        "run_count": len(runs),
        "comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact result view written by the single benchmark entry."""

    if "run_count" in report:
        lines = [
            "# Benchmark results",
            "",
            "| Comparison | Load case | Metric | Objective | Valid runs | Candidate/reference |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
        for comparison in report.get("comparisons", []):
            lines.append(
                f"| {comparison.get('comparison')} | "
                f"{comparison.get('load_case')} | {comparison.get('metric')} | "
                f"{comparison.get('objective')} | "
                f"{comparison.get('valid_run_count', 0)}/{comparison.get('run_count', 0)} | "
                f"{_number(comparison.get('geometric_mean'))} |"
            )
        return "\n".join(lines) + "\n"

    lines = ["# Benchmark results", ""]
    points = report.get("points", [])
    if points:
        lines.extend(
            [
                "| Point | Group | Task | Load case | Success | Elapsed (s) | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Images/s | Mean image latency (ms) | Mean transition latency (ms) | Text→image (ms) | Image→text (ms) |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for point in points:
            metrics = point.get("metrics", {})
            image_metrics = metrics.get("t2i", metrics)
            text_metrics = metrics.get("i2t", metrics)
            image_metrics = image_metrics if isinstance(image_metrics, dict) else {}
            text_metrics = text_metrics if isinstance(text_metrics, dict) else {}
            if isinstance(metrics.get("images"), dict):
                image_metrics = metrics["images"]
            image_latency = image_metrics.get("image_latency_ms", {})
            interleave = metrics.get("modality_interleave")
            transition_timing = (
                interleave.get("transition_timing") if isinstance(interleave, dict) else {}
            )
            transition_timing = transition_timing if isinstance(transition_timing, dict) else {}
            transition_latency = transition_timing.get("transition_latency_ms", {})
            text_to_image = transition_timing.get("text_to_image_transition_latency_ms", {})
            image_to_text = transition_timing.get("image_to_text_transition_latency_ms", {})
            lines.append(
                f"| {point.get('benchmark')} | {point.get('group')} | {point.get('task')} | "
                f"{point.get('load_case')} | {point.get('ok_count', 0)}/{point.get('request_count', 0)} | "
                f"{_number(point.get('elapsed_s'))} | {_number(text_metrics.get('output_throughput'))} | "
                f"{_number(text_metrics.get('mean_ttft_ms'))} | "
                f"{_number(text_metrics.get('mean_tpot_ms'))} | "
                f"{_number(image_metrics.get('images_per_second'))} | "
                f"{_number(image_latency.get('mean') if isinstance(image_latency, dict) else None)} | "
                f"{_number(transition_latency.get('mean') if isinstance(transition_latency, dict) else None)} | "
                f"{_number(text_to_image.get('mean') if isinstance(text_to_image, dict) else None)} | "
                f"{_number(image_to_text.get('mean') if isinstance(image_to_text, dict) else None)} |"
            )
        lines.extend(["", "## Paired comparisons", ""])
    lines.extend(
        [
            "| Comparison | Load case | Metric | Objective | Reference | Candidate | Candidate/reference | Work |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | :---: |",
        ]
    )
    for comparison in report.get("comparisons", []):
        for metric in comparison.get("metrics", []):
            lines.append(
                f"| {comparison.get('comparison')} | {comparison.get('load_case')} | "
                f"{metric.get('name')} | {metric.get('objective')} | "
                f"{_number(metric.get('reference'))} | {_number(metric.get('candidate'))} | "
                f"{_number(metric.get('candidate_over_reference'))} | "
                f"{'pass' if comparison.get('work', {}).get('passed') else 'fail'} |"
            )
        if "text_canary" in comparison:
            canary = comparison["text_canary"]
            lines.append(
                f"\nText canary for `{comparison.get('comparison')}` at load case "
                f"`{comparison.get('load_case')}`: "
                f"{'pass' if canary.get('passed') else 'fail'} "
                f"({len(canary.get('mismatch_request_ids', []))} mismatched requests)."
            )
        if "image_smoke" in comparison:
            smoke = comparison["image_smoke"]
            lines.append(
                f"\nImage smoke check for `{comparison.get('comparison')}` at load case "
                f"`{comparison.get('load_case')}`: "
                f"{'pass' if smoke.get('canary_passed') else 'fail'}."
            )
    return "\n".join(lines) + "\n"


def _load_point(directory: Path, *, expected_role: str) -> dict[str, Any]:
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"valid": False, "role": expected_role, "metrics": {}}
    artifact = summary.get("artifact") if isinstance(summary.get("artifact"), dict) else {}
    matrix = artifact.get("matrix_contract") if isinstance(artifact, dict) else None
    matrix = matrix if isinstance(matrix, dict) else {}
    parity = matrix.get("parity_contract")
    parity = parity if isinstance(parity, dict) else {}
    harness = parity.get("harness")
    harness = harness if isinstance(harness, dict) else {}
    spec = harness.get("spec")
    spec = spec if isinstance(spec, dict) else {}
    role = matrix.get("comparison_role")
    return {
        "valid": bool(
            artifact.get("valid") is True
            and artifact.get("valid_marker") == "canonical-valid-v4"
            and role == expected_role
        ),
        "role": role,
        "task": summary.get("task"),
        "parity_group": matrix.get("parity_group"),
        "load_case": (
            matrix.get("benchmark_definition", {}).get("load_case_id")
            if isinstance(matrix.get("benchmark_definition"), dict)
            else None
        ),
        "metrics": summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {},
        "request_count": summary.get("request_count"),
        "ok_count": summary.get("ok_count"),
        "failed_count": summary.get("failed_count"),
        "elapsed_s": summary.get("elapsed_s"),
        "generation_conformance": bool(
            isinstance(artifact.get("generation_conformance"), dict)
            and artifact["generation_conformance"].get("valid") is True
        ),
        "parity_fingerprint": parity.get("fingerprint"),
        "ignore_eos": spec.get("ignore_eos"),
    }


def _metric(point: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = point.get("metrics", {})
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _comparison_metric_specs(task: Any, load_case: Any) -> tuple[_MetricSpec, ...]:
    if task == "text":
        return (
            ("output_throughput", ("output_throughput",), "maximize"),
            ("mean_ttft_ms", ("mean_ttft_ms",), "minimize"),
            ("mean_tpot_ms", ("mean_tpot_ms",), "minimize"),
        )
    if task == "i2t":
        return (("output_throughput", ("output_throughput",), "maximize"),)
    if task in _IMAGE_TASKS and load_case == "c1":
        return (("image_latency_ms.mean", ("image_latency_ms", "mean"), "minimize"),)
    if task in _IMAGE_TASKS:
        return (("images_per_second", ("images_per_second",), "maximize"),)
    return ()


def _point_result(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_count": point.get("request_count"),
        "ok_count": point.get("ok_count"),
        "failed_count": point.get("failed_count"),
        "elapsed_s": point.get("elapsed_s"),
    }


def _is_fingerprint(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.6g}"
