"""Offline comparison of matched benchmark result bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


def compare_pair(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    max_regression: float,
) -> dict[str, Any]:
    reference = _load_summary(reference_directory)
    candidate = _load_summary(candidate_directory)
    failures: list[str] = []
    for field in ("benchmark", "task", "workload", "selected_rows", "metric_definitions"):
        if reference.get(field) != candidate.get(field):
            failures.append(f"{field}_mismatch")
    if reference.get("validation", {}).get("valid") is not True:
        failures.append("reference_invalid")
    if candidate.get("validation", {}).get("valid") is not True:
        failures.append("candidate_invalid")

    metrics: list[dict[str, Any]] = []
    for definition in reference.get("metric_definitions", []):
        path = str(definition.get("path", ""))
        direction = definition.get("direction")
        ref_value = _metric(reference.get("metrics", {}), path)
        cand_value = _metric(candidate.get("metrics", {}), path)
        normalized_ratio = None
        raw_change_percent = None
        passed = False
        if ref_value is None or cand_value is None or ref_value <= 0 or cand_value <= 0:
            failures.append(f"metric_unavailable:{path}")
        else:
            raw_change_percent = (cand_value / ref_value - 1.0) * 100.0
            normalized_ratio = (
                cand_value / ref_value if direction == "higher" else ref_value / cand_value
            )
            passed = normalized_ratio >= 1.0 - max_regression
        metrics.append(
            {
                "path": path,
                "direction": direction,
                "reference": ref_value,
                "candidate": cand_value,
                "raw_change_percent": raw_change_percent,
                "normalized_ratio": normalized_ratio,
                "minimum_ratio": 1.0 - max_regression,
                "passed": passed,
            }
        )
    comparable = not failures
    return {
        "benchmark": reference.get("benchmark") or candidate.get("benchmark"),
        "comparable": comparable,
        "passed": comparable and bool(metrics) and all(metric["passed"] for metric in metrics),
        "metrics": metrics,
        "failures": failures,
        "warnings": {
            "reference": reference.get("warnings", []),
            "candidate": candidate.get("warnings", []),
        },
    }


def compare_suite(
    reference_root: str | Path,
    candidate_root: str | Path,
    points: Sequence[str],
    *,
    max_regression: float,
) -> dict[str, Any]:
    reference_root = Path(reference_root)
    candidate_root = Path(candidate_root)
    comparisons = [
        compare_pair(
            reference_root / point,
            candidate_root / point,
            max_regression=max_regression,
        )
        for point in points
    ]
    return {
        "max_regression": max_regression,
        "valid": all(comparison["comparable"] for comparison in comparisons),
        "passed": bool(comparisons) and all(comparison["passed"] for comparison in comparisons),
        "comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark comparison",
        "",
        f"Overall: {'pass' if report.get('passed') else 'fail'}",
        "",
        "| Benchmark | Metric | Reference | Candidate | Change | Ratio | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in report.get("comparisons", []):
        if not comparison.get("metrics"):
            lines.append(
                f"| {comparison.get('benchmark')} | n/a | n/a | n/a | n/a | n/a | fail |"
            )
        for metric in comparison.get("metrics", []):
            lines.append(
                "| {benchmark} | `{path}` | {reference} | {candidate} | {change} | {ratio} | {result} |".format(
                    benchmark=comparison.get("benchmark"),
                    path=metric.get("path"),
                    reference=_number(metric.get("reference")),
                    candidate=_number(metric.get("candidate")),
                    change=_percent(metric.get("raw_change_percent")),
                    ratio=_number(metric.get("normalized_ratio")),
                    result="pass" if metric.get("passed") else "fail",
                )
            )
    return "\n".join(lines) + "\n"


def _load_summary(directory: str | Path) -> dict[str, Any]:
    try:
        value = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"validation": {"valid": False}, "warnings": []}
    return value if isinstance(value, dict) else {"validation": {"valid": False}}


def _metric(metrics: Any, path: str) -> float | None:
    value = metrics
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"
