#!/usr/bin/env python3
"""Compare matched serving result bundles offline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from uniserve_eval.config import DEFAULT_CONFIG, load_config


def compare_pair(
    reference_directory: str | Path,
    candidate_directory: str | Path,
    *,
    max_regression: float | None = None,
    max_latency_regression_ms: float | None = None,
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
        raw_change_percent = None
        if ref_value is None or cand_value is None or ref_value <= 0 or cand_value <= 0:
            failures.append(f"metric_unavailable:{path}")
        else:
            raw_change_percent = (cand_value / ref_value - 1.0) * 100.0
        metric = {
            "path": path,
            "direction": direction,
            "reference": ref_value,
            "candidate": cand_value,
            "raw_change_percent": raw_change_percent,
        }
        if max_regression is not None:
            normalized_ratio = None
            passed = False
            if ref_value is not None and cand_value is not None and ref_value > 0 and cand_value > 0:
                normalized_ratio = (
                    cand_value / ref_value if direction == "higher" else ref_value / cand_value
                )
                passed = normalized_ratio >= 1.0 - max_regression
            metric.update(
                {
                    "normalized_ratio": normalized_ratio,
                    "minimum_ratio": 1.0 - max_regression,
                    "passed": passed,
                }
            )
        elif max_latency_regression_ms is not None and _is_latency_metric(path, direction):
            regression_ms = None
            passed = False
            if ref_value is not None and cand_value is not None:
                regression_ms = cand_value - ref_value
                passed = regression_ms <= max_latency_regression_ms
            metric.update(
                {
                    "regression_ms": regression_ms,
                    "maximum_regression_ms": max_latency_regression_ms,
                    "passed": passed,
                }
            )
        metrics.append(metric)
    comparable = not failures
    result = {
        "benchmark": reference.get("benchmark") or candidate.get("benchmark"),
        "comparable": comparable,
        "metrics": metrics,
        "failures": failures,
        "warnings": {
            "reference": reference.get("warnings", []),
            "candidate": candidate.get("warnings", []),
        },
    }
    if max_regression is not None or max_latency_regression_ms is not None:
        screened_metrics = [metric for metric in metrics if "passed" in metric]
        result["passed"] = comparable and bool(screened_metrics) and all(
            metric["passed"] for metric in screened_metrics
        )
    return result


def compare_suite(
    reference_root: str | Path,
    candidate_root: str | Path,
    points: Sequence[str],
    *,
    max_regression: float | None = None,
    max_latency_regression_ms: float | None = None,
) -> dict[str, Any]:
    reference_root = Path(reference_root)
    candidate_root = Path(candidate_root)
    comparisons = [
        compare_pair(
            reference_root / point,
            candidate_root / point,
            max_regression=max_regression,
            max_latency_regression_ms=max_latency_regression_ms,
        )
        for point in points
    ]
    result = {
        "valid": all(comparison["comparable"] for comparison in comparisons),
        "comparisons": comparisons,
    }
    if max_regression is not None:
        result.update(
            {
                "max_regression": max_regression,
                "passed": bool(comparisons)
                and all(comparison["passed"] for comparison in comparisons),
            }
        )
    elif max_latency_regression_ms is not None:
        result.update(
            {
                "max_latency_regression_ms": max_latency_regression_ms,
                "passed": bool(comparisons)
                and all(comparison["passed"] for comparison in comparisons),
            }
        )
    return result


def render_markdown(report: dict[str, Any]) -> str:
    if report.get("max_latency_regression_ms") is not None:
        return _render_latency_screen_markdown(report)
    if report.get("max_regression") is None:
        return _render_unscreened_markdown(report)
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


def _render_latency_screen_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark comparison",
        "",
        f"Overall: {'pass' if report.get('passed') else 'fail'}",
        "",
        "| Benchmark | Latency metric | Reference | Candidate | Regression | Limit | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in report.get("comparisons", []):
        screened_metrics = [
            metric for metric in comparison.get("metrics", []) if "regression_ms" in metric
        ]
        if not screened_metrics:
            lines.append(
                f"| {comparison.get('benchmark')} | n/a | n/a | n/a | n/a | n/a | fail |"
            )
        for metric in screened_metrics:
            lines.append(
                "| {benchmark} | `{path}` | {reference} ms | {candidate} ms | {regression} ms | {limit} ms | {result} |".format(
                    benchmark=comparison.get("benchmark"),
                    path=metric.get("path"),
                    reference=_number(metric.get("reference")),
                    candidate=_number(metric.get("candidate")),
                    regression=_signed_number(metric.get("regression_ms")),
                    limit=_number(metric.get("maximum_regression_ms")),
                    result="pass" if metric.get("passed") else "fail",
                )
            )
    return "\n".join(lines) + "\n"


def _render_unscreened_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark comparison",
        "",
        f"Bundle validity: {'valid' if report.get('valid') else 'invalid'}",
        "",
        "| Benchmark | Metric | Direction | Reference | Candidate | Raw change |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for comparison in report.get("comparisons", []):
        if not comparison.get("metrics"):
            lines.append(
                f"| {comparison.get('benchmark')} | n/a | n/a | n/a | n/a | n/a |"
            )
        for metric in comparison.get("metrics", []):
            lines.append(
                "| {benchmark} | `{path}` | {direction} | {reference} | {candidate} | {change} |".format(
                    benchmark=comparison.get("benchmark"),
                    path=metric.get("path"),
                    direction=metric.get("direction"),
                    reference=_number(metric.get("reference")),
                    candidate=_number(metric.get("candidate")),
                    change=_percent(metric.get("raw_change_percent")),
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


def _is_latency_metric(path: str, direction: Any) -> bool:
    return direction == "lower" and (path.endswith("_ms") or "_ms." in path)


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"


def _signed_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.6g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    regression_group = parser.add_mutually_exclusive_group()
    regression_group.add_argument("--max-regression", type=float)
    regression_group.add_argument("--max-latency-regression-ms", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.max_regression is not None and not 0 <= args.max_regression < 1:
        parser.error("--max-regression must be in [0, 1)")
    if args.max_latency_regression_ms is not None and args.max_latency_regression_ms < 0:
        parser.error("--max-latency-regression-ms must be non-negative")

    config = load_config(args.config)
    points = config.selected_points(args.selection)
    report = compare_suite(
        args.reference_root,
        args.candidate_root,
        [point.name for point in points],
        max_regression=args.max_regression,
        max_latency_regression_ms=args.max_latency_regression_ms,
    )
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    screened = args.max_regression is not None or args.max_latency_regression_ms is not None
    if report["valid"] is not True or (screened and report["passed"] is not True):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
