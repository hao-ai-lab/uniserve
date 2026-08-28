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
    if max_regression is not None:
        result["passed"] = comparable and bool(metrics) and all(
            metric["passed"] for metric in metrics
        )
    return result


def compare_suite(
    reference_root: str | Path,
    candidate_root: str | Path,
    points: Sequence[str],
    *,
    max_regression: float | None = None,
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
    return result


def render_markdown(report: dict[str, Any]) -> str:
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


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--max-regression", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.max_regression is not None and not 0 <= args.max_regression < 1:
        parser.error("--max-regression must be in [0, 1)")

    config = load_config(args.config)
    points = config.selected_points(args.selection)
    report = compare_suite(
        args.reference_root,
        args.candidate_root,
        [point.name for point in points],
        max_regression=args.max_regression,
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
    if report["valid"] is not True or (
        args.max_regression is not None and report["passed"] is not True
    ):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
