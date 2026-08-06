from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve_eval.harness.comparison import compare_pair, compare_suite, render_markdown

pytestmark = pytest.mark.unit


def _write_summary(directory: Path, *, throughput: float, latency: float) -> None:
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "benchmark": "point",
                "task": "text",
                "workload": {"name": "point", "request_rate": 16},
                "selected_rows": {"count": 2, "sha256": "a" * 64},
                "metric_definitions": [
                    {"path": "output_throughput", "direction": "higher"},
                    {"path": "mean_ttft_ms", "direction": "lower"},
                ],
                "metrics": {
                    "output_throughput": throughput,
                    "mean_ttft_ms": latency,
                },
                "validation": {"valid": True},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_comparison_normalizes_metric_direction(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_summary(reference, throughput=100, latency=10)
    _write_summary(candidate, throughput=96, latency=10.4)
    result = compare_pair(reference, candidate, max_regression=0.05)
    assert result["comparable"] is True
    assert result["passed"] is True
    assert result["metrics"][0]["normalized_ratio"] == pytest.approx(0.96)
    assert result["metrics"][1]["normalized_ratio"] == pytest.approx(10 / 10.4)


def test_comparison_without_a_screen_reports_raw_metrics(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_summary(reference, throughput=100, latency=10)
    _write_summary(candidate, throughput=96, latency=10.4)
    result = compare_pair(reference, candidate)
    assert result["comparable"] is True
    assert "passed" not in result
    assert result["metrics"][0]["raw_change_percent"] == pytest.approx(-4.0)
    assert "normalized_ratio" not in result["metrics"][0]
    assert "minimum_ratio" not in result["metrics"][0]
    assert "passed" not in result["metrics"][0]
    report = compare_suite(tmp_path, tmp_path, (), max_regression=None)
    report["valid"] = True
    report["comparisons"] = [result]
    markdown = render_markdown(report)
    assert "Bundle validity: valid" in markdown
    assert "Raw change" in markdown
    assert "Result" not in markdown


def test_comparison_rejects_different_selected_rows(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_summary(reference, throughput=100, latency=10)
    _write_summary(candidate, throughput=100, latency=10)
    summary_path = candidate / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selected_rows"]["sha256"] = "b" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = compare_pair(reference, candidate, max_regression=0.05)
    assert result["comparable"] is False
    assert "selected_rows_mismatch" in result["failures"]
