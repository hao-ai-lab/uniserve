from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "compare_eval.py"
_SPEC = importlib.util.spec_from_file_location("compare_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
compare_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_eval)


def _write_summary(
    directory: Path, *, throughput: float, latency: float
) -> None:
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
    result = compare_eval.compare_pair(
        reference, candidate, max_regression=0.05
    )
    assert result["comparable"] is True
    assert result["passed"] is True
    assert result["metrics"][0]["slowdown"] == pytest.approx(100 / 96)
    assert result["metrics"][1]["slowdown"] == pytest.approx(10.4 / 10)


def test_comparison_without_a_screen_reports_raw_metrics(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_summary(reference, throughput=100, latency=10)
    _write_summary(candidate, throughput=96, latency=10.4)
    result = compare_eval.compare_pair(reference, candidate)
    assert result["comparable"] is True
    assert "passed" not in result
    assert result["metrics"][0]["raw_change_percent"] == pytest.approx(-4.0)
    assert "passed" not in result["metrics"][0]
    report = compare_eval.compare_suite(
        tmp_path, tmp_path, (), max_regression=None
    )
    report["valid"] = True
    report["comparisons"] = [result]
    markdown = compare_eval.render_markdown(report)
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
    result = compare_eval.compare_pair(
        reference, candidate, max_regression=0.05
    )
    assert result["comparable"] is False
    assert "selected_rows_mismatch" in result["failures"]


def test_latency_screen_uses_absolute_millisecond_headroom(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    passing_candidate = tmp_path / "passing"
    failing_candidate = tmp_path / "failing"
    _write_summary(reference, throughput=100, latency=10_000)
    _write_summary(passing_candidate, throughput=1, latency=10_300)
    _write_summary(failing_candidate, throughput=100, latency=10_301)

    passing = compare_eval.compare_pair(
        reference,
        passing_candidate,
        max_latency_regression_ms=300,
    )
    failing = compare_eval.compare_pair(
        reference,
        failing_candidate,
        max_latency_regression_ms=300,
    )

    assert passing["passed"] is True
    assert "passed" not in passing["metrics"][0]
    assert passing["metrics"][1]["regression_seconds"] == pytest.approx(0.3)
    assert passing["metrics"][1]["passed"] is True
    assert failing["passed"] is False
    assert failing["metrics"][1]["passed"] is False


@pytest.mark.parametrize("slowdown, passed", [(1.05, True), (1.051, False)])
@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_relative_budget_is_an_exact_slowdown_limit(
    tmp_path, slowdown, passed, direction
):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _write_summary(reference, throughput=105, latency=100)
    _write_summary(
        candidate,
        throughput=105 / slowdown if direction == "higher" else 105,
        latency=100 * slowdown if direction == "lower" else 100,
    )
    result = compare_eval.compare_pair(
        reference, candidate, max_regression=0.05
    )
    assert result["passed"] is passed


@pytest.mark.parametrize(
    "duration, latency, passed",
    [(10.3, 10_300, True), (10.301, 10_000, False), (10.0, 10_301, False)],
)
def test_video_budget_checks_latency_and_reciprocal_throughput(
    tmp_path, duration, latency, passed
):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for directory, seconds, latency_ms in (
        (reference, 10.0, 10_000),
        (candidate, duration, latency),
    ):
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "benchmark": "video",
                    "task": "video",
                    "workload": {"samples": 3},
                    "selected_rows": {"count": 3, "sha256": "a" * 64},
                    "metric_definitions": [
                        {"path": "videos_per_second", "direction": "higher"},
                        {"path": "video_latency_ms.mean", "direction": "lower"},
                    ],
                    "metrics": {
                        "videos_per_second": 1.0 / seconds,
                        "video_latency_ms": {"mean": latency_ms},
                    },
                    "validation": {"valid": True},
                }
            )
        )
    result = compare_eval.compare_pair(
        reference, candidate, max_latency_regression_ms=300
    )
    assert result["passed"] is passed
    assert result["metrics"][0]["regression_seconds"] == pytest.approx(
        duration - 10
    )
    assert result["metrics"][1]["regression_seconds"] == pytest.approx(
        latency / 1000 - 10
    )
    report = {
        "max_latency_regression_ms": 300,
        "passed": passed,
        "comparisons": [result],
    }
    assert "videos_per_second" in compare_eval.render_markdown(report)
