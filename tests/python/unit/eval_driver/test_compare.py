"""Compare-command behaviors over fabricated workload summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve_eval import compare

pytestmark = pytest.mark.unit


def _write_summary(root: Path, workload: str, summary: dict) -> None:
    out_dir = root / "workloads" / workload
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture()
def image_config(tmp_path: Path) -> dict:
    config = {"artifact_root": str(tmp_path)}
    _write_summary(
        tmp_path,
        "base-t2i",
        {
            "metric_family": "image",
            "request_count": 4,
            "ok_count": 4,
            "gpu_memory": {"peak_single_gpu_mib": 60000},
            "metrics": {
                "image_latency_ms": {"mean": 5000.0, "p50": 4900.0},
                "images_per_minute": 12.0,
            },
        },
    )
    _write_summary(
        tmp_path,
        "cand-t2i",
        {
            "metric_family": "image",
            "request_count": 4,
            "ok_count": 4,
            "gpu_memory": {"peak_single_gpu_mib": 36000},
            "metrics": {
                "image_latency_ms": {"mean": 10000.0, "p50": 9800.0},
                "images_per_minute": 6.0,
            },
        },
    )
    return config


def test_compare_image_family_table_and_ratio_math(image_config, capsys) -> None:
    data = compare.compare_workloads(image_config, ["base-t2i", "cand-t2i"])

    rows = {row["metric"]: row for row in data["rows"]}
    assert data["metric_family"] == "image"
    assert data["baseline"] == "base-t2i"
    # Latency metric: baseline/candidate (candidate twice as slow -> 0.5x).
    assert rows["image_latency_ms.mean"]["ratios"]["cand-t2i"] == pytest.approx(0.5)
    assert rows["image_latency_ms.p50"]["ratios"]["cand-t2i"] == pytest.approx(4900.0 / 9800.0)
    # Higher-is-better metric: candidate/baseline.
    assert rows["images_per_minute"]["ratios"]["cand-t2i"] == pytest.approx(0.5)
    # Common rows present when the summaries carry them.
    assert rows["ok_count"]["values"] == {"base-t2i": 4, "cand-t2i": 4}
    assert rows["gpu_memory.peak_single_gpu_mib"]["ratios"]["cand-t2i"] == pytest.approx(60000 / 36000)

    table = capsys.readouterr().out
    assert "| metric | base-t2i (baseline) | cand-t2i | cand-t2i (x vs baseline) |" in table
    assert "| images_per_minute | 12 | 6 | 0.50x |" in table

    out_dir = Path(image_config["artifact_root"]) / "comparisons" / "base-t2i_vs_cand-t2i"
    assert (out_dir / "compare.md").exists()
    persisted = json.loads((out_dir / "compare.json").read_text(encoding="utf-8"))
    assert persisted["workloads"] == ["base-t2i", "cand-t2i"]


def test_compare_stream_family_metric_set(tmp_path: Path) -> None:
    config = {"artifact_root": str(tmp_path)}
    for name, throughput, ttft, itl, e2e in [("base-i2t", 160.0, 200.0, 5.0, 1500.0), ("cand-i2t", 10.0, 400.0, 100.0, 34000.0)]:
        _write_summary(
            tmp_path,
            name,
            {
                "metric_family": "stream",
                "request_count": 4,
                "ok_count": 4,
                "metrics": {
                    "output_throughput": throughput,
                    "mean_ttft_ms": ttft,
                    "mean_itl_ms": itl,
                    "mean_e2e_latency_ms": e2e,
                },
            },
        )

    data = compare.compare_workloads(config, ["base-i2t", "cand-i2t"])

    rows = {row["metric"]: row for row in data["rows"]}
    assert set(rows) == {
        "output_throughput",
        "mean_ttft_ms",
        "mean_itl_ms",
        "mean_e2e_latency_ms",
        "request_count",
        "ok_count",
    }
    assert rows["output_throughput"]["ratios"]["cand-i2t"] == pytest.approx(10.0 / 160.0)
    assert rows["mean_itl_ms"]["ratios"]["cand-i2t"] == pytest.approx(5.0 / 100.0)


def test_compare_refuses_mixed_metric_families(tmp_path: Path) -> None:
    config = {"artifact_root": str(tmp_path)}
    _write_summary(tmp_path, "img", {"metric_family": "image", "metrics": {}})
    _write_summary(tmp_path, "txt", {"metric_family": "stream", "metrics": {}})

    with pytest.raises(SystemExit, match="mixed metric families"):
        compare.compare_workloads(config, ["img", "txt"])


def test_compare_errors_when_summary_missing(tmp_path: Path) -> None:
    config = {"artifact_root": str(tmp_path)}
    _write_summary(tmp_path, "have", {"metric_family": "image", "metrics": {}})

    with pytest.raises(SystemExit, match="run the point first"):
        compare.compare_workloads(config, ["have", "missing"])
