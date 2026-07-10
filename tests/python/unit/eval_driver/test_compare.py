"""Compare-command behaviors over fabricated workload summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniserve_eval import compare
from uniserve_eval.harness.artifacts import ArtifactWriter
from uniserve_eval.harness.report import record_collection_contract, write_summary_artifacts

pytestmark = pytest.mark.unit


def _write_summary(root: Path, workload: str, summary: dict) -> None:
    request_count = int(summary.get("request_count", 0))
    request_records = [{"request_id": f"request-{index}"} for index in range(request_count)]
    gpu_samples: list[dict] = []
    summary.setdefault("request_count", request_count)
    summary.setdefault("spec", {"workload": workload})
    summary.setdefault("base_url", "http://127.0.0.1:8000")
    summary.setdefault(
        "artifact",
        {
            "schema_version": 2,
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "checks": {
                "requests": True,
                "plan_evidence": True,
                "profile_contract": True,
                "request_records": True,
                "gpu_samples": True,
            },
            "contract": {"fingerprint": f"harness-{workload}"},
            "profile_contract": {
                "schema_version": 2,
                "fingerprint": f"profile-{workload}",
                "model_contract": {
                    "kind": "directory",
                    "tree_sha256": "shared-model",
                },
            },
            "request_records": record_collection_contract(request_records),
            "gpu_samples": record_collection_contract(gpu_samples),
        },
    )
    out_dir = root / "workloads" / workload
    writer = ArtifactWriter(out_dir)
    writer.write_jsonl("requests.jsonl", request_records)
    writer.write_jsonl("gpu_samples.jsonl", gpu_samples)
    writer.write_json(
        "run.json",
        {
            "harness_status": "completed",
            "artifact_valid": True,
            "items": request_count,
            "spec": summary["spec"],
            "base_url": summary["base_url"],
        },
    )
    write_summary_artifacts(out_dir, summary)


@pytest.fixture(autouse=True)
def current_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compare,
        "resolved_perf_command",
        lambda _config, name: ([name], {"name": name}, {"name": name}),
    )
    monkeypatch.setattr(compare, "spec_from_harness_command", lambda command: command[0])
    monkeypatch.setattr(compare, "load_benchmark_inputs", lambda _spec: ([{"id": "row"}], None))
    monkeypatch.setattr(
        compare,
        "benchmark_contract",
        lambda spec, _rows: {"fingerprint": f"harness-{spec}"},
    )
    monkeypatch.setattr(
        compare,
        "benchmark_parity_contract",
        lambda _contract: {"fingerprint": "shared-parity"},
    )
    monkeypatch.setattr(
        compare,
        "perf_profile_contract_fingerprint",
        lambda _command, workload, _server, _contract, **_kwargs: f"profile-{workload['name']}",
    )


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


def test_compare_rejects_protocol_or_workload_parity_mismatch(
    image_config, monkeypatch
) -> None:
    monkeypatch.setattr(
        compare,
        "benchmark_parity_contract",
        lambda contract: {"fingerprint": contract["fingerprint"]},
    )

    with pytest.raises(SystemExit, match="different protocol or workload contracts"):
        compare.compare_workloads(image_config, ["base-t2i", "cand-t2i"])


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


def test_compare_refuses_noncanonical_artifact(tmp_path: Path) -> None:
    config = {"artifact_root": str(tmp_path)}
    _write_summary(
        tmp_path,
        "invalid",
        {"metric_family": "image", "metrics": {}, "artifact": {"valid": False}},
    )
    _write_summary(tmp_path, "valid", {"metric_family": "image", "metrics": {}})

    with pytest.raises(SystemExit, match="outside the current contract"):
        compare.compare_workloads(config, ["invalid", "valid"])


def test_compare_refuses_stale_contract(tmp_path: Path) -> None:
    config = {"artifact_root": str(tmp_path)}
    _write_summary(tmp_path, "stale", {"metric_family": "image", "metrics": {}})
    _write_summary(tmp_path, "current", {"metric_family": "image", "metrics": {}})
    path = tmp_path / "workloads" / "stale" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["artifact"]["contract"]["fingerprint"] = "stale"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(SystemExit, match="outside the current contract"):
        compare.compare_workloads(config, ["stale", "current"])
