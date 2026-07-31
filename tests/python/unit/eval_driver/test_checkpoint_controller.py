"""Behavioral tests for the decode-runtime checkpoint acceptance controller.

The protocol uses the worse regression against the immutable anchor and previous
accepted checkpoint, admits every required performance metric through a single 20%
limit, and retains correctness, work, artifact-validity, interleave-conformance,
and provenance checks as hard gates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]


def _load_controller():
    spec = importlib.util.spec_from_file_location(
        "checkpoint_controller", ROOT / "scripts" / "checkpoint_controller.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_controller"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_controller()


def _base_checks() -> dict[str, bool]:
    return {
        "declared_request_count": True,
        "minimum_successful_requests": True,
        "maximum_failed_requests": True,
        "request_records": True,
        "summary_payload": True,
    }


def _write_point(
    directory: Path,
    *,
    task: str,
    metrics: dict[str, Any],
    request_count: int = 32,
    ok_count: int | None = None,
    failed_count: int = 0,
    elapsed_s: float = 10.0,
    valid: bool = True,
    checks: dict[str, Any] | None = None,
    generation_valid: bool = True,
    interleave_valid: bool | None = None,
    head: str = "0" * 40,
    dirty: bool | None = False,
    dataset_revision: str = "d" * 40,
    load_case: str = "c32",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ok_count = request_count if ok_count is None else ok_count
    artifact: dict[str, Any] = {
        "valid": valid,
        "valid_marker": "canonical-valid-v2" if valid else "invalid",
        "checks": _base_checks() if checks is None else checks,
        "generation_conformance": {"valid": generation_valid},
        "contract": {"spec": {"dataset_revision": dataset_revision}},
        "matrix_contract": {
            "benchmark_definition": {"load_case_id": load_case},
            "execution_policy": {
                "build_manifest": {
                    "source_state": {
                        "head": head,
                        "dirty": dirty,
                        "tracked_changes_sha256": "e" * 64,
                        "untracked_files_sha256": "f" * 64,
                    }
                }
            },
            "model_revision_contract": {"revision": "9" * 40},
        },
    }
    if interleave_valid is not None:
        artifact["interleave_latency_conformance"] = {"valid": interleave_valid}
    summary = {
        "task": task,
        "request_count": request_count,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "elapsed_s": elapsed_s,
        "model": "TestModel",
        "metrics": metrics,
        "artifact": artifact,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "run.json").write_text(
        json.dumps(
            {
                "model": "TestModel",
                "dataset_revision": dataset_revision,
                "sampling_seed": 42,
                "server_topology": "single_server_single_harness",
                "runtime_profile_id": "test",
            }
        ),
        encoding="utf-8",
    )
    return directory


def _interleave_metrics(
    *,
    ttft: float,
    tpot: float,
    image_latency: float,
    transition: float,
    text_to_image: float,
    image_to_text: float,
    coverage: float = 1.0,
    signatures: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ttft_ms": {"mean": ttft, "count": 32},
        "tpot_ms": {"mean": tpot, "count": 32},
        "images": {"image_latency_ms": {"mean": image_latency, "count": 32}},
        "modality_interleave": {
            "transition_timing": {
                "transition_latency_ms": {"mean": transition, "count": 32},
                "text_to_image_transition_latency_ms": {
                    "mean": text_to_image,
                    "count": 16,
                },
                "image_to_text_transition_latency_ms": {
                    "mean": image_to_text,
                    "count": 16,
                },
                "timestamp_coverage": coverage,
                "transition_sample_coverage": coverage,
                "expected_transition_count": 2,
                "request_signatures": signatures or {"r0": "text->image->text"},
            }
        },
    }


def _triplet(
    root: Path,
    *,
    throughput: float,
    ttft: float,
    tpot: float,
    images_per_s: float,
    head: str,
    dirty: bool = False,
) -> None:
    _write_point(
        root / "qwen3_sharegpt" / "uniserve_r16",
        task="text",
        metrics={
            "output_throughput": throughput,
            "mean_ttft_ms": ttft,
            "mean_tpot_ms": tpot,
        },
        request_count=200,
        load_case="r16",
        head=head,
        dirty=dirty,
    )
    _write_point(
        root / "sensenova_mjhq_t2i" / "uniserve_c32",
        task="t2i",
        metrics={"images_per_second": images_per_s},
        head=head,
        dirty=dirty,
    )
    _write_point(
        root / "sensenova_beans_i2t" / "uniserve_c32",
        task="i2t",
        metrics={"output_throughput": throughput},
        head=head,
        dirty=dirty,
    )


def _write_default_travel(
    directory: Path,
    *,
    head: str,
    elapsed_s: float = 42.0,
    dirty: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "elapsed_s": elapsed_s,
        "artifact": {
            "valid": True,
            "valid_marker": "verify-valid-v1",
            "checks": {
                "error_free": True,
                "image_steps_per_image": True,
                "finish_reason": True,
                "image_dimensions": True,
            },
            "provenance": {
                "source_state": {"head": head, "dirty": dirty},
                "profile_contract": {"workload": "gate/sensenova/default-travel"},
            },
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("candidate", "baseline", "objective", "expected"),
    [
        (80.0, 100.0, "maximize", 0.20),
        (100.0, 100.0, "maximize", 0.0),
        (120.0, 100.0, "maximize", 0.0),
        (120.0, 100.0, "minimize", 0.20),
        (100.0, 100.0, "minimize", 0.0),
        (80.0, 100.0, "minimize", 0.0),
    ],
)
def test_regression_formula(
    candidate: float,
    baseline: float,
    objective: str,
    expected: float,
) -> None:
    assert cc.regression(candidate, baseline, objective) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("candidate", "baseline"),
    [(None, 100.0), (100.0, None), (100.0, 0.0), (100.0, -1.0)],
)
def test_regression_requires_positive_values(
    candidate: float | None,
    baseline: float | None,
) -> None:
    assert cc.regression(candidate, baseline, "maximize") is None


@pytest.mark.parametrize(
    ("regression", "expected"),
    [
        (0.0, "pass"),
        (0.20, "pass"),
        (0.200001, "block"),
        (None, "unavailable"),
    ],
)
def test_performance_band(regression: float | None, expected: str) -> None:
    assert cc._band(regression) == expected


def test_metric_uses_worse_regression_against_both_baselines(tmp_path: Path) -> None:
    spec = ("output_throughput", ("output_throughput",), "maximize")
    candidate = cc.load_point(
        _write_point(
            tmp_path / "candidate",
            task="i2t",
            metrics={"output_throughput": 82.0},
        )
    )
    anchor = cc.load_point(
        _write_point(
            tmp_path / "anchor",
            task="i2t",
            metrics={"output_throughput": 100.0},
        )
    )
    previous = cc.load_point(
        _write_point(
            tmp_path / "previous",
            task="i2t",
            metrics={"output_throughput": 90.0},
        )
    )

    outcome = cc.evaluate_metric(spec, candidate, anchor, previous)

    assert outcome.regression_anchor == pytest.approx(0.18)
    assert outcome.regression_previous == pytest.approx(1.0 - 82.0 / 90.0)
    assert outcome.effective_regression == pytest.approx(0.18)
    assert outcome.band == "pass"


def _metric_outcome(band: str) -> Any:
    return cc.MetricOutcome(
        "metric",
        "maximize",
        1.0,
        1.0,
        None,
        0.0,
        None,
        0.0,
        band,
    )


def test_classification_requires_every_metric_to_pass() -> None:
    assert cc.classify([_metric_outcome("pass"), _metric_outcome("pass")]) == "pass"
    assert cc.classify([_metric_outcome("pass"), _metric_outcome("block")]) == "block"
    assert cc.classify([_metric_outcome("unavailable")]) == "block"


def test_hard_gate_accepts_complete_valid_point(tmp_path: Path) -> None:
    point = cc.load_point(
        _write_point(
            tmp_path / "point",
            task="i2t",
            metrics={"output_throughput": 100.0},
        )
    )
    assert cc.hard_gate_failures(point, None, interleave=False) == []


@pytest.mark.parametrize(
    ("point_kwargs", "expected"),
    [
        ({"failed_count": 1, "ok_count": 31}, "failed_requests=1"),
        ({"valid": False}, "artifact_invalid"),
        ({"generation_valid": False}, "generation_conformance_invalid"),
        ({"dirty": True}, "source_tree_dirty"),
        ({"dirty": None}, "source_tree_dirty"),
    ],
)
def test_hard_gate_rejects_invalid_candidate_evidence(
    tmp_path: Path,
    point_kwargs: dict[str, Any],
    expected: str,
) -> None:
    point = cc.load_point(
        _write_point(
            tmp_path / "point",
            task="i2t",
            metrics={"output_throughput": 100.0},
            **point_kwargs,
        )
    )
    assert expected in cc.hard_gate_failures(point, None, interleave=False)


def test_interleave_hard_gate_requires_complete_timestamped_conformance(
    tmp_path: Path,
) -> None:
    signatures = {"r0": "text->image->text"}
    anchor = cc.load_point(
        _write_point(
            tmp_path / "anchor",
            task="interleave",
            metrics=_interleave_metrics(
                ttft=100.0,
                tpot=10.0,
                image_latency=1000.0,
                transition=2000.0,
                text_to_image=3000.0,
                image_to_text=5.0,
                signatures=signatures,
            ),
            interleave_valid=True,
        )
    )
    candidate = cc.load_point(
        _write_point(
            tmp_path / "candidate",
            task="interleave",
            metrics=_interleave_metrics(
                ttft=100.0,
                tpot=10.0,
                image_latency=1000.0,
                transition=2000.0,
                text_to_image=3000.0,
                image_to_text=5.0,
                coverage=0.5,
                signatures=signatures,
            ),
            interleave_valid=False,
        )
    )

    failures = cc.hard_gate_failures(candidate, anchor, interleave=True)

    assert "interleave_latency_invalid" in failures
    assert "timestamp_coverage=0.5" in failures
    assert "transition_sample_coverage=0.5" in failures


def test_checkpoint_passes_at_the_performance_limit(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=80.0,
        ttft=120.0,
        tpot=24.0,
        images_per_s=0.8,
        head="b" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp1",
        major_boundary=False,
    )

    assert report.verdict == "pass"
    assert {metric.band for point in report.points for metric in point.metrics} == {"pass"}


def test_checkpoint_blocks_when_one_metric_exceeds_limit(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=79.9,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="b" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp2",
        major_boundary=False,
    )

    assert report.verdict == "block"


def test_previous_checkpoint_comparison_can_block_candidate(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "previous",
        throughput=120.0,
        ttft=80.0,
        tpot=16.0,
        images_per_s=1.2,
        head="b" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=95.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="c" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        tmp_path / "previous",
        checkpoint="cp2",
        major_boundary=False,
    )

    assert report.verdict == "block"


def test_major_boundary_uses_the_same_performance_limit(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="b" * 40,
    )
    anchor_interleave = _interleave_metrics(
        ttft=100.0,
        tpot=10.0,
        image_latency=1000.0,
        transition=2000.0,
        text_to_image=3000.0,
        image_to_text=5.0,
    )
    candidate_interleave = _interleave_metrics(
        ttft=120.0,
        tpot=12.0,
        image_latency=1200.0,
        transition=2400.0,
        text_to_image=3600.0,
        image_to_text=6.0,
    )
    _write_point(
        tmp_path / "anchor" / "sensenova_ueval_interleave" / "uniserve_c4",
        task="interleave",
        metrics=anchor_interleave,
        interleave_valid=True,
        head="a" * 40,
        load_case="c4",
    )
    _write_point(
        tmp_path / "candidate" / "sensenova_ueval_interleave" / "uniserve_c4",
        task="interleave",
        metrics=candidate_interleave,
        interleave_valid=True,
        head="b" * 40,
        load_case="c4",
    )
    default_travel = _write_default_travel(
        tmp_path / "candidate_default_travel",
        head="b" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp5",
        major_boundary=True,
        default_travel_dir=default_travel,
    )

    assert report.verdict == "pass"
    interleave_names = {
        metric.name
        for point in report.points
        if point.task == "interleave"
        for metric in point.metrics
    }
    assert interleave_names == {
        "ttft_mean_ms",
        "tpot_mean_ms",
        "image_latency_mean_ms",
        "transition_latency_mean_ms",
        "text_to_image_transition_mean_ms",
        "image_to_text_transition_mean_ms",
    }


def test_later_ueval_boundary_compares_the_previous_major_point(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="b" * 40,
    )
    metrics = _interleave_metrics(
        ttft=100.0,
        tpot=10.0,
        image_latency=1000.0,
        transition=2000.0,
        text_to_image=3000.0,
        image_to_text=5.0,
    )
    _write_point(
        tmp_path / "anchor" / "sensenova_ueval_interleave" / "uniserve_c4",
        task="interleave",
        metrics=metrics,
        interleave_valid=True,
        head="a" * 40,
        load_case="c4",
    )
    _write_point(
        tmp_path / "candidate" / "sensenova_ueval_interleave" / "uniserve_c4",
        task="interleave",
        metrics=_interleave_metrics(
            ttft=120.0,
            tpot=12.0,
            image_latency=1200.0,
            transition=2400.0,
            text_to_image=3600.0,
            image_to_text=6.0,
        ),
        interleave_valid=True,
        head="b" * 40,
        load_case="c4",
    )
    _write_point(
        tmp_path / "previous_major" / "sensenova_ueval_interleave" / "uniserve_c4",
        task="interleave",
        metrics=_interleave_metrics(
            ttft=90.0,
            tpot=9.0,
            image_latency=900.0,
            transition=1800.0,
            text_to_image=2700.0,
            image_to_text=4.5,
        ),
        interleave_valid=True,
        head="c" * 40,
        load_case="c4",
    )
    default_travel = _write_default_travel(
        tmp_path / "candidate_default_travel",
        head="b" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp7",
        major_boundary=True,
        default_travel_dir=default_travel,
        previous_major_root=tmp_path / "previous_major",
    )

    assert report.verdict == "block"
    point = next(point for point in report.points if point.task == "interleave")
    assert point.hard_failures == []
    assert {metric.previous for metric in point.metrics} == {
        90.0,
        9.0,
        900.0,
        1800.0,
        2700.0,
        4.5,
    }
    assert {metric.band for metric in point.metrics} == {"block"}


def test_cp3_major_boundary_accepts_triplet_and_default_travel(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=90.0,
        ttft=110.0,
        tpot=22.0,
        images_per_s=0.9,
        head="b" * 40,
    )
    default_travel = _write_default_travel(
        tmp_path / "candidate_default_travel",
        head="b" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp3",
        major_boundary=True,
        default_travel_dir=default_travel,
    )

    assert report.verdict == "pass"
    assert report.source_revision == "b" * 40
    assert {point.task for point in report.points} == {
        "text",
        "t2i",
        "i2t",
        "default_travel",
    }


def test_default_travel_is_bound_to_the_candidate_source_revision(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=90.0,
        ttft=110.0,
        tpot=22.0,
        images_per_s=0.9,
        head="b" * 40,
    )
    default_travel = _write_default_travel(
        tmp_path / "candidate_default_travel",
        head="c" * 40,
    )

    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp3",
        major_boundary=True,
        default_travel_dir=default_travel,
    )

    assert report.verdict == "block"
    point = next(point for point in report.points if point.task == "default_travel")
    assert point.hard_failures == ["source_revision_mismatch"]


def test_acceptance_artifact_declares_the_fixed_limit(tmp_path: Path) -> None:
    _triplet(
        tmp_path / "anchor",
        throughput=100.0,
        ttft=100.0,
        tpot=20.0,
        images_per_s=1.0,
        head="a" * 40,
    )
    _triplet(
        tmp_path / "candidate",
        throughput=90.0,
        ttft=110.0,
        tpot=22.0,
        images_per_s=0.9,
        head="b" * 40,
    )
    report = cc.evaluate_checkpoint(
        tmp_path / "candidate",
        tmp_path / "anchor",
        None,
        checkpoint="cp1",
        major_boundary=False,
    )

    artifact = cc._report_to_dict(report)

    assert artifact["schema_version"] == 3
    assert artifact["source_revision"] == "b" * 40
    assert artifact["regression_limit"] == pytest.approx(0.20)
    assert artifact["verdict"] == "pass"
