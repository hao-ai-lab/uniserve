"""Unit tests for the decode-runtime checkpoint acceptance controller.

These derive the controller's behavior from the acceptance rules in
``specs/decode-runtime-construction.md`` and ``docs/benchmark-protocol.md``:
the regression formulas, the 5% target / 7% grace bands, the worse-of-two-baselines
rule, the checkpoint-only availability of the grace band, and the hard gates for
failures, artifact validity, interleave conformance, and provenance.
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


# --------------------------------------------------------------------------- #
# Synthetic point fixtures
# --------------------------------------------------------------------------- #

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
    dirty: bool = False,
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
    t2i: float,
    i2t: float,
    t2i_count: int = 1,
    i2t_count: int = 1,
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
                "text_to_image_transition_latency_ms": {"mean": t2i, "count": t2i_count},
                "image_to_text_transition_latency_ms": {"mean": i2t, "count": i2t_count},
                "timestamp_coverage": coverage,
                "transition_sample_coverage": coverage,
                "expected_transition_count": 2,
                "request_signatures": signatures or {"r0": "text->image->text"},
            }
        },
    }


# --------------------------------------------------------------------------- #
# regression + band
# --------------------------------------------------------------------------- #

def test_regression_maximize():
    assert cc.regression(95.0, 100.0, "maximize") == pytest.approx(0.05)
    assert cc.regression(100.0, 100.0, "maximize") == 0.0
    assert cc.regression(110.0, 100.0, "maximize") == 0.0  # improvement clamps to 0


def test_regression_minimize():
    assert cc.regression(105.0, 100.0, "minimize") == pytest.approx(0.05)
    assert cc.regression(100.0, 100.0, "minimize") == 0.0
    assert cc.regression(90.0, 100.0, "minimize") == 0.0  # improvement clamps to 0


def test_regression_missing_or_nonpositive_baseline():
    assert cc.regression(None, 100.0, "maximize") is None
    assert cc.regression(100.0, None, "maximize") is None
    assert cc.regression(100.0, 0.0, "maximize") is None
    assert cc.regression(100.0, -1.0, "minimize") is None


def test_band_boundaries():
    assert cc._band(0.0) == "target"
    assert cc._band(0.05) == "target"
    assert cc._band(0.0500001) == "grace"
    assert cc._band(0.07) == "grace"
    assert cc._band(0.0700001) == "block"
    assert cc._band(None) == "unavailable"


# --------------------------------------------------------------------------- #
# dual-baseline worst-case
# --------------------------------------------------------------------------- #

def test_evaluate_metric_takes_worse_of_two_baselines(tmp_path):
    spec = ("output_throughput", ("output_throughput",), "maximize")
    cand = cc.load_point(_write_point(tmp_path / "c", task="i2t", metrics={"output_throughput": 94.0}))
    anchor = cc.load_point(_write_point(tmp_path / "a", task="i2t", metrics={"output_throughput": 100.0}))
    prev = cc.load_point(_write_point(tmp_path / "p", task="i2t", metrics={"output_throughput": 96.0}))
    out = cc.evaluate_metric(spec, cand, anchor, prev)
    # vs anchor: 6% regression; vs previous: ~2.08%. Worse (effective) is vs anchor.
    assert out.regression_anchor == pytest.approx(0.06)
    assert out.regression_previous == pytest.approx(1 - 94.0 / 96.0)
    assert out.effective_regression == pytest.approx(0.06)
    assert out.band == "grace"


def test_evaluate_metric_unavailable_when_candidate_missing(tmp_path):
    spec = ("output_throughput", ("output_throughput",), "maximize")
    cand = cc.load_point(_write_point(tmp_path / "c", task="i2t", metrics={}))
    anchor = cc.load_point(_write_point(tmp_path / "a", task="i2t", metrics={"output_throughput": 100.0}))
    out = cc.evaluate_metric(spec, cand, anchor, None)
    assert out.effective_regression is None
    assert out.band == "unavailable"


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #

def _mo(band: str):
    return cc.MetricOutcome("m", "maximize", 1.0, 1.0, None, 0.0, None, 0.0, band)


def test_classify_all_target():
    assert cc.classify([_mo("target"), _mo("target")], allow_grace=True) == "target-pass"


def test_classify_grace_allowed():
    assert cc.classify([_mo("target"), _mo("grace")], allow_grace=True) == "grace-pass"


def test_classify_grace_forbidden_blocks():
    # Major boundary: a grace-band metric is a block.
    assert cc.classify([_mo("target"), _mo("grace")], allow_grace=False) == "block"


def test_classify_block_band():
    assert cc.classify([_mo("target"), _mo("block")], allow_grace=True) == "block"


def test_classify_unavailable_blocks():
    assert cc.classify([_mo("unavailable")], allow_grace=True) == "block"


# --------------------------------------------------------------------------- #
# hard gates
# --------------------------------------------------------------------------- #

def test_hard_gate_clean_point_passes(tmp_path):
    p = cc.load_point(_write_point(tmp_path / "p", task="i2t", metrics={"output_throughput": 100.0}))
    assert cc.hard_gate_failures(p, None, interleave=False) == []


def test_hard_gate_failed_requests(tmp_path):
    p = cc.load_point(_write_point(tmp_path / "p", task="i2t", metrics={}, failed_count=1, ok_count=31))
    failures = cc.hard_gate_failures(p, None, interleave=False)
    assert any("failed_requests" in f for f in failures)
    assert any("incomplete_success" in f for f in failures)


def test_hard_gate_invalid_artifact_and_check(tmp_path):
    checks = _base_checks()
    checks["request_records"] = False
    p = cc.load_point(_write_point(tmp_path / "p", task="i2t", metrics={}, valid=False, checks=checks))
    failures = cc.hard_gate_failures(p, None, interleave=False)
    assert "artifact_invalid" in failures
    assert "check:request_records" in failures


def test_hard_gate_dirty_tree(tmp_path):
    p = cc.load_point(_write_point(tmp_path / "p", task="i2t", metrics={}, dirty=True))
    assert "source_tree_dirty" in cc.hard_gate_failures(p, None, interleave=False)


def test_hard_gate_generation_conformance(tmp_path):
    p = cc.load_point(_write_point(tmp_path / "p", task="t2i", metrics={}, generation_valid=False))
    assert "generation_conformance_invalid" in cc.hard_gate_failures(p, None, interleave=False)


def test_hard_gate_interleave_coverage_and_signature(tmp_path):
    cand = cc.load_point(
        _write_point(
            tmp_path / "c",
            task="interleave",
            metrics=_interleave_metrics(
                ttft=1, tpot=1, image_latency=1, transition=1, t2i=1, i2t=1,
                coverage=0.5, signatures={"r0": "text->image"},
            ),
            interleave_valid=False,
        )
    )
    anchor = cc.load_point(
        _write_point(
            tmp_path / "a",
            task="interleave",
            metrics=_interleave_metrics(
                ttft=1, tpot=1, image_latency=1, transition=1, t2i=1, i2t=1,
                signatures={"r0": "text->image->text"},
            ),
            interleave_valid=True,
        )
    )
    failures = cc.hard_gate_failures(cand, anchor, interleave=True)
    assert "interleave_latency_invalid" in failures
    assert any("timestamp_coverage" in f for f in failures)
    assert any("transition_sample_coverage" in f for f in failures)
    assert "modality_signature_mismatch" in failures


# --------------------------------------------------------------------------- #
# end-to-end evaluate_checkpoint
# --------------------------------------------------------------------------- #

def _triplet(root: Path, *, throughput: float, ttft: float, tpot: float, images_per_s: float, head: str = "0" * 40, dirty: bool = False):
    _write_point(
        root / "qwen3_sharegpt" / "uniserve_r16",
        task="text",
        metrics={"output_throughput": throughput, "mean_ttft_ms": ttft, "mean_tpot_ms": tpot},
        request_count=200, load_case="r16", head=head, dirty=dirty,
    )
    _write_point(
        root / "sensenova_mjhq_t2i" / "uniserve_c32",
        task="t2i",
        metrics={"images_per_second": images_per_s},
        head=head, dirty=dirty,
    )
    _write_point(
        root / "sensenova_beans_i2t" / "uniserve_c32",
        task="i2t",
        metrics={"output_throughput": throughput},
        head=head, dirty=dirty,
    )


def test_checkpoint_target_pass(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    # Candidate within 5% on every metric (throughput 97 -> 3% down; ttft 103 -> 3% up).
    _triplet(tmp_path / "cand", throughput=97, ttft=103, tpot=20.5, images_per_s=0.98, head="b" * 40)
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp1", major_boundary=False
    )
    assert report.verdict == "target-pass"


def test_checkpoint_grace_pass(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    # One metric at 6% (grace band); others within target.
    _triplet(tmp_path / "cand", throughput=94, ttft=103, tpot=20.5, images_per_s=0.99, head="b" * 40)
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp2", major_boundary=False
    )
    assert report.verdict == "grace-pass"


def test_checkpoint_block_on_regression(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    # 8% throughput regression exceeds the 7% hard boundary.
    _triplet(tmp_path / "cand", throughput=92, ttft=100, tpot=20, images_per_s=1.0, head="b" * 40)
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp2", major_boundary=False
    )
    assert report.verdict == "block"


def test_checkpoint_block_on_hard_gate(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    _triplet(tmp_path / "cand", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="b" * 40)
    # Corrupt one candidate point with a failed request.
    _write_point(
        tmp_path / "cand" / "sensenova_beans_i2t" / "uniserve_c32",
        task="i2t", metrics={"output_throughput": 100.0}, failed_count=1, ok_count=31, head="b" * 40,
    )
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp2", major_boundary=False
    )
    assert report.verdict == "block"


def test_major_boundary_requires_interleave_and_no_grace(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    _triplet(tmp_path / "cand", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="b" * 40)
    anchor_il = _interleave_metrics(ttft=100, tpot=10, image_latency=1000, transition=2000, t2i=3000, i2t=5)
    # Candidate transition latency 6% worse -> grace band, but major boundary forbids grace -> block.
    cand_il = _interleave_metrics(ttft=100, tpot=10, image_latency=1000, transition=2120, t2i=3000, i2t=5)
    _write_point(tmp_path / "anchor" / "sensenova_ueval_interleave" / "uniserve_c4", task="interleave", metrics=anchor_il, interleave_valid=True, head="a" * 40, load_case="c4")
    _write_point(tmp_path / "cand" / "sensenova_ueval_interleave" / "uniserve_c4", task="interleave", metrics=cand_il, interleave_valid=True, head="b" * 40, load_case="c4")
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp3", major_boundary=True
    )
    assert report.verdict == "block"
    tasks = {p.task for p in report.points}
    assert "interleave" in tasks


def test_major_boundary_target_pass(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    _triplet(tmp_path / "cand", throughput=98, ttft=102, tpot=20.5, images_per_s=0.99, head="b" * 40)
    il = _interleave_metrics(ttft=100, tpot=10, image_latency=1000, transition=2000, t2i=3000, i2t=5)
    _write_point(tmp_path / "anchor" / "iv" / "c4", task="interleave", metrics=il, interleave_valid=True, head="a" * 40, load_case="c4")
    _write_point(tmp_path / "cand" / "iv" / "c4", task="interleave", metrics=il, interleave_valid=True, head="b" * 40, load_case="c4")
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp3", major_boundary=True
    )
    assert report.verdict == "target-pass"


def test_missing_required_point_blocks(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    # Candidate missing the t2i point entirely.
    _write_point(tmp_path / "cand" / "qwen3_sharegpt" / "uniserve_r16", task="text", metrics={"output_throughput": 100, "mean_ttft_ms": 100, "mean_tpot_ms": 20}, request_count=200, load_case="r16", head="b" * 40)
    _write_point(tmp_path / "cand" / "sensenova_beans_i2t" / "uniserve_c32", task="i2t", metrics={"output_throughput": 100}, head="b" * 40)
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp1", major_boundary=False
    )
    assert report.verdict == "block"


def test_directional_transition_skipped_when_baseline_count_zero(tmp_path):
    _triplet(tmp_path / "anchor", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="a" * 40)
    _triplet(tmp_path / "cand", throughput=100, ttft=100, tpot=20, images_per_s=1.0, head="b" * 40)
    # Anchor has zero image_to_text samples: that directional metric must be skipped,
    # so a large candidate i2t value cannot block.
    anchor_il = _interleave_metrics(ttft=100, tpot=10, image_latency=1000, transition=2000, t2i=3000, i2t=0, i2t_count=0)
    cand_il = _interleave_metrics(ttft=100, tpot=10, image_latency=1000, transition=2000, t2i=3000, i2t=99999, i2t_count=1)
    _write_point(tmp_path / "anchor" / "iv" / "c4", task="interleave", metrics=anchor_il, interleave_valid=True, head="a" * 40, load_case="c4")
    _write_point(tmp_path / "cand" / "iv" / "c4", task="interleave", metrics=cand_il, interleave_valid=True, head="b" * 40, load_case="c4")
    report = cc.evaluate_checkpoint(
        tmp_path / "cand", tmp_path / "anchor", None, checkpoint="cp3", major_boundary=True
    )
    names = {m.name for p in report.points for m in p.metrics}
    assert "image_to_text_transition_mean_ms" not in names
    assert report.verdict == "target-pass"
