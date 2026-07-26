"""Single-entry benchmark comparison behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uniserve_eval.harness.comparison import compare_pair, summarize_runs

pytestmark = pytest.mark.unit

_PARITY_FINGERPRINT = "a" * 64


def _request(request_id: str, text: str) -> dict:
    payload = text.encode("utf-8")
    return {
        "request_id": request_id,
        "success": True,
        "prompt_len": 8,
        "prompt_len_source": "server_usage",
        "output_len": 13,
        "output_len_source": "server_usage",
        "requested_output_len": 13,
        "finish_reason": "length",
        "generated_text_bytes": len(payload),
        "generated_text_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _point(directory: Path, *, role: str, throughput: float, text: str) -> None:
    directory.mkdir()
    summary = {
        "task": "text",
        "request_count": 1,
        "ok_count": 1,
        "failed_count": 0,
        "elapsed_s": 1.0,
        "metrics": {
            "output_throughput": throughput,
            "mean_ttft_ms": 40.0 if role == "reference" else 30.0,
            "mean_tpot_ms": 12.0 if role == "reference" else 10.0,
        },
        "artifact": {
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "matrix_contract": {
                "comparison_role": role,
                "parity_group": "qwen3_sharegpt",
                "benchmark_definition": {"load_case_id": "r16"},
                "parity_contract": {
                    "fingerprint": _PARITY_FINGERPRINT,
                    "harness": {"spec": {"request_rate": 16.0}},
                },
            },
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "requests.jsonl").write_text(
        json.dumps(_request("request-1", text)) + "\n",
        encoding="utf-8",
    )


def test_text_comparison_runs_only_fixed_work_by_default(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _point(reference, role="reference", throughput=100.0, text="reference")
    _point(candidate, role="candidate", throughput=125.0, text="candidate")

    result = compare_pair(reference, candidate)

    assert result["valid"] is True
    assert result["work"]["passed"] is True
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics["output_throughput"]["candidate_over_reference"] == 1.25
    assert metrics["mean_ttft_ms"]["candidate_over_reference"] == 0.75
    assert metrics["mean_tpot_ms"]["candidate_over_reference"] == pytest.approx(10.0 / 12.0)
    assert "text_canary" not in result
    assert "image_smoke" not in result


def test_text_canary_is_added_only_when_enabled(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _point(reference, role="reference", throughput=100.0, text="reference")
    _point(candidate, role="candidate", throughput=125.0, text="candidate")

    result = compare_pair(reference, candidate, text_canary=True)

    assert result["valid"] is True
    assert result["work"]["passed"] is True
    assert result["text_canary"]["passed"] is False
    assert result["text_canary"]["mismatch_request_ids"] == ["request-1"]


def _i2t_point(
    directory: Path,
    *,
    role: str,
    prompt_len: int,
    output_len: int,
    ignore_eos: bool,
    parity_fingerprint: str = _PARITY_FINGERPRINT,
) -> None:
    directory.mkdir()
    summary = {
        "task": "i2t",
        "request_count": 1,
        "ok_count": 1,
        "failed_count": 0,
        "elapsed_s": 1.0,
        "metrics": {"output_throughput": float(output_len)},
        "artifact": {
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "matrix_contract": {
                "comparison_role": role,
                "parity_group": "image_to_text",
                "benchmark_definition": {"load_case_id": "r1"},
                "parity_contract": {
                    "fingerprint": parity_fingerprint,
                    "harness": {"spec": {"ignore_eos": ignore_eos}},
                },
            },
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    record = _request("request-1", "answer")
    record.update(
        {
            "prompt_len": prompt_len,
            "output_len": output_len,
            "requested_output_len": 256,
            "finish_reason": "length" if output_len == 256 else "stop",
        }
    )
    (directory / "requests.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def test_i2t_comparison_uses_declared_natural_eos_work_contract(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _i2t_point(
        reference,
        role="reference",
        prompt_len=5,
        output_len=160,
        ignore_eos=False,
    )
    _i2t_point(
        candidate,
        role="candidate",
        prompt_len=16,
        output_len=256,
        ignore_eos=False,
    )

    result = compare_pair(reference, candidate)

    assert result["valid"] is True
    assert result["work"]["passed"] is True
    assert result["work"]["checks"]["output_counts_within_requested_limits"] is True
    assert result["metrics"][0]["candidate_over_reference"] == pytest.approx(1.6)


def test_comparison_rejects_different_parity_contracts(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _i2t_point(
        reference,
        role="reference",
        prompt_len=5,
        output_len=160,
        ignore_eos=False,
    )
    _i2t_point(
        candidate,
        role="candidate",
        prompt_len=16,
        output_len=256,
        ignore_eos=False,
        parity_fingerprint="b" * 64,
    )

    result = compare_pair(reference, candidate)

    assert result["valid"] is False
    assert result["failures"] == ["parity_contract_mismatch"]


def test_run_summary_keeps_each_ratio_and_reports_geometric_mean() -> None:
    def run(ratio: float) -> dict:
        return {
            "comparisons": [
                {
                    "comparison": "qwen3_sharegpt",
                    "load_case": "r16",
                    "valid": True,
                    "metrics": [
                        {
                            "name": "output_throughput",
                            "objective": "maximize",
                            "candidate_over_reference": ratio,
                        }
                    ],
                }
            ]
        }

    summary = summarize_runs([run(1.0), run(1.21)])

    assert summary["run_count"] == 2
    assert summary["comparisons"] == [
        {
            "comparison": "qwen3_sharegpt",
            "load_case": "r16",
            "metric": "output_throughput",
            "objective": "maximize",
            "run_count": 2,
            "valid_run_count": 2,
            "candidate_over_reference": [1.0, 1.21],
            "geometric_mean": pytest.approx(1.1),
        }
    ]


def _image_point(
    directory: Path,
    *,
    role: str,
    load_case: str,
    latency_ms: float,
    throughput: float,
) -> None:
    directory.mkdir()
    summary = {
        "task": "t2i",
        "request_count": 32,
        "ok_count": 32,
        "failed_count": 0,
        "elapsed_s": 32.0,
        "metrics": {
            "images_per_second": throughput,
            "image_latency_ms": {"mean": latency_ms, "p50": latency_ms},
        },
        "artifact": {
            "valid": True,
            "valid_marker": "canonical-valid-v2",
            "generation_conformance": {"valid": True},
            "matrix_contract": {
                "comparison_role": role,
                "parity_group": "t2i_pair",
                "benchmark_definition": {"load_case_id": load_case},
                "parity_contract": {"fingerprint": _PARITY_FINGERPRINT},
            },
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.parametrize(
    ("load_case", "expected_name", "expected_objective", "expected_ratio"),
    [
        ("c1", "image_latency_ms.mean", "minimize", 0.8),
        ("c32", "images_per_second", "maximize", 1.25),
    ],
)
def test_image_comparison_uses_concurrency_case_metric(
    tmp_path: Path,
    load_case: str,
    expected_name: str,
    expected_objective: str,
    expected_ratio: float,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _image_point(
        reference, role="reference", load_case=load_case, latency_ms=1000.0, throughput=4.0
    )
    _image_point(candidate, role="candidate", load_case=load_case, latency_ms=800.0, throughput=5.0)

    result = compare_pair(reference, candidate)

    assert result["valid"] is True
    assert result["metrics"][0]["name"] == expected_name
    assert result["metrics"][0]["objective"] == expected_objective
    assert result["metrics"][0]["candidate_over_reference"] == pytest.approx(expected_ratio)
