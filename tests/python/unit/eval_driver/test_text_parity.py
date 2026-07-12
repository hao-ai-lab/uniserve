"""Fixed-work conformance and strict text canary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uniserve_eval.harness.text_parity import (
    evaluate_text_canary,
    evaluate_text_work_conformance,
)

pytestmark = pytest.mark.unit


def _record(
    request_id: str,
    *,
    text: str = "answer",
    prompt_len: int = 17,
    output_len: int = 5,
) -> dict:
    payload = text.encode("utf-8")
    return {
        "request_id": request_id,
        "success": True,
        "prompt_len": prompt_len,
        "prompt_len_source": "server_usage",
        "output_len": output_len,
        "requested_output_len": output_len,
        "output_len_source": "server_usage",
        "finish_reason": "length",
        "generated_text_bytes": len(payload),
        "generated_text_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_text_canary_passes_without_exposing_response_text(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference_records = [_record("request-0"), _record("request-1", text="second")]
    candidate_records = [dict(record) for record in reference_records]
    reference_records[0]["generated_text"] = "private-reference-response"
    candidate_records[0]["generated_text"] = "private-candidate-response"
    _write_jsonl(reference, reference_records)
    _write_jsonl(candidate, candidate_records)

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is True
    assert evidence["reference_request_count"] == 2
    assert evidence["candidate_request_count"] == 2
    assert evidence["compared_request_count"] == 2
    assert evidence["mismatch_request_ids"] == []
    assert evidence["mismatches"] == []
    assert all(evidence["checks"].values())
    serialized = json.dumps(evidence, sort_keys=True)
    assert "private-reference-response" not in serialized
    assert "private-candidate-response" not in serialized


def test_text_canary_reports_accounting_finish_and_text_mismatches(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference_record = _record("request-0")
    candidate_record = _record("request-0", text="different", prompt_len=18, output_len=6)
    candidate_record.update(
        {
            "prompt_len_source": "request_fallback",
            "output_len_source": "requested_fallback",
            "requested_output_len": 7,
            "finish_reason": "stop",
        }
    )
    _write_jsonl(reference, [reference_record])
    _write_jsonl(candidate, [candidate_record])

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is False
    assert evidence["mismatch_request_ids"] == ["request-0"]
    mismatch = evidence["mismatches"][0]
    assert mismatch["request_id"] == "request-0"
    assert {reason["code"] for reason in mismatch["reasons"]} == {
        "prompt_len_source_not_server_usage",
        "output_len_source_not_server_usage",
        "requested_output_work_mismatch",
        "prompt_len_mismatch",
        "requested_output_len_mismatch",
        "output_len_mismatch",
        "finish_reason_mismatch",
        "generated_text_bytes_mismatch",
        "generated_text_sha256_mismatch",
    }


def test_fixed_work_conformance_does_not_run_the_text_canary(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(reference, [_record("request-0", text="reference answer")])
    _write_jsonl(candidate, [_record("request-0", text="candidate answer")])

    evidence = evaluate_text_work_conformance(reference, candidate)

    assert evidence["passed"] is True
    assert evidence["mismatch_request_ids"] == []
    assert evidence["canary"] == {
        "enabled": False,
        "passed": False,
        "checks": {},
        "mismatch_request_ids": [],
        "mismatches": [],
    }


def test_text_canary_requires_identical_order_and_unique_request_ids(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(reference, [_record("request-0"), _record("request-1")])
    _write_jsonl(
        candidate,
        [_record("request-1"), _record("request-0"), _record("request-0")],
    )

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is False
    assert evidence["checks"]["ordered_request_ids"] is False
    assert evidence["checks"]["unique_request_ids"] is False
    assert evidence["mismatch_request_ids"] == ["request-0", "request-1"]
    reasons = {
        request["request_id"]: {reason["code"] for reason in request["reasons"]}
        for request in evidence["mismatches"]
    }
    assert "duplicate_request_id" in reasons["request-0"]
    assert "request_id_order_mismatch" in reasons["request-0"]
    assert "request_id_order_mismatch" in reasons["request-1"]


def test_text_canary_localizes_missing_requests(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(reference, [_record("request-0"), _record("request-1")])
    _write_jsonl(candidate, [_record("request-0"), _record("request-2")])

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is False
    assert evidence["mismatch_request_ids"] == ["request-1", "request-2"]
    assert evidence["mismatches"] == [
        {
            "request_id": "request-1",
            "reasons": [{"code": "missing_candidate_request"}],
        },
        {
            "request_id": "request-2",
            "reasons": [{"code": "missing_reference_request"}],
        },
    ]


def test_text_canary_returns_structured_input_failures(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference.write_text('{"request_id": "request-0"}\nnot-json\n', encoding="utf-8")
    _write_jsonl(candidate, [_record("request-0")])

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is False
    assert evidence["checks"]["inputs_readable"] is False
    assert evidence["input_failures"] == [{"side": "reference", "code": "invalid_json", "line": 2}]
    assert "not-json" not in json.dumps(evidence, sort_keys=True)


def test_text_canary_rejects_empty_inputs(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    reference.write_text("", encoding="utf-8")
    candidate.write_text("", encoding="utf-8")

    evidence = evaluate_text_canary(reference, candidate)

    assert evidence["passed"] is False
    assert evidence["checks"]["nonempty_inputs"] is False
    assert evidence["input_failures"] == [
        {"side": "reference", "code": "empty_input"},
        {"side": "candidate", "code": "empty_input"},
    ]
