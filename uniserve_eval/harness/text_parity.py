"""Fixed-work conformance and an optional strict text canary.

The fixed-work path compares request identity and token accounting. The optional
canary additionally compares termination metadata plus the byte length and
SHA-256 digest of generated text. Raw response text is never copied into the
returned evidence.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_WORK_CHECK_NAMES = (
    "inputs_readable",
    "nonempty_inputs",
    "ordered_request_ids",
    "unique_request_ids",
    "successful_requests",
    "server_reported_prompt_counts",
    "server_reported_output_counts",
    "requested_output_work",
    "matching_prompt_counts",
    "matching_requested_output_work",
    "matching_output_counts",
)
_CANARY_CHECK_NAMES = (
    "matching_finish_reason",
    "matching_generated_text_bytes",
    "matching_generated_text_sha256",
)
_CHECK_NAMES = _WORK_CHECK_NAMES + _CANARY_CHECK_NAMES
_CANARY_REASON_CODES = frozenset(
    {
        "invalid_finish_reason",
        "finish_reason_mismatch",
        "invalid_generated_text_bytes",
        "generated_text_bytes_mismatch",
        "invalid_generated_text_sha256",
        "generated_text_sha256_mismatch",
    }
)


def evaluate_text_canary(
    reference_requests: str | Path,
    candidate_requests: str | Path,
) -> dict[str, Any]:
    """Return a strict, redacted text canary for two request files.

    Request identity is positional: both files must contain the same unique
    request IDs in the same order. Common requests are compared by ID so a
    missing row produces one localized failure rather than shifting every
    subsequent comparison. Input failures and semantic mismatches are returned
    as data; the function performs no writes.
    """

    evidence = _evaluate_text_parity(reference_requests, candidate_requests, include_canary=True)
    diagnostic = evidence["canary"]
    mismatch_ids = _unique_in_order(
        evidence["mismatch_request_ids"] + diagnostic["mismatch_request_ids"]
    )
    work_by_id = {
        mismatch["request_id"]: mismatch["reasons"] for mismatch in evidence["mismatches"]
    }
    canary_by_id = {
        mismatch["request_id"]: mismatch["reasons"] for mismatch in diagnostic["mismatches"]
    }
    evidence.update(
        {
            "kind": "strict_text_canary",
            "passed": evidence["work_conformance_passed"] and diagnostic["passed"],
            "mismatch_request_ids": mismatch_ids,
            "mismatches": [
                {
                    "request_id": request_id,
                    "reasons": work_by_id.get(request_id, []) + canary_by_id.get(request_id, []),
                }
                for request_id in mismatch_ids
            ],
        }
    )
    return evidence


def evaluate_text_work_conformance(
    reference_requests: str | Path,
    candidate_requests: str | Path,
) -> dict[str, Any]:
    """Return request and token-work conformance without running the canary."""

    evidence = _evaluate_text_parity(reference_requests, candidate_requests, include_canary=False)
    evidence.update(
        {
            "kind": "fixed_work_text_conformance",
            "passed": evidence["work_conformance_passed"],
        }
    )
    return evidence


def _evaluate_text_parity(
    reference_requests: str | Path,
    candidate_requests: str | Path,
    *,
    include_canary: bool,
) -> dict[str, Any]:
    check_names = _CHECK_NAMES if include_canary else _WORK_CHECK_NAMES
    checks = {name: True for name in check_names}
    reference_rows, reference_failures = _load_jsonl(Path(reference_requests), side="reference")
    candidate_rows, candidate_failures = _load_jsonl(Path(candidate_requests), side="candidate")
    input_failures = reference_failures + candidate_failures
    if input_failures:
        checks["inputs_readable"] = False
    if not reference_rows or not candidate_rows:
        checks["nonempty_inputs"] = False

    reasons_by_id: dict[str, list[dict[str, Any]]] = {}
    reference_ids = _request_ids(
        reference_rows,
        side="reference",
        checks=checks,
        input_failures=input_failures,
    )
    candidate_ids = _request_ids(
        candidate_rows,
        side="candidate",
        checks=checks,
        input_failures=input_failures,
    )
    if input_failures:
        checks["inputs_readable"] = False

    _record_duplicate_ids(
        reference_ids,
        side="reference",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _record_duplicate_ids(
        candidate_ids,
        side="candidate",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_request_id_sequences(
        reference_ids,
        candidate_ids,
        checks=checks,
        reasons_by_id=reasons_by_id,
    )

    reference_by_id = _first_rows_by_id(reference_rows)
    candidate_by_id = _first_rows_by_id(candidate_rows)
    common_ids = [
        request_id
        for request_id in _unique_in_order(reference_ids)
        if request_id in candidate_by_id
    ]
    for request_id in common_ids:
        _compare_request(
            request_id,
            reference_by_id[request_id],
            candidate_by_id[request_id],
            checks=checks,
            reasons_by_id=reasons_by_id,
            include_canary=include_canary,
        )

    mismatch_order = _unique_in_order(reference_ids + candidate_ids)
    canary_mismatch_request_ids = [
        request_id
        for request_id in mismatch_order
        if any(
            reason.get("code") in _CANARY_REASON_CODES
            for reason in reasons_by_id.get(request_id, ())
        )
    ]
    work_mismatch_request_ids = [
        request_id
        for request_id in mismatch_order
        if any(
            reason.get("code") not in _CANARY_REASON_CODES
            for reason in reasons_by_id.get(request_id, ())
        )
    ]
    work_mismatches = [
        {
            "request_id": request_id,
            "reasons": [
                reason
                for reason in reasons_by_id[request_id]
                if reason.get("code") not in _CANARY_REASON_CODES
            ],
        }
        for request_id in work_mismatch_request_ids
    ]
    canary_mismatches = [
        {
            "request_id": request_id,
            "reasons": [
                reason
                for reason in reasons_by_id[request_id]
                if reason.get("code") in _CANARY_REASON_CODES
            ],
        }
        for request_id in canary_mismatch_request_ids
    ]
    work_checks = {name: checks[name] for name in _WORK_CHECK_NAMES}
    canary_checks = {name: checks[name] for name in _CANARY_CHECK_NAMES} if include_canary else {}
    work_conformance_passed = not input_failures and all(work_checks.values())
    canary_passed = include_canary and not input_failures and all(canary_checks.values())
    return {
        "schema_version": 1,
        "work_conformance_passed": work_conformance_passed,
        "reference_request_count": len(reference_rows),
        "candidate_request_count": len(candidate_rows),
        "compared_request_count": len(common_ids),
        "checks": checks,
        "work_checks": work_checks,
        "mismatch_request_ids": work_mismatch_request_ids,
        "mismatches": work_mismatches,
        "canary": {
            "enabled": include_canary,
            "passed": canary_passed,
            "checks": canary_checks if include_canary else {},
            "mismatch_request_ids": canary_mismatch_request_ids,
            "mismatches": canary_mismatches,
        },
        "input_failures": input_failures,
    }


def _load_jsonl(
    path: Path,
    *,
    side: str,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return records, [{"side": side, "code": "input_not_found"}]
    except UnicodeDecodeError:
        return records, [{"side": side, "code": "invalid_utf8"}]
    except OSError:
        return records, [{"side": side, "code": "input_unreadable"}]

    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            failures.append({"side": side, "code": "invalid_json", "line": line_number})
            continue
        if not isinstance(value, dict):
            failures.append({"side": side, "code": "record_not_object", "line": line_number})
            continue
        records.append((line_number, value))
    if not records and not failures:
        failures.append({"side": side, "code": "empty_input"})
    return records, failures


def _request_ids(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    side: str,
    checks: dict[str, bool],
    input_failures: list[dict[str, Any]],
) -> list[str]:
    request_ids: list[str] = []
    for line_number, record in rows:
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            checks["inputs_readable"] = False
            checks["ordered_request_ids"] = False
            input_failures.append({"side": side, "code": "invalid_request_id", "line": line_number})
            continue
        request_ids.append(request_id)
    return request_ids


def _record_duplicate_ids(
    request_ids: list[str],
    *,
    side: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    counts = Counter(request_ids)
    for request_id, count in counts.items():
        if count <= 1:
            continue
        checks["unique_request_ids"] = False
        _add_reason(
            reasons_by_id,
            request_id,
            "duplicate_request_id",
            side=side,
            count=count,
        )


def _compare_request_id_sequences(
    reference_ids: list[str],
    candidate_ids: list[str],
    *,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    if reference_ids == candidate_ids:
        return
    checks["ordered_request_ids"] = False
    reference_unique = _unique_in_order(reference_ids)
    candidate_unique = _unique_in_order(candidate_ids)
    reference_set = set(reference_unique)
    candidate_set = set(candidate_unique)

    for request_id in reference_unique:
        if request_id not in candidate_set:
            _add_reason(reasons_by_id, request_id, "missing_candidate_request")
    for request_id in candidate_unique:
        if request_id not in reference_set:
            _add_reason(reasons_by_id, request_id, "missing_reference_request")

    common = reference_set & candidate_set
    reference_common = [request_id for request_id in reference_unique if request_id in common]
    candidate_common = [request_id for request_id in candidate_unique if request_id in common]
    if reference_common == candidate_common:
        return
    reference_positions = {request_id: index for index, request_id in enumerate(reference_common)}
    candidate_positions = {request_id: index for index, request_id in enumerate(candidate_common)}
    for request_id in reference_common:
        reference_position = reference_positions[request_id]
        candidate_position = candidate_positions[request_id]
        if reference_position != candidate_position:
            _add_reason(
                reasons_by_id,
                request_id,
                "request_id_order_mismatch",
                reference_position=reference_position,
                candidate_position=candidate_position,
            )


def _first_rows_by_id(
    rows: list[tuple[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for _, record in rows:
        request_id = record.get("request_id")
        if isinstance(request_id, str) and request_id:
            by_id.setdefault(request_id, record)
    return by_id


def _compare_request(
    request_id: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
    include_canary: bool,
) -> None:
    for side, record in (("reference", reference), ("candidate", candidate)):
        if record.get("success") is not True:
            checks["successful_requests"] = False
            _add_reason(reasons_by_id, request_id, "request_unsuccessful", side=side)

    reference_prompt = _server_count(
        request_id,
        reference,
        side="reference",
        field="prompt_len",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_prompt = _server_count(
        request_id,
        candidate,
        side="candidate",
        field="prompt_len",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_values(
        request_id,
        reference_prompt,
        candidate_prompt,
        check="matching_prompt_counts",
        mismatch_code="prompt_len_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )

    reference_output = _server_count(
        request_id,
        reference,
        side="reference",
        field="output_len",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_output = _server_count(
        request_id,
        candidate,
        side="candidate",
        field="output_len",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    reference_requested = _nonnegative_int(
        request_id,
        reference,
        side="reference",
        field="requested_output_len",
        check="requested_output_work",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_requested = _nonnegative_int(
        request_id,
        candidate,
        side="candidate",
        field="requested_output_len",
        check="requested_output_work",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    for side, requested, actual in (
        ("reference", reference_requested, reference_output),
        ("candidate", candidate_requested, candidate_output),
    ):
        if requested is None or actual is None:
            checks["requested_output_work"] = False
        elif requested != actual:
            checks["requested_output_work"] = False
            _add_reason(
                reasons_by_id,
                request_id,
                "requested_output_work_mismatch",
                side=side,
            )
    _compare_values(
        request_id,
        reference_requested,
        candidate_requested,
        check="matching_requested_output_work",
        mismatch_code="requested_output_len_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_values(
        request_id,
        reference_output,
        candidate_output,
        check="matching_output_counts",
        mismatch_code="output_len_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )

    if not include_canary:
        return

    reference_finish = _nonempty_string(
        request_id,
        reference,
        side="reference",
        field="finish_reason",
        check="matching_finish_reason",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_finish = _nonempty_string(
        request_id,
        candidate,
        side="candidate",
        field="finish_reason",
        check="matching_finish_reason",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_values(
        request_id,
        reference_finish,
        candidate_finish,
        check="matching_finish_reason",
        mismatch_code="finish_reason_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )

    reference_bytes = _nonnegative_int(
        request_id,
        reference,
        side="reference",
        field="generated_text_bytes",
        check="matching_generated_text_bytes",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_bytes = _nonnegative_int(
        request_id,
        candidate,
        side="candidate",
        field="generated_text_bytes",
        check="matching_generated_text_bytes",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_values(
        request_id,
        reference_bytes,
        candidate_bytes,
        check="matching_generated_text_bytes",
        mismatch_code="generated_text_bytes_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )

    reference_sha = _sha256(
        request_id,
        reference,
        side="reference",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    candidate_sha = _sha256(
        request_id,
        candidate,
        side="candidate",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )
    _compare_values(
        request_id,
        reference_sha,
        candidate_sha,
        check="matching_generated_text_sha256",
        mismatch_code="generated_text_sha256_mismatch",
        checks=checks,
        reasons_by_id=reasons_by_id,
    )


def _server_count(
    request_id: str,
    record: dict[str, Any],
    *,
    side: str,
    field: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> int | None:
    check = (
        "server_reported_prompt_counts"
        if field == "prompt_len"
        else "server_reported_output_counts"
    )
    source_field = f"{field}_source"
    if record.get(source_field) != "server_usage":
        checks[check] = False
        _add_reason(
            reasons_by_id,
            request_id,
            f"{source_field}_not_server_usage",
            side=side,
        )
    return _nonnegative_int(
        request_id,
        record,
        side=side,
        field=field,
        check=check,
        checks=checks,
        reasons_by_id=reasons_by_id,
    )


def _nonnegative_int(
    request_id: str,
    record: dict[str, Any],
    *,
    side: str,
    field: str,
    check: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> int | None:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        checks[check] = False
        _add_reason(
            reasons_by_id,
            request_id,
            f"invalid_{field}",
            side=side,
        )
        return None
    return value


def _nonempty_string(
    request_id: str,
    record: dict[str, Any],
    *,
    side: str,
    field: str,
    check: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> str | None:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        checks[check] = False
        _add_reason(
            reasons_by_id,
            request_id,
            f"invalid_{field}",
            side=side,
        )
        return None
    return value


def _sha256(
    request_id: str,
    record: dict[str, Any],
    *,
    side: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> str | None:
    value = record.get("generated_text_sha256")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        checks["matching_generated_text_sha256"] = False
        _add_reason(
            reasons_by_id,
            request_id,
            "invalid_generated_text_sha256",
            side=side,
        )
        return None
    return value


def _compare_values(
    request_id: str,
    reference: Any | None,
    candidate: Any | None,
    *,
    check: str,
    mismatch_code: str,
    checks: dict[str, bool],
    reasons_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    if reference is None or candidate is None:
        checks[check] = False
    elif reference != candidate:
        checks[check] = False
        _add_reason(reasons_by_id, request_id, mismatch_code)


def _add_reason(
    reasons_by_id: dict[str, list[dict[str, Any]]],
    request_id: str,
    code: str,
    **details: Any,
) -> None:
    reason = {"code": code, **details}
    reasons = reasons_by_id.setdefault(request_id, [])
    if reason not in reasons:
        reasons.append(reason)


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
