"""Python projections of the canonical cross-language vocabularies."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from uniserve_worker.contracts.op_kinds import OP_KIND_TABLE, OP_KINDS
from uniserve_worker.foundation.errors import _POLICY, ErrorCode
from uniserve_worker.server.worker_kind import WorkerKind

pytestmark = pytest.mark.contract


def _vocab_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "crates" / "protocol" / "vocab"
        if candidate.is_dir():
            return candidate
    return None


def _load(name: str) -> dict:
    vocab = _vocab_dir()
    if vocab is None:
        pytest.skip("canonical vocab schema not present in this checkout")
    path = vocab / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_op_kinds_match_canonical_schema():
    """Operation wire strings and modes match the canonical schema."""
    schema = _load("op_kinds.toml")
    schema_wire = {op["wire"] for op in schema["op"]}
    assert schema_wire == set(OP_KINDS), (
        f"op-kind set drift: schema={sorted(schema_wire)} python={sorted(OP_KINDS)}"
    )
    schema_modes = {op["wire"]: op["mode"] for op in schema["op"]}
    python_modes = {wire: spec.mode for wire, spec in OP_KIND_TABLE.items()}
    assert schema_modes == python_modes, (
        f"op-kind mode drift: schema={schema_modes} python={python_modes}"
    )


def test_worker_kinds_match_canonical_schema():
    """Python WorkerKind tokens and operation envelopes match the schema."""
    schema = _load("worker_kinds.toml")
    entries = schema["kind"]

    schema_tokens = {k["token"] for k in entries}
    python_tokens = set(WorkerKind.wire_values())
    assert schema_tokens == python_tokens, (
        f"worker-kind set drift: schema={sorted(schema_tokens)} python={sorted(python_tokens)}"
    )

    schema_ops = {k["token"]: set(k["supported_ops"]) for k in entries}
    python_ops = {worker_kind.value: set(worker_kind.supported_ops) for worker_kind in WorkerKind}
    assert schema_ops == python_ops, f"supported_ops drift: schema={schema_ops} python={python_ops}"


def test_error_taxonomy_matches_canonical_schema():
    """Error codes and policies match the canonical schema."""
    schema = _load("worker_errors.toml")
    entries = schema["error"]

    # Wire-string code set matches the ErrorCode values.
    schema_codes = {e["code"] for e in entries}
    python_codes = {str(c) for c in ErrorCode}
    assert schema_codes == python_codes, (
        f"error-code set drift: schema={sorted(schema_codes)} python={sorted(python_codes)}"
    )

    # Each schema member's `python` column names the ErrorCode whose value is its
    # `code`, pinning the member identifiers.
    for e in entries:
        member = ErrorCode[e["python"]]
        assert str(member) == e["code"], (
            f"ErrorCode.{e['python']} value {member!r} != schema code {e['code']!r}"
        )

    # Policy bits (retryable, fatal, capture_trace) per code.
    schema_policy = {e["code"]: (e["retryable"], e["fatal"], e["capture_trace"]) for e in entries}
    python_policy = {
        str(code): (pol.retryable, pol.fatal, pol.capture_trace) for code, pol in _POLICY.items()
    }
    assert schema_policy == python_policy, (
        f"error-policy drift: schema={schema_policy} python={python_policy}"
    )
