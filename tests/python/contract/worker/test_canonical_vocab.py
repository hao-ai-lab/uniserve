"""Drift guard: Python vocabularies vs the canonical cross-language schema.

The single source of truth lives in ``crates/protocol/vocab/*.toml``. Each Rust
crate has a ``#[test]`` asserting its own enums match those files; this is the
Python half. Adding (or renaming/repolicy-ing) an op kind, worker kind, or error
class now requires updating the schema AND every language, or one of these tests
fails.

The tests skip (rather than fail) when the schema files are absent — e.g. a
Python-only checkout/wheel — so they are meaningful in the full repo and inert
elsewhere.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from uniserve_worker.contracts.op_kinds import OP_KIND_TABLE, OP_KINDS
from uniserve_worker.foundation.errors import _POLICY, ErrorCode
from uniserve_worker.server.worker_kind import (
    RUNNER_BACKED_KINDS,
    SUPPORTED_OPS,
    WORKER_KINDS,
)

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
    """Item 25: Python OP_KIND_TABLE == op_kinds.toml (wire strings + modes)."""
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
    """Item 21: Python WORKER_KINDS/SUPPORTED_OPS/RUNNER_BACKED_KINDS == worker_kinds.toml."""
    schema = _load("worker_kinds.toml")
    entries = schema["kind"]

    schema_tokens = {k["token"] for k in entries}
    assert schema_tokens == set(WORKER_KINDS), (
        f"worker-kind set drift: schema={sorted(schema_tokens)} "
        f"python={sorted(WORKER_KINDS)}"
    )

    schema_ops = {k["token"]: set(k["supported_ops"]) for k in entries}
    python_ops = {token: set(ops) for token, ops in SUPPORTED_OPS.items()}
    assert schema_ops == python_ops, (
        f"supported_ops drift: schema={schema_ops} python={python_ops}"
    )

    schema_runner_backed = {k["token"] for k in entries if k["runner_backed"]}
    assert schema_runner_backed == set(RUNNER_BACKED_KINDS), (
        f"runner-backed drift: schema={sorted(schema_runner_backed)} "
        f"python={sorted(RUNNER_BACKED_KINDS)}"
    )


def test_error_taxonomy_matches_canonical_schema():
    """Item 26: Python ErrorCode + _POLICY == worker_errors.toml (codes + policy)."""
    schema = _load("worker_errors.toml")
    entries = schema["error"]

    # Wire-string code set matches the ErrorCode values.
    schema_codes = {e["code"] for e in entries}
    python_codes = {str(c) for c in ErrorCode}
    assert schema_codes == python_codes, (
        f"error-code set drift: schema={sorted(schema_codes)} "
        f"python={sorted(python_codes)}"
    )

    # Each schema member's `python` column names the ErrorCode whose value is its
    # `code`, pinning the member identifiers.
    for e in entries:
        member = ErrorCode[e["python"]]
        assert str(member) == e["code"], (
            f"ErrorCode.{e['python']} value {member!r} != schema code {e['code']!r}"
        )

    # Policy bits (retryable, fatal, capture_trace) per code.
    schema_policy = {
        e["code"]: (e["retryable"], e["fatal"], e["capture_trace"]) for e in entries
    }
    python_policy = {
        str(code): (pol.retryable, pol.fatal, pol.capture_trace)
        for code, pol in _POLICY.items()
    }
    assert schema_policy == python_policy, (
        f"error-policy drift: schema={schema_policy} python={python_policy}"
    )
