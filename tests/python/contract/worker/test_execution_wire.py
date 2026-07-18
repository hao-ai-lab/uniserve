"""Execution wire envelope: round trips, pins, and fail-closed decoding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.python.contract.worker.test_execution_contracts_vectors import (
    _shared_vector,
)
from uniserve_worker.contracts.execution import OperationTag, validate_execute_batch
from uniserve_worker.contracts.execution_wire import (
    WireDecodeError,
    decode_execute_batch,
    encode_execute_batch,
)

pytestmark = pytest.mark.contract


def _pinned() -> dict[str, str]:
    for parent in Path(__file__).resolve().parents:
        fixture = parent / "crates" / "protocol" / "vocab" / "execution_wire.json"
        if fixture.exists():
            return json.loads(fixture.read_text(encoding="utf-8"))
    pytest.skip("execution wire fixture not present in this checkout")


def test_shared_vectors_round_trip_and_match_the_pinned_encoding():
    pinned = _pinned()
    assert pinned
    for name, wire in pinned.items():
        batch = _shared_vector(name)
        assert encode_execute_batch(batch) == wire, name
        decoded = decode_execute_batch(wire)
        assert decoded == batch, name
        validate_execute_batch(decoded, advertised_operations=frozenset(OperationTag))


def test_unknown_schema_majors_and_fields_fail_closed():
    wire = encode_execute_batch(_shared_vector("single_sequence_row"))
    with pytest.raises(WireDecodeError, match="unsupported execution wire schema"):
        decode_execute_batch(wire.replace('"schema_major":1', '"schema_major":9'))
    with pytest.raises(WireDecodeError, match="unknown fields"):
        decode_execute_batch(
            wire.replace('"engine_epoch":7,"step_id"', '"engine_epoch":7,"x":1,"step_id"')
        )
    with pytest.raises(WireDecodeError, match="exactly one sealed variant"):
        mangled = json.loads(wire)
        mangled["batch"]["rows"][0]["operation"]["extra"] = {}
        decode_execute_batch(json.dumps(mangled, separators=(",", ":")))
    with pytest.raises(WireDecodeError, match="malformed"):
        decode_execute_batch("not json")
