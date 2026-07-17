"""Cross-language canonical-fingerprint vectors (Stage 1 acceptance pin).

Python half of the byte-for-byte execution-identity agreement. The batches
here mirror, value for value, the constructors in
``crates/foundation/core/src/execution_identity_vectors.rs``; both sides must
reproduce the digests pinned in
``crates/protocol/vocab/execution_fingerprint.toml``. Skips (rather than
fails) when the vocab fixture is absent, e.g. a Python-only checkout.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from uniserve_worker.contracts.execution import (
    CacheLease,
    CandidateVerification,
    EncodeStep,
    EngineRef,
    ExecuteBatch,
    ExecuteRow,
    FlowStep,
    NewSession,
    OperationTag,
    ProductLease,
    ProductLifetime,
    RepresentationKind,
    SamplingSpec,
    SequenceStep,
    SessionRef,
    TransferKind,
    canonical_payload_fingerprint,
    validate_execute_batch,
)

pytestmark = pytest.mark.contract

_ENGINE = EngineRef(deployment_id=1, engine_epoch=7)


def _session(request_id: int, incarnation: int, version: int) -> SessionRef:
    return SessionRef(
        engine=_ENGINE,
        request_id=request_id,
        incarnation=incarnation,
        session_version=version,
    )


def _shared_vector(name: str) -> ExecuteBatch:
    if name == "single_sequence_row":
        return ExecuteBatch(
            engine_epoch=7,
            step_id=9,
            acknowledged_through=4,
            rows=(
                ExecuteRow(
                    row_id=0,
                    session=_session(41, 1, 3),
                    operation=SequenceStep((5, 6, 7), 12, 12, 1),
                    admission=None,
                    cache_leases=(),
                    product_leases=(),
                    scheduler_op_id=100,
                ),
            ),
        )
    if name == "admission_verification_and_leases":
        return ExecuteBatch(
            engine_epoch=7,
            step_id=10,
            acknowledged_through=9,
            rows=(
                ExecuteRow(
                    row_id=0,
                    session=_session(55, 2, 0),
                    operation=SequenceStep(
                        (),
                        8,
                        8,
                        1,
                        CandidateVerification((11, 12, 13), (8, 9, 10)),
                    ),
                    admission=NewSession(
                        request_id=55,
                        incarnation=2,
                        sampling=SamplingSpec(0.0, 1.0, 20, 0.0, 1.0, 0.0, 0.0),
                        base_seed=42,
                        max_history_tokens=4096,
                    ),
                    cache_leases=(
                        CacheLease(5, 7, bytes(32), 1, 128, 2, 77),
                    ),
                    product_leases=(
                        ProductLease(
                            lease_id=9,
                            schema_id=3,
                            producer=_session(50, 1, 4),
                            product_version=2,
                            extent_rows=64,
                            lifetime=ProductLifetime.REQUEST,
                            transfer=TransferKind.LOCAL_RESIDENCY,
                        ),
                    ),
                    scheduler_op_id=200,
                ),
            ),
        )
    if name == "flow_and_encode_rows":
        return ExecuteBatch(
            engine_epoch=7,
            step_id=11,
            acknowledged_through=10,
            rows=(
                ExecuteRow(
                    row_id=0,
                    session=_session(60, 1, 5),
                    operation=FlowStep(1, 17, 50, 3, (4.0, 1.0, 1.5), (1, 2), 5),
                    admission=None,
                    cache_leases=(),
                    product_leases=(),
                    scheduler_op_id=300,
                ),
                ExecuteRow(
                    row_id=1,
                    session=_session(61, 1, 6),
                    operation=EncodeStep(
                        RepresentationKind.IMAGE_PATCH, 8, 9, (1, 64, 36)
                    ),
                    admission=None,
                    cache_leases=(),
                    product_leases=(),
                    scheduler_op_id=301,
                ),
            ),
        )
    raise AssertionError(f"unknown shared vector {name}")


def _pinned_digests() -> list[tuple[str, str]]:
    for parent in Path(__file__).resolve().parents:
        fixture = parent / "crates" / "protocol" / "vocab" / "execution_fingerprint.toml"
        if fixture.exists():
            schema = tomllib.loads(fixture.read_text(encoding="utf-8"))
            return [(entry["name"], entry["digest"]) for entry in schema["vector"]]
    pytest.skip("execution fingerprint fixture not present in this checkout")


def test_shared_vectors_match_the_pinned_cross_language_digests():
    pinned = _pinned_digests()
    assert pinned, "vocab fixture lists no vectors"
    for name, expected in pinned:
        batch = _shared_vector(name)
        validate_execute_batch(batch, advertised_operations=frozenset(OperationTag))
        assert canonical_payload_fingerprint(batch) == expected, (
            f"fingerprint drift for shared vector {name}"
        )
