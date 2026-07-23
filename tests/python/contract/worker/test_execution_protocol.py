"""Canonical scheduler-to-worker protocol conformance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
    EncodeKind,
    EncodeOperation,
    FlowAdmission,
    FlowDelta,
    FlowOperation,
    Guidance,
    ImageParams,
    InlineImage,
    KvLeaseDelta,
    LatentProduct,
    MaterializeKind,
    MaterializeOperation,
    OperationEnvelope,
    OperationResult,
    PublishedProduct,
    SequenceAdmission,
    SequenceDelta,
    SequenceEffect,
    SequenceMode,
    SequenceOperation,
    TokenInput,
    TokenPolicy,
    TransferKind,
    TransferOperation,
)
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.server.app import WorkerServer, dispatch

pytestmark = pytest.mark.contract

_RESPONSE_FIELDS = {
    "kind",
    "call_id",
    "capabilities",
    "result",
    "metrics",
    "pressure",
    "message",
    "code",
    "retryable",
    "fatal",
    "phase",
    "route",
    "operations",
    "snapshot",
}


def _operations() -> tuple[tuple[Admission, object], ...]:
    return (
        (
            Admission.create(1, sequence=SequenceAdmission()),
            SequenceOperation(
                SequenceMode.EXTEND,
                KvLeaseDelta(new_blocks=(1,)),
                (0, 2),
                TokenPolicy(),
                TokenInput((3, 4)),
            ),
        ),
        (
            Admission.create(2, flow=FlowAdmission(ImageParams(height=16, width=16))),
            FlowOperation(
                latent_handle=7,
                position=0,
                start_step=0,
                step_count=1,
                conditioning_position=0,
                conditioning=None,
                guidance=Guidance(1, 1.0, 1.0, "none", 0.0, (0.0, 1.0)),
                image_prompt="",
            ),
        ),
        (
            Admission.create(3, sequence=SequenceAdmission()),
            EncodeOperation(
                EncodeKind.VISION,
                KvLeaseDelta(),
                (0, 1),
                0,
                InlineImage("aW1hZ2U=", 9),
            ),
        ),
        (
            Admission.create(4, flow=FlowAdmission(ImageParams(height=16, width=16))),
            MaterializeOperation(
                MaterializeKind.IMAGE,
                KvLeaseDelta(),
                0,
                0,
                TokenPolicy(),
                LatentProduct(11),
            ),
        ),
        (
            Admission.create(5, sequence=SequenceAdmission()),
            TransferOperation(
                TransferKind.PRODUCT,
                KvLeaseDelta(),
                0,
                0,
                TokenPolicy(),
                PublishedProduct(13, "runtime-locator"),
            ),
        ),
    )


def _batch() -> Batch:
    values = _operations()
    admissions = tuple(admission for admission, _operation in values)
    operations = tuple(
        OperationEnvelope.create(
            session_id=admission.session_id,
            epoch=1,
            op_id=100 + admission.session_id,
            base_version=0,
            admission_digest=admission.digest,
            model_spec_digest="a" * 64,
            weight_digest="b" * 64,
            operation=operation,  # type: ignore[arg-type]
        )
        for admission, operation in values
    )
    return Batch(step_id=17, admissions=admissions, projections=(), operations=operations)


def test_closed_operation_union_round_trips_exactly():
    batch = _batch()

    assert Batch.from_wire(batch.to_wire()) == batch


def test_worker_response_variants_project_the_complete_typed_schema():
    worker = execution_worker()
    capabilities = dispatch(worker, {"kind": "get_capabilities"})
    error = WorkerServer(worker, None).handle({"kind": "unknown"})

    assert set(capabilities) == _RESPONSE_FIELDS
    assert capabilities["operations"] == []
    assert set(error) == _RESPONSE_FIELDS
    assert error["kind"] == "error"


def test_unknown_protocol_and_union_variants_fail_during_parse():
    protocol = deepcopy(_batch().to_wire())
    protocol["protocol_version"] = 2
    with pytest.raises(WorkerError, match="unsupported execution protocol"):
        Batch.from_wire(protocol)

    variant = deepcopy(_batch().to_wire())
    variant["operations"][0]["operation"]["kind"] = "unknown_sequence"
    with pytest.raises(WorkerError, match="unknown variant"):
        Batch.from_wire(variant)


def test_tampered_digest_fails_before_a_batch_is_constructed():
    value = deepcopy(_batch().to_wire())
    value["operations"][0]["digest"] = "0" * 64

    with pytest.raises(WorkerError, match="digest mismatch"):
        Batch.from_wire(value)


def test_result_variant_and_version_must_match_the_operation():
    operation = _batch().operations[0]
    valid = OperationResult.for_operation(
        operation,
        SequenceDelta(SequenceEffect(sampled_token_ids=(8,), kv_tokens=2)),
    )
    valid.validate_for(operation)

    with pytest.raises(WorkerError, match="identity or version"):
        replace(valid, result_version=valid.result_version + 1).validate_for(operation)
    with pytest.raises(WorkerError, match="delta variant"):
        replace(valid, delta=FlowDelta(1, False)).validate_for(operation)
