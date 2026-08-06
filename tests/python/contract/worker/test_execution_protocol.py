"""Canonical scheduler-to-worker protocol conformance at the worker server."""

from __future__ import annotations

import pytest

from tests.python.fixtures.depth_one import materialize_operation, root_parent, und_admission
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Bounds,
    DeviceDim,
    Domain,
    DType,
    Operation,
    PointRange,
    ProductKind,
    ProductRef,
    ShapeBound,
    StorageClass,
    Work,
    WorkVariant,
)
from uniserve_worker.capabilities import operation_type
from uniserve_worker.server.app import WorkerServer, dispatch
from uniserve_worker.spec import OperationType
from uniserve_worker.worker.protocol import WorkerContract, model_free_capabilities

pytestmark = pytest.mark.contract

_RESPONSE_FIELDS = {
    "kind",
    "call_id",
    "capabilities",
    "completion_report",
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


def test_worker_response_variants_project_the_complete_typed_schema():
    worker = execution_worker()
    capabilities = dispatch(worker, {"kind": "get_capabilities"})
    error = WorkerServer(worker, None).handle({"kind": "unknown"})

    assert set(capabilities) == _RESPONSE_FIELDS
    assert capabilities["operations"] == []
    assert set(error) == _RESPONSE_FIELDS
    assert error["kind"] == "error"


def test_worker_contract_separates_executable_and_advertised_work():
    declared = model_free_capabilities(
        block_size=64,
        supported_work=(WorkVariant.TOKEN_EXTEND, WorkVariant.TOKEN_DECODE),
        pipeline_depth=2,
        completion_payload_bytes=1 << 20,
    )
    operations = frozenset(
        {
            OperationType.SEQUENCE_EXTEND,
            OperationType.SEQUENCE_DECODE,
            OperationType.MATERIALIZE_FRAME,
        }
    )

    contract = WorkerContract.compile(
        declared,
        allowed_operation_types=operations,
        implemented_operation_types=operations,
        pipeline_depth=2,
        owner="ExecutionWorker",
    )

    expected = (WorkVariant.TOKEN_EXTEND, WorkVariant.TOKEN_DECODE)
    assert contract.capabilities.supported_work == expected
    assert (
        contract.capabilities.execution_constraints.route_capabilities[0].supported_work == expected
    )
    assert contract.effective_operation_types == operations


def test_materialization_operation_type_is_fixed_by_its_registered_input_product():
    admission = und_admission(17)
    rk = admission.request_key
    parent = root_parent(admission)
    latent = ProductRef(
        request_key=rk,
        producer_op_id=1,
        output_index=0,
        generation=1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(768),)),
        point_range=PointRange(),
    )
    image = materialize_operation(rk, op_id=2, parent=parent, latent=latent)
    transported_frame = ProductRef(
        request_key=rk,
        producer_op_id=3,
        output_index=0,
        generation=2,
        kind=ProductKind.ARTIFACT,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(768),)),
        point_range=PointRange(),
    )
    frame = Operation.registered(
        request_key=rk,
        op_id=4,
        parent=parent,
        work=Work("materialize"),
        route=0,
        domain=Domain.GEN,
        bounds=Bounds(max_completion_bytes=1 << 20),
        inputs=(transported_frame,),
    )

    assert operation_type(image) is OperationType.MATERIALIZE_IMAGE
    assert operation_type(frame) is OperationType.MATERIALIZE_FRAME
