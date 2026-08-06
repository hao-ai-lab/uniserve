"""Canonical scheduler-to-worker protocol conformance at the worker server."""

from __future__ import annotations

from dataclasses import replace

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
from uniserve_worker.server.app import WorkerServer, dispatch
from uniserve_worker.server.stub import StubModel
from uniserve_worker.spec import (
    OperationSpec,
    OperationStageCondition,
    OperationStagePurpose,
    OperationStageSpec,
    RouteRowKind,
)
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
    work = frozenset(
        {
            WorkVariant.TOKEN_EXTEND,
            WorkVariant.TOKEN_DECODE,
            WorkVariant.MATERIALIZE,
        }
    )

    contract = WorkerContract.compile(
        declared,
        allowed_work_variants=work,
        implemented_work_variants=work,
        pipeline_depth=2,
        owner="ExecutionWorker",
    )

    expected = (WorkVariant.TOKEN_EXTEND, WorkVariant.TOKEN_DECODE)
    assert contract.capabilities.supported_work == expected
    assert (
        contract.capabilities.execution_constraints.route_capabilities[0].supported_work == expected
    )
    assert contract.effective_work_variants == work


def _materialize_state_model() -> StubModel:
    """A stub whose materialize op declares a retained-image state stage."""

    model = StubModel()
    operations = tuple(
        OperationSpec(
            WorkVariant.MATERIALIZE,
            (
                OperationStageSpec(
                    "stub",
                    RouteRowKind.FLOW,
                    OperationStagePurpose.STATE,
                    OperationStageCondition.RETAIN_IMAGE,
                ),
            ),
        )
        if operation.kind is WorkVariant.MATERIALIZE
        else operation
        for operation in model.spec.operations
    )
    model.spec = replace(model.spec, operations=operations)
    return model


def test_materialization_stages_are_fixed_by_its_registered_input_product():
    executor = execution_worker(_materialize_state_model()).executor
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

    # A latent-backed materialize lowers to the model's declared image stages;
    # a transported frame is model-free and lowers to no neural stage.
    assert executor._operation_stages_for(image) == executor._stages(WorkVariant.MATERIALIZE)
    assert executor._operation_stages_for(image)
    assert executor._operation_stages_for(frame) == ()
