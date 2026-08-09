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
from uniserve_worker.models.runtime import LoweredStage, RowKind
from uniserve_worker.server.app import WorkerServer, dispatch
from uniserve_worker.server.stub import StubModel

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


class _MaterializingModel(StubModel):
    def lower(
        self,
        variant: WorkVariant,
        *,
        retain_image: bool = False,
    ) -> tuple[LoweredStage, ...]:
        del retain_image
        if variant is WorkVariant.MATERIALIZE:
            return (LoweredStage("encode", RowKind.DECODE),)
        return super().lower(variant)


def test_materialization_stages_are_fixed_by_its_registered_input_product():
    executor = execution_worker(_MaterializingModel()).executor
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

    # A latent-backed materialize invokes the model image path; a transported
    # frame is already materialized and needs no neural call.
    assert executor._operation_stages_for(image) == executor._stages(WorkVariant.MATERIALIZE)
    assert executor._operation_stages_for(image)
    assert executor._operation_stages_for(frame) == ()
