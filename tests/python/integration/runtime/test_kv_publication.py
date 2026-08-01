from __future__ import annotations

import json

from tests.python.fixtures.depth_one import (
    commit_resolved,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
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
    TokenMode,
    TransferMode,
    VersionRef,
    Work,
)
from uniserve_worker.runtime.kv_store import KvSnapshot


def _publication_operation(
    admission: Admission,
    *,
    op_id: int,
    parent: VersionRef,
    control_seq: int,
) -> tuple[Operation, ProductRef]:
    request_key = admission.request_key
    product = ProductRef(
        request_key=request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 10 + 1,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
        dtype=DType.U8,
        shape_bound=ShapeBound((DeviceDim(1 << 20),)),
        point_range=PointRange(),
    )
    return (
        Operation.registered(
            request_key=request_key,
            op_id=op_id,
            parent=parent,
            work=Work("transfer", TransferMode.KV_PUBLISH.value),
            route=0,
            domain=Domain.UND,
            bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
            outputs=(product,),
            control_seq=control_seq,
        ),
        product,
    )


def test_tail_closure_precedes_exact_incremental_publication() -> None:
    worker = execution_worker()
    admission = und_admission(41, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        Batch(
            step_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )
    first_commit = commit_resolved(worker.sessions.get(41))
    closure_template, closure_input = token_operation(
        admission.request_key,
        op_id=2,
        parent=first_commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(5,),
        control_seq=first_commit.control_seq,
    )
    closure = Operation.registered(
        request_key=closure_template.request_key,
        op_id=closure_template.op_id,
        parent=closure_template.parent,
        work=closure_template.work,
        route=closure_template.route,
        domain=closure_template.domain,
        bounds=closure_template.bounds,
        inputs=closure_template.inputs,
        outputs=(),
        control_seq=closure_template.control_seq,
    )
    closure_result = worker.execute(
        Batch(
            step_id=2,
            admissions=(),
            operations=(closure,),
            controls=(first_commit,),
            input_products=(closure_input,),
        )
    )
    closure_record = closure_result.completions[0]
    assert closure_record.logical_lengths.token_len == 2
    assert closure_record.logical_lengths.kv_visible_len == 3
    assert closure_record.logical_lengths.kv_initialized_len == 3

    second_commit = commit_resolved(worker.sessions.get(41))
    publication, publication_product = _publication_operation(
        admission,
        op_id=3,
        parent=second_commit.selected,
        control_seq=second_commit.control_seq,
    )
    publication_result = worker.execute(
        Batch(
            step_id=3,
            admissions=(),
            operations=(publication,),
            controls=(second_commit,),
        )
    )

    assert publication_result.completions[0].selected_point == 0
    assert publication_result.completions[0].logical_lengths.kv_published_len == 3
    payload = publication_result.products[0]
    snapshot = KvSnapshot.from_wire(json.loads(payload.payload))
    assert payload.product == publication_product
    assert snapshot.source_version == second_commit.selected
    assert snapshot.base_extent == 0
    assert snapshot.published_extent == 3
    assert worker.kv.validate_conditioning(41, publication_product) == snapshot

    suffix_template, suffix_input = token_operation(
        admission.request_key,
        op_id=4,
        parent=second_commit.selected,
        mode=TokenMode.EXTEND,
        tokens=(6,),
        control_seq=second_commit.control_seq,
    )
    suffix_closure = Operation.registered(
        request_key=suffix_template.request_key,
        op_id=suffix_template.op_id,
        parent=suffix_template.parent,
        work=suffix_template.work,
        route=suffix_template.route,
        domain=suffix_template.domain,
        bounds=suffix_template.bounds,
        inputs=suffix_template.inputs,
        outputs=(),
        control_seq=suffix_template.control_seq,
    )
    suffix_result = worker.execute(
        Batch(
            step_id=4,
            admissions=(),
            operations=(suffix_closure,),
            input_products=(suffix_input,),
        )
    )
    assert suffix_result.completions[0].logical_lengths.kv_visible_len == 4

    suffix_commit = commit_resolved(worker.sessions.get(41))
    incremental, incremental_product = _publication_operation(
        admission,
        op_id=5,
        parent=suffix_commit.selected,
        control_seq=suffix_commit.control_seq,
    )
    incremental_result = worker.execute(
        Batch(
            step_id=5,
            admissions=(),
            operations=(incremental,),
            controls=(suffix_commit,),
        )
    )
    incremental_payload = incremental_result.products[0]
    incremental_snapshot = KvSnapshot.from_wire(json.loads(incremental_payload.payload))
    assert incremental_payload.product == incremental_product
    assert incremental_snapshot.source_version == suffix_commit.selected
    assert incremental_snapshot.base_version == second_commit.selected
    assert incremental_snapshot.base_extent == 3
    assert incremental_snapshot.published_extent == 4
    assert worker.kv.validate_conditioning(41, publication_product) == snapshot
    assert worker.kv.validate_conditioning(41, incremental_product) == incremental_snapshot
