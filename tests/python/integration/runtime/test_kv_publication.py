from __future__ import annotations

import time
from dataclasses import replace

from tests.python.fixtures.depth_one import (
    commit_resolved,
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.depth_one import (
    kv_publication_operation as _publication_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Bounds,
    DeviceDim,
    Domain,
    DType,
    KvPlacement,
    Operation,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    Release,
    ShapeBound,
    StorageClass,
    TokenMode,
    TransferMode,
    VersionRef,
    Work,
)
from uniserve_worker.runtime.transfer import (
    Locator,
    decode_transfer_descriptor,
    encode_transfer_descriptor,
)
from uniserve_worker.transfer.cache import CachePublication


def _installation_operation(
    admission: Admission,
    *,
    op_id: int,
    parent: VersionRef,
    source: ProductRef,
) -> tuple[Operation, ProductRef]:
    product = ProductRef(
        request_key=admission.request_key,
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
            request_key=admission.request_key,
            op_id=op_id,
            parent=parent,
            work=Work("transfer", TransferMode.KV_INSTALL.value),
            route=0,
            domain=Domain.PREFILL,
            bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
            inputs=(source,),
            outputs=(product,),
            kv_capacity_pages=1,
        ),
        product,
    )


def _installation_placement(operation: Operation, length: int) -> KvPlacement:
    return KvPlacement(
        request_key=operation.request_key,
        op_id=operation.op_id,
        group_id=0,
        block_table=(1,),
        pages_to_zero=(1,),
        prefix_length=0,
        input_length=int(length),
        visible_length=0,
        resulting_length=int(length),
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
        execution_batch(
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
        kv_capacity_pages=closure_template.kv_capacity_pages,
        control_seq=closure_template.control_seq,
    )
    closure_result = worker.execute(
        execution_batch(
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
        admission.request_key,
        op_id=3,
        parent=second_commit.selected,
        control_seq=second_commit.control_seq,
    )
    publication_result = worker.execute(
        execution_batch(
            step_id=3,
            admissions=(),
            operations=(publication,),
            controls=(second_commit,),
        )
    )

    assert publication_result.completions[0].selected_point == 0
    assert publication_result.completions[0].logical_lengths.kv_published_len == 3
    payload = publication_result.products[0]
    kind, descriptor, producer_plan_digest = decode_transfer_descriptor(payload.payload)
    assert kind == "kv"
    assert producer_plan_digest == publication.plan_digest
    snapshot = CachePublication.from_wire(descriptor["snapshot"])
    assert payload.product == publication_product
    assert snapshot.source_version == second_commit.selected
    assert snapshot.base_extent == 0
    assert snapshot.published_extent == 3

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
        kv_capacity_pages=suffix_template.kv_capacity_pages,
        control_seq=suffix_template.control_seq,
    )
    suffix_result = worker.execute(
        execution_batch(
            step_id=4,
            admissions=(),
            operations=(suffix_closure,),
            input_products=(suffix_input,),
        )
    )
    assert suffix_result.completions[0].logical_lengths.kv_visible_len == 4

    suffix_commit = commit_resolved(worker.sessions.get(41))
    incremental, incremental_product = _publication_operation(
        admission.request_key,
        op_id=5,
        parent=suffix_commit.selected,
        control_seq=suffix_commit.control_seq,
    )
    incremental_result = worker.execute(
        execution_batch(
            step_id=5,
            admissions=(),
            operations=(incremental,),
            controls=(suffix_commit,),
        )
    )
    incremental_payload = incremental_result.products[0]
    kind, descriptor, producer_plan_digest = decode_transfer_descriptor(incremental_payload.payload)
    assert kind == "kv"
    assert producer_plan_digest == incremental.plan_digest
    incremental_snapshot = CachePublication.from_wire(descriptor["snapshot"])
    assert incremental_payload.product == incremental_product
    assert incremental_snapshot.source_version == suffix_commit.selected
    assert incremental_snapshot.base_version == second_commit.selected
    assert incremental_snapshot.base_extent == 3
    assert incremental_snapshot.published_extent == 4


def test_cross_stage_kv_install_uses_query_ready_exact_snapshot() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    released_consumer = execution_worker(transfer_backend="shm")
    admission = und_admission(42, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    try:
        producer.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
        commit = commit_resolved(producer.sessions.get(42))
        publication, source = _publication_operation(
            admission.request_key,
            op_id=2,
            parent=commit.selected,
            control_seq=commit.control_seq,
        )
        published = producer.execute(
            execution_batch(
                step_id=2,
                operations=(publication,),
                controls=(commit,),
            )
        )
        assert len(published.products) == 1
        installation, installed = _installation_operation(
            admission,
            op_id=3,
            parent=root_parent(admission),
            source=source,
        )
        batch = execution_batch(
            step_id=3,
            admissions=(admission,),
            operations=(installation,),
            input_products=(published.products[0],),
            kv_placements=(_installation_placement(installation, 2),),
        )
        prepared = consumer.prepare_execute(batch)
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()
        report = consumer.execute_prepared(prepared)
        assert report.completions[0].status.value == "ok"
        assert report.completions[0].logical_lengths.kv_visible_len == 2
        assert report.products[0].product == installed
        producer.execute(
            execution_batch(
                step_id=4,
                controls=(Release(admission.request_key, publication.op_id),),
            )
        )
        expired_install, _ = _installation_operation(
            admission,
            op_id=4,
            parent=root_parent(admission),
            source=source,
        )
        expired = released_consumer.prepare_execute(
            execution_batch(
                step_id=5,
                admissions=(admission,),
                operations=(expired_install,),
                input_products=(published.products[0],),
                kv_placements=(_installation_placement(expired_install, 2),),
            )
        )
        assert expired is not None
        deadline = time.monotonic() + 5.0
        while not expired.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert expired.ready()
        expired_report = released_consumer.execute_prepared(expired)
        assert expired_report.completions[0].status.value == "error"
    finally:
        producer.close()
        consumer.close()
        released_consumer.close()


def test_failed_cross_stage_kv_read_preserves_source_and_rolls_back_destination() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = und_admission(43, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    try:
        producer.execute(
            execution_batch(
                step_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
        commit = commit_resolved(producer.sessions.get(43))
        publication, source = _publication_operation(
            admission.request_key,
            op_id=2,
            parent=commit.selected,
            control_seq=commit.control_seq,
        )
        published = producer.execute(
            execution_batch(step_id=2, operations=(publication,), controls=(commit,))
        ).products[0]
        kind, value, producer_digest = decode_transfer_descriptor(published.payload)
        snapshot = CachePublication.from_wire(value["snapshot"])
        first = Locator.from_wire_json(snapshot.locators[0])
        missing = replace(first, handle=b"uniserve-missing-transfer-segment")
        broken = replace(
            snapshot,
            locators=(missing.to_wire_json(), *snapshot.locators[1:]),
        )
        payload = ProductPayload(
            product=published.product,
            payload=encode_transfer_descriptor(
                kind,
                {"snapshot": broken.to_wire()},
                producer_digest,
            ),
        )
        installation, _installed = _installation_operation(
            admission,
            op_id=3,
            parent=root_parent(admission),
            source=source,
        )
        prepared = consumer.prepare_execute(
            execution_batch(
                step_id=3,
                admissions=(admission,),
                operations=(installation,),
                input_products=(payload,),
                kv_placements=(_installation_placement(installation, 2),),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()

        report = consumer.execute_prepared(prepared)

        assert report.completions[0].status.value == "error"
        assert report.completions[0].error_code is not None

        retry, installed = _installation_operation(
            admission,
            op_id=4,
            parent=root_parent(admission),
            source=source,
        )
        prepared_retry = consumer.prepare_execute(
            execution_batch(
                step_id=4,
                admissions=(admission,),
                operations=(retry,),
                input_products=(published,),
                kv_placements=(_installation_placement(retry, 2),),
            )
        )
        assert prepared_retry is not None
        deadline = time.monotonic() + 5.0
        while not prepared_retry.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared_retry.ready()
        retry_report = consumer.execute_prepared(prepared_retry)
        assert retry_report.completions[0].status.value == "ok"
        assert retry_report.completions[0].logical_lengths.kv_visible_len == 2
        assert retry_report.products[0].product == installed
    finally:
        producer.close()
        consumer.close()
