from __future__ import annotations

import time
from dataclasses import replace

from tests.python.fixtures.depth_one import (
    ar_params,
    commit_for_completion,
    execution_run,
    finalized_report,
    root_parent,
    token_operation,
)
from tests.python.fixtures.depth_one import (
    kv_publication_operation as _publication_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    BlockTable,
    Bounds,
    CachePageAllocation,
    Checkpoint,
    DeviceDim,
    DType,
    Free,
    NewRequest,
    Operation,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RunKind,
    ShapeBound,
    StorageClass,
    TokenMode,
)
from uniserve_worker.transfer.connector import CachePublication
from uniserve_worker.transfer.tickets import (
    decode_transfer_handle,
    encode_transfer_handle,
)


def _installation_operation(
    admission: NewRequest,
    *,
    op_id: int,
    parent: Checkpoint,
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
            kind=RunKind.TRANSFER_KV_INSTALL,
            bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
            inputs=(source,),
            outputs=(product,),
        ),
        product,
    )


def _installation_placement(operation: Operation, length: int) -> dict[str, object]:
    request_pool_idx = int(operation.request_key.request_id) + 1
    return {
        "block_tables": (BlockTable(request_pool_idx, 0, (1,), max(1, int(length))),),
        "new_cache_pages": (CachePageAllocation(request_pool_idx, 0, (1,)),),
    }


def test_tail_closure_precedes_exact_incremental_publication() -> None:
    worker = execution_worker()
    admission = ar_params(41, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    first_result = worker.execute(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(extend,),
            input_products=(extend_input,),
        )
    )
    first_commit = commit_for_completion(extend, first_result)
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
        kind=closure_template.kind,
        bounds=closure_template.bounds,
        inputs=closure_template.inputs,
        outputs=(),
        control_seq=closure_template.control_seq,
    )
    closure_result = finalized_report(worker.execute(
        execution_run(
            run_id=2,
            admissions=(),
            operations=(closure,),
            commands=(first_commit,),
            input_products=(closure_input,),
        )
    ))
    closure_record = closure_result.completions[0]
    assert closure_record.logical_lengths.token_len == 2
    assert closure_record.logical_lengths.kv_visible_len == 3
    assert closure_record.logical_lengths.kv_computed_len == 3

    second_commit = commit_for_completion(closure, closure_result)
    publication, publication_product = _publication_operation(
        admission.request_key,
        op_id=3,
        parent=second_commit.selected,
        control_seq=second_commit.control_seq,
    )
    publication_result = finalized_report(
        worker.execute(
            execution_run(
                run_id=3,
                admissions=(),
                operations=(publication,),
                commands=(second_commit,),
            )
        )
    )

    assert publication_result.completions[0].selected_point == 0
    assert publication_result.completions[0].logical_lengths.kv_visible_len == 3
    assert (
        worker.execution.cache_publications.published_extent(admission.request_key.request_id) == 3
    )
    payload = publication_result.products[0]
    kind, descriptor = decode_transfer_handle(payload.payload)
    assert kind == "kv"
    snapshot = CachePublication.from_mapping(descriptor["snapshot"])
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
        kind=suffix_template.kind,
        bounds=suffix_template.bounds,
        inputs=suffix_template.inputs,
        outputs=(),
        control_seq=suffix_template.control_seq,
    )
    suffix_result = finalized_report(worker.execute(
        execution_run(
            run_id=4,
            admissions=(),
            operations=(suffix_closure,),
            input_products=(suffix_input,),
        )
    ))
    assert suffix_result.completions[0].logical_lengths.kv_visible_len == 4

    suffix_commit = commit_for_completion(suffix_closure, suffix_result)
    incremental, incremental_product = _publication_operation(
        admission.request_key,
        op_id=5,
        parent=suffix_commit.selected,
        control_seq=suffix_commit.control_seq,
    )
    incremental_result = finalized_report(
        worker.execute(
            execution_run(
                run_id=5,
                admissions=(),
                operations=(incremental,),
                commands=(suffix_commit,),
            )
        )
    )
    incremental_payload = incremental_result.products[0]
    kind, descriptor = decode_transfer_handle(incremental_payload.payload)
    assert kind == "kv"
    incremental_snapshot = CachePublication.from_mapping(descriptor["snapshot"])
    assert incremental_payload.product == incremental_product
    assert incremental_snapshot.source_version == suffix_commit.selected
    assert incremental_snapshot.base_version == second_commit.selected
    assert incremental_snapshot.base_extent == 3
    assert incremental_snapshot.published_extent == 4


def test_cross_stage_kv_install_uses_query_ready_exact_snapshot() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    released_consumer = execution_worker(transfer_backend="shm")
    admission = ar_params(42, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    try:
        extended = producer.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
        commit = commit_for_completion(extend, extended)
        publication, source = _publication_operation(
            admission.request_key,
            op_id=2,
            parent=commit.selected,
            control_seq=commit.control_seq,
        )
        published = finalized_report(
            producer.execute(
                execution_run(
                    run_id=2,
                    operations=(publication,),
                    commands=(commit,),
                )
            )
        )
        assert len(published.products) == 1
        installation, installed = _installation_operation(
            admission,
            op_id=3,
            parent=root_parent(admission),
            source=source,
        )
        batch = execution_run(
            run_id=3,
            admissions=(admission,),
            operations=(installation,),
            input_products=(published.products[0],),
            **_installation_placement(installation, 2),
        )
        prepared = consumer.prepare_execute(batch)
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared.ready()
        report = finalized_report(consumer.execute_prepared(prepared))
        assert report.completions[0].status.value == "ok"
        assert report.completions[0].logical_lengths.kv_visible_len == 2
        assert report.products[0].product == installed
        finalized_report(producer.execute(
            execution_run(
                run_id=4,
                commands=(Free(source.buffer_id),),
            )
        ))
        expired_install, _ = _installation_operation(
            admission,
            op_id=4,
            parent=root_parent(admission),
            source=source,
        )
        expired = released_consumer.prepare_execute(
            execution_run(
                run_id=5,
                admissions=(admission,),
                operations=(expired_install,),
                input_products=(published.products[0],),
                **_installation_placement(expired_install, 2),
            )
        )
        assert expired is not None
        deadline = time.monotonic() + 5.0
        while not expired.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert expired.ready()
        expired_report = finalized_report(released_consumer.execute_prepared(expired))
        assert expired_report.completions[0].status.value == "error"
    finally:
        producer.close()
        consumer.close()
        released_consumer.close()


def test_failed_cross_stage_kv_read_preserves_source_and_destination_state() -> None:
    producer = execution_worker(transfer_backend="shm")
    consumer = execution_worker(transfer_backend="shm")
    admission = ar_params(43, block_ids=(0,))
    extend, extend_input = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    try:
        extended = producer.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
                input_products=(extend_input,),
            )
        )
        commit = commit_for_completion(extend, extended)
        publication, source = _publication_operation(
            admission.request_key,
            op_id=2,
            parent=commit.selected,
            control_seq=commit.control_seq,
        )
        published = finalized_report(
            producer.execute(
                execution_run(run_id=2, operations=(publication,), commands=(commit,))
            )
        ).products[0]
        kind, value = decode_transfer_handle(published.payload)
        snapshot = CachePublication.from_mapping(value["snapshot"])
        first = snapshot.locators[0]
        missing = replace(first, handle=b"uniserve-missing-transfer-segment")
        broken = replace(
            snapshot,
            locators=(missing, *snapshot.locators[1:]),
        )
        payload = ProductPayload(
            product=published.product,
            payload=encode_transfer_handle(
                kind,
                {"generation": value["generation"], "snapshot": broken.to_mapping()},
            ),
        )
        installation, _installed = _installation_operation(
            admission,
            op_id=3,
            parent=root_parent(admission),
            source=source,
        )
        prepared = consumer.prepare_execute(
            execution_run(
                run_id=3,
                admissions=(admission,),
                operations=(installation,),
                input_products=(payload,),
                **_installation_placement(installation, 2),
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
            execution_run(
                run_id=4,
                admissions=(admission,),
                operations=(retry,),
                input_products=(published,),
                **_installation_placement(retry, 2),
            )
        )
        assert prepared_retry is not None
        deadline = time.monotonic() + 5.0
        while not prepared_retry.ready() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert prepared_retry.ready()
        retry_report = finalized_report(consumer.execute_prepared(prepared_retry))
        assert retry_report.completions[0].status.value == "ok"
        assert retry_report.completions[0].logical_lengths.kv_visible_len == 2
        assert retry_report.products[0].product == installed
    finally:
        producer.close()
        consumer.close()
