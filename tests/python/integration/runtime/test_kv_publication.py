from __future__ import annotations

import hashlib
import json
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import torch

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
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from uniserve_worker.execution.batch import (
    BlockTable,
    Bounds,
    CachePageAllocation,
    Checkpoint,
    CloseReason,
    DeviceDim,
    DType,
    Finish,
    Free,
    KvTransferValue,
    Locator,
    NewRequest,
    OpCode,
    Operation,
    OpStatus,
    PointRange,
    PosixShmTransfer,
    ProductKind,
    ProductPayload,
    ProductRef,
    RunResult,
    ShapeBound,
    StorageClass,
    TokenMode,
    TransferHandle,
)
from uniserve_worker.process import WorkerProcess


def _read_request(locator: Locator) -> bytes:
    """Encode a host reader's registration request at the external SHM boundary."""

    assert isinstance(locator.transport, PosixShmTransfer)
    descriptor = json.dumps(locator.to_mapping(), sort_keys=True, separators=(",", ":"))
    return (
        hashlib.sha256(locator.transport.name.encode()).digest()
        + hashlib.sha256(descriptor.encode()).digest()
    )


def test_kv_install_waits_for_storage_and_input_without_blocking_independent_work() -> None:
    producer = execution_worker(transfer_backends=("shm",))
    worker = execution_worker(transfer_backends=("shm",), pipeline_depth=3)
    incoming = ar_params(45, block_ids=(0,))
    admission = ar_params(44, block_ids=(0,))
    server_started = False
    grant = Event()
    accepted = Event()
    endpoint = f"uniserve-test-kv-{uuid.uuid4().hex}"
    try:
        publications = []
        commits = []
        for owner, request, tokens in (
            (producer, incoming, (8, 9)),
            (worker, admission, (3, 4)),
        ):
            extend, input_product = token_operation(
                request.request_key,
                op_id=1,
                parent=root_parent(request),
                mode=TokenMode.EXTEND,
                tokens=tokens,
            )
            extended = finalized_report(
                owner.execute(
                    execution_run(
                        run_id=1,
                        admissions=(request,),
                        operations=(extend,),
                        input_products=(input_product,),
                    )
                )
            )
            commit = commit_for_completion(extend, extended)
            publication, _product = _publication_operation(
                request.request_key,
                op_id=2,
                parent=commit.selected,
                control_seq=commit.control_seq,
            )
            published = finalized_report(
                owner.execute(
                    execution_run(run_id=2, operations=(publication,), commands=(commit,))
                )
            )
            publications.append(published.products[0])
            commits.append(commit)

        source, resident = publications
        assert isinstance(source.payload, TransferHandle)
        assert isinstance(source.payload.value, KvTransferValue)
        assert isinstance(resident.payload, TransferHandle)
        assert isinstance(resident.payload.value, KvTransferValue)
        old_locator = resident.payload.value.tensors[0].locations[0]
        assert isinstance(old_locator.transport, PosixShmTransfer)

        # The external publisher owns real SHM bytes, but gates permission to
        # read them so storage retirement and input completion remain distinct.
        tensors = tuple(
            replace(
                tensor,
                locations=tuple(
                    replace(locator, transport=replace(locator.transport, endpoint=endpoint))
                    for locator in tensor.locations
                ),
            )
            for tensor in source.payload.value.tensors
        )
        incoming_payload = replace(
            source, payload=TransferHandle(replace(source.payload.value, tensors=tensors))
        )
        requests = {_read_request(locator) for tensor in tensors for locator in tensor.locations}
        with (
            socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener,
            socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as reader,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            listener.bind("\0" + endpoint)
            listener.listen(len(requests))
            listener.settimeout(10)

            def serve() -> None:
                pending = set(requests)
                while pending:
                    connection, _address = listener.accept()
                    with connection:
                        connection.settimeout(10)
                        pending.remove(connection.recv(128))
                        accepted.set()
                        assert grant.wait(10), "publisher was never permitted to expose its bytes"
                        connection.sendall(b"G")
                        assert connection.recv(1) == b"A"
                        connection.sendall(b"D")

            reader.settimeout(10)
            reader.connect("\0" + old_locator.transport.endpoint)
            reader.sendall(_read_request(old_locator))
            assert reader.recv(1) == b"G"
            commit = commits[1]
            finish = Finish(
                admission.request_key,
                commit.control_seq + 1,
                commit.selected,
                CloseReason.COMPLETED,
            )
            installation, installed = _installation_operation(
                incoming, op_id=3, parent=root_parent(incoming), source=source.product
            )
            independent = ar_params(46, block_ids=(1,))
            operation, independent_input = token_operation(
                independent.request_key,
                op_id=1,
                parent=root_parent(independent),
                mode=TokenMode.EXTEND,
                tokens=(6, 7),
            )
            runs = (
                execution_run(run_id=3, commands=(finish,)),
                execution_run(
                    run_id=5,
                    admissions=(incoming,),
                    operations=(installation,),
                    input_products=(incoming_payload,),
                    **_installation_allocation(installation, 2),
                ),
                execution_run(
                    run_id=4,
                    admissions=(independent,),
                    operations=(operation,),
                    input_products=(independent_input,),
                ),
            )
            ipc = QueuedWorkerIpc(
                tuple({"kind": "submit", "call_id": run.run_id, "run": run} for run in runs)
            )
            server = WorkerProcess(worker, ipc)
            serving = executor.submit(serve)
            processing = executor.submit(server.serve)
            server_started = True
            reader_held = True
            try:
                response = ipc.receive()
                assert response["call_id"] == 4, response
                completed = RunResult.from_mapping(response["result"])
                assert completed.completions[0].status is OpStatus.OK

                reader.sendall(b"A")
                assert reader.recv(1) == b"D"
                reader_held = False
                response = ipc.receive()
                assert response["call_id"] == 3, response
                assert RunResult.from_mapping(response["result"]).done
                assert accepted.wait(5), "retiring storage did not start the dependent read"
                grant.set()

                # No new IPC request drives this transition: the completed
                # physical import must wake the sleeping process itself.
                response = ipc.receive()
                assert response["call_id"] == 5, response
                report = RunResult.from_mapping(response["result"])
                assert report.completions[0].status is OpStatus.OK
                assert report.completions[0].logical_lengths.kv_visible_len == 2
                assert report.products[0].product == installed
                serving.result(timeout=5)
                for layer in range(worker.cache_pool.num_layers):
                    expected = producer.cache_pool.read(layer, (1,), start=0, length=2)
                    actual = worker.cache_pool.read(layer, (1,), start=0, length=2)
                    for left, right in zip(actual, expected, strict=True):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
            finally:
                grant.set()
                if reader_held:
                    reader.sendall(b"A")
                    assert reader.recv(1) == b"D"
                ipc.submit({"kind": "close", "call_id": 6})
                processing.result(timeout=10)
    finally:
        grant.set()
        if not server_started:
            worker.close()
        producer.close()


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
            kind=OpCode.TRANSFER_KV_INSTALL,
            bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
            inputs=(source,),
            outputs=(product,),
        ),
        product,
    )


def _installation_allocation(operation: Operation, length: int) -> dict[str, object]:
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
    closure_result = finalized_report(
        worker.execute(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(closure,),
                commands=(first_commit,),
                input_products=(closure_input,),
            )
        )
    )
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
    payload = publication_result.products[0]
    assert isinstance(payload.payload, TransferHandle)
    snapshot = payload.payload.value
    assert isinstance(snapshot, KvTransferValue)
    assert payload.product == publication_product
    assert snapshot.source == second_commit.selected
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
    suffix_result = finalized_report(
        worker.execute(
            execution_run(
                run_id=4,
                admissions=(),
                operations=(suffix_closure,),
                input_products=(suffix_input,),
            )
        )
    )
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
    assert isinstance(incremental_payload.payload, TransferHandle)
    incremental_snapshot = incremental_payload.payload.value
    assert isinstance(incremental_snapshot, KvTransferValue)
    assert incremental_payload.product == incremental_product
    assert incremental_snapshot.source == suffix_commit.selected
    assert incremental_snapshot.base == second_commit.selected
    assert incremental_snapshot.base_extent == 3
    assert incremental_snapshot.published_extent == 4


def test_cross_stage_kv_install_uses_query_ready_exact_snapshot() -> None:
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    released_consumer = execution_worker(transfer_backends=("shm",))
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
            **_installation_allocation(installation, 2),
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

        # Republishing an unchanged visible extent carries a valid empty suffix.
        # Installation must preserve the cache and acknowledge its new product.
        repeated_publication, repeated_source = _publication_operation(
            admission.request_key,
            op_id=4,
            parent=commit.selected,
            control_seq=commit.control_seq,
        )
        repeated = finalized_report(
            producer.execute(execution_run(run_id=4, operations=(repeated_publication,)))
        )
        repeated_install, repeated_installed = _installation_operation(
            admission,
            op_id=5,
            parent=root_parent(admission),
            source=repeated_source,
        )
        repeated_batch = execution_run(
            run_id=5,
            operations=(repeated_install,),
            input_products=(repeated.products[0],),
            block_tables=(BlockTable(admission.request_pool_idx, 0, (1,), 2),),
        )
        repeated_prepared = consumer.prepare_execute(repeated_batch)
        assert repeated_prepared is not None
        assert repeated_prepared.ready()
        repeated_report = finalized_report(consumer.execute_prepared(repeated_prepared))
        assert repeated_report.completions[0].status.value == "ok"
        assert repeated_report.completions[0].logical_lengths.kv_visible_len == 2
        assert repeated_report.products[0].product == repeated_installed
        finalized_report(
            producer.execute(
                execution_run(
                    run_id=6,
                    commands=(Free(source.buffer_id),),
                )
            )
        )
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
                **_installation_allocation(expired_install, 2),
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
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
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
            producer.execute(execution_run(run_id=2, operations=(publication,), commands=(commit,)))
        ).products[0]
        assert isinstance(published.payload, TransferHandle)
        snapshot = published.payload.value
        assert isinstance(snapshot, KvTransferValue)
        first = snapshot.tensors[0].locations[0]
        assert isinstance(first.transport, PosixShmTransfer)
        missing = replace(
            first, transport=replace(first.transport, name="uniserve-missing-transfer-segment")
        )
        broken = replace(
            snapshot,
            tensors=(replace(snapshot.tensors[0], locations=(missing,)), *snapshot.tensors[1:]),
        )
        payload = ProductPayload(
            product=published.product,
            payload=TransferHandle(broken),
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
                **_installation_allocation(installation, 2),
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
                **_installation_allocation(retry, 2),
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
