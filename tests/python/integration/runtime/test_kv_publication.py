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
    execution_run,
    finalized_report,
    record_completion,
    root_parent,
    token_operation,
)
from tests.python.fixtures.depth_one import (
    kv_publication_operation as _publication_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from uniserve_worker.protocol.batch import (
    BatchOutput,
    BlockTable,
    Bounds,
    BufferId,
    CachePageAllocation,
    ComputationId,
    Finish,
    ForwardMode,
    Free,
    KvTransfer,
    Locator,
    NewRequest,
    OpStatus,
    PosixShmTransfer,
    ScheduledRequest,
    TransferMode,
)


def _read_request(locator: Locator) -> bytes:
    """Encode a host reader's registration request at the external SHM boundary."""

    assert isinstance(locator.transport, PosixShmTransfer)
    descriptor = json.dumps(locator.to_mapping(), sort_keys=True, separators=(",", ":"))
    return (
        hashlib.sha256(locator.transport.name.encode()).digest()
        + hashlib.sha256(descriptor.encode()).digest()
    )


def test_kv_install_waits_for_storage_and_input_without_blocking_independent_work() -> None:
    with (
        execution_worker(transfer_backends=("shm",)) as producer,
        execution_worker(transfer_backends=("shm",), pipeline_depth=3) as worker,
    ):
        incoming = ar_params(45, block_ids=(0,))
        admission = ar_params(44, block_ids=(0,))
        worker.warmup()
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
                extend = token_operation(
                    request.request_key,
                    op_id=ComputationId(1, 0),
                    predecessor=root_parent(request),
                    mode=ForwardMode.PREFILL,
                    tokens=tokens,
                )
                extended = finalized_report(
                    owner,
                    owner.submit(
                        execution_run(
                            run_id=1,
                            admissions=(request,),
                            operations=(extend,),
                        )
                    ),
                )
                observation = record_completion(extend, extended)
                publication, _product = _publication_operation(
                    request.request_key,
                    op_id=ComputationId(2, 0),
                    predecessor=observation.op_id,
                )
                published = finalized_report(
                    owner,
                    owner.submit(execution_run(run_id=2, operations=(publication,), commands=())),
                )
                publications.append(published.completions[0].kv_output)
                commits.append(observation)

            source, resident = publications
            assert isinstance(source, KvTransfer)
            assert isinstance(resident, KvTransfer)
            old_locator = resident.tensors[0].locations[0]
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
                for tensor in source.tensors
            )
            incoming_payload = replace(source, tensors=tensors)
            requests = {
                _read_request(locator) for tensor in tensors for locator in tensor.locations
            }
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
                            assert grant.wait(10), (
                                "publisher was never permitted to expose its bytes"
                            )
                            connection.sendall(b"G")
                            assert connection.recv(1) == b"A"
                            connection.sendall(b"D")

                reader.settimeout(10)
                reader.connect("\0" + old_locator.transport.endpoint)
                reader.sendall(_read_request(old_locator))
                assert reader.recv(1) == b"G"
                observation = commits[1]
                finish = Finish(
                    admission.request_key,
                )
                installation, installed = _installation_operation(
                    incoming,
                    op_id=ComputationId(3, 0),
                    predecessor=root_parent(incoming),
                    source=source.source,
                )
                independent = ar_params(46, block_ids=(1,))
                operation = token_operation(
                    independent.request_key,
                    op_id=ComputationId(1, 0),
                    predecessor=root_parent(independent),
                    mode=ForwardMode.PREFILL,
                    tokens=(6, 7),
                )
                runs = (
                    execution_run(run_id=3, commands=(finish,)),
                    execution_run(
                        run_id=4,
                        admissions=(incoming,),
                        operations=(installation,),
                        kv_inputs=(incoming_payload,),
                        **_installation_allocation(installation, 2),
                    ),
                    execution_run(
                        run_id=5,
                        admissions=(independent,),
                        operations=(operation,),
                    ),
                )
                ipc = QueuedWorkerIpc(
                    tuple({"kind": "submit", "call_id": run.run_id, "run": run} for run in runs)
                )
                worker.bind(ipc)
                serving = executor.submit(serve)
                processing = executor.submit(worker.run)
                reader_held = True
                try:
                    response = ipc.receive()
                    assert response["call_id"] == 5, response
                    completed = BatchOutput.from_mapping(response["result"])
                    assert completed.completions[0].status is OpStatus.OK

                    reader.sendall(b"A")
                    assert reader.recv(1) == b"D"
                    reader_held = False
                    response = ipc.receive()
                    assert response["call_id"] == 3, response
                    assert BatchOutput.from_mapping(response["result"]).done
                    assert accepted.wait(5), "retiring storage did not start the dependent read"
                    grant.set()

                    # No new IPC request drives this transition: the completed
                    # physical import must wake the sleeping process itself.
                    response = ipc.receive()
                    assert response["call_id"] == 4, response
                    report = BatchOutput.from_mapping(response["result"])
                    assert report.completions[0].status is OpStatus.OK
                    assert report.completions[0].kv_visible_len == 2
                    serving.result(timeout=5)
                    for layer in range(worker.kv_cache.cache.config.num_layers):
                        expected = producer.kv_cache.cache.layer(layer).read(
                            (1,), start=0, length=2
                        )
                        actual = worker.kv_cache.cache.layer(layer).read((1,), start=0, length=2)
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


def _installation_operation(
    admission: NewRequest,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    source: BufferId,
) -> tuple[ScheduledRequest, BufferId]:
    product = BufferId(
        owner=admission.request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 10 + 1,
    )
    return (
        ScheduledRequest(
            request_key=admission.request_key,
            op_id=op_id,
            predecessor=predecessor,
            kind=TransferMode.KV_INSTALL,
            bounds=Bounds(max_transfer_bytes=1 << 20),
            kv_input=source,
            kv_output=product,
        ),
        product,
    )


def _installation_allocation(operation: ScheduledRequest, length: int) -> dict[str, object]:
    request_pool_idx = int(operation.request_key.request_id) + 1
    return {
        "block_tables": (BlockTable(request_pool_idx, 0, (1,), max(1, int(length))),),
        "new_cache_pages": (CachePageAllocation(request_pool_idx, 0, (1,)),),
    }


def test_tail_closure_precedes_exact_incremental_publication() -> None:
    worker = execution_worker()
    admission = ar_params(41, block_ids=(0,))
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    first_result = worker.submit(
        execution_run(
            run_id=1,
            admissions=(admission,),
            operations=(extend,),
        )
    )
    first_result = finalized_report(worker, first_result)
    first_observation = record_completion(extend, first_result)
    closure_template = token_operation(
        admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=first_observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(5,),
    )
    closure = ScheduledRequest(
        request_key=closure_template.request_key,
        op_id=closure_template.op_id,
        predecessor=closure_template.predecessor,
        kind=closure_template.kind,
        bounds=closure_template.bounds,
        inputs=closure_template.inputs,
        input_token_ids=closure_template.input_token_ids,
        outputs=(),
    )
    closure_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                admissions=(),
                operations=(closure,),
                commands=(),
            )
        ),
    )
    closure_record = closure_result.completions[0]
    assert closure_record.position == 2
    assert closure_record.kv_visible_len == 3
    assert closure_record.kv_computed_len == 3

    second_observation = record_completion(closure, closure_result)
    publication, publication_product = _publication_operation(
        admission.request_key,
        op_id=ComputationId(3, 0),
        predecessor=second_observation.op_id,
    )
    publication_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=3,
                admissions=(),
                operations=(publication,),
                commands=(),
            )
        ),
    )

    assert publication_result.completions[0].kv_visible_len == 3
    snapshot = publication_result.completions[0].kv_output
    assert isinstance(snapshot, KvTransfer)
    assert snapshot.source == publication_product
    assert snapshot.base_extent == 0
    assert snapshot.published_extent == 3

    suffix_template = token_operation(
        admission.request_key,
        op_id=ComputationId(4, 0),
        predecessor=second_observation.op_id,
        mode=ForwardMode.PREFILL,
        tokens=(6,),
    )
    suffix_closure = ScheduledRequest(
        request_key=suffix_template.request_key,
        op_id=suffix_template.op_id,
        predecessor=suffix_template.predecessor,
        kind=suffix_template.kind,
        bounds=suffix_template.bounds,
        inputs=suffix_template.inputs,
        input_token_ids=suffix_template.input_token_ids,
        outputs=(),
    )
    suffix_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=4,
                admissions=(),
                operations=(suffix_closure,),
            )
        ),
    )
    assert suffix_result.completions[0].kv_visible_len == 4

    suffix_observation = record_completion(suffix_closure, suffix_result)
    incremental, incremental_product = _publication_operation(
        admission.request_key,
        op_id=ComputationId(5, 0),
        predecessor=suffix_observation.op_id,
    )
    incremental_result = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=5,
                admissions=(),
                operations=(incremental,),
                commands=(),
            )
        ),
    )
    incremental_snapshot = incremental_result.completions[0].kv_output
    assert isinstance(incremental_snapshot, KvTransfer)
    assert incremental_snapshot.source == incremental_product
    assert incremental_snapshot.base == publication_product
    assert incremental_snapshot.base_extent == 3
    assert incremental_snapshot.published_extent == 4


def test_cross_stage_kv_install_uses_query_ready_exact_snapshot() -> None:
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    released_consumer = execution_worker(transfer_backends=("shm",))
    admission = ar_params(42, block_ids=(0,))
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        extended = producer.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
            )
        )
        extended = finalized_report(producer, extended)
        observation = record_completion(extend, extended)
        publication, source = _publication_operation(
            admission.request_key,
            op_id=ComputationId(2, 0),
            predecessor=observation.op_id,
        )
        published = finalized_report(
            producer,
            producer.submit(
                execution_run(
                    run_id=2,
                    operations=(publication,),
                    commands=(),
                )
            ),
        )
        assert isinstance(published.completions[0].kv_output, KvTransfer)
        installation, installed = _installation_operation(
            admission,
            op_id=ComputationId(3, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        batch = execution_run(
            run_id=3,
            admissions=(admission,),
            operations=(installation,),
            kv_inputs=(published.completions[0].kv_output,),
            **_installation_allocation(installation, 2),
        )
        prepared = consumer.submit(batch)
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()
        prepared = finalized_report(consumer, prepared)
        report = prepared
        assert report.completions[0].status.value == "ok"
        assert report.completions[0].kv_visible_len == 2

        # Republishing an unchanged visible extent carries a valid empty suffix.
        # Installation must preserve the cache and acknowledge its new product.
        repeated_publication, repeated_source = _publication_operation(
            admission.request_key,
            op_id=ComputationId(4, 0),
            predecessor=observation.op_id,
        )
        repeated = finalized_report(
            producer, producer.submit(execution_run(run_id=4, operations=(repeated_publication,)))
        )
        repeated_install, repeated_installed = _installation_operation(
            admission,
            op_id=ComputationId(5, 0),
            predecessor=root_parent(admission),
            source=repeated_source,
        )
        repeated_batch = execution_run(
            run_id=5,
            operations=(repeated_install,),
            kv_inputs=(repeated.completions[0].kv_output,),
            block_tables=(BlockTable(admission.request_pool_idx, 0, (1,), 2),),
        )
        repeated_prepared = consumer.submit(repeated_batch)
        assert repeated_prepared is not None
        assert repeated_prepared.inputs_ready()
        repeated_prepared = finalized_report(consumer, repeated_prepared)
        repeated_report = repeated_prepared
        assert repeated_report.completions[0].status.value == "ok"
        assert repeated_report.completions[0].kv_visible_len == 2
        finalized_report(
            producer,
            producer.submit(
                execution_run(
                    run_id=6,
                    commands=(Free(source),),
                )
            ),
        )
        expired_install, _ = _installation_operation(
            admission,
            op_id=ComputationId(4, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        expired = released_consumer.submit(
            execution_run(
                run_id=5,
                admissions=(admission,),
                operations=(expired_install,),
                kv_inputs=(published.completions[0].kv_output,),
                **_installation_allocation(expired_install, 2),
            )
        )
        assert expired is not None
        deadline = time.monotonic() + 5.0
        while not expired.inputs_ready() and time.monotonic() < deadline:
            released_consumer.advance_inputs(expired)
            time.sleep(0.001)
        assert expired.inputs_ready()
        expired = finalized_report(released_consumer, expired)
        expired_report = expired
        assert expired_report.completions[0].status.value == "error"
    finally:
        producer.close()
        consumer.close()
        released_consumer.close()


def test_failed_cross_stage_kv_read_preserves_source_and_destination_state() -> None:
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    admission = ar_params(43, block_ids=(0,))
    extend = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    try:
        extended = producer.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(extend,),
            )
        )
        extended = finalized_report(producer, extended)
        observation = record_completion(extend, extended)
        publication, source = _publication_operation(
            admission.request_key,
            op_id=ComputationId(2, 0),
            predecessor=observation.op_id,
        )
        published = (
            finalized_report(
                producer,
                producer.submit(execution_run(run_id=2, operations=(publication,), commands=())),
            )
            .completions[0]
            .kv_output
        )
        snapshot = published
        assert isinstance(snapshot, KvTransfer)
        first = snapshot.tensors[0].locations[0]
        assert isinstance(first.transport, PosixShmTransfer)
        missing = replace(
            first, transport=replace(first.transport, name="uniserve-missing-transfer-segment")
        )
        broken = replace(
            snapshot,
            tensors=(replace(snapshot.tensors[0], locations=(missing,)), *snapshot.tensors[1:]),
        )
        payload = broken
        installation, _installed = _installation_operation(
            admission,
            op_id=ComputationId(3, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        prepared = consumer.submit(
            execution_run(
                run_id=3,
                admissions=(admission,),
                operations=(installation,),
                kv_inputs=(payload,),
                **_installation_allocation(installation, 2),
            )
        )
        assert prepared is not None
        deadline = time.monotonic() + 5.0
        while not prepared.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared)
            time.sleep(0.001)
        assert prepared.inputs_ready()

        report = prepared

        report = finalized_report(consumer, report)
        assert report.completions[0].status.value == "error"
        assert report.completions[0].error_code is not None

        retry, installed = _installation_operation(
            admission,
            op_id=ComputationId(4, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        prepared_retry = consumer.submit(
            execution_run(
                run_id=4,
                admissions=(admission,),
                operations=(retry,),
                kv_inputs=(published,),
                **_installation_allocation(retry, 2),
            )
        )
        assert prepared_retry is not None
        deadline = time.monotonic() + 5.0
        while not prepared_retry.inputs_ready() and time.monotonic() < deadline:
            consumer.advance_inputs(prepared_retry)
            time.sleep(0.001)
        assert prepared_retry.inputs_ready()
        prepared_retry = finalized_report(consumer, prepared_retry)
        retry_report = prepared_retry
        assert retry_report.completions[0].status.value == "ok"
        assert retry_report.completions[0].kv_visible_len == 2
    finally:
        producer.close()
        consumer.close()
