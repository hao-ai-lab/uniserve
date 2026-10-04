from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from multiprocessing import shared_memory

import pytest
import torch

from tests.python.fixtures.depth_one import (
    ar_params,
    execution_batch,
    finalized_report,
    record_completion,
    root_parent,
    token_call,
)
from tests.python.fixtures.depth_one import (
    kv_publication_call as _publication_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from uniserve_worker.protocol.batch import (
    BlockTable,
    CacheUnitAllocation,
    Finish,
    Free,
    NewRequest,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    CallStatus,
    ForwardMode,
    TransferMode,
)
from uniserve_worker.protocol.identity import BufferId, CallId
from uniserve_worker.protocol.output import BatchOutput
from uniserve_worker.protocol.transfer import (
    KvTransfer,
    Locator,
    PosixShmTransfer,
)
from uniserve_worker.transport import segment
from uniserve_worker.transport.shared_storage import open_shared_storage

pytestmark = pytest.mark.integration


def _map_locations(publication: KvTransfer, locate) -> KvTransfer:
    """Replace every published tensor's locations, group by group."""
    return replace(
        publication,
        groups=tuple(
            replace(
                group,
                tensors=tuple(
                    replace(tensor, locations=locate(tensor))
                    for tensor in group.tensors
                ),
            )
            for group in publication.groups
        ),
    )


def _gated_copy(locator: Locator) -> tuple[Locator, shared_memory.SharedMemory]:
    """Copy a ready publication into a segment whose readiness the test holds.

    The reader stays pending until the test announces readiness in the header.
    """
    assert isinstance(locator.transport, PosixShmTransfer)
    source = open_shared_storage(
        locator.transport.name, segment.HEADER_BYTES + locator.nbytes
    )
    try:
        header = memoryview(source)
        segment.await_ready(header)
        payload = segment.HEADER_BYTES
        data = bytes(source[payload : payload + locator.nbytes])
        header.release()
    finally:
        source.close()
    storage = shared_memory.SharedMemory(
        create=True, size=segment.HEADER_BYTES + locator.nbytes
    )
    gated = replace(
        locator, transport=replace(locator.transport, name=storage.name)
    )
    segment.initialize(storage.buf)
    storage.buf[segment.HEADER_BYTES : segment.HEADER_BYTES + len(data)] = data
    return gated, storage


def test_kv_install_waits_for_storage_and_input_without_blocking_independent_work(  # noqa: E501
) -> None:
    """A KV installation waits for its storage and its input, and no more.

    The storage is held by a retired publication a consumer has not yet
    acknowledged; the input is a publication whose producer has not yet
    announced it. Independent work completes meanwhile, the acknowledgment
    releases the storage, and readiness completes the installation.
    """
    with (
        execution_worker(transfer_backends=("shm",)) as producer,
        execution_worker(transfer_backends=("shm",), queue_depth=3) as worker,
    ):
        incoming = ar_params(45, block_ids=(0,))
        admission = ar_params(44, block_ids=(0,))
        worker.warmup()
        gated_segments: list[shared_memory.SharedMemory] = []
        try:
            publications = []
            commits = []
            for owner, request, tokens in (
                (producer, incoming, (8, 9)),
                (worker, admission, (3, 4)),
            ):
                extend = token_call(
                    request.request_key,
                    call_id=CallId(1, 0),
                    predecessor=root_parent(request),
                    mode=ForwardMode.PREFILL,
                    tokens=tokens,
                )
                extended = finalized_report(
                    owner,
                    owner.submit(
                        execution_batch(
                            batch_id=1,
                            admissions=(request,),
                            calls=(extend,),
                        )
                    ),
                )
                observation = record_completion(extend, extended)
                publication, _product = _publication_call(
                    request.request_key,
                    call_id=CallId(2, 0),
                    predecessor=observation.call_id,
                )
                if owner is worker:
                    # This publication is read by an external consumer whose
                    # acknowledgment the test controls.
                    publication = replace(publication, consumer_slots=(1,))
                published = finalized_report(
                    owner,
                    owner.submit(
                        execution_batch(
                            batch_id=2, calls=(publication,), commands=()
                        )
                    ),
                )
                publications.append(published.completions[0].kv_output)
                commits.append(observation)

            source, resident = publications
            assert isinstance(source, KvTransfer)
            assert isinstance(resident, KvTransfer)
            old_locator = resident.tensors[0].locations[0]
            assert isinstance(old_locator.transport, PosixShmTransfer)

            # The incoming publication's bytes are real, but the test holds
            # their readiness, so storage retirement and input completion
            # remain distinct.
            def gate(tensor):
                locations = []
                for locator in tensor.locations:
                    gated, storage = _gated_copy(locator)
                    gated_segments.append(storage)
                    locations.append(gated)
                return tuple(locations)

            incoming_payload = _map_locations(source, gate)

            # An external consumer holds every segment of the worker's own
            # publication unacknowledged across the request's Finish.
            held: list[tuple[object, memoryview]] = []
            for tensor in resident.tensors:
                for locator in tensor.locations:
                    assert isinstance(locator.transport, PosixShmTransfer)
                    mapping = open_shared_storage(
                        locator.transport.name,
                        segment.HEADER_BYTES + locator.nbytes,
                    )
                    header = memoryview(mapping)
                    segment.await_ready(header)
                    # A reader claims its word before its first read; the
                    # producer waits only for readers that claimed.
                    segment.claim(header, 1)
                    held.append((mapping, header))

            def acknowledge_held() -> None:
                for _mapping, header in held:
                    segment.acknowledge(header, 1)

            # The worker runs on its own thread; a failed assertion must not
            # wait for a worker that no longer reads its channel.
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                finish = Finish(
                    admission.request_key,
                )
                installation, installed = _installation_call(
                    incoming,
                    call_id=CallId(4, 0),
                    predecessor=root_parent(incoming),
                    source=source.source,
                )
                independent = ar_params(46, block_ids=(1,))
                call = token_call(
                    independent.request_key,
                    call_id=CallId(5, 0),
                    predecessor=root_parent(independent),
                    mode=ForwardMode.PREFILL,
                    tokens=(6, 7),
                )
                runs = (
                    execution_batch(batch_id=3, commands=(finish,)),
                    execution_batch(
                        batch_id=4,
                        admissions=(incoming,),
                        calls=(installation,),
                        kv_inputs=(incoming_payload,),
                        **_installation_allocation(installation, 2),
                    ),
                    execution_batch(
                        batch_id=5,
                        admissions=(independent,),
                        calls=(call,),
                    ),
                )
                ipc = QueuedWorkerIpc(
                    tuple(
                        {
                            "kind": "submit",
                            "message_id": run.batch_id,
                            "batch": run,
                        }
                        for run in runs
                    )
                )
                worker.bind(ipc)
                processing = executor.submit(worker.run)
                reader_held = True
                try:
                    response = ipc.receive()
                    assert response["message_id"] == 5, response
                    completed = BatchOutput.from_mapping(response["result"])
                    assert completed.completions[0].status is CallStatus.OK

                    acknowledge_held()
                    reader_held = False
                    response = ipc.receive()
                    assert response["message_id"] == 3, response
                    for storage in gated_segments:
                        segment.set_state(storage.buf, segment.READY)

                    # No new IPC request drives this transition: the completed
                    # physical import must wake the sleeping process itself.
                    response = ipc.receive()
                    assert response["message_id"] == 4, response
                    report = BatchOutput.from_mapping(response["result"])
                    assert report.completions[0].status is CallStatus.OK
                    assert report.completions[0].kv_visible_len == 2
                    for layer in worker.kv_cache.cache.config.layers:
                        expected = producer.kv_cache.cache.state(layer).read(
                            (1,), start=0, length=2
                        )
                        actual = worker.kv_cache.cache.state(layer).read(
                            (1,), start=0, length=2
                        )
                        for left, right in zip(actual, expected, strict=True):
                            torch.testing.assert_close(
                                left, right, rtol=0, atol=0
                            )
                finally:
                    if reader_held:
                        acknowledge_held()
                    for storage in gated_segments:
                        segment.set_state(storage.buf, segment.READY)
                    for mapping, header in held:
                        header.release()
                        mapping.close()
                    ipc.submit({"kind": "close", "message_id": 6})
                    processing.result(timeout=10)
            finally:
                executor.shutdown(wait=False)
        finally:
            for storage in gated_segments:
                storage.close()
                with suppress(FileNotFoundError):
                    storage.unlink()


def _installation_call(
    admission: NewRequest,
    *,
    call_id: CallId,
    predecessor: CallId,
    source: BufferId,
) -> tuple[Call, BufferId]:
    product = BufferId(
        owner=admission.request_key,
        producer_call_id=call_id,
        output_index=0,
        generation=call_id.batch_id * 10 + 1,
    )
    return (
        Call(
            request_key=admission.request_key,
            call_id=call_id,
            coordinates=CallCoordinates(),
            kind=TransferMode.KV_INSTALL,
            bounds=Bounds(max_transfer_bytes=1 << 20),
            kv_input=source,
            kv_output=product,
        ),
        product,
    )


def _installation_allocation(call: Call, length: int) -> dict[str, object]:
    request_pool_idx = int(call.request_key.request_id) + 1
    return {
        "block_tables": (
            BlockTable(request_pool_idx, 0, 0, (1,), max(1, int(length))),
        ),
        "new_cache_units": (CacheUnitAllocation(request_pool_idx, 0, (1,)),),
    }


def test_tail_closure_precedes_exact_incremental_publication() -> None:
    worker = execution_worker()
    admission = ar_params(41, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    first_result = worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(admission,),
            calls=(extend,),
        )
    )
    first_result = finalized_report(worker, first_result)
    first_observation = record_completion(extend, first_result)
    closure_template = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=first_observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(5,),
    )
    closure = Call(
        request_key=closure_template.request_key,
        call_id=closure_template.call_id,
        coordinates=closure_template.coordinates,
        kind=closure_template.kind,
        bounds=closure_template.bounds,
        inputs=closure_template.inputs,
        input_token_ids=closure_template.input_token_ids,
        outputs=(),
    )
    closure_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(),
                calls=(closure,),
                commands=(),
            )
        ),
    )
    closure_record = closure_result.completions[0]
    assert closure_record.position == 2
    assert closure_record.kv_visible_len == 3
    assert closure_record.kv_computed_len == 3

    second_observation = record_completion(closure, closure_result)
    publication, publication_product = _publication_call(
        admission.request_key,
        call_id=CallId(3, 0),
        predecessor=second_observation.call_id,
    )
    publication_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(),
                calls=(publication,),
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

    suffix_template = token_call(
        admission.request_key,
        call_id=CallId(4, 0),
        predecessor=second_observation.call_id,
        mode=ForwardMode.PREFILL,
        tokens=(6,),
    )
    suffix_closure = Call(
        request_key=suffix_template.request_key,
        call_id=suffix_template.call_id,
        coordinates=suffix_template.coordinates,
        kind=suffix_template.kind,
        bounds=suffix_template.bounds,
        inputs=suffix_template.inputs,
        input_token_ids=suffix_template.input_token_ids,
        outputs=(),
    )
    suffix_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=4,
                admissions=(),
                calls=(suffix_closure,),
            )
        ),
    )
    assert suffix_result.completions[0].kv_visible_len == 4

    suffix_observation = record_completion(suffix_closure, suffix_result)
    incremental, incremental_product = _publication_call(
        admission.request_key,
        call_id=CallId(5, 0),
        predecessor=suffix_observation.call_id,
    )
    incremental_result = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=5,
                admissions=(),
                calls=(incremental,),
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
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        extended = producer.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        )
        extended = finalized_report(producer, extended)
        observation = record_completion(extend, extended)
        publication, source = _publication_call(
            admission.request_key,
            call_id=CallId(2, 0),
            predecessor=observation.call_id,
        )
        published = finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=2,
                    calls=(publication,),
                    commands=(),
                )
            ),
        )
        assert isinstance(published.completions[0].kv_output, KvTransfer)
        installation, installed = _installation_call(
            admission,
            call_id=CallId(3, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        batch = execution_batch(
            batch_id=3,
            admissions=(admission,),
            calls=(installation,),
            kv_inputs=(published.completions[0].kv_output,),
            **_installation_allocation(installation, 2),
        )
        prepared = consumer.submit(batch)
        prepared = finalized_report(consumer, prepared)
        report = prepared
        assert report.completions[0].status.value == "ok"
        assert report.completions[0].kv_visible_len == 2

        # Republishing an unchanged visible extent carries a valid empty suffix.
        # Installation must preserve the cache and acknowledge its new product.
        repeated_publication, repeated_source = _publication_call(
            admission.request_key,
            call_id=CallId(4, 0),
            predecessor=observation.call_id,
        )
        repeated = finalized_report(
            producer,
            producer.submit(
                execution_batch(batch_id=4, calls=(repeated_publication,))
            ),
        )
        repeated_install, repeated_installed = _installation_call(
            admission,
            call_id=CallId(5, 0),
            predecessor=root_parent(admission),
            source=repeated_source,
        )
        repeated_batch = execution_batch(
            batch_id=5,
            calls=(repeated_install,),
            kv_inputs=(repeated.completions[0].kv_output,),
            block_tables=(
                BlockTable(admission.request_pool_idx, 0, 0, (1,), 2),
            ),
        )
        repeated_prepared = consumer.submit(repeated_batch)
        repeated_prepared = finalized_report(consumer, repeated_prepared)
        repeated_report = repeated_prepared
        assert repeated_report.completions[0].status.value == "ok"
        assert repeated_report.completions[0].kv_visible_len == 2
        finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=6,
                    commands=(Free(source),),
                )
            ),
        )
        expired_install, _ = _installation_call(
            admission,
            call_id=CallId(4, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        expired = released_consumer.submit(
            execution_batch(
                batch_id=5,
                admissions=(admission,),
                calls=(expired_install,),
                kv_inputs=(published.completions[0].kv_output,),
                **_installation_allocation(expired_install, 2),
            )
        )
        expired = finalized_report(released_consumer, expired)
        expired_report = expired
        assert expired_report.completions[0].status.value == "error"
    finally:
        producer.close()
        consumer.close()
        released_consumer.close()


def test_failed_cross_stage_kv_read_preserves_source_and_destination_state() -> (  # noqa: E501
    None
):
    producer = execution_worker(transfer_backends=("shm",))
    consumer = execution_worker(transfer_backends=("shm",))
    admission = ar_params(43, block_ids=(0,))
    extend = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    try:
        extended = producer.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(extend,),
            )
        )
        extended = finalized_report(producer, extended)
        observation = record_completion(extend, extended)
        publication, source = _publication_call(
            admission.request_key,
            call_id=CallId(2, 0),
            predecessor=observation.call_id,
        )
        published = (
            finalized_report(
                producer,
                producer.submit(
                    execution_batch(
                        batch_id=2, calls=(publication,), commands=()
                    )
                ),
            )
            .completions[0]
            .kv_output
        )
        snapshot = published
        assert isinstance(snapshot, KvTransfer)
        first = snapshot.tensors[0].locations[0]
        assert isinstance(first.transport, PosixShmTransfer)
        missing = replace(
            first,
            transport=replace(
                first.transport, name="uniserve-missing-transfer-segment"
            ),
        )
        broken = _map_locations(
            snapshot,
            lambda tensor: (
                (missing,)
                if tensor is snapshot.tensors[0]
                else tensor.locations
            ),
        )
        payload = broken
        installation, _installed = _installation_call(
            admission,
            call_id=CallId(3, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        prepared = consumer.submit(
            execution_batch(
                batch_id=3,
                admissions=(admission,),
                calls=(installation,),
                kv_inputs=(payload,),
                **_installation_allocation(installation, 2),
            )
        )

        report = prepared

        report = finalized_report(consumer, report)
        assert report.completions[0].status.value == "error"
        assert report.completions[0].error_code is not None

        retry, installed = _installation_call(
            admission,
            call_id=CallId(4, 0),
            predecessor=root_parent(admission),
            source=source,
        )
        prepared_retry = consumer.submit(
            execution_batch(
                batch_id=4,
                admissions=(admission,),
                calls=(retry,),
                kv_inputs=(published,),
                **_installation_allocation(retry, 2),
            )
        )
        prepared_retry = finalized_report(consumer, prepared_retry)
        retry_report = prepared_retry
        assert retry_report.completions[0].status.value == "ok"
        assert retry_report.completions[0].kv_visible_len == 2
    finally:
        producer.close()
        consumer.close()
