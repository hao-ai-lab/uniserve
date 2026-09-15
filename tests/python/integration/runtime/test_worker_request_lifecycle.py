from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from tests.python.fixtures.depth_one import (
    ar_params,
    bind_request_allocation,
    encode_operation,
    execution_run,
    finalized_report,
    record_completion,
    root_parent,
    token_operation,
    visual_state_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from tests.python.fixtures.simulation import expected_successor
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.protocol.batch import (
    ComputationId,
    ErrorCode,
    Finish,
    ForwardMode,
    Free,
    NewRequest,
    OpStatus,
    RequestKey,
    TransferMode,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("device", "warm_start"),
    (("cpu", False), ("cpu", True), pytest.param("cuda:0", True, marks=pytest.mark.gpu)),
)
def test_independent_product_work_preserves_request_progress(device, warm_start) -> None:
    from dataclasses import replace

    from uniserve_worker.protocol.batch import Bounds, ScheduledRequest

    worker = execution_worker(device=device)
    admission = ar_params(79, block_ids=(0,))
    first = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(7,),
    )
    with worker:
        if warm_start:
            worker.warmup()
        produced = finalized_report(
            worker,
            worker.submit(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(first,),
                )
            ),
        )
        assert produced.completions[0].status is OpStatus.OK
        observation = record_completion(first, produced)
        source = first.token_output
        copy = replace(source, producer_op_id=ComputationId(2, 0), generation=2)
        independent = ScheduledRequest(
            request_key=admission.request_key,
            op_id=ComputationId(2, 0),
            predecessor=None,
            kind=TransferMode.TENSOR,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(copy,),
        )
        copied = finalized_report(
            worker,
            worker.submit(
                execution_run(
                    run_id=2,
                    operations=(independent,),
                    commands=(),
                )
            ),
        )
        assert copied.completions[0].status is OpStatus.OK
        next_op = token_operation(
            admission.request_key,
            op_id=ComputationId(3, 0),
            predecessor=observation.op_id,
            mode=ForwardMode.DECODE,
            tokens=(produced.completions[0].committed_tokens[0],),
        )
        continued = finalized_report(
            worker,
            worker.submit(
                execution_run(
                    run_id=3,
                    operations=(next_op,),
                )
            ),
        )
        assert continued.completions[0].status is OpStatus.OK
        assert continued.completions[0].committed_tokens == (1001,)
        assert continued.completions[0].position == 2


@pytest.mark.parametrize(
    "backends,device",
    (
        (("local",), "cpu"),
        (("shm",), "cpu"),
        (("shm", "local"), "cpu"),
        pytest.param(("shm", "cuda_ipc"), "cuda:0", marks=pytest.mark.gpu),
        pytest.param(("cuda_ipc", "shm"), "cuda:0", marks=pytest.mark.gpu),
    ),
)
def test_retained_encoder_product_outlives_its_producer_request(
    backends: tuple[str, ...], device: str
) -> None:
    backend = backends[0]
    producer = execution_worker(transfer_backends=backends, device=device)
    consumer = (
        producer
        if backend == "local"
        else execution_worker(transfer_backends=(backend,), device=device)
    )
    admission = ar_params(91, block_ids=(0,))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (64, 96, 128)).save(image, format="PNG")
    encode = encode_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        image_base64=base64.b64encode(image.getvalue()).decode("ascii"),
        encoder_handle=11,
    )
    product = encode.encoder_output
    try:
        produced = finalized_report(
            producer,
            producer.submit(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(encode,),
                )
            ),
        )
        assert produced.completions[0].status is OpStatus.OK
        finish = Finish(
            admission.request_key,
            retained_buffers=(product.buffer_id,),
        )
        assert finalized_report(
            producer, producer.submit(execution_run(run_id=2, commands=(finish,)))
        ).done

        template = ar_params(92, block_ids=(1,))
        replacement = NewRequest(
            template.request_key,
            request_pool_idx=admission.request_pool_idx,
            ar=template.ar,
        )
        bind_request_allocation(
            replacement.request_key,
            request_pool_idx=replacement.request_pool_idx,
            page_ids=(1,),
        )
        visual = visual_state_operation(
            replacement.request_key,
            op_id=ComputationId(1, 0),
            predecessor=root_parent(replacement),
            feature=product,
            sample_continuation=True,
            max_tokens=2,
        )
        batch = execution_run(
            run_id=3,
            admissions=(replacement,),
            operations=(visual,),
            input_products=produced.products,
        )
        if backend != "local":
            prepared = consumer.submit(batch)
            assert prepared is not None
            deadline = time.monotonic() + 5
            while not prepared.inputs_ready() and time.monotonic() < deadline:
                consumer.advance_inputs(prepared)
                time.sleep(0.001)
            assert prepared.inputs_ready()
            prepared = finalized_report(consumer, prepared)
            consumed = prepared
        else:
            consumed = finalized_report(consumer, consumer.submit(batch))
        assert consumed.completions[0].status is OpStatus.OK
        assert consumed.completions[0].kv_visible_len == 2
        assert consumed.completions[0].committed_tokens == (expected_successor(1007),)

        read = producer.tensor_store.consume(product, consumer_op_id=ComputationId(2, 0))
        freed = producer.submit(execution_run(run_id=4, commands=(Free(product.buffer_id),)))
        assert not freed.complete
        producer.tensor_store.complete_reads((read,))
        freed = finalized_report(producer, freed)
        assert freed.done
        with pytest.raises(WorkerError):
            producer.tensor_store.consume(product, consumer_op_id=ComputationId(3, 0))
        assert finalized_report(
            producer, producer.submit(execution_run(run_id=5, commands=(finish,)))
        ).done
    finally:
        if consumer is not producer:
            consumer.close()
        producer.close()


@pytest.mark.parametrize("retirement", ("free", "finish"))
def test_command_acknowledgement_waits_for_readers_without_delaying_other_results(
    retirement: str,
) -> None:
    worker = execution_worker(pipeline_depth=2)
    worker.warmup()
    admission = ar_params(87, block_ids=(0,))
    operation = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        produced = finalized_report(
            worker,
            worker.submit(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(operation,),
                )
            ),
        )
        record_completion(operation, produced)

        product = operation.token_output
        read = worker.tensor_store.consume(product, consumer_op_id=ComputationId(2, 0))
        command = (
            Finish(
                admission.request_key,
            )
            if retirement == "finish"
            else Free(product.buffer_id)
        )
        independent = ar_params(88, block_ids=(1,))
        next_operation = token_operation(
            independent.request_key,
            op_id=ComputationId(1, 0),
            predecessor=root_parent(independent),
            mode=ForwardMode.PREFILL,
            tokens=(7, 8),
        )
        batch = execution_run(
            run_id=3,
            admissions=(independent,),
            operations=(next_operation,),
            commands=(command,),
        )

        later = ar_params(89, block_ids=(2,))
        later_operation = token_operation(
            later.request_key,
            op_id=ComputationId(1, 0),
            predecessor=root_parent(later),
            mode=ForwardMode.PREFILL,
            tokens=(9,),
        )
        later_run = execution_run(
            run_id=4,
            admissions=(later,),
            operations=(later_operation,),
        )

        class Endpoint(QueuedWorkerIpc):
            def respond(self, response):
                super().respond(response)
                if response.get("call_id") == 1:
                    partial = response["result"]
                    assert not partial["done"]
                    assert len(partial["completions"]) == 1
                    self.submit({"kind": "submit", "run": batch, "call_id": 2})
                elif response.get("call_id") == 2:
                    assert response["kind"] == "error"
                    assert response["code"] == "InvalidDescriptor"
                    self.submit({"kind": "poll", "run_id": 3, "call_id": 4})
                    self.submit({"kind": "submit", "run": later_run, "call_id": 3})
                elif response.get("call_id") == 3:
                    result = response["result"]
                    assert result["done"]
                    assert len(result["completions"]) == 1
                    worker.tensor_store.complete_reads((read,))
                elif response.get("call_id") == 4:
                    terminal = response["result"]
                    assert terminal["done"]
                    assert not terminal["completions"]
                    self.submit({"kind": "close", "call_id": 5})

        endpoint = Endpoint(({"kind": "submit", "run": batch, "call_id": 1},))
        worker.bind(endpoint).run()
        assert [response["call_id"] for response in endpoint.responses] == [1, 2, 3, 4, 5]
        assert endpoint.responses[-1]["kind"] == "ok"
    finally:
        worker.close()


def test_cancelled_admission_cannot_publish_over_a_reused_request_slot() -> None:
    worker = execution_worker()
    admission = ar_params(89, block_ids=(0,))
    operation = token_operation(
        admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        pending = worker.submit(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
            )
        )
        finish = Finish(
            admission.request_key,
        )
        finalized_report(worker, worker.submit(execution_run(run_id=2, commands=(finish,))))
        template = ar_params(90, block_ids=(1,))
        replacement = NewRequest(
            template.request_key,
            request_pool_idx=admission.request_pool_idx,
            ar=template.ar,
        )
        bind_request_allocation(
            replacement.request_key, request_pool_idx=replacement.request_pool_idx, page_ids=(1,)
        )
        next_operation = token_operation(
            replacement.request_key,
            op_id=ComputationId(1, 0),
            predecessor=root_parent(replacement),
            mode=ForwardMode.PREFILL,
            tokens=(7, 8),
        )
        current = worker.submit(
            execution_run(
                run_id=3,
                admissions=(replacement,),
                operations=(next_operation,),
            )
        )
        pending = finalized_report(worker, pending)
        pending
        current = finalized_report(worker, current)
        completed = current
        assert completed.completions[0].status is OpStatus.OK
        assert completed.completions[0].request_key == replacement.request_key
        assert (
            worker.requests.get(replacement.request_key.request_id).request_key
            == replacement.request_key
        )
    finally:
        worker.close()


def test_close_rejects_descendants_without_affecting_another_request() -> None:
    worker = execution_worker()
    closed_admission = ar_params(81, block_ids=(0,))
    active_admission = ar_params(82, block_ids=(1,))
    closed_extend = token_operation(
        closed_admission.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(closed_admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    active_extend = token_operation(
        active_admission.request_key,
        op_id=ComputationId(1, 1),
        predecessor=root_parent(active_admission),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=1,
                admissions=(closed_admission, active_admission),
                operations=(closed_extend, active_extend),
            )
        ),
    )
    closed_observation = record_completion(closed_extend, report)
    active_observation = record_completion(active_extend, report)

    close = Finish(
        request_key=closed_admission.request_key,
    )
    worker.submit(execution_run(run_id=3, commands=(close,)))

    closed_decode = token_operation(
        closed_admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=closed_observation.op_id,
        mode=ForwardMode.DECODE,
        tokens=(report.completions[0].committed_tokens[0],),
    )
    closed_report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=4,
                operations=(closed_decode,),
            )
        ),
    )
    assert closed_report.completions[0].status is OpStatus.ERROR
    assert closed_report.completions[0].error_code is ErrorCode.INVALID_OPERATION

    active_decode = token_operation(
        active_admission.request_key,
        op_id=ComputationId(2, 0),
        predecessor=active_observation.op_id,
        mode=ForwardMode.DECODE,
        tokens=(report.completions[1].committed_tokens[0],),
    )
    active_report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=5,
                operations=(active_decode,),
            )
        ),
    )
    assert active_report.completions[0].status is OpStatus.OK
    assert active_report.completions[0].kv_visible_len == 3
    worker.close()


def test_drop_reuses_the_slot_and_rejects_the_retired_request_key() -> None:
    worker = execution_worker()
    retired = ar_params(83, block_ids=(0,))
    retired_operation = token_operation(
        retired.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(retired),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    worker.submit(
        execution_run(
            run_id=1,
            admissions=(retired,),
            operations=(retired_operation,),
        )
    )
    worker.drop_request(retired.request_key.request_id)

    replacement_template = ar_params(84, block_ids=(1,))
    replacement = NewRequest(
        RequestKey(
            engine_id=retired.request_key.engine_id,
            request_id=retired.request_key.request_id,
            request_epoch=retired.request_key.request_epoch + 1,
        ),
        request_pool_idx=retired.request_pool_idx,
        ar=replacement_template.ar,
    )
    bind_request_allocation(
        replacement.request_key,
        request_pool_idx=replacement.request_pool_idx,
        page_ids=(1,),
    )
    replacement_operation = token_operation(
        replacement.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(replacement),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    replacement_report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=2,
                admissions=(replacement,),
                operations=(replacement_operation,),
            )
        ),
    )
    assert replacement_report.completions[0].status is OpStatus.OK

    # A late close belongs to the retired epoch even after the slot and request
    # identifier have both been reused. The replacement must continue normally.
    finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=3,
                commands=(Finish(retired.request_key),),
            )
        ),
    )
    decoded = token_operation(
        replacement.request_key,
        op_id=ComputationId(2, 0),
        predecessor=replacement_operation.op_id,
        mode=ForwardMode.DECODE,
        tokens=(replacement_report.completions[0].committed_tokens[0],),
    )
    continued = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=4,
                operations=(decoded,),
            )
        ),
    )
    assert continued.completions[0].status is OpStatus.OK
    assert continued.completions[0].kv_visible_len == 3

    retired_report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=5,
                operations=(retired_operation,),
            )
        ),
    )
    assert retired_report.completions[0].status is OpStatus.ERROR
    assert retired_report.completions[0].error_code is ErrorCode.INVALID_OPERATION
    worker.close()


def test_finish_of_uninstalled_admission_allows_slot_reuse() -> None:
    worker = execution_worker()
    admission = ar_params(85, block_ids=(0,))
    finish = Finish(
        request_key=admission.request_key,
    )
    for run_id in (1, 2):
        report = finalized_report(
            worker, worker.submit(execution_run(run_id=run_id, commands=(finish,)))
        )
        assert report.done

    replacement = ar_params(86, block_ids=(0,))
    operation = token_operation(
        replacement.request_key,
        op_id=ComputationId(1, 0),
        predecessor=root_parent(replacement),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_run(
                run_id=3,
                admissions=(replacement,),
                operations=(operation,),
            )
        ),
    )
    assert report.completions[0].status is OpStatus.OK
    worker.close()
