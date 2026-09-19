from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from tests.python.fixtures.depth_one import (
    ar_params,
    bind_request_allocation,
    encode_call,
    execution_batch,
    finalized_report,
    record_completion,
    root_parent,
    stamp_batch,
    token_call,
    visual_state_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.simulation import expected_successor
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.protocol.batch import Finish, Free, NewRequest
from uniserve_worker.protocol.call import (
    CallCoordinates,
    CallStatus,
    ErrorCode,
    ForwardMode,
    TransferMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("device", "warm_start"),
    (
        ("cpu", False),
        ("cpu", True),
        pytest.param("cuda:0", True, marks=pytest.mark.gpu),
    ),
)
def test_independent_product_work_preserves_request_progress(
    device, warm_start
) -> None:
    from dataclasses import replace

    from uniserve_worker.protocol.call import Bounds, Call

    worker = execution_worker(device=device)
    admission = ar_params(79, block_ids=(0,))
    first = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
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
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(first,),
                )
            ),
        )
        assert produced.completions[0].status is CallStatus.OK
        observation = record_completion(first, produced)
        source = first.token_output
        copy = replace(source, producer_call_id=CallId(2, 0), generation=2)
        independent = Call(
            request_key=admission.request_key,
            call_id=CallId(2, 0),
            coordinates=CallCoordinates(),
            kind=TransferMode.TENSOR,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(copy,),
        )
        copied = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=2,
                    calls=(independent,),
                    commands=(),
                )
            ),
        )
        assert copied.completions[0].status is CallStatus.OK
        next_op = token_call(
            admission.request_key,
            call_id=CallId(3, 0),
            predecessor=observation.call_id,
            mode=ForwardMode.DECODE,
            tokens=(produced.completions[0].committed_tokens[0],),
        )
        continued = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=3,
                    calls=(next_op,),
                )
            ),
        )
        assert continued.completions[0].status is CallStatus.OK
        assert continued.completions[0].committed_tokens == (1001,)
        assert continued.completions[0].position == 2


@pytest.mark.parametrize(
    "backends,device",
    (
        (("local",), "cpu"),
        (("shm",), "cpu"),
        (("shm", "local"), "cpu"),
        pytest.param(("shm", "cuda_vmm"), "cuda:0", marks=pytest.mark.gpu),
        pytest.param(("cuda_vmm", "shm"), "cuda:0", marks=pytest.mark.gpu),
    ),
)
def test_retained_encoder_product_outlives_its_producer_request(
    backends: tuple[str, ...], device: str
) -> None:
    # A consumer binds the mechanisms its edge to the producer carries, which
    # a placement gives both ends alike: a device product travels on the
    # device mechanism and a host product on the host mechanism.
    producer = execution_worker(transfer_backends=backends, device=device)
    consumer = (
        producer
        if backends == ("local",)
        else execution_worker(transfer_backends=backends, device=device)
    )
    admission = ar_params(91, block_ids=(0,))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (64, 96, 128)).save(image, format="PNG")
    encode = encode_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        image_base64=base64.b64encode(image.getvalue()).decode("ascii"),
        encoder_handle=11,
        entry="vision_encoder",
    )
    product = encode.encoder_output
    try:
        produced = finalized_report(
            producer,
            producer.submit(
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(encode,),
                )
            ),
        )
        assert produced.completions[0].status is CallStatus.OK
        finish = Finish(
            admission.request_key,
            retained_buffers=(product.buffer_id,),
        )
        finalized_report(
            producer,
            producer.submit(execution_batch(batch_id=2, commands=(finish,))),
        )

        template = ar_params(92, block_ids=(1,))
        replacement = NewRequest(
            template.request_key,
            request_pool_idx=admission.request_pool_idx,
            generation=template.generation,
        )
        bind_request_allocation(
            replacement.request_key,
            request_pool_idx=replacement.request_pool_idx,
            page_ids=(1,),
        )
        visual = visual_state_call(
            replacement.request_key,
            call_id=CallId(1, 0),
            predecessor=root_parent(replacement),
            feature=product,
            sample_continuation=True,
            max_tokens=2,
        )
        batch = execution_batch(
            batch_id=3,
            admissions=(replacement,),
            calls=(visual,),
            input_products=produced.products,
        )
        if backends != ("local",):
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
        assert consumed.completions[0].status is CallStatus.OK
        assert consumed.completions[0].kv_visible_len == 2
        assert consumed.completions[0].committed_tokens == (
            expected_successor(1007),
        )

        read = producer.tensor_store.consume(
            product, consumer_call_id=CallId(2, 0)
        )
        freed = producer.submit(
            execution_batch(batch_id=4, commands=(Free(product.buffer_id),))
        )
        assert not freed.complete
        producer.tensor_store.complete_reads((read,))
        freed = finalized_report(producer, freed)
        with pytest.raises(WorkerError):
            producer.tensor_store.consume(
                product, consumer_call_id=CallId(3, 0)
            )
        finalized_report(
            producer,
            producer.submit(execution_batch(batch_id=5, commands=(finish,))),
        )
    finally:
        if consumer is not producer:
            consumer.close()
        producer.close()


@pytest.mark.parametrize("retirement", ("free", "finish"))
def test_command_acknowledgement_waits_for_readers_without_delaying_other_results(  # noqa: E501
    retirement: str,
) -> None:
    worker = execution_worker(queue_depth=2)
    worker.warmup()
    admission = ar_params(87, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        produced = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(call,),
                )
            ),
        )
        record_completion(call, produced)

        product = call.token_output
        read = worker.tensor_store.consume(
            product, consumer_call_id=CallId(2, 0)
        )
        command = (
            Finish(
                admission.request_key,
            )
            if retirement == "finish"
            else Free(product.buffer_id)
        )
        independent = ar_params(88, block_ids=(1,))
        next_call = token_call(
            independent.request_key,
            call_id=CallId(3, 0),
            predecessor=root_parent(independent),
            mode=ForwardMode.PREFILL,
            tokens=(7, 8),
        )
        batch = execution_batch(
            batch_id=3,
            admissions=(independent,),
            calls=(next_call,),
            commands=(command,),
        )

        later = ar_params(89, block_ids=(2,))
        later_call = token_call(
            later.request_key,
            call_id=CallId(4, 0),
            predecessor=root_parent(later),
            mode=ForwardMode.PREFILL,
            tokens=(9,),
        )
        later_run = execution_batch(
            batch_id=4,
            admissions=(later,),
            calls=(later_call,),
        )

        class Endpoint(QueuedWorkerIpc):
            def respond(self, response):
                super().respond(response)
                if response.get("message_id") == 2:
                    # The later batch owns no retired storage, so its result
                    # must arrive while the first batch still waits.
                    assert len(response["result"]["completions"]) == 1
                    worker.tensor_store.complete_reads((read,))
                elif response.get("message_id") == 1:
                    assert len(response["result"]["completions"]) == 1
                    self.submit({"kind": "close", "message_id": 3})

        endpoint = Endpoint(
            (
                {"kind": "submit", "batch": batch, "message_id": 1},
                {"kind": "submit", "batch": later_run, "message_id": 2},
            )
        )
        worker.bind(endpoint).run()
        # The retiring batch answers after the reader retires, behind the
        # independent batch it does not order.
        assert [response["message_id"] for response in endpoint.responses] == [
            2,
            1,
            3,
        ]
        assert endpoint.responses[-1]["kind"] == "ok"
    finally:
        worker.close()


def test_cancelled_admission_cannot_publish_over_a_reused_request_slot() -> (
    None
):
    worker = execution_worker()
    admission = ar_params(89, block_ids=(0,))
    call = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    try:
        pending = worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(admission,),
                calls=(call,),
            )
        )
        finish = Finish(
            admission.request_key,
        )
        finalized_report(
            worker,
            worker.submit(execution_batch(batch_id=2, commands=(finish,))),
        )
        template = ar_params(90, block_ids=(1,))
        replacement = NewRequest(
            template.request_key,
            request_pool_idx=admission.request_pool_idx,
            generation=template.generation,
        )
        bind_request_allocation(
            replacement.request_key,
            request_pool_idx=replacement.request_pool_idx,
            page_ids=(1,),
        )
        next_call = token_call(
            replacement.request_key,
            call_id=CallId(3, 0),
            predecessor=root_parent(replacement),
            mode=ForwardMode.PREFILL,
            tokens=(7, 8),
        )
        current = worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(replacement,),
                calls=(next_call,),
            )
        )
        pending = finalized_report(worker, pending)
        pending
        current = finalized_report(worker, current)
        completed = current
        assert completed.completions[0].status is CallStatus.OK
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
    closed_extend = token_call(
        closed_admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(closed_admission),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    active_extend = token_call(
        active_admission.request_key,
        call_id=CallId(1, 1),
        predecessor=root_parent(active_admission),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=1,
                admissions=(closed_admission, active_admission),
                calls=(closed_extend, active_extend),
            )
        ),
    )
    closed_observation = record_completion(closed_extend, report)
    active_observation = record_completion(active_extend, report)

    close = Finish(
        request_key=closed_admission.request_key,
    )
    worker.submit(execution_batch(batch_id=3, commands=(close,)))

    closed_decode = token_call(
        closed_admission.request_key,
        call_id=CallId(2, 0),
        predecessor=closed_observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(report.completions[0].committed_tokens[0],),
    )
    closed_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=4,
                calls=(closed_decode,),
            )
        ),
    )
    assert closed_report.completions[0].status is CallStatus.ERROR
    assert closed_report.completions[0].error_code is ErrorCode.INVALID_CALL

    active_decode = token_call(
        active_admission.request_key,
        call_id=CallId(2, 0),
        predecessor=active_observation.call_id,
        mode=ForwardMode.DECODE,
        tokens=(report.completions[1].committed_tokens[0],),
    )
    active_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=5,
                calls=(active_decode,),
            )
        ),
    )
    assert active_report.completions[0].status is CallStatus.OK
    assert active_report.completions[0].kv_visible_len == 3
    worker.close()


def test_drop_reuses_the_slot_and_rejects_the_retired_request_key() -> None:
    worker = execution_worker()
    retired = ar_params(83, block_ids=(0,))
    retired_call = token_call(
        retired.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(retired),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    worker.submit(
        execution_batch(
            batch_id=1,
            admissions=(retired,),
            calls=(retired_call,),
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
        generation=replacement_template.generation,
    )
    bind_request_allocation(
        replacement.request_key,
        request_pool_idx=replacement.request_pool_idx,
        page_ids=(1,),
    )
    replacement_call = token_call(
        replacement.request_key,
        call_id=CallId(2, 0),
        predecessor=root_parent(replacement),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    replacement_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=2,
                admissions=(replacement,),
                calls=(replacement_call,),
            )
        ),
    )
    assert replacement_report.completions[0].status is CallStatus.OK

    # A late close belongs to the retired epoch even after the slot and request
    # identifier have both been reused. The replacement must continue normally.
    finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                commands=(Finish(retired.request_key),),
            )
        ),
    )
    decoded = token_call(
        replacement.request_key,
        call_id=CallId(4, 0),
        predecessor=replacement_call.call_id,
        mode=ForwardMode.DECODE,
        tokens=(replacement_report.completions[0].committed_tokens[0],),
    )
    continued = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=4,
                calls=(decoded,),
            )
        ),
    )
    assert continued.completions[0].status is CallStatus.OK
    assert continued.completions[0].kv_visible_len == 3

    # The retired key is refused whatever call names it, so this repeats the
    # original call under its own identity rather than resubmitting one the
    # first batch still owns.
    retired_again = token_call(
        retired.request_key,
        call_id=CallId(5, 0),
        predecessor=root_parent(retired),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    retired_report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=5,
                calls=(retired_again,),
            )
        ),
    )
    assert retired_report.completions[0].status is CallStatus.ERROR
    assert retired_report.completions[0].error_code is ErrorCode.INVALID_CALL
    worker.close()


def test_finish_of_uninstalled_admission_allows_slot_reuse() -> None:
    worker = execution_worker()
    admission = ar_params(85, block_ids=(0,))
    finish = Finish(
        request_key=admission.request_key,
    )
    for batch_id in (1, 2):
        finalized_report(
            worker,
            worker.submit(
                execution_batch(batch_id=batch_id, commands=(finish,))
            ),
        )

    replacement = ar_params(86, block_ids=(0,))
    call = token_call(
        replacement.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(replacement),
        mode=ForwardMode.PREFILL,
        tokens=(3, 4),
    )
    report = finalized_report(
        worker,
        worker.submit(
            execution_batch(
                batch_id=3,
                admissions=(replacement,),
                calls=(call,),
            )
        ),
    )
    assert report.completions[0].status is CallStatus.OK
    worker.close()


def test_a_call_whose_coordinates_contradict_the_request_is_refused() -> None:
    """The engine states where a call executes; a rank refuses a wrong claim.

    A rank must not silently execute a call at a position other than the one
    it was scheduled for, so a claim that disagrees with the request's own
    progress ends the call rather than producing a token at the wrong place.
    """
    from dataclasses import replace

    worker = execution_worker()
    admission = ar_params(91, block_ids=(0,))
    prefill = token_call(
        admission.request_key,
        call_id=CallId(1, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(7, 8),
    )
    with worker:
        primed = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(prefill,),
                )
            ),
        )
        assert primed.completions[0].status is CallStatus.OK
        assert primed.completions[0].position == 2
        observation = record_completion(prefill, primed)

        decode = token_call(
            admission.request_key,
            call_id=CallId(2, 0),
            predecessor=observation.call_id,
            mode=ForwardMode.DECODE,
            tokens=(primed.completions[0].committed_tokens[0],),
        )
        run = stamp_batch(worker, execution_batch(batch_id=2, calls=(decode,)))
        # The prompt left the request at position two; claim the origin.
        contradicted = replace(
            run,
            calls=(replace(run.calls[0], coordinates=CallCoordinates()),),
        )
        refused = finalized_report(worker, worker.submit(contradicted))

        assert refused.completions[0].status is CallStatus.ERROR
        assert refused.completions[0].error_code is ErrorCode.INVALID_CALL
        assert refused.completions[0].committed_tokens == ()
