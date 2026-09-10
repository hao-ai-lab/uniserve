from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from tests.python.fixtures.depth_one import (
    ar_params,
    bind_request_allocation,
    commit_for_completion,
    encode_operation,
    execution_run,
    finalized_report,
    root_parent,
    token_operation,
    visual_state_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    CloseReason,
    ErrorCode,
    Finish,
    Free,
    NewRequest,
    OpStatus,
    ProductKind,
    Retire,
    TokenMode,
)
from uniserve_worker.execution.output import run_result_ready
from uniserve_worker.execution.run import RunReader, WorkerRun
from uniserve_worker.execution.step import execute_startup
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.models.stub import _next_token

pytestmark = pytest.mark.integration


def test_independent_product_work_preserves_the_committed_request_state() -> None:
    from dataclasses import replace

    from uniserve_worker.execution.batch import Bounds, OpCode, Operation

    worker = execution_worker()
    admission = ar_params(79, block_ids=(0,))
    first, tokens = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(7,),
    )
    try:
        produced = finalized_report(
            worker.execute(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(first,),
                    input_products=(tokens,),
                )
            )
        )
        assert produced.completions[0].status is OpStatus.OK
        commit = commit_for_completion(first, produced)
        source = next(value for value in first.outputs if value.kind is ProductKind.TOKEN)
        copy = replace(source, producer_op_id=2, generation=2)
        independent = Operation.registered(
            request_key=admission.request_key,
            op_id=2,
            parent=None,
            kind=OpCode.TRANSFER_PRODUCT,
            bounds=Bounds(max_transfer_bytes=source.max_bytes),
            inputs=(source,),
            outputs=(copy,),
        )
        copied = finalized_report(
            worker.execute(
                execution_run(
                    run_id=2,
                    operations=(independent,),
                    commands=(commit,),
                )
            )
        )
        assert copied.completions[0].status is OpStatus.OK
        next_op, next_tokens = token_operation(
            admission.request_key,
            op_id=3,
            parent=commit.selected,
            mode=TokenMode.DECODE,
            tokens=(produced.completions[0].committed_tokens[0],),
            control_seq=commit.control_seq,
        )
        continued = finalized_report(
            worker.execute(
                execution_run(
                    run_id=3,
                    operations=(next_op,),
                    input_products=(next_tokens,),
                )
            )
        )
        assert continued.completions[0].status is OpStatus.OK
        assert continued.completions[0].committed_tokens == (1001,)
        assert continued.completions[0].logical_lengths.token_len == 2
    finally:
        worker.close()


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
@pytest.mark.parametrize("retirement", ("finish", "retire"))
def test_retained_encoder_product_outlives_its_producer_request(
    backends: tuple[str, ...], device: str, retirement: str
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
    encode, payload = encode_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        image_base64=base64.b64encode(image.getvalue()).decode("ascii"),
        encoder_handle=11,
    )
    product = encode.outputs[0]
    try:
        produced = finalized_report(
            producer.execute(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(encode,),
                    input_products=(payload,),
                )
            )
        )
        assert produced.completions[0].status is OpStatus.OK
        finish = Finish(
            admission.request_key,
            1,
            root_parent(admission),
            CloseReason.COMPLETED,
            retained_buffers=(product.buffer_id,),
        )
        if retirement == "retire":
            finish = Retire(admission.request_key, retained_buffers=(product.buffer_id,))
        assert finalized_report(producer.execute(execution_run(run_id=2, commands=(finish,)))).done

        template = ar_params(92, block_ids=(1,))
        replacement = NewRequest.create(
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
            op_id=1,
            parent=root_parent(replacement),
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
            prepared = consumer.prepare_execute(batch)
            assert prepared is not None
            deadline = time.monotonic() + 5
            while not prepared.ready() and time.monotonic() < deadline:
                time.sleep(0.001)
            assert prepared.ready()
            consumed = finalized_report(consumer.execute_prepared(prepared))
        else:
            consumed = finalized_report(consumer.execute(batch))
        assert consumed.completions[0].status is OpStatus.OK
        assert consumed.completions[0].logical_lengths.kv_visible_len == 2
        assert consumed.completions[0].committed_tokens == (_next_token(1007),)

        read = producer.encoder_cache.consume(product, consumer_op_id=2)
        freed = producer.execute(execution_run(run_id=4, commands=(Free(product.buffer_id),)))
        assert not run_result_ready(freed)
        producer.encoder_cache.record_readers((read,))
        assert finalized_report(freed).done
        with pytest.raises(WorkerError):
            producer.encoder_cache.consume(product, consumer_op_id=3)
        assert finalized_report(producer.execute(execution_run(run_id=5, commands=(finish,)))).done
    finally:
        if consumer is not producer:
            consumer.close()
        producer.close()


@pytest.mark.parametrize("retirement", ("free", "finish", "retire"))
@pytest.mark.parametrize("startup", (False, True))
def test_command_acknowledgement_waits_for_readers_without_delaying_other_results(
    retirement: str,
    startup: bool,
) -> None:
    worker = execution_worker(pipeline_depth=2)
    admission = ar_params(87, block_ids=(0,))
    operation, payload = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    try:
        produced = finalized_report(
            worker.execute(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(operation,),
                    input_products=(payload,),
                )
            )
        )
        commit = commit_for_completion(operation, produced)
        finalized_report(worker.execute(execution_run(run_id=2, commands=(commit,))))
        product = next(output for output in operation.outputs if output.kind is ProductKind.TOKEN)
        read = worker.device_products.consume(product, consumer_op_id=2)
        command = (
            Finish(
                admission.request_key,
                commit.control_seq + 1,
                commit.selected,
                CloseReason.CANCELLED,
            )
            if retirement == "finish"
            else Retire(admission.request_key)
            if retirement == "retire"
            else Free(product.buffer_id)
        )
        independent = ar_params(88, block_ids=(1,))
        next_operation, next_payload = token_operation(
            independent.request_key,
            op_id=1,
            parent=root_parent(independent),
            mode=TokenMode.EXTEND,
            tokens=(7, 8),
        )
        batch = worker.plan_run(
            execution_run(
                run_id=3,
                admissions=(independent,),
                operations=(next_operation,),
                input_products=(next_payload,),
                commands=(command,),
            )
        )
        run = WorkerRun(
            batch,
            on_successors_ready=lambda _: None,
            on_ready=lambda _: None,
            on_terminal=lambda _: None,
        )
        run.attach(execute_startup(worker, batch) if startup else worker.execute(batch))
        reader = RunReader(run, lambda _: None)
        replay = RunReader(run, lambda _: None)
        assert reader.ready()
        partial = reader.take_ready()
        assert not partial.done
        assert partial.completions[0].request_key == independent.request_key
        assert partial.completions[0].status is OpStatus.OK
        assert reader.pending()
        assert not reader.ready()
        assert replay.take_ready().to_mapping() == partial.to_mapping()

        worker.device_products.record_readers((read,))
        assert reader.ready()
        terminal = reader.take_ready()
        assert terminal.done
        assert not terminal.completions
        assert not reader.pending()
        assert replay.take_ready().to_mapping() == terminal.to_mapping()
        assert not replay.pending()
    finally:
        worker.close()


def test_cancelled_admission_cannot_publish_over_a_reused_request_slot() -> None:
    worker = execution_worker()
    admission = ar_params(89, block_ids=(0,))
    operation, payload = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    try:
        pending = worker.execute(
            execution_run(
                run_id=1,
                admissions=(admission,),
                operations=(operation,),
                input_products=(payload,),
            )
        )
        finish = Finish(admission.request_key, 1, root_parent(admission), CloseReason.CANCELLED)
        finalized_report(worker.execute(execution_run(run_id=2, commands=(finish,))))
        template = ar_params(90, block_ids=(1,))
        replacement = NewRequest.create(
            template.request_key,
            request_pool_idx=admission.request_pool_idx,
            ar=template.ar,
        )
        bind_request_allocation(
            replacement.request_key, request_pool_idx=replacement.request_pool_idx, page_ids=(1,)
        )
        next_operation, next_payload = token_operation(
            replacement.request_key,
            op_id=1,
            parent=root_parent(replacement),
            mode=TokenMode.EXTEND,
            tokens=(7, 8),
        )
        current = worker.execute(
            execution_run(
                run_id=3,
                admissions=(replacement,),
                operations=(next_operation,),
                input_products=(next_payload,),
            )
        )
        finalized_report(pending)
        completed = finalized_report(current)
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
    closed_extend, closed_input = token_operation(
        closed_admission.request_key,
        op_id=1,
        parent=root_parent(closed_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    active_extend, active_input = token_operation(
        active_admission.request_key,
        op_id=1,
        parent=root_parent(active_admission),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    report = finalized_report(
        worker.execute(
            execution_run(
                run_id=1,
                admissions=(closed_admission, active_admission),
                operations=(closed_extend, active_extend),
                input_products=(closed_input, active_input),
            )
        )
    )
    closed_commit = commit_for_completion(closed_extend, report)
    active_commit = commit_for_completion(active_extend, report)
    worker.execute(execution_run(run_id=2, commands=(closed_commit, active_commit)))
    close = Finish(
        request_key=closed_admission.request_key,
        control_seq=closed_commit.control_seq + 1,
        cutoff=closed_commit.selected,
        reason=CloseReason.CANCELLED,
    )
    worker.execute(execution_run(run_id=3, commands=(close,)))

    closed_decode, closed_decode_input = token_operation(
        closed_admission.request_key,
        op_id=2,
        parent=closed_commit.selected,
        mode=TokenMode.DECODE,
        tokens=(report.completions[0].committed_tokens[0],),
        control_seq=close.control_seq,
    )
    closed_report = finalized_report(
        worker.execute(
            execution_run(
                run_id=4,
                operations=(closed_decode,),
                input_products=(closed_decode_input,),
            )
        )
    )
    assert closed_report.completions[0].status is OpStatus.ERROR
    assert closed_report.completions[0].error_code is ErrorCode.INVALID_OPERATION

    active_decode, active_decode_input = token_operation(
        active_admission.request_key,
        op_id=2,
        parent=active_commit.selected,
        mode=TokenMode.DECODE,
        tokens=(report.completions[1].committed_tokens[0],),
        control_seq=active_commit.control_seq,
    )
    active_report = finalized_report(
        worker.execute(
            execution_run(
                run_id=5,
                operations=(active_decode,),
                input_products=(active_decode_input,),
            )
        )
    )
    assert active_report.completions[0].status is OpStatus.OK
    assert active_report.completions[0].logical_lengths.kv_visible_len == 3
    worker.close()


def test_drop_reuses_the_slot_and_rejects_the_retired_request_key() -> None:
    worker = execution_worker()
    retired = ar_params(83, block_ids=(0,))
    retired_operation, retired_input = token_operation(
        retired.request_key,
        op_id=1,
        parent=root_parent(retired),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        execution_run(
            run_id=1,
            admissions=(retired,),
            operations=(retired_operation,),
            input_products=(retired_input,),
        )
    )
    worker.drop_request(retired.request_key.request_id)

    replacement_template = ar_params(84, block_ids=(1,))
    replacement = NewRequest.create(
        replacement_template.request_key,
        request_pool_idx=retired.request_pool_idx,
        ar=replacement_template.ar,
    )
    bind_request_allocation(
        replacement.request_key,
        request_pool_idx=replacement.request_pool_idx,
        page_ids=(1,),
    )
    replacement_operation, replacement_input = token_operation(
        replacement.request_key,
        op_id=1,
        parent=root_parent(replacement),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    replacement_report = finalized_report(
        worker.execute(
            execution_run(
                run_id=2,
                admissions=(replacement,),
                operations=(replacement_operation,),
                input_products=(replacement_input,),
            )
        )
    )
    assert replacement_report.completions[0].status is OpStatus.OK

    retired_report = finalized_report(
        worker.execute(
            execution_run(
                run_id=3,
                operations=(retired_operation,),
                input_products=(retired_input,),
            )
        )
    )
    assert retired_report.completions[0].status is OpStatus.ERROR
    assert retired_report.completions[0].error_code is ErrorCode.INVALID_OPERATION
    worker.close()


def test_finish_of_uninstalled_admission_allows_slot_reuse() -> None:
    worker = execution_worker()
    admission = ar_params(85, block_ids=(0,))
    finish = Finish(
        request_key=admission.request_key,
        control_seq=1,
        cutoff=root_parent(admission),
        reason=CloseReason.ERROR,
    )
    for run_id in (1, 2):
        report = finalized_report(worker.execute(execution_run(run_id=run_id, commands=(finish,))))
        assert report.done

    replacement = ar_params(86, block_ids=(0,))
    operation, input_product = token_operation(
        replacement.request_key,
        op_id=1,
        parent=root_parent(replacement),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    report = finalized_report(
        worker.execute(
            execution_run(
                run_id=3,
                admissions=(replacement,),
                operations=(operation,),
                input_products=(input_product,),
            )
        )
    )
    assert report.completions[0].status is OpStatus.OK
    worker.close()
