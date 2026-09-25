"""Worker protocol and ownership.

The ownership covers concrete execution and completion resources.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError, replace

import pytest

from tests.python.fixtures.depth_one import (
    ar_params,
    execution_batch,
    finalized_report,
    root_parent,
    token_call,
)
from tests.python.fixtures.execution_worker import execution_worker
from tests.python.fixtures.worker_ipc import QueuedWorkerIpc
from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.executor import Submission
from uniserve_worker.protocol.batch import Batch, Finish, NewRequest
from uniserve_worker.protocol.call import Call, CallStatus, ForwardMode
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.integration


def _request(message_id: int, batch: Batch) -> dict[str, object]:
    return {"kind": "submit", "message_id": message_id, "batch": batch}


def _token_run(
    *,
    request_id: int,
    call_id: CallId,
    batch_id: int,
    tokens: tuple[int, ...],
) -> tuple[NewRequest, Call, Batch]:
    admission = ar_params(request_id, block_ids=(request_id,))
    call = token_call(
        admission.request_key,
        call_id=call_id,
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=tokens,
    )
    return (
        admission,
        call,
        execution_batch(
            batch_id=batch_id,
            admissions=(admission,),
            calls=(call,),
        ),
    )


def _by_call(endpoint: QueuedWorkerIpc) -> dict[int, dict[str, object]]:
    return {
        int(response["message_id"]): response
        for response in endpoint.responses
        if response.get("message_id") is not None
    }


def test_info_request_is_served_before_close() -> None:
    endpoint = QueuedWorkerIpc(
        (
            {"kind": "info", "message_id": 1},
            {"kind": "close", "message_id": 2},
        )
    )
    with execution_worker(queue_depth=1) as worker:
        worker.bind(endpoint)
        worker.run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "info"
    assert responses[1]["info"] == worker.info.to_mapping()
    assert [response["message_id"] for response in endpoint.responses] == [1, 2]
    assert responses[2]["kind"] == "ok"


def test_direct_admission_preserves_identity_and_bounded_delivery():
    commands = (Finish(RequestKey(1, 1, 1)),)
    with execution_worker(queue_depth=1) as worker:
        first = worker.submit(Batch(batch_id=7, commands=commands))
        with pytest.raises(FrozenInstanceError):
            first.batch_id = 9
        with pytest.raises(WorkerError, match="no longer owned"):
            worker.poll(Submission(first.batch_id))
        with pytest.raises(WorkerError, match="admission queue is full"):
            worker.submit(Batch(batch_id=8, commands=commands))
        assert worker.poll(first).batch_id == 7
        with pytest.raises(WorkerError, match="no longer owned"):
            worker.poll(first)
        with pytest.raises(RuntimeError, match="warmup must precede"):
            worker.warmup()
        with pytest.raises(WorkerError, match="must exceed"):
            worker.submit(Batch(batch_id=7, commands=commands))
        # Capacity rejection applied nothing: this identity can now be used.
        second = worker.submit(Batch(batch_id=8, commands=commands))
        assert worker.poll(second).batch_id == 8


@pytest.mark.parametrize("queue_depth", (1, 3))
def test_duplicate_submissions_are_rejected_while_the_original_completes(
    queue_depth,
) -> None:
    _admission, _call, run = _token_run(
        request_id=11,
        call_id=CallId(21, 0),
        batch_id=7,
        tokens=(8, 9),
    )
    endpoint = QueuedWorkerIpc(
        (
            _request(1, run),
            _request(2, run),
            _request(3, run),
            {"kind": "close", "message_id": 4},
        )
    )
    with execution_worker(queue_depth=queue_depth) as worker:
        worker.bind(endpoint)
        worker.run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "result"
    for message_id in (2, 3):
        assert responses[message_id]["kind"] == "error"
        assert responses[message_id]["code"] == "InvalidDescriptor"
    assert responses[4]["kind"] == "ok"


def test_failed_submissions_cannot_be_reused_and_allow_shutdown() -> None:
    with execution_worker(queue_depth=2) as worker:
        admission = replace(
            ar_params(91), request_pool_idx=worker.info.request_slots + 1
        )
        run = execution_batch(batch_id=1, admissions=(admission,))
        endpoint = QueuedWorkerIpc(
            (
                _request(1, run),
                _request(2, run),
                {"kind": "close", "message_id": 3},
            )
        )
        worker.bind(endpoint).run()

    responses = _by_call(endpoint)
    for message_id in (1, 2):
        response = responses[message_id]
        assert response["kind"] == "error"
        assert response["code"] == "InvalidDescriptor"
        assert response["fatal"] is False
    assert [response["message_id"] for response in endpoint.responses] == [
        1,
        2,
        3,
    ]
    assert responses[3]["kind"] == "ok"


def test_execution_failure_is_logged_under_the_submit_request(
    caplog,
) -> None:
    """A batch accepted for execution and failed there names its request.

    The failure reaches the engine as an error response to the submission and
    is logged as a failed ``submit`` request.
    """
    with execution_worker(queue_depth=1) as worker:
        # The admission is decoded and accepted, then refused when execution
        # installs it into a request slot the worker does not have.
        admission = replace(
            ar_params(92), request_pool_idx=worker.info.request_slots + 1
        )
        endpoint = QueuedWorkerIpc(
            (
                _request(
                    1, execution_batch(batch_id=1, admissions=(admission,))
                ),
                {"kind": "close", "message_id": 2},
            )
        )
        with caplog.at_level(logging.WARNING, logger="uniserve_worker"):
            worker.bind(endpoint).run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "error"
    assert responses[1]["code"] == "InvalidDescriptor"
    failures = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("worker request ")
    ]
    assert len(failures) == 1
    assert failures[0].startswith("worker request 'submit' failed")


def test_execution_failure_reports_the_call_kind_as_its_route(caplog) -> None:
    """A failure inside a batch's execution names the kind it executed.

    Every call of a batch shares one kind, so the batch failure log and an
    error raised to a propagating caller both report it as the route.
    """
    admissions = (ar_params(93, block_ids=(0,)), ar_params(94, block_ids=(1,)))
    # A text extension without input tokens is refused while the batch runs.
    calls = tuple(
        replace(
            token_call(
                admission.request_key,
                call_id=CallId(index + 1, 0),
                predecessor=root_parent(admission),
                mode=ForwardMode.PREFILL,
                tokens=(3, 4),
            ),
            input_token_ids=(),
        )
        for index, admission in enumerate(admissions)
    )
    with (
        execution_worker(queue_depth=1) as worker,
        caplog.at_level(
            logging.WARNING, logger="uniserve_worker.execution.step"
        ),
    ):
        served = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=1, admissions=admissions[:1], calls=calls[:1]
                )
            ),
        )
        with pytest.raises(WorkerError) as raised:
            worker.submit(
                execution_batch(
                    batch_id=2, admissions=admissions[1:], calls=calls[1:]
                ),
                propagate_errors=True,
            )

    assert served.completions[0].status is CallStatus.ERROR
    assert raised.value.to_mapping()["route"] == "prefill"
    # One failure log per batch, the served and the propagated one.
    logged = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("batch failed")
    ]
    assert len(logged) == 2
    assert all("route=prefill " in message for message in logged)


def test_refused_call_batch_reports_its_call_kind_route_over_ipc() -> None:
    """A batch refused before its calls launch still names their kind."""
    with execution_worker(queue_depth=1) as worker:
        admission = replace(
            ar_params(95), request_pool_idx=worker.info.request_slots + 1
        )
        call = token_call(
            admission.request_key,
            call_id=CallId(1, 0),
            predecessor=root_parent(admission),
            mode=ForwardMode.PREFILL,
            tokens=(3, 4),
        )
        endpoint = QueuedWorkerIpc(
            (
                _request(
                    1,
                    execution_batch(
                        batch_id=1, admissions=(admission,), calls=(call,)
                    ),
                ),
                {"kind": "close", "message_id": 2},
            )
        )
        worker.bind(endpoint).run()

    response = _by_call(endpoint)[1]
    assert response["kind"] == "error"
    assert response["code"] == "InvalidDescriptor"
    assert response["route"] == "prefill"


def test_a_batch_id_that_does_not_advance_is_refused_before_new_admission() -> (
    None
):
    """`batch_id` is the submission identity and must strictly advance.

    A refused submission applies nothing: neither its admission nor its
    call_kinds take effect, and the same work still runs once it is
    submitted under an advancing identity.
    """
    admission, call, run = _token_run(
        request_id=12,
        call_id=CallId(2, 0),
        batch_id=8,
        tokens=(4, 5),
    )
    conflicting = token_call(
        admission.request_key,
        call_id=CallId(2, 0),
        predecessor=root_parent(admission),
        mode=ForwardMode.PREFILL,
        tokens=(4, 5, 6),
    )
    conflicting_run = execution_batch(
        batch_id=8,
        calls=(conflicting,),
    )
    next_admission, next_call, _superseded = _token_run(
        request_id=13,
        call_id=CallId(2, 1),
        batch_id=8,
        tokens=(7,),
    )
    # Reuses the accepted identity while carrying a new admission.
    reused_identity_run = execution_batch(
        batch_id=8,
        admissions=(next_admission,),
        calls=(call, next_call),
    )
    _, _, advancing_run = _token_run(
        request_id=13,
        call_id=CallId(3, 1),
        batch_id=9,
        tokens=(7,),
    )
    endpoint = QueuedWorkerIpc(
        (
            _request(1, run),
            _request(2, conflicting_run),
            _request(3, reused_identity_run),
            _request(4, advancing_run),
            {"kind": "close", "message_id": 5},
        )
    )
    with execution_worker(queue_depth=3) as worker:
        worker.bind(endpoint)
        worker.run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "result"
    assert responses[2]["kind"] == "error"
    assert responses[2]["code"] == "InvalidDescriptor"
    assert responses[3]["kind"] == "error"
    assert responses[3]["code"] == "InvalidDescriptor"
    # Rejected work must not admit its request; the same computation still
    # executes once its batch identity advances.
    result = responses[4]["result"]
    assert result["batch_id"] == 3
    assert result["completions"][0]["call_id"] == {
        "batch_id": 3,
        "request_index": 1,
    }
    assert result["completions"][0]["status"] == "ok"
    assert responses[5]["kind"] == "ok"


def test_rebinding_rejects_both_same_and_different_endpoints() -> None:
    endpoint = QueuedWorkerIpc(({"kind": "close", "message_id": 1},))
    other = QueuedWorkerIpc(({"kind": "info", "message_id": 2},))
    worker = execution_worker()
    try:
        assert worker.bind(endpoint) is worker
        for candidate in (endpoint, other):
            with pytest.raises(RuntimeError, match="already.*bound"):
                worker.bind(candidate)
        worker.run()
        assert endpoint.responses[0]["message_id"] == 1
        assert other.try_recv() == {"kind": "info", "message_id": 2}
        assert not endpoint.closed
        assert not other.closed
    finally:
        worker.close()


def test_unbound_run_failure_closes_worker_at_scope_exit() -> None:
    worker = execution_worker()
    with pytest.raises(RuntimeError, match="no bound IPC endpoint"):
        with worker:
            worker.run()
    with pytest.raises(RuntimeError, match="closed"):
        worker.bind(QueuedWorkerIpc())
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(
            execution_batch(batch_id=1, commands=(Finish(RequestKey(1, 1, 1)),))
        )
    worker.close()


def test_closed_endpoint_is_rejected_without_consuming_worker() -> None:
    endpoint = QueuedWorkerIpc()
    endpoint.close()
    worker = execution_worker()
    try:
        with pytest.raises(ValueError, match="open IPC endpoint"):
            worker.bind(endpoint)
        worker.bind(QueuedWorkerIpc(({"kind": "close"},))).run()
    finally:
        worker.close()


def test_service_is_single_use_and_scope_exit_prevents_reuse() -> None:
    worker = execution_worker().bind(QueuedWorkerIpc(({"kind": "close"},)))
    with worker:
        worker.run()
        with pytest.raises(RuntimeError, match="only run once"):
            worker.run()
    for action in (worker.run, worker.warmup):
        with pytest.raises(RuntimeError, match="closed"):
            action()
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(
            execution_batch(batch_id=1, commands=(Finish(RequestKey(1, 1, 1)),))
        )
    worker.close()


def test_close_releases_borrowed_endpoint_references() -> None:
    import weakref

    endpoint = QueuedWorkerIpc()
    reference = weakref.ref(endpoint)
    worker = execution_worker().bind(endpoint)
    worker.close()
    assert not endpoint.closed
    del endpoint
    assert reference() is None


@pytest.mark.parametrize("entrypoint", ("run", "warmup"))
def test_warmup_failure_preserves_error_and_leaves_requests_unconsumed(
    monkeypatch, tmp_path, entrypoint: str
) -> None:
    import torch

    from tests.python.fixtures.launch import worker_args
    from uniserve_worker.worker import Worker

    config = worker_args(
        tmp_path,
        max_batch_tokens=256,
        max_batch_calls=2,
        no_model=True,
        allow_stub=True,
        kv_token_capacity=4096,
    )
    request = {"kind": "info", "message_id": 1}
    endpoint = QueuedWorkerIpc((request,))
    failure = RuntimeError("numerical backend unavailable")

    def unavailable(*args, **kwargs):
        raise failure

    # PyTorch is the external numerical backend. Construction and binding must
    # succeed even when entering numerical warmup cannot succeed.
    # Decorated numerical entry points already retain context instances;
    # failing their entry exercises the backend boundary those instances use.
    monkeypatch.setattr(torch.inference_mode, "__enter__", unavailable)
    worker = Worker.from_config(config)
    with pytest.raises(RuntimeError) as caught:
        with worker:
            if entrypoint == "run":
                worker.bind(endpoint)
            getattr(worker, entrypoint)()
    assert caught.value is failure
    assert endpoint.try_recv() == request
    assert endpoint.responses == []
    assert not endpoint.closed
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(
            execution_batch(batch_id=1, commands=(Finish(RequestKey(1, 1, 1)),))
        )


@pytest.mark.parametrize("gc_enabled", (False, True))
def test_transport_failure_releases_worker_and_restores_gc(
    gc_enabled: bool,
) -> None:
    import gc

    failure = OSError("IPC connection lost")

    class DisconnectedEndpoint(QueuedWorkerIpc):
        def try_recv(self):
            raise failure

    was_enabled = gc.isenabled()
    endpoint = DisconnectedEndpoint()
    worker = execution_worker().bind(endpoint)
    try:
        (gc.enable if gc_enabled else gc.disable)()
        with pytest.raises(OSError) as caught:
            with worker:
                worker.run()
        assert caught.value is failure
        assert gc.isenabled() == gc_enabled
        assert not endpoint.closed
        with pytest.raises(RuntimeError, match="closed"):
            worker.submit(
                execution_batch(
                    batch_id=1, commands=(Finish(RequestKey(1, 1, 1)),)
                )
            )
    finally:
        (gc.enable if was_enabled else gc.disable)()
        worker.close()


def test_successful_manual_warmup_is_retained_by_run(monkeypatch) -> None:
    import torch

    worker = execution_worker()
    try:
        worker.warmup()

        def unavailable(*args, **kwargs):
            raise RuntimeError("numerical startup is unavailable")

        # Administrative serving needs no further numerical startup once the
        # execution-only caller has successfully warmed up the worker.
        monkeypatch.setattr(torch.inference_mode, "__enter__", unavailable)
        endpoint = QueuedWorkerIpc(
            ({"kind": "info", "message_id": 1}, {"kind": "close"})
        )
        worker.bind(endpoint).run()
        assert endpoint.responses[0]["kind"] == "info"
        assert endpoint.responses[-1]["kind"] == "ok"
    finally:
        worker.close()


def test_startup_failure_keeps_resources_until_scope_exit(monkeypatch) -> None:
    import weakref

    import torch

    worker = execution_worker()
    model = weakref.ref(worker.model)
    endpoint = QueuedWorkerIpc()

    def unavailable(*args, **kwargs):
        raise ValueError("startup unavailable")

    monkeypatch.setattr(torch.inference_mode, "__enter__", unavailable)
    with worker:
        with pytest.raises(ValueError):
            worker.warmup()
        assert model() is not None

    assert model() is None
    assert not endpoint.closed


@pytest.mark.parametrize("gc_enabled", (False, True))
def test_normal_shutdown_reports_cleanup_failure_and_restores_gc(
    monkeypatch, gc_enabled
) -> None:
    import concurrent.futures
    import gc

    endpoint = QueuedWorkerIpc(({"kind": "close", "message_id": 1},))
    worker = execution_worker().bind(endpoint)
    shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    failure = OSError("executor shutdown failed")

    def failed_shutdown(executor, *args, **kwargs):
        shutdown(executor, *args, **kwargs)
        raise failure

    monkeypatch.setattr(
        concurrent.futures.ThreadPoolExecutor, "shutdown", failed_shutdown
    )
    was_enabled = gc.isenabled()
    try:
        (gc.enable if gc_enabled else gc.disable)()
        with pytest.raises(OSError) as caught:
            with worker:
                worker.run()
        assert caught.value is failure
        assert gc.isenabled() == gc_enabled
        assert endpoint.responses[0]["kind"] == "ok"
        assert endpoint.responses[0]["message_id"] == 1
        assert not endpoint.closed
        with pytest.raises(RuntimeError, match="closed"):
            worker.run()
    finally:
        (gc.enable if was_enabled else gc.disable)()
        worker.close()


def test_scope_retains_model_after_run_and_releases_it_on_exit() -> None:
    import gc
    import weakref

    from uniserve_models.stub import Model

    model = Model()
    reference = weakref.ref(model)
    worker = execution_worker(model).bind(QueuedWorkerIpc(({"kind": "close"},)))
    del model
    with worker as owned:
        assert owned is worker
        worker.run()
        gc.collect()
        assert reference() is not None
    gc.collect()
    assert reference() is None
    with pytest.raises(RuntimeError, match="closed"):
        with worker:
            pytest.fail("closed workers cannot enter another resource scope")
