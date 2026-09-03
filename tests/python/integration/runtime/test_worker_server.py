"""Worker-server idempotence over concrete execution and completion resources."""

from __future__ import annotations

from collections import deque

import pytest

from tests.python.fixtures.depth_one import (
    ar_params,
    execution_run,
    root_parent,
    token_operation,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.execution.batch import (
    NewRequest,
    Operation,
    ProductPayload,
    Run,
    TokenMode,
)
from uniserve_worker.process import WorkerProcess

pytestmark = pytest.mark.integration


class _Endpoint:
    def __init__(self, requests: tuple[dict[str, object], ...]) -> None:
        self._requests = deque(requests)
        self.responses: list[dict[str, object]] = []

    def try_recv(self) -> dict[str, object] | None:
        return self._requests.popleft() if self._requests else None

    def recv(self) -> dict[str, object]:
        return self._requests.popleft()

    def wait_incoming(self, timeout_us: int) -> None:
        del timeout_us

    def respond(self, response: dict[str, object]) -> None:
        self.responses.append(response)


def _request(call_id: int, run: Run) -> dict[str, object]:
    return {"kind": "submit", "call_id": call_id, "run": run}


def _token_run(
    *,
    request_id: int,
    op_id: int,
    run_id: int,
    tokens: tuple[int, ...],
) -> tuple[NewRequest, Operation, ProductPayload, Run]:
    admission = ar_params(request_id, block_ids=(request_id,))
    operation, payload = token_operation(
        admission.request_key,
        op_id=op_id,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=tokens,
    )
    return (
        admission,
        operation,
        payload,
        execution_run(
            run_id=run_id,
            admissions=(admission,),
            operations=(operation,),
            input_products=(payload,),
        ),
    )


def _by_call(endpoint: _Endpoint) -> dict[int, dict[str, object]]:
    return {
        int(response["call_id"]): response
        for response in endpoint.responses
        if response.get("call_id") is not None
    }


def test_info_request_is_served_by_the_process_queue() -> None:
    endpoint = _Endpoint(
        (
            {"kind": "info", "call_id": 1},
            {"kind": "close", "call_id": 2},
        )
    )
    server = WorkerProcess(execution_worker(pipeline_depth=1), endpoint)

    server.serve()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "info"
    assert responses[1]["info"] == server.worker.info.to_mapping()


def test_inflight_and_terminal_duplicates_return_one_terminal_report() -> None:
    _admission, _operation, _payload, run = _token_run(
        request_id=11,
        op_id=21,
        run_id=7,
        tokens=(8, 9),
    )
    endpoint = _Endpoint(
        (
            _request(1, run),
            _request(2, run),
            _request(3, run),
            {"kind": "close", "call_id": 4},
        )
    )
    server = WorkerProcess(execution_worker(pipeline_depth=2), endpoint)

    server.serve()

    responses = _by_call(endpoint)
    first = responses[1]["result"]
    assert responses[2]["result"] == first
    assert responses[3]["result"] == first


def test_conflicting_run_identity_fails_before_new_admission() -> None:
    admission, operation, payload, run = _token_run(
        request_id=12,
        op_id=31,
        run_id=8,
        tokens=(4, 5),
    )
    conflicting, conflicting_payload = token_operation(
        admission.request_key,
        op_id=31,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(4, 5, 6),
    )
    conflicting_run = execution_run(
        run_id=8,
        operations=(conflicting,),
        input_products=(conflicting_payload,),
    )
    next_admission, next_operation, next_payload, _next_run = _token_run(
        request_id=13,
        op_id=32,
        run_id=8,
        tokens=(7,),
    )
    mixed_run = execution_run(
        run_id=8,
        admissions=(next_admission,),
        operations=(operation, next_operation),
        input_products=(payload, next_payload),
    )
    endpoint = _Endpoint(
        (
            _request(1, run),
            _request(2, conflicting_run),
            _request(3, mixed_run),
            {"kind": "close", "call_id": 4},
        )
    )
    server = WorkerProcess(execution_worker(pipeline_depth=3), endpoint)

    server.serve()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "result"
    assert responses[2]["kind"] == "error"
    assert responses[2]["code"] == "InvalidDescriptor"
    assert responses[3]["kind"] == "error"
    assert responses[3]["code"] == "InvalidDescriptor"


def test_completed_report_remains_retained_after_later_execution() -> None:
    _first_admission, first_operation, _first_payload, first_run = _token_run(
        request_id=5,
        op_id=41,
        run_id=11,
        tokens=(1, 2),
    )
    _second_admission, _second_operation, _second_payload, second_run = _token_run(
        request_id=6,
        op_id=42,
        run_id=12,
        tokens=(3, 4),
    )
    endpoint = _Endpoint(
        (
            _request(1, first_run),
            _request(2, second_run),
            _request(3, first_run),
            {"kind": "close", "call_id": 4},
        )
    )
    server = WorkerProcess(
        execution_worker(
            pipeline_depth=1,
                max_batch_operations=1,
            max_request_pool_size=8,
        ),
        endpoint,
        replay_capacity=4,
    )

    server.serve()

    responses = _by_call(endpoint)
    first = responses[1]["result"]
    retained = responses[3]["result"]
    assert responses[2]["kind"] == "result"
    assert retained == first
