"""Worker-server idempotence over concrete execution and completion resources."""

from __future__ import annotations

from collections import deque

import pytest

from tests.python.fixtures.depth_one import (
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import (
    Admission,
    Batch,
    Operation,
    ProductPayload,
    TokenMode,
)
from uniserve_worker.server.app import WorkerServer
from uniserve_worker.server.process import WorkerServeLoop

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


def _request(call_id: int, batch: Batch) -> dict[str, object]:
    return {"kind": "execute", "call_id": call_id, "batch": batch}


def _token_batch(
    *,
    session_id: int,
    op_id: int,
    step_id: int,
    tokens: tuple[int, ...],
) -> tuple[Admission, Operation, ProductPayload, Batch]:
    admission = und_admission(session_id, block_ids=(session_id,))
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
        execution_batch(
            step_id=step_id,
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


def test_inflight_join_and_completed_replay_return_one_terminal_report() -> None:
    _admission, _operation, _payload, batch = _token_batch(
        session_id=11,
        op_id=21,
        step_id=7,
        tokens=(8, 9),
    )
    endpoint = _Endpoint(
        (
            _request(1, batch),
            _request(2, batch),
            _request(3, batch),
            {"kind": "shutdown", "call_id": 4},
        )
    )
    server = WorkerServer(execution_worker(pipeline_depth=2), endpoint)

    WorkerServeLoop(server, endpoint).run()

    responses = _by_call(endpoint)
    first = responses[1]["completion_report"]
    assert responses[2]["completion_report"] == first
    assert responses[3]["completion_report"] == first


def test_conflicting_digest_and_mixed_registration_fail_before_new_admission() -> None:
    admission, operation, payload, batch = _token_batch(
        session_id=12,
        op_id=31,
        step_id=8,
        tokens=(4, 5),
    )
    conflicting, conflicting_payload = token_operation(
        admission.request_key,
        op_id=31,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(4, 5, 6),
    )
    conflicting_batch = execution_batch(
        step_id=9,
        operations=(conflicting,),
        input_products=(conflicting_payload,),
    )
    next_admission, next_operation, next_payload, _next_batch = _token_batch(
        session_id=13,
        op_id=32,
        step_id=10,
        tokens=(7,),
    )
    mixed_batch = execution_batch(
        step_id=10,
        admissions=(next_admission,),
        operations=(operation, next_operation),
        input_products=(payload, next_payload),
    )
    endpoint = _Endpoint(
        (
            _request(1, batch),
            _request(2, conflicting_batch),
            _request(3, mixed_batch),
            {"kind": "shutdown", "call_id": 4},
        )
    )
    server = WorkerServer(execution_worker(pipeline_depth=3), endpoint)

    WorkerServeLoop(server, endpoint).run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "result"
    assert responses[2]["kind"] == "error"
    assert responses[2]["code"] == "InvalidDescriptor"
    assert responses[3]["kind"] == "error"
    assert responses[3]["code"] == "InvalidDescriptor"


def test_completed_report_remains_replayable_after_later_execution() -> None:
    _first_admission, first_operation, _first_payload, first_batch = _token_batch(
        session_id=5,
        op_id=41,
        step_id=11,
        tokens=(1, 2),
    )
    _second_admission, _second_operation, _second_payload, second_batch = _token_batch(
        session_id=6,
        op_id=42,
        step_id=12,
        tokens=(3, 4),
    )
    endpoint = _Endpoint(
        (
            _request(1, first_batch),
            {
                "kind": "release_products",
                "call_id": 2,
                "product_handles": [
                    int(output.generation) for output in first_operation.outputs
                ],
            },
            _request(3, second_batch),
            _request(4, first_batch),
            {"kind": "shutdown", "call_id": 5},
        )
    )
    server = WorkerServer(
        execution_worker(
            pipeline_depth=1,
            max_batch_operations=1,
            max_request_pool_size=8,
        ),
        endpoint,
        replay_capacity=4,
    )

    WorkerServeLoop(server, endpoint).run()

    responses = _by_call(endpoint)
    first = responses[1]["completion_report"]
    replayed = responses[4]["completion_report"]
    assert responses[3]["kind"] == "result"
    assert replayed == first


def test_prompt_launches_after_an_earlier_queued_same_session_control() -> None:
    _admission, _operation, _payload, batch = _token_batch(
        session_id=19,
        op_id=43,
        step_id=14,
        tokens=(3, 5),
    )
    endpoint = _Endpoint(
        (
            {"kind": "drop_session", "call_id": 1, "session_id": 19},
            _request(2, batch),
            {"kind": "shutdown", "call_id": 3},
        )
    )
    server = WorkerServer(execution_worker(pipeline_depth=2), endpoint)

    WorkerServeLoop(server, endpoint).run()

    responses = _by_call(endpoint)
    assert responses[1]["kind"] == "ok"
    assert responses[2]["kind"] == "result"


def test_atomic_replay_remains_available_until_every_participant_epoch_ends() -> None:
    first_admission, first_operation, first_payload, _first_batch = _token_batch(
        session_id=6,
        op_id=51,
        step_id=13,
        tokens=(5,),
    )
    second_admission, second_operation, second_payload, _second_batch = _token_batch(
        session_id=7,
        op_id=52,
        step_id=13,
        tokens=(6,),
    )
    batch = execution_batch(
        step_id=13,
        admissions=(first_admission, second_admission),
        operations=(first_operation, second_operation),
        input_products=(first_payload, second_payload),
    )
    endpoint = _Endpoint(
        (
            _request(1, batch),
            {"kind": "drop_session", "call_id": 2, "session_id": 6},
            _request(3, batch),
            {"kind": "shutdown", "call_id": 4},
        )
    )
    server = WorkerServer(
        execution_worker(
            pipeline_depth=1,
            max_batch_operations=2,
            max_request_pool_size=8,
        ),
        endpoint,
    )

    WorkerServeLoop(server, endpoint).run()

    responses = _by_call(endpoint)
    assert responses[2]["kind"] == "ok"
    assert responses[3]["completion_report"] == responses[1]["completion_report"]
