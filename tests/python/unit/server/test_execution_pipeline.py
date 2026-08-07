"""Worker request-pipeline scheduling properties."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from tests.python.fixtures.depth_one import (
    execution_batch,
    root_parent,
    token_operation,
    und_admission,
)
from uniserve_worker.batch import Batch, CompletionReport, PartitionCompletion, TokenMode
from uniserve_worker.capabilities import ResponseKind
from uniserve_worker.server.app import WorkerServer, _response
from uniserve_worker.server.process import WorkerServeLoop

pytestmark = pytest.mark.unit


class _Metrics:
    def __init__(self) -> None:
        self.clock = 0

    def now_ns(self) -> int:
        self.clock += 1
        return self.clock

    def record_pipeline(self, _phase: str, _duration: int) -> None:
        pass


class _Profiler:
    def close(self) -> None:
        pass


class _Worker:
    def __init__(self) -> None:
        self.contract = SimpleNamespace(capabilities=SimpleNamespace(pipeline_depth=2))
        self.execute_calls = 0

    def execute(self, _batch: object) -> CompletionReport:
        self.execute_calls += 1
        raise AssertionError("completion polling must not execute model work")

    def close(self) -> None:
        pass


class _Endpoint:
    def __init__(self, requests: tuple[dict[str, object], ...]) -> None:
        self.requests = deque(requests)
        self.responses: list[dict[str, object]] = []

    def try_recv(self) -> dict[str, object] | None:
        return self.requests.popleft() if self.requests else None

    def recv(self) -> dict[str, object]:
        return self.requests.popleft()

    def wait_incoming(self, timeout_us: int) -> None:
        return None

    def respond(self, response: dict[str, object]) -> None:
        self.responses.append(response)


class _Server:
    pipeline_depth = 2

    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.metrics = _Metrics()
        self.profiler = _Profiler()
        self.worker = _Worker()

    def handle(self, request: dict[str, object]) -> dict[str, object]:
        call_id = int(request["call_id"])
        self.actions.append(f"handle:{call_id}")
        return {"kind": "ok", "call_id": call_id}

    def respond(self, response: dict[str, object]) -> None:
        self.actions.append(f"respond:{int(response['call_id'])}")


def test_pipeline_launches_to_depth_before_finalizing_the_oldest_response() -> None:
    actions: list[str] = []
    endpoint = _Endpoint(
        (
            {"kind": "execute", "call_id": 1},
            {"kind": "execute", "call_id": 2},
            {"kind": "shutdown", "call_id": 3},
        )
    )
    WorkerServeLoop(_Server(actions), endpoint).run()  # type: ignore[arg-type]

    assert actions == [
        "handle:1",
        "handle:2",
        "respond:1",
        "handle:3",
        "respond:2",
        "respond:3",
    ]


def test_pipeline_dispatches_ready_responses_in_lineage_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[str] = []
    server = _Server(actions)
    loop = WorkerServeLoop(server, _Endpoint(()))  # type: ignore[arg-type]
    loop.inflight.extend(
        (
            (
                {
                    "call_id": 1,
                    "batch": {
                        "partitions": [{"operations": [{"request_key": {"session_id": 7}}]}],
                    },
                },
                {"call_id": 1, "ready": False},
            ),
            (
                {
                    "call_id": 2,
                    "batch": {
                        "partitions": [{"operations": [{"request_key": {"session_id": 7}}]}],
                    },
                },
                {"call_id": 2, "ready": True},
            ),
            (
                {
                    "call_id": 3,
                    "batch": {
                        "partitions": [{"operations": [{"request_key": {"session_id": 9}}]}],
                    },
                },
                {"call_id": 3, "ready": True},
            ),
        )
    )
    monkeypatch.setattr(
        "uniserve_worker.server.app._response_ready",
        lambda response: bool(response["ready"]),
    )

    assert loop._respond_ready()
    loop.inflight[0][1]["ready"] = True
    assert loop._respond_ready()
    assert loop._respond_ready()
    assert actions == ["respond:3", "respond:1", "respond:2"]


def test_server_classifies_partition_readiness_once_and_polls_the_deferred_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _Worker()
    endpoint = _Endpoint(())
    server = WorkerServer(worker, endpoint)  # type: ignore[arg-type]
    readiness_checks: dict[int, int] = {}

    def readiness(partition: PartitionCompletion) -> bool:
        count = readiness_checks.get(partition.partition_id, 0) + 1
        readiness_checks[partition.partition_id] = count
        return partition.partition_id == 1 or count > 1

    monkeypatch.setattr(
        "uniserve_worker.server.app.partition_completion_ready",
        readiness,
    )
    report = CompletionReport(
        step_id=44,
        partitions=(
            PartitionCompletion(partition_id=1, completions=()),
            PartitionCompletion(partition_id=2, completions=()),
        ),
    )
    response = _response(ResponseKind.RESULT, completion_report=report)
    response["call_id"] = 101

    server.respond(response)

    first = endpoint.responses[0]["completion_report"]
    assert isinstance(first, dict)
    assert [partition["partition_id"] for partition in first["partitions"]] == [1]
    assert tuple(server._pending_completion_reports) == (44,)

    polled = server.handle({"kind": "poll_completions", "call_id": 102, "step_id": 44})
    assert polled["call_id"] == 102
    assert isinstance(polled["completion_report"], CompletionReport)
    assert [partition.partition_id for partition in polled["completion_report"].partitions] == [2]

    server.respond(polled)

    second = endpoint.responses[1]["completion_report"]
    assert isinstance(second, dict)
    assert [partition["partition_id"] for partition in second["partitions"]] == [2]
    assert server._pending_completion_reports == {}
    assert readiness_checks == {1: 1, 2: 2}
    assert worker.execute_calls == 0


class _DeferredPrepared:
    def __init__(self, ready_after: int) -> None:
        self.ready_after = int(ready_after)
        self.queries = 0

    def ready(self) -> bool:
        self.queries += 1
        return self.queries >= self.ready_after


class _TransferWorker:
    def __init__(self, batches: tuple[Batch, ...], actions: list[str]) -> None:
        work = tuple(operation.work.variant for batch in batches for operation in batch.operations)
        self.contract = SimpleNamespace(
            capabilities=SimpleNamespace(
                pipeline_depth=2,
                supported_work=work,
                supported_controls=(),
            )
        )
        self.actions = actions
        self.prepared = _DeferredPrepared(ready_after=3)

    def prepare_execute(self, batch: Batch) -> object | None:
        if batch.step_id == 1:
            self.actions.append("transfer-submitted:1")
            return self.prepared
        return None

    def execute_prepared(self, prepared: object) -> CompletionReport:
        assert prepared is self.prepared
        self.actions.append("execute:1")
        return CompletionReport(step_id=1, partitions=())

    def execute(self, batch: Batch) -> CompletionReport:
        self.actions.append(f"execute:{batch.step_id}")
        return CompletionReport(step_id=batch.step_id, partitions=())

    def close(self) -> None:
        pass


def _token_batch(step_id: int, session_id: int) -> Batch:
    admission = und_admission(session_id)
    operation, payload = token_operation(
        admission.request_key,
        op_id=1,
        parent=root_parent(admission),
        mode=TokenMode.EXTEND,
        tokens=(1,),
    )
    return execution_batch(
        step_id=step_id,
        admissions=(admission,),
        operations=(operation,),
        input_products=(payload,),
    )


def test_pending_stage_transfer_does_not_block_an_unrelated_request() -> None:
    actions: list[str] = []
    batches = (_token_batch(1, 101), _token_batch(2, 202))
    endpoint = _Endpoint(
        (
            {"kind": "execute", "call_id": 1, "batch": batches[0]},
            {"kind": "execute", "call_id": 2, "batch": batches[1]},
            {"kind": "shutdown", "call_id": 3},
        )
    )
    worker = _TransferWorker(batches, actions)
    WorkerServeLoop(WorkerServer(worker, endpoint), endpoint).run()  # type: ignore[arg-type]

    assert actions == ["transfer-submitted:1", "execute:2", "execute:1"]
    assert [response["call_id"] for response in endpoint.responses] == [2, 1, 3]
