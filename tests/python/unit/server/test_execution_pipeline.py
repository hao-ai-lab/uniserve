"""Worker request-pipeline scheduling properties."""

from __future__ import annotations

from collections import deque

import pytest

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
    def close(self) -> None:
        pass


class _Endpoint:
    def __init__(self, requests: tuple[dict[str, object], ...]) -> None:
        self.requests = deque(requests)

    def try_recv(self) -> dict[str, object] | None:
        return self.requests.popleft() if self.requests else None

    def recv(self) -> dict[str, object]:
        return self.requests.popleft()

    def respond(self, _response: dict[str, object]) -> None:
        pass


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
