"""Real rank channels for worker service integration tests."""

from __future__ import annotations

import pytest

from uniserve_worker._uniserve_ipc import Client, Server


class WorkerChannel:
    """Own a TCP endpoint and its client; receive results on the test thread."""

    def __init__(self, requests: tuple[dict[str, object], ...] = ()) -> None:
        self.endpoint = Server("127.0.0.1", max_inflight=32, transport="tcp")
        self.client = Client(
            self.endpoint.endpoint(""), max_inflight=32, transport="tcp"
        )
        self._responses: list[dict[str, object]] = []
        self._sent = 0
        self._next_id = 1
        for request in requests:
            self.submit(request)

    def submit(self, request: dict[str, object]) -> None:
        request = dict(request)
        if "batch" in request:
            request["batch"] = request["batch"].to_mapping()
        if request.get("message_id") is None:
            request["message_id"] = self._next_id
        self._next_id = max(self._next_id, request["message_id"] + 1)
        self.client.send(request)
        self._sent += 1

    def receive(self, timeout: float = 5) -> dict[str, object]:
        response = self.client.recv(timeout)
        if response is None:
            raise TimeoutError("worker response did not arrive")
        self._responses.append(response)
        return response

    @property
    def responses(self) -> list[dict[str, object]]:
        while len(self._responses) < self._sent:
            self.receive()
        return self._responses

    def close(self) -> None:
        self.client.close()
        self.endpoint.close()


@pytest.fixture
def worker_channel():
    channels = []

    def create(requests=()):
        channel = WorkerChannel(requests)
        channels.append(channel)
        return channel

    yield create
    for channel in reversed(channels):
        channel.close()
