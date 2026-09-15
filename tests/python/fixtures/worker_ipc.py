"""Queued CPU IPC boundary for worker service integration tests."""

from __future__ import annotations

from queue import Empty, Full, Queue


class QueuedWorkerIpc:
    """Deliver requests, responses and latched wake signals.

    Delivery happens between test threads.
    """

    def __init__(self, requests: tuple[dict[str, object], ...] = ()) -> None:
        self.closed = False
        self._requests: Queue[dict[str, object]] = Queue()
        self._responses: Queue[dict[str, object]] = Queue()
        self._wake: Queue[None] = Queue(maxsize=1)
        self.responses: list[dict[str, object]] = []
        for request in requests:
            self.submit(request)

    def close(self) -> None:
        self.closed = True

    def submit(self, request: dict[str, object]) -> None:
        self._requests.put(request)
        self.wake()

    def receive(self, timeout: float = 5) -> dict[str, object]:
        return self._responses.get(timeout=timeout)

    def try_recv(self) -> dict[str, object] | None:
        try:
            return self._requests.get_nowait()
        except Empty:
            return None

    def recv(self) -> dict[str, object]:
        return self._requests.get()

    def wait_incoming(self, timeout_us: int) -> None:
        try:
            self._wake.get(timeout=timeout_us / 1_000_000)
        except Empty:
            pass

    def wake(self) -> None:
        try:
            self._wake.put_nowait(None)
        except Full:
            pass

    def wake_on_stream(self, _stream: int) -> None:
        raise AssertionError("CPU IPC endpoint received CUDA work")

    def respond(self, response: dict[str, object]) -> None:
        self.responses.append(response)
        self._responses.put(response)
