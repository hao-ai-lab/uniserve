"""Native asynchronous request server exposed to the Python worker."""

from types import TracebackType
from typing import Any, Self

class Server:
    """Receives bounded IPC requests and publishes their responses."""

    def __init__(
        self,
        service_name: str,
        max_payload: int = 1_048_576,
        max_inflight: int = 1,
    ) -> None:
        """Bind a named service.

        The service bounds payload size and in-flight request capacity.
        """
        ...
    def __enter__(self) -> Self:
        """Return this open endpoint and close it when the scope exits."""
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the endpoint.

        An exception raised inside the scope is preserved.
        """
        ...
    @property
    def closed(self) -> bool:
        """Report whether the endpoint owner has released this service."""
        ...
    def close(self) -> None:
        """Idempotently release the service.

        The release happens after all endpoint operations have stopped.
        """
        ...
    def recv(self) -> Any:
        """Block until the next validated request is available."""
        ...
    def try_recv(self) -> Any | None:
        """Return the next request immediately.

        Return ``None`` when the queue is empty.
        """
        ...
    def wait_incoming(self, timeout_us: int) -> None:
        """Wait up to ``timeout_us`` for request or wake activity."""
        ...
    def wake(self) -> None:
        """Interrupt a pending receive wait from another thread."""
        ...
    def wake_on_stream(self, stream: int) -> None:
        """Schedule a server wake.

        The wake fires after the CUDA stream reaches its current point.
        """
        ...
    def respond(self, response: Any) -> None:
        """Publish one response to the request identified by its envelope."""
        ...

class StreamSignal:
    """Bridges CUDA stream completion into an asyncio-readable signal."""

    def __init__(self) -> None:
        """Create an owned completion descriptor.

        The descriptor receives CUDA stream notifications.
        """
        ...
    def fileno(self) -> int:
        """Return the readable descriptor signaled by completed stream work."""
        ...
    def schedule(self, stream: int) -> None:
        """Signal the descriptor.

        The signal fires after the CUDA stream reaches its current point.
        """
        ...
    def consume(self) -> None:
        """Drain pending readiness notifications from the descriptor."""
        ...
