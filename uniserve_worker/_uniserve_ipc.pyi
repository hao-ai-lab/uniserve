"""Native asynchronous request server exposed to the Python worker."""

from types import TracebackType
from typing import Any, Self, final

__all__ = [
    "Server",
    "StreamSignal",
    "atomic_load_u32",
    "atomic_store_u32",
    "service_name",
]

@final
class Server:
    """Receives bounded IPC requests and publishes their responses."""

    def __new__(
        cls,
        service_name: str,
        max_payload: int = 1_048_576,
        max_inflight: int = 1,
        transport: str = ...,
    ) -> Self:
        """Bind this rank's channel.

        The channel bounds payload size and in-flight request capacity.
        ``transport`` is ``"iceoryx2"``, the default, for a rank on the head's
        host, which serves shared storage under ``service_name``, or ``"tcp"``
        for a rank elsewhere, where ``service_name`` is the interface to bind.
        """
        ...
    def endpoint(self, service: str) -> str:
        """Return the endpoint this rank reports to the head.

        A shared-storage endpoint is ``service`` itself; a socket endpoint is
        the address its bind produced.
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

@final
class StreamSignal:
    """Bridges CUDA stream completion into an asyncio-readable signal."""

    def __new__(cls) -> Self:
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

def service_name(id: str) -> str:
    """Return the shared-storage service name for one endpoint identifier."""
    ...

def atomic_store_u32(buffer: memoryview, offset: int, value: int) -> None:
    """Store a 32-bit word with release ordering.

    ``buffer`` must be writable and contiguous, and the word at ``offset``
    must be four-byte aligned and lie inside it; otherwise ``RuntimeError``
    is raised.
    """
    ...

def atomic_load_u32(buffer: memoryview, offset: int) -> int:
    """Load a 32-bit word with acquire ordering.

    ``buffer`` carries the same requirements as for ``atomic_store_u32``.
    """
    ...
