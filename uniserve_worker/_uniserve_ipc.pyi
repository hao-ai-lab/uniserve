"""Type stub for the common native worker and IPC extension.

``Server`` is the rank's end of its channel to the engine: it receives the
engine's requests and publishes the worker's responses over iceoryx2 shared
storage or a TCP socket. ``StreamSignal`` turns CUDA stream completion into a
readable descriptor for selector loops. ``atomic_store_u32`` and
``atomic_load_u32`` order the header words of shared-storage segments, which
``uniserve_worker.transport.segment`` reads and writes across processes.
``Request`` and ``RequestPool`` own the lifecycle shared by serving and direct
numerical execution.
"""

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self, final

import torch

from uniserve.sampling import SamplingParams
from uniserve.tensors import BufferConfig
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.request import RequestProgress, RequestResult
from uniserve_worker.protocol.batch import BatchCommand, NewRequest
from uniserve_worker.protocol.call import Call, ImageParams
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.storage.request_slots import RequestSlots

__all__ = [
    "Request",
    "RequestPool",
    "Server",
    "StreamSignal",
    "atomic_load_u32",
    "atomic_store_u32",
    "service_name",
]

@final
class Request:
    """An admitted epoch whose lifecycle is mutated only by its request pool."""

    @property
    def request_id(self) -> int: ...
    @property
    def request_key(self) -> RequestKey: ...
    @property
    def request_pool_idx(self) -> int: ...
    @property
    def admission(self) -> NewRequest: ...
    @property
    def sampling(self) -> SamplingParams | None: ...
    @property
    def image(self) -> ImageParams | None: ...
    @property
    def negative_token_ids(self) -> tuple[int, ...]: ...
    @property
    def finish_token_ids(self) -> tuple[int, ...]: ...
    @property
    def accepted_progress(self) -> RequestProgress: ...
    @property
    def prompt_logits_ready(self) -> bool: ...
    @property
    def rng_counter(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    @property
    def retired(self) -> bool: ...
    diffusion: DiffusionState | None

@final
class RequestPool:
    """Bind scheduler slots, order request calls, and retire drained state."""

    def __new__(
        cls,
        max_request_pool_size: int,
        *,
        state_buffers: Mapping[str, BufferConfig] | None = None,
        device: torch.device | str = "cpu",
    ) -> Self: ...
    @property
    def max_request_pool_size(self) -> int: ...
    @property
    def storage(self) -> RequestSlots: ...
    def close(self) -> None: ...
    def get(self, request_id: int) -> Request: ...
    def peek(self, request_id: int) -> Request | None: ...
    def request_ids(self) -> tuple[int, ...]: ...
    def has_open_requests(self) -> bool: ...
    def bind_calls(
        self, calls: Sequence[Call], request_pool_indices: Sequence[int]
    ) -> tuple[Request, ...]: ...
    def validate_pending(self, calls: Sequence[Call]) -> None: ...
    def add_pending(self, calls: Sequence[Call]) -> None: ...
    def predecessors(
        self, calls: Sequence[Call]
    ) -> dict[CallId, CallId | None]: ...
    def apply_result(self, result: RequestResult) -> None: ...
    def cancel_calls(self, calls: Sequence[Call]) -> None: ...
    def start(self, admission: NewRequest) -> int | None: ...
    def finish(self, request_key: RequestKey) -> None: ...
    def apply_commands(
        self, commands: Sequence[BatchCommand]
    ) -> tuple[int, ...]: ...
    def retirement_ready(self, request_key: RequestKey) -> bool: ...
    def drop(self, request_id: int) -> None: ...
    def retire(self, request_id: int) -> None: ...

@final
class Server:
    """Receives bounded IPC requests and publishes their responses.

    ``recv``, ``try_recv``, ``wait_incoming`` and ``respond`` take exclusive
    use of the endpoint and release the GIL while they hold it. A call to any
    of them, to ``endpoint`` or to ``close`` made meanwhile from another
    thread raises ``RuntimeError`` instead of waiting. ``wake`` and
    ``wake_on_stream`` do not need the endpoint and may be called from any
    thread while it is in use.
    """

    def __new__(
        cls,
        service_name: str,
        max_payload: int = 1_048_576,
        max_inflight: int = 1,
        transport: str = ...,
    ) -> Self:
        """Bind this rank's channel.

        The channel bounds payload size in bytes and in-flight capacity:
        ``max_inflight`` counts outstanding requests for shared storage and
        queued response frames for a socket. ``transport`` is ``"iceoryx2"``,
        the default, for a rank on the head's host, which serves shared
        storage under ``service_name``, or ``"tcp"`` for a rank elsewhere,
        where ``service_name`` is the interface to bind on a system-chosen
        port. A socket bind does not wait for the engine to connect.

        Raises:
            RuntimeError: The bind fails or ``transport`` names neither
                mechanism.
        """
        ...
    def endpoint(self, service: str) -> str:
        """Return the endpoint this rank reports to the head.

        A shared-storage endpoint is ``service`` itself; a socket endpoint is
        the address its bind produced, whose host may be a wildcard that
        ``uniserve_worker.bootstrap.launch.register_endpoint`` replaces.
        """
        ...
    def __enter__(self) -> Self:
        """Return this open endpoint and close it when the scope exits.

        Raises ``RuntimeError`` when the endpoint is already closed.
        """
        ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the endpoint.

        An exception raised inside the scope is preserved; a close failure is
        attached to it as a note instead of replacing it.
        """
        ...
    @property
    def closed(self) -> bool:
        """Report whether the endpoint owner has released this service."""
        ...
    def close(self) -> None:
        """Idempotently release the service.

        The caller must have stopped every endpoint call first: close does not
        wait for one, and raises ``RuntimeError`` while another thread holds
        the endpoint.
        """
        ...
    def recv(self) -> Any:
        """Block until the next validated request is available.

        There is no deadline, and ``wake`` does not end the wait. A submit
        request arrives as ``{kind, message_id, batch}`` with a constructed
        ``uniserve_worker.protocol.batch.Batch``; every other request kind
        arrives in its schema-derived Python representation.
        """
        ...
    def try_recv(self) -> Any | None:
        """Return the next request immediately.

        Requests take the same form as from ``recv``. Return ``None`` when the
        queue is empty.
        """
        ...
    def wait_incoming(self, timeout_us: int) -> None:
        """Wait up to ``timeout_us`` microseconds for a request or a wake.

        Returns the same way on a request, a wake, and the timeout, and may
        consume a pending wake; the caller re-checks every progress source
        afterwards.
        """
        ...
    def wake(self) -> None:
        """Fire the completion wake that ends a ``wait_incoming``.

        A wake fired while no wait is pending ends the next one unless a
        receive consumes it first. Wakes fired before one is consumed
        coalesce into one.
        """
        ...
    def wake_on_stream(self, stream: int) -> None:
        """Schedule a server wake.

        The wake fires after the CUDA stream reaches its current point.
        ``stream`` is the native stream handle, ``torch.cuda.Stream``'s
        ``cuda_stream``.
        """
        ...
    def respond(self, response: Any) -> None:
        """Publish one response to the request identified by its envelope.

        A shared-storage endpoint refuses a message id that matches no
        received, unanswered request; a socket endpoint sends whatever id it
        is given.

        Raises:
            ValueError: ``response`` does not decode as a worker response.
            RuntimeError: The endpoint is closed or in use, or encoding or
                publication fails.
        """
        ...

@final
class StreamSignal:
    """Bridges CUDA stream completion into an asyncio-readable signal.

    A signal is one-shot: it can be scheduled successfully once. The
    descriptor stays open until both this object and a scheduled callback
    that has not yet run release it.
    """

    def __new__(cls) -> Self:
        """Create an owned, non-blocking completion descriptor.

        The descriptor receives CUDA stream notifications.
        """
        ...
    def fileno(self) -> int:
        """Return the readable descriptor signaled by completed stream work.

        The descriptor is borrowed; the signal owns and closes it.
        """
        ...
    def schedule(self, stream: int) -> None:
        """Signal the descriptor.

        The signal fires after the CUDA stream reaches its current point.
        ``stream`` is the native stream handle. Raises ``RuntimeError`` when
        the signal was already scheduled, the CUDA runtime cannot be loaded,
        or CUDA rejects the callback; only a rejected callback leaves the
        signal schedulable again.
        """
        ...
    def consume(self) -> None:
        """Read the fired signal from the descriptor.

        Call it once the descriptor is readable: the read does not block, and
        ``RuntimeError`` is raised when no signal is pending.
        """
        ...

def service_name(id: str) -> str:
    """Return the shared-storage service name for one endpoint identifier.

    A rank names its own channel endpoint and reports it to the head, so both
    sides must spell the name the same way; this function is that spelling.
    """
    ...

def atomic_store_u32(buffer: memoryview, offset: int, value: int) -> None:
    """Store a 32-bit word with release ordering.

    Every write the calling thread made before the store is visible to a
    process that loads the word with ``atomic_load_u32`` and observes
    ``value``. ``buffer`` must be writable and contiguous, and the word at
    ``offset`` must be four-byte aligned and lie inside it; otherwise
    ``RuntimeError`` is raised.
    """
    ...

def atomic_load_u32(buffer: memoryview, offset: int) -> int:
    """Load a 32-bit word with acquire ordering.

    ``buffer`` carries the same requirements as for ``atomic_store_u32``,
    including writability.
    """
    ...
