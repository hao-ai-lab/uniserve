"""Type stub for the common native worker and IPC extension.

``Server`` is the rank's end of its channel to the engine: it receives the
engine's requests and publishes the worker's responses over iceoryx2 shared
storage or a TCP socket. ``StreamSignal`` turns CUDA stream completion into a
readable descriptor for selector loops. ``atomic_store_u32`` and
``atomic_load_u32`` order the header words of shared-storage segments, which
``uniserve_worker.transport.segment`` reads and writes across processes.
``Request`` and ``RequestPool`` own the lifecycle shared by serving and direct
numerical execution. ``BufferPool`` binds scheduler-assigned storage and issues
its numerical ``BufferBinding`` views.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Generic, ParamSpec, Self, TypeVar, final

import torch

import uniserve_worker.storage.block_tables as block_tables
import uniserve_worker.storage.kv_cache as kv_cache
from uniserve.runtime.execution import ExecutionContext
from uniserve.sampling import SamplingParams
from uniserve.tensors import BufferConfig
from uniserve_worker.execution.diffusion_state import DiffusionState
from uniserve_worker.execution.output import (
    LatentResult,
    TokenUpdate,
)
from uniserve_worker.execution.request import RequestResult
from uniserve_worker.protocol.batch import (
    Batch,
    BufferAllocation,
    LatentParams,
    NewRequest,
    TensorPublication,
)
from uniserve_worker.protocol.call import (
    Call,
    CallKind,
    CallStatus,
    ErrorCode,
    ImageParams,
)
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    CallIdentity,
    RequestKey,
)
from uniserve_worker.protocol.output import (
    BatchOutput,
    ForwardStats,
    MediaOutput,
    RequestOutput,
)
from uniserve_worker.protocol.tensor import TensorRef
from uniserve_worker.protocol.transfer import (
    KvTransfer,
    Locator,
    TensorTransfer,
    WorkerEndpoint,
)
from uniserve_worker.storage.latent_pool import LatentStaging
from uniserve_worker.storage.request_slots import RequestSlots
from uniserve_worker.storage.tensor_store import FeatureMetadata, ImageMetadata
from uniserve_worker.transport.exports import ExportLocations
from uniserve_worker.transport.interface import Transport
from uniserve_worker.worker import Worker

__all__ = [
    "BatchInputs",
    "BatchState",
    "BlockTables",
    "Buffer",
    "BufferBinding",
    "BufferPool",
    "BufferRegistry",
    "Completion",
    "CUDAEvent",
    "CUDAStream",
    "EventPool",
    "EventPoolError",
    "Executor",
    "ExpertExchange",
    "GraphStorage",
    "GroupShape",
    "GroupTable",
    "HostLane",
    "HostTask",
    "KVCacheManager",
    "KVImport",
    "KVImporter",
    "LatentExport",
    "LatentImport",
    "LatentPool",
    "LatentUpdate",
    "Microbatches",
    "OutputBuffer",
    "OutputPool",
    "PendingOutput",
    "Request",
    "RequestProgress",
    "RequestPool",
    "ReadReservation",
    "Server",
    "StreamSignal",
    "Submission",
    "TensorImport",
    "TensorRead",
    "TensorStore",
    "TransferCapacity",
    "TransferPool",
    "TransferTicket",
    "WeightPrefetch",
    "atomic_load_u32",
    "atomic_store_u32",
    "service_name",
    "release_exports",
    "graph_storage_budget_bytes",
    "yield_microbatch",
]

Source = TypeVar("Source")
Args = ParamSpec("Args")

@final
class ExpertExchange:
    """Rank readiness, expert-step selection and layer participation."""

    def __init__(
        self,
        ranks: Sequence[int],
        rank: int,
        memberships: Sequence[Sequence[int]],
        *,
        attention_ranks: int,
        max_tokens: int,
        fused: bool,
        local: object,
        records: object,
        gather: Callable[[], None],
        prepare: Callable[[], None] | None,
        broadcast: Callable[[], None] | None,
    ) -> None: ...
    def agree(
        self, tokens: int, *, kind: int = 0, leaving: bool = False
    ) -> int: ...
    def warmup(self, capacity: int) -> int: ...
    def begin(self, capacity: int) -> None: ...
    def end(self) -> None: ...
    def enter(self, module: int) -> None: ...
    def pending_layers(self, modules: Sequence[int]) -> list[int]: ...
    def reset_layers(self) -> None: ...
    def record_layers(self, modules: Iterable[int]) -> None: ...
    @property
    def invoked(self) -> frozenset[int]: ...
    @property
    def capacities(self) -> tuple[int, ...]: ...
    @property
    def capacity(self) -> int: ...
    @property
    def kind(self) -> int: ...
    @property
    def active(self) -> bool: ...
    @property
    def released(self) -> bool: ...
    def close(self) -> None: ...

@final
class WeightPrefetch:
    """Peer-weight copy plans, CUDA dependencies and backing ownership."""

    def __init__(
        self,
        device: int,
        layers: Sequence[int],
        copies: Sequence[Sequence[tuple[int, torch.Tensor, torch.Tensor]]],
        backing: Sequence[torch.Tensor],
    ) -> None: ...
    def begin(self) -> None: ...
    def end(self, stream: int) -> None: ...
    def contains(self, module: torch.nn.Module) -> bool: ...
    def before(self, module: torch.nn.Module, stream: int) -> None: ...
    def after(self, module: torch.nn.Module, stream: int) -> None: ...
    def copy(self, index: int) -> None: ...
    def set_capture(
        self, submit: Callable[[Callable[[], None]], None] | None
    ) -> Callable[[Callable[[], None]], None] | None: ...
    def close(self) -> None: ...

def yield_microbatch() -> None: ...

@final
class Microbatches:
    """Cooperative numerical calls on persistent native host threads."""

    contexts: tuple[ExecutionContext, ...]
    device: torch.device

    def __init__(self, contexts: Sequence[ExecutionContext]) -> None: ...
    def __call__(
        self, calls: Sequence[Callable[[], Source]]
    ) -> list[Source]: ...
    def close(self) -> None: ...

def graph_storage_budget_bytes(total_device_bytes: int) -> int: ...

@final
class GraphStorage:
    """Shared graph pools and startup residency limits, owned in Rust."""

    def __init__(
        self, *, budgets: dict[torch.device | str, int] | None = None
    ) -> None: ...
    def reserve(
        self,
        owner: object,
        devices: Iterable[torch.device | str],
        *,
        share: object | None = None,
    ) -> dict[torch.device, torch.cuda.MemPool]: ...
    def allocate(self, owner: object) -> AbstractContextManager[None]: ...
    def check(self) -> None: ...
    def set_budget(self, device: torch.device | str, amount: int) -> None: ...
    def seal(self) -> None: ...
    def resident_bytes(self) -> dict[torch.device, int]: ...
    def pool_bytes(self) -> dict[torch.device, int]: ...
    def owner_bytes(self) -> dict[tuple[Any, torch.device], int]: ...
    def release(self, owner: object) -> None: ...
    def close(self) -> None: ...

def release_exports(
    exports: dict[BufferId, ExportLocations], buffers: Iterable[BufferId]
) -> None: ...

@final
class BatchState:
    """Batch-owned input leases, output rows and borrowed numerical views.

    The executor creates and retires each state. Numerical callbacks borrow
    its resources without advancing execution or request progress.
    """

    batch: Batch
    inputs: BatchInputs
    predicate_entries: list[tuple[CallIdentity, tuple[int, int], int]]
    predicate_transfers: tuple[tuple[CallIdentity, BufferId, int], ...]
    input_products: tuple[TensorPublication, ...]
    kv_inputs: tuple[KvTransfer, ...]
    stream: torch.cuda.Stream | None
    started_ns: int
    forward_stats: list[ForwardStats]
    component_us: dict[str, int]
    forward_indices: dict[CallIdentity, tuple[int, ...]]
    products: tuple[TensorPublication, ...]

    @property
    def batch_id(self) -> int: ...
    @property
    def route(self) -> str | None: ...
    @property
    def output_buffer(self) -> OutputBuffer: ...
    def scope(self) -> AbstractContextManager[None]: ...
    def bind_outputs(
        self,
        requests: RequestPool,
        buffer: OutputBuffer,
        started_ns: int,
        predicated: set[int],
    ) -> None:
        """Bind all output rows atomically in scheduler call order."""
    def pending_outputs(self) -> tuple[PendingOutput, ...]: ...
    def pending_output(self, request_id: int) -> PendingOutput: ...
    def predicate_values(self) -> dict[CallIdentity, bool]:
        """Read completed U8 predicates once, retaining device predicates."""

@final
class BatchInputs:
    """Batch input leases, preparation progress and completion observers."""

    started: int
    submitted: bool
    predicate: OutputBuffer | None

    def __new__(cls) -> Self: ...
    @property
    def closed(self) -> bool: ...
    def set_dependencies(self, dependencies: Sequence[Completion]) -> None: ...
    def storage_ready(self) -> bool: ...
    def require_storage(self) -> None: ...
    def ready(self) -> bool: ...
    def input_ready(self, buffer: BufferId) -> bool: ...
    def is_borrowed(self, buffer: BufferId) -> bool: ...
    def add(
        self,
        buffer: BufferId,
        value: TensorRead | LatentImport | KVImport | None = None,
    ) -> None: ...
    def remove(self, buffer: BufferId) -> None: ...
    def tensor(self, buffer: BufferId) -> TensorRead | None: ...
    def latent(self, buffer: BufferId) -> LatentImport | None: ...
    def cache(self, buffer: BufferId) -> KVImport | None: ...
    def cache_imports(self) -> list[KVImport]: ...
    def add_image(self, call: CallId, task: HostTask) -> None: ...
    def image(self, call: CallId) -> HostTask | None: ...
    def on_ready(self, callback: Callable[[], None]) -> None:
        """Notify once this snapshot of dependencies is consumable."""
    def close(
        self,
        tensor_store: TensorStore,
        latent_pool: LatentPool | None,
        kv_importer: KVImporter | None,
    ) -> None:
        """Release consumers once, attempting every resource after failures."""

@final
class PendingOutput:
    """Own result decoding and output retirement for one numerical call."""

    def __new__(
        cls, call: Call, request: Request, buffer: OutputBuffer, row: int
    ) -> Self: ...

    call: Call
    request: Request
    token_update: TokenUpdate
    latent: LatentResult
    tensor_exports: dict[BufferId, ExportLocations]
    cache_exports: dict[BufferId, ExportLocations]
    exported_locators: list[Locator]
    cache_publication: tuple[BufferId, KvTransfer] | None
    cache_installation: tuple[BufferId, BufferId, KvTransfer] | None
    device_reads: list[TensorRead]
    feature_reads: list[TensorRead]
    writes: list[Buffer]
    predicate: tuple[torch.Tensor, bool] | None
    token_write: Buffer | None
    transition_write: Buffer | None
    completion_write: Buffer | None
    producer_write: Buffer | None
    products: tuple[TensorPublication, ...]

    def set_sampling(
        self,
        sampling: tuple[int, int, int],
        logprobs: tuple[int, int, int] | None = None,
    ) -> None: ...
    def add_prompt_logprobs(
        self, spans: Sequence[tuple[int, int, int]]
    ) -> None: ...
    def set_candidates(self, span: tuple[int, int]) -> None: ...
    def set_canvas(self, span: tuple[int, int]) -> None: ...
    def advance_tokens(
        self,
        tokens: int,
        *,
        cache_length: int | None = None,
        position: int | None = None,
        sampled: bool = False,
    ) -> None: ...
    def set_cache_length(self, length: int) -> None: ...
    def set_flow_step(self, step: int) -> None: ...
    def set_prompt_logits(self, logits: torch.Tensor) -> None: ...
    def cache_coordinates(
        self, tables: block_tables.BlockTables | None
    ) -> tuple[int, int, int]: ...
    def set_speculation(
        self,
        draft_tokens: Sequence[int],
        terminal_prefix: int | None,
        visible: int,
        initialized: int,
    ) -> None: ...
    @property
    def host_tasks(self) -> tuple[HostTask[Any], ...]: ...
    def set_host_tasks(
        self,
        tasks: Sequence[HostTask[Any]],
        finish: Callable[[tuple[object, ...]], None] | None = None,
    ) -> None: ...
    def set_media(self, media: MediaOutput) -> None: ...
    def set_kv_output(self, output: KvTransfer) -> None: ...
    @property
    def request_key(self) -> RequestKey: ...
    @property
    def call_id(self) -> CallId: ...
    @property
    def kind(self) -> CallKind: ...
    @property
    def value(self) -> RequestOutput | None: ...
    @property
    def progress(self) -> RequestProgress: ...
    @property
    def status(self) -> CallStatus: ...
    @property
    def error_code(self) -> ErrorCode | None: ...
    def ready(self) -> bool: ...
    def materialize(self) -> RequestOutput:
        """Resolve completed work once; retain accepted progress on failure."""
    def abandon(self) -> None:
        """Discard delivery while in-flight readers keep their storage."""

@final
class OutputBuffer:
    """One batch's captures and readback fence; only its storage is recycled."""

    @property
    def sealed(self) -> bool: ...

    event_pool: EventPool
    devices: tuple[torch.device, ...]

    def register_device(self, device: torch.device | str) -> None: ...
    def begin_device(self, device: torch.device | str) -> None: ...
    def capture(self, tokens: torch.Tensor) -> tuple[int, int]: ...
    def capture_bytes(self, value: torch.Tensor) -> torch.Tensor:
        """Borrow copied bytes; wait for completion and retain CPU readers."""
    def seal(self) -> None: ...
    def ready(self) -> bool: ...
    def completion(self) -> Completion: ...
    def read_tokens(self, offset: int, count: int) -> tuple[int, ...]: ...
    def register_logprobs(
        self,
        span: tuple[int, int],
        rows: tuple[int, ...],
        counts: tuple[int, ...],
        requested_ids: tuple[tuple[int, ...], ...],
        max_count: int,
        max_requested: int,
    ) -> None:
        """Describe one captured score column before sealing the buffer."""
    def observe(self, row: int) -> tuple[int, int]: ...
    def timing(self) -> tuple[int, int, int, int]: ...
    def discard(self, row: int) -> None: ...
    def abandon(self) -> None: ...
    def retain_cpu_reader(self) -> Callable[[], None]:
        """Return a release callable that the reader must call exactly once."""

@final
class OutputPool:
    """Bounded readback storage for sampling, predicates and media."""

    def __new__(
        cls, *, capacity: int, max_words: int, event_pool: EventPool
    ) -> Self: ...
    @property
    def capacity(self) -> int: ...
    @property
    def max_words(self) -> int: ...
    @property
    def event_pool(self) -> EventPool: ...
    def acquire(
        self,
        rows: int,
        *,
        token_capacity: int,
        devices: Sequence[torch.device | str] = (),
    ) -> OutputBuffer: ...
    def close(self) -> None: ...

@final
class HostBuffers:
    """Round-robin host inputs retained until asynchronous copies finish.

    Pair each acquire with record_copy after enqueueing the copy on the
    current stream. Close drains copies and releases the host allocations.
    """

    def __new__(
        cls,
        shape: tuple[int, ...] | int,
        *,
        dtype: torch.dtype,
        depth: int,
        device: torch.device | str,
    ) -> Self: ...
    @property
    def device(self) -> torch.device: ...
    def acquire(self) -> tuple[int, torch.Tensor]:
        """Wait for the next slot's previous copy and return its host tensor."""
    def record_copy(self, slot: int) -> None:
        """Fence this slot's copy on the current CUDA stream."""
    def close(self) -> None:
        """Wait for copies and release storage; subsequent acquire fails."""

@final
class CUDAStream:
    """Native stream ownership, SM partitions and reusable execution fences."""

    def __new__(
        cls, device: int, handle: int, event_slots: int = 2
    ) -> Self: ...
    @staticmethod
    def sibling(
        device: int, origin: int, event_slots: int = 2
    ) -> CUDAStream: ...
    @staticmethod
    def partition(
        device: int, counts: Sequence[int], slots: Sequence[int]
    ) -> list[CUDAStream]: ...
    @property
    def device(self) -> int: ...
    @property
    def handle(self) -> int: ...
    @property
    def sm_count(self) -> int: ...
    @property
    def full_device(self) -> bool: ...
    @property
    def closed(self) -> bool: ...
    def fork(self) -> CUDAStream: ...
    def wait(self, producer: int) -> None: ...
    def record(self, consumer: int) -> CUDAEvent | None: ...
    def synchronize(self) -> None: ...
    def close(self, *, aborted: bool = False) -> None: ...

class EventPoolError(RuntimeError):
    """Invalid event lease ownership or stream ordering."""

@final
class CUDAEvent:
    """A native completion event from execution, a pool or CUDA IPC."""

    def query(self) -> bool: ...
    def synchronize(self) -> None: ...
    def wait(self, stream: torch.cuda.Stream | None = None) -> None: ...
    def elapsed_time(self, end: CUDAEvent) -> float: ...
    def ipc_handle(self) -> bytes: ...
    @staticmethod
    def from_ipc_handle(
        device: torch.device | str, handle: bytes
    ) -> CUDAEvent: ...

@final
class EventPool:
    """Share event references until producers and deferred owners retire."""

    def __new__(cls) -> Self: ...
    def set_completion_wake(
        self, wake_on_stream: Callable[[int], None] | None
    ) -> None: ...
    def schedule_completion_wake(
        self, device: torch.device | str, event: CUDAEvent
    ) -> None: ...
    def acquire(
        self,
        device: torch.device | str,
        *,
        timing: bool = False,
        interprocess: bool = False,
    ) -> CUDAEvent: ...
    def declare_stream(
        self, event: CUDAEvent, device: torch.device | str
    ) -> int: ...
    def record(self, event: CUDAEvent, device: torch.device | str) -> int: ...
    def retain(
        self, event: CUDAEvent, device: torch.device | str, count: int = 1
    ) -> None: ...
    def release(self, event: CUDAEvent, count: int = 1) -> None: ...
    def defer_release(
        self,
        events: Sequence[CUDAEvent],
        owner: object,
        *,
        completed: Callable[[], None] | None = None,
    ) -> None: ...
    def reap(self) -> None: ...
    def close(self) -> None: ...

@final
class HostLane:
    """Bounded host threads, with capacity reserved before task inputs exist."""

    def __new__(
        cls, *, max_inflight: int, workers: int, name: str = "worker-host-lane"
    ) -> Self: ...
    @property
    def max_inflight(self) -> int: ...
    @property
    def reserved(self) -> int: ...
    def set_completion_wake(self, wake: Callable[[], None] | None) -> None: ...
    def reserve(self) -> HostTask[Any]: ...
    def abort(self) -> None:
        """Stop admission without joining running work; cancel queued work."""
    def close(self) -> None:
        """Cancel unsubmitted tasks, then drain all submitted work."""

@final
class HostTask(Generic[Source]):
    """An admitted action, its result, and its physical input lease.

    Result observers may immediately reserve returned lane capacity. A failed
    or cancelled input producer keeps its lease until physical retirement is
    known. Cancelling queued work removes it before returning capacity.
    """

    def configure(
        self,
        action: Callable[[], Source],
        *,
        dependencies: Sequence[HostTask[Any] | Completion] = (),
        input_ready: Callable[[], bool] | None = None,
        input_completion: Callable[[], Completion] | None = None,
        release: Callable[[], None] | None = None,
        profile_name: str = "uniserve.host",
    ) -> Self:
        """Attach numerical work; take the input lease only on success."""
    def submit(
        self,
        function: Callable[Args, Source],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Self:
        """Configure and submit an immediate action without an input lease."""
    def submit_if_ready(self) -> None:
        """Submit once the input is CPU-readable; repeat calls are harmless."""
    def done(self) -> bool: ...
    def cancelled(self) -> bool: ...
    def result(self, timeout: float | None = None) -> Source:
        """Wait without holding the GIL; preserve the original action error."""
    def exception(
        self, timeout: float | None = None
    ) -> BaseException | None: ...
    def add_done_callback(self, callback: Callable[[Self], object]) -> None: ...
    def abandon(self) -> None:
        """Cancel only unsubmitted work; submitted readers keep their inputs."""
    def cancel(self) -> bool:
        """Withdraw unsubmitted or queued work; running actions finish."""

@final
class KVImport:
    """KV destination retained until release and physical drain."""

    @property
    def request_pool_idx(self) -> int: ...
    @property
    def tables(self) -> tuple[GroupTable, ...]: ...
    @property
    def initialized_units(self) -> tuple[int, ...]: ...
    @property
    def publication(self) -> KvTransfer: ...
    @property
    def completion(self) -> HostTask[None] | Completion: ...
    @property
    def retirement(self) -> Completion: ...
    @property
    def cancelled(self) -> bool: ...
    @property
    def released(self) -> bool: ...

@final
class KVImporter:
    """Bounded KV copies sharing conversion storage and native stream waits."""

    def __new__(
        cls, pool: kv_cache.KVCacheManager, *, capacity: int
    ) -> Self: ...
    def set_completion_wake(self, wake: Callable[[], None] | None) -> None: ...
    def reserve(
        self,
        publication: KvTransfer,
        *,
        request_pool_idx: int,
        tables: tuple[GroupTable, ...],
        initialized_units: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> KVImport: ...
    def owns(self, write: KVImport) -> bool: ...
    def adopt(self, write: KVImport) -> None: ...
    def abandon(self, write: KVImport) -> None: ...
    def release(self, buffers: Sequence[BufferId]) -> None: ...
    def cancel_requests(
        self,
        requests: frozenset[RequestKey],
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None: ...
    def stop(self) -> None: ...
    def require_retired(self) -> None: ...
    def _require_active(self, write: KVImport) -> None: ...
    def _retain(self, write: KVImport, ticket: TransferTicket) -> None: ...
    def _consume(
        self, write: KVImport, tickets: Sequence[TransferTicket]
    ) -> None: ...
    def _drain(self, write: KVImport) -> None: ...

@final
class KVCacheManager:
    """Resident KV transfers, incremental bases, and physical accesses."""

    def __new__(cls) -> Self: ...
    def resident(self, buffer: BufferId) -> KvTransfer | None: ...
    def destination_base(
        self, request: RequestKey, destination: str
    ) -> tuple[BufferId, int] | None: ...
    def validate_install(self, transfer: KvTransfer) -> None: ...
    def validate_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None: ...
    def apply_publications(
        self,
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> None: ...
    def release_calls(
        self, calls: Sequence[tuple[RequestKey, CallId]]
    ) -> tuple[BufferId, ...]: ...
    def drop_request(self, request_id: int) -> None: ...
    def clear_resident(self) -> None: ...
    @property
    def has_pending_accesses(self) -> bool: ...
    @property
    def has_transfers(self) -> bool: ...
    def retain_execution(
        self,
        request: RequestKey,
        spans: Sequence[tuple[int, int, int]],
        completion: Completion,
    ) -> None: ...
    def reserve_export(
        self, buffer: BufferId, spans: Sequence[tuple[int, int, int]]
    ) -> None: ...
    def retain_export(
        self, buffer: BufferId, retirement: Completion
    ) -> None: ...
    def release_exports(self, buffers: Iterable[BufferId]) -> None: ...
    def exported_buffers(self) -> tuple[BufferId, ...]: ...
    def reserve_import(
        self,
        buffer: BufferId,
        spans: Sequence[tuple[int, int, int]],
        retirement: Completion,
    ) -> None: ...
    def discard_import(self, buffer: BufferId) -> None: ...
    def write_dependencies(
        self, spans: Sequence[tuple[int, int, int]]
    ) -> tuple[Completion, ...]: ...
    def require_writable(
        self, spans: Sequence[tuple[int, int, int]]
    ) -> None: ...
    def require_reusable(
        self, spans: Sequence[tuple[int, int, int]]
    ) -> None: ...
    def retirement_ready(
        self,
        buffers: Iterable[BufferId],
        requests: Iterable[RequestKey],
        retained: Iterable[BufferId],
    ) -> bool: ...
    def require_retired(self) -> None: ...

@final
class Completion:
    """Native completion shared by a producer and its storage consumers.

    Cancellation and failure wake observers but do not authorize storage reuse.
    Callbacks receive this completion, even when registered after it finishes.
    Waits release the GIL; errors retain the original exception.
    """

    def __new__(cls) -> Self: ...
    def done(self) -> bool: ...
    def succeeded(self) -> bool: ...
    def cancelled(self) -> bool: ...
    def cancel(self) -> bool: ...
    def result(self, timeout: float | None = None) -> None: ...
    def exception(
        self, timeout: float | None = None
    ) -> BaseException | None: ...
    def set_result(self, result: None) -> None: ...
    def set_exception(self, error: BaseException) -> None: ...
    def add_done_callback(
        self, callback: Callable[[Completion], object]
    ) -> None: ...

@final
class BufferRegistry(Generic[Source]):
    """Retain registered storage until its producer and readers finish.

    Reclaim hands the source back and completes its retirement signal. Drain
    waits for an in-flight reclamation at shutdown. Settled inspects backend
    acknowledgments; it must not mutate the registry or invoke observers.
    """

    def __new__(
        cls,
        *,
        capacity: int,
        reclaim: Callable[[Source, Completion], None],
        drain: Callable[[Source], None],
        settled: Callable[[Source], bool],
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    def register(
        self, locator: Locator, source: Source, *, pending: bool = False
    ) -> None: ...
    def source(self, locator: Locator) -> Source: ...
    def acquire(self, locator: Locator) -> Source: ...
    def release_reader(self, locator: Locator) -> None: ...
    def complete(
        self,
        locator: Locator,
        *,
        error: BaseException | None = None,
        producer_completed: bool = True,
    ) -> None: ...
    def release(self, locator: Locator) -> Completion | None: ...
    def retirement(self, locator: Locator) -> Completion: ...
    def awaiting_acknowledgment(self) -> bool: ...
    def reap(self) -> None: ...
    def close(self) -> None: ...

@final
class TransferCapacity:
    """One rank's nonblocking byte and read credits shared by its backends."""

    def __new__(cls, byte_capacity: int, ticket_capacity: int) -> Self: ...
    @property
    def capacity(self) -> int: ...
    @property
    def ticket_capacity(self) -> int: ...
    @property
    def used(self) -> int: ...
    def take_reads(
        self,
        count: int = 1,
        *,
        message: str = "asynchronous transfer ticket capacity is exhausted",
    ) -> None: ...
    def return_reads(self, count: int = 1) -> None: ...
    def notify_reads_returned(
        self, callback: Callable[[], None], *, after: int
    ) -> None: ...
    def acquire(self, amount: int) -> None: ...
    def release(self, amount: int) -> None: ...

@final
class ReadReservation:
    """A fetch's read credits, handed off individually to submitted reads."""

    def __new__(cls, capacity: TransferCapacity, count: int) -> Self: ...
    def use(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> None: ...

@final
class TransferTicket:
    """A read's consumable views, failure, and physical retirement."""

    def __new__(
        cls, event_pool: EventPool, *, release: Callable[[], None] | None = None
    ) -> Self: ...
    def ready(self) -> bool: ...
    def retired(self) -> bool: ...
    def retirement_ready(self) -> bool: ...
    def result(
        self, stream: torch.cuda.Stream | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
    def add_done_callback(self, callback: Callable[[], None]) -> None: ...
    def add_retirement_callback(self, callback: Callable[[], None]) -> None: ...
    def _require_active(self) -> None: ...
    def _complete(
        self,
        value: torch.Tensor | tuple[torch.Tensor, ...],
        event: CUDAEvent | None = None,
    ) -> None: ...
    def _fail(self, error: BaseException) -> bool: ...
    def _drain_consumers(self) -> None: ...

@final
class TransferPool:
    """Own bounded read submission, copy streams, and credit retirement."""

    def __new__(
        cls,
        *,
        workers: int,
        capacity: TransferCapacity,
        name: str,
        event_pool: EventPool,
    ) -> Self: ...
    def set_completion_wake(self, wake: Callable[[], None] | None) -> None: ...
    def submit(
        self,
        call: Callable[..., Any],
        *args: Any,
        nbytes: int,
        destination: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        reservation: ReadReservation | None = None,
    ) -> TransferTicket: ...
    def copy(
        self,
        ticket: TransferTicket,
        source: torch.Tensor | tuple[torch.Tensor, ...],
        destination: torch.Tensor | tuple[torch.Tensor, ...],
        producer: CUDAEvent | None = None,
        acknowledgment: torch.Tensor | None = None,
    ) -> None: ...
    def close(self) -> None: ...

@final
class LatentUpdate:
    """A batch's prepared trajectory commit or release."""

    request_pool_idx: int
    params: LatentParams | None
    expected_generation: int
    expected_step: int
    generation: int
    step: int
    release: bool

    def __new__(
        cls,
        request_pool_idx: int,
        params: LatentParams | None = None,
        expected_generation: int = 0,
        expected_step: int = 0,
        generation: int = 0,
        step: int = 0,
        release: bool = False,
    ) -> Self: ...

@final
class LatentImport:
    """Preassigned pages kept until their copies are adopted or abandoned."""

    @property
    def product(self) -> TensorRef: ...
    @property
    def request_pool_idx(self) -> int: ...
    @property
    def page_table(self) -> tuple[int, ...]: ...
    @property
    def spans(self) -> tuple[torch.Tensor, ...]: ...
    @property
    def transfers(self) -> tuple[TransferTicket, ...]: ...
    @property
    def adopted(self) -> bool: ...
    @property
    def released(self) -> bool: ...

@final
class LatentExport:
    """An immutable page-bank version held by transport readers."""

    @property
    def buffer(self) -> BufferId: ...
    @property
    def request_pool_idx(self) -> int: ...
    @property
    def bank(self) -> int: ...
    @property
    def page_table(self) -> tuple[int, ...]: ...
    @property
    def spans(self) -> tuple[torch.Tensor, ...]: ...

@final
class LatentPool:
    """Own latent backing, page assignments, and committed trajectories."""

    def __new__(
        cls,
        *,
        request_pool_size: int,
        num_pages: int,
        page_units: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device | str,
        staging: bool = True,
    ) -> Self: ...
    @property
    def request_pool_size(self) -> int: ...
    @property
    def num_pages(self) -> int: ...
    @property
    def page_units(self) -> int: ...
    @property
    def latent_width(self) -> int: ...
    @property
    def capacity_units(self) -> int: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def storage(self) -> torch.Tensor: ...
    @property
    def step_buffer(self) -> torch.Tensor: ...
    @property
    def page_table_buffer(self) -> torch.Tensor: ...
    @property
    def timesteps(self) -> torch.Tensor: ...
    @property
    def page_rows(self) -> torch.Tensor: ...
    @property
    def persistent_bytes(self) -> int: ...
    @property
    def exports(self) -> dict[BufferId, ExportLocations]: ...
    def startup_values(
        self, rows: int, units: int
    ) -> AbstractContextManager[tuple[torch.Tensor, ...]]: ...
    def _startup_staging(
        self, rows: int, units: int
    ) -> tuple[torch.Tensor, ...]: ...
    def stage(
        self,
        page_tables: Sequence[Sequence[int]],
        latent_units: Sequence[int],
        *,
        occupied: Sequence[LatentStaging] = (),
    ) -> tuple[LatentStaging, ...]: ...
    def initialize(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        latent_units: int,
    ) -> None: ...
    def bank_view(
        self, bank: int, page_table: Sequence[int]
    ) -> torch.Tensor: ...
    def initial_bank(
        self,
        request_pool_idx: int,
        page_table: Sequence[int],
        *,
        latent_units: int,
    ) -> int: ...
    def step_banks(
        self,
        request_pool_idx: int,
        page_table: Sequence[int],
        *,
        step: int,
        generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> tuple[int, int]: ...
    def gather_current(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        step: int,
        generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> torch.Tensor: ...
    def write_inactive(
        self,
        request_pool_idx: int,
        staging: LatentStaging,
        *,
        expected_step: int,
        expected_generation: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> None: ...
    def reserve_export(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentExport: ...
    def reserve_current_export(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        generation: int,
        step: int,
        latent_units: int,
        height: int,
        width: int,
    ) -> LatentExport: ...
    def retain_export(
        self, source: LatentExport, retirement: Completion
    ) -> None: ...
    def release_buffers(self, buffers: Sequence[BufferId]) -> None: ...
    def write_dependencies(
        self, request_pool_idx: int, page_table: Sequence[int]
    ) -> tuple[Completion, ...]: ...
    def stage_timestep(
        self, request_pool_idx: int, value: float
    ) -> torch.Tensor: ...
    def validate_updates(self, updates: Sequence[LatentUpdate]) -> None: ...
    def apply_updates(self, updates: Sequence[LatentUpdate]) -> None: ...
    def reserve_import(
        self,
        product: TensorRef,
        *,
        request_pool_idx: int,
        page_table: Sequence[int],
        latent_units: int,
    ) -> LatentImport: ...
    def retain_transfer(
        self, write: LatentImport, ticket: TransferTicket
    ) -> None: ...
    def adopt_import(
        self,
        write: LatentImport,
        *,
        generation: int,
        step: int,
        height: int,
        width: int,
    ) -> None: ...
    def abandon_import(self, write: LatentImport) -> None: ...
    def retirement_ready(self, requests: Sequence[RequestKey]) -> bool: ...
    def cancel_imports(self, requests: Sequence[RequestKey]) -> None: ...
    def release_slots(self, request_pool_indices: Sequence[int]) -> None: ...
    def close(self) -> None: ...

@final
class Buffer:
    """An allocation-backed value owned and mutated by its tensor store."""

    @property
    def reference(self) -> TensorRef: ...
    @property
    def tensor(self) -> torch.Tensor: ...
    @property
    def logical_shape(self) -> tuple[int, ...]: ...
    @property
    def region(self) -> tuple[slice, ...] | None: ...
    @property
    def metadata(self) -> ImageMetadata | FeatureMetadata | None: ...
    @property
    def producer_recorded(self) -> bool: ...
    @property
    def feature(self) -> bool: ...

@final
class TensorRead:
    """A read lease retaining the tensor view acquired from its buffer."""

    @property
    def tensor(self) -> torch.Tensor: ...
    @property
    def region(self) -> tuple[slice, ...] | None: ...
    @property
    def metadata(self) -> ImageMetadata | FeatureMetadata | None: ...
    @property
    def imported(self) -> TensorImport | None: ...

@final
class TensorImport:
    """The transfers filling missing regions for one or more read leases."""

    @property
    def tickets(self) -> tuple[TransferTicket, ...]: ...

@final
class TensorStore:
    """Own buffer visibility, import coordination, and physical retirement."""

    def __new__(
        cls,
        *,
        capacity: int = 0,
        byte_capacity: int | None = None,
        max_feature_bytes: int = 1,
        devices: tuple[torch.device | str, ...] = (),
        request_capacity: int = 0,
        relay_depth: int = 0,
        buffer_pool: BufferPool,
        event_pool: EventPool | None = None,
    ) -> Self: ...
    @property
    def capacity(self) -> int: ...
    @property
    def byte_capacity(self) -> int: ...
    @property
    def max_feature_bytes(self) -> int: ...
    @property
    def devices(self) -> tuple[torch.device, ...]: ...
    @property
    def request_capacity(self) -> int: ...
    @property
    def relay_depth(self) -> int: ...
    @property
    def buffer_pool(self) -> BufferPool: ...
    @property
    def event_pool(self) -> EventPool: ...
    @property
    def exports(self) -> dict[BufferId, ExportLocations]: ...
    def bind_outputs(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[Buffer, ...]: ...
    def reserve_features(
        self,
        bindings: tuple[tuple[TensorRef, torch.device | str], ...],
        *,
        buffer_allocations: Mapping[BufferId, BufferAllocation],
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[Buffer, ...]: ...
    def bind_output_groups(
        self,
        groups: tuple[tuple[tuple[TensorRef, torch.device | str], ...], ...],
        *,
        request_slots: Mapping[RequestKey, int] | None = None,
        buffer_allocations: Mapping[BufferId, BufferAllocation] | None = None,
        regions: Mapping[TensorRef, tuple[slice, ...]] | None = None,
        shapes: Mapping[TensorRef, tuple[int, ...]] | None = None,
    ) -> tuple[tuple[Buffer, ...], ...]: ...
    def producer_write_views(
        self, writes: tuple[Buffer, ...]
    ) -> tuple[torch.Tensor, ...]: ...
    def publish_write(
        self,
        write: Buffer,
        value: torch.Tensor,
        *,
        producer_event: CUDAEvent | None = None,
        metadata: ImageMetadata | FeatureMetadata | None = None,
    ) -> torch.Tensor: ...
    def publish_writes(
        self,
        writes: tuple[Buffer, ...],
        values: torch.Tensor,
        *,
        producer_event: CUDAEvent | None = None,
    ) -> tuple[torch.Tensor, ...]: ...
    def publish_scalar_write(
        self,
        write: Buffer,
        value: bool | int,
        *,
        producer_event: CUDAEvent | None = None,
    ) -> torch.Tensor: ...
    def consume(
        self,
        reference: TensorRef,
        *,
        consumer_call_id: CallId,
        device: torch.device | str | None = None,
    ) -> TensorRead: ...
    def consume_batch(
        self,
        requests: tuple[
            tuple[TensorRef, CallId, torch.device | str | None], ...
        ],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[TensorRead, ...]: ...
    def complete_reads(
        self,
        reads: tuple[TensorRead, ...],
        *,
        device: torch.device | str | None = None,
        after_writes: tuple[Buffer, ...] = (),
    ) -> None: ...
    def import_tensor(
        self,
        reference: TensorRef,
        tensor: TensorTransfer,
        *,
        device: torch.device | str,
        bindings: Mapping[tuple[WorkerEndpoint, str], Transport],
        request_slots: Mapping[RequestKey, int],
        buffer_allocations: Mapping[BufferId, BufferAllocation],
        metadata: ImageMetadata | FeatureMetadata | None = None,
    ) -> TensorRead: ...
    def wait_import(self, read: TensorRead) -> None: ...
    def complete_import(self, read: TensorRead) -> None: ...
    def defer_write(self, write: Buffer) -> None: ...
    def validate_writes(self, writes: tuple[Buffer, ...]) -> None: ...
    def commit_writes(self, writes: tuple[Buffer, ...]) -> None: ...
    def retain_export(self, write: Buffer, retirement: Completion) -> None: ...
    def retain_transfer(
        self, write: Buffer, ticket: TransferTicket
    ) -> None: ...
    def abandon_writes(self, writes: tuple[Buffer, ...]) -> None: ...
    def release_calls(
        self, releases: Iterable[tuple[RequestKey, CallId]]
    ) -> None: ...
    def release_buffers(self, buffers: Iterable[BufferId]) -> None: ...
    def release_requests(
        self,
        requests: Iterable[RequestKey],
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None: ...
    def retirement_ready(
        self,
        *,
        buffers: frozenset[BufferId],
        requests: frozenset[RequestKey],
        retained: frozenset[BufferId] = frozenset(),
    ) -> bool: ...
    def resident_bytes(self, device: torch.device | str) -> int: ...
    def close(self) -> None: ...

@final
class BufferBinding:
    """A pool-issued tensor view of a reserved physical byte range."""

    @property
    def buffer(self) -> BufferId: ...
    @property
    def physical_offset(self) -> int: ...
    @property
    def physical_bytes(self) -> int: ...
    @property
    def device_name(self) -> str: ...
    @property
    def tensor(self) -> torch.Tensor: ...

@final
class BufferPool:
    """Back scheduler-assigned buffers with nonoverlapping device views."""

    def __new__(
        cls,
        *,
        byte_capacity: int,
        devices: tuple[torch.device | str, ...],
        compact: bool = False,
    ) -> Self: ...
    @property
    def byte_capacity(self) -> int: ...
    @property
    def compact(self) -> bool: ...
    @property
    def devices(self) -> tuple[torch.device, ...]: ...
    def bind(
        self,
        reference: TensorRef,
        allocation: BufferAllocation,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> BufferBinding: ...
    def release(self, binding: BufferBinding) -> None: ...
    def close(self) -> None: ...

@final
class GroupShape:
    def __new__(
        cls, page_tokens: int, units_per_page: int, window: int | None = None
    ) -> Self: ...
    @property
    def page_tokens(self) -> int: ...
    @property
    def units_per_page(self) -> int: ...
    @property
    def window(self) -> int | None: ...

@final
class GroupTable:
    def __new__(
        cls,
        shape: GroupShape,
        start_page: int,
        units: Sequence[int],
        allocated_tokens: int,
    ) -> Self: ...
    @property
    def shape(self) -> GroupShape: ...
    @property
    def start_page(self) -> int: ...
    @property
    def end_page(self) -> int: ...
    @property
    def units(self) -> tuple[int, ...]: ...
    @property
    def allocated_tokens(self) -> int: ...
    def row(self, index: int) -> tuple[int, ...]: ...
    def spans(
        self, start: int, length: int
    ) -> tuple[tuple[int, int, int], ...]: ...

@final
class BlockTables:
    """Own host KV page tables; copy callbacks borrow numerical updates."""

    def __new__(
        cls, groups: Sequence[GroupShape], request_pool_size: int, width: int
    ) -> Self: ...
    @property
    def first_table(self) -> tuple[int, ...]: ...
    def install(
        self,
        tables: Sequence[tuple[int, int, int, Sequence[int], int]],
        copy: Callable[..., None],
    ) -> None: ...
    def table(self, request_pool_idx: int, group_id: int) -> GroupTable: ...
    def allocated_length(self, request_pool_idx: int) -> int: ...
    def retain_prefix(self, request: RequestKey, slot: int) -> None: ...
    def release_prefixes(
        self,
        request: RequestKey,
        copy: Callable[[Sequence[int]], None],
        slots: Sequence[int] | None = None,
    ) -> None: ...
    def release(
        self, slots: Sequence[int], copy: Callable[[Sequence[int]], None]
    ) -> None: ...
    def clear(self) -> None: ...

@final
class Submission:
    """A native batch handle consumed by the executor that admitted it."""

    @property
    def batch_id(self) -> int: ...
    def notify_ready(self) -> None: ...

@final
class Executor:
    """Own admission, dependencies, collective order, and result delivery."""

    def __new__(cls, worker: Worker) -> Self: ...
    @property
    def has_work(self) -> bool: ...
    @property
    def started(self) -> bool: ...
    def submit(
        self, batch: Batch, *, propagate_errors: bool = False
    ) -> Submission: ...
    def advance(self) -> bool: ...
    def serve(self, endpoint: Server) -> None: ...
    def poll(self, submission: Submission) -> BatchOutput | None: ...
    def reset(self) -> None: ...
    def drop_request(self, request_id: int) -> None: ...
    def close(self) -> None: ...

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
class RequestProgress:
    """Immutable native progress.

    KV lengths count tokens; flow_step counts completed solver steps.
    """

    def __new__(
        cls,
        logical_position: int = 0,
        rng_counter: int = 0,
        flow_step: int = 0,
        kv_visible_len: int = 0,
        kv_computed_len: int = 0,
        prompt_logits_ready: bool = False,
    ) -> Self: ...
    @property
    def logical_position(self) -> int: ...
    @property
    def rng_counter(self) -> int: ...
    @property
    def flow_step(self) -> int: ...
    @property
    def kv_visible_len(self) -> int: ...
    @property
    def kv_computed_len(self) -> int: ...
    @property
    def prompt_logits_ready(self) -> bool: ...

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
    def add_pending(self, calls: Sequence[Call]) -> None: ...
    def apply_result(self, result: RequestResult) -> None: ...
    def cancel_calls(self, calls: Sequence[Call]) -> None: ...
    def start(self, admission: NewRequest) -> int | None: ...
    def finish(self, request_key: RequestKey) -> None: ...
    def retirement_ready(self, request_key: RequestKey) -> bool: ...
    def drop(self, request_id: int) -> None: ...
    def retire(self, request_id: int) -> None:
        """Retire a closed request after device and host staging drain."""
        ...

@final
class Client:
    """Send requests and receive independently ready results on a rank channel.

    Each outstanding request needs a distinct message_id. One thread owns
    channel operations; blocking calls release the GIL.
    """

    def __new__(
        cls,
        endpoint: str,
        max_payload: int = 1_048_576,
        max_inflight: int = 1,
        transport: str = ...,
        timeout: float = 5.0,
    ) -> Self: ...
    def send(self, request: Mapping[str, Any]) -> None: ...
    def recv(self, timeout: float = 0.0) -> dict[str, Any] | None: ...
    def close(self) -> None: ...

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

        Envelopes use native response fields: ``info``, ``result`` or
        ``error`` holds the corresponding payload beside ``message_id``.
        A shared-storage endpoint refuses a message id that matches no
        received, unanswered request; a socket endpoint sends whatever id it
        is given.

        Raises:
            TypeError, ValueError: ``response`` does not decode.
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
