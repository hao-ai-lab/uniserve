"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import gc
import logging
from bisect import insort
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from queue import SimpleQueue
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, cast

import torch

from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.nn.parallel import EntryConfig

from ..bootstrap.capacity import (
    check_startup_memory,
    decode_context_blocks,
    device_total_bytes,
    resolve_request_capacity,
)
from ..bootstrap.model_loader import load_worker_model
from ..bootstrap.worker_info import RequestKind, ResponseKind, WorkerInfo
from ..bootstrap.worker_info_builder import build_worker_layout
from ..config import (
    WorkerConfig,
    graph_memory_budget_bytes,
)
from ..execution.attention import supports_flow_attention
from ..execution.batch import (
    BufferId,
    Finish,
    Free,
    OpCode,
    RequestKey,
    Retire,
    Run,
    RunResult,
    WorkerEndpoint,
)
from ..execution.forward_batch import AttentionSelection
from ..execution.model_runner import ModelRunner
from ..execution.output import OutputPool
from ..execution.prepare import plan_run, prepare_batch
from ..execution.retirement import (
    drop_request as drop_execution_request,
)
from ..execution.retirement import release_buffers
from ..execution.rows import PreparedExecution
from ..execution.run import WorkerRun
from ..execution.step import execute_batch
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    unsupported_setup,
)
from ..foundation.math import ceil_div
from ..foundation.resources import close_resources
from ..models.runtime import ExecutionModel
from ..nn.mesh import Communicator
from ..profiling import WorkerProfiler, profile_range, record_failure, worker_range_name
from ..runtime.cache_pool import CachePool
from ..runtime.cache_publications import CachePublications
from ..runtime.cpu import CpuPool
from ..runtime.device import canonical_device, device_memory_budget
from ..runtime.device_events import DeviceEventPool
from ..runtime.device_products import DeviceProducts
from ..runtime.distributed import DistributedEnvironment, init_distributed_environment
from ..runtime.encoder_cache import EncoderCache
from ..runtime.latent_pool import LatentPool
from ..runtime.persistent_buffers import PersistentBuffers
from ..runtime.req_to_token_pool import ReqToTokenPool
from ..runtime.request import RequestPool
from ..runtime.runtime_states import RuntimeStates
from ..transfer.publications import TransferPublications
from ..transfer.tickets import make_transports
from . import messages
from .messages import PendingResponse, ServiceRequest
from .warmup import warmup_requests

if TYPE_CHECKING:
    from ..bootstrap.config import WorkerProcessArgs
    from ..bootstrap.ipc import WorkerIpcEndpoint

logger = logging.getLogger(__name__)

_IPC_WAIT_TIMEOUT_US = 60_000_000


class Worker:
    """Own a model, execution resources, and an optional single-use IPC service."""

    model: ExecutionModel
    worker_config: WorkerConfig
    runner: ModelRunner
    runtime_states: RuntimeStates | None
    cache_pool: CachePool | None
    req_to_token_pool: ReqToTokenPool | None
    latent_pool: LatentPool | None
    _completion_wake: Callable[[], None] | None = None

    def __enter__(self) -> Self:
        """Enter the owning scope for this worker's execution and service resources."""

        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources without suppressing or replacing the scope's error."""

        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise

            exc_value.add_note(f"Resource cleanup also failed: {cleanup_error!r}")

    @classmethod
    def from_config(cls, config: WorkerProcessArgs) -> Self:
        """Build a worker without numerical warmup, graph capture, or IPC I/O.

        Construction failures release all resources acquired here. The returned
        worker owns those resources until its context exits or the caller closes it.
        """

        # Establish device and collective membership before loading rank-local weights.
        distributed = init_distributed_environment(
            rank=config.execution.rank,
            local_rank=config.local_rank,
            world_size=config.execution.world_size,
            device=config.execution.device,
            backend=config.distributed_backend,
            init_method=config.distributed_init_method,
        )

        try:
            bindings = distributed.initialize_entries(dict(config.components))

            loaded = load_worker_model(config, bindings)

            # Sampling uses the language model's TP group, independently of other
            # components' parallel layouts. Backend selection belongs to the runner.
            model_mesh = bindings.meshes.get("model")
            sampling_group = None if model_mesh is None else model_mesh.get_group("tp")

            # The constructor owns partial execution allocations on failure. On
            # success the worker also takes responsibility for distributed.close().
            return cls(
                loaded.model,
                sampling_group=sampling_group,
                worker_config=loaded.worker_config,
                tokenizer=loaded.tokenizer,
                allowed_work_variants=config.supported_ops,
                transfer_backends=config.data_plane.backends,
                publication_backends=config.data_plane.publication_backends,
                worker_id=config.worker_id,
                schedule=loaded.schedule,
                pipeline_depth=config.ipc.pipeline_depth,
                completion_payload_bytes=config.ipc.max_payload_bytes,
                components=config.components,
                distributed_environment=distributed,
            )
        except BaseException as error:
            try:
                distributed.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")

            raise

    def __init__(
        self,
        model: ExecutionModel,
        *,
        worker_config: WorkerConfig,
        sampling_group: Communicator | None,
        tokenizer: Any | None,
        allowed_work_variants: frozenset[OpCode],
        pipeline_depth: int,
        completion_payload_bytes: int,
        attention: AttentionSelection | None = None,
        transfer_backends: tuple[str, ...] = ("local",),
        publication_backends: tuple[str, ...] = ("local",),
        worker_id: str = "worker",
        schedule: DiffusionSchedule | None = None,
        components: tuple[tuple[str, EntryConfig], ...] = (),
        distributed_environment: DistributedEnvironment | None = None,
    ) -> None:
        """Allocate execution resources for an already-loaded model.

        Resolve unspecified attention from worker_config. Construction performs
        no warmup or IPC I/O, and rolls back partial resource allocations on error.
        The caller closes the worker or uses its owning context after success.
        """

        self._closed = False
        self._run_started = False
        self._warmed_up = False

        self.ipc_endpoint: WorkerIpcEndpoint | None = None
        self.profiler: WorkerProfiler | None = None

        startup = ExitStack()

        try:
            if not isinstance(model, ExecutionModel):
                raise unsupported_setup("worker model has no supported execution surface")

            if not isinstance(worker_config, WorkerConfig):
                raise unsupported_setup("model worker requires a WorkerConfig")

            if pipeline_depth <= 0:
                raise unsupported_setup("worker pipeline depth must be positive")

            if (
                not publication_backends
                or len(set(publication_backends)) != len(publication_backends)
                or not set(publication_backends).issubset(transfer_backends)
            ):
                raise unsupported_setup("publication backends must be unique bound transports")

            if (
                "cuda_ipc" in transfer_backends
                and torch.device(worker_config.device).type != "cuda"
            ):
                raise unsupported_setup("CUDA IPC requires a CUDA worker device")

            self.model = model
            self.distributed_environment = distributed_environment
            self.sampling_group = sampling_group
            self.tokenizer = tokenizer

            self._device = canonical_device(worker_config.device)
            self._generation_device = canonical_device(
                worker_config.generation_device or worker_config.device
            )
            self._last_collective_seq = -1

            runner = ModelRunner(
                model,
                worker_config,
                environment=distributed_environment,
                attention=attention,
                schedule=schedule,
            )
            self.runner = runner
            startup.callback(runner.close)

            attention = runner.attention
            self.attention = attention

            # Measure the remaining grant after the runner binds persistent model
            # inputs and workspaces. Sizing earlier would treat occupied memory as free.
            # Request capacity is the only configuration resolved after binding.
            endpoint = WorkerEndpoint.local(worker_id, int(worker_config.rank))

            worker_config = resolve_request_capacity(
                model,
                worker_config,
                pipeline_depth=pipeline_depth,
                capacity_group=(
                    None
                    if distributed_environment is None
                    else distributed_environment.process_group
                ),
            )
            self.worker_config = worker_config
            runner.worker_config = worker_config

            layout = build_worker_layout(
                model,
                worker_config,
                endpoint=endpoint,
                queue_depth=int(pipeline_depth),
                completion_payload_bytes=int(completion_payload_bytes),
                allowed_work_variants=allowed_work_variants,
                transfer_backends=transfer_backends,
                components=components,
                attention_identity=attention.identity,
                # The scheduler's page indices are shared across all resident layer
                # and head regions, including stages with different memory grants.
                capacity_group=(
                    distributed_environment.process_group
                    if distributed_environment is not None
                    else sampling_group
                ),
            )

            # Auxiliary devices have separate grants; primary-device reservations
            # were already included in the layout's capacity calculation.
            for device, fixed_bytes in layout.fixed_device_bytes:
                if device == worker_config.device or canonical_device(device).type != "cuda":
                    continue

                available, _free = device_memory_budget(device, worker_config.kv_memory_fraction)
                reserved = fixed_bytes + graph_memory_budget_bytes(device_total_bytes(device))

                if reserved > available:
                    raise unsupported_setup(
                        f"runtime storage on {device} requires {reserved} bytes, "
                        f"but its static memory grant is {available} bytes"
                    )

            self._layout = layout
            info = layout.info
            arena = layout.arena

            owns_kv = bool(model.resource_geometry.kv)
            cache = model.cache_geometry if owns_kv else None

            self.cache_pool = None
            self.req_to_token_pool = None
            max_blocks_per_row = 0

            # KV pages and request-to-token tables share group geometry; bind them to
            # the model only after attention compatibility has been established.
            if cache is not None:
                kv_cache = info.kv_cache
                if kv_cache is None:
                    raise unsupported_setup("KV model worker has no KV-cache configuration")

                cache_dtype = getattr(torch, str(cache.dtype).removeprefix("torch."), None)
                if not isinstance(cache_dtype, torch.dtype):
                    raise unsupported_setup(f"unsupported cache dtype {cache.dtype!r}")

                max_blocks_per_row = max(
                    1,
                    ceil_div(int(model.text_max_tokens), int(worker_config.block_size)),
                )

                # Cache groups occupy consecutive ranges in the shared page pool.
                group_ranges: list[tuple[int, int]] = []
                group_offset = 0
                for group in kv_cache.groups:
                    group_ranges.append((group_offset, int(group.num_blocks)))
                    group_offset += int(group.num_blocks)

                self.cache_pool = CachePool(
                    num_layers=int(cache.num_layers),
                    num_pages=int(kv_cache.num_blocks),
                    page_size=int(kv_cache.block_size),
                    num_kv_heads=int(cache.num_kv_heads),
                    total_kv_heads=int(cache.total_kv_heads),
                    kv_head_offset=int(cache.kv_head_offset),
                    total_layers=cast(int, cache.total_layers),
                    layer_offset=int(cache.layer_offset),
                    head_dim=int(cache.head_dim),
                    device=worker_config.device,
                    dtype=cache_dtype,
                    store_dtype=cache.store_dtype,
                    group_ranges=tuple(group_ranges) if group_ranges else None,
                    import_capacity=int(info.max_unresolved_ops),
                )
                startup.callback(self.cache_pool.close)

                assert attention is not None
                if OpCode.DIFFUSION_STEP in info.supported_ops and not supports_flow_attention(
                    attention,
                    cache,
                    self.cache_pool,
                    torch.device(worker_config.device),
                ):
                    raise unsupported_setup(
                        "image generation requires paged-prefix plus dense-current attention"
                    )

                self.req_to_token_pool = ReqToTokenPool(
                    group_count=self.cache_pool.group_count,
                    request_pool_size=int(info.request_slots),
                    max_blocks_per_request=max_blocks_per_row,
                    block_size=int(kv_cache.block_size),
                    device=worker_config.device,
                    staging_depth=int(pipeline_depth),
                )
                startup.callback(self.req_to_token_pool.close)

                # Execution begins only after both physical pages and request
                # mappings exist; the model borrows this worker's cache storage.
                model.bind_cache_pool(self.cache_pool, attention)

            # Admission, lineage, and persistent tensors share one slot owner.
            self.requests = RequestPool(
                int(info.request_slots),
                tensor_schema=model.resource_geometry.request_tensors or None,
                device=worker_config.device,
            )

            torch_dtype = getattr(
                torch,
                str(worker_config.model_dtype).removeprefix("torch."),
                None,
            )
            if not isinstance(torch_dtype, torch.dtype):
                raise unsupported_setup(f"unsupported model dtype {worker_config.model_dtype!r}")

            if self.req_to_token_pool is not None:
                self.runtime_states = RuntimeStates(
                    request_pool_size=int(info.request_slots),
                    vocab_size=int(model.vocab_size),
                    continuation_width=1,
                    device=worker_config.device,
                    logits_dtype=torch_dtype,
                    valid_cache_lengths=self.req_to_token_pool.verified_lens,
                )
            else:
                self.runtime_states = None

            # Generation state has its own page pool and may live on another device.
            flow = model.generation
            latent_dtype = getattr(torch, str(layout.latent_dtype).removeprefix("torch."), None)
            if flow is not None and not isinstance(latent_dtype, torch.dtype):
                raise unsupported_setup(f"unsupported latent dtype {layout.latent_dtype!r}")

            if flow is None:
                self.latent_pool = None
            else:
                assert isinstance(latent_dtype, torch.dtype)
                self.latent_pool = LatentPool(
                    request_pool_size=int(info.request_slots),
                    num_pages=int(info.latent_pages),
                    page_units=int(info.latent_page_units),
                    latent_width=int(layout.latent_width),
                    dtype=latent_dtype,
                    device=worker_config.generation_device or worker_config.device,
                )
                startup.callback(self.latent_pool.close)

            if (
                self.latent_pool is not None
                and self.latent_pool.persistent_bytes != arena.latent_pool_bytes
            ):
                raise RuntimeError("latent pool allocation disagrees with its exact capacity plan")

            # These bounded stores own all asynchronous products, copies, CPU tasks,
            # and transfer lifetimes exposed by an in-flight pipeline.
            owner_devices = tuple(
                dict.fromkeys(
                    (
                        worker_config.device,
                        worker_config.generation_device or worker_config.device,
                    )
                )
            )

            self.device_events = DeviceEventPool()
            startup.callback(self.device_events.close)

            self.output_pool = OutputPool(
                capacity=int(pipeline_depth) * int(info.max_batch_ops),
                max_words=int(info.max_batch_ops) * (4 + (int(completion_payload_bytes) + 3) // 4),
                event_pool=self.device_events,
            )
            startup.callback(self.output_pool.close)

            self.persistent_buffers = PersistentBuffers(
                byte_capacity=int(layout.physical_buffer_pool_bytes),
                devices=owner_devices,
                compact=layout.physical_buffer_pool_bytes < info.buffer_pool_bytes,
            )
            startup.callback(self.persistent_buffers.close)

            self.device_products = DeviceProducts(
                capacity=arena.device_products,
                byte_capacity=arena.device_product_bytes,
                request_capacity=int(info.request_slots),
                relay_depth=int(info.max_unresolved_ops) + 1,
                persistent_buffers=self.persistent_buffers,
                event_pool=self.device_events,
            )
            startup.callback(self.device_products.close)

            self.encoder_cache = EncoderCache(
                entry_capacity=int(model.resource_geometry.encoder_cache_entries),
                max_entry_bytes=max(
                    1,
                    int(layout.max_latent_feature_bytes),
                    int(layout.max_vision_feature_bytes),
                ),
                devices=owner_devices,
                persistent_buffers=self.persistent_buffers,
                event_pool=self.device_events,
            )
            startup.callback(self.encoder_cache.close)

            self.cpu_tasks = CpuPool(
                capacity=int(arena.cpu_tasks),
                workers=min(4, int(arena.cpu_tasks)),
            )
            startup.callback(self.cpu_tasks.close)

            transfer_byte_capacity = int(arena.transfer_bytes)
            if model.resource_geometry.request_tensors:
                # Each live request tensor reserves one credit per publication
                # representation and one read credit on every possible remote rank.
                # These credits bound ownership lifetimes; they allocate no storage.
                transfer_byte_capacity *= len(publication_backends) + max(
                    0, int(worker_config.world_size) - 1
                )

            self.transports = make_transports(
                transfer_backends,
                source=endpoint,
                byte_capacity=transfer_byte_capacity,
                ticket_capacity=arena.transfer_tickets,
                event_pool=self.device_events,
            )
            for transport in self.transports.values():
                startup.callback(transport.close)

            self.transfer_publications = TransferPublications(self.transports)
            self.publication_transports = {
                name: self.transports[name] for name in publication_backends
            }

            if owns_kv:
                assert layout.input_geometry is not None and self.runtime_states is not None

                runner.configure_packed(
                    geometry=layout.input_geometry,
                    cache_pool=self.cache_pool,
                    latent_pool=self.latent_pool,
                    decode_predicates=self.runtime_states.predicates,
                    max_operations=int(info.max_batch_ops),
                    request_slots=int(info.request_slots),
                    max_tokens=int(info.max_batch_tokens),
                    latent_capacity_units=int(info.latent_capacity_units),
                    decode_context_blocks=decode_context_blocks(
                        model, worker_config, self.cache_pool
                    ),
                    variants=frozenset(info.supported_ops),
                    max_inflight=int(pipeline_depth),
                )

            # Operation handlers borrow the resources owned by this rank.
            from ..execution.video import create_media_resources
            from ..models.video import VideoModel

            self.media_mux, self.media_output_ring = (
                create_media_resources(
                    model,
                    device=self._generation_device,
                    rank=worker_config.rank,
                    owns_output=model.owns_media_output
                    and worker_config.rank == info.output_rank("output"),
                    state_slots=info.request_slots,
                    unresolved_window=info.max_unresolved_ops,
                )
                if isinstance(model, VideoModel)
                else (None, None)
            )
            if self.media_mux is not None:
                startup.callback(self.media_mux.close)

            self.cache_publications = (
                CachePublications(self.cache_pool, self.req_to_token_pool)
                if self.cache_pool is not None and self.req_to_token_pool is not None
                else None
            )
        except BaseException as error:
            try:
                startup.close()
            except BaseException as cleanup_error:
                error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")

            raise

        # All registered resources now belong to this Worker and its close method.
        startup.pop_all()

    def bind(self, endpoint: WorkerIpcEndpoint) -> Self:
        """Borrow an open endpoint and install bounded service queues and completion wakes.

        The caller owns the endpoint and must keep it open until the Worker scope
        exits or close() finishes. Binding is exclusive and cannot be replaced.
        The surrounding Worker scope owns cleanup on success or failure.
        """

        self._require_open()

        if self.ipc_endpoint is not None:
            raise RuntimeError("worker already has a bound IPC endpoint")

        if endpoint is None or endpoint.closed:
            raise ValueError("worker binding requires an open IPC endpoint")

        self._init_request_scheduling()
        self._init_run_tracking()

        self.profiler = WorkerProfiler.from_env()

        # Publish the binding only after all service state exists. Register
        # callbacks last so completion notifications always find that state.
        self.ipc_endpoint = endpoint
        self.set_completion_wake(endpoint.wake, endpoint.wake_on_stream)

        return self

    def _init_request_scheduling(self) -> None:
        """Initialize admission limits, dependency chains, and ready-request queues."""

        self.pipeline_depth = max(1, int(self.info.queue_depth))
        self._next_sequence = 1
        self._transport_occupancy = 0
        self._admission_closed = False
        self._shutdown_response: dict[str, Any] | None = None

        # Ready requests follow submission order; dependencies and capacity can
        # hold one request while unrelated ready work continues.
        self._pending_requests: dict[int, ServiceRequest] = {}
        self._request_tails: dict[int, ServiceRequest] = {}
        self._ready_requests: list[ServiceRequest] = []

        # Collective participants must observe the host's submission order even
        # when their local dependency chains become ready at different times.
        self._collective_submission_tail: ServiceRequest | None = None
        self._preserve_collective_order = (
            int(self.info.world_size) > 1 and self.model.ordered_collective_execution
        )

    def _init_run_tracking(self) -> None:
        """Keep bounded in-flight work and a constant-size admission high-water mark."""

        # Physical IDs increase in transport submission order, independently of
        # logical batch IDs. Completion order does not affect admission.
        self._last_run_id = -1
        self.runs: dict[int, WorkerRun] = {}
        self._preparation_ready: SimpleQueue[WorkerRun] = SimpleQueue()
        self._executing_runs: deque[WorkerRun] = deque()

        self.pending_responses: deque[PendingResponse] = deque()
        self._waiting_responses: dict[int, PendingResponse] = {}

    def run(self) -> None:
        """Warm up and serve synchronously once, within the owner's resource scope.

        No service requests are consumed before warmup completes. Returning or
        raising leaves resource cleanup to the surrounding Worker context.
        """

        self._require_open()

        if self._run_started:
            raise RuntimeError("worker can only run once")

        if self.ipc_endpoint is None:
            raise RuntimeError("worker has no bound IPC endpoint")

        self._run_started = True

        self.warmup()
        self._process_requests()

    def _process_requests(self) -> None:
        """Advance accepted work and deliver responses until the service drains."""

        endpoint = self.ipc_endpoint
        assert endpoint is not None

        # Suspend cyclic collection only during serving, restoring the caller's
        # setting before either normal shutdown or exceptional resource cleanup.
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            while True:
                # Give accepted work and ready responses priority over admission.
                self.device_events.reap()

                if self._advance_executing_runs():
                    continue

                if self._send_one_ready_response():
                    continue

                if self._launch_one_ready_request():
                    continue

                # Close is acknowledged after execution and claimed responses drain.
                if self._admission_closed:
                    self._close_completed_polls()

                    if self._service_drained():
                        if self._shutdown_response is not None:
                            self._transport_respond(self._shutdown_response)

                        return

                if not self._admission_closed and self._transport_occupancy < self.pipeline_depth:
                    request = endpoint.try_recv()
                    if request is not None:
                        self._accept(request)
                        continue

                # Outstanding work needs completion wakes as well as IPC arrivals.
                if (
                    self._pending_requests
                    or self.pending_responses
                    or self._waiting_responses
                    or self.runs
                ):
                    endpoint.wait_incoming(_IPC_WAIT_TIMEOUT_US)
                    continue

                if self._admission_closed:
                    return

                self._accept(endpoint.recv())
        finally:
            if gc_was_enabled:
                gc.enable()

    def _dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve an administrative request without transport I/O."""

        kind = messages.request_kind(request)
        if kind is RequestKind.INFO:
            return messages.response(ResponseKind.INFO, info=self.info.to_mapping())
        if kind is RequestKind.CLOSE:
            return messages.response(ResponseKind.OK)
        raise invalid_descriptor(f"unsupported administrative request {kind.value!r}")

    def _boxed_error(self, request: Mapping[str, Any], error: BaseException) -> dict[str, Any]:
        """Classify an exception, record it, and encode the protocol error response."""

        classified = (
            error
            if isinstance(error, WorkerError)
            else classify(error, context=str(request.get("kind", "unknown")))
        )
        record_failure(
            request.get("kind"),
            classified,
            unexpected=not isinstance(error, WorkerError),
        )
        return messages.error_response(classified, request)

    def _accept(self, request: dict[str, Any]) -> int:
        """Assign transport order, decode the request, and link per-request dependencies."""

        # Sequence and occupancy are assigned before parsing so even malformed
        # requests produce an ordered response and release one transport slot.
        sequence = self._next_sequence
        self._next_sequence += 1
        self._transport_occupancy += 1
        requests = messages.raw_request_ids(request)
        try:
            kind = messages.request_kind(request)

            # Close commands stop admission immediately but retain their
            # sequence position until all earlier responses have drained.
            if kind is RequestKind.CLOSE:
                self._admission_closed = True
                self._shutdown_response = messages.with_call_id(self._dispatch(request), request)
                return sequence
            run: Run | None = None
            if kind is RequestKind.SUBMIT:
                raw_run = messages.required(request, "run", kind)
                raw_run_id = (
                    int(raw_run.run_id)
                    if isinstance(raw_run, Run)
                    else int(raw_run.get("run_id", -1))
                    if isinstance(raw_run, Mapping)
                    else -1
                )
                with profile_range(
                    worker_range_name("run_decode", run_id=raw_run_id, rank=self.info.endpoint.rank)
                ):
                    run = raw_run if isinstance(raw_run, Run) else Run.from_mapping(raw_run)
                if run.run_id <= self._last_run_id:
                    raise invalid_descriptor(
                        f"run id {run.run_id} must exceed previously submitted id {self._last_run_id}"
                    )
                self._last_run_id = run.run_id

                # Product release is independent of request-state transitions.
                # An earlier numerical run may need this retired allocation,
                # so revocation cannot wait behind that run's execution FIFO.
                # Existing readers retain storage; the command's ordinary
                # terminal acknowledgement still waits for their completion.
                freed = tuple(
                    command.buffer for command in run.commands if isinstance(command, Free)
                )
                if freed:
                    release_buffers(
                        freed,
                        cache_pool=self.cache_pool,
                        device_products=self.device_products,
                        encoder_cache=self.encoder_cache,
                        latent_pool=self.latent_pool,
                        transfer_publications=self.transfer_publications,
                    )
                requests = messages.run_requests(run)

            # Each request identifier forms a FIFO dependency chain. Multi-key
            # work waits once per distinct predecessor to avoid double counts.
            pending = ServiceRequest(
                sequence=sequence,
                request=request,
                requests=requests,
                kind=kind,
                run=run,
            )
            predecessors = {
                id(predecessor): predecessor
                for request_id in requests
                if (predecessor := self._request_tails.get(request_id)) is not None
            }
            # Component participants can finish the same operation at different
            # times. Preserve the host's cooperative submission order even when
            # a later, independent request has already satisfied its own lineage.
            # Release at successor visibility keeps device and CPU completion
            # overlap under the existing execution credits.
            if kind is RequestKind.SUBMIT and self._preserve_collective_order:
                if self._collective_submission_tail is not None:
                    predecessors[id(self._collective_submission_tail)] = (
                        self._collective_submission_tail
                    )
                self._collective_submission_tail = pending
            pending.dependencies = len(predecessors)
            for predecessor in predecessors.values():
                predecessor.successors.append(pending)
            for request_id in requests:
                self._request_tails[request_id] = pending
            self._pending_requests[sequence] = pending
            if pending.dependencies == 0:
                self._enqueue(pending)
        except BaseException as error:
            # Parse and admission failures enter the same ordered response queue
            # as successfully launched requests.
            self.pending_responses.append(
                PendingResponse(sequence, requests, self._boxed_error(request, error))
            )
        return sequence

    def _enqueue(self, pending: ServiceRequest) -> None:
        """Insert dependency-ready work in the caller's submission order."""

        insort(self._ready_requests, pending, key=lambda request: request.sequence)

    def _release(self, pending: ServiceRequest) -> None:
        """Release one completed request and wake successors whose dependencies reach zero."""

        if pending.released:
            return
        pending.released = True
        self._pending_requests.pop(pending.sequence, None)
        if self._collective_submission_tail is pending:
            self._collective_submission_tail = None
        for request in pending.requests:
            if self._request_tails.get(request) is pending:
                del self._request_tails[request]
        for successor in pending.successors:
            successor.dependencies -= 1
            if successor.dependencies == 0:
                self._enqueue(successor)
        pending.successors.clear()

    def _launch_one_ready_request(self) -> bool:
        """Select and launch one dependency-ready administrative or execution request."""

        execution_full = len(self.runs) >= self.pipeline_depth
        position = next(
            (
                index
                for index, request in enumerate(self._ready_requests)
                if request.kind is not RequestKind.SUBMIT or not execution_full
            ),
            None,
        )
        if position is None:
            return False
        pending = self._ready_requests.pop(position)

        try:
            if pending.kind is RequestKind.SUBMIT:
                self._launch_execute(pending)
            elif pending.kind is RequestKind.POLL:
                self._launch_poll(pending)
                self._release(pending)
            else:
                self._launch_admin(pending)
                self._release(pending)
        except BaseException as error:
            self.pending_responses.append(
                PendingResponse(
                    pending.sequence,
                    pending.requests,
                    self._boxed_error(pending.request, error),
                )
            )
            self._release(pending)
        return True

    def _launch_admin(self, pending: ServiceRequest) -> None:
        """Dispatch an administrative request and queue its ordered response."""

        response = self._dispatch(pending.request)
        self.pending_responses.append(
            PendingResponse(
                pending.sequence,
                pending.requests,
                messages.with_call_id(response, pending.request),
            )
        )

    def _launch_execute(self, pending: ServiceRequest) -> None:
        """Start one accepted run and queue its first result fragment."""

        if pending.run is None:
            raise RuntimeError("accepted execute request lost its run")
        run = WorkerRun(
            self.plan_run(pending.run),
            on_successors_ready=lambda _: self._release(pending),
            on_ready=self._run_ready,
        )
        self.runs[run.run_id] = run
        self._start_execution(run)
        self._queue_result(pending, run)

    def _start_execution(self, run: WorkerRun) -> None:
        """Prepare transfers and predicates, then execute now or return a readiness-gated future."""

        try:
            unsupported = tuple(
                operation.kind
                for operation in run.run.operations
                if not self.supports_run_kind(operation.kind)
            )
            if unsupported:
                names = sorted({value.value for value in unsupported})
                raise invalid_descriptor(
                    f"execution run contains work variants unsupported by this worker: {names!r}"
                )

            source: RunResult | PreparedExecution
            if run.run.operations:
                source = self.prepare_execute(run.run)
            else:
                with profile_range(
                    worker_range_name(
                        "model_execute", run_id=run.run_id, rank=self.info.endpoint.rank
                    )
                ):
                    source = self.execute(run.run)

            run.attach(source)
            if run.advance_execution():
                if not run.complete:
                    self._executing_runs.append(run)
            else:
                if not isinstance(source, PreparedExecution):
                    raise RuntimeError("pending execution source has no readiness owner")
                source.on_dependencies_ready(partial(self._preparation_completed, run))
        except BaseException as error:
            run.fail(error)

    def _launch_poll(self, pending: ServiceRequest) -> None:
        """Claim the next undelivered fragment of a previously submitted run."""

        run_id = messages.integer(pending.request, "run_id", RequestKind.POLL)
        run = self.runs.get(run_id)
        if run is None or not run.awaiting_poll:
            raise invalid_descriptor(f"poll names run {run_id} with no pending results")
        run.awaiting_poll = False
        self._queue_result(pending, run)

    def _queue_result(self, request: ServiceRequest, run: WorkerRun) -> None:
        """Queue a Submit or Poll response when its next result fragment is ready."""

        pending = PendingResponse(
            request.sequence,
            request.requests | run.request_ids,
            messages.with_call_id(messages.response(ResponseKind.RESULT), request.request),
            run,
        )
        if self._pending_ready(pending):
            self.pending_responses.append(pending)
        else:
            self._waiting_responses[run.run_id] = pending

    def _run_ready(self, run: WorkerRun) -> None:
        """Wake the single response waiting for this run's next fragment."""

        pending = self._waiting_responses.pop(run.run_id, None)
        if pending is not None:
            if self._pending_ready(pending):
                self.pending_responses.append(pending)
            else:
                self._waiting_responses[run.run_id] = pending

    def _preparation_completed(self, run: WorkerRun) -> None:
        """Enqueue readiness before waking the IPC loop that consumes it."""

        self._preparation_ready.put(run)
        if self.ipc_endpoint is not None:
            self.ipc_endpoint.wake()

    def _advance_executing_runs(self) -> bool:
        """Launch preparation-ready work and advance executing runs in launch order."""

        advanced = False
        if not self._preparation_ready.empty():
            run = self._preparation_ready.get_nowait()
            if not run.complete:
                launched = run.advance_execution()
                if not run.complete:
                    if launched:
                        self._executing_runs.append(run)
                    else:
                        source = run.source
                        if not isinstance(source, PreparedExecution):
                            raise RuntimeError("pending execution source has no readiness owner")
                        source.on_dependencies_ready(partial(self._preparation_completed, run))
            advanced = True
        # Query every launched run: one pending host read or retirement must not
        # hide an independent completion behind it.
        for _ in range(len(self._executing_runs)):
            run = self._executing_runs.popleft()
            before = run.state
            run.advance()
            advanced |= run.complete or run.state != before
            if not run.complete:
                self._executing_runs.append(run)
        return advanced

    def _pending_ready(self, pending: PendingResponse) -> bool:
        """Query completion without blocking the service thread."""

        run = pending.run
        if run is None:
            return True
        with profile_range(
            worker_range_name("completion", run_id=run.run_id, rank=self.info.endpoint.rank)
        ):
            return run.ready()

    def _send_one_ready_response(self) -> bool:
        """Send the oldest transport-ready response if one exists."""

        if not self.pending_responses:
            return False
        self._send_pending(self.pending_responses.popleft())
        return True

    def _send_pending(self, pending: PendingResponse) -> None:
        """Serialize and send one ready response while preserving transport sequence order."""

        response = dict(pending.response)
        run = pending.run
        run_id = run.run_id if run is not None else None
        if run is not None:
            if run.error is not None:
                response = messages.error_response(run.take_error(), pending.response)
            else:
                response["result"] = run.take_ready()
            if run.pending():
                run.awaiting_poll = True
            else:
                del self.runs[run.run_id]
                run.close()

        fatal = bool(response.get("fatal"))
        with profile_range(
            worker_range_name("finalize_response", run_id=run_id, rank=self.info.endpoint.rank)
        ):
            finalized = messages.finalize_response(response)
        self._transport_respond(finalized)
        if pending.sequence > 0:
            self._transport_occupancy -= 1
            if self._transport_occupancy < 0:
                raise RuntimeError("worker transport occupancy underflow")
        if fatal:
            self._admission_closed = True

    def _transport_respond(self, response: dict[str, Any]) -> None:
        """Send one finalized response through the bound IPC endpoint."""

        if self.ipc_endpoint is None:
            raise RuntimeError("worker has no bound IPC endpoint")
        with profile_range("uniserve.worker.respond"):
            self.ipc_endpoint.respond(response)

    def _close_completed_polls(self) -> None:
        """Close discards unclaimed fragments after their execution safely retires."""

        for run_id, run in tuple(self.runs.items()):
            if not run.awaiting_poll:
                continue
            run.advance()
            if run.complete:
                del self.runs[run_id]
                run.close()

    def _service_drained(self) -> bool:
        """Return whether all accepted work and claimed responses have drained."""

        return (
            not self._pending_requests
            and not self.pending_responses
            and not self._waiting_responses
            and not self.runs
        )

    @property
    def info(self) -> WorkerInfo:
        """Expose immutable params, capacity, and model metadata advertised to the scheduler."""

        return self._layout.info

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("worker is closed and cannot be reused")

    def execute(self, batch: Run) -> RunResult:
        """Plan and synchronously resolve one physical run."""

        self._require_open()
        return self._execute_batch(self.plan_run(batch))

    def _execute_batch(
        self,
        batch: Run,
        *,
        prepared: PreparedExecution | None = None,
        propagate_errors: bool = False,
    ) -> RunResult:
        """Use the same execution resources and retirement rules for every run."""

        # Cooperative ranks launch computation in the same order. Preparation
        # and host completion may overlap; neither retains old run identities.
        if self.worker_config.world_size > 1 and batch.operations:
            if batch.collective_seq <= self._last_collective_seq:
                raise invalid_descriptor("collective sequence does not advance")
            self._last_collective_seq = batch.collective_seq

        report = execute_batch(
            batch,
            prepared=prepared,
            propagate_errors=propagate_errors,
            cache_pool=self.cache_pool,
            cache_registry=self.cache_publications,
            cpu_tasks=self.cpu_tasks,
            device_products=self.device_products,
            encoder_cache=self.encoder_cache,
            worker_info=self.info,
            latent_pool=self.latent_pool,
            media_mux=self.media_mux,
            media_output_ring=self.media_output_ring,
            execution_model=self.model,
            output_pool=self.output_pool,
            publication_transports=self.publication_transports,
            request_tables=self.req_to_token_pool,
            request_pool=self.requests,
            model_runner=self.runner,
            runtime_states=self.runtime_states,
            sampling_group=self.sampling_group,
            tokenizer=self.tokenizer,
            transfer_publications=self.transfer_publications,
            transfer_backends=self.transports,
            config=self.worker_config,
        )
        return self._retire_commands(batch, report)

    def plan_run(self, batch: Run) -> Run:
        """Derive the worker-private execution lanes for one physical run."""

        self._require_open()

        entries = self.info.components
        if entries:
            for operation in batch.operations:
                entry = next((entry for entry in entries if entry.name == operation.entry), None)
                if entry is None or self.worker_config.rank not in entry.config.ranks:
                    raise invalid_descriptor(
                        f"operation targets entry {operation.entry!r} outside this rank"
                    )
        return plan_run(batch, worker_info=self.info, model_runner=self.runner)

    def supports_run_kind(self, kind: OpCode) -> bool:
        """Return whether this worker can execute one physical run variant."""

        return kind in self.info.supported_ops

    def prepare_execute(self, batch: Run) -> PreparedExecution:
        """Stage a run and retain its asynchronous inputs until execution or abandonment.

        The caller must execute or abandon the returned preparation before
        leaving the Worker scope.
        """

        self._require_open()

        batch = self.plan_run(batch)
        prepared = prepare_batch(
            batch,
            cache_pool=self.cache_pool,
            cache_registry=self.cache_publications,
            device_products=self.device_products,
            encoder_cache=self.encoder_cache,
            worker_info=self.info,
            latent_pool=self.latent_pool,
            execution_model=self.model,
            output_pool=self.output_pool,
            request_tables=self.req_to_token_pool,
            request_pool=self.requests,
            model_runner=self.runner,
            runtime_states=self.runtime_states,
            transfer_publications=self.transfer_publications,
            transfer_backends=self.transports,
            config=self.worker_config,
        )

        return prepared.bind(
            lambda value: self._execute_batch(value.batch, prepared=value),
        )

    def _retire_commands(
        self,
        batch: Run,
        report: RunResult,
    ) -> RunResult:
        """Delay command acknowledgement until physical readers and request storage retire."""

        closed = frozenset(
            command.request_key
            for command in batch.commands
            if isinstance(command, (Finish, Retire))
        )
        local_closed = frozenset(
            key
            for key in closed
            if (row := self.requests.peek(key.request_id)) is not None and row.request_key == key
        )
        freed = frozenset(command.buffer for command in batch.commands if isinstance(command, Free))
        if not closed and not freed:
            return report
        retained = (
            frozenset(
                buffer
                for command in batch.commands
                if isinstance(command, (Finish, Retire))
                for buffer in command.retained_buffers
            )
            - freed
        )
        self.device_products.release_requests(closed, retained=retained)
        self.encoder_cache.release_requests(closed, retained=retained)
        if self.latent_pool is not None:
            self.latent_pool.cancel_imports(tuple(closed))
        publications = self.transfer_publications
        selected = publications.retiring(buffers=freed, requests=closed, retained=retained)
        releases = publications.release(selected)
        if self.latent_pool is not None:
            self.latent_pool.release_buffers(selected)
        if self.cache_pool is not None:
            self.cache_pool.imports.cancel_requests(closed, retained=retained)
            self.cache_pool.release_buffers(selected)
        wake = self._completion_wake
        if wake is not None:
            for future in releases:
                future.add_done_callback(lambda _future: wake())

        def record_fences() -> tuple[torch.cuda.Event, ...]:
            events = []
            for device in self.persistent_buffers.devices:
                if device.type != "cuda":
                    continue
                event = self.device_events.acquire(device)
                self.device_events.retain(event, device)
                self.device_events.record(event, device)
                self.device_events.schedule_completion_wake(device, event)
                events.append(event)
            return tuple(events)

        # Output completion precedes some request-state writes on the execution
        # stream. Finish must include those writes before resetting the slot.
        pending = record_fences() if closed else ()
        cleaned = False

        def retirement_ready() -> bool:
            nonlocal pending, cleaned
            self.device_events.reap()
            if not all(event.query() for event in pending):
                return False
            for event in pending:
                self.device_events.release(event)
            pending = ()
            if cleaned:
                return True
            for request_key in local_closed:
                if not self.requests.retirement_ready(request_key):
                    return False
            if not self.device_products.retirement_ready(
                buffers=freed, requests=closed, retained=retained
            ):
                return False
            if not self.encoder_cache.retirement_ready(
                buffers=freed, requests=closed, retained=retained
            ):
                return False
            if self.latent_pool is not None and not self.latent_pool.retirement_ready(
                tuple(closed)
            ):
                return False
            if self.cache_pool is not None and not self.cache_pool.retirement_ready(
                buffers=freed, requests=closed, retained=retained
            ):
                return False
            for future in releases:
                if not future.done():
                    return False
                future.result()
            publications.forget(selected)
            for request_key in local_closed:
                self.retire_request(request_key, retained=retained)
            # Slot reset can itself submit device writes. Its completion is
            # part of the acknowledgement that authorizes address reuse.
            pending = record_fences() if closed else ()
            cleaned = True
            return not pending

        return replace(report, retirement=retirement_ready)

    def execute_prepared(self, prepared: PreparedExecution) -> RunResult:
        """Resolve an already staged execution after validating its ownership type."""

        self._require_open()

        if not isinstance(prepared, PreparedExecution):
            raise invalid_descriptor("prepared execution has an invalid type")
        return prepared.resolve()

    def warmup(self) -> None:
        """Prepare numerical execution once without serving requests.

        Successful warmup is retained for run(). The owner must exit the Worker
        scope if startup fails, just as for a binding or service failure.
        """

        self._require_open()
        if self._warmed_up:
            return

        self.runner.prepare_fixed_modules()
        self.runner.warmup(self.requests.tensor_slots)
        self.runner.capture(tokenizer=self.tokenizer, latents=self.latent_pool)
        warmup_requests(self)
        self.runner.complete_startup()

        # Startup scenarios must release their requests before service admission.
        if self.requests.request_ids():
            raise RuntimeError("startup completed with resident requests")
        self._last_collective_seq = -1
        check_startup_memory(
            self.worker_config, self._layout.arena.device_product_bytes, self.device_products
        )
        self._warmed_up = True

    def drop_request(self, request_id: int) -> None:
        """Release all runtime, cache, latent, product, and transfer state for one request."""

        request_id = int(request_id)
        request = self.requests.peek(request_id)
        drop_execution_request(
            request_id,
            cache_pool=self.cache_pool,
            cache_registry=self.cache_publications,
            request_tables=self.req_to_token_pool,
            request_pool=self.requests,
            runtime_states=self.runtime_states,
            transfer_publications=self.transfer_publications,
            transfer_backends=self.transports,
        )
        if request is not None:
            self.device_products.release_requests((request.request_key,))
            self.encoder_cache.release_requests((request.request_key,))
        if self.media_mux is not None:
            self.media_mux.drop(request_id)
        if request is not None and self.latent_pool is not None:
            self.latent_pool.release_slots((int(request.request_pool_idx),))
        self.requests.drop(request_id)

    def retire_request(
        self, request_key: RequestKey, *, retained: frozenset[BufferId] = frozenset()
    ) -> None:
        """Retire the exact epoch while keeping independently owned persistent products."""

        request_id = int(request_key.request_id)
        request = self.requests.peek(request_id)
        if request is None or request.request_key != request_key or request.retired:
            return
        drop_execution_request(
            request_id,
            retained=retained,
            cache_pool=self.cache_pool,
            cache_registry=self.cache_publications,
            request_tables=self.req_to_token_pool,
            request_pool=self.requests,
            runtime_states=self.runtime_states,
            transfer_publications=self.transfer_publications,
            transfer_backends=self.transports,
        )
        self.device_products.release_requests((request.request_key,), retained=retained)
        self.encoder_cache.release_requests((request.request_key,), retained=retained)
        if self.media_mux is not None:
            self.media_mux.drop(request_id)
        if self.latent_pool is not None:
            self.latent_pool.release_slots((int(request.request_pool_idx),))
        self.requests.retire(request_id)

    def close(self) -> None:
        """Idempotently drain and release owned resources; never close the borrowed IPC endpoint.

        Every owner is given a chance to release even if another release fails.
        Direct execution and service startup are both forbidden after closing.
        """

        if self._closed:
            return

        self._closed = True

        actions: list[Callable[[], object]] = [self.runner.synchronize]
        if self.profiler is not None:
            actions.append(self.profiler.close)
        if self.ipc_endpoint is not None:
            actions.append(self._release_service_runs)

        # Submitted jobs retain their mux sessions until host work has finished.
        actions.append(self.cpu_tasks.close)
        if self.media_mux is not None:
            actions.append(self.media_mux.close)

        # Stop imports and transports before releasing the storage they borrow.
        if self.cache_pool is not None:
            actions.append(self.cache_pool.imports.stop)
        actions.extend(transport.close for transport in self.transports.values())
        actions.append(self.transfer_publications.clear)

        actions.append(self.output_pool.close)
        if self.cache_pool is not None:
            actions.append(self.cache_pool.close)
        if self.req_to_token_pool is not None:
            actions.append(self.req_to_token_pool.close)
        if self.latent_pool is not None:
            actions.append(self.latent_pool.close)

        actions.extend(
            (
                self.encoder_cache.close,
                self.device_products.close,
                self.persistent_buffers.close,
                self.device_events.close,
                self.runner.close,
            )
        )
        if self.distributed_environment is not None:
            actions.append(self.distributed_environment.close)

        # Async producers have stopped before callback references are removed.
        actions.append(partial(self.set_completion_wake, None, None))

        try:
            close_resources(*actions)
        finally:
            self.ipc_endpoint = None

            # Keep immutable configuration/info available to the caller, but
            # retaining a closed Worker must not retain its model and tensors.
            del (
                self.model,
                self.runner,
                self.tokenizer,
                self.attention,
                self.sampling_group,
                self.distributed_environment,
                self.requests,
                self.runtime_states,
                self.req_to_token_pool,
                self.cache_pool,
                self.cache_publications,
                self.latent_pool,
                self.media_mux,
                self.media_output_ring,
                self.output_pool,
                self.encoder_cache,
                self.device_products,
                self.persistent_buffers,
                self.device_events,
                self.cpu_tasks,
                self.transports,
                self.publication_transports,
                self.transfer_publications,
            )

    def _release_service_runs(self) -> None:
        """Release in-flight preparations and outputs before their runtime owners close."""

        try:
            close_resources(*(run.close for run in self.runs.values()))
        finally:
            self.runs.clear()
            self.pending_responses.clear()
            self._waiting_responses.clear()

            self._pending_requests.clear()
            self._request_tails.clear()
            self._ready_requests.clear()
            self._collective_submission_tail = None
            self._executing_runs.clear()

            while not self._preparation_ready.empty():
                self._preparation_ready.get_nowait()

    def set_completion_wake(
        self,
        wake: Callable[[], None] | None,
        wake_on_stream: Callable[[int], None] | None,
    ) -> None:
        """Register host and CUDA-stream callbacks used to wake result polling."""

        self._completion_wake = wake
        self.device_events.set_completion_wake(wake_on_stream)
        self.cpu_tasks.set_completion_wake(wake)

        for transport in self.transports.values():
            transport.set_completion_wake(wake)

        if self.cache_pool is not None:
            self.cache_pool.imports.set_completion_wake(wake)


__all__ = ["Worker"]
