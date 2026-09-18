"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import gc
import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from queue import SimpleQueue
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.model import CausalLM, VideoPostprocessor
from uniserve.processing import FlowPrompt, ImageProcessor
from uniserve.profiling import profile_range
from uniserve.quantization import Quantizer
from uniserve.runtime import EventPool, PrefixCache
from uniserve.runtime.device import canonical_device, device_memory_budget
from uniserve.runtime.process_groups import (
    ProcessGroups,
    initialize_process_groups,
)
from uniserve.runtime.resources import close_resources
from uniserve_worker.profiling import (
    WorkerProfiler,
    record_failure,
    worker_range_name,
)
from uniserve_worker.protocol.call import CallKind
from uniserve_worker.protocol.identity import (
    BufferId,
    CallId,
    RequestKey,
)

from ..bootstrap.capacity import (
    check_startup_memory,
    decode_context_blocks,
    device_total_bytes,
    resolve_request_capacity,
)
from ..bootstrap.distributed import initialize_entries
from ..bootstrap.inputs import capability, image_builder
from ..bootstrap.model_loader import load_worker_model, prepare_worker_model
from ..bootstrap.worker_info import RequestKind, ResponseKind, WorkerInfo
from ..bootstrap.worker_info_builder import build_worker_layout
from ..config import (
    WorkerConfig,
    graph_memory_budget_bytes,
)
from ..execution.batch_state import BatchState
from ..execution.model_entry import ModelEntry
from ..execution.model_runner import ModelRunner
from ..execution.output import OutputPool, PendingOutput
from ..execution.prepare import (
    capture_predicates,
    prepare_batch,
    prepare_inputs,
    validate_batch,
)
from ..execution.step import execute_batch
from ..foundation.errors import (
    WorkerError,
    classify,
    invalid_descriptor,
    unsupported_setup,
)
from ..protocol.batch import Batch, Finish, Free
from ..protocol.output import BatchOutput
from ..protocol.transfer import WorkerEndpoint
from ..runtime.block_tables import BlockTables
from ..runtime.buffer_pool import BufferPool
from ..runtime.cache_manager import CacheManager
from ..runtime.decode_state import DecodeState
from ..runtime.host_lane import HostLane
from ..runtime.latent_pool import LatentPool
from ..runtime.request import RequestPool
from ..runtime.tensor_store import TensorStore
from ..transfer.exports import forget_exports, release_exports, retiring_exports
from ..transfer.tickets import make_transports
from . import messages
from .messages import PendingResponse, ServiceRequest
from .warmup import warmup_requests

if TYPE_CHECKING:
    from ..bootstrap.config import ComponentConfig, WorkerProcessArgs
    from ..bootstrap.launch import WorkerIpcEndpoint

logger = logging.getLogger(__name__)

_IPC_WAIT_TIMEOUT_US = 60_000_000


def _input_producers(
    batch: Batch,
) -> set[tuple[RequestKey, CallId]]:
    """Identify producers requiring extended visibility.

    Their values must remain visible until input acquisition.
    """
    sources = {
        (reference.request_key, reference.producer_call_id)
        for call in batch.calls
        for reference in (
            *call.tensor_inputs(),
            *(() if call.predicate is None else (call.predicate,)),
        )
    }
    sources.update(
        (call.kv_input.owner, call.kv_input.producer_call_id)
        for call in batch.calls
        if call.kv_input is not None
    )
    return sources


class Worker:
    """Own a model and its execution resources.

    Also owns an optional single-use IPC service.
    """

    model: nn.Module
    worker_config: WorkerConfig
    runner: ModelRunner
    decode_state: DecodeState | None
    kv_cache: CacheManager | None
    block_tables: BlockTables | None
    latent_pool: LatentPool | None
    _completion_wake: Callable[[], None] | None = None

    def __enter__(self) -> Self:
        """Enter the owning scope.

        The scope covers this worker's execution and service resources.
        """
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release resources on scope exit.

        The scope's error is neither suppressed nor replaced.
        """
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise

            exc_value.add_note(
                f"Resource cleanup also failed: {cleanup_error!r}"
            )

    @classmethod
    def from_config(cls, config: WorkerProcessArgs) -> Self:
        """Build a worker without numerical warmup, graph capture, or IPC I/O.

        Construction failures release all resources acquired here. The
        returned worker owns those resources until its context exits or the
        caller closes it.
        """
        source = prepare_worker_model(config)

        # Mathematical constraints are checked before physical groups exist.
        try:
            distributed = initialize_process_groups(
                rank=config.execution.rank,
                local_rank=config.local_rank,
                world_size=config.execution.world_size,
                device=config.execution.device,
                backend=config.distributed_backend,
                init_method=config.distributed_init_method,
            )
        except ValueError as error:
            # Public numerical resources report ordinary argument errors;
            # the serving boundary assigns the worker's configuration code.
            raise unsupported_setup(str(error)) from error

        try:
            try:
                bindings = initialize_entries(
                    distributed, dict(config.components)
                )
            except ValueError as error:
                raise unsupported_setup(str(error)) from error

            loaded = load_worker_model(config, bindings, source=source)

            # Sampling uses the language model's TP group, independently of
            # other components' parallel layouts. Backend selection belongs
            # to the runner.
            model_mesh = bindings["model"].mesh if "model" in bindings else None
            sampling_group = (
                None if model_mesh is None else model_mesh.get_group("tp")
            )

            # The constructor owns partial execution allocations on failure.
            # On success the worker also takes responsibility for
            # distributed.close().
            return cls(
                loaded.model,
                bindings=bindings,
                sampling_group=sampling_group,
                worker_config=loaded.config,
                tokenizer=loaded.tokenizer,
                image_processor=loaded.image_processor,
                flow_prompt=loaded.flow_prompt,
                allowed_work_variants=config.supported_ops,
                transfer_backends=config.data_plane.backends,
                publication_backends=config.data_plane.publication_backends,
                worker_id=config.worker_id,
                queue_depth=config.ipc.queue_depth,
                completion_payload_bytes=config.ipc.max_payload_bytes,
                acknowledgment_slot=config.ipc.acknowledgment_slot,
                product_consumers=config.ipc.product_consumers,
                components=config.components,
                process_groups=distributed,
            )
        except BaseException as error:
            try:
                distributed.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Resource cleanup also failed: {cleanup_error!r}"
                )

            raise

    def __init__(
        self,
        model: nn.Module,
        *,
        worker_config: WorkerConfig,
        sampling_group: Communicator | None,
        tokenizer: Any | None,
        allowed_work_variants: frozenset[CallKind],
        queue_depth: int,
        completion_payload_bytes: int,
        acknowledgment_slot: int = 0,
        product_consumers: tuple[int, ...] = (),
        attention: str | None = None,
        transfer_backends: tuple[str, ...] = ("local",),
        publication_backends: tuple[str, ...] = ("local",),
        worker_id: str = "worker",
        image_processor: ImageProcessor | None = None,
        flow_prompt: FlowPrompt | None = None,
        components: tuple[tuple[str, ComponentConfig], ...] = (),
        process_groups: ProcessGroups | None = None,
        bindings: Mapping[str, ModelEntry] | None = None,
    ) -> None:
        """Allocate execution resources for an already-loaded model.

        Resolve unspecified attention from worker_config. Construction
        performs no warmup or IPC I/O, and rolls back partial resource
        allocations on error. The caller closes the worker or uses its
        owning context after success.
        """
        self._closed = False
        self._run_started = False
        self._warmed_up = False

        self.ipc_endpoint: WorkerIpcEndpoint | None = None
        self.profiler: WorkerProfiler | None = None

        startup = ExitStack()

        try:
            if not isinstance(model, nn.Module):
                raise unsupported_setup(
                    "worker model has no supported execution surface"
                )

            if not isinstance(worker_config, WorkerConfig):
                raise unsupported_setup("model worker requires a WorkerConfig")

            if queue_depth <= 0:
                raise unsupported_setup(
                    "worker pipeline depth must be positive"
                )

            if (
                not publication_backends
                or len(set(publication_backends)) != len(publication_backends)
                or not set(publication_backends).issubset(transfer_backends)
            ):
                raise unsupported_setup(
                    "publication backends must be unique bound transports"
                )

            if (
                "cuda_vmm" in transfer_backends
                and torch.device(worker_config.device).type != "cuda"
            ):
                raise unsupported_setup(
                    "CUDA VMM requires a CUDA worker device"
                )

            self.model = model
            self.process_groups = process_groups
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
                bindings=bindings,
                attention=attention,
                image_processor=image_processor,
                flow_prompt=flow_prompt,
            )
            self.runner = runner
            startup.callback(runner.close)

            attention = runner.attention
            self.attention = attention

            # Measure the remaining grant after the runner binds persistent
            # model inputs and workspaces. Sizing earlier would treat
            # occupied memory as free. Request capacity is the only
            # configuration resolved after binding.
            endpoint = WorkerEndpoint.local(worker_id, int(worker_config.rank))

            worker_config = resolve_request_capacity(
                model,
                worker_config,
                state_buffers=runner.state_buffers,
                queue_depth=queue_depth,
                bindings=bindings,
                capacity_group=(
                    None
                    if process_groups is None
                    else process_groups.process_group
                ),
            )
            self.worker_config = worker_config
            runner.worker_config = worker_config

            layout = build_worker_layout(
                model,
                worker_config,
                image_processor=image_processor,
                state_buffers=runner.state_buffers,
                endpoint=endpoint,
                bindings=bindings,
                queue_depth=int(queue_depth),
                completion_payload_bytes=int(completion_payload_bytes),
                allowed_work_variants=allowed_work_variants,
                transfer_backends=transfer_backends,
                components=components,
                attention_identity=f"{type(attention).__module__}.{type(attention).__qualname__}",
                # The scheduler's page indices are shared across all
                # resident layer and head regions, including stages with
                # different memory grants.
                capacity_group=(
                    process_groups.process_group
                    if process_groups is not None
                    else sampling_group
                ),
            )

            # Auxiliary devices have separate grants; primary-device
            # reservations were already included in the layout's capacity
            # calculation.
            for device, fixed_bytes in layout.fixed_device_bytes:
                if (
                    device == worker_config.device
                    or canonical_device(device).type != "cuda"
                ):
                    continue

                available, _free = device_memory_budget(
                    device, worker_config.kv_memory_fraction
                )
                reserved = fixed_bytes + graph_memory_budget_bytes(
                    device_total_bytes(device)
                )

                if reserved > available:
                    raise unsupported_setup(
                        f"runtime storage on {device} requires {reserved} "
                        "bytes, but its static memory grant is "
                        f"{available} bytes"
                    )

            self._layout = layout
            info = layout.info
            arena = layout.arena

            text = capability(model, CausalLM)
            owns_kv = text is not None
            cache = None if text is None else text.cache_config

            self.kv_cache = None
            self.block_tables = None
            max_blocks_per_row = 0

            # KV pages and request-to-token tables share group dimensions;
            # bind them to the model only after attention compatibility has
            # been established.
            if cache is not None:
                assert text is not None
                kv_cache = info.kv_cache
                if kv_cache is None:
                    raise unsupported_setup(
                        "KV model worker has no KV-cache configuration"
                    )

                max_blocks_per_row = max(
                    1,
                    ceil_div(
                        worker_config.max_sequence_tokens,
                        int(worker_config.block_size),
                    ),
                )

                # Cache groups occupy consecutive ranges in the shared page
                # pool.
                group_ranges: list[tuple[int, int]] = []
                group_offset = 0
                for group in kv_cache.groups:
                    group_ranges.append((group_offset, int(group.num_blocks)))
                    group_offset += int(group.num_blocks)

                encoded = kv_cache.dtype == "float8_e4m3fn"
                self.kv_cache = CacheManager(
                    PrefixCache(
                        cache,
                        num_blocks=kv_cache.num_blocks,
                        block_size=kv_cache.block_size,
                        device=worker_config.device,
                        dtype=None
                        if encoded
                        else getattr(torch, kv_cache.dtype),
                        quantization={
                            name: Quantizer("fp8", axis=0)
                            for name in cache.layers
                        }
                        if encoded
                        else None,
                    ),
                    info=kv_cache,
                    group_ranges=tuple(group_ranges) if group_ranges else None,
                    import_capacity=int(info.max_unresolved_ops),
                    request_pool_size=int(info.request_slots),
                    max_blocks_per_request=max_blocks_per_row,
                    staging_depth=int(queue_depth),
                )
                startup.callback(self.kv_cache.close)

                self.block_tables = self.kv_cache.block_tables

            # Admission, lineage, and persistent tensors share one slot owner.
            self.requests = RequestPool(
                int(info.request_slots),
                state_buffers=runner.state_buffers or None,
                device=worker_config.device,
            )

            torch_dtype = getattr(
                torch,
                str(worker_config.model_dtype).removeprefix("torch."),
                None,
            )
            if not isinstance(torch_dtype, torch.dtype):
                raise unsupported_setup(
                    f"unsupported model dtype {worker_config.model_dtype!r}"
                )

            if self.block_tables is not None:
                assert text is not None
                self.decode_state = DecodeState(
                    request_pool_size=int(info.request_slots),
                    vocab_size=text.backbone.vocab_size,
                    continuation_width=1,
                    device=worker_config.device,
                    logits_dtype=torch_dtype,
                    valid_cache_lengths=self.block_tables.verified_lengths,
                )
            else:
                self.decode_state = None

            # Generation state has its own page pool and may live on
            # another device.
            flow = image_builder(model)
            latent_dtype = getattr(
                torch, str(layout.latent_dtype).removeprefix("torch."), None
            )
            if flow is not None and not isinstance(latent_dtype, torch.dtype):
                raise unsupported_setup(
                    f"unsupported latent dtype {layout.latent_dtype!r}"
                )

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
                    device=worker_config.generation_device
                    or worker_config.device,
                )
                startup.callback(self.latent_pool.close)

            if (
                self.latent_pool is not None
                and self.latent_pool.persistent_bytes != arena.latent_pool_bytes
            ):
                raise RuntimeError(
                    "latent pool allocation disagrees with its exact "
                    "capacity plan"
                )

            # These bounded stores own all asynchronous products, copies,
            # CPU tasks, and transfer lifetimes exposed by an in-flight
            # pipeline.
            owner_devices = tuple(
                dict.fromkeys(
                    (
                        worker_config.device,
                        worker_config.generation_device or worker_config.device,
                    )
                )
            )

            self.device_events = EventPool()
            startup.callback(self.device_events.close)

            self.output_pool = OutputPool(
                capacity=int(queue_depth) * int(info.max_batch_ops),
                max_words=int(info.max_batch_ops)
                * (4 + (int(completion_payload_bytes) + 3) // 4),
                event_pool=self.device_events,
            )
            startup.callback(self.output_pool.close)

            self.buffer_pool = BufferPool(
                byte_capacity=int(layout.physical_buffer_pool_bytes),
                devices=owner_devices,
                compact=layout.physical_buffer_pool_bytes
                < info.buffer_pool_bytes,
            )
            startup.callback(self.buffer_pool.close)

            self.tensor_store = TensorStore(
                capacity=arena.tensor_store,
                byte_capacity=arena.device_product_bytes,
                entry_capacity=int(layout.encoder_cache_entries),
                max_entry_bytes=max(
                    1,
                    int(layout.max_latent_feature_bytes),
                    int(layout.max_vision_feature_bytes),
                ),
                devices=owner_devices,
                request_capacity=int(info.request_slots),
                relay_depth=int(info.max_unresolved_ops) + 1,
                buffer_pool=self.buffer_pool,
                event_pool=self.device_events,
            )
            startup.callback(self.tensor_store.close)

            self.host_tasks = HostLane(
                max_inflight=int(arena.host_lane_inflight),
                workers=min(4, int(arena.host_lane_inflight)),
            )
            startup.callback(self.host_tasks.close)

            transfer_byte_capacity = int(arena.transfer_bytes)
            if (
                capability(model, VideoPostprocessor) is not None
                or runner.state_buffers
            ):
                # Each live request tensor reserves one credit per
                # publication representation and one read credit on every
                # possible remote rank. These credits bound ownership
                # lifetimes; they allocate no storage.
                transfer_byte_capacity *= len(publication_backends) + max(
                    0, int(worker_config.world_size) - 1
                )

            self.transports = make_transports(
                transfer_backends,
                source=endpoint,
                byte_capacity=transfer_byte_capacity,
                ticket_capacity=arena.transfer_tickets,
                event_pool=self.device_events,
                consumers=product_consumers,
                acknowledgment_slot=acknowledgment_slot,
            )
            for transport in self.transports.values():
                startup.callback(transport.close)

            self.publication_transports = {
                name: self.transports[name] for name in publication_backends
            }

            if owns_kv:
                assert (
                    layout.input_config is not None
                    and self.decode_state is not None
                )

                runner.configure_inputs(
                    input_config=layout.input_config,
                    kv_cache=self.kv_cache,
                    latent_pool=self.latent_pool,
                    decode_predicates=self.decode_state.predicates,
                    max_calls=int(info.max_batch_ops),
                    request_slots=int(info.request_slots),
                    max_tokens=int(info.max_batch_tokens),
                    latent_capacity_units=int(info.latent_capacity_units),
                    decode_context_blocks=decode_context_blocks(
                        model, worker_config, self.kv_cache
                    ),
                    variants=frozenset(info.supported_ops),
                    max_inflight=int(queue_depth),
                )

            # Call handlers borrow the resources owned by this rank.
            from ..execution.video import create_media_resources

            self.media_mux, self.media_buffers = (
                create_media_resources(
                    self.runner,
                    rank=worker_config.rank,
                    worker_info=info,
                    state_slots=info.request_slots,
                    unresolved_window=info.max_unresolved_ops,
                )
                if capability(model, VideoPostprocessor) is not None
                else (None, None)
            )
            if self.media_mux is not None:
                startup.callback(self.media_mux.close)

        except BaseException as error:
            try:
                startup.close()
            except BaseException as cleanup_error:
                error.add_note(
                    f"Resource cleanup also failed: {cleanup_error!r}"
                )

            raise

        # All registered resources now belong to this Worker and its close
        # method.
        startup.pop_all()
        self._init_request_scheduling()
        self._init_run_tracking()

    def bind(self, endpoint: WorkerIpcEndpoint) -> Self:
        """Borrow an open endpoint and install service state.

        Installs bounded service queues and completion wakes. The caller
        owns the endpoint and must keep it open until the Worker scope
        exits or close() finishes. Binding is exclusive and cannot be
        replaced. The surrounding Worker scope owns cleanup on success or
        failure.
        """
        self._require_open()

        if self.ipc_endpoint is not None:
            raise RuntimeError("worker already has a bound IPC endpoint")

        if endpoint is None or endpoint.closed:
            raise ValueError("worker binding requires an open IPC endpoint")

        self.profiler = WorkerProfiler.from_env()

        # Publish the binding only after all service state exists. Register
        # callbacks last so completion notifications always find that state.
        self.ipc_endpoint = endpoint
        self.set_completion_wake(endpoint.wake, endpoint.wake_on_stream)

        return self

    def _init_request_scheduling(self) -> None:
        """Initialize admission scheduling state.

        Covers admission limits, dependency chains, and ready-request queues.
        """
        self.queue_depth = max(1, int(self.info.queue_depth))
        self._next_sequence = 1
        self._transport_occupancy = 0
        self._admission_closed = False
        self._shutdown_response: dict[str, Any] | None = None

        # A rank executes batches in channel order. Requests launch from the
        # head of this queue in the order the channel delivered them, which is
        # the order the engine dispatched them in.
        self._pending_requests: dict[int, ServiceRequest] = {}
        self._ready_requests: deque[ServiceRequest] = deque()

        # A component whose calls are collective needs its ranks inside the
        # same collective, not merely issuing collectives in the same order. A
        # rank may hold no media unit of a batch and skip it, and depth lets a
        # rank start the next batch while a peer is still in this one, so one
        # rank can reach a denoiser's capture_required all-reduce that its
        # peers have not. Such a rank therefore holds one batch in flight.
        self._launched_submissions = 0
        self._collective_component = any(
            group.size > 1
            for entry in self.runner.bindings.values()
            if entry.owns
            for group in entry.groups
        ) or (self.sampling_group is not None and self.sampling_group.size > 1)

    def _init_run_tracking(self) -> None:
        """Initialize batch tracking state.

        Keeps bounded in-flight work and a constant-size admission
        high-water mark.
        """
        # Batch IDs increase in transport submission order and are refused
        # when they do not advance. Completion order does not affect admission.
        self._last_batch_id = -1
        self.inflight: dict[int, BatchState] = {}
        self._batch_submissions: dict[int, ServiceRequest] = {}
        self._preparation_ready: SimpleQueue[BatchState] = SimpleQueue()
        self._executing_batches: deque[BatchState] = deque()

        self.pending_responses: deque[PendingResponse] = deque()
        self._waiting_responses: dict[int, PendingResponse] = {}

    def run(self) -> None:
        """Warm up and serve synchronously once.

        Runs within the owner's resource scope. No service requests are
        consumed before warmup completes. Returning or raising leaves
        resource cleanup to the surrounding Worker context.
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
        """Advance accepted work until the service drains.

        Delivers responses as they become ready.
        """
        endpoint = self.ipc_endpoint
        assert endpoint is not None

        # Suspend cyclic collection only during serving, restoring the caller's
        # setting before either normal shutdown or exceptional resource cleanup.
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            while True:
                # Give accepted work and ready responses priority over
                # admission.
                self.device_events.reap()

                if self._advance_executing_batches():
                    continue

                if self._send_one_ready_response():
                    continue

                if self._launch_one_ready_request():
                    continue

                # Close is acknowledged after execution and claimed
                # responses drain.
                if self._admission_closed:
                    if self._service_drained():
                        if self._shutdown_response is not None:
                            self._transport_respond(self._shutdown_response)

                        return

                if (
                    not self._admission_closed
                    and self._transport_occupancy < self.queue_depth
                ):
                    request = endpoint.try_recv()
                    if request is not None:
                        self._accept(request)
                        continue

                # Outstanding work needs completion wakes as well as IPC
                # arrivals.
                if (
                    self._pending_requests
                    or self.pending_responses
                    or self._waiting_responses
                    or self.inflight
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
            return messages.response(
                ResponseKind.INFO, info=self.info.to_mapping()
            )
        if kind is RequestKind.CLOSE:
            return messages.response(ResponseKind.OK)
        raise invalid_descriptor(
            f"unsupported administrative request {kind.value!r}"
        )

    def _boxed_error(
        self, request: Mapping[str, Any], error: BaseException
    ) -> dict[str, Any]:
        """Classify an exception and record it.

        Also encodes the protocol error response.
        """
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
        """Assign transport order and decode the request.

        Also links per-request dependencies.
        """
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
                self._shutdown_response = messages.with_message_id(
                    self._dispatch(request), request
                )
                return sequence

            batch: Batch | None = None
            if kind is RequestKind.SUBMIT:
                raw_batch = messages.required(request, "batch", kind)
                raw_batch_id = (
                    int(raw_batch.batch_id)
                    if isinstance(raw_batch, Batch)
                    else int(raw_batch.get("batch_id", -1))
                    if isinstance(raw_batch, Mapping)
                    else -1
                )
                with profile_range(
                    worker_range_name(
                        "batch_decode",
                        batch_id=raw_batch_id,
                        rank=self.info.endpoint.rank,
                    )
                ):
                    batch = (
                        raw_batch
                        if isinstance(raw_batch, Batch)
                        else Batch.from_mapping(raw_batch)
                    )

                if batch.batch_id <= self._last_batch_id:
                    raise invalid_descriptor(
                        f"batch id {batch.batch_id} must exceed previously "
                        f"submitted id {self._last_batch_id}"
                    )
                self._last_batch_id = batch.batch_id

                # Product release is independent of request-state transitions.
                # An earlier numerical batch may need this retired allocation,
                # so revocation cannot wait behind that batch's execution FIFO.
                # Existing readers retain storage; the command's ordinary
                # terminal acknowledgement still waits for their completion.
                freed = tuple(
                    command.buffer
                    for command in batch.commands
                    if isinstance(command, Free)
                )
                if freed:
                    self.release_buffers(freed)

                requests = messages.batch_requests(batch)

            pending = ServiceRequest(
                sequence=sequence,
                request=request,
                requests=requests,
                kind=kind,
                batch=batch,
            )
            self._pending_requests[sequence] = pending
            self._ready_requests.append(pending)
        except BaseException as error:
            # Parse and admission failures enter the same ordered response queue
            # as successfully launched requests.
            self.pending_responses.append(
                PendingResponse(
                    sequence, requests, self._boxed_error(request, error)
                )
            )
        return sequence

    def _release(self, pending: ServiceRequest) -> None:
        """Retire one launched request from the admitted set."""
        if pending.released:
            return
        pending.released = True
        if pending.kind is RequestKind.SUBMIT:
            self._launched_submissions -= 1
        self._pending_requests.pop(pending.sequence, None)

    def _launch_one_ready_request(self) -> bool:
        """Select and launch one dependency-ready request.

        Handles administrative and execution requests.
        """
        if not self._ready_requests:
            return False

        # Channel order is execution order, so only the head launches. A batch
        # that cannot start yet holds the queue rather than letting a later one
        # overtake it, which is what lets cooperative ranks launch their
        # collectives in the order the engine dispatched them.
        head = self._ready_requests[0]
        if head.kind is RequestKind.SUBMIT and (
            len(self.inflight) >= self.queue_depth
            or (self._collective_component and self._launched_submissions)
        ):
            return False

        pending = self._ready_requests.popleft()

        try:
            if pending.kind is RequestKind.SUBMIT:
                self._launched_submissions += 1
                self._launch_execute(pending)
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
                messages.with_message_id(response, pending.request),
            )
        )

    def _launch_execute(self, pending: ServiceRequest) -> None:
        """Start one accepted batch and queue its result."""
        if pending.batch is None:
            raise RuntimeError("accepted execute request lost its batch")
        batch = BatchState(pending.batch)
        self._batch_submissions[batch.batch_id] = pending
        self.inflight[batch.batch_id] = batch

        self._start_execution(batch)
        self._queue_result(pending, batch)

    def _start_execution(self, batch: BatchState) -> None:
        """Submit physical inputs for the batch.

        Directly launches the batch when the inputs are ready.
        """
        try:
            unsupported = tuple(
                call.kind
                for call in batch.batch.calls
                if not self.supports_computation(call.kind)
            )
            if unsupported:
                names = sorted({value.value for value in unsupported})
                raise invalid_descriptor(
                    "execution batch contains work variants unsupported by "
                    f"this worker: {names!r}"
                )

            self._prepare_execution(batch)

            if self._advance_execution(batch):
                if not batch.complete:
                    self._executing_batches.append(batch)
            else:
                batch.on_dependencies_ready(
                    partial(self._preparation_completed, batch)
                )
        except BaseException as error:
            self._fail_run(batch, error)

    def _advance_execution(self, batch: BatchState) -> bool:
        """Execute prepared inputs on the worker thread.

        Never executes from a notification callback.
        """
        if batch.complete or batch.launched:
            return True
        try:
            self.advance_inputs(batch)
            if not batch.inputs_ready():
                return False
            self._execute_prepared(batch)
            self._notify_batch(batch)
        except BaseException as error:
            self._fail_run(batch, error)
        return True

    def _advance_batch(self, batch: BatchState) -> None:
        """Materialize complete groups.

        Also advances physical command retirement.
        """
        if batch.complete or not batch.launched:
            return
        try:
            # CPU work is submitted by the Worker, never by a readiness query.
            for output in batch.outputs:
                if isinstance(output, PendingOutput) and output.value is None:
                    for task in output.completion_tasks:
                        task.submit_if_ready()

            for group, indexes in batch.output_groups.items():
                if group in batch.completed_groups:
                    continue
                outputs = tuple(batch.outputs[index] for index in indexes)
                if any(value is None for value in outputs):
                    raise RuntimeError(
                        "launched batch is missing an call output"
                    )
                if any(
                    isinstance(value, PendingOutput) and not value.ready()
                    for value in outputs
                ):
                    continue
                pending = tuple(
                    value
                    for value in outputs
                    if isinstance(value, PendingOutput)
                )
                values = tuple(
                    value.materialize()
                    if isinstance(value, PendingOutput)
                    else value
                    for value in outputs
                )
                self.requests.apply_outputs(pending)
                # Acceptance remains visible after pending rows become wire
                # values.
                if len(pending) == len(outputs):
                    batch.accepted_groups.add(group)
                for index, value in zip(indexes, values, strict=True):
                    batch.outputs[index] = value
                batch.completed_groups.add(group)

            if len(batch.completed_groups) == len(batch.output_groups):
                batch.complete = self._advance_retirement(batch)
            self._notify_batch(batch)
        except BaseException as error:
            self._fail_run(batch, error, context="completion materialization")

    def _notify_batch(self, batch: BatchState) -> None:
        """Retire the batch's submission and wake its waiting IPC response."""
        submission = self._batch_submissions.pop(batch.batch_id, None)
        if submission is not None:
            self._release(submission)
        self._batch_ready(batch)

    def _fail_run(
        self,
        batch: BatchState,
        error: BaseException,
        *,
        context: str = "execute",
    ) -> None:
        """Record a classified failure and close the batch.

        Also wakes its waiting responses.
        """
        if batch.complete:
            return
        batch.error = (
            error
            if isinstance(error, WorkerError)
            else classify(error, context=context)
        )
        try:
            self._close_batch(batch)
        except BaseException as cleanup_error:
            batch.error.add_note(f"batch cleanup failed: {cleanup_error!r}")
        batch.complete = True
        self._notify_batch(batch)

    def _queue_result(self, request: ServiceRequest, batch: BatchState) -> None:
        """Queue the response to one Submit.

        The response is queued once the batch's single result is ready.
        """
        pending = PendingResponse(
            request.sequence,
            request.requests | batch.request_ids,
            messages.with_message_id(
                messages.response(ResponseKind.RESULT), request.request
            ),
            batch,
        )
        self._advance_batch(batch)
        if self._pending_ready(pending):
            self.pending_responses.append(pending)
        else:
            self._waiting_responses[batch.batch_id] = pending

    def _batch_ready(self, batch: BatchState) -> None:
        """Wake the response waiting for this batch's result."""
        pending = self._waiting_responses.pop(batch.batch_id, None)
        if pending is not None:
            if self._pending_ready(pending):
                self.pending_responses.append(pending)
            else:
                self._waiting_responses[batch.batch_id] = pending

    def _preparation_completed(self, batch: BatchState) -> None:
        """Enqueue readiness before waking the IPC loop that consumes it."""
        self._preparation_ready.put(batch)
        if self.ipc_endpoint is not None:
            self.ipc_endpoint.wake()

    def _advance_executing_batches(self) -> bool:
        """Launch preparation-ready work.

        Also advances executing runs in launch order.
        """
        advanced = False
        if not self._preparation_ready.empty():
            batch = self._preparation_ready.get_nowait()
            if not batch.complete:
                launched = self._advance_execution(batch)
                if not batch.complete:
                    if launched:
                        self._executing_batches.append(batch)
                    else:
                        batch.on_dependencies_ready(
                            partial(self._preparation_completed, batch)
                        )
            advanced = True

        # Query every launched batch: one pending host read or retirement
        # must not hide an independent completion behind it.
        for _ in range(len(self._executing_batches)):
            batch = self._executing_batches.popleft()
            before = len(batch.completed_groups)
            self._advance_batch(batch)
            advanced |= batch.complete or len(batch.completed_groups) != before
            if not batch.complete:
                self._executing_batches.append(batch)
        return advanced

    def _pending_ready(self, pending: PendingResponse) -> bool:
        """Query completion without blocking the service thread."""
        batch = pending.batch
        if batch is None:
            return True
        with profile_range(
            worker_range_name(
                "completion",
                batch_id=batch.batch_id,
                rank=self.info.endpoint.rank,
            )
        ):
            return batch.ready()

    def _send_one_ready_response(self) -> bool:
        """Send the oldest transport-ready response if one exists."""
        if not self.pending_responses:
            return False
        self._send_pending(self.pending_responses.popleft())
        return True

    def _send_pending(self, pending: PendingResponse) -> None:
        """Serialize and send one ready response.

        Transport sequence order is preserved.
        """
        response = dict(pending.response)
        batch = pending.batch
        batch_id = batch.batch_id if batch is not None else None

        if batch is not None:
            if batch.error is not None:
                response = messages.error_response(
                    batch.take_error(), pending.response
                )
            else:
                response["result"] = batch.take_output()
            del self.inflight[batch.batch_id]
            self._close_batch(batch)

        fatal = bool(response.get("fatal"))
        with profile_range(
            worker_range_name(
                "finalize_response",
                batch_id=batch_id,
                rank=self.info.endpoint.rank,
            )
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

    def _service_drained(self) -> bool:
        """Return whether the service has drained.

        Covers all accepted work and claimed responses.
        """
        return (
            not self._pending_requests
            and not self.pending_responses
            and not self._waiting_responses
            and not self.inflight
        )

    @property
    def info(self) -> WorkerInfo:
        """Expose immutable worker info.

        Params, capacity, and model metadata advertised to the scheduler.
        """
        return self._layout.info

    def _require_open(self) -> None:
        """Raise when this worker's resource scope has already been closed."""
        if self._closed:
            raise RuntimeError("worker is closed and cannot be reused")

    def submit(
        self, batch: Batch, *, propagate_errors: bool = False
    ) -> BatchState:
        """Accept one batch and submit available work.

        Retains the batch's asynchronous state. Call advance to progress
        pending inputs, CPU work, and retirement, then poll to consume the
        batch's result. A batch identity remains owned until its result is
        consumed or the Worker closes.
        """
        self._require_open()
        if batch.batch_id in self.inflight:
            raise invalid_descriptor("batch ID already has an in-flight batch")

        state = BatchState(batch, propagate_errors=propagate_errors)
        self.inflight[state.batch_id] = state

        try:
            self._prepare_execution(state)
            self._advance_execution(state)
            self._advance_batch(state)
        except BaseException:
            self.inflight.pop(state.batch_id, None)
            self._close_batch(state)
            raise

        if state.error is not None and propagate_errors:
            error = state.error
            self.inflight.pop(state.batch_id, None)
            self._close_batch(state)
            raise error
        return state

    def advance(self) -> None:
        """Progress pending batch state on the caller thread.

        Covers physical dependencies and completed outputs.
        """
        self._require_open()
        for state in tuple(self.inflight.values()):
            self._advance_execution(state)
            self._advance_batch(state)

    def poll(self, state: BatchState) -> BatchOutput | None:
        """Consume the batch's result without launching computation."""
        self._require_open()
        if self.inflight.get(state.batch_id) is not state:
            raise invalid_descriptor(
                "poll names a batch no longer owned by this Worker"
            )
        if not state.ready():
            return None

        if state.error is not None:
            error = state.take_error()
            del self.inflight[state.batch_id]
            self._close_batch(state)
            raise error

        output = state.take_output()
        del self.inflight[state.batch_id]
        self._close_batch(state)
        return output

    def _execute_batch(self, state: BatchState) -> None:
        """Execute prepared numerical work.

        Retains the work's physical retirement facts.
        """
        batch = state.batch
        # Cooperative ranks launch computation in the same order. Preparation
        # and host completion may overlap; neither retains old batch identities.
        if self.worker_config.world_size > 1 and batch.calls:
            if batch.collective_seq <= self._last_collective_seq:
                raise invalid_descriptor(
                    f"collective sequence does not advance: batch "
                    f"{batch.batch_id} carries {batch.collective_seq} after "
                    f"{self._last_collective_seq}"
                )
            self._last_collective_seq = batch.collective_seq

        execute_batch(
            state,
            propagate_errors=state.propagate_errors,
            kv_cache=self.kv_cache,
            host_tasks=self.host_tasks,
            tensor_store=self.tensor_store,
            worker_info=self.info,
            latent_pool=self.latent_pool,
            media_mux=self.media_mux,
            media_buffers=self.media_buffers,
            output_pool=self.output_pool,
            publication_transports=self.publication_transports,
            request_tables=self.block_tables,
            request_pool=self.requests,
            model_runner=self.runner,
            decode_state=self.decode_state,
            sampling_group=self.sampling_group,
            tokenizer=self.tokenizer,
            transfer_backends=self.transports,
            config=self.worker_config,
        )

        consumed = _input_producers(batch)
        self._release_predecessors(
            tuple(
                (call.request_key, call.predecessor)
                for call in batch.calls
                if call.predecessor is not None
                and call.predecessor.batch_id > 0
                and (call.request_key, call.predecessor) in consumed
            )
        )

        self.tensor_store.release_buffers(
            tuple(
                predicate.buffer_id
                for call in batch.calls
                if (predicate := call.predicate) is not None
                and (
                    call.predecessor is None
                    or predicate.producer_call_id != call.predecessor
                )
            )
        )
        state.launched = True
        self._retire_commands(state)

    def supports_computation(self, kind: CallKind) -> bool:
        """Return whether this worker can execute this computation."""
        return kind in self.info.supported_ops

    def _prepare_execution(self, state: BatchState) -> None:
        """Validate the batch and apply its commands.

        Also prepares the batch's physical inputs.
        """
        batch = state.batch
        validate_batch(
            batch,
            worker_info=self.info,
            model_runner=self.runner,
            config=self.worker_config,
        )

        for command in batch.commands:
            slots = self.requests.apply_commands((command,))
            if slots and self.decode_state is not None:
                self.decode_state.reset(slots)

        consumed = _input_producers(batch)
        self._release_predecessors(
            tuple(
                (call.request_key, call.predecessor)
                for call in batch.calls
                if call.predecessor is not None
                and call.predecessor.batch_id > 0
                and (call.request_key, call.predecessor) not in consumed
            )
        )
        self._release_commands(batch)
        prepare_batch(
            state,
            kv_cache=self.kv_cache,
            latent_pool=self.latent_pool,
            request_tables=self.block_tables,
            request_pool=self.requests,
        )
        self.advance_inputs(state)

    def advance_inputs(self, state: BatchState) -> None:
        """Submit ready physical reads and predicate copies.

        Does not launch a model.
        """
        if state.inputs_closed:
            return
        if state.inputs_submitted:
            capture_predicates(state, self.tensor_store)
            return
        # A destination remains unavailable until the previous physical reader
        # retires. Dependencies with no import still gate model execution.
        if (state.input_products or state.kv_inputs) and not all(
            dependency.done() for dependency in state.storage_dependencies
        ):
            return
        if state.input_products or state.kv_inputs:
            for dependency in state.storage_dependencies:
                dependency.result()

        prepare_inputs(
            state,
            kv_cache=self.kv_cache,
            tensor_store=self.tensor_store,
            latent_pool=self.latent_pool,
            output_pool=self.output_pool,
            request_tables=self.block_tables,
            request_pool=self.requests,
            model_runner=self.runner,
            transfer_backends=self.transports,
            config=self.worker_config,
        )
        state.inputs_submitted = True
        capture_predicates(state, self.tensor_store)

    def _retire_commands(self, state: BatchState) -> None:
        """Submit release work for the batch's commands.

        Retains the events and futures required by its acknowledgement.
        """
        batch = state.batch
        closed = frozenset(
            command.request_key
            for command in batch.commands
            if isinstance(command, Finish)
        )
        local_closed = frozenset(
            key
            for key in closed
            if (row := self.requests.peek(key.request_id)) is not None
            and row.request_key == key
        )
        freed = frozenset(
            command.buffer
            for command in batch.commands
            if isinstance(command, Free)
        )

        if not closed and not freed:
            state.retirement_cleaned = True
            return

        retained = (
            frozenset(
                buffer
                for command in batch.commands
                if isinstance(command, Finish)
                for buffer in command.retained_buffers
            )
            - freed
        )
        self.tensor_store.release_requests(closed, retained=retained)

        if self.latent_pool is not None:
            self.latent_pool.cancel_imports(tuple(closed))

        stores = tuple(
            store
            for store in (self.tensor_store, self.kv_cache, self.latent_pool)
            if store is not None
        )
        selected = tuple(
            buffer
            for store in stores
            for buffer in retiring_exports(
                store.exports, buffers=freed, requests=closed, retained=retained
            )
        )
        releases = tuple(
            future
            for store in stores
            for future in release_exports(
                store.exports, store.export_releases, selected
            )
        )

        if self.latent_pool is not None:
            self.latent_pool.release_buffers(selected)
        if self.kv_cache is not None:
            self.kv_cache.imports.cancel_requests(closed, retained=retained)
            self.kv_cache.release_buffers(selected)

        wake = self._completion_wake
        if wake is not None:
            for future in releases:
                future.add_done_callback(lambda _future: wake())

        state.retirement_requests = closed
        state.retirement_local_requests = local_closed
        state.retirement_buffers = freed
        state.retained_buffers = retained
        state.retirement_exports = selected
        state.retirement_releases = releases
        # Finish includes request-state writes issued after output capture.
        state.retirement_events = (
            self._record_retirement_events() if closed else ()
        )

    def _record_retirement_events(self) -> tuple[torch.cuda.Event, ...]:
        """Record tracked completion events.

        One event per CUDA device in the buffer pool.
        """
        events = []
        for device in self.buffer_pool.devices:
            if device.type != "cuda":
                continue
            event = self.device_events.acquire(device)
            self.device_events.retain(event, device)
            self.device_events.record(event, device)
            self.device_events.schedule_completion_wake(device, event)
            events.append(event)
        return tuple(events)

    def _advance_retirement(self, state: BatchState) -> bool:
        """Reset retired request slots.

        Reset happens only after every physical reader has finished.
        """
        self.device_events.reap()
        # A device product is held until its consumers acknowledge it. They do
        # so by writing into the chunk they read, which arrives with no local
        # notification, so the producing rank looks for it here.
        for transport in self.transports.values():
            transport.reap()
        if not all(event.query() for event in state.retirement_events):
            return False

        for event in state.retirement_events:
            self.device_events.release(event)
        state.retirement_events = ()

        if state.retirement_cleaned:
            return True

        closed = state.retirement_requests
        freed = state.retirement_buffers
        retained = state.retained_buffers

        if any(
            not self.requests.retirement_ready(key)
            for key in state.retirement_local_requests
        ):
            return False
        if not self.tensor_store.retirement_ready(
            buffers=freed, requests=closed, retained=retained
        ):
            return False
        if (
            self.latent_pool is not None
            and not self.latent_pool.retirement_ready(tuple(closed))
        ):
            return False
        if self.kv_cache is not None and not self.kv_cache.retirement_ready(
            buffers=freed, requests=closed, retained=retained
        ):
            return False
        for future in state.retirement_releases:
            if not future.done():
                return False
            future.result()

        for store in (self.tensor_store, self.kv_cache, self.latent_pool):
            if store is not None:
                forget_exports(
                    store.exports,
                    store.export_releases,
                    state.retirement_exports,
                )
        for key in state.retirement_local_requests:
            self.retire_request(key, retained=retained)

        # Slot reset itself submits writes; their completion permits
        # address reuse.
        state.retirement_events = (
            self._record_retirement_events() if closed else ()
        )
        state.retirement_cleaned = True
        return not state.retirement_events

    def _close_batch(self, state: BatchState) -> None:
        """Cancel unresolved acceptance.

        Real CPU and GPU readers retain storage meanwhile.
        """
        pending = tuple(
            output
            for output in state.outputs
            if isinstance(output, PendingOutput)
        )
        self.requests.cancel_outputs(pending)
        if state.retirement_events:
            self.device_events.defer_release(state.retirement_events, state)
            state.retirement_events = ()
        state.close(self.tensor_store, self.latent_pool, self.kv_cache)

    def _execute_prepared(self, state: BatchState) -> None:
        """Directly execute physical inputs once.

        Releases their preparation leases.
        """
        self._require_open()
        if state.inputs_closed:
            raise RuntimeError("batch inputs have already been consumed")

        self.advance_inputs(state)
        if not state.inputs_ready():
            raise RuntimeError("batch was observed before dependency readiness")

        try:
            for dependency in state.storage_dependencies:
                dependency.result()
            self._execute_batch(state)
        except BaseException as error:
            try:
                state.close_inputs(
                    self.tensor_store, self.latent_pool, self.kv_cache
                )
            except BaseException as cleanup_error:
                error.add_note(f"batch input cleanup failed: {cleanup_error!r}")
            raise
        state.close_inputs(self.tensor_store, self.latent_pool, self.kv_cache)

    def warmup(self) -> None:
        """Prepare numerical execution once without serving requests.

        Successful warmup is retained for run(). The owner must exit the Worker
        scope if startup fails, just as for a binding or service failure.
        """
        self._require_open()
        if self._warmed_up:
            return

        self.runner.warmup(self.requests.tensor_slots)
        self.runner.capture(tokenizer=self.tokenizer, latents=self.latent_pool)
        warmup_requests(self)
        self.runner.complete_startup()

        # Startup scenarios must release their requests before service
        # admission.
        if self.requests.request_ids():
            raise RuntimeError("startup completed with resident requests")
        self._last_collective_seq = -1
        check_startup_memory(
            self.worker_config,
            self._layout.arena.device_product_bytes,
            self.tensor_store,
        )
        self._warmed_up = True

    def release_buffers(self, buffers: Sequence[BufferId]) -> None:
        """Revoke new reads immediately.

        Storage owners retain existing readers.
        """
        self.tensor_store.release_buffers(buffers)
        if self.kv_cache is not None:
            self.kv_cache.release_buffers(buffers)
        if self.latent_pool is not None:
            self.latent_pool.release_buffers(buffers)

    def _release_predecessors(
        self, predecessors: tuple[tuple[RequestKey, CallId], ...]
    ) -> None:
        """Revoke predecessor outputs.

        Revocation happens after every declared consumer has acquired them.
        """
        self.tensor_store.release_calls(predecessors)
        if self.kv_cache is not None:
            released = self.kv_cache.release_calls(predecessors)
            self.kv_cache.release_buffers(released)

    def _release_commands(self, batch: Batch) -> None:
        """Apply Free/Finish visibility.

        Visibility applies before work can wait for their reusable storage.
        """
        freed = {
            command.buffer
            for command in batch.commands
            if isinstance(command, Free)
        }
        closed = {
            command.request_key: frozenset(command.retained_buffers) - freed
            for command in batch.commands
            if isinstance(command, Finish)
        }

        closing = tuple(
            buffer
            for request_key, retained in closed.items()
            for store in (self.tensor_store, self.kv_cache, self.latent_pool)
            if store is not None
            for buffer in retiring_exports(
                store.exports,
                requests=frozenset((request_key,)),
                retained=retained | freed,
            )
        )
        self.release_buffers((*freed, *closing))

        if self.kv_cache is not None:
            for request_key, retained in closed.items():
                self.kv_cache.imports.cancel_requests(
                    frozenset((request_key,)), retained=retained
                )

    def _release_request(
        self, request_id: int, retained: frozenset[BufferId]
    ) -> None:
        """Reset a drained request's storage.

        Independently owned products are preserved.
        """
        request = self.requests.peek(request_id)
        if request is not None:
            if self.kv_cache is not None:
                self.kv_cache.imports.cancel_requests(
                    frozenset((request.request_key,)), retained=retained
                )
            if self.decode_state is not None:
                self.decode_state.reset((request.request_pool_idx,))
            if self.block_tables is not None:
                self.block_tables.release((request.request_pool_idx,))

        if self.kv_cache is not None:
            self.kv_cache.drop(request_id)
        if request is not None and self.block_tables is not None:
            self.block_tables.release_prefixes(request.request_key)

        for store in (self.tensor_store, self.kv_cache, self.latent_pool):
            if store is not None:
                selected = tuple(
                    buffer
                    for buffer in store.exports
                    if int(buffer.owner.request_id) == request_id
                    and buffer not in retained
                )
                store.release_buffers(selected)

        if request is not None:
            self.tensor_store.release_requests(
                (request.request_key,), retained=retained
            )
        if self.media_mux is not None:
            self.media_mux.drop(request_id)
        if request is not None and self.latent_pool is not None:
            self.latent_pool.release_slots((request.request_pool_idx,))

    def drop_request(self, request_id: int) -> None:
        """Release a drained request and remove its admission from the pool."""
        request_id = int(request_id)
        self._release_request(request_id, frozenset())
        self.requests.drop(request_id)

    def retire_request(
        self,
        request_key: RequestKey,
        *,
        retained: frozenset[BufferId] = frozenset(),
    ) -> None:
        """Retire the exact epoch after readers drain.

        Independent products are retained.
        """
        request = self.requests.peek(request_key.request_id)
        if (
            request is None
            or request.request_key != request_key
            or request.retired
        ):
            return
        self._release_request(request_key.request_id, retained)
        self.requests.retire(request_key.request_id)

    def close(self) -> None:
        """Idempotently drain and release owned resources.

        Never closes the borrowed IPC endpoint. Every owner is given a
        chance to release even if another release fails. Direct execution
        and service startup are both forbidden after closing.
        """
        if self._closed:
            return

        self._closed = True

        actions: list[Callable[[], object]] = [self.runner.synchronize]
        if self.profiler is not None:
            actions.append(self.profiler.close)
        actions.append(self._release_service_runs)

        # Submitted jobs retain their mux sessions until host work has finished.
        actions.append(self.host_tasks.close)
        if self.media_mux is not None:
            actions.append(self.media_mux.close)

        # Stop imports and transports before releasing the storage they borrow.
        if self.kv_cache is not None:
            actions.append(self.kv_cache.imports.stop)
        actions.extend(
            transport.close for transport in self.transports.values()
        )

        actions.append(self.output_pool.close)
        # Producers and consumers have drained; destroy captured
        # executables before releasing the cache, latent, and product
        # backing they reference.
        actions.append(self.runner.close_graphs)
        if self.kv_cache is not None:
            actions.append(self.kv_cache.close)
        if self.latent_pool is not None:
            actions.append(self.latent_pool.close)

        actions.extend(
            (
                self.tensor_store.close,
                self.buffer_pool.close,
                self.device_events.close,
                self.runner.close,
                self.requests.close,
            )
        )
        if self.process_groups is not None:
            actions.append(self.process_groups.close)

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
                self.process_groups,
                self.requests,
                self.decode_state,
                self.block_tables,
                self.kv_cache,
                self.latent_pool,
                self.media_mux,
                self.media_buffers,
                self.output_pool,
                self.tensor_store,
                self.buffer_pool,
                self.device_events,
                self.host_tasks,
                self.transports,
                self.publication_transports,
            )

    def _release_service_runs(self) -> None:
        """Release in-flight preparations and outputs.

        Release happens before their runtime owners close.
        """
        try:
            close_resources(
                *(
                    partial(self._close_batch, batch)
                    for batch in self.inflight.values()
                )
            )
        finally:
            self.inflight.clear()
            self._batch_submissions.clear()
            self.pending_responses.clear()
            self._waiting_responses.clear()

            self._pending_requests.clear()
            self._ready_requests.clear()
            self._launched_submissions = 0
            self._executing_batches.clear()

            while not self._preparation_ready.empty():
                self._preparation_ready.get_nowait()

    def set_completion_wake(
        self,
        wake: Callable[[], None] | None,
        wake_on_stream: Callable[[int], None] | None,
    ) -> None:
        """Register callbacks used to wake result polling.

        Covers host and CUDA-stream callbacks.
        """
        self._completion_wake = wake
        self.device_events.set_completion_wake(wake_on_stream)
        self.host_tasks.set_completion_wake(wake)

        for transport in self.transports.values():
            transport.set_completion_wake(wake)

        if self.kv_cache is not None:
            self.kv_cache.imports.set_completion_wake(wake)


__all__ = ["Worker"]
