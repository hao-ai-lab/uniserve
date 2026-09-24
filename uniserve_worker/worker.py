"""Resource scope and public entry point for one model-backed worker.

`Worker` owns the execution resources of one worker rank, including the
model runner (`ModelExecutor`), the paged KV cache and block tables of a
worker with a ``CausalLM``, the latent pool of a denoising worker, the
request slots (`RequestPool`), the product, buffer and output stores, the
transports, the host lane, the muxer rank's `MediaMux`, and the process
groups created by `from_config`.

Both entry paths drive the worker's one `Executor`. The worker process
(``uniserve_worker.bootstrap.launch``) builds a worker with `from_config`,
borrows its IPC endpoint with `bind`, and calls `run`, which warms up and
then runs `Service` on the caller's thread. A Python caller may instead call
`warmup` and drive batches directly through `submit`, `advance` and `poll`.
Construction performs no warmup, graph capture or IPC I/O.

`close` releases the owners in dependency order. After a failure on this
rank it releases only what completes without peers or the device and keeps
the rest alive until process exit (`retain_until_exit`), so the process must
exit afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.model import CausalLM, VideoPostprocessor
from uniserve.processing import FlowPrompt, ImageProcessor
from uniserve.quantization import Quantizer
from uniserve.runtime import EventPool, PrefixCache
from uniserve.runtime.device import canonical_device, device_storage_budget
from uniserve.runtime.process_groups import (
    ProcessGroups,
    initialize_process_groups,
)
from uniserve.runtime.resources import close_resources
from uniserve_worker.bootstrap.capacity import (
    check_startup_storage,
    decode_context_blocks,
    device_total_bytes,
    latent_pool_plan,
    resolve_request_capacity,
)
from uniserve_worker.bootstrap.components import (
    MUXER_COMPONENT,
    holds_host_components,
)
from uniserve_worker.bootstrap.distributed import initialize_components
from uniserve_worker.bootstrap.inputs import capability, image_builder
from uniserve_worker.bootstrap.model_loader import (
    load_worker_model,
    prepare_worker_model,
)
from uniserve_worker.bootstrap.report import build_worker_layout
from uniserve_worker.bootstrap.warmup import warmup_requests
from uniserve_worker.config.execution import (
    WorkerConfig,
    graph_storage_budget_bytes,
)
from uniserve_worker.errors import (
    unsupported_setup,
)
from uniserve_worker.execution.executor import Executor, Submission
from uniserve_worker.execution.host import HostLane
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.execution.request import RequestPool
from uniserve_worker.media.container import (
    AUDIO_CODEC,
    VIDEO_CODEC,
    require_media_codecs,
)
from uniserve_worker.media.mux import MediaMux
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.profiling import (
    WorkerProfiler,
)
from uniserve_worker.protocol.batch import Batch
from uniserve_worker.protocol.call import CallKind
from uniserve_worker.protocol.output import BatchOutput
from uniserve_worker.protocol.transfer import WorkerEndpoint
from uniserve_worker.protocol.worker_info import (
    WorkerInfo,
)
from uniserve_worker.service import Service
from uniserve_worker.storage.block_tables import BlockTables
from uniserve_worker.storage.buffer_pool import BufferPool
from uniserve_worker.storage.decode_state import DecodeState
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.latent_pool import LatentPool
from uniserve_worker.storage.output import OutputPool
from uniserve_worker.storage.tensor_store import TensorStore
from uniserve_worker.transport import make_transports

if TYPE_CHECKING:
    from uniserve_worker.bootstrap.launch import WorkerIpcEndpoint
    from uniserve_worker.config.deployment import (
        ComponentConfig,
        WorkerProcessArgs,
    )

logger = logging.getLogger(__name__)


class Worker:
    """Own numerical resources and expose direct execution or an IPC service."""

    model: nn.Module
    worker_config: WorkerConfig
    runner: ModelExecutor
    decode_state: DecodeState | None
    kv_cache: KVCacheManager | None
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

        A scope leaving on an error closes with ``aborted=True`` (see
        `close`). The scope's error is neither suppressed nor replaced; a
        cleanup failure is attached to it as a note.
        """
        if exc_value is not None:
            # Native cleanup can itself stall after a device failure. Record
            # the initiating error before entering teardown so it is not lost.
            logger.error(
                "worker execution failed before resource cleanup",
                exc_info=(type(exc_value), exc_value, traceback),
            )
        try:
            # A scope leaving on an error releases without its peers: they are
            # not leaving with it, and every collective step would wait for
            # them instead of letting the error reach the caller.
            self.close(aborted=exc_value is not None)
        except BaseException as cleanup_error:
            if exc_value is None:
                raise

            exc_value.add_note(
                f"Resource cleanup also failed: {cleanup_error!r}"
            )

    @classmethod
    def from_config(cls, config: WorkerProcessArgs) -> Self:
        """Build a worker without numerical warmup, graph capture, or IPC I/O.

        The returned worker owns the process groups and every execution
        resource until its context exits or the caller closes it.

        A failure after the process groups exist closes them with
        ``aborted=True``, which retains them until process exit instead of
        destroying them collectively; `__init__` likewise retains its own
        partial allocations. The caller must end the process after a
        construction failure that reaches that point.
        """
        # The placement is validated against the model's declarations on a
        # meta-device skeleton, so a placement error surfaces before any
        # process group exists.
        source, description, declarations = prepare_worker_model(config)

        try:
            distributed = initialize_process_groups(
                rank=config.execution.rank,
                local_rank=config.local_rank,
                world_size=config.execution.world_size,
                device=config.execution.device,
                backend=config.distributed_backend,
                rendezvous=config.rendezvous,
            )
        except ValueError as error:
            # Public numerical resources report ordinary argument errors;
            # the serving boundary assigns the worker's configuration code.
            raise unsupported_setup(str(error)) from error

        try:
            try:
                bindings = initialize_components(
                    distributed,
                    dict(config.components),
                    declarations=declarations,
                )
            except ValueError as error:
                raise unsupported_setup(str(error)) from error

            loaded = load_worker_model(
                config,
                bindings,
                source=source,
                description=description,
                declarations=declarations,
            )

            # Sampling broadcasts selected tokens over the `tp` group of the
            # component holding the `CausalLM`, independently of other
            # components' parallel layouts. `initialize_components` binds a
            # `tp` axis for such a component even without tensor-parallel
            # layers. Without such a component, or on a rank outside it, the
            # sampling group is None.
            model_mesh = next(
                (
                    bindings[name].mesh
                    for name, calls in declarations.items()
                    if name in bindings
                    and any(isinstance(call.module, CausalLM) for call in calls)
                ),
                None,
            )
            sampling_group = (
                None if model_mesh is None else model_mesh.get_group("tp")
            )

            # The constructor handles its own partial allocations on failure.
            # On success the worker also takes responsibility for
            # `distributed.close()`. `attention` is left unset, so the runner
            # resolves the backend from the worker configuration.
            return cls(
                loaded.model,
                bindings=bindings,
                sampling_group=sampling_group,
                entry_points=loaded.entry_points,
                worker_config=loaded.config,
                tokenizer=loaded.tokenizer,
                image_processor=loaded.image_processor,
                flow_prompt=loaded.flow_prompt,
                allowed_calls=config.supported_calls,
                transfer_backends=config.data_plane.backends,
                publication_backends=config.data_plane.publication_backends,
                worker_id=config.worker_id,
                checkpoint_identity=loaded.checkpoint_identity,
                queue_depth=config.ipc.queue_depth,
                completion_payload_bytes=config.ipc.max_payload_bytes,
                acknowledgment_slot=config.ipc.acknowledgment_slot,
                host_slots=config.ipc.host_slots,
                products_cross_hosts=config.ipc.products_cross_hosts,
                components=config.components,
                process_groups=distributed,
            )
        except BaseException as error:
            try:
                distributed.close(aborted=True)
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
        allowed_calls: frozenset[CallKind],
        queue_depth: int,
        completion_payload_bytes: int,
        acknowledgment_slot: int = 0,
        host_slots: Sequence[int] = (),
        products_cross_hosts: bool = False,
        attention: str | None = None,
        transfer_backends: tuple[str, ...] = ("local",),
        publication_backends: tuple[str, ...] = ("local",),
        worker_id: str = "worker",
        image_processor: ImageProcessor | None = None,
        flow_prompt: FlowPrompt | None = None,
        components: tuple[tuple[str, ComponentConfig], ...] = (),
        process_groups: ProcessGroups | None = None,
        bindings: Mapping[str, ComponentBinding] | None = None,
        checkpoint_identity: str = "",
        entry_points=None,
    ) -> None:
        """Allocate execution resources for an already-loaded model.

        Resolve unspecified attention from worker_config. Construction
        performs no warmup or IPC I/O. Sizing the request capacity and the
        KV page count can run collectives across the worker's ranks, so
        every rank must construct its worker together.

        On error, the host lane and the runner are aborted and every other
        partial allocation is retained until process exit rather than
        released, since a failed CUDA owner cannot prove that its device
        accesses ended; the caller must end the process. After success the
        caller closes the worker or uses its owning context.
        """
        self._closed = False
        self._run_started = False
        self._warmed_up = False

        self.ipc_endpoint: WorkerIpcEndpoint | None = None
        self.profiler: WorkerProfiler | None = None

        # `startup` collects each allocated owner's release but is never
        # unwound: on success `close` takes the owners over, and on failure
        # they are retained until process exit.
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

            runner = ModelExecutor(
                model,
                worker_config,
                bindings=bindings,
                attention=attention,
                entry_points=entry_points,
                image_processor=image_processor,
                flow_prompt=flow_prompt,
                max_inflight=queue_depth,
            )
            self.runner = runner
            startup.callback(runner.close)

            self.attention = runner.attention

            endpoint = WorkerEndpoint.local(worker_id, int(worker_config.rank))

            # Measure the remaining grant after the runner binds persistent
            # model inputs and workspaces. Sizing earlier would treat
            # occupied storage as free. Request capacity is the only
            # configuration resolved after binding; the runner receives the
            # resolved configuration too.
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
                allowed_calls=allowed_calls,
                transfer_backends=transfer_backends,
                components=components,
                checkpoint_identity=checkpoint_identity,
                attention_backend=runner.attention.name,
                # The scheduler's page indices address every rank's resident
                # layer and head regions, including stages with different
                # storage grants, so a multi-rank token worker's ranks agree
                # on the minimum KV page count over this group.
                capacity_group=(
                    process_groups.process_group
                    if process_groups is not None
                    else sampling_group
                ),
            )

            # Only the token-worker layout reports fixed device bytes; with a
            # KV cache, its primary-device bytes were charged against the KV
            # pool's grant when the layout sized it. This loop checks each
            # auxiliary CUDA device's own grant, which must hold that
            # device's fixed bytes plus the default graph allowance.
            for device, fixed_bytes in layout.fixed_device_bytes:
                if (
                    device == worker_config.device
                    or canonical_device(device).type != "cuda"
                ):
                    continue

                available, _free = device_storage_budget(
                    device, worker_config.kv_storage_fraction
                )
                reserved = fixed_bytes + graph_storage_budget_bytes(
                    device_total_bytes(device)
                )

                if reserved > available:
                    raise unsupported_setup(
                        f"runtime storage on {device} requires {reserved} "
                        "bytes, but its static storage grant is "
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

            # A worker with a `CausalLM` owns the paged KV cache and its
            # per-request block tables, sized by the layout's `KVCacheInfo`.
            # The runner borrows both through `configure_inputs` below.
            if cache is not None:
                assert text is not None
                kv_cache = info.kv_cache
                if kv_cache is None:
                    raise unsupported_setup(
                        "KV model worker has no KV-cache configuration"
                    )

                # Block-table width: the pages of the longest sequence.
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

                # An FP8 cache passes each layer an FP8 `Quantizer` in place
                # of a storage dtype.
                encoded = kv_cache.dtype == "float8_e4m3fn"
                self.kv_cache = KVCacheManager(
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
                    import_capacity=int(info.max_unresolved_calls),
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

            # Denoised samples have their own page pool, which may live on
            # another device. An image worker always holds it; a standalone
            # denoiser's pool lives on the ranks that advance its samples.
            plan = latent_pool_plan(model, worker_config)
            if plan is None or (
                image_builder(model) is None and not runner.denoises
            ):
                self.latent_pool = None
            else:
                self.latent_pool = LatentPool(
                    request_pool_size=int(info.request_slots),
                    num_pages=int(info.latent_pages),
                    page_units=int(info.latent_page_units),
                    latent_width=plan.latent_width,
                    dtype=plan.dtype,
                    device=worker_config.generation_device
                    or worker_config.device,
                    staging=plan.staging,
                )
                startup.callback(self.latent_pool.close)
            if runner.denoises:
                # Ladders gather a request's tables from its slot and its
                # samples from its pages, so runners borrow both owners.
                assert self.latent_pool is not None
                runner.bind_diffusion_storage(
                    self.requests.storage.bank, self.latent_pool
                )

            # Storage sizing charged the latent pool's planned bytes
            # (`LatentPoolPlan.capacity_bytes`), so the allocation must
            # match that plan exactly.
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

            # `max_words` bounds a batch's completion lease. Each call takes
            # `SAMPLING_COMPLETION_FIELDS` (4) int64 words plus its payload
            # bytes counted at 4 bytes per word; `_completion_words` in
            # `uniserve_worker.execution.prepare` sizes each lease by the
            # same rule, so the two change together.
            self.output_pool = OutputPool(
                capacity=int(queue_depth) * int(info.max_batch_calls),
                max_words=int(info.max_batch_calls)
                * (4 + (int(completion_payload_bytes) + 3) // 4),
                event_pool=self.device_events,
            )
            startup.callback(self.output_pool.close)

            # A rank that backs only the products it produces or consumes has
            # a physical arena smaller than the engine's logical pool;
            # `compact` then places bindings first-fit instead of at the
            # scheduler's offsets.
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
                entry_capacity=int(info.encoder_cache_entries),
                max_entry_bytes=max(1, int(info.encoder_entry_bytes)),
                devices=owner_devices,
                request_capacity=int(info.request_slots),
                relay_depth=int(info.max_unresolved_calls) + 1,
                buffer_pool=self.buffer_pool,
                event_pool=self.device_events,
            )
            startup.callback(self.tensor_store.close)

            # A rank holding host components is one codec slot: its lane runs
            # one codec task at a time on one thread. Any other rank's host
            # work is bounded by its arena. The lane's capacity must equal the
            # `host_lane_capacity` advertised to the engine; the
            # request-tensor layout in `build_worker_layout` applies the same
            # rule, so the two change together.
            self.codec_slot = holds_host_components(
                name for name, _ in components
            )
            self.host_tasks = (
                HostLane(max_inflight=1, workers=1)
                if self.codec_slot
                else HostLane(
                    max_inflight=int(arena.host_lane_inflight),
                    workers=min(4, int(arena.host_lane_inflight)),
                )
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
                acknowledgment_slot=acknowledgment_slot,
                host_slots=host_slots,
                cross_host_consumers=products_cross_hosts,
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
                    max_calls=int(info.max_batch_calls),
                    request_slots=int(info.request_slots),
                    max_tokens=int(info.max_batch_tokens),
                    latent_capacity_units=int(info.latent_capacity_units),
                    decode_context_blocks=decode_context_blocks(
                        model, worker_config, self.kv_cache
                    ),
                    max_inflight=int(queue_depth),
                )

            # Only the muxer member that publishes the component's host
            # products (`WorkerInfo.output_rank`) holds a `MediaMux`; it
            # assembles each request's container on its host lane.
            muxer = self.runner.bindings.get(MUXER_COMPONENT)
            self.media_mux = (
                MediaMux(rank=worker_config.rank)
                if muxer is not None
                and muxer.owns
                and worker_config.rank == info.output_rank(MUXER_COMPONENT)
                else None
            )
            if self.media_mux is not None:
                startup.callback(self.media_mux.close)

        except BaseException as error:
            # Releasing partially started owners could wait on device work
            # or peers and replace this error with a stall, so keep them
            # alive until process exit. The host lane and the runner take
            # only their aborted releases, which do not wait.
            try:
                from uniserve.runtime.resources import retain_until_exit

                self._closed = True
                retain_until_exit((self, startup.pop_all()))
                if "host_tasks" in self.__dict__:
                    self.host_tasks.abort()
                if "runner" in self.__dict__:
                    self.runner.close(aborted=True)
            except BaseException as cleanup_error:
                error.add_note(
                    f"Resource cleanup also failed: {cleanup_error!r}"
                )

            raise

        # All registered resources now belong to this Worker and its close
        # method.
        startup.pop_all()
        self.executor = Executor(self)
        self.service: Service | None = None

    @property
    def info(self) -> WorkerInfo:
        """Return the immutable description this worker advertises.

        It carries the endpoint, supported calls, capacity bounds and model
        metadata that `build_worker_layout` resolved for the engine. It
        stays readable after `close`.
        """
        return self._layout.info

    def _require_open(self) -> None:
        """Raise when this worker's resource scope has already been closed."""
        if self._closed:
            raise RuntimeError("worker is closed and cannot be reused")

    def supports_computation(self, kind: CallKind) -> bool:
        """Return whether this worker can execute this computation."""
        return kind in self.info.supported_calls

    def warmup(self) -> None:
        """Prepare numerical execution once without serving requests.

        A numerical worker binds its graph storage budgets, warms and
        captures its runners, and drives synthetic requests through
        `warmup_requests`; a codec-slot rank checks that its media codecs
        load. Startup is then sealed and resident storage is checked
        against the device grants. Successful warmup is retained, so a later
        call (including the one in `run`) returns immediately.

        Errors from warmup, capture and the storage checks propagate. The
        owner must exit the Worker scope if startup fails, just as for a
        binding or service failure.

        Raises:
            RuntimeError: The worker is closed, a batch was already
                submitted, or startup left a request resident.
        """
        self._require_open()
        if self._warmed_up:
            return
        if self.executor._last_batch_id >= 0:
            raise RuntimeError(
                "warmup must precede the first serving submission"
            )

        if self.runner.numerical:
            # Bind each CUDA device's graph budget before capture, which
            # fails when graph residency exceeds it. The budget is the graph
            # pools' current residency plus this process's remaining grant on
            # the device, less the device's share of products not yet
            # resident. A token worker's remaining grant includes the graph
            # allowance that KV sizing held back; a media worker's graph
            # share is what its fixed request banks and lazy products leave,
            # not a fraction of total device memory.
            devices = tuple(
                dict.fromkeys(
                    (
                        self.worker_config.device,
                        self.worker_config.generation_device
                        or self.worker_config.device,
                    )
                )
            )
            storage = self.runner.graph_storage
            resident = storage.resident_bytes()
            products = self._layout.arena.device_product_bytes // len(devices)
            for device in devices:
                target = canonical_device(device)
                if target.type != "cuda":
                    continue
                available, _free = device_storage_budget(
                    target, self.worker_config.kv_storage_fraction
                )
                pending = max(
                    0, products - self.tensor_store.resident_bytes(target)
                )
                storage.set_budget(
                    target,
                    resident.get(target, 0) + max(0, available - pending),
                )
            self.runner.warmup(self.requests.storage.tensor_slots)
            self.runner.capture(
                tokenizer=self.tokenizer, latents=self.latent_pool
            )
            warmup_requests(self)
        # A host rank is ready once its codecs load, so a missing codec
        # surfaces here and not under a request.
        if self.codec_slot:
            require_media_codecs(VIDEO_CODEC, AUDIO_CODEC)
        self.runner.complete_startup()

        # Startup scenarios must release their requests before service
        # admission.
        if self.requests.request_ids():
            raise RuntimeError("startup completed with resident requests")

        # Warmup batches advance the executor's batch id and collective
        # sequence. The executor rejects a non-increasing batch id at
        # admission and, on a multi-rank worker, a non-increasing collective
        # sequence at launch, so reset both to accept the engine's first
        # batch.
        self.executor._last_batch_id = -1
        self.executor._last_collective_seq = -1
        check_startup_storage(
            self.worker_config,
            self._layout.arena.device_product_bytes,
            self.tensor_store,
        )
        self._warmed_up = True

    def close(self, *, aborted: bool = False) -> None:
        """Idempotently drain and release owned resources.

        Never closes the borrowed IPC endpoint. Every owner is given a
        chance to release even if another release fails. Direct execution
        and service startup are both forbidden after closing.

        ``aborted`` releases after a failure on this rank. A normal release
        drains the device and retires this rank's communicators and process
        groups collectively, which the peers complete only when they are
        releasing too; a rank that fails alone would wait for ranks that are
        still serving. An aborted release therefore takes only the steps that
        complete on this rank, so the failure reaches the caller.
        """
        if self._closed:
            return

        self._closed = True

        if aborted:
            from uniserve.runtime.resources import retain_until_exit

            # Transfers, outputs and graphs can borrow the same backing. Keep
            # the complete tree; no failed access is acknowledged as retired.
            retain_until_exit(self)
            close_resources(
                partial(self.set_completion_wake, None, None),
                self.host_tasks.abort,
                partial(self.runner.close, aborted=True),
                *(
                    (partial(self.process_groups.close, aborted=True),)
                    if self.process_groups is not None
                    else ()
                ),
            )
            self.ipc_endpoint = None
            self.service = None
            return

        actions: list[Callable[[], object]] = [self.runner.synchronize]
        if self.profiler is not None:
            actions.append(self.profiler.close)
        actions.append(self.executor.close)

        # Discarding mux sessions before the host lane drains is safe: a
        # session held by a running lane task is closed when that task
        # releases it (`MuxSession.discard`).
        if self.media_mux is not None:
            actions.append(self.media_mux.close)
        actions.append(self.host_tasks.close)

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
                partial(self.runner.close, aborted=aborted),
                self.requests.close,
            )
        )
        if self.process_groups is not None:
            actions.append(partial(self.process_groups.close, aborted=aborted))

        # Async producers have stopped before callback references are removed.
        actions.append(partial(self.set_completion_wake, None, None))

        try:
            close_resources(*actions)
        finally:
            self.ipc_endpoint = None
            self.service = None

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
                self.output_pool,
                self.tensor_store,
                self.buffer_pool,
                self.device_events,
                self.host_tasks,
                self.transports,
                self.publication_transports,
            )

    def set_completion_wake(
        self,
        wake: Callable[[], None] | None,
        wake_on_stream: Callable[[int], None] | None,
    ) -> None:
        """Register callbacks used to wake result polling.

        ``wake`` is installed on the host lane, every transport and, on a KV
        worker, the KV imports (`CacheImports`); they call it from their own
        threads or callbacks as work completes.
        ``wake_on_stream`` is installed on the device event pool, which calls
        it with the handle of a CUDA stream that waits on a producer event.
        `bind` registers the IPC endpoint's callbacks; passing ``None`` for
        both unregisters them.
        """
        self._completion_wake = wake
        self.device_events.set_completion_wake(wake_on_stream)
        self.host_tasks.set_completion_wake(wake)

        for transport in self.transports.values():
            transport.set_completion_wake(wake)

        if self.kv_cache is not None:
            self.kv_cache.imports.set_completion_wake(wake)

    def bind(self, endpoint: WorkerIpcEndpoint) -> Self:
        """Borrow an open IPC endpoint for one synchronous service run.

        Also creates the profiler from the environment and registers the
        endpoint's wake callbacks. `close` drops the endpoint without closing
        it; its owner closes it.

        Raises:
            RuntimeError: The worker is closed or already bound.
            ValueError: The endpoint is missing or closed.
        """
        self._require_open()
        if self.service is not None:
            raise RuntimeError("worker already has a bound IPC endpoint")
        if endpoint is None or endpoint.closed:
            raise ValueError("worker binding requires an open IPC endpoint")
        self.profiler = WorkerProfiler.from_env()
        self.service = Service(self, endpoint)
        self.ipc_endpoint = endpoint
        self.set_completion_wake(endpoint.wake, endpoint.wake_on_stream)
        return self

    def run(self) -> None:
        """Warm up, then run the bound IPC service once in the caller thread.

        Raises:
            RuntimeError: The worker is closed, has already run, or has no
                bound endpoint.
        """
        self._require_open()
        if self._run_started:
            raise RuntimeError("worker can only run once")
        if self.service is None:
            raise RuntimeError("worker has no bound IPC endpoint")
        self._run_started = True
        self.warmup()
        self.service.run()

    def submit(
        self, batch: Batch, *, propagate_errors: bool = False
    ) -> Submission:
        """Accept a batch and return its immutable, single-consumption handle.

        Call advance to progress work and poll to consume the result. Batch
        ids must strictly increase from one submission to the next. A full
        queue raises ResourceError without consuming the batch identity.
        With ``propagate_errors``, a failure recorded during submission is
        raised after the batch is released instead of at poll.
        """
        return self.executor.submit(batch, propagate_errors=propagate_errors)

    def advance(self) -> None:
        """Progress accepted work without consuming its results."""
        self.executor.advance()

    def poll(self, submission: Submission) -> BatchOutput | None:
        """Consume a result, or return None while the submission is pending."""
        return self.executor.poll(submission)

    def drop_request(self, request_id: int) -> None:
        """Release a drained request and remove its admission."""
        self.executor.drop_request(request_id)


__all__ = ["Submission", "Worker"]
