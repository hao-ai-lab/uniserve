"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from threading import Condition, RLock
from typing import TYPE_CHECKING, Any

import torch

from uniserve_worker.nn.diffusion.schedule import DiffusionSchedule
from uniserve_worker.nn.parallel import EntryConfig

from ..bootstrap.capacity import (
    device_total_bytes,
    request_tensor_arena_capacity,
    tensor_slot_capacity,
)
from ..bootstrap.worker_info import EntryInfo, WorkerInfo
from ..bootstrap.worker_info_builder import build_worker_layout, configuration_identity
from ..config import (
    DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
    LaneConfig,
    WorkerConfig,
    graph_memory_budget_bytes,
)
from ..execution.batch import (
    BufferId,
    Domain,
    Finish,
    FixedCheckpoint,
    Free,
    Locator,
    LogicalLengths,
    OpCode,
    Operation,
    RequestKey,
    Retire,
    Run,
    RunResult,
    WorkerEndpoint,
)
from ..execution.cuda_graph import (
    CudaGraphRunner,
    PrefixCapture,
    select_flow_captures,
    select_mixed_captures,
    select_prefill_captures,
)
from ..execution.forward_batch import AttentionMode, AttentionSelection, ModelPhase
from ..execution.model_runner import ForwardResult, ModelRunner, capture_image_parameters
from ..execution.output import OutputPool
from ..execution.rows import ForwardRow, LaneState, LatentExecution, OperationIdentity
from ..execution.step import (
    PreparedExecution,
    complete_startup,
    execute_batch,
    execute_prepared,
    plan_run,
    prepare_batch,
)
from ..execution.step import (
    drop_request as drop_execution_request,
)
from ..execution.trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..foundation.errors import invalid_descriptor, unsupported_setup
from ..foundation.math import ceil_div
from ..loader.source import WeightSourceConfig
from ..loader.update import WeightUpdater
from ..loader.weight_set import WeightSet
from ..models.generation import GenerationPipeline
from ..models.inputs import ImageProcessor
from ..models.runtime import ExecutionModel
from ..nn.diffusion.cfg import build_flow_cfg_plan
from ..nn.mesh import Communicator, DeviceMesh, EntryBindings
from ..runtime.cache_pool import CachePool
from ..runtime.cache_publications import CachePublications
from ..runtime.cpu import CpuPool
from ..runtime.device import canonical_device, device_memory_budget
from ..runtime.device_events import DeviceEventPool
from ..runtime.device_products import DeviceProducts
from ..runtime.encoder_cache import EncoderCache
from ..runtime.latent_pool import LatentPool
from ..runtime.persistent_buffers import PersistentBuffers
from ..runtime.req_to_token_pool import ReqToTokenPool
from ..runtime.request import RequestDraft, RequestPool
from ..runtime.runtime_states import RuntimeStates
from ..transfer.tickets import make_transports
from . import warmup as packed_warmup

if TYPE_CHECKING:
    from ..bootstrap.config import WorkerProcessArgs
    from ..execution.video import VideoMuxCoordinator, VideoOutputRing
    from ..runtime.distributed import DistributedEnvironment

logger = logging.getLogger(__name__)


class Worker:
    """Own one configured process and its sole model execution root."""

    model: ExecutionModel
    worker_config: WorkerConfig
    weights: WeightSet
    runner: ModelRunner
    runtime_states: RuntimeStates | None
    cache_pool: CachePool | None
    req_to_token_pool: ReqToTokenPool | None
    latent_pool: LatentPool | None
    _completion_wake: Callable[[], None] | None = None

    @classmethod
    def from_config(
        cls,
        config: WorkerProcessArgs,
    ) -> Worker:
        """Build model, mesh, runtime stores, lanes, transport, and capacity from validated launch configuration."""

        from ..backends.attention import resolve_attention_selection
        from ..backends.triton import configure_triton_toolchain
        from ..bootstrap.model_loader import materialize_worker_model
        from ..runtime.distributed import init_distributed_environment, initialize_model_parallel

        configure_triton_toolchain()
        environment = init_distributed_environment(
            rank=config.execution.rank,
            local_rank=config.local_rank,
            world_size=config.execution.world_size,
            device=config.execution.device,
            backend=config.distributed_backend,
            init_method=config.distributed_init_method,
        )
        meshes = initialize_model_parallel(
            environment,
            {
                name: (component.ranks, component.parallel_config)
                for name, component in config.components
                if component.distribution is None
            },
        )
        # A temporal decoder is a local model on each of its assigned ranks.
        # Its temporal width remains component params, not a parallel axis.
        for name, component in config.components:
            if config.execution.rank not in component.ranks:
                continue
            if component.distribution is not None:
                meshes[name] = DeviceMesh(
                    (config.execution.rank,),
                    config.execution.rank,
                    component.parallel_config,
                    environment.local_device,
                )
        bindings = EntryBindings(dict(config.components), meshes, environment.process_group)
        loaded = materialize_worker_model(config, bindings)
        model_mesh = meshes.get("model")
        sampling_group = None if model_mesh is None else model_mesh.get_group("tp")
        attention = (
            resolve_attention_selection(
                loaded.worker_config.attention_backend or "auto",
                tuning=config.execution.flashinfer,
                block_size=loaded.worker_config.block_size,
            )
            if loaded.model.resource_geometry.kv
            else None
        )
        worker = cls(
            loaded.model,
            sampling_group=sampling_group,
            worker_config=loaded.worker_config,
            attention=attention,
            tokenizer=loaded.tokenizer,
            allowed_work_variants=config.supported_ops,
            transfer_backends=config.data_plane.backends,
            publication_backends=config.data_plane.publication_backends,
            worker_id=config.worker_id,
            weights=loaded.weights,
            schedule=loaded.schedule,
            weight_sidecars=loaded.weight_sidecars,
            weight_sources=loaded.weight_sources,
            pipeline_depth=config.ipc.pipeline_depth,
            completion_payload_bytes=config.ipc.max_payload_bytes,
            components=config.components,
            distributed_environment=environment,
        )
        return worker

    def __init__(
        self,
        model: ExecutionModel,
        *,
        sampling_group: Communicator | None,
        worker_config: WorkerConfig,
        attention: AttentionSelection | None,
        tokenizer: Any | None,
        allowed_work_variants: frozenset[OpCode],
        transfer_backends: tuple[str, ...] = ("local",),
        publication_backends: tuple[str, ...] = ("local",),
        worker_id: str = "worker",
        weights: WeightSet | None = None,
        schedule: DiffusionSchedule | None = None,
        weight_sidecars: tuple[str, ...] = ("config.json",),
        weight_sources: tuple[WeightSourceConfig, ...] = (WeightSourceConfig(),),
        pipeline_depth: int,
        completion_payload_bytes: int,
        components: tuple[tuple[str, EntryConfig], ...] = (),
        distributed_environment: DistributedEnvironment | None = None,
    ) -> None:
        """Assemble all bounded runtime stores and execution lanes for one configured model."""

        if (
            not publication_backends
            or len(set(publication_backends)) != len(publication_backends)
            or not set(publication_backends).issubset(transfer_backends)
        ):
            raise unsupported_setup("publication backends must be unique bound transports")
        if "cuda_ipc" in transfer_backends and torch.device(worker_config.device).type != "cuda":
            raise unsupported_setup("CUDA IPC requires a CUDA worker device")
        # Freeze model identity, weight generation, and advertised capacity from
        # the same worker_config geometry before allocating runtime state.
        if not isinstance(model, ExecutionModel):
            raise unsupported_setup("worker model has no supported execution surface")
        if not isinstance(worker_config, WorkerConfig):
            raise unsupported_setup("model worker requires a worker worker_config")
        self.model = model
        self.distributed_environment = distributed_environment
        self.worker_config = worker_config
        self.sampling_group = sampling_group
        self.tokenizer = tokenizer
        self.attention = attention
        self._device = canonical_device(worker_config.device)
        self._generation_device = canonical_device(
            worker_config.generation_device or worker_config.device
        )
        self._collective_history: OrderedDict[int, object] = OrderedDict()
        self._transport_publications: dict[BufferId, tuple[Locator, ...]] = {}
        self._flow_prefix_slots: dict[RequestKey, set[int]] = {}
        self._weight_condition = Condition(RLock())
        self._active_model_calls = 0
        self._weight_update_active = False
        installed_weights = WeightSet.from_module(model) if weights is None else weights
        self.weights = installed_weights
        self.trace = ExecutionTrace(self.model.architecture)
        runner = ModelRunner(
            model,
            worker_config,
            self.trace,
            environment=distributed_environment,
            schedule=schedule,
        )
        self.runner = runner
        # Fixed executables and their storage are resident before variable pools are sized.
        runner.capture_modules()
        endpoint = WorkerEndpoint.local(worker_id, int(worker_config.rank))
        if canonical_device(worker_config.device).type == "cuda":
            available, _free = device_memory_budget(
                worker_config.device, worker_config.kv_memory_fraction
            )
            worker_config = replace(worker_config, pool_memory_bytes=available)
            schema = model.resource_geometry.request_tensors
            if schema:
                if distributed_environment is None:
                    raise unsupported_setup("request tensor sizing requires its rank group")
                public_arena = request_tensor_arena_capacity(
                    worker_config,
                    pipeline_depth=pipeline_depth,
                    product_bytes_per_request=model.product_storage_bytes,
                )
                slots = tensor_slot_capacity(
                    schema,
                    distributed_environment.process_group,
                    maximum=worker_config.max_request_pool_size,
                    minimum=worker_config.min_request_pool_size,
                    available_bytes=max(0, available - public_arena.device_product_bytes),
                    product_bytes_per_request=model.product_storage_bytes,
                )
                worker_config = replace(
                    worker_config,
                    max_request_pool_size=slots,
                    max_batch_operations=min(slots, worker_config.max_batch_operations),
                    max_batch_tokens=min(slots, worker_config.max_batch_tokens),
                )
            self.worker_config = worker_config
            runner.worker_config = worker_config
        layout = build_worker_layout(
            model,
            worker_config,
            endpoint=endpoint,
            model_name=self.model.architecture,
            weight_version=self.weights.version,
            queue_depth=int(pipeline_depth),
            completion_payload_bytes=int(completion_payload_bytes),
            capacity_group=sampling_group,
        )
        layout = replace(
            layout,
            info=replace(
                layout.info,
                configuration_id=configuration_identity(
                    model,
                    worker_config,
                    layout,
                    components,
                    None if attention is None else attention.identity,
                ),
                components=tuple(
                    EntryInfo(name, config, model.entry_outputs.get(name, ()))
                    for name, config in components
                ),
            ),
        )
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
        declared = layout.info
        arena = layout.arena
        if int(pipeline_depth) <= 0:
            raise unsupported_setup("worker pipeline depth must be positive")

        # The scheduler sees only routes supported by both its resolved plan and
        # the concrete model, further capped by physical lane geometry.
        implemented_work = model.supported_work
        self._effective_work_variants = allowed_work_variants & implemented_work
        if not self._effective_work_variants:
            raise unsupported_setup(
                f"{type(self).__name__} implements none of the requested work variants "
                f"{sorted(value.value for value in allowed_work_variants)!r}"
            )
        advertised_work = self._effective_work_variants
        if not advertised_work:
            raise unsupported_setup(f"{type(self).__name__} advertises no executable work")
        lane_operation_bound = min(
            (
                int(lane.max_batch_operations or declared.max_batch_ops)
                for lane in worker_config.lanes
            ),
            default=int(declared.max_batch_ops),
        )
        lane_token_bound = min(
            (
                int(lane.max_batch_tokens or declared.max_batch_tokens)
                for lane in worker_config.lanes
            ),
            default=int(declared.max_batch_tokens),
        )
        self._info = replace(
            declared,
            device=str(worker_config.device),
            transfer_backends=transfer_backends,
            supported_ops=tuple(code for code in OpCode if code in advertised_work),
            queue_depth=int(pipeline_depth),
            max_batch_ops=min(
                int(declared.max_batch_ops),
                lane_operation_bound,
            ),
            max_batch_tokens=min(int(declared.max_batch_tokens), lane_token_bound),
        )
        owns_kv = bool(model.resource_geometry.kv)
        packed_model = model
        if owns_kv != (attention is not None):
            raise unsupported_setup(
                "attention selection must exactly match model-owned KV resources"
            )
        cache = packed_model.cache_geometry if owns_kv else None
        self.cache_pool = None
        self.req_to_token_pool = None
        max_blocks_per_row = 0

        # KV pages and request-to-token tables share group geometry; bind them to
        # the model only after attention compatibility has been established.
        if cache is not None:
            kv_cache = self._info.kv_cache
            if kv_cache is None:
                raise unsupported_setup("KV model worker has no KV-cache configuration")
            cache_dtype = getattr(torch, str(cache.dtype).removeprefix("torch."), None)
            if not isinstance(cache_dtype, torch.dtype):
                raise unsupported_setup(f"unsupported cache dtype {cache.dtype!r}")
            max_blocks_per_row = max(
                1,
                ceil_div(int(packed_model.text_max_tokens), int(worker_config.block_size)),
            )
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
                head_dim=int(cache.head_dim),
                device=worker_config.device,
                dtype=cache_dtype,
                store_dtype=cache.store_dtype,
                group_ranges=tuple(group_ranges) if group_ranges else None,
                import_capacity=int(self._info.max_unresolved_ops),
            )
            assert attention is not None
            if (
                OpCode.DIFFUSION_STEP in self._effective_work_variants
                and not _supports_flow_attention(
                    attention,
                    cache,
                    self.cache_pool,
                    torch.device(worker_config.device),
                )
            ):
                raise unsupported_setup(
                    "image generation requires paged-prefix plus dense-current attention"
                )
            self.req_to_token_pool = ReqToTokenPool(
                group_count=self.cache_pool.group_count,
                request_pool_size=int(self._info.request_slots),
                max_blocks_per_request=max_blocks_per_row,
                block_size=int(kv_cache.block_size),
                device=worker_config.device,
                staging_depth=int(pipeline_depth),
            )
            packed_model.bind_cache_pool(self.cache_pool, attention)

        # Admission, lineage, and persistent tensors share one slot owner.
        self.requests = RequestPool(
            int(self._info.request_slots),
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
                request_pool_size=int(self._info.request_slots),
                vocab_size=int(packed_model.vocab_size),
                continuation_width=1,
                device=worker_config.device,
                logits_dtype=torch_dtype,
                valid_cache_lengths=self.req_to_token_pool.verified_lens,
            )
        else:
            self.runtime_states = None
        flow = None if packed_model is None else packed_model.generation
        latent_dtype = getattr(torch, str(layout.latent_dtype).removeprefix("torch."), None)
        if flow is not None and not isinstance(latent_dtype, torch.dtype):
            raise unsupported_setup(f"unsupported latent dtype {layout.latent_dtype!r}")
        if flow is None:
            self.latent_pool = None
        else:
            assert isinstance(latent_dtype, torch.dtype)
            self.latent_pool = LatentPool(
                request_pool_size=int(self._info.request_slots),
                num_pages=int(self._info.latent_pages),
                page_units=int(self._info.latent_page_units),
                latent_width=int(layout.latent_width),
                dtype=latent_dtype,
                device=worker_config.generation_device or worker_config.device,
            )
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
        self.output_pool = OutputPool(
            capacity=int(pipeline_depth) * int(self._info.max_batch_ops),
            max_words=int(self._info.max_batch_ops)
            * (4 + (int(completion_payload_bytes) + 3) // 4),
            event_pool=self.device_events,
        )
        self.persistent_buffers = PersistentBuffers(
            byte_capacity=int(self._info.buffer_pool_bytes),
            devices=owner_devices,
        )
        self.device_products = DeviceProducts(
            capacity=arena.device_products,
            byte_capacity=arena.device_product_bytes,
            request_capacity=int(self._info.request_slots),
            relay_depth=int(self._info.max_unresolved_ops) + 1,
            persistent_buffers=self.persistent_buffers,
            event_pool=self.device_events,
        )
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
        self.cpu_tasks = CpuPool(
            capacity=int(arena.cpu_tasks),
            workers=min(4, int(arena.cpu_tasks)),
        )
        self.transports = make_transports(
            transfer_backends,
            source=endpoint,
            byte_capacity=arena.transfer_bytes,
            ticket_capacity=arena.transfer_tickets,
            event_pool=self.device_events,
        )
        self.publication_transports = {name: self.transports[name] for name in publication_backends}
        max_rows = min(
            int(self._info.max_batch_ops),
            int(self._info.request_slots),
        )

        # Derive fixed staging and graph catalogs from the intersection of model,
        # lane, cache, latent, and scheduler capacities.
        decode_lane = next(
            (lane for lane in worker_config.lanes if Domain.DECODE in lane.domains),
            None,
        )
        prefill_lane = next(
            (lane for lane in worker_config.lanes if Domain.PREFILL in lane.domains),
            None,
        )
        flow_lane = next(
            (lane for lane in worker_config.lanes if Domain.FLOW in lane.domains),
            None,
        )
        decode_max_operations = min(
            max_rows,
            max_rows if decode_lane is None else int(decode_lane.max_batch_operations or max_rows),
        )
        prefill_max_tokens = min(
            int(self._info.max_batch_tokens),
            (
                int(self._info.max_batch_tokens)
                if prefill_lane is None
                else int(prefill_lane.max_batch_tokens or self._info.max_batch_tokens)
            ),
        )
        flow_max_operations = min(
            max_rows,
            max_rows if flow_lane is None else int(flow_lane.max_batch_operations or max_rows),
        )
        decode_graph_batch_sizes = tuple(
            value
            for value in worker_config.decode_graph_batch_sizes
            if 0 < int(value) <= decode_max_operations
            and owns_kv
            and self._info.kv_cache is not None
            and int(value) < int(self._info.kv_cache.num_blocks)
        )
        prefill_capacity = (
            min(
                int(self._info.max_batch_tokens),
                int(packed_model.text_max_tokens),
                max(0, int(self._info.kv_cache.num_blocks) - 1) * int(worker_config.block_size),
            )
            if owns_kv and self._info.kv_cache is not None
            else 0
        )
        prefill_graph_token_sizes = tuple(
            value
            for value in worker_config.prefill_graph_token_sizes
            if 0 < int(value) <= min(prefill_capacity, prefill_max_tokens)
        )
        prefill_graph_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        flow_cfg_branches: tuple[int, ...] = ()
        if flow is not None:
            branch_counts: list[int] = []
            for cfg_branches in range(1, int(flow.max_cfg_branches) + 1):
                image = capture_image_parameters(
                    cfg_branches,
                    steps=1,
                    height=16,
                    width=16,
                )
                guide = build_flow_cfg_plan(
                    cfg_text_scale=float(image.cfg_text_scale),
                    cfg_img_scale=float(image.cfg_img_scale),
                    recipe=flow.cfg_recipe,
                    renorm=image.cfg_renorm_type,
                    renorm_min=float(image.cfg_renorm_min),
                    use_cfg=True,
                )
                if len(guide.branches) != cfg_branches:
                    raise invalid_descriptor(
                        "flow CFG startup parameters do not realize their branch geometry"
                    )
                branch_counts.append(cfg_branches)
            flow_cfg_branches = tuple(branch_counts)
        flow_graph_buckets = (
            select_flow_captures(
                worker_config.flow_graph_shapes,
                worker_config.flow_graph_batch_sizes,
                flow_cfg_branches,
                max_operations=flow_max_operations,
                max_tokens=(
                    layout.input_geometry.max_tokens
                    if layout.input_geometry is not None
                    else self._info.max_batch_tokens
                ),
                per_image_capacity=int(self._info.latent_capacity_units),
                latent_capacity=int(self.latent_pool.capacity_units),
                physical_tokens=flow.physical_tokens,
                image_tokens=flow.image_tokens,
            )
            if flow is not None and self.latent_pool is not None
            else ()
        )
        shared_mixed_limit = (
            max_rows
            if not worker_config.lanes
            else max(
                (
                    min(
                        max_rows,
                        int(lane.max_batch_operations or max_rows),
                    )
                    for lane in worker_config.lanes
                    if {Domain.DECODE, Domain.FLOW} <= set(lane.domains)
                ),
                default=0,
            )
        )
        mixed_text_batch_sizes = (
            ()
            if (
                not flow_graph_buckets
                or not packed_model.tensorized_mixed
                or (
                    worker_config.lanes
                    and not any(
                        {Domain.DECODE, Domain.FLOW} <= set(lane.domains)
                        for lane in worker_config.lanes
                    )
                )
                or not {
                    OpCode.AR_DECODE,
                    OpCode.DIFFUSION_STEP,
                }.issubset(self._effective_work_variants)
            )
            else tuple(
                range(
                    1,
                    min(
                        decode_max_operations,
                        shared_mixed_limit - 1,
                        max(int(batch_size) for batch_size in worker_config.flow_graph_batch_sizes),
                    )
                    + 1,
                )
            )
        )
        mixed_flow_graph_buckets = select_mixed_captures(flow_graph_buckets, mixed_text_batch_sizes)
        flow_prefix_lengths: dict[int, tuple[int, ...]] = {}
        if flow is not None and owns_kv and packed_model.tensorized_mixed and flow_graph_buckets:
            for cfg_branches in flow_cfg_branches:
                image = capture_image_parameters(
                    cfg_branches,
                    steps=1,
                    height=16,
                    width=16,
                )
                guide = build_flow_cfg_plan(
                    cfg_text_scale=float(image.cfg_text_scale),
                    cfg_img_scale=float(image.cfg_img_scale),
                    recipe=flow.cfg_recipe,
                    renorm=image.cfg_renorm_type,
                    renorm_min=float(image.cfg_renorm_min),
                    use_cfg=True,
                )
                flow_prefix_lengths[cfg_branches] = tuple(
                    len(prefix)
                    for branch in guide.branches
                    for prefix, copy_conditioning in (
                        flow.prefix(
                            flow.branch_source(branch),
                            image_prompt="",
                            negative_prompt=image.negative_prompt,
                            negative_token_ids=(),
                            tokenizer=tokenizer,
                        ),
                    )
                    if prefix and not copy_conditioning
                )
        flow_prefix_graph_buckets = tuple(
            PrefixCapture(
                rows=rows,
                prefix_lengths=flow_prefix_lengths[cfg_branches],
            )
            for rows, cfg_branches in sorted(
                {(bucket.rows, bucket.cfg_branches) for bucket in flow_graph_buckets}
            )
            if flow_prefix_lengths.get(cfg_branches)
            and sum(
                1
                for candidate in flow_graph_buckets
                if candidate.rows == rows and candidate.cfg_branches == cfg_branches
            )
            > 1
        )
        self._flow_cfg_branches = flow_cfg_branches
        graph_budget = graph_memory_budget_bytes(device_total_bytes(worker_config.device))

        def graph_factory(
            device: torch.device,
            lane: LaneConfig | None,
            stream: torch.cuda.Stream | None,
            expected_context: int | None,
        ) -> CudaGraphRunner:
            """Construct a lane-scoped graph catalog within its row, token, and memory bounds."""

            if (
                packed_model is None
                or attention is None
                or self.cache_pool is None
                or self.runtime_states is None
            ):
                raise RuntimeError("packed graph construction lost model-owned KV resources")
            domains = tuple(Domain) if lane is None else lane.domains
            owns_model_compute = str(device) == str(torch.device(worker_config.device))

            # Intersect global graph buckets with this physical lane's advertised capacity.
            lane_max_operations = (
                max_rows if lane is None else int(lane.max_batch_operations or max_rows)
            )
            lane_max_tokens = (
                prefill_capacity if lane is None else int(lane.max_batch_tokens or prefill_capacity)
            )
            lane_decode_buckets = (
                tuple(
                    value for value in decode_graph_batch_sizes if int(value) <= lane_max_operations
                )
                if owns_model_compute and Domain.DECODE in domains
                else ()
            )
            lane_prefill_buckets = (
                tuple(value for value in prefill_graph_token_sizes if int(value) <= lane_max_tokens)
                if owns_model_compute and Domain.PREFILL in domains
                else ()
            )
            lane_prefill_row_sizes = (
                prefill_graph_row_sizes if owns_model_compute and Domain.PREFILL in domains else ()
            )
            lane_prefill_catalog = select_prefill_captures(
                lane_prefill_buckets,
                lane_prefill_row_sizes,
                max_rows=lane_max_operations,
                max_tokens=lane_max_tokens,
            )
            lane_flow_buckets = (
                tuple(value for value in flow_graph_buckets if value.rows <= lane_max_operations)
                if owns_model_compute and Domain.FLOW in domains
                else ()
            )
            lane_mixed_flow_buckets = (
                tuple(
                    value
                    for value in mixed_flow_graph_buckets
                    if value.decode_rows + value.flow_rows <= lane_max_operations
                )
                if owns_model_compute and {Domain.DECODE, Domain.FLOW} <= set(domains)
                else ()
            )
            lane_flow_prefix_buckets = (
                tuple(
                    value
                    for value in flow_prefix_graph_buckets
                    if value.rows <= lane_max_operations
                    and value.rows * sum(value.prefix_lengths) <= lane_max_tokens
                )
                if owns_model_compute and Domain.FLOW in domains
                else ()
            )
            # The expected count reserves catalog metadata and provides a precise
            # post-warmup completeness bound for this lane.
            expected_resident_executables = 0
            if worker_config.cuda_graph:
                if OpCode.AR_DECODE in self._effective_work_variants:
                    expected_resident_executables += len(lane_decode_buckets)
                if (
                    worker_config.prefill_cuda_graph
                    and OpCode.AR_EXTEND in self._effective_work_variants
                ):
                    expected_resident_executables += (
                        len(lane_prefill_buckets)
                        if packed_model.tensorized_mixed
                        else len(lane_prefill_catalog)
                    )
                if worker_config.prefill_cuda_graph and {
                    OpCode.DIFFUSION_PREPARE,
                    OpCode.DIFFUSION_STEP,
                }.issubset(self._effective_work_variants):
                    expected_resident_executables += len(
                        {bucket.executable_key for bucket in lane_flow_buckets}
                    )
                    if OpCode.AR_DECODE in self._effective_work_variants:
                        expected_resident_executables += len(
                            {bucket.executable_key for bucket in lane_mixed_flow_buckets}
                        )
                    expected_resident_executables += len(
                        {bucket.executable_key for bucket in lane_flow_prefix_buckets}
                    )
            output_slots = int(
                (pipeline_depth if lane is None else lane.max_inflight or pipeline_depth) + 1
            )
            return CudaGraphRunner(
                enabled=worker_config.cuda_graph,
                prefill_enabled=worker_config.prefill_cuda_graph,
                cache=packed_model.cache_geometry,
                cache_pool=self.cache_pool,
                attention=attention,
                block_size=worker_config.block_size,
                weight_version=self.weights.version,
                memory_budget_bytes=graph_budget,
                decode_batch_sizes=lane_decode_buckets,
                decode_predicates=(
                    self.runtime_states.predicates
                    if owns_model_compute and Domain.DECODE in domains
                    else None
                ),
                decode_context_blocks=self._decode_context_blocks(),
                packed_context_blocks=max_blocks_per_row,
                prefill_token_sizes=(() if packed_model.tensorized_mixed else lane_prefill_buckets),
                prefill_row_sizes=lane_prefill_row_sizes,
                stream=stream,
                expected_context=expected_context,
                expected_resident_executables=expected_resident_executables,
                output_slot_count=output_slots,
            )

        if owns_kv:
            assert layout.input_geometry is not None
            runner.bind_packed(
                geometry=layout.input_geometry,
                lanes=worker_config.lanes,
                max_inflight=int(pipeline_depth),
                graph_factory=graph_factory,
                flow_captures=flow_graph_buckets,
                mixed_captures=mixed_flow_graph_buckets,
                prefill_tokens=prefill_graph_token_sizes,
                prefill_rows=prefill_graph_row_sizes,
            )

        # Operation handlers borrow the resources owned by this rank.
        from ..execution.video import create_media_resources
        from ..models.video import VideoModel

        self.media_mux, self.media_output_ring = (
            create_media_resources(
                model,
                rank=worker_config.rank,
                output_rank=worker_config.output_rank,
                state_slots=self._info.request_slots,
                unresolved_window=self._info.max_unresolved_ops,
            )
            if isinstance(model, VideoModel)
            else (None, None)
        )
        self.cache_publications = (
            CachePublications(self.cache_pool, self.req_to_token_pool)
            if self.cache_pool is not None and self.req_to_token_pool is not None
            else None
        )
        self.weight_updater = (
            WeightUpdater(
                self.model,
                sources=weight_sources,
                sidecars=weight_sidecars,
                weights=self.weights,
                publish=self._publish_weight_set,
                exclusive=self._exclusive_weight_update,
            )
            if self.model.supports_weight_updates
            else None
        )

    @property
    def info(self) -> WorkerInfo:
        """Expose immutable params, capacity, and model metadata advertised to the scheduler."""

        return self._info

    def _decode_context_blocks(self) -> int:
        """Return the maximum paged-decode context blocks supported by this worker."""

        model = self.model
        if not model.resource_geometry.kv:
            return 0
        worker_config = self.worker_config
        max_tokens = int(model.text_max_tokens)
        if max_tokens < 1:
            return 0
        blocks = (max_tokens + int(worker_config.block_size) - 1) // int(worker_config.block_size)
        pool = self.cache_pool
        if pool is None:
            return 0
        return min(blocks, max(0, int(pool.num_pages) - 1))

    @staticmethod
    def operation_identity(operation: Operation) -> OperationIdentity:
        """Form the lane-local identity from request generation and operation id."""

        return operation.request_key, int(operation.op_id)

    def request_row(self, scope: LaneState, request_id: int) -> RequestDraft:
        """Return the unique staged request draft for an identifier within the current lane."""

        try:
            return scope.request_rows[int(request_id)]
        except KeyError:
            raise invalid_descriptor(f"lane has no request row for request {request_id}") from None

    def generation(self) -> GenerationPipeline:
        """Require the model's diffusion generation contract for the active operation."""

        value = self.model.generation
        if not isinstance(value, GenerationPipeline):
            raise invalid_descriptor("operation requires model generation behavior")
        return value

    def image_processor(self) -> ImageProcessor:
        """Require the model's image preprocessing contract for the active operation."""

        value = self.model.image_processor
        if not isinstance(value, ImageProcessor):
            raise invalid_descriptor("operation requires model image processing")
        return value

    def require_media_mux(self) -> VideoMuxCoordinator:
        """Require the rank-local coordinator that finalizes encoded media artifacts."""

        if self.media_mux is None:
            raise unsupported_setup("operation requires media mux resources")
        return self.media_mux

    def require_media_output_ring(self) -> VideoOutputRing:
        """Require output-owner bounded storage for asynchronous encoded media output."""

        if self.media_output_ring is None:
            raise unsupported_setup("operation requires the media output owner's ring")
        return self.media_output_ring

    def cache_coordinates(
        self,
        operation: Operation,
        scope: LaneState,
        *,
        group_id: int = 0,
    ) -> tuple[int, int, int, int]:
        """Resolve visible, computed, and physical KV coordinates for an operation and cache group."""

        request = self.request_row(scope, operation.request_key.request_id)
        slot = int(request.request.request_pool_idx)
        rows = scope.forward_rows.get(self.operation_identity(operation), ())
        descriptor = next(
            (row for row in rows if int(row.request_pool_index) == slot),
            None,
        )
        parent = request.request.parent_runtime(operation.parent)
        visible = int(parent.kv_visible_len) if descriptor is None else int(descriptor.seq_len)
        pool = self.req_to_token_pool
        if pool is None:
            raise unsupported_setup("operation requires request-to-token storage")
        pool.pages(slot, group_id)
        capacity = pool.allocated_length(slot)
        if visible > capacity:
            raise invalid_descriptor("operation visibility exceeds scheduler block table")
        return slot, int(group_id), visible, capacity

    def latent_row(self, operation: Operation, scope: LaneState) -> LatentExecution:
        """Resolve the staged physical latent params for an operation in this lane."""

        row = scope.latent_rows.get(self.operation_identity(operation))
        if row is None:
            raise invalid_descriptor("trajectory operation has no staged latent params")
        return row

    def require_latent_pool(self) -> LatentPool:
        """Require the worker-owned resident latent page pool."""

        if self.latent_pool is None:
            raise unsupported_setup("operation requires a physical latent pool")
        return self.latent_pool

    def logical_lengths(
        self,
        operation: Operation,
        request: RequestDraft,
        cache: tuple[int, int, int, int] | None,
        *,
        latent_len: int | None = None,
        computed_len: int | None = None,
    ) -> LogicalLengths:
        """Construct post-operation semantic lengths from request state and optional cache coordinates."""

        parent = request.request.parent_runtime(operation.parent)
        if cache is None:
            visible = parent.kv_visible_len
            computed = parent.kv_computed_len
        else:
            _slot, _group, visible, _capacity = cache
            computed = visible if computed_len is None else int(computed_len)
        return LogicalLengths(
            token_len=request.logical_position,
            kv_visible_len=visible,
            kv_computed_len=computed,
            latent_len=request.flow_step if latent_len is None else int(latent_len),
        )

    @staticmethod
    def output_generations(operation: Operation) -> tuple[int, ...]:
        """List logical generations in descriptor output order."""

        return tuple(int(reference.generation) for reference in operation.outputs)

    def operation_device(self, operation: Operation) -> torch.device:
        """Return the model or generation device assigned to an operation kind."""

        if operation.kind in {
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
            OpCode.DIFFUSION_DECODE,
            OpCode.MEDIA_APPEND,
            OpCode.DIFFUSION_FINALIZE,
        }:
            return self._generation_device
        return self._device

    def phase_device(self, phase: ModelPhase) -> torch.device:
        """Select the generation device for latent codecs and the model device otherwise."""

        if phase in {ModelPhase.ENCODE_LATENT, ModelPhase.DECODE_LATENT}:
            return self._generation_device
        return self._device

    def broadcast_tp_selection(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcast device-selected scalar values from tensor-parallel rank zero."""

        group = self.sampling_group
        return value if group is None else group.broadcast(value, src=0)

    def run_observed_forward_group(
        self,
        tasks: tuple[ForwardRow, ...],
        scope: LaneState,
    ) -> ForwardResult:
        """Run a forward group and accumulate its path, token, timing, and attention observations."""

        from ..execution.step import _run_forward_group

        result = _run_forward_group(self, tasks, scope)
        scope.observations.append(result.observation)
        return result

    @staticmethod
    def record_component(scope: LaneState, name: str, started_ns: int) -> None:
        """Accumulate elapsed microseconds under a lane-scoped execution component."""

        elapsed_us = max(0, (time.perf_counter_ns() - int(started_ns)) // 1000)
        scope.component_us[name] = scope.component_us.get(name, 0) + elapsed_us

    @staticmethod
    def fixed_parent(operation: Operation) -> FixedCheckpoint:
        """Require a host-resolved parent checkpoint for a depth-one state transition."""

        point = operation.state_parent.point
        if not isinstance(point, FixedCheckpoint):
            raise invalid_descriptor("operation names a device parent; depth one commits fixed")
        return point

    def execute(self, batch: Run) -> RunResult:
        """Plan and synchronously resolve one physical run under model-call exclusion."""

        with self._model_call():
            batch = self.plan_run(batch)
            report = execute_batch(self, batch)
            return self._retire_commands(batch, report)

    def plan_run(self, batch: Run) -> Run:
        """Derive the worker-private execution lanes for one physical run."""

        entries = self.info.components
        for operation in batch.operations:
            if entries:
                entry = next((entry for entry in entries if entry.name == operation.entry), None)
                if entry is None or self.worker_config.rank not in entry.config.ranks:
                    raise invalid_descriptor(
                        f"operation targets entry {operation.entry!r} outside this rank"
                    )
        return plan_run(self, batch)

    def supports_run_kind(self, kind: OpCode) -> bool:
        """Return whether this worker can execute one physical run variant."""

        return kind in self._effective_work_variants

    def prepare_execute(self, batch: Run) -> PreparedExecution:
        """Validate and stage a scheduler run while retaining all asynchronous input ownership."""

        self._begin_model_call()
        try:
            batch = self.plan_run(batch)
            prepared = prepare_batch(self, batch)
        except BaseException:
            self._end_model_call()
            raise
        return prepared.bind(
            lambda value: self._retire_commands(
                value.batch,
                execute_prepared(self, value),
            ),
            self._end_model_call,
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
        publications = self._transport_publications
        selected = tuple(
            buffer
            for buffer in publications
            if buffer in freed or (buffer.owner in closed and buffer not in retained)
        )
        releases = tuple(
            future
            for buffer in selected
            for locator in publications[buffer]
            if (future := self.transports[locator.backend].release(locator)) is not None
        )
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
            for buffer in selected:
                publications.pop(buffer, None)
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

        if not isinstance(prepared, PreparedExecution):
            raise invalid_descriptor("prepared execution has an invalid type")
        return prepared.resolve()

    @contextmanager
    def _model_call(self) -> Generator[None, None, None]:
        """Return the model-call context that enforces weight-update exclusion."""

        self._begin_model_call()
        try:
            yield
        finally:
            self._end_model_call()

    def _begin_model_call(self) -> None:
        """Register an active model call unless an exclusive update is pending."""

        with self._weight_condition:
            updater = getattr(self, "weight_updater", None)
            if updater is not None and updater.unhealthy:
                raise RuntimeError("worker weight graph is unhealthy")
            if self._weight_update_active:
                raise RuntimeError("worker weight update is draining model execution")
            self._active_model_calls += 1

    def _end_model_call(self) -> None:
        """Release an active model-call registration and wake update waiters."""

        with self._weight_condition:
            if self._active_model_calls < 1:
                raise RuntimeError("worker model-call accounting underflow")
            self._active_model_calls -= 1
            self._weight_condition.notify_all()

    @contextmanager
    def _exclusive_weight_update(self) -> Generator[None, None, None]:
        """Enter the model's optional exclusive weight-update context."""

        with self._weight_condition:
            if self._weight_update_active:
                raise RuntimeError("worker already has an active weight update")
            self._weight_update_active = True
            while self._active_model_calls:
                self._weight_condition.wait()
        try:
            yield
        finally:
            with self._weight_condition:
                self._weight_update_active = False
                self._weight_condition.notify_all()

    def _publish_weight_set(self, weights: WeightSet) -> None:
        """Atomically publish a loaded weight generation to the live registry."""

        if weights.version <= self.weights.version:
            raise ValueError("installed weight version must increase")
        self.runner.invalidate_graphs(weights.version)
        self.weights = weights
        self._info = replace(self._info, weight_version=weights.version)

    def warmup(self) -> None:
        """Materialize startup workloads and capture configured CUDA graph shapes."""

        context = packed_warmup.WarmupContext(self)
        packed_warmup.warmup(context)
        self.runner.warmup_modules(self.requests.tensor_slots)
        complete_startup(self)
        # Warmup may retain backend plans and graph pools in addition to the
        # explicit arenas. Readiness requires that these resident allocations
        # leave room for every still-lazy public product within the same grant.
        devices = tuple(
            dict.fromkeys(
                (
                    self.worker_config.device,
                    self.worker_config.generation_device or self.worker_config.device,
                )
            )
        )
        product_bytes = self._layout.arena.device_product_bytes // len(devices)
        for device in devices:
            if canonical_device(device).type != "cuda":
                continue
            available, free = device_memory_budget(device, self.worker_config.kv_memory_fraction)
            total = device_total_bytes(device)
            remaining = max(0, product_bytes - self.device_products.resident_bytes(device))
            if remaining > available or total - free > int(
                total * self.worker_config.kv_memory_fraction
            ):
                raise unsupported_setup(
                    f"initialized runtime on {device} exceeds its static memory grant: "
                    f"{total - free} resident bytes and {remaining} reserved product bytes"
                )

    def drop_request(self, request_id: int) -> None:
        """Release all runtime, cache, latent, product, and transfer state for one request."""

        request_id = int(request_id)
        request = self.requests.peek(request_id)
        drop_execution_request(self, request_id)
        if request is not None:
            self.device_products.release_requests((request.request_key,))
            self.encoder_cache.release_requests((request.request_key,))
        if self.media_mux is not None:
            self.media_mux.drop(request_id)
        if request is not None and self.latent_pool is not None:
            self.latent_pool.release_slots((int(request.request_pool_idx),))
        self.requests.drop(request_id)
        if request is not None:
            self.trace.emit(
                ExecutionPhase.CLEANUP,
                (
                    OperationTrace(
                        authority_id=request.request_key.authority_id,
                        request_id=request.request_id,
                        epoch=request.epoch,
                        op_id=0 if request.last_op_id is None else request.last_op_id,
                        version=request.version,
                    ),
                ),
            )

    def retire_request(
        self, request_key: RequestKey, *, retained: frozenset[BufferId] = frozenset()
    ) -> None:
        """Retire the exact epoch while keeping independently owned persistent products."""

        request_id = int(request_key.request_id)
        request = self.requests.peek(request_id)
        if request is None or request.request_key != request_key or request.retired:
            return
        drop_execution_request(self, request_id, retained=retained)
        self.device_products.release_requests((request.request_key,), retained=retained)
        self.encoder_cache.release_requests((request.request_key,), retained=retained)
        if self.media_mux is not None:
            self.media_mux.drop(request_id)
        if self.latent_pool is not None:
            self.latent_pool.release_slots((int(request.request_pool_idx),))
        self.requests.retire(request_id)
        self.trace.emit(
            ExecutionPhase.CLEANUP,
            (
                OperationTrace(
                    authority_id=request.request_key.authority_id,
                    request_id=request.request_id,
                    epoch=request.epoch,
                    op_id=0 if request.last_op_id is None else request.last_op_id,
                    version=request.version,
                ),
            ),
        )

    def free_products(self, buffers: tuple[BufferId, ...]) -> None:
        """Release exact scheduler buffers from device-product and encoder-cache ownership."""

        self.device_products.release_buffers(buffers)
        self.encoder_cache.release_buffers(buffers)
        if self.cache_pool is not None:
            self.cache_pool.release_buffers(buffers)

    def close(self) -> None:
        """Release execution, transport, model-state, and distributed resources owned by the worker."""

        self.runner.synchronize()
        self._collective_history.clear()
        self._transport_publications.clear()
        self._flow_prefix_slots.clear()
        if self.media_mux is not None:
            self.media_mux.close()
        self.cpu_tasks.close()
        if self.cache_pool is not None:
            self.cache_pool.imports.stop()
        for transport in self.transports.values():
            transport.close()
        self.output_pool.close()
        if self.cache_pool is not None:
            self.cache_pool.close()
        if self.latent_pool is not None:
            self.latent_pool.close()
        self.encoder_cache.close()
        self.device_products.close()
        self.persistent_buffers.close()
        self.device_events.close()
        self.runner.close()
        if self.distributed_environment is not None:
            self.distributed_environment.close()

    def set_completion_wake(
        self,
        wake: Callable[[], None],
        wake_on_stream: Callable[[int], None],
    ) -> None:
        """Register host and CUDA-stream callbacks used to wake result polling."""

        self._completion_wake = wake
        self.device_events.set_completion_wake(wake_on_stream)
        self.cpu_tasks.set_completion_wake(wake)
        for transport in self.transports.values():
            transport.set_completion_wake(wake)
        if self.cache_pool is not None:
            self.cache_pool.imports.set_completion_wake(wake)


def _supports_flow_attention(
    selection: AttentionSelection,
    geometry: object,
    pool: CachePool,
    device: torch.device,
) -> bool:
    """Return whether the selected backend can execute the model's flow-attention geometry."""

    if not pool.supports_paged_attention_storage:
        return False
    head_dim = int(getattr(geometry, "head_dim"))
    for provider in selection.providers:
        if provider.can_bind(
            AttentionMode.PACKED,
            head_dim=head_dim,
            block_size=pool.block_size,
            device=device,
        ):
            return True
    return False


__all__ = ["Worker"]
