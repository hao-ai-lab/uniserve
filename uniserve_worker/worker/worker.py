"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from threading import Condition, RLock
from typing import TYPE_CHECKING

import torch

from ..bootstrap.capacity import (
    device_total_bytes,
    model_arena_capacity,
)
from ..bootstrap.execution_config import (
    DEFAULT_PREFILL_GRAPH_ROW_BUCKETS,
    ExecutionConfig,
    LaneConfig,
    graph_memory_budget_bytes,
)
from ..bootstrap.worker_info import WorkerInfo
from ..bootstrap.worker_info_builder import build_worker_layout
from ..execution.batch import (
    Domain,
    Finish,
    Run,
    RunKind,
    RunResult,
    logical_op_kinds,
)
from ..execution.cuda_graph import CudaGraphRunner
from ..execution.forward_batch import AttentionMode, AttentionSelection
from ..execution.graph_bucket import GraphBucket
from ..execution.model_runner import ModelRunner
from ..execution.output import OutputPool
from ..execution.step import (
    PreparedExecution,
    close_execution,
    complete_startup,
    create_execution_resources,
    execute_batch,
    execute_prepared,
    install_weights,
    plan_run,
    prepare_batch,
)
from ..execution.step import (
    drop_request as drop_execution_request,
)
from ..execution.trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..foundation.errors import invalid_descriptor, unsupported_setup
from ..foundation.math import ceil_div
from ..loader.update import WeightUpdater
from ..loader.weight_set import WeightSet
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.diffusion.cfg import build_flow_cfg_plan
from ..nn.mesh import DeviceMesh
from ..runtime.cache_pool import CachePool
from ..runtime.cpu import CpuPool
from ..runtime.device_events import DeviceEventPool
from ..runtime.device_products import DeviceProducts
from ..runtime.encoder_cache import EncoderCache
from ..runtime.latent_pool import LatentPool
from ..runtime.persistent_buffers import PersistentBuffers
from ..runtime.req_to_token_pool import ReqToTokenPool
from ..runtime.request import RequestPool
from ..runtime.runtime_states import RuntimeStates
from ..transfer.connector import TransferConnector
from . import warmup as packed_warmup
from .warmup import (
    _flow_graph_executable,
    _flow_prefix_graph_executable,
    _FlowGraphBucket,
    _FlowPrefixGraphBucket,
    _has_decode_flow_lane,
    _mixed_flow_graph_executable,
    _paged_prefill_graph_buckets,
    _startup_image_parameters,
)

if TYPE_CHECKING:
    from ..bootstrap.config import WorkerProcessArgs

logger = logging.getLogger(__name__)


class Worker:
    """Own one configured process and its sole model execution root."""

    model: ExecutionModel
    deployment: WorkerDeployment
    weights: WeightSet
    runner: ModelRunner | None
    runtime_states: RuntimeStates | None
    cache_pool: CachePool | None
    req_to_token_pool: ReqToTokenPool | None
    latent_pool: LatentPool | None

    @classmethod
    def from_config(cls, config: WorkerProcessArgs) -> Worker:
        """Build model, mesh, runtime stores, lanes, transport, and capacity from validated launch configuration."""

        from ..backends.attention import resolve_attention_selection
        from ..backends.triton import configure_triton_toolchain
        from ..bootstrap.model_loader import materialize_worker_model
        from ..bootstrap.plan import resolve_worker_plan
        from ..nn.placement import place_towers
        from ..runtime.distributed import build_device_mesh

        plan = resolve_worker_plan(config.supported_ops)
        configure_triton_toolchain()
        mesh = build_device_mesh(
            tp_rank=config.placement.tp_rank,
            tp_size=config.placement.tp_size,
            device=config.placement.device,
            tower_devices=config.placement.tower_devices,
            tower_primary=0,
            tp_backend=config.placement.tp_backend,
            tp_init_method=config.placement.tp_init_method,
        )
        loaded = materialize_worker_model(config, plan, mesh)
        place_towers(loaded.model, mesh)
        attention = (
            resolve_attention_selection(
                loaded.deployment.attention_backend or "auto",
                tuning=config.execution.flashinfer,
                block_size=loaded.deployment.block_size,
            )
            if loaded.model.resource_geometry.kv
            else None
        )
        return cls(
            loaded.model,
            mesh=mesh,
            deployment=loaded.deployment,
            attention=attention,
            execution=config.execution,
            tokenizer=loaded.tokenizer,
            allowed_work_variants=plan.allowed_work_variants,
            transfer_backend=config.data_plane.backend,
            cross_process=config.data_plane.backend != "local",
            weights=loaded.weights,
            weight_sidecars=loaded.weight_sidecars,
            pipeline_depth=config.ipc.pipeline_depth,
            completion_payload_bytes=config.ipc.max_payload_bytes,
        )

    def __init__(
        self,
        model: ExecutionModel,
        *,
        mesh: DeviceMesh,
        deployment: WorkerDeployment,
        attention: AttentionSelection | None,
        execution: ExecutionConfig,
        tokenizer: object | None,
        allowed_work_variants: frozenset[RunKind],
        transfer_backend: str = "local",
        cross_process: bool = False,
        weights: WeightSet | None = None,
        weight_sidecars: tuple[str, ...] = ("config.json",),
        pipeline_depth: int,
        completion_payload_bytes: int,
    ) -> None:
        """Assemble all bounded runtime stores and execution lanes for one model replica."""

        # Freeze model identity, weight generation, and advertised capacity from
        # the same deployment geometry before allocating runtime state.
        if not isinstance(model, ExecutionModel):
            raise unsupported_setup("worker model has no supported execution surface")
        if not isinstance(deployment, WorkerDeployment):
            raise unsupported_setup("model worker requires a worker deployment")
        self.model = model
        self.mesh = mesh
        self.deployment = deployment
        self._weight_condition = Condition(RLock())
        self._active_model_calls = 0
        self._weight_update_active = False
        installed_weights = WeightSet.from_module(model) if weights is None else weights
        self.weights = installed_weights
        self.architecture = model.architecture
        self.weight_version = installed_weights.version
        layout = build_worker_layout(
            model,
            deployment,
            model_name=self.architecture,
            weight_version=self.weight_version,
            queue_depth=int(pipeline_depth),
            completion_payload_bytes=int(completion_payload_bytes),
        )
        self._layout = layout
        declared = layout.info
        declared_kv = declared.kv_cache
        arena = model_arena_capacity(
            model,
            deployment,
            pipeline_depth=int(pipeline_depth),
            completion_payload_bytes=int(completion_payload_bytes),
            num_blocks=0 if declared_kv is None else int(declared_kv.num_blocks),
            request_pool_size=int(declared.request_slots),
            num_latent_pages=int(declared.latent_pages),
            latent_page_units=int(declared.latent_page_units),
            latent_width=int(layout.latent_width),
            max_latent_feature_bytes=int(layout.max_latent_feature_bytes),
            max_vision_feature_bytes=int(layout.max_vision_feature_bytes),
            bytes_per_token=0 if declared_kv is None else int(declared_kv.bytes_per_token),
        )
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
                for lane in execution.lanes
            ),
            default=int(declared.max_batch_ops),
        )
        lane_token_bound = min(
            (int(lane.max_batch_tokens or declared.max_batch_tokens) for lane in execution.lanes),
            default=int(declared.max_batch_tokens),
        )
        self._info = replace(
            declared,
            supported_ops=logical_op_kinds(tuple(advertised_work)),
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
                ceil_div(int(packed_model.text_max_tokens), int(deployment.block_size)),
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
                head_dim=int(cache.head_dim),
                device=deployment.device,
                dtype=cache_dtype,
                store_dtype=cache.store_dtype,
                group_ranges=tuple(group_ranges) if group_ranges else None,
            )
            assert attention is not None
            if (
                RunKind.DIFFUSION_STEP in self._effective_work_variants
                and not _supports_flow_attention(
                    attention,
                    cache,
                    self.cache_pool,
                    torch.device(deployment.device),
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
                device=deployment.device,
                staging_depth=int(pipeline_depth),
            )
            packed_model.bind_cache_pool(self.cache_pool, attention)

        # Generic request rows own admission and lineage. Model-specific state,
        # continuation vectors, and latent pages retain the same one-based slots.
        self.requests = RequestPool(int(self._info.request_slots))
        self.requests.attach_model_state(model.create_request_state())
        torch_dtype = getattr(
            torch,
            str(deployment.model_dtype).removeprefix("torch."),
            None,
        )
        if not isinstance(torch_dtype, torch.dtype):
            raise unsupported_setup(f"unsupported model dtype {deployment.model_dtype!r}")
        if self.req_to_token_pool is not None:
            self.runtime_states = RuntimeStates(
                request_pool_size=int(self._info.request_slots),
                vocab_size=int(packed_model.vocab_size),
                continuation_width=1,
                device=deployment.device,
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
                device=deployment.generation_device or deployment.device,
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
                    deployment.device,
                    deployment.generation_device or deployment.device,
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
        self.transfers = TransferConnector(
            backend=transfer_backend,
            byte_capacity=arena.transfer_bytes,
            ticket_capacity=arena.transfer_tickets,
            cross_process=bool(cross_process),
        )
        max_rows = min(
            int(self._info.max_batch_ops),
            int(self._info.request_slots),
        )

        # Derive fixed staging and graph catalogs from the intersection of model,
        # lane, cache, latent, and scheduler capacities.
        max_staged_rows = max_rows * (1 if flow is None else int(flow.max_cfg_branches))
        max_text_staged_tokens = int(self._info.max_batch_tokens)
        max_flow_staged_tokens = (
            0
            if flow is None
            else max(
                (
                    int(flow.max_latent_tokens),
                    *(
                        flow.physical_tokens(int(height), int(width))
                        for height, width in execution.flow_graph_shapes
                    ),
                )
            )
            * int(flow.max_cfg_branches)
        )
        # Flow may be the final operation admitted after earlier text consumes
        # the nominal token budget. Keep the fixed sequence arena large enough
        # for that text span plus one exact CFG-expanded flow operation.
        max_staged_tokens = max_text_staged_tokens + max_flow_staged_tokens
        decode_lane = next(
            (lane for lane in execution.lanes if Domain.DECODE in lane.domains),
            None,
        )
        prefill_lane = next(
            (lane for lane in execution.lanes if Domain.PREFILL in lane.domains),
            None,
        )
        flow_lane = next(
            (lane for lane in execution.lanes if Domain.FLOW in lane.domains),
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
            for value in execution.decode_graph_batch_sizes
            if 0 < int(value) <= decode_max_operations
            and owns_kv
            and self._info.kv_cache is not None
            and int(value) < int(self._info.kv_cache.num_blocks)
        )
        prefill_capacity = (
            min(
                int(self._info.max_batch_tokens),
                int(packed_model.text_max_tokens),
                max(0, int(self._info.kv_cache.num_blocks) - 1) * int(deployment.block_size),
            )
            if owns_kv
            else 0
        )
        prefill_graph_token_sizes = tuple(
            value
            for value in execution.prefill_graph_token_sizes
            if 0 < int(value) <= min(prefill_capacity, prefill_max_tokens)
        )
        prefill_graph_row_sizes = DEFAULT_PREFILL_GRAPH_ROW_BUCKETS
        flow_cfg_branches: tuple[int, ...] = ()
        if flow is not None:
            branch_counts: list[int] = []
            for cfg_branches in range(1, int(flow.max_cfg_branches) + 1):
                image = _startup_image_parameters(
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
            ()
            if flow is None
            else tuple(
                _FlowGraphBucket(
                    rows=int(batch_size),
                    height=int(height),
                    width=int(width),
                    cfg_branches=cfg_branches,
                )
                for height, width in execution.flow_graph_shapes
                for batch_size in execution.flow_graph_batch_sizes
                for cfg_branches in flow_cfg_branches
                if 0 < int(batch_size) <= flow_max_operations
                and int(batch_size) * flow.physical_tokens(int(height), int(width)) * cfg_branches
                <= max_staged_tokens
                and flow.image_tokens(int(height), int(width))
                <= int(self._info.latent_capacity_units)
                and self.latent_pool is not None
                and int(batch_size) * flow.image_tokens(int(height), int(width))
                <= int(self.latent_pool.capacity_units)
            )
        )
        shared_mixed_limit = (
            max_rows
            if not execution.lanes
            else max(
                (
                    min(
                        max_rows,
                        int(lane.max_batch_operations or max_rows),
                    )
                    for lane in execution.lanes
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
                or not _has_decode_flow_lane(execution.lanes)
                or not {
                    RunKind.AR_DECODE,
                    RunKind.DIFFUSION_STEP,
                }.issubset(self._effective_work_variants)
            )
            else tuple(
                range(
                    1,
                    min(
                        decode_max_operations,
                        shared_mixed_limit - 1,
                        max(int(batch_size) for batch_size in execution.flow_graph_batch_sizes),
                    )
                    + 1,
                )
            )
        )
        mixed_flow_graph_buckets = tuple(
            GraphBucket(
                decode_rows=text_batch_size,
                flow_rows=1,
                height=bucket.height,
                width=bucket.width,
                cfg_branches=bucket.cfg_branches,
            )
            for bucket in flow_graph_buckets
            if bucket.rows == 1
            for text_batch_size in mixed_text_batch_sizes
        )
        flow_prefix_lengths: dict[int, tuple[int, ...]] = {}
        if (
            flow is not None
            and owns_kv
            and packed_model.tensorized_mixed
            and flow_graph_buckets
        ):
            for cfg_branches in flow_cfg_branches:
                image = _startup_image_parameters(
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
            _FlowPrefixGraphBucket(
                rows=rows,
                cfg_branches=cfg_branches,
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
        self._prefill_graph_token_sizes = prefill_graph_token_sizes
        self._prefill_graph_row_sizes = prefill_graph_row_sizes
        self._flow_cfg_branches = flow_cfg_branches
        self._flow_graph_buckets = flow_graph_buckets
        self._mixed_flow_graph_buckets = mixed_flow_graph_buckets
        self._mixed_buckets = mixed_flow_graph_buckets
        graph_budget = graph_memory_budget_bytes(device_total_bytes(deployment.device))

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
            owns_model_compute = str(device) == str(torch.device(deployment.device))

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
            lane_prefill_catalog = _paged_prefill_graph_buckets(
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
            if execution.cuda_graph:
                if RunKind.AR_DECODE in self._effective_work_variants:
                    expected_resident_executables += len(lane_decode_buckets)
                if (
                    execution.prefill_cuda_graph
                    and RunKind.AR_EXTEND in self._effective_work_variants
                ):
                    expected_resident_executables += (
                        len(lane_prefill_buckets)
                        if packed_model.tensorized_mixed
                        else len(lane_prefill_catalog)
                    )
                if execution.prefill_cuda_graph and {
                    RunKind.DIFFUSION_PREPARE,
                    RunKind.DIFFUSION_STEP,
                }.issubset(self._effective_work_variants):
                    expected_resident_executables += len(
                        {_flow_graph_executable(bucket) for bucket in lane_flow_buckets}
                    )
                    if RunKind.AR_DECODE in self._effective_work_variants:
                        expected_resident_executables += len(
                            {
                                _mixed_flow_graph_executable(bucket)
                                for bucket in lane_mixed_flow_buckets
                            }
                        )
                    expected_resident_executables += len(
                        {
                            _flow_prefix_graph_executable(bucket)
                            for bucket in lane_flow_prefix_buckets
                        }
                    )
            output_slots = int(
                (pipeline_depth if lane is None else lane.max_inflight or pipeline_depth) + 1
            )
            return CudaGraphRunner(
                enabled=execution.cuda_graph,
                prefill_enabled=execution.prefill_cuda_graph,
                cache=packed_model.cache_geometry,
                cache_pool=self.cache_pool,
                attention=attention,
                block_size=deployment.block_size,
                weight_version=self.weight_version,
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

        self._execution = execution
        self.trace = ExecutionTrace(self.architecture)
        devices = (
            (deployment.device,)
            if deployment.generation_device is None
            else (deployment.device, deployment.generation_device)
        )
        runner = (
            ModelRunner(
                packed_model,
                deployment,
                self.trace,
                max_rows=max_staged_rows,
                max_tokens=max_staged_tokens,
                max_text_tokens=max_text_staged_tokens,
                max_blocks_per_row=max_blocks_per_row,
                hidden_size=int(packed_model.hidden_size),
                devices=devices,
                lanes=execution.lanes,
                max_inflight=int(pipeline_depth),
                graph_factory=graph_factory,
            )
            if owns_kv
            else None
        )
        self.runner = runner

        # ExecutionResources is the sole façade passed to operation handlers;
        # every referenced store above remains worker-owned until close.
        self.media_mux, self.media_output_ring = model.create_media_runtime(
            self._info.max_unresolved_ops
        )
        self.execution = create_execution_resources(
            runner=runner,
            model=model,
            deployment=deployment,
            attention=attention,
            requests=self.requests,
            runtime_states=self.runtime_states,
            cache_pool=self.cache_pool,
            req_to_token_pool=self.req_to_token_pool,
            latent_pool=self.latent_pool,
            device_products=self.device_products,
            encoder_cache=self.encoder_cache,
            device_events=self.device_events,
            outputs=self.output_pool,
            cpu_tasks=self.cpu_tasks,
            weights=self.weights,
            mesh=mesh,
            transport=self.transfers.transport,
            tokenizer=tokenizer,
            model_name=self.architecture,
            weight_version=self.weight_version,
            allowed_work_variants=self._effective_work_variants,
            mixed_buckets=self._mixed_buckets,
            trace=self.trace,
            media_mux=self.media_mux,
            media_output_ring=self.media_output_ring,
        )
        self.weight_updater = (
            WeightUpdater(
                self.model,
                architecture=self.architecture,
                scope=self.deployment.model_scope,
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
        """Expose immutable placement, capacity, and model metadata advertised to the scheduler."""

        return self._info

    def _decode_context_blocks(self) -> int:
        """Return the maximum paged-decode context blocks supported by this worker."""

        model = self.model
        if not model.resource_geometry.kv:
            return 0
        deployment = self.deployment
        max_tokens = int(model.text_max_tokens)
        if max_tokens < 1:
            return 0
        blocks = (max_tokens + int(deployment.block_size) - 1) // int(deployment.block_size)
        pool = self.cache_pool
        if pool is None:
            return 0
        return min(blocks, max(0, int(pool.num_pages) - 1))

    def execute(self, batch: Run) -> RunResult:
        """Plan and synchronously resolve one physical run under model-call exclusion."""

        with self._model_call():
            batch = self.plan_run(batch)
            report = execute_batch(self.execution, batch)
            return self._finish_closed_requests(batch, report)

    def plan_run(self, batch: Run) -> Run:
        """Derive the worker-private execution lanes for one physical run."""

        return plan_run(self.execution, batch)

    def supports_run_kind(self, kind: RunKind) -> bool:
        """Return whether this worker can execute one physical run variant."""

        return kind in self._effective_work_variants

    def prepare_execute(self, batch: Run) -> PreparedExecution | None:
        """Validate and stage a scheduler run while retaining all asynchronous input ownership."""

        self._begin_model_call()
        try:
            batch = self.plan_run(batch)
            prepared = prepare_batch(self.execution, batch)
        except BaseException:
            self._end_model_call()
            raise
        if prepared is None:
            if int(self.info.rank.tp_size) > 1:
                prepared = PreparedExecution(batch=batch, transfers=())
            else:
                self._end_model_call()
                return None
        return prepared.bind(
            lambda value: self._finish_closed_requests(
                value.batch,
                execute_prepared(self.execution, value),
            ),
            self._end_model_call,
        )

    def _finish_closed_requests(
        self,
        batch: Run,
        report: RunResult,
    ) -> RunResult:
        """Release model-owned state after terminal request completions become visible."""

        closed = {
            int(command.request_key.request_id)
            for command in batch.commands
            if isinstance(command, Finish)
        }
        for request_id in closed:
            self.retire_request(request_id)
        return report

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

        install_weights(self.execution, weights)
        self.weights = weights
        self.weight_version = weights.version
        self._info = replace(self._info, weight_version=weights.version)

    def warmup(self) -> None:
        """Materialize startup workloads and capture configured CUDA graph shapes."""

        context = packed_warmup.WarmupContext(self)
        packed_warmup.warmup(context)
        self.model.warmup(context)
        complete_startup(self.execution)
        self._mixed_buckets = tuple(
            bucket for bucket in self._mixed_buckets if bucket in self.execution.mixed_buckets
        )

    def drop_request(self, request_id: int) -> None:
        """Release all runtime, cache, latent, product, and transfer state for one request."""

        request_id = int(request_id)
        request = self.requests.peek(request_id)
        drop_execution_request(self.execution, request_id)
        self.device_products.drop_request(request_id)
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

    def retire_request(self, request_id: int) -> None:
        """Retire a completed request after releasing its model and data-plane state."""

        request_id = int(request_id)
        request = self.requests.peek(request_id)
        if request is None or request.retired:
            return
        drop_execution_request(self.execution, request_id)
        self.device_products.drop_request(request_id)
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

    def free_products(self, handles: tuple[int, ...]) -> None:
        """Release scheduler generations from device-product and encoder-cache ownership."""

        generations = tuple(int(handle) for handle in handles)
        self.device_products.release_generations(generations)
        self.encoder_cache.release_generations(generations)

    def close(self) -> None:
        """Release execution, transport, model-state, and distributed resources owned by the worker."""

        runner = self.runner
        if runner is not None:
            runner.synchronize()
        close_execution(self.execution)
        if runner is not None:
            runner.close()
        self.model.synchronize_runtime()
        if self.media_mux is not None:
            self.media_mux.close()
        self.cpu_tasks.close()
        self.transfers.close()
        if self.latent_pool is not None:
            self.latent_pool.close()
        self.encoder_cache.close()
        self.device_products.close()
        self.persistent_buffers.close()
        self.output_pool.close()
        self.device_events.close()

    def set_completion_wake(
        self,
        wake: Callable[[], None],
        wake_on_stream: Callable[[int], None],
    ) -> None:
        """Register host and CUDA-stream callbacks used to wake result polling."""

        self.device_events.set_completion_wake(wake_on_stream)
        self.cpu_tasks.set_completion_wake(wake)
        self.transfers.set_completion_wake(wake)


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
