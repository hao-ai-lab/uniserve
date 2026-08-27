"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from threading import Condition, RLock
from typing import TYPE_CHECKING

import torch

from ..batch import (
    Batch,
    CacheCopy,
    CompletionReport,
    Domain,
    ForwardMode,
    RecoveryPlacement,
    RequestKey,
    SnapshotRef,
)
from ..bootstrap.capabilities import resolve_capabilities
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
from ..capabilities import (
    GraphBucketCapability,
    LaneCapabilities,
    MixedExecutionCapability,
    RequestKind,
    WorkerCapabilities,
)
from ..execution.cuda_graph import CudaGraphRunner
from ..execution.forward_batch import AttentionSelection
from ..execution.model_runner import ModelRunner
from ..execution.step import (
    PreparedExecution,
    close_execution,
    create_execution_resources,
    execute_batch,
    execute_prepared,
    install_weights,
    prepare_batch,
)
from ..execution.step import (
    drop_session as drop_execution_session,
)
from ..execution.trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.math import ceil_div
from ..loader.update import WeightUpdater
from ..loader.weight_set import WeightSet
from ..models.identity import ModelIdentity, architecture_identity
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.diffusion.cfg import build_flow_cfg_plan
from ..nn.mesh import DeviceMesh
from ..recovery.snapshot import SnapshotRecovery
from ..runtime.cache_pool import CachePool
from ..runtime.device_events import DeviceEventPool
from ..runtime.device_products import DeviceProducts
from ..runtime.encoder_cache import EncoderCache
from ..runtime.latent_pool import LatentPool
from ..runtime.req_to_token_pool import ReqToTokenPool
from ..runtime.runtime_states import RuntimeStates
from ..server.cpu_tasks import BoundedCpuTaskPool
from ..server.request_state import RequestTable
from ..transfer.connector import TransferConnector
from . import warmup as packed_warmup
from .warmup import (
    _flow_graph_executable,
    _flow_prefix_graph_executable,
    _FlowGraphBucket,
    _FlowPrefixGraphBucket,
    _has_decode_flow_partition,
    _mixed_flow_graph_executable,
    _paged_prefill_graph_buckets,
    _startup_image_parameters,
)

if TYPE_CHECKING:
    from ..bootstrap.config import WorkerLaunchConfig

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
    snapshot_recovery: SnapshotRecovery | None
    _warmup_kv_pages: dict[tuple[RequestKey, int], list[int]]
    _warmup_prefix_pages: dict[RequestKey, list[int]]
    _warmup_prefix_slots: dict[RequestKey, int]
    _warmup_latent_pages: dict[RequestKey, list[int]]

    @classmethod
    def from_config(cls, config: WorkerLaunchConfig) -> Worker:
        from ..backends.attention import resolve_attention_selection
        from ..backends.triton import configure_triton_toolchain
        from ..bootstrap.model_loader import materialize_worker_model
        from ..bootstrap.plan import resolve_worker_plan
        from ..nn.placement import place_towers
        from ..server.distributed import build_device_mesh
        from ..server.worker_kind import WorkerKind

        plan = resolve_worker_plan(config.worker_kind)
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
        return cls(
            loaded.model,
            mesh=mesh,
            deployment=loaded.deployment,
            attention=resolve_attention_selection(
                loaded.deployment.attention_backend or "auto",
                tuning=config.execution.flashinfer,
                block_size=loaded.deployment.block_size,
            ),
            execution=config.execution,
            tokenizer=loaded.tokenizer,
            allowed_work_variants=plan.allowed_work_variants,
            transfer_backend=config.data_plane.backend,
            cross_process=config.worker_kind is not WorkerKind.FULL,
            architecture_digest=loaded.identity.architecture_digest,
            weight_digest=loaded.identity.weight_digest,
            weights=WeightSet.from_module(loaded.model, digest=loaded.identity.weight_digest),
            weight_sidecars=loaded.weight_sidecars,
            pipeline_depth=config.ipc.pipeline_depth,
            completion_payload_bytes=config.ipc.max_payload_bytes,
            snapshot_dir=config.snapshot_dir,
        )

    def __init__(
        self,
        model: ExecutionModel,
        *,
        mesh: DeviceMesh,
        deployment: WorkerDeployment,
        attention: AttentionSelection,
        execution: ExecutionConfig,
        tokenizer: object | None,
        allowed_work_variants: frozenset[ForwardMode],
        transfer_backend: str = "local",
        cross_process: bool = False,
        architecture_digest: str | None = None,
        weight_digest: str | None = None,
        weights: WeightSet | None = None,
        weight_sidecars: tuple[str, ...] = ("config.json",),
        pipeline_depth: int,
        completion_payload_bytes: int,
        snapshot_dir: str | None = None,
    ) -> None:
        if not isinstance(model, ExecutionModel):
            raise capability_mismatch("worker model must implement ExecutionModel")
        if not isinstance(deployment, WorkerDeployment):
            raise capability_mismatch("model worker requires a worker deployment")
        self.model = model
        self.deployment = deployment
        self._weight_condition = Condition(RLock())
        self._active_model_calls = 0
        self._weight_update_active = False
        installed_weights = (
            WeightSet.from_module(model, digest=weight_digest) if weights is None else weights
        )
        if weight_digest is not None and installed_weights.digest != weight_digest:
            raise capability_mismatch(
                "worker weight identity does not match the installed WeightSet"
            )
        self.weights = installed_weights
        self.weight_digest = installed_weights.digest
        self.identity = ModelIdentity(
            architecture=model.architecture,
            architecture_digest=architecture_digest
            or architecture_identity(model.architecture, {"architecture": model.architecture}),
            weight_digest=self.weight_digest,
        )
        declared = resolve_capabilities(
            model,
            deployment,
            architecture_digest=self.identity.architecture_digest,
            weight_digest=self.weight_digest,
            pipeline_depth=int(pipeline_depth),
            completion_payload_bytes=int(completion_payload_bytes),
        )
        arena = model_arena_capacity(
            model,
            deployment,
            pipeline_depth=int(pipeline_depth),
            completion_payload_bytes=int(completion_payload_bytes),
            num_blocks=int(declared.num_blocks),
            request_pool_size=int(declared.max_request_pool_size),
            num_latent_pages=int(declared.num_latent_pages),
            latent_page_units=int(declared.latent_page_units),
            latent_width=int(declared.latent_width),
            max_latent_feature_bytes=int(declared.max_latent_feature_bytes),
            max_vision_feature_bytes=int(declared.max_vision_feature_bytes),
            bytes_per_token=int(declared.bytes_per_token),
        )
        if snapshot_dir is not None and not model.resource_geometry.kv:
            raise capability_mismatch("session snapshots require model-owned KV resources")
        if snapshot_dir is not None:
            declared = replace(
                declared,
                supported_controls=(
                    *declared.supported_controls,
                    RequestKind.SNAPSHOT_SESSION,
                    RequestKind.RESTORE_SESSION,
                ),
            )
        if int(pipeline_depth) <= 0:
            raise capability_mismatch("worker pipeline depth must be positive")
        implemented_work = model.supported_work
        self._effective_work_variants = allowed_work_variants & implemented_work
        if not self._effective_work_variants:
            raise capability_mismatch(
                f"{type(self).__name__} implements none of the requested work variants "
                f"{sorted(value.value for value in allowed_work_variants)!r}"
            )
        advertised_work = self._effective_work_variants
        if not advertised_work:
            raise capability_mismatch(f"{type(self).__name__} advertises no executable work")
        self._capabilities = replace(
            declared,
            supported_work=tuple(variant for variant in ForwardMode if variant in advertised_work),
            pipeline_depth=int(pipeline_depth),
        )
        owns_kv = bool(model.resource_geometry.kv)
        cache = model.cache_geometry if owns_kv else None
        self.cache_pool = None
        self.req_to_token_pool = None
        max_blocks_per_row = 0
        if cache is not None:
            cache_dtype = getattr(torch, str(cache.dtype).removeprefix("torch."), None)
            if not isinstance(cache_dtype, torch.dtype):
                raise capability_mismatch(f"unsupported cache dtype {cache.dtype!r}")
            max_blocks_per_row = max(
                1,
                ceil_div(int(model.text_max_tokens), int(deployment.block_size)),
            )
            self.cache_pool = CachePool(
                num_layers=int(cache.num_layers),
                num_pages=int(self._capabilities.num_blocks),
                page_size=int(self._capabilities.block_size),
                num_kv_heads=int(cache.num_kv_heads),
                head_dim=int(cache.head_dim),
                device=deployment.device,
                dtype=cache_dtype,
                store_dtype=cache.store_dtype,
                group_ranges=(
                    tuple(
                        (int(group.block_offset), int(group.num_blocks))
                        for group in self._capabilities.groups
                    )
                    if self._capabilities.groups
                    else None
                ),
            )
            if (
                ForwardMode.GEN_FLOW in self._effective_work_variants
                and not _supports_flow_attention(
                    attention,
                    cache,
                    self.cache_pool,
                    torch.device(deployment.device),
                )
            ):
                raise capability_mismatch(
                    "image generation requires paged-prefix plus dense-current attention"
                )
            self.req_to_token_pool = ReqToTokenPool(
                group_count=self.cache_pool.group_count,
                request_pool_size=int(self._capabilities.max_request_pool_size),
                max_blocks_per_request=max_blocks_per_row,
                block_size=int(self._capabilities.block_size),
                device=deployment.device,
                staging_depth=int(pipeline_depth),
            )
            model.bind_cache_pool(self.cache_pool, attention)
        self.requests = RequestTable(int(self._capabilities.max_request_pool_size))
        torch_dtype = getattr(
            torch,
            str(deployment.model_dtype).removeprefix("torch."),
            None,
        )
        if not isinstance(torch_dtype, torch.dtype):
            raise capability_mismatch(f"unsupported model dtype {deployment.model_dtype!r}")
        self.runtime_states = (
            RuntimeStates(
                request_pool_size=int(self._capabilities.max_request_pool_size),
                vocab_size=int(model.vocab_size),
                continuation_width=1,
                device=deployment.device,
                logits_dtype=torch_dtype,
                valid_cache_lengths=self.req_to_token_pool.verified_lens,
            )
            if self.req_to_token_pool is not None
            else None
        )
        flow = model.generation
        latent_dtype = getattr(
            torch,
            str(self._capabilities.latent_dtype).removeprefix("torch."),
            None,
        )
        if flow is not None and not isinstance(latent_dtype, torch.dtype):
            raise capability_mismatch(
                f"unsupported latent dtype {self._capabilities.latent_dtype!r}"
            )
        if flow is None:
            self.latent_pool = None
        else:
            assert isinstance(latent_dtype, torch.dtype)
            self.latent_pool = LatentPool(
                request_pool_size=int(self._capabilities.max_request_pool_size),
                num_pages=int(self._capabilities.num_latent_pages),
                page_units=int(self._capabilities.latent_page_units),
                latent_width=int(self._capabilities.latent_width),
                dtype=latent_dtype,
                device=deployment.generation_device or deployment.device,
            )
        if (
            self.latent_pool is not None
            and self.latent_pool.persistent_bytes != arena.latent_pool_bytes
        ):
            raise RuntimeError("latent pool allocation disagrees with its exact capacity plan")
        owner_devices = tuple(
            dict.fromkeys(
                (
                    deployment.device,
                    deployment.generation_device or deployment.device,
                )
            )
        )
        self.device_events = DeviceEventPool()
        self.device_products = DeviceProducts(
            capacity=arena.device_products,
            byte_capacity=arena.device_product_bytes,
            event_pool=self.device_events,
        )
        self.encoder_cache = EncoderCache(
            entry_capacity=int(model.resource_geometry.encoder_cache_entries),
            max_entry_bytes=max(
                1,
                int(self._capabilities.max_latent_feature_bytes),
                int(self._capabilities.max_vision_feature_bytes),
            ),
            devices=owner_devices,
            event_pool=self.device_events,
        )
        self.cpu_tasks = BoundedCpuTaskPool(
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
            int(self._capabilities.max_batch_operations),
            int(self._capabilities.max_request_pool_size),
        )
        max_staged_rows = max_rows * (1 if flow is None else int(flow.max_cfg_branches))
        max_text_staged_tokens = int(self._capabilities.max_batch_tokens)
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
            int(self._capabilities.max_batch_tokens),
            (
                int(self._capabilities.max_batch_tokens)
                if prefill_lane is None
                else int(prefill_lane.max_batch_tokens or self._capabilities.max_batch_tokens)
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
            and int(value) < int(self._capabilities.num_blocks)
        )
        prefill_capacity = (
            min(
                int(self._capabilities.max_batch_tokens),
                int(model.text_max_tokens),
                max(0, int(self._capabilities.num_blocks) - 1) * int(deployment.block_size),
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
                <= int(self._capabilities.latent_capacity_units)
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
                or not model.tensorized_mixed
                or not _has_decode_flow_partition(execution.lanes)
                or not {
                    ForwardMode.TOKEN_DECODE,
                    ForwardMode.GEN_FLOW,
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
            MixedExecutionCapability(
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
        if flow is not None and model.tensorized_mixed and flow_graph_buckets:
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
        self._capabilities = replace(
            self._capabilities,
            mixed_buckets=mixed_flow_graph_buckets,
        )
        graph_budget = graph_memory_budget_bytes(device_total_bytes(deployment.device))

        def graph_factory(
            device: torch.device,
            lane: LaneConfig | None,
            stream: torch.cuda.Stream | None,
            expected_context: int | None,
        ) -> CudaGraphRunner:
            domains = tuple(Domain) if lane is None else lane.domains
            owns_model_compute = str(device) == str(torch.device(deployment.device))
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
            expected_resident_executables = 0
            if execution.cuda_graph:
                if ForwardMode.TOKEN_DECODE in self._effective_work_variants:
                    expected_resident_executables += len(lane_decode_buckets)
                if (
                    execution.prefill_cuda_graph
                    and ForwardMode.TOKEN_EXTEND in self._effective_work_variants
                ):
                    expected_resident_executables += (
                        len(lane_prefill_buckets)
                        if model.tensorized_mixed
                        else len(lane_prefill_catalog)
                    )
                if execution.prefill_cuda_graph and {
                    ForwardMode.GEN_TRANSITION,
                    ForwardMode.GEN_FLOW,
                }.issubset(self._effective_work_variants):
                    expected_resident_executables += len(
                        {_flow_graph_executable(bucket) for bucket in lane_flow_buckets}
                    )
                    if ForwardMode.TOKEN_DECODE in self._effective_work_variants:
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
                cache=model.cache_geometry,
                cache_pool=self.cache_pool,
                attention=attention,
                block_size=deployment.block_size,
                weight_digest=self.weight_digest,
                memory_budget_bytes=graph_budget,
                decode_batch_sizes=lane_decode_buckets,
                decode_predicates=(
                    self.runtime_states.predicates
                    if owns_model_compute and Domain.DECODE in domains
                    else None
                ),
                decode_context_blocks=self._decode_context_blocks(),
                packed_context_blocks=max_blocks_per_row,
                prefill_token_sizes=(() if model.tensorized_mixed else lane_prefill_buckets),
                prefill_row_sizes=lane_prefill_row_sizes,
                stream=stream,
                expected_context=expected_context,
                expected_resident_executables=expected_resident_executables,
                output_slot_count=output_slots,
            )

        self._execution = execution
        self.trace = ExecutionTrace(self.identity.architecture_digest)
        devices = (
            (deployment.device,)
            if deployment.generation_device is None
            else (deployment.device, deployment.generation_device)
        )
        runner = (
            ModelRunner(
                model,
                deployment,
                self.trace,
                max_rows=max_staged_rows,
                max_tokens=max_staged_tokens,
                max_text_tokens=max_text_staged_tokens,
                max_blocks_per_row=max_blocks_per_row,
                hidden_size=int(model.hidden_size),
                devices=devices,
                lanes=execution.lanes,
                max_inflight=int(pipeline_depth),
                graph_factory=graph_factory,
            )
            if owns_kv
            else None
        )
        if execution.lanes and runner is not None:
            lane_by_id = {lane.lane_id: lane for lane in execution.lanes}
            lane_capabilities: list[LaneCapabilities] = []
            for partition in runner.partitions:
                if partition.lane_id is None:
                    continue
                lane = lane_by_id[partition.lane_id]
                max_operations = min(
                    int(self._capabilities.max_batch_operations),
                    int(lane.max_batch_operations or self._capabilities.max_batch_operations),
                )
                max_tokens = min(
                    int(self._capabilities.max_batch_tokens),
                    int(lane.max_batch_tokens or self._capabilities.max_batch_tokens),
                )
                buckets: list[GraphBucketCapability] = []
                if execution.cuda_graph and Domain.DECODE in lane.domains:
                    buckets.extend(
                        GraphBucketCapability(
                            phase="text_decode",
                            batch_size=int(batch_size),
                            token_bucket=int(batch_size),
                            attention_form="paged_decode",
                            height=0,
                            width=0,
                            cfg_branches=1,
                        )
                        for batch_size in decode_graph_batch_sizes
                        if int(batch_size) <= max_operations
                    )
                if (
                    execution.cuda_graph
                    and execution.prefill_cuda_graph
                    and Domain.PREFILL in lane.domains
                ):
                    paged_catalog = (
                        ()
                        if model.tensorized_mixed
                        else _paged_prefill_graph_buckets(
                            prefill_graph_token_sizes,
                            prefill_graph_row_sizes,
                            max_rows=max_operations,
                            max_tokens=max_tokens,
                        )
                    )
                    buckets.extend(
                        GraphBucketCapability(
                            phase="text_prefill",
                            batch_size=item.row_bucket,
                            token_bucket=item.token_bucket,
                            attention_form="paged_varlen",
                            height=0,
                            width=0,
                            cfg_branches=1,
                        )
                        for item in paged_catalog
                    )
                    if model.tensorized_mixed:
                        buckets.extend(
                            GraphBucketCapability(
                                phase="text_prefill",
                                batch_size=1,
                                token_bucket=int(token_size),
                                attention_form="packed",
                                height=0,
                                width=0,
                                cfg_branches=1,
                            )
                            for token_size in prefill_graph_token_sizes
                            if int(token_size) <= max_tokens
                        )
                if (
                    execution.cuda_graph
                    and execution.prefill_cuda_graph
                    and Domain.FLOW in lane.domains
                    and flow is not None
                ):
                    buckets.extend(
                        GraphBucketCapability(
                            phase="text_prefill",
                            batch_size=bucket.rows * len(bucket.prefix_lengths),
                            token_bucket=bucket.rows * sum(bucket.prefix_lengths),
                            attention_form="packed",
                            height=0,
                            width=0,
                            cfg_branches=bucket.cfg_branches,
                            layout="flow_prefix",
                        )
                        for bucket in flow_prefix_graph_buckets
                        if bucket.rows <= max_operations
                        and bucket.rows * sum(bucket.prefix_lengths) <= max_tokens
                    )
                    buckets.extend(
                        GraphBucketCapability(
                            phase="denoise",
                            batch_size=bucket.rows,
                            token_bucket=0,
                            attention_form="packed",
                            height=bucket.height,
                            width=bucket.width,
                            cfg_branches=bucket.cfg_branches,
                            layout="flow",
                        )
                        for bucket in flow_graph_buckets
                        if bucket.rows <= max_operations
                    )
                    if ForwardMode.TOKEN_DECODE in self._effective_work_variants and {
                        Domain.DECODE,
                        Domain.FLOW,
                    } <= set(lane.domains):
                        buckets.extend(
                            GraphBucketCapability(
                                phase="denoise",
                                batch_size=bucket.decode_rows + bucket.flow_rows,
                                token_bucket=bucket.decode_rows,
                                attention_form="packed",
                                height=bucket.height,
                                width=bucket.width,
                                cfg_branches=bucket.cfg_branches,
                                layout="decode_flow",
                            )
                            for bucket in mixed_flow_graph_buckets
                            if bucket.decode_rows + bucket.flow_rows <= max_operations
                        )
                lane_capabilities.append(
                    LaneCapabilities(
                        lane_id=lane.lane_id,
                        domains=lane.domains,
                        resolved_sm_count=partition.sm_count,
                        kv_capacity_tokens=lane.kv_capacity_tokens,
                        latent_capacity_units=lane.latent_capacity_units,
                        max_batch_operations=max_operations,
                        max_batch_tokens=max_tokens,
                        max_inflight=int(lane.max_inflight or pipeline_depth),
                        graph_buckets=tuple(buckets),
                        eager_max_batch_operations=max_operations,
                        eager_max_batch_tokens=max_tokens,
                    )
                )
            self._capabilities = replace(self._capabilities, lanes=tuple(lane_capabilities))
        self.runner = runner
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
            cpu_tasks=self.cpu_tasks,
            weights=self.weights,
            mesh=mesh,
            transport=self.transfers.transport,
            tokenizer=tokenizer,
            architecture_digest=self.identity.architecture_digest,
            weight_digest=self.weight_digest,
            allowed_work_variants=self._effective_work_variants,
            mixed_buckets=self._capabilities.mixed_buckets,
            trace=self.trace,
        )
        self._warmup_kv_pages = {}
        self._warmup_prefix_pages = {}
        self._warmup_prefix_slots = {}
        self._warmup_latent_pages = {}
        self._warmup_step_id = 0
        self.snapshot_recovery = None
        if snapshot_dir is not None:
            caps = self._capabilities
            self.snapshot_recovery = SnapshotRecovery(
                snapshot_dir,
                model_identity=self.identity.architecture_digest,
                weight_digest=self.weight_digest,
                topology={
                    "rank": caps.rank.to_mapping(),
                    "model_scope": deployment.model_scope,
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "num_layers": caps.num_layers,
                    "max_request_pool_size": caps.max_request_pool_size,
                    "latent_page_units": caps.latent_page_units,
                    "num_latent_pages": caps.num_latent_pages,
                    "latent_width": caps.latent_width,
                    "latent_dtype": caps.latent_dtype,
                    "latent_downsample": caps.latent_downsample,
                },
                device=deployment.device,
                requests=self.requests,
                cache_pool=self.cache_pool,
                req_to_token_pool=self.req_to_token_pool,
                cache_publications=self.execution.cache_publications,
                latent_pool=self.latent_pool,
                device_products=self.device_products,
                encoder_cache=self.encoder_cache,
                runtime_states=self.runtime_states,
                transport=self.transfers.transport,
            )
        self.weight_updater = WeightUpdater(
            self.model,
            architecture=self.identity.architecture,
            scope=self.deployment.model_scope,
            sidecars=weight_sidecars,
            weights=self.weights,
            publish=self._publish_weight_set,
            exclusive=self._exclusive_weight_update,
            gather_rank_digests=self._gather_rank_weight_digests,
        )

    @property
    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    def _decode_context_blocks(self) -> int:
        model = self.model
        deployment = self.deployment
        max_tokens = int(model.text_max_tokens)
        if max_tokens < 1:
            return 0
        blocks = (max_tokens + int(deployment.block_size) - 1) // int(deployment.block_size)
        pool = self.cache_pool
        if pool is None:
            return 0
        return min(blocks, max(0, int(pool.num_pages) - 1))

    def execute(self, batch: Batch) -> CompletionReport:
        with self._model_call():
            return execute_batch(self.execution, batch)

    def prepare_execute(self, batch: Batch) -> PreparedExecution | None:
        self._begin_model_call()
        try:
            prepared = prepare_batch(self.execution, batch)
        except BaseException:
            self._end_model_call()
            raise
        if prepared is None:
            self._end_model_call()
            return None
        return prepared.bind(
            lambda value: execute_prepared(self.execution, value),
            self._end_model_call,
        )

    def execute_prepared(self, prepared: PreparedExecution) -> CompletionReport:
        if not isinstance(prepared, PreparedExecution):
            raise invalid_descriptor("prepared execution has an invalid type")
        return prepared.resolve()

    @contextmanager
    def _model_call(self) -> Generator[None, None, None]:
        self._begin_model_call()
        try:
            yield
        finally:
            self._end_model_call()

    def _begin_model_call(self) -> None:
        with self._weight_condition:
            updater = getattr(self, "weight_updater", None)
            if updater is not None and updater.unhealthy:
                raise RuntimeError("worker weight graph is unhealthy")
            if self._weight_update_active:
                raise RuntimeError("worker weight update is draining model execution")
            self._active_model_calls += 1

    def _end_model_call(self) -> None:
        with self._weight_condition:
            if self._active_model_calls < 1:
                raise RuntimeError("worker model-call accounting underflow")
            self._active_model_calls -= 1
            self._weight_condition.notify_all()

    @contextmanager
    def _exclusive_weight_update(self) -> Generator[None, None, None]:
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
        install_weights(self.execution, weights)
        self.weights = weights
        self.weight_digest = weights.digest
        self.identity = ModelIdentity(
            architecture=self.identity.architecture,
            architecture_digest=self.identity.architecture_digest,
            weight_digest=weights.digest,
        )
        self._capabilities = replace(self._capabilities, weight_digest=weights.digest)
        if self.snapshot_recovery is not None:
            self.snapshot_recovery.rebind_weight_digest(weights.digest)

    def _gather_rank_weight_digests(self, rank_digest: str) -> Sequence[str]:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return (rank_digest,)
        gathered: list[object] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, rank_digest)
        if any(not isinstance(value, str) for value in gathered):
            raise RuntimeError("tensor-parallel ranks did not publish weight digests")
        return tuple(str(value) for value in gathered)

    def warmup(self) -> None:
        packed_warmup.warmup(self)

    def drop_session(self, session_id: int) -> None:
        session_id = int(session_id)
        session = self.requests.peek(session_id)
        drop_execution_session(self.execution, session_id)
        self.device_products.drop_session(session_id)
        if session is not None and self.latent_pool is not None:
            self.latent_pool.release_slots((int(session.request_pool_idx),))
        self.requests.drop(session_id)
        if session is not None:
            for group_id in range(self.cache_pool.group_count):
                self._warmup_kv_pages.pop((session.request_key, group_id), None)
            self._warmup_prefix_pages.pop(session.request_key, None)
            self._warmup_prefix_slots.pop(session.request_key, None)
            self._warmup_latent_pages.pop(session.request_key, None)
        if self.snapshot_recovery is not None:
            self.snapshot_recovery.drop_session(session_id)
        if session is not None:
            self.trace.emit(
                ExecutionPhase.CLEANUP,
                (
                    OperationTrace(
                        session_id=session.session_id,
                        epoch=session.epoch,
                        op_id=0 if session.last_op_id is None else session.last_op_id,
                        version=session.version,
                    ),
                ),
            )

    def copy_kv(self, copies: tuple[CacheCopy, ...]) -> None:
        for group_id in {copy.group_id for copy in copies}:
            selected = tuple(copy for copy in copies if copy.group_id == group_id)
            self.cache_pool.copy_pages(
                group_id,
                tuple(copy.source_page for copy in selected),
                tuple(copy.destination_page for copy in selected),
            )

    def release_products(self, handles: tuple[int, ...]) -> None:
        generations = tuple(int(handle) for handle in handles)
        self.device_products.release_generations(generations)
        self.encoder_cache.release_generations(generations)

    def snapshot_session(self, placement: RecoveryPlacement) -> SnapshotRef:
        if self.snapshot_recovery is None:
            raise capability_mismatch("this worker has no configured snapshot recovery")
        runner = self.runner
        if runner is None:
            raise capability_mismatch("session snapshots require packed-forward execution")
        runner.synchronize()
        return self.snapshot_recovery.snapshot_session(placement)

    def restore_session(
        self,
        reference: SnapshotRef,
        placement: RecoveryPlacement,
    ) -> None:
        if self.snapshot_recovery is None:
            raise capability_mismatch("this worker has no configured snapshot recovery")
        runner = self.runner
        if runner is None:
            raise capability_mismatch("session snapshots require packed-forward execution")
        runner.synchronize()
        self.snapshot_recovery.restore(reference, placement)

    def resource_pressure(self) -> list[dict[str, object]]:
        caps = self._capabilities
        counts = {
            "image_latent": (
                0 if self.latent_pool is None else self.latent_pool.resident_byte_count()
            ),
            "encoder_output": self.encoder_cache.resident_entries,
        }
        totals = {
            "image_latent": (
                0 if self.latent_pool is None else int(self.latent_pool.capacity_bytes)
            ),
            "encoder_output": int(caps.encoder_cache_budget),
        }
        return [
            _pressure(value.value, counts[value.value], totals[value.value])
            for value in caps.resource_classes
            if value.value in counts
        ]

    def close(self) -> None:
        runner = self.runner
        if runner is not None:
            runner.synchronize()
        close_execution(self.execution)
        if runner is not None:
            runner.close()
        self.cpu_tasks.close()
        self.transfers.close()
        if self.latent_pool is not None:
            self.latent_pool.close()
        self.encoder_cache.close()
        self.device_products.close()
        self.device_events.close()

    def set_completion_wake(
        self,
        wake: Callable[[], None],
        wake_on_stream: Callable[[int], None],
    ) -> None:
        self.device_events.set_completion_wake(wake_on_stream)
        self.cpu_tasks.set_completion_wake(wake)
        self.transfers.set_completion_wake(wake)


def _supports_flow_attention(
    selection: AttentionSelection,
    geometry: object,
    pool: CachePool,
    device: torch.device,
) -> bool:
    if not pool.supports_paged_attention_storage:
        return False
    head_dim = int(getattr(geometry, "head_dim"))
    for provider in selection.providers:
        capabilities = provider.capabilities()
        if not capabilities.available or not capabilities.segmented_attention:
            continue
        if head_dim < int(capabilities.min_head_dim):
            continue
        multiple = max(1, int(capabilities.paged_block_size_multiple))
        if pool.block_size % max(1, multiple) != 0:
            continue
        if not capabilities.supports_trunk_geometry(head_dim, head_dim, head_dim):
            continue
        if capabilities.cuda_only and device.type != "cuda":
            continue
        minimum = capabilities.min_cuda_capability
        if minimum is not None:
            if device.type != "cuda":
                continue
            major, minor = torch.cuda.get_device_capability(device)
            if (int(major), int(minor)) < (int(minimum[0]), int(minimum[1])):
                continue
        return True
    return False


def _pressure(resource_class: str, used: int, total: int) -> dict[str, object]:
    if used < 0 or total < 0 or used > total:
        raise RuntimeError(
            f"resource pressure invariant failed for {resource_class}: used={used}, total={total}"
        )
    return {
        "class": resource_class,
        "total": total,
        "used": used,
        "evictable": 0,
        "free": total - used,
    }


__all__ = ["Worker"]
