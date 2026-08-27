"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Condition, RLock
from typing import TYPE_CHECKING

import torch

from ..batch import (
    Admission,
    AttentionRegime,
    Batch,
    BatchPartition,
    BlockTable,
    CacheCopy,
    CachePageAllocation,
    CompletionReport,
    Domain,
    ExecutionCapability,
    ForwardMode,
    ImageParams,
    LatentPlacement,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RecoveryPlacement,
    RequestKey,
    RowGeometry,
    SnapshotRef,
    StorageClass,
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
    complete_startup,
    create_execution_resources,
    execute_batch,
    execute_prepared,
    execute_startup,
    install_weights,
    parent_runtime,
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
from ..models.generation import GenerationPipeline
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
from ..server.completion import (
    completion_report_ready,
    finalize_completion_report,
)
from ..server.cpu_tasks import BoundedCpuTaskPool
from ..server.request_state import RequestTable
from ..transfer.connector import TransferConnector

if TYPE_CHECKING:
    from ..bootstrap.config import WorkerLaunchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FlowGraphBucket:
    rows: int
    height: int
    width: int
    cfg_branches: int


@dataclass(frozen=True, slots=True)
class _FlowPrefixGraphBucket:
    rows: int
    cfg_branches: int
    prefix_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PagedPrefillGraphBucket:
    token_bucket: int
    row_bucket: int
    live_rows: int


def _flow_graph_executable(bucket: _FlowGraphBucket) -> tuple[object, ...]:
    return (
        "flow",
        bucket.rows * bucket.cfg_branches,
        bucket.height,
        bucket.width,
    )


def _flow_prefix_graph_executable(bucket: _FlowPrefixGraphBucket) -> tuple[object, ...]:
    return (
        "flow_prefix",
        bucket.prefix_lengths * bucket.rows,
    )


def _mixed_flow_graph_executable(
    bucket: MixedExecutionCapability,
) -> tuple[object, ...]:
    return (
        "decode_flow",
        bucket.decode_rows,
        bucket.flow_rows * bucket.cfg_branches,
        bucket.height,
        bucket.width,
    )


def _paged_prefill_graph_buckets(
    token_sizes: Sequence[int],
    row_sizes: Sequence[int],
    *,
    max_rows: int,
    max_tokens: int,
) -> tuple[_PagedPrefillGraphBucket, ...]:
    buckets: list[_PagedPrefillGraphBucket] = []
    minimum_rows = 1
    for row_bucket in sorted({int(value) for value in row_sizes if int(value) > 1}):
        if minimum_rows > int(max_rows):
            break
        minimum_tokens = minimum_rows if minimum_rows == 1 else minimum_rows + 1
        for token_bucket in sorted(
            {int(value) for value in token_sizes if minimum_tokens <= int(value) <= int(max_tokens)}
        ):
            buckets.append(
                _PagedPrefillGraphBucket(
                    token_bucket=token_bucket,
                    row_bucket=row_bucket,
                    live_rows=minimum_rows,
                )
            )
        minimum_rows = row_bucket
    return tuple(buckets)


def _startup_image_parameters(
    cfg_branches: int,
    *,
    steps: int,
    height: int,
    width: int,
) -> ImageParams:
    scales = {
        1: (1.0, 1.0),
        2: (4.0, 1.0),
        3: (4.0, 2.0),
    }
    try:
        text_scale, image_scale = scales[int(cfg_branches)]
    except KeyError as error:
        raise invalid_descriptor(
            "flow CFG branch geometry exceeds the concrete branch set"
        ) from error
    return ImageParams(
        steps=int(steps),
        cfg_text_scale=text_scale,
        cfg_img_scale=image_scale,
        height=int(height),
        width=int(width),
        seed=0,
    )


def _has_decode_flow_partition(lanes: tuple[LaneConfig, ...]) -> bool:
    """Return whether one physical partition can run tensorized decode+flow."""

    required = {Domain.DECODE, Domain.FLOW}
    return not lanes or any(required <= set(lane.domains) for lane in lanes)


def _warmup_batch(
    *,
    step_id: int,
    admissions: tuple[Admission, ...],
    operations: tuple[Operation, ...],
    block_tables: dict[tuple[RequestKey, int], tuple[BlockTable, ...]],
    new_cache_pages: dict[tuple[RequestKey, int], tuple[CachePageAllocation, ...]],
    forward_rows: dict[tuple[RequestKey, int], tuple[RowGeometry, ...]],
    latent_placements: dict[tuple[RequestKey, int], LatentPlacement],
    input_products: tuple[ProductPayload, ...] = (),
    tensorized_mixed: bool = False,
) -> Batch:
    groups: list[tuple[Domain, int, list[Operation]]] = []
    for operation in operations:
        existing = next(
            (
                members
                for domain, route, members in groups
                if domain is operation.domain and route == operation.route
            ),
            None,
        )
        if existing is None:
            groups.append((operation.domain, operation.route, [operation]))
        else:
            existing.append(operation)
    partitions = tuple(
        BatchPartition(
            partition_id=index,
            submission_group=1 if tensorized_mixed else index,
            collective_seq=max(
                1,
                int(step_id) * 16 + (1 if tensorized_mixed else index),
            ),
            domain=domain,
            route=route,
            execution=(
                ExecutionCapability.TENSORIZED_MIXED
                if tensorized_mixed
                else ExecutionCapability.DOMAIN_HOMOGENEOUS
            ),
            attention=AttentionRegime.HYBRID,
            shape_class=0,
            operations=tuple(members),
            block_tables=tuple(
                table
                for operation in members
                for table in block_tables.get(
                    (operation.request_key, operation.op_id),
                    (),
                )
            ),
            new_cache_pages=tuple(
                allocation
                for operation in members
                for allocation in new_cache_pages.get(
                    (operation.request_key, operation.op_id),
                    (),
                )
            ),
            forward_rows=tuple(
                replace(row, operation_index=operation_index)
                for operation_index, operation in enumerate(members)
                for row in forward_rows.get((operation.request_key, operation.op_id), ())
            ),
            latent_placements=tuple(
                latent_placements[(operation.request_key, operation.op_id)]
                for operation in members
                if operation.work
                in {
                    ForwardMode.GEN_TRANSITION,
                    ForwardMode.GEN_FLOW,
                    ForwardMode.MATERIALIZE,
                }
            ),
        )
        for index, (domain, route, members) in enumerate(groups, start=1)
    )
    return Batch(
        step_id=step_id,
        admissions=admissions,
        partitions=partitions,
        input_products=input_products,
    )


def _warmup_token_outputs(
    request_key: RequestKey,
    op_id: int,
    first_generation: int,
    *,
    finish_candidate: bool = False,
) -> tuple[ProductRef, ...]:
    from ..batch import (
        DType,
        PointRange,
        ProductKind,
        ShapeBound,
        StorageClass,
    )

    definitions = [(0, ProductKind.TOKEN, DType.U32, ShapeBound())]
    if finish_candidate:
        definitions.append((4, ProductKind.FINISH, DType.U8, ShapeBound()))
    return tuple(
        ProductRef(
            request_key=request_key,
            producer_op_id=op_id,
            output_index=output_index,
            generation=first_generation + generation_offset,
            kind=kind,
            storage_class=StorageClass.DEVICE_TENSOR,
            dtype=dtype,
            shape_bound=shape,
            point_range=PointRange(base_point=0, max_points=1),
        )
        for generation_offset, (output_index, kind, dtype, shape) in enumerate(definitions)
    )


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

    def _execute_warmup(
        self,
        batch: Batch,
        *,
        retain_device_outputs: bool = False,
        catalog_graphs: bool = True,
    ) -> CompletionReport:
        report = execute_startup(self.execution, batch, catalog_graphs=catalog_graphs)
        while not completion_report_ready(report):
            time.sleep(0.00005)
        finalized = finalize_completion_report(report)
        device_generations = tuple(
            int(output.generation)
            for operation in batch.operations
            for output in operation.outputs
            if output.storage_class is StorageClass.DEVICE_TENSOR
        )
        failures = tuple(
            completion
            for completion in finalized.completions
            if completion.status is OpStatus.ERROR
        )
        if failures or not retain_device_outputs:
            self.release_products(device_generations)
        if failures:
            details = ", ".join(
                f"session={completion.request_key.session_id} op={completion.op_id} "
                f"code={completion.error_code.value if completion.error_code is not None else 'internal'}"
                for completion in failures
            )
            raise RuntimeError(f"startup warmup execution failed: {details}")
        return finalized

    def _build_warmup_batch(
        self,
        *,
        admissions: tuple[Admission, ...],
        operations: tuple[Operation, ...],
        input_products: tuple[ProductPayload, ...] = (),
        tensorized_mixed: bool = False,
        image_geometry: tuple[int, int] | None = None,
    ) -> Batch:
        self._warmup_step_id += 1
        admissions_by_key = {admission.request_key: admission for admission in admissions}
        occupied_blocks = {page for pages in self._warmup_kv_pages.values() for page in pages}
        request_pool_indices: dict[RequestKey, int] = {}
        block_tables: dict[tuple[RequestKey, int], tuple[BlockTable, ...]] = {}
        new_cache_pages: dict[tuple[RequestKey, int], tuple[CachePageAllocation, ...]] = {}
        forward_rows: dict[tuple[RequestKey, int], tuple[RowGeometry, ...]] = {}
        latent_placements: dict[tuple[RequestKey, int], LatentPlacement] = {}
        for operation in operations:
            session = self.requests.peek(int(operation.request_key.session_id))
            admission = admissions_by_key.get(operation.request_key)
            if session is None and admission is None:
                raise invalid_descriptor("warmup operation has no request-pool binding")
            if session is None:
                assert admission is not None
                request_pool_indices[operation.request_key] = admission.request_pool_idx
            else:
                request_pool_indices[operation.request_key] = session.request_pool_idx
            if operation.work not in {
                ForwardMode.TOKEN_EXTEND,
                ForwardMode.TOKEN_DECODE,
                ForwardMode.TOKEN_VERIFY,
                ForwardMode.DRAFT,
                ForwardMode.TRANSFER_KV_PUBLISH,
                ForwardMode.TRANSFER_KV_INSTALL,
                ForwardMode.GEN_TRANSITION,
                ForwardMode.GEN_FLOW,
            }:
                continue
            if (
                session is None
                and admission is not None
                and admission.und is not None
                and admission.und.initial_position != 0
            ):
                raise invalid_descriptor("warmup KV admission requires an empty prefix")
            visible = 0
            if session is not None:
                runtime = parent_runtime(self.execution, operation, session)
                visible = int(runtime.kv_visible_len)
            input_length = (
                int(operation.bounds.max_tokens)
                if operation.work
                in {
                    ForwardMode.TOKEN_EXTEND,
                    ForwardMode.TOKEN_DECODE,
                    ForwardMode.TOKEN_VERIFY,
                    ForwardMode.DRAFT,
                }
                else 0
            )
            tables: list[BlockTable] = []
            allocations: list[CachePageAllocation] = []
            for group_id in range(self.cache_pool.group_count):
                lease_key = (operation.request_key, group_id)
                block_table = self._warmup_kv_pages.setdefault(lease_key, [])
                target_pages = ceil_div(
                    visible + input_length,
                    int(self.cache_pool.block_size),
                )
                missing = target_pages - len(block_table)
                if missing < 0:
                    raise invalid_descriptor("warmup operation regresses its KV capacity")
                allocated = tuple(
                    candidate
                    for candidate in self.cache_pool.page_ids(group_id)
                    if candidate not in occupied_blocks
                )[:missing]
                if len(allocated) != missing:
                    raise invalid_descriptor("warmup KV placement exceeds resident capacity")
                block_table.extend(allocated)
                occupied_blocks.update(allocated)
                tables.append(
                    BlockTable(
                        request_pool_idx=request_pool_indices[operation.request_key],
                        group_id=group_id,
                        page_ids=tuple(block_table),
                        allocated_tokens=len(block_table) * self.cache_pool.block_size,
                    )
                )
                if allocated:
                    allocations.append(
                        CachePageAllocation(
                            request_pool_idx=request_pool_indices[operation.request_key],
                            group_id=group_id,
                            page_ids=allocated,
                        )
                    )
            identity = (operation.request_key, operation.op_id)
            block_tables[identity] = tuple(tables)
            new_cache_pages[identity] = tuple(allocations)
            if input_length > 0:
                forward_rows[identity] = (
                    RowGeometry(
                        operation_index=0,
                        request_pool_index=request_pool_indices[operation.request_key],
                        seq_len=visible,
                        query_len=input_length,
                    ),
                )
        height, width = image_geometry or self._warmup_image_geometry()
        latent_units = max(
            1,
            (height // max(1, int(self._capabilities.latent_downsample)))
            * (width // max(1, int(self._capabilities.latent_downsample))),
        )
        page_units = int(self._capabilities.latent_page_units)
        latent_page_count = (latent_units + page_units - 1) // page_units if page_units > 0 else 0
        occupied_latent_pages = {
            page for pages in self._warmup_latent_pages.values() for page in pages
        }
        for operation in operations:
            if operation.work not in {
                ForwardMode.GEN_TRANSITION,
                ForwardMode.GEN_FLOW,
            } and not any(product.kind is ProductKind.LATENT for product in operation.inputs):
                continue
            page_table = self._warmup_latent_pages.setdefault(operation.request_key, [])
            missing = latent_page_count - len(page_table)
            if missing < 0:
                raise invalid_descriptor("warmup latent placement regresses its physical extent")
            allocated = tuple(
                page
                for page in range(1, int(self._capabilities.num_latent_pages))
                if page not in occupied_latent_pages
            )[:missing]
            if len(allocated) != missing:
                raise invalid_descriptor("warmup latent placement exceeds resident capacity")
            page_table.extend(allocated)
            occupied_latent_pages.update(allocated)
            session = self.requests.peek(int(operation.request_key.session_id))
            start_step = 0 if session is None else int(session.flow_step)
            latent_placements[(operation.request_key, operation.op_id)] = LatentPlacement(
                request_key=operation.request_key,
                op_id=operation.op_id,
                page_table=tuple(page_table),
                latent_units=latent_units,
                height=height,
                width=width,
                start_step=start_step,
                step_count=(
                    int(operation.bounds.max_tokens)
                    if operation.work is ForwardMode.GEN_FLOW
                    else 0
                ),
            )
            if operation.work is ForwardMode.GEN_FLOW:
                extra_tables, extra_allocations, flow_rows = self._warmup_flow_tables(
                    operation,
                    request_pool_indices[operation.request_key],
                    height,
                    width,
                )
                identity = (operation.request_key, operation.op_id)
                block_tables[identity] = (*block_tables.get(identity, ()), *extra_tables)
                new_cache_pages[identity] = (
                    *new_cache_pages.get(identity, ()),
                    *extra_allocations,
                )
                forward_rows[identity] = flow_rows
        return _warmup_batch(
            step_id=self._warmup_step_id,
            admissions=admissions,
            operations=operations,
            block_tables=block_tables,
            new_cache_pages=new_cache_pages,
            forward_rows=forward_rows,
            latent_placements=latent_placements,
            input_products=input_products,
            tensorized_mixed=tensorized_mixed,
        )

    def _warmup_flow_tables(
        self,
        operation: Operation,
        main_slot: int,
        height: int,
        width: int,
    ) -> tuple[
        tuple[BlockTable, ...],
        tuple[CachePageAllocation, ...],
        tuple[RowGeometry, ...],
    ]:
        session = self.requests.get(operation.request_key.session_id)
        image = session.image
        generation = self.model.generation
        if image is None or generation is None:
            raise invalid_descriptor("generation warmup has no admitted image runtime")
        guide = build_flow_cfg_plan(
            cfg_text_scale=float(image.cfg_text_scale),
            cfg_img_scale=float(image.cfg_img_scale),
            recipe=generation.cfg_recipe,
            renorm=image.cfg_renorm_type,
            renorm_min=float(image.cfg_renorm_min),
            use_cfg=True,
        )
        runtime = parent_runtime(self.execution, operation, session)
        query = generation.physical_tokens(height, width)
        image_prompt = image.image_prompts[0] if image.image_prompts else ""
        branch_prefixes: list[tuple[tuple[int, ...], bool]] = []
        for branch in guide.branches:
            prefix, copy_conditioning = generation.prefix(
                generation.branch_source(branch),
                image_prompt=image_prompt,
                negative_prompt=image.negative_prompt,
                negative_token_ids=session.negative_token_ids,
                tokenizer=self.execution.tokenizer,
            )
            branch_prefixes.append((prefix, copy_conditioning))
        alternatives = {
            prefix for prefix, copy_conditioning in branch_prefixes if not copy_conditioning
        }
        if len(alternatives) > 1:
            raise invalid_descriptor("warmup flow has multiple distinct alternative prefixes")
        alternative = next(iter(alternatives), ())
        required = ceil_div(len(alternative), self.cache_pool.block_size)
        lease = self._warmup_prefix_pages.setdefault(operation.request_key, [])
        missing = required - len(lease)
        occupied = {
            page
            for request_key, pages in self._warmup_prefix_pages.items()
            if request_key != operation.request_key
            for page in pages
        }
        occupied.update(page for pages in self._warmup_kv_pages.values() for page in pages)
        allocated = tuple(page for page in self.cache_pool.page_ids(0) if page not in occupied)[
            :missing
        ]
        if len(allocated) != missing:
            raise invalid_descriptor("warmup alternative prefix exceeds KV capacity")
        lease.extend(allocated)
        tables: tuple[BlockTable, ...] = ()
        allocations: tuple[CachePageAllocation, ...] = ()
        alternative_slot = main_slot
        rows: list[RowGeometry] = []
        if alternative:
            alternative_slot = self._warmup_prefix_slots.setdefault(
                operation.request_key,
                int(self._capabilities.max_request_pool_size) - len(self._warmup_prefix_slots),
            )
            if alternative_slot == main_slot or alternative_slot < 1:
                raise invalid_descriptor("warmup has no request slot for an alternative prefix")
            tables = (
                BlockTable(
                    request_pool_idx=alternative_slot,
                    group_id=0,
                    page_ids=tuple(lease),
                    allocated_tokens=len(lease) * self.cache_pool.block_size,
                ),
            )
            if allocated:
                allocations = (
                    CachePageAllocation(
                        request_pool_idx=alternative_slot,
                        group_id=0,
                        page_ids=allocated,
                    ),
                )
            rows.append(
                RowGeometry(
                    operation_index=0,
                    request_pool_index=alternative_slot,
                    seq_len=0,
                    query_len=len(alternative),
                )
            )
        for prefix, copy_conditioning in branch_prefixes:
            rows.append(
                RowGeometry(
                    operation_index=0,
                    request_pool_index=main_slot if copy_conditioning else alternative_slot,
                    seq_len=int(runtime.kv_visible_len) if copy_conditioning else len(prefix),
                    query_len=query,
                )
            )
        return tables, allocations, tuple(rows)

    def warmup(self) -> None:
        """Complete pre-admission kernel JIT and open the serving epoch.

        The ``fa4_cute`` attention backend JIT-compiles its CUTLASS kernels the
        first time each variant runs, costing tens of seconds on the first real
        request. Representative operations run through the real execution path;
        startup succeeds only after every configured warmup completes and its
        private collective identities are retired.
        """

        import torch

        product_devices = (
            self.deployment.device,
            self.deployment.generation_device or self.deployment.device,
        )
        self.device_products.warmup_scattered_publication(product_devices)
        if torch.device(self.deployment.device).type == "cuda":
            self._warmup_sequence()
            logger.info("completed token CUDA graph warmup")
            self._warmup_flow()
            logger.info("completed flow CUDA graph warmup")
        elif self._capabilities.mixed_buckets:
            self._warmup_flow()
            logger.info("completed mixed execution warmup")
        complete_startup(self.execution)
        logger.info("completed execution partition startup verification")

    def _warmup_image_geometry(self) -> tuple[int, int]:
        """Largest square image whose latent grid fits the declared capacity."""

        import math

        caps = self._capabilities
        downsample = max(1, int(caps.latent_downsample))
        capacity = int(caps.latent_capacity_units)
        if int(caps.max_vae_grid_tokens) > 0:
            capacity = min(capacity, int(caps.max_vae_grid_tokens))
        side = max(1, math.isqrt(max(1, capacity)))
        return side * downsample, side * downsample

    def _warmup_sequence(self) -> None:
        """Warm the real token forward paths and capture the configured graphs.

        One prompt extend across the largest configured decode batch pays the
        first-use kernel JIT; the paged-prefill CUDA graph is captured for
        every configured token bucket; the decode CUDA graph is captured for
        every configured batch size (two rounds each: capture, then replay).
        """

        import torch

        from ..batch import (
            Admission,
            Bounds,
            DevicePoint,
            Domain,
            DType,
            FixedPoint,
            Operation,
            PointRange,
            ProductKind,
            ProductPayload,
            ProductRef,
            RequestKey,
            SamplingParams,
            ShapeBound,
            StaticDim,
            StorageClass,
            UndAdmission,
            VersionRef,
            encode_token_product_bytes,
        )

        variants = self._effective_work_variants
        if ForwardMode.TOKEN_EXTEND not in variants:
            return
        pool = self.cache_pool
        if self.requests.request_ids():
            return
        if self._execution.cuda_graph and self._execution.prefill_cuda_graph:
            self._warmup_prefill_graphs()
        configured = (
            tuple(
                sorted(
                    {
                        batch_size
                        for partition in self.runner.partitions
                        if Domain.DECODE in partition.domains
                        for batch_size in partition.graphs.decode_batch_sizes
                    }
                )
            )
            if (self._execution.cuda_graph and ForwardMode.TOKEN_DECODE in variants)
            else (1,)
        )
        batch_sizes = tuple(
            sorted(
                {int(value) for value in configured if 0 < int(value) < int(pool.num_pages)},
                reverse=True,
            )
        )
        if not batch_sizes:
            return
        logger.info("warming %d decode CUDA graph executables", len(batch_sizes))
        session_ids = tuple(range(1, max(batch_sizes) + 1))
        keys = {sid: RequestKey(0, sid, 1) for sid in session_ids}
        admissions = {
            sid: Admission.create(
                keys[sid],
                request_pool_idx=sid,
                und=UndAdmission(
                    sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                    initial_position=0,
                ),
            )
            for sid in session_ids
        }

        next_product_generation = 1

        def prompt_op(
            sid: int,
            op_id: int,
            parent: VersionRef,
            tokens: tuple[int, ...],
        ) -> tuple[Operation, ProductPayload]:
            nonlocal next_product_generation
            token_ref = ProductRef(
                request_key=keys[sid],
                producer_op_id=op_id,
                output_index=(1 << 16) - 1,
                generation=op_id,
                kind=ProductKind.TOKEN,
                storage_class=StorageClass.HOST_STAGING,
                dtype=DType.U32,
                shape_bound=ShapeBound((StaticDim(max(1, len(tokens))),)),
                point_range=PointRange(),
            )
            outputs = _warmup_token_outputs(keys[sid], op_id, next_product_generation)
            next_product_generation += len(outputs)
            operation = Operation.registered(
                request_key=keys[sid],
                op_id=op_id,
                parent=parent,
                work=ForwardMode.TOKEN_EXTEND,
                route=0,
                domain=Domain.PREFILL,
                bounds=Bounds(max_points=1, max_tokens=max(1, len(tokens))),
                inputs=(token_ref,),
                outputs=outputs,
            )
            return operation, ProductPayload(
                product=token_ref, payload=encode_token_product_bytes(tokens)
            )

        def decode_op(sid: int, op_id: int, predecessor: Operation) -> Operation:
            nonlocal next_product_generation
            token_output = next(
                output for output in predecessor.outputs if output.kind is ProductKind.TOKEN
            )
            outputs = _warmup_token_outputs(keys[sid], op_id, next_product_generation)
            next_product_generation += len(outputs)
            return Operation.registered(
                request_key=keys[sid],
                op_id=op_id,
                parent=VersionRef(
                    keys[sid],
                    predecessor.op_id,
                    DevicePoint(1, None, predecessor.plan_digest),
                ),
                work=ForwardMode.TOKEN_DECODE,
                route=0,
                domain=Domain.DECODE,
                bounds=Bounds(max_points=1, max_tokens=1),
                outputs=outputs,
                predicate=token_output,
            )

        op_ids = {sid: 0 for sid in session_ids}
        predecessors: dict[int, Operation] = {}
        try:
            operations = []
            payloads = []
            for sid in session_ids:
                root = VersionRef(keys[sid], 0, FixedPoint(0, admissions[sid].digest))
                op_ids[sid] += 1
                operation, payload = prompt_op(sid, op_ids[sid], root, (0,))
                operations.append(operation)
                payloads.append(payload)
            self._execute_warmup(
                self._build_warmup_batch(
                    admissions=tuple(admissions[sid] for sid in session_ids),
                    operations=tuple(operations),
                    input_products=tuple(payloads),
                ),
                retain_device_outputs=ForwardMode.TOKEN_DECODE in variants,
            )
            predecessors.update(zip(session_ids, operations, strict=True))
            if ForwardMode.TOKEN_DECODE not in variants:
                return
            repeats = 2 if self._execution.cuda_graph else 1
            for _ in range(repeats):
                for batch_size in batch_sizes:
                    selected = session_ids[:batch_size]
                    operations = []
                    for sid in selected:
                        op_ids[sid] += 1
                        operations.append(decode_op(sid, op_ids[sid], predecessors[sid]))
                    self._execute_warmup(
                        self._build_warmup_batch(
                            admissions=(),
                            operations=tuple(operations),
                        ),
                        retain_device_outputs=True,
                    )
                    self.release_products(
                        tuple(
                            int(output.generation)
                            for sid in selected
                            for output in predecessors[sid].outputs
                            if output.storage_class is StorageClass.DEVICE_TENSOR
                        )
                    )
                    predecessors.update(zip(selected, operations, strict=True))
        finally:
            device = torch.device(self.deployment.device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for sid in session_ids:
                self.drop_session(sid)

    def _warmup_prefill_graphs(self) -> None:
        """Capture the paged-prefill CUDA graph for every configured token bucket."""

        from ..batch import (
            Admission,
            Bounds,
            Domain,
            DType,
            FixedPoint,
            Operation,
            PointRange,
            ProductKind,
            ProductPayload,
            ProductRef,
            RequestKey,
            SamplingParams,
            ShapeBound,
            StaticDim,
            StorageClass,
            UndAdmission,
            VersionRef,
            encode_token_product_bytes,
        )

        pool = self.cache_pool
        if self.requests.request_ids():
            return
        max_route_tokens = int(self.model.text_max_tokens)
        capacity = min(
            max_route_tokens,
            max(0, int(pool.num_pages) - 1) * int(pool.block_size),
        )
        token_buckets = tuple(
            sorted(
                {
                    int(value)
                    for value in self._prefill_graph_token_sizes
                    if 0 < int(value) <= capacity
                },
                reverse=True,
            )
        )
        if not token_buckets:
            return
        catalog = (
            tuple(_PagedPrefillGraphBucket(value, 1, 1) for value in token_buckets)
            if self.model.tensorized_mixed
            else _paged_prefill_graph_buckets(
                token_buckets,
                self._prefill_graph_row_sizes,
                max_rows=int(self._capabilities.max_request_pool_size),
                max_tokens=capacity,
            )
        )
        catalog = tuple(
            sorted(
                catalog,
                key=lambda value: (value.token_bucket * value.row_bucket, value.token_bucket),
                reverse=True,
            )
        )
        logger.info("warming %d paged-prefill CUDA graph executables", len(catalog))
        session_id = 0
        # First warm every configured physical call in descending footprint,
        # then capture every bucket in the same order.
        for _ in range(2):
            for bucket in catalog:
                live_rows = bucket.live_rows
                token_counts = (
                    bucket.token_bucket - live_rows + 1,
                    *(1 for _ in range(live_rows - 1)),
                )
                admissions: list[Admission] = []
                operations: list[Operation] = []
                input_products: list[ProductPayload] = []
                active_sessions: list[int] = []
                for row, token_count in enumerate(token_counts):
                    tokens = (0,) * token_count
                    session_id += 1
                    active_sessions.append(session_id)
                    rk = RequestKey(0, session_id, 1)
                    admission = Admission.create(
                        rk,
                        request_pool_idx=row + 1,
                        und=UndAdmission(
                            sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                            initial_position=0,
                        ),
                    )
                    token_ref = ProductRef(
                        request_key=rk,
                        producer_op_id=1,
                        output_index=(1 << 16) - 1,
                        generation=1,
                        kind=ProductKind.TOKEN,
                        storage_class=StorageClass.HOST_STAGING,
                        dtype=DType.U32,
                        shape_bound=ShapeBound((StaticDim(token_count),)),
                        point_range=PointRange(),
                    )
                    operations.append(
                        Operation.registered(
                            request_key=rk,
                            op_id=1,
                            parent=VersionRef(rk, 0, FixedPoint(0, admission.digest)),
                            work=ForwardMode.TOKEN_EXTEND,
                            route=0,
                            domain=Domain.PREFILL,
                            bounds=Bounds(max_points=1, max_tokens=token_count),
                            inputs=(token_ref,),
                            outputs=_warmup_token_outputs(rk, 1, 2),
                        )
                    )
                    admissions.append(admission)
                    input_products.append(
                        ProductPayload(
                            product=token_ref,
                            payload=encode_token_product_bytes(tokens),
                        )
                    )
                try:
                    self._execute_warmup(
                        self._build_warmup_batch(
                            admissions=tuple(admissions),
                            operations=tuple(operations),
                            input_products=tuple(input_products),
                        )
                    )
                finally:
                    for active_session in active_sessions:
                        self.drop_session(active_session)

    def _warmup_flow(self) -> None:
        """Drive one denoise quantum through the real flow forward path."""

        from ..batch import (
            Admission,
            Bounds,
            DeviceDim,
            DevicePoint,
            Domain,
            DrawLayout,
            DType,
            FixedPoint,
            GenAdmission,
            Operation,
            PointRange,
            ProductKind,
            ProductPayload,
            ProductRef,
            RequestKey,
            Rng,
            SamplingParams,
            ShapeBound,
            StaticDim,
            StorageClass,
            UndAdmission,
            VersionRef,
            encode_token_product_bytes,
        )

        generation = self.model.generation
        if not {
            ForwardMode.GEN_TRANSITION,
            ForwardMode.GEN_FLOW,
        }.issubset(self._effective_work_variants) or not isinstance(generation, GenerationPipeline):
            return
        if self.requests.request_ids():
            return
        configured = tuple(
            sorted(
                self._flow_graph_buckets,
                key=lambda value: (
                    value.rows * value.height * value.width * value.cfg_branches,
                    value.rows,
                    value.height,
                    value.width,
                    value.cfg_branches,
                ),
                reverse=True,
            )
        )
        if not configured:
            if self._execution.cuda_graph:
                return
            height, width = self._warmup_image_geometry()
            configured = tuple(
                _FlowGraphBucket(1, height, width, cfg_branches)
                for cfg_branches in self._flow_cfg_branches
            )
        next_session_id = 1
        next_generation = 1
        for bucket in configured:
            batch_size = bucket.rows
            height = bucket.height
            width = bucket.width
            cfg_branches = bucket.cfg_branches
            if batch_size > int(self._capabilities.max_request_pool_size):
                continue
            mixed_text_sizes = tuple(
                mixed.decode_rows
                for mixed in self._mixed_flow_graph_buckets
                if mixed.flow_rows == batch_size
                and mixed.height == height
                and mixed.width == width
                and mixed.cfg_branches == cfg_branches
                and mixed.decode_rows + batch_size <= int(self._capabilities.max_request_pool_size)
            )
            mixed_rounds = (
                3 if self._execution.cuda_graph and self._execution.prefill_cuda_graph else 1
            )
            session_ids = tuple(range(next_session_id, next_session_id + batch_size))
            next_session_id += batch_size
            keys = tuple(RequestKey(0, session_id, 1) for session_id in session_ids)
            admissions = tuple(
                Admission.create(
                    key,
                    request_pool_idx=index,
                    gen_admission=GenAdmission(
                        image=_startup_image_parameters(
                            cfg_branches,
                            steps=2 + mixed_rounds * len(mixed_text_sizes),
                            height=height,
                            width=width,
                        )
                    ),
                )
                for index, key in enumerate(keys, start=1)
            )
            text_session_count = max(mixed_text_sizes, default=0)
            text_session_ids = tuple(range(next_session_id, next_session_id + text_session_count))
            next_session_id += text_session_count
            text_keys = {
                session_id: RequestKey(0, session_id, 1) for session_id in text_session_ids
            }
            text_admissions = {
                session_id: Admission.create(
                    text_keys[session_id],
                    request_pool_idx=batch_size + index,
                    und=UndAdmission(
                        sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                        initial_position=0,
                    ),
                )
                for index, session_id in enumerate(text_session_ids, start=1)
            }
            roots = tuple(
                VersionRef(key, 0, FixedPoint(0, admission.digest))
                for key, admission in zip(keys, admissions, strict=True)
            )
            conditionings: list[ProductRef] = []
            publications: list[Operation] = []
            for key, root in zip(keys, roots, strict=True):
                conditioning = ProductRef(
                    request_key=key,
                    producer_op_id=1,
                    output_index=0,
                    generation=next_generation,
                    kind=ProductKind.KV,
                    storage_class=StorageClass.PAGED_KV,
                    dtype=DType.U8,
                    shape_bound=ShapeBound((DeviceDim(1 << 20),)),
                    point_range=PointRange(),
                )
                next_generation += 1
                conditionings.append(conditioning)
                publications.append(
                    Operation.registered(
                        request_key=key,
                        op_id=1,
                        parent=root,
                        work=ForwardMode.TRANSFER_KV_PUBLISH,
                        route=0,
                        domain=Domain.PREFILL,
                        bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
                        outputs=(conditioning,),
                    )
                )
            try:
                self._execute_warmup(
                    self._build_warmup_batch(
                        admissions=admissions,
                        operations=tuple(publications),
                        image_geometry=(height, width),
                    )
                )
                text_predecessors: dict[int, Operation] = {}
                text_op_ids = {session_id: 1 for session_id in text_session_ids}
                if text_session_ids:
                    prompt_operations: list[Operation] = []
                    prompt_payloads: list[ProductPayload] = []
                    for session_id in text_session_ids:
                        key = text_keys[session_id]
                        token_ref = ProductRef(
                            request_key=key,
                            producer_op_id=1,
                            output_index=(1 << 16) - 1,
                            generation=next_generation,
                            kind=ProductKind.TOKEN,
                            storage_class=StorageClass.HOST_STAGING,
                            dtype=DType.U32,
                            shape_bound=ShapeBound((StaticDim(1),)),
                            point_range=PointRange(),
                        )
                        next_generation += 1
                        prompt_outputs = _warmup_token_outputs(key, 1, next_generation)
                        next_generation += len(prompt_outputs)
                        operation = Operation.registered(
                            request_key=key,
                            op_id=1,
                            parent=VersionRef(
                                key,
                                0,
                                FixedPoint(0, text_admissions[session_id].digest),
                            ),
                            work=ForwardMode.TOKEN_EXTEND,
                            route=0,
                            domain=Domain.PREFILL,
                            bounds=Bounds(max_points=1, max_tokens=1),
                            inputs=(token_ref,),
                            outputs=prompt_outputs,
                        )
                        prompt_operations.append(operation)
                        prompt_payloads.append(
                            ProductPayload(
                                product=token_ref,
                                payload=encode_token_product_bytes((0,)),
                            )
                        )
                    self._execute_warmup(
                        self._build_warmup_batch(
                            admissions=tuple(
                                text_admissions[session_id] for session_id in text_session_ids
                            ),
                            operations=tuple(prompt_operations),
                            input_products=tuple(prompt_payloads),
                            image_geometry=(height, width),
                        ),
                        retain_device_outputs=True,
                        catalog_graphs=False,
                    )
                    text_predecessors.update(zip(text_session_ids, prompt_operations, strict=True))
                max_latent_elements = max(
                    1,
                    math.prod(generation.latent_shape(height, width)),
                )
                initial_latents: list[ProductRef] = []
                transitions: list[Operation] = []
                for key, root, conditioning in zip(keys, roots, conditionings, strict=True):
                    initial_latent = ProductRef(
                        request_key=key,
                        producer_op_id=2,
                        output_index=0,
                        generation=next_generation,
                        kind=ProductKind.LATENT,
                        storage_class=StorageClass.LATENT_ARENA,
                        dtype=DType.BF16,
                        shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                        point_range=PointRange(),
                    )
                    next_generation += 1
                    ready = ProductRef(
                        request_key=key,
                        producer_op_id=2,
                        output_index=1,
                        generation=next_generation,
                        kind=ProductKind.COMPLETION,
                        storage_class=StorageClass.DEVICE_TENSOR,
                        dtype=DType.U32,
                        shape_bound=ShapeBound((StaticDim(1),)),
                        point_range=PointRange(),
                    )
                    next_generation += 1
                    initial_latents.append(initial_latent)
                    transitions.append(
                        Operation.registered(
                            request_key=key,
                            op_id=2,
                            parent=root,
                            work=ForwardMode.GEN_TRANSITION,
                            route=0,
                            domain=Domain.FLOW,
                            bounds=Bounds(
                                max_points=1,
                                max_tokens=1,
                                max_latent_bytes=max_latent_elements * 2,
                            ),
                            inputs=(conditioning,),
                            outputs=(initial_latent, ready),
                            rng=Rng(
                                seed=0,
                                semantic_index_base=1,
                                draw_layout=DrawLayout.FLOW_NOISE,
                            ),
                        )
                    )
                self._execute_warmup(
                    self._build_warmup_batch(
                        admissions=(),
                        operations=tuple(transitions),
                        image_geometry=(height, width),
                    )
                )
                current_latents = tuple(initial_latents)
                flow_predecessors = dict(zip(session_ids, transitions, strict=True))
                for op_id in (3, 4):
                    outputs: list[ProductRef] = []
                    flows: list[Operation] = []
                    for session_id, key, conditioning, current in zip(
                        session_ids, keys, conditionings, current_latents, strict=True
                    ):
                        output = ProductRef(
                            request_key=key,
                            producer_op_id=op_id,
                            output_index=0,
                            generation=next_generation,
                            kind=ProductKind.LATENT,
                            storage_class=StorageClass.LATENT_ARENA,
                            dtype=DType.BF16,
                            shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                            point_range=PointRange(),
                        )
                        next_generation += 1
                        outputs.append(output)
                        flows.append(
                            Operation.registered(
                                request_key=key,
                                op_id=op_id,
                                parent=VersionRef(
                                    key,
                                    flow_predecessors[session_id].op_id,
                                    DevicePoint(
                                        1,
                                        None,
                                        flow_predecessors[session_id].plan_digest,
                                    ),
                                ),
                                work=ForwardMode.GEN_FLOW,
                                route=0,
                                domain=Domain.FLOW,
                                bounds=Bounds(
                                    max_points=1,
                                    max_tokens=1,
                                    max_latent_bytes=max_latent_elements * 2,
                                ),
                                inputs=(conditioning, current),
                                outputs=(output,),
                            )
                        )
                    self._execute_warmup(
                        self._build_warmup_batch(
                            admissions=(),
                            operations=tuple(flows),
                            image_geometry=(height, width),
                        )
                    )
                    current_latents = tuple(outputs)
                    flow_predecessors.update(zip(session_ids, flows, strict=True))
                flow_op_id = 5
                for text_batch_size in mixed_text_sizes:
                    selected_text = text_session_ids[:text_batch_size]
                    for _ in range(mixed_rounds):
                        text_operations: list[Operation] = []
                        for session_id in selected_text:
                            predecessor = text_predecessors[session_id]
                            token_output = next(
                                output
                                for output in predecessor.outputs
                                if output.kind is ProductKind.TOKEN
                            )
                            text_op_ids[session_id] += 1
                            op_id = text_op_ids[session_id]
                            token_outputs = _warmup_token_outputs(
                                text_keys[session_id], op_id, next_generation
                            )
                            next_generation += len(token_outputs)
                            text_operations.append(
                                Operation.registered(
                                    request_key=text_keys[session_id],
                                    op_id=op_id,
                                    parent=VersionRef(
                                        text_keys[session_id],
                                        predecessor.op_id,
                                        DevicePoint(1, None, predecessor.plan_digest),
                                    ),
                                    work=ForwardMode.TOKEN_DECODE,
                                    route=0,
                                    domain=Domain.DECODE,
                                    bounds=Bounds(max_points=1, max_tokens=1),
                                    outputs=token_outputs,
                                    predicate=token_output,
                                )
                            )

                        flow_outputs: list[ProductRef] = []
                        flow_operations: list[Operation] = []
                        for session_id, key, conditioning, current in zip(
                            session_ids,
                            keys,
                            conditionings,
                            current_latents,
                            strict=True,
                        ):
                            output = ProductRef(
                                request_key=key,
                                producer_op_id=flow_op_id,
                                output_index=0,
                                generation=next_generation,
                                kind=ProductKind.LATENT,
                                storage_class=StorageClass.LATENT_ARENA,
                                dtype=DType.BF16,
                                shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                                point_range=PointRange(),
                            )
                            next_generation += 1
                            flow_outputs.append(output)
                            flow_operations.append(
                                Operation.registered(
                                    request_key=key,
                                    op_id=flow_op_id,
                                    parent=VersionRef(
                                        key,
                                        flow_predecessors[session_id].op_id,
                                        DevicePoint(
                                            1,
                                            None,
                                            flow_predecessors[session_id].plan_digest,
                                        ),
                                    ),
                                    work=ForwardMode.GEN_FLOW,
                                    route=0,
                                    domain=Domain.FLOW,
                                    bounds=Bounds(
                                        max_points=1,
                                        max_tokens=1,
                                        max_latent_bytes=max_latent_elements * 2,
                                    ),
                                    inputs=(conditioning, current),
                                    outputs=(output,),
                                )
                            )
                        flow_op_id += 1
                        self._execute_warmup(
                            self._build_warmup_batch(
                                admissions=(),
                                operations=(*text_operations, *flow_operations),
                                tensorized_mixed=True,
                                image_geometry=(height, width),
                            ),
                            retain_device_outputs=True,
                        )
                        self.release_products(
                            tuple(
                                int(output.generation)
                                for session_id in selected_text
                                for output in text_predecessors[session_id].outputs
                                if output.storage_class is StorageClass.DEVICE_TENSOR
                            )
                        )
                        text_predecessors.update(zip(selected_text, text_operations, strict=True))
                        current_latents = tuple(flow_outputs)
                        flow_predecessors.update(zip(session_ids, flow_operations, strict=True))
            finally:
                for session_id in (*session_ids, *text_session_ids):
                    self.drop_session(session_id)

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
