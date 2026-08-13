"""Composition root for one canonical model-backed worker."""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch import nn

from ..batch import (
    Admission,
    AttentionRegime,
    Batch,
    BatchPartition,
    CacheCopy,
    CompletionReport,
    Domain,
    ExecutionCapability,
    ImageParams,
    KvBranchPlacement,
    KvPlacement,
    LatentPlacement,
    Operation,
    OpStatus,
    ProductPayload,
    ProductRef,
    RecoveryPlacement,
    RequestKey,
    SnapshotRef,
    StorageClass,
    WorkVariant,
)
from ..capabilities import (
    GraphBucketCapability,
    LaneCapabilities,
    RequestKind,
    WorkerCapabilities,
)
from ..execution import ModelExecutor, ModelRunner
from ..execution.cuda_graph import CudaGraphRunner
from ..execution.executor import completion_report_ready, finalize_completion_report
from ..execution.forward_batch import AttentionSelection
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import ExecutionConfig, LaneConfig, graph_memory_budget_bytes
from ..foundation.sizing import ceil_div, device_total_bytes
from ..foundation.sync_detector import (
    sync_detection_active,
    sync_detection_enforced,
    sync_detector,
)
from ..loader.weight_set import WeightSet
from ..models.generation import GenerationPipeline
from ..models.identity import ModelIdentity, architecture_identity
from ..models.runtime import ExecutionModel, WorkerDeployment
from ..nn.diffusion.cfg import build_flow_cfg_plan
from ..nn.mesh import DeviceMesh
from ..runtime.arena_capacity import model_arena_capacity
from ..runtime.cache_pool import CachePool
from ..runtime.capabilities import resolve_capabilities
from ..runtime.execution_trace import ExecutionPhase, ExecutionTrace, OperationTrace
from ..runtime.latent_store import LatentStore
from ..runtime.mesh_store import MeshStore
from ..runtime.mover import Mover
from ..runtime.product_store import ProductStore
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore
from ..runtime.runtime_states import RuntimeStates
from ..runtime.snapshot_store import SnapshotProvider

logger = logging.getLogger(__name__)


def _warmup_batch(
    *,
    step_id: int,
    admissions: tuple[Admission, ...],
    operations: tuple[Operation, ...],
    request_pool_indices: dict[RequestKey, int],
    kv_placements: dict[tuple[RequestKey, int], tuple[KvPlacement, ...]],
    kv_branch_placements: dict[tuple[RequestKey, int], tuple[KvBranchPlacement, ...]],
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
            request_pool_indices=tuple(
                request_pool_indices[operation.request_key] for operation in members
            ),
            kv_placements=tuple(
                placement
                for operation in members
                for placement in kv_placements.get(
                    (operation.request_key, operation.op_id),
                    (),
                )
                if (operation.request_key, operation.op_id) in kv_placements
            ),
            kv_branch_placements=tuple(
                placement
                for operation in members
                for placement in kv_branch_placements.get(
                    (operation.request_key, operation.op_id),
                    (),
                )
            ),
            latent_placements=tuple(
                latent_placements[(operation.request_key, operation.op_id)]
                for operation in members
                if operation.work.variant
                in {
                    WorkVariant.GEN_TRANSITION,
                    WorkVariant.GEN_FLOW,
                    WorkVariant.MATERIALIZE,
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


class ModelWorker:
    """Own one ready model and all system authorities around its raw forward."""

    def __init__(
        self,
        model: ExecutionModel,
        *,
        mesh: DeviceMesh,
        deployment: WorkerDeployment,
        attention: AttentionSelection,
        execution: ExecutionConfig,
        tokenizer: object | None,
        allowed_work_variants: frozenset[WorkVariant],
        defer_sampling: bool = False,
        transfer_backend: str = "local",
        cross_process: bool = False,
        architecture_digest: str | None = None,
        weight_digest: str | None = None,
        pipeline_depth: int,
        completion_payload_bytes: int,
        snapshot_dir: str | None = None,
    ) -> None:
        if not isinstance(model, ExecutionModel) or type(model).forward is nn.Module.forward:
            raise capability_mismatch(
                "model worker requires forward(input_ids, positions, forward_batch)"
            )
        if not isinstance(deployment, WorkerDeployment):
            raise capability_mismatch("model worker requires a worker deployment")
        self.model = model
        self.deployment = deployment
        self.weights = WeightSet.from_module(model, digest=weight_digest)
        self.weight_digest = self.weights.digest
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
            scratch_capacity_tokens=int(declared.scratch_capacity_tokens),
            latent_capacity_units=int(declared.latent_capacity_units),
            max_latent_feature_bytes=int(declared.max_latent_feature_bytes),
            max_vision_feature_bytes=int(declared.max_vision_feature_bytes),
            bytes_per_token=int(declared.bytes_per_token),
        )
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
        advertised_work = self._effective_work_variants.intersection(declared.supported_work)
        if not advertised_work:
            raise capability_mismatch(f"{type(self).__name__} advertises no executable work")
        self._capabilities = replace(
            declared,
            supported_work=tuple(variant for variant in WorkVariant if variant in advertised_work),
            pipeline_depth=int(pipeline_depth),
        )
        cache = model.cache_geometry
        cache_dtype = getattr(torch, str(cache.dtype).removeprefix("torch."), None)
        if not isinstance(cache_dtype, torch.dtype):
            raise capability_mismatch(f"unsupported cache dtype {cache.dtype!r}")
        self.cache_pool = CachePool(
            num_layers=int(cache.num_layers),
            request_pages=int(self._capabilities.num_blocks),
            scratch_pages=ceil_div(
                int(self._capabilities.scratch_capacity_tokens),
                int(self._capabilities.block_size),
            ),
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
        self.sessions = SessionStore()
        self.runtime_states = RuntimeStates(
            request_pool_size=int(self._capabilities.max_request_pool_size),
            vocab_size=int(model.vocab_size),
            continuation_width=1,
            device=deployment.device,
        )
        self.latents = LatentStore(capacity_bytes=arena.latent_bytes)
        self.products = ProductStore(
            encoder_cache_budget=model.resource_geometry.encoder_cache_entries,
            device_product_capacity=arena.device_products,
            device_product_byte_capacity=arena.device_product_bytes,
        )
        self.replay = ReplayStore()
        self.mover = Mover(
            transfer_backend=transfer_backend,
            transfer_byte_capacity=arena.transfer_bytes,
            transfer_ticket_capacity=arena.transfer_tickets,
            cross_process=bool(cross_process),
        )
        max_rows = min(
            int(self._capabilities.max_batch_operations),
            int(self._capabilities.max_request_pool_size),
        )
        flow = model.generation
        max_staged_rows = max_rows * (
            1 if flow is None else int(flow.max_cfg_branches)
        )
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
        und_lane = next(
            (lane for lane in execution.lanes if Domain.UND in lane.domains),
            None,
        )
        gen_lane = next(
            (lane for lane in execution.lanes if Domain.GEN in lane.domains),
            None,
        )
        und_max_operations = min(
            max_rows,
            max_rows if und_lane is None else int(und_lane.max_batch_operations or max_rows),
        )
        und_max_tokens = min(
            int(self._capabilities.max_batch_tokens),
            (
                int(self._capabilities.max_batch_tokens)
                if und_lane is None
                else int(und_lane.max_batch_tokens or self._capabilities.max_batch_tokens)
            ),
        )
        gen_max_operations = min(
            max_rows,
            max_rows if gen_lane is None else int(gen_lane.max_batch_operations or max_rows),
        )
        decode_graph_batch_sizes = tuple(
            value
            for value in execution.decode_graph_batch_sizes
            if 0 < int(value) <= und_max_operations
            and int(value) < int(self._capabilities.num_blocks)
        )
        prefill_capacity = min(
            int(self._capabilities.max_batch_tokens),
            int(model.text_max_tokens),
            max(0, int(self._capabilities.num_blocks) - 1) * int(deployment.block_size),
        )
        prefill_graph_token_sizes = tuple(
            value
            for value in execution.prefill_graph_token_sizes
            if 0 < int(value) <= min(prefill_capacity, und_max_tokens)
        )
        flow_graph_buckets = (
            ()
            if flow is None
            else tuple(
                (int(batch_size), int(height), int(width))
                for height, width in execution.flow_graph_shapes
                for batch_size in execution.flow_graph_batch_sizes
                if 0 < int(batch_size) <= gen_max_operations
                and int(batch_size)
                * flow.physical_tokens(int(height), int(width))
                * int(flow.max_cfg_branches)
                <= max_staged_tokens
                and flow.image_tokens(int(height), int(width))
                <= int(self._capabilities.latent_capacity_units)
                and int(batch_size)
                * math.prod(flow.latent_shape(int(height), int(width)))
                * torch.empty((), dtype=cache_dtype).element_size()
                * 2
                <= int(self.latents.capacity_bytes)
            )
        )
        mixed_text_batch_sizes = (
            ()
            if not flow_graph_buckets or not model.tensorized_mixed
            else tuple(
                range(
                    1,
                    max(int(batch_size) for batch_size in execution.flow_graph_batch_sizes) + 1,
                )
            )
        )
        mixed_flow_graph_buckets = tuple(
            (text_batch_size, height, width)
            for batch_size, height, width in flow_graph_buckets
            if batch_size == 1
            for text_batch_size in mixed_text_batch_sizes
        )
        flow_prefix_lengths: tuple[int, ...] = ()
        if flow is not None and model.tensorized_mixed and flow_graph_buckets:
            image = ImageParams()
            guide = build_flow_cfg_plan(
                cfg_text_scale=float(image.cfg_text_scale),
                cfg_img_scale=float(image.cfg_img_scale),
                recipe=flow.cfg_recipe,
                renorm=image.cfg_renorm_type,
                renorm_min=float(image.cfg_renorm_min),
                use_cfg=True,
            )
            flow_prefix_lengths = tuple(
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
        flow_prefix_graph_batches = tuple(
            batch_size
            for batch_size in sorted({bucket[0] for bucket in flow_graph_buckets})
            if flow_prefix_lengths
            and sum(1 for candidate in flow_graph_buckets if candidate[0] == batch_size) > 1
        )
        self._decode_graph_batch_sizes = decode_graph_batch_sizes
        self._prefill_graph_token_sizes = prefill_graph_token_sizes
        self._flow_graph_buckets = flow_graph_buckets
        self._mixed_flow_graph_buckets = mixed_flow_graph_buckets
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
                if owns_model_compute and Domain.UND in domains
                else ()
            )
            lane_prefill_buckets = (
                tuple(value for value in prefill_graph_token_sizes if int(value) <= lane_max_tokens)
                if owns_model_compute and Domain.UND in domains
                else ()
            )
            lane_flow_buckets = (
                tuple(value for value in flow_graph_buckets if value[0] <= lane_max_operations)
                if owns_model_compute and Domain.GEN in domains
                else ()
            )
            lane_mixed_flow_buckets = (
                tuple(
                    value
                    for value in mixed_flow_graph_buckets
                    if value[0] + 1 <= lane_max_operations
                )
                if owns_model_compute and {Domain.UND, Domain.GEN} <= set(domains)
                else ()
            )
            expected_captures = 0
            if execution.cuda_graph:
                if WorkVariant.TOKEN_DECODE in self._effective_work_variants:
                    expected_captures += len(lane_decode_buckets)
                if (
                    execution.prefill_cuda_graph
                    and WorkVariant.TOKEN_EXTEND in self._effective_work_variants
                ):
                    expected_captures += len(lane_prefill_buckets)
                if execution.prefill_cuda_graph and {
                    WorkVariant.GEN_TRANSITION,
                    WorkVariant.GEN_FLOW,
                }.issubset(self._effective_work_variants):
                    expected_captures += len(lane_flow_buckets)
                    if WorkVariant.TOKEN_DECODE in self._effective_work_variants:
                        expected_captures += len(lane_mixed_flow_buckets)
                    if Domain.UND in domains:
                        expected_captures += len(flow_prefix_graph_batches)
            output_slots = int(
                (pipeline_depth if lane is None else lane.max_inflight or pipeline_depth) + 1
            )
            return CudaGraphRunner(
                enabled=execution.cuda_graph,
                prefill_enabled=execution.prefill_cuda_graph,
                cache=model.cache_geometry,
                block_size=deployment.block_size,
                weight_digest=self.weight_digest,
                memory_budget_bytes=graph_budget,
                decode_batch_sizes=lane_decode_buckets,
                decode_context_blocks=self._decode_context_blocks(),
                packed_context_blocks=max_blocks_per_row,
                prefill_token_sizes=lane_prefill_buckets,
                stream=stream,
                expected_context=expected_context,
                expected_captures=expected_captures,
                output_slot_count=output_slots,
            )

        self._execution = execution
        self.trace = ExecutionTrace(self.identity.architecture_digest)
        max_blocks_per_row = max(
            1,
            ceil_div(int(model.text_max_tokens), int(deployment.block_size)),
            ceil_div(
                int(self._capabilities.scratch_capacity_tokens),
                int(deployment.block_size),
            ),
        )
        max_latent_pages_per_row = (
            0 if flow is None else ceil_div(int(flow.max_latent_tokens), int(deployment.block_size))
        )
        devices = (
            (deployment.device,)
            if deployment.generation_device is None
            else (deployment.device, deployment.generation_device)
        )
        self.runner = ModelRunner(
            model,
            None,
            self.trace,
            max_rows=max_staged_rows,
            max_tokens=max_staged_tokens,
            max_text_tokens=max_text_staged_tokens,
            max_blocks_per_row=max_blocks_per_row,
            max_latent_pages_per_row=max_latent_pages_per_row,
            hidden_size=int(model.hidden_size),
            devices=devices,
            lanes=execution.lanes,
            max_inflight=int(pipeline_depth),
            graph_factory=graph_factory,
        )
        if execution.lanes:
            lane_by_id = {lane.lane_id: lane for lane in execution.lanes}
            lane_capabilities: list[LaneCapabilities] = []
            for partition in self.runner.partitions:
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
                if execution.cuda_graph and Domain.UND in lane.domains:
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
                    if execution.prefill_cuda_graph:
                        buckets.extend(
                            GraphBucketCapability(
                                phase="text_prefill",
                                batch_size=8,
                                token_bucket=int(token_size),
                                attention_form="paged_varlen",
                                height=0,
                                width=0,
                                cfg_branches=1,
                            )
                            for token_size in prefill_graph_token_sizes
                            if int(token_size) <= max_tokens
                        )
                        buckets.extend(
                            GraphBucketCapability(
                                phase="text_prefill",
                                batch_size=int(batch_size) * len(flow_prefix_lengths),
                                token_bucket=int(batch_size) * sum(flow_prefix_lengths),
                                attention_form="packed",
                                height=0,
                                width=0,
                                cfg_branches=1,
                                layout="flow_prefix",
                            )
                            for batch_size in flow_prefix_graph_batches
                            if int(batch_size) * len(flow_prefix_lengths) <= max_operations
                            and int(batch_size) * sum(flow_prefix_lengths) <= max_tokens
                        )
                if (
                    execution.cuda_graph
                    and execution.prefill_cuda_graph
                    and Domain.GEN in lane.domains
                    and flow is not None
                ):
                    buckets.extend(
                        GraphBucketCapability(
                            phase="denoise",
                            batch_size=int(batch_size),
                            token_bucket=0,
                            attention_form="none",
                            height=int(height),
                            width=int(width),
                            cfg_branches=int(flow.max_cfg_branches),
                        )
                        for batch_size, height, width in flow_graph_buckets
                        if batch_size <= max_operations
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
        self.executor = ModelExecutor(
            model=model,
            deployment=deployment,
            runner=self.runner,
            attention=attention,
            sessions=self.sessions,
            runtime_states=self.runtime_states,
            cache_pool=self.cache_pool,
            latents=self.latents,
            products=self.products,
            replay=self.replay,
            weights=self.weights,
            mesh=MeshStore(mesh),
            transport=self.mover.transport,
            tokenizer=tokenizer,
            architecture_digest=self.identity.architecture_digest,
            weight_digest=self.weight_digest,
            allowed_work_variants=self._effective_work_variants,
            trace=self.trace,
            pipeline_depth=pipeline_depth,
            defer_sampling=defer_sampling,
            completion_payload_bytes=completion_payload_bytes,
            cpu_task_capacity=arena.cpu_tasks,
        )
        self._warmup_kv_pages: dict[tuple[RequestKey, int], list[int]] = {}
        self._warmup_scratch_pages: dict[RequestKey, list[int]] = {}
        self._warmup_step_id = 0
        self.snapshot_provider: SnapshotProvider | None = None
        if snapshot_dir is not None:
            caps = self._capabilities
            self.snapshot_provider = SnapshotProvider(
                snapshot_dir,
                model_identity=self.identity.architecture_digest,
                weight_digest=self.weight_digest,
                topology={
                    "rank": caps.rank.to_wire(),
                    "model_scope": deployment.model_scope,
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "num_layers": caps.num_layers,
                },
                device=deployment.device,
                sessions=self.sessions,
                cache_pool=self.cache_pool,
                cache_publications=self.executor.cache_publications,
                latents=self.latents,
                products=self.products,
                replay=self.replay,
                transport=self.mover.transport,
            )

    @property
    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    def _decode_context_blocks(self) -> int:
        max_tokens = int(self.model.text_max_tokens)
        if max_tokens < 1:
            return 0
        blocks = (max_tokens + int(self.deployment.block_size) - 1) // int(
            self.deployment.block_size
        )
        return min(blocks, max(0, int(self.cache_pool.request_pages) - 1))

    def execute(self, batch: Batch) -> CompletionReport:
        with self._sync_guard("execute"):
            return self.executor.execute(batch)

    def prepare_execute(self, batch: Batch) -> object | None:
        with self._sync_guard("prepare"):
            return self.executor.prepare(batch)

    def execute_prepared(self, prepared: object) -> CompletionReport:
        from ..execution.executor import PreparedExecution

        if not isinstance(prepared, PreparedExecution):
            raise invalid_descriptor("prepared execution has an invalid type")
        with self._sync_guard("execute_prepared"):
            return self.executor.execute_prepared(prepared)

    def _sync_guard(self, label: str):
        """Run a steady-state execute region under the forbidden-sync detector.

        Inert unless zero-blocking detection is activated, so the production hot
        path is unchanged until a qualification run enables it.
        """
        if not sync_detection_active():
            return nullcontext()
        return sync_detector().guard(label, enforce=sync_detection_enforced())

    def _execute_warmup(
        self,
        batch: Batch,
        *,
        retain_device_outputs: bool = False,
    ) -> CompletionReport:
        report = self.executor.execute_startup(batch)
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
            self.products.release(device_generations)
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
        kv_placements: dict[tuple[RequestKey, int], tuple[KvPlacement, ...]] = {}
        kv_branch_placements: dict[tuple[RequestKey, int], tuple[KvBranchPlacement, ...]] = {}
        latent_placements: dict[tuple[RequestKey, int], LatentPlacement] = {}
        for operation in operations:
            session = self.sessions.peek(int(operation.request_key.session_id))
            admission = admissions_by_key.get(operation.request_key)
            if session is None and admission is None:
                raise invalid_descriptor("warmup operation has no request-pool binding")
            if session is None:
                assert admission is not None
                request_pool_indices[operation.request_key] = admission.request_pool_idx
            else:
                request_pool_indices[operation.request_key] = session.request_pool_idx
            if operation.work.variant not in {
                WorkVariant.TOKEN_EXTEND,
                WorkVariant.TOKEN_DECODE,
                WorkVariant.TOKEN_VERIFY,
                WorkVariant.DRAFT,
                WorkVariant.TRANSFER_KV_PUBLISH,
                WorkVariant.TRANSFER_KV_INSTALL,
                WorkVariant.GEN_TRANSITION,
                WorkVariant.GEN_FLOW,
            }:
                continue
            if (
                session is None
                and admission is not None
                and admission.und is not None
                and admission.und.kv.prefix_len != 0
            ):
                raise invalid_descriptor("warmup KV admission requires an empty prefix")
            visible = 0
            if session is not None:
                runtime = self.executor._parent_runtime(operation, session)
                visible = int(runtime.kv_visible_len)
            input_length = (
                int(operation.bounds.max_tokens)
                if operation.work.variant
                in {
                    WorkVariant.TOKEN_EXTEND,
                    WorkVariant.TOKEN_DECODE,
                    WorkVariant.TOKEN_VERIFY,
                    WorkVariant.DRAFT,
                }
                else 0
            )
            placements: list[KvPlacement] = []
            for group_id in range(self.cache_pool.group_count):
                lease_key = (operation.request_key, group_id)
                block_table = self._warmup_kv_pages.setdefault(lease_key, [])
                missing = int(operation.kv_capacity_pages) - len(block_table)
                if missing < 0:
                    raise invalid_descriptor("warmup operation regresses its KV capacity")
                allocated = tuple(
                    candidate
                    for candidate in self.cache_pool.request_page_ids(group_id)
                    if candidate not in occupied_blocks
                )[:missing]
                if len(allocated) != missing:
                    raise invalid_descriptor("warmup KV placement exceeds resident capacity")
                block_table.extend(allocated)
                occupied_blocks.update(allocated)
                placements.append(
                    KvPlacement(
                        request_key=operation.request_key,
                        op_id=operation.op_id,
                        group_id=group_id,
                        block_table=tuple(block_table),
                        pages_to_zero=allocated,
                        prefix_length=visible,
                        input_length=input_length,
                        visible_length=visible,
                        resulting_length=visible + input_length,
                    )
                )
            kv_placements[(operation.request_key, operation.op_id)] = tuple(placements)
        height, width = image_geometry or self._warmup_image_geometry()
        latent_units = max(
            1,
            (height // max(1, int(self._capabilities.latent_downsample)))
            * (width // max(1, int(self._capabilities.latent_downsample))),
        )
        page_units = int(self._capabilities.latent_page_units)
        latent_page_count = (latent_units + page_units - 1) // page_units if page_units > 0 else 0
        for operation in operations:
            if operation.work.variant not in {
                WorkVariant.GEN_TRANSITION,
                WorkVariant.GEN_FLOW,
                WorkVariant.MATERIALIZE,
            }:
                continue
            latent_placements[(operation.request_key, operation.op_id)] = LatentPlacement(
                request_key=operation.request_key,
                op_id=operation.op_id,
                page_table=tuple(range(1, latent_page_count + 1)),
                latent_units=latent_units,
                height=height,
                width=width,
                start_step=0,
                step_count=(
                    int(operation.bounds.max_tokens)
                    if operation.work.variant is WorkVariant.GEN_FLOW
                    else 0
                ),
            )
            if operation.work.variant is WorkVariant.GEN_FLOW:
                kv_branch_placements[(operation.request_key, operation.op_id)] = (
                    self._warmup_branch_placements(operation, height, width)
                )
        return _warmup_batch(
            step_id=self._warmup_step_id,
            admissions=admissions,
            operations=operations,
            request_pool_indices=request_pool_indices,
            kv_placements=kv_placements,
            kv_branch_placements=kv_branch_placements,
            latent_placements=latent_placements,
            input_products=input_products,
            tensorized_mixed=tensorized_mixed,
        )

    def _warmup_branch_placements(
        self,
        operation: Operation,
        height: int,
        width: int,
    ) -> tuple[KvBranchPlacement, ...]:
        session = self.sessions.get(operation.request_key.session_id)
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
        runtime = self.executor._parent_runtime(operation, session)
        query = generation.physical_tokens(height, width)
        image_prompt = image.image_prompts[0] if image.image_prompts else ""
        prefix_lengths = []
        for branch in guide.branches:
            prefix, copy_conditioning = generation.prefix(
                generation.branch_source(branch),
                image_prompt=image_prompt,
                negative_prompt=image.negative_prompt,
                negative_token_ids=session.negative_token_ids,
                tokenizer=self.executor.tokenizer,
            )
            prefix_lengths.append(int(runtime.kv_visible_len) if copy_conditioning else len(prefix))
        widths = tuple(
            ceil_div(query + prefix_length, self.cache_pool.block_size)
            for prefix_length in prefix_lengths
        )
        required = sum(widths)
        lease = self._warmup_scratch_pages.setdefault(operation.request_key, [])
        missing = required - len(lease)
        if missing < 0:
            raise invalid_descriptor("warmup generation scratch geometry changed")
        occupied = {
            page
            for request_key, pages in self._warmup_scratch_pages.items()
            if request_key != operation.request_key
            for page in pages
        }
        allocated = tuple(
            page
            for page in range(
                self.cache_pool.scratch_page_offset,
                self.cache_pool.num_pages,
            )
            if page not in occupied
        )[:missing]
        if len(allocated) != missing:
            raise invalid_descriptor("warmup generation scratch exceeds fixed capacity")
        lease.extend(allocated)
        fresh = set(allocated)
        placements: list[KvBranchPlacement] = []
        offset = 0
        for branch_index, width in enumerate(widths, start=1):
            block_table = tuple(lease[offset : offset + width])
            placements.append(
                KvBranchPlacement(
                    request_key=operation.request_key,
                    op_id=operation.op_id,
                    branch_index=branch_index,
                    group_id=0,
                    block_table=block_table,
                    pages_to_zero=tuple(page for page in block_table if page in fresh),
                )
            )
            offset += width
        return tuple(placements)

    def warmup(self) -> None:
        """Complete pre-admission kernel JIT and open the serving epoch.

        The ``fa4_cute`` attention backend JIT-compiles its CUTLASS kernels the
        first time each variant runs, costing tens of seconds on the first real
        request. Representative operations run through the real execution path;
        startup succeeds only after every configured warmup completes and its
        private collective identities are retired.
        """

        import torch

        if torch.device(self.deployment.device).type == "cuda":
            self._warmup_sequence()
            self._warmup_flow()
        self.runner.complete_startup()
        self.executor.complete_startup()

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
            KvAdmission,
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
            TokenMode,
            UndAdmission,
            VersionRef,
            Work,
            encode_token_product_bytes,
        )

        variants = self._effective_work_variants
        if WorkVariant.TOKEN_EXTEND not in variants:
            return
        pool = self.cache_pool
        if self.sessions.session_ids():
            return
        if self._execution.cuda_graph and self._execution.prefill_cuda_graph:
            self._warmup_prefill_graphs()
        configured = (
            self._decode_graph_batch_sizes
            if (self._execution.cuda_graph and WorkVariant.TOKEN_DECODE in variants)
            else (1,)
        )
        batch_sizes = tuple(
            sorted(
                {int(value) for value in configured if 0 < int(value) < int(pool.request_pages)},
                reverse=True,
            )
        )
        if not batch_sizes:
            return
        session_ids = tuple(range(1, max(batch_sizes) + 1))
        keys = {sid: RequestKey(0, sid, 1) for sid in session_ids}
        admissions = {
            sid: Admission.create(
                keys[sid],
                request_pool_idx=sid,
                und=UndAdmission(
                    sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                    kv=KvAdmission(),
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
                work=Work.token(TokenMode.EXTEND),
                route=0,
                domain=Domain.UND,
                bounds=Bounds(max_points=1, max_tokens=max(1, len(tokens))),
                inputs=(token_ref,),
                outputs=outputs,
                kv_capacity_pages=1,
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
                work=Work.token(TokenMode.DECODE),
                route=0,
                domain=Domain.UND,
                bounds=Bounds(max_points=1, max_tokens=1),
                outputs=outputs,
                kv_capacity_pages=ceil_div(op_id, int(pool.block_size)),
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
                retain_device_outputs=WorkVariant.TOKEN_DECODE in variants,
            )
            predecessors.update(zip(session_ids, operations, strict=True))
            if WorkVariant.TOKEN_DECODE not in variants:
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
                    self.products.release(
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
            KvAdmission,
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
            TokenMode,
            UndAdmission,
            VersionRef,
            Work,
            encode_token_product_bytes,
        )

        pool = self.cache_pool
        if self.sessions.session_ids():
            return
        max_route_tokens = int(self.model.text_max_tokens)
        capacity = min(
            max_route_tokens,
            max(0, int(pool.request_pages) - 1) * int(pool.block_size),
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
        logger.info("warming %d paged-prefill CUDA graph token buckets", len(token_buckets))
        session_id = 0
        # First warm every configured physical call in descending footprint,
        # then capture every bucket in the same order.
        for _ in range(2):
            for token_count in token_buckets:
                block_count = (token_count + int(pool.block_size) - 1) // int(pool.block_size)
                tokens = (0,) * token_count
                session_id += 1
                rk = RequestKey(0, session_id, 1)
                admission = Admission.create(
                    rk,
                    request_pool_idx=1,
                    und=UndAdmission(
                        sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                        kv=KvAdmission(),
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
                operation = Operation.registered(
                    request_key=rk,
                    op_id=1,
                    parent=VersionRef(rk, 0, FixedPoint(0, admission.digest)),
                    work=Work.token(TokenMode.EXTEND),
                    route=0,
                    domain=Domain.UND,
                    bounds=Bounds(max_points=1, max_tokens=token_count),
                    inputs=(token_ref,),
                    outputs=_warmup_token_outputs(rk, 1, 2),
                    kv_capacity_pages=block_count,
                )
                try:
                    self._execute_warmup(
                        self._build_warmup_batch(
                            admissions=(admission,),
                            operations=(operation,),
                            input_products=(
                                ProductPayload(
                                    product=token_ref,
                                    payload=encode_token_product_bytes(tokens),
                                ),
                            ),
                        )
                    )
                finally:
                    self.drop_session(session_id)

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
            ImageParams,
            KvAdmission,
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
            TokenMode,
            TransferMode,
            UndAdmission,
            VersionRef,
            Work,
            encode_token_product_bytes,
        )

        if not {
            WorkVariant.GEN_TRANSITION,
            WorkVariant.GEN_FLOW,
        }.issubset(self._effective_work_variants) or not isinstance(
            self.model.generation, GenerationPipeline
        ):
            return
        if self.sessions.session_ids():
            return
        configured: tuple[tuple[int, int, int], ...]
        if not self._execution.cuda_graph:
            configured = ((1, *self._warmup_image_geometry()),)
        else:
            configured = tuple(
                sorted(
                    (
                        (int(batch_size), int(height), int(width))
                        for batch_size, height, width in self._flow_graph_buckets
                    ),
                    key=lambda value: value[0] * value[1] * value[2],
                    reverse=True,
                )
            )
        next_session_id = 1
        next_generation = 1
        for batch_size, height, width in configured:
            if batch_size > int(self._capabilities.max_request_pool_size):
                continue
            mixed_text_sizes = tuple(
                text_batch_size
                for text_batch_size, mixed_height, mixed_width in self._mixed_flow_graph_buckets
                if batch_size == 1
                and mixed_height == height
                and mixed_width == width
                and text_batch_size + batch_size <= int(self._capabilities.max_request_pool_size)
            )
            session_ids = tuple(range(next_session_id, next_session_id + batch_size))
            next_session_id += batch_size
            keys = tuple(RequestKey(0, session_id, 1) for session_id in session_ids)
            admissions = tuple(
                Admission.create(
                    key,
                    request_pool_idx=index,
                    gen_admission=GenAdmission(
                        image=ImageParams(
                            steps=2 + 2 * len(mixed_text_sizes),
                            height=height,
                            width=width,
                            seed=0,
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
                        kv=KvAdmission(),
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
                        work=Work("transfer", TransferMode.KV_PUBLISH.value),
                        route=0,
                        domain=Domain.UND,
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
                            work=Work.token(TokenMode.EXTEND),
                            route=0,
                            domain=Domain.UND,
                            bounds=Bounds(max_points=1, max_tokens=1),
                            inputs=(token_ref,),
                            outputs=prompt_outputs,
                            kv_capacity_pages=1,
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
                    )
                    text_predecessors.update(zip(text_session_ids, prompt_operations, strict=True))
                max_latent_elements = max(
                    1,
                    math.prod(self.model.generation.latent_shape(height, width)),
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
                            work=Work("gen", "transition"),
                            route=0,
                            domain=Domain.GEN,
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
                                work=Work("gen", "flow"),
                                route=0,
                                domain=Domain.GEN,
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
                    released_latents = tuple(
                        (reference.request_key, int(reference.producer_op_id))
                        for reference in current_latents
                    )
                    self.latents.release_operations(released_latents)
                    self.products.device_products.release_operations(released_latents)
                    current_latents = tuple(outputs)
                    flow_predecessors.update(zip(session_ids, flows, strict=True))
                flow_op_id = 5
                for text_batch_size in mixed_text_sizes:
                    selected_text = text_session_ids[:text_batch_size]
                    for _ in range(2):
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
                                    work=Work.token(TokenMode.DECODE),
                                    route=0,
                                    domain=Domain.UND,
                                    bounds=Bounds(max_points=1, max_tokens=1),
                                    outputs=token_outputs,
                                    kv_capacity_pages=ceil_div(
                                        op_id, int(self.cache_pool.block_size)
                                    ),
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
                                    work=Work("gen", "flow"),
                                    route=0,
                                    domain=Domain.GEN,
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
                        self.products.release(
                            tuple(
                                int(output.generation)
                                for session_id in selected_text
                                for output in text_predecessors[session_id].outputs
                                if output.storage_class is StorageClass.DEVICE_TENSOR
                            )
                        )
                        text_predecessors.update(zip(selected_text, text_operations, strict=True))
                        released_latents = tuple(
                            (reference.request_key, int(reference.producer_op_id))
                            for reference in current_latents
                        )
                        self.latents.release_operations(released_latents)
                        self.products.device_products.release_operations(released_latents)
                        current_latents = tuple(flow_outputs)
                        flow_predecessors.update(zip(session_ids, flow_operations, strict=True))
            finally:
                for session_id in (*session_ids, *text_session_ids):
                    self.drop_session(session_id)

    def drop_session(self, session_id: int) -> None:
        session_id = int(session_id)
        session = self.sessions.peek(session_id)
        self.executor.drop_session(session_id)
        self._release_records(self.products.session_records(session_id))
        self.products.drop(session_id)
        self.latents.drop_session(session_id)
        self.replay.drop_session(session_id)
        self.sessions.drop(session_id)
        if session is not None:
            for group_id in range(self.cache_pool.group_count):
                self._warmup_kv_pages.pop((session.request_key, group_id), None)
            self._warmup_scratch_pages.pop(session.request_key, None)
        if self.snapshot_provider is not None:
            self.snapshot_provider.drop_session(session_id)
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
        records = tuple(
            record for handle in handles if (record := self.products.get(int(handle))) is not None
        )
        self._release_records(records)
        self.products.release(tuple(int(handle) for handle in handles))
        self.sessions.discard_product_handles({int(handle) for handle in handles})

    def snapshot_session(self, placement: RecoveryPlacement) -> SnapshotRef:
        if self.snapshot_provider is None:
            raise capability_mismatch("this worker has no configured snapshot provider")
        return self.snapshot_provider.snapshot_session(placement)

    def restore_session(
        self,
        reference: SnapshotRef,
        placement: RecoveryPlacement,
    ) -> None:
        if self.snapshot_provider is None:
            raise capability_mismatch("this worker has no configured snapshot provider")
        self.snapshot_provider.restore(reference, placement)
        session = self.sessions.get(placement.request_key.session_id)
        valid_cache_length = max(
            (int(group.length) for group in placement.cache_groups),
            default=0,
        )
        self.runtime_states.reset(
            (int(session.request_pool_idx),),
            valid_cache_lengths=(valid_cache_length,),
            logical_lengths=(int(session.logical_position),),
            sampling_positions=(int(session.rng_counter),),
        )

    def resource_pressure(self) -> list[dict[str, object]]:
        caps = self._capabilities
        counts = {
            "image_latent": self.latents.resident_byte_count(),
            "encoder_output": self.products.encoder_output_count(),
        }
        totals = {
            "image_latent": int(self.latents.capacity_bytes),
            "encoder_output": int(caps.encoder_cache_budget),
        }
        return [
            _pressure(value.value, counts[value.value], totals[value.value])
            for value in caps.resource_classes
            if value.value in counts
        ]

    def close(self) -> None:
        self.runner.synchronize()
        self.executor.close()
        self.mover.close()
        self.latents.close()
        self.products.close()
        self.runner.close()

    def _release_records(self, records: tuple[object, ...]) -> None:
        from ..runtime.product_store import ProductRecord
        from ..runtime.transfer import Locator

        for record in records:
            if isinstance(record, ProductRecord) and record.locator:
                self.mover.transport.release(Locator.from_wire_json(record.locator))


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


__all__ = ["ModelWorker"]
