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
from ..capabilities import RequestKind, WorkerCapabilities
from ..execution import ModelExecutor, ModelRunner
from ..execution.executor import completion_report_ready, finalize_completion_report
from ..forward import AttentionSelection
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..foundation.runtime_config import ExecutionConfig, graph_memory_budget_bytes
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
from ..runtime.graph_store import GraphStore
from ..runtime.latent_store import LatentStore
from ..runtime.mesh_store import MeshStore
from ..runtime.mover import Mover
from ..runtime.product_store import ProductStore
from ..runtime.replay import ReplayStore
from ..runtime.request_session import SessionStore
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
            submission_group=index,
            collective_seq=max(1, int(step_id) * 16 + index),
            domain=domain,
            route=route,
            execution=ExecutionCapability.DOMAIN_HOMOGENEOUS,
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
            raise capability_mismatch("model worker requires nn.Module.forward(ForwardBatch)")
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
        self.graphs = GraphStore(
            enabled=execution.cuda_graph,
            prefill_enabled=execution.prefill_cuda_graph,
            cache=model.cache_geometry,
            block_size=deployment.block_size,
            weight_digest=self.weight_digest,
            memory_budget_bytes=graph_memory_budget_bytes(device_total_bytes(deployment.device)),
            decode_batch_sizes=execution.cuda_graph_warmup_batches,
            decode_context_blocks=self._decode_context_blocks(),
            prefill_token_sizes=execution.prefill_cuda_graph_warmup_tokens,
        )
        self._execution = execution
        self.trace = ExecutionTrace(self.identity.architecture_digest)
        self.executor = ModelExecutor(
            model=model,
            deployment=deployment,
            runner=ModelRunner(model, self.graphs, self.trace),
            attention=attention,
            sessions=self.sessions,
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
            pinned_staging_capacity=arena.pinned_staging_bytes,
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
        max_tokens = max(
            (
                self.model.route_max_tokens(stage.route)
                for variant in self.model.supported_work
                for stage in self.model.lower(variant)
                if stage.row.value == "token"
            ),
            default=0,
        )
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
        height, width = self._warmup_image_geometry()
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
        if (
            self._execution.cuda_graph
            and self._execution.prefill_cuda_graph
            and self._execution.prefill_cuda_graph_warmup
        ):
            self._warmup_prefill_graphs()
        configured = (
            self._execution.cuda_graph_warmup_batches
            if (
                self._execution.cuda_graph
                and self._execution.cuda_graph_warmup
                and WorkVariant.TOKEN_DECODE in variants
            )
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
                kv_capacity_pages=1,
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
            repeats = 2 if self._execution.cuda_graph and self._execution.cuda_graph_warmup else 1
            for batch_size in batch_sizes:
                selected = session_ids[:batch_size]
                for _ in range(repeats):
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
        max_route_tokens = max(
            (
                self.model.route_max_tokens(stage.route)
                for variant in self.model.supported_work
                for stage in self.model.lower(variant)
                if stage.row.value == "token"
            ),
            default=0,
        )
        capacity = min(
            max_route_tokens,
            max(0, int(pool.request_pages) - 1) * int(pool.block_size),
        )
        token_buckets = tuple(
            sorted(
                {
                    int(value)
                    for value in self._execution.prefill_cuda_graph_warmup_tokens
                    if 0 < int(value) <= capacity
                },
                reverse=True,
            )
        )
        if not token_buckets:
            return
        logger.info("warming %d paged-prefill CUDA graph token buckets", len(token_buckets))
        session_id = 0
        for token_count in token_buckets:
            block_count = (token_count + int(pool.block_size) - 1) // int(pool.block_size)
            tokens = (0,) * token_count
            # Two rounds per bucket: the first captures the graph, the second
            # replays it. Every round uses a fresh session and operation id so
            # no replay or session state carries between rounds.
            for _ in range(2):
                session_id += 1
                rk = RequestKey(0, session_id, 1)
                admission = Admission.create(
                    rk,
                    request_pool_idx=session_id,
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
            Domain,
            DrawLayout,
            DType,
            FixedPoint,
            GenAdmission,
            ImageParams,
            Operation,
            PointRange,
            ProductKind,
            ProductRef,
            RequestKey,
            Rng,
            ShapeBound,
            StaticDim,
            StorageClass,
            TransferMode,
            VersionRef,
            Work,
        )

        if WorkVariant.GEN_TRANSITION not in self._effective_work_variants or not isinstance(
            self.model.generation, GenerationPipeline
        ):
            return
        if self.sessions.session_ids():
            return
        session_id = 2
        rk = RequestKey(0, session_id, 1)
        height, width = self._warmup_image_geometry()
        admission = Admission.create(
            rk,
            request_pool_idx=session_id,
            gen_admission=GenAdmission(
                image=ImageParams(steps=1, height=height, width=width, seed=0)
            ),
        )
        try:
            root = VersionRef(rk, 0, FixedPoint(0, admission.digest))
            conditioning = ProductRef(
                request_key=rk,
                producer_op_id=1,
                output_index=0,
                generation=4,
                kind=ProductKind.KV,
                storage_class=StorageClass.PAGED_KV,
                dtype=DType.U8,
                shape_bound=ShapeBound((DeviceDim(1 << 20),)),
                point_range=PointRange(),
            )
            publication = Operation.registered(
                request_key=rk,
                op_id=1,
                parent=root,
                work=Work("transfer", TransferMode.KV_PUBLISH.value),
                route=0,
                domain=Domain.UND,
                bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
                outputs=(conditioning,),
            )
            self._execute_warmup(
                self._build_warmup_batch(admissions=(admission,), operations=(publication,))
            )
            max_latent_elements = max(
                1,
                math.prod(self.model.generation.latent_shape(height, width)),
            )
            initial_latent = ProductRef(
                request_key=rk,
                producer_op_id=2,
                output_index=0,
                generation=5,
                kind=ProductKind.LATENT,
                storage_class=StorageClass.LATENT_ARENA,
                dtype=DType.BF16,
                shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                point_range=PointRange(),
            )
            transition_ready = ProductRef(
                request_key=rk,
                producer_op_id=2,
                output_index=1,
                generation=6,
                kind=ProductKind.COMPLETION,
                storage_class=StorageClass.DEVICE_TENSOR,
                dtype=DType.U32,
                shape_bound=ShapeBound((StaticDim(1),)),
                point_range=PointRange(),
            )
            transition = Operation.registered(
                request_key=rk,
                op_id=2,
                parent=root,
                work=Work("gen", "transition"),
                route=0,
                domain=Domain.GEN,
                bounds=Bounds(max_points=1, max_tokens=1, max_latent_bytes=max_latent_elements * 2),
                inputs=(conditioning,),
                outputs=(initial_latent, transition_ready),
                rng=Rng(
                    seed=0,
                    semantic_index_base=1,
                    draw_layout=DrawLayout.FLOW_NOISE,
                ),
            )
            self._execute_warmup(self._build_warmup_batch(admissions=(), operations=(transition,)))
            next_latent = ProductRef(
                request_key=rk,
                producer_op_id=3,
                output_index=0,
                generation=7,
                kind=ProductKind.LATENT,
                storage_class=StorageClass.LATENT_ARENA,
                dtype=DType.BF16,
                shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                point_range=PointRange(),
            )
            flow = Operation.registered(
                request_key=rk,
                op_id=3,
                parent=self.sessions.get(session_id).committed_version(),
                work=Work("gen", "flow"),
                route=0,
                domain=Domain.GEN,
                bounds=Bounds(max_points=1, max_tokens=1, max_latent_bytes=max_latent_elements * 2),
                inputs=(conditioning, initial_latent),
                outputs=(next_latent,),
            )
            self._execute_warmup(
                self._build_warmup_batch(admissions=(), operations=(flow,), input_products=())
            )
        finally:
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
        self.executor.close()
        self.graphs.close()
        self.mover.close()

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
