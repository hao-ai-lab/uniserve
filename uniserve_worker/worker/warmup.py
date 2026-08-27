"""Packed-forward graph catalog and startup workload construction."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from ..batch import (
    Admission,
    AttentionRegime,
    Batch,
    BatchPartition,
    BlockTable,
    CachePageAllocation,
    CompletionReport,
    Domain,
    ForwardMode,
    ImageParams,
    LatentPlacement,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    RowGeometry,
    StorageClass,
)
from ..bootstrap.execution_config import (
    LaneConfig,
)
from ..capabilities import (
    MixedExecutionCapability,
)
from ..execution.step import (
    complete_startup,
    execute_startup,
    parent_runtime,
)
from ..foundation.errors import invalid_descriptor
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline
from ..nn.diffusion.cfg import build_flow_cfg_plan
from ..server.completion import (
    completion_report_ready,
    finalize_completion_report,
)

if TYPE_CHECKING:
    from .worker import Worker

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


def _execute_warmup(
    self: Worker,
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
        completion for completion in finalized.completions if completion.status is OpStatus.ERROR
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
    self: Worker,
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
    height, width = image_geometry or _warmup_image_geometry(self)
    latent_units = max(
        1,
        (height // max(1, int(self._capabilities.latent_downsample)))
        * (width // max(1, int(self._capabilities.latent_downsample))),
    )
    page_units = int(self._capabilities.latent_page_units)
    latent_page_count = (latent_units + page_units - 1) // page_units if page_units > 0 else 0
    occupied_latent_pages = {page for pages in self._warmup_latent_pages.values() for page in pages}
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
                int(operation.bounds.max_tokens) if operation.work is ForwardMode.GEN_FLOW else 0
            ),
        )
        if operation.work is ForwardMode.GEN_FLOW:
            extra_tables, extra_allocations, flow_rows = _warmup_flow_tables(
                self,
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
    self: Worker,
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


def warmup(self: Worker) -> None:
    """Complete pre-admission kernel JIT and open the serving epoch.

    The ``fa4_cute`` attention backend JIT-compiles its CUTLASS kernels the
    first time each variant runs, costing tens of seconds on the first real
    request. Representative operations run through the real execution path;
    startup succeeds only after every configured warmup completes and its
    private collective identities are retired.
    """

    product_devices = (
        self.deployment.device,
        self.deployment.generation_device or self.deployment.device,
    )
    self.device_products.warmup_scattered_publication(product_devices)
    if torch.device(self.deployment.device).type == "cuda":
        _warmup_sequence(self)
        logger.info("completed token CUDA graph warmup")
        _warmup_flow(self)
        logger.info("completed flow CUDA graph warmup")
    elif self._capabilities.mixed_buckets:
        _warmup_flow(self)
        logger.info("completed mixed execution warmup")
    complete_startup(self.execution)
    logger.info("completed execution partition startup verification")


def _warmup_image_geometry(self: Worker) -> tuple[int, int]:
    """Largest square image whose latent grid fits the declared capacity."""

    import math

    caps = self._capabilities
    downsample = max(1, int(caps.latent_downsample))
    capacity = int(caps.latent_capacity_units)
    if int(caps.max_vae_grid_tokens) > 0:
        capacity = min(capacity, int(caps.max_vae_grid_tokens))
    side = max(1, math.isqrt(max(1, capacity)))
    return side * downsample, side * downsample


def _warmup_sequence(self: Worker) -> None:
    """Warm the real token forward paths and capture the configured graphs.

    One prompt extend across the largest configured decode batch pays the
    first-use kernel JIT; the paged-prefill CUDA graph is captured for
    every configured token bucket; the decode CUDA graph is captured for
    every configured batch size (two rounds each: capture, then replay).
    """

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
        _warmup_prefill_graphs(self)
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
        _execute_warmup(
            self,
            _build_warmup_batch(
                self,
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
                _execute_warmup(
                    self,
                    _build_warmup_batch(
                        self,
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


def _warmup_prefill_graphs(self: Worker) -> None:
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
            {int(value) for value in self._prefill_graph_token_sizes if 0 < int(value) <= capacity},
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
                _execute_warmup(
                    self,
                    _build_warmup_batch(
                        self,
                        admissions=tuple(admissions),
                        operations=tuple(operations),
                        input_products=tuple(input_products),
                    ),
                )
            finally:
                for active_session in active_sessions:
                    self.drop_session(active_session)


def _warmup_flow(self: Worker) -> None:
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
        height, width = _warmup_image_geometry(self)
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
        mixed_rounds = 3 if self._execution.cuda_graph and self._execution.prefill_cuda_graph else 1
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
        text_keys = {session_id: RequestKey(0, session_id, 1) for session_id in text_session_ids}
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
            _execute_warmup(
                self,
                _build_warmup_batch(
                    self,
                    admissions=admissions,
                    operations=tuple(publications),
                    image_geometry=(height, width),
                ),
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
                _execute_warmup(
                    self,
                    _build_warmup_batch(
                        self,
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
            _execute_warmup(
                self,
                _build_warmup_batch(
                    self,
                    admissions=(),
                    operations=tuple(transitions),
                    image_geometry=(height, width),
                ),
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
                _execute_warmup(
                    self,
                    _build_warmup_batch(
                        self,
                        admissions=(),
                        operations=tuple(flows),
                        image_geometry=(height, width),
                    ),
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
                    _execute_warmup(
                        self,
                        _build_warmup_batch(
                            self,
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
