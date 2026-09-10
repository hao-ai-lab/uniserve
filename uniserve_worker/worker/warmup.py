"""Bounded request, sampling, and product startup scenarios."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..execution.batch import (
    AttentionRegime,
    BatchCommand,
    BlockTable,
    BufferAllocation,
    BufferId,
    CachePageAllocation,
    Domain,
    Free,
    LatentParams,
    ModelOutput,
    NewRequest,
    OpCode,
    Operation,
    OpStatus,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    Retire,
    RowGeometry,
    Run,
    RunLane,
    RunResult,
    Start,
    StorageClass,
)
from ..execution.model_runner import capture_image_parameters
from ..execution.output import (
    finalize_run_result,
    run_result_ready,
)
from ..execution.runners.packed import FlowCapture
from ..foundation.errors import invalid_descriptor
from ..foundation.math import ceil_div
from ..models.generation import GenerationPipeline
from ..nn.diffusion.cfg import build_flow_cfg_plan

if TYPE_CHECKING:
    from .worker import Worker

logger = logging.getLogger(__name__)


class _WarmupRequests:
    """Track synthetic request allocations while borrowing Worker execution resources."""

    def __init__(self, worker: Worker) -> None:
        """Borrow startup-owned worker resources needed to exercise every execution shape."""

        self.worker = worker
        self._kv_pages: dict[tuple[RequestKey, int], list[int]] = {}
        self._prefix_pages: dict[RequestKey, list[int]] = {}
        self._prefix_slots: dict[RequestKey, int] = {}
        self._latent_pages: dict[RequestKey, list[int]] = {}
        self._buffers: dict[BufferId, BufferAllocation] = {}
        self._free_buffer_ranges: list[tuple[int, int]] = [
            (0, int(self.worker.info.buffer_pool_bytes))
        ]
        self._run_id = 0

    def drop_request(self, request_id: int) -> None:
        """Release warmup request, runtime, cache, latent, and product state for one identifier."""

        request = self.worker.requests.peek(int(request_id))
        if request is None:
            return
        self._execute_controls((Retire(request.request_key),))
        # Serving keeps a terminal row until its slot is reassigned. Synthetic
        # requests have no further scheduler messages and can leave the table.
        self.worker.requests.drop(request_id)
        for group_id in range(
            0 if self.worker.cache_pool is None else self.worker.cache_pool.group_count
        ):
            self._kv_pages.pop((request.request_key, group_id), None)
        self._prefix_pages.pop(request.request_key, None)
        self._prefix_slots.pop(request.request_key, None)
        self._latent_pages.pop(request.request_key, None)
        released = tuple(
            buffer
            for buffer, allocation in self._buffers.items()
            if int(allocation.buffer.owner.request_id) == int(request_id)
        )
        self._release_buffer_allocations(released)

    def free_products(self, buffers: tuple[BufferId, ...]) -> None:
        """Release warmup products and recycle their synthetic persistent-buffer allocations."""

        if not buffers:
            return
        self._execute_controls(tuple(Free(buffer) for buffer in buffers))
        self._release_buffer_allocations(buffers)

    def _execute_controls(self, commands: tuple[BatchCommand, ...]) -> None:
        """Wait for the same physical retirement acknowledgement used by serving."""

        self._run_id += 1
        _execute_warmup(
            self,
            Run(
                batch_id=self._run_id,
                run_id=self._run_id,
                commands=commands,
            ),
            retain_device_outputs=True,
        )

    def _release_buffer_allocations(self, buffers: tuple[BufferId, ...]) -> None:
        """Release persistent warmup allocations by exact buffer identity."""

        for buffer in buffers:
            allocation = self._buffers.pop(buffer, None)
            if allocation is not None:
                self._free_buffer_ranges.append((allocation.offset, allocation.bytes))
        if not self._free_buffer_ranges:
            return
        merged: list[tuple[int, int]] = []
        for offset, extent in sorted(self._free_buffer_ranges):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous, size = merged[-1]
                merged[-1] = (previous, size + extent)
            else:
                merged.append((offset, extent))
        self._free_buffer_ranges = merged

    def buffer_allocation(self, product: ProductRef) -> BufferAllocation:
        """Allocate a deterministic aligned slice of warmup persistent storage for a product."""

        existing = self._buffers.get(product.buffer_id)
        if existing is not None:
            return existing
        alignment = 256
        required = int(product.max_bytes)
        for index, (offset, extent) in enumerate(self._free_buffer_ranges):
            aligned = (offset + alignment - 1) & ~(alignment - 1)
            end = aligned + required
            if end > offset + extent:
                continue
            replacement: list[tuple[int, int]] = []
            if aligned > offset:
                replacement.append((offset, aligned - offset))
            if end < offset + extent:
                replacement.append((end, offset + extent - end))
            self._free_buffer_ranges[index : index + 1] = replacement
            allocation = BufferAllocation(product.buffer_id, aligned, required)
            self._buffers[product.buffer_id] = allocation
            return allocation
        raise invalid_descriptor("warmup persistent buffer allocation exceeds resident capacity")


def _warmup_batch(
    *,
    run_id: int,
    admissions: tuple[NewRequest, ...],
    operations: tuple[Operation, ...],
    block_tables: dict[tuple[RequestKey, int], tuple[BlockTable, ...]],
    new_cache_pages: dict[tuple[RequestKey, int], tuple[CachePageAllocation, ...]],
    forward_rows: dict[tuple[RequestKey, int], tuple[RowGeometry, ...]],
    latent_params: dict[tuple[RequestKey, int], LatentParams],
    buffer_allocations: tuple[BufferAllocation, ...],
    input_products: tuple[ProductPayload, ...] = (),
    tensorized_mixed: bool = False,
) -> Run:
    """Assemble warmup operations into domain lanes with their physical allocations."""

    # Preserve operation order within each execution domain while assigning a
    # shared launch identity only for tensorized mixed qualification.
    groups: list[tuple[Domain, int, list[Operation]]] = []
    for operation in operations:
        existing = next(
            (members for domain, _route, members in groups if domain is operation.domain),
            None,
        )
        if existing is None:
            groups.append((operation.domain, 0, [operation]))
        else:
            existing.append(operation)
    # Every lane carries only the tables, rows, and buffers referenced by its members.
    lanes = tuple(
        RunLane(
            lane_id=index,
            launch_id=1 if tensorized_mixed else index,
            collective_seq=max(
                1,
                int(run_id) * 16 + (1 if tensorized_mixed else index),
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
            latent_params=tuple(
                latent_params[(operation.request_key, operation.op_id)]
                for operation in members
                if operation.kind
                in {
                    OpCode.DIFFUSION_PREPARE,
                    OpCode.DIFFUSION_STEP,
                    OpCode.DIFFUSION_FINALIZE,
                }
            ),
            buffer_allocations=tuple(
                allocation
                for allocation in buffer_allocations
                if any(
                    product.buffer_id == allocation.buffer
                    for operation in members
                    for product in (
                        *operation.inputs,
                        *operation.outputs,
                        *((operation.predicate,) if operation.predicate is not None else ()),
                    )
                )
            ),
        )
        for index, (domain, route, members) in enumerate(groups, start=1)
    )
    return Run(
        batch_id=run_id,
        run_id=run_id,
        lanes=lanes,
        input_products=input_products,
        commands=tuple(Start(request) for request in admissions),
    )


def _warmup_token_outputs(
    request_key: RequestKey,
    op_id: int,
    first_generation: int,
) -> tuple[ProductRef, ...]:
    """Declare generation-tagged token and transition products for warmup sampling."""

    from ..execution.batch import (
        DType,
        PointRange,
        ProductKind,
        ShapeBound,
        StorageClass,
    )

    definitions = [(0, ProductKind.TOKEN, DType.U32, ShapeBound())]
    return tuple(
        ProductRef(
            request_key=request_key,
            producer_op_id=op_id,
            output_index=output_index,
            generation=first_generation + generation_offset,
            kind=kind,
            storage_class=StorageClass.REQUEST_RELAY,
            dtype=dtype,
            shape_bound=shape,
            point_range=PointRange(base_point=0, max_points=1),
        )
        for generation_offset, (output_index, kind, dtype, shape) in enumerate(definitions)
    )


def _execute_warmup(
    requests: _WarmupRequests,
    batch: Run,
    *,
    retain_device_outputs: bool = False,
) -> RunResult:
    """Execute a runtime scenario and optionally retain its published outputs."""

    report = requests.worker._execute_batch(batch, propagate_errors=True)
    while not run_result_ready(report):
        time.sleep(0.00005)
    finalized = finalize_run_result(report)
    device_buffers = tuple(
        output.buffer_id
        for operation in batch.operations
        for output in operation.outputs
        if output.storage_class in {StorageClass.DEVICE_TENSOR, StorageClass.REQUEST_RELAY}
    )
    failures: list[ModelOutput] = []
    for completion in finalized.completions:
        if not isinstance(completion, ModelOutput):
            raise RuntimeError("finalized warmup result retains unresolved device output")
        if completion.status is OpStatus.ERROR:
            failures.append(completion)
    if failures:
        details = ", ".join(
            f"request={completion.request_key.request_id} op={completion.op_id} "
            f"code={completion.error_code.value if completion.error_code is not None else 'internal'}"
            for completion in failures
        )
        raise RuntimeError(f"startup warmup execution failed: {details}")
    if not retain_device_outputs:
        requests.free_products(device_buffers)
    return finalized


def _build_warmup_batch(
    requests: _WarmupRequests,
    *,
    admissions: tuple[NewRequest, ...],
    operations: tuple[Operation, ...],
    input_products: tuple[ProductPayload, ...] = (),
    tensorized_mixed: bool = False,
    image_geometry: tuple[int, int] | None = None,
) -> Run:
    """Derive cache, latent, buffer, and row allocations for a warmup submission."""

    requests._run_id += 1
    admissions_by_key = {admission.request_key: admission for admission in admissions}
    occupied_blocks = {page for pages in requests._kv_pages.values() for page in pages}
    request_pool_indices: dict[RequestKey, int] = {}
    block_tables: dict[tuple[RequestKey, int], tuple[BlockTable, ...]] = {}
    new_cache_pages: dict[tuple[RequestKey, int], tuple[CachePageAllocation, ...]] = {}
    forward_rows: dict[tuple[RequestKey, int], tuple[RowGeometry, ...]] = {}
    latent_params: dict[tuple[RequestKey, int], LatentParams] = {}
    buffer_allocations: dict[BufferId, BufferAllocation] = {}
    # Persistent products reserve stable buffer allocations before lane construction.
    for operation in operations:
        for product in (
            *operation.inputs,
            *operation.outputs,
            *((operation.predicate,) if operation.predicate is not None else ()),
        ):
            if not product.uses_persistent_buffer():
                continue
            allocation = requests.buffer_allocation(product)
            buffer_allocations[allocation.buffer] = allocation
    # Bind request slots and grow reusable KV leases to each operation's maximum shape.
    for operation in operations:
        request = requests.worker.requests.peek(int(operation.request_key.request_id))
        admission = admissions_by_key.get(operation.request_key)
        if request is None and admission is None:
            raise invalid_descriptor("warmup operation has no request-pool binding")
        if request is None:
            assert admission is not None
            request_pool_indices[operation.request_key] = admission.request_pool_idx
        else:
            request_pool_indices[operation.request_key] = request.request_pool_idx
        if operation.kind not in {
            OpCode.AR_EXTEND,
            OpCode.AR_DECODE,
            OpCode.AR_VERIFY,
            OpCode.TRANSFER_KV_PUBLISH,
            OpCode.TRANSFER_KV_INSTALL,
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
        }:
            continue
        if (
            request is None
            and admission is not None
            and admission.ar is not None
            and admission.ar.initial_position != 0
        ):
            raise invalid_descriptor("warmup KV admission requires an empty prefix")
        visible = 0
        if request is not None:
            runtime = request.parent_runtime(operation.parent)
            visible = int(runtime.kv_visible_len)
        input_length = (
            int(operation.bounds.max_tokens)
            if operation.kind
            in {
                OpCode.AR_EXTEND,
                OpCode.AR_DECODE,
                OpCode.AR_VERIFY,
            }
            else 0
        )
        tables: list[BlockTable] = []
        allocations: list[CachePageAllocation] = []
        if requests.worker.cache_pool is None:
            raise invalid_descriptor("warmup KV operation requires cache storage")
        for group_id in range(requests.worker.cache_pool.group_count):
            lease_key = (operation.request_key, group_id)
            block_table = requests._kv_pages.setdefault(lease_key, [])
            target_pages = ceil_div(
                visible + input_length,
                int(requests.worker.cache_pool.block_size),
            )
            missing = target_pages - len(block_table)
            if missing < 0:
                raise invalid_descriptor("warmup operation regresses its KV capacity")
            allocated = tuple(
                candidate
                for candidate in requests.worker.cache_pool.page_ids(group_id)
                if candidate not in occupied_blocks
            )[:missing]
            if len(allocated) != missing:
                raise invalid_descriptor(
                    "warmup KV allocation exceeds resident capacity: "
                    f"request={operation.request_key.request_id}, group={group_id}, "
                    f"required_pages={missing}, available_pages={len(allocated)}, "
                    f"resident_pages={len(requests.worker.cache_pool.page_ids(group_id))}, "
                    f"leased_pages={len(occupied_blocks)}"
                )
            block_table.extend(allocated)
            occupied_blocks.update(allocated)
            tables.append(
                BlockTable(
                    request_pool_idx=request_pool_indices[operation.request_key],
                    group_id=group_id,
                    page_ids=tuple(block_table),
                    allocated_tokens=len(block_table) * requests.worker.cache_pool.block_size,
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
                    write_kv=True,
                ),
            )
    height, width = image_geometry or _warmup_image_geometry(requests)
    latent_units = max(
        1,
        (height // max(1, int(requests.worker._layout.latent_downsample)))
        * (width // max(1, int(requests.worker._layout.latent_downsample))),
    )
    page_units = int(requests.worker.info.latent_page_units)
    latent_page_count = (latent_units + page_units - 1) // page_units if page_units > 0 else 0
    occupied_latent_pages = {page for pages in requests._latent_pages.values() for page in pages}
    for operation in operations:
        if operation.kind not in {
            OpCode.DIFFUSION_PREPARE,
            OpCode.DIFFUSION_STEP,
        } and not any(product.kind is ProductKind.LATENT for product in operation.inputs):
            continue
        page_table = requests._latent_pages.setdefault(operation.request_key, [])
        missing = latent_page_count - len(page_table)
        if missing < 0:
            raise invalid_descriptor("warmup latent allocation regresses its physical extent")
        allocated = tuple(
            page
            for page in range(1, int(requests.worker.info.latent_pages))
            if page not in occupied_latent_pages
        )[:missing]
        if len(allocated) != missing:
            raise invalid_descriptor("warmup latent allocation exceeds resident capacity")
        page_table.extend(allocated)
        occupied_latent_pages.update(allocated)
        request = requests.worker.requests.peek(int(operation.request_key.request_id))
        start_step = 0 if request is None else int(request.flow_step)
        latent_params[(operation.request_key, operation.op_id)] = LatentParams(
            request_key=operation.request_key,
            op_id=operation.op_id,
            page_table=tuple(page_table),
            latent_units=latent_units,
            height=height,
            width=width,
            start_step=start_step,
            step_count=(
                int(operation.bounds.max_tokens) if operation.kind is OpCode.DIFFUSION_STEP else 0
            ),
        )
        if operation.kind is OpCode.DIFFUSION_STEP:
            extra_tables, extra_allocations, flow_rows = _warmup_flow_tables(
                requests,
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
        run_id=requests._run_id,
        admissions=admissions,
        operations=operations,
        block_tables=block_tables,
        new_cache_pages=new_cache_pages,
        forward_rows=forward_rows,
        latent_params=latent_params,
        buffer_allocations=tuple(buffer_allocations.values()),
        input_products=input_products,
        tensorized_mixed=tensorized_mixed,
    )


def _warmup_flow_tables(
    requests: _WarmupRequests,
    operation: Operation,
    main_slot: int,
    height: int,
    width: int,
) -> tuple[
    tuple[BlockTable, ...],
    tuple[CachePageAllocation, ...],
    tuple[RowGeometry, ...],
]:
    """Build alternative-prefix KV tables and forward rows for all active CFG branches."""

    request = requests.worker.requests.get(operation.request_key.request_id)
    image = request.image
    generation = requests.worker.model.generation
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
    runtime = request.parent_runtime(operation.parent)
    query = generation.physical_tokens(height, width)
    image_prompt = image.image_prompts[0] if image.image_prompts else ""
    # Branches either reuse the conditioned request slot or share one alternative prefix.
    branch_prefixes: list[tuple[tuple[int, ...], bool]] = []
    for branch in guide.branches:
        prefix, copy_conditioning = generation.prefix(
            generation.branch_source(branch),
            image_prompt=image_prompt,
            negative_prompt=image.negative_prompt,
            negative_token_ids=request.negative_token_ids,
            tokenizer=requests.worker.tokenizer,
        )
        branch_prefixes.append((prefix, copy_conditioning))
    alternatives = {
        prefix for prefix, copy_conditioning in branch_prefixes if not copy_conditioning
    }
    if len(alternatives) > 1:
        raise invalid_descriptor("warmup flow has multiple distinct alternative prefixes")
    alternative = next(iter(alternatives), ())
    if requests.worker.cache_pool is None:
        raise invalid_descriptor("warmup flow requires KV cache storage")
    required = ceil_div(len(alternative), requests.worker.cache_pool.block_size)
    lease = requests._prefix_pages.setdefault(operation.request_key, [])
    missing = required - len(lease)
    occupied = {
        page
        for request_key, pages in requests._prefix_pages.items()
        if request_key != operation.request_key
        for page in pages
    }
    occupied.update(page for pages in requests._kv_pages.values() for page in pages)
    # Prefix pages persist across warmup shapes so graph capture observes stable tables.
    allocated = tuple(
        page for page in requests.worker.cache_pool.page_ids(0) if page not in occupied
    )[:missing]
    if len(allocated) != missing:
        raise invalid_descriptor("warmup alternative prefix exceeds KV capacity")
    lease.extend(allocated)
    tables: tuple[BlockTable, ...] = ()
    allocations: tuple[CachePageAllocation, ...] = ()
    alternative_slot = main_slot
    rows: list[RowGeometry] = []
    if alternative:
        alternative_slot = requests._prefix_slots.setdefault(
            operation.request_key,
            int(requests.worker.info.request_slots) - len(requests._prefix_slots),
        )
        if alternative_slot == main_slot or alternative_slot < 1:
            raise invalid_descriptor("warmup has no request slot for an alternative prefix")
        tables = (
            BlockTable(
                request_pool_idx=alternative_slot,
                group_id=0,
                page_ids=tuple(lease),
                allocated_tokens=len(lease) * requests.worker.cache_pool.block_size,
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
                write_kv=True,
            )
        )
    # Emit row geometry in exact guidance-branch evaluation order.
    for prefix, copy_conditioning in branch_prefixes:
        rows.append(
            RowGeometry(
                operation_index=0,
                request_pool_index=main_slot if copy_conditioning else alternative_slot,
                seq_len=int(runtime.kv_visible_len) if copy_conditioning else len(prefix),
                query_len=query,
                write_kv=False,
            )
        )
    return tables, allocations, tuple(rows)


def warmup_requests(worker: Worker) -> None:
    """Exercise synthetic requests through the configured execution paths.

    Successful scenarios retire their requests before returning. If execution
    fails, leave resource release to the enclosing Worker scope instead of
    issuing more execution commands that could replace the startup error.
    """

    requests = _WarmupRequests(worker)
    product_devices = (
        worker.worker_config.device,
        worker.worker_config.generation_device or worker.worker_config.device,
    )
    worker.device_products.warmup_scattered_publication(product_devices)
    if torch.device(worker.worker_config.device).type == "cuda":
        if OpCode.AR_EXTEND in worker.info.supported_ops:
            _warmup_tokens(requests)
            logger.info("completed token runtime warmup")
        if isinstance(worker.model.generation, GenerationPipeline):
            _warmup_flow(requests)
            logger.info("completed flow runtime warmup")
    elif worker.runner.mixed_captures:
        _warmup_flow(requests)
        logger.info("completed mixed execution warmup")


def _warmup_image_geometry(requests: _WarmupRequests) -> tuple[int, int]:
    """Derive the largest square image whose latent grid fits the declared capacity."""

    downsample = max(1, int(requests.worker._layout.latent_downsample))
    capacity = int(requests.worker.info.latent_capacity_units)
    if int(requests.worker._layout.max_vae_grid_tokens) > 0:
        capacity = min(capacity, int(requests.worker._layout.max_vae_grid_tokens))
    side = max(1, math.isqrt(max(1, capacity)))
    return side * downsample, side * downsample


def _warmup_tokens(requests: _WarmupRequests) -> None:
    """Exercise extend-to-decode token handoff and release its synthetic request."""

    from ..execution.batch import (
        ArRequestParams,
        Bounds,
        Checkpoint,
        DeviceSelected,
        DType,
        FixedCheckpoint,
        NewRequest,
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
        encode_token_product_bytes,
    )

    variants = requests.worker.info.supported_ops
    if OpCode.AR_EXTEND not in variants:
        return
    pool = requests.worker.cache_pool
    if pool is None:
        raise invalid_descriptor("autoregressive warmup requires KV cache storage")
    if requests.worker.requests.request_ids():
        return
    batch_sizes = (1,)
    request_ids = tuple(range(1, max(batch_sizes) + 1))
    keys = {sid: RequestKey(0, sid, 1) for sid in request_ids}
    admissions = {
        sid: NewRequest.create(
            keys[sid],
            request_pool_idx=sid,
            ar=ArRequestParams(
                sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                initial_position=0,
            ),
        )
        for sid in request_ids
    }

    next_product_generation = 1

    def prompt_op(
        sid: int,
        op_id: int,
        parent: Checkpoint,
        tokens: tuple[int, ...],
    ) -> tuple[Operation, ProductPayload]:
        """Build one prompt operation and its synthetic token input publication."""

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
            kind=OpCode.AR_EXTEND,
            bounds=Bounds(max_points=1, max_tokens=max(1, len(tokens))),
            inputs=(token_ref,),
            outputs=outputs,
        )
        return operation, ProductPayload(
            product=token_ref, payload=encode_token_product_bytes(tokens)
        )

    def decode_op(sid: int, op_id: int, predecessor: Operation) -> Operation:
        """Build one decode operation consuming the predecessor's token product."""

        nonlocal next_product_generation
        token_output = next(
            output for output in predecessor.outputs if output.kind is ProductKind.TOKEN
        )
        outputs = _warmup_token_outputs(keys[sid], op_id, next_product_generation)
        next_product_generation += len(outputs)
        return Operation.registered(
            request_key=keys[sid],
            op_id=op_id,
            parent=Checkpoint(
                predecessor.op_id,
                DeviceSelected(),
            ),
            kind=OpCode.AR_DECODE,
            bounds=Bounds(max_points=1, max_tokens=1),
            outputs=outputs,
            predicate=token_output,
        )

    op_ids = {sid: 0 for sid in request_ids}
    predecessors: dict[int, Operation] = {}
    operations = []
    payloads = []
    for sid in request_ids:
        root = Checkpoint(0, FixedCheckpoint(0))
        op_ids[sid] += 1
        operation, payload = prompt_op(sid, op_ids[sid], root, (0,))
        operations.append(operation)
        payloads.append(payload)
    _execute_warmup(
        requests,
        _build_warmup_batch(
            requests,
            admissions=tuple(admissions[sid] for sid in request_ids),
            operations=tuple(operations),
            input_products=tuple(payloads),
        ),
        retain_device_outputs=OpCode.AR_DECODE in variants,
    )
    predecessors.update(zip(request_ids, operations, strict=True))
    if OpCode.AR_DECODE in variants:
        for batch_size in batch_sizes:
            selected = request_ids[:batch_size]
            operations = []
            for sid in selected:
                op_ids[sid] += 1
                operations.append(decode_op(sid, op_ids[sid], predecessors[sid]))
            _execute_warmup(
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=(),
                    operations=tuple(operations),
                ),
                retain_device_outputs=True,
            )
            requests.free_products(
                tuple(
                    output.buffer_id
                    for sid in selected
                    for output in predecessors[sid].outputs
                    if output.storage_class
                    in {StorageClass.DEVICE_TENSOR, StorageClass.REQUEST_RELAY}
                )
            )
            predecessors.update(zip(selected, operations, strict=True))

    device = torch.device(requests.worker.worker_config.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for sid in request_ids:
        requests.drop_request(sid)


def _warmup_flow(requests: _WarmupRequests) -> None:
    """Drive one denoise quantum through the real flow forward path."""

    from ..execution.batch import (
        ArRequestParams,
        Bounds,
        Checkpoint,
        DeviceDim,
        DeviceSelected,
        DrawLayout,
        DType,
        FixedCheckpoint,
        NewRequest,
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
        UmmRequestParams,
        encode_token_product_bytes,
    )

    generation = requests.worker.model.generation
    if not {
        OpCode.DIFFUSION_PREPARE,
        OpCode.DIFFUSION_STEP,
    }.issubset(requests.worker.info.supported_ops) or not isinstance(
        generation, GenerationPipeline
    ):
        return
    if requests.worker.requests.request_ids():
        return
    # One scenario per CFG execution branch; numerical shape catalogs belong
    # to the execution owners and are already resident before these requests.
    configured = tuple(
        next(
            (
                shape
                for shape in reversed(requests.worker.runner.flow_captures)
                if shape.cfg_branches == branches
            ),
            FlowCapture(1, *_warmup_image_geometry(requests), branches),
        )
        for branches in requests.worker.runner.flow_cfg_branches
    )
    # Warmup identities and generations are private to this bounded startup sequence.
    next_request_id = 1
    next_generation = 1
    for bucket in configured:
        batch_size = bucket.rows
        height = bucket.height
        width = bucket.width
        cfg_branches = bucket.cfg_branches
        if batch_size > int(requests.worker.info.request_slots):
            continue
        mixed_text_sizes = tuple(
            dict.fromkeys(
                mixed.decode_rows
                for mixed in requests.worker.runner.mixed_captures
                if mixed.flow_rows == batch_size
                and mixed.height == height
                and mixed.width == width
                and mixed.cfg_branches == cfg_branches
            )
        )[:1]
        mixed_rounds = 1
        request_ids = tuple(range(next_request_id, next_request_id + batch_size))
        next_request_id += batch_size
        keys = tuple(RequestKey(0, request_id, 1) for request_id in request_ids)
        admissions = tuple(
            NewRequest.create(
                key,
                request_pool_idx=index,
                umm=UmmRequestParams(
                    image=capture_image_parameters(
                        cfg_branches,
                        steps=2 + mixed_rounds * len(mixed_text_sizes),
                        height=height,
                        width=width,
                    )
                ),
            )
            for index, key in enumerate(keys, start=1)
        )
        text_request_count = max(mixed_text_sizes, default=0)
        text_request_ids = tuple(range(next_request_id, next_request_id + text_request_count))
        next_request_id += text_request_count
        text_keys = {request_id: RequestKey(0, request_id, 1) for request_id in text_request_ids}
        text_admissions = {
            request_id: NewRequest.create(
                text_keys[request_id],
                request_pool_idx=batch_size + index,
                ar=ArRequestParams(
                    sampling=SamplingParams(temperature=0.0, ignore_eos=True),
                    initial_position=0,
                ),
            )
            for index, request_id in enumerate(text_request_ids, start=1)
        }
        roots = tuple(
            Checkpoint(0, FixedCheckpoint(0))
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
                    kind=OpCode.TRANSFER_KV_PUBLISH,
                    bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
                    outputs=(conditioning,),
                )
            )
        _execute_warmup(
            requests,
            _build_warmup_batch(
                requests,
                admissions=admissions,
                operations=tuple(publications),
                image_geometry=(height, width),
            ),
        )
        text_predecessors: dict[int, Operation] = {}
        text_op_ids = {request_id: 1 for request_id in text_request_ids}
        if text_request_ids:
            prompt_operations: list[Operation] = []
            prompt_payloads: list[ProductPayload] = []
            for request_id in text_request_ids:
                key = text_keys[request_id]
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
                    parent=Checkpoint(
                        0,
                        FixedCheckpoint(0),
                    ),
                    kind=OpCode.AR_EXTEND,
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
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=tuple(
                        text_admissions[request_id] for request_id in text_request_ids
                    ),
                    operations=tuple(prompt_operations),
                    input_products=tuple(prompt_payloads),
                    image_geometry=(height, width),
                ),
                retain_device_outputs=True,
            )
            text_predecessors.update(zip(text_request_ids, prompt_operations, strict=True))
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
                storage_class=StorageClass.REQUEST_RELAY,
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
                    kind=OpCode.DIFFUSION_PREPARE,
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
            requests,
            _build_warmup_batch(
                requests,
                admissions=(),
                operations=tuple(transitions),
                image_geometry=(height, width),
            ),
        )
        current_latents = tuple(initial_latents)
        flow_predecessors = dict(zip(request_ids, transitions, strict=True))
        for op_id in (3, 4):
            outputs: list[ProductRef] = []
            flows: list[Operation] = []
            for request_id, key, conditioning, current in zip(
                request_ids, keys, conditionings, current_latents, strict=True
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
                        parent=Checkpoint(
                            flow_predecessors[request_id].op_id,
                            DeviceSelected(),
                        ),
                        kind=OpCode.DIFFUSION_STEP,
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
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=(),
                    operations=tuple(flows),
                    image_geometry=(height, width),
                ),
            )
            requests.free_products(tuple(product.buffer_id for product in current_latents))
            current_latents = tuple(outputs)
            flow_predecessors.update(zip(request_ids, flows, strict=True))
        flow_op_id = 5
        for text_batch_size in mixed_text_sizes:
            selected_text = text_request_ids[:text_batch_size]
            for _ in range(mixed_rounds):
                text_operations: list[Operation] = []
                for request_id in selected_text:
                    predecessor = text_predecessors[request_id]
                    token_output = next(
                        output for output in predecessor.outputs if output.kind is ProductKind.TOKEN
                    )
                    text_op_ids[request_id] += 1
                    op_id = text_op_ids[request_id]
                    token_outputs = _warmup_token_outputs(
                        text_keys[request_id], op_id, next_generation
                    )
                    next_generation += len(token_outputs)
                    text_operations.append(
                        Operation.registered(
                            request_key=text_keys[request_id],
                            op_id=op_id,
                            parent=Checkpoint(
                                predecessor.op_id,
                                DeviceSelected(),
                            ),
                            kind=OpCode.AR_DECODE,
                            bounds=Bounds(max_points=1, max_tokens=1),
                            outputs=token_outputs,
                            predicate=token_output,
                        )
                    )

                flow_outputs: list[ProductRef] = []
                flow_operations: list[Operation] = []
                for request_id, key, conditioning, current in zip(
                    request_ids,
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
                            parent=Checkpoint(
                                flow_predecessors[request_id].op_id,
                                DeviceSelected(),
                            ),
                            kind=OpCode.DIFFUSION_STEP,
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
                    requests,
                    _build_warmup_batch(
                        requests,
                        admissions=(),
                        operations=(*text_operations, *flow_operations),
                        tensorized_mixed=True,
                        image_geometry=(height, width),
                    ),
                    retain_device_outputs=True,
                )
                requests.free_products(
                    tuple(
                        output.buffer_id
                        for request_id in selected_text
                        for output in text_predecessors[request_id].outputs
                        if output.storage_class
                        in {StorageClass.DEVICE_TENSOR, StorageClass.REQUEST_RELAY}
                    )
                )
                text_predecessors.update(zip(selected_text, text_operations, strict=True))
                requests.free_products(tuple(product.buffer_id for product in current_latents))
                current_latents = tuple(flow_outputs)
                flow_predecessors.update(zip(request_ids, flow_operations, strict=True))

        for request_id in (*request_ids, *text_request_ids):
            requests.drop_request(request_id)
