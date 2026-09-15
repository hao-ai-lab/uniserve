"""Bounded request, sampling, and product startup scenarios."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import torch

from uniserve.math import ceil_div
from uniserve.media.image import Config as ImageConfig
from ..execution.flow import image_state
from uniserve_worker.protocol.batch import (
    ComputationId,
    ForwardMode,
    PipelineStage,
    TransferMode,
)

from ..execution.diffusion_state import resolve_prefix
from ..execution.graph_inputs import DiffusionShape
from ..execution.model_runner import capture_image_parameters
from ..foundation.errors import invalid_descriptor
from ..protocol.batch import (
    BatchCommand,
    BatchOutput,
    BlockTable,
    BufferAllocation,
    BufferId,
    CachePageAllocation,
    Finish,
    Free,
    LatentParams,
    NewRequest,
    OpStatus,
    RequestKey,
    RequestOutput,
    ScheduleBatch,
    ScheduledRequest,
    Start,
    TensorPublication,
    TensorRef,
)

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
        self._execute_controls((Finish(request.request_key),))
        # Serving keeps a terminal row until its slot is reassigned. Synthetic
        # requests have no further scheduler messages and can leave the table.
        self.worker.requests.drop(request_id)
        for group_id in range(
            0 if self.worker.kv_cache is None else self.worker.kv_cache.group_count
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
            ScheduleBatch(
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

    def buffer_allocation(self, product: TensorRef) -> BufferAllocation:
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
    operations: tuple[ScheduledRequest, ...],
    block_tables: dict[tuple[RequestKey, ComputationId], tuple[BlockTable, ...]],
    new_cache_pages: dict[tuple[RequestKey, ComputationId], tuple[CachePageAllocation, ...]],
    forward_inputs: dict[
        tuple[RequestKey, ComputationId],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]],
    ],
    latent_params: dict[tuple[RequestKey, ComputationId], LatentParams],
    buffer_allocations: tuple[BufferAllocation, ...],
    input_products: tuple[TensorPublication, ...] = (),
) -> ScheduleBatch:
    """Assemble warmup operations and their physical input columns."""

    return ScheduleBatch(
        batch_id=run_id,
        run_id=run_id,
        collective_seq=max(1, int(run_id) * 16 + 1),
        operations=operations,
        block_tables=tuple(
            table
            for operation in operations
            for table in block_tables.get(
                (operation.request_key, operation.op_id),
                (),
            )
        ),
        new_cache_pages=tuple(
            allocation
            for operation in operations
            for allocation in new_cache_pages.get(
                (operation.request_key, operation.op_id),
                (),
            )
        ),
        forward_operation_indices=tuple(
            operation_index
            for operation_index, operation in enumerate(operations)
            for _ in forward_inputs.get((operation.request_key, operation.op_id), ((), (), (), ()))[
                0
            ]
        ),
        request_pool_indices=tuple(
            value
            for operation in operations
            for value in forward_inputs.get(
                (operation.request_key, operation.op_id), ((), (), (), ())
            )[0]
        ),
        seq_lens=tuple(
            value
            for operation in operations
            for value in forward_inputs.get(
                (operation.request_key, operation.op_id), ((), (), (), ())
            )[1]
        ),
        query_lens=tuple(
            value
            for operation in operations
            for value in forward_inputs.get(
                (operation.request_key, operation.op_id), ((), (), (), ())
            )[2]
        ),
        write_kv=tuple(
            value
            for operation in operations
            for value in forward_inputs.get(
                (operation.request_key, operation.op_id), ((), (), (), ())
            )[3]
        ),
        latent_params=tuple(
            latent_params[(operation.request_key, operation.op_id)]
            for operation in operations
            if operation.kind
            in {
                PipelineStage.LATENT_PREPARATION,
                PipelineStage.DENOISING,
                PipelineStage.IMAGE_DECODING,
            }
        ),
        buffer_allocations=tuple(
            allocation
            for allocation in buffer_allocations
            if any(
                product.buffer_id == allocation.buffer
                for operation in operations
                for product in (
                    *operation.tensor_inputs(),
                    *operation.tensor_outputs(),
                    *((operation.predicate,) if operation.predicate is not None else ()),
                )
            )
        ),
        input_products=input_products,
        commands=tuple(Start(request) for request in admissions),
    )


def _warmup_token_output(
    request_key: RequestKey, op_id: ComputationId, generation: int
) -> TensorRef:
    """Declare a packed int64 token relay for warmup sampling."""

    from ..protocol.batch import DType, ShapeBound

    return TensorRef(
        request_key=request_key,
        producer_op_id=op_id,
        output_index=0,
        generation=generation,
        dtype=DType.I64,
        shape_bound=ShapeBound(),
    )


def _execute_warmup(
    requests: _WarmupRequests,
    batch: ScheduleBatch,
    *,
    retain_device_outputs: bool = False,
) -> BatchOutput:
    """Execute a runtime scenario and optionally retain its published outputs."""

    worker = requests.worker
    state = worker.submit(batch, propagate_errors=True)
    fragments: list[BatchOutput] = []
    while True:
        worker.advance()
        output = worker.poll(state)
        if output is None:
            time.sleep(0.00005)
            continue
        fragments.append(output)
        if output.done:
            break
    finalized = BatchOutput.combine(fragments)
    device_buffers = tuple(
        output.buffer_id
        for operation in batch.operations
        for output in (
            *operation.outputs,
            *(
                value
                for value in (
                    operation.token_output,
                    operation.completion_output,
                    operation.transition_output,
                    operation.image_output,
                )
                if value is not None
            ),
        )
    )
    failures: list[RequestOutput] = []
    for completion in finalized.completions:
        if not isinstance(completion, RequestOutput):
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
    operations: tuple[ScheduledRequest, ...],
    input_products: tuple[TensorPublication, ...] = (),
    image_size: tuple[int, int] | None = None,
) -> ScheduleBatch:
    """Derive cache, latent, buffer, and row allocations for a warmup submission."""

    requests._run_id += 1
    admissions_by_key = {admission.request_key: admission for admission in admissions}
    occupied_blocks = {page for pages in requests._kv_pages.values() for page in pages}
    request_pool_indices: dict[RequestKey, int] = {}
    block_tables: dict[tuple[RequestKey, ComputationId], tuple[BlockTable, ...]] = {}
    new_cache_pages: dict[tuple[RequestKey, ComputationId], tuple[CachePageAllocation, ...]] = {}
    forward_inputs: dict[
        tuple[RequestKey, ComputationId],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]],
    ] = {}
    latent_params: dict[tuple[RequestKey, ComputationId], LatentParams] = {}
    buffer_allocations: dict[BufferId, BufferAllocation] = {}
    # Persistent products reserve stable buffer allocations before lane construction.
    for operation in operations:
        for product in (*operation.buffer_inputs(), *operation.buffer_outputs()):
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
            ForwardMode.PREFILL,
            ForwardMode.DECODE,
            ForwardMode.VERIFY,
            TransferMode.KV_PUBLISH,
            TransferMode.KV_INSTALL,
            PipelineStage.LATENT_PREPARATION,
            PipelineStage.DENOISING,
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
            runtime = request.accepted_progress
            visible = int(runtime.kv_visible_len)
        input_length = (
            int(operation.bounds.max_tokens)
            if operation.kind
            in {
                ForwardMode.PREFILL,
                ForwardMode.DECODE,
                ForwardMode.VERIFY,
            }
            else 0
        )
        tables: list[BlockTable] = []
        allocations: list[CachePageAllocation] = []
        if requests.worker.kv_cache is None:
            raise invalid_descriptor("warmup KV operation requires cache storage")
        for group_id in range(requests.worker.kv_cache.group_count):
            lease_key = (operation.request_key, group_id)
            block_table = requests._kv_pages.setdefault(lease_key, [])
            target_pages = ceil_div(
                visible + input_length,
                int(requests.worker.kv_cache.info.block_size),
            )
            missing = target_pages - len(block_table)
            if missing < 0:
                raise invalid_descriptor("warmup operation regresses its KV capacity")
            allocated = tuple(
                candidate
                for candidate in requests.worker.kv_cache.page_ids(group_id)
                if candidate not in occupied_blocks
            )[:missing]
            if len(allocated) != missing:
                raise invalid_descriptor(
                    "warmup KV allocation exceeds resident capacity: "
                    f"request={operation.request_key.request_id}, group={group_id}, "
                    f"required_pages={missing}, available_pages={len(allocated)}, "
                    f"resident_pages={len(requests.worker.kv_cache.page_ids(group_id))}, "
                    f"leased_pages={len(occupied_blocks)}"
                )
            block_table.extend(allocated)
            occupied_blocks.update(allocated)
            tables.append(
                BlockTable(
                    request_pool_idx=request_pool_indices[operation.request_key],
                    group_id=group_id,
                    page_ids=tuple(block_table),
                    allocated_tokens=len(block_table) * requests.worker.kv_cache.info.block_size,
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
            forward_inputs[identity] = (
                (request_pool_indices[operation.request_key],),
                (visible + input_length,),
                (input_length,),
                (True,),
            )
    height, width = image_size or _warmup_image_size(requests)
    latent_units = max(
        1,
        (height // max(1, int(requests.worker._layout.latent_downsample)))
        * (width // max(1, int(requests.worker._layout.latent_downsample))),
    )
    page_units = int(requests.worker.info.latent_page_units)
    latent_page_count = (latent_units + page_units - 1) // page_units if page_units > 0 else 0
    occupied_latent_pages = {page for pages in requests._latent_pages.values() for page in pages}
    for operation in operations:
        if (
            operation.kind
            not in {
                PipelineStage.LATENT_PREPARATION,
                PipelineStage.DENOISING,
            }
            and operation.latent_input is None
        ):
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
        start_step = 0 if request is None else int(request.accepted_progress.flow_step)
        latent_params[(operation.request_key, operation.op_id)] = LatentParams(
            request_key=operation.request_key,
            op_id=operation.op_id,
            page_table=tuple(page_table),
            latent_units=latent_units,
            height=height,
            width=width,
            start_step=start_step,
            step_count=(
                int(operation.bounds.max_tokens) if operation.kind is PipelineStage.DENOISING else 0
            ),
        )
        if operation.kind is PipelineStage.DENOISING:
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
            forward_inputs[identity] = flow_rows
    return _warmup_batch(
        run_id=requests._run_id,
        admissions=admissions,
        operations=operations,
        block_tables=block_tables,
        new_cache_pages=new_cache_pages,
        forward_inputs=forward_inputs,
        latent_params=latent_params,
        buffer_allocations=tuple(buffer_allocations.values()),
        input_products=input_products,
    )


def _warmup_flow_tables(
    requests: _WarmupRequests,
    operation: ScheduledRequest,
    main_slot: int,
    height: int,
    width: int,
) -> tuple[
    tuple[BlockTable, ...],
    tuple[CachePageAllocation, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[bool, ...]],
]:
    """Build alternative-prefix KV tables and forward rows for all active CFG branches."""

    request = requests.worker.requests.get(operation.request_key.request_id)
    image = request.image
    generation = requests.worker.runner.images
    if image is None or generation is None:
        raise invalid_descriptor("generation warmup has no admitted image runtime")
    trajectory = image_state(generation, ImageConfig(height, width), image)
    branches = trajectory.guidance.branches(
        trajectory.schedule, request.accepted_progress.flow_step
    )
    runtime = request.accepted_progress
    query = generation.sequence_length(ImageConfig(height, width))
    image_prompt = image.image_prompts[0] if image.image_prompts else ""
    # Branches either reuse the conditioned request slot or share one alternative prefix.
    branch_prefixes: list[tuple[tuple[int, ...], bool]] = []
    for branch in branches:
        prefix, copy_conditioning = resolve_prefix(
            requests.worker.runner.flow_prompt,
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
    if requests.worker.kv_cache is None:
        raise invalid_descriptor("warmup flow requires KV cache storage")
    required = ceil_div(len(alternative), requests.worker.kv_cache.info.block_size)
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
        page for page in requests.worker.kv_cache.page_ids(0) if page not in occupied
    )[:missing]
    if len(allocated) != missing:
        raise invalid_descriptor("warmup alternative prefix exceeds KV capacity")
    lease.extend(allocated)
    tables: tuple[BlockTable, ...] = ()
    allocations: tuple[CachePageAllocation, ...] = ()
    alternative_slot = main_slot
    request_pool_indices: list[int] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_kv: list[bool] = []
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
                allocated_tokens=len(lease) * requests.worker.kv_cache.info.block_size,
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
        request_pool_indices.append(alternative_slot)
        seq_lens.append(len(alternative))
        query_lens.append(len(alternative))
        write_kv.append(True)
    # Append every column in the exact guidance-branch evaluation order.
    for prefix, copy_conditioning in branch_prefixes:
        request_pool_indices.append(main_slot if copy_conditioning else alternative_slot)
        seq_lens.append((int(runtime.kv_visible_len) if copy_conditioning else len(prefix)) + query)
        query_lens.append(query)
        write_kv.append(False)
    return (
        tables,
        allocations,
        (tuple(request_pool_indices), tuple(seq_lens), tuple(query_lens), tuple(write_kv)),
    )


def warmup_requests(worker: Worker) -> None:
    """Exercise synthetic requests through the configured execution paths.

    Successful scenarios retire their requests before returning. If execution
    fails, leave resource release to the enclosing Worker scope instead of
    issuing more execution commands that could replace the startup error.
    """

    requests = _WarmupRequests(worker)
    if torch.device(worker.worker_config.device).type == "cuda":
        if ForwardMode.PREFILL in worker.info.supported_ops:
            _warmup_tokens(requests)
            logger.info("completed token runtime warmup")
        if worker.runner.images is not None:
            _warmup_flow(requests)
            logger.info("completed flow runtime warmup")


def _warmup_image_size(requests: _WarmupRequests) -> tuple[int, int]:
    """Derive the largest square image whose latent grid fits the declared capacity."""

    downsample = max(1, int(requests.worker._layout.latent_downsample))
    capacity = int(requests.worker.info.latent_capacity_units)
    if int(requests.worker._layout.max_vae_grid_tokens) > 0:
        capacity = min(capacity, int(requests.worker._layout.max_vae_grid_tokens))
    side = max(1, math.isqrt(max(1, capacity)))
    return side * downsample, side * downsample


def _warmup_tokens(requests: _WarmupRequests) -> None:
    """Exercise extend-to-decode token handoff and release its synthetic request."""

    from ..protocol.batch import (
        ArRequestParams,
        Bounds,
        NewRequest,
        RequestKey,
        SamplingParams,
        ScheduledRequest,
    )

    variants = requests.worker.info.supported_ops
    if ForwardMode.PREFILL not in variants:
        return
    pool = requests.worker.kv_cache
    if pool is None:
        raise invalid_descriptor("autoregressive warmup requires KV cache storage")
    if requests.worker.requests.request_ids():
        return
    batch_sizes = (1,)
    request_ids = tuple(range(1, max(batch_sizes) + 1))
    keys = {sid: RequestKey(0, sid, 1) for sid in request_ids}
    admissions = {
        sid: NewRequest(
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
        op_id: ComputationId,
        predecessor: ComputationId,
        tokens: tuple[int, ...],
    ) -> ScheduledRequest:
        """Build one prompt computation with direct token inputs."""

        nonlocal next_product_generation
        outputs = _warmup_token_output(keys[sid], op_id, next_product_generation)
        next_product_generation += 1
        operation = ScheduledRequest(
            request_key=keys[sid],
            op_id=op_id,
            predecessor=predecessor,
            kind=ForwardMode.PREFILL,
            bounds=Bounds(max_tokens=max(1, len(tokens))),
            input_token_ids=tokens,
            token_output=outputs,
        )
        return operation

    def decode_op(
        sid: int, op_id: ComputationId, predecessor: ScheduledRequest
    ) -> ScheduledRequest:
        """Build one decode operation consuming the predecessor's token product."""

        nonlocal next_product_generation
        token_output = predecessor.token_output
        assert token_output is not None
        outputs = _warmup_token_output(keys[sid], op_id, next_product_generation)
        next_product_generation += 1
        return ScheduledRequest(
            request_key=keys[sid],
            op_id=op_id,
            predecessor=predecessor.op_id,
            kind=ForwardMode.DECODE,
            bounds=Bounds(max_tokens=1),
            token_output=outputs,
            predicate=token_output,
        )

    predecessors: dict[int, ScheduledRequest] = {}
    operations: list[ScheduledRequest] = []
    for sid in request_ids:
        root = ComputationId(0, 0)
        op_id = ComputationId(requests._run_id + 1, len(operations))
        operation = prompt_op(sid, op_id, root, (0,))
        operations.append(operation)
    _execute_warmup(
        requests,
        _build_warmup_batch(
            requests,
            admissions=tuple(admissions[sid] for sid in request_ids),
            operations=tuple(operations),
        ),
        retain_device_outputs=ForwardMode.DECODE in variants,
    )
    predecessors.update(zip(request_ids, operations, strict=True))
    if ForwardMode.DECODE in variants:
        for batch_size in batch_sizes:
            selected = request_ids[:batch_size]
            operations = []
            for sid in selected:
                op_id = ComputationId(requests._run_id + 1, len(operations))
                operations.append(decode_op(sid, op_id, predecessors[sid]))
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
                    for output in predecessors[sid].tensor_outputs()
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

    from ..protocol.batch import (
        Bounds,
        DeviceDim,
        DrawLayout,
        DType,
        NewRequest,
        RequestKey,
        Rng,
        ScheduledRequest,
        ShapeBound,
        StaticDim,
        TensorRef,
        UmmRequestParams,
    )

    generation = requests.worker.runner.images
    if (
        not {
            PipelineStage.LATENT_PREPARATION,
            PipelineStage.DENOISING,
        }.issubset(requests.worker.info.supported_ops)
        or generation is None
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
            DiffusionShape(1, *_warmup_image_size(requests), branches),
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
        request_ids = tuple(range(next_request_id, next_request_id + batch_size))
        next_request_id += batch_size
        keys = tuple(RequestKey(0, request_id, 1) for request_id in request_ids)
        admissions = tuple(
            NewRequest(
                key,
                request_pool_idx=index,
                umm=UmmRequestParams(
                    image=capture_image_parameters(
                        cfg_branches,
                        steps=2,
                        height=height,
                        width=width,
                    )
                ),
            )
            for index, key in enumerate(keys, start=1)
        )
        roots = tuple(ComputationId(0, 0) for key in keys)
        conditionings: list[BufferId] = []
        publications: list[ScheduledRequest] = []
        for key, root in zip(keys, roots, strict=True):
            op_id = ComputationId(requests._run_id + 1, len(publications))
            conditioning = BufferId(
                owner=key,
                producer_op_id=op_id,
                output_index=0,
                generation=next_generation,
            )
            next_generation += 1
            conditionings.append(conditioning)
            publications.append(
                ScheduledRequest(
                    request_key=key,
                    op_id=op_id,
                    predecessor=root,
                    kind=TransferMode.KV_PUBLISH,
                    bounds=Bounds(max_transfer_bytes=1 << 20),
                    kv_output=conditioning,
                )
            )
        _execute_warmup(
            requests,
            _build_warmup_batch(
                requests,
                admissions=admissions,
                operations=tuple(publications),
                image_size=(height, width),
            ),
        )
        max_latent_elements = max(
            1,
            math.prod(generation.denoiser.latent_shape("image", ImageConfig(height, width))),
        )
        initial_latents: list[TensorRef] = []
        transitions: list[ScheduledRequest] = []
        for key, root, conditioning in zip(keys, roots, conditionings, strict=True):
            op_id = ComputationId(requests._run_id + 1, len(transitions))
            initial_latent = TensorRef(
                request_key=key,
                producer_op_id=op_id,
                output_index=0,
                generation=next_generation,
                dtype=DType.BF16,
                shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
            )
            next_generation += 1
            ready = TensorRef(
                request_key=key,
                producer_op_id=op_id,
                output_index=1,
                generation=next_generation,
                dtype=DType.U8,
                shape_bound=ShapeBound((StaticDim(1),)),
            )
            next_generation += 1
            initial_latents.append(initial_latent)
            transitions.append(
                ScheduledRequest(
                    request_key=key,
                    op_id=op_id,
                    predecessor=root,
                    kind=PipelineStage.LATENT_PREPARATION,
                    bounds=Bounds(
                        max_tokens=1,
                        max_latent_bytes=max_latent_elements * 2,
                    ),
                    kv_input=conditioning,
                    latent_output=initial_latent,
                    completion_output=ready,
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
                image_size=(height, width),
            ),
        )
        current_latents = tuple(initial_latents)
        flow_predecessors = dict(zip(request_ids, transitions, strict=True))
        for _ in range(2):
            outputs: list[TensorRef] = []
            flows: list[ScheduledRequest] = []
            for request_id, key, conditioning, current in zip(
                request_ids, keys, conditionings, current_latents, strict=True
            ):
                op_id = ComputationId(requests._run_id + 1, len(flows))
                output = TensorRef(
                    request_key=key,
                    producer_op_id=op_id,
                    output_index=0,
                    generation=next_generation,
                    dtype=DType.BF16,
                    shape_bound=ShapeBound((DeviceDim(max_latent_elements),)),
                )
                next_generation += 1
                outputs.append(output)
                flows.append(
                    ScheduledRequest(
                        request_key=key,
                        op_id=op_id,
                        predecessor=flow_predecessors[request_id].op_id,
                        kind=PipelineStage.DENOISING,
                        bounds=Bounds(
                            max_tokens=1,
                            max_latent_bytes=max_latent_elements * 2,
                        ),
                        kv_input=conditioning,
                        latent_input=current,
                        latent_output=output,
                    )
                )
            _execute_warmup(
                requests,
                _build_warmup_batch(
                    requests,
                    admissions=(),
                    operations=tuple(flows),
                    image_size=(height, width),
                ),
            )
            requests.free_products(tuple(product.buffer_id for product in current_latents))
            current_latents = tuple(outputs)
            flow_predecessors.update(zip(request_ids, flows, strict=True))
        for request_id in request_ids:
            requests.drop_request(request_id)
