"""Depth-one ``ScheduledRequest``/``ScheduleBatch`` builders.

These are builders for worker forward-behavior tests.

Each builder produces the records the scheduler supplies at depth one: an
:class:`NewRequest`, an :class:`ScheduledRequest` whose ``predecessor``
names accepted execution progress, and the host-staged token payload
consumed by token work.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from uniserve.sampling import SamplingParams
from uniserve_worker.execution.batch_state import BatchState

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker

from uniserve_worker.protocol.batch import (
    BatchCommand,
    BlockTable,
    BufferAllocation,
    CachePageAllocation,
    GenerationParams,
    LatentParams,
    NewRequest,
    ScheduleBatch,
    Start,
    TensorPublication,
)
from uniserve_worker.protocol.identity import (
    BufferId,
    ComputationId,
    RequestKey,
)
from uniserve_worker.protocol.operation import (
    Bounds,
    DrawLayout,
    ForwardMode,
    ImageParams,
    OpStatus,
    PipelineStage,
    Rng,
    ScheduledRequest,
    TransferMode,
)
from uniserve_worker.protocol.output import BatchOutput, RequestOutput
from uniserve_worker.protocol.tensor import (
    DeviceDim,
    DType,
    ShapeBound,
    TensorRef,
)
from uniserve_worker.protocol.transfer import KvTransfer

AUTHORITY = 0
_BLOCK_TABLES: dict[RequestKey, list[int]] = {}
_REQUEST_POOL_INDICES: dict[RequestKey, int] = {}
_PAGES_TO_ZERO: dict[tuple[RequestKey, ComputationId], tuple[int, ...]] = {}
_UNBOUND_PAGES: dict[RequestKey, list[int]] = {}
_IMAGE_PARAMS: dict[RequestKey, ImageParams] = {}
_OP_KV_LENGTHS: dict[
    tuple[RequestKey, ComputationId], tuple[int, int, int, int]
] = {}
_OP_KV_RESULTS: dict[tuple[RequestKey, ComputationId], int] = {}
_LATENT_STEPS: dict[TensorRef, int] = {}
_MAX_CFG_BRANCHES = 1
_REQUEST_POOL_SIZE = 1
_CACHE_PAGES = 1
_BLOCK_SIZE = 1
_COMMIT_MARKER_TOKENS = 1
_LATENT_PAGE_UNITS = 1
_LATENT_DOWNSAMPLE = 1
_ALTERNATIVE_SLOTS: dict[RequestKey, int] = {}
_ALTERNATIVE_PAGES: dict[RequestKey, tuple[int, ...]] = {}
_BUFFER_ALLOCATIONS: dict[BufferId, BufferAllocation] = {}


def configure_physical_pool(
    *,
    cache_pages: int,
    request_pool_size: int,
    block_size: int,
    commit_marker_tokens: int,
    max_cfg_branches: int,
    latent_page_units: int,
    latent_downsample: int,
) -> None:
    global _CACHE_PAGES, _REQUEST_POOL_SIZE, _BLOCK_SIZE, _COMMIT_MARKER_TOKENS
    global _MAX_CFG_BRANCHES, _LATENT_PAGE_UNITS, _LATENT_DOWNSAMPLE
    _CACHE_PAGES = max(1, int(cache_pages))
    _REQUEST_POOL_SIZE = max(1, int(request_pool_size))
    _BLOCK_SIZE = max(1, int(block_size))
    _COMMIT_MARKER_TOKENS = max(0, int(commit_marker_tokens))
    _MAX_CFG_BRANCHES = max(1, int(max_cfg_branches))
    _LATENT_PAGE_UNITS = max(1, int(latent_page_units))
    _LATENT_DOWNSAMPLE = max(1, int(latent_downsample))


def _reset_request(rk: RequestKey) -> None:
    _BLOCK_TABLES[rk] = []
    _REQUEST_POOL_INDICES.pop(rk, None)
    _UNBOUND_PAGES[rk] = []
    _ALTERNATIVE_SLOTS.pop(rk, None)
    _ALTERNATIVE_PAGES.pop(rk, None)
    _IMAGE_PARAMS.pop(rk, None)
    for table in (_PAGES_TO_ZERO, _OP_KV_LENGTHS, _OP_KV_RESULTS):
        for identity in tuple(
            identity for identity in table if identity[0] == rk
        ):
            table.pop(identity, None)
    for product in tuple(
        product for product in _LATENT_STEPS if product.request_key == rk
    ):
        _LATENT_STEPS.pop(product, None)
    for buffer in tuple(
        buffer for buffer in _BUFFER_ALLOCATIONS if buffer.owner == rk
    ):
        _BUFFER_ALLOCATIONS.pop(buffer, None)
    _OP_KV_RESULTS[(rk, ComputationId(0, 0))] = 0


def _kv_page(value: int) -> int:
    return int(value) + 1


def _latent_params(operation: ScheduledRequest) -> LatentParams:
    image = _IMAGE_PARAMS[operation.request_key]
    latent_units = max(
        1,
        (int(image.height) // _LATENT_DOWNSAMPLE)
        * (int(image.width) // _LATENT_DOWNSAMPLE),
    )
    page_count = (latent_units + _LATENT_PAGE_UNITS - 1) // _LATENT_PAGE_UNITS
    latent_input = operation.latent_input
    start_step = (
        0 if latent_input is None else _LATENT_STEPS.get(latent_input, 0)
    )
    return LatentParams(
        request_key=operation.request_key,
        op_id=operation.op_id,
        page_table=tuple(range(1, page_count + 1)),
        latent_units=latent_units,
        height=int(image.height),
        width=int(image.width),
        start_step=start_step,
        step_count=(
            int(operation.bounds.max_tokens)
            if operation.kind is PipelineStage.DENOISING
            else 0
        ),
    )


def _parent_kv_length(rk: RequestKey, predecessor: ComputationId) -> int:
    return _OP_KV_RESULTS.get((rk, predecessor), 0)


def record_kv_result(
    rk: RequestKey, op_id: ComputationId, visible_length: int
) -> None:
    _OP_KV_RESULTS[(rk, op_id)] = int(visible_length)


def bind_request_allocation(
    rk: RequestKey,
    *,
    request_pool_idx: int,
    page_ids: Sequence[int],
) -> None:
    pages = [int(page) for page in page_ids]
    if int(request_pool_idx) < 1 or any(page < 1 for page in pages):
        raise ValueError("request params identifiers must be positive")
    if len(set(pages)) != len(pages):
        raise ValueError("request params repeats a KV page")
    _REQUEST_POOL_INDICES[rk] = int(request_pool_idx)
    _BLOCK_TABLES[rk] = pages
    _UNBOUND_PAGES[rk] = []


def _record_existing_kv(
    rk: RequestKey,
    op_id: ComputationId,
    predecessor: ComputationId,
    input_length: int,
) -> int:
    prefix = _parent_kv_length(rk, predecessor)
    resulting = prefix + int(input_length)
    block_table = _BLOCK_TABLES.get(rk, ())
    _OP_KV_LENGTHS[(rk, op_id)] = (prefix, int(input_length), prefix, resulting)
    _OP_KV_RESULTS[(rk, op_id)] = resulting
    return len(block_table)


def _alternative_slot(rk: RequestKey) -> int:
    existing = _ALTERNATIVE_SLOTS.get(rk)
    if existing is not None:
        return existing
    occupied = set(_REQUEST_POOL_INDICES.values()) | set(
        _ALTERNATIVE_SLOTS.values()
    )
    slot = next(
        (
            candidate
            for candidate in range(_REQUEST_POOL_SIZE, 0, -1)
            if candidate not in occupied
        ),
        None,
    )
    if slot is None:
        raise RuntimeError(
            "test scheduler has no request slot for a flow prefix"
        )
    _ALTERNATIVE_SLOTS[rk] = slot
    return slot


def _alternative_pages(rk: RequestKey, tokens: int) -> tuple[int, ...]:
    needed = (int(tokens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    existing = _ALTERNATIVE_PAGES.get(rk, ())
    if len(existing) >= needed:
        return existing[:needed]
    occupied = {
        page
        for pages in (*_BLOCK_TABLES.values(), *_ALTERNATIVE_PAGES.values())
        for page in pages
    }
    selected = tuple(
        candidate
        for candidate in range(_CACHE_PAGES - 1, 0, -1)
        if candidate not in occupied
    )[:needed]
    if len(selected) != needed:
        raise RuntimeError(
            "test scheduler has no ordinary KV pages for a flow prefix"
        )
    _ALTERNATIVE_PAGES[rk] = selected
    return selected


def execution_run(
    *,
    run_id: int,
    admissions: Sequence[NewRequest] = (),
    operations: Sequence[ScheduledRequest] = (),
    input_products: Sequence[TensorPublication] = (),
    kv_inputs: Sequence[KvTransfer] = (),
    commands: Sequence[BatchCommand] = (),
    block_tables: Sequence[BlockTable] = (),
    new_cache_pages: Sequence[CachePageAllocation] = (),
) -> ScheduleBatch:
    """Build scheduler columns and physical allocations.

    The columns and allocations drive observable worker behavior.
    """
    for admission in admissions:
        if admission.image is not None:
            _IMAGE_PARAMS[admission.request_key] = admission.image
        _REQUEST_POOL_INDICES[admission.request_key] = int(
            admission.request_pool_idx
        )
    for operation in operations:
        for product in (
            *operation.buffer_inputs(),
            *operation.buffer_outputs(),
        ):
            if product.buffer_id in _BUFFER_ALLOCATIONS:
                continue
            required = int(product.max_bytes)
            offset = 0
            for params in sorted(
                _BUFFER_ALLOCATIONS.values(), key=lambda value: value.offset
            ):
                offset = (offset + 255) & ~255
                if offset + required <= params.offset:
                    break
                offset = max(offset, params.offset + params.bytes)
            offset = (offset + 255) & ~255
            _BUFFER_ALLOCATIONS[product.buffer_id] = BufferAllocation(
                product.buffer_id,
                offset,
                required,
            )
    explicit_tables = {
        (int(table.request_pool_idx), int(table.group_id)): table
        for table in block_tables
    }

    def table_for(operation: ScheduledRequest) -> BlockTable | None:
        slot = _REQUEST_POOL_INDICES.get(
            operation.request_key,
            int(operation.request_key.request_id) + 1,
        )
        explicit = explicit_tables.get((slot, 0))
        if explicit is not None:
            return explicit
        lengths = _OP_KV_LENGTHS.get((operation.request_key, operation.op_id))
        if lengths is None:
            return None
        pages = tuple(_BLOCK_TABLES.get(operation.request_key, ()))
        return BlockTable(slot, 0, pages, len(pages) * _BLOCK_SIZE)

    tables: dict[tuple[int, int], BlockTable] = {}
    allocations: dict[tuple[int, int], set[int]] = {}
    forward_operation_indices: list[int] = []
    request_pool_indices: list[int] = []
    seq_lens: list[int] = []
    query_lens: list[int] = []
    write_kv: list[bool] = []
    for operation_index, operation in enumerate(operations):
        table = table_for(operation)
        if table is not None:
            identity = (table.request_pool_idx, table.group_id)
            tables[identity] = table
            pages = _PAGES_TO_ZERO.get(
                (operation.request_key, operation.op_id), ()
            )
            if pages:
                allocations.setdefault(identity, set()).update(pages)
        lengths = _OP_KV_LENGTHS.get((operation.request_key, operation.op_id))
        if lengths is not None and lengths[1] > 0:
            forward_operation_indices.append(operation_index)
            request_pool_indices.append(
                _REQUEST_POOL_INDICES[operation.request_key]
            )
            seq_lens.append(lengths[2] + lengths[1])
            query_lens.append(lengths[1])
            write_kv.append(True)
        if operation.kind is PipelineStage.DENOISING:
            image = _IMAGE_PARAMS[operation.request_key]
            main_slot = _REQUEST_POOL_INDICES[operation.request_key]
            main_len = 0 if lengths is None else lengths[2]
            text_off = abs(float(image.cfg_text_scale) - 1.0) <= 1e-6
            image_off = abs(float(image.cfg_img_scale) - 1.0) <= 1e-6
            branches = (
                1
                if text_off and image_off
                else 2
                if text_off or image_off
                else 3
            )
            branches = min(branches, _MAX_CFG_BRANCHES)
            query_len = (
                max(1, int(image.height) // _LATENT_DOWNSAMPLE)
                * max(1, int(image.width) // _LATENT_DOWNSAMPLE)
                + _COMMIT_MARKER_TOKENS
            )
            alternative: tuple[int, int] | None = None
            if branches > 1:
                alt_slot = _alternative_slot(operation.request_key)
                negative = next(
                    (
                        admission.generation.negative_token_ids
                        for admission in admissions
                        if admission.request_key == operation.request_key
                        and admission.generation is not None
                    ),
                    (),
                )
                alt_pages = _alternative_pages(
                    operation.request_key, len(negative)
                )
                alt_table = BlockTable(
                    alt_slot,
                    0,
                    alt_pages,
                    len(alt_pages) * _BLOCK_SIZE,
                )
                tables[(alt_slot, 0)] = alt_table
                if alt_pages:
                    allocations.setdefault((alt_slot, 0), set()).update(
                        alt_pages
                    )
                if negative:
                    forward_operation_indices.append(operation_index)
                    request_pool_indices.append(alt_slot)
                    seq_lens.append(len(negative))
                    query_lens.append(len(negative))
                    write_kv.append(True)
                alternative = (alt_slot, len(negative))
            for branch in range(branches):
                slot, seq_len = (
                    (main_slot, main_len)
                    if branch == 0 or alternative is None
                    else alternative
                )
                forward_operation_indices.append(operation_index)
                request_pool_indices.append(slot)
                seq_lens.append(seq_len + query_len)
                query_lens.append(query_len)
                write_kv.append(False)
    for allocation in new_cache_pages:
        identity = (allocation.request_pool_idx, allocation.group_id)
        allocations.setdefault(identity, set()).update(allocation.page_ids)
    return ScheduleBatch(
        batch_id=operations[0].op_id.batch_id if operations else int(run_id),
        run_id=int(run_id),
        collective_seq=int(run_id) * 1024 + 2,
        operations=tuple(operations),
        block_tables=tuple(tables.values()),
        new_cache_pages=tuple(
            CachePageAllocation(slot, group, tuple(sorted(pages)))
            for (slot, group), pages in allocations.items()
            if pages
        ),
        forward_operation_indices=tuple(forward_operation_indices),
        request_pool_indices=tuple(request_pool_indices),
        seq_lens=tuple(seq_lens),
        query_lens=tuple(query_lens),
        write_kv=tuple(write_kv),
        latent_params=tuple(
            _latent_params(operation)
            for operation in operations
            if operation.kind
            in {PipelineStage.LATENT_PREPARATION, PipelineStage.DENOISING}
            or operation.latent_input is not None
        ),
        buffer_allocations=tuple(
            {
                product.buffer_id: _BUFFER_ALLOCATIONS[product.buffer_id]
                for operation in operations
                for product in (
                    *operation.buffer_inputs(),
                    *operation.buffer_outputs(),
                )
            }.values()
        ),
        input_products=tuple(input_products),
        kv_inputs=tuple(kv_inputs),
        commands=tuple(Start(request) for request in admissions)
        + tuple(commands),
    )


def request_key(request_id: int, request_epoch: int = 1) -> RequestKey:
    return RequestKey(AUTHORITY, request_id, request_epoch)


def ar_params(
    request_id: int,
    *,
    block_ids: Sequence[int] = (),
    prefix_len: int = 0,
    request_epoch: int = 1,
    sampling: SamplingParams | None = None,
) -> NewRequest:
    rk = request_key(request_id, request_epoch)
    _reset_request(rk)
    _BLOCK_TABLES[rk] = [_kv_page(value) for value in block_ids]
    _UNBOUND_PAGES[rk] = list(_BLOCK_TABLES[rk])
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    _OP_KV_RESULTS[(rk, ComputationId(0, 0))] = int(prefix_len)
    return NewRequest(
        rk,
        request_pool_idx=request_id + 1,
        generation=GenerationParams(
            sampling=(
                sampling
                if sampling is not None
                else SamplingParams(temperature=0.0, ignore_eos=True)
            ),
            initial_position=int(prefix_len),
        ),
    )


def umm_params(
    request_id: int, image: ImageParams, *, request_epoch: int = 1
) -> NewRequest:
    rk = request_key(request_id, request_epoch)
    _reset_request(rk)
    _IMAGE_PARAMS[rk] = image
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    return NewRequest(
        rk,
        request_pool_idx=request_id + 1,
        image=image,
    )


def root_parent(admission: NewRequest) -> ComputationId:
    """The ordering sentinel for a request's first state operation."""
    return ComputationId(0, 0)


def finalized_report(worker: Worker, state: BatchState) -> BatchOutput:
    """Drive the public Worker interface.

    The drive continues until every response fragment is delivered.
    """
    deadline = time.monotonic() + 10.0
    fragments: list[BatchOutput] = []
    while True:
        worker.advance()
        output = worker.poll(state)
        if output is not None:
            fragments.append(output)
            if output.done:
                return BatchOutput.combine(fragments)
        if time.monotonic() >= deadline:
            raise TimeoutError("worker completion did not become query-ready")
        time.sleep(0.00005)


def record_completion(
    operation: ScheduledRequest, report: BatchOutput
) -> RequestOutput:
    """Observe accepted output and carry its visible KV extent.

    The extent is carried into the next test input.
    """
    resolved = report
    matches = tuple(
        record
        for record in resolved.completions
        if record.request_key == operation.request_key
        and record.op_id == operation.op_id
    )
    if len(matches) != 1 or matches[0].status is not OpStatus.OK:
        raise ValueError("operation has no unique successful completion")
    record = matches[0]
    record_kv_result(
        operation.request_key, operation.op_id, record.kv_visible_len
    )
    return record


def token_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    mode: ForwardMode,
    tokens: Sequence[int],
    block_table_delta: Sequence[int] = (),
    predicate: TensorRef | None = None,
    logprobs: bool = False,
    rng: Rng | None = None,
) -> ScheduledRequest:
    """Build a token computation with its actual model input IDs."""
    block_table = _BLOCK_TABLES.setdefault(rk, [])
    added = [_kv_page(value) for value in block_table_delta]
    if set(added) & set(block_table):
        raise ValueError("KV block-table delta repeats an existing page")
    block_table.extend(added)
    pending = _UNBOUND_PAGES.setdefault(rk, [])
    pending.extend(added)
    _PAGES_TO_ZERO[(rk, op_id)] = tuple(pending)
    pending.clear()
    prefix_length = _parent_kv_length(rk, predecessor)
    input_length = len(tokens)
    _OP_KV_LENGTHS[(rk, op_id)] = (
        prefix_length,
        input_length,
        prefix_length,
        prefix_length + input_length,
    )
    if mode is not ForwardMode.VERIFY:
        _OP_KV_RESULTS[(rk, op_id)] = prefix_length + input_length

    token_output = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 4 + 1,
        dtype=DType.I64,
        shape_bound=ShapeBound(),
    )
    operation = ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=mode,
        bounds=Bounds(
            max_tokens=max(1, len(tokens)),
            max_kv_pages=len(added),
            max_completion_bytes=((1 << 16) - 1 if logprobs else 0),
        ),
        input_token_ids=tuple(int(value) for value in tokens),
        token_output=token_output,
        predicate=predicate,
        rng=rng,
    )
    return operation


def encode_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    image_base64: str | None,
    encoder_handle: int,
    mode: PipelineStage = PipelineStage.VISION_ENCODING,
    source_product: TensorRef | None = None,
) -> ScheduledRequest:
    """An encoder computation with an encoded image or a resident image source.

    The scheduler stamps the encode output reference's ``generation`` with the
    content-stable encoder handle; the worker echoes it in
    ``completion.product_generations``.
    """
    if (image_base64 is None) == (source_product is None):
        raise ValueError("encode operation requires exactly one image source")
    output_ref = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=int(encoder_handle),
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(4_096),)),
    )
    operation = ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=mode,
        bounds=Bounds(max_tokens=64, max_latent_bytes=8_192),
        input_image=image_base64,
        image_input=source_product,
        encoder_output=output_ref,
    )
    return operation


def diffusion_prepare_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    conditioning: BufferId,
    seed: int = 29,
    image_index: int = 1,
) -> tuple[ScheduledRequest, TensorRef]:
    image = _IMAGE_PARAMS[rk]
    latent = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 3 + 1,
        dtype=DType.BF16,
        shape_bound=ShapeBound(
            (DeviceDim(3 * int(image.height) * int(image.width)),)
        ),
    )
    ready = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=1,
        generation=op_id.batch_id * 3 + 2,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    _record_existing_kv(rk, op_id, predecessor, 0)
    operation = ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=PipelineStage.LATENT_PREPARATION,
        bounds=Bounds(max_tokens=1, max_latent_bytes=latent.max_bytes),
        kv_input=conditioning,
        latent_output=latent,
        completion_output=ready,
        rng=Rng(
            seed=int(seed),
            semantic_index_base=int(image_index),
            draw_layout=DrawLayout.FLOW_NOISE,
        ),
    )
    _LATENT_STEPS[latent] = 0
    return operation, latent


def diffusion_step_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    conditioning: BufferId,
    latent: TensorRef,
    steps: int,
) -> tuple[ScheduledRequest, TensorRef]:
    output = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 3 + 1,
        dtype=DType.BF16,
        shape_bound=latent.shape_bound,
    )
    _record_existing_kv(rk, op_id, predecessor, 0)
    operation = ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=PipelineStage.DENOISING,
        bounds=Bounds(max_tokens=int(steps), max_latent_bytes=output.max_bytes),
        kv_input=conditioning,
        latent_input=latent,
        latent_output=output,
    )
    _LATENT_STEPS[output] = _LATENT_STEPS.get(latent, 0) + int(steps)
    return operation, output


def kv_publication_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
) -> tuple[ScheduledRequest, BufferId]:
    product = BufferId(
        owner=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 3 + 1,
    )
    _record_existing_kv(rk, op_id, predecessor, 0)
    operation = ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=TransferMode.KV_PUBLISH,
        bounds=Bounds(max_transfer_bytes=1 << 20),
        kv_output=product,
    )
    return operation, product


def diffusion_finalize_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    latent: TensorRef,
    feedback_source: bool = False,
) -> ScheduledRequest:
    image_output = None
    if feedback_source:
        image_output = TensorRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=1,
            generation=op_id.batch_id * 3 + 1,
            dtype=DType.BF16,
            shape_bound=ShapeBound((DeviceDim(3 * 16 * 16),)),
        )
    return ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=PipelineStage.IMAGE_DECODING,
        bounds=Bounds(
            max_latent_bytes=(3 * 16 * 16 * 2 if feedback_source else 0),
            max_completion_bytes=65_536,
        ),
        latent_input=latent,
        image_output=image_output,
    )


def visual_state_operation(
    rk: RequestKey,
    *,
    op_id: ComputationId,
    predecessor: ComputationId,
    feature: TensorRef,
    sample_continuation: bool,
    max_tokens: int,
) -> ScheduledRequest:
    completion = TensorRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id.batch_id * 3,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
    )
    token = None
    if sample_continuation:
        token = TensorRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=1,
            generation=op_id.batch_id * 3 + 1,
            dtype=DType.I64,
            shape_bound=ShapeBound(),
        )
    _record_existing_kv(rk, op_id, predecessor, max_tokens)
    return ScheduledRequest(
        request_key=rk,
        op_id=op_id,
        predecessor=predecessor,
        kind=ForwardMode.PREFILL,
        bounds=Bounds(max_tokens=max_tokens),
        vision_input=feature,
        completion_output=completion,
        token_output=token,
    )


__all__ = [
    "AUTHORITY",
    "bind_request_allocation",
    "record_completion",
    "encode_operation",
    "execution_run",
    "finalized_report",
    "diffusion_step_operation",
    "diffusion_prepare_operation",
    "umm_params",
    "diffusion_finalize_operation",
    "kv_publication_operation",
    "record_kv_result",
    "request_key",
    "root_parent",
    "token_operation",
    "ar_params",
    "visual_state_operation",
]
