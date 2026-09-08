"""Depth-one ``Operation``/``Run`` builders for worker forward-behavior tests.

Each builder produces the records the scheduler supplies at depth one: an
:class:`NewRequest`, an :class:`Operation` whose ``parent`` names committed state,
and the host-staged token payload consumed by token work.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from uniserve_worker.execution.batch import (
    ArRequestParams,
    AttentionRegime,
    BatchCommand,
    BlockTable,
    Bounds,
    BufferAllocation,
    BufferId,
    CachePageAllocation,
    Checkpoint,
    Commit,
    DeviceDim,
    Disposition,
    Domain,
    DrawLayout,
    DType,
    EncodeMode,
    FixedCheckpoint,
    ImageParams,
    LatentParams,
    NewRequest,
    OpCode,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    Rng,
    RowGeometry,
    Run,
    RunLane,
    RunResult,
    SamplingParams,
    ShapeBound,
    Start,
    StaticDim,
    StorageClass,
    TokenMode,
    UmmRequestParams,
    encode_token_product_bytes,
)
from uniserve_worker.execution.output import (
    finalize_run_result,
    run_result_ready,
)

AUTHORITY = 0
_BLOCK_TABLES: dict[RequestKey, list[int]] = {}
_REQUEST_POOL_INDICES: dict[RequestKey, int] = {}
_PAGES_TO_ZERO: dict[tuple[RequestKey, int], tuple[int, ...]] = {}
_UNBOUND_PAGES: dict[RequestKey, list[int]] = {}
_IMAGE_PARAMS: dict[RequestKey, ImageParams] = {}
_OP_KV_LENGTHS: dict[tuple[RequestKey, int], tuple[int, int, int, int]] = {}
_OP_KV_RESULTS: dict[tuple[RequestKey, int], int] = {}
_OP_KV_VERIFY_BASES: dict[tuple[RequestKey, int], int] = {}
_LATENT_STEPS: dict[ProductRef, int] = {}
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
    for table in (_PAGES_TO_ZERO, _OP_KV_LENGTHS, _OP_KV_RESULTS, _OP_KV_VERIFY_BASES):
        for identity in tuple(identity for identity in table if identity[0] == rk):
            table.pop(identity, None)
    for product in tuple(product for product in _LATENT_STEPS if product.request_key == rk):
        _LATENT_STEPS.pop(product, None)
    for buffer in tuple(buffer for buffer in _BUFFER_ALLOCATIONS if buffer.owner == rk):
        _BUFFER_ALLOCATIONS.pop(buffer, None)
    _OP_KV_RESULTS[(rk, 0)] = 0


def _kv_page(value: int) -> int:
    return int(value) + 1


def _latent_params(operation: Operation) -> LatentParams:
    image = _IMAGE_PARAMS[operation.request_key]
    latent_units = max(
        1,
        (int(image.height) // _LATENT_DOWNSAMPLE) * (int(image.width) // _LATENT_DOWNSAMPLE),
    )
    page_count = (latent_units + _LATENT_PAGE_UNITS - 1) // _LATENT_PAGE_UNITS
    latent_input = next(
        (product for product in operation.inputs if product.kind is ProductKind.LATENT),
        None,
    )
    start_step = 0 if latent_input is None else _LATENT_STEPS.get(latent_input, 0)
    return LatentParams(
        request_key=operation.request_key,
        op_id=operation.op_id,
        page_table=tuple(range(1, page_count + 1)),
        latent_units=latent_units,
        height=int(image.height),
        width=int(image.width),
        start_step=start_step,
        step_count=(
            int(operation.bounds.max_tokens) if operation.kind is OpCode.DIFFUSION_STEP else 0
        ),
    )


def _parent_kv_length(rk: RequestKey, parent: Checkpoint) -> int:
    identity = (rk, int(parent.op_id))
    if identity in _OP_KV_VERIFY_BASES:
        return _OP_KV_VERIFY_BASES[identity] + int(parent.point.point_index)
    return _OP_KV_RESULTS.get(identity, 0)


def record_kv_result(rk: RequestKey, op_id: int, visible_length: int) -> None:
    _OP_KV_RESULTS[(rk, int(op_id))] = int(visible_length)


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
    op_id: int,
    parent: Checkpoint,
    input_length: int,
) -> int:
    prefix = _parent_kv_length(rk, parent)
    resulting = prefix + int(input_length)
    block_table = _BLOCK_TABLES.get(rk, ())
    _OP_KV_LENGTHS[(rk, op_id)] = (prefix, int(input_length), prefix, resulting)
    _OP_KV_RESULTS[(rk, op_id)] = resulting
    return len(block_table)


def _alternative_slot(rk: RequestKey) -> int:
    existing = _ALTERNATIVE_SLOTS.get(rk)
    if existing is not None:
        return existing
    occupied = set(_REQUEST_POOL_INDICES.values()) | set(_ALTERNATIVE_SLOTS.values())
    slot = next(
        (candidate for candidate in range(_REQUEST_POOL_SIZE, 0, -1) if candidate not in occupied),
        None,
    )
    if slot is None:
        raise RuntimeError("test scheduler has no request slot for a flow prefix")
    _ALTERNATIVE_SLOTS[rk] = slot
    return slot


def _alternative_pages(rk: RequestKey, tokens: int) -> tuple[int, ...]:
    needed = (int(tokens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    existing = _ALTERNATIVE_PAGES.get(rk, ())
    if len(existing) >= needed:
        return existing[:needed]
    occupied = {
        page for pages in (*_BLOCK_TABLES.values(), *_ALTERNATIVE_PAGES.values()) for page in pages
    }
    selected = tuple(
        candidate for candidate in range(_CACHE_PAGES - 1, 0, -1) if candidate not in occupied
    )[:needed]
    if len(selected) != needed:
        raise RuntimeError("test scheduler has no ordinary KV pages for a flow prefix")
    _ALTERNATIVE_PAGES[rk] = selected
    return selected


def execution_run(
    *,
    run_id: int,
    admissions: Sequence[NewRequest] = (),
    operations: Sequence[Operation] = (),
    input_products: Sequence[ProductPayload] = (),
    commands: Sequence[BatchCommand] = (),
    block_tables: Sequence[BlockTable] = (),
    new_cache_pages: Sequence[CachePageAllocation] = (),
) -> Run:
    """Build the explicit physical lanes used by ModelRunner behavior tests."""

    for admission in admissions:
        if admission.umm is not None:
            _IMAGE_PARAMS[admission.request_key] = admission.umm.image
        _REQUEST_POOL_INDICES[admission.request_key] = int(admission.request_pool_idx)
    for operation in operations:
        for product in (*operation.inputs, *operation.outputs):
            if not product.uses_persistent_buffer() or product.buffer_id in _BUFFER_ALLOCATIONS:
                continue
            required = int(product.max_bytes)
            offset = 0
            for params in sorted(_BUFFER_ALLOCATIONS.values(), key=lambda value: value.offset):
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
    by_route: dict[int, list[Operation]] = {0: list(operations)} if operations else {}
    lanes: list[RunLane] = []
    explicit_tables = {
        (int(table.request_pool_idx), int(table.group_id)): table for table in block_tables
    }

    def table_for(operation: Operation) -> BlockTable | None:
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

    lane_id = 1
    for group_id, (route, routed) in enumerate(sorted(by_route.items()), start=1):
        domains = tuple(domain for domain in Domain if any(op.domain is domain for op in routed))
        variants = {operation.kind for operation in routed}
        attention = (
            AttentionRegime.HYBRID
            if any(variant.value == "diffusion_step" for variant in variants)
            else AttentionRegime.CAUSAL
            if any(
                variant.value.startswith("token_") or variant.value == "draft"
                for variant in variants
            )
            else AttentionRegime.NONE
        )
        for domain in domains:
            domain_operations = tuple(
                operation for operation in routed if operation.domain is domain
            )
            tables: dict[tuple[int, int], BlockTable] = {}
            allocations: dict[tuple[int, int], set[int]] = {}
            forward_rows: list[RowGeometry] = []
            for operation_index, operation in enumerate(domain_operations):
                table = table_for(operation)
                if table is not None:
                    identity = (table.request_pool_idx, table.group_id)
                    tables[identity] = table
                    pages = _PAGES_TO_ZERO.get((operation.request_key, operation.op_id), ())
                    if pages:
                        allocations.setdefault(identity, set()).update(pages)
                lengths = _OP_KV_LENGTHS.get((operation.request_key, operation.op_id))
                if lengths is not None and lengths[1] > 0:
                    forward_rows.append(
                        RowGeometry(
                            operation_index,
                            _REQUEST_POOL_INDICES[operation.request_key],
                            lengths[2],
                            lengths[1],
                            True,
                        )
                    )
                if operation.kind is OpCode.DIFFUSION_STEP:
                    image = _IMAGE_PARAMS[operation.request_key]
                    main_slot = _REQUEST_POOL_INDICES[operation.request_key]
                    main_len = 0 if lengths is None else lengths[2]
                    text_off = abs(float(image.cfg_text_scale) - 1.0) <= 1e-6
                    image_off = abs(float(image.cfg_img_scale) - 1.0) <= 1e-6
                    branches = 1 if text_off and image_off else 2 if text_off or image_off else 3
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
                                admission.ar.negative_token_ids
                                for admission in admissions
                                if admission.request_key == operation.request_key
                                and admission.ar is not None
                            ),
                            (),
                        )
                        alt_pages = _alternative_pages(operation.request_key, len(negative))
                        alt_table = BlockTable(
                            alt_slot,
                            0,
                            alt_pages,
                            len(alt_pages) * _BLOCK_SIZE,
                        )
                        tables[(alt_slot, 0)] = alt_table
                        if alt_pages:
                            allocations.setdefault((alt_slot, 0), set()).update(alt_pages)
                        if negative:
                            forward_rows.append(
                                RowGeometry(operation_index, alt_slot, 0, len(negative), True)
                            )
                        alternative = (alt_slot, len(negative))
                    for branch in range(branches):
                        slot, seq_len = (
                            (main_slot, main_len)
                            if branch == 0 or alternative is None
                            else alternative
                        )
                        forward_rows.append(
                            RowGeometry(operation_index, slot, seq_len, query_len, False)
                        )
            for allocation in new_cache_pages:
                identity = (allocation.request_pool_idx, allocation.group_id)
                allocations.setdefault(identity, set()).update(allocation.page_ids)
            lanes.append(
                RunLane(
                    lane_id=lane_id,
                    launch_id=group_id,
                    collective_seq=int(run_id) * 1024 + group_id + 1,
                    domain=domain,
                    route=route,
                    attention=attention,
                    shape_class=0,
                    operations=domain_operations,
                    block_tables=tuple(tables.values()),
                    new_cache_pages=tuple(
                        CachePageAllocation(slot, group, tuple(sorted(pages)))
                        for (slot, group), pages in allocations.items()
                        if pages
                    ),
                    forward_rows=tuple(forward_rows),
                    latent_params=tuple(
                        _latent_params(operation)
                        for operation in domain_operations
                        if operation.kind in {OpCode.DIFFUSION_PREPARE, OpCode.DIFFUSION_STEP}
                        or any(product.kind is ProductKind.LATENT for product in operation.inputs)
                    ),
                    buffer_allocations=tuple(
                        {
                            product.buffer_id: _BUFFER_ALLOCATIONS[product.buffer_id]
                            for operation in domain_operations
                            for product in (*operation.inputs, *operation.outputs)
                            if product.uses_persistent_buffer()
                        }.values()
                    ),
                )
            )
            lane_id += 1
    return Run(
        batch_id=int(run_id),
        run_id=int(run_id),
        lanes=tuple(lanes),
        input_products=tuple(input_products),
        commands=tuple(Start(request) for request in admissions) + tuple(commands),
    )


def request_key(request_id: int, epoch: int = 1) -> RequestKey:
    return RequestKey(AUTHORITY, request_id, epoch)


def ar_params(
    request_id: int,
    *,
    block_ids: Sequence[int] = (),
    prefix_len: int = 0,
    epoch: int = 1,
    sampling: SamplingParams | None = None,
) -> NewRequest:
    rk = request_key(request_id, epoch)
    _reset_request(rk)
    _BLOCK_TABLES[rk] = [_kv_page(value) for value in block_ids]
    _UNBOUND_PAGES[rk] = list(_BLOCK_TABLES[rk])
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    _OP_KV_RESULTS[(rk, 0)] = int(prefix_len)
    return NewRequest.create(
        rk,
        request_pool_idx=request_id + 1,
        ar=ArRequestParams(
            sampling=(
                sampling
                if sampling is not None
                else SamplingParams(temperature=0.0, ignore_eos=True)
            ),
            initial_position=int(prefix_len),
        ),
    )


def umm_params(request_id: int, image: ImageParams, *, epoch: int = 1) -> NewRequest:
    rk = request_key(request_id, epoch)
    _reset_request(rk)
    _IMAGE_PARAMS[rk] = image
    _REQUEST_POOL_INDICES[rk] = request_id + 1
    return NewRequest.create(
        rk,
        request_pool_idx=request_id + 1,
        umm=UmmRequestParams(image=image),
    )


def root_parent(admission: NewRequest) -> Checkpoint:
    """The admission-root fixed version a request's first operation parents on."""

    return Checkpoint(0, FixedCheckpoint(0))


def finalized_report(report: RunResult) -> RunResult:
    deadline = time.monotonic() + 10.0
    while not run_result_ready(report):
        if time.monotonic() >= deadline:
            raise TimeoutError("worker completion did not become query-ready")
        time.sleep(0.00005)
    return finalize_run_result(report)


def commit_for_completion(
    operation: Operation,
    report: RunResult,
    *,
    expected_parent: Checkpoint | None = None,
    control_seq: int | None = None,
    public_event_limit: int = 0,
) -> Commit:
    if not operation.advances_state:
        raise ValueError("only a state-advancing completion can be committed")
    resolved = finalized_report(report)
    matches = tuple(
        record
        for record in resolved.completions
        if record.request_key == operation.request_key and int(record.op_id) == int(operation.op_id)
    )
    if len(matches) != 1 or matches[0].status is not OpStatus.OK:
        raise ValueError("operation has no unique successful completion")
    record = matches[0]
    selected = Checkpoint(
        int(operation.op_id),
        FixedCheckpoint(int(record.selected_point)),
    )
    _OP_KV_RESULTS[(operation.request_key, int(operation.op_id))] = int(
        record.logical_lengths.kv_visible_len
    )
    return Commit(
        request_key=operation.request_key,
        control_seq=(int(operation.control_seq) + 1 if control_seq is None else int(control_seq)),
        expected_parent=operation.parent if expected_parent is None else expected_parent,
        selected=selected,
        public_event_limit=public_event_limit,
        disposition=Disposition.PUBLISH,
    )


def _token_input_ref(rk: RequestKey, op_id: int, token_count: int) -> ProductRef:
    return ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=(1 << 16) - 1,
        generation=op_id * 3,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U32,
        shape_bound=ShapeBound((StaticDim(max(1, int(token_count))),)),
        point_range=PointRange(),
    )


def token_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    mode: TokenMode,
    tokens: Sequence[int],
    block_table_delta: Sequence[int] = (),
    predicate: ProductRef | None = None,
    logprobs: bool = False,
    rng: Rng | None = None,
    control_seq: int = 0,
) -> tuple[Operation, ProductPayload]:
    """A token operation plus the input token product the worker decodes for it."""

    block_table = _BLOCK_TABLES.setdefault(rk, [])
    added = [_kv_page(value) for value in block_table_delta]
    if set(added) & set(block_table):
        raise ValueError("KV block-table delta repeats an existing page")
    block_table.extend(added)
    pending = _UNBOUND_PAGES.setdefault(rk, [])
    pending.extend(added)
    _PAGES_TO_ZERO[(rk, op_id)] = tuple(pending)
    pending.clear()
    prefix_length = _parent_kv_length(rk, parent)
    input_length = len(tokens)
    _OP_KV_LENGTHS[(rk, op_id)] = (
        prefix_length,
        input_length,
        prefix_length,
        prefix_length + input_length,
    )
    if mode is TokenMode.VERIFY:
        _OP_KV_VERIFY_BASES[(rk, op_id)] = prefix_length
    else:
        _OP_KV_RESULTS[(rk, op_id)] = prefix_length + input_length

    reference = _token_input_ref(rk, op_id, len(tokens))
    token_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 4 + 1,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.REQUEST_RELAY,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(
            base_point=0,
            max_points=(len(tokens) if mode is TokenMode.VERIFY else 1),
        ),
    )
    max_points = len(tokens) if mode is TokenMode.VERIFY else 1
    selected_point_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=1,
        generation=op_id * 6 + 2,
        kind=ProductKind.SELECTED_POINT,
        storage_class=StorageClass.REQUEST_RELAY,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(base_point=0, max_points=max_points),
    )
    outputs = [token_output]
    if mode is TokenMode.VERIFY:
        outputs.append(selected_point_output)
    if logprobs:
        outputs.append(
            ProductRef(
                request_key=rk,
                producer_op_id=op_id,
                output_index=2,
                generation=op_id * 6 + 6,
                kind=ProductKind.LOGPROB,
                storage_class=StorageClass.HOST_STAGING,
                dtype=DType.U8,
                shape_bound=ShapeBound((StaticDim((1 << 16) - 1),)),
                point_range=PointRange(),
            )
        )
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.token(mode),
        bounds=Bounds(
            max_points=max_points,
            max_tokens=max(1, len(tokens)),
            max_kv_pages=len(added),
            max_completion_bytes=((1 << 16) - 1 if logprobs else 0),
        ),
        inputs=(reference,),
        outputs=tuple(outputs),
        predicate=predicate,
        rng=rng,
        control_seq=control_seq,
    )
    payload = ProductPayload(
        product=reference,
        payload=encode_token_product_bytes(tuple(int(value) for value in tokens)),
    )
    return operation, payload


def encode_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    image_base64: str | None,
    encoder_handle: int,
    mode: EncodeMode = EncodeMode.VISION,
    source_product: ProductRef | None = None,
    control_seq: int = 0,
) -> tuple[Operation, ProductPayload | None]:
    """An encode operation plus its input image product.

    The scheduler stamps the encode output reference's ``generation`` with the
    content-stable encoder handle; the worker echoes it in
    ``completion.product_generations``.
    """

    kind = ProductKind.VISION_FEATURE if mode is EncodeMode.VISION else ProductKind.LATENT_FEATURE
    if (image_base64 is None) == (source_product is None):
        raise ValueError("encode operation requires exactly one image source")
    image_bytes = None if image_base64 is None else image_base64.encode("utf-8")
    image_ref = source_product
    if image_ref is None:
        assert image_bytes is not None
        image_ref = ProductRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=0xFFFF,
            generation=op_id * 3 + 2,
            kind=ProductKind.ARTIFACT,
            storage_class=StorageClass.HOST_STAGING,
            dtype=DType.U8,
            shape_bound=ShapeBound((StaticDim(len(image_bytes)),)),
            point_range=PointRange(),
        )
    output_ref = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=int(encoder_handle),
        kind=kind,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(4_096),)),
        point_range=PointRange(),
    )
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.encoder(mode),
        bounds=Bounds(max_points=1, max_tokens=64, max_latent_bytes=8_192),
        inputs=(image_ref,),
        outputs=(output_ref,),
        control_seq=control_seq,
    )
    payload = (
        None if image_bytes is None else ProductPayload(product=image_ref, payload=image_bytes)
    )
    return operation, payload


def diffusion_prepare_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    conditioning: ProductRef,
    seed: int = 29,
    image_index: int = 1,
    control_seq: int = 0,
) -> tuple[Operation, ProductRef]:
    image = _IMAGE_PARAMS[rk]
    latent = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 3 + 1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(3 * int(image.height) * int(image.width)),)),
        point_range=PointRange(),
    )
    ready = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=1,
        generation=op_id * 3 + 2,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.REQUEST_RELAY,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    _record_existing_kv(rk, op_id, parent, 0)
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.DIFFUSION_PREPARE,
        bounds=Bounds(max_points=1, max_tokens=1, max_latent_bytes=latent.max_bytes),
        inputs=(conditioning,),
        outputs=(latent, ready),
        rng=Rng(
            seed=int(seed),
            semantic_index_base=int(image_index),
            draw_layout=DrawLayout.FLOW_NOISE,
        ),
        control_seq=control_seq,
    )
    _LATENT_STEPS[latent] = 0
    return operation, latent


def diffusion_step_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    conditioning: ProductRef,
    latent: ProductRef,
    steps: int,
    control_seq: int = 0,
) -> tuple[Operation, ProductRef]:
    output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 3 + 1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=latent.shape_bound,
        point_range=PointRange(),
    )
    _record_existing_kv(rk, op_id, parent, 0)
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.DIFFUSION_STEP,
        bounds=Bounds(max_points=1, max_tokens=int(steps), max_latent_bytes=output.max_bytes),
        inputs=(conditioning, latent),
        outputs=(output,),
        control_seq=control_seq,
    )
    _LATENT_STEPS[output] = _LATENT_STEPS.get(latent, 0) + int(steps)
    return operation, output


def kv_publication_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    control_seq: int = 0,
) -> tuple[Operation, ProductRef]:
    product = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 3 + 1,
        kind=ProductKind.KV,
        storage_class=StorageClass.PAGED_KV,
        dtype=DType.U8,
        shape_bound=ShapeBound((DeviceDim(1 << 20),)),
        point_range=PointRange(),
    )
    _record_existing_kv(rk, op_id, parent, 0)
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.TRANSFER_KV_PUBLISH,
        bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
        outputs=(product,),
        control_seq=control_seq,
    )
    return operation, product


def diffusion_finalize_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    latent: ProductRef,
    feedback_source: bool = False,
    control_seq: int = 0,
) -> Operation:
    outputs: tuple[ProductRef, ...] = (
        ProductRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=0,
            generation=op_id * 3,
            kind=ProductKind.ARTIFACT,
            storage_class=StorageClass.PINNED_OUTPUT,
            dtype=DType.U8,
            shape_bound=ShapeBound((DeviceDim(65_536),)),
            point_range=PointRange(),
        ),
    )
    if feedback_source:
        outputs += (
            ProductRef(
                request_key=rk,
                producer_op_id=op_id,
                output_index=1,
                generation=op_id * 3 + 1,
                kind=ProductKind.ARTIFACT,
                storage_class=StorageClass.LATENT_ARENA,
                dtype=DType.BF16,
                shape_bound=ShapeBound((DeviceDim(3 * 16 * 16),)),
                point_range=PointRange(),
            ),
        )
    return Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.DIFFUSION_FINALIZE,
        bounds=Bounds(
            max_latent_bytes=(3 * 16 * 16 * 2 if feedback_source else 0),
            max_completion_bytes=65_536,
        ),
        inputs=(latent,),
        outputs=outputs,
        control_seq=control_seq,
    )


def visual_state_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: Checkpoint,
    feature: ProductRef,
    sample_continuation: bool,
    max_tokens: int,
    control_seq: int = 0,
) -> Operation:
    outputs: tuple[ProductRef, ...] = (
        ProductRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=0,
            generation=op_id * 3,
            kind=ProductKind.COMPLETION,
            storage_class=StorageClass.REQUEST_RELAY,
            dtype=DType.U8,
            shape_bound=ShapeBound(),
            point_range=PointRange(),
        ),
    )
    if sample_continuation:
        outputs += (
            ProductRef(
                request_key=rk,
                producer_op_id=op_id,
                output_index=1,
                generation=op_id * 3 + 1,
                kind=ProductKind.TOKEN,
                storage_class=StorageClass.REQUEST_RELAY,
                dtype=DType.U32,
                shape_bound=ShapeBound(),
                point_range=PointRange(),
            ),
        )
    _record_existing_kv(rk, op_id, parent, max_tokens)
    return Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        kind=OpCode.token(TokenMode.EXTEND),
        bounds=Bounds(max_points=1, max_tokens=max_tokens),
        inputs=(feature,),
        outputs=outputs,
        control_seq=control_seq,
    )


__all__ = [
    "AUTHORITY",
    "bind_request_allocation",
    "commit_for_completion",
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
