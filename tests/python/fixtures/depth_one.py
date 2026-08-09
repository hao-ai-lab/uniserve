"""Depth-one ``Operation``/``Batch`` builders for worker forward-behavior tests.

Each builder produces the records the scheduler supplies at depth one: an
:class:`Admission`, an :class:`Operation` whose ``parent`` names committed state,
and — for token work — the input token :class:`ProductPayload` the worker decodes
into its durable token store before running the forward.
"""

from __future__ import annotations

from collections.abc import Sequence

from uniserve_worker.batch import (
    Admission,
    AttentionRegime,
    Batch,
    BatchPartition,
    Bounds,
    Commit,
    DeviceDim,
    Disposition,
    Domain,
    DrawLayout,
    DType,
    EncodeMode,
    ExecutionCapability,
    FixedPoint,
    GenAdmission,
    ImageParams,
    KvAdmission,
    KvPlacement,
    LatentPlacement,
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
    WorkVariant,
    encode_token_product_bytes,
)

AUTHORITY = 0
_BLOCK_TABLES: dict[RequestKey, list[int]] = {}
_PAGES_TO_ZERO: dict[tuple[RequestKey, int], tuple[int, ...]] = {}
_UNBOUND_PAGES: dict[RequestKey, list[int]] = {}
_IMAGE_PARAMS: dict[RequestKey, ImageParams] = {}


def _kv_page(value: int) -> int:
    return int(value) + 1


def _latent_placement(operation: Operation) -> LatentPlacement:
    image = _IMAGE_PARAMS[operation.request_key]
    latent_units = max(1, (int(image.height) // 16) * (int(image.width) // 16))
    page_count = (latent_units + 15) // 16
    return LatentPlacement(
        request_key=operation.request_key,
        op_id=operation.op_id,
        page_table=tuple(range(1, page_count + 1)),
        latent_units=latent_units,
        height=int(image.height),
        width=int(image.width),
        start_step=0,
        step_count=(
            int(operation.bounds.max_tokens)
            if operation.work.variant is WorkVariant.GEN_FLOW
            else 0
        ),
    )


def execution_batch(
    *,
    step_id: int,
    admissions: Sequence[Admission] = (),
    operations: Sequence[Operation] = (),
    input_products: Sequence[ProductPayload] = (),
    controls: Sequence[object] = (),
) -> Batch:
    """Build the explicit physical partitions used by executor behavior tests."""

    for admission in admissions:
        if admission.gen_admission is not None:
            _IMAGE_PARAMS[admission.request_key] = admission.gen_admission.image
    by_route: dict[int, list[Operation]] = {}
    for operation in operations:
        by_route.setdefault(int(operation.route), []).append(operation)
    partitions: list[BatchPartition] = []
    partition_id = 1
    for group_id, (route, routed) in enumerate(sorted(by_route.items()), start=1):
        domains = tuple(
            domain
            for domain in (Domain.UND, Domain.GEN)
            if any(op.domain is domain for op in routed)
        )
        execution = (
            ExecutionCapability.TENSORIZED_MIXED
            if len(domains) > 1
            else ExecutionCapability.DOMAIN_HOMOGENEOUS
        )
        variants = {operation.work.variant for operation in routed}
        attention = (
            AttentionRegime.HYBRID
            if any(variant.value == "gen_flow" for variant in variants)
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
            partitions.append(
                BatchPartition(
                    partition_id=partition_id,
                    submission_group=group_id,
                    collective_seq=int(step_id) * 1024 + group_id + 1,
                    domain=domain,
                    route=route,
                    execution=execution,
                    attention=attention,
                    shape_class=0,
                    operations=domain_operations,
                    request_pool_indices=tuple(
                        next(
                            (
                                admission.request_pool_idx
                                for admission in admissions
                                if admission.request_key == operation.request_key
                            ),
                            int(operation.request_key.session_id) + 1,
                        )
                        for operation in domain_operations
                    ),
                    kv_placements=tuple(
                        KvPlacement(
                            request_key=operation.request_key,
                            op_id=operation.op_id,
                            group_id=0,
                            block_table=tuple(
                                _BLOCK_TABLES.get(operation.request_key, ())[
                                    : operation.kv_capacity_pages
                                ]
                            ),
                            pages_to_zero=_PAGES_TO_ZERO.get(
                                (operation.request_key, operation.op_id), ()
                            ),
                            prefix_length=0,
                            input_length=operation.bounds.max_tokens,
                            visible_length=0,
                            resulting_length=operation.bounds.max_tokens,
                        )
                        for operation in domain_operations
                        if operation.kv_capacity_pages > 0
                    ),
                    latent_placements=tuple(
                        _latent_placement(operation)
                        for operation in domain_operations
                        if operation.work.variant
                        in {
                            WorkVariant.GEN_TRANSITION,
                            WorkVariant.GEN_FLOW,
                            WorkVariant.MATERIALIZE,
                        }
                    ),
                )
            )
            partition_id += 1
    return Batch(
        step_id=int(step_id),
        admissions=tuple(admissions),
        partitions=tuple(partitions),
        input_products=tuple(input_products),
        controls=tuple(controls),
    )


def request_key(session_id: int, epoch: int = 1) -> RequestKey:
    return RequestKey(AUTHORITY, session_id, epoch)


def und_admission(
    session_id: int,
    *,
    block_ids: Sequence[int] = (),
    prefix_len: int = 0,
    epoch: int = 1,
    sampling: SamplingParams | None = None,
) -> Admission:
    rk = request_key(session_id, epoch)
    _BLOCK_TABLES[rk] = [_kv_page(value) for value in block_ids]
    _UNBOUND_PAGES[rk] = list(_BLOCK_TABLES[rk])
    return Admission.create(
        rk,
        request_pool_idx=session_id + 1,
        und=UndAdmission(
            sampling=(
                sampling
                if sampling is not None
                else SamplingParams(temperature=0.0, ignore_eos=True)
            ),
            kv=KvAdmission(prefix_len=int(prefix_len)),
        ),
    )


def gen_admission(session_id: int, image: ImageParams, *, epoch: int = 1) -> Admission:
    rk = request_key(session_id, epoch)
    _IMAGE_PARAMS[rk] = image
    return Admission.create(
        rk,
        request_pool_idx=session_id + 1,
        gen_admission=GenAdmission(image=image),
    )


def root_parent(admission: Admission) -> VersionRef:
    """The admission-root fixed version a request's first operation parents on."""

    return VersionRef(admission.request_key, 0, FixedPoint(0, admission.digest))


def commit_resolved(session: object, *, public_event_limit: int = 0) -> Commit:
    expected_parent = session.committed_version()
    selected = session.resolved_version()
    return Commit(
        request_key=session.request_key,
        control_seq=session.applied_control_seq + 1,
        expected_parent=expected_parent,
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
    parent: VersionRef,
    mode: TokenMode,
    tokens: Sequence[int],
    block_table_delta: Sequence[int] = (),
    predicate: ProductRef | None = None,
    produces_finish_candidate: bool = True,
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

    reference = _token_input_ref(rk, op_id, len(tokens))
    token_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 4 + 1,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.DEVICE_TENSOR,
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
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(base_point=0, max_points=max_points),
    )
    accepted_span_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=2,
        generation=op_id * 6 + 3,
        kind=ProductKind.ACCEPTED_SPAN,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U32,
        shape_bound=ShapeBound((StaticDim(max_points + 1),)),
        point_range=PointRange(base_point=0, max_points=max_points),
    )
    continuation_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=3,
        generation=op_id * 6 + 4,
        kind=ProductKind.CONTINUATION,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.I64,
        shape_bound=ShapeBound((StaticDim(4),)),
        point_range=PointRange(base_point=0, max_points=max_points),
    )
    finish_output = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=4,
        generation=op_id * 6 + 5,
        kind=ProductKind.FINISH,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    outputs = [token_output]
    if mode is TokenMode.VERIFY:
        outputs.extend((selected_point_output, accepted_span_output, continuation_output))
    if produces_finish_candidate:
        outputs.append(finish_output)
    if logprobs:
        outputs.append(
            ProductRef(
                request_key=rk,
                producer_op_id=op_id,
                output_index=5,
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
        work=Work.token(mode),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(
            max_points=max_points,
            max_tokens=max(1, len(tokens)),
            max_kv_pages=len(added),
            max_completion_bytes=((1 << 16) - 1 if logprobs else 0),
        ),
        inputs=(reference,),
        outputs=tuple(outputs),
        kv_capacity_pages=len(block_table),
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
    parent: VersionRef,
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
    image_ref = source_product or ProductRef(
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
        work=Work("encode", mode.value),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_tokens=64, max_latent_bytes=8_192),
        inputs=(image_ref,),
        outputs=(output_ref,),
        control_seq=control_seq,
    )
    payload = (
        None if image_base64 is None else ProductPayload(product=image_ref, payload=image_bytes)
    )
    return operation, payload


def gen_transition_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: VersionRef,
    conditioning: ProductRef,
    seed: int = 29,
    image_index: int = 1,
    control_seq: int = 0,
) -> tuple[Operation, ProductRef]:
    latent = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=op_id * 3 + 1,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.BF16,
        shape_bound=ShapeBound((DeviceDim(4_096),)),
        point_range=PointRange(),
    )
    ready = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=1,
        generation=op_id * 3 + 2,
        kind=ProductKind.COMPLETION,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("gen", "transition"),
        route=0,
        domain=Domain.GEN,
        bounds=Bounds(max_points=1, max_tokens=1, max_latent_bytes=8_192),
        inputs=(conditioning,),
        outputs=(latent, ready),
        rng=Rng(
            seed=int(seed),
            semantic_index_base=int(image_index),
            draw_layout=DrawLayout.FLOW_NOISE,
        ),
        control_seq=control_seq,
    )
    return operation, latent


def flow_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: VersionRef,
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
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("gen", "flow"),
        route=0,
        domain=Domain.GEN,
        bounds=Bounds(max_points=1, max_tokens=int(steps), max_latent_bytes=8_192),
        inputs=(conditioning, latent),
        outputs=(output,),
        control_seq=control_seq,
    )
    return operation, output


def kv_publication_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: VersionRef,
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
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("transfer", TransferMode.KV_PUBLISH.value),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_transfer_bytes=1 << 20),
        outputs=(product,),
        control_seq=control_seq,
    )
    return operation, product


def materialize_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: VersionRef,
    latent: ProductRef,
    feedback_source: bool = False,
    control_seq: int = 0,
) -> Operation:
    outputs = (
        ProductRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=0,
            generation=op_id * 3,
            kind=ProductKind.ARTIFACT,
            storage_class=StorageClass.COMPLETION_ARENA,
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
        work=Work("materialize", None),
        route=0,
        domain=Domain.GEN,
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
    parent: VersionRef,
    feature: ProductRef,
    sample_continuation: bool,
    max_tokens: int,
    control_seq: int = 0,
) -> Operation:
    outputs = (
        ProductRef(
            request_key=rk,
            producer_op_id=op_id,
            output_index=0,
            generation=op_id * 3,
            kind=ProductKind.COMPLETION,
            storage_class=StorageClass.DEVICE_TENSOR,
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
                storage_class=StorageClass.DEVICE_TENSOR,
                dtype=DType.U32,
                shape_bound=ShapeBound(),
                point_range=PointRange(),
            ),
        )
    return Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work.token(TokenMode.EXTEND),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_tokens=max_tokens),
        inputs=(feature,),
        outputs=outputs,
        control_seq=control_seq,
    )


__all__ = [
    "AUTHORITY",
    "commit_resolved",
    "encode_operation",
    "execution_batch",
    "flow_operation",
    "gen_transition_operation",
    "gen_admission",
    "materialize_operation",
    "kv_publication_operation",
    "request_key",
    "root_parent",
    "token_operation",
    "und_admission",
    "visual_state_operation",
]
