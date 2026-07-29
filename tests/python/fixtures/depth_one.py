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
    Bounds,
    Domain,
    DType,
    EncodeMode,
    FixedPoint,
    GenAdmission,
    ImageParams,
    KvAllocation,
    Operation,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    SamplingParams,
    ShapeBound,
    StorageClass,
    TokenMode,
    UndAdmission,
    VersionRef,
    Work,
    encode_token_product_bytes,
)

AUTHORITY = 0


def request_key(session_id: int, epoch: int = 1) -> RequestKey:
    return RequestKey(AUTHORITY, session_id, epoch)


def und_admission(
    session_id: int,
    *,
    block_ids: Sequence[int] = (),
    epoch: int = 1,
    sampling: SamplingParams | None = None,
) -> Admission:
    return Admission.create(
        request_key(session_id, epoch),
        und=UndAdmission(
            sampling=(
                sampling
                if sampling is not None
                else SamplingParams(temperature=0.0, ignore_eos=True)
            ),
            kv=KvAllocation(block_ids=tuple(int(value) for value in block_ids)),
        ),
    )


def gen_admission(session_id: int, image: ImageParams, *, epoch: int = 1) -> Admission:
    return Admission.create(request_key(session_id, epoch), gen_admission=GenAdmission(image=image))


def root_parent(admission: Admission) -> VersionRef:
    """The admission-root fixed version a request's first operation parents on."""

    return VersionRef(admission.request_key, 0, FixedPoint(0, admission.digest))


def _token_input_ref(rk: RequestKey) -> ProductRef:
    return ProductRef(
        request_key=rk,
        producer_op_id=0,
        output_index=0,
        generation=0,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U32,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )


def token_operation(
    rk: RequestKey,
    *,
    op_id: int,
    parent: VersionRef,
    mode: TokenMode,
    tokens: Sequence[int],
    new_kv_blocks: Sequence[int] = (),
) -> tuple[Operation, ProductPayload]:
    """A token operation plus the input token product the worker decodes for it.

    ``new_kv_blocks`` are the KV blocks this step appends to the session's lease,
    exactly as the scheduler supplies them when a sequence crosses a page
    boundary.
    """

    reference = _token_input_ref(rk)
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work.token(mode),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_tokens=max(1, len(tokens))),
        inputs=(reference,),
        new_kv_blocks=tuple(int(value) for value in new_kv_blocks),
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
    image_base64: str,
    encoder_handle: int,
    mode: EncodeMode = EncodeMode.VISION,
) -> tuple[Operation, ProductPayload]:
    """An encode operation plus its input image product.

    The scheduler stamps the encode output reference's ``generation`` with the
    content-stable encoder handle; the worker echoes it in
    ``completion.product_generations``.
    """

    kind = ProductKind.VISION_FEATURE if mode is EncodeMode.VISION else ProductKind.LATENT_FEATURE
    image_ref = ProductRef(
        request_key=rk,
        producer_op_id=0,
        output_index=1,
        generation=0,
        kind=kind,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U8,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    output_ref = ProductRef(
        request_key=rk,
        producer_op_id=op_id,
        output_index=0,
        generation=int(encoder_handle),
        kind=kind,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.BF16,
        shape_bound=ShapeBound(),
        point_range=PointRange(),
    )
    operation = Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("encode", mode.value),
        route=0,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_tokens=64),
        inputs=(image_ref,),
        outputs=(output_ref,),
    )
    return operation, ProductPayload(product=image_ref, payload=image_base64.encode("utf-8"))


def flow_operation(rk: RequestKey, *, op_id: int, parent: VersionRef, steps: int) -> Operation:
    return Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("gen", "flow"),
        route=0,
        domain=Domain.GEN,
        bounds=Bounds(max_points=int(steps)),
    )


def materialize_operation(rk: RequestKey, *, op_id: int, parent: VersionRef) -> Operation:
    return Operation.registered(
        request_key=rk,
        op_id=op_id,
        parent=parent,
        work=Work("materialize", None),
        route=0,
        domain=Domain.GEN,
        bounds=Bounds(),
    )


__all__ = [
    "AUTHORITY",
    "encode_operation",
    "flow_operation",
    "gen_admission",
    "materialize_operation",
    "request_key",
    "root_parent",
    "token_operation",
    "und_admission",
]
