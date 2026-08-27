"""Worker protocol conformance for the Python record mirror.

The digest-parity case reads the canonical fixture emitted by the Rust
``uniserve-worker-ipc`` crate test ``emit_digest_parity_fixture`` (run
``cargo test -p uniserve-worker-ipc`` first), reconstructs the records from
their wire form, recomputes both host digests, and asserts they are
byte-identical to the Rust-computed digests.
"""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import pytest

from uniserve_worker.batch import (
    Admission,
    AttentionRegime,
    Batch,
    BatchPartition,
    BlockTable,
    Bounds,
    CachePageAllocation,
    Commit,
    DevicePoint,
    Disposition,
    Domain,
    DrawLayout,
    DType,
    ErrorCode,
    FinishFlags,
    FixedPoint,
    ForwardMode,
    LatentPlacement,
    LogicalLengths,
    ModelOutput,
    Operation,
    OpStatus,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RequestKey,
    Rng,
    RowGeometry,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TimingCounters,
    TokenMode,
    TokenSpan,
    UndAdmission,
    VersionRef,
    control_from_wire,
    control_to_wire,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
    encode_sampling_state_bytes,
    encode_token_product_bytes,
    execution_domain,
    mark_typed_wire,
)
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parents[2] / "generated" / "digest_parity.json"


def _fixture() -> dict:
    if not _FIXTURE.exists():
        raise AssertionError(
            f"missing digest-parity fixture {_FIXTURE}; run "
            "`cargo test -p uniserve-worker-ipc` to emit it"
        )
    return json.loads(_FIXTURE.read_text())


def _request_key() -> RequestKey:
    return RequestKey(authority_id=4, session_id=7, epoch=2)


def _fixed_parent() -> VersionRef:
    return VersionRef(
        request_key=_request_key(),
        producer_op_id=1,
        point=FixedPoint(point_index=0, semantic_digest="aa" * 32),
    )


def _token_output() -> ProductRef:
    return ProductRef(
        request_key=_request_key(),
        producer_op_id=11,
        output_index=0,
        generation=3,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.DEVICE_TENSOR,
        dtype=DType.I32,
        shape_bound=ShapeBound((StaticDim(1),)),
        point_range=PointRange(base_point=0, max_points=1),
    )


def _decode_operation(input_product: ProductRef | None = None) -> Operation:
    return Operation.registered(
        request_key=_request_key(),
        op_id=11,
        parent=_fixed_parent(),
        work=ForwardMode.token(TokenMode.DECODE),
        route=1,
        domain=Domain.DECODE,
        bounds=Bounds(max_points=1, max_tokens=1, max_kv_pages=1),
        inputs=(() if input_product is None else (input_product,)),
        outputs=(_token_output(),),
        rng=Rng(seed=99, semantic_index_base=4, draw_layout=DrawLayout.TARGET_SAMPLING),
    )


def _partition(*operations: Operation) -> BatchPartition:
    first = operations[0]
    return BatchPartition(
        partition_id=1,
        submission_group=1,
        collective_seq=1,
        domain=first.domain,
        route=first.route,
        attention=AttentionRegime.CAUSAL,
        shape_class=0,
        operations=operations,
        block_tables=(BlockTable(8, 0, (7,), 1),),
        new_cache_pages=(CachePageAllocation(8, 0, (7,)),),
        forward_rows=tuple(
            RowGeometry(index, 8, 0, 1) for index, _operation in enumerate(operations)
        ),
    )


def _latent_product(
    request_key: RequestKey,
    *,
    producer_op_id: int,
    generation: int,
) -> ProductRef:
    return ProductRef(
        request_key=request_key,
        producer_op_id=producer_op_id,
        output_index=0,
        generation=generation,
        kind=ProductKind.LATENT,
        storage_class=StorageClass.LATENT_ARENA,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(3), StaticDim(2))),
        point_range=PointRange(base_point=0, max_points=1),
    )


def _trajectory_operation(request_key: RequestKey, work: ForwardMode, *, op_id: int) -> Operation:
    input_product = _latent_product(
        request_key,
        producer_op_id=op_id - 1,
        generation=1,
    )
    outputs = (
        (_latent_product(request_key, producer_op_id=op_id, generation=2),)
        if work in {ForwardMode.GEN_TRANSITION, ForwardMode.GEN_FLOW}
        else ()
    )
    inputs = () if work is ForwardMode.GEN_TRANSITION else (input_product,)
    return Operation.registered(
        request_key=request_key,
        op_id=op_id,
        parent=VersionRef(
            request_key=request_key,
            producer_op_id=1,
            point=FixedPoint(point_index=0, semantic_digest="aa" * 32),
        ),
        work=work,
        route=1,
        domain=execution_domain(work),
        bounds=Bounds(max_points=1, max_tokens=1, max_latent_bytes=24),
        inputs=inputs,
        outputs=outputs,
    )


def _trajectory_partition(
    operation: Operation,
    *,
    partition_id: int,
    submission_group: int,
    pages: tuple[int, ...],
) -> BatchPartition:
    return BatchPartition(
        partition_id=partition_id,
        submission_group=submission_group,
        collective_seq=submission_group,
        domain=operation.domain,
        route=operation.route,
        attention=AttentionRegime.NONE,
        shape_class=0,
        operations=(operation,),
        latent_placements=(
            LatentPlacement(
                request_key=operation.request_key,
                op_id=operation.op_id,
                page_table=pages,
                latent_units=3,
                height=16,
                width=48,
                start_step=(0 if operation.work is ForwardMode.GEN_TRANSITION else 1),
                step_count=(1 if operation.work is ForwardMode.GEN_FLOW else 0),
            ),
        ),
    )


# --- Rust <-> Python digest parity -----------------------------------------


def test_plan_digest_matches_rust() -> None:
    fixture = _fixture()
    operation = Operation.from_mapping(fixture["operation"])
    assert operation.plan_digest == fixture["plan_digest"]
    assert operation.compute_plan_digest() == fixture["plan_digest"]


def test_semantic_digest_matches_rust() -> None:
    fixture = _fixture()
    completion = ModelOutput.from_mapping(fixture["completion"])
    assert completion.committed_tokens, "fixture must exercise non-empty committed tokens"
    recomputed = completion.compute_semantic_digest(
        fixture["parent_semantic_digest"], fixture["plan_digest"]
    )
    assert recomputed == fixture["semantic_digest"]
    assert completion.semantic_digest == fixture["semantic_digest"]
    # Token values are lineage identity: a different committed token at the same
    # span changes the semantic digest.
    shifted = replace(completion, committed_tokens=(completion.committed_tokens[0] + 1,))
    assert (
        shifted.compute_semantic_digest(fixture["parent_semantic_digest"], fixture["plan_digest"])
        != fixture["semantic_digest"]
    )


def test_identity_is_invariant_to_batch_allocation_topology_and_completion_order() -> None:
    operation = _decode_operation()
    other_key = RequestKey(0, 12, 1)
    other_outputs = tuple(
        replace(product, request_key=other_key, producer_op_id=12) for product in operation.outputs
    )
    other = replace(
        operation,
        request_key=other_key,
        op_id=12,
        parent=VersionRef(other_key, 1, operation.parent.point),
        outputs=other_outputs,
    )
    other = replace(other, plan_digest=other.compute_plan_digest())
    expected_plan = operation.plan_digest
    assert replace(operation, control_seq=1).compute_plan_digest() != expected_plan

    for index, (ordered, placements) in enumerate(
        (
            ((operation, other), (7, 8)),
            ((other, operation), (80, 70)),
        ),
        start=1,
    ):
        slots = tuple(int(item.request_key.session_id) + index for item in ordered)
        batch = Batch(
            step_id=100 + index,
            partitions=(
                BatchPartition(
                    partition_id=10 + index,
                    submission_group=20 + index,
                    collective_seq=30 + index,
                    domain=Domain.DECODE,
                    route=operation.route,
                    attention=AttentionRegime.CAUSAL,
                    shape_class=index,
                    operations=ordered,
                    block_tables=tuple(
                        BlockTable(slot, 0, (placements[position],), 1)
                        for position, slot in enumerate(slots)
                    ),
                    new_cache_pages=tuple(
                        CachePageAllocation(slot, 0, (placements[position],))
                        for position, slot in enumerate(slots)
                    ),
                    forward_rows=tuple(
                        RowGeometry(position, slot, 0, 1) for position, slot in enumerate(slots)
                    ),
                ),
            ),
        )
        restored = Batch.from_mapping(batch.to_mapping())
        target = next(
            item for item in restored.operations if item.request_key == operation.request_key
        )
        assert target.plan_digest == expected_plan

    fixture = _fixture()
    completion = ModelOutput.from_mapping(fixture["completion"])
    other_completion = replace(completion, request_key=other_key, op_id=12)
    plans = {
        operation.request_key: operation.plan_digest,
        other.request_key: other.plan_digest,
    }
    expected = {
        (record.request_key, record.op_id): record.compute_semantic_digest(
            fixture["parent_semantic_digest"],
            plans[record.request_key],
        )
        for record in (completion, other_completion)
    }
    for ordering in permutations((completion, other_completion)):
        observed = {
            (record.request_key, record.op_id): record.compute_semantic_digest(
                fixture["parent_semantic_digest"],
                plans[record.request_key],
            )
            for record in ordering
        }
        assert observed == expected


# --- Validation ------------------------------------------------------------


def test_trajectory_operations_require_exact_nonoverlapping_latent_placements() -> None:
    first_key = _request_key()
    first = _trajectory_operation(first_key, ForwardMode.GEN_TRANSITION, op_id=20)
    with pytest.raises(WorkerError, match="has no latent placement"):
        BatchPartition(
            partition_id=1,
            submission_group=1,
            collective_seq=1,
            domain=Domain.FLOW,
            route=1,
            attention=AttentionRegime.NONE,
            shape_class=0,
            operations=(first,),
        )

    valid = _trajectory_partition(
        first,
        partition_id=1,
        submission_group=1,
        pages=(3, 4),
    )

    second_key = RequestKey(authority_id=4, session_id=9, epoch=2)
    second = _trajectory_operation(second_key, ForwardMode.GEN_TRANSITION, op_id=21)
    first_placement = valid.latent_placements[0]
    with pytest.raises(WorkerError, match="overlap"):
        BatchPartition(
            partition_id=2,
            submission_group=2,
            collective_seq=2,
            domain=Domain.FLOW,
            route=1,
            attention=AttentionRegime.NONE,
            shape_class=0,
            operations=(first, second),
            latent_placements=(
                first_placement,
                replace(
                    first_placement,
                    request_key=second_key,
                    op_id=second.op_id,
                    page_table=(4, 5),
                ),
            ),
        )


def test_submission_group_has_one_fixed_latent_staging_partition() -> None:
    flow_operation = _trajectory_operation(
        _request_key(),
        ForwardMode.MATERIALIZE,
        op_id=20,
    )
    transfer_operation = _trajectory_operation(
        RequestKey(authority_id=4, session_id=9, epoch=2),
        ForwardMode.TRANSFER_PRODUCT,
        op_id=21,
    )
    partitions = (
        _trajectory_partition(
            flow_operation,
            partition_id=1,
            submission_group=7,
            pages=(3, 4),
        ),
        _trajectory_partition(
            transfer_operation,
            partition_id=2,
            submission_group=7,
            pages=(5, 6),
        ),
    )
    with pytest.raises(WorkerError, match="multiple latent staging partitions"):
        Batch(step_id=1, partitions=partitions)


def test_operation_rejects_a_work_domain_mismatch() -> None:
    operation = replace(_decode_operation(), domain=Domain.FLOW)
    operation = replace(operation, plan_digest=operation.compute_plan_digest())
    with pytest.raises(WorkerError, match="domain is inconsistent"):
        operation.validate()


def test_kv_publication_requires_a_fixed_semantic_parent() -> None:
    operation = Operation.registered(
        request_key=_request_key(),
        op_id=12,
        parent=VersionRef(
            request_key=_request_key(),
            producer_op_id=9,
            point=DevicePoint(1, None, "cc" * 32),
        ),
        work=ForwardMode.TRANSFER_KV_PUBLISH,
        route=1,
        domain=Domain.PREFILL,
        bounds=Bounds(),
    )

    with pytest.raises(WorkerError, match="fixed semantic parent"):
        operation.validate()


def test_work_variants_bind_state_advancement_and_execution_domain() -> None:
    variants = [
        (ForwardMode.TOKEN_EXTEND, True, Domain.PREFILL),
        (ForwardMode.TOKEN_DECODE, True, Domain.DECODE),
        (ForwardMode.TOKEN_VERIFY, True, Domain.DECODE),
        (ForwardMode.DRAFT, False, Domain.DECODE),
        (ForwardMode.ENCODE_VISION, False, Domain.PREFILL),
        (ForwardMode.ENCODE_LATENT, False, Domain.PREFILL),
        (ForwardMode.TRANSFER_PRODUCT, False, Domain.PREFILL),
        (ForwardMode.TRANSFER_KV_PUBLISH, False, Domain.PREFILL),
        (ForwardMode.TRANSFER_KV_INSTALL, False, Domain.PREFILL),
        (ForwardMode.GEN_TRANSITION, True, Domain.FLOW),
        (ForwardMode.GEN_FLOW, True, Domain.FLOW),
        (ForwardMode.MATERIALIZE, False, Domain.FLOW),
    ]
    for work, advances, domain in variants:
        assert work.advances_state is advances
        assert execution_domain(work) is domain


def test_error_completion_requires_error_code() -> None:
    with pytest.raises(WorkerError):
        ModelOutput(
            request_key=_request_key(),
            op_id=11,
            completion_slot_generation=1,
            status=OpStatus.ERROR,
            selected_point=0,
            logical_lengths=LogicalLengths(),
            token_span=TokenSpan(),
            committed_tokens=(),
            finish_flags=FinishFlags(),
            product_generations=(),
            semantic_digest="bb" * 32,
            error_code=None,
            timing_counters=TimingCounters(),
        ).validate()

    ModelOutput(
        request_key=_request_key(),
        op_id=11,
        completion_slot_generation=1,
        status=OpStatus.ERROR,
        selected_point=0,
        logical_lengths=LogicalLengths(),
        token_span=TokenSpan(),
        committed_tokens=(),
        finish_flags=FinishFlags(),
        product_generations=(),
        semantic_digest="bb" * 32,
        error_code=ErrorCode.COMPUTE_ERROR,
        timing_counters=TimingCounters(),
    ).validate()


def test_commit_requires_a_fixed_selected_version() -> None:
    device_point = VersionRef(
        request_key=_request_key(),
        producer_op_id=9,
        point=DevicePoint(point_index=1, selected_point=None, producer_plan_digest="cc" * 32),
    )
    wire = control_to_wire(
        Commit(
            request_key=_request_key(),
            control_seq=1,
            expected_parent=_fixed_parent(),
            selected=device_point,
            public_event_limit=1,
            disposition=Disposition.PUBLISH,
        )
    )
    with pytest.raises(WorkerError):
        control_from_wire(wire)


def test_batch_rejects_conflicting_control_identity() -> None:
    def commit(limit: int) -> Commit:
        return Commit(
            request_key=_request_key(),
            control_seq=1,
            expected_parent=_fixed_parent(),
            selected=_fixed_parent(),
            public_event_limit=limit,
            disposition=Disposition.PUBLISH,
        )

    admission = Admission.create(_request_key(), request_pool_idx=8, und=UndAdmission())
    with pytest.raises(WorkerError):
        Batch(
            step_id=1,
            admissions=(admission,),
            partitions=(_partition(_decode_operation()),),
            controls=(commit(1), commit(2)),
        )


def test_batch_rejects_two_operations_for_one_request() -> None:
    with pytest.raises(WorkerError):
        Batch(step_id=1, partitions=(_partition(_decode_operation(), _decode_operation()),))


def test_operation_rejects_a_forged_plan_digest() -> None:
    wire = _decode_operation().to_mapping()
    wire["plan_digest"] = "00" * 32
    with pytest.raises(WorkerError):
        Operation.from_mapping(wire)


def test_operation_accepts_shared_encoder_features_but_not_foreign_lineage_state() -> None:
    feature = replace(
        _token_output(),
        request_key=RequestKey(0, 99, 1),
        producer_op_id=3,
        kind=ProductKind.VISION_FEATURE,
    )
    operation = replace(_decode_operation(), inputs=(feature,))
    operation = replace(operation, plan_digest=operation.compute_plan_digest())
    operation.validate()

    foreign_token = replace(feature, kind=ProductKind.TOKEN)
    invalid = replace(operation, inputs=(foreign_token,))
    invalid = replace(invalid, plan_digest=invalid.compute_plan_digest())
    with pytest.raises(WorkerError, match="request-local input"):
        Operation.from_mapping(invalid.to_mapping())


def test_admission_payload_digest_is_invariant_to_pool_index() -> None:
    admission = Admission.create(_request_key(), request_pool_idx=8, und=UndAdmission())
    relocated = replace(admission, request_pool_idx=19)
    assert relocated.payload_digest() == admission.digest


def test_token_product_bytes_match_the_rust_codec() -> None:
    fixture = bytes([3, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0])
    assert encode_token_product_bytes([7, 8, 9]) == fixture
    assert decode_token_product_bytes(fixture) == (7, 8, 9)
    with pytest.raises(WorkerError):
        decode_token_product_bytes(bytes([0, 0]))
    with pytest.raises(WorkerError):
        decode_token_product_bytes(bytes([2, 0, 0, 0, 9, 0, 0, 0]))


def test_sampling_state_bytes_preserve_branch_local_processor_semantics() -> None:
    encoded = encode_sampling_state_bytes(
        SamplingState(
            allowed_token_ids=(),
            suppressed_token_ids=(7, 2, 7),
            finish_token_ids=(11, 5, 11),
            transition_token_ids=(29, 13, 29),
            force_finish=True,
        )
    )

    assert decode_sampling_state_bytes(encoded) == SamplingState(
        allowed_token_ids=(),
        suppressed_token_ids=(2, 7),
        finish_token_ids=(5, 11),
        transition_token_ids=(13, 29),
        force_finish=True,
    )


def test_batch_carries_host_supplied_input_products() -> None:
    token_input = ProductRef(
        request_key=_request_key(),
        producer_op_id=11,
        output_index=0,
        generation=1,
        kind=ProductKind.TOKEN,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.I32,
        shape_bound=ShapeBound((StaticDim(3),)),
        point_range=PointRange(base_point=0, max_points=1),
    )
    payload = ProductPayload(product=token_input, payload=encode_token_product_bytes([7, 8, 9]))
    admission = Admission.create(_request_key(), request_pool_idx=8, und=UndAdmission())
    batch = Batch(
        step_id=1,
        admissions=(admission,),
        partitions=(_partition(_decode_operation(token_input)),),
        input_products=(payload,),
    )
    restored = Batch.from_mapping(batch.to_mapping())
    assert restored.input_products == (payload,)
    assert decode_token_product_bytes(restored.input_products[0].payload) == (7, 8, 9)


def test_host_visible_output_fits_the_operation_completion_bound() -> None:
    operation = _decode_operation()
    logprob = ProductRef(
        request_key=operation.request_key,
        producer_op_id=operation.op_id,
        output_index=2,
        generation=7,
        kind=ProductKind.LOGPROB,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.U8,
        shape_bound=ShapeBound((StaticDim(17),)),
        point_range=PointRange(base_point=0, max_points=1),
    )
    invalid = replace(
        operation,
        outputs=(*operation.outputs, logprob),
        bounds=replace(operation.bounds, max_completion_bytes=16),
    )
    invalid = replace(invalid, plan_digest=invalid.compute_plan_digest())

    with pytest.raises(WorkerError, match="host-visible output"):
        Operation.from_mapping(invalid.to_mapping())


def test_typed_wire_batch_decodes_to_the_validated_batch() -> None:
    batch = Batch(step_id=7, partitions=(_partition(_decode_operation()),))
    wire = batch.to_mapping()
    validated = Batch.from_mapping(wire)
    typed = dict(wire)
    mark_typed_wire(typed)
    assert Batch.from_mapping(typed) == validated
