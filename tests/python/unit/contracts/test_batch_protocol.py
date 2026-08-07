"""Worker protocol conformance for the Python record mirror.

The digest-parity case reads the canonical fixture emitted by the Rust
``worker-wire`` crate test ``emit_digest_parity_fixture`` (run
``cargo test -p uniserve-worker-wire`` first), reconstructs the records from
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
    Bounds,
    Close,
    CloseReason,
    Commit,
    CompletionRecord,
    CompletionReport,
    Control,
    DevicePoint,
    Disposition,
    Domain,
    DrawLayout,
    DType,
    ErrorCode,
    ExecutionCapability,
    FinishFlags,
    FixedPoint,
    KvReservation,
    LogicalLengths,
    Operation,
    OpStatus,
    PartitionCompletion,
    PointRange,
    ProductKind,
    ProductPayload,
    ProductRef,
    RegistrationAck,
    Release,
    RequestKey,
    Rng,
    SamplingOwnership,
    SamplingState,
    ShapeBound,
    StaticDim,
    StorageClass,
    TimingCounters,
    TokenMode,
    TokenSpan,
    UndAdmission,
    VersionRef,
    Work,
    WorkerForwardStats,
    WorkVariant,
    control_content_digest,
    control_from_wire,
    control_to_wire,
    decode_sampling_state_bytes,
    decode_token_product_bytes,
    encode_sampling_state_bytes,
    encode_token_product_bytes,
    protocol_layout_digest,
    route_capability_digest,
)
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parents[2] / "generated" / "digest_parity.json"


def _fixture() -> dict:
    if not _FIXTURE.exists():
        raise AssertionError(
            f"missing digest-parity fixture {_FIXTURE}; run "
            "`cargo test -p uniserve-worker-wire` to emit it"
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
        work=Work.token(TokenMode.DECODE),
        route=1,
        domain=Domain.UND,
        bounds=Bounds(max_points=1, max_tokens=1, max_kv_pages=1),
        inputs=(() if input_product is None else (input_product,)),
        outputs=(_token_output(),),
        kv_capacity_pages=1,
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
        execution=ExecutionCapability.DOMAIN_HOMOGENEOUS,
        attention=AttentionRegime.CAUSAL,
        shape_class=0,
        operations=operations,
        kv_reservations=tuple(
            KvReservation(
                request_key=operation.request_key,
                op_id=operation.op_id,
                logical_page_delta=(7,),
            )
            for operation in operations
        ),
    )


# --- Rust <-> Python digest parity -----------------------------------------


def test_plan_digest_matches_rust() -> None:
    fixture = _fixture()
    operation = Operation.from_wire(fixture["operation"])
    assert operation.plan_digest == fixture["plan_digest"]
    assert operation.compute_plan_digest() == fixture["plan_digest"]


def test_semantic_digest_matches_rust() -> None:
    fixture = _fixture()
    completion = CompletionRecord.from_wire(fixture["completion"])
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
        reservations = tuple(
            KvReservation(
                request_key=item.request_key,
                op_id=item.op_id,
                logical_page_delta=(placements[position],),
            )
            for position, item in enumerate(ordered)
        )
        batch = Batch(
            step_id=100 + index,
            partitions=(
                BatchPartition(
                    partition_id=10 + index,
                    submission_group=20 + index,
                    collective_seq=30 + index,
                    domain=Domain.UND,
                    route=operation.route,
                    execution=ExecutionCapability.DOMAIN_HOMOGENEOUS,
                    attention=AttentionRegime.CAUSAL,
                    shape_class=index,
                    operations=ordered,
                    kv_reservations=reservations,
                ),
            ),
        )
        restored = Batch.from_wire(batch.to_wire())
        target = next(
            item for item in restored.operations if item.request_key == operation.request_key
        )
        assert target.plan_digest == expected_plan

    fixture = _fixture()
    completion = CompletionRecord.from_wire(fixture["completion"])
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


def test_protocol_layout_digest_matches_rust() -> None:
    fixture = _fixture()
    assert protocol_layout_digest() == fixture["protocol_layout_digest"]


def test_route_capability_digest_matches_rust() -> None:
    fixture = _fixture()
    sample = fixture["route_capability_sample"]
    assert (
        route_capability_digest(
            [WorkVariant(name) for name in sample["supported_work"]],
            sample["max_cfg_branches"],
            sample["max_latent_size"],
            sample["max_vae_grid_tokens"],
            sample["max_vit_grid_tokens"],
            sample["max_latent_feature_bytes"],
            sample["max_vision_feature_bytes"],
            sample["max_batch_operations"],
            sample["max_speculative_points"],
            sample["max_unresolved_window"],
            sample["device_sequence_lengths"],
            sample["device_append_offsets"],
            sample["incremental_kv_publication"],
            tuple(
                (
                    capability["route"],
                    tuple(WorkVariant(variant) for variant in capability["supported_work"]),
                    capability["tensorized_mixed"],
                    SamplingOwnership(capability["sampling_ownership"]),
                    capability["preemptible"],
                    (
                        tuple(capability["credits"]["per_request"]),
                        tuple(capability["credits"]["worker"]),
                    ),
                    capability["max_unresolved_window"],
                    capability["legal_feature_bitset"],
                    capability["sampler_processors"],
                    capability["processor_order_revision"],
                    capability["rng_layouts"],
                    capability["graph_eligible"],
                    capability["gen_conditioning"],
                    capability["max_points_per_operation"],
                    tuple(capability["mixed_row_combinations"]),
                )
                for capability in sample["route_capabilities"]
            ),
            sample["kv_dtype"],
            sample["model_dtype"],
            sample["attention_backend"],
        )
        == fixture["route_capability_digest"]
    )


# --- Round-trip and validation ---------------------------------------------


def test_operation_round_trips_through_wire() -> None:
    operation = _decode_operation()
    assert Operation.from_wire(operation.to_wire()) == operation


def test_every_work_variant_round_trips() -> None:
    variants = [
        (Work("token", "extend"), True),
        (Work("token", "decode"), True),
        (Work("token", "verify"), True),
        (Work("draft", None), False),
        (Work("encode", "vision"), False),
        (Work("encode", "latent"), False),
        (Work("transfer", "product"), False),
        (Work("transfer", "kv_publish"), False),
        (Work("transfer", "kv_install"), False),
        (Work("gen", "transition"), True),
        (Work("gen", "flow"), True),
        (Work("materialize", None), False),
    ]
    for index, (work, advances) in enumerate(variants):
        assert work.advances_state is advances
        assert Work.from_wire(work.to_wire()) == work
        assert work.variant_index == index


def test_completion_record_round_trips() -> None:
    record = CompletionRecord(
        request_key=_request_key(),
        op_id=11,
        completion_slot_generation=2,
        status=OpStatus.OK,
        selected_point=1,
        logical_lengths=LogicalLengths(token_len=5, kv_visible_len=5),
        token_span=TokenSpan(base=4, len=1),
        committed_tokens=(271,),
        finish_flags=FinishFlags(stop=True),
        product_generations=(3, 5),
        semantic_digest="bb" * 32,
        error_code=None,
        timing_counters=TimingCounters(),
    )
    assert CompletionRecord.from_wire(record.to_wire()) == record


def test_error_completion_requires_error_code() -> None:
    with pytest.raises(WorkerError):
        CompletionRecord(
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

    CompletionRecord(
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


def test_every_control_variant_round_trips() -> None:
    controls: list[Control] = [
        Commit(
            request_key=_request_key(),
            control_seq=1,
            expected_parent=_fixed_parent(),
            selected=_fixed_parent(),
            public_event_limit=7,
            disposition=Disposition.PUBLISH,
        ),
        Close(
            request_key=_request_key(),
            control_seq=2,
            cutoff=_fixed_parent(),
            reason=CloseReason.COMPLETED,
        ),
        Release(request_key=_request_key(), op_id=11),
    ]
    for control in controls:
        assert control_from_wire(control_to_wire(control)) == control


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

    admission = Admission.create(_request_key(), und=UndAdmission())
    with pytest.raises(WorkerError):
        Batch(
            step_id=1,
            admissions=(admission,),
            partitions=(_partition(_decode_operation()),),
            controls=(commit(1), commit(2)),
        )


def test_batch_allows_duplicate_identical_control() -> None:
    control = Commit(
        request_key=_request_key(),
        control_seq=1,
        expected_parent=_fixed_parent(),
        selected=_fixed_parent(),
        public_event_limit=1,
        disposition=Disposition.PUBLISH,
    )
    admission = Admission.create(_request_key(), und=UndAdmission())
    batch = Batch(
        step_id=1,
        admissions=(admission,),
        partitions=(_partition(_decode_operation()),),
        controls=(control, control),
    )
    assert len(batch.controls) == 2
    assert control_content_digest(control) == control_content_digest(control)


def test_batch_rejects_two_operations_for_one_request() -> None:
    with pytest.raises(WorkerError):
        Batch(step_id=1, partitions=(_partition(_decode_operation(), _decode_operation()),))


def test_operation_rejects_a_forged_plan_digest() -> None:
    wire = _decode_operation().to_wire()
    wire["plan_digest"] = "00" * 32
    with pytest.raises(WorkerError):
        Operation.from_wire(wire)


def test_operation_accepts_shared_encoder_features_but_not_foreign_lineage_state() -> None:
    feature = replace(
        _token_output(),
        request_key=RequestKey(0, 99, 1),
        producer_op_id=3,
        kind=ProductKind.VISION_FEATURE,
    )
    operation = replace(_decode_operation(), inputs=(feature,))
    operation = replace(operation, plan_digest=operation.compute_plan_digest())
    assert Operation.from_wire(operation.to_wire()) == operation

    foreign_token = replace(feature, kind=ProductKind.TOKEN)
    invalid = replace(operation, inputs=(foreign_token,))
    invalid = replace(invalid, plan_digest=invalid.compute_plan_digest())
    with pytest.raises(WorkerError, match="request-local input"):
        Operation.from_wire(invalid.to_wire())


def test_admission_round_trips_and_binds_digest() -> None:
    admission = Admission.create(_request_key(), und=UndAdmission())
    assert Admission.from_wire(admission.to_wire()) == admission


def test_token_product_bytes_round_trip() -> None:
    for tokens in ([], [42], [1, 2, 3, 4, 5], [0xFFFFFFFF, 0, 7]):
        encoded = encode_token_product_bytes(tokens)
        assert decode_token_product_bytes(encoded) == tuple(tokens)
    # The exact fixture bytes the Rust codec produces for [7, 8, 9].
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
            force_finish=True,
        )
    )

    assert decode_sampling_state_bytes(encoded) == SamplingState(
        allowed_token_ids=(),
        suppressed_token_ids=(2, 7),
        finish_token_ids=(5, 11),
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
    admission = Admission.create(_request_key(), und=UndAdmission())
    batch = Batch(
        step_id=1,
        admissions=(admission,),
        partitions=(_partition(_decode_operation(token_input)),),
        input_products=(payload,),
    )
    restored = Batch.from_wire(batch.to_wire())
    assert restored.input_products == (payload,)
    assert decode_token_product_bytes(restored.input_products[0].payload) == (7, 8, 9)


def test_completion_report_round_trips_with_product_payloads() -> None:
    logprob = ProductRef(
        request_key=_request_key(),
        producer_op_id=11,
        output_index=2,
        generation=7,
        kind=ProductKind.LOGPROB,
        storage_class=StorageClass.HOST_STAGING,
        dtype=DType.F32,
        shape_bound=ShapeBound((StaticDim(4),)),
        point_range=PointRange(base_point=0, max_points=1),
    )
    report = CompletionReport(
        step_id=5,
        partitions=(
            PartitionCompletion(
                partition_id=1,
                completions=(
                    CompletionRecord(
                        request_key=_request_key(),
                        op_id=11,
                        completion_slot_generation=2,
                        status=OpStatus.OK,
                        selected_point=1,
                        logical_lengths=LogicalLengths(token_len=1, kv_visible_len=1),
                        token_span=TokenSpan(base=0, len=1),
                        committed_tokens=(271,),
                        finish_flags=FinishFlags(),
                        product_generations=(3,),
                        semantic_digest="bb" * 32,
                        error_code=None,
                        timing_counters=TimingCounters(),
                    ),
                ),
                products=(ProductPayload(product=logprob, payload=bytes([1, 2, 3, 4])),),
                registration=RegistrationAck(visible=True),
                worker_exec_us=10,
                forward_stats=WorkerForwardStats(
                    mode_counts={"text": 1},
                    mode_tokens={"text": 1},
                    mode_us={"text": 7},
                    component_us={"forward": 7, "text_sample": 2},
                    cuda_graph_replays=1,
                    cuda_graph_unpadded_tokens=1,
                    cuda_graph_padded_tokens=8,
                    cuda_graph_runtime_mode_counts={"graph_replay": 1},
                ),
            ),
        ),
    )
    assert CompletionReport.from_wire(report.to_wire()) == report


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
        Operation.from_wire(invalid.to_wire())
