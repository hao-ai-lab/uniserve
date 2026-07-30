//! Tests derived from the Checkpoint 1 protocol specification: round-trip
//! encoding for every record and `Work`/`Control` variant, identity validation,
//! and digest determinism.

use uniserve_core::{BlockId, RequestId};

use super::*;
use crate::flat::{decode_request, decode_response, encode_request, encode_response};

fn digest_string(seed: u8) -> String {
    format!("{seed:02x}").repeat(32)
}

fn request_key() -> RequestKey {
    RequestKey::new(4, RequestId(7), 2)
}

fn fixed_parent() -> VersionRef {
    VersionRef::admission_root(request_key(), OpId(1), digest_string(0xaa))
}

fn output_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 0,
        generation: 3,
        kind: ProductKind::Token,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::I32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1), DimBound::Device { max: 8 }],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn kv_output(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 1,
        generation: 5,
        kind: ProductKind::Kv,
        storage_class: StorageClass::PagedKv,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Device { max: 64 }],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn token_decode_operation() -> Operation {
    Operation::registered(
        request_key(),
        OpId(11),
        fixed_parent(),
        Work::Token(TokenMode::Decode),
        RouteId(1),
        Domain::Und,
        Bounds {
            max_points: 1,
            max_tokens: 1,
            max_kv_pages: 1,
            ..Bounds::default()
        },
        Vec::new(),
        vec![output_product(OpId(11)), kv_output(OpId(11))],
        vec![BlockId(7)],
        None,
        Some(Rng {
            seed: 99,
            semantic_index_base: 4,
            draw_layout: DrawLayout::TargetSampling,
        }),
        0,
    )
}

fn operation_for(work: Work, op_id: OpId, advances: bool) -> Operation {
    Operation::registered(
        request_key(),
        op_id,
        fixed_parent(),
        work,
        RouteId(1),
        Domain::Und,
        Bounds {
            max_points: if advances { 1 } else { 0 },
            ..Bounds::default()
        },
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        None,
        0,
    )
}

fn completion_record() -> CompletionRecord {
    CompletionRecord {
        request_key: request_key(),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 5,
            kv_visible_len: 5,
            latent_len: 0,
        },
        token_span: TokenSpan { base: 4, len: 1 },
        committed_tokens: vec![271],
        finish_flags: FinishFlags {
            eos: false,
            length: false,
            stop: false,
        },
        product_generations: vec![3, 5],
        semantic_digest: digest_string(0xbb),
        error_code: None,
        timing_counters: TimingCounters::default(),
    }
}

fn admission() -> Admission {
    Admission::new(
        request_key(),
        Some(UndAdmission {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            kv: KvAllocation::default(),
        }),
        None,
        None,
    )
    .unwrap()
}

fn execute_round_trip(batch: Batch) -> Batch {
    let request = WorkerRequest::execute(batch);
    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    decoded.batch.unwrap()
}

#[test]
fn every_work_variant_round_trips_through_the_wire() {
    let variants = [
        (Work::Token(TokenMode::Extend), true),
        (Work::Token(TokenMode::Decode), true),
        (Work::Token(TokenMode::Verify), true),
        (Work::Draft, false),
        (Work::Encode(EncodeMode::Vision), false),
        (Work::Encode(EncodeMode::Latent), false),
        (Work::Transfer(TransferMode::Product), false),
        (Work::Transfer(TransferMode::KvPublish), false),
        (Work::Transfer(TransferMode::KvInstall), false),
        (Work::Gen(GenMode::Transition), true),
        (Work::Gen(GenMode::Flow), true),
        (Work::Materialize, false),
    ];
    for (index, (work, advances)) in variants.into_iter().enumerate() {
        assert_eq!(
            work.advances_state(),
            advances,
            "work table effect mismatch"
        );
        let operation = operation_for(work, OpId(100 + index as u64), advances);
        let batch = execute_round_trip(Batch::new(1, Vec::new(), vec![operation.clone()]));
        assert_eq!(batch.operations[0], operation);
        assert_eq!(batch.operations[0].work, work);
    }
}

#[test]
fn version_ref_device_point_round_trips() {
    let device_parent = VersionRef {
        request_key: request_key(),
        producer_op_id: OpId(9),
        point: Point::Device {
            selected_point: output_product(OpId(9)),
            producer_plan_digest: digest_string(0xcc),
        },
    };
    let operation = Operation::registered(
        request_key(),
        OpId(12),
        device_parent.clone(),
        Work::Token(TokenMode::Decode),
        RouteId(1),
        Domain::Und,
        Bounds {
            max_points: 1,
            ..Bounds::default()
        },
        Vec::new(),
        vec![output_product(OpId(12))],
        Vec::new(),
        None,
        None,
        0,
    );
    let batch = execute_round_trip(Batch::new(2, Vec::new(), vec![operation]));
    assert_eq!(batch.operations[0].parent, device_parent);
}

#[test]
fn operation_carries_new_kv_blocks_across_the_wire() {
    let base = token_decode_operation();
    assert_eq!(base.new_kv_blocks, vec![BlockId(7)]);
    let batch = execute_round_trip(Batch::new(9, Vec::new(), vec![base.clone()]));
    assert_eq!(batch.operations[0].new_kv_blocks, vec![BlockId(7)]);
    assert_eq!(batch.operations[0], base);
    // The appended KV blocks are a registration field: changing them changes the
    // plan digest.
    let mut more = token_decode_operation();
    more.new_kv_blocks = vec![BlockId(7), BlockId(8)];
    assert_ne!(more.compute_plan_digest(), base.plan_digest);
}

#[test]
fn completion_report_round_trips_records_and_product_payloads() {
    let mut logprob = output_product(OpId(11));
    logprob.output_index = 2;
    logprob.kind = ProductKind::Logprob;
    let report = CompletionReport {
        step_id: 5,
        completions: vec![completion_record()],
        products: vec![ProductPayload {
            product: logprob,
            bytes: vec![1, 2, 3, 4],
        }],
        registration: RegistrationAck { visible: true },
        worker_exec_us: Some(10),
        forward_stats: None,
    };
    let response = WorkerResponse::completion_report(report.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    assert_eq!(decoded.completion_report.unwrap(), report);
}

#[test]
fn completion_product_value_respects_its_registered_bound() {
    let mut logprob = output_product(OpId(11));
    logprob.output_index = 2;
    logprob.kind = ProductKind::Logprob;
    logprob.storage_class = StorageClass::HostStaging;
    logprob.dtype = DType::U8;
    logprob.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(3)],
    };
    let report = CompletionReport {
        step_id: 5,
        completions: vec![completion_record()],
        products: vec![ProductPayload {
            product: logprob,
            bytes: vec![1, 2, 3, 4],
        }],
        registration: RegistrationAck { visible: true },
        worker_exec_us: Some(10),
        forward_stats: None,
    };

    assert!(report.validate().is_err());
}

#[test]
fn error_completion_round_trips_with_its_error_code() {
    let mut record = completion_record();
    record.status = OpStatus::Error;
    record.error_code = Some(ErrorCode::ComputeError);
    let report = CompletionReport {
        step_id: 6,
        completions: vec![record.clone()],
        products: Vec::new(),
        registration: RegistrationAck { visible: false },
        worker_exec_us: None,
        forward_stats: None,
    };
    let decoded =
        decode_response(&encode_response(&WorkerResponse::completion_report(report)).unwrap())
            .unwrap();
    assert_eq!(decoded.completion_report.unwrap().completions[0], record);
}

#[test]
fn every_control_variant_round_trips_through_the_wire() {
    let commit = Control::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: 7,
        disposition: Disposition::Publish,
    };
    let close = Control::Close {
        request_key: request_key(),
        control_seq: 2,
        cutoff: fixed_parent(),
        reason: CloseReason::Completed,
    };
    let release = Control::Release {
        request_key: request_key(),
        op_id: OpId(11),
    };
    let batch = Batch::new(3, vec![admission()], vec![token_decode_operation()])
        .with_controls(vec![commit.clone(), close.clone(), release.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(decoded.controls, vec![commit, close, release]);
}

#[test]
fn admission_round_trips_and_binds_its_operation() {
    let batch = execute_round_trip(Batch::new(
        4,
        vec![admission()],
        vec![token_decode_operation()],
    ));
    assert_eq!(batch.admissions[0], admission());
}

#[test]
fn plan_digest_is_deterministic_and_excludes_control_seq() {
    let mut a = token_decode_operation();
    let b = token_decode_operation();
    assert_eq!(a.plan_digest, b.plan_digest);
    assert_eq!(a.plan_digest, a.compute_plan_digest());
    // control_seq is an ordering field, not registration identity.
    a.control_seq = 999;
    assert_eq!(a.compute_plan_digest(), b.plan_digest);
}

#[test]
fn semantic_digest_binds_parent_plan_and_output_delta() {
    let operation = token_decode_operation();
    let completion = completion_record();
    let parent = digest_string(0xaa);
    let base = completion.compute_semantic_digest(&parent, &operation.plan_digest);
    // Deterministic.
    assert_eq!(
        base,
        completion.compute_semantic_digest(&parent, &operation.plan_digest)
    );
    // A different selected result yields a different semantic digest.
    let mut other = completion.clone();
    other.selected_point = 2;
    assert_ne!(
        base,
        other.compute_semantic_digest(&parent, &operation.plan_digest)
    );
    // A different parent lineage yields a different semantic digest.
    assert_ne!(
        base,
        completion.compute_semantic_digest(&digest_string(0x11), &operation.plan_digest)
    );
    // A different committed token at the same span yields a different digest:
    // token values, not just positions, are lineage identity.
    let mut other_token = completion.clone();
    other_token.committed_tokens = vec![272];
    assert_ne!(
        base,
        other_token.compute_semantic_digest(&parent, &operation.plan_digest)
    );
}

#[test]
fn validation_rejects_a_forged_plan_digest() {
    let mut operation = token_decode_operation();
    operation.plan_digest = digest_string(0x00);
    assert!(operation.validate().is_err());
}

#[test]
fn validation_rejects_inconsistent_advances_state() {
    let mut operation = token_decode_operation();
    operation.advances_state = false;
    operation.plan_digest = operation.compute_plan_digest();
    assert!(operation.validate().is_err());
}

#[test]
fn validation_rejects_an_output_owned_by_another_operation() {
    let mut operation = token_decode_operation();
    operation.outputs[0].producer_op_id = OpId(999);
    operation.plan_digest = operation.compute_plan_digest();
    assert!(operation.validate().is_err());
}

#[test]
fn product_validation_enforces_generation_and_shape_bounds() {
    let mut product = output_product(OpId(11));
    product.generation = 0;
    assert!(product.validate().is_err());
    product.generation = 1;
    product.shape_bound.dims = vec![DimBound::Device { max: 2 }, DimBound::Device { max: 4 }];
    assert!(product.validate().is_err());
}

#[test]
fn batch_rejects_two_operations_for_one_request() {
    let batch = Batch {
        step_id: 1,
        admissions: Vec::new(),
        operations: vec![token_decode_operation(), token_decode_operation()],
        controls: Vec::new(),
        input_products: Vec::new(),
    };
    assert!(batch.validate().is_err());
}

#[test]
fn token_product_bytes_round_trip() {
    for tokens in [vec![], vec![42], vec![1, 2, 3, 4, 5], vec![u32::MAX, 0, 7]] {
        let bytes = encode_token_product_bytes(&tokens);
        assert_eq!(decode_token_product_bytes(&bytes).unwrap(), tokens);
    }
    // A truncated count is rejected rather than silently misread.
    assert!(decode_token_product_bytes(&[0, 0]).is_err());
    // A length that disagrees with the declared count is a fault.
    assert!(decode_token_product_bytes(&[2, 0, 0, 0, 9, 0, 0, 0]).is_err());
}

#[test]
fn sampling_state_bytes_preserve_empty_allowed_and_canonical_sets() {
    let state = SamplingState {
        recent_counts: vec![(9, 1), (3, 2), (9, 4)],
        allowed_token_ids: Some(Vec::new()),
        suppressed_token_ids: vec![7, 2, 7],
        finish_token_ids: vec![11, 5, 11],
        force_finish: true,
    };
    let decoded =
        decode_sampling_state_bytes(&encode_sampling_state_bytes(&state)).expect("decode state");
    assert_eq!(
        decoded,
        SamplingState {
            recent_counts: vec![(3, 2), (9, 5)],
            allowed_token_ids: Some(Vec::new()),
            suppressed_token_ids: vec![2, 7],
            finish_token_ids: vec![5, 11],
            force_finish: true,
        }
    );
}

#[test]
fn batch_carries_host_supplied_input_product_values() {
    let mut token_input = output_product(OpId(11));
    token_input.output_index = 0;
    token_input.kind = ProductKind::Token;
    token_input.storage_class = StorageClass::HostStaging;
    token_input.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(3)],
    };
    let payload = ProductPayload {
        product: token_input.clone(),
        bytes: encode_token_product_bytes(&[7, 8, 9]),
    };
    let mut operation = token_decode_operation();
    operation.inputs.push(token_input);
    operation.plan_digest = operation.compute_plan_digest();
    let batch = Batch::new(1, vec![admission()], vec![operation])
        .with_input_products(vec![payload.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(decoded.input_products, vec![payload.clone()]);
    assert_eq!(
        decode_token_product_bytes(&decoded.input_products[0].bytes).unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn batch_rejects_a_conflicting_control_identity() {
    let commit = |limit| Control::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: limit,
        disposition: Disposition::Publish,
    };
    let batch = Batch::new(1, vec![admission()], vec![token_decode_operation()])
        .with_controls(vec![commit(1), commit(2)]);
    assert!(batch.validate().is_err());
}

#[test]
fn batch_allows_a_duplicate_identical_control() {
    let commit = Control::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    let batch = Batch::new(1, vec![admission()], vec![token_decode_operation()])
        .with_controls(vec![commit.clone(), commit]);
    assert!(batch.validate().is_ok());
}

#[test]
fn control_content_digest_distinguishes_variants() {
    let commit = Control::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: fixed_parent(),
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    let close = Control::Close {
        request_key: request_key(),
        control_seq: 1,
        cutoff: fixed_parent(),
        reason: CloseReason::Completed,
    };
    assert_ne!(commit.content_digest(), close.content_digest());
}

#[test]
fn commit_control_requires_a_fixed_selected_version() {
    let device_selected = VersionRef {
        request_key: request_key(),
        producer_op_id: OpId(9),
        point: Point::Device {
            selected_point: output_product(OpId(9)),
            producer_plan_digest: digest_string(0xcc),
        },
    };
    let control = Control::Commit {
        request_key: request_key(),
        control_seq: 1,
        expected_parent: fixed_parent(),
        selected: device_selected,
        public_event_limit: 1,
        disposition: Disposition::Publish,
    };
    assert!(control.validate().is_err());
}

#[test]
fn capabilities_round_trip_and_carry_the_layout_agreement() {
    let caps = EngineCaps::default();
    assert_eq!(caps.protocol_layout_digest, protocol_layout_digest());
    let response = WorkerResponse::capabilities(caps.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    let decoded = decoded.capabilities.unwrap();
    assert_eq!(decoded, caps);
    assert!(caps.agrees_with(&decoded));
}

#[test]
fn capabilities_with_a_disagreeing_layout_digest_are_rejected() {
    let caps = EngineCaps {
        protocol_layout_digest: digest_string(0x00),
        ..Default::default()
    };
    assert!(caps.validate().is_err());
}
