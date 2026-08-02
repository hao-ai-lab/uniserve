//! Tests derived from the Checkpoint 1 protocol specification: round-trip
//! encoding for every record and `Work`/`Control` variant, identity validation,
//! and digest determinism.

use uniserve_core::{BlockId, KvGroupKind, RequestId};

use super::*;
use crate::flat::{
    decode_request, decode_request_unpack, decode_response, decode_response_unpack, encode_request,
    encode_response,
};

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
        dtype: DType::U32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn selected_point_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 1,
        generation: 4,
        kind: ProductKind::SelectedPoint,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::U32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(1)],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn accepted_span_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 2,
        generation: 5,
        kind: ProductKind::AcceptedSpan,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::U32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(2)],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn continuation_product(op: OpId) -> ProductRef {
    ProductRef {
        request_key: request_key(),
        producer_op_id: op,
        output_index: 3,
        generation: 6,
        kind: ProductKind::Continuation,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::I64,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(4)],
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
        vec![
            output_product(OpId(11)),
            selected_point_product(OpId(11)),
            accepted_span_product(OpId(11)),
            continuation_product(OpId(11)),
        ],
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
            ..LogicalLengths::default()
        },
        token_span: TokenSpan { base: 4, len: 1 },
        committed_tokens: vec![271],
        finish_flags: FinishFlags {
            eos: false,
            length: false,
            stop: false,
        },
        product_generations: vec![3, 4, 5, 6],
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
            finish_token_ids: vec![2, 7],
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

fn partitions_for_operations(operations: Vec<Operation>) -> Vec<BatchPartition> {
    let mut groups: Vec<(Domain, RouteId, Vec<Operation>)> = Vec::new();
    for operation in operations {
        if let Some((_, _, members)) = groups
            .iter_mut()
            .find(|(domain, route, _)| *domain == operation.domain && *route == operation.route)
        {
            members.push(operation);
        } else {
            groups.push((operation.domain, operation.route, vec![operation]));
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(index, (domain, route, operations))| BatchPartition {
            partition_id: index as u32 + 1,
            submission_group: index as u32 + 1,
            collective_seq: index as u64 + 1,
            domain,
            route,
            execution: ExecutionCapability::DomainHomogeneous,
            attention: AttentionRegime::Hybrid,
            shape_class: 0,
            operations,
        })
        .collect()
}

fn batch_with_operations(
    step_id: u64,
    admissions: Vec<Admission>,
    operations: Vec<Operation>,
) -> Batch {
    Batch::new(step_id, admissions, partitions_for_operations(operations))
}

fn partition_report(
    step_id: u64,
    completions: Vec<CompletionRecord>,
    products: Vec<ProductPayload>,
    visible: bool,
    worker_exec_us: Option<u64>,
    forward_stats: Option<WorkerForwardStats>,
) -> CompletionReport {
    CompletionReport {
        step_id,
        partitions: vec![PartitionCompletion {
            partition_id: 1,
            completions,
            products,
            registration: RegistrationAck { visible },
            worker_exec_us,
            forward_stats,
        }],
    }
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
        let batch = execute_round_trip(batch_with_operations(
            1,
            Vec::new(),
            vec![operation.clone()],
        ));
        assert_eq!(batch.operations().next().unwrap(), &operation);
        assert_eq!(batch.operations().next().unwrap().work, work);
    }
}

#[test]
fn version_ref_device_point_round_trips() {
    let device_parent = VersionRef {
        request_key: request_key(),
        producer_op_id: OpId(9),
        point: Point::Device {
            point_index: 0,
            selected_point: Some(selected_point_product(OpId(9))),
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
    let batch = execute_round_trip(batch_with_operations(2, Vec::new(), vec![operation]));
    assert_eq!(batch.operations().next().unwrap().parent, device_parent);
}

#[test]
fn operation_carries_new_kv_blocks_across_the_wire() {
    let base = token_decode_operation();
    assert_eq!(base.new_kv_blocks, vec![BlockId(7)]);
    let batch = execute_round_trip(batch_with_operations(9, Vec::new(), vec![base.clone()]));
    assert_eq!(
        batch.operations().next().unwrap().new_kv_blocks,
        vec![BlockId(7)]
    );
    assert_eq!(batch.operations().next().unwrap(), &base);
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
    let report = partition_report(
        5,
        vec![completion_record()],
        vec![ProductPayload {
            product: logprob,
            bytes: vec![1, 2, 3, 4],
        }],
        true,
        Some(10),
        None,
    );
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
    let report = partition_report(
        5,
        vec![completion_record()],
        vec![ProductPayload {
            product: logprob,
            bytes: vec![1, 2, 3, 4],
        }],
        true,
        Some(10),
        None,
    );

    assert!(report.validate().is_err());
}

#[test]
fn error_completion_round_trips_with_its_error_code() {
    let mut record = completion_record();
    record.status = OpStatus::Error;
    record.error_code = Some(ErrorCode::ComputeError);
    let report = partition_report(6, vec![record.clone()], Vec::new(), false, None, None);
    let decoded =
        decode_response(&encode_response(&WorkerResponse::completion_report(report)).unwrap())
            .unwrap();
    assert_eq!(
        decoded
            .completion_report
            .unwrap()
            .completions()
            .next()
            .unwrap(),
        &record
    );
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
    let batch = batch_with_operations(3, vec![admission()], vec![token_decode_operation()])
        .with_controls(vec![commit.clone(), close.clone(), release.clone()]);
    let decoded = execute_round_trip(batch);
    assert_eq!(decoded.controls, vec![commit, close, release]);
}

#[test]
fn admission_round_trips_and_binds_its_operation() {
    let batch = execute_round_trip(batch_with_operations(
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
        partitions: partitions_for_operations(vec![
            token_decode_operation(),
            token_decode_operation(),
        ]),
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
    let batch = batch_with_operations(1, vec![admission()], vec![operation])
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
    let batch = batch_with_operations(1, vec![admission()], vec![token_decode_operation()])
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
    let batch = batch_with_operations(1, vec![admission()], vec![token_decode_operation()])
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
            point_index: 0,
            selected_point: Some(selected_point_product(OpId(9))),
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

// ---------------------------------------------------------------------------
// Accessor-decode equivalence. The fixtures below cover every request and
// response kind, every `Work` and `Control` variant, fixed and device parents,
// admissions with full sampling and image parameters, and non-empty product
// payload bytes. Every scalar field carries a distinct value so a transposed
// field mapping cannot cancel out. Each fixture is asserted to (1) round-trip
// exactly through encode/decode and (2) decode byte-identically through the
// retired unpack-based path and the accessor-based path.
// ---------------------------------------------------------------------------

fn session_key(session: u64) -> RequestKey {
    RequestKey::new(4, RequestId(session), 2)
}

fn product_for(key: RequestKey, op: OpId, output_index: u16, kind: ProductKind) -> ProductRef {
    ProductRef {
        request_key: key,
        producer_op_id: op,
        output_index,
        generation: 3 + u32::from(output_index),
        kind,
        storage_class: match kind {
            ProductKind::Kv => StorageClass::PagedKv,
            ProductKind::Latent | ProductKind::LatentFeature => StorageClass::LatentArena,
            ProductKind::Completion => StorageClass::DeviceTensor,
            _ => StorageClass::DeviceTensor,
        },
        dtype: match kind {
            ProductKind::Token | ProductKind::SelectedPoint | ProductKind::AcceptedSpan => {
                DType::U32
            }
            ProductKind::Continuation => DType::I64,
            ProductKind::Kv => DType::BF16,
            ProductKind::Logprob => DType::F32,
            ProductKind::Completion => DType::U8,
            _ => DType::F16,
        },
        shape_bound: ShapeBound {
            dims: match kind {
                ProductKind::Token | ProductKind::SelectedPoint | ProductKind::Finish => {
                    vec![DimBound::Static(1)]
                }
                ProductKind::AcceptedSpan => vec![DimBound::Static(2)],
                ProductKind::Continuation => vec![DimBound::Static(4)],
                _ => vec![DimBound::Static(2), DimBound::Device { max: 16 }],
            },
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    }
}

fn full_sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        ignore_eos: true,
        seed: Some(0xdead_beef),
        min_p: 0.05,
        repetition_penalty: 1.1,
        frequency_penalty: 0.25,
        presence_penalty: -0.5,
        logit_bias: vec![(5, 1.5), (11, -2.0)],
        min_tokens: 3,
        n_logprobs: 4,
        return_logprobs: true,
        return_prompt_logprobs: true,
        n_prompt_logprobs: 2,
        logprob_token_ids: vec![7, 8, 9],
        bad_words_ids: vec![vec![1, 2, 3], vec![42]],
        allowed_token_ids: Some(vec![10, 11, 12]),
    }
}

fn full_image() -> ImageParams {
    ImageParams {
        steps: 20,
        cfg_text_scale: 5.0,
        cfg_img_scale: 1.5,
        cfg_renorm_type: "global".into(),
        cfg_renorm_min: 0.1,
        cfg_interval: (0.2, 0.9),
        timestep_shift: 3.0,
        height: 1024,
        width: 768,
        seed: Some(77),
        negative_prompt: "blurry".into(),
        max_images: 2,
        image_prompts: vec!["p0".into(), "p1".into()],
        retain_images: false,
    }
}

/// One operation per closed `Work` variant, each on its own request key so the
/// batch admits them together; the first two keys also carry admissions.
fn comprehensive_batch() -> Batch {
    let variants = [
        Work::Token(TokenMode::Extend),
        Work::Token(TokenMode::Decode),
        Work::Token(TokenMode::Verify),
        Work::Draft,
        Work::Encode(EncodeMode::Vision),
        Work::Encode(EncodeMode::Latent),
        Work::Transfer(TransferMode::Product),
        Work::Transfer(TransferMode::KvPublish),
        Work::Transfer(TransferMode::KvInstall),
        Work::Gen(GenMode::Transition),
        Work::Gen(GenMode::Flow),
        Work::Materialize,
    ];
    let mut operations = Vec::new();
    for (index, work) in variants.into_iter().enumerate() {
        let key = session_key(100 + index as u64);
        let op_id = OpId(11 + index as u64);
        // Alternate fixed and device parents across the set.
        let parent = if index % 2 == 0 {
            VersionRef::admission_root(key, OpId(1), digest_string(0xaa))
        } else {
            VersionRef {
                request_key: key,
                producer_op_id: OpId(9),
                point: Point::Device {
                    point_index: 0,
                    selected_point: Some(product_for(key, OpId(9), 0, ProductKind::SelectedPoint)),
                    producer_plan_digest: digest_string(0xcc),
                },
            }
        };
        let domain = if matches!(work, Work::Gen(_)) {
            Domain::Gen
        } else {
            Domain::Und
        };
        operations.push(Operation::registered(
            key,
            op_id,
            parent,
            work,
            RouteId(1 + index as u32),
            domain,
            Bounds {
                max_points: if work.advances_state() { 1 } else { 0 },
                max_tokens: 7 + index as u32,
                max_kv_pages: 3,
                max_latent_bytes: 1 << 20,
                max_completion_bytes: 4096,
                max_transfer_bytes: 1 << 16,
            },
            vec![product_for(
                session_key(50),
                OpId(2),
                0,
                ProductKind::VisionFeature,
            )],
            vec![
                product_for(key, op_id, 0, ProductKind::Token),
                product_for(key, op_id, 1, ProductKind::Kv),
            ],
            vec![BlockId(70 + index as u32), BlockId(90 + index as u32)],
            Some(product_for(
                session_key(51),
                OpId(3),
                0,
                ProductKind::Completion,
            )),
            Some(Rng {
                seed: 99 + index as u64,
                semantic_index_base: 4,
                draw_layout: match index % 3 {
                    0 => DrawLayout::TargetSampling,
                    1 => DrawLayout::SpeculativeProposal,
                    _ => DrawLayout::FlowNoise,
                },
            }),
            index as u64,
        ));
    }
    let und_admission = Admission::new(
        session_key(100),
        Some(UndAdmission {
            sampling: full_sampling(),
            negative_token_ids: vec![100, 101],
            finish_token_ids: vec![2, 7],
            kv: KvAllocation {
                block_ids: vec![BlockId(3), BlockId(4), BlockId(5)],
                prefix_len: 128,
                group_id: 1,
            },
        }),
        None,
        Some(6),
    )
    .unwrap();
    let gen_admission = Admission::new(
        session_key(110),
        None,
        Some(GenAdmission {
            image: full_image(),
        }),
        None,
    )
    .unwrap();
    let controls = vec![
        Control::Commit {
            request_key: session_key(200),
            control_seq: 1,
            expected_parent: VersionRef::admission_root(
                session_key(200),
                OpId(1),
                digest_string(0xaa),
            ),
            selected: VersionRef::admission_root(session_key(200), OpId(2), digest_string(0xab)),
            public_event_limit: 7,
            disposition: Disposition::Retain,
        },
        Control::Close {
            request_key: session_key(201),
            control_seq: 2,
            cutoff: VersionRef::admission_root(session_key(201), OpId(1), digest_string(0xac)),
            reason: CloseReason::Preempted,
        },
        Control::Release {
            request_key: session_key(202),
            op_id: OpId(33),
        },
    ];
    let mut input_product = product_for(
        operations[0].request_key,
        operations[0].op_id,
        2,
        ProductKind::Token,
    );
    input_product.storage_class = StorageClass::HostStaging;
    input_product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(4)],
    };
    operations[0].inputs.push(input_product.clone());
    operations[0].plan_digest = operations[0].compute_plan_digest();
    let input_products = vec![ProductPayload {
        product: input_product,
        bytes: encode_token_product_bytes(&[7, 8, 9, 10]),
    }];
    batch_with_operations(42, vec![und_admission, gen_admission], operations)
        .with_controls(controls)
        .with_input_products(input_products)
}

fn snapshot_fixture() -> SnapshotRef {
    SnapshotRef {
        version: VersionRef {
            request_key: RequestKey::new(1, RequestId(9), 3),
            producer_op_id: OpId(17),
            point: Point::Fixed {
                point_index: 17,
                semantic_digest: digest_string(0xdc),
            },
        },
        digest: digest_string(0xdd),
        locator: digest_string(0xdd),
    }
}

/// One fixture per `RequestKind`, plus call-id coverage on the execute frame.
fn request_fixtures() -> Vec<WorkerRequest> {
    let mut execute = WorkerRequest::execute(comprehensive_batch());
    execute.call_id = Some(91);
    vec![
        WorkerRequest::get_capabilities(),
        execute,
        WorkerRequest::poll_completions(42),
        WorkerRequest::drop_session(RequestId(42)),
        WorkerRequest::shutdown(),
        WorkerRequest::copy_kv(vec![(BlockId(1), BlockId(2)), (BlockId(3), BlockId(4))]),
        WorkerRequest::load_adapter(3, "adapters/alpha".into()),
        WorkerRequest::unload_adapter(3),
        WorkerRequest::release_products(vec![1, 2, 3]),
        WorkerRequest::reset_prefix_cache(),
        WorkerRequest::get_metrics(),
        WorkerRequest::get_pressure(),
        WorkerRequest::snapshot_session(RequestId(9)),
        WorkerRequest::restore_session(snapshot_fixture()),
    ]
}

fn full_caps() -> EngineCaps {
    let mut caps = EngineCaps {
        supported_work: WorkVariant::ALL.to_vec(),
        quantization: Some("fp8".into()),
        groups: vec![
            KvCacheGroupSpec {
                group_id: 0,
                block_offset: 0,
                num_blocks: 2048,
                kind: KvGroupKind::Full,
            },
            KvCacheGroupSpec {
                group_id: 1,
                block_offset: 2048,
                num_blocks: 1024,
                kind: KvGroupKind::SlidingWindow {
                    window: 4096,
                    sink: 64,
                },
            },
        ],
        rank: RankInfo {
            tp_rank: 1,
            tp_size: 2,
            pp_rank: 3,
            pp_size: 4,
            dp_rank: 5,
            dp_size: 6,
        },
        pipeline_depth: 2,
        encoder_cache_budget: 77,
        supported_controls: vec![RequestKind::Execute, RequestKind::DropSession],
        adapter_mode: AdapterMode::PerRequest,
        execution_constraints: ExecutionConstraints {
            max_batch_operations: 64,
            route_capabilities: vec![RouteExecutionCapability {
                route: RouteId(0),
                supported_work: WorkVariant::ALL.to_vec(),
                tensorized_mixed: true,
                sampling_ownership: SamplingOwnership::DesignatedRank,
                preemptible: false,
                credits: ExecutionConstraints::default().route_capabilities[0].credits,
            }],
            ..ExecutionConstraints::default()
        },
        resource_classes: vec![ResourceClass::KvBlock, ResourceClass::Adapter],
        model_spec_digest: digest_string(0x21),
        weight_digest: digest_string(0x22),
        restored_snapshots: vec![
            SnapshotRef {
                version: VersionRef::admission_root(session_key(5), OpId(1), digest_string(0x23)),
                digest: digest_string(0x24),
                locator: digest_string(0x24),
            },
            SnapshotRef {
                version: VersionRef::admission_root(session_key(6), OpId(1), digest_string(0x25)),
                digest: digest_string(0x26),
                locator: digest_string(0x26),
            },
        ],
        ..EngineCaps::default()
    };
    caps.route_capability_digest = caps.compute_route_capability_digest();
    caps
}

fn full_forward_stats() -> WorkerForwardStats {
    let map = |prefix: &str, base: u64| {
        BTreeMap::from([
            (format!("{prefix}.a"), base),
            (format!("{prefix}.b"), base + 1),
        ])
    };
    WorkerForwardStats {
        mode_counts: map("mode_counts", 1),
        mode_tokens: map("mode_tokens", 3),
        mode_us: map("mode_us", 5),
        component_us: map("component_us", 7),
        attention_launches: 11,
        attention_us: 12,
        attention_backend_counts: map("backend", 13),
        cuda_graph_captures: 15,
        cuda_graph_replays: 16,
        cuda_graph_misses: 17,
        cuda_graph_fallbacks: 18,
        cuda_graph_unpadded_tokens: 19,
        cuda_graph_padded_tokens: 20,
        cuda_graph_runtime_mode_counts: map("runtime_mode", 21),
        text_decode_token_relay_hits: 23,
        text_decode_token_relay_misses: 24,
        text_decode_position_relay_hits: 25,
        text_decode_position_relay_misses: 26,
        flashinfer_decode_plan_calls: 27,
        flashinfer_decode_plan_reuses: 28,
        flashinfer_decode_plan_rows: 29,
        flashinfer_decode_plan_indices: 30,
        flashinfer_decode_graph_plan_calls: 31,
        flashinfer_decode_graph_plan_reuses: 32,
        spec_verify_rows: 33,
        spec_verify_draft_tokens: 34,
        spec_verify_accepted_tokens: 35,
        spec_verify_rejected_tokens: 36,
        spec_verify_committed_tokens: 37,
        spec_verify_path_counts: map("spec_path", 38),
    }
}

fn full_completion_report() -> CompletionReport {
    let ok_record = CompletionRecord {
        request_key: session_key(100),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 5,
            kv_visible_len: 6,
            latent_len: 7,
            kv_reserved_len: 64,
            kv_initialized_len: 8,
            kv_committed_len: 6,
            kv_published_len: 5,
        },
        token_span: TokenSpan { base: 4, len: 2 },
        committed_tokens: vec![271, 272],
        finish_flags: FinishFlags {
            eos: true,
            length: false,
            stop: false,
        },
        product_generations: vec![3, 5],
        semantic_digest: digest_string(0xbb),
        error_code: None,
        timing_counters: TimingCounters {
            queued_us: 41,
            device_us: 42,
            copy_us: 43,
            host_us: 44,
        },
    };
    let mut predicated_record = ok_record.clone();
    predicated_record.request_key = session_key(101);
    predicated_record.op_id = OpId(12);
    predicated_record.status = OpStatus::Predicated;
    predicated_record.token_span.len = 0;
    predicated_record.committed_tokens = Vec::new();
    predicated_record.finish_flags = FinishFlags::default();
    predicated_record.product_generations = Vec::new();
    let mut error_record = ok_record.clone();
    error_record.request_key = session_key(102);
    error_record.op_id = OpId(13);
    error_record.status = OpStatus::Error;
    error_record.error_code = Some(ErrorCode::ResourceExhausted);
    error_record.finish_flags = FinishFlags {
        eos: false,
        length: false,
        stop: true,
    };
    let mut artifact = product_for(session_key(102), OpId(13), 3, ProductKind::Artifact);
    artifact.storage_class = StorageClass::CompletionArena;
    artifact.dtype = DType::U8;
    artifact.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(256)],
    };
    let products = vec![
        ProductPayload {
            product: product_for(session_key(100), OpId(11), 2, ProductKind::Logprob),
            bytes: vec![1, 2, 3, 4, 5],
        },
        ProductPayload {
            product: artifact,
            bytes: (0..=255).collect(),
        },
    ];
    partition_report(
        5,
        vec![ok_record, predicated_record, error_record],
        products,
        true,
        Some(1234),
        Some(full_forward_stats()),
    )
}

fn full_metrics() -> WorkerMetrics {
    let map = |prefix: &str, base: u64| {
        BTreeMap::from([
            (format!("{prefix}.a"), base),
            (format!("{prefix}.b"), base + 1),
        ])
    };
    WorkerMetrics {
        executes: 51,
        operations_total: 52,
        exec_us_total: 53,
        last_exec_us: 54,
        operation_counts: map("operation_counts", 55),
        operation_us: map("operation_us", 57),
        control_ok: map("control_ok", 59),
        control_err: map("control_err", 61),
        error_counts: map("error_counts", 63),
        cuda_graph_captures: 65,
        cuda_graph_replays: 66,
        cuda_graph_misses: 67,
        cuda_graph_fallbacks: 68,
        cuda_graph_unpadded_tokens: 69,
        cuda_graph_padded_tokens: 70,
        cuda_graph_runtime_mode_counts: map("runtime_mode", 71),
        forward: Some(full_forward_stats()),
    }
}

/// One fixture per `ResponseKind`, plus a second capabilities frame that fills
/// every optional field the default leaves empty.
fn response_fixtures() -> Vec<WorkerResponse> {
    let bare = |kind: ResponseKind| WorkerResponse {
        kind,
        call_id: Some(17),
        capabilities: None,
        completion_report: None,
        metrics: None,
        pressure: None,
        message: None,
        code: None,
        retryable: None,
        fatal: None,
        phase: None,
        route: None,
        operations: Vec::new(),
        snapshot: None,
    };
    vec![
        WorkerResponse::capabilities(EngineCaps::default()),
        WorkerResponse::capabilities(full_caps()),
        WorkerResponse::completion_report(full_completion_report()),
        WorkerResponse::ok(),
        WorkerResponse {
            message: Some("device fault on decode".into()),
            code: Some("compute_error".into()),
            retryable: Some(true),
            fatal: Some(false),
            phase: Some("execute".into()),
            route: Some("und.decode".into()),
            operations: vec![
                ErrorOperationIdentity {
                    request_key: session_key(100),
                    op_id: OpId(11),
                },
                ErrorOperationIdentity {
                    request_key: session_key(101),
                    op_id: OpId(12),
                },
            ],
            ..bare(ResponseKind::Error)
        },
        WorkerResponse {
            metrics: Some(full_metrics()),
            ..bare(ResponseKind::Metrics)
        },
        WorkerResponse {
            pressure: Some(vec![
                ResourcePressure {
                    class: ResourceClass::KvBlock,
                    total: 81,
                    used: 82,
                    evictable: 83,
                    free: 84,
                },
                ResourcePressure {
                    class: ResourceClass::ImageLatent,
                    total: 85,
                    used: 86,
                    evictable: 87,
                    free: 88,
                },
            ]),
            ..bare(ResponseKind::Pressure)
        },
        WorkerResponse::snapshot(snapshot_fixture()),
    ]
}

#[test]
fn accessor_decode_round_trips_and_matches_unpack_for_every_request_kind() {
    let fixtures = request_fixtures();
    for kind in RequestKind::ALL {
        assert!(
            fixtures.iter().any(|request| request.kind == kind),
            "no fixture covers request kind {kind:?}"
        );
    }
    for request in fixtures {
        let bytes = encode_request(&request).unwrap();
        let accessor = decode_request(&bytes).unwrap();
        let unpack = decode_request_unpack(&bytes).unwrap();
        assert_eq!(accessor, request, "round trip for {:?}", request.kind);
        assert_eq!(
            accessor, unpack,
            "accessor vs unpack for {:?}",
            request.kind
        );
    }
}

#[test]
fn accessor_decode_round_trips_and_matches_unpack_for_every_response_kind() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Capabilities,
        ResponseKind::Result,
        ResponseKind::Ok,
        ResponseKind::Error,
        ResponseKind::Metrics,
        ResponseKind::Pressure,
        ResponseKind::Snapshot,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for response in fixtures {
        let bytes = encode_response(&response).unwrap();
        let accessor = decode_response(&bytes).unwrap();
        let unpack = decode_response_unpack(&bytes).unwrap();
        assert_eq!(accessor, response, "round trip for {:?}", response.kind);
        assert_eq!(
            accessor, unpack,
            "accessor vs unpack for {:?}",
            response.kind
        );
    }
}

#[test]
fn accessor_decode_rejects_malformed_frames_like_unpack_decode() {
    // Both paths must agree on rejection too, not just success.
    let garbage: &[u8] = &[0x01, 0x02, 0x03];
    assert!(decode_request(garbage).is_err());
    assert!(decode_request_unpack(garbage).is_err());
    // A structurally valid frame whose payload shape is invalid for its kind:
    // an execute request with no batch.
    let mut request = WorkerRequest::get_capabilities();
    request.kind = RequestKind::Execute;
    let bytes = encode_request(&request).unwrap();
    let accessor_err = decode_request(&bytes).unwrap_err().to_string();
    let unpack_err = decode_request_unpack(&bytes).unwrap_err().to_string();
    assert_eq!(accessor_err, unpack_err);
    assert_eq!(accessor_err, "execute request has no batch");
}

/// A realistic decode-step submission: `operations` token-decode operations on
/// distinct sessions, mirroring the measured 35-operation production batch.
fn decode_step_batch(operations: usize) -> Batch {
    let ops = (0..operations)
        .map(|index| {
            let key = session_key(1000 + index as u64);
            let op_id = OpId(11);
            Operation::registered(
                key,
                op_id,
                VersionRef::admission_root(key, OpId(1), digest_string(0xaa)),
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
                vec![
                    product_for(key, op_id, 0, ProductKind::Token),
                    product_for(key, op_id, 1, ProductKind::Kv),
                ],
                vec![BlockId(7 + index as u32)],
                None,
                Some(Rng {
                    seed: 99 + index as u64,
                    semantic_index_base: 4,
                    draw_layout: DrawLayout::TargetSampling,
                }),
                0,
            )
        })
        .collect();
    batch_with_operations(1, Vec::new(), ops)
}

#[test]
#[ignore = "micro-benchmark: cargo test -p uniserve-worker-wire --release -- --ignored --nocapture"]
fn bench_decode_unpack_vs_accessor() {
    let request = WorkerRequest::execute(decode_step_batch(35));
    let request_bytes = encode_request(&request).unwrap();
    let mut payload_batch = decode_step_batch(35);
    let payload_operation = payload_batch
        .partitions
        .first_mut()
        .expect("decode benchmark has a partition")
        .operations
        .first_mut()
        .expect("decode benchmark has operations");
    let mut input_product = product_for(
        payload_operation.request_key,
        payload_operation.op_id,
        2,
        ProductKind::Artifact,
    );
    input_product.storage_class = StorageClass::HostStaging;
    input_product.dtype = DType::U8;
    input_product.shape_bound = ShapeBound {
        dims: vec![DimBound::Static(1 << 20)],
    };
    payload_operation.inputs.push(input_product.clone());
    payload_operation.plan_digest = payload_operation.compute_plan_digest();
    let payload_request =
        WorkerRequest::execute(payload_batch.with_input_products(vec![ProductPayload {
            product: input_product,
            bytes: vec![0x5a; 1 << 20],
        }]));
    let payload_request_bytes = encode_request(&payload_request).unwrap();
    let mut report = full_completion_report();
    report.partitions[0].products[1].bytes = vec![0xa5; 4 << 20];
    let response_bytes = encode_response(&WorkerResponse::completion_report(report)).unwrap();

    let time = |mut run: Box<dyn FnMut()>| {
        for _ in 0..50 {
            run();
        }
        let iters = 500u32;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            run();
        }
        start.elapsed().as_nanos() / u128::from(iters)
    };
    let unpack_request_ns = time(Box::new(|| {
        std::hint::black_box(decode_request_unpack(&request_bytes).unwrap());
    }));
    let accessor_request_ns = time(Box::new(|| {
        std::hint::black_box(decode_request(&request_bytes).unwrap());
    }));
    let unpack_payload_ns = time(Box::new(|| {
        std::hint::black_box(decode_request_unpack(&payload_request_bytes).unwrap());
    }));
    let accessor_payload_ns = time(Box::new(|| {
        std::hint::black_box(decode_request(&payload_request_bytes).unwrap());
    }));
    let unpack_response_ns = time(Box::new(|| {
        std::hint::black_box(decode_response_unpack(&response_bytes).unwrap());
    }));
    let accessor_response_ns = time(Box::new(|| {
        std::hint::black_box(decode_response(&response_bytes).unwrap());
    }));
    println!(
        "decode_request (35-op decode batch, {} bytes): unpack {unpack_request_ns} ns/iter, \
         accessor {accessor_request_ns} ns/iter ({:.2}x)",
        request_bytes.len(),
        unpack_request_ns as f64 / accessor_request_ns as f64
    );
    println!(
        "decode_request (35-op batch + 1 MiB input product, {} bytes): unpack \
         {unpack_payload_ns} ns/iter, accessor {accessor_payload_ns} ns/iter ({:.2}x)",
        payload_request_bytes.len(),
        unpack_payload_ns as f64 / accessor_payload_ns as f64
    );
    println!(
        "decode_response (completion report with 4 MiB product, {} bytes): unpack \
         {unpack_response_ns} ns/iter, accessor {accessor_response_ns} ns/iter ({:.2}x)",
        response_bytes.len(),
        unpack_response_ns as f64 / accessor_response_ns as f64
    );
}
