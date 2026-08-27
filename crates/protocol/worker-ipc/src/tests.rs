//! Protocol round trips, canonical identity, and descriptor validation.

use std::collections::BTreeMap;

use uniserve_core::{BlockId, CfgRenorm, Digest, KvGroupKind, ModelDtype, RequestId};

use super::*;
use crate::codec::{decode_request, decode_response, encode_request, encode_response};

fn digest_string(seed: u8) -> Digest {
    Digest::try_from(format!("{seed:02x}").repeat(32)).unwrap()
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
    Operation {
        request_key: request_key(),
        op_id: OpId(11),
        parent: fixed_parent(),
        work: ForwardMode::TokenDecode,
        route: RouteId(1),
        domain: Domain::Decode,
        advances_state: false,
        bounds: Bounds {
            max_points: 1,
            max_tokens: 1,
            max_kv_pages: 1,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: vec![
            output_product(OpId(11)),
            selected_point_product(OpId(11)),
            accepted_span_product(OpId(11)),
            continuation_product(OpId(11)),
        ],
        predicate: None,
        rng: Some(Rng {
            seed: 99,
            semantic_index_base: 4,
            draw_layout: DrawLayout::TargetSampling,
        }),
        control_seq: 0,
        plan_digest: Digest::zero(),
    }
    .sealed()
}

fn operation_for(work: ForwardMode, op_id: OpId, advances: bool) -> Operation {
    Operation {
        request_key: request_key(),
        op_id,
        parent: fixed_parent(),
        work,
        route: RouteId(1),
        domain: work.domain(),
        advances_state: false,
        bounds: Bounds {
            max_points: if advances { 1 } else { 0 },
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: Vec::new(),
        predicate: None,
        rng: None,
        control_seq: 0,
        plan_digest: Digest::zero(),
    }
    .sealed()
}

fn completion_record() -> ModelOutput {
    ModelOutput {
        request_key: request_key(),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 5,
            kv_visible_len: 5,
            kv_computed_len: 5,
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
        u32::try_from(request_key().session_id.0).unwrap(),
        Some(UndAdmission {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: vec![2, 7],
            initial_position: 0,
        }),
        None,
    )
    .unwrap()
}

fn execute_round_trip(batch: Batch) -> Batch {
    let request = WorkerRequest::execute(batch);
    let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
    decoded.batch().unwrap().clone()
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
        .map(|(index, (domain, route, operations))| {
            let mut block_tables = Vec::new();
            let mut new_cache_pages = Vec::new();
            let mut forward_rows = Vec::new();
            for (operation_index, operation) in operations.iter().enumerate() {
                let capacity_pages = operation.bounds.max_kv_pages;
                if capacity_pages == 0 {
                    continue;
                }
                let request_pool_idx = u32::try_from(operation.request_key.session_id.0).unwrap();
                let page_ids = (1..=capacity_pages).map(BlockId).collect::<Vec<_>>();
                if !block_tables.iter().any(|table: &BlockTable| {
                    table.request_pool_idx == request_pool_idx && table.group_id == 0
                }) {
                    block_tables.push(BlockTable {
                        request_pool_idx,
                        group_id: 0,
                        page_ids: page_ids.clone(),
                        allocated_tokens: operation.bounds.max_tokens.max(1),
                    });
                    new_cache_pages.push(CachePageAllocation {
                        request_pool_idx,
                        group_id: 0,
                        page_ids,
                    });
                }
                forward_rows.push(RowGeometry {
                    operation_index: operation_index as u32,
                    request_pool_index: request_pool_idx,
                    seq_len: 0,
                    query_len: operation.bounds.max_tokens.max(1),
                });
            }
            let latent_placements = operations
                .iter()
                .filter(|operation| {
                    matches!(
                        operation.work,
                        ForwardMode::GenTransition | ForwardMode::GenFlow
                    ) || operation
                        .inputs
                        .iter()
                        .any(|reference| reference.kind == ProductKind::Latent)
                })
                .map(|operation| LatentPlacement {
                    request_key: operation.request_key,
                    op_id: operation.op_id,
                    page_table: vec![1],
                    latent_units: 1,
                    height: 1,
                    width: 1,
                    start_step: 0,
                    step_count: u32::from(operation.work == ForwardMode::GenFlow),
                })
                .collect();
            BatchPartition {
                partition_id: index as u32 + 1,
                submission_group: index as u32 + 1,
                collective_seq: index as u64 + 1,
                domain,
                route,
                attention: AttentionRegime::Hybrid,
                shape_class: 0,
                operations,
                block_tables,
                new_cache_pages,
                forward_rows,
                latent_placements,
                decode_placements: Vec::new(),
            }
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
    completions: Vec<ModelOutput>,
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
        (ForwardMode::TokenExtend, true, Domain::Prefill),
        (ForwardMode::TokenDecode, true, Domain::Decode),
        (ForwardMode::TokenVerify, true, Domain::Decode),
        (ForwardMode::Draft, false, Domain::Decode),
        (ForwardMode::EncodeVision, false, Domain::Prefill),
        (ForwardMode::EncodeLatent, false, Domain::Prefill),
        (ForwardMode::TransferProduct, false, Domain::Prefill),
        (ForwardMode::TransferKvPublish, false, Domain::Prefill),
        (ForwardMode::TransferKvInstall, false, Domain::Prefill),
        (ForwardMode::GenTransition, true, Domain::Flow),
        (ForwardMode::GenFlow, true, Domain::Flow),
        (ForwardMode::Materialize, false, Domain::Flow),
    ];
    for (index, (work, advances, domain)) in variants.into_iter().enumerate() {
        assert_eq!(
            work.advances_state(),
            advances,
            "work table effect mismatch"
        );
        assert_eq!(work.domain(), domain, "work table domain mismatch");
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
    let operation = Operation {
        request_key: request_key(),
        op_id: OpId(12),
        parent: device_parent.clone(),
        work: ForwardMode::TokenDecode,
        route: RouteId(1),
        domain: Domain::Decode,
        advances_state: false,
        bounds: Bounds {
            max_points: 1,
            ..Bounds::default()
        },
        inputs: Vec::new(),
        outputs: vec![output_product(OpId(12))],
        predicate: None,
        rng: None,
        control_seq: 0,
        plan_digest: Digest::zero(),
    }
    .sealed();
    let batch = execute_round_trip(batch_with_operations(2, Vec::new(), vec![operation]));
    assert_eq!(batch.operations().next().unwrap().parent, device_parent);
}

#[test]
fn block_table_placement_round_trips_without_changing_operation_identity() {
    let base = token_decode_operation();
    let batch = execute_round_trip(batch_with_operations(9, Vec::new(), vec![base.clone()]));
    assert_eq!(batch.operations().next().unwrap(), &base);
    assert_eq!(
        batch.partitions[0].block_tables[0].page_ids,
        vec![BlockId(1)]
    );
    let mut relocated = batch.clone();
    relocated.partitions[0].block_tables[0].page_ids = vec![BlockId(17)];
    relocated.partitions[0].new_cache_pages[0].page_ids = vec![BlockId(17)];
    assert_eq!(
        relocated.operations().next().unwrap().plan_digest,
        base.plan_digest
    );
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
    assert_eq!(decoded.report().unwrap(), &report);
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
        decoded.report().unwrap().completions().next().unwrap(),
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
    let mut relocated = admission();
    relocated.request_pool_idx = 19;
    assert_eq!(relocated.payload_digest(), admission().digest);
}

#[test]
fn plan_digest_is_deterministic_and_binds_control_seq() {
    let mut a = token_decode_operation();
    let b = token_decode_operation();
    assert_eq!(a.plan_digest, b.plan_digest);
    assert_eq!(a.plan_digest, a.compute_plan_digest());
    a.control_seq = 999;
    assert_ne!(a.compute_plan_digest(), b.plan_digest);
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
fn validation_rejects_a_work_domain_mismatch() {
    let mut operation = token_decode_operation();
    operation.domain = Domain::Flow;
    operation.plan_digest = operation.compute_plan_digest();
    assert!(operation.validate().is_err());
}

#[test]
fn kv_publication_requires_a_fixed_semantic_parent() {
    let mut operation = operation_for(ForwardMode::TransferKvPublish, OpId(12), false);
    operation.parent = VersionRef {
        request_key: request_key(),
        producer_op_id: OpId(9),
        point: Point::Device {
            point_index: 1,
            selected_point: None,
            producer_plan_digest: digest_string(0xcc),
        },
    };
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
fn validation_allows_shared_encoder_features_and_rejects_foreign_lineage_state() {
    let foreign_key = RequestKey::new(4, RequestId(8), 2);
    let mut feature = output_product(OpId(3));
    feature.request_key = foreign_key;
    feature.kind = ProductKind::VisionFeature;
    let mut operation = token_decode_operation();
    operation.inputs = vec![feature];
    operation.plan_digest = operation.compute_plan_digest();
    operation.validate().unwrap();

    operation.inputs[0].kind = ProductKind::Token;
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
        allowed_token_ids: Some(Vec::new()),
        suppressed_token_ids: vec![7, 2, 7],
        finish_token_ids: vec![11, 5, 11],
        transition_token_ids: vec![29, 13, 29],
        force_finish: true,
    };
    let decoded =
        decode_sampling_state_bytes(&encode_sampling_state_bytes(&state)).expect("decode state");
    assert_eq!(
        decoded,
        SamplingState {
            allowed_token_ids: Some(Vec::new()),
            suppressed_token_ids: vec![2, 7],
            finish_token_ids: vec![5, 11],
            transition_token_ids: vec![13, 29],
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
fn capabilities_round_trip_with_the_canonical_layout() {
    let caps = WorkerCapabilities::default();
    let response = WorkerResponse::capabilities(caps.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    let WorkerResponse::Capabilities {
        capabilities: decoded,
        ..
    } = decoded
    else {
        panic!("decoded response must preserve its capabilities variant");
    };
    assert_eq!(decoded, caps);
}

#[test]
fn kv_free_capabilities_round_trip_without_kv_geometry() {
    let caps = WorkerCapabilities {
        block_size: 0,
        num_blocks: 0,
        num_layers: 0,
        num_kv_heads: 0,
        head_dim: 0,
        supported_work: vec![ForwardMode::GenTransition, ForwardMode::GenFlow],
        latent_page_units: 64,
        num_latent_pages: 3,
        latent_width: 1,
        latent_dtype: Some(ModelDtype::Float32),
        bytes_per_token: 0,
        groups: Vec::new(),
        kv_dtype: None,
        resource_classes: vec![ResourceClass::ImageLatent],
        ..WorkerCapabilities::default()
    };
    let response = WorkerResponse::capabilities(caps.clone());
    let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
    let WorkerResponse::Capabilities {
        capabilities: decoded,
        ..
    } = decoded
    else {
        panic!("decoded response must preserve its capabilities variant");
    };
    assert_eq!(decoded, caps);
}

#[test]
fn capabilities_reject_duplicate_set_members() {
    let caps = WorkerCapabilities {
        supported_work: vec![
            ForwardMode::TokenExtend,
            ForwardMode::TokenDecode,
            ForwardMode::TokenExtend,
        ],
        ..Default::default()
    };
    assert!(encode_response(&WorkerResponse::capabilities(caps)).is_err());

    let caps = WorkerCapabilities {
        supported_controls: vec![RequestKind::Execute, RequestKind::Execute],
        ..Default::default()
    };
    assert!(encode_response(&WorkerResponse::capabilities(caps)).is_err());

    let caps = WorkerCapabilities {
        resource_classes: vec![ResourceClass::KvBlock, ResourceClass::KvBlock],
        ..Default::default()
    };
    assert!(encode_response(&WorkerResponse::capabilities(caps)).is_err());
}

#[test]
fn capabilities_require_group_totals_to_match_the_cache() {
    let mut caps = full_caps();
    caps.groups[1].num_blocks = 2047;
    assert!(encode_response(&WorkerResponse::capabilities(caps)).is_err());
}

#[test]
fn image_generation_requires_incremental_kv_publication() {
    let mut caps = full_caps();
    caps.incremental_kv_publication = false;
    assert!(
        !caps
            .generation_runtime_capabilities()
            .features
            .contains(uniserve_core::GenerationFeatures::IMAGE_GENERATION)
    );

    caps.incremental_kv_publication = true;
    assert!(
        caps.generation_runtime_capabilities()
            .features
            .contains(uniserve_core::GenerationFeatures::IMAGE_GENERATION)
    );
}

// ---------------------------------------------------------------------------
// Canonical wire fixtures cover every request and response kind, every `ForwardMode`
// and `Control` variant, fixed and device parents, full admissions, and non-empty
// product payloads. Distinct scalar values expose transposed field mappings.
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
        typical_p: 0.8,
        forced_token_ids: vec![13, 14],
    }
}

fn full_image() -> ImageParams {
    ImageParams {
        steps: 20,
        cfg_text_scale: 5.0,
        cfg_img_scale: 1.5,
        cfg_renorm_type: CfgRenorm::Global,
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

/// One operation per closed `ForwardMode` variant, each on its own request key so the
/// batch admits them together; the first two keys also carry admissions.
fn comprehensive_batch() -> Batch {
    let variants = [
        ForwardMode::TokenExtend,
        ForwardMode::TokenDecode,
        ForwardMode::TokenVerify,
        ForwardMode::Draft,
        ForwardMode::EncodeVision,
        ForwardMode::EncodeLatent,
        ForwardMode::TransferProduct,
        ForwardMode::TransferKvPublish,
        ForwardMode::TransferKvInstall,
        ForwardMode::GenTransition,
        ForwardMode::GenFlow,
        ForwardMode::Materialize,
    ];
    let mut operations = Vec::new();
    for (index, work) in variants.into_iter().enumerate() {
        let key = session_key(100 + index as u64);
        let op_id = OpId(11 + index as u64);
        let parent = if work.requires_fixed_parent() || index % 2 == 0 {
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
        let domain = work.domain();
        operations.push(
            Operation {
                request_key: key,
                op_id,
                parent,
                work,
                route: RouteId(1 + index as u32),
                domain,
                advances_state: false,
                bounds: Bounds {
                    max_points: if work.advances_state() { 1 } else { 0 },
                    max_tokens: 7 + index as u32,
                    max_kv_pages: 3,
                    max_latent_bytes: 1 << 20,
                    max_completion_bytes: 4096,
                    max_transfer_bytes: 1 << 16,
                },
                inputs: vec![product_for(key, OpId(2), 0, ProductKind::VisionFeature)],
                outputs: vec![
                    product_for(key, op_id, 0, ProductKind::Token),
                    product_for(key, op_id, 1, ProductKind::Kv),
                ],
                predicate: Some(product_for(key, OpId(3), 0, ProductKind::Completion)),
                rng: Some(Rng {
                    seed: 99 + index as u64,
                    semantic_index_base: 4,
                    draw_layout: match index % 3 {
                        0 => DrawLayout::TargetSampling,
                        1 => DrawLayout::SpeculativeProposal,
                        _ => DrawLayout::FlowNoise,
                    },
                }),
                control_seq: index as u64,
                plan_digest: Digest::zero(),
            }
            .sealed(),
        );
    }
    let und_admission = Admission::new(
        session_key(100),
        100,
        Some(UndAdmission {
            sampling: full_sampling(),
            negative_token_ids: vec![100, 101],
            finish_token_ids: vec![2, 7],
            initial_position: 128,
        }),
        None,
    )
    .unwrap();
    let gen_admission = Admission::new(
        session_key(110),
        110,
        None,
        Some(GenAdmission {
            image: full_image(),
        }),
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

/// One fixture per `RequestKind`, plus call-id coverage on the execute frame.
fn request_fixtures() -> Vec<WorkerRequest> {
    let mut execute = WorkerRequest::execute(comprehensive_batch());
    execute.set_call_id(Some(91));
    vec![
        WorkerRequest::get_capabilities(),
        execute,
        WorkerRequest::poll_completions(42),
        WorkerRequest::drop_session(RequestId(42)),
        WorkerRequest::shutdown(),
        WorkerRequest::release_products(vec![1, 2, 3]),
        WorkerRequest::get_pressure(),
    ]
}

fn full_caps() -> WorkerCapabilities {
    WorkerCapabilities {
        supported_work: ForwardMode::ALL.to_vec(),
        groups: vec![
            KvCacheGroupSpec {
                num_blocks: 2048,
                kind: KvGroupKind::Full,
            },
            KvCacheGroupSpec {
                num_blocks: 2048,
                kind: KvGroupKind::SlidingWindow {
                    window: 4096,
                    sink: 64,
                },
            },
        ],
        rank: RankInfo {
            tp_rank: 1,
            tp_size: 2,
        },
        pipeline_depth: 2,
        encoder_cache_budget: 77,
        supported_controls: vec![RequestKind::Execute, RequestKind::DropSession],
        max_batch_operations: 64,
        max_batch_tokens: 4096,
        max_request_pool_size: 96,
        max_unresolved_window: 3,
        mixed_buckets: vec![MixedExecutionCapability {
            decode_rows: 1,
            flow_rows: 1,
            height: 1152,
            width: 2048,
            cfg_branches: 3,
        }],
        sampling_ownership: SamplingOwnership::DesignatedRank,
        resource_classes: vec![ResourceClass::KvBlock, ResourceClass::ImageLatent],
        latent_page_units: 64,
        num_latent_pages: 17,
        latent_width: 16,
        latent_dtype: Some(ModelDtype::BFloat16),
        model_identity: Some(digest_string(0x21)),
        weight_digest: Some(digest_string(0x22)),
        ..WorkerCapabilities::default()
    }
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
    let ok_record = ModelOutput {
        request_key: session_key(100),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 5,
            kv_visible_len: 6,
            latent_len: 7,
            kv_computed_len: 8,
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
    artifact.storage_class = StorageClass::PinnedOutput;
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

/// One fixture per `ResponseKind`, plus a second capabilities frame that
/// exercises non-default capability values.
fn response_fixtures() -> Vec<WorkerResponse> {
    vec![
        WorkerResponse::Capabilities {
            call_id: None,
            capabilities: WorkerCapabilities::default(),
        },
        WorkerResponse::Capabilities {
            call_id: Some(17),
            capabilities: full_caps(),
        },
        WorkerResponse::Result {
            call_id: Some(17),
            completion_report: full_completion_report(),
        },
        WorkerResponse::Ok { call_id: Some(17) },
        WorkerResponse::Error {
            call_id: Some(17),
            error: WorkerResponseError {
                message: "device fault on decode".into(),
                code: Some("compute_error".into()),
                retryable: true,
                fatal: false,
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
            },
        },
        WorkerResponse::Pressure {
            call_id: Some(17),
            pressure: vec![
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
            ],
        },
    ]
}

#[test]
fn every_request_kind_round_trips_through_the_wire() {
    let fixtures = request_fixtures();
    for kind in RequestKind::ALL {
        assert!(
            fixtures.iter().any(|request| request.kind() == kind),
            "no fixture covers request kind {kind:?}"
        );
    }
    for request in fixtures {
        let bytes = encode_request(&request).unwrap();
        assert_eq!(
            decode_request(&bytes).unwrap(),
            request,
            "round trip for {:?}",
            request.kind()
        );
    }
}

#[test]
fn every_response_kind_round_trips_through_the_wire() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Capabilities,
        ResponseKind::Result,
        ResponseKind::Ok,
        ResponseKind::Error,
        ResponseKind::Pressure,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind() == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for response in fixtures {
        let bytes = encode_response(&response).unwrap();
        assert_eq!(
            decode_response(&bytes).unwrap(),
            response,
            "round trip for {:?}",
            response.kind()
        );
    }
}

#[test]
fn wire_decode_rejects_malformed_frames() {
    let garbage: &[u8] = &[0x01, 0x02, 0x03];
    assert!(decode_request(garbage).is_err());
    assert!(decode_response(garbage).is_err());

    let mut request = encode_request(&WorkerRequest::get_capabilities()).unwrap();
    request.truncate(request.len() / 2);
    assert!(decode_request(&request).is_err());

    let mut response = encode_response(&WorkerResponse::ok()).unwrap();
    response.truncate(response.len() / 2);
    assert!(decode_response(&response).is_err());
}
