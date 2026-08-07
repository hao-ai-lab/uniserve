//! Emits the canonical Rust-computed digest-parity fixture consumed by the
//! Python mirror test (`tests/python/unit/contracts/test_batch_protocol.py`).
//!
//! The fixture carries one `Operation` and one `CompletionRecord` in their
//! serde wire form plus the Rust-computed `plan_digest` and `semantic_digest`.
//! The Python side reconstructs the records from the same wire form, recomputes
//! both digests, and asserts they are byte-identical, proving the host digest
//! algebra agrees across the two implementations.

use std::path::PathBuf;

use uniserve_core::RequestId;
use uniserve_worker_wire::{
    Bounds, CompletionRecord, CreditDimension, CreditVector, DType, DimBound, Domain, DrawLayout,
    EngineCaps, ExecutionConstraints, FinishFlags, LogicalLengths, OpId, OpStatus, Operation,
    Point, PointRange, ProductKind, ProductRef, RequestKey, Rng, RouteCreditLimits,
    RouteExecutionCapability, RouteId, SamplingOwnership, ShapeBound, StorageClass, TimingCounters,
    TokenMode, TokenSpan, VersionRef, Work, WorkVariant, protocol_layout_digest,
};

fn digest_string(seed: u8) -> String {
    format!("{seed:02x}").repeat(32)
}

fn request_key() -> RequestKey {
    RequestKey::new(4, RequestId(7), 2)
}

fn canonical_operation() -> Operation {
    let parent = VersionRef::admission_root(request_key(), OpId(1), digest_string(0xaa));
    let input = ProductRef {
        request_key: request_key(),
        producer_op_id: OpId(1),
        output_index: 0,
        generation: 1,
        kind: ProductKind::VisionFeature,
        storage_class: StorageClass::DeviceTensor,
        dtype: DType::BF16,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(3), DimBound::Device { max: 256 }],
        },
        point_range: PointRange {
            base_point: 0,
            max_points: 1,
        },
    };
    let token_output = ProductRef {
        request_key: request_key(),
        producer_op_id: OpId(11),
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
    };
    let selected_point_output = ProductRef {
        request_key: request_key(),
        producer_op_id: OpId(11),
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
    };
    let accepted_span_output = ProductRef {
        request_key: request_key(),
        producer_op_id: OpId(11),
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
    };
    let continuation_output = ProductRef {
        request_key: request_key(),
        producer_op_id: OpId(11),
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
    };
    Operation::registered(
        request_key(),
        OpId(11),
        parent,
        Work::Token(TokenMode::Decode),
        RouteId(9),
        Domain::Und,
        Bounds {
            max_points: 1,
            max_tokens: 1,
            max_kv_pages: 2,
            max_latent_bytes: 0,
            max_completion_bytes: 4096,
            max_transfer_bytes: 0,
        },
        vec![input],
        vec![
            token_output,
            selected_point_output,
            accepted_span_output,
            continuation_output,
        ],
        3,
        None,
        Some(Rng {
            seed: 0x0123_4567_89ab_cdef,
            semantic_index_base: 40,
            draw_layout: DrawLayout::TargetSampling,
        }),
        7,
    )
}

fn canonical_completion(semantic_digest: String) -> CompletionRecord {
    CompletionRecord {
        request_key: request_key(),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 41,
            kv_visible_len: 41,
            latent_len: 0,
            kv_reserved_len: 64,
            kv_initialized_len: 42,
            kv_committed_len: 41,
            kv_published_len: 40,
        },
        token_span: TokenSpan { base: 40, len: 1 },
        committed_tokens: vec![50256],
        finish_flags: FinishFlags {
            eos: false,
            length: false,
            stop: true,
        },
        product_generations: vec![3, 4, 5, 6],
        semantic_digest,
        error_code: None,
        timing_counters: TimingCounters::default(),
    }
}

fn fixture_path() -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..");
    workspace
        .join("tests")
        .join("python")
        .join("generated")
        .join("digest_parity.json")
}

#[test]
fn emit_digest_parity_fixture() {
    let parent_semantic_digest =
        VersionRef::admission_root(request_key(), OpId(1), digest_string(0xaa));
    let parent_semantic_digest = match parent_semantic_digest.point {
        Point::Fixed {
            semantic_digest, ..
        } => semantic_digest,
        Point::Device { .. } => unreachable!("admission root is fixed"),
    };

    let operation = canonical_operation();
    operation.validate().expect("canonical operation is valid");
    let plan_digest = operation.compute_plan_digest();
    assert_eq!(plan_digest, operation.plan_digest);

    // The completion carries the semantic digest it names once host-validated.
    let semantic_digest = canonical_completion(String::new())
        .compute_semantic_digest(&parent_semantic_digest, &plan_digest);
    let completion = canonical_completion(semantic_digest.clone());
    completion
        .validate()
        .expect("canonical completion is valid");

    // A fixed sample capability tuple for the cross-language route-capability
    // agreement. The supported-work list carries a duplicate to exercise the
    // sort-and-dedup step.
    let sample_work = [
        WorkVariant::TokenDecode,
        WorkVariant::TokenExtend,
        WorkVariant::GenFlow,
        WorkVariant::EncodeVision,
        WorkVariant::TokenDecode,
    ];
    let caps = EngineCaps {
        supported_work: sample_work.to_vec(),
        max_cfg_branches: 3,
        max_latent_size: 4096,
        max_vae_grid_tokens: 64,
        max_vit_grid_tokens: 64,
        max_latent_feature_bytes: 1 << 20,
        max_vision_feature_bytes: 1 << 21,
        execution_constraints: ExecutionConstraints {
            max_batch_operations: 16,
            route_capabilities: vec![RouteExecutionCapability {
                route: RouteId(0),
                supported_work: vec![
                    WorkVariant::TokenDecode,
                    WorkVariant::TokenExtend,
                    WorkVariant::GenFlow,
                    WorkVariant::EncodeVision,
                ],
                tensorized_mixed: true,
                sampling_ownership: SamplingOwnership::DesignatedRank,
                preemptible: false,
                credits: RouteCreditLimits {
                    per_request: CreditVector {
                        registered_operations: 2,
                        execution_slots: 2,
                        completion_slots: 2,
                        device_products: 10,
                        kv_pages: 64,
                        rollback_deltas: 34,
                        latent_artifact_bytes: 1 << 20,
                        pinned_completion_staging_bytes: 1 << 21,
                        transfer_bytes: 1 << 20,
                        transfer_tickets: 2,
                        cpu_tasks: 1,
                        output_journal_bytes: 1 << 24,
                    },
                    worker: CreditVector {
                        registered_operations: 16,
                        execution_slots: 16,
                        completion_slots: 16,
                        device_products: 80,
                        kv_pages: 256,
                        rollback_deltas: 272,
                        latent_artifact_bytes: 1 << 24,
                        pinned_completion_staging_bytes: 1 << 25,
                        transfer_bytes: 1 << 24,
                        transfer_tickets: 16,
                        cpu_tasks: 16,
                        output_journal_bytes: 1 << 28,
                    },
                },
                max_unresolved_window: 4,
                legal_feature_bitset: 0b0001_1111,
                sampler_processors: 0x3FFF,
                processor_order_revision: 1,
                rng_layouts: 0b101,
                graph_eligible: true,
                gen_conditioning: 2,
                max_points_per_operation: 17,
                mixed_row_combinations: vec![0b011, 0b101],
            }],
            ..ExecutionConstraints::default()
        },
        kv_dtype: "bfloat16".into(),
        model_dtype: "bfloat16".into(),
        attention_backend: "flashinfer".into(),
        ..Default::default()
    };
    let route_capability_digest = caps.compute_route_capability_digest();
    let layout_digest = protocol_layout_digest();

    let fixture = serde_json::json!({
        "parent_semantic_digest": parent_semantic_digest,
        "operation": operation,
        "completion": completion,
        "plan_digest": plan_digest,
        "semantic_digest": semantic_digest,
        "protocol_layout_digest": layout_digest,
        "route_capability_sample": {
            "supported_work": sample_work
                .iter()
                .map(|variant| variant.as_wire_str())
                .collect::<Vec<_>>(),
            "max_cfg_branches": caps.max_cfg_branches,
            "max_latent_size": caps.max_latent_size,
            "max_vae_grid_tokens": caps.max_vae_grid_tokens,
            "max_vit_grid_tokens": caps.max_vit_grid_tokens,
            "max_latent_feature_bytes": caps.max_latent_feature_bytes,
            "max_vision_feature_bytes": caps.max_vision_feature_bytes,
            "max_batch_operations": caps.execution_constraints.max_batch_operations,
            "max_speculative_points": caps.execution_constraints.max_speculative_points,
            "max_unresolved_window": caps.execution_constraints.max_unresolved_window,
            "device_sequence_lengths": caps.execution_constraints.device_sequence_lengths,
            "device_append_offsets": caps.execution_constraints.device_append_offsets,
            "incremental_kv_publication": caps.execution_constraints.incremental_kv_publication,
            "route_capabilities": caps.execution_constraints.route_capabilities.iter().map(|capability| serde_json::json!({
                "route": capability.route.0,
                "supported_work": capability.supported_work.iter().map(|variant| variant.as_wire_str()).collect::<Vec<_>>(),
                "tensorized_mixed": capability.tensorized_mixed,
                "sampling_ownership": match capability.sampling_ownership {
                    SamplingOwnership::DesignatedRank => "designated_rank",
                    SamplingOwnership::DeterministicSharded => "deterministic_sharded",
                },
                "preemptible": capability.preemptible,
                "credits": {
                    "per_request": CreditDimension::ALL.map(|dimension| capability.credits.per_request.get(dimension)),
                    "worker": CreditDimension::ALL.map(|dimension| capability.credits.worker.get(dimension)),
                },
                "max_unresolved_window": capability.max_unresolved_window,
                "legal_feature_bitset": capability.legal_feature_bitset,
                "sampler_processors": capability.sampler_processors,
                "processor_order_revision": capability.processor_order_revision,
                "rng_layouts": capability.rng_layouts,
                "graph_eligible": capability.graph_eligible,
                "gen_conditioning": capability.gen_conditioning,
                "max_points_per_operation": capability.max_points_per_operation,
                "mixed_row_combinations": capability.mixed_row_combinations,
            })).collect::<Vec<_>>(),
            "kv_dtype": caps.kv_dtype,
            "model_dtype": caps.model_dtype,
            "attention_backend": caps.attention_backend,
        },
        "route_capability_digest": route_capability_digest,
    });

    let path = fixture_path();
    std::fs::create_dir_all(path.parent().unwrap()).expect("create fixture directory");
    std::fs::write(&path, serde_json::to_vec_pretty(&fixture).unwrap()).expect("write fixture");
}
