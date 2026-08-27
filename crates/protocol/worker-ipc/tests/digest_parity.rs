//! Emits the canonical Rust-computed digest-parity fixture consumed by the
//! Python mirror test (`tests/python/unit/contracts/test_batch_protocol.py`).
//!
//! The fixture carries one `Operation` and one `ModelOutput` in their
//! serde wire form plus the Rust-computed `plan_digest` and `semantic_digest`.
//! The Python side reconstructs the records from the same wire form, recomputes
//! both digests, and asserts they are byte-identical, proving the host digest
//! algebra agrees across the two implementations.

use std::path::PathBuf;

use uniserve_core::{Digest, RequestId};
use uniserve_worker_ipc::{
    Bounds, DType, DimBound, Domain, DrawLayout, FinishFlags, ForwardMode, LogicalLengths,
    ModelOutput, OpId, OpStatus, Operation, OperationSpec, Point, PointRange, ProductKind,
    ProductRef, RequestKey, Rng, RouteId, ShapeBound, StorageClass, TimingCounters, TokenSpan,
    VersionRef,
};

fn digest_string(seed: u8) -> Digest {
    Digest::try_from(format!("{seed:02x}").repeat(32)).unwrap()
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
    Operation::registered(OperationSpec {
        request_key: request_key(),
        op_id: OpId(11),
        parent,
        work: ForwardMode::TokenDecode,
        route: RouteId(9),
        domain: Domain::Decode,
        bounds: Bounds {
            max_points: 1,
            max_tokens: 1,
            max_kv_pages: 2,
            max_latent_bytes: 0,
            max_completion_bytes: 4096,
            max_transfer_bytes: 0,
        },
        inputs: vec![input],
        outputs: vec![
            token_output,
            selected_point_output,
            accepted_span_output,
            continuation_output,
        ],
        predicate: None,
        rng: Some(Rng {
            seed: 0x0123_4567_89ab_cdef,
            semantic_index_base: 40,
            draw_layout: DrawLayout::TargetSampling,
        }),
        control_seq: 7,
    })
}

fn canonical_completion(semantic_digest: Digest) -> ModelOutput {
    ModelOutput {
        request_key: request_key(),
        op_id: OpId(11),
        completion_slot_generation: 2,
        status: OpStatus::Ok,
        selected_point: 1,
        logical_lengths: LogicalLengths {
            token_len: 41,
            kv_visible_len: 41,
            latent_len: 0,
            kv_computed_len: 42,
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
    let semantic_digest = canonical_completion(Digest::zero())
        .compute_semantic_digest(&parent_semantic_digest, &plan_digest);
    let completion = canonical_completion(semantic_digest.clone());
    completion
        .validate()
        .expect("canonical completion is valid");

    let fixture = serde_json::json!({
        "parent_semantic_digest": parent_semantic_digest,
        "operation": operation,
        "completion": completion,
        "plan_digest": plan_digest,
        "semantic_digest": semantic_digest,
    });

    let path = fixture_path();
    std::fs::create_dir_all(path.parent().unwrap()).expect("create fixture directory");
    std::fs::write(&path, serde_json::to_vec_pretty(&fixture).unwrap()).expect("write fixture");
}
