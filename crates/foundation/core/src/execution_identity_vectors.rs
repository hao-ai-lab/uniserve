//! Shared cross-language fingerprint vectors (test support).
//!
//! Each constructor here mirrors, value for value, a batch built by the
//! Python side in `tests/python/contract/worker/test_execution_contracts_vectors.py`.
//! The pinned digests live in `crates/protocol/vocab/execution_fingerprint.toml`;
//! both languages must reproduce them.

use crate::execution_identity::{
    ExecCacheLease, ExecCandidateVerification, ExecEngineRef, ExecExecuteBatch, ExecExecuteRow,
    ExecNewSession, ExecOperation, ExecProductLease, ExecProductLifetime, ExecSamplingSpec,
    ExecSessionRef, ExecTransferKind,
};

const ENGINE: ExecEngineRef = ExecEngineRef {
    deployment_id: 1,
    engine_epoch: 7,
};

fn session(request_id: i64, incarnation: i64, version: i64) -> ExecSessionRef {
    ExecSessionRef {
        engine: ENGINE,
        request_id,
        incarnation,
        session_version: version,
    }
}

pub fn shared_vector(name: &str) -> ExecExecuteBatch {
    match name {
        "single_sequence_row" => ExecExecuteBatch {
            engine_epoch: 7,
            step_id: 9,
            acknowledged_through: 4,
            rows: vec![ExecExecuteRow {
                row_id: 0,
                session: session(41, 1, 3),
                operation: ExecOperation::Sequence {
                    input_tokens: vec![5, 6, 7],
                    history_length: 12,
                    position_begin: 12,
                    requested_outputs: 1,
                    verification: None,
                },
                admission: None,
                cache_leases: vec![],
                product_leases: vec![],
                scheduler_op_id: 100,
            }],
        },
        "admission_verification_and_leases" => ExecExecuteBatch {
            engine_epoch: 7,
            step_id: 10,
            acknowledged_through: 9,
            rows: vec![ExecExecuteRow {
                row_id: 0,
                session: session(55, 2, 0),
                operation: ExecOperation::Sequence {
                    input_tokens: vec![],
                    history_length: 8,
                    position_begin: 8,
                    requested_outputs: 1,
                    verification: Some(ExecCandidateVerification {
                        candidate_tokens: vec![11, 12, 13],
                        candidate_positions: vec![8, 9, 10],
                    }),
                },
                admission: Some(ExecNewSession {
                    request_id: 55,
                    incarnation: 2,
                    sampling: ExecSamplingSpec {
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 20,
                        min_p: 0.0,
                        repetition_penalty: 1.0,
                        frequency_penalty: 0.0,
                        presence_penalty: 0.0,
                    },
                    base_seed: 42,
                    max_history_tokens: 4096,
                }),
                cache_leases: vec![ExecCacheLease {
                    lease_id: 5,
                    engine_epoch: 7,
                    identity_digest: vec![0; 32],
                    identity_schema: 1,
                    charge: 128,
                    version: 2,
                    residency_handle: 77,
                }],
                product_leases: vec![ExecProductLease {
                    lease_id: 9,
                    schema_id: 3,
                    producer: session(50, 1, 4),
                    product_version: 2,
                    extent_rows: 64,
                    lifetime: ExecProductLifetime::Request,
                    transfer: ExecTransferKind::LocalResidency,
                }],
                scheduler_op_id: 200,
            }],
        },
        "flow_and_encode_rows" => ExecExecuteBatch {
            engine_epoch: 7,
            step_id: 11,
            acknowledged_through: 10,
            rows: vec![
                ExecExecuteRow {
                    row_id: 0,
                    session: session(60, 1, 5),
                    operation: ExecOperation::Flow {
                        schedule_id: 1,
                        step_index: 17,
                        total_steps: 50,
                        input_product: 3,
                        branch_coefficients: vec![4.0, 1.0, 1.5],
                        conditioning_products: vec![1, 2],
                        output_schema: 5,
                    },
                    admission: None,
                    cache_leases: vec![],
                    product_leases: vec![],
                    scheduler_op_id: 300,
                },
                ExecExecuteRow {
                    row_id: 1,
                    session: session(61, 1, 6),
                    operation: ExecOperation::Encode {
                        kind: 1,
                        input_product: 8,
                        output_schema: 9,
                        grid: (1, 64, 36),
                    },
                    admission: None,
                    cache_leases: vec![],
                    product_leases: vec![],
                    scheduler_op_id: 301,
                },
            ],
        },
        other => panic!("unknown shared vector {other}"),
    }
}

pub fn vector_digests_from_vocab() -> Vec<(String, String)> {
    const SCHEMA: &str = include_str!("../../../protocol/vocab/execution_fingerprint.toml");
    let schema: toml::Value = toml::from_str(SCHEMA).expect("valid fixture");
    schema["vector"]
        .as_array()
        .expect("vector array")
        .iter()
        .map(|entry| {
            (
                entry["name"].as_str().expect("name").to_owned(),
                entry["digest"].as_str().expect("digest").to_owned(),
            )
        })
        .collect()
}
