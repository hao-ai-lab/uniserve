//! Canonical execution-payload identity (Rust twin).
//!
//! Stage 1 of `specs/unified_forward_execution.md` requires Python and Rust to
//! agree byte-for-byte on canonical execution identity. This module mirrors
//! the value tree and exact byte encoding of
//! `uniserve_worker/contracts/execution.py::canonical_payload_fingerprint`;
//! the shared vectors in `crates/protocol/vocab/execution_fingerprint.toml`
//! pin the agreement from both sides, so any unilateral field reorder,
//! retype, or tag change fails a test in one language.
//!
//! Encoding layout (all integers little-endian):
//!
//! * `0x01` + i64        — integer field
//! * `0x02` + f64        — float field (IEEE 754 bits)
//! * `0x03` + u32 + raw  — bytes field
//! * `0x04` + u32        — sequence header (element count; elements follow)
//! * `0x05` + u32        — struct header (field count incl. the tag slot)
//! * `0x06` + u32        — enum value (operation tag or 0 in the tag slot)
//! * `0x07`              — absent optional
//!
//! The encoding is positional: field names never enter the digest. The
//! cumulative acknowledgement is envelope metadata and is excluded, so a
//! retry that only advances acknowledgement keeps its transaction identity.
//!
//! Dormant: nothing on the production wire consumes these values yet.

use sha2::{Digest, Sha256};

/// Sealed operation tags shared with the Python contract and the program
/// algebra; adding a variant is a cross-language protocol change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ExecOperationTag {
    SequenceStep = 1,
    FlowStep = 2,
    EncodeStep = 3,
    MaterializeStep = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecEngineRef {
    pub deployment_id: i64,
    pub engine_epoch: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecSessionRef {
    pub engine: ExecEngineRef,
    pub request_id: i64,
    pub incarnation: i64,
    pub session_version: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecCacheLease {
    pub lease_id: i64,
    pub engine_epoch: i64,
    pub identity_digest: Vec<u8>,
    pub identity_schema: i64,
    pub charge: i64,
    pub version: i64,
    pub residency_handle: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ExecProductLifetime {
    Transition = 1,
    Request = 2,
    Session = 3,
    Cache = 4,
    AcknowledgedOutput = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ExecTransferKind {
    LocalResidency = 1,
    CudaIpc = 2,
    SharedMemory = 3,
    Mooncake = 4,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecProductLease {
    pub lease_id: i64,
    pub schema_id: i64,
    pub producer: ExecSessionRef,
    pub product_version: i64,
    pub extent_rows: i64,
    pub lifetime: ExecProductLifetime,
    pub transfer: ExecTransferKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecSamplingSpec {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecNewSession {
    pub request_id: i64,
    pub incarnation: i64,
    pub sampling: ExecSamplingSpec,
    pub base_seed: i64,
    pub max_history_tokens: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecCandidateVerification {
    pub candidate_tokens: Vec<i64>,
    pub candidate_positions: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecOperation {
    Sequence {
        input_tokens: Vec<i64>,
        history_length: i64,
        position_begin: i64,
        requested_outputs: i64,
        verification: Option<ExecCandidateVerification>,
    },
    Flow {
        schedule_id: i64,
        step_index: i64,
        total_steps: i64,
        input_product: i64,
        branch_coefficients: Vec<f64>,
        conditioning_products: Vec<i64>,
        output_schema: i64,
    },
    Encode {
        kind: u32,
        input_product: i64,
        output_schema: i64,
        grid: (i64, i64, i64),
    },
    Materialize {
        input_product: i64,
        output_schema: i64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecExecuteRow {
    pub row_id: i64,
    pub session: ExecSessionRef,
    pub operation: ExecOperation,
    pub admission: Option<ExecNewSession>,
    pub cache_leases: Vec<ExecCacheLease>,
    pub product_leases: Vec<ExecProductLease>,
    pub scheduler_op_id: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecExecuteBatch {
    pub engine_epoch: i64,
    pub step_id: i64,
    pub acknowledged_through: i64,
    pub rows: Vec<ExecExecuteRow>,
}

#[derive(Default)]
struct Encoder {
    parts: Vec<u8>,
}

impl Encoder {
    fn integer(&mut self, value: i64) {
        self.parts.push(0x01);
        self.parts.extend_from_slice(&value.to_le_bytes());
    }

    fn real(&mut self, value: f64) {
        self.parts.push(0x02);
        self.parts.extend_from_slice(&value.to_le_bytes());
    }

    fn raw(&mut self, value: &[u8]) {
        self.parts.push(0x03);
        self.parts
            .extend_from_slice(&u32::try_from(value.len()).expect("bounded").to_le_bytes());
        self.parts.extend_from_slice(value);
    }

    fn sequence(&mut self, count: usize) {
        self.parts.push(0x04);
        self.parts
            .extend_from_slice(&u32::try_from(count).expect("bounded").to_le_bytes());
    }

    fn structure(&mut self, field_count: u32) {
        self.parts.push(0x05);
        self.parts.extend_from_slice(&field_count.to_le_bytes());
    }

    fn enumeration(&mut self, value: u32) {
        self.parts.push(0x06);
        self.parts.extend_from_slice(&value.to_le_bytes());
    }

    fn absent(&mut self) {
        self.parts.push(0x07);
    }

    fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.parts);
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn encode_engine(encoder: &mut Encoder, engine: &ExecEngineRef) {
    encoder.structure(3);
    encoder.enumeration(0);
    encoder.integer(engine.deployment_id);
    encoder.integer(engine.engine_epoch);
}

fn encode_session(encoder: &mut Encoder, session: &ExecSessionRef) {
    encoder.structure(5);
    encoder.enumeration(0);
    encode_engine(encoder, &session.engine);
    encoder.integer(session.request_id);
    encoder.integer(session.incarnation);
    encoder.integer(session.session_version);
}

fn encode_int_seq(encoder: &mut Encoder, values: &[i64]) {
    encoder.sequence(values.len());
    for value in values {
        encoder.integer(*value);
    }
}

fn encode_cache_lease(encoder: &mut Encoder, lease: &ExecCacheLease) {
    encoder.structure(8);
    encoder.enumeration(0);
    encoder.integer(lease.lease_id);
    encoder.integer(lease.engine_epoch);
    encoder.raw(&lease.identity_digest);
    encoder.integer(lease.identity_schema);
    encoder.integer(lease.charge);
    encoder.integer(lease.version);
    encoder.integer(lease.residency_handle);
}

fn encode_product_lease(encoder: &mut Encoder, lease: &ExecProductLease) {
    encoder.structure(8);
    encoder.enumeration(0);
    encoder.integer(lease.lease_id);
    encoder.integer(lease.schema_id);
    encode_session(encoder, &lease.producer);
    encoder.integer(lease.product_version);
    encoder.integer(lease.extent_rows);
    encoder.enumeration(lease.lifetime as u32);
    encoder.enumeration(lease.transfer as u32);
}

fn encode_admission(encoder: &mut Encoder, admission: &ExecNewSession) {
    encoder.structure(6);
    encoder.enumeration(0);
    encoder.integer(admission.request_id);
    encoder.integer(admission.incarnation);
    let sampling = &admission.sampling;
    encoder.structure(8);
    encoder.enumeration(0);
    encoder.real(sampling.temperature);
    encoder.real(sampling.top_p);
    encoder.integer(sampling.top_k);
    encoder.real(sampling.min_p);
    encoder.real(sampling.repetition_penalty);
    encoder.real(sampling.frequency_penalty);
    encoder.real(sampling.presence_penalty);
    encoder.integer(admission.base_seed);
    encoder.integer(admission.max_history_tokens);
}

fn encode_operation(encoder: &mut Encoder, operation: &ExecOperation) {
    match operation {
        ExecOperation::Sequence {
            input_tokens,
            history_length,
            position_begin,
            requested_outputs,
            verification,
        } => {
            encoder.structure(6);
            encoder.enumeration(ExecOperationTag::SequenceStep as u32);
            encode_int_seq(encoder, input_tokens);
            encoder.integer(*history_length);
            encoder.integer(*position_begin);
            encoder.integer(*requested_outputs);
            match verification {
                None => encoder.absent(),
                Some(verification) => {
                    encoder.structure(3);
                    encoder.enumeration(0);
                    encode_int_seq(encoder, &verification.candidate_tokens);
                    encode_int_seq(encoder, &verification.candidate_positions);
                }
            }
        }
        ExecOperation::Flow {
            schedule_id,
            step_index,
            total_steps,
            input_product,
            branch_coefficients,
            conditioning_products,
            output_schema,
        } => {
            encoder.structure(8);
            encoder.enumeration(ExecOperationTag::FlowStep as u32);
            encoder.integer(*schedule_id);
            encoder.integer(*step_index);
            encoder.integer(*total_steps);
            encoder.integer(*input_product);
            encoder.sequence(branch_coefficients.len());
            for coefficient in branch_coefficients {
                encoder.real(*coefficient);
            }
            encode_int_seq(encoder, conditioning_products);
            encoder.integer(*output_schema);
        }
        ExecOperation::Encode {
            kind,
            input_product,
            output_schema,
            grid,
        } => {
            encoder.structure(5);
            encoder.enumeration(ExecOperationTag::EncodeStep as u32);
            encoder.enumeration(*kind);
            encoder.integer(*input_product);
            encoder.integer(*output_schema);
            encoder.sequence(3);
            encoder.integer(grid.0);
            encoder.integer(grid.1);
            encoder.integer(grid.2);
        }
        ExecOperation::Materialize {
            input_product,
            output_schema,
        } => {
            encoder.structure(3);
            encoder.enumeration(ExecOperationTag::MaterializeStep as u32);
            encoder.integer(*input_product);
            encoder.integer(*output_schema);
        }
    }
}

fn encode_row(encoder: &mut Encoder, row: &ExecExecuteRow) {
    encoder.structure(8);
    encoder.enumeration(0);
    encoder.integer(row.row_id);
    encode_session(encoder, &row.session);
    encode_operation(encoder, &row.operation);
    match &row.admission {
        None => encoder.absent(),
        Some(admission) => encode_admission(encoder, admission),
    }
    encoder.sequence(row.cache_leases.len());
    for lease in &row.cache_leases {
        encode_cache_lease(encoder, lease);
    }
    encoder.sequence(row.product_leases.len());
    for lease in &row.product_leases {
        encode_product_lease(encoder, lease);
    }
    encoder.integer(row.scheduler_op_id);
}

/// Canonical execution-payload identity (sha256 hex). Excludes the cumulative
/// acknowledgement — see the module docs.
pub fn canonical_payload_fingerprint(batch: &ExecExecuteBatch) -> String {
    let mut encoder = Encoder::default();
    encoder.structure(3);
    encoder.integer(batch.engine_epoch);
    encoder.integer(batch.step_id);
    encoder.sequence(batch.rows.len());
    for row in &batch.rows {
        encode_row(&mut encoder, row);
    }
    encoder.digest()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_identity_vectors::{shared_vector, vector_digests_from_vocab};

    #[test]
    fn shared_vectors_match_the_pinned_cross_language_digests() {
        let pinned = vector_digests_from_vocab();
        assert!(!pinned.is_empty(), "vocab fixture lists no vectors");
        for (name, expected) in pinned {
            let batch = shared_vector(&name);
            assert_eq!(
                canonical_payload_fingerprint(&batch),
                expected,
                "fingerprint drift for shared vector {name}"
            );
        }
    }

    #[test]
    fn acknowledgement_is_excluded_from_identity() {
        let mut batch = shared_vector("single_sequence_row");
        let baseline = canonical_payload_fingerprint(&batch);
        batch.acknowledged_through += 5;
        assert_eq!(canonical_payload_fingerprint(&batch), baseline);
    }

    #[test]
    fn row_values_change_identity() {
        let batch = shared_vector("single_sequence_row");
        let baseline = canonical_payload_fingerprint(&batch);
        let mut changed = batch.clone();
        if let ExecOperation::Sequence { input_tokens, .. } = &mut changed.rows[0].operation {
            input_tokens[2] += 1;
        }
        assert_ne!(canonical_payload_fingerprint(&changed), baseline);
    }
}
