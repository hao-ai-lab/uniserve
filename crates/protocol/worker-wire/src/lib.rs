//! Versioned worker execution protocol.
//!
//! Protocol v3 defines one closed operation algebra: sequence, flow, encode,
//! materialize, and transfer. Every effect is returned through the matching
//! typed delta.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uniserve_core::{BlockId, ImageParams, KvCacheGroupSpec, RankInfo, RequestId, SamplingParams};
pub use uniserve_core::{
    EncodeKind, MaterializeKind, OperationKind, OperationType, SequenceMode, TransferKind,
};

pub mod flat;
pub mod resources;
#[allow(warnings)]
pub mod schema {
    include!(concat!(env!("OUT_DIR"), "/flatbuffers/mod.rs"));
}

pub use resources::{
    LeasePolicy, ResourceClass, ResourceEvent, ResourceEventKind, ResourceHandle, ResourceLease,
    ResourcePressure,
};

pub const EXECUTION_PROTOCOL_VERSION: u16 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    Wire,
    LastSampled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvAllocation {
    pub block_ids: Vec<BlockId>,
    pub prefix_len: u32,
    pub group_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceAdmission {
    pub sampling: SamplingParams,
    pub negative_token_ids: Vec<u32>,
    pub kv: KvAllocation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowAdmission {
    pub image: ImageParams,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    pub session_id: RequestId,
    pub digest: String,
    pub sequence: Option<SequenceAdmission>,
    pub flow: Option<FlowAdmission>,
    pub adapter_id: Option<u32>,
}

impl Admission {
    pub fn new(
        session_id: RequestId,
        sequence: Option<SequenceAdmission>,
        flow: Option<FlowAdmission>,
        adapter_id: Option<u32>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            sequence.is_some() || flow.is_some(),
            "admission must declare sequence or flow state"
        );
        let mut admission = Self {
            session_id,
            digest: String::new(),
            sequence,
            flow,
            adapter_id,
        };
        admission.digest = admission.payload_digest(EXECUTION_PROTOCOL_VERSION);
        Ok(admission)
    }

    pub fn validate(&self, protocol_version: u16) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.sequence.is_some() || self.flow.is_some(),
            "admission must declare sequence or flow state"
        );
        anyhow::ensure!(
            is_digest(&self.digest),
            "admission digest must be a lowercase SHA-256 digest"
        );
        anyhow::ensure!(
            self.digest == self.payload_digest(protocol_version),
            "admission digest mismatch for session {}",
            self.session_id.0
        );
        if let Some(sequence) = &self.sequence {
            sequence.sampling.validate()?;
            anyhow::ensure!(
                sequence.kv.prefix_len == 0 || !sequence.kv.block_ids.is_empty(),
                "a non-empty KV prefix requires allocated blocks"
            );
            anyhow::ensure!(
                sequence.kv.block_ids.iter().collect::<HashSet<_>>().len()
                    == sequence.kv.block_ids.len(),
                "KV allocation repeats a logical block"
            );
        }
        if let Some(flow) = &self.flow {
            flow.image.validate()?;
        }
        Ok(())
    }

    pub fn payload_digest(&self, protocol_version: u16) -> String {
        let mut digest = CanonicalDigest::new(b"uniserve-admission-v3\0", protocol_version);
        digest.u64(self.session_id.0);
        digest.option(self.sequence.as_ref(), |digest, sequence| {
            digest.sampling(&sequence.sampling);
            digest.u32s(sequence.negative_token_ids.iter().copied());
            digest.u32s(sequence.kv.block_ids.iter().map(|block| block.0));
            digest.u32(sequence.kv.prefix_len);
            digest.u32(sequence.kv.group_id);
        });
        digest.option(self.flow.as_ref(), |digest, flow| digest.image(&flow.image));
        digest.option(self.adapter_id, CanonicalDigest::u32);
        digest.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvLeaseDelta {
    pub group_id: u32,
    pub new_blocks: Vec<BlockId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TokenPolicy {
    pub allowed_tokens: Vec<u32>,
    pub suppress_tokens: Vec<u32>,
    pub recent_tokens: Vec<u32>,
    pub publish_kv: bool,
    pub publish_kv_on_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenInput {
    pub token_ids: Vec<u32>,
    pub source: TokenSource,
    pub draft_token_ids: Vec<u32>,
    pub return_all_logits: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedProduct {
    pub handle: u64,
    pub locator: String,
}

impl PublishedProduct {
    fn validate(&self, label: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.handle > 0 || !self.locator.is_empty(),
            "{label} requires a handle or locator"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedKv {
    pub handle: u64,
    /// Data-plane locators grouped contiguously in tensor-parallel rank order.
    pub locators: Vec<String>,
    pub source_version: u64,
    pub kv_tokens: u32,
    pub block_ids: Vec<BlockId>,
    pub group_id: u32,
    pub position: u32,
}

impl PublishedKv {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.handle > 0 || !self.locators.is_empty(),
            "published KV requires a local handle or data-plane locators"
        );
        anyhow::ensure!(
            self.source_version > 0,
            "published KV source version must be positive"
        );
        anyhow::ensure!(
            self.kv_tokens == 0 || !self.block_ids.is_empty(),
            "non-empty published KV requires logical blocks"
        );
        anyhow::ensure!(
            self.block_ids.iter().collect::<HashSet<_>>().len() == self.block_ids.len(),
            "published KV repeats a logical block"
        );
        anyhow::ensure!(
            self.locators.iter().all(|locator| !locator.is_empty()),
            "published KV contains an empty locator"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SequenceInput {
    Tokens(TokenInput),
    PublishedLogits(PublishedProduct),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceOperation {
    pub mode: SequenceMode,
    pub lease: KvLeaseDelta,
    pub position: (u32, u32),
    pub policy: TokenPolicy,
    pub input: SequenceInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guidance {
    pub branch_count: u8,
    pub text_scale: f32,
    pub image_scale: f32,
    pub renorm_type: String,
    pub renorm_min: f32,
    pub interval: (f32, f32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowOperation {
    pub latent_handle: u64,
    pub position: u32,
    pub start_step: u16,
    pub step_count: u16,
    pub conditioning_position: u32,
    pub conditioning: Option<PublishedKv>,
    pub guidance: Guidance,
    pub image_prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum EncodeInput {
    InlineImage { base64: String, content_hash: u64 },
    StagedProduct { handle: u64, content_hash: u64 },
    CachedProduct { content_hash: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodeOperation {
    pub kind: EncodeKind,
    pub lease: KvLeaseDelta,
    pub position: (u32, u32),
    pub conditioning_position: u32,
    pub input: EncodeInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum MaterializeInput {
    Latent { handle: u64 },
    Published(PublishedProduct),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializeOperation {
    pub kind: MaterializeKind,
    pub lease: KvLeaseDelta,
    pub position: u32,
    pub conditioning_position: u32,
    pub policy: TokenPolicy,
    pub input: MaterializeInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferOperation {
    pub kind: TransferKind,
    pub lease: KvLeaseDelta,
    pub position: u32,
    pub conditioning_position: u32,
    pub policy: TokenPolicy,
    pub source: PublishedProduct,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Operation {
    Sequence(SequenceOperation),
    Flow(FlowOperation),
    Encode(EncodeOperation),
    Materialize(MaterializeOperation),
    Transfer(TransferOperation),
}

impl Operation {
    pub const fn kind(&self) -> OperationKind {
        match self {
            Self::Sequence(_) => OperationKind::Sequence,
            Self::Flow(_) => OperationKind::Flow,
            Self::Encode(_) => OperationKind::Encode,
            Self::Materialize(_) => OperationKind::Materialize,
            Self::Transfer(_) => OperationKind::Transfer,
        }
    }

    pub const fn operation_type(&self) -> OperationType {
        match self {
            Self::Sequence(operation) => match operation.mode {
                SequenceMode::Extend => OperationType::SequenceExtend,
                SequenceMode::Decode => OperationType::SequenceDecode,
                SequenceMode::Verify => OperationType::SequenceVerify,
                SequenceMode::Sample => OperationType::SequenceSample,
            },
            Self::Flow(_) => OperationType::Flow,
            Self::Encode(operation) => match operation.kind {
                EncodeKind::Vision => OperationType::EncodeVision,
                EncodeKind::Latent => OperationType::EncodeLatent,
            },
            Self::Materialize(operation) => match operation.kind {
                MaterializeKind::Image => OperationType::MaterializeImage,
                MaterializeKind::Frame => OperationType::MaterializeFrame,
            },
            Self::Transfer(operation) => match operation.kind {
                TransferKind::Product => OperationType::TransferProduct,
                TransferKind::Kv => OperationType::TransferKv,
            },
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Sequence(operation) => {
                anyhow::ensure!(
                    operation.position.1 >= operation.position.0,
                    "sequence position range is inverted"
                );
                match (&operation.mode, &operation.input) {
                    (SequenceMode::Sample, SequenceInput::PublishedLogits(product)) => {
                        product.validate("published logits")
                    }
                    (SequenceMode::Sample, SequenceInput::Tokens(_)) => {
                        anyhow::bail!("sample sequence requires published logits")
                    }
                    (_, SequenceInput::PublishedLogits(_)) => {
                        anyhow::bail!("model sequence requires token input")
                    }
                    (_, SequenceInput::Tokens(input)) => {
                        anyhow::ensure!(
                            !input.token_ids.is_empty(),
                            "model sequence requires at least one token id"
                        );
                        match operation.mode {
                            SequenceMode::Extend => {
                                anyhow::ensure!(
                                    input.source == TokenSource::Wire,
                                    "sequence extend requires wire token input"
                                );
                                anyhow::ensure!(
                                    input.draft_token_ids.is_empty(),
                                    "sequence extend cannot carry draft work"
                                );
                            }
                            SequenceMode::Decode => {
                                anyhow::ensure!(
                                    input.token_ids.len() == 1 && input.draft_token_ids.is_empty(),
                                    "sequence decode requires one input token and no draft"
                                );
                            }
                            SequenceMode::Verify => {
                                anyhow::ensure!(
                                    input.token_ids.len() == 1 && !input.draft_token_ids.is_empty(),
                                    "sequence verify requires one input token and a draft"
                                );
                            }
                            SequenceMode::Sample => {
                                unreachable!("sample/token mismatch is rejected by the outer match")
                            }
                        }
                        Ok(())
                    }
                }
            }
            Self::Flow(operation) => {
                anyhow::ensure!(
                    operation.latent_handle > 0,
                    "flow operation requires a latent handle"
                );
                anyhow::ensure!(operation.step_count > 0, "flow step count must be positive");
                anyhow::ensure!(
                    operation.guidance.branch_count > 0,
                    "flow guidance requires a branch"
                );
                if let Some(conditioning) = &operation.conditioning {
                    conditioning.validate()?;
                }
                validate_finite_guidance(&operation.guidance)
            }
            Self::Encode(operation) => {
                anyhow::ensure!(
                    operation.position.1 >= operation.position.0,
                    "encode position range is inverted"
                );
                match &operation.input {
                    EncodeInput::InlineImage {
                        base64,
                        content_hash,
                    } => {
                        anyhow::ensure!(!base64.is_empty(), "inline encode input is empty");
                        anyhow::ensure!(*content_hash > 0, "encode content hash must be positive");
                    }
                    EncodeInput::StagedProduct {
                        handle,
                        content_hash,
                    } => {
                        anyhow::ensure!(
                            *handle > 0 && *content_hash > 0,
                            "staged encode input is invalid"
                        );
                    }
                    EncodeInput::CachedProduct { content_hash } => {
                        anyhow::ensure!(*content_hash > 0, "cached encode input is invalid")
                    }
                }
                Ok(())
            }
            Self::Materialize(operation) => match &operation.input {
                MaterializeInput::Latent { handle } => {
                    anyhow::ensure!(
                        *handle > 0,
                        "materialize operation requires a latent handle"
                    );
                    Ok(())
                }
                MaterializeInput::Published(product) => product.validate("materialize input"),
            },
            Self::Transfer(operation) => operation.source.validate("transfer source"),
        }
    }
}

fn validate_finite_guidance(guidance: &Guidance) -> anyhow::Result<()> {
    for value in [
        guidance.text_scale,
        guidance.image_scale,
        guidance.renorm_min,
        guidance.interval.0,
        guidance.interval.1,
    ] {
        anyhow::ensure!(value.is_finite(), "guidance values must be finite");
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationEnvelope {
    pub session_id: RequestId,
    pub epoch: u64,
    pub op_id: u64,
    pub base_version: u64,
    pub digest: String,
    pub admission_digest: String,
    pub model_spec_digest: String,
    pub weight_digest: String,
    pub operation: Operation,
}

impl OperationEnvelope {
    pub fn unsealed(session_id: RequestId, operation: Operation) -> Self {
        Self {
            session_id,
            epoch: 0,
            op_id: 0,
            base_version: 0,
            digest: String::new(),
            admission_digest: String::new(),
            model_spec_digest: String::new(),
            weight_digest: String::new(),
            operation,
        }
    }

    pub const fn kind(&self) -> OperationKind {
        self.operation.kind()
    }

    pub const fn operation_type(&self) -> OperationType {
        self.operation.operation_type()
    }

    pub fn seal(&mut self, epoch: u64, op_id: u64, base_version: u64) {
        self.epoch = epoch;
        self.op_id = op_id;
        self.base_version = base_version;
        self.refresh_digest();
    }

    pub fn refresh_digest(&mut self) {
        self.digest = self.payload_digest(EXECUTION_PROTOCOL_VERSION);
    }

    pub fn validate(&self, protocol_version: u16) -> anyhow::Result<()> {
        anyhow::ensure!(
            protocol_version == EXECUTION_PROTOCOL_VERSION,
            "unsupported execution protocol version {protocol_version}"
        );
        anyhow::ensure!(self.epoch > 0, "operation epoch must be positive");
        anyhow::ensure!(self.op_id > 0, "operation id must be positive");
        anyhow::ensure!(
            is_digest(&self.admission_digest),
            "operation admission digest is invalid"
        );
        anyhow::ensure!(
            is_digest(&self.model_spec_digest),
            "operation model spec digest is invalid"
        );
        anyhow::ensure!(
            is_digest(&self.weight_digest),
            "operation weight digest is invalid"
        );
        self.operation.validate()?;
        anyhow::ensure!(is_digest(&self.digest), "operation digest is invalid");
        anyhow::ensure!(
            self.digest == self.payload_digest(protocol_version),
            "operation digest mismatch for session {}",
            self.session_id.0
        );
        Ok(())
    }

    pub fn payload_digest(&self, protocol_version: u16) -> String {
        let mut digest = CanonicalDigest::new(b"uniserve-operation-v3\0", protocol_version);
        digest.u64(self.session_id.0);
        digest.u64(self.epoch);
        digest.u64(self.op_id);
        digest.u64(self.base_version);
        digest.string(&self.admission_digest);
        digest.string(&self.model_spec_digest);
        digest.string(&self.weight_digest);
        digest.operation(&self.operation);
        digest.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    pub protocol_version: u16,
    pub step_id: u64,
    pub admissions: Vec<Admission>,
    pub projections: Vec<SessionProjection>,
    pub operations: Vec<OperationEnvelope>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionProjection {
    pub session_id: RequestId,
    pub epoch: u64,
    pub version: u64,
    pub last_op_id: u64,
    pub admission_digest: String,
    pub source_digest: String,
    pub last_sampled_token: Option<u32>,
}

impl SessionProjection {
    pub fn validate_for(&self, operation: &OperationEnvelope) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.session_id == operation.session_id,
            "session projection targets a different session"
        );
        anyhow::ensure!(
            self.epoch == operation.epoch,
            "session projection epoch does not match operation"
        );
        anyhow::ensure!(
            self.version == operation.base_version,
            "session projection version does not match operation base version"
        );
        anyhow::ensure!(
            self.version > 0 && self.last_op_id > 0,
            "session projection must identify committed state"
        );
        anyhow::ensure!(
            self.admission_digest == operation.admission_digest,
            "session projection admission identity does not match operation"
        );
        anyhow::ensure!(
            is_digest(&self.admission_digest),
            "session projection admission digest is invalid"
        );
        anyhow::ensure!(
            is_digest(&self.source_digest),
            "session projection source digest is invalid"
        );
        Ok(())
    }
}

impl Batch {
    pub fn new(
        step_id: u64,
        admissions: Vec<Admission>,
        operations: Vec<OperationEnvelope>,
    ) -> Self {
        Self {
            protocol_version: EXECUTION_PROTOCOL_VERSION,
            step_id,
            admissions,
            projections: Vec::new(),
            operations,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.protocol_version == EXECUTION_PROTOCOL_VERSION,
            "unsupported execution protocol version {}",
            self.protocol_version
        );
        anyhow::ensure!(
            !self.operations.is_empty(),
            "execution batch must contain an operation"
        );
        let mut sessions = HashSet::with_capacity(self.operations.len());
        for operation in &self.operations {
            operation.validate(self.protocol_version)?;
            anyhow::ensure!(
                sessions.insert(operation.session_id),
                "batch contains multiple operations for session {}",
                operation.session_id.0
            );
        }
        let mut admitted = HashSet::with_capacity(self.admissions.len());
        for admission in &self.admissions {
            admission.validate(self.protocol_version)?;
            anyhow::ensure!(
                admitted.insert(admission.session_id),
                "batch contains duplicate admission for session {}",
                admission.session_id.0
            );
            let operation = self
                .operations
                .iter()
                .find(|operation| operation.session_id == admission.session_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "batch admits session {} without an operation",
                        admission.session_id.0
                    )
                })?;
            anyhow::ensure!(
                operation.admission_digest == admission.digest,
                "operation admission digest mismatch for session {}",
                admission.session_id.0
            );
        }
        let mut projected = HashSet::with_capacity(self.projections.len());
        for projection in &self.projections {
            anyhow::ensure!(
                projected.insert(projection.session_id),
                "batch contains duplicate projection for session {}",
                projection.session_id.0
            );
            let operation = self
                .operations
                .iter()
                .find(|operation| operation.session_id == projection.session_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "batch projects session {} without an operation",
                        projection.session_id.0
                    )
                })?;
            projection.validate_for(operation)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob(pub u32, pub f32, pub u32);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SequenceEffect {
    pub sampled_token_ids: Vec<u32>,
    pub sampled_logprob: Option<f32>,
    pub top_logprobs: Vec<TokenLogprob>,
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    pub accepted_draft_tokens: Option<u32>,
    pub kv_tokens: Option<u32>,
    pub published_logits: Option<PublishedProduct>,
    pub published_kv: Option<PublishedKv>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceDelta {
    pub effect: SequenceEffect,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlowDelta {
    pub steps_completed: u16,
    pub done: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodeDelta {
    pub product_handle: u64,
    pub kv_tokens: u32,
    pub image_size: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub png_base64: String,
    pub height: u32,
    pub width: u32,
    pub handle: u64,
    pub locator: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum MaterializedProduct {
    Image(ImageArtifact),
    Published(PublishedProduct),
    Frame { count: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterializeDelta {
    pub product: MaterializedProduct,
    pub kv_tokens: Option<u32>,
    pub sequence: Option<SequenceEffect>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransferDelta {
    pub product: Option<PublishedProduct>,
    pub kv_tokens: Option<u32>,
    pub sequence: Option<SequenceEffect>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ResultDelta {
    Sequence(SequenceDelta),
    Flow(FlowDelta),
    Encode(EncodeDelta),
    Materialize(MaterializeDelta),
    Transfer(TransferDelta),
}

impl ResultDelta {
    pub const fn kind(&self) -> OperationKind {
        match self {
            Self::Sequence(_) => OperationKind::Sequence,
            Self::Flow(_) => OperationKind::Flow,
            Self::Encode(_) => OperationKind::Encode,
            Self::Materialize(_) => OperationKind::Materialize,
            Self::Transfer(_) => OperationKind::Transfer,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationResult {
    pub session_id: RequestId,
    pub epoch: u64,
    pub op_id: u64,
    pub base_version: u64,
    pub result_version: u64,
    pub delta: ResultDelta,
}

impl OperationResult {
    pub fn validate_for(&self, operation: &OperationEnvelope) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.session_id == operation.session_id,
            "result session does not match operation"
        );
        anyhow::ensure!(
            self.epoch == operation.epoch,
            "result epoch does not match operation"
        );
        anyhow::ensure!(
            self.op_id == operation.op_id,
            "result identity does not match operation"
        );
        anyhow::ensure!(
            self.base_version == operation.base_version,
            "result base version does not match operation"
        );
        anyhow::ensure!(
            self.result_version == operation.base_version.saturating_add(1),
            "result version does not advance exactly once"
        );
        anyhow::ensure!(
            self.delta.kind() == operation.kind(),
            "result delta variant does not match operation"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub step_id: u64,
    pub operations: Vec<OperationResult>,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<WorkerForwardStats>,
}

impl ExecutionResult {
    pub fn validate_for(&self, batch: &Batch) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.step_id == batch.step_id,
            "result step does not match batch"
        );
        anyhow::ensure!(
            self.operations.len() == batch.operations.len(),
            "result operation count does not match batch"
        );
        for (result, operation) in self.operations.iter().zip(&batch.operations) {
            result.validate_for(operation)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerForwardStats {
    pub mode_counts: BTreeMap<String, u64>,
    pub mode_tokens: BTreeMap<String, u64>,
    pub mode_us: BTreeMap<String, u64>,
    pub component_us: BTreeMap<String, u64>,
    pub attention_launches: u64,
    pub attention_us: u64,
    pub attention_backend_counts: BTreeMap<String, u64>,
    pub cuda_graph_captures: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_misses: u64,
    pub cuda_graph_fallbacks: u64,
    pub cuda_graph_unpadded_tokens: u64,
    pub cuda_graph_padded_tokens: u64,
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    pub text_decode_token_relay_hits: u64,
    pub text_decode_token_relay_misses: u64,
    pub text_decode_position_relay_hits: u64,
    pub text_decode_position_relay_misses: u64,
    pub flashinfer_decode_plan_calls: u64,
    pub flashinfer_decode_plan_reuses: u64,
    pub flashinfer_decode_plan_rows: u64,
    pub flashinfer_decode_plan_indices: u64,
    pub flashinfer_decode_graph_plan_calls: u64,
    pub flashinfer_decode_graph_plan_reuses: u64,
    pub spec_verify_rows: u64,
    pub spec_verify_draft_tokens: u64,
    pub spec_verify_accepted_tokens: u64,
    pub spec_verify_rejected_tokens: u64,
    pub spec_verify_committed_tokens: u64,
    pub spec_verify_path_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AdapterMode {
    #[default]
    None,
    EngineWide,
    PerRequest,
    MultiAdapter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExecutionConstraints {
    pub max_batch_operations: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCaps {
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub scratch_capacity_tokens: u64,
    pub supported_operation_types: Vec<OperationType>,
    pub max_latent_size: u32,
    pub latent_downsample: u32,
    pub max_vae_grid_tokens: u32,
    pub max_vit_grid_tokens: u32,
    pub commit_marker_tokens: u32,
    pub gen_rope_advance: u32,
    pub max_cfg_branches: u32,
    pub bytes_per_token: u64,
    pub groups: Vec<KvCacheGroupSpec>,
    pub kv_dtype: String,
    pub model_dtype: String,
    pub attention_backend: String,
    pub quantization: Option<String>,
    pub rank: RankInfo,
    pub pipeline_depth: u32,
    pub encoder_cache_budget: u32,
    pub supported_controls: Vec<RequestKind>,
    pub adapter_mode: AdapterMode,
    pub execution_constraints: ExecutionConstraints,
    pub resource_classes: Vec<ResourceClass>,
    pub model_spec_digest: String,
    pub weight_digest: String,
    pub restored_sessions: Vec<RequestId>,
}

impl Default for EngineCaps {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_blocks: 4096,
            num_layers: 28,
            scratch_capacity_tokens: 1 << 20,
            supported_operation_types: vec![OperationType::SequenceExtend],
            max_latent_size: 0,
            latent_downsample: 1,
            max_vae_grid_tokens: 0,
            max_vit_grid_tokens: 0,
            commit_marker_tokens: 2,
            gen_rope_advance: 2,
            max_cfg_branches: 3,
            bytes_per_token: 57_344,
            groups: Vec::new(),
            kv_dtype: "bfloat16".into(),
            model_dtype: "bfloat16".into(),
            attention_backend: "flashinfer".into(),
            quantization: None,
            rank: RankInfo::default(),
            pipeline_depth: 1,
            encoder_cache_budget: 0,
            supported_controls: Vec::new(),
            adapter_mode: AdapterMode::None,
            execution_constraints: ExecutionConstraints::default(),
            resource_classes: Vec::new(),
            model_spec_digest: String::new(),
            weight_digest: String::new(),
            restored_sessions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    GetCapabilities,
    Execute,
    DropSession,
    Shutdown,
    CopyKv,
    LoadAdapter,
    UnloadAdapter,
    ReleaseProducts,
    ResetPrefixCache,
    GetMetrics,
    GetPressure,
    SnapshotSession,
    RestoreSession,
}

impl RequestKind {
    pub const ALL: [Self; 13] = [
        Self::GetCapabilities,
        Self::Execute,
        Self::DropSession,
        Self::Shutdown,
        Self::CopyKv,
        Self::LoadAdapter,
        Self::UnloadAdapter,
        Self::ReleaseProducts,
        Self::ResetPrefixCache,
        Self::GetMetrics,
        Self::GetPressure,
        Self::SnapshotSession,
        Self::RestoreSession,
    ];

    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::GetCapabilities => "get_capabilities",
            Self::Execute => "execute",
            Self::DropSession => "drop_session",
            Self::Shutdown => "shutdown",
            Self::CopyKv => "copy_kv",
            Self::LoadAdapter => "load_adapter",
            Self::UnloadAdapter => "unload_adapter",
            Self::ReleaseProducts => "release_products",
            Self::ResetPrefixCache => "reset_prefix_cache",
            Self::GetMetrics => "get_metrics",
            Self::GetPressure => "get_pressure",
            Self::SnapshotSession => "snapshot_session",
            Self::RestoreSession => "restore_session",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRef {
    pub session_id: RequestId,
    pub epoch: u64,
    pub version: u64,
    pub digest: String,
    pub locator: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub kind: RequestKind,
    pub call_id: Option<u64>,
    pub batch: Option<Batch>,
    pub session_id: Option<RequestId>,
    pub copies: Option<Vec<(BlockId, BlockId)>>,
    pub adapter_id: Option<u32>,
    pub adapter_path: Option<String>,
    pub product_handles: Option<Vec<u64>>,
    pub snapshot: Option<SnapshotRef>,
}

impl WorkerRequest {
    fn bare(kind: RequestKind) -> Self {
        Self {
            kind,
            call_id: None,
            batch: None,
            session_id: None,
            copies: None,
            adapter_id: None,
            adapter_path: None,
            product_handles: None,
            snapshot: None,
        }
    }

    pub fn get_capabilities() -> Self {
        Self::bare(RequestKind::GetCapabilities)
    }
    pub fn execute(batch: Batch) -> Self {
        Self {
            batch: Some(batch),
            ..Self::bare(RequestKind::Execute)
        }
    }
    pub fn drop_session(session_id: RequestId) -> Self {
        Self {
            session_id: Some(session_id),
            ..Self::bare(RequestKind::DropSession)
        }
    }
    pub fn shutdown() -> Self {
        Self::bare(RequestKind::Shutdown)
    }
    pub fn copy_kv(copies: Vec<(BlockId, BlockId)>) -> Self {
        Self {
            copies: Some(copies),
            ..Self::bare(RequestKind::CopyKv)
        }
    }
    pub fn load_adapter(adapter_id: u32, adapter_path: String) -> Self {
        Self {
            adapter_id: Some(adapter_id),
            adapter_path: Some(adapter_path),
            ..Self::bare(RequestKind::LoadAdapter)
        }
    }
    pub fn unload_adapter(adapter_id: u32) -> Self {
        Self {
            adapter_id: Some(adapter_id),
            ..Self::bare(RequestKind::UnloadAdapter)
        }
    }
    pub fn release_products(product_handles: Vec<u64>) -> Self {
        Self {
            product_handles: Some(product_handles),
            ..Self::bare(RequestKind::ReleaseProducts)
        }
    }
    pub fn reset_prefix_cache() -> Self {
        Self::bare(RequestKind::ResetPrefixCache)
    }
    pub fn get_metrics() -> Self {
        Self::bare(RequestKind::GetMetrics)
    }
    pub fn get_pressure() -> Self {
        Self::bare(RequestKind::GetPressure)
    }
    pub fn snapshot_session(session_id: RequestId) -> Self {
        Self {
            session_id: Some(session_id),
            ..Self::bare(RequestKind::SnapshotSession)
        }
    }
    pub fn restore_session(snapshot: SnapshotRef) -> Self {
        Self {
            snapshot: Some(snapshot),
            ..Self::bare(RequestKind::RestoreSession)
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub executes: u64,
    pub operations_total: u64,
    pub exec_us_total: u64,
    pub last_exec_us: u64,
    pub operation_counts: BTreeMap<String, u64>,
    pub operation_us: BTreeMap<String, u64>,
    pub control_ok: BTreeMap<String, u64>,
    pub control_err: BTreeMap<String, u64>,
    pub error_counts: BTreeMap<String, u64>,
    pub cuda_graph_captures: u64,
    pub cuda_graph_replays: u64,
    pub cuda_graph_misses: u64,
    pub cuda_graph_fallbacks: u64,
    pub cuda_graph_unpadded_tokens: u64,
    pub cuda_graph_padded_tokens: u64,
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    pub forward: Option<WorkerForwardStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseKind {
    Capabilities,
    Result,
    Ok,
    Error,
    Metrics,
    Pressure,
    Snapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorOperationIdentity {
    pub session_id: u64,
    pub epoch: u64,
    pub op_id: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub kind: ResponseKind,
    pub call_id: Option<u64>,
    pub capabilities: Option<EngineCaps>,
    pub result: Option<ExecutionResult>,
    pub metrics: Option<WorkerMetrics>,
    pub pressure: Option<Vec<ResourcePressure>>,
    pub message: Option<String>,
    pub code: Option<String>,
    pub retryable: Option<bool>,
    pub fatal: Option<bool>,
    pub phase: Option<String>,
    pub route: Option<String>,
    pub operations: Vec<ErrorOperationIdentity>,
    pub snapshot: Option<SnapshotRef>,
}

impl WorkerResponse {
    pub fn capabilities(capabilities: EngineCaps) -> Self {
        Self {
            kind: ResponseKind::Capabilities,
            call_id: None,
            capabilities: Some(capabilities),
            result: None,
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
        }
    }

    pub fn result(result: ExecutionResult) -> Self {
        Self {
            kind: ResponseKind::Result,
            call_id: None,
            capabilities: None,
            result: Some(result),
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
        }
    }

    pub fn ok() -> Self {
        Self {
            kind: ResponseKind::Ok,
            call_id: None,
            capabilities: None,
            result: None,
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
        }
    }

    pub fn snapshot(snapshot: SnapshotRef) -> Self {
        Self {
            kind: ResponseKind::Snapshot,
            call_id: None,
            capabilities: None,
            result: None,
            metrics: None,
            pressure: None,
            message: None,
            code: None,
            retryable: None,
            fatal: None,
            phase: None,
            route: None,
            operations: Vec::new(),
            snapshot: Some(snapshot),
        }
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

struct CanonicalDigest(Sha256);

impl CanonicalDigest {
    fn new(domain: &[u8], version: u16) -> Self {
        let mut digest = Sha256::new();
        digest.update(domain);
        digest.update(version.to_le_bytes());
        Self(digest)
    }

    fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }
    fn u8(&mut self, value: u8) {
        self.0.update([value]);
    }
    fn u16(&mut self, value: u16) {
        self.0.update(value.to_le_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.0.update(value.to_le_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }
    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }
    fn string(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.0.update(value.as_bytes());
    }
    fn u32s(&mut self, values: impl IntoIterator<Item = u32>) {
        let values: Vec<_> = values.into_iter().collect();
        self.u64(values.len() as u64);
        for value in values {
            self.u32(value);
        }
    }
    fn option<T>(&mut self, value: Option<T>, encode: impl FnOnce(&mut Self, T)) {
        match value {
            Some(value) => {
                self.u8(1);
                encode(self, value);
            }
            None => self.u8(0),
        }
    }

    fn sampling(&mut self, value: &SamplingParams) {
        self.f32(value.temperature);
        self.u32(value.top_k);
        self.f32(value.top_p);
        self.bool(value.ignore_eos);
        self.option(value.seed, Self::u64);
        self.f32(value.min_p);
        self.f32(value.repetition_penalty);
        self.f32(value.frequency_penalty);
        self.f32(value.presence_penalty);
        self.u64(value.logit_bias.len() as u64);
        for (token, bias) in &value.logit_bias {
            self.u32(*token);
            self.f32(*bias);
        }
        self.u64(value.min_tokens as u64);
        self.bool(value.return_logprobs);
        self.u32(value.n_logprobs);
        self.bool(value.return_prompt_logprobs);
        self.u32(value.n_prompt_logprobs);
        self.u32s(value.logprob_token_ids.iter().copied());
        self.u64(value.bad_words_ids.len() as u64);
        for tokens in &value.bad_words_ids {
            self.u32s(tokens.iter().copied());
        }
        self.option(value.allowed_token_ids.as_deref(), |digest, tokens| {
            digest.u32s(tokens.iter().copied())
        });
    }

    fn image(&mut self, value: &ImageParams) {
        self.u16(value.steps);
        self.f32(value.cfg_text_scale);
        self.f32(value.cfg_img_scale);
        self.string(&value.cfg_renorm_type);
        self.f32(value.cfg_renorm_min);
        self.f32(value.cfg_interval.0);
        self.f32(value.cfg_interval.1);
        self.f32(value.timestep_shift);
        self.u32(value.height);
        self.u32(value.width);
        self.option(value.seed, Self::u64);
        self.string(&value.negative_prompt);
        self.u16(value.max_images);
        self.u64(value.image_prompts.len() as u64);
        for prompt in &value.image_prompts {
            self.string(prompt);
        }
        self.bool(value.retain_images);
    }

    fn lease(&mut self, value: &KvLeaseDelta) {
        self.u32(value.group_id);
        self.u32s(value.new_blocks.iter().map(|block| block.0));
    }
    fn policy(&mut self, value: &TokenPolicy) {
        self.u32s(value.allowed_tokens.iter().copied());
        self.u32s(value.suppress_tokens.iter().copied());
        self.u32s(value.recent_tokens.iter().copied());
        self.bool(value.publish_kv);
        self.u32s(value.publish_kv_on_tokens.iter().copied());
    }
    fn published(&mut self, value: &PublishedProduct) {
        self.u64(value.handle);
    }
    fn published_kv(&mut self, value: &PublishedKv) {
        self.u64(value.handle);
        self.u64(value.source_version);
        self.u32(value.kv_tokens);
        self.u32s(value.block_ids.iter().map(|block| block.0));
        self.u32(value.group_id);
        self.u32(value.position);
    }

    fn operation(&mut self, value: &Operation) {
        match value {
            Operation::Sequence(operation) => {
                self.u8(0);
                self.u8(match operation.mode {
                    SequenceMode::Extend => 0,
                    SequenceMode::Decode => 1,
                    SequenceMode::Verify => 2,
                    SequenceMode::Sample => 3,
                });
                self.lease(&operation.lease);
                self.u32(operation.position.0);
                self.u32(operation.position.1);
                self.policy(&operation.policy);
                match &operation.input {
                    SequenceInput::Tokens(input) => {
                        self.u8(0);
                        self.u32s(input.token_ids.iter().copied());
                        self.u8(match input.source {
                            TokenSource::Wire => 0,
                            TokenSource::LastSampled => 1,
                        });
                        self.u32s(input.draft_token_ids.iter().copied());
                        self.bool(input.return_all_logits);
                    }
                    SequenceInput::PublishedLogits(product) => {
                        self.u8(1);
                        self.published(product);
                    }
                }
            }
            Operation::Flow(operation) => {
                self.u8(1);
                self.u64(operation.latent_handle);
                self.u32(operation.position);
                self.u16(operation.start_step);
                self.u16(operation.step_count);
                self.u32(operation.conditioning_position);
                self.option(operation.conditioning.as_ref(), |digest, conditioning| {
                    digest.published_kv(conditioning)
                });
                self.u8(operation.guidance.branch_count);
                self.f32(operation.guidance.text_scale);
                self.f32(operation.guidance.image_scale);
                self.string(&operation.guidance.renorm_type);
                self.f32(operation.guidance.renorm_min);
                self.f32(operation.guidance.interval.0);
                self.f32(operation.guidance.interval.1);
                self.string(&operation.image_prompt);
            }
            Operation::Encode(operation) => {
                self.u8(2);
                self.u8(match operation.kind {
                    EncodeKind::Vision => 0,
                    EncodeKind::Latent => 1,
                });
                self.lease(&operation.lease);
                self.u32(operation.position.0);
                self.u32(operation.position.1);
                self.u32(operation.conditioning_position);
                match &operation.input {
                    EncodeInput::InlineImage {
                        base64,
                        content_hash,
                    } => {
                        self.u8(0);
                        self.string(base64);
                        self.u64(*content_hash);
                    }
                    EncodeInput::StagedProduct {
                        handle,
                        content_hash,
                    } => {
                        self.u8(1);
                        self.u64(*handle);
                        self.u64(*content_hash);
                    }
                    EncodeInput::CachedProduct { content_hash } => {
                        self.u8(2);
                        self.u64(*content_hash);
                    }
                }
            }
            Operation::Materialize(operation) => {
                self.u8(3);
                self.u8(match operation.kind {
                    MaterializeKind::Image => 0,
                    MaterializeKind::Frame => 1,
                });
                self.lease(&operation.lease);
                self.u32(operation.position);
                self.u32(operation.conditioning_position);
                self.policy(&operation.policy);
                match &operation.input {
                    MaterializeInput::Latent { handle } => {
                        self.u8(0);
                        self.u64(*handle);
                    }
                    MaterializeInput::Published(product) => {
                        self.u8(1);
                        self.published(product);
                    }
                }
            }
            Operation::Transfer(operation) => {
                self.u8(4);
                self.u8(match operation.kind {
                    TransferKind::Product => 0,
                    TransferKind::Kv => 1,
                });
                self.lease(&operation.lease);
                self.u32(operation.position);
                self.u32(operation.conditioning_position);
                self.policy(&operation.policy);
                self.published(&operation.source);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "ab".repeat(32)
    }

    #[test]
    fn operation_union_is_closed_and_versioned() {
        assert_eq!(EXECUTION_PROTOCOL_VERSION, 3);
        assert_eq!(OperationKind::ALL.len(), 5);
        let mut operation = OperationEnvelope::unsealed(
            RequestId(7),
            Operation::Sequence(SequenceOperation {
                mode: SequenceMode::Decode,
                lease: KvLeaseDelta::default(),
                position: (3, 4),
                policy: TokenPolicy::default(),
                input: SequenceInput::Tokens(TokenInput {
                    token_ids: vec![9],
                    source: TokenSource::Wire,
                    draft_token_ids: Vec::new(),
                    return_all_logits: false,
                }),
            }),
        );
        operation.admission_digest = digest();
        operation.model_spec_digest = digest();
        operation.weight_digest = digest();
        operation.seal(2, 11, 4);
        operation.validate(EXECUTION_PROTOCOL_VERSION).unwrap();
    }

    #[test]
    fn result_variant_must_match_operation() {
        let mut operation = OperationEnvelope::unsealed(
            RequestId(1),
            Operation::Flow(FlowOperation {
                latent_handle: 1,
                position: 0,
                start_step: 0,
                step_count: 1,
                conditioning_position: 0,
                conditioning: None,
                guidance: Guidance {
                    branch_count: 1,
                    text_scale: 1.0,
                    image_scale: 1.0,
                    renorm_type: "none".into(),
                    renorm_min: 0.0,
                    interval: (0.0, 1.0),
                },
                image_prompt: String::new(),
            }),
        );
        operation.admission_digest = digest();
        operation.model_spec_digest = digest();
        operation.weight_digest = digest();
        operation.seal(1, 2, 0);
        let result = OperationResult {
            session_id: RequestId(1),
            epoch: 1,
            op_id: 2,
            base_version: 0,
            result_version: 1,
            delta: ResultDelta::Sequence(SequenceDelta {
                effect: SequenceEffect::default(),
            }),
        };
        assert!(result.validate_for(&operation).is_err());
    }

    #[test]
    fn operation_identity_excludes_runtime_locator_addresses() {
        let operation = |locator: &str| {
            let mut envelope = OperationEnvelope::unsealed(
                RequestId(5),
                Operation::Transfer(TransferOperation {
                    kind: TransferKind::Product,
                    lease: KvLeaseDelta::default(),
                    position: 8,
                    conditioning_position: 8,
                    policy: TokenPolicy::default(),
                    source: PublishedProduct {
                        handle: 17,
                        locator: locator.into(),
                    },
                }),
            );
            envelope.admission_digest = digest();
            envelope.model_spec_digest = digest();
            envelope.weight_digest = digest();
            envelope.seal(3, 9, 2);
            envelope
        };

        let before_recovery = operation("physical-locator-a");
        let after_recovery = operation("physical-locator-b");

        assert_eq!(before_recovery.digest, after_recovery.digest);
        before_recovery
            .validate(EXECUTION_PROTOCOL_VERSION)
            .unwrap();
        after_recovery.validate(EXECUTION_PROTOCOL_VERSION).unwrap();
    }
}
