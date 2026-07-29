//! Worker execution protocol.
//!
//! The scheduler and worker exchange four cross-layer records — [`Operation`],
//! [`VersionRef`], [`ProductRef`], and [`CompletionRecord`] — plus a request
//! [`Control`] command. Every operation names one closed [`Work`] variant, one
//! exact parent version, and its declared input and output products. The worker
//! returns exactly one [`CompletionRecord`] per operation. Two host-computed
//! digests fix identity: an operation [`Operation::plan_digest`] over immutable
//! registration fields, and a [`CompletionRecord::compute_semantic_digest`] over
//! the selected result.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use uniserve_core::{BlockId, ImageParams, KvCacheGroupSpec, RankInfo, RequestId, SamplingParams};

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

/// A lowercase 64-character SHA-256 digest string. Both protocol digests and
/// the model and route identities use this canonical form.
pub type Digest = String;

// ---------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------

/// `(authority_id, session_id, epoch)`. The epoch advances whenever an admitted
/// identity is reused, so no operation or product reference aliases across
/// requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    pub authority_id: u64,
    pub session_id: RequestId,
    pub epoch: u64,
}

impl RequestKey {
    pub const fn new(authority_id: u64, session_id: RequestId, epoch: u64) -> Self {
        Self {
            authority_id,
            session_id,
            epoch,
        }
    }
}

/// Unique within one scheduler-authority lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OpId(pub u64);

/// Route identity assigned by the scheduler for capability negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteId(pub u32);

// ---------------------------------------------------------------------------
// Closed `Work` algebra
// ---------------------------------------------------------------------------

/// State effect and role of a token operation's device work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenMode {
    Extend,
    Decode,
    Verify,
}

/// Encoder role producing an immutable auxiliary feature product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodeMode {
    Vision,
    Latent,
}

/// Movement role for an immutable product or a committed KV view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferMode {
    Product,
    KvPublish,
    KvInstall,
}

/// Generation-lineage role for a Gen route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenMode {
    Transition,
    Flow,
}

/// The single closed work algebra. The state effect and role of each variant
/// are fixed; sampling is device postprocessing inside `Token(*)`, not a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Work {
    Token(TokenMode),
    Draft,
    Encode(EncodeMode),
    Transfer(TransferMode),
    Gen(GenMode),
    Materialize,
}

/// The flat exhaustive tag for one [`Work`] leaf, used on the wire and in
/// capability negotiation. Declaration order is the canonical index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum WorkVariant {
    TokenExtend = 0,
    TokenDecode = 1,
    TokenVerify = 2,
    Draft = 3,
    EncodeVision = 4,
    EncodeLatent = 5,
    TransferProduct = 6,
    TransferKvPublish = 7,
    TransferKvInstall = 8,
    GenTransition = 9,
    GenFlow = 10,
    Materialize = 11,
}

impl WorkVariant {
    pub const ALL: [Self; 12] = [
        Self::TokenExtend,
        Self::TokenDecode,
        Self::TokenVerify,
        Self::Draft,
        Self::EncodeVision,
        Self::EncodeLatent,
        Self::TransferProduct,
        Self::TransferKvPublish,
        Self::TransferKvInstall,
        Self::GenTransition,
        Self::GenFlow,
        Self::Materialize,
    ];

    /// Whether a variant advances the authoritative request lineage.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::TokenExtend
                | Self::TokenDecode
                | Self::TokenVerify
                | Self::GenTransition
                | Self::GenFlow
        )
    }

    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::TokenExtend => "token_extend",
            Self::TokenDecode => "token_decode",
            Self::TokenVerify => "token_verify",
            Self::Draft => "draft",
            Self::EncodeVision => "encode_vision",
            Self::EncodeLatent => "encode_latent",
            Self::TransferProduct => "transfer_product",
            Self::TransferKvPublish => "transfer_kv_publish",
            Self::TransferKvInstall => "transfer_kv_install",
            Self::GenTransition => "gen_transition",
            Self::GenFlow => "gen_flow",
            Self::Materialize => "materialize",
        }
    }
}

impl Work {
    pub const fn variant(self) -> WorkVariant {
        match self {
            Self::Token(TokenMode::Extend) => WorkVariant::TokenExtend,
            Self::Token(TokenMode::Decode) => WorkVariant::TokenDecode,
            Self::Token(TokenMode::Verify) => WorkVariant::TokenVerify,
            Self::Draft => WorkVariant::Draft,
            Self::Encode(EncodeMode::Vision) => WorkVariant::EncodeVision,
            Self::Encode(EncodeMode::Latent) => WorkVariant::EncodeLatent,
            Self::Transfer(TransferMode::Product) => WorkVariant::TransferProduct,
            Self::Transfer(TransferMode::KvPublish) => WorkVariant::TransferKvPublish,
            Self::Transfer(TransferMode::KvInstall) => WorkVariant::TransferKvInstall,
            Self::Gen(GenMode::Transition) => WorkVariant::GenTransition,
            Self::Gen(GenMode::Flow) => WorkVariant::GenFlow,
            Self::Materialize => WorkVariant::Materialize,
        }
    }

    pub const fn from_variant(variant: WorkVariant) -> Self {
        match variant {
            WorkVariant::TokenExtend => Self::Token(TokenMode::Extend),
            WorkVariant::TokenDecode => Self::Token(TokenMode::Decode),
            WorkVariant::TokenVerify => Self::Token(TokenMode::Verify),
            WorkVariant::Draft => Self::Draft,
            WorkVariant::EncodeVision => Self::Encode(EncodeMode::Vision),
            WorkVariant::EncodeLatent => Self::Encode(EncodeMode::Latent),
            WorkVariant::TransferProduct => Self::Transfer(TransferMode::Product),
            WorkVariant::TransferKvPublish => Self::Transfer(TransferMode::KvPublish),
            WorkVariant::TransferKvInstall => Self::Transfer(TransferMode::KvInstall),
            WorkVariant::GenTransition => Self::Gen(GenMode::Transition),
            WorkVariant::GenFlow => Self::Gen(GenMode::Flow),
            WorkVariant::Materialize => Self::Materialize,
        }
    }

    /// The canonical state effect fixed by the work table.
    pub const fn advances_state(self) -> bool {
        self.variant().advances_state()
    }
}

// ---------------------------------------------------------------------------
// Product references and bounded shapes
// ---------------------------------------------------------------------------

/// The role a product plays for its consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ProductKind {
    Token = 0,
    Logprob = 1,
    Draft = 2,
    VisionFeature = 3,
    LatentFeature = 4,
    Kv = 5,
    Latent = 6,
    Artifact = 7,
    Completion = 8,
}

/// The worker store family that backs a product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum StorageClass {
    DeviceTensor = 0,
    PagedKv = 1,
    LatentArena = 2,
    HostStaging = 3,
    CompletionArena = 4,
}

/// Element type of a product's backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DType {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    #[serde(rename = "bf16")]
    BF16 = 6,
    F32 = 7,
}

/// One dimension of a bounded shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DimBound {
    /// A host-static extent.
    Static(u32),
    /// The single device-actual axis, bounded by this fixed maximum.
    Device { max: u32 },
}

/// A shape that is host-static except for at most one device-actual axis, which
/// carries a fixed maximum. A product reference never carries an unbounded shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ShapeBound {
    pub dims: Vec<DimBound>,
}

impl ShapeBound {
    pub fn validate(&self) -> anyhow::Result<()> {
        let device_dims = self
            .dims
            .iter()
            .filter(|dim| matches!(dim, DimBound::Device { .. }))
            .count();
        anyhow::ensure!(
            device_dims <= 1,
            "a shape bound carries more than one device-actual dimension"
        );
        Ok(())
    }
}

/// The state points a product spans, rooted at `base_point`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct PointRange {
    pub base_point: u32,
    pub max_points: u32,
}

/// A generation-tagged reference to a declared device or host product. Physical
/// slots, tensors, and events stay worker-local and never appear here.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductRef {
    pub request_key: RequestKey,
    pub producer_op_id: OpId,
    pub output_index: u16,
    pub generation: u32,
    pub kind: ProductKind,
    pub storage_class: StorageClass,
    pub dtype: DType,
    pub shape_bound: ShapeBound,
    pub point_range: PointRange,
}

impl ProductRef {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.shape_bound.validate()
    }
}

// ---------------------------------------------------------------------------
// Version references
// ---------------------------------------------------------------------------

/// One exact state point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Point {
    /// A host-observed state point named by its index and semantic digest.
    Fixed {
        point_index: u32,
        semantic_digest: Digest,
    },
    /// A device-selected point a successor may consume before host observation.
    Device {
        selected_point: ProductRef,
        producer_plan_digest: Digest,
    },
}

/// Names one exact state point of a producer operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionRef {
    pub request_key: RequestKey,
    pub producer_op_id: OpId,
    pub point: Point,
}

impl VersionRef {
    /// The admission root: point zero of the request-admission operation.
    pub fn admission_root(
        request_key: RequestKey,
        producer_op_id: OpId,
        semantic_digest: Digest,
    ) -> Self {
        Self {
            request_key,
            producer_op_id,
            point: Point::Fixed {
                point_index: 0,
                semantic_digest,
            },
        }
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self.point, Point::Fixed { .. })
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match &self.point {
            Point::Fixed {
                semantic_digest, ..
            } => anyhow::ensure!(
                is_digest(semantic_digest),
                "fixed version reference has an invalid semantic digest"
            ),
            Point::Device {
                selected_point,
                producer_plan_digest,
            } => {
                selected_point.validate()?;
                anyhow::ensure!(
                    is_digest(producer_plan_digest),
                    "device version reference has an invalid producer plan digest"
                );
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

/// Whether an operation belongs to the understanding or generation branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Domain {
    Und = 0,
    Gen = 1,
}

/// Hard resource maxima the scheduler reserves before an operation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
    pub max_points: u32,
    pub max_tokens: u32,
    pub max_kv_pages: u32,
    pub max_latent_bytes: u64,
    pub max_completion_bytes: u64,
    pub max_transfer_bytes: u64,
}

/// How the common sampler consumes deterministic random draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DrawLayout {
    TargetSampling = 0,
    SpeculativeProposal = 1,
    FlowNoise = 2,
}

/// Deterministic random-draw coordinates for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rng {
    pub seed: u64,
    pub semantic_index_base: u64,
    pub draw_layout: DrawLayout,
}

/// One immutable unit of registered work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub parent: VersionRef,
    pub work: Work,
    pub route: RouteId,
    pub domain: Domain,
    pub advances_state: bool,
    pub bounds: Bounds,
    pub inputs: Vec<ProductRef>,
    pub outputs: Vec<ProductRef>,
    /// The KV blocks this operation appends to its session's KV view as the
    /// sequence grows (empty when this step adds no block). The scheduler's block
    /// manager allocates them; the worker grows its KV entry by exactly these
    /// blocks during registration.
    pub new_kv_blocks: Vec<BlockId>,
    pub predicate: Option<ProductRef>,
    pub rng: Option<Rng>,
    pub control_seq: u64,
    pub plan_digest: Digest,
}

impl Operation {
    /// Build an operation and fill in its plan digest.
    #[allow(clippy::too_many_arguments)]
    pub fn registered(
        request_key: RequestKey,
        op_id: OpId,
        parent: VersionRef,
        work: Work,
        route: RouteId,
        domain: Domain,
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        new_kv_blocks: Vec<BlockId>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    ) -> Self {
        let mut operation = Self {
            request_key,
            op_id,
            parent,
            work,
            route,
            domain,
            advances_state: work.advances_state(),
            bounds,
            inputs,
            outputs,
            new_kv_blocks,
            predicate,
            rng,
            control_seq,
            plan_digest: String::new(),
        };
        operation.plan_digest = operation.compute_plan_digest();
        operation
    }

    /// The immutable registration identity digest.
    pub fn compute_plan_digest(&self) -> Digest {
        let mut digest = CanonicalDigest::new(b"uniserve-operation\0");
        digest.request_key(self.request_key);
        digest.op_id(self.op_id);
        digest.version_ref(&self.parent);
        digest.u8(self.work.variant() as u8);
        digest.u32(self.route.0);
        digest.u8(self.domain as u8);
        digest.bool(self.advances_state);
        digest.bounds(&self.bounds);
        digest.u64(self.inputs.len() as u64);
        for input in &self.inputs {
            digest.product_ref(input);
        }
        digest.u64(self.outputs.len() as u64);
        for output in &self.outputs {
            digest.product_ref(output);
        }
        digest.option(self.predicate.as_ref(), CanonicalDigest::product_ref);
        digest.option(self.rng.as_ref(), CanonicalDigest::rng);
        digest.u32s(self.new_kv_blocks.iter().map(|block| block.0));
        digest.finish()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.op_id.0 > 0, "operation id must be positive");
        anyhow::ensure!(
            self.advances_state == self.work.advances_state(),
            "operation declares an advances_state inconsistent with its work variant"
        );
        self.parent.validate()?;
        self.bounds_are_finite()?;
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        for output in &self.outputs {
            output.validate()?;
            anyhow::ensure!(
                output.request_key == self.request_key && output.producer_op_id == self.op_id,
                "an output product is not owned by its producing operation"
            );
            anyhow::ensure!(
                output.point_range.max_points <= self.bounds.max_points.max(1),
                "an output product exceeds the operation point bound"
            );
            anyhow::ensure!(
                output_indices.insert(output.output_index),
                "operation repeats an output index"
            );
        }
        for input in &self.inputs {
            input.validate()?;
        }
        if let Some(predicate) = &self.predicate {
            predicate.validate()?;
        }
        anyhow::ensure!(
            is_digest(&self.plan_digest),
            "operation plan digest is not a lowercase SHA-256 digest"
        );
        anyhow::ensure!(
            self.plan_digest == self.compute_plan_digest(),
            "operation plan digest does not match its registration fields"
        );
        Ok(())
    }

    fn bounds_are_finite(&self) -> anyhow::Result<()> {
        // Bounds are unsigned integers; the invariant enforced here is that a
        // state-advancing operation can advance by at least one point.
        if self.advances_state {
            anyhow::ensure!(
                self.bounds.max_points >= 1,
                "a state-advancing operation must admit at least one point"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Completion record
// ---------------------------------------------------------------------------

/// Terminal status of one operation's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum OpStatus {
    Ok = 0,
    Predicated = 1,
    Error = 2,
}

/// A deterministic error class for a failed completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ErrorCode {
    InvalidOperation = 0,
    ResourceExhausted = 1,
    ComputeError = 2,
    Cancelled = 3,
    Internal = 4,
}

/// Accounting lengths carried by a completion. These are accounting fields, not
/// state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct LogicalLengths {
    pub token_len: u32,
    pub kv_visible_len: u32,
    pub latent_len: u32,
}

/// The span of tokens an operation contributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TokenSpan {
    pub base: u32,
    pub len: u32,
}

/// Device-observed finish candidates for a token operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct FinishFlags {
    pub eos: bool,
    pub length: bool,
    pub stop: bool,
}

/// Per-operation timing counters. Accounting only; never state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TimingCounters {
    pub queued_us: u64,
    pub device_us: u64,
    pub copy_us: u64,
    pub host_us: u64,
}

/// The fixed-layout record a worker emits once for every operation, after its
/// copy event is query-ready and its pinned fields are validated on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionRecord {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub completion_slot_generation: u32,
    pub status: OpStatus,
    pub selected_point: u32,
    pub logical_lengths: LogicalLengths,
    pub token_span: TokenSpan,
    /// The tokens this operation sampled or selected on device — the semantic
    /// output delta for a `Token(*)` operation, empty for every other work
    /// variant. Bounded by the operation's `max_tokens`.
    pub committed_tokens: Vec<u32>,
    pub finish_flags: FinishFlags,
    pub product_generations: Vec<u32>,
    pub semantic_digest: Digest,
    pub error_code: Option<ErrorCode>,
    pub timing_counters: TimingCounters,
}

impl CompletionRecord {
    /// The selected-result identity digest, host-computed from the ready record.
    /// The committed token values are part of the semantic output delta, so two
    /// different tokens selected at the same span do not share a lineage.
    pub fn compute_semantic_digest(&self, parent_semantic: &str, plan_digest: &str) -> Digest {
        let mut digest = CanonicalDigest::new(b"uniserve-semantic\0");
        digest.string(parent_semantic);
        digest.string(plan_digest);
        digest.u32(self.selected_point);
        digest.u8(self.status as u8);
        digest.u32(self.logical_lengths.token_len);
        digest.u32(self.logical_lengths.kv_visible_len);
        digest.u32(self.logical_lengths.latent_len);
        digest.u32(self.token_span.base);
        digest.u32(self.token_span.len);
        digest.u32s(self.committed_tokens.iter().copied());
        digest.bool(self.finish_flags.eos);
        digest.bool(self.finish_flags.length);
        digest.bool(self.finish_flags.stop);
        digest.u32s(self.product_generations.iter().copied());
        digest.finish()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.op_id.0 > 0, "completion op id must be positive");
        anyhow::ensure!(
            is_digest(&self.semantic_digest),
            "completion semantic digest is not a lowercase SHA-256 digest"
        );
        match self.status {
            OpStatus::Error => anyhow::ensure!(
                self.error_code.is_some(),
                "an error completion must carry an error code"
            ),
            OpStatus::Ok | OpStatus::Predicated => anyhow::ensure!(
                self.error_code.is_none(),
                "a non-error completion must not carry an error code"
            ),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// What to do with a committed selected point's public output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Disposition {
    Publish = 0,
    Retain = 1,
    Discard = 2,
}

/// Why a request lineage is being closed at a cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CloseReason {
    Completed = 0,
    Cancelled = 1,
    Error = 2,
    Preempted = 3,
}

/// The entire request-runtime command channel. Administrative worker commands
/// (drop session, copy KV, adapters, prefix-cache reset, snapshot and restore)
/// are a separate channel and are not part of `Control`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Control {
    /// Commit a selected fixed point and expose public output up to a limit.
    Commit {
        request_key: RequestKey,
        control_seq: u64,
        expected_parent: VersionRef,
        selected: VersionRef,
        public_event_limit: u64,
        disposition: Disposition,
    },
    /// Close a lineage at a fixed cutoff, dominating every uncommitted descendant.
    Close {
        request_key: RequestKey,
        control_seq: u64,
        cutoff: VersionRef,
        reason: CloseReason,
    },
    /// Drop the scheduler's logical ownership of one operation.
    Release {
        request_key: RequestKey,
        op_id: OpId,
    },
}

impl Control {
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Commit { request_key, .. }
            | Self::Close { request_key, .. }
            | Self::Release { request_key, .. } => *request_key,
        }
    }

    /// The variant tag used in the idempotency identity.
    pub const fn variant_index(&self) -> u8 {
        match self {
            Self::Commit { .. } => 0,
            Self::Close { .. } => 1,
            Self::Release { .. } => 2,
        }
    }

    /// The `control_seq` for commit and close; releases carry no sequence.
    pub const fn control_seq(&self) -> Option<u64> {
        match self {
            Self::Commit { control_seq, .. } | Self::Close { control_seq, .. } => {
                Some(*control_seq)
            }
            Self::Release { .. } => None,
        }
    }

    /// The canonical content digest for idempotency: two controls with the same
    /// `(request_key, control_seq, variant)` but different content conflict.
    pub fn content_digest(&self) -> Digest {
        let mut digest = CanonicalDigest::new(b"uniserve-control\0");
        digest.u8(self.variant_index());
        digest.request_key(self.request_key());
        match self {
            Self::Commit {
                control_seq,
                expected_parent,
                selected,
                public_event_limit,
                disposition,
                ..
            } => {
                digest.u64(*control_seq);
                digest.version_ref(expected_parent);
                digest.version_ref(selected);
                digest.u64(*public_event_limit);
                digest.u8(*disposition as u8);
            }
            Self::Close {
                control_seq,
                cutoff,
                reason,
                ..
            } => {
                digest.u64(*control_seq);
                digest.version_ref(cutoff);
                digest.u8(*reason as u8);
            }
            Self::Release { op_id, .. } => {
                digest.op_id(*op_id);
            }
        }
        digest.finish()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Commit {
                expected_parent,
                selected,
                ..
            } => {
                expected_parent.validate()?;
                selected.validate()?;
                anyhow::ensure!(
                    selected.is_fixed(),
                    "a commit control must select a fixed version"
                );
            }
            Self::Close { cutoff, .. } => {
                cutoff.validate()?;
                anyhow::ensure!(
                    cutoff.is_fixed(),
                    "a close control must name a fixed cutoff version"
                );
            }
            Self::Release { op_id, .. } => {
                anyhow::ensure!(op_id.0 > 0, "a release control must name a valid operation");
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Admission framing (session establishment)
// ---------------------------------------------------------------------------

/// A logical KV allocation established at admission for a token lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvAllocation {
    pub block_ids: Vec<BlockId>,
    pub prefix_len: u32,
    pub group_id: u32,
}

/// Understanding-branch admission: sampling parameters, negative tokens, and KV.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UndAdmission {
    pub sampling: SamplingParams,
    pub negative_token_ids: Vec<u32>,
    pub kv: KvAllocation,
}

/// Generation-branch admission: image parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenAdmission {
    pub image: ImageParams,
}

/// Session establishment framing. Carries the per-domain parameters a lineage
/// needs before its operations run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Admission {
    pub request_key: RequestKey,
    pub digest: Digest,
    pub und: Option<UndAdmission>,
    pub gen_admission: Option<GenAdmission>,
    pub adapter_id: Option<u32>,
}

impl Admission {
    pub fn new(
        request_key: RequestKey,
        und: Option<UndAdmission>,
        gen_admission: Option<GenAdmission>,
        adapter_id: Option<u32>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            und.is_some() || gen_admission.is_some(),
            "admission must declare an understanding or generation branch"
        );
        let mut admission = Self {
            request_key,
            digest: String::new(),
            und,
            gen_admission,
            adapter_id,
        };
        admission.digest = admission.payload_digest();
        Ok(admission)
    }

    pub fn payload_digest(&self) -> Digest {
        let mut digest = CanonicalDigest::new(b"uniserve-admission\0");
        digest.request_key(self.request_key);
        digest.option(self.und.as_ref(), |digest, und| {
            digest.sampling(&und.sampling);
            digest.u32s(und.negative_token_ids.iter().copied());
            digest.u32s(und.kv.block_ids.iter().map(|block| block.0));
            digest.u32(und.kv.prefix_len);
            digest.u32(und.kv.group_id);
        });
        digest.option(self.gen_admission.as_ref(), |digest, branch| {
            digest.image(&branch.image)
        });
        digest.option(self.adapter_id, CanonicalDigest::u32);
        digest.finish()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.und.is_some() || self.gen_admission.is_some(),
            "admission must declare an understanding or generation branch"
        );
        anyhow::ensure!(
            is_digest(&self.digest),
            "admission digest is not a lowercase SHA-256 digest"
        );
        anyhow::ensure!(
            self.digest == self.payload_digest(),
            "admission digest does not match its payload"
        );
        if let Some(und) = &self.und {
            und.sampling.validate()?;
            anyhow::ensure!(
                und.kv.prefix_len == 0 || !und.kv.block_ids.is_empty(),
                "a non-empty KV prefix requires allocated blocks"
            );
            anyhow::ensure!(
                und.kv.block_ids.iter().collect::<HashSet<_>>().len() == und.kv.block_ids.len(),
                "KV allocation repeats a logical block"
            );
        }
        if let Some(branch) = &self.gen_admission {
            branch.image.validate()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Batch and response framing
// ---------------------------------------------------------------------------

/// A submission window: a topologically ordered set of operations plus any
/// controls, and the admissions that establish their lineages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    pub step_id: u64,
    pub admissions: Vec<Admission>,
    pub operations: Vec<Operation>,
    pub controls: Vec<Control>,
    /// Host-supplied input product values the operations reference through
    /// `Operation::inputs`, matched by `ProductRef` identity. These are
    /// transported values, not lineage identity: `plan_digest` already covers
    /// the input product references, so `input_products` enters no digest.
    pub input_products: Vec<ProductPayload>,
}

impl Batch {
    pub fn new(step_id: u64, admissions: Vec<Admission>, operations: Vec<Operation>) -> Self {
        Self {
            step_id,
            admissions,
            operations,
            controls: Vec::new(),
            input_products: Vec::new(),
        }
    }

    pub fn with_controls(mut self, controls: Vec<Control>) -> Self {
        self.controls = controls;
        self
    }

    pub fn with_input_products(mut self, input_products: Vec<ProductPayload>) -> Self {
        self.input_products = input_products;
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.operations.is_empty() || !self.controls.is_empty(),
            "a submission batch must carry at least one operation or control"
        );
        // Depth one: at most one runnable operation per request per batch.
        let mut request_keys = HashSet::with_capacity(self.operations.len());
        for operation in &self.operations {
            operation.validate()?;
            anyhow::ensure!(
                request_keys.insert(operation.request_key),
                "a submission batch carries multiple operations for one request"
            );
        }
        let mut admitted = HashSet::with_capacity(self.admissions.len());
        for admission in &self.admissions {
            admission.validate()?;
            anyhow::ensure!(
                admitted.insert(admission.request_key),
                "a submission batch carries a duplicate admission"
            );
            anyhow::ensure!(
                self.operations
                    .iter()
                    .any(|operation| operation.request_key == admission.request_key),
                "a submission batch admits a request without an operation"
            );
        }
        // Idempotency identity: (request_key, control_seq, variant, content).
        let mut control_identities: std::collections::HashMap<
            (RequestKey, Option<u64>, u8),
            Digest,
        > = std::collections::HashMap::new();
        for control in &self.controls {
            control.validate()?;
            let identity = (
                control.request_key(),
                control.control_seq(),
                control.variant_index(),
            );
            let content = control.content_digest();
            if let Some(existing) = control_identities.get(&identity) {
                anyhow::ensure!(
                    existing == &content,
                    "a submission batch reuses a control identity with different content"
                );
            } else {
                control_identities.insert(identity, content);
            }
        }
        for payload in &self.input_products {
            payload.validate()?;
        }
        Ok(())
    }
}

/// The per-operation forward statistics a worker attaches to a completion report.
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

/// Envelope metadata reporting whether an atomic registration became visible. It
/// carries no semantic lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RegistrationAck {
    pub visible: bool,
}

/// A resolved product value carried across the boundary: a host-supplied input
/// value the worker consumes (prompt, forced, or draft token ids; encode image
/// bytes) referenced through `Operation::inputs`, or a worker-produced output
/// value the host consumes (requested logprob blobs, materialized image bytes).
/// The `product` identifies what the value is by `ProductRef` identity; the
/// bytes are the value. A product payload is never a lineage identity and enters
/// no digest.
///
/// The protocol layer treats `bytes` as opaque; the codec agreement is enforced
/// by consumers. The one convention this crate fixes and shares with the Python
/// worker is the `ProductKind::Token` layout, produced and parsed by
/// [`encode_token_product_bytes`] and [`decode_token_product_bytes`]: a
/// little-endian `u32` count `N`, then `N` little-endian `u32` token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductPayload {
    pub product: ProductRef,
    /// Raw value bytes. `serde_bytes` keeps the pyo3 boundary on the
    /// bytes fast path (one buffer copy) instead of a per-element
    /// integer-sequence walk, which costs hundreds of milliseconds for a
    /// multi-megabyte image artifact.
    #[serde(with = "serde_bytes")]
    pub bytes: Vec<u8>,
}

impl ProductPayload {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.product.validate()
    }
}

/// Encode a `ProductKind::Token` product value: a little-endian `u32` count
/// followed by that many little-endian `u32` token ids.
pub fn encode_token_product_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + tokens.len() * 4);
    bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes
}

/// Decode a `ProductKind::Token` product value produced by
/// [`encode_token_product_bytes`].
pub fn decode_token_product_bytes(bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        bytes.len() >= 4,
        "token product bytes are too short to carry a count"
    );
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected = 4 + count * 4;
    anyhow::ensure!(
        bytes.len() == expected,
        "token product byte length {} does not match declared count {count}",
        bytes.len()
    );
    Ok(bytes[4..]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

/// A worker response frame: one completion per submitted operation, the resolved
/// output-product values for host consumption, and the registration
/// acknowledgement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionReport {
    pub step_id: u64,
    pub completions: Vec<CompletionRecord>,
    pub products: Vec<ProductPayload>,
    pub registration: RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<WorkerForwardStats>,
}

impl CompletionReport {
    pub fn validate(&self) -> anyhow::Result<()> {
        for completion in &self.completions {
            completion.validate()?;
        }
        for payload in &self.products {
            payload.validate()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Capabilities and startup agreement
// ---------------------------------------------------------------------------

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

/// A worker's advertised capabilities. Admission requires every rank, worker,
/// and frontend to agree on both the protocol-layout and route-capability
/// digests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineCaps {
    pub block_size: u32,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub scratch_capacity_tokens: u64,
    pub supported_work: Vec<WorkVariant>,
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
    pub model_spec_digest: Digest,
    pub weight_digest: Digest,
    pub protocol_layout_digest: Digest,
    pub route_capability_digest: Digest,
    pub restored_sessions: Vec<RequestId>,
}

impl EngineCaps {
    /// The canonical protocol-layout digest: a disagreement means a peer reads a
    /// different record layout and must not be admitted.
    pub fn canonical_protocol_layout_digest() -> Digest {
        protocol_layout_digest()
    }

    /// The route-capability digest: per-route supported work, sampler and shape
    /// regime, and mixed-submission capability summarized for the agreement.
    pub fn compute_route_capability_digest(&self) -> Digest {
        let mut digest = CanonicalDigest::new(b"uniserve-route-capability\0");
        let mut variants: Vec<u8> = self
            .supported_work
            .iter()
            .map(|variant| *variant as u8)
            .collect();
        variants.sort_unstable();
        variants.dedup();
        digest.u64(variants.len() as u64);
        for variant in variants {
            digest.u8(variant);
        }
        digest.u32(self.max_cfg_branches);
        digest.u32(self.max_latent_size);
        digest.u32(self.max_vae_grid_tokens);
        digest.u32(self.max_vit_grid_tokens);
        digest.u8(self.adapter_mode as u8);
        digest.u32(self.execution_constraints.max_batch_operations);
        digest.string(&self.kv_dtype);
        digest.string(&self.model_dtype);
        digest.string(&self.attention_backend);
        digest.finish()
    }

    /// Whether this process may be admitted alongside `other`.
    pub fn agrees_with(&self, other: &Self) -> bool {
        self.protocol_layout_digest == other.protocol_layout_digest
            && self.route_capability_digest == other.route_capability_digest
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.supported_work.is_empty(),
            "worker capabilities declare no work variants"
        );
        anyhow::ensure!(
            self.protocol_layout_digest == protocol_layout_digest(),
            "worker capabilities carry a disagreeing protocol-layout digest"
        );
        anyhow::ensure!(
            self.route_capability_digest == self.compute_route_capability_digest(),
            "worker capabilities carry an inconsistent route-capability digest"
        );
        anyhow::ensure!(
            (self.model_spec_digest.is_empty() && self.weight_digest.is_empty())
                || (is_digest(&self.model_spec_digest) && is_digest(&self.weight_digest)),
            "worker capability model and weight identities are incomplete"
        );
        Ok(())
    }
}

impl Default for EngineCaps {
    fn default() -> Self {
        let mut caps = Self {
            block_size: 64,
            num_blocks: 4096,
            num_layers: 28,
            scratch_capacity_tokens: 1 << 20,
            supported_work: vec![WorkVariant::TokenExtend, WorkVariant::TokenDecode],
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
            protocol_layout_digest: protocol_layout_digest(),
            route_capability_digest: String::new(),
            restored_sessions: Vec::new(),
        };
        caps.route_capability_digest = caps.compute_route_capability_digest();
        caps
    }
}

/// The canonical protocol-layout digest over the closed `Work` and `Control`
/// variants and the fixed record field layouts.
pub fn protocol_layout_digest() -> Digest {
    let mut digest = CanonicalDigest::new(b"uniserve-protocol-layout\0");
    digest.u64(WorkVariant::ALL.len() as u64);
    for variant in WorkVariant::ALL {
        digest.string(variant.as_wire_str());
    }
    for control in ["commit", "close", "release"] {
        digest.string(control);
    }
    // Record field layouts, in declaration order.
    let record_layouts: [&[&str]; 4] = [
        &[
            "request_key",
            "op_id",
            "parent",
            "work",
            "route",
            "domain",
            "advances_state",
            "bounds",
            "inputs",
            "outputs",
            "predicate",
            "rng",
            "control_seq",
            "plan_digest",
        ],
        &["request_key", "producer_op_id", "point"],
        &[
            "request_key",
            "producer_op_id",
            "output_index",
            "generation",
            "kind",
            "storage_class",
            "dtype",
            "shape_bound",
            "point_range",
        ],
        &[
            "request_key",
            "op_id",
            "completion_slot_generation",
            "status",
            "selected_point",
            "logical_lengths",
            "token_span",
            "finish_flags",
            "product_generations",
            "semantic_digest",
            "error_code",
            "timing_counters",
        ],
    ];
    for record in record_layouts {
        digest.u64(record.len() as u64);
        for field in record {
            digest.string(field);
        }
    }
    digest.finish()
}

// ---------------------------------------------------------------------------
// Administrative request and response framing
// ---------------------------------------------------------------------------

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
    pub digest: Digest,
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
    pub request_key: RequestKey,
    pub op_id: OpId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub kind: ResponseKind,
    pub call_id: Option<u64>,
    pub capabilities: Option<EngineCaps>,
    pub completion_report: Option<CompletionReport>,
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
    fn bare(kind: ResponseKind) -> Self {
        Self {
            kind,
            call_id: None,
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
        }
    }

    pub fn capabilities(capabilities: EngineCaps) -> Self {
        Self {
            capabilities: Some(capabilities),
            ..Self::bare(ResponseKind::Capabilities)
        }
    }

    pub fn completion_report(report: CompletionReport) -> Self {
        Self {
            completion_report: Some(report),
            ..Self::bare(ResponseKind::Result)
        }
    }

    pub fn ok() -> Self {
        Self::bare(ResponseKind::Ok)
    }

    pub fn snapshot(snapshot: SnapshotRef) -> Self {
        Self {
            snapshot: Some(snapshot),
            ..Self::bare(ResponseKind::Snapshot)
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical digest helper
// ---------------------------------------------------------------------------

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// A little-endian, length-prefixed SHA-256 builder. Each digest is domain
/// separated by a neutral tag; the byte layout is mirrored exactly by the Python
/// worker so both sides compute identical digests.
struct CanonicalDigest(Sha256);

impl CanonicalDigest {
    fn new(domain: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(domain);
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

    fn request_key(&mut self, value: RequestKey) {
        self.u64(value.authority_id);
        self.u64(value.session_id.0);
        self.u64(value.epoch);
    }

    fn op_id(&mut self, value: OpId) {
        self.u64(value.0);
    }

    fn shape_bound(&mut self, value: &ShapeBound) {
        self.u64(value.dims.len() as u64);
        for dim in &value.dims {
            match dim {
                DimBound::Static(extent) => {
                    self.u8(0);
                    self.u32(*extent);
                }
                DimBound::Device { max } => {
                    self.u8(1);
                    self.u32(*max);
                }
            }
        }
    }

    fn product_ref(&mut self, value: &ProductRef) {
        self.request_key(value.request_key);
        self.op_id(value.producer_op_id);
        self.u16(value.output_index);
        self.u32(value.generation);
        self.u8(value.kind as u8);
        self.u8(value.storage_class as u8);
        self.u8(value.dtype as u8);
        self.shape_bound(&value.shape_bound);
        self.u32(value.point_range.base_point);
        self.u32(value.point_range.max_points);
    }

    fn version_ref(&mut self, value: &VersionRef) {
        self.request_key(value.request_key);
        self.op_id(value.producer_op_id);
        match &value.point {
            Point::Fixed {
                point_index,
                semantic_digest,
            } => {
                self.u8(0);
                self.u32(*point_index);
                self.string(semantic_digest);
            }
            Point::Device {
                selected_point,
                producer_plan_digest,
            } => {
                self.u8(1);
                self.product_ref(selected_point);
                self.string(producer_plan_digest);
            }
        }
    }

    fn bounds(&mut self, value: &Bounds) {
        self.u32(value.max_points);
        self.u32(value.max_tokens);
        self.u32(value.max_kv_pages);
        self.u64(value.max_latent_bytes);
        self.u64(value.max_completion_bytes);
        self.u64(value.max_transfer_bytes);
    }

    fn rng(&mut self, value: &Rng) {
        self.u64(value.seed);
        self.u64(value.semantic_index_base);
        self.u8(value.draw_layout as u8);
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
}

#[cfg(test)]
mod tests;
