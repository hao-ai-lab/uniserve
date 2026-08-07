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

use std::collections::{BTreeMap, HashMap, HashSet};

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
    CreditDimension, CreditVector, ResourceClass, ResourcePressure, RouteCreditLimits,
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
    SamplingState = 9,
    Finish = 10,
    SelectedPoint = 11,
    AcceptedSpan = 12,
    Continuation = 13,
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
        anyhow::ensure!(
            self.dims.iter().all(|dim| match dim {
                DimBound::Static(value) => *value > 0,
                DimBound::Device { max } => *max > 0,
            }),
            "a shape bound contains a zero extent"
        );
        Ok(())
    }

    fn max_elements(&self) -> u64 {
        self.dims.iter().fold(1_u64, |elements, dim| {
            elements.saturating_mul(u64::from(match dim {
                DimBound::Static(value) => *value,
                DimBound::Device { max } => *max,
            }))
        })
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
        anyhow::ensure!(
            self.generation > 0,
            "product reference has no logical generation"
        );
        self.shape_bound.validate()
    }

    pub fn max_bytes(&self) -> u64 {
        let element_bytes = match self.dtype {
            DType::U8 => 1,
            DType::U16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::I64 => 8,
        };
        self.shape_bound
            .max_elements()
            .saturating_mul(element_bytes)
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
        /// The producer-local point when selection is statically determined.
        point_index: u32,
        /// The producer's dynamic selection product for multi-point work.
        selected_point: Option<ProductRef>,
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
                point_index,
                selected_point,
                producer_plan_digest,
            } => {
                if let Some(selected_point) = selected_point {
                    anyhow::ensure!(
                        *point_index == 0,
                        "a dynamic device version also declares a fixed point"
                    );
                    selected_point.validate()?;
                    anyhow::ensure!(
                        selected_point.request_key == self.request_key
                            && selected_point.producer_op_id == self.producer_op_id,
                        "device version selected point is not owned by its producer"
                    );
                    anyhow::ensure!(
                        selected_point.generation > 0,
                        "device version selected point has no logical generation"
                    );
                    anyhow::ensure!(
                        selected_point.kind == ProductKind::SelectedPoint
                            && selected_point.storage_class == StorageClass::DeviceTensor
                            && selected_point.dtype == DType::U32
                            && selected_point.shape_bound.max_elements() == 1,
                        "device version does not name a scalar selected-point product"
                    );
                } else {
                    anyhow::ensure!(
                        *point_index > 0,
                        "a static device version must name a positive producer point"
                    );
                }
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

/// The physical execution contract for one scheduler batch partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ExecutionCapability {
    /// The physical submission contains one independently described domain.
    DomainHomogeneous = 0,
    /// Multiple independently described domain partitions share only the final
    /// tensorized runner call under a route-static capability declaration.
    TensorizedMixed = 1,
}

/// The attention structure one partition presents to the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum AttentionRegime {
    None = 0,
    Causal = 1,
    Bidirectional = 2,
    Hybrid = 3,
}

/// Tensor-parallel ownership of device token selection for one route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum SamplingOwnership {
    DesignatedRank = 0,
    DeterministicSharded = 1,
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
    /// Exact logical KV capacity visible after registration. Physical pages are
    /// selected by the worker from the partition's logical lease reservation.
    pub kv_capacity_pages: u32,
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
        kv_capacity_pages: u32,
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
            kv_capacity_pages,
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
        digest.u32(self.kv_capacity_pages);
        digest.option(self.predicate.as_ref(), CanonicalDigest::product_ref);
        digest.option(self.rng.as_ref(), CanonicalDigest::rng);
        digest.u64(self.control_seq);
        digest.finish()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.op_id.0 > 0, "operation id must be positive");
        anyhow::ensure!(
            self.advances_state == self.work.advances_state(),
            "operation declares an advances_state inconsistent with its work variant"
        );
        self.parent.validate()?;
        anyhow::ensure!(
            self.parent.request_key == self.request_key,
            "operation parent belongs to another request lineage"
        );
        self.bounds_are_finite()?;
        anyhow::ensure!(
            self.bounds.max_kv_pages <= self.kv_capacity_pages,
            "operation KV growth bound exceeds its logical capacity"
        );
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        for output in &self.outputs {
            output.validate()?;
            anyhow::ensure!(
                output.request_key == self.request_key && output.producer_op_id == self.op_id,
                "an output product is not owned by its producing operation"
            );
            anyhow::ensure!(
                output.generation > 0,
                "an output product has no logical generation"
            );
            anyhow::ensure!(
                output.point_range.max_points <= self.bounds.max_points.max(1),
                "an output product exceeds the operation point bound"
            );
            match output.storage_class {
                StorageClass::LatentArena => anyhow::ensure!(
                    output.max_bytes() <= self.bounds.max_latent_bytes,
                    "a latent-arena output exceeds the operation latent-byte bound"
                ),
                StorageClass::HostStaging | StorageClass::CompletionArena => anyhow::ensure!(
                    output.max_bytes() <= self.bounds.max_completion_bytes,
                    "a host-visible output exceeds the operation completion-byte bound"
                ),
                StorageClass::PagedKv => anyhow::ensure!(
                    output.max_bytes() <= self.bounds.max_transfer_bytes,
                    "a paged-KV output exceeds the operation transfer-byte bound"
                ),
                _ => {}
            }
            anyhow::ensure!(
                output_indices.insert(output.output_index),
                "operation repeats an output index"
            );
        }
        for input in &self.inputs {
            input.validate()?;
            anyhow::ensure!(
                input.request_key == self.request_key
                    || matches!(
                        input.kind,
                        ProductKind::VisionFeature | ProductKind::LatentFeature
                    ),
                "a request-local input product belongs to another request lineage"
            );
        }
        if let Some(predicate) = &self.predicate {
            predicate.validate()?;
            anyhow::ensure!(
                predicate.request_key == self.request_key,
                "operation predicate belongs to another request lineage"
            );
            let continuation_token = predicate.kind == ProductKind::Token
                && predicate.dtype == DType::U32
                && predicate.shape_bound.max_elements() == 1;
            anyhow::ensure!(
                predicate.generation > 0
                    && predicate.storage_class == StorageClass::DeviceTensor
                    && (predicate.kind == ProductKind::Completion || continuation_token),
                "operation predicate is not a generation-tagged device decision product"
            );
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
    pub kv_reserved_len: u32,
    pub kv_initialized_len: u32,
    pub kv_committed_len: u32,
    pub kv_published_len: u32,
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
            self.completion_slot_generation > 0,
            "completion slot generation must be positive"
        );
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
        if self.status == OpStatus::Predicated {
            anyhow::ensure!(
                self.token_span.len == 0
                    && self.committed_tokens.is_empty()
                    && self.product_generations.is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
            anyhow::ensure!(
                !self.finish_flags.eos && !self.finish_flags.length && !self.finish_flags.stop,
                "a predicated completion must not select a terminal outcome"
            );
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
/// (drop session, copy KV, prefix-cache reset, snapshot and restore)
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

/// KV lineage metadata established at admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct KvAdmission {
    pub prefix_len: u32,
    pub group_id: u32,
}

/// Understanding-branch admission: invariant sampling policy, negative tokens,
/// terminal token ids, and KV lineage metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UndAdmission {
    pub sampling: SamplingParams,
    pub negative_token_ids: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub kv: KvAdmission,
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
}

impl Admission {
    pub fn new(
        request_key: RequestKey,
        und: Option<UndAdmission>,
        gen_admission: Option<GenAdmission>,
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
            digest.u32s(und.finish_token_ids.iter().copied());
            digest.u32(und.kv.prefix_len);
            digest.u32(und.kv.group_id);
        });
        digest.option(self.gen_admission.as_ref(), |digest, branch| {
            digest.image(&branch.image)
        });
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
                und.finish_token_ids
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "und admission finish token ids are not canonical"
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

/// One independently owned scheduler batch partition. `submission_group`
/// identifies the physical runner call: a domain-homogeneous group has exactly
/// one partition, while a qualified tensorized-mixed group has one partition
/// per participating domain. `collective_seq` is the exact TP collective order
/// all ranks validate before enqueue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchPartition {
    pub partition_id: u32,
    pub submission_group: u32,
    pub collective_seq: u64,
    pub domain: Domain,
    pub route: RouteId,
    pub execution: ExecutionCapability,
    pub attention: AttentionRegime,
    pub shape_class: u64,
    pub operations: Vec<Operation>,
    /// Logical page deltas bound atomically while this partition is registered.
    /// Logical identity preserves prefix sharing; worker-local mappings determine
    /// every physical page address.
    pub kv_reservations: Vec<KvReservation>,
}

/// One operation's logical KV page growth at registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvReservation {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub logical_page_delta: Vec<BlockId>,
}

impl KvReservation {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.op_id.0 > 0,
            "KV reservation operation id must be positive"
        );
        anyhow::ensure!(
            self.logical_page_delta.iter().collect::<HashSet<_>>().len()
                == self.logical_page_delta.len(),
            "KV reservation repeats a logical page"
        );
        Ok(())
    }
}

impl BatchPartition {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.partition_id > 0, "batch partition id must be positive");
        anyhow::ensure!(
            self.submission_group > 0,
            "batch partition submission group must be positive"
        );
        anyhow::ensure!(
            self.collective_seq > 0,
            "batch partition collective sequence must be positive"
        );
        anyhow::ensure!(
            !self.operations.is_empty(),
            "batch partition must carry at least one operation"
        );
        for operation in &self.operations {
            operation.validate()?;
            anyhow::ensure!(
                operation.domain == self.domain && operation.route == self.route,
                "batch partition operation disagrees with its domain or route"
            );
        }
        let operations = self
            .operations
            .iter()
            .map(|operation| ((operation.request_key, operation.op_id), operation))
            .collect::<HashMap<_, _>>();
        let mut reservation_ids = HashSet::with_capacity(self.kv_reservations.len());
        for reservation in &self.kv_reservations {
            reservation.validate()?;
            let identity = (reservation.request_key, reservation.op_id);
            anyhow::ensure!(
                reservation_ids.insert(identity),
                "batch partition repeats a KV reservation identity"
            );
            let operation = operations.get(&identity).ok_or_else(|| {
                anyhow::anyhow!("KV reservation does not name a partition operation")
            })?;
            anyhow::ensure!(
                reservation.logical_page_delta.len() <= operation.kv_capacity_pages as usize,
                "KV reservation exceeds operation logical capacity"
            );
        }
        for operation in &self.operations {
            anyhow::ensure!(
                operation.kv_capacity_pages == 0
                    || reservation_ids.contains(&(operation.request_key, operation.op_id)),
                "operation with logical KV capacity has no reservation"
            );
        }
        Ok(())
    }
}

/// A submission window containing independently owned physical partitions plus
/// controls and the admissions that establish their lineages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    pub step_id: u64,
    pub admissions: Vec<Admission>,
    pub partitions: Vec<BatchPartition>,
    pub controls: Vec<Control>,
    /// Host-supplied input product values the operations reference through
    /// `Operation::inputs`, matched by `ProductRef` identity. These are
    /// transported values, not lineage identity: `plan_digest` already covers
    /// the input product references, so `input_products` enters no digest.
    pub input_products: Vec<ProductPayload>,
}

impl Batch {
    pub fn new(step_id: u64, admissions: Vec<Admission>, partitions: Vec<BatchPartition>) -> Self {
        Self {
            step_id,
            admissions,
            partitions,
            controls: Vec::new(),
            input_products: Vec::new(),
        }
    }

    pub fn operations(&self) -> impl Iterator<Item = &Operation> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.operations.iter())
    }

    pub fn operation_count(&self) -> usize {
        self.partitions
            .iter()
            .map(|partition| partition.operations.len())
            .sum()
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
            !self.partitions.is_empty() || !self.controls.is_empty(),
            "a submission batch must carry at least one operation or control"
        );
        let mut partition_ids = HashSet::with_capacity(self.partitions.len());
        let mut submission_groups: std::collections::HashMap<u32, Vec<&BatchPartition>> =
            std::collections::HashMap::new();
        for partition in &self.partitions {
            partition.validate()?;
            anyhow::ensure!(
                partition_ids.insert(partition.partition_id),
                "a submission batch repeats a partition id"
            );
            submission_groups
                .entry(partition.submission_group)
                .or_default()
                .push(partition);
        }
        for partitions in submission_groups.values() {
            let execution = partitions[0].execution;
            let collective_seq = partitions[0].collective_seq;
            let attention = partitions[0].attention;
            let shape_class = partitions[0].shape_class;
            anyhow::ensure!(
                partitions.iter().all(|partition| {
                    partition.execution == execution
                        && partition.collective_seq == collective_seq
                        && partition.attention == attention
                        && partition.shape_class == shape_class
                }),
                "physical submission partitions disagree on execution, attention, shape, or collective order"
            );
            match execution {
                ExecutionCapability::DomainHomogeneous => anyhow::ensure!(
                    partitions.len() == 1,
                    "a domain-homogeneous submission group must contain one partition"
                ),
                ExecutionCapability::TensorizedMixed => {
                    anyhow::ensure!(
                        partitions.len() >= 2,
                        "a tensorized-mixed submission group must contain multiple partitions"
                    );
                    anyhow::ensure!(
                        partitions
                            .iter()
                            .map(|partition| partition.domain)
                            .collect::<HashSet<_>>()
                            .len()
                            == partitions.len(),
                        "a tensorized-mixed submission group repeats a domain"
                    );
                    anyhow::ensure!(
                        partitions
                            .iter()
                            .all(|partition| partition.route == partitions[0].route),
                        "a tensorized-mixed submission group spans route capabilities"
                    );
                }
            }
        }
        // Depth one: at most one runnable operation per request per batch.
        let mut request_keys = HashSet::with_capacity(self.operation_count());
        for operation in self.operations() {
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
                self.operations()
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
        let declared_inputs = self
            .operations()
            .flat_map(|operation| operation.inputs.iter())
            .collect::<HashSet<_>>();
        for operation in self.operations() {
            for input in operation
                .inputs
                .iter()
                .filter(|input| input.storage_class == StorageClass::HostStaging)
            {
                anyhow::ensure!(
                    input.request_key == operation.request_key
                        && input.producer_op_id == operation.op_id,
                    "a host-staging input is not owned by its consuming operation"
                );
            }
        }
        let mut supplied_inputs = HashSet::with_capacity(self.input_products.len());
        for payload in &self.input_products {
            anyhow::ensure!(
                declared_inputs.contains(&payload.product),
                "an input product payload is not declared by any operation"
            );
            let transferred = is_transfer_descriptor(&payload.bytes);
            if payload.product.storage_class == StorageClass::HostStaging {
                anyhow::ensure!(
                    !transferred,
                    "host-staging input cannot carry a cross-stage transfer descriptor"
                );
            } else {
                anyhow::ensure!(
                    transferred && payload.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                    "cross-stage product input has an invalid transfer descriptor frame"
                );
            }
            anyhow::ensure!(
                supplied_inputs.insert(&payload.product),
                "a submission batch repeats an input product payload"
            );
            payload.validate_input_value()?;
        }
        for input in declared_inputs {
            if input.storage_class == StorageClass::HostStaging {
                anyhow::ensure!(
                    supplied_inputs.contains(input),
                    "a host-staging operation input has no product payload"
                );
            }
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
/// The protocol layer treats other product bytes as opaque. This crate fixes
/// the cross-language layouts for token inputs and branch-local sampling state.
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

    fn validate_input_value(&self) -> anyhow::Result<()> {
        if is_transfer_descriptor(&self.bytes) {
            anyhow::ensure!(
                self.product.storage_class != StorageClass::HostStaging
                    && self.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                "cross-stage product input has an invalid transfer descriptor frame"
            );
            return Ok(());
        }
        match self.product.kind {
            ProductKind::Token => {
                let tokens = decode_token_product_bytes(&self.bytes)?;
                anyhow::ensure!(
                    tokens.len() as u64 <= self.product.shape_bound.max_elements(),
                    "token input product exceeds its registered element bound"
                );
            }
            ProductKind::SamplingState => {
                decode_sampling_state_bytes(&self.bytes)?;
                anyhow::ensure!(
                    self.bytes.len() as u64 <= self.product.max_bytes(),
                    "sampling-state input exceeds its registered byte bound"
                );
            }
            _ => anyhow::ensure!(
                self.bytes.len() as u64 <= self.product.max_bytes(),
                "input product payload exceeds its registered byte bound"
            ),
        }
        Ok(())
    }

    fn validate_output_value(&self) -> anyhow::Result<()> {
        self.validate()?;
        if is_transfer_descriptor(&self.bytes) {
            anyhow::ensure!(
                self.product.storage_class != StorageClass::HostStaging
                    && self.product.storage_class != StorageClass::CompletionArena
                    && self.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                "output transfer descriptor has an invalid storage class or byte bound"
            );
            return Ok(());
        }
        anyhow::ensure!(
            self.bytes.len() as u64 <= self.product.max_bytes(),
            "output product payload exceeds its registered byte bound"
        );
        Ok(())
    }
}

pub const TRANSFER_DESCRIPTOR_PREFIX: &[u8] = b"uniserve-transfer\0";
pub const MAX_TRANSFER_DESCRIPTOR_BYTES: usize = 64 * 1024;

pub fn is_transfer_descriptor(bytes: &[u8]) -> bool {
    bytes.starts_with(TRANSFER_DESCRIPTOR_PREFIX)
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

/// Branch-local token processor inputs for one sampling operation.
///
/// Token ids in every field are strictly increasing. `allowed_token_ids`
/// distinguishes no whitelist (`None`) from a present empty whitelist, which
/// deterministically represents an invalid all-masked distribution.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SamplingState {
    pub recent_counts: Vec<(u32, u32)>,
    pub allowed_token_ids: Option<Vec<u32>>,
    pub suppressed_token_ids: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub force_finish: bool,
}

impl SamplingState {
    pub fn canonicalize(&mut self) {
        self.recent_counts.retain(|(_, count)| *count > 0);
        self.recent_counts.sort_unstable_by_key(|(token, _)| *token);
        self.recent_counts.dedup_by(|next, prior| {
            if next.0 == prior.0 {
                prior.1 = prior.1.saturating_add(next.1);
                true
            } else {
                false
            }
        });
        if let Some(allowed) = &mut self.allowed_token_ids {
            allowed.sort_unstable();
            allowed.dedup();
        }
        self.suppressed_token_ids.sort_unstable();
        self.suppressed_token_ids.dedup();
        self.finish_token_ids.sort_unstable();
        self.finish_token_ids.dedup();
    }
}

/// Encode canonical branch-local sampling state.
///
/// Layout: recent count and `(token, count)` pairs; one allowed-presence byte;
/// an allowed count and ids when present; then a suppressed count and ids.
pub fn encode_sampling_state_bytes(state: &SamplingState) -> Vec<u8> {
    let mut canonical = state.clone();
    canonical.canonicalize();
    let allowed_len = canonical.allowed_token_ids.as_ref().map_or(0, Vec::len);
    let mut bytes = Vec::with_capacity(
        9 + canonical.recent_counts.len() * 8
            + allowed_len * 4
            + canonical.suppressed_token_ids.len() * 4
            + canonical.finish_token_ids.len() * 4
            + 1,
    );
    bytes.extend_from_slice(&(canonical.recent_counts.len() as u32).to_le_bytes());
    for (token, count) in canonical.recent_counts {
        bytes.extend_from_slice(&token.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    match canonical.allowed_token_ids {
        Some(allowed) => {
            bytes.push(1);
            bytes.extend_from_slice(&(allowed.len() as u32).to_le_bytes());
            for token in allowed {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&(canonical.suppressed_token_ids.len() as u32).to_le_bytes());
    for token in canonical.suppressed_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(canonical.finish_token_ids.len() as u32).to_le_bytes());
    for token in canonical.finish_token_ids {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.push(u8::from(canonical.force_finish));
    bytes
}

/// Decode and validate canonical branch-local sampling state.
pub fn decode_sampling_state_bytes(bytes: &[u8]) -> anyhow::Result<SamplingState> {
    fn take_u32(bytes: &[u8], offset: &mut usize) -> anyhow::Result<u32> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| anyhow::anyhow!("sampling-state offset overflow"))?;
        anyhow::ensure!(end <= bytes.len(), "sampling-state bytes are truncated");
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        Ok(value)
    }
    fn take_ids(bytes: &[u8], offset: &mut usize, count: u32) -> anyhow::Result<Vec<u32>> {
        let mut values = Vec::with_capacity(count as usize);
        for _ in 0..count {
            values.push(take_u32(bytes, offset)?);
        }
        anyhow::ensure!(
            values.windows(2).all(|pair| pair[0] < pair[1]),
            "sampling-state token ids are not canonical"
        );
        Ok(values)
    }

    let mut offset = 0;
    let recent_len = take_u32(bytes, &mut offset)?;
    let mut recent_counts = Vec::with_capacity(recent_len as usize);
    for _ in 0..recent_len {
        let token = take_u32(bytes, &mut offset)?;
        let count = take_u32(bytes, &mut offset)?;
        anyhow::ensure!(count > 0, "sampling-state recent count must be positive");
        recent_counts.push((token, count));
    }
    anyhow::ensure!(
        recent_counts.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "sampling-state recent token ids are not canonical"
    );
    anyhow::ensure!(
        offset < bytes.len(),
        "sampling-state bytes omit allowed presence"
    );
    let allowed_token_ids = match bytes[offset] {
        0 => {
            offset += 1;
            None
        }
        1 => {
            offset += 1;
            let count = take_u32(bytes, &mut offset)?;
            Some(take_ids(bytes, &mut offset, count)?)
        }
        other => anyhow::bail!("sampling-state allowed presence {other} is invalid"),
    };
    let suppressed_len = take_u32(bytes, &mut offset)?;
    let suppressed_token_ids = take_ids(bytes, &mut offset, suppressed_len)?;
    let finish_len = take_u32(bytes, &mut offset)?;
    let finish_token_ids = take_ids(bytes, &mut offset, finish_len)?;
    anyhow::ensure!(
        offset < bytes.len(),
        "sampling-state bytes omit force-finish"
    );
    let force_finish = match bytes[offset] {
        0 => false,
        1 => true,
        other => anyhow::bail!("sampling-state force-finish {other} is invalid"),
    };
    offset += 1;
    anyhow::ensure!(
        offset == bytes.len(),
        "sampling-state bytes contain trailing data"
    );
    Ok(SamplingState {
        recent_counts,
        allowed_token_ids,
        suppressed_token_ids,
        finish_token_ids,
        force_finish,
    })
}

/// One independently completed partition in a worker response frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionCompletion {
    pub partition_id: u32,
    pub completions: Vec<CompletionRecord>,
    pub products: Vec<ProductPayload>,
    pub registration: RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<WorkerForwardStats>,
}

impl PartitionCompletion {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.partition_id > 0,
            "partition completion id must be positive"
        );
        for completion in &self.completions {
            completion.validate()?;
        }
        for payload in &self.products {
            payload.validate_output_value()?;
        }
        Ok(())
    }
}

/// A worker response frame containing independently completed scheduler
/// partitions. Partition order is framing only; operation identity restores
/// scheduler order without imposing a cross-partition readiness dependency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionReport {
    pub step_id: u64,
    pub partitions: Vec<PartitionCompletion>,
}

impl CompletionReport {
    pub fn completions(&self) -> impl Iterator<Item = &CompletionRecord> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.completions.iter())
    }

    pub fn products(&self) -> impl Iterator<Item = &ProductPayload> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.products.iter())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let mut partition_ids = HashSet::with_capacity(self.partitions.len());
        for partition in &self.partitions {
            partition.validate()?;
            anyhow::ensure!(
                partition_ids.insert(partition.partition_id),
                "completion report repeats a partition id"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Capabilities and startup agreement
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteExecutionCapability {
    pub route: RouteId,
    pub supported_work: Vec<WorkVariant>,
    pub tensorized_mixed: bool,
    pub sampling_ownership: SamplingOwnership,
    pub preemptible: bool,
    pub credits: RouteCreditLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    pub max_batch_operations: u32,
    pub max_speculative_points: u32,
    /// Upper bound on submitted-but-unresolved operations the worker admits per
    /// request, making the previously implicit registration window an explicit
    /// declared fact the scheduler bounds against.
    pub max_unresolved_window: u32,
    pub device_sequence_lengths: bool,
    pub device_append_offsets: bool,
    pub incremental_kv_publication: bool,
    pub route_capabilities: Vec<RouteExecutionCapability>,
}

impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self {
            max_batch_operations: 1,
            max_speculative_points: 1,
            max_unresolved_window: 1,
            device_sequence_lengths: true,
            device_append_offsets: true,
            incremental_kv_publication: true,
            route_capabilities: vec![RouteExecutionCapability {
                route: RouteId(0),
                supported_work: vec![WorkVariant::TokenExtend, WorkVariant::TokenDecode],
                tensorized_mixed: false,
                sampling_ownership: SamplingOwnership::DesignatedRank,
                preemptible: false,
                credits: RouteCreditLimits {
                    per_request: CreditVector {
                        registered_operations: 1,
                        execution_slots: 1,
                        completion_slots: 1,
                        ..CreditVector::ZERO
                    },
                    worker: CreditVector {
                        registered_operations: 1,
                        execution_slots: 1,
                        completion_slots: 1,
                        ..CreditVector::ZERO
                    },
                },
            }],
        }
    }
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
    pub max_latent_feature_bytes: u64,
    pub max_vision_feature_bytes: u64,
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
    pub execution_constraints: ExecutionConstraints,
    pub resource_classes: Vec<ResourceClass>,
    pub model_spec_digest: Digest,
    pub weight_digest: Digest,
    pub protocol_layout_digest: Digest,
    pub route_capability_digest: Digest,
    pub restored_snapshots: Vec<SnapshotRef>,
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
        digest.u64(self.max_latent_feature_bytes);
        digest.u64(self.max_vision_feature_bytes);
        digest.u32(self.execution_constraints.max_batch_operations);
        digest.u32(self.execution_constraints.max_speculative_points);
        digest.u32(self.execution_constraints.max_unresolved_window);
        digest.bool(self.execution_constraints.device_sequence_lengths);
        digest.bool(self.execution_constraints.device_append_offsets);
        digest.bool(self.execution_constraints.incremental_kv_publication);
        let mut route_capabilities = self.execution_constraints.route_capabilities.to_vec();
        route_capabilities.sort_unstable_by_key(|capability| capability.route.0);
        digest.u64(route_capabilities.len() as u64);
        for capability in route_capabilities {
            digest.u32(capability.route.0);
            let mut supported_work = capability
                .supported_work
                .iter()
                .map(|variant| *variant as u8)
                .collect::<Vec<_>>();
            supported_work.sort_unstable();
            supported_work.dedup();
            digest.u64(supported_work.len() as u64);
            for variant in supported_work {
                digest.u8(variant);
            }
            digest.bool(capability.tensorized_mixed);
            digest.u8(capability.sampling_ownership as u8);
            digest.bool(capability.preemptible);
            digest.credit_vector(capability.credits.per_request);
            digest.credit_vector(capability.credits.worker);
        }
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
            self.execution_constraints.max_batch_operations > 0
                && self.execution_constraints.max_speculative_points > 0
                && self.execution_constraints.max_unresolved_window > 0,
            "worker execution constraints declare a zero bound"
        );
        anyhow::ensure!(
            !self.execution_constraints.route_capabilities.is_empty()
                && self
                    .execution_constraints
                    .route_capabilities
                    .iter()
                    .map(|capability| capability.route)
                    .collect::<HashSet<_>>()
                    .len()
                    == self.execution_constraints.route_capabilities.len(),
            "worker execution constraints declare missing or repeated route capabilities"
        );
        let declared_work = self.supported_work.iter().copied().collect::<HashSet<_>>();
        let routed_work = self
            .execution_constraints
            .route_capabilities
            .iter()
            .flat_map(|capability| capability.supported_work.iter().copied())
            .collect::<HashSet<_>>();
        anyhow::ensure!(
            self.execution_constraints
                .route_capabilities
                .iter()
                .all(|capability| {
                    !capability.supported_work.is_empty()
                        && capability
                            .supported_work
                            .iter()
                            .all(|variant| declared_work.contains(variant))
                        && capability
                            .supported_work
                            .iter()
                            .collect::<HashSet<_>>()
                            .len()
                            == capability.supported_work.len()
                        && capability.credits.validate().is_ok()
                        && capability.credits.per_request.registered_operations > 0
                        && capability.credits.per_request.execution_slots > 0
                        && capability.credits.per_request.completion_slots > 0
                })
                && routed_work == declared_work,
            "worker route capabilities do not partition the declared work"
        );
        let shared_worker_credits = self.execution_constraints.route_capabilities[0]
            .credits
            .worker;
        anyhow::ensure!(
            self.execution_constraints
                .route_capabilities
                .iter()
                .all(|capability| capability.credits.worker == shared_worker_credits),
            "worker routes disagree on the shared worker-wide credit vector"
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
        let restored_session_ids = self
            .restored_snapshots
            .iter()
            .map(|snapshot| snapshot.version.request_key.session_id)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            self.restored_snapshots
                .iter()
                .all(|snapshot| snapshot.validate().is_ok())
                && restored_session_ids
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "worker restored snapshot identities are invalid or non-canonical"
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
            max_latent_feature_bytes: 0,
            max_vision_feature_bytes: 0,
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
            execution_constraints: ExecutionConstraints::default(),
            resource_classes: Vec::new(),
            model_spec_digest: String::new(),
            weight_digest: String::new(),
            protocol_layout_digest: protocol_layout_digest(),
            route_capability_digest: String::new(),
            restored_snapshots: Vec::new(),
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
    let product_kinds = [
        "token",
        "logprob",
        "draft",
        "vision_feature",
        "latent_feature",
        "kv",
        "latent",
        "artifact",
        "completion",
        "sampling_state",
        "finish",
        "selected_point",
        "accepted_span",
        "continuation",
    ];
    digest.u64(product_kinds.len() as u64);
    for kind in product_kinds {
        digest.string(kind);
    }
    for control in ["commit", "close", "release"] {
        digest.string(control);
    }
    // Record field layouts, in declaration order.
    let record_layouts: [&[&str]; 10] = [
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
            "kv_capacity_pages",
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
            "committed_tokens",
            "finish_flags",
            "product_generations",
            "semantic_digest",
            "error_code",
            "timing_counters",
        ],
        &["version", "digest", "locator"],
        &["prefix_len", "group_id"],
        &["sampling", "negative_token_ids", "finish_token_ids", "kv"],
        &["request_key", "digest", "und", "gen_admission"],
        &["request_key", "op_id", "logical_page_delta"],
        &[
            "partition_id",
            "submission_group",
            "collective_seq",
            "domain",
            "route",
            "execution",
            "attention",
            "shape_class",
            "operations",
            "kv_reservations",
        ],
    ];
    for record in record_layouts {
        digest.u64(record.len() as u64);
        for field in record {
            digest.string(field);
        }
    }
    let logical_lengths = [
        "token_len",
        "kv_visible_len",
        "latent_len",
        "kv_reserved_len",
        "kv_initialized_len",
        "kv_committed_len",
        "kv_published_len",
    ];
    digest.u64(logical_lengths.len() as u64);
    for field in logical_lengths {
        digest.string(field);
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
    PollCompletions,
    DropSession,
    Shutdown,
    CopyKv,
    ReleaseProducts,
    GetMetrics,
    GetPressure,
    SnapshotSession,
    RestoreSession,
}

impl RequestKind {
    pub const ALL: [Self; 11] = [
        Self::GetCapabilities,
        Self::Execute,
        Self::PollCompletions,
        Self::DropSession,
        Self::Shutdown,
        Self::CopyKv,
        Self::ReleaseProducts,
        Self::GetMetrics,
        Self::GetPressure,
        Self::SnapshotSession,
        Self::RestoreSession,
    ];

    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::GetCapabilities => "get_capabilities",
            Self::Execute => "execute",
            Self::PollCompletions => "poll_completions",
            Self::DropSession => "drop_session",
            Self::Shutdown => "shutdown",
            Self::CopyKv => "copy_kv",
            Self::ReleaseProducts => "release_products",
            Self::GetMetrics => "get_metrics",
            Self::GetPressure => "get_pressure",
            Self::SnapshotSession => "snapshot_session",
            Self::RestoreSession => "restore_session",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRef {
    pub version: VersionRef,
    pub digest: Digest,
    pub locator: String,
}

impl SnapshotRef {
    pub fn validate(&self) -> anyhow::Result<()> {
        self.version.validate()?;
        anyhow::ensure!(
            self.version.is_fixed(),
            "snapshot reference version is not fixed"
        );
        anyhow::ensure!(
            is_digest(&self.digest) && self.locator == self.digest,
            "snapshot reference artifact digest or locator is invalid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub kind: RequestKind,
    pub call_id: Option<u64>,
    pub batch: Option<Batch>,
    pub step_id: Option<u64>,
    pub session_id: Option<RequestId>,
    pub copies: Option<Vec<(BlockId, BlockId)>>,
    pub product_handles: Option<Vec<u64>>,
    pub snapshot: Option<SnapshotRef>,
}

impl WorkerRequest {
    fn bare(kind: RequestKind) -> Self {
        Self {
            kind,
            call_id: None,
            batch: None,
            step_id: None,
            session_id: None,
            copies: None,
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
    pub fn poll_completions(step_id: u64) -> Self {
        Self {
            step_id: Some(step_id),
            ..Self::bare(RequestKind::PollCompletions)
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
    pub fn release_products(product_handles: Vec<u64>) -> Self {
        Self {
            product_handles: Some(product_handles),
            ..Self::bare(RequestKind::ReleaseProducts)
        }
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
                point_index,
                selected_point,
                producer_plan_digest,
            } => {
                self.u8(1);
                self.u32(*point_index);
                self.u8(u8::from(selected_point.is_some()));
                if let Some(selected_point) = selected_point {
                    self.product_ref(selected_point);
                }
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

    fn credit_vector(&mut self, value: CreditVector) {
        for dimension in CreditDimension::ALL {
            self.u64(value.get(dimension));
        }
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
