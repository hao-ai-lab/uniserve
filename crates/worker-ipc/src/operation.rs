//! Request identity, planned operations, static `NewRequest` state, and batches.

use super::*;

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

/// Route identity assigned by the scheduler for worker selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteId(pub u32);

// ---------------------------------------------------------------------------
// Closed `ForwardMode` algebra
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ReconstructionKind {
    Video,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaProfileId {
    MinimaxH3T2va,
}

/// Closed forward-mode tag for one operation. Declaration order is the
/// canonical index in IPC and worker selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ForwardMode {
    TokenExtend = 0,
    TokenDecode = 1,
    TokenVerify = 2,
    EncodeVision = 3,
    EncodeLatent = 4,
    TransferProduct = 5,
    TransferKvPublish = 6,
    TransferKvInstall = 7,
    MediaPrepare = 8,
    MediaDenoise = 9,
    Materialize = 10,
    MediaReconstruct = 11,
}

/// Configured or resolved worker attention implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttentionBackend {
    Auto,
    TrtllmMha,
    SglKernel,
    FlashInfer,
    FlashAttn,
    Fa4Cute,
    TorchSdpa,
    H3VsaSm100,
    Composite(Vec<AttentionBackend>),
}

impl AttentionBackend {
    pub fn as_name(&self) -> String {
        match self {
            Self::Auto => "auto".to_owned(),
            Self::TrtllmMha => "trtllm_mha".to_owned(),
            Self::SglKernel => "sgl_kernel".to_owned(),
            Self::FlashInfer => "flashinfer".to_owned(),
            Self::FlashAttn => "flash_attn".to_owned(),
            Self::Fa4Cute => "fa4_cute".to_owned(),
            Self::TorchSdpa => "torch_sdpa".to_owned(),
            Self::H3VsaSm100 => "h3_vsa_sm100".to_owned(),
            Self::Composite(backends) => backends
                .iter()
                .map(Self::as_name)
                .collect::<Vec<_>>()
                .join("+"),
        }
    }
}

impl std::str::FromStr for AttentionBackend {
    type Err = AttentionBackendParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.contains('+') {
            let backends = value
                .split('+')
                .map(str::parse)
                .collect::<Result<Vec<_>, _>>()?;
            if backends.len() < 2
                || backends
                    .iter()
                    .any(|backend| matches!(backend, Self::Auto | Self::Composite(_)))
            {
                return Err(AttentionBackendParseError(value.to_owned()));
            }
            return Ok(Self::Composite(backends));
        }
        match value {
            "auto" => Ok(Self::Auto),
            "trtllm_mha" => Ok(Self::TrtllmMha),
            "sgl_kernel" => Ok(Self::SglKernel),
            "flashinfer" => Ok(Self::FlashInfer),
            "flash_attn" => Ok(Self::FlashAttn),
            "fa4_cute" => Ok(Self::Fa4Cute),
            "torch_sdpa" => Ok(Self::TorchSdpa),
            "h3_vsa_sm100" => Ok(Self::H3VsaSm100),
            _ => Err(AttentionBackendParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for AttentionBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.as_name())
    }
}

impl Serialize for AttentionBackend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.as_name())
    }
}

impl<'de> Deserialize<'de> for AttentionBackend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported attention backend {0:?}")]
pub struct AttentionBackendParseError(String);

impl ForwardMode {
    pub const ALL: [Self; 12] = [
        Self::TokenExtend,
        Self::TokenDecode,
        Self::TokenVerify,
        Self::EncodeVision,
        Self::EncodeLatent,
        Self::TransferProduct,
        Self::TransferKvPublish,
        Self::TransferKvInstall,
        Self::MediaPrepare,
        Self::MediaDenoise,
        Self::Materialize,
        Self::MediaReconstruct,
    ];

    /// Whether a variant advances the authoritative request lineage.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::TokenExtend
                | Self::TokenDecode
                | Self::TokenVerify
                | Self::MediaPrepare
                | Self::MediaDenoise
                | Self::MediaReconstruct
        )
    }

    /// Whether this work leaf can execute only over a host-resolved semantic root.
    pub const fn requires_fixed_parent(self) -> bool {
        matches!(self, Self::TransferKvPublish)
    }

    /// The device execution domain that owns this work leaf.
    pub const fn domain(self) -> Domain {
        match self {
            Self::TokenDecode | Self::TokenVerify => Domain::Decode,
            Self::MediaPrepare
            | Self::MediaDenoise
            | Self::MediaReconstruct
            | Self::Materialize => Domain::Flow,
            Self::TokenExtend
            | Self::EncodeVision
            | Self::EncodeLatent
            | Self::TransferProduct
            | Self::TransferKvPublish
            | Self::TransferKvInstall => Domain::Prefill,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenExtend => "token_extend",
            Self::TokenDecode => "token_decode",
            Self::TokenVerify => "token_verify",
            Self::EncodeVision => "encode_vision",
            Self::EncodeLatent => "encode_latent",
            Self::TransferProduct => "transfer_product",
            Self::TransferKvPublish => "transfer_kv_publish",
            Self::TransferKvInstall => "transfer_kv_install",
            Self::MediaPrepare => "media_prepare",
            Self::MediaDenoise => "media_denoise",
            Self::Materialize => "materialize",
            Self::MediaReconstruct => "media_reconstruct",
        }
    }
}

// ---------------------------------------------------------------------------
// Version references
// ---------------------------------------------------------------------------

/// One exact state point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Point {
    /// A host-observed state point named by its producer-local index.
    Fixed { point_index: u32 },
    /// A device-selected point a successor may consume before host observation.
    Device {
        /// The producer-local point when selection is statically determined.
        point_index: u32,
        /// The producer's dynamic selection product for multi-point work.
        selected_point: Option<ProductRef>,
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
    pub fn admission_root(request_key: RequestKey, producer_op_id: OpId) -> Self {
        Self {
            request_key,
            producer_op_id,
            point: Point::Fixed { point_index: 0 },
        }
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self.point, Point::Fixed { .. })
    }

    pub fn validate(&self) -> ValidationResult<()> {
        match &self.point {
            Point::Fixed { .. } => {}
            Point::Device {
                point_index,
                selected_point,
            } => {
                if let Some(selected_point) = selected_point {
                    ensure_valid!(
                        *point_index == 0,
                        "a dynamic device version also declares a fixed point"
                    );
                    selected_point.validate()?;
                    ensure_valid!(
                        selected_point.request_key == self.request_key
                            && selected_point.producer_op_id == self.producer_op_id,
                        "device version selected point is not owned by its producer"
                    );
                    ensure_valid!(
                        selected_point.generation > 0,
                        "device version selected point has no logical generation"
                    );
                    ensure_valid!(
                        selected_point.kind == ProductKind::SelectedPoint
                            && matches!(
                                selected_point.storage_class,
                                StorageClass::DeviceTensor | StorageClass::RequestRelay
                            )
                            && selected_point.dtype == DType::U32
                            && selected_point.shape_bound.max_elements() == 1,
                        "device version does not name a scalar selected-point product"
                    );
                } else {
                    ensure_valid!(
                        *point_index > 0,
                        "a static device version must name a positive producer point"
                    );
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Operation
// ---------------------------------------------------------------------------

/// The device execution class used for static lane binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Domain {
    Prefill = 0,
    Decode = 1,
    Flow = 2,
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
    pub work: ForwardMode,
    pub route: RouteId,
    pub domain: Domain,
    pub advances_state: bool,
    pub bounds: Bounds,
    pub inputs: Vec<ProductRef>,
    pub outputs: Vec<ProductRef>,
    pub predicate: Option<ProductRef>,
    pub rng: Option<Rng>,
    pub control_seq: u64,
}

impl Operation {
    /// Seal a literally constructed operation with its derived state advance.
    pub fn sealed(mut self) -> Self {
        self.advances_state = self.work.advances_state();
        self
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.op_id.0 > 0, "operation id must be positive");
        ensure_valid!(
            self.domain == self.work.domain(),
            "operation domain is inconsistent with its work variant"
        );
        ensure_valid!(
            self.advances_state == self.work.advances_state(),
            "operation declares an advances_state inconsistent with its work variant"
        );
        self.parent.validate()?;
        ensure_valid!(
            self.parent.request_key == self.request_key,
            "operation parent belongs to another request lineage"
        );
        ensure_valid!(
            !self.work.requires_fixed_parent() || self.parent.is_fixed(),
            "operation requires a fixed semantic parent"
        );
        self.bounds_are_finite()?;
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        for output in &self.outputs {
            output.validate()?;
            ensure_valid!(
                output.request_key == self.request_key && output.producer_op_id == self.op_id,
                "an output product is not owned by its producing operation"
            );
            ensure_valid!(
                output.generation > 0,
                "an output product has no logical generation"
            );
            ensure_valid!(
                output.point_range.max_points <= self.bounds.max_points.max(1),
                "an output product exceeds the operation point bound"
            );
            match output.storage_class {
                StorageClass::LatentArena => ensure_valid!(
                    output.max_bytes() <= self.bounds.max_latent_bytes,
                    "a latent-arena output exceeds the operation latent-byte bound"
                ),
                StorageClass::HostStaging | StorageClass::PinnedOutput => ensure_valid!(
                    output.max_bytes() <= self.bounds.max_completion_bytes,
                    "a host-visible output exceeds the operation completion-byte bound"
                ),
                StorageClass::PagedKv => ensure_valid!(
                    output.max_bytes() <= self.bounds.max_transfer_bytes,
                    "a paged-KV output exceeds the operation transfer-byte bound"
                ),
                _ => {}
            }
            ensure_valid!(
                output_indices.insert(output.output_index),
                "operation repeats an output index"
            );
        }
        for input in &self.inputs {
            input.validate()?;
            ensure_valid!(
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
            ensure_valid!(
                predicate.request_key == self.request_key,
                "operation predicate belongs to another request lineage"
            );
            let continuation_token = predicate.kind == ProductKind::Token
                && predicate.dtype == DType::U32
                && predicate.shape_bound.max_elements() == 1;
            ensure_valid!(
                predicate.generation > 0
                    && matches!(
                        predicate.storage_class,
                        StorageClass::DeviceTensor | StorageClass::RequestRelay
                    )
                    && (predicate.kind == ProductKind::Completion || continuation_token),
                "operation predicate is not a generation-tagged device decision product"
            );
        }
        Ok(())
    }

    fn bounds_are_finite(&self) -> ValidationResult<()> {
        // Bounds are unsigned integers; the invariant enforced here is that a
        // state-advancing operation can advance by at least one point.
        if self.advances_state {
            ensure_valid!(
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
    pub kv_computed_len: u32,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaOutput {
    pub handle: String,
    pub bytes: u64,
}

/// The fixed-layout record a worker emits once for every operation, after its
/// copy event is query-ready and its pinned fields are validated on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelOutput {
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
    pub error_code: Option<ErrorCode>,
    pub timing_counters: TimingCounters,
    pub media_output: Option<MediaOutput>,
}

impl ModelOutput {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.op_id.0 > 0, "completion op id must be positive");
        ensure_valid!(
            self.logical_lengths.kv_visible_len <= self.logical_lengths.kv_computed_len,
            "completion selected KV length exceeds computed length"
        );
        ensure_valid!(
            self.completion_slot_generation > 0,
            "completion slot generation must be positive"
        );
        match self.status {
            OpStatus::Error => ensure_valid!(
                self.error_code.is_some(),
                "an error completion must carry an error code"
            ),
            OpStatus::Ok | OpStatus::Predicated => ensure_valid!(
                self.error_code.is_none(),
                "a non-error completion must not carry an error code"
            ),
        }
        if self.status == OpStatus::Predicated {
            ensure_valid!(
                self.token_span.len == 0
                    && self.committed_tokens.is_empty()
                    && self.product_generations.is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
            ensure_valid!(
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

    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::Commit {
                expected_parent,
                selected,
                ..
            } => {
                expected_parent.validate()?;
                selected.validate()?;
                ensure_valid!(
                    selected.is_fixed(),
                    "a commit control must select a fixed version"
                );
            }
            Self::Close { cutoff, .. } => {
                cutoff.validate()?;
                ensure_valid!(
                    cutoff.is_fixed(),
                    "a close control must name a fixed cutoff version"
                );
            }
            Self::Release { op_id, .. } => {
                ensure_valid!(op_id.0 > 0, "a release control must name a valid operation");
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NewRequest framing (session establishment)
// ---------------------------------------------------------------------------

/// Understanding-branch admission: invariant sampling policy, negative tokens,
/// terminal token ids, and the semantic position visible at admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UndAdmission {
    pub sampling: SamplingParams,
    pub negative_token_ids: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub initial_position: u32,
}

/// Generation-branch admission: image parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenAdmission {
    pub image: ImageParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    pub frame_count: u32,
    pub video_reconstruction_units: u32,
    pub audio_latent_frames: u32,
    pub prompt_tokens: u32,
    pub denoise_steps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaAdmission {
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub profile: MediaProfileId,
    pub geometry: MediaGeometry,
}

/// Session establishment framing. Carries the per-domain parameters a lineage
/// needs before its operations run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewRequest {
    pub request_key: RequestKey,
    /// Scheduler-assigned stable request-state row. Index zero is reserved for
    /// inactive graph padding and never identifies a live request.
    pub request_pool_idx: u32,
    pub und: Option<UndAdmission>,
    pub gen_admission: Option<GenAdmission>,
    pub media: Option<MediaAdmission>,
}

impl NewRequest {
    pub fn new(
        request_key: RequestKey,
        request_pool_idx: u32,
        und: Option<UndAdmission>,
        gen_admission: Option<GenAdmission>,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        ensure_valid!(
            und.is_some() || gen_admission.is_some(),
            "admission must declare an understanding or generation branch"
        );
        let admission = Self {
            request_key,
            request_pool_idx,
            und,
            gen_admission,
            media: None,
        };
        Ok(admission)
    }

    pub fn new_media(
        request_key: RequestKey,
        request_pool_idx: u32,
        media: MediaAdmission,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        let admission = Self {
            request_key,
            request_pool_idx,
            und: None,
            gen_admission: None,
            media: Some(media),
        };
        admission.validate()?;
        Ok(admission)
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "request-pool index must be positive"
        );
        ensure_valid!(
            self.und.is_some() || self.gen_admission.is_some() || self.media.is_some(),
            "admission must declare an understanding, generation, or media branch"
        );
        if let Some(und) = &self.und {
            und.sampling.validate()?;
            ensure_valid!(
                und.finish_token_ids
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
                "und admission finish token ids are not canonical"
            );
        }
        if let Some(branch) = &self.gen_admission {
            branch.image.validate()?;
        }
        if let Some(media) = &self.media {
            ensure_valid!(
                !media.prompt_token_ids.is_empty(),
                "media admission prompt tokens must not be empty"
            );
            ensure_valid!(
                media.geometry.frame_count >= 22
                    && media.geometry.frame_count % 17 == 5
                    && media.geometry.video_reconstruction_units
                        == (media.geometry.frame_count - 5) / 17
                    && u64::from(media.geometry.audio_latent_frames)
                        == uniserve_core::MediaGeometry::required_audio_latent_frames(
                            media.geometry.frame_count,
                        )
                    && media.geometry.prompt_tokens > 0
                    && usize::try_from(media.geometry.prompt_tokens).ok()
                        == Some(media.prompt_token_ids.len())
                    && media.geometry.denoise_steps == 4,
                "media admission geometry is invalid"
            );
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
    pub attention: AttentionRegime,
    pub shape_class: u64,
    pub operations: Vec<Operation>,
    /// Complete scheduler-owned logical-page mappings installed this step.
    pub block_tables: Vec<BlockTable>,
    /// Newly acquired physical pages that require worker-side initialization.
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// CPU row-packing metadata. One operation may contribute multiple rows.
    pub forward_rows: Vec<RowGeometry>,
    /// Complete scheduler-owned latent mappings for operations that address a
    /// generation trajectory.
    pub latent_placements: Vec<LatentPlacement>,
    /// Fixed-profile media reconstruction units owned by MediaReconstruct operations.
    pub reconstruction_placements: Vec<ReconstructionPlacement>,
}

/// One scheduler-owned request-slot block table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTable {
    pub request_pool_idx: u32,
    pub group_id: u32,
    pub page_ids: Vec<BlockId>,
    pub allocated_tokens: u32,
}

/// Physical pages newly acquired for one installed block table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePageAllocation {
    pub request_pool_idx: u32,
    pub group_id: u32,
    pub page_ids: Vec<BlockId>,
}

impl BlockTable {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "block table request slot must be positive"
        );
        ensure_valid!(
            self.page_ids.iter().all(|page| page.0 > 0)
                && self.page_ids.iter().collect::<HashSet<_>>().len() == self.page_ids.len(),
            "block table repeats a page or carries page zero"
        );
        ensure_valid!(
            !self.page_ids.is_empty() || self.allocated_tokens == 0,
            "empty block table carries allocated tokens"
        );
        Ok(())
    }
}

impl CachePageAllocation {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "cache-page allocation request slot must be positive"
        );
        ensure_valid!(
            !self.page_ids.is_empty()
                && self.page_ids.iter().all(|page| page.0 > 0)
                && self.page_ids.iter().collect::<HashSet<_>>().len() == self.page_ids.len(),
            "cache-page allocation is empty, repeats a page, or carries page zero"
        );
        Ok(())
    }
}

/// Row-aligned model-forward metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowGeometry {
    pub operation_index: u32,
    pub request_pool_index: u32,
    pub seq_len: u32,
    pub query_len: u32,
}

impl RowGeometry {
    pub fn validate(self, operation_count: usize) -> ValidationResult<()> {
        ensure_valid!(
            (self.operation_index as usize) < operation_count,
            "forward row operation index is outside its partition"
        );
        ensure_valid!(
            self.request_pool_index > 0,
            "forward row carries the reserved request slot"
        );
        ensure_valid!(
            self.query_len > 0,
            "forward row query length must be positive"
        );
        Ok(())
    }
}

/// One operation's complete scheduler-owned latent trajectory placement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPlacement {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub page_table: Vec<u32>,
    pub latent_units: u32,
    pub height: u32,
    pub width: u32,
    pub start_step: u32,
    pub step_count: u32,
}

impl LatentPlacement {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0,
            "latent placement operation id must be positive"
        );
        ensure_valid!(
            self.latent_units > 0 && self.height > 0 && self.width > 0,
            "latent placement geometry must be positive"
        );
        ensure_valid!(
            !self.page_table.is_empty()
                && self.page_table.iter().all(|page| *page > 0)
                && self.page_table.iter().collect::<HashSet<_>>().len() == self.page_table.len(),
            "latent placement page table is empty, repeats a page, or carries page zero"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionPlacement {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub kind: ReconstructionKind,
    pub start_unit: u32,
    pub unit_count: u32,
}

impl ReconstructionPlacement {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0,
            "reconstruction placement operation id must be positive"
        );
        ensure_valid!(
            self.unit_count > 0,
            "reconstruction placement unit count must be positive"
        );
        Ok(())
    }
}

impl BatchPartition {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.partition_id > 0, "batch partition id must be positive");
        ensure_valid!(
            self.submission_group > 0,
            "batch partition submission group must be positive"
        );
        ensure_valid!(
            self.collective_seq > 0,
            "batch partition collective sequence must be positive"
        );
        ensure_valid!(
            !self.operations.is_empty(),
            "batch partition must carry at least one operation"
        );
        for operation in &self.operations {
            operation.validate()?;
            ensure_valid!(
                operation.domain == self.domain && operation.route == self.route,
                "batch partition operation disagrees with its domain or route"
            );
        }
        let mut table_ids = HashSet::with_capacity(self.block_tables.len());
        for table in &self.block_tables {
            table.validate()?;
            ensure_valid!(
                table_ids.insert((table.request_pool_idx, table.group_id)),
                "batch partition repeats a block table"
            );
        }
        let tables = self
            .block_tables
            .iter()
            .map(|table| ((table.request_pool_idx, table.group_id), table))
            .collect::<HashMap<_, _>>();
        let mut allocation_ids = HashSet::with_capacity(self.new_cache_pages.len());
        for allocation in &self.new_cache_pages {
            allocation.validate()?;
            let identity = (allocation.request_pool_idx, allocation.group_id);
            ensure_valid!(
                allocation_ids.insert(identity),
                "batch partition repeats a cache-page allocation"
            );
            let table = tables.get(&identity).ok_or_else(|| {
                invalid_message!("cache-page allocation has no matching block table")
            })?;
            let table_pages = table.page_ids.iter().collect::<HashSet<_>>();
            ensure_valid!(
                allocation
                    .page_ids
                    .iter()
                    .all(|page| table_pages.contains(page)),
                "cache-page allocation is outside its block table"
            );
        }
        for row in &self.forward_rows {
            row.validate(self.operations.len())?;
        }
        let operations = self
            .operations
            .iter()
            .map(|operation| ((operation.request_key, operation.op_id), operation))
            .collect::<HashMap<_, _>>();
        let mut latent_ids = HashSet::with_capacity(self.latent_placements.len());
        let mut latent_pages = HashSet::new();
        for placement in &self.latent_placements {
            placement.validate()?;
            let identity = (placement.request_key, placement.op_id);
            ensure_valid!(
                latent_ids.insert(identity),
                "batch partition repeats a latent placement identity"
            );
            let operation = operations.get(&identity).ok_or_else(|| {
                invalid_message!("latent placement does not name a partition operation")
            })?;
            let addresses_trajectory = matches!(
                operation.work,
                ForwardMode::MediaPrepare | ForwardMode::MediaDenoise
            ) || operation
                .inputs
                .iter()
                .any(|reference| reference.kind == ProductKind::Latent);
            ensure_valid!(
                addresses_trajectory,
                "latent placement names an operation that does not address a trajectory"
            );
            ensure_valid!(
                placement
                    .page_table
                    .iter()
                    .all(|page| latent_pages.insert(*page)),
                "latent placements overlap physical pages"
            );
        }
        for operation in &self.operations {
            let needs_latent = matches!(
                operation.work,
                ForwardMode::MediaPrepare | ForwardMode::MediaDenoise
            ) || operation
                .inputs
                .iter()
                .any(|reference| reference.kind == ProductKind::Latent);
            ensure_valid!(
                !needs_latent || latent_ids.contains(&(operation.request_key, operation.op_id)),
                "operation that addresses a trajectory has no latent placement"
            );
        }
        let mut reconstruction_ids = HashSet::with_capacity(self.reconstruction_placements.len());
        for placement in &self.reconstruction_placements {
            placement.validate()?;
            let identity = (placement.request_key, placement.op_id);
            ensure_valid!(
                reconstruction_ids.insert(identity),
                "batch partition repeats a reconstruction placement identity"
            );
            let operation = operations.get(&identity).ok_or_else(|| {
                invalid_message!("reconstruction placement does not name a partition operation")
            })?;
            ensure_valid!(
                operation.work == ForwardMode::MediaReconstruct,
                "reconstruction placement does not name a media reconstruction operation"
            );
        }
        for operation in &self.operations {
            ensure_valid!(
                operation.work != ForwardMode::MediaReconstruct
                    || reconstruction_ids.contains(&(operation.request_key, operation.op_id)),
                "media reconstruction operation has no reconstruction placement"
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
    pub admissions: Vec<NewRequest>,
    pub partitions: Vec<BatchPartition>,
    pub controls: Vec<Control>,
    /// Host-supplied input product values matched by `ProductRef` identity.
    pub input_products: Vec<ProductPayload>,
}

impl Batch {
    pub fn new(step_id: u64, admissions: Vec<NewRequest>, partitions: Vec<BatchPartition>) -> Self {
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

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.partitions.is_empty() || !self.controls.is_empty(),
            "a submission batch must carry at least one operation or control"
        );
        let mut partition_ids = HashSet::with_capacity(self.partitions.len());
        let mut submission_groups: std::collections::HashMap<u32, Vec<&BatchPartition>> =
            std::collections::HashMap::new();
        for partition in &self.partitions {
            partition.validate()?;
            ensure_valid!(
                partition_ids.insert(partition.partition_id),
                "a submission batch repeats a partition id"
            );
            submission_groups
                .entry(partition.submission_group)
                .or_default()
                .push(partition);
        }
        for partitions in submission_groups.values() {
            ensure_valid!(
                partitions
                    .iter()
                    .filter(|partition| !partition.latent_placements.is_empty())
                    .count()
                    <= 1,
                "a physical submission group has multiple latent staging partitions"
            );
            let collective_seq = partitions[0].collective_seq;
            let attention = partitions[0].attention;
            let shape_class = partitions[0].shape_class;
            ensure_valid!(
                partitions.iter().all(|partition| {
                    partition.collective_seq == collective_seq
                        && partition.attention == attention
                        && partition.shape_class == shape_class
                }),
                "physical submission partitions disagree on attention, shape, or collective order"
            );
            if partitions.len() >= 2 {
                ensure_valid!(
                    partitions
                        .iter()
                        .map(|partition| partition.domain)
                        .collect::<HashSet<_>>()
                        .len()
                        == partitions.len(),
                    "a tensorized-mixed submission group repeats a domain"
                );
                ensure_valid!(
                    partitions
                        .iter()
                        .all(|partition| partition.route == partitions[0].route),
                    "a tensorized-mixed submission group spans route info"
                );
            }
        }
        // Depth one: at most one runnable operation per request per batch.
        let mut request_keys = HashSet::with_capacity(self.operation_count());
        for partition in &self.partitions {
            for operation in &partition.operations {
                ensure_valid!(
                    request_keys.insert(operation.request_key),
                    "a submission batch carries multiple operations for one request"
                );
            }
        }
        let mut admitted = HashSet::with_capacity(self.admissions.len());
        for admission in &self.admissions {
            admission.validate()?;
            ensure_valid!(
                admitted.insert(admission.request_key),
                "a submission batch carries a duplicate admission"
            );
            ensure_valid!(
                self.operations()
                    .any(|operation| operation.request_key == admission.request_key),
                "a submission batch admits a request without an operation"
            );
        }
        // A repeated control identity must carry exactly the same command.
        let mut control_identities: std::collections::HashMap<
            (RequestKey, Option<u64>, u8),
            Control,
        > = std::collections::HashMap::new();
        for control in &self.controls {
            control.validate()?;
            let identity = (
                control.request_key(),
                control.control_seq(),
                control.variant_index(),
            );
            if let Some(existing) = control_identities.get(&identity) {
                ensure_valid!(
                    existing == control,
                    "a submission batch reuses a control identity with different content"
                );
            } else {
                control_identities.insert(identity, control.clone());
            }
        }
        for payload in &self.input_products {
            payload.validate()?;
        }
        let declared_inputs = self
            .operations()
            .flat_map(|operation| operation.inputs.iter().chain(operation.predicate.iter()))
            .collect::<HashSet<_>>();
        for operation in self.operations() {
            for input in operation
                .inputs
                .iter()
                .filter(|input| input.storage_class == StorageClass::HostStaging)
            {
                ensure_valid!(
                    input.request_key == operation.request_key
                        && input.producer_op_id == operation.op_id,
                    "a host-staging input is not owned by its consuming operation"
                );
            }
        }
        let mut supplied_inputs = HashSet::with_capacity(self.input_products.len());
        for payload in &self.input_products {
            ensure_valid!(
                declared_inputs.contains(&payload.product),
                "an input product payload is not declared by any operation"
            );
            let transferred = is_transfer_descriptor(&payload.bytes);
            if payload.product.storage_class == StorageClass::HostStaging {
                ensure_valid!(
                    !transferred,
                    "host-staging input cannot carry a cross-stage transfer descriptor"
                );
            } else {
                ensure_valid!(
                    transferred && payload.bytes.len() <= MAX_TRANSFER_DESCRIPTOR_BYTES,
                    "cross-stage product input has an invalid transfer descriptor frame"
                );
            }
            ensure_valid!(
                supplied_inputs.insert(&payload.product),
                "a submission batch repeats an input product payload"
            );
            payload.validate_input_value()?;
        }
        for input in declared_inputs {
            if input.storage_class == StorageClass::HostStaging {
                ensure_valid!(
                    supplied_inputs.contains(input),
                    "a host-staging operation input has no product payload"
                );
            }
        }
        Ok(())
    }
}

/// One independently completed partition in a worker response frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionCompletion {
    pub partition_id: u32,
    pub completions: Vec<ModelOutput>,
    pub products: Vec<ProductPayload>,
    pub registration: RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<WorkerForwardStats>,
}

impl PartitionCompletion {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
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
    pub fn completions(&self) -> impl Iterator<Item = &ModelOutput> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.completions.iter())
    }

    pub fn products(&self) -> impl Iterator<Item = &ProductPayload> {
        self.partitions
            .iter()
            .flat_map(|partition| partition.products.iter())
    }

    pub fn validate(&self) -> ValidationResult<()> {
        let mut partition_ids = HashSet::with_capacity(self.partitions.len());
        for partition in &self.partitions {
            partition.validate()?;
            ensure_valid!(
                partition_ids.insert(partition.partition_id),
                "completion report repeats a partition id"
            );
        }
        Ok(())
    }
}
