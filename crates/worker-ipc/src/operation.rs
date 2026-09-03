//! Request identity, planned operations, static `NewRequest` state, and batches.

use super::*;

// ---------------------------------------------------------------------------
// Identities
// ---------------------------------------------------------------------------

/// `(authority_id, request_id, epoch)`. The epoch advances whenever an admitted
/// identity is reused, so no operation or product reference aliases across
/// requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    pub authority_id: u64,
    pub request_id: RequestId,
    pub epoch: u64,
}

impl RequestKey {
    pub const fn new(authority_id: u64, request_id: RequestId, epoch: u64) -> Self {
        Self {
            authority_id,
            request_id,
            epoch,
        }
    }
}

// ---------------------------------------------------------------------------
// Closed logical and physical operation algebras
// ---------------------------------------------------------------------------

/// Stable operation kinds understood by Runtime, Scheduler, and Router.
/// Physical worker modes are represented by [`RunKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum OpKind {
    ArExtend = 0,
    ArDecode = 1,
    ArVerify = 2,
    EncoderExecute = 3,
    DiffusionPrepare = 4,
    DiffusionStep = 5,
    DiffusionDecode = 6,
}

/// Worker-private execution modes carried by a physical [`Run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum RunKind {
    ArExtend = 0,
    ArDecode = 1,
    ArVerify = 2,
    EncoderVision = 3,
    EncoderLatent = 4,
    TransferProduct = 5,
    TransferKvPublish = 6,
    TransferKvInstall = 7,
    DiffusionPrepare = 8,
    DiffusionStep = 9,
    DiffusionFinalize = 10,
    DiffusionDecode = 11,
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

impl OpKind {
    pub const ALL: [Self; 7] = [
        Self::ArExtend,
        Self::ArDecode,
        Self::ArVerify,
        Self::EncoderExecute,
        Self::DiffusionPrepare,
        Self::DiffusionStep,
        Self::DiffusionDecode,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArExtend => "ar_extend",
            Self::ArDecode => "ar_decode",
            Self::ArVerify => "ar_verify",
            Self::EncoderExecute => "encoder_execute",
            Self::DiffusionPrepare => "diffusion_prepare",
            Self::DiffusionStep => "diffusion_step",
            Self::DiffusionDecode => "diffusion_decode",
        }
    }

    /// Physical modes enabled when a worker pool advertises this logical capability.
    pub const fn run_kinds(self) -> &'static [RunKind] {
        match self {
            Self::ArExtend => &[RunKind::ArExtend],
            Self::ArDecode => &[RunKind::ArDecode, RunKind::TransferKvPublish],
            Self::ArVerify => &[RunKind::ArVerify],
            Self::EncoderExecute => &[
                RunKind::EncoderVision,
                RunKind::EncoderLatent,
                RunKind::TransferProduct,
            ],
            Self::DiffusionPrepare => &[RunKind::DiffusionPrepare],
            Self::DiffusionStep => &[RunKind::DiffusionStep, RunKind::TransferKvInstall],
            Self::DiffusionDecode => &[RunKind::DiffusionDecode, RunKind::DiffusionFinalize],
        }
    }
}

impl RunKind {
    pub const ALL: [Self; 12] = [
        Self::ArExtend,
        Self::ArDecode,
        Self::ArVerify,
        Self::EncoderVision,
        Self::EncoderLatent,
        Self::TransferProduct,
        Self::TransferKvPublish,
        Self::TransferKvInstall,
        Self::DiffusionPrepare,
        Self::DiffusionStep,
        Self::DiffusionFinalize,
        Self::DiffusionDecode,
    ];

    pub const fn op_kind(self) -> OpKind {
        match self {
            Self::ArExtend => OpKind::ArExtend,
            Self::ArDecode => OpKind::ArDecode,
            Self::ArVerify => OpKind::ArVerify,
            Self::EncoderVision | Self::EncoderLatent => OpKind::EncoderExecute,
            Self::DiffusionPrepare => OpKind::DiffusionPrepare,
            Self::DiffusionStep => OpKind::DiffusionStep,
            Self::DiffusionFinalize | Self::DiffusionDecode => OpKind::DiffusionDecode,
            Self::TransferProduct => OpKind::EncoderExecute,
            Self::TransferKvPublish => OpKind::ArDecode,
            Self::TransferKvInstall => OpKind::DiffusionStep,
        }
    }

    /// Whether a variant advances the authoritative request lineage.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::ArExtend
                | Self::ArDecode
                | Self::ArVerify
                | Self::DiffusionPrepare
                | Self::DiffusionStep
                | Self::DiffusionDecode
        )
    }

    /// Whether this work leaf can execute only over a host-resolved semantic root.
    pub const fn requires_fixed_parent(self) -> bool {
        matches!(self, Self::TransferKvPublish)
    }

    /// The device execution domain that owns this work leaf.
    pub const fn domain(self) -> Domain {
        match self {
            Self::ArDecode | Self::ArVerify => Domain::Decode,
            Self::DiffusionPrepare
            | Self::DiffusionStep
            | Self::DiffusionDecode
            | Self::DiffusionFinalize => Domain::Flow,
            Self::ArExtend
            | Self::EncoderVision
            | Self::EncoderLatent
            | Self::TransferProduct
            | Self::TransferKvPublish
            | Self::TransferKvInstall => Domain::Prefill,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArExtend => "ar_extend",
            Self::ArDecode => "ar_decode",
            Self::ArVerify => "ar_verify",
            Self::EncoderVision => "encoder_vision",
            Self::EncoderLatent => "encoder_latent",
            Self::TransferProduct => "transfer_product",
            Self::TransferKvPublish => "transfer_kv_publish",
            Self::TransferKvInstall => "transfer_kv_install",
            Self::DiffusionPrepare => "diffusion_prepare",
            Self::DiffusionStep => "diffusion_step",
            Self::DiffusionFinalize => "diffusion_finalize",
            Self::DiffusionDecode => "diffusion_decode",
        }
    }
}

// ---------------------------------------------------------------------------
// Version references
// ---------------------------------------------------------------------------

/// One exact state point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CheckpointPoint {
    Fixed(u32),
    DeviceSelected,
}

/// Names one exact state point of a producer operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Checkpoint {
    pub op_id: OpId,
    pub point: CheckpointPoint,
}

impl Checkpoint {
    pub const fn admission_root(op_id: OpId) -> Self {
        Self {
            op_id,
            point: CheckpointPoint::Fixed(0),
        }
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self.point, CheckpointPoint::Fixed(_))
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0 || matches!(self.point, CheckpointPoint::Fixed(0)),
            "checkpoint operation id is invalid"
        );
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "family")]
pub enum OpPayload {
    Ar {
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    },
    Encoder {
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    },
    Diffusion {
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    },
    Transfer {
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    },
}

impl OpPayload {
    pub fn new(
        kind: RunKind,
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    ) -> Self {
        let fields = || (bounds, inputs, outputs, predicate, rng, control_seq);
        match kind {
            RunKind::ArExtend | RunKind::ArDecode | RunKind::ArVerify => {
                let (bounds, inputs, outputs, predicate, rng, control_seq) = fields();
                Self::Ar {
                    bounds,
                    inputs,
                    outputs,
                    predicate,
                    rng,
                    control_seq,
                }
            }
            RunKind::EncoderVision | RunKind::EncoderLatent => {
                let (bounds, inputs, outputs, predicate, rng, control_seq) = fields();
                Self::Encoder {
                    bounds,
                    inputs,
                    outputs,
                    predicate,
                    rng,
                    control_seq,
                }
            }
            RunKind::DiffusionPrepare
            | RunKind::DiffusionStep
            | RunKind::DiffusionDecode
            | RunKind::DiffusionFinalize => {
                let (bounds, inputs, outputs, predicate, rng, control_seq) = fields();
                Self::Diffusion {
                    bounds,
                    inputs,
                    outputs,
                    predicate,
                    rng,
                    control_seq,
                }
            }
            RunKind::TransferProduct | RunKind::TransferKvPublish | RunKind::TransferKvInstall => {
                let (bounds, inputs, outputs, predicate, rng, control_seq) = fields();
                Self::Transfer {
                    bounds,
                    inputs,
                    outputs,
                    predicate,
                    rng,
                    control_seq,
                }
            }
        }
    }

    fn fields(
        &self,
    ) -> (
        &Bounds,
        &[ProductRef],
        &[ProductRef],
        Option<&ProductRef>,
        Option<Rng>,
        u64,
    ) {
        match self {
            Self::Ar {
                bounds,
                inputs,
                outputs,
                predicate,
                rng,
                control_seq,
            }
            | Self::Encoder {
                bounds,
                inputs,
                outputs,
                predicate,
                rng,
                control_seq,
            }
            | Self::Diffusion {
                bounds,
                inputs,
                outputs,
                predicate,
                rng,
                control_seq,
            }
            | Self::Transfer {
                bounds,
                inputs,
                outputs,
                predicate,
                rng,
                control_seq,
            } => (
                bounds,
                inputs,
                outputs,
                predicate.as_ref(),
                *rng,
                *control_seq,
            ),
        }
    }

    pub const fn family_matches(&self, kind: RunKind) -> bool {
        matches!(
            (self, kind),
            (
                Self::Ar { .. },
                RunKind::ArExtend | RunKind::ArDecode | RunKind::ArVerify
            ) | (
                Self::Encoder { .. },
                RunKind::EncoderVision | RunKind::EncoderLatent
            ) | (
                Self::Diffusion { .. },
                RunKind::DiffusionPrepare
                    | RunKind::DiffusionStep
                    | RunKind::DiffusionDecode
                    | RunKind::DiffusionFinalize
            ) | (
                Self::Transfer { .. },
                RunKind::TransferProduct | RunKind::TransferKvPublish | RunKind::TransferKvInstall
            )
        )
    }
}

/// One immutable unit of registered work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub parent: Checkpoint,
    pub kind: RunKind,
    pub payload: OpPayload,
}

impl Operation {
    pub const fn sealed(self) -> Self {
        self
    }

    pub const fn domain(&self) -> Domain {
        self.kind.domain()
    }
    pub const fn advances_state(&self) -> bool {
        self.kind.advances_state()
    }
    pub fn bounds(&self) -> &Bounds {
        self.payload.fields().0
    }
    pub fn inputs(&self) -> &[ProductRef] {
        self.payload.fields().1
    }
    pub fn outputs(&self) -> &[ProductRef] {
        self.payload.fields().2
    }
    pub fn inputs_mut(&mut self) -> &mut Vec<ProductRef> {
        match &mut self.payload {
            OpPayload::Ar { inputs, .. }
            | OpPayload::Encoder { inputs, .. }
            | OpPayload::Diffusion { inputs, .. }
            | OpPayload::Transfer { inputs, .. } => inputs,
        }
    }
    pub fn outputs_mut(&mut self) -> &mut Vec<ProductRef> {
        match &mut self.payload {
            OpPayload::Ar { outputs, .. }
            | OpPayload::Encoder { outputs, .. }
            | OpPayload::Diffusion { outputs, .. }
            | OpPayload::Transfer { outputs, .. } => outputs,
        }
    }
    pub fn predicate(&self) -> Option<&ProductRef> {
        self.payload.fields().3
    }
    pub fn set_predicate(&mut self, value: Option<ProductRef>) {
        match &mut self.payload {
            OpPayload::Ar { predicate, .. }
            | OpPayload::Encoder { predicate, .. }
            | OpPayload::Diffusion { predicate, .. }
            | OpPayload::Transfer { predicate, .. } => *predicate = value,
        }
    }
    pub fn rng(&self) -> Option<Rng> {
        self.payload.fields().4
    }
    pub fn control_seq(&self) -> u64 {
        self.payload.fields().5
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.op_id.0 > 0, "operation id must be positive");
        ensure_valid!(
            self.payload.family_matches(self.kind),
            "operation payload family disagrees with its kind"
        );
        self.parent.validate()?;
        ensure_valid!(
            !self.kind.requires_fixed_parent() || self.parent.is_fixed(),
            "operation requires a fixed semantic parent"
        );
        self.bounds_are_finite()?;
        let mut output_indices = HashSet::with_capacity(self.outputs().len());
        for output in self.outputs() {
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
                output.point_range.max_points <= self.bounds().max_points.max(1),
                "an output product exceeds the operation point bound"
            );
            match output.storage_class {
                StorageClass::LatentArena => ensure_valid!(
                    output.max_bytes() <= self.bounds().max_latent_bytes,
                    "a latent-arena output exceeds the operation latent-byte bound"
                ),
                StorageClass::HostStaging | StorageClass::PinnedOutput => ensure_valid!(
                    output.max_bytes() <= self.bounds().max_completion_bytes,
                    "a host-visible output exceeds the operation completion-byte bound"
                ),
                StorageClass::PagedKv => ensure_valid!(
                    output.max_bytes() <= self.bounds().max_transfer_bytes,
                    "a paged-KV output exceeds the operation transfer-byte bound"
                ),
                _ => {}
            }
            ensure_valid!(
                output_indices.insert(output.output_index),
                "operation repeats an output index"
            );
        }
        for input in self.inputs() {
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
        if let Some(predicate) = self.predicate() {
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
        if self.advances_state() {
            ensure_valid!(
                self.bounds().max_points >= 1,
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
#[serde(rename_all = "snake_case", tag = "transport", content = "value")]
pub enum ArtifactHandle {
    PosixShm { name: String },
}

impl ArtifactHandle {
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::PosixShm { name } => ensure_valid!(
                !name.is_empty() && !name.contains('/'),
                "POSIX shared-memory artifact name is invalid"
            ),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaOutput {
    pub handle: ArtifactHandle,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResultData {
    pub logical_lengths: LogicalLengths,
    pub token_span: TokenSpan,
    pub committed_tokens: Vec<u32>,
    pub finish_flags: FinishFlags,
    pub media_output: Option<MediaOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DiffusionResult {
    pub data: ResultData,
    pub next_cursor: u32,
    pub done: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "family", content = "value")]
pub enum ResultPayload {
    Ar(ResultData),
    Encoder(ResultData),
    Diffusion(DiffusionResult),
    Transfer(ResultData),
}

impl ResultPayload {
    pub const fn family_matches(&self, kind: RunKind) -> bool {
        matches!(
            (self, kind),
            (
                Self::Ar(_),
                RunKind::ArExtend | RunKind::ArDecode | RunKind::ArVerify
            ) | (
                Self::Encoder(_),
                RunKind::EncoderVision | RunKind::EncoderLatent
            ) | (
                Self::Diffusion(_),
                RunKind::DiffusionPrepare
                    | RunKind::DiffusionStep
                    | RunKind::DiffusionDecode
                    | RunKind::DiffusionFinalize
            ) | (
                Self::Transfer(_),
                RunKind::TransferProduct | RunKind::TransferKvPublish | RunKind::TransferKvInstall
            )
        )
    }

    pub fn for_kind(kind: RunKind, data: ResultData) -> Self {
        match kind {
            RunKind::ArExtend | RunKind::ArDecode | RunKind::ArVerify => Self::Ar(data),
            RunKind::EncoderVision | RunKind::EncoderLatent => Self::Encoder(data),
            RunKind::DiffusionPrepare
            | RunKind::DiffusionStep
            | RunKind::DiffusionDecode
            | RunKind::DiffusionFinalize => Self::Diffusion(DiffusionResult {
                done: data.finish_flags.eos
                    || data.finish_flags.length
                    || data.finish_flags.stop
                    || data.media_output.is_some(),
                data,
                next_cursor: 0,
            }),
            RunKind::TransferProduct | RunKind::TransferKvPublish | RunKind::TransferKvInstall => {
                Self::Transfer(data)
            }
        }
    }

    fn data(&self) -> &ResultData {
        match self {
            Self::Ar(data) | Self::Encoder(data) | Self::Transfer(data) => data,
            Self::Diffusion(result) => &result.data,
        }
    }

    pub fn data_mut(&mut self) -> &mut ResultData {
        match self {
            Self::Ar(data) | Self::Encoder(data) | Self::Transfer(data) => data,
            Self::Diffusion(result) => &mut result.data,
        }
    }
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
    pub product_generations: Vec<u32>,
    pub error_code: Option<ErrorCode>,
    pub timing_counters: TimingCounters,
    pub payload: ResultPayload,
}

impl ModelOutput {
    pub fn logical_lengths(&self) -> &LogicalLengths {
        &self.payload.data().logical_lengths
    }
    pub fn logical_lengths_mut(&mut self) -> &mut LogicalLengths {
        &mut self.payload.data_mut().logical_lengths
    }
    pub fn token_span(&self) -> TokenSpan {
        self.payload.data().token_span
    }
    pub fn token_span_mut(&mut self) -> &mut TokenSpan {
        &mut self.payload.data_mut().token_span
    }
    pub fn committed_tokens(&self) -> &[u32] {
        &self.payload.data().committed_tokens
    }
    pub fn committed_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.payload.data_mut().committed_tokens
    }
    pub fn finish_flags(&self) -> FinishFlags {
        self.payload.data().finish_flags
    }
    pub fn finish_flags_mut(&mut self) -> &mut FinishFlags {
        &mut self.payload.data_mut().finish_flags
    }
    pub fn media_output(&self) -> Option<&MediaOutput> {
        self.payload.data().media_output.as_ref()
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.op_id.0 > 0, "completion op id must be positive");
        ensure_valid!(
            self.logical_lengths().kv_visible_len <= self.logical_lengths().kv_computed_len,
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
                self.token_span().len == 0
                    && self.committed_tokens().is_empty()
                    && self.product_generations.is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
            ensure_valid!(
                !self.finish_flags().eos
                    && !self.finish_flags().length
                    && !self.finish_flags().stop,
                "a predicated completion must not select a terminal outcome"
            );
        }
        if let Some(output) = self.media_output() {
            output.handle.validate()?;
            ensure_valid!(
                output.bytes > 0,
                "media output byte length must be positive"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Run commands
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

/// Ordered request and buffer commands carried with logical work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum BatchCommand {
    /// Establish one request lineage before its first operation on a pool.
    Start { request: NewRequest },
    /// Commit a selected fixed point and expose public output up to a limit.
    Commit {
        request_key: RequestKey,
        control_seq: u64,
        expected_parent: Checkpoint,
        selected: Checkpoint,
        public_event_limit: u64,
        disposition: Disposition,
    },
    /// Finish a lineage at a fixed cutoff, dominating every uncommitted descendant.
    Finish {
        request_key: RequestKey,
        control_seq: u64,
        cutoff: Checkpoint,
        reason: CloseReason,
    },
    /// Free one exact persistent product after its final consumer.
    Free { buffer: BufferId },
}

impl BatchCommand {
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Start { request } => request.request_key,
            Self::Commit { request_key, .. } | Self::Finish { request_key, .. } => *request_key,
            Self::Free { buffer } => buffer.owner,
        }
    }

    pub const fn variant_index(&self) -> u8 {
        match self {
            Self::Start { .. } => 0,
            Self::Commit { .. } => 1,
            Self::Finish { .. } => 2,
            Self::Free { .. } => 3,
        }
    }

    /// The `control_seq` for commit and close; releases carry no sequence.
    pub const fn control_seq(&self) -> Option<u64> {
        match self {
            Self::Commit { control_seq, .. } | Self::Finish { control_seq, .. } => {
                Some(*control_seq)
            }
            Self::Start { .. } | Self::Free { .. } => None,
        }
    }

    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::Start { request } => request.validate()?,
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
            Self::Finish { cutoff, .. } => {
                cutoff.validate()?;
                ensure_valid!(
                    cutoff.is_fixed(),
                    "a finish command must name a fixed cutoff version"
                );
            }
            Self::Free { buffer } => buffer.validate()?,
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NewRequest framing (request start)
// ---------------------------------------------------------------------------

/// Autoregressive request parameters fixed for the worker request lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArRequestParams {
    pub sampling: SamplingParams,
    pub negative_token_ids: Vec<u32>,
    pub finish_token_ids: Vec<u32>,
    pub initial_position: u32,
}

/// Unified-multimodal request parameters fixed for the worker request lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UmmRequestParams {
    pub image: ImageParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    pub frame_count: u32,
    pub decode_units: u32,
    pub prompt_tokens: u32,
    pub denoise_steps: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequestParams {
    pub prompt_token_ids: Vec<u32>,
    pub seed: u64,
    pub geometry: MediaGeometry,
}

/// Request-start framing. Carries the per-domain parameters a request
/// needs before its operations run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewRequest {
    pub request_key: RequestKey,
    /// Scheduler-assigned stable request-state row. Index zero is reserved for
    /// inactive graph padding and never identifies a live request.
    pub request_pool_idx: u32,
    pub ar: Option<ArRequestParams>,
    pub umm: Option<UmmRequestParams>,
    pub diffusion: Option<DiffusionRequestParams>,
}

impl NewRequest {
    pub fn new(
        request_key: RequestKey,
        request_pool_idx: u32,
        ar: Option<ArRequestParams>,
        umm: Option<UmmRequestParams>,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        ensure_valid!(
            ar.is_some() || umm.is_some(),
            "request start must declare autoregressive or unified-multimodal parameters"
        );
        let admission = Self {
            request_key,
            request_pool_idx,
            ar,
            umm,
            diffusion: None,
        };
        Ok(admission)
    }

    pub fn new_media(
        request_key: RequestKey,
        request_pool_idx: u32,
        diffusion: DiffusionRequestParams,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        let admission = Self {
            request_key,
            request_pool_idx,
            ar: None,
            umm: None,
            diffusion: Some(diffusion),
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
            self.ar.is_some() || self.umm.is_some() || self.diffusion.is_some(),
            "request start must declare one runtime-family parameter set"
        );
        if let Some(ar) = &self.ar {
            ar.sampling.validate()?;
            ensure_valid!(
                ar.finish_token_ids.windows(2).all(|pair| pair[0] < pair[1]),
                "autoregressive finish token ids are not canonical"
            );
        }
        if let Some(branch) = &self.umm {
            branch.image.validate()?;
        }
        if let Some(diffusion) = &self.diffusion {
            ensure_valid!(
                !diffusion.prompt_token_ids.is_empty(),
                "diffusion prompt tokens must not be empty"
            );
            ensure_valid!(
                diffusion.geometry.frame_count > 0
                    && diffusion.geometry.decode_units > 0
                    && diffusion.geometry.prompt_tokens > 0
                    && usize::try_from(diffusion.geometry.prompt_tokens).ok()
                        == Some(diffusion.prompt_token_ids.len())
                    && diffusion.geometry.denoise_steps > 0,
                "diffusion request geometry is invalid"
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Run and response framing
// ---------------------------------------------------------------------------

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
            "forward row operation index is outside its lane"
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
pub struct DecodePlacement {
    pub request_key: RequestKey,
    pub op_id: OpId,
    pub cursor: u32,
    pub max_units: u32,
}

impl DecodePlacement {
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0,
            "decode placement operation id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode placement unit bound must be positive"
        );
        Ok(())
    }
}

/// Scheduler-selected address span for one cross-operation persistent buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferPlacement {
    pub buffer: BufferId,
    pub offset: u64,
    pub bytes: u64,
}

impl BufferPlacement {
    pub fn validate(self) -> ValidationResult<()> {
        self.buffer.validate()?;
        ensure_valid!(
            self.bytes > 0,
            "buffer placement byte extent must be positive"
        );
        ensure_valid!(
            self.offset.checked_add(self.bytes).is_some(),
            "buffer placement span overflows"
        );
        Ok(())
    }
}

/// One executor-produced physical worker invocation. Execution domains,
/// attention selection, and captured buckets are derived by the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub batch_id: u64,
    pub run_id: u64,
    pub collective_seq: u64,
    pub operations: Vec<Operation>,
    pub block_tables: Vec<BlockTable>,
    pub new_cache_pages: Vec<CachePageAllocation>,
    pub forward_rows: Vec<RowGeometry>,
    pub latent_placements: Vec<LatentPlacement>,
    pub decode_placements: Vec<DecodePlacement>,
    pub buffer_placements: Vec<BufferPlacement>,
    pub commands: Vec<BatchCommand>,
    /// Host-supplied input product values matched by `ProductRef` identity.
    pub input_products: Vec<ProductPayload>,
}

impl Run {
    pub fn new(batch_id: u64, admissions: Vec<NewRequest>, operations: Vec<Operation>) -> Self {
        Self {
            batch_id,
            run_id: batch_id,
            collective_seq: batch_id.max(1),
            operations,
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward_rows: Vec::new(),
            latent_placements: Vec::new(),
            decode_placements: Vec::new(),
            buffer_placements: Vec::new(),
            commands: admissions
                .into_iter()
                .map(|request| BatchCommand::Start { request })
                .collect(),
            input_products: Vec::new(),
        }
    }

    pub fn operations(&self) -> impl Iterator<Item = &Operation> {
        self.operations.iter()
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn with_commands(mut self, commands: Vec<BatchCommand>) -> Self {
        self.commands.extend(commands);
        self
    }

    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request),
            _ => None,
        })
    }

    pub fn with_input_products(mut self, input_products: Vec<ProductPayload>) -> Self {
        self.input_products = input_products;
        self
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.operations.is_empty() || !self.commands.is_empty(),
            "a submission batch must carry at least one operation or control"
        );
        ensure_valid!(
            self.collective_seq > 0,
            "run collective sequence must be positive"
        );
        for operation in &self.operations {
            operation.validate()?;
        }
        let operations = self
            .operations
            .iter()
            .map(|operation| ((operation.request_key, operation.op_id), operation))
            .collect::<HashMap<_, _>>();
        let mut table_ids = HashSet::with_capacity(self.block_tables.len());
        for table in &self.block_tables {
            table.validate()?;
            ensure_valid!(
                table_ids.insert((table.request_pool_idx, table.group_id)),
                "run repeats a block table"
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
                "run repeats a cache-page allocation"
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
        let mut latent_ids = HashSet::with_capacity(self.latent_placements.len());
        let mut latent_pages = HashSet::new();
        for placement in &self.latent_placements {
            placement.validate()?;
            let identity = (placement.request_key, placement.op_id);
            ensure_valid!(
                latent_ids.insert(identity),
                "run repeats a latent placement identity"
            );
            let operation = operations.get(&identity).ok_or_else(|| {
                invalid_message!("latent placement does not name a run operation")
            })?;
            let addresses_trajectory = matches!(
                operation.kind,
                RunKind::DiffusionPrepare | RunKind::DiffusionStep
            ) || operation
                .inputs()
                .iter()
                .any(|value| value.kind == ProductKind::Latent);
            ensure_valid!(
                addresses_trajectory,
                "latent placement names work without a trajectory"
            );
            ensure_valid!(
                placement
                    .page_table
                    .iter()
                    .all(|page| latent_pages.insert(*page)),
                "run latent placements overlap physical pages"
            );
        }
        for operation in &self.operations {
            let needs_latent = matches!(
                operation.kind,
                RunKind::DiffusionPrepare | RunKind::DiffusionStep
            ) || operation
                .inputs()
                .iter()
                .any(|value| value.kind == ProductKind::Latent);
            ensure_valid!(
                !needs_latent || latent_ids.contains(&(operation.request_key, operation.op_id)),
                "operation that addresses a trajectory has no latent placement"
            );
        }
        let mut decode_ids = HashSet::with_capacity(self.decode_placements.len());
        for placement in &self.decode_placements {
            placement.validate()?;
            let identity = (placement.request_key, placement.op_id);
            ensure_valid!(
                decode_ids.insert(identity),
                "run repeats a decode placement identity"
            );
            let operation = operations.get(&identity).ok_or_else(|| {
                invalid_message!("decode placement does not name a run operation")
            })?;
            ensure_valid!(
                matches!(
                    operation.kind,
                    RunKind::DiffusionDecode | RunKind::DiffusionFinalize
                ),
                "decode placement does not name media decode work"
            );
        }
        let mut buffer_ids = HashSet::with_capacity(self.buffer_placements.len());
        let mut buffer_spans = self
            .buffer_placements
            .iter()
            .map(|placement| {
                placement.validate()?;
                ensure_valid!(
                    buffer_ids.insert(placement.buffer),
                    "run repeats a buffer placement identity"
                );
                Ok((placement.offset, placement.offset + placement.bytes))
            })
            .collect::<ValidationResult<Vec<_>>>()?;
        buffer_spans.sort_unstable();
        ensure_valid!(
            buffer_spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
            "run buffer placements overlap"
        );
        for operation in &self.operations {
            for output in operation
                .outputs()
                .iter()
                .filter(|output| output.uses_persistent_buffer())
            {
                let placement = self
                    .buffer_placements
                    .iter()
                    .find(|placement| placement.buffer == output.buffer_id())
                    .ok_or_else(|| {
                        invalid_message!("persistent operation output has no buffer placement")
                    })?;
                ensure_valid!(
                    placement.bytes >= output.max_bytes(),
                    "buffer placement is smaller than its declared output"
                );
            }
        }
        // Depth one: at most one runnable operation per request per batch.
        let mut request_keys = HashSet::with_capacity(self.operation_count());
        for operation in &self.operations {
            ensure_valid!(
                request_keys.insert(operation.request_key),
                "a submission batch carries multiple operations for one request"
            );
        }
        let mut admitted = HashSet::new();
        for admission in self.admissions() {
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
        let mut command_identities: std::collections::HashMap<
            (RequestKey, Option<u64>, u8, Option<u32>),
            BatchCommand,
        > = std::collections::HashMap::new();
        for command in &self.commands {
            command.validate()?;
            let identity = (
                command.request_key(),
                command.control_seq(),
                command.variant_index(),
                match command {
                    BatchCommand::Free { buffer } => Some(buffer.generation),
                    _ => None,
                },
            );
            if let Some(existing) = command_identities.get(&identity) {
                ensure_valid!(
                    existing == command,
                    "a submission batch reuses a command identity with different content"
                );
            } else {
                command_identities.insert(identity, command.clone());
            }
        }
        for payload in &self.input_products {
            payload.validate()?;
        }
        let declared_inputs = self
            .operations()
            .flat_map(|operation| {
                operation
                    .inputs()
                    .iter()
                    .chain(operation.predicate().into_iter())
            })
            .collect::<HashSet<_>>();
        for operation in self.operations() {
            for input in operation
                .inputs()
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
            if payload.product.storage_class == StorageClass::HostStaging {
                ensure_valid!(
                    matches!(payload.value, InlineValue::Bytes(_)),
                    "host-staging input cannot carry a cross-stage transfer handle"
                );
            } else {
                ensure_valid!(
                    matches!(payload.value, InlineValue::Transfer(_)),
                    "cross-stage product input has no transfer handle"
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

/// One independently ready subset of a physical run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunResult {
    pub batch_id: u64,
    pub run_id: u64,
    pub completions: Vec<ModelOutput>,
    pub products: Vec<ProductPayload>,
    pub registration: RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<WorkerForwardStats>,
    pub done: bool,
}

impl RunResult {
    pub fn completions(&self) -> impl Iterator<Item = &ModelOutput> {
        self.completions.iter()
    }

    pub fn products(&self) -> impl Iterator<Item = &ProductPayload> {
        self.products.iter()
    }

    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            !self.completions.is_empty() || self.done,
            "an empty completion report must terminate its run"
        );
        let mut identities = HashSet::with_capacity(self.completions.len());
        for completion in &self.completions {
            completion.validate()?;
            ensure_valid!(
                identities.insert((completion.request_key, completion.op_id)),
                "completion report repeats an operation"
            );
        }
        for payload in &self.products {
            payload.validate_output_value()?;
        }
        Ok(())
    }
}
