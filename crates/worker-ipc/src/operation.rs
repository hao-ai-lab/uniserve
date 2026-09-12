//! Request identities, operation descriptors, admissions, and execution batches.

use super::*;

/// `(authority_id, request_id, epoch)`. The epoch advances whenever an admitted
/// identity is reused, so no operation or product reference aliases across
/// requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    /// Authority namespace that allocates request identifiers.
    pub authority_id: u64,
    /// Identifier assigned to the request within its authority namespace.
    pub request_id: RequestId,
    /// Admission generation that distinguishes reuse of the same request identifier.
    pub epoch: u64,
}

impl RequestKey {
    /// Constructs an identity from authority, request, and admission epoch.
    pub const fn new(authority_id: u64, request_id: RequestId, epoch: u64) -> Self {
        Self {
            authority_id,
            request_id,
            epoch,
        }
    }
}

/// Worker-private execution modes carried by a physical [`Run`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum OpCode {
    /// Extends an autoregressive prefix with prompt tokens.
    ArExtend = 0,
    /// Produces the next autoregressive token.
    ArDecode = 1,
    /// Verifies speculative autoregressive tokens.
    ArVerify = 2,
    /// Encodes image input into vision features.
    EncoderVision = 3,
    /// Encodes input into latent features.
    EncoderLatent = 4,
    /// Publishes a product for another execution stage.
    TransferProduct = 5,
    /// Publishes paged KV state for another worker pool.
    TransferKvPublish = 6,
    /// Installs published paged KV state in the destination pool.
    TransferKvInstall = 7,
    /// Initializes a diffusion trajectory.
    DiffusionPrepare = 8,
    /// Advances a diffusion trajectory.
    DiffusionStep = 9,
    /// Finalizes a diffusion trajectory for decoding.
    DiffusionFinalize = 10,
    /// Decodes a completed diffusion trajectory into media.
    DiffusionDecode = 11,
    /// Encodes prompt tokens into a conditioning tensor.
    EncoderText = 12,
    /// Consumes decoded media units into their ordered output stream.
    MediaAppend = 13,
}

/// Configured or resolved worker attention implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttentionBackend {
    /// Lets the worker select a compatible backend.
    Auto,
    /// TensorRT-LLM multi-head attention.
    TrtllmMha,
    /// SGLang attention kernel.
    SglKernel,
    /// FlashInfer attention implementation.
    FlashInfer,
    /// FlashAttention implementation.
    FlashAttn,
    /// CuTe DSL FlashAttention 4 implementation.
    Fa4Cute,
    /// PyTorch scaled dot-product attention.
    TorchSdpa,
    /// FastH3 variable-sparse attention for SM100 devices.
    H3VsaSm100,
    /// Ordered composition of two or more concrete backends.
    Composite(Vec<AttentionBackend>),
}

impl AttentionBackend {
    /// Returns the stable configuration name for this backend selection.
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
    /// Error returned when a backend expression is unsupported or malformed.
    type Err = AttentionBackendParseError;

    /// Parses a single backend name or an ordered composite expression.
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
    /// Formats the backend using its stable configuration name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.as_name())
    }
}

impl Serialize for AttentionBackend {
    /// Serializes the backend as its stable configuration name.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.as_name())
    }
}

impl<'de> Deserialize<'de> for AttentionBackend {
    /// Deserializes and validates a backend configuration name.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Error returned for an unsupported attention backend name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported attention backend {0:?}")]
pub struct AttentionBackendParseError(String);

impl OpCode {
    /// Physical run kinds supported by the protocol.
    pub const ALL: [Self; 14] = [
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
        Self::EncoderText,
        Self::MediaAppend,
    ];

    /// Returns whether this variant advances the authoritative request lineage.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::ArExtend
                | Self::ArDecode
                | Self::ArVerify
                | Self::DiffusionPrepare
                | Self::DiffusionStep
        )
    }

    /// Returns whether this work requires a host-resolved semantic parent.
    pub const fn requires_fixed_parent(self) -> bool {
        matches!(self, Self::TransferKvPublish)
    }

    /// Returns the device execution domain that owns this work leaf.
    pub const fn domain(self) -> Domain {
        match self {
            Self::ArDecode | Self::ArVerify => Domain::Decode,
            Self::DiffusionPrepare
            | Self::DiffusionStep
            | Self::DiffusionDecode
            | Self::DiffusionFinalize
            | Self::MediaAppend => Domain::Flow,
            Self::ArExtend
            | Self::EncoderText
            | Self::EncoderVision
            | Self::EncoderLatent
            | Self::TransferProduct
            | Self::TransferKvPublish
            | Self::TransferKvInstall => Domain::Prefill,
        }
    }

    /// Returns the stable wire name for this physical run kind.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArExtend => "ar_extend",
            Self::ArDecode => "ar_decode",
            Self::ArVerify => "ar_verify",
            Self::EncoderVision => "encoder_vision",
            Self::EncoderText => "encoder_text",
            Self::EncoderLatent => "encoder_latent",
            Self::TransferProduct => "transfer_product",
            Self::TransferKvPublish => "transfer_kv_publish",
            Self::TransferKvInstall => "transfer_kv_install",
            Self::DiffusionPrepare => "diffusion_prepare",
            Self::DiffusionStep => "diffusion_step",
            Self::DiffusionFinalize => "diffusion_finalize",
            Self::DiffusionDecode => "diffusion_decode",
            Self::MediaAppend => "media_append",
        }
    }
}

/// One exact state point.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum CheckpointPoint {
    /// Host-selected semantic point.
    Fixed(u32),
    /// Point selected by device execution.
    DeviceSelected,
}

/// Names one exact state point of a producer operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Operation that produces the checkpoint.
    pub op_id: OpId,
    /// Semantic point within the producing operation.
    pub point: CheckpointPoint,
}

impl Checkpoint {
    /// Constructs the fixed admission-root checkpoint for an operation.
    pub const fn admission_root(op_id: OpId) -> Self {
        Self {
            op_id,
            point: CheckpointPoint::Fixed(0),
        }
    }

    /// Returns whether this checkpoint names a host-known fixed point.
    pub fn is_fixed(&self) -> bool {
        matches!(self.point, CheckpointPoint::Fixed(_))
    }

    /// Validates checkpoint identity and point constraints.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0 || matches!(self.point, CheckpointPoint::Fixed(0)),
            "checkpoint operation id is invalid"
        );
        Ok(())
    }
}

/// The device execution class used for static lane binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Domain {
    /// Prompt and encoder work that expands request state.
    Prefill = 0,
    /// Latency-sensitive autoregressive token work.
    Decode = 1,
    /// Diffusion trajectory work.
    Flow = 2,
}

/// Hard resource maxima the scheduler reserves before an operation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
    /// Maximum semantic points the operation may advance.
    pub max_points: u32,
    /// Maximum tokens the operation may process or produce.
    pub max_tokens: u32,
    /// Maximum paged-KV blocks the operation may consume.
    pub max_kv_pages: u32,
    /// Maximum latent storage in bytes.
    pub max_latent_bytes: u64,
    /// Maximum host-visible completion data in bytes.
    pub max_completion_bytes: u64,
    /// Maximum cross-stage transfer data in bytes.
    pub max_transfer_bytes: u64,
}

/// How the common sampler consumes deterministic random draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DrawLayout {
    /// One draw used to sample a target token.
    TargetSampling = 0,
    /// Draw sequence used to propose speculative tokens.
    SpeculativeProposal = 1,
    /// Draw sequence used to generate diffusion noise.
    FlowNoise = 2,
}

/// Deterministic random-draw coordinates for one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rng {
    /// Request-level random seed.
    pub seed: u64,
    /// First semantic draw index assigned to the operation.
    pub semantic_index_base: u64,
    /// Mapping from semantic work to deterministic draws.
    pub draw_layout: DrawLayout,
}

/// Executable tag, data dependencies and authorized bounds for a bounded operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpPayload {
    pub code: OpCode,
    pub bounds: Bounds,
    pub inputs: Vec<ProductRef>,
    pub outputs: Vec<ProductRef>,
    pub predicate: Option<ProductRef>,
    pub rng: Option<Rng>,
    pub control_seq: u64,
}

impl OpPayload {
    pub fn new(
        code: OpCode,
        bounds: Bounds,
        inputs: Vec<ProductRef>,
        outputs: Vec<ProductRef>,
        predicate: Option<ProductRef>,
        rng: Option<Rng>,
        control_seq: u64,
    ) -> Self {
        Self {
            code,
            bounds,
            inputs,
            outputs,
            predicate,
            rng,
            control_seq,
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
        (
            &self.bounds,
            &self.inputs,
            &self.outputs,
            self.predicate.as_ref(),
            self.rng,
            self.control_seq,
        )
    }
}

/// One immutable unit of registered work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    /// Request lineage that owns the operation.
    pub request_key: RequestKey,
    /// Request-local operation identifier.
    pub op_id: OpId,
    /// Exact state dependency, absent for computation determined only by its inputs.
    pub parent: Option<Checkpoint>,
    /// Computation entry bound within the selected Worker.
    pub entry: String,
    /// Family-specific products, bounds, and control state.
    pub payload: OpPayload,
}

impl Operation {
    /// Derives the execution code from the closed payload tag.
    pub const fn kind(&self) -> OpCode {
        self.payload.code
    }

    /// Marks a payload immutable for subsequent scheduler deltas.
    pub const fn sealed(self) -> Self {
        self
    }

    /// Returns the static execution domain for this payload.
    pub const fn domain(&self) -> Domain {
        self.kind().domain()
    }
    /// Returns whether successful execution advances request state.
    pub const fn advances_state(&self) -> bool {
        self.kind().advances_state()
    }
    /// Returns the scheduler-declared resource maxima.
    pub fn bounds(&self) -> &Bounds {
        self.payload.fields().0
    }
    /// Returns product references consumed by the operation.
    pub fn inputs(&self) -> &[ProductRef] {
        self.payload.fields().1
    }
    /// Returns product references produced by the operation.
    pub fn outputs(&self) -> &[ProductRef] {
        self.payload.fields().2
    }
    /// Returns mutable input references for transport lowering.
    pub fn inputs_mut(&mut self) -> &mut Vec<ProductRef> {
        &mut self.payload.inputs
    }

    /// Returns mutable output references for transport lowering.
    pub fn outputs_mut(&mut self) -> &mut Vec<ProductRef> {
        &mut self.payload.outputs
    }

    /// Returns the optional predicate product controlling execution.
    pub fn predicate(&self) -> Option<&ProductRef> {
        self.payload.fields().3
    }
    /// Replaces the optional predicate product.
    pub fn set_predicate(&mut self, value: Option<ProductRef>) {
        self.payload.predicate = value;
    }

    /// Returns deterministic random coordinates when the operation samples.
    pub fn rng(&self) -> Option<Rng> {
        self.payload.fields().4
    }
    /// Returns the request control sequence observed by this operation.
    pub fn control_seq(&self) -> u64 {
        self.payload.fields().5
    }

    /// Validates family-specific products, bounds, predicates, and RNG state.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish operation identity, family, lineage, and declared capacity.
        ensure_valid!(self.op_id.0 > 0, "operation id must be positive");
        ensure_valid!(!self.entry.is_empty(), "operation entry must not be empty");

        ensure_valid!(
            !(self.advances_state()
                || self.kind() == OpCode::TransferKvInstall
                || self
                    .inputs()
                    .iter()
                    .any(|value| value.kind == ProductKind::Latent))
                || self.parent.is_some(),
            "state-changing operation requires a predecessor"
        );
        if let Some(parent) = &self.parent {
            parent.validate()?;
        } else {
            ensure_valid!(
                self.control_seq() == 0,
                "computation without a predecessor has no state control sequence"
            );
        }
        ensure_valid!(
            !self.kind().requires_fixed_parent()
                || self.parent.as_ref().is_some_and(Checkpoint::is_fixed),
            "operation requires a fixed semantic parent"
        );
        self.bounds_are_finite()?;

        // Every output must be uniquely owned by this producer and fit the
        // resource class reserved for the operation.
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

        // Inputs are request-local except for shareable encoder features.
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

        // Predicates are generation-tagged device decisions in this lineage.
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

    /// Validates resource bounds that depend on state-advancement semantics.
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

/// Terminal status of one operation's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum OpStatus {
    /// Execution completed successfully.
    Ok = 0,
    /// The operation was skipped because its predicate was false.
    Predicated = 1,
    /// Execution failed with a deterministic error code.
    Error = 2,
}

/// A deterministic error class for a failed completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ErrorCode {
    /// Operation metadata or product declarations are invalid.
    InvalidOperation = 0,
    /// A reserved or physical resource could not satisfy the operation.
    ResourceExhausted = 1,
    /// Device computation failed.
    ComputeError = 2,
    /// The request was cancelled before completion.
    Cancelled = 3,
    /// An invariant failed within the worker.
    Internal = 4,
}

/// Accounting lengths carried by a completion. These are accounting fields, not
/// state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct LogicalLengths {
    /// Number of logical tokens committed for the request.
    pub token_len: u32,
    /// Number of KV tokens visible to successor operations.
    pub kv_visible_len: u32,
    /// Number of KV tokens materialized on the device.
    pub kv_computed_len: u32,
    /// Number of logical latent units committed for the request.
    pub latent_len: u32,
}

/// The span of tokens an operation contributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TokenSpan {
    /// First logical token position contributed by the operation.
    pub base: u32,
    /// Number of tokens contributed by the operation.
    pub len: u32,
}

/// Device-observed finish candidates for a token operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct FinishFlags {
    /// Whether an end-of-sequence token was selected.
    pub eos: bool,
    /// Whether the configured length limit was reached.
    pub length: bool,
    /// Whether a configured stop condition matched.
    pub stop: bool,
}

/// Per-operation timing counters. Accounting only; never state identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TimingCounters {
    /// Time spent waiting for worker execution, in microseconds.
    pub queued_us: u64,
    /// Time spent executing device work, in microseconds.
    pub device_us: u64,
    /// Time spent transferring completion data, in microseconds.
    pub copy_us: u64,
    /// Time spent processing the completion on the host, in microseconds.
    pub host_us: u64,
}

/// Transport handle for a completed media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "transport", content = "value")]
pub enum ArtifactHandle {
    /// Artifact stored in a POSIX shared-memory object.
    PosixShm {
        /// Shared-memory object name without a path separator.
        name: String,
    },
}

impl ArtifactHandle {
    /// Validates the transport-specific artifact identifier.
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

/// Metadata describing one materialized media artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaOutput {
    /// Transport descriptor for the materialized artifact.
    pub handle: ArtifactHandle,
    /// Artifact length in bytes.
    pub bytes: u64,
}

/// Common scalar outputs produced by an operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResultData {
    /// Logical request lengths after the operation.
    pub logical_lengths: LogicalLengths,
    /// Token positions contributed by the operation.
    pub token_span: TokenSpan,
    /// Token identifiers committed by the operation.
    pub committed_tokens: Vec<u32>,
    /// Device-observed terminal conditions.
    pub finish_flags: FinishFlags,
    /// Materialized media artifact, when produced.
    pub media_output: Option<MediaOutput>,
}

/// Cursor and terminal state produced by a diffusion operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DiffusionResult {
    /// Common operation result fields.
    pub data: ResultData,
    /// Cursor for the next diffusion operation.
    pub next_cursor: u32,
    /// Whether the diffusion request has reached a terminal state.
    pub done: bool,
}

/// Completion payload selected by the run kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "family", content = "value")]
pub enum ResultPayload {
    /// Autoregressive operation result.
    Ar(ResultData),
    /// Multimodal encoder result.
    Encoder(ResultData),
    /// Diffusion operation result with trajectory state.
    Diffusion(DiffusionResult),
    /// Cross-stage transfer result.
    Transfer(ResultData),
}

impl ResultPayload {
    /// Returns whether this result payload accepts `kind`.
    pub const fn family_matches(&self, kind: OpCode) -> bool {
        matches!(
            (self, kind),
            (
                Self::Ar(_),
                OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify
            ) | (
                Self::Encoder(_),
                OpCode::EncoderVision | OpCode::EncoderLatent | OpCode::EncoderText
            ) | (
                Self::Diffusion(_),
                OpCode::DiffusionPrepare
                    | OpCode::DiffusionStep
                    | OpCode::DiffusionDecode
                    | OpCode::DiffusionFinalize
                    | OpCode::MediaAppend
            ) | (
                Self::Transfer(_),
                OpCode::TransferProduct | OpCode::TransferKvPublish | OpCode::TransferKvInstall
            )
        )
    }

    /// Wraps common result data in the payload family required by `kind`.
    pub fn for_kind(kind: OpCode, data: ResultData) -> Self {
        match kind {
            OpCode::ArExtend | OpCode::ArDecode | OpCode::ArVerify => Self::Ar(data),
            OpCode::EncoderVision | OpCode::EncoderLatent | OpCode::EncoderText => {
                Self::Encoder(data)
            }
            OpCode::DiffusionPrepare
            | OpCode::DiffusionStep
            | OpCode::DiffusionDecode
            | OpCode::DiffusionFinalize
            | OpCode::MediaAppend => Self::Diffusion(DiffusionResult {
                done: data.finish_flags.eos
                    || data.finish_flags.length
                    || data.finish_flags.stop
                    || data.media_output.is_some(),
                data,
                next_cursor: 0,
            }),
            OpCode::TransferProduct | OpCode::TransferKvPublish | OpCode::TransferKvInstall => {
                Self::Transfer(data)
            }
        }
    }

    /// Returns common result data independently of the family variant.
    fn data(&self) -> &ResultData {
        match self {
            Self::Ar(data) | Self::Encoder(data) | Self::Transfer(data) => data,
            Self::Diffusion(result) => &result.data,
        }
    }

    /// Returns mutable access to common result data.
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
    /// Request lineage completed by the operation.
    pub request_key: RequestKey,
    /// Request-local operation identifier.
    pub op_id: OpId,
    /// Generation of the completion slot written by the worker.
    pub completion_slot_generation: u32,
    /// Terminal execution status.
    pub status: OpStatus,
    /// Semantic point selected by device execution.
    pub selected_point: u32,
    /// Allocation generations for products emitted by the operation.
    pub product_generations: Vec<u32>,
    /// Error classification when `status` is [`OpStatus::Error`].
    pub error_code: Option<ErrorCode>,
    /// Worker timing measurements for the operation.
    pub timing_counters: TimingCounters,
    /// Family-specific result data.
    pub payload: ResultPayload,
}

impl ModelOutput {
    /// Returns logical lengths from the family-specific payload.
    pub fn logical_lengths(&self) -> &LogicalLengths {
        &self.payload.data().logical_lengths
    }
    /// Returns mutable logical lengths from the family-specific payload.
    pub fn logical_lengths_mut(&mut self) -> &mut LogicalLengths {
        &mut self.payload.data_mut().logical_lengths
    }
    /// Returns the committed token span from the family-specific payload.
    pub fn token_span(&self) -> TokenSpan {
        self.payload.data().token_span
    }
    /// Returns mutable access to the committed token span.
    pub fn token_span_mut(&mut self) -> &mut TokenSpan {
        &mut self.payload.data_mut().token_span
    }
    /// Returns committed token identifiers from the family-specific payload.
    pub fn committed_tokens(&self) -> &[u32] {
        &self.payload.data().committed_tokens
    }
    /// Returns mutable committed token storage.
    pub fn committed_tokens_mut(&mut self) -> &mut Vec<u32> {
        &mut self.payload.data_mut().committed_tokens
    }
    /// Returns terminal flags from the family-specific payload.
    pub fn finish_flags(&self) -> FinishFlags {
        self.payload.data().finish_flags
    }
    /// Returns mutable terminal flags from the family-specific payload.
    pub fn finish_flags_mut(&mut self) -> &mut FinishFlags {
        &mut self.payload.data_mut().finish_flags
    }
    /// Returns media metadata when the operation materialized an artifact.
    pub fn media_output(&self) -> Option<&MediaOutput> {
        self.payload.data().media_output.as_ref()
    }

    /// Validates completion identity, status, payload, and product generations.
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

/// What to do with a committed selected point's public output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Disposition {
    /// Expose committed output to the public event stream.
    Publish = 0,
    /// Keep committed output for downstream execution without publishing it.
    Retain = 1,
    /// Release committed output without publishing it.
    Discard = 2,
}

/// Why a request lineage is being closed at a cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CloseReason {
    /// The request reached its normal terminal condition.
    Completed = 0,
    /// The caller cancelled the request.
    Cancelled = 1,
    /// Execution terminated because of an error.
    Error = 2,
    /// Scheduling preemption terminated the request lineage.
    Preempted = 3,
}

/// Ordered request and buffer commands carried with logical work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum BatchCommand {
    /// Establish one request lineage before its first operation on a pool.
    Start {
        /// Static request state installed by the worker.
        request: NewRequest,
    },
    /// Commit a selected fixed point and expose public output up to a limit.
    Commit {
        /// Request lineage receiving the commit.
        request_key: RequestKey,
        /// Monotonic request-control sequence.
        control_seq: u64,
        /// Checkpoint that must currently own the request lineage.
        expected_parent: Checkpoint,
        /// Fixed checkpoint selected for commit.
        selected: Checkpoint,
        /// Exclusive public-event sequence limit exposed by the commit.
        public_event_limit: u64,
        /// Policy for the selected checkpoint's public output.
        disposition: Disposition,
    },
    /// Finish a lineage at a fixed cutoff, dominating every uncommitted descendant.
    Finish {
        /// Request lineage being closed.
        request_key: RequestKey,
        /// Monotonic request-control sequence.
        control_seq: u64,
        /// Last fixed checkpoint retained by the request.
        cutoff: Checkpoint,
        /// Terminal reason recorded for the request.
        reason: CloseReason,
        /// Persistent products whose allocations remain owned outside this request.
        /// They stay readable until a subsequent explicit Free command.
        retained_buffers: Vec<BufferId>,
    },
    /// Free one exact persistent product after its final consumer.
    Free {
        /// Persistent buffer identity to release.
        buffer: BufferId,
    },
    /// Retire physical request state without selecting or publishing a checkpoint.
    /// Used after terminal failure or on auxiliary owners of a finished request.
    /// Acknowledgement still waits for operations, readers and storage reset.
    Retire {
        /// Exact request epoch whose rank-local resources are being retired.
        request_key: RequestKey,
        /// Persistent allocations retained by independent owners until Free.
        retained_buffers: Vec<BufferId>,
    },
}

impl BatchCommand {
    /// Returns the request lineage targeted by this command.
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Start { request } => request.request_key,
            Self::Commit { request_key, .. }
            | Self::Finish { request_key, .. }
            | Self::Retire { request_key, .. } => *request_key,
            Self::Free { buffer } => buffer.owner,
        }
    }

    /// Returns the stable discriminant used to order command variants.
    pub const fn variant_index(&self) -> u8 {
        match self {
            Self::Start { .. } => 0,
            Self::Commit { .. } => 1,
            Self::Finish { .. } => 2,
            Self::Free { .. } => 3,
            Self::Retire { .. } => 4,
        }
    }

    /// Returns the control sequence for commit and close commands.
    pub const fn control_seq(&self) -> Option<u64> {
        match self {
            Self::Commit { control_seq, .. } | Self::Finish { control_seq, .. } => {
                Some(*control_seq)
            }
            Self::Start { .. } | Self::Free { .. } | Self::Retire { .. } => None,
        }
    }

    /// Validates command identity, cutoff, and resource constraints.
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
            Self::Retire { .. } => {}
        }
        if let Self::Finish {
            request_key,
            retained_buffers,
            ..
        }
        | Self::Retire {
            request_key,
            retained_buffers,
        } = self
        {
            let mut retained = std::collections::HashSet::new();
            for buffer in retained_buffers {
                buffer.validate()?;
                ensure_valid!(
                    buffer.owner == *request_key,
                    "retained buffer belongs to another request"
                );
                ensure_valid!(
                    retained.insert(*buffer),
                    "request retirement repeats a retained buffer"
                );
            }
        }
        Ok(())
    }
}

/// Autoregressive request parameters fixed for the worker request lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArRequestParams {
    /// Sampling policy shared by autoregressive operations.
    pub sampling: SamplingParams,
    /// Token identifiers used for negative-conditioning input.
    pub negative_token_ids: Vec<u32>,
    /// Canonical token identifiers that terminate generation.
    pub finish_token_ids: Vec<u32>,
    /// Logical position assigned to the first request token.
    pub initial_position: u32,
}

/// Unified-multimodal request parameters fixed for the worker request lifetime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UmmRequestParams {
    /// Image-generation or transformation parameters.
    pub image: ImageParams,
}

/// Fixed geometry for a media-generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaGeometry {
    /// Number of frames produced by the request.
    pub frame_count: u32,
    /// Number of decoder work units in the terminal media stage.
    pub video_units: u32,
    /// Number of prompt tokens supplied to the diffusion model.
    pub prompt_tokens: u32,
    /// Number of denoising steps in the trajectory.
    pub denoise_steps: u32,
}

/// Ordered decoded reference media. Pixels are RGB u8 `[frames, height, width, 3]`;
/// audio is prepared stereo f32 `[2, samples]` at 32 kHz. Products use the
/// ordinary request-owned transport and lifetime, never source URLs or payloads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodedReference {
    pub kind: String,
    pub task: String,
    pub role: String,
    pub include_audio: bool,
    pub pixels: Option<ProductRef>,
    pub audio: Option<ProductRef>,
    /// Exact presentation frame rate; images and standalone audio use 0/1.
    pub fps_num: u32,
    pub fps_den: u32,
}

impl DecodedReference {
    /// Validate resolved geometry before any decoder or device allocation.
    pub fn validate(&self, owner: RequestKey) -> ValidationResult<()> {
        let visual = matches!(self.kind.as_str(), "image" | "video");
        ensure_valid!(visual || self.kind == "audio", "invalid reference modality");
        let valid_task = match self.task.as_str() {
            "reference" => self.role == "reference",
            "first_frame" => self.kind == "image" && self.role == "first_frame",
            "first_last_frame" => {
                self.kind == "image" && matches!(self.role.as_str(), "first_frame" | "last_frame")
            }
            "continue_scene" | "continue_shot" => self.kind == "video" && self.role == "preceding",
            _ => false,
        };
        ensure_valid!(valid_task, "invalid reference task or role");
        ensure_valid!(
            self.pixels.is_some() == visual,
            "reference pixels disagree with modality"
        );
        ensure_valid!(
            !self.include_audio || self.kind == "video",
            "soundtrack selection requires video"
        );
        ensure_valid!(
            self.audio.is_some() == (self.kind == "audio" || self.include_audio),
            "reference audio disagrees with soundtrack policy"
        );
        ensure_valid!(
            self.fps_den > 0
                && if self.kind == "video" {
                    self.fps_num > 0
                } else {
                    self.fps_num == 0 && self.fps_den == 1
                },
            "invalid reference frame rate"
        );
        for product in self.pixels.iter().chain(self.audio.iter()) {
            product.validate()?;
            ensure_valid!(
                product.request_key == owner && product.kind == ProductKind::Tensor,
                "reference tensor must belong to its request"
            );
            ensure_valid!(
                product
                    .shape_bound
                    .dims
                    .iter()
                    .all(|dim| matches!(dim, DimBound::Static(n) if *n > 0)),
                "decoded reference dimensions must be static and positive"
            );
        }
        if let Some(pixels) = &self.pixels {
            let dims: Vec<u32> = pixels
                .shape_bound
                .dims
                .iter()
                .map(|dim| match dim {
                    DimBound::Static(n) => *n,
                    _ => unreachable!(),
                })
                .collect();
            ensure_valid!(
                pixels.dtype == DType::U8 && dims.len() == 4 && dims[3] == 3,
                "reference pixels must be RGB u8 THWC"
            );
            ensure_valid!(
                dims[0] <= 720
                    && dims[1] <= 4096
                    && dims[2] <= 4096
                    && u64::from(dims[0]) * u64::from(dims[1]) * u64::from(dims[2])
                        <= 128 * 1024 * 1024,
                "reference pixel budget exceeded"
            );
            ensure_valid!(
                self.kind != "image" || dims[0] == 1,
                "image reference must have one frame"
            );
        }
        if let Some(audio) = &self.audio {
            ensure_valid!(
                audio.dtype == DType::F32
                    && matches!(audio.shape_bound.dims.as_slice(), [DimBound::Static(2), DimBound::Static(samples)] if *samples <= 960_000),
                "reference audio must be stereo f32 with at most 30 seconds at 32 kHz"
            );
        }
        Ok(())
    }
}

/// Static diffusion parameters admitted for one request lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionRequestParams {
    /// Tokenized text conditioning for the trajectory.
    pub prompt_token_ids: Vec<u32>,
    /// Request-level random seed.
    pub seed: u64,
    /// Fixed media and trajectory dimensions.
    pub geometry: MediaGeometry,
    /// Ordered decoded conditioning products, empty for text-only generation.
    #[serde(default)]
    pub references: Vec<DecodedReference>,
}

/// Request-start framing. Carries the per-domain parameters a request
/// needs before its operations run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewRequest {
    /// Globally unique request lineage identity.
    pub request_key: RequestKey,
    /// Scheduler-assigned stable request-state row. Index zero is reserved for
    /// inactive graph padding and never identifies a live request.
    pub request_pool_idx: u32,
    /// Autoregressive request parameters, when applicable.
    pub ar: Option<ArRequestParams>,
    /// Unified-multimodal request parameters, when applicable.
    pub umm: Option<UmmRequestParams>,
    /// Diffusion request parameters, when applicable.
    pub diffusion: Option<DiffusionRequestParams>,
}

impl NewRequest {
    /// Constructs static autoregressive or multimodal admission state.
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

    /// Constructs static terminal media admission state.
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

    /// Validates request identity and the selected family parameters.
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
                diffusion.references.len() <= 1,
                "references permits at most 1 image"
            );
            for reference in &diffusion.references {
                reference.validate(self.request_key)?;
                ensure_valid!(
                    reference.kind == "image" && reference.task == "reference" && reference.role == "reference",
                    "references requires image with task=reference and role=reference"
                );
                let pixels = reference.pixels.as_ref().expect("validated image pixels");
                ensure_valid!(
                    pixels.shape_bound.dims[1..3].iter().all(|dim| matches!(dim, DimBound::Static(n) if n % 32 == 0)),
                    "references image dimensions must be multiples of 32"
                );
            }
            ensure_valid!(
                !diffusion.prompt_token_ids.is_empty(),
                "diffusion prompt tokens must not be empty"
            );
            ensure_valid!(
                diffusion.geometry.frame_count > 0
                    && diffusion.geometry.video_units > 0
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

/// One scheduler-owned request-slot block table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTable {
    /// Scheduler-assigned request-state row that owns the table.
    pub request_pool_idx: u32,
    /// KV cache group addressed by the table.
    pub group_id: u32,
    /// Physical KV page identifiers in logical order.
    pub page_ids: Vec<BlockId>,
    /// Token capacity covered by the installed pages.
    pub allocated_tokens: u32,
}

/// Physical pages newly acquired for one installed block table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePageAllocation {
    /// Scheduler-assigned request-state row receiving the pages.
    pub request_pool_idx: u32,
    /// KV cache group receiving the pages.
    pub group_id: u32,
    /// Newly allocated physical page identifiers.
    pub page_ids: Vec<BlockId>,
}

impl BlockTable {
    /// Validates table identity, page uniqueness, and allocated length.
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
    /// Validates newly allocated page identities and ownership.
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
    /// Index of the operation represented by this model row.
    pub operation_index: u32,
    /// Scheduler-assigned request-state row.
    pub request_pool_index: u32,
    /// Total sequence length visible to attention.
    pub seq_len: u32,
    /// Number of query tokens evaluated by the row.
    pub query_len: u32,
    /// Whether the query interval is persisted in the cache after the visible prefix.
    pub write_kv: bool,
}

impl RowGeometry {
    /// Validates row ranges against the run's operation count.
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

/// Solver-step range and optional paged storage for one request trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentParams {
    /// Request lineage that owns the trajectory.
    pub request_key: RequestKey,
    /// Operation that addresses the trajectory.
    pub op_id: OpId,
    /// Physical latent pages in logical order; empty for request-owned tensors.
    pub page_table: Vec<u32>,
    /// Logical units stored in pages; zero when the trajectory uses request tensors.
    pub latent_units: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// First denoising step assigned to the operation.
    pub start_step: u32,
    /// Number of denoising steps assigned to the operation.
    pub step_count: u32,
}

impl LatentParams {
    /// Validates latent page identities, shape, and byte bounds.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0,
            "latent params operation id must be positive"
        );
        ensure_valid!(
            self.height > 0 && self.width > 0,
            "latent params geometry must be positive"
        );
        ensure_valid!(
            (!self.page_table.is_empty() == (self.latent_units > 0))
                && self.page_table.iter().all(|page| *page > 0)
                && self.page_table.iter().collect::<HashSet<_>>().len() == self.page_table.len(),
            "latent params page table disagrees with its units, repeats a page, or carries page zero"
        );
        Ok(())
    }
}

/// Independent media stream selected by decoding or ordered output consumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaTrack {
    /// Overlapping temporal video units.
    Video = 0,
    /// Interleaved audio samples.
    Audio = 1,
}

impl MediaTrack {
    /// Stable name used by physical request descriptors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

/// Cursor and unit bounds for one diffusion decode operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request lineage that owns the media output.
    pub request_key: RequestKey,
    /// Decode operation receiving the params.
    pub op_id: OpId,
    /// Independent media stream addressed by this range.
    pub track: MediaTrack,
    /// First decoder unit assigned to the operation.
    pub cursor: u32,
    /// Maximum decoder units the operation may process.
    pub max_units: u32,
}

impl DecodeRange {
    /// Validates decode identity and unit capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.0 > 0,
            "decode params operation id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode params unit bound must be positive"
        );
        ensure_valid!(
            self.track != MediaTrack::Audio || (self.cursor == 0 && self.max_units == 1),
            "audio decode range must address its single sample stream"
        );
        Ok(())
    }
}

/// Scheduler-selected address span for one cross-operation persistent buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferAllocation {
    /// Persistent buffer identity receiving the address span.
    pub buffer: BufferId,
    /// Byte offset within the persistent-buffer arena.
    pub offset: u64,
    /// Length of the reserved span in bytes.
    pub bytes: u64,
}

impl BufferAllocation {
    /// Validates aligned non-empty buffer params.
    pub fn validate(self) -> ValidationResult<()> {
        self.buffer.validate()?;
        ensure_valid!(self.bytes > 0, "buffer params byte extent must be positive");
        ensure_valid!(
            self.offset.checked_add(self.bytes).is_some(),
            "buffer params span overflows"
        );
        Ok(())
    }
}

/// One executor-produced physical worker invocation. Execution domains,
/// attention selection, and captured buckets are derived by the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// Submission batch identity assigned by the executor.
    pub batch_id: u64,
    /// Physical invocation identity, strictly increasing in each worker's Submit order.
    /// Logical batch allocation and result completion may occur in a different order.
    pub run_id: u64,
    /// Monotonic sequence shared by collective participants.
    pub collective_seq: u64,
    /// Operations executed by this invocation.
    pub operations: Vec<Operation>,
    /// Complete scheduler-owned KV page tables required by the operations.
    pub block_tables: Vec<BlockTable>,
    /// Physical KV pages newly allocated for this invocation.
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Row-aligned model-forward metadata.
    pub forward_rows: Vec<RowGeometry>,
    /// Scheduler-owned latent trajectory allocations.
    pub latent_params: Vec<LatentParams>,
    /// Scheduler-owned media decode allocations.
    pub decode_ranges: Vec<DecodeRange>,
    /// Address spans for persistent cross-operation buffers.
    pub buffer_allocations: Vec<BufferAllocation>,
    /// Ordered request-state and buffer-lifetime commands.
    pub commands: Vec<BatchCommand>,
    /// Host-supplied input product values matched by `ProductRef` identity.
    pub input_products: Vec<ProductPayload>,
}

impl Run {
    /// Constructs a run with admissions and operations using default metadata.
    pub fn new(batch_id: u64, admissions: Vec<NewRequest>, operations: Vec<Operation>) -> Self {
        Self {
            batch_id,
            run_id: batch_id,
            collective_seq: batch_id.max(1),
            operations,
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward_rows: Vec::new(),
            latent_params: Vec::new(),
            decode_ranges: Vec::new(),
            buffer_allocations: Vec::new(),
            commands: admissions
                .into_iter()
                .map(|request| BatchCommand::Start { request })
                .collect(),
            input_products: Vec::new(),
        }
    }

    /// Iterates over operations in submission order.
    pub fn operations(&self) -> impl Iterator<Item = &Operation> {
        self.operations.iter()
    }

    /// Returns the number of operations in this run.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Attaches ordered control commands to the run.
    pub fn with_commands(mut self, commands: Vec<BatchCommand>) -> Self {
        self.commands.extend(commands);
        self
    }

    /// Iterates over static request admissions in submission order.
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request),
            _ => None,
        })
    }

    /// Attaches resolved input products to the run.
    pub fn with_input_products(mut self, input_products: Vec<ProductPayload>) -> Self {
        self.input_products = input_products;
        self
    }

    /// Validates identities, families, allocations, products, and commands.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish the run envelope before validating relationships within it.
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

        // KV allocations are subsets of unique block tables for the same pool group.
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

        // Latent pages are exclusive across every trajectory placed in a run.
        let mut latent_ids = HashSet::with_capacity(self.latent_params.len());
        let mut latent_pages = HashSet::new();
        for params in &self.latent_params {
            params.validate()?;
            let identity = (params.request_key, params.op_id);
            ensure_valid!(
                latent_ids.insert(identity),
                "run repeats a latent params identity"
            );
            let operation = operations
                .get(&identity)
                .ok_or_else(|| invalid_message!("latent params does not name a run operation"))?;
            let addresses_trajectory = matches!(
                operation.kind(),
                OpCode::DiffusionPrepare | OpCode::DiffusionStep
            ) || operation
                .inputs()
                .iter()
                .any(|value| value.kind == ProductKind::Latent);
            ensure_valid!(
                addresses_trajectory,
                "latent params names work without a trajectory"
            );
            ensure_valid!(
                params
                    .page_table
                    .iter()
                    .all(|page| latent_pages.insert(*page)),
                "run latent allocations overlap physical pages"
            );
        }

        for operation in &self.operations {
            let needs_latent = matches!(
                operation.kind(),
                OpCode::DiffusionPrepare | OpCode::DiffusionStep
            ) || operation
                .inputs()
                .iter()
                .any(|value| value.kind == ProductKind::Latent);
            ensure_valid!(
                !needs_latent || latent_ids.contains(&(operation.request_key, operation.op_id)),
                "operation that addresses a trajectory has no latent params"
            );
        }

        // Decode params is restricted to operations that materialize media.
        let mut decode_ids = HashSet::with_capacity(self.decode_ranges.len());
        for params in &self.decode_ranges {
            params.validate()?;
            let identity = (params.request_key, params.op_id);
            ensure_valid!(
                decode_ids.insert(identity),
                "run repeats a decode params identity"
            );
            let operation = operations
                .get(&identity)
                .ok_or_else(|| invalid_message!("decode params does not name a run operation"))?;
            ensure_valid!(
                matches!(
                    operation.kind(),
                    OpCode::DiffusionDecode | OpCode::MediaAppend
                ),
                "decode params does not name media decode work"
            );
        }

        for operation in &self.operations {
            let needs_range = matches!(
                operation.kind(),
                OpCode::DiffusionDecode | OpCode::MediaAppend
            );
            ensure_valid!(
                !needs_range || decode_ids.contains(&(operation.request_key, operation.op_id)),
                "media reconstruction operation has no decode range"
            );
        }

        // Persistent buffers use non-overlapping spans and cover every declared output.
        let mut buffer_ids = HashSet::with_capacity(self.buffer_allocations.len());
        let mut buffer_spans = self
            .buffer_allocations
            .iter()
            .map(|params| {
                params.validate()?;
                ensure_valid!(
                    buffer_ids.insert(params.buffer),
                    "run repeats a buffer params identity"
                );
                Ok((params.offset, params.offset + params.bytes))
            })
            .collect::<ValidationResult<Vec<_>>>()?;
        buffer_spans.sort_unstable();

        ensure_valid!(
            buffer_spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
            "run buffer allocations overlap"
        );

        for operation in &self.operations {
            for output in operation
                .outputs()
                .iter()
                .filter(|output| output.uses_persistent_buffer())
            {
                let params = self
                    .buffer_allocations
                    .iter()
                    .find(|params| params.buffer == output.buffer_id())
                    .ok_or_else(|| {
                        invalid_message!("persistent operation output has no buffer params")
                    })?;
                ensure_valid!(
                    params.bytes >= output.max_bytes(),
                    "buffer params is smaller than its declared output"
                );
            }
        }

        // Submission depth is one runnable operation per request.
        let mut request_keys = HashSet::with_capacity(self.operation_count());
        for operation in &self.operations {
            ensure_valid!(
                request_keys.insert(operation.request_key),
                "a submission batch carries multiple operations for one request"
            );
        }

        // Admissions create request state independently of entry execution.
        let mut admitted = HashSet::new();
        for admission in self.admissions() {
            admission.validate()?;
            ensure_valid!(
                admitted.insert(admission.request_key),
                "a submission batch carries a duplicate admission"
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

        // Product payloads must be declared, uniquely supplied, and represented
        // according to whether their storage crosses a stage boundary.
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
    /// Submission batch identity copied from the run.
    pub batch_id: u64,
    /// Physical invocation identity copied from the run.
    pub run_id: u64,
    /// Operation completions contained in this fragment.
    pub completions: Vec<ModelOutput>,
    /// Product values published by completed operations.
    pub products: Vec<ProductPayload>,
    /// Visibility result for atomic product registration.
    pub registration: RegistrationAck,
    /// Aggregate worker execution time in microseconds, when measured.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported by the worker.
    pub forward_stats: Option<WorkerForwardStats>,
    /// Whether all operations and command-owned physical resources have retired.
    /// Free/Finish/Retire runs publish a separate empty terminal fragment after any
    /// operation fragments, even when retirement is immediately ready.
    pub done: bool,
}

impl RunResult {
    /// Iterates over operation completions in report order.
    pub fn completions(&self) -> impl Iterator<Item = &ModelOutput> {
        self.completions.iter()
    }

    /// Iterates over resolved product payloads in report order.
    pub fn products(&self) -> impl Iterator<Item = &ProductPayload> {
        self.products.iter()
    }

    /// Validates completion and product identities for this run.
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
