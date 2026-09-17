//! Request identities, operation descriptors, admissions, and execution batches.

use super::*;

/// `(engine_id, request_id, request_epoch)`. The epoch advances whenever an admitted
/// identity is reused, so no operation or product reference aliases across
/// requests or epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestKey {
    /// Engine instance that allocates request identifiers.
    pub engine_id: u64,
    /// Identifier assigned to the request within its engine instance.
    pub request_id: RequestId,
    /// Admission generation that distinguishes reuse of the same request identifier.
    pub request_epoch: u64,
}

impl RequestKey {
    /// Constructs an identity from engine, request, and admission epoch.
    pub const fn new(engine_id: u64, request_id: RequestId, request_epoch: u64) -> Self {
        Self {
            engine_id,
            request_id,
            request_epoch,
        }
    }
}

/// The numerical mode of an autoregressive model forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ForwardMode {
    Prefill,
    Decode,
    Verify,
}

impl ForwardMode {
    pub const ALL: [Self; 3] = [Self::Prefill, Self::Decode, Self::Verify];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
        }
    }
}

/// A concrete encoder, diffusion, decoder, or media-output computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum PipelineStage {
    VisionEncoding,
    LatentEncoding,
    TextEncoding,
    LatentPreparation,
    Denoising,
    ImageDecoding,
    VideoDecoding,
    AudioDecoding,
    VideoEncoding,
    AudioEncoding,
    Muxing,
}

impl PipelineStage {
    pub const ALL: [Self; 11] = [
        Self::VisionEncoding,
        Self::LatentEncoding,
        Self::TextEncoding,
        Self::LatentPreparation,
        Self::Denoising,
        Self::ImageDecoding,
        Self::VideoDecoding,
        Self::AudioDecoding,
        Self::VideoEncoding,
        Self::AudioEncoding,
        Self::Muxing,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VisionEncoding => "vision_encoding",
            Self::LatentEncoding => "latent_encoding",
            Self::TextEncoding => "text_encoding",
            Self::LatentPreparation => "latent_preparation",
            Self::Denoising => "denoising",
            Self::ImageDecoding => "image_decoding",
            Self::VideoDecoding => "video_decoding",
            Self::AudioDecoding => "audio_decoding",
            Self::VideoEncoding => "video_encoding",
            Self::AudioEncoding => "audio_encoding",
            Self::Muxing => "muxing",
        }
    }
}

impl PipelineStage {
    /// Fixed stages needed to produce video and audio.
    pub const VIDEO: [Self; 8] = [
        Self::TextEncoding,
        Self::LatentPreparation,
        Self::Denoising,
        Self::VideoDecoding,
        Self::AudioDecoding,
        Self::VideoEncoding,
        Self::AudioEncoding,
        Self::Muxing,
    ];
}

/// The storage action performed between concrete producers and consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TransferMode {
    Tensor,
    KvPublish,
    KvInstall,
}

impl TransferMode {
    pub const ALL: [Self; 3] = [Self::Tensor, Self::KvPublish, Self::KvInstall];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::KvPublish => "kv_publish",
            Self::KvInstall => "kv_install",
        }
    }
}

/// Exactly one computation classification. The sum preserves the distinct
/// forward, pipeline, and storage contracts without parallel opcode metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Computation {
    Forward(ForwardMode),
    Pipeline(PipelineStage),
    Transfer(TransferMode),
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

impl Computation {
    /// Computations accepted as individual scheduled request items.
    pub const ALL: [Self; 17] = [
        Self::Forward(ForwardMode::Prefill),
        Self::Forward(ForwardMode::Decode),
        Self::Forward(ForwardMode::Verify),
        Self::Pipeline(PipelineStage::VisionEncoding),
        Self::Pipeline(PipelineStage::LatentEncoding),
        Self::Pipeline(PipelineStage::TextEncoding),
        Self::Pipeline(PipelineStage::LatentPreparation),
        Self::Pipeline(PipelineStage::Denoising),
        Self::Pipeline(PipelineStage::ImageDecoding),
        Self::Pipeline(PipelineStage::VideoDecoding),
        Self::Pipeline(PipelineStage::AudioDecoding),
        Self::Pipeline(PipelineStage::VideoEncoding),
        Self::Pipeline(PipelineStage::AudioEncoding),
        Self::Pipeline(PipelineStage::Muxing),
        Self::Transfer(TransferMode::Tensor),
        Self::Transfer(TransferMode::KvPublish),
        Self::Transfer(TransferMode::KvInstall),
    ];

    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::Forward(_)
                | Self::Pipeline(PipelineStage::LatentPreparation | PipelineStage::Denoising)
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward(mode) => mode.as_str(),
            Self::Pipeline(stage) => stage.as_str(),
            Self::Transfer(mode) => mode.as_str(),
        }
    }
}

/// Hard resource maxima the scheduler reserves before an operation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
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

/// Branch-local token processor inputs for one sampling operation.
///
/// Token ids in every field are strictly increasing. `allowed_token_ids`
/// distinguishes no whitelist (`None`) from a present empty whitelist, which
/// deterministically represents an invalid all-masked distribution. Penalty
/// token counts are not carried here: they are a device-resident committed base
/// plus bounded per-operation deltas folded after sampling accepts tokens, so no host
/// token history participates in a successor's sampling input.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SamplingState {
    /// Optional whitelist of token identifiers eligible for sampling.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Token identifiers excluded from sampling.
    pub suppressed_token_ids: Vec<u32>,
    /// Token identifiers that terminate generation when selected.
    pub finish_token_ids: Vec<u32>,
    /// Token identifiers that advance a structured-generation transition.
    pub transition_token_ids: Vec<u32>,
    /// Whether the current grammar state requires immediate termination.
    pub force_finish: bool,
}

impl SamplingState {
    /// Sorts and deduplicates token identifier sets for deterministic encoding.
    pub fn canonicalize(&mut self) {
        if let Some(allowed) = &mut self.allowed_token_ids {
            allowed.sort_unstable();
            allowed.dedup();
        }
        self.suppressed_token_ids.sort_unstable();
        self.suppressed_token_ids.dedup();
        self.finish_token_ids.sort_unstable();
        self.finish_token_ids.dedup();
        self.transition_token_ids.sort_unstable();
        self.transition_token_ids.dedup();
    }
}

/// One immutable computation with its identity, data dependencies, and output limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledRequest {
    /// Request lineage that owns the operation.
    pub request_key: RequestKey,
    /// Logical batch and selection ordinal of the completed computation.
    pub op_id: ComputationId,
    /// Execution-order dependency; zero identifies admission, absent for independent work.
    pub predecessor: Option<ComputationId>,
    /// Computation entry bound within the selected Worker.
    pub entry: String,
    /// Computation performed by this operation.
    pub code: Computation,
    /// Scheduler-declared limits for the computation and its outputs.
    pub bounds: Bounds,
    /// Data dependencies consumed in input order.
    pub inputs: Vec<TensorRef>,
    /// Outputs allocated to this producer.
    pub outputs: Vec<TensorRef>,
    /// Source token scalar transported between computation entries.
    pub token_input: Option<TensorRef>,
    /// Sampled token and continuation bit in one I64 scalar.
    pub token_output: Option<TensorRef>,
    /// Vision features consumed by multimodal forward.
    pub vision_input: Option<TensorRef>,
    /// VAE features consumed by multimodal forward.
    pub latent_feature_input: Option<TensorRef>,
    /// Features produced by the selected image encoder.
    pub encoder_output: Option<TensorRef>,
    /// Diffusion trajectory consumed by this computation.
    pub latent_input: Option<TensorRef>,
    /// Diffusion trajectory produced by this computation.
    pub latent_output: Option<TensorRef>,
    /// Resident image consumed by feedback encoding.
    pub image_input: Option<TensorRef>,
    /// Decoded image retained for feedback.
    pub image_output: Option<TensorRef>,
    /// Boolean device completion consumed by a dependent computation.
    pub completion_output: Option<TensorRef>,
    /// Boolean device decision that selects an image transition.
    pub transition_output: Option<TensorRef>,
    /// Device predicate that enables execution.
    pub predicate: Option<TensorRef>,
    /// Deterministic sampling coordinates.
    pub rng: Option<Rng>,
    /// Host-side sampler constraints for this computation; absent uses admission defaults.
    pub sampling_state: Option<SamplingState>,
    /// Host-known prompt, draft, or decode input tokens in model input order.
    /// Empty when continuation reads its predecessor's device token.
    pub input_token_ids: Vec<u32>,
    /// Encoded source image in base64, consumed by a vision or latent encoder.
    /// Dispatch and in-flight matching share these immutable bytes.
    pub input_image: Option<std::sync::Arc<str>>,
    /// Published or installed cache extent consumed by this computation.
    pub kv_input: Option<BufferId>,
    /// Identity under which this computation publishes or installs cache state.
    pub kv_output: Option<BufferId>,
}

impl ScheduledRequest {
    /// Tensor dependencies in the computation signature, excluding its predicate.
    pub fn tensor_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.token_input.iter())
            .chain(self.vision_input.iter())
            .chain(self.latent_feature_input.iter())
            .chain(self.latent_input.iter())
            .chain(self.image_input.iter())
    }

    /// All tensor declarations owned by this computation.
    pub fn tensor_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs
            .iter()
            .chain(self.token_output.iter())
            .chain(self.completion_output.iter())
            .chain(self.transition_output.iter())
            .chain(self.encoder_output.iter())
            .chain(self.latent_output.iter())
            .chain(self.image_output.iter())
    }

    /// Mutable declarations while assigning computation identities before submission.
    pub fn tensor_outputs_mut(&mut self) -> impl Iterator<Item = &mut TensorRef> {
        self.outputs
            .iter_mut()
            .chain(self.token_output.iter_mut())
            .chain(self.completion_output.iter_mut())
            .chain(self.transition_output.iter_mut())
            .chain(self.encoder_output.iter_mut())
            .chain(self.latent_output.iter_mut())
            .chain(self.image_output.iter_mut())
    }

    /// Tensor inputs that require persistent destination storage.
    pub fn buffer_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.vision_input.iter())
            .chain(self.latent_feature_input.iter())
            .chain(self.image_input.iter())
    }

    /// Tensor outputs backed by scheduler-allocated persistent buffers.
    pub fn buffer_outputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.outputs
            .iter()
            .chain(self.encoder_output.iter())
            .chain(self.image_output.iter())
    }

    /// Exact data and predicate dependencies, independent of physical representation.
    pub fn input_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_inputs()
            .chain(self.predicate.iter())
            .map(TensorRef::buffer_id)
            .chain(self.kv_input)
    }

    /// Resource identities published by this computation.
    pub fn output_buffers(&self) -> impl Iterator<Item = BufferId> + '_ {
        self.tensor_outputs()
            .map(TensorRef::buffer_id)
            .chain(self.kv_output)
    }

    /// Returns whether successful execution advances request state.
    pub const fn advances_state(&self) -> bool {
        self.code.advances_state()
    }
    /// Validates family-specific products, bounds, predicates, and RNG state.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish operation identity, family, lineage, and declared capacity.
        ensure_valid!(self.op_id.batch_id > 0, "operation id must be positive");
        ensure_valid!(!self.entry.is_empty(), "operation entry must not be empty");

        ensure_valid!(
            !(self.advances_state()
                || self.code == Computation::Transfer(TransferMode::KvInstall)
                || self.latent_input.is_some())
                || self.predecessor.is_some(),
            "state-changing operation requires a predecessor"
        );
        if let Some(predecessor) = self.predecessor {
            ensure_valid!(
                predecessor.batch_id != 0 || predecessor.request_index == 0,
                "admission predecessor must use index zero"
            );
            ensure_valid!(
                predecessor < self.op_id,
                "predecessor must precede operation"
            );
        }
        ensure_valid!(
            self.input_token_ids.len() <= self.bounds.max_tokens as usize,
            "input token count exceeds the computation token bound"
        );
        if let Some(state) = &self.sampling_state {
            for ids in [
                state.allowed_token_ids.as_deref(),
                Some(state.suppressed_token_ids.as_slice()),
                Some(state.finish_token_ids.as_slice()),
                Some(state.transition_token_ids.as_slice()),
            ]
            .into_iter()
            .flatten()
            {
                ensure_valid!(
                    ids.windows(2).all(|pair| pair[0] < pair[1]),
                    "sampling-state token ids are not canonical"
                );
            }
        }

        if let Some(image) = &self.input_image {
            ensure_valid!(
                !image.is_empty()
                    && matches!(
                        self.code,
                        Computation::Pipeline(
                            PipelineStage::VisionEncoding | PipelineStage::LatentEncoding
                        )
                    )
                    && self.image_input.is_none(),
                "encoded image requires an image encoder without another image source"
            );
        }

        // Every output must be uniquely owned by this producer and fit the
        // resource class reserved for the operation.
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        let publishes_kv = matches!(
            self.code,
            Computation::Transfer(TransferMode::KvPublish | TransferMode::KvInstall)
        );
        ensure_valid!(
            self.kv_output.is_some() == publishes_kv,
            "KV publication or installation requires one cache output identity"
        );
        if let Some(output) = self.kv_output {
            output.validate()?;
            ensure_valid!(
                output.owner == self.request_key && output.producer_op_id == self.op_id,
                "KV output is not owned by its producing computation"
            );
            output_indices.insert(output.output_index);
        }
        let consumes_kv = matches!(
            self.code,
            Computation::Transfer(TransferMode::KvInstall)
                | Computation::Pipeline(
                    PipelineStage::LatentPreparation | PipelineStage::Denoising
                )
        );
        if let Some(input) = self.kv_input {
            input.validate()?;
            ensure_valid!(
                consumes_kv && input.owner == self.request_key,
                "KV input is incompatible with its computation or request"
            );
        }
        ensure_valid!(
            self.code != Computation::Transfer(TransferMode::KvInstall) || self.kv_input.is_some(),
            "KV installation requires a source publication"
        );

        for output in self.tensor_outputs() {
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
                output_indices.insert(output.output_index),
                "operation repeats an output index"
            );
        }

        for output in self
            .encoder_output
            .iter()
            .chain(self.latent_output.iter())
            .chain(self.image_output.iter())
        {
            ensure_valid!(
                output.max_bytes() <= self.bounds.max_latent_bytes,
                "image or trajectory output exceeds its declared byte capacity"
            );
        }
        if let Some(input) = &self.token_input {
            ensure_valid!(
                self.code == Computation::Transfer(TransferMode::Tensor),
                "token transfer input requires a tensor-transfer computation"
            );
            ensure_valid!(
                input.dtype == DType::I64 && input.shape_bound.max_elements() == 1,
                "token transfer input requires one int64 element"
            );
        }
        for tensor in self.token_output.iter() {
            ensure_valid!(
                tensor.dtype == DType::I64 && tensor.shape_bound.max_elements() == 1,
                "device token relay requires one int64 element"
            );
        }
        for tensor in self
            .completion_output
            .iter()
            .chain(self.transition_output.iter())
        {
            ensure_valid!(
                tensor.dtype == DType::U8 && tensor.shape_bound.max_elements() == 1,
                "device completion requires one uint8 element"
            );
        }

        // Only encoder features can be shared across request lineages.
        for input in self.tensor_inputs() {
            input.validate()?;
            ensure_valid!(
                input.request_key == self.request_key
                    || Some(input) == self.vision_input.as_ref()
                    || Some(input) == self.latent_feature_input.as_ref(),
                "request-local tensor belongs to another request lineage"
            );
        }
        if let Some(predicate) = &self.predicate {
            predicate.validate()?;
            ensure_valid!(
                predicate.request_key == self.request_key,
                "computation predicate belongs to another request lineage"
            );
            ensure_valid!(
                matches!(predicate.dtype, DType::U8 | DType::I64)
                    && predicate.shape_bound.max_elements() == 1,
                "device predicate requires a boolean or packed continuation scalar"
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

/// The fixed-layout record a worker emits once for every operation, after its
/// copy event is query-ready and its pinned fields are validated on the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestOutput {
    /// Request lineage completed by the operation.
    pub request_key: RequestKey,
    /// Request-local operation identifier.
    pub op_id: ComputationId,
    /// Terminal execution status.
    pub status: OpStatus,
    /// Allocation generations for products emitted by the operation.
    pub product_generations: Vec<u32>,
    /// Error classification when `status` is [`OpStatus::Error`].
    pub error_code: Option<ErrorCode>,
    /// Worker timing measurements for the operation.
    pub timing_counters: TimingCounters,
    /// Computation that produced this result; must match the submitted operation.
    pub code: Computation,
    /// Model position for the next input, including image-feedback position advances.
    pub position: u32,
    /// Accepted KV prefix available to a successor.
    pub kv_visible_len: u32,
    /// KV extent initialized by execution, including rejected speculative positions.
    pub kv_computed_len: u32,
    /// Completed denoising steps, independent of latent tensor dimensions.
    pub num_completed_steps: u32,
    /// Token identifiers accepted by the sampler.
    pub committed_tokens: Vec<u32>,
    /// Natural-log probability of the final accepted token, when requested.
    pub sampled_logprob: Option<f32>,
    /// Ranked candidates for the final accepted token.
    pub top_logprobs: Vec<TokenLogprob>,
    /// Ranked candidates for each scored prompt position, in input order.
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    /// Device-observed terminal conditions.
    pub finish_flags: FinishFlags,
    /// Completed artifact and its external storage lifetime, when produced.
    pub media_output: Option<MediaOutput>,
    /// Physical KV publication, including this rank's tensor locations.
    pub kv_output: Option<KvTransfer>,
}

impl RequestOutput {
    /// Validates completion identity, status, accepted lengths, and product generations.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(self.op_id.batch_id > 0, "completion op id must be positive");
        if let Some(publication) = &self.kv_output {
            publication.validate()?;
            ensure_valid!(
                self.status == OpStatus::Ok
                    && self.code == Computation::Transfer(TransferMode::KvPublish)
                    && publication.source.owner == self.request_key
                    && publication.source.producer_op_id == self.op_id,
                "KV publication does not belong to its successful completion"
            );
        }

        ensure_valid!(
            self.kv_visible_len <= self.kv_computed_len,
            "completion selected KV length exceeds computed length"
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
                self.committed_tokens.as_slice().is_empty() && self.product_generations.is_empty(),
                "a predicated completion must select its parent without semantic output"
            );
            ensure_valid!(
                !self.finish_flags.eos && !self.finish_flags.length && !self.finish_flags.stop,
                "a predicated completion must not select a terminal outcome"
            );
        }
        if let Some(output) = self.media_output.as_ref() {
            output.handle.validate()?;
            ensure_valid!(
                output.bytes > 0,
                "media output byte length must be positive"
            );
        }
        Ok(())
    }
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
    /// Close an exact request epoch after admitted work and physical readers finish.
    Finish {
        request_key: RequestKey,
        /// Products that remain readable under independent ownership until Free.
        retained_buffers: Vec<BufferId>,
    },
    /// Free one exact persistent product after its final consumer.
    Free {
        /// Persistent buffer identity to release.
        buffer: BufferId,
    },
}

impl BatchCommand {
    /// Returns the request lineage targeted by this command.
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Start { request } => request.request_key,
            Self::Finish { request_key, .. } => *request_key,
            Self::Free { buffer } => buffer.owner,
        }
    }

    /// Returns the stable discriminant used to order command variants.
    pub const fn variant_index(&self) -> u8 {
        match self {
            Self::Start { .. } => 0,
            Self::Finish { .. } => 1,
            Self::Free { .. } => 2,
        }
    }

    /// Validates command identity and retained resource constraints.
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::Start { request } => request.validate()?,
            Self::Finish { .. } => {}
            Self::Free { buffer } => buffer.validate()?,
        }
        if let Self::Finish {
            request_key,
            retained_buffers,
            ..
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

pub use uniserve_core::DiffusionSamplingParams;

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
    pub diffusion: Option<DiffusionSamplingParams>,
    /// Tokenized conditioning supplied at diffusion admission.
    pub prompt_token_ids: Vec<u32>,
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
            prompt_token_ids: Vec::new(),
        };
        Ok(admission)
    }

    /// Constructs static terminal media admission state.
    pub fn new_media(
        request_key: RequestKey,
        request_pool_idx: u32,
        prompt_token_ids: Vec<u32>,
        diffusion: DiffusionSamplingParams,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        let admission = Self {
            request_key,
            request_pool_idx,
            ar: None,
            umm: None,
            diffusion: Some(diffusion),
            prompt_token_ids,
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
                !self.prompt_token_ids.is_empty(),
                "diffusion prompt tokens must not be empty"
            );
            ensure_valid!(
                diffusion.num_frames > 0
                    && diffusion.num_decode_chunks > 0
                    && self.prompt_token_ids.len() <= u32::MAX as usize
                    && diffusion.num_inference_steps > 0,
                "diffusion parameters are invalid"
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

/// Aligned columns defining the model's forward rows, independent of wire framing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardBatch {
    /// Physical operation index for each row; logical computation IDs remain unchanged.
    #[serde(rename = "forward_operation_indices")]
    pub operation_indices: Vec<u32>,
    /// Scheduler-owned request slots, including alternative CFG prefixes.
    pub request_pool_indices: Vec<u32>,
    /// Total attention lengths (cached prefix plus query), not allocated capacities.
    pub seq_lens: Vec<u32>,
    /// Query tokens evaluated by each row.
    pub query_lens: Vec<u32>,
    /// Whether the query interval persists in KV after its visible prefix.
    pub write_kv: Vec<bool>,
}

impl ForwardBatch {
    /// Append one model row to every column in the same order.
    pub fn push(
        &mut self,
        operation_index: u32,
        request_pool_index: u32,
        seq_len: u32,
        query_len: u32,
        write_kv: bool,
    ) {
        self.operation_indices.push(operation_index);
        self.request_pool_indices.push(request_pool_index);
        self.seq_lens.push(seq_len);
        self.query_lens.push(query_len);
        self.write_kv.push(write_kv);
    }

    /// Move forward inputs into a physical batch, offsetting only physical row ownership.
    pub fn append(&mut self, other: Self, operation_offset: u32) {
        self.operation_indices
            .extend(other.operation_indices.into_iter().map(|index| {
                index
                    .checked_add(operation_offset)
                    .expect("physical operation index fits u32")
            }));
        self.request_pool_indices.extend(other.request_pool_indices);
        self.seq_lens.extend(other.seq_lens);
        self.query_lens.extend(other.query_lens);
        self.write_kv.extend(other.write_kv);
    }

    /// Select a rank's operations and map their forward rows to the local operation array.
    pub fn select(&self, operations: &[usize]) -> Self {
        let mut selected = Self::default();
        for (row, index) in self.operation_indices.iter().enumerate() {
            if let Some(local) = operations
                .iter()
                .position(|operation| *operation == *index as usize)
            {
                selected.push(
                    local as u32,
                    self.request_pool_indices[row],
                    self.seq_lens[row],
                    self.query_lens[row],
                    self.write_kv[row],
                );
            }
        }
        selected
    }

    /// Validate column alignment and numerical ranges before indexing model inputs.
    pub fn validate(&self, operation_count: usize) -> ValidationResult<()> {
        let rows = self.operation_indices.len();
        ensure_valid!(
            self.request_pool_indices.len() == rows
                && self.seq_lens.len() == rows
                && self.query_lens.len() == rows
                && self.write_kv.len() == rows,
            "forward input columns have different lengths"
        );
        ensure_valid!(
            self.operation_indices
                .iter()
                .all(|index| (*index as usize) < operation_count),
            "forward operation index is outside its batch"
        );
        ensure_valid!(
            self.request_pool_indices.iter().all(|slot| *slot > 0),
            "forward input carries the reserved request slot"
        );
        ensure_valid!(
            self.query_lens.iter().all(|length| *length > 0),
            "forward query length must be positive"
        );
        ensure_valid!(
            self.seq_lens
                .iter()
                .zip(&self.query_lens)
                .all(|(total, query)| total >= query),
            "forward sequence length is shorter than its query"
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
    pub op_id: ComputationId,
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
            self.op_id.batch_id > 0,
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

/// Cursor and unit bounds for one diffusion decode operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request lineage that owns the media output.
    pub request_key: RequestKey,
    /// Decode operation receiving the params.
    pub op_id: ComputationId,
    /// First decoder unit assigned to the operation.
    pub cursor: u32,
    /// Maximum decoder units the operation may process.
    pub max_units: u32,
}

impl DecodeRange {
    /// Validates decode identity and unit capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.op_id.batch_id > 0,
            "decode params operation id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode params unit bound must be positive"
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
pub struct ScheduleBatch {
    /// Submission batch identity assigned by the executor.
    pub batch_id: u64,
    /// Physical invocation identity, strictly increasing in each worker's Submit order.
    /// Logical batch allocation and result completion may occur in a different order.
    pub run_id: u64,
    /// Monotonic sequence shared by collective participants.
    pub collective_seq: u64,
    /// Operations executed by this invocation.
    pub operations: Vec<ScheduledRequest>,
    /// Complete scheduler-owned KV page tables required by the operations.
    pub block_tables: Vec<BlockTable>,
    /// Physical KV pages newly allocated for this invocation.
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Columnar model inputs carried directly by the worker message.
    #[serde(flatten)]
    pub forward: ForwardBatch,
    /// Scheduler-owned latent trajectory allocations.
    pub latent_params: Vec<LatentParams>,
    /// Scheduler-owned media decode allocations.
    pub decode_ranges: Vec<DecodeRange>,
    /// Address spans for persistent cross-operation buffers.
    pub buffer_allocations: Vec<BufferAllocation>,
    /// Ordered request-state and buffer-lifetime commands.
    pub commands: Vec<BatchCommand>,
    /// Host-supplied input product values matched by `TensorRef` identity.
    pub input_products: Vec<TensorPublication>,
    /// Imported KV publications consumed by explicit cache installation computations.
    pub kv_inputs: Vec<KvTransfer>,
}

impl ScheduleBatch {
    /// Constructs a run with admissions and operations using default metadata.
    pub fn new(
        batch_id: u64,
        admissions: Vec<NewRequest>,
        operations: Vec<ScheduledRequest>,
    ) -> Self {
        Self {
            batch_id,
            run_id: batch_id,
            collective_seq: batch_id.max(1),
            operations,
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward: ForwardBatch::default(),
            latent_params: Vec::new(),
            decode_ranges: Vec::new(),
            buffer_allocations: Vec::new(),
            commands: admissions
                .into_iter()
                .map(|request| BatchCommand::Start { request })
                .collect(),
            input_products: Vec::new(),
            kv_inputs: Vec::new(),
        }
    }

    /// Iterates over operations in submission order.
    pub fn operations(&self) -> impl Iterator<Item = &ScheduledRequest> {
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
    pub fn with_input_products(mut self, input_products: Vec<TensorPublication>) -> Self {
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

        let mut computation_ids = HashSet::with_capacity(self.operations.len());
        for operation in &self.operations {
            operation.validate()?;
            ensure_valid!(
                operation.op_id.batch_id == self.batch_id,
                "computation identity belongs to another logical batch"
            );
            ensure_valid!(
                computation_ids.insert(operation.op_id),
                "a submission batch repeats a computation identity"
            );
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

        self.forward.validate(self.operations.len())?;

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
                operation.code,
                Computation::Pipeline(PipelineStage::LatentPreparation)
                    | Computation::Pipeline(PipelineStage::Denoising)
            ) || operation.latent_input.is_some();
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
                operation.code,
                Computation::Pipeline(PipelineStage::LatentPreparation)
                    | Computation::Pipeline(PipelineStage::Denoising)
            ) || operation.latent_input.is_some();
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
                    operation.code,
                    Computation::Pipeline(
                        PipelineStage::VideoDecoding
                            | PipelineStage::AudioDecoding
                            | PipelineStage::VideoEncoding
                            | PipelineStage::AudioEncoding
                    )
                ),
                "decode params does not name media decode work"
            );
            // Audio reconstruction divides the sample timeline into as many
            // media units as its component has ranks, and one round covers
            // them all, so an audio decode range starts at the first unit.
            ensure_valid!(
                !matches!(
                    operation.code,
                    Computation::Pipeline(PipelineStage::AudioDecoding)
                ) || (params.cursor == 0 && params.max_units >= 1),
                "audio decode range must start at the first media unit"
            );
            // Audio encoding consumes the assembled track as one host call.
            ensure_valid!(
                !matches!(
                    operation.code,
                    Computation::Pipeline(PipelineStage::AudioEncoding)
                ) || (params.cursor == 0 && params.max_units == 1),
                "audio encode range must address its single sample stream"
            );
        }

        for operation in &self.operations {
            let needs_range = matches!(
                operation.code,
                Computation::Pipeline(
                    PipelineStage::VideoDecoding
                        | PipelineStage::AudioDecoding
                        | PipelineStage::VideoEncoding
                        | PipelineStage::AudioEncoding
                )
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
            for output in operation.buffer_outputs() {
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
            (RequestKey, u8, Option<BufferId>),
            BatchCommand,
        > = std::collections::HashMap::new();
        for command in &self.commands {
            command.validate()?;
            let identity = (
                command.request_key(),
                command.variant_index(),
                match command {
                    BatchCommand::Free { buffer } => Some(*buffer),
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
                    .tensor_inputs()
                    .chain(operation.predicate.as_ref().into_iter())
            })
            .collect::<HashSet<_>>();

        let mut supplied_inputs = HashSet::with_capacity(self.input_products.len());
        for payload in &self.input_products {
            ensure_valid!(
                declared_inputs.contains(&payload.product),
                "an input product payload is not declared by any operation"
            );
            ensure_valid!(
                supplied_inputs.insert(&payload.product),
                "a submission batch repeats an input product payload"
            );
            payload.validate()?;
        }

        let mut kv_sources = HashSet::new();
        for publication in &self.kv_inputs {
            publication.validate()?;
            ensure_valid!(
                kv_sources.insert(publication.source),
                "run repeats a KV input"
            );
            let consumers = self
                .operations()
                .filter(|operation| operation.kv_input == Some(publication.source))
                .collect::<Vec<_>>();
            ensure_valid!(
                consumers.len() == 1
                    && consumers[0].code == Computation::Transfer(TransferMode::KvInstall),
                "KV transfer requires one installation consumer"
            );
            let bytes = publication.tensors.iter().try_fold(0_u64, |sum, tensor| {
                Ok::<_, ValidationError>(sum.saturating_add(tensor.validate()?))
            })?;
            ensure_valid!(
                bytes <= consumers[0].bounds.max_transfer_bytes,
                "KV input exceeds its installation transfer-byte bound"
            );
        }

        Ok(())
    }
}

/// One independently ready subset of a physical run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchOutput {
    /// Submission batch identity copied from the run.
    pub batch_id: u64,
    /// Physical invocation identity copied from the run.
    pub run_id: u64,
    /// Operation completions contained in this fragment.
    pub completions: Vec<RequestOutput>,
    /// Product values published by completed operations.
    pub products: Vec<TensorPublication>,
    /// Visibility result for atomic product registration.
    pub registration: RegistrationAck,
    /// Aggregate worker execution time in microseconds, when measured.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported by the worker.
    pub forward_stats: Option<ForwardStats>,
    /// Whether all operations and command-owned physical resources have retired.
    /// Free/Finish runs publish a separate empty terminal fragment after any
    /// operation fragments, even when retirement is immediately ready.
    pub done: bool,
}

impl BatchOutput {
    /// Iterates over operation completions in report order.
    pub fn completions(&self) -> impl Iterator<Item = &RequestOutput> {
        self.completions.iter()
    }

    /// Iterates over resolved product payloads in report order.
    pub fn products(&self) -> impl Iterator<Item = &TensorPublication> {
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
                completion.op_id.batch_id == self.batch_id,
                "completion identity belongs to another logical batch"
            );
            ensure_valid!(
                identities.insert(completion.op_id),
                "completion report repeats an operation"
            );
        }
        for payload in &self.products {
            payload.validate()?;
        }
        Ok(())
    }
}
