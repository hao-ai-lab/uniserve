//! Request identities, call descriptors, admissions, and execution batches.
//!
//! These are the semantic types of the scheduler-to-worker protocol,
//! independent of wire framing. The engine lowers each scheduled batch into a
//! [`Batch`] (`ExecutionBatch::into_protocol`, which validates it), projects
//! it onto the calls each rank owns (`rank_projection`, which narrows forward
//! rows with `ForwardBatch::select` and sets each call's `consumer_slots`),
//! and [`crate::codec`] encodes it as FlatBuffers. A rank answers with one
//! [`BatchOutput`] of per-call [`RequestOutput`] completions. The Python
//! worker exposes native request, call and buffer identifiers alongside the
//! numerical records in `uniserve_worker.protocol`.
//!
//! Types with invariants carry a `validate` method. [`Batch::validate`]
//! reaches each nested one and adds the checks that span calls. The
//! `ensure_valid!` and `invalid_message!` macros come from the crate root.

use super::*;

/// `(engine_id, request_id, request_epoch)`.
///
/// The scheduler stamps each admission with the next value of its epoch
/// counter, so a reused request identifier yields a distinct key and no call
/// or product reference aliases across requests or epochs.
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

/// The numerical mode of a token model forward.
///
/// `Prefill`, `Decode` and `Verify` extend a request's KV cache causally.
/// `TokenDenoising` runs one denoising pass over rows of a token canvas that
/// read the request's cached prefix without writing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ForwardMode {
    Prefill,
    Decode,
    Verify,
    TokenDenoising,
}

impl ForwardMode {
    /// Every mode in declaration order.
    ///
    /// `worker-ipc-py` builds its Python enum table from this array and
    /// indexes it with `mode as usize`, so the order must match the
    /// declaration order.
    pub const ALL: [Self; 4] = [
        Self::Prefill,
        Self::Decode,
        Self::Verify,
        Self::TokenDenoising,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::TokenDenoising => "token_denoising",
        }
    }
}

/// A concrete media-input, encoder, diffusion, decoder, or media-output
/// computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum MediaCall {
    /// Decodes a video request's condition media on a host rank into the
    /// inputs of the vision and latent encoders.
    MediaReading,
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

impl MediaCall {
    /// Every media call in declaration order.
    ///
    /// `worker-ipc-py` indexes a Python enum table built from this array with
    /// `call as usize`, so the order must match the declaration order.
    pub const ALL: [Self; 12] = [
        Self::MediaReading,
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
            Self::MediaReading => "media_reading",
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

/// The storage action performed between concrete producers and consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum TransferMode {
    Tensor,
    KvExport,
    KvInstall,
}

impl TransferMode {
    /// Every transfer mode in declaration order.
    ///
    /// `worker-ipc-py` indexes a Python enum table built from this array with
    /// `mode as usize`, so the order must match the declaration order.
    pub const ALL: [Self; 3] = [Self::Tensor, Self::KvExport, Self::KvInstall];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tensor => "tensor",
            Self::KvExport => "kv_export",
            Self::KvInstall => "kv_install",
        }
    }
}

/// Exactly one computation classification. The sum preserves the distinct
/// forward, media, and storage contracts without parallel opcode metadata.
///
/// Serde represents a kind by its inner enum's snake_case name alone
/// (`untagged`), which round-trips only while the names of [`ForwardMode`],
/// [`MediaCall`], and [`TransferMode`] stay disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CallKind {
    Forward(ForwardMode),
    Media(MediaCall),
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

impl CallKind {
    /// Computations accepted as individual scheduled request items.
    ///
    /// Mirrored, in the same order, by `CALL_KINDS` in
    /// `uniserve_worker.protocol.call`; the worker's reports list supported
    /// calls in this order.
    pub const ALL: [Self; 19] = [
        Self::Forward(ForwardMode::Prefill),
        Self::Forward(ForwardMode::Decode),
        Self::Forward(ForwardMode::Verify),
        Self::Media(MediaCall::MediaReading),
        Self::Forward(ForwardMode::TokenDenoising),
        Self::Media(MediaCall::VisionEncoding),
        Self::Media(MediaCall::LatentEncoding),
        Self::Media(MediaCall::TextEncoding),
        Self::Media(MediaCall::LatentPreparation),
        Self::Media(MediaCall::Denoising),
        Self::Media(MediaCall::ImageDecoding),
        Self::Media(MediaCall::VideoDecoding),
        Self::Media(MediaCall::AudioDecoding),
        Self::Media(MediaCall::VideoEncoding),
        Self::Media(MediaCall::AudioEncoding),
        Self::Media(MediaCall::Muxing),
        Self::Transfer(TransferMode::Tensor),
        Self::Transfer(TransferMode::KvExport),
        Self::Transfer(TransferMode::KvInstall),
    ];

    /// Returns whether a call of this kind advances its request's state.
    ///
    /// The worker mirrors this set in `uniserve_worker.protocol.call`; its
    /// `RequestPool` uses it to choose the call a request's next call follows
    /// and the `Ok` results that move the request's `state_call_id`.
    pub const fn advances_state(self) -> bool {
        matches!(
            self,
            Self::Forward(_) | Self::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forward(mode) => mode.as_str(),
            Self::Media(call) => call.as_str(),
            Self::Transfer(mode) => mode.as_str(),
        }
    }
}

/// Hard resource maxima the scheduler reserves before a call runs.
///
/// [`Call::validate`] checks `max_tokens` against `input_token_ids` and
/// `max_latent_bytes` against encoder, latent, and image outputs;
/// [`Batch::validate`] checks `max_transfer_bytes` against installed KV
/// inputs. Neither checks `max_kv_units` or `max_completion_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Bounds {
    /// Maximum tokens the call may process or produce.
    pub max_tokens: u32,
    /// Paged-KV units newly allocated to the request's tables and first
    /// declared to workers by this call.
    pub max_kv_units: u32,
    /// Maximum latent storage in bytes.
    pub max_latent_bytes: u64,
    /// Maximum host-visible completion data in bytes.
    pub max_completion_bytes: u64,
    /// Maximum cross-call transfer data in bytes.
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

/// Deterministic random-draw coordinates for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rng {
    /// Request-level random seed.
    pub seed: u64,
    /// First semantic draw index assigned to the call.
    pub semantic_index_base: u64,
    /// Mapping from semantic work to deterministic draws.
    pub draw_layout: DrawLayout,
}

/// Branch-local token processor inputs for one sampling call.
///
/// Token ids in every field are strictly increasing: [`Call::validate`]
/// rejects any other order, and [`Self::canonicalize`] produces it.
/// `allowed_token_ids` distinguishes no whitelist (`None`) from a present empty
/// whitelist, which deterministically represents an invalid all-masked
/// distribution. Penalty token counts are not carried here: they are a
/// device-resident committed base plus bounded per-call deltas folded after
/// sampling accepts tokens, so no host token history participates in a
/// successor's sampling input.
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

/// The coordinates one call executes at.
///
/// A rank would otherwise derive these by chaining from the calls that
/// preceded it. The engine holds the request state they come from, so it states
/// them and the rank asserts its own ledger agrees.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallCoordinates {
    /// Position of this call's first token in the request's logical sequence.
    pub logical_position: u32,
    /// Tokens whose KV a numerical call may attend to at submission.
    pub kv_visible_len: u32,
    /// Tokens whose KV is initialized at submission; never below the visible extent.
    pub kv_computed_len: u32,
    /// Denoising steps completed for this request at submission.
    pub flow_step: u32,
}

impl CallCoordinates {
    /// Validates the containment the coordinates must satisfy.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.kv_visible_len <= self.kv_computed_len,
            "call coordinates place visible KV beyond the computed extent"
        );
        Ok(())
    }
}

/// One denoising step of a block-diffusion request's resident canvas.
///
/// The request's admitted `ArRequestParams::canvas` fixes the sampling; the
/// worker holds the canvas, its self-conditioning input and its stopping
/// history in the request's slot. `block` counts the blocks already
/// committed to the request's context and `step` the steps already run on
/// this canvas; step zero starts the canvas from random tokens. The step
/// that stops the canvas, by convergence or by reaching the admitted step
/// limit, reports the canvas's argmax tokens as its committed tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CanvasStep {
    /// Blocks committed to the request's context before this canvas.
    pub block: u32,
    /// Denoising steps already run on this canvas.
    pub step: u32,
}

/// Candidate log-probabilities a token-denoising call reads at canvas slots.
///
/// The call's `input_token_ids` hold its canvas rows back to back. Slot `i`
/// is canvas token `slot_tokens[i]` of that sequence and reads the candidate
/// token ids `candidate_ids[candidate_offsets[i]..candidate_offsets[i + 1]]`.
/// The worker reports, in `candidate_ids` order, each candidate's natural-log
/// probability under the log-softmax over the full vocabulary of the model's
/// logits at its slot (`RequestOutput::candidate_logprobs`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Readout {
    /// Index of each slot's token in the call's concatenated canvas rows.
    pub slot_tokens: Vec<u32>,
    /// Offsets of each slot's candidates in `candidate_ids`: one more entry
    /// than there are slots, starting at zero and ending at the id count.
    pub candidate_offsets: Vec<u32>,
    /// Candidate token ids of every slot, slot after slot.
    pub candidate_ids: Vec<u32>,
}

impl Readout {
    /// Number of candidate log-probabilities the readout reports.
    pub fn candidate_count(&self) -> usize {
        self.candidate_ids.len()
    }

    /// Validates the slot layout against a call of `canvas_tokens` canvas
    /// tokens.
    ///
    /// Slots address canvas tokens in increasing order, each reads at least
    /// one candidate, and the offsets partition `candidate_ids` in slot
    /// order.
    pub fn validate(&self, canvas_tokens: usize) -> ValidationResult<()> {
        ensure_valid!(!self.slot_tokens.is_empty(), "readout reads no slot");
        ensure_valid!(
            self.candidate_offsets.len() == self.slot_tokens.len() + 1,
            "readout candidate offsets do not bound every slot"
        );
        ensure_valid!(
            self.candidate_offsets.first() == Some(&0)
                && self.candidate_offsets.last().map(|&end| end as usize)
                    == Some(self.candidate_ids.len())
                && self
                    .candidate_offsets
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "readout candidate offsets do not partition the candidates"
        );
        ensure_valid!(
            self.slot_tokens
                .iter()
                .all(|&token| (token as usize) < canvas_tokens),
            "readout slot lies outside the call's canvas rows"
        );
        ensure_valid!(
            self.slot_tokens.windows(2).all(|pair| pair[0] < pair[1]),
            "readout slots are not in canvas order"
        );
        Ok(())
    }
}

/// Component used when a deployment does not partition a model by capability.
pub const DEFAULT_COMPONENT: &str = "model";

/// One image block of a context prefill.
///
/// The block writes the vision-encoder product `feature` into KV as one
/// attention block, before the call's input token `offset`: the call's
/// context is its input tokens `[0, offset)`, then the block, then the
/// tokens from `offset` on. Blocks sharing an offset follow in list order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionInput {
    /// Input tokens of the call that precede the block.
    pub offset: u32,
    /// Vision-encoder features the block injects.
    pub feature: TensorRef,
}

/// One immutable computation with its identity, data dependencies, and output limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Call {
    /// Request that owns the call.
    pub request_key: RequestKey,
    /// Logical batch and selection ordinal, preserved when the batch is
    /// projected onto a rank.
    pub call_id: CallId,
    /// Coordinates this call executes at, so a rank does not derive them.
    pub coordinates: CallCoordinates,
    /// Component bound within the selected Worker.
    pub component: String,
    /// Kind performed by this call.
    pub code: CallKind,
    /// Scheduler-declared limits for the computation and its outputs.
    pub bounds: Bounds,
    /// Data dependencies consumed in input order.
    pub inputs: Vec<TensorRef>,
    /// Outputs allocated to this producer.
    pub outputs: Vec<TensorRef>,
    /// Acknowledgment slots of the ranks that read this call's products.
    ///
    /// A producing rank cannot name its readers: it knows its own component,
    /// not which component consumes what it publishes, and a product is read
    /// in a later batch than the one producing it. The head states them, and
    /// a published product retires once each has acknowledged it. The
    /// executing rank's own slot is never listed.
    pub consumer_slots: Vec<u32>,
    /// Source token scalar transported between components.
    pub token_input: Option<TensorRef>,
    /// Sampled token and continuation bit in one I64 scalar. The worker packs
    /// it with `tagged_token_values` in `uniserve_worker.sampling.sampler`.
    pub token_output: Option<TensorRef>,
    /// Image blocks a prefill injects, in context order: each block's
    /// vision-encoder feature and the input token it precedes.
    pub vision_inputs: Vec<VisionInput>,
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
    /// Device predicate that enables execution: a U8 flag or a packed I64
    /// continuation scalar. A false predicate yields [`CallStatus::Predicated`].
    pub predicate: Option<TensorRef>,
    /// Deterministic sampling coordinates.
    pub rng: Option<Rng>,
    /// Host-side sampler constraints for this computation; absent uses admission defaults.
    pub sampling_state: Option<SamplingState>,
    /// Host-known prompt, draft, or decode input tokens in model input order,
    /// or a token-denoising call's canvas rows back to back. Empty when
    /// continuation reads its predecessor's device token.
    pub input_token_ids: Vec<u32>,
    /// Candidate slots a token-denoising readout reports; absent for every
    /// other call.
    pub readout: Option<Readout>,
    /// The step a token-denoising call runs on its request's generation
    /// canvas; absent for every other call.
    pub canvas: Option<CanvasStep>,
    /// Encoded source image in base64, consumed by a vision or latent encoder.
    /// Dispatch and in-flight matching share these immutable bytes.
    pub input_image: Option<std::sync::Arc<str>>,
    /// Published or installed cache extent consumed by this computation.
    pub kv_input: Option<BufferId>,
    /// Identity under which this computation publishes or installs cache state.
    pub kv_output: Option<BufferId>,
}

impl Call {
    /// A context prefill interleaves prompt tokens and input-image blocks.
    pub fn writes_context(&self) -> bool {
        self.code == CallKind::Forward(ForwardMode::Prefill)
            && !self.vision_inputs.is_empty()
            && self.completion_output.is_none()
    }

    /// Image feedback or a latent input writes one visual-state row.
    /// Context-prefill vision blocks instead belong to its prompt segments.
    pub fn writes_visual_state(&self) -> bool {
        !self.writes_context()
            && (!self.vision_inputs.is_empty() || self.latent_feature_input.is_some())
    }

    /// Tensor dependencies in the computation signature, excluding its predicate.
    ///
    /// Includes `token_input` and `latent_input`, which [`Self::buffer_inputs`]
    /// omits.
    pub fn tensor_inputs(&self) -> impl Iterator<Item = &TensorRef> {
        self.inputs
            .iter()
            .chain(self.token_input.iter())
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
            .chain(self.latent_input.iter())
            .chain(self.image_input.iter())
    }

    /// All tensor declarations owned by this computation.
    ///
    /// Includes the device scalars (token, completion, transition) and
    /// `latent_output`, which [`Self::buffer_outputs`] omits.
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
            .chain(self.vision_inputs.iter().map(|input| &input.feature))
            .chain(self.latent_feature_input.iter())
            .chain(self.image_input.iter())
    }

    /// Tensor outputs backed by scheduler-allocated persistent buffers.
    ///
    /// [`Batch::validate`] requires a [`BufferAllocation`] large enough for each
    /// of these. Latent trajectories are addressed through [`LatentParams`]
    /// instead.
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

    /// Validates the invariants one call carries on its own.
    ///
    /// Checks identity, component, coordinates, the token bound, canonical
    /// sampling-state sets, the encoded-image source, KV export
    /// identities, product ownership, generations, output indices, dtypes and
    /// byte bounds, cross-request inputs, and the predicate. `rng` is not
    /// checked. Relationships to other calls and to the batch's allocation
    /// tables belong to [`Batch::validate`].
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish call identity, component, coordinates, and token capacity.
        ensure_valid!(self.call_id.batch_id > 0, "call id must be positive");
        ensure_valid!(
            !self.component.is_empty(),
            "call component must not be empty"
        );
        self.coordinates.validate()?;

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
                        CallKind::Media(MediaCall::VisionEncoding | MediaCall::LatentEncoding)
                    )
                    && self.image_input.is_none(),
                "encoded image requires an image encoder without another image source"
            );
        }

        // A token-denoising call either reads candidate log-probabilities at
        // slots of the canvas rows it carries, or runs one step of its
        // request's resident generation canvas of `max_tokens` tokens. Either
        // way it produces no sampled token output.
        let denoises = self.code == CallKind::Forward(ForwardMode::TokenDenoising);
        ensure_valid!(
            u8::from(self.readout.is_some()) + u8::from(self.canvas.is_some())
                == u8::from(denoises),
            "a token-denoising call carries exactly one readout or canvas step"
        );
        ensure_valid!(
            !denoises || (self.token_output.is_none() && self.transition_output.is_none()),
            "token denoising samples no token output"
        );
        if let Some(readout) = &self.readout {
            ensure_valid!(
                !self.input_token_ids.is_empty(),
                "a readout requires canvas tokens"
            );
            readout.validate(self.input_token_ids.len())?;
        }
        if self.canvas.is_some() {
            ensure_valid!(
                self.input_token_ids.is_empty() && self.bounds.max_tokens > 0,
                "a canvas step denoises its resident canvas of max_tokens tokens"
            );
        }

        // Every output must be uniquely owned by this producer and fit the
        // resource class reserved for the call. The KV output shares the
        // output-index space with the tensor outputs.
        let mut output_indices = HashSet::with_capacity(self.outputs.len());
        let publishes_kv = matches!(
            self.code,
            CallKind::Transfer(TransferMode::KvExport | TransferMode::KvInstall)
        );
        ensure_valid!(
            self.kv_output.is_some() == publishes_kv,
            "KV export or installation requires one cache output identity"
        );
        if let Some(output) = self.kv_output {
            output.validate()?;
            ensure_valid!(
                output.owner == self.request_key && output.producer_call_id == self.call_id,
                "KV output is not owned by its producing computation"
            );
            output_indices.insert(output.output_index);
        }
        let consumes_kv = matches!(
            self.code,
            CallKind::Transfer(TransferMode::KvInstall)
                | CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
        );
        if let Some(input) = self.kv_input {
            input.validate()?;
            ensure_valid!(
                consumes_kv && input.owner == self.request_key,
                "KV input is incompatible with its computation or request"
            );
        }
        ensure_valid!(
            self.code != CallKind::Transfer(TransferMode::KvInstall) || self.kv_input.is_some(),
            "KV installation requires a source export"
        );

        for output in self.tensor_outputs() {
            output.validate()?;
            ensure_valid!(
                output.request_key == self.request_key && output.producer_call_id == self.call_id,
                "an output product is not owned by its producing call"
            );
            ensure_valid!(
                output.generation > 0,
                "an output product has no logical generation"
            );
            ensure_valid!(
                output_indices.insert(output.output_index),
                "call repeats an output index"
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
                self.code == CallKind::Transfer(TransferMode::Tensor),
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

        // Only encoder features (vision and latent-feature inputs) can be
        // shared across requests.
        for input in self.tensor_inputs() {
            input.validate()?;
            ensure_valid!(
                input.request_key == self.request_key
                    || self
                        .vision_inputs
                        .iter()
                        .any(|vision| &vision.feature == input)
                    || Some(input) == self.latent_feature_input.as_ref(),
                "request-local tensor belongs to another request"
            );
        }
        if let Some(predicate) = &self.predicate {
            predicate.validate()?;
            ensure_valid!(
                predicate.request_key == self.request_key,
                "computation predicate belongs to another request"
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

/// Terminal status of one call's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CallStatus {
    /// Execution completed successfully.
    Ok = 0,
    /// The call was skipped because its predicate was false.
    Predicated = 1,
    /// Execution failed with a deterministic error code.
    Error = 2,
}

/// A deterministic error class for a failed completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum ErrorCode {
    /// Call metadata or product declarations are invalid.
    InvalidCall = 0,
    /// A reserved or physical resource could not satisfy the call.
    ResourceExhausted = 1,
    /// Device computation failed.
    ComputeError = 2,
    /// The request was cancelled before completion.
    Cancelled = 3,
    /// An invariant failed within the worker.
    Internal = 4,
}

/// Device-observed finish candidates for a token call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct FinishFlags {
    /// Whether an end-of-sequence token was selected.
    pub eos: bool,
    /// Whether the configured length limit was reached.
    pub length: bool,
    /// Whether a configured stop condition matched.
    pub stop: bool,
}

/// Per-call timing counters. Accounting only; never state identity.
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
    /// Artifact stored in a POSIX shared-storage object.
    PosixShm {
        /// Shared-storage object name without a path separator.
        name: String,
    },
}

impl ArtifactHandle {
    /// Validates the transport-specific artifact identifier.
    pub fn validate(&self) -> ValidationResult<()> {
        match self {
            Self::PosixShm { name } => ensure_valid!(
                !name.is_empty() && !name.contains('/'),
                "POSIX shared-storage artifact name is invalid"
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

/// The completion record a worker emits once for every call, after its copy
/// event is query-ready and its pinned fields are validated on the host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestOutput {
    /// Request completed by the call.
    pub request_key: RequestKey,
    /// Request-local call identifier.
    pub call_id: CallId,
    /// Terminal execution status.
    pub status: CallStatus,
    /// Allocation generations for products emitted by the call.
    pub product_generations: Vec<u32>,
    /// Error classification when `status` is [`CallStatus::Error`].
    pub error_code: Option<ErrorCode>,
    /// Worker timing measurements for the call.
    pub timing_counters: TimingCounters,
    /// Kind that produced this result; must match the submitted call.
    pub code: CallKind,
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
    /// Natural-log probability of every candidate of a token-denoising
    /// readout, in the call's `Readout::candidate_ids` order.
    pub candidate_logprobs: Vec<f32>,
    /// Device-observed terminal conditions.
    pub finish_flags: FinishFlags,
    /// Completed artifact and its external storage lifetime, when produced.
    pub media_output: Option<MediaOutput>,
    /// Physical KV export, including this rank's tensor locations.
    pub kv_output: Option<KvTransfer>,
}

impl RequestOutput {
    /// Validates completion identity, status, accepted lengths, and product generations.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "completion call id must be positive"
        );
        if let Some(export) = &self.kv_output {
            export.validate()?;
            ensure_valid!(
                self.status == CallStatus::Ok
                    && self.code == CallKind::Transfer(TransferMode::KvExport)
                    && export.source.owner == self.request_key
                    && export.source.producer_call_id == self.call_id,
                "KV export does not belong to its successful completion"
            );
        }

        ensure_valid!(
            self.kv_visible_len <= self.kv_computed_len,
            "completion selected KV length exceeds computed length"
        );
        match self.status {
            CallStatus::Error => ensure_valid!(
                self.error_code.is_some(),
                "an error completion must carry an error code"
            ),
            CallStatus::Ok | CallStatus::Predicated => ensure_valid!(
                self.error_code.is_none(),
                "a non-error completion must not carry an error code"
            ),
        }
        ensure_valid!(
            self.candidate_logprobs.is_empty()
                || (self.status == CallStatus::Ok
                    && self.code == CallKind::Forward(ForwardMode::TokenDenoising)),
            "candidate log-probabilities belong to a successful token-denoising completion"
        );
        if self.status == CallStatus::Predicated {
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
    /// Establish one request before its first call on a pool.
    Start {
        /// Static request state installed by the worker.
        request: Box<NewRequest>,
    },
    /// Close an exact request epoch after admitted work and physical readers finish.
    Finish {
        /// Exact request epoch to close.
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
    /// Returns the request targeted by this command.
    pub fn request_key(&self) -> RequestKey {
        match self {
            Self::Start { request } => request.request_key,
            Self::Finish { request_key, .. } => *request_key,
            Self::Free { buffer } => buffer.owner,
        }
    }

    /// Returns a discriminant that distinguishes command variants.
    ///
    /// [`Batch::validate`] uses it in the identity that detects repeated
    /// commands.
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
    /// Sampling policy shared by autoregressive calls.
    pub sampling: SamplingParams,
    /// Token identifiers used for negative-conditioning input.
    pub negative_token_ids: Vec<u32>,
    /// Canonical token identifiers that terminate generation.
    pub finish_token_ids: Vec<u32>,
    /// Logical position assigned to the first request token.
    pub initial_position: u32,
    /// Block-diffusion sampling of a request that generates its text in
    /// canvases; its seed is `sampling.seed`.
    pub canvas: Option<CanvasSampling>,
}

pub use uniserve_core::{CanvasSampling, DiffusionSamplingParams};

/// What a video request admits besides its sampling controls.
///
/// The engine forms it from the `DiffusionRequest` it admits; the conditions'
/// media stay published by the engine until the request retires, so a
/// worker reads them by locator while the request lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoAdmission {
    /// The task the request runs.
    pub task: uniserve_core::VideoTask,
    /// The denoiser's AdaLN tag of each prompt token: 0 for a vision token
    /// or vision marker, 1 for text.
    pub text_tags: Vec<u8>,
    /// The conditions in request order.
    pub conditions: Vec<uniserve_core::VideoCondition>,
}

/// Request-start framing. Carries the per-domain parameters a request
/// needs before its calls run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewRequest {
    /// Globally unique request identity.
    pub request_key: RequestKey,
    /// Scheduler-assigned stable request-state row. Index zero is reserved for
    /// inactive graph padding and never identifies a live request.
    pub request_pool_idx: u32,
    /// Autoregressive request parameters, when applicable.
    pub ar: Option<ArRequestParams>,
    /// Image request parameters, when applicable.
    pub image: Option<ImageParams>,
    /// Diffusion request parameters, when applicable.
    pub diffusion: Option<DiffusionSamplingParams>,
    /// A video request's task, presentation tags and conditions; present
    /// exactly with `diffusion`.
    pub video: Option<VideoAdmission>,
    /// Tokenized conditioning supplied at diffusion admission.
    pub prompt_token_ids: Vec<u32>,
    /// Number of input images the request carries. Model image preprocessing
    /// that divides a pixel budget among a request's input images reads it
    /// when it encodes one of them.
    pub input_images: u32,
}

impl NewRequest {
    /// Constructs static autoregressive or multimodal admission state.
    ///
    /// Checks only the request-pool index and that `ar` or `image` is present;
    /// it does not run [`Self::validate`] on the family parameters.
    pub fn new(
        request_key: RequestKey,
        request_pool_idx: u32,
        ar: Option<ArRequestParams>,
        image: Option<ImageParams>,
        input_images: u32,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        ensure_valid!(
            ar.is_some() || image.is_some(),
            "request start must declare autoregressive or image parameters"
        );
        let admission = Self {
            request_key,
            request_pool_idx,
            ar,
            image,
            diffusion: None,
            video: None,
            prompt_token_ids: Vec::new(),
            input_images,
        };
        Ok(admission)
    }

    /// Constructs static video admission state and validates it.
    pub fn new_media(
        request_key: RequestKey,
        request_pool_idx: u32,
        prompt_token_ids: Vec<u32>,
        diffusion: DiffusionSamplingParams,
        video: VideoAdmission,
    ) -> ValidationResult<Self> {
        ensure_valid!(request_pool_idx > 0, "request-pool index must be positive");
        let admission = Self {
            request_key,
            request_pool_idx,
            ar: None,
            image: None,
            diffusion: Some(diffusion),
            video: Some(video),
            prompt_token_ids,
            input_images: 0,
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
            self.ar.is_some() || self.image.is_some() || self.diffusion.is_some(),
            "request start must declare one runtime-family parameter set"
        );
        if let Some(ar) = &self.ar {
            ar.sampling.validate()?;
            if let Some(canvas) = &ar.canvas
                && let Some(parameter) = canvas.invalid_parameter()
            {
                bail_invalid!("invalid canvas sampling {parameter}");
            }

            ensure_valid!(
                ar.finish_token_ids.windows(2).all(|pair| pair[0] < pair[1]),
                "autoregressive finish token ids are not canonical"
            );
        }
        if let Some(branch) = &self.image {
            branch.validate()?;
        }
        if let Some(diffusion) = &self.diffusion {
            ensure_valid!(
                !self.prompt_token_ids.is_empty(),
                "diffusion prompt tokens must not be empty"
            );
            ensure_valid!(
                diffusion.num_frames > 0
                    && diffusion.video_units > 0
                    && self.prompt_token_ids.len() <= u32::MAX as usize
                    && diffusion.num_inference_steps > 0,
                "diffusion parameters are invalid"
            );
        }
        ensure_valid!(
            self.video.is_some() == self.diffusion.is_some(),
            "a video admission carries both its sampling and its inputs"
        );
        if let Some(video) = &self.video {
            ensure_valid!(
                video.text_tags.len() == self.prompt_token_ids.len(),
                "every video prompt token requires one tag"
            );
            ensure_valid!(
                (video.task == uniserve_core::VideoTask::T2va) == video.conditions.is_empty(),
                "only a conditioned video task carries conditions"
            );
            for condition in &video.conditions {
                if let Err(message) = condition.validate() {
                    bail_invalid!("{message}");
                }
            }
        }
        Ok(())
    }
}

/// One request slot's complete unit table in one KV cache group.
///
/// `unit_ids` lists the units of logical pages `start_page..`, page-major:
/// the group's `units_per_page` units of page `start_page` come first. Pages
/// before `start_page` have been released by a sliding-window group, and a
/// worker reads none of them. `allocated_tokens` is the absolute token extent
/// the table covers, up to the end of its last page; the worker checks it
/// against the group's page shape, which this record does not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTable {
    /// Scheduler-assigned request-state row that owns the table.
    pub request_pool_idx: u32,
    /// KV cache group addressed by the table.
    pub group_id: u32,
    /// First logical page the units cover; earlier pages are retired.
    pub start_page: u32,
    /// Physical unit identifiers of pages `start_page..`, page-major.
    pub unit_ids: Vec<UnitId>,
    /// Absolute token extent the table covers.
    pub allocated_tokens: u32,
}

/// Physical units newly acquired for one installed block table.
///
/// The worker resets every listed unit before a call reads or writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheUnitAllocation {
    /// Scheduler-assigned request-state row receiving the units.
    pub request_pool_idx: u32,
    /// KV cache group receiving the units.
    pub group_id: u32,
    /// Newly allocated physical unit identifiers.
    pub unit_ids: Vec<UnitId>,
}

/// Whether `units` are real units (never the zero sentinel) without repeats.
fn distinct_units(units: &[UnitId]) -> bool {
    units.iter().all(|unit| unit.0 > 0) && units.iter().collect::<HashSet<_>>().len() == units.len()
}

impl BlockTable {
    /// Validates table identity and unit uniqueness.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "block table request slot must be positive"
        );
        ensure_valid!(
            distinct_units(&self.unit_ids),
            "block table repeats a unit or carries unit zero"
        );
        Ok(())
    }
}

impl CacheUnitAllocation {
    /// Validates newly allocated unit identities and ownership.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "cache-unit allocation request slot must be positive"
        );
        ensure_valid!(
            !self.unit_ids.is_empty() && distinct_units(&self.unit_ids),
            "cache-unit allocation is empty, repeats a unit, or carries unit zero"
        );
        Ok(())
    }
}

/// Aligned columns defining the model's forward rows, independent of wire framing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardBatch {
    /// Physical call index for each row; logical computation IDs remain unchanged.
    ///
    /// [`Batch`] flattens this struct; the serialized key matches the
    /// `forward_call_indices` field of the worker schema and of the Python
    /// worker's `Batch`.
    #[serde(rename = "forward_call_indices")]
    pub call_indices: Vec<u32>,
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
        call_index: u32,
        request_pool_index: u32,
        seq_len: u32,
        query_len: u32,
        write_kv: bool,
    ) {
        self.call_indices.push(call_index);
        self.request_pool_indices.push(request_pool_index);
        self.seq_lens.push(seq_len);
        self.query_lens.push(query_len);
        self.write_kv.push(write_kv);
    }

    /// Move forward inputs into a physical batch, offsetting only physical row ownership.
    ///
    /// Fails without modifying this batch when an offset call index exceeds `u32`.
    pub fn append(&mut self, other: Self, call_offset: u32) -> ValidationResult<()> {
        // Offset every index before extending any column so a failure leaves
        // the columns aligned.
        let call_indices = other
            .call_indices
            .into_iter()
            .map(|index| {
                index.checked_add(call_offset).ok_or_else(|| {
                    invalid_message!(
                        "physical call index {index} with offset {call_offset} exceeds u32"
                    )
                })
            })
            .collect::<ValidationResult<Vec<_>>>()?;

        self.call_indices.extend(call_indices);
        self.request_pool_indices.extend(other.request_pool_indices);
        self.seq_lens.extend(other.seq_lens);
        self.query_lens.extend(other.query_lens);
        self.write_kv.extend(other.write_kv);
        Ok(())
    }

    /// Select a rank's calls and map their forward rows to the local call array.
    ///
    /// `calls` lists the batch-level call indices the rank owns, in the order
    /// of its projected call array. Rows of other calls are dropped, and each
    /// kept row's call index becomes its position in `calls`.
    pub fn select(&self, calls: &[usize]) -> Self {
        let mut selected = Self::default();
        for (row, index) in self.call_indices.iter().enumerate() {
            if let Some(local) = calls.iter().position(|call| *call == *index as usize) {
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
    pub fn validate(&self, call_count: usize) -> ValidationResult<()> {
        let rows = self.call_indices.len();
        ensure_valid!(
            self.request_pool_indices.len() == rows
                && self.seq_lens.len() == rows
                && self.query_lens.len() == rows
                && self.write_kv.len() == rows,
            "forward input columns have different lengths"
        );
        ensure_valid!(
            self.call_indices
                .iter()
                .all(|index| (*index as usize) < call_count),
            "forward call index is outside its batch"
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
    /// Request that owns the trajectory.
    pub request_key: RequestKey,
    /// Call that addresses the trajectory.
    pub call_id: CallId,
    /// Physical latent pages in logical order; empty for request-owned tensors.
    pub page_table: Vec<u32>,
    /// Logical units stored in pages; zero when the trajectory uses request tensors.
    pub latent_units: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// First denoising step assigned to the call.
    pub start_step: u32,
    /// Number of denoising steps assigned to the call.
    pub step_count: u32,
}

impl LatentParams {
    /// Validates latent page identities, shape, and byte bounds.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "latent params call id must be positive"
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

/// Cursor and unit bounds for one diffusion decode call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request that owns the media output.
    pub request_key: RequestKey,
    /// Decode call receiving the params.
    pub call_id: CallId,
    /// First decoder unit assigned to the call.
    pub cursor: u32,
    /// Maximum decoder units the call may process.
    pub max_units: u32,
}

impl DecodeRange {
    /// Validates decode identity and unit capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "decode params call id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode params unit bound must be positive"
        );
        Ok(())
    }
}

/// Scheduler-selected address span for one cross-call persistent buffer.
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
    /// Validates the buffer identity and a non-empty span whose end does not
    /// overflow `u64`.
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

/// One numerical call on one component, with the calls of every request that
/// participates in it. Execution domains, attention selection, and captured
/// buckets are derived by the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// Batch identity assigned by the executor, strictly increasing in each
    /// worker's Submit order.
    pub batch_id: u64,
    /// Monotonic sequence shared by collective participants.
    pub collective_seq: u64,
    /// Calls executed by this invocation.
    pub calls: Vec<Call>,
    /// Complete scheduler-owned KV page tables required by the calls.
    pub block_tables: Vec<BlockTable>,
    /// Physical KV units newly allocated for this invocation.
    pub new_cache_units: Vec<CacheUnitAllocation>,
    /// Columnar model inputs carried directly by the worker message.
    #[serde(flatten)]
    pub forward: ForwardBatch,
    /// Scheduler-owned latent trajectory allocations.
    pub latent_params: Vec<LatentParams>,
    /// Scheduler-owned media decode allocations.
    pub decode_ranges: Vec<DecodeRange>,
    /// Address spans for persistent cross-call buffers.
    pub buffer_allocations: Vec<BufferAllocation>,
    /// Ordered request-state and buffer-lifetime commands.
    pub commands: Vec<BatchCommand>,
    /// Host-supplied input product values matched by `TensorRef` identity.
    pub input_products: Vec<TensorExport>,
    /// Imported KV exports consumed by explicit cache installation call kinds.
    pub kv_inputs: Vec<KvTransfer>,
}

impl Batch {
    /// Constructs a run with admissions and calls using default metadata.
    ///
    /// `collective_seq` is `batch_id`, or one when `batch_id` is zero;
    /// [`Self::validate`] requires it to be positive.
    pub fn new(batch_id: u64, admissions: Vec<NewRequest>, calls: Vec<Call>) -> Self {
        Self {
            batch_id,
            collective_seq: batch_id.max(1),
            calls,
            block_tables: Vec::new(),
            new_cache_units: Vec::new(),
            forward: ForwardBatch::default(),
            latent_params: Vec::new(),
            decode_ranges: Vec::new(),
            buffer_allocations: Vec::new(),
            commands: admissions
                .into_iter()
                .map(|request| BatchCommand::Start {
                    request: Box::new(request),
                })
                .collect(),
            input_products: Vec::new(),
            kv_inputs: Vec::new(),
        }
    }

    /// Iterates over calls in submission order.
    pub fn calls(&self) -> impl Iterator<Item = &Call> {
        self.calls.iter()
    }

    /// Returns the number of calls in this run.
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Attaches ordered control commands to the run.
    pub fn with_commands(mut self, commands: Vec<BatchCommand>) -> Self {
        self.commands.extend(commands);
        self
    }

    /// Iterates over static request admissions in submission order.
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request.as_ref()),
            _ => None,
        })
    }

    /// Attaches resolved input products to the run.
    pub fn with_input_products(mut self, input_products: Vec<TensorExport>) -> Self {
        self.input_products = input_products;
        self
    }

    /// Validates identities, call kinds, allocations, products, and commands.
    ///
    /// Runs every nested `validate`, including [`Call::validate`] for each
    /// call, and checks what spans calls, including one call per request, one
    /// kind and component per batch, allocation tables that match the calls
    /// addressing them, non-overlapping latent pages and buffer spans, and
    /// input products and KV inputs that some call declares.
    pub fn validate(&self) -> ValidationResult<()> {
        // Establish the run envelope before validating relationships within it.
        ensure_valid!(
            !self.calls.is_empty() || !self.commands.is_empty(),
            "a submission batch must carry at least one call or control"
        );
        ensure_valid!(
            self.collective_seq > 0,
            "run collective sequence must be positive"
        );

        let mut computation_ids = HashSet::with_capacity(self.calls.len());
        for call in &self.calls {
            call.validate()?;
            ensure_valid!(
                call.call_id.batch_id == self.batch_id,
                "computation identity belongs to another logical batch"
            );
            ensure_valid!(
                computation_ids.insert(call.call_id),
                "a submission batch repeats a computation identity"
            );
        }
        // A batch is one numerical call on one component: every call in it
        // performs the same computation through the same component, so a rank
        // executes it as a single homogeneous group and returns one result.
        ensure_valid!(
            self.calls.windows(2).all(|pair| {
                pair[0].code == pair[1].code && pair[0].component == pair[1].component
            }),
            "a submission batch mixes call kinds or components"
        );

        let calls = self
            .calls
            .iter()
            .map(|call| ((call.request_key, call.call_id), call))
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

        let mut allocation_ids = HashSet::with_capacity(self.new_cache_units.len());
        for allocation in &self.new_cache_units {
            allocation.validate()?;
            let identity = (allocation.request_pool_idx, allocation.group_id);
            ensure_valid!(
                allocation_ids.insert(identity),
                "run repeats a cache-unit allocation"
            );
            let table = tables.get(&identity).ok_or_else(|| {
                invalid_message!("cache-unit allocation has no matching block table")
            })?;
            let table_units = table.unit_ids.iter().collect::<HashSet<_>>();
            ensure_valid!(
                allocation
                    .unit_ids
                    .iter()
                    .all(|unit| table_units.contains(unit)),
                "cache-unit allocation is outside its block table"
            );
        }

        self.forward.validate(self.calls.len())?;

        // Latent pages are exclusive across every trajectory placed in a run.
        let mut latent_ids = HashSet::with_capacity(self.latent_params.len());
        let mut latent_pages = HashSet::new();
        for params in &self.latent_params {
            params.validate()?;
            let identity = (params.request_key, params.call_id);
            ensure_valid!(
                latent_ids.insert(identity),
                "run repeats a latent params identity"
            );
            let call = calls
                .get(&identity)
                .ok_or_else(|| invalid_message!("latent params does not name a run call"))?;
            let addresses_trajectory = matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising)
            ) || call.latent_input.is_some();
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

        for call in &self.calls {
            let needs_latent = matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising)
            ) || call.latent_input.is_some();
            ensure_valid!(
                !needs_latent || latent_ids.contains(&(call.request_key, call.call_id)),
                "call that addresses a trajectory has no latent params"
            );
        }

        // Decode ranges belong exactly to video and audio decoding and encoding
        // calls: each such call has one, and no other call does.
        let mut decode_ids = HashSet::with_capacity(self.decode_ranges.len());
        for params in &self.decode_ranges {
            params.validate()?;
            let identity = (params.request_key, params.call_id);
            ensure_valid!(
                decode_ids.insert(identity),
                "run repeats a decode params identity"
            );
            let call = calls
                .get(&identity)
                .ok_or_else(|| invalid_message!("decode params does not name a run call"))?;
            // A video request's latent encoding also covers a range: the
            // visual condition units of one round, or its single audio call.
            ensure_valid!(
                matches!(
                    call.code,
                    CallKind::Media(
                        MediaCall::VideoDecoding
                            | MediaCall::AudioDecoding
                            | MediaCall::VideoEncoding
                            | MediaCall::AudioEncoding
                            | MediaCall::LatentEncoding
                    )
                ),
                "decode params does not name media decode work"
            );
            // Audio reconstruction divides the sample timeline into as many
            // media units as its component has ranks, and one round covers
            // them all, so an audio decode range starts at the first unit.
            ensure_valid!(
                !matches!(call.code, CallKind::Media(MediaCall::AudioDecoding))
                    || (params.cursor == 0 && params.max_units >= 1),
                "audio decode range must start at the first media unit"
            );
            // Audio encoding consumes the assembled track as one host call.
            ensure_valid!(
                !matches!(call.code, CallKind::Media(MediaCall::AudioEncoding))
                    || (params.cursor == 0 && params.max_units == 1),
                "audio encode range must address its single sample stream"
            );
        }

        for call in &self.calls {
            let needs_range = matches!(
                call.code,
                CallKind::Media(
                    MediaCall::VideoDecoding
                        | MediaCall::AudioDecoding
                        | MediaCall::VideoEncoding
                        | MediaCall::AudioEncoding
                )
            );
            ensure_valid!(
                !needs_range || decode_ids.contains(&(call.request_key, call.call_id)),
                "media reconstruction call has no decode range"
            );
        }

        // Persistent buffers use non-overlapping half-open spans and cover every
        // declared output. `BufferAllocation::validate` has already rejected
        // an overflowing span end, so the addition below cannot wrap.
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

        for call in &self.calls {
            for output in call.buffer_outputs() {
                let params = self
                    .buffer_allocations
                    .iter()
                    .find(|params| params.buffer == output.buffer_id())
                    .ok_or_else(|| {
                        invalid_message!("persistent call output has no buffer params")
                    })?;
                ensure_valid!(
                    params.bytes >= output.max_bytes(),
                    "buffer params is smaller than its declared output"
                );
            }
        }

        // Submission depth is one runnable call per request.
        let mut request_keys = HashSet::with_capacity(self.call_count());
        for call in &self.calls {
            ensure_valid!(
                request_keys.insert(call.request_key),
                "a submission batch carries multiple calls for one request"
            );
        }

        // Admissions create request state independently of call execution.
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

        // Each input product payload must be valid, declared as an input or
        // predicate by some call, and supplied at most once.
        let mut supplied_inputs = HashSet::with_capacity(self.input_products.len());
        for payload in &self.input_products {
            ensure_valid!(
                self.calls().any(|call| match &payload.value {
                    TransferHandle::Encoder {
                        payload_kind: FeatureKind::Vision,
                        ..
                    } => {
                        call.vision_inputs
                            .iter()
                            .any(|block| block.feature == payload.product)
                    }
                    TransferHandle::Encoder {
                        payload_kind: FeatureKind::Latent,
                        ..
                    } => {
                        call.latent_feature_input.as_ref() == Some(&payload.product)
                    }
                    TransferHandle::DeviceProduct { .. } => {
                        call.inputs.contains(&payload.product)
                            || [&call.token_input, &call.image_input, &call.predicate]
                                .into_iter()
                                .any(|input| input.as_ref() == Some(&payload.product))
                    }
                    TransferHandle::Latent { .. } => {
                        call.latent_input.as_ref() == Some(&payload.product)
                    }
                }),
                "an input transfer has no consumer for its payload kind"
            );
            ensure_valid!(
                supplied_inputs.insert(&payload.product),
                "a submission batch repeats an input product payload"
            );
            payload.validate()?;
        }

        // Each imported KV export feeds exactly one installation call and
        // fits that call's transfer-byte bound.
        let mut kv_sources = HashSet::new();
        for export in &self.kv_inputs {
            export.validate()?;
            ensure_valid!(kv_sources.insert(export.source), "run repeats a KV input");
            let consumers = self
                .calls()
                .filter(|call| call.kv_input == Some(export.source))
                .collect::<Vec<_>>();
            ensure_valid!(
                consumers.len() == 1
                    && consumers[0].code == CallKind::Transfer(TransferMode::KvInstall),
                "KV transfer requires one installation consumer"
            );
            let bytes = export.tensors().try_fold(0_u64, |sum, tensor| {
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

/// One batch's complete result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchOutput {
    /// Batch identity copied from the submission.
    pub batch_id: u64,
    /// Call completions, one per call the batch carried.
    pub completions: Vec<RequestOutput>,
    /// Product values published by completed calls.
    pub products: Vec<TensorExport>,
    /// Aggregate worker execution time in microseconds, when measured.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported by the worker.
    pub forward_stats: Option<ForwardStats>,
}

impl BatchOutput {
    /// Iterates over call completions in report order.
    pub fn completions(&self) -> impl Iterator<Item = &RequestOutput> {
        self.completions.iter()
    }

    /// Iterates over resolved product payloads in report order.
    pub fn products(&self) -> impl Iterator<Item = &TensorExport> {
        self.products.iter()
    }

    /// Validates completion and product identities for this run.
    pub fn validate(&self) -> ValidationResult<()> {
        let mut identities = HashSet::with_capacity(self.completions.len());
        for completion in &self.completions {
            completion.validate()?;
            ensure_valid!(
                completion.call_id.batch_id == self.batch_id,
                "completion identity belongs to another logical batch"
            );
            ensure_valid!(
                identities.insert(completion.call_id),
                "completion report repeats a call"
            );
        }
        for payload in &self.products {
            payload.validate()?;
        }
        Ok(())
    }
}
