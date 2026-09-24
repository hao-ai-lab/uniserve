//! Typed serving inputs before and after model-owned tokenization.
//!
//! [`PromptInput`] and the control structs ([`SamplingConfig`], [`StopConfig`],
//! [`ImageGenControls`], [`DecodeControls`]) are the protocol-neutral request
//! vocabulary. `serving::preprocessing` lowers the OpenAI wire requests into
//! them, and [`TextPromptRequest`] bundles the control structs with a
//! programmatic text prompt and optional input images.
//! `InputProcessor::preprocess_generation` consumes them and splits each
//! request into a `GenerationRequest`, which moves into the engine, and a
//! [`ResponseOptions`], which the frontend keeps to assemble the response
//! stream.

use std::collections::HashMap;

use crate::serving::text::TextDecodeOptions;
use crate::serving::text::tokenizer::DynTokenizer;
use crate::serving::{CacheAccounting, ResourceAccounting, ServeRequestId};

/// One top-level input image of a [`TextPromptRequest`].
///
/// The omni preprocessors decode it and compute its prompt position; models
/// without image input reject it during feature validation. Chat images arrive
/// separately as `image_url` parts inside the chat messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    /// Standard base64 encoding of the image file bytes, without a `data:` URL
    /// prefix.
    pub b64: String,
}

/// The public prompt: raw text or a chat conversation. Input images live in
/// programmatic image inputs (and inside chat parts for the chat variant).
#[derive(Debug, Clone, PartialEq)]
pub enum PromptInput {
    /// Plain text prompt.
    Text(String),
    /// Conversation, rendering options, and tools consumed by the chat
    /// renderer (`HfChatRenderer`).
    Chat(crate::serving::chat::ChatRequest),
}

/// Closed output-modality selection. The input side is inferred from the prompt
/// and images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalitySelection {
    /// Text output only.
    Text,
    /// Image output only.
    Image,
    /// Interleaved text and image output.
    TextAndImage,
}

impl ModalitySelection {
    /// Returns whether text output is requested.
    pub const fn includes_text(self) -> bool {
        matches!(self, Self::Text | Self::TextAndImage)
    }

    /// Returns whether image output is requested.
    pub const fn includes_image(self) -> bool {
        matches!(self, Self::Image | Self::TextAndImage)
    }
}

impl Default for ModalitySelection {
    /// Returns [`ModalitySelection::Text`].
    fn default() -> Self {
        Self::Text
    }
}

/// The single canonical sampling configuration. All values are optional so that
/// model and generation-config defaults apply when omitted.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SamplingConfig {
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Maximum number of highest-probability candidates considered per step.
    pub top_k: Option<u32>,
    /// Minimum token probability relative to the most likely candidate.
    pub min_p: Option<f32>,
    /// Optional deterministic sampler seed. For SenseNova and Bagel requests,
    /// `ImageGenControls::seed` takes precedence when both are set.
    pub seed: Option<i64>,
    /// Maximum number of generated tokens.
    pub max_tokens: Option<u32>,
    /// Minimum number of generated tokens before stopping is allowed.
    pub min_tokens: Option<u32>,
    /// Penalty proportional to a token's prior frequency.
    pub frequency_penalty: Option<f32>,
    /// Penalty applied once to tokens that already appear.
    pub presence_penalty: Option<f32>,
    /// Multiplicative repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Whether end-of-sequence tokens are ignored as stopping conditions.
    pub ignore_eos: bool,
}

/// Stop-token, stop-string, bad-word, allowed-token, logit-bias, and logprob
/// controls supported by the configured sampler.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StopConfig {
    /// Additional token identifiers that terminate generation.
    pub stop_token_ids: Vec<u32>,
    /// Decoded text sequences that terminate generation.
    pub stop_strings: Vec<String>,
    /// Text sequences excluded from generated output.
    pub bad_words: Vec<String>,
    /// Optional allowlist of token identifiers available to the sampler.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Additive logit adjustments keyed by token identifier.
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Number of generated-token alternatives included in log probabilities.
    pub logprobs: Option<i32>,
    /// Number of prompt-token alternatives included in log probabilities.
    pub prompt_logprobs: Option<i32>,
    /// Additional token identifiers whose probabilities are reported.
    pub logprob_token_ids: Option<Vec<u32>>,
}

/// Image-generation controls: dimensions, denoise steps, guidance controls,
/// seed, and image-count bounds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageGenControls {
    /// Named output-resolution preset.
    pub resolution: Option<crate::profile::omni::resolution::ResolutionName>,
    /// Explicit output width in pixels.
    pub width: Option<u32>,
    /// Explicit output height in pixels.
    pub height: Option<u32>,
    /// Diffusion step count.
    pub steps: Option<u16>,
    /// Classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: Option<f32>,
    /// Classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: Option<f32>,
    /// Fractional diffusion interval over which guidance applies.
    pub cfg_interval: Option<[f32; 2]>,
    /// Guidance renormalization algorithm.
    pub cfg_renorm_type: Option<uniserve_core::CfgRenorm>,
    /// Minimum scale at which guidance renormalization applies.
    pub cfg_renorm_min: Option<f32>,
    /// Diffusion timestep shift.
    pub timestep_shift: Option<f32>,
    /// Optional deterministic seed. For SenseNova and Bagel requests it
    /// replaces `SamplingConfig::seed` as the sampler seed; Qwen3
    /// preprocessing resolves its seed from `SamplingConfig::seed` alone.
    pub seed: Option<u64>,
    /// Maximum number of generated images.
    pub max_images: Option<u16>,
    /// Additional text prompts used by image-generation branches.
    pub prompts: Vec<String>,
    /// Whether generated image products remain available after response assembly.
    pub retain_images: Option<bool>,
}

/// Per-token detail included in the public output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputDetail {
    /// Emits decoded user-visible text without token metadata.
    #[default]
    VisibleText,
    /// Emits generated token identifiers in addition to visible text.
    Tokens,
    /// Emits generated token identifiers and candidate log probabilities.
    Logprobs,
}

/// Decode controls applied to the response path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeControls {
    /// Whether tokenizer special tokens are omitted from decoded text.
    pub skip_special_tokens: bool,
    /// Whether the matched stop string remains in visible output.
    pub include_stop_string_in_output: bool,
}

impl Default for DecodeControls {
    /// Skips special tokens and omits the matched stop string.
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_string_in_output: false,
        }
    }
}

/// A programmatic text prompt with optional context images. HTTP endpoints use
/// their own request schemas and do not construct this value.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPromptRequest {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Plain text passed through the model's text-prompt preprocessing.
    pub prompt: String,
    /// Top-level input images associated with the prompt.
    pub images: Vec<ImageInput>,
    /// Requested output modalities.
    pub modalities: ModalitySelection,
    /// Token sampling controls.
    pub sampling: SamplingConfig,
    /// Generation constraints and probability-reporting controls.
    pub stop: StopConfig,
    /// Optional negative prompt used for classifier-free guidance.
    pub negative_text: Option<String>,
    /// Image-generation controls when image output is requested.
    pub image_gen: Option<ImageGenControls>,
    /// Optional cache namespace isolating otherwise identical requests.
    /// Combined with `cache_salt` into the engine cache isolation key by
    /// `cache_isolation_key`; with both absent, the request uses the shared
    /// cache partition.
    pub cache_namespace: Option<String>,
    /// Optional caller-provided value mixed into the cache isolation key.
    pub cache_salt: Option<String>,
    /// Whether existing prefix and encoder cache entries are ignored.
    pub bypass_cache_read: bool,
    /// Whether this request is excluded from cache storage.
    pub no_cache_store: bool,
    /// Scheduler priority for this request.
    pub priority: i32,
    /// Requested response detail.
    pub output: OutputDetail,
    /// Incremental text decoding controls.
    pub decode: DecodeControls,
}

impl TextPromptRequest {
    /// Creates a programmatic text-prompt request with model-default controls.
    pub fn new(request_id: impl Into<ServeRequestId>, prompt: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            prompt: prompt.into(),
            images: Vec::new(),
            modalities: ModalitySelection::Text,
            sampling: SamplingConfig::default(),
            stop: StopConfig::default(),
            negative_text: None,
            image_gen: None,
            cache_namespace: None,
            cache_salt: None,
            bypass_cache_read: false,
            no_cache_store: false,
            priority: 0,
            output: OutputDetail::VisibleText,
            decode: DecodeControls::default(),
        }
    }
}

/// Model-supplied committed-event processor selection, built by the per-model
/// preprocessing that `InputProcessor::preprocess_generation` dispatches to.
///
/// `submit_and_stream` selects the response assembler (see
/// `serving::assembly`) from this value.
pub enum OutputProcessorPolicy {
    /// Raw visible text.
    None,
    /// Qwen3 chat reasoning and tool-call parsing over decoded text, selected
    /// for Qwen3 chat prompts (Qwen3 text prompts use `None`). The processor
    /// parses `<think>` reasoning only when the server's `reasoning_parsing`
    /// setting is on, and otherwise passes that text through as visible
    /// text; it parses tool calls only when
    /// `ChatRequest::tool_parsing_enabled` holds.
    Qwen3(crate::serving::chat::Qwen3ChatOutputProcessor),
    /// SenseNova reasoning and visible-answer filtering over committed text.
    SenseNova(crate::profile::omni::OutputFilterPolicy),
}

/// Model identity stamped onto `Accepted` events.
#[derive(Debug, Clone)]
pub struct ModelEventIdentity {
    /// Model name exposed through serving events.
    pub served_name: String,
    /// Stable profile description exposed through serving events.
    pub description: String,
}

/// Tokenization resources and output requirements retained by the frontend.
/// The corresponding generation request moves directly into the engine.
pub struct ResponseOptions {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// The model-bound tokenizer that owns decoding for this request.
    pub tokenizer: DynTokenizer,
    /// Copy of the prompt token identifiers submitted to the engine, kept
    /// because detokenization needs the prompt after the generation request
    /// has moved into the engine.
    pub prompt_token_ids: Vec<u32>,
    /// Incremental detokenization behavior.
    pub decode: TextDecodeOptions,
    /// Whether generated token identifiers are included in public events.
    pub emit_token_ids: bool,
    /// Whether prompt log probabilities are requested.
    pub prompt_logprobs_requested: bool,
    /// Whether generated-token log probabilities are requested.
    pub generated_logprobs_requested: bool,
    /// Model-selected semantic output processor.
    pub output_processor: OutputProcessorPolicy,
    /// Model identity stamped onto accepted events.
    pub identity: ModelEventIdentity,
    /// Cache accounting limits propagated to the engine request.
    pub cache: CacheAccounting,
    /// Resource bounds propagated to admission control.
    pub resources: ResourceAccounting,
}
