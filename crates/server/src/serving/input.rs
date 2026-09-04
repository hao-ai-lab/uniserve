//! Typed serving inputs before and after model-owned tokenization.
//!
//! [`GenerateReqInput`] carries transport-independent user intent.
//! [`TokenizedGenerateReqInput`] carries the fully resolved engine request and
//! output policy.

use std::collections::{BTreeMap, HashMap};

use uniserve_core::GenerationRequest;

use crate::serving::chat::{ChatMessage, ChatToolChoice, ReasoningEffort, Tool};
use crate::serving::text::TextDecodeOptions;
use crate::serving::text::tokenizer::DynTokenizer;
use crate::serving::{CacheAccounting, ResourceAccounting, ServeRequestId};

/// One supported public input image. Model-specific placement is resolved by
/// [`crate::serving::model::ResolvedModel::tokenize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    /// Base64-encoded image payload.
    pub b64: String,
}

/// The public prompt: raw text or a chat conversation. Input images live in
/// [`GenerateReqInput::images`] (and inside chat parts for the chat variant).
#[derive(Debug, Clone, PartialEq)]
pub enum PromptInput {
    /// Plain text prompt.
    Text(String),
    /// Structured conversation prompt with optional function tools.
    Chat {
        /// Ordered conversation history.
        messages: Vec<ChatMessage>,
        /// Functions available for the next assistant turn.
        tools: Vec<Tool>,
        /// Tool-selection policy for the next assistant turn.
        tool_choice: ChatToolChoice,
        /// Optional model reasoning budget.
        reasoning_effort: Option<ReasoningEffort>,
    },
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
    /// Returns the default value.
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
    /// Optional deterministic sampler seed.
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
    /// Optional deterministic image-sampling seed.
    pub seed: Option<u64>,
    /// Maximum number of generated images.
    pub max_images: Option<u16>,
    /// Additional text prompts used by image-generation branches.
    pub prompts: Vec<String>,
    /// Whether generated image products remain available after response assembly.
    pub retain_images: Option<bool>,
}

/// Cache bounds required for admission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheBounds {
    /// Optional cache namespace isolating otherwise identical requests.
    pub namespace: Option<String>,
    /// Optional caller-provided value mixed into the cache key.
    pub salt: Option<String>,
    /// Whether existing cache entries are ignored.
    pub bypass_read: bool,
    /// Whether products from this request are excluded from cache storage.
    pub no_store: bool,
}

/// Scheduling bounds required for admission (no deadline).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SchedulingBounds {
    /// Request scheduling priority; larger values receive preference.
    pub priority: i32,
    /// Distributed trace context propagated into engine execution.
    pub trace_context: BTreeMap<String, String>,
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
    /// Returns the default value.
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_string_in_output: false,
        }
    }
}

/// The sole internal generate-class admission value.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateReqInput {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Whether transport output should be streamed incrementally.
    pub stream: bool,
    /// Text or structured chat prompt.
    pub prompt: PromptInput,
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
    /// Prefix and product cache controls.
    pub cache: CacheBounds,
    /// Admission and scheduling controls.
    pub scheduling: SchedulingBounds,
    /// Requested response detail.
    pub output: OutputDetail,
    /// Incremental text decoding controls.
    pub decode: DecodeControls,
}

impl GenerateReqInput {
    /// Builds a minimal text-prompt request with default controls.
    pub fn text(request_id: impl Into<ServeRequestId>, prompt: impl Into<String>) -> Self {
        Self::from_prompt(request_id.into(), PromptInput::Text(prompt.into()))
    }

    /// Builds a minimal chat request with default controls.
    pub fn chat(request_id: impl Into<ServeRequestId>, messages: Vec<ChatMessage>) -> Self {
        Self::from_prompt(
            request_id.into(),
            PromptInput::Chat {
                messages,
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                reasoning_effort: None,
            },
        )
    }

    /// Builds serving input from a text prompt.
    fn from_prompt(request_id: ServeRequestId, prompt: PromptInput) -> Self {
        Self {
            request_id,
            stream: true,
            prompt,
            images: Vec::new(),
            modalities: ModalitySelection::default(),
            sampling: SamplingConfig::default(),
            stop: StopConfig::default(),
            negative_text: None,
            image_gen: None,
            cache: CacheBounds::default(),
            scheduling: SchedulingBounds::default(),
            output: OutputDetail::default(),
            decode: DecodeControls::default(),
        }
    }

    /// Returns whether the request declares any input image.
    pub fn has_input_image(&self) -> bool {
        !self.images.is_empty()
            || matches!(&self.prompt, PromptInput::Chat { messages, .. } if messages.iter().any(ChatMessage::has_multimodal))
    }

    /// Returns whether the request uses function-tool syntax in either the
    /// current turn or its chat history.
    pub fn uses_tools(&self) -> bool {
        let PromptInput::Chat {
            messages, tools, ..
        } = &self.prompt
        else {
            return false;
        };
        !tools.is_empty()
            || messages.iter().any(|message| match message {
                ChatMessage::Developer { tools, .. } => {
                    tools.as_ref().is_some_and(|tools| !tools.is_empty())
                }
                ChatMessage::Assistant { content } => content.has_tool_calls(),
                ChatMessage::ToolResponse { .. } => true,
                ChatMessage::System { .. } | ChatMessage::User { .. } => false,
            })
    }

    /// Returns whether the request asks the model's chat template to select a
    /// reasoning effort.
    pub fn requests_reasoning(&self) -> bool {
        matches!(
            &self.prompt,
            PromptInput::Chat {
                reasoning_effort: Some(_),
                ..
            }
        )
    }
}

/// Model-supplied committed-event processor selection, built inside
/// [`crate::serving::model::ResolvedModel::tokenize`].
pub enum OutputProcessorPolicy {
    /// Raw visible text.
    None,
    /// Qwen3 chat reasoning + tool parsing over decoded text. The flag
    /// selects whether `<think>` delimiters are parsed into reasoning blocks
    /// or streamed verbatim as content.
    Qwen3(crate::serving::chat::Qwen3ChatOutputProcessor),
    /// SenseNova reasoning and visible-answer filtering over committed text.
    SenseNova(crate::profile::omni::OutputFilterPolicy),
    /// Bagel committed-event output policy.
    Bagel,
}

/// Model identity stamped onto `Accepted` events.
#[derive(Debug, Clone)]
pub struct ModelEventIdentity {
    /// Model name exposed through serving events.
    pub served_name: String,
    /// Stable profile description exposed through serving events.
    pub description: String,
}

/// The sole value submitted to the engine client, produced by
/// [`crate::serving::model::ResolvedModel::tokenize`].
pub struct TokenizedGenerateReqInput {
    /// Caller-visible request identifier.
    pub request_id: ServeRequestId,
    /// Canonical engine request from the public funnel.
    pub request: GenerationRequest,
    /// The model-bound tokenizer that owns decoding for this request.
    pub tokenizer: DynTokenizer,
    /// Prompt token identifiers submitted to the engine.
    pub prompt_token_ids: Vec<u32>,
    /// Incremental detokenization behavior.
    pub decode: TextDecodeOptions,
    /// Whether generated token identifiers are included in public events.
    pub emit_token_ids: bool,
    /// Whether prompt log probabilities are requested.
    pub prompt_logprobs_requested: bool,
    /// Whether generated-token log probabilities are requested.
    pub generated_logprobs_requested: bool,
    /// Whether special tokens are omitted from decoded response text.
    pub skip_special_tokens: bool,
    /// Model-selected semantic output processor.
    pub output_processor: OutputProcessorPolicy,
    /// Model identity stamped onto accepted events.
    pub identity: ModelEventIdentity,
    /// Cache accounting limits propagated to the engine request.
    pub cache: CacheAccounting,
    /// Resource bounds propagated to admission control.
    pub resources: ResourceAccounting,
}
