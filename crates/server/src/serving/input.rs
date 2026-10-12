//! Typed serving inputs before and after model-owned tokenization.
//!
//! [`PromptInput`] and the control structs ([`SamplingConfig`], [`StopConfig`],
//! [`ImageGenControls`], [`DecodeControls`]) are the protocol-neutral request
//! vocabulary. `serving::preprocessing` lowers the OpenAI wire requests into
//! them. `InputProcessor::preprocess_generation` consumes them and splits each
//! request into a `GenerationRequest`, which moves into the engine, and a
//! [`ResponseOptions`], which the frontend keeps to assemble the response
//! stream.

use std::collections::HashMap;

use crate::serving::text::TextDecodeOptions;
use crate::serving::text::tokenizer::DynTokenizer;

/// The public prompt: the text of an image-generation request or a chat
/// conversation. Input images live inside the parts of a chat conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptInput {
    /// Plain text prompt of an image-generation request.
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

/// Model-supplied committed-event processor selection, built by the per-model
/// preprocessing that `InputProcessor::preprocess_generation` dispatches to.
///
/// `submit_and_stream` selects the response assembler (see
/// `serving::assembly`) from this value.
pub enum OutputProcessorPolicy {
    /// Raw visible text.
    None,
    /// Chat reasoning and tool-call parsing over decoded text, selected for
    /// the chat prompts of Qwen3 and DiffusionGemma. The processor parses
    /// reasoning only when the server's
    /// `reasoning_parsing` setting is on, and otherwise passes that text
    /// through as visible text; it parses tool calls only when
    /// `ChatRequest::tool_parsing_enabled` holds.
    Chat(crate::serving::chat::ChatOutputProcessor),
    /// SenseNova reasoning and visible-answer filtering over committed text.
    SenseNova(crate::profile::omni::OutputFilterPolicy),
}

/// Tokenization resources and output requirements retained by the frontend.
/// The corresponding generation request moves directly into the engine.
pub struct ResponseOptions {
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
}
