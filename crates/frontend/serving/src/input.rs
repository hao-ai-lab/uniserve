//! Sole internal generate-class admission value ([`GenerateReqInput`]) and the
//! sole value submitted to the engine gateway ([`TokenizedGenerateReqInput`]).
//!
//! Model-private prompt recipes, token placement, generation policies, and
//! output filters live behind [`crate::model::ResolvedModel::tokenize`].

use std::collections::{BTreeMap, HashMap};

use uniserve_core::GenerationRequest;

use crate::chat::{ChatMessage, ChatRequest, ChatTool, ChatToolChoice, ReasoningEffort};
use crate::text::TextDecodeOptions;
use crate::{CacheAccounting, ResourceAccounting, ServeRequestId};

/// One supported public input image. Model-specific placement is resolved by
/// [`crate::model::ResolvedModel::tokenize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub b64: String,
}

/// The public prompt: raw text or a chat conversation. Input images live in
/// [`GenerateReqInput::images`] (and inside chat parts for the chat variant).
#[derive(Debug, Clone, PartialEq)]
pub enum PromptInput {
    Text(String),
    Chat {
        messages: Vec<ChatMessage>,
        tools: Vec<ChatTool>,
        tool_choice: ChatToolChoice,
        reasoning_effort: Option<ReasoningEffort>,
    },
}

/// Output-modality selection: text output, image output, or an admitted
/// combination. The input side is inferred from the prompt and images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModalitySelection {
    pub output_text: bool,
    pub output_image: bool,
}

impl Default for ModalitySelection {
    fn default() -> Self {
        Self {
            output_text: true,
            output_image: false,
        }
    }
}

/// The single canonical sampling configuration. All values are optional so that
/// model and generation-config defaults apply when omitted.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SamplingConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub seed: Option<i64>,
    pub max_tokens: Option<u32>,
    pub min_tokens: Option<u32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub ignore_eos: bool,
}

/// Stop-token, stop-string, bad-word, allowed-token, logit-bias, and logprob
/// controls supported by the configured sampler capability.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StopConfig {
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub bad_words: Vec<String>,
    pub allowed_token_ids: Option<Vec<u32>>,
    pub logit_bias: Option<HashMap<u32, f32>>,
    pub logprobs: Option<i32>,
    pub prompt_logprobs: Option<i32>,
    pub logprob_token_ids: Option<Vec<u32>>,
}

/// Image-generation controls: dimensions, denoise steps, guidance controls,
/// seed, and image-count bounds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageGenControls {
    pub resolution: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u16>,
    pub cfg_text_scale: Option<f32>,
    pub cfg_img_scale: Option<f32>,
    pub cfg_interval: Option<[f32; 2]>,
    pub cfg_renorm_type: Option<String>,
    pub cfg_renorm_min: Option<f32>,
    pub timestep_shift: Option<f32>,
    pub seed: Option<u64>,
    pub max_images: Option<u16>,
    pub prompts: Vec<String>,
    pub retain_images: Option<bool>,
}

/// Cache bounds required for admission.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheBounds {
    pub namespace: Option<String>,
    pub salt: Option<String>,
    pub bypass_read: bool,
    pub no_store: bool,
}

/// Scheduling bounds required for admission (no deadline).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SchedulingBounds {
    pub priority: i32,
    pub trace_context: BTreeMap<String, String>,
}

/// Public output contract: which per-token detail the caller receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputContract {
    #[default]
    VisibleText,
    Tokens,
    Logprobs,
}

/// Decode controls applied to the response path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeControls {
    pub skip_special_tokens: bool,
    pub include_stop_string_in_output: bool,
}

impl Default for DecodeControls {
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
    pub request_id: ServeRequestId,
    pub stream: bool,
    pub prompt: PromptInput,
    pub images: Vec<ImageInput>,
    pub modalities: ModalitySelection,
    pub sampling: SamplingConfig,
    pub stop: StopConfig,
    pub negative_text: Option<String>,
    pub image_gen: Option<ImageGenControls>,
    pub cache: CacheBounds,
    pub scheduling: SchedulingBounds,
    pub output: OutputContract,
    pub decode: DecodeControls,
}

impl GenerateReqInput {
    /// Build a minimal text-prompt request with default controls.
    pub fn text(request_id: impl Into<ServeRequestId>, prompt: impl Into<String>) -> Self {
        Self::from_prompt(request_id.into(), PromptInput::Text(prompt.into()))
    }

    /// Build a minimal chat request with default controls.
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
            output: OutputContract::default(),
            decode: DecodeControls::default(),
        }
    }

    /// True when the request declares any input image (top-level or chat part).
    pub fn has_input_image(&self) -> bool {
        !self.images.is_empty()
            || matches!(&self.prompt, PromptInput::Chat { messages, .. } if messages.iter().any(ChatMessage::has_multimodal))
    }
}

/// Model-supplied committed-event processor selection, built inside
/// [`crate::model::ResolvedModel::tokenize`].
#[derive(Debug, Clone)]
pub enum OutputProcessorPolicy {
    /// Raw visible text.
    None,
    /// Qwen3 chat reasoning + tool parsing over decoded text.
    Qwen3(Box<ChatRequest>),
    /// SenseNova reasoning and visible-answer filtering over committed text.
    SenseNova(uniserve_model_profile::omni::OutputFilterPolicy),
    /// Bagel committed-event output policy.
    Bagel,
}

/// Submission-envelope inputs carried alongside the engine request.
#[derive(Debug, Clone)]
pub struct SubmissionMetadata {
    pub trace_headers: Option<BTreeMap<String, String>>,
}

/// Model identity stamped onto `Accepted` events.
#[derive(Debug, Clone)]
pub struct ModelEventIdentity {
    pub profile_id: String,
    pub description_id: String,
}

/// The sole value submitted to the engine gateway, produced by
/// [`crate::model::ResolvedModel::tokenize`].
#[derive(Debug, Clone)]
pub struct TokenizedGenerateReqInput {
    pub request_id: ServeRequestId,
    /// Canonical engine request from the public funnel.
    pub request: GenerationRequest,
    pub prompt_token_ids: Vec<u32>,
    pub decode: TextDecodeOptions,
    pub emit_token_ids: bool,
    pub prompt_logprobs_requested: bool,
    pub generated_logprobs_requested: bool,
    pub skip_special_tokens: bool,
    pub output_processor: OutputProcessorPolicy,
    pub submission: SubmissionMetadata,
    pub identity: ModelEventIdentity,
    pub cache: CacheAccounting,
    pub resources: ResourceAccounting,
}
