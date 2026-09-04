//! OpenAI-compatible chat-completion request and response schemas.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::SerializeDisplay;
use validator::Validate;

use super::common::{
    ChatLogProbs, ChatMessage, ContentPart, MessageContent, Normalizable, ReasoningEffort,
    StreamOptions, StringOrArray, Tool, ToolCall, ToolCallDelta, ToolChoice, ToolChoiceValue,
    Usage, default_true, validate_stop, validate_top_p_value,
};

/// The configured OpenAI-compatible chat request.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[serde(deny_unknown_fields)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
    /// Ordered conversation messages used to build the model prompt.
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,
    /// Served model name.
    pub model: String,
    /// Frequency-based token penalty in the inclusive range `[-2, 2]`.
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    /// Additive sampling biases keyed by decimal token identifier.
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Whether generated-token log probabilities are returned.
    #[serde(default)]
    pub logprobs: bool,
    /// Number of alternate generated-token logprobs, or `-1` for all.
    #[validate(range(min = -1))]
    pub top_logprobs: Option<i32>,
    /// Maximum number of completion tokens.
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,
    /// Presence-based token penalty in the inclusive range `[-2, 2]`.
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    /// Deterministic sampling seed.
    pub seed: Option<i64>,
    /// Text sequences that terminate generation.
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,
    /// Whether the response is delivered as server-sent event chunks.
    #[serde(default)]
    pub stream: bool,
    /// Options that alter streamed response framing.
    pub stream_options: Option<StreamOptions>,
    /// Output modalities requested from the model.
    #[serde(default = "default_modalities")]
    pub modalities: Vec<ChatModality>,
    /// Image-generation controls when image output is requested.
    pub image_config: Option<ChatImageConfig>,
    /// Sampling temperature in the inclusive range `[0, 2]`.
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass in the interval `(0, 1]`.
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    /// Maximum candidate tokens retained before sampling.
    pub top_k: Option<u32>,
    /// Minimum token probability relative to the most likely token.
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    /// Multiplicative penalty applied to previously generated tokens.
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    /// Function tools available to the assistant.
    pub tools: Option<Vec<Tool>>,
    /// Policy controlling whether the assistant may call a tool.
    pub tool_choice: Option<ToolChoice>,
    /// Requested reasoning budget.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Token identifiers that terminate generation.
    pub stop_token_ids: Option<Vec<u32>>,
    /// Whether matched stop strings remain in returned text.
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    /// Whether end-of-sequence tokens are excluded from termination checks.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Minimum tokens generated before stop conditions may terminate output.
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,
    /// Whether special tokens are omitted during text decoding.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    /// Number of alternate prompt-token logprobs, or `-1` for all.
    pub prompt_logprobs: Option<i32>,
    /// Optional whitelist of token identifiers eligible for sampling.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Text sequences excluded from generated output.
    pub bad_words: Option<Vec<String>>,
    /// Whether parsed reasoning content is included in responses.
    #[serde(default = "default_true")]
    pub include_reasoning: bool,
    /// Scheduler priority assigned to the request.
    pub priority: Option<i32>,
    /// Whether token strings are rendered as token-identifier placeholders.
    pub return_tokens_as_token_ids: Option<bool>,
    /// Whether response objects include generated token identifiers.
    pub return_token_ids: Option<bool>,
    /// Caller-provided salt used to isolate prompt-cache entries.
    pub cache_salt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
/// Output modality requested from a chat completion.
pub enum ChatModality {
    /// Text output.
    Text,
    /// Image output.
    Image,
}

/// Returns the default response modalities.
fn default_modalities() -> Vec<ChatModality> {
    vec![ChatModality::Text]
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Image-generation controls embedded in a chat request.
pub struct ChatImageConfig {
    /// Named output resolution preset.
    pub resolution: Option<crate::profile::omni::resolution::ResolutionName>,
    /// Explicit output height in pixels.
    pub height: Option<u32>,
    /// Explicit output width in pixels.
    pub width: Option<u32>,
    /// Number of denoising steps per image.
    pub steps: Option<u16>,
    /// Text classifier-free-guidance scale.
    pub guidance_scale: Option<f32>,
    /// Image classifier-free-guidance scale.
    pub image_guidance_scale: Option<f32>,
    /// Image-generation random seed.
    pub seed: Option<u64>,
    /// Number of images to generate.
    pub num_images: Option<u16>,
    /// Classifier-free-guidance renormalization policy.
    pub cfg_norm: Option<uniserve_core::CfgRenorm>,
    /// Minimum scale applied by guidance renormalization.
    pub cfg_renorm_min: Option<f32>,
    /// Fractional denoising interval in which guidance is active.
    pub cfg_interval: Option<[f32; 2]>,
    /// Diffusion scheduler timestep shift.
    pub timestep_shift: Option<f32>,
}

impl Normalizable for ChatCompletionRequest {
    /// Normalizes the value into its canonical representation.
    fn normalize(&mut self) {
        if self.tool_choice.is_none()
            && let Some(tools) = &self.tools
        {
            self.tool_choice = Some(ToolChoice(if tools.is_empty() {
                ToolChoiceValue::None
            } else {
                ToolChoiceValue::Auto
            }));
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// Non-streaming chat-completion response body.
pub struct ChatCompletionResponse {
    /// OpenAI-compatible completion identifier.
    pub id: String,
    /// Object type, always `chat.completion`.
    pub object: String,
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Served model name.
    pub model: String,
    /// Completed assistant choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Request usage accounting, when available.
    pub usage: Option<Usage>,
    /// Backend fingerprint, when reported.
    pub system_fingerprint: Option<String>,
    /// Prompt logprobs aligned to prompt positions, when requested.
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    /// Prompt token identifiers, when requested.
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// One completed assistant choice.
pub struct ChatCompletionChoice {
    /// Choice position in the response.
    pub index: u32,
    /// Completed assistant message.
    pub message: ChatCompletionMessage,
    /// Generated-token logprob details, when requested.
    pub logprobs: Option<ChatLogProbs>,
    /// Stable terminal reason, when generation is complete.
    pub finish_reason: Option<String>,
    /// Token identifier or string that triggered stopping, when applicable.
    pub stop_reason: Option<Value>,
    /// Generated token identifiers, when requested.
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay)]
/// Serializer for the constant assistant role label.
pub struct AssistantRole;

impl fmt::Display for AssistantRole {
    /// Formats the value for diagnostic output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("assistant")
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// Assistant message returned by a non-streaming completion.
pub struct ChatCompletionMessage {
    /// Constant assistant role marker.
    pub role: AssistantRole,
    /// Visible assistant text, when produced.
    pub content: Option<String>,
    /// Completed tool calls requested by the assistant.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Parsed reasoning content, when requested.
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    /// Generated image content, when produced.
    pub images: Option<Vec<ContentPart>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
/// One server-sent chat-completion chunk.
pub struct ChatCompletionStreamResponse {
    /// OpenAI-compatible completion identifier.
    pub id: String,
    /// Object type, always `chat.completion.chunk`.
    pub object: String,
    /// Response creation timestamp in Unix seconds.
    pub created: u64,
    /// Served model name.
    pub model: String,
    /// Incremental assistant choices carried by this chunk.
    pub choices: Vec<ChatCompletionStreamChoice>,
    /// Terminal request usage when requested by stream options.
    pub usage: Option<Usage>,
    /// Prompt token identifiers, when supplied with initial metadata.
    pub prompt_token_ids: Option<Vec<u32>>,
}

impl ChatCompletionStreamResponse {
    /// Constructs the initial stream response for one completion.
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
            prompt_token_ids: None,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
/// One incremental choice carried by a stream chunk.
pub struct ChatCompletionStreamChoice {
    /// Choice position in the response.
    pub index: u32,
    /// Incremental assistant message fields.
    pub delta: ChatMessageDelta,
    /// Generated-token logprob details for this chunk.
    pub logprobs: Option<ChatLogProbs>,
    /// Stable terminal reason on the final choice chunk.
    pub finish_reason: Option<String>,
    /// Token identifier or string that triggered stopping, when applicable.
    pub stop_reason: Option<Value>,
    /// Generated token identifiers carried by this chunk.
    pub token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
/// Incremental assistant-message fields in a stream chunk.
pub struct ChatMessageDelta {
    /// Assistant role marker, normally present on the first chunk.
    pub role: Option<AssistantRole>,
    /// Incremental visible text.
    pub content: Option<String>,
    /// Incremental tool-call fields.
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Incremental parsed reasoning text.
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    /// Generated image content carried by this chunk.
    pub images: Option<Vec<ContentPart>>,
}

/// Validates conversation-level presence and non-empty user content.
fn validate_messages(messages: &[ChatMessage]) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }
    for message in messages {
        if let ChatMessage::User { content, .. } = message {
            match content {
                MessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                MessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Validates dependencies and bounds that span multiple chat request fields.
fn validate_chat_cross_parameters(
    request: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    if request.top_logprobs.is_some() && !request.logprobs {
        return Err(validator::ValidationError::new(
            "top_logprobs_requires_logprobs",
        ));
    }
    if request.stream_options.is_some() && !request.stream {
        return Err(validator::ValidationError::new(
            "stream_options_requires_stream",
        ));
    }
    if let (Some(minimum), Some(maximum)) = (request.min_tokens, request.max_completion_tokens)
        && minimum > maximum
    {
        return Err(validator::ValidationError::new(
            "min_tokens_exceeds_max_completion_tokens",
        ));
    }
    if request.modalities.is_empty() {
        return Err(validator::ValidationError::new("modalities_empty"));
    }
    let unique = request
        .modalities
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != request.modalities.len() {
        return Err(validator::ValidationError::new("modalities_duplicate"));
    }
    let image_output = request.modalities.contains(&ChatModality::Image);
    if !image_output && request.image_config.is_some() {
        return Err(validator::ValidationError::new(
            "image_config_without_image",
        ));
    }
    if let Some(config) = &request.image_config {
        if config.width.is_some() != config.height.is_some() {
            return Err(validator::ValidationError::new("image_config_dimensions"));
        }
        if config.num_images == Some(0) || config.steps == Some(0) {
            return Err(validator::ValidationError::new("image_config_bounds"));
        }
    }
    if let Some(ToolChoice(choice)) = &request.tool_choice
        && *choice != ToolChoiceValue::None
        && !request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    {
        return Err(validator::ValidationError::new(
            "tool_choice_requires_tools",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ChatCompletionRequest;

    #[test]
    fn configured_request_schema_rejects_unknown_controls() {
        let value = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "unsupported_option": true
        });
        assert!(serde_json::from_value::<ChatCompletionRequest>(value).is_err());
    }
}
