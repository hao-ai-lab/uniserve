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
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,
    pub model: String,
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub logprobs: bool,
    #[validate(range(min = -1))]
    pub top_logprobs: Option<i32>,
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    #[serde(default = "default_modalities")]
    pub modalities: Vec<ChatModality>,
    pub image_config: Option<ChatImageConfig>,
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    #[serde(default)]
    pub ignore_eos: bool,
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    pub prompt_logprobs: Option<i32>,
    pub allowed_token_ids: Option<Vec<u32>>,
    pub bad_words: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub include_reasoning: bool,
    pub priority: Option<i32>,
    pub return_tokens_as_token_ids: Option<bool>,
    pub return_token_ids: Option<bool>,
    pub cache_salt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatModality {
    Text,
    Image,
}

fn default_modalities() -> Vec<ChatModality> {
    vec![ChatModality::Text]
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatImageConfig {
    pub resolution: Option<crate::profile::omni::resolution::ResolutionName>,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub steps: Option<u16>,
    pub guidance_scale: Option<f32>,
    pub image_guidance_scale: Option<f32>,
    pub seed: Option<u64>,
    pub num_images: Option<u16>,
    pub cfg_norm: Option<uniserve_core::CfgRenorm>,
    pub cfg_renorm_min: Option<f32>,
    pub cfg_interval: Option<[f32; 2]>,
    pub timestep_shift: Option<f32>,
}

impl Normalizable for ChatCompletionRequest {
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
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay)]
pub struct AssistantRole;

impl fmt::Display for AssistantRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("assistant")
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionMessage {
    pub role: AssistantRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    pub images: Option<Vec<ContentPart>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    pub usage: Option<Usage>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub public_commit: Option<StreamPublicCommit>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamPublicCommit {
    pub event_seq: u64,
    pub modality: String,
    pub committed_at: f64,
    pub semantic_root: StreamSemanticRoot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StreamSemanticRoot {
    pub producer_op_id: u64,
    pub point_index: u32,
}

impl ChatCompletionStreamResponse {
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
            prompt_token_ids: None,
            public_commit: None,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub token_ids: Option<Vec<u32>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatMessageDelta {
    pub role: Option<AssistantRole>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(rename = "reasoning_content")]
    pub reasoning: Option<String>,
    pub images: Option<Vec<ContentPart>>,
}

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
