//! Schema values shared across OpenAI-compatible request families.

use std::slice;

use llm_multimodal::ImageDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Returns `true` for fields whose wire default is enabled.
pub(super) fn default_true() -> bool {
    true
}

/// A wire value accepted as either one string or an array of strings.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    /// One string value.
    String(String),
    /// Ordered array of string values.
    Array(Vec<String>),
}

impl StringOrArray {
    /// Borrows the value uniformly as a string slice.
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::String(s) => slice::from_ref(s),
            StringOrArray::Array(arr) => arr,
        }
    }

    #[allow(unused)]
    /// Converts the value into an owned string vector.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s],
            StringOrArray::Array(arr) => arr,
        }
    }
}

/// Rejects empty stop strings.
pub(super) fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    if stop.as_slice().iter().any(|s| s.is_empty()) {
        return Err(validator::ValidationError::new(
            "stop strings cannot be empty",
        ));
    }
    Ok(())
}

/// Validates that `top_p` lies in `(0, 1]`.
pub(super) fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

/// Effort level accepted by OpenAI-compatible reasoning requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disables explicit reasoning effort.
    None,
    /// Requests the smallest available reasoning budget.
    Minimal,
    /// Requests a low reasoning budget.
    Low,
    /// Requests a medium reasoning budget.
    Medium,
    /// Requests a high reasoning budget.
    High,
    /// Requests an extra-high reasoning budget.
    XHigh,
    /// Requests the largest available reasoning budget.
    Max,
}

/// One typed part of multimodal message content.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ContentPart {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// Text carried by the content part.
        text: String,
    },
    /// Image content referenced by URL or data URL.
    #[serde(rename = "image_url")]
    ImageUrl {
        /// Image source and requested detail level.
        image_url: ImageUrl,
        /// Optional caller-provided image identity.
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Remote or data URL and requested image-detail level.
pub struct ImageUrl {
    /// Remote URL or data URL containing the image.
    pub url: String,
    /// Requested image preprocessing detail.
    pub detail: Option<ImageDetail>,
}

/// Options that alter streamed chat-completion responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    /// Whether the stream includes a terminal usage-only chunk.
    pub include_usage: Option<bool>,
}

/// A tool definition accepted by chat-completion requests.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Tool {
    /// Function tool callable by the assistant.
    Function {
        /// Function declaration exposed to the model.
        function: Function,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Function-tool name, description, and JSON parameter schema.
pub struct Function {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional human-readable function description.
    pub description: Option<String>,
    /// JSON Schema describing accepted function arguments.
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Completed assistant tool call.
pub struct ToolCall {
    /// Model-assigned tool call identifier.
    pub id: String,
    /// Tool type, currently `function`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function name and serialized arguments.
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Completed function name and serialized arguments.
pub struct FunctionCallResponse {
    /// Function name selected by the model.
    pub name: String,
    /// Serialized JSON arguments, when supplied.
    #[serde(default)]
    pub arguments: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
/// Incremental fields for one indexed tool call.
pub struct ToolCallDelta {
    /// Position of the tool call within the assistant message.
    pub index: u32,
    /// Tool call identifier when first announced.
    pub id: Option<String>,
    /// Tool type when first announced.
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    /// Incremental function fields.
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
/// Incremental function name and argument fragments.
pub struct FunctionCallDelta {
    /// Function name when first announced.
    pub name: Option<String>,
    /// Incremental serialized argument fragment.
    pub arguments: Option<String>,
}

/// Tool choice value for simple string options.
///
/// Only the `auto` and `none` string forms are accepted; `required` and
/// named-function objects fail deserialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    /// Lets the model decide whether to call a tool.
    Auto,
    /// Prevents the model from calling tools.
    None,
}

/// Tool choice for the Chat Completion API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ToolChoice(pub ToolChoiceValue);

impl Default for ToolChoice {
    /// Returns the default value.
    fn default() -> Self {
        Self(ToolChoiceValue::Auto)
    }
}

/// A role-tagged OpenAI-compatible chat message.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role")]
#[serde(deny_unknown_fields)]
pub enum ChatMessage {
    /// System instruction message.
    #[serde(rename = "system")]
    System {
        /// Instruction content.
        content: MessageContent,
        /// Optional participant name.
        name: Option<String>,
    },
    /// User-authored message.
    #[serde(rename = "user")]
    User {
        /// User content.
        content: MessageContent,
        /// Optional participant name.
        name: Option<String>,
    },
    /// Assistant-authored message.
    #[serde(rename = "assistant")]
    Assistant {
        /// Optional visible assistant content.
        content: Option<MessageContent>,
        /// Optional participant name.
        name: Option<String>,
        /// Tool calls requested by the assistant.
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for reasoning-capable models. The request field
        /// is named `reasoning`, while responses emit `reasoning_content`.
        reasoning: Option<String>,
    },
    /// Tool result message.
    #[serde(rename = "tool")]
    Tool {
        /// Tool result content.
        content: MessageContent,
        /// Identifier of the assistant tool call being answered.
        tool_call_id: String,
    },
    /// Developer instruction message with optional tool declarations.
    #[serde(rename = "developer")]
    Developer {
        /// Developer instruction content.
        content: MessageContent,
        /// Tools introduced by the developer message.
        tools: Option<Vec<Tool>>,
        /// Optional participant name.
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
/// Chat message content accepted as text or typed parts.
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Ordered multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// OpenAI usage fields plus optional generated-image lifecycle accounting.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    /// Tokens consumed by the prompt.
    pub prompt_tokens: u32,
    /// Sum of prompt and completion tokens.
    pub total_tokens: u32,
    /// Tokens generated by the model, when applicable, including tokens
    /// retained internally for reasoning or model control.
    pub completion_tokens: Option<u32>,
    /// Prompt token breakdown, when available.
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
    /// Completion token breakdown, when available.
    pub completion_tokens_details: Option<CompletionTokenUsageInfo>,
    /// Number of generated images, when applicable.
    pub image_count: Option<u32>,
    /// Total denoising steps across generated images.
    pub image_steps: Option<u32>,
    /// Denoising steps observed for each generated image.
    pub image_steps_per_image: Option<Vec<u32>>,
}

impl Usage {
    /// Creates usage from prompt and completion token counts.
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            // `prompt_tokens + completion_tokens` can exceed `u32::MAX`, which
            // would panic in debug builds and wrap in release builds.
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: None,
            completion_tokens_details: None,
            image_count: None,
            image_steps: None,
            image_steps_per_image: None,
        }
    }

    /// Creates usage for a response that includes generated images.
    pub fn from_generation_counts(
        prompt_tokens: u32,
        completion_tokens: u32,
        image_count: u32,
        image_steps: u32,
    ) -> Self {
        Self {
            image_count: Some(image_count),
            image_steps: Some(image_steps),
            ..Self::from_counts(prompt_tokens, completion_tokens)
        }
    }

    /// Attaches the observed denoising-step count for each completed image.
    ///
    /// An empty list sets the field to `None`, which omits it from the wire.
    pub fn with_image_steps_per_image(mut self, image_steps_per_image: Vec<u32>) -> Self {
        self.image_steps_per_image =
            (!image_steps_per_image.is_empty()).then_some(image_steps_per_image);
        self
    }
}

/// Prompt-token usage breakdown.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokenUsageInfo {
    /// Prompt tokens reused from cache, when reported.
    pub cached_tokens: Option<u32>,
}

/// Completion-token usage breakdown.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokenUsageInfo {
    /// Completion tokens retained as reasoning content.
    pub reasoning_tokens: Option<u32>,
}

/// Generated-token logprob response payload.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    /// Per-token logprob entries, or `None` when unavailable.
    pub content: Option<Vec<ChatLogProbsContent>>,
}

/// Logprob details for one generated token.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbsContent {
    /// Decoded token text, or the `token_id:<id>` placeholder when the
    /// request sets `return_tokens_as_token_ids`.
    pub token: String,
    /// Natural logarithm of the selected token probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Option<Vec<u8>>,
    /// Candidate tokens at this position, including the selected token.
    pub top_logprobs: Vec<TopLogProb>,
}

/// One candidate token and its log probability.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    /// Decoded token text, or the `token_id:<id>` placeholder when the
    /// request sets `return_tokens_as_token_ids`.
    pub token: String,
    /// Natural logarithm of the token probability.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Option<Vec<u8>>,
}

/// Wire response containing one structured API error.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    /// Structured API error.
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
/// Structured OpenAI-compatible error details.
pub struct ErrorDetail {
    /// Human-readable error description.
    pub message: String,
    /// Stable OpenAI-compatible error category.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Request parameter associated with the error, when known.
    pub param: Option<String>,
    /// Stable machine-readable error code, when available.
    pub code: Option<String>,
}

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelObject {
    /// Served model identifier.
    pub id: String,
    /// Object type, always `model`.
    pub object: String,
    /// Unix timestamp in seconds; the listing reports when it was produced.
    pub created: i64,
    /// Owner label; the listing reports `uniserve`.
    pub owned_by: String,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// Object type, always `list`.
    pub object: String,
    /// Models served by this configuration.
    pub data: Vec<ModelObject>,
}

/// Trait for request types that need post-deserialization normalization.
///
/// The route's `ValidatedJson` extractor calls `normalize` after
/// deserialization and before `validator` runs.
pub trait Normalizable {
    /// Applies request defaults and canonical transformations after deserialization.
    fn normalize(&mut self) {
        // Most request schemas require no post-deserialization transformation.
    }
}
