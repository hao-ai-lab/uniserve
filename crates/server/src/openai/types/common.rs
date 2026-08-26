use std::slice;

use llm_multimodal::ImageDetail;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Default value helpers
// ============================================================================

/// Helper function for serde default value (returns true).
pub(super) fn default_true() -> bool {
    true
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::String(s) => slice::from_ref(s),
            StringOrArray::Array(arr) => arr,
        }
    }

    #[allow(unused)]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s],
            StringOrArray::Array(arr) => arr,
        }
    }
}

/// Validates stop sequences (non-empty strings)
pub(super) fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    if stop.as_slice().iter().any(|s| s.is_empty()) {
        return Err(validator::ValidationError::new(
            "stop strings cannot be empty",
        ));
    }
    Ok(())
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validates top_p: 0.0 < top_p <= 1.0.
pub(super) fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

// ============================================================================
// Reasoning controls
// ============================================================================

/// Effort level accepted by OpenAI-compatible reasoning requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        uuid: Option<String>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<ImageDetail>,
}

// ============================================================================
// Streaming
// ============================================================================

/// Mirrors the `StreamOptions` class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Tool choice value for simple string options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    None,
}

/// Tool choice for the Chat Completion API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ToolChoice(pub ToolChoiceValue);

impl Default for ToolChoice {
    fn default() -> Self {
        Self(ToolChoiceValue::Auto)
    }
}

// ============================================================================
// Chat Messages
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role")]
#[serde(deny_unknown_fields)]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<MessageContent>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for reasoning-capable models.
        reasoning: Option<String>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
    #[serde(rename = "developer")]
    Developer {
        content: MessageContent,
        tools: Option<Vec<Tool>>,
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

// ============================================================================
// Usage and Logging
// ============================================================================

/// OpenAI usage fields plus optional generated-image lifecycle accounting.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    pub completion_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
    pub completion_tokens_details: Option<CompletionTokenUsageInfo>,
    pub image_count: Option<u32>,
    pub image_steps: Option<u32>,
    pub image_steps_per_image: Option<Vec<u32>>,
}

impl Usage {
    /// Create a Usage from prompt and completion token counts.
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            // `prompt_tokens + completion_tokens` can exceed u32::MAX, which would
            // panic in debug builds and silently wrap in release; saturate instead so
            // total_tokens stays a monotone, non-wrapping value.
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: None,
            completion_tokens_details: None,
            image_count: None,
            image_steps: None,
            image_steps_per_image: None,
        }
    }

    /// Create usage for a response that includes generated images.
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

    /// Attach the observed denoising-step count for each completed image.
    pub fn with_image_steps_per_image(mut self, image_steps_per_image: Vec<u32>) -> Self {
        self.image_steps_per_image =
            (!image_steps_per_image.is_empty()).then_some(image_steps_per_image);
        self
    }
}

/// Mirrors the `PromptTokenUsageInfo` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: Option<u32>,
}

/// Mirrors the `CompletionTokenUsageInfo` class.
/// Breakdown of completion-token usage required by the current OpenAI spec,
/// notably `reasoning_tokens` for reasoning-capable models.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct CompletionTokenUsageInfo {
    pub reasoning_tokens: Option<u32>,
}

/// Mirrors the `ChatCompletionLogProbs` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    pub content: Option<Vec<ChatLogProbsContent>>,
}

/// Mirrors the `ChatCompletionLogProbsContent` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

/// Mirrors the `ChatCompletionLogProb` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Model types
// ============================================================================

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub identity: ServedModelIdentity,
    pub capabilities: ModelCapabilities,
}

/// Load-bound identity of the configured model description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServedModelIdentity {
    pub profile_id: String,
    pub description_id: String,
    pub config_fingerprint: String,
}

/// Typed public capabilities for the configured model route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub endpoints: Vec<ModelEndpoint>,
    pub input_modalities: Vec<ModelModality>,
    pub output_modalities: Vec<ModelModality>,
    pub features: Vec<ModelFeature>,
    pub sampling_controls: Vec<ModelSamplingControl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEndpoint {
    ChatCompletions,
    ImageGenerations,
    VideoGenerations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Text,
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFeature {
    Streaming,
    Usage,
    Logprobs,
    Reasoning,
    ToolCalling,
    RepeatedInterleave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSamplingControl {
    Greedy,
    Temperature,
    TopK,
    TopP,
    MinP,
    RepetitionPenalty,
    FrequencyPenalty,
    PresencePenalty,
    LogitBias,
    AllowedTokenIds,
    BadWords,
    MinTokens,
    Logprobs,
    StopTokenIds,
    Eos,
    StopStrings,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

// ============================================================================
// Normalizable trait
// ============================================================================

/// Trait for request types that need post-deserialization normalization.
pub trait Normalizable {
    /// Normalize the request by applying defaults and transformations.
    fn normalize(&mut self) {
        // Default: no-op
    }
}
