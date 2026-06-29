//! OpenAI-compatible request and response schemas.
//!
//! This crate owns the JSON-shaped DTOs used by OpenAI-compatible routes. It
//! intentionally does not depend on Axum, server state, or generation runtime
//! crates; higher layers validate and lower these values into UniServe domain
//! requests.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod chat_completions;
pub mod common;
pub mod completions;
pub mod structured_outputs;

pub use chat_completions::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatMessageDelta,
};
pub use common::{
    ChatLogProbs, ChatLogProbsContent, ChatMessage, ContentPart, ErrorDetail, ErrorResponse,
    Function, FunctionCallDelta, FunctionCallResponse, FunctionChoice, ImageUrl,
    ListModelsResponse, LogProbs, MessageContent, ModelObject, Normalizable, Prompt,
    PromptTokenUsageInfo, ReasoningEffort, StreamOptions, StringOrArray, Tool, ToolCall,
    ToolCallDelta, ToolChoice, ToolChoiceValue, ToolReference, TopLogProb, Usage, VideoUrl,
};
pub use completions::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionSseChunk,
    CompletionStreamChoice, CompletionStreamResponse,
};
pub use structured_outputs::{JsonSchemaFormat, ResponseFormat};
