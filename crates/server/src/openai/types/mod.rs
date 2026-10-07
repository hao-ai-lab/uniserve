//! Public OpenAI-compatible wire schemas.
//!
//! The submodules are private; their public types are re-exported here and
//! again from `crate::openai`, so callers outside `crate::openai` name them
//! as `crate::openai::Type`.

mod chat_completions;
mod common;
mod images;
mod videos;

pub use chat_completions::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatImageConfig, ChatMessageDelta, ChatModality,
};
pub use common::{
    ChatLogProbs, ChatLogProbsContent, ChatMessage, CompletionTokenUsageInfo, ContentPart,
    ErrorDetail, ErrorResponse, Function, FunctionCallDelta, FunctionCallResponse, ImageUrl,
    ListModelsResponse, MessageContent, ModelObject, Normalizable, PromptTokenUsageInfo,
    ReasoningEffort, StreamOptions, StringOrArray, Tool, ToolCall, ToolCallDelta, ToolChoice,
    ToolChoiceValue, TopLogProb, Usage,
};
pub use images::{GeneratedImageData, ImageGenerationRequest, ImageGenerationResponse};
pub use videos::{DEFAULT_VIDEO_SEED, VideoCondition, VideoGenerationRequest, VideoTarget};
