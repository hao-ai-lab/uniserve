//! Public OpenAI-compatible wire schemas.

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
pub use videos::VideoGenerationRequest;
