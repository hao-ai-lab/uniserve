//! Chat rendering and output-processing library.
//!
//! This description-owned library provides the fixed Hugging Face chat renderer,
//! chat protocol values, and the request-scoped Qwen3 output processor.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use crate::text::{FinishReason, StopReason};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
pub use output::Qwen3ChatOutputProcessor;
pub use renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, GenerationPromptMode, ReasoningEffort,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use template::ChatTemplateLoadOptions;
pub use uniserve_model_profile::reasoning::{ReasoningDelta, ReasoningError};
pub use uniserve_model_profile::tools::ToolParserError;

mod error;
pub mod output;
pub mod protocol;
pub mod template;
pub mod parser {
    pub use crate::chat::output::parser::*;
}
pub mod renderer {
    pub use crate::chat::template::renderer::*;
}
pub mod event {
    pub use crate::chat::protocol::{
        AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
        AssistantToolCall, ChatEvent,
    };
}
pub mod request {
    pub use crate::chat::protocol::{
        ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
        ChatToolChoice, GenerationPromptMode, ReasoningEffort,
    };
}
pub mod stream;
