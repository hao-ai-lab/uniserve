//! Chat rendering and output-processing library.
//!
//! This description-owned library provides the fixed Hugging Face chat renderer,
//! chat protocol values, and the request-scoped Qwen3 output processor.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use crate::profile::reasoning::{ReasoningDelta, ReasoningError};
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::text::{FinishReason, StopReason};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
pub use output::Qwen3ChatOutputProcessor;
pub use renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort, Tool,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use template::ChatTemplateLoadOptions;

mod error;
pub mod output;
pub mod protocol;
pub mod template;
pub mod parser {
    pub use crate::serving::chat::output::parser::*;
}
pub mod renderer {
    pub use crate::serving::chat::template::renderer::*;
}
pub mod event {
    pub use crate::serving::chat::protocol::{
        AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
        AssistantToolCall, ChatEvent,
    };
}
pub mod request {
    pub use crate::serving::chat::protocol::{
        ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole,
        ChatToolChoice, GenerationPromptMode, ReasoningEffort, Tool,
    };
}
pub mod stream;
