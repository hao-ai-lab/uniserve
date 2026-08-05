//! Chat rendering and output-processing library.
//!
//! Under the S02 funnel this module is a description-owned library: it provides
//! the fixed Hugging Face chat renderer, the chat protocol request/event types,
//! and the request-scoped chat output processor. There is no chat backend tower,
//! no runtime facade, and no string-keyed renderer/parser selection on the
//! configured request path — [`crate::model::Qwen3Desc`] binds the concrete
//! renderer and parser policy directly.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub use crate::text::{FinishReason, StopReason};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
pub use output::{
    ChatOutputProcessor, DefaultChatOutputProcessor, DynChatOutputProcessor,
    HarmonyChatOutputProcessor,
};
pub use parser::ParserSelection;
pub use renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer};
pub use renderer::{ChatRenderer, DynChatRenderer, RenderedPrompt};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, GenerationPromptMode, ReasoningEffort, SamplingParams,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use template::ChatTemplateLoadOptions;
pub use uniserve_model_profile::reasoning::{ReasoningDelta, ReasoningError, ReasoningParser};
pub use uniserve_model_profile::tools::{ToolParser, ToolParserError};

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
        ChatToolChoice, GenerationPromptMode, ReasoningEffort, SamplingParams,
    };
}
mod stream;
