//! Chat assistant output processing and parser selection.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod parser;
pub mod stream;

#[path = "output/mod.rs"]
pub(crate) mod processor;

pub use error::{Error, Result};
pub use parser::ParserSelection;
pub use parser::reasoning::ReasoningParserFactory;
pub use parser::tool::ToolParserFactory;
pub use processor::{
    ChatOutputProcessor, DefaultChatOutputProcessor, DynChatEventStream, DynChatOutputProcessor,
    DynDecodedTextEventStream, HarmonyChatOutputProcessor, validate_harmony_parser_overrides,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};

pub use crate::chat::protocol::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent, ChatRequest, ChatTool, ChatToolChoice,
};
pub use crate::text::FinishReason;
pub use uniserve_model_profile::reasoning::{ReasoningDelta, ReasoningParser};
pub use uniserve_model_profile::tools::{ToolParser, ToolParserError};

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
