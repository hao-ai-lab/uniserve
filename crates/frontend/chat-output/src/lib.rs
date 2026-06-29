//! Chat assistant output processing and parser selection.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub mod error;
pub mod output;
pub mod parser;
pub mod stream;

pub use error::{Error, Result};
pub use output::{
    ChatOutputProcessor, DefaultChatOutputProcessor, DynChatEventStream, DynChatOutputProcessor,
    DynDecodedTextEventStream, HarmonyChatOutputProcessor,
};
pub use parser::ParserSelection;
pub use parser::reasoning::ReasoningParserFactory;
pub use parser::tool::ToolParserFactory;
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};

pub use uniserve_chat_protocol::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent, ChatRequest, ChatTool, ChatToolChoice,
};
pub use uniserve_llm::FinishReason;
pub use uniserve_reasoning_parser::{ReasoningDelta, ReasoningParser};
pub use uniserve_tool_parser::{ToolParser, ToolParserError};

pub mod event {
    pub use uniserve_chat_protocol::{
        AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
        AssistantToolCall, ChatEvent,
    };
}

pub mod request {
    pub use uniserve_chat_protocol::{
        ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
        ChatToolChoice, GenerationPromptMode, ReasoningEffort, SamplingParams,
    };
}
