//! Assistant output processing for chat generation.
//!
//! Decoded text passes through the reasoning stage (`reasoning`) and then the
//! tool-call stage (`tool`), which produce the [`AssistantEvent`] stream
//! defined in `processor`. Each model family composes the two stages with its
//! own parsers: `qwen3` for Qwen3 and `gemma4` for Gemma-4, and
//! [`ChatOutputProcessor`] names the family a request's model selects.
//! `assemble_chat_event_stream` drives that stream and uses
//! `structured::OutputProcessor` to turn assistant events into content-block
//! and tool-call serving events.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod gemma4;
pub(crate) mod processor;
mod qwen3;
mod reasoning;
pub(crate) mod structured;
mod tool;
pub use processor::AssistantEvent;

pub use gemma4::Gemma4ChatOutputProcessor;
pub use qwen3::{Qwen3ChatOutputProcessor, template_instructs_json_tool_calls};

use futures::stream::BoxStream;

use crate::serving::chat::Result as ChatResult;

/// The chat output processor of a request, chosen by its model's chat format.
pub enum ChatOutputProcessor {
    /// Qwen3 `<think>` reasoning and JSON tool calls.
    Qwen3(Qwen3ChatOutputProcessor),
    /// Gemma-4 thought channels and native tool calls.
    Gemma4(Gemma4ChatOutputProcessor),
}

impl ChatOutputProcessor {
    /// Parses decoded text into reasoning, visible text, and tool calls with
    /// the selected family's parsers.
    ///
    /// Decoder errors propagate as `Error::Text`.
    pub fn parse(
        self,
        decoded: impl futures::Stream<
            Item = crate::serving::text::Result<crate::serving::text::DecodedTextEvent>,
        > + Send
        + 'static,
    ) -> BoxStream<'static, ChatResult<AssistantEvent>> {
        match self {
            Self::Qwen3(processor) => Box::pin(processor.parse(decoded)),
            Self::Gemma4(processor) => Box::pin(processor.parse(decoded)),
        }
    }
}

pub use crate::profile::reasoning::ReasoningDelta;
pub use crate::profile::tools::ToolParserError;
pub use crate::serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantToolCall, ChatRequest,
    ChatToolChoice, Tool,
};
pub use crate::serving::text::FinishReason;
