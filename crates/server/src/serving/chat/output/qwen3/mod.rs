//! Qwen3 reasoning and tool-call output processor composition.
//!
//! Decoded text flows through two chained stream stages: the reasoning stage
//! splits `<think>` sections into reasoning deltas, and the tool stage extracts
//! Qwen3 XML tool calls from the remaining visible text. A stage without a
//! parser forwards text unchanged.

use super::reasoning::reasoning_event_stream;
use super::tool::tool_event_stream;
use crate::profile::reasoning::Qwen3ReasoningParser;
use crate::profile::tools::Qwen3XmlToolParser;
use crate::serving::chat::{ChatRequest, ChatToolChoice};
use crate::serving::chat::{Error, Result as ChatResult};
use crate::serving::text::tokenizer::DynTokenizer;

/// Request-scoped Qwen3 reasoning and tool-call processor.
pub struct Qwen3ChatOutputProcessor {
    reasoning_parser: Option<Qwen3ReasoningParser>,
    tool_parser: Option<Qwen3XmlToolParser>,
}

impl Qwen3ChatOutputProcessor {
    /// Creates a request-scoped Qwen3 reasoning and tool output processor.
    ///
    /// Tool-call parsing is enabled only when `tool_choice` is `Auto` and the
    /// request declares tools, the same condition as
    /// `ChatRequest::tool_parsing_enabled`, which also decides whether
    /// `HfChatRenderer` exposes the tools to the template. Reasoning parsing is
    /// enabled when `parse_reasoning` is set. The request is only read.
    ///
    /// # Errors
    ///
    /// Returns `Error::ParserInitialization` when the reasoning parser cannot
    /// be built, for example because the tokenizer lacks the `<think>` or
    /// `</think>` token.
    pub fn new(
        request: &mut ChatRequest,
        tokenizer: DynTokenizer,
        parse_reasoning: bool,
    ) -> ChatResult<Self> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let tool_parser = if tool_parsing_enabled {
            let parser = Qwen3XmlToolParser::new(&request.tools);
            Some(parser)
        } else {
            None
        };
        let reasoning_parser = parse_reasoning
            .then(|| Qwen3ReasoningParser::new(tokenizer))
            .transpose()
            .map_err(|error| Error::ParserInitialization {
                kind: "reasoning",
                name: "qwen3".to_string(),
                error: Box::new(error),
            })?;
        Ok(Self {
            reasoning_parser,
            tool_parser,
        })
    }

    /// Parses decoded text into reasoning, visible text, and incremental tool calls.
    ///
    /// Decoder errors propagate as `Error::Text`.
    pub fn parse(
        self,
        decoded: impl futures::Stream<
            Item = crate::serving::text::Result<crate::serving::text::DecodedTextEvent>,
        > + Send,
    ) -> impl futures::Stream<Item = ChatResult<super::processor::AssistantEvent>> + Send {
        let reasoning = reasoning_event_stream(decoded, self.reasoning_parser);
        tool_event_stream(reasoning, self.tool_parser)
    }
}
