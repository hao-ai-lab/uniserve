//! Gemma-4 reasoning and tool-call output processor composition.
//!
//! Decoded text flows through two chained stream stages: the reasoning stage
//! splits `<|channel>thought\n...<channel|>` sections into reasoning deltas,
//! and the tool stage extracts native Gemma-4 tool calls from the remaining
//! visible text. A stage without a parser forwards text unchanged.
//!
//! DiffusionGemma commits output in blocks of tokens, so one decoded delta can
//! hold a whole thought channel, visible text, and complete tool calls; both
//! stages accept any number of delimiters per delta, as well as delimiters
//! split across deltas.

use super::reasoning::reasoning_event_stream;
use super::tool::tool_event_stream;
use crate::profile::reasoning::Gemma4ReasoningParser;
use crate::profile::tools::Gemma4ToolParser;
use crate::serving::chat::ChatRequest;
use crate::serving::chat::{Error, Result as ChatResult};
use crate::serving::text::tokenizer::DynTokenizer;

/// Request-scoped Gemma-4 reasoning and tool-call processor.
pub struct Gemma4ChatOutputProcessor {
    reasoning_parser: Option<Gemma4ReasoningParser>,
    tool_parser: Option<Gemma4ToolParser>,
}

impl Gemma4ChatOutputProcessor {
    /// Creates a request-scoped Gemma-4 reasoning and tool output processor.
    ///
    /// Tool-call parsing is enabled when `request.tool_parsing_enabled()`
    /// holds, the condition under which `HfChatRenderer` exposes the tools to
    /// the template. Reasoning parsing is enabled when `parse_reasoning` is
    /// set. The thinking mode needs no separate input: the prompt suffix
    /// decides whether generation starts inside a thought channel, and
    /// otherwise the model's own channel delimiters mark reasoning.
    ///
    /// Sets `request.decode_options.skip_special_tokens` to `false`. Gemma-4
    /// delimits thought channels, tool calls, and string arguments with
    /// special tokens, which the parsers must see and which stream verbatim
    /// as content when parsing is disabled.
    ///
    /// # Errors
    ///
    /// Returns `Error::ParserInitialization` when the reasoning parser cannot
    /// be built because the tokenizer lacks the `<|channel>` or `<channel|>`
    /// token.
    pub fn new(
        request: &mut ChatRequest,
        tokenizer: DynTokenizer,
        parse_reasoning: bool,
    ) -> ChatResult<Self> {
        request.decode_options.skip_special_tokens = false;

        let tool_parser = request.tool_parsing_enabled().then(Gemma4ToolParser::new);
        let reasoning_parser = parse_reasoning
            .then(|| Gemma4ReasoningParser::new(tokenizer))
            .transpose()
            .map_err(|error| Error::ParserInitialization {
                kind: "reasoning",
                name: "gemma4".to_string(),
                error: Box::new(error),
            })?;

        Ok(Self {
            reasoning_parser,
            tool_parser,
        })
    }

    /// Parses decoded text into reasoning, visible text, and tool calls.
    ///
    /// Each tool call is published whole, name and complete JSON arguments
    /// together, once its end marker arrives. Decoder errors propagate as
    /// `Error::Text`.
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
