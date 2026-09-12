//! Qwen3 reasoning and tool-call output processor composition.

mod reasoning;
mod tool;

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
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
    pub fn new(
        request: &mut ChatRequest,
        tokenizer: DynTokenizer,
        parse_reasoning: bool,
    ) -> ChatResult<Self> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let tool_parser = if tool_parsing_enabled {
            let parser = Qwen3XmlToolParser::new(&request.tools);
            if parser.preserve_special_tokens() {
                request.decode_options.skip_special_tokens = false;
            }
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
        if reasoning_parser
            .as_ref()
            .is_some_and(Qwen3ReasoningParser::preserve_special_tokens)
        {
            request.decode_options.skip_special_tokens = false;
        }
        Ok(Self {
            reasoning_parser,
            tool_parser,
        })
    }

    /// Parses decoded text into reasoning, visible text, and incremental tool calls.
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
