//! Qwen3 reasoning and tool-call output processing pipeline.

mod reasoning;
mod tool;

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use super::structured::structured_chat_event_stream;
use crate::profile::reasoning::Qwen3ReasoningParser;
use crate::profile::tools::Qwen3XmlToolParser;
use crate::serving::chat::output::{Error, Result as ChatResult};
use crate::serving::chat::{ChatRequest, ChatToolChoice};
use crate::serving::text::tokenizer::DynTokenizer;

/// Request-scoped Qwen3 reasoning and tool-call processor.
pub struct Qwen3ChatOutputProcessor {
    reasoning_parser: Option<Qwen3ReasoningParser>,
    tool_parser: Option<Qwen3XmlToolParser>,
}

impl Qwen3ChatOutputProcessor {
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

    /// Transforms a committed token stream into structured chat
    /// events through three sequential stages once text decoding has
    /// already happened:
    ///
    /// 1. [`reasoning_event_stream`] — reasoning/content separation
    /// 2. [`tool_event_stream`] — tool-call parsing
    /// 3. [`structured_chat_event_stream`] — final block assembly
    pub fn process(
        self,
        decoded: impl futures::Stream<
            Item = crate::serving::text::Result<crate::serving::text::DecodedTextEvent>,
        > + Send,
    ) -> ChatResult<impl futures::Stream<Item = ChatResult<crate::serving::chat::ChatEvent>> + Send>
    {
        let reasoning = reasoning_event_stream(decoded, self.reasoning_parser);
        let tool = tool_event_stream(reasoning, self.tool_parser);
        let structured = structured_chat_event_stream(tool);

        Ok(structured)
    }
}
