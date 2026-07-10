//! Default output processing pipeline.

mod reasoning;
mod tool;

use std::sync::Once;

use crate::text::tokenizer::DynTokenizer;
use futures::{Stream, StreamExt as _};
use tracing::{info, warn};
use trait_set::trait_set;

use self::reasoning::reasoning_event_stream;
use self::tool::tool_event_stream;
use super::structured::structured_chat_event_stream;
use crate::chat::output::error::Result;
use crate::chat::output::parser::ParserSelection;
use crate::chat::output::parser::reasoning::{ReasoningParser, ReasoningParserFactory};
use crate::chat::output::parser::tool::{ToolParser, ToolParserFactory};
use crate::chat::output::processor::{
    AssistantEvent, ChatOutputProcessor, ContentEvent, DynChatEventStream,
    DynDecodedTextEventStream,
};
use crate::chat::output::request::{ChatRequest, ChatToolChoice};
use crate::chat::output::{Error, Result as ChatResult};

trait_set! {
    trait ContentEventStream = Stream<Item = Result<ContentEvent>> + Send + 'static;
}

/// Default request-scoped output processor used by Hugging Face style chat
/// backends.
///
/// This implementation assumes the backend already emitted decoded text deltas,
/// then optionally layers reasoning parsing and tool-call parsing before
/// assembling final structured chat events.
pub struct DefaultChatOutputProcessor {
    uniserve_reasoning_parser: Option<Box<dyn ReasoningParser>>,
    uniserve_tool_parser: Option<Box<dyn ToolParser>>,
}

impl DefaultChatOutputProcessor {
    /// Build the default output processor and apply any parser-specific request
    /// adjustments.
    ///
    /// Parser resolution happens here so that request validation, prompt
    /// rendering, and streaming all observe the same parser-adjusted
    /// request state.
    pub fn new(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        tool_call_parser: &ParserSelection,
        uniserve_reasoning_parser: &ParserSelection,
    ) -> ChatResult<Self> {
        let tool_parsing_enabled =
            matches!(request.tool_choice, ChatToolChoice::Auto) && !request.tools.is_empty();
        let uniserve_tool_parser = if tool_parsing_enabled {
            Some(Self::resolve_tool_parser(
                request,
                model_id,
                tool_call_parser,
            )?)
        } else {
            None
        };
        let uniserve_reasoning_parser = Self::resolve_optional_reasoning_parser(
            request,
            model_id,
            tokenizer,
            uniserve_reasoning_parser,
        )?;

        Ok(Self {
            uniserve_reasoning_parser,
            uniserve_tool_parser,
        })
    }

    /// Build the plain-text-only default output processor.
    ///
    /// This keeps the default structured chat-event assembly but disables both
    /// reasoning parsing and tool-call parsing completely, so that all
    /// content is treated as opaque text.
    pub fn plain_text_only() -> Self {
        Self {
            uniserve_reasoning_parser: None,
            uniserve_tool_parser: None,
        }
    }

    fn resolve_tool_parser(
        request: &mut ChatRequest,
        model_id: &str,
        selection: &ParserSelection,
    ) -> ChatResult<Box<dyn ToolParser>> {
        let factory = ToolParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => factory.resolve_name_for_model(model_id).ok_or_else(|| {
                Error::ParserUnavailableForModel {
                    kind: "tool",
                    model_id: model_id.to_string(),
                }
            })?,
            ParserSelection::None => return Err(Error::ParserDisabled { kind: "tool" }),
            ParserSelection::Explicit(name) => name.as_str(),
        };

        let parser = factory.create(parser_name, &request.tools)?;

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        TOOL_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using tool parser"));
        Ok(parser)
    }

    fn resolve_optional_reasoning_parser(
        request: &mut ChatRequest,
        model_id: &str,
        tokenizer: DynTokenizer,
        selection: &ParserSelection,
    ) -> ChatResult<Option<Box<dyn ReasoningParser>>> {
        let factory = ReasoningParserFactory::global();
        let parser_name = match selection {
            ParserSelection::Auto => factory.resolve_name_for_model(model_id),
            ParserSelection::None => None,
            ParserSelection::Explicit(name) => Some(name.as_str()),
        };

        let Some(parser_name) = parser_name else {
            REASONING_PARSER_LOG_ONCE.call_once(|| info!("reasoning parsing disabled"));
            return Ok(None);
        };

        let parser = match factory.create(parser_name, tokenizer) {
            Ok(parser) => parser,
            Err(error) if matches!(selection, ParserSelection::Auto) => {
                AUTO_REASONING_PARSER_FALLBACK_LOG_ONCE.call_once(|| {
                    warn!(
                        parser_name,
                        error = %error,
                        "failed to initialize auto-selected reasoning parser; falling back to plain text"
                    );
                });
                return Ok(None);
            }
            Err(error) => return Err(error),
        };

        if parser.preserve_special_tokens() {
            request.decode_options.skip_special_tokens = false;
        }

        REASONING_PARSER_LOG_ONCE.call_once(|| info!(parser_name, "using reasoning parser"));
        Ok(Some(parser))
    }
}

static TOOL_PARSER_LOG_ONCE: Once = Once::new();
static REASONING_PARSER_LOG_ONCE: Once = Once::new();
static AUTO_REASONING_PARSER_FALLBACK_LOG_ONCE: Once = Once::new();

impl ChatOutputProcessor for DefaultChatOutputProcessor {
    /// Transforms a raw generate-output token stream into structured chat
    /// events through three sequential stages once text decoding has
    /// already happened:
    ///
    /// 1. [`reasoning_event_stream`] — reasoning/content separation
    /// 2. [`tool_event_stream`] — tool-call parsing
    /// 3. [`structured_chat_event_stream`] — final block assembly
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream> {
        let reasoning = reasoning_event_stream(decoded, self.uniserve_reasoning_parser);
        let tool = tool_event_stream(reasoning, self.uniserve_tool_parser);
        let structured = structured_chat_event_stream(tool);

        Ok(structured.boxed())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uniserve_model_profile::tokenizer::Tokenizer;

    use super::*;
    use crate::chat::output::request::ChatRequest;

    struct TokenizerWithoutReasoningDelimiters;

    impl Tokenizer for TokenizerWithoutReasoningDelimiters {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<Vec<u32>> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> uniserve_model_profile::tokenizer::Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
                .collect())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
    }

    #[test]
    fn auto_reasoning_parser_init_failure_falls_back_to_plain_text() {
        let mut request = ChatRequest::for_test();
        let parser = DefaultChatOutputProcessor::resolve_optional_reasoning_parser(
            &mut request,
            "Qwen/Qwen3-0.6B-Base",
            Arc::new(TokenizerWithoutReasoningDelimiters),
            &ParserSelection::Auto,
        )
        .expect("auto reasoning parser init should not fail the request");

        assert!(parser.is_none());
    }

    #[test]
    fn explicit_reasoning_parser_init_failure_is_reported() {
        let mut request = ChatRequest::for_test();
        let error = match DefaultChatOutputProcessor::resolve_optional_reasoning_parser(
            &mut request,
            "Qwen/Qwen3-0.6B-Base",
            Arc::new(TokenizerWithoutReasoningDelimiters),
            &ParserSelection::Explicit("qwen3".to_string()),
        ) {
            Ok(_) => panic!("explicit parser initialization failures should remain actionable"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("failed to initialize reasoning parser `qwen3`")
        );
    }
}
