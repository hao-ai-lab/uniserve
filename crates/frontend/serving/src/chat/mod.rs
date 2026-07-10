//! Minimal chat facade above [`text`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request ->
//! streamed structured assistant events`. The request side remains text-first,
//! while the response side can emit structured reasoning and final-answer
//! blocks. It is closer to the reference internal chat-rendering flow than to a full
//! OpenAI-compatible surface.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub use crate::text::{FinishReason, StopReason};
pub use backend::hf::HfChatBackend;
pub use backend::{
    ChatBackend, ChatTextBackend, DynChatBackend, DynChatTextBackend, LoadModelBackendsOptions,
    LoadedModelBackends, NewChatOutputProcessorOptions, load_model_backends,
};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
use futures::{StreamExt, TryStreamExt as _};
pub use output::{
    ChatOutputProcessor, DefaultChatOutputProcessor, DynChatOutputProcessor,
    HarmonyChatOutputProcessor,
};
pub use parser::ParserSelection;
pub use parser::reasoning::ReasoningParserFactory;
pub use parser::tool::ToolParserFactory;
pub use renderer::hf::ChatTemplateContentFormatOption;
pub use renderer::{
    ChatRenderer, DeepSeekV4ChatRenderer, DeepSeekV32ChatRenderer, DynChatRenderer, RenderedPrompt,
    RendererSelection,
};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, GenerationPromptMode, ReasoningEffort, SamplingParams,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use uniserve_model_profile::reasoning::{ReasoningDelta, ReasoningError, ReasoningParser};
pub use uniserve_model_profile::tools::{ToolParser, ToolParserError};

mod backend;
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

use crate::text::{PreparedTextRequest, TextRequest, TextRuntime};
use uniserve_engine_gateway::EngineGateway;

/// One chat request after profile rendering and text-runtime lowering has
/// completed, but before scheduler admission.
#[derive(Debug)]
pub(crate) struct PreparedChatRequest {
    /// Parser-adjusted semantic chat request used for output processing.
    pub chat_request: ChatRequest,
    /// Tokenized text request ready for the lower text runtime.
    pub prepared_text_request: PreparedTextRequest,
}

/// Validate explicit parser override names without starting request processing.
pub fn validate_parser_overrides(
    tool_call_parser: &ParserSelection,
    uniserve_reasoning_parser: &ParserSelection,
) -> Result<()> {
    let tool_parser_factory = ToolParserFactory::global();
    if let ParserSelection::Explicit(name) = tool_call_parser
        && !tool_parser_factory.contains(name)
    {
        return Err(Error::ParserUnavailableByName {
            kind: "tool",
            name: name.clone(),
            available_names: tool_parser_factory.list(),
        });
    }

    let reasoning_parser_factory = ReasoningParserFactory::global();
    if let ParserSelection::Explicit(name) = uniserve_reasoning_parser
        && !reasoning_parser_factory.contains(name)
    {
        return Err(Error::ParserUnavailableByName {
            kind: "reasoning",
            name: name.clone(),
            available_names: reasoning_parser_factory.list(),
        });
    }

    Ok(())
}

/// Structured chat facade above [`TextRuntime`].
///
/// This layer stays above raw text semantics: it takes care of chat-template
/// rendering, exposes structured assistant events, and adds chat-specific
/// request semantics such as tool calls.
#[derive(Clone)]
pub(crate) struct ChatRuntime {
    text: TextRuntime,
    backend: DynChatBackend,
    /// Tool-call parser selection.
    tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    uniserve_reasoning_parser: ParserSelection,
}

impl ChatRuntime {
    /// Create a new chat facade from a text-generation facade plus a chat
    /// backend.
    pub(crate) fn new(text: TextRuntime, backend: DynChatBackend) -> Self {
        Self {
            text,
            backend,
            tool_call_parser: ParserSelection::Auto,
            uniserve_reasoning_parser: ParserSelection::Auto,
        }
    }

    /// Set tool-call parser selection.
    pub(crate) fn with_tool_call_parser(mut self, selection: ParserSelection) -> Self {
        self.tool_call_parser = selection;
        self
    }

    /// Set reasoning parser selection.
    pub(crate) fn with_reasoning_parser(mut self, selection: ParserSelection) -> Self {
        self.uniserve_reasoning_parser = selection;
        self
    }

    pub(crate) fn with_text_max_model_len(mut self, max_model_len: u32) -> Self {
        self.text = self.text.with_max_model_len(max_model_len);
        self
    }

    /// Expose the underlying text facade for raw text-generation routes such as
    /// `/v1/completions`.
    pub(crate) fn text(&self) -> &TextRuntime {
        &self.text
    }

    /// Compile one chat request through parser policy, profile rendering,
    /// multimodal preprocessing, tokenization, and text-runtime lowering
    /// without submitting it.
    pub(crate) async fn compile(&self, request: ChatRequest) -> Result<PreparedChatRequest> {
        if request.has_multimodal() {
            return Err(Error::UnsupportedMultimodalContent("image_url"));
        }

        let (request, rendered) = self.render(request).await?;
        let prompt = rendered.prompt;

        let text_request = TextRequest {
            request_id: request.request_id.clone(),
            prompt,
            sampling_params: request.sampling_params.clone(),
            decode_options: request.decode_options.clone(),
            intermediate: request.intermediate,
            priority: request.priority,
            cache_salt: request.cache_salt.clone(),
            add_special_tokens: request.add_special_tokens,
            data_parallel_rank: request.data_parallel_rank,
            trace_context: request.trace_context.clone(),
            adapter: request.adapter.clone(),
        };
        let prepared_text_request = self.text.compile(text_request)?;

        Ok(PreparedChatRequest {
            chat_request: request,
            prepared_text_request,
        })
    }

    pub(crate) async fn render(
        &self,
        mut request: ChatRequest,
    ) -> Result<(ChatRequest, RenderedPrompt)> {
        request.validate()?;
        let _output_processor = self.new_output_processor(&mut request)?;
        self.render_validated(request).await
    }

    async fn render_validated(
        &self,
        request: ChatRequest,
    ) -> Result<(ChatRequest, RenderedPrompt)> {
        let chat_renderer = self.backend.chat_renderer();
        let (request, rendered) =
            tokio::task::spawn_blocking(move || -> Result<(ChatRequest, RenderedPrompt)> {
                let rendered = chat_renderer.render(&request)?;
                Ok((request, rendered))
            })
            .await
            .map_err(|error| Error::ChatTemplate(format!("chat render task failed: {error}")))??;
        Ok((request, rendered))
    }

    /// Submit a previously compiled chat request and stream structured chat
    /// events.
    pub(crate) async fn chat_prepared(
        &self,
        gateway: &EngineGateway,
        mut prepared: PreparedChatRequest,
    ) -> Result<ChatEventStream> {
        let output_processor = self.new_output_processor(&mut prepared.chat_request)?;
        let request_id = prepared.chat_request.request_id.clone();
        let decoded_stream = self
            .text
            .generate_prepared(gateway, prepared.prepared_text_request)
            .await?
            .map_err(crate::chat::output::Error::from)
            .boxed();
        let structured_stream = output_processor.process(decoded_stream)?;

        Ok(ChatEventStream::new(request_id, structured_stream))
    }

    pub(crate) fn new_output_processor(
        &self,
        request: &mut ChatRequest,
    ) -> Result<DynChatOutputProcessor> {
        self.backend.new_chat_output_processor(
            request,
            NewChatOutputProcessorOptions {
                tool_call_parser: &self.tool_call_parser,
                uniserve_reasoning_parser: &self.uniserve_reasoning_parser,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ParserSelection, validate_parser_overrides};
    use crate::chat::parser::reasoning::names;

    #[test]
    fn validate_parser_overrides_accepts_registered_names() {
        validate_parser_overrides(
            &ParserSelection::Explicit("llama3_json".to_string()),
            &ParserSelection::Explicit(names::QWEN3.to_string()),
        )
        .unwrap();
    }

    #[test]
    fn validate_parser_overrides_accepts_auto_and_none() {
        validate_parser_overrides(&ParserSelection::Auto, &ParserSelection::None).unwrap();
    }
}
