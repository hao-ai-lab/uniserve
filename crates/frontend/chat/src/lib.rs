//! Minimal chat facade above [`text`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request ->
//! streamed structured assistant events`. The request side remains text-first,
//! while the response side can emit structured reasoning and final-answer
//! blocks. It is closer to the reference internal chat-rendering flow than to a full
//! OpenAI-compatible surface.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
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
pub use uniserve_llm::FinishReason;
pub use uniserve_reasoning_parser::{ReasoningDelta, ReasoningError, ReasoningParser};
pub use uniserve_tool_parser::{ToolParser, ToolParserError};

mod backend;
mod error;
pub mod multimodal;
pub mod output {
    pub use uniserve_chat_output::output::*;
}
pub mod parser {
    pub use uniserve_chat_output::parser::*;
}
pub mod renderer {
    pub use uniserve_chat_template::renderer::*;
}
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
mod stream;

use uniserve_engine_client::EngineCoreClient;
use uniserve_engine_client::protocol::ModelDtype;
use uniserve_llm::Llm;
use uniserve_text::{TextLlm, TextRequest};

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

/// Structured chat facade above [`TextLlm`].

/// This layer stays above raw text semantics: it takes care of chat-template
/// rendering, exposes structured assistant events, and adds chat-specific
/// request semantics such as tool calls.
pub struct ChatLlm {
    text: TextLlm,
    backend: DynChatBackend,
 /// Effective model dtype reported by the engine.
    model_dtype: ModelDtype,
 /// Tool-call parser selection.
    tool_call_parser: ParserSelection,
 /// Reasoning parser selection.
    uniserve_reasoning_parser: ParserSelection,
}

impl ChatLlm {
 /// Create a new chat facade from a text-generation facade plus a chat
 /// backend.
    pub fn new(text: TextLlm, backend: DynChatBackend) -> Self {
        let model_dtype = text.uniserve_engine_client().model_dtype();

        Self {
            text,
            backend,
            model_dtype,
            tool_call_parser: ParserSelection::Auto,
            uniserve_reasoning_parser: ParserSelection::Auto,
        }
    }

 /// Convenience constructor for one shared backend object that implements
 /// both text and chat responsibilities.
    #[allow(
        clippy::clone_on_ref_ptr,
        reason = "clone performs an Arc trait-object upcast from ChatTextBackend to TextBackend"
    )]
    pub fn from_shared_backend(llm: Llm, backend: DynChatTextBackend) -> Self {
        let text_backend: uniserve_text::DynTextBackend = backend.clone();
        let text = TextLlm::new(llm, text_backend);
        Self::new(text, backend)
    }

 /// Set tool-call parser selection.
    pub fn with_tool_call_parser(mut self, selection: ParserSelection) -> Self {
        self.tool_call_parser = selection;
        self
    }

 /// Set reasoning parser selection.
    pub fn with_reasoning_parser(mut self, selection: ParserSelection) -> Self {
        self.uniserve_reasoning_parser = selection;
        self
    }

 /// Override the effective model dtype used for multimodal tensor encoding.
    pub fn with_model_dtype(mut self, model_dtype: ModelDtype) -> Self {
        self.model_dtype = model_dtype;
        self
    }

 /// Expose the underlying text facade for raw text-generation routes such as
 /// `/v1/completions`.
    pub fn text(&self) -> &TextLlm {
        &self.text
    }

 /// Return the model ID reported by the underlying text backend.
    pub fn model_id(&self) -> &str {
        self.text.model_id()
    }

 /// Expose the underlying engine client for low-level utility/admin
 /// calls.
    pub fn uniserve_engine_client(&self) -> &EngineCoreClient {
        self.text.uniserve_engine_client()
    }

 /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, mut request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let output_processor = self.backend.new_chat_output_processor(
            &mut request,
            NewChatOutputProcessorOptions {
                tool_call_parser: &self.tool_call_parser,
                uniserve_reasoning_parser: &self.uniserve_reasoning_parser,
            },
        )?;
 // Chat-template rendering is CPU-bound and can be expensive for large
 // conversations, so run it on the blocking pool rather than stalling the
 // async executor. The request is moved in and handed back so downstream
 // lowering can keep using it without an extra clone.
        let chat_renderer = self.backend.chat_renderer();
        let (request, rendered) =
            tokio::task::spawn_blocking(move || -> Result<(ChatRequest, RenderedPrompt)> {
                let rendered = chat_renderer.render(&request)?;
                Ok((request, rendered))
            })
            .await
            .map_err(|error| Error::ChatTemplate(format!("chat render task failed: {error}")))??;

        let (prompt, mm_features) = multimodal::finalize_rendered_prompt(
            &request,
            rendered,
            self.backend.multimodal_model_info(),
            self.model_dtype,
        )
        .await?;

        let text_request = TextRequest {
            request_id: request.request_id.clone(),
            prompt,
            mm_features,
            sampling_params: request.sampling_params,
            decode_options: request.decode_options,
            intermediate: request.intermediate,
            priority: request.priority,
            cache_salt: request.cache_salt,
            add_special_tokens: request.add_special_tokens,
            data_parallel_rank: request.data_parallel_rank,
            lora_request: request.lora_request,
        };
        let decoded_stream = self
            .text
            .generate(text_request)
            .await?
            .map_err(uniserve_chat_output::Error::from)
            .boxed();

        let structured_stream = output_processor.process(decoded_stream)?;

        Ok(ChatEventStream::new(request.request_id, structured_stream))
    }

 /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.text.shutdown().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use thiserror_ext::AsReport;

    use super::{ParserSelection, validate_parser_overrides};
    use crate::parser::reasoning::names;

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

    #[test]
    fn validate_parser_overrides_rejects_unknown_tool_parser() {
        let error = validate_parser_overrides(
            &ParserSelection::Explicit("definitely_missing_tool_parser".to_string()),
            &ParserSelection::Auto,
        )
        .unwrap_err();

        expect_test::expect!["tool parser `definitely_missing_tool_parser` is not registered (choose from: deepseek_v3, deepseek_v31, deepseek_v32, deepseek_v4, gemma4, glm45, glm47, hermes, hy_v3, internlm, kimi_k2, llama3_json, llama4_json, minimax_m2, mistral, phi4_mini_json, qwen3_coder, qwen3_xml)"].assert_eq(&error.to_report_string());
    }

    #[test]
    fn validate_parser_overrides_rejects_unknown_reasoning_parser() {
        let error = validate_parser_overrides(
            &ParserSelection::Auto,
            &ParserSelection::Explicit("definitely_missing_reasoning_parser".to_string()),
        )
        .unwrap_err();

        expect_test::expect!["reasoning parser `definitely_missing_reasoning_parser` is not registered (choose from: cohere_cmd, deepseek_r1, deepseek_v3, deepseek_v4, gemma4, glm45, kimi, kimi_k2, minimax_m2, nemotron_v3, qwen3, step3)"].assert_eq(&error.to_report_string());
    }
}
