//! Metadata retained for OpenAI response construction.

use crate::openai::ChatCompletionRequest;

/// Public response metadata retained by HTTP response construction.
///
/// The chat-completions route handler builds this before calling
/// `ServingRuntime::generate_chat`, which takes the request by value, and
/// passes the fields to `collect_chat_completion` or
/// `chat_completion_chunk_stream`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponseContext {
    /// OpenAI-compatible response identifier.
    pub request_id: String,
    /// Model name reported in response objects.
    pub response_model: String,
    /// Whether streaming responses include a terminal usage chunk
    /// (`stream_options.include_usage`). Non-streaming responses always
    /// carry usage.
    pub include_usage: bool,
    /// Whether generated-token logprobs were requested.
    pub requested_logprobs: bool,
    /// Whether prompt-token logprobs were requested. Only non-streaming
    /// responses carry them; `validate_request_compat` rejects a streamed
    /// request that asks for a positive or `-1` count.
    pub include_prompt_logprobs: bool,
    /// Whether parsed reasoning is exposed in responses.
    pub include_reasoning: bool,
    /// Whether token identifiers are exposed in responses.
    pub return_token_ids: bool,
    /// Whether textual tokens use token-identifier placeholders.
    pub return_tokens_as_token_ids: bool,
}

impl ChatResponseContext {
    /// Captures transport-only choices before the request enters preprocessing.
    pub fn from_request(
        request: &ChatCompletionRequest,
        request_id: String,
        served_model_name: &str,
    ) -> Self {
        Self {
            request_id,
            response_model: served_model_name.to_string(),
            include_usage: request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
            requested_logprobs: request.logprobs,
            include_prompt_logprobs: request.prompt_logprobs.is_some(),
            include_reasoning: request.include_reasoning,
            return_token_ids: request.return_token_ids.unwrap_or(false),
            return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
        }
    }
}
