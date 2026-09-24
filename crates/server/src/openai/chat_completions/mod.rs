//! Chat request validation and OpenAI response construction.
//!
//! Lowering a chat request into a generation request belongs to
//! `serving::InputProcessor::preprocess_chat_request`. This module checks the
//! served model name and `prompt_logprobs` (`validate_request_compat`) and
//! turns the resulting `RequestOutput` stream into a buffered
//! `ChatCompletionResponse` or into streamed chunks and SSE events.

mod context;
mod response;
mod validate;

pub use context::ChatResponseContext;
pub use response::{
    chat_completion_chunk_stream, chat_completion_sse_stream, collect_chat_completion,
};
pub use validate::validate_request_compat;
