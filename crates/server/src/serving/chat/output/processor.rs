//! Assistant event contract shared by the chat output-processing stages.

use std::sync::Arc;

use uuid::Uuid;

use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::FinishReason;
use crate::serving::text::output::{DecodedLogprobs, DecodedPromptLogprobs};

#[derive(Debug, Clone, PartialEq)]
/// Incremental assistant event after reasoning and tool parsing.
///
/// A well-formed stream starts with one `Start` and ends with `Done`;
/// `assemble_chat_event_stream` rejects output before `Start`, a second
/// `Start`, and a stream that closes before `Done`. Parsed text and sampled
/// token metadata travel in separate events; because the decoder and the
/// parsers hold back incomplete input, a text delta need not match the tokens
/// of any single `SampleDelta`.
pub enum AssistantEvent {
    /// Prompt metadata, published once before any output.
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        /// Engine queue-entry Unix timestamp in seconds.
        queued_at: Option<f64>,
        /// Engine admission Unix timestamp in seconds.
        scheduled_at: Option<f64>,
    },
    /// Parsed text of kind `Text` or `Reasoning`; never `ToolCall`.
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Token IDs and logprobs of newly sampled tokens, without text.
    SampleDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    /// Opens a tool call; subsequent argument deltas belong to it.
    ToolCallStart {
        /// Tool-call identifier from `generate_tool_call_id`.
        id: String,
        /// Function name selected by the model.
        name: String,
    },
    /// Argument text for the most recently started tool call.
    ToolCallArgumentsDelta {
        /// Next chunk of the JSON arguments text.
        delta: String,
    },
    /// Terminal token accounting and finish reason.
    Done {
        prompt_token_count: usize,
        output_token_count: usize,
        internal_token_count: usize,
        finish_reason: FinishReason,
    },
}

/// Generates an OpenAI-compatible tool-call identifier: `call_` followed by
/// 24 hex digits of a random UUID.
pub(crate) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
