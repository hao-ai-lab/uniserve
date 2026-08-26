use std::pin::Pin;
use std::sync::Arc;

use crate::serving::text::output::{DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent};
use futures::Stream;
use subenum::subenum;
use trait_set::trait_set;
use uuid::Uuid;

use crate::serving::chat::output::FinishReason;
use crate::serving::chat::output::error::Result;
use crate::serving::chat::output::event::{AssistantBlockKind, ChatEvent};

/// Internal assistant event before final assembly.
///
/// - [`ContentEvent`]: subenum after reasoning parsing, carries only text content.
/// - [`AssistantEvent`]: full event after tool parsing, adds tool-call variants.
#[subenum(ContentEvent)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    #[subenum(ContentEvent)]
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
    },
    #[subenum(ContentEvent)]
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    #[subenum(ContentEvent)]
    LogprobsDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    /// The start of a new tool call, with its declared name and generated ID.
    ToolCallStart { id: String, name: String },
    /// A delta for the arguments of the currently open tool call. Must follow a
    /// `ToolCallStart`.
    ToolCallArgumentsDelta { delta: String },
    #[subenum(ContentEvent)]
    Done {
        prompt_token_count: usize,
        output_token_count: usize,
        internal_token_count: usize,
        finish_reason: FinishReason,
    },
}

/// Boxed stream of decoded text events coming from [`text`].
pub type DynDecodedTextEventStream = Pin<Box<dyn Stream<Item = Result<DecodedTextEvent>> + Send>>;
/// Boxed stream of structured chat events exposed by the chat runtime.
pub type DynChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>;

trait_set! {
 /// Boxed-stream constraint for decoded text updates.
    pub trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
 /// Boxed-stream constraint for internal assistant events.
    pub trait AssistantEventStream = Stream<Item = Result<AssistantEvent>> + Send + 'static;
 /// Boxed-stream constraint for public chat events.
    pub trait ChatEventStream = Stream<Item = Result<ChatEvent>> + Send + 'static;
}

/// Generate the northbound tool-call ID using the OpenAI-style `call_<id>`
/// format.
pub(crate) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
