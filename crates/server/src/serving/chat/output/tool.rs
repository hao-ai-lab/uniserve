//! Converts assistant text into tool-call-aware updates.
//!
//! This is the second chat output stage. Visible `Text` deltas pass through
//! the model's `ToolParser`; every other event is forwarded unchanged. A
//! parser error does not fail the request: the stage logs it and forwards all
//! later text unparsed. After a parse error, the input the parser has not
//! represented in its output, such as a whole call whose header failed, is
//! re-emitted as visible text. Finalization fails only while a published call
//! is open; that call keeps the arguments already streamed.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::warn;

use crate::profile::tools::{ToolCallDelta, ToolParser, ToolParserItem, ToolParserOutput};
use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::processor::AssistantEvent;
use crate::serving::chat::output::processor::generate_tool_call_id;
use crate::serving::chat::{Error, Result};

/// Tool-stage state for one request.
struct ToolState<P> {
    parser: P,
    /// Set after a parse or finalization error; later text bypasses the parser.
    parser_failed: bool,
    /// Parser-local index of the tool call that later argument deltas extend.
    /// Cleared when visible text or a non-`Text` delta follows the call, or
    /// when the parser fails.
    open_call_index: Option<usize>,
}

impl<P: ToolParser> ToolState<P> {
    /// Creates the stage state around the request's tool parser.
    fn new(parser: P) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_call_index: None,
        }
    }

    /// Feeds visible text through incremental tool parsing with a visible-text
    /// fallback.
    ///
    /// On a parser error, the events parsed before the error are still
    /// emitted, followed by `ToolParser::reset`'s unrepresented input as
    /// visible text: for example the whole attempted call when its header
    /// failed, or the input after the arguments of an already published call.
    fn process_text_delta(
        &mut self,
        kind: AssistantBlockKind,
        delta: String,
    ) -> Result<Vec<AssistantEvent>> {
        let mut events = Vec::new();
        if kind != AssistantBlockKind::Text || self.parser_failed {
            self.open_call_index = None;
            events.push(AssistantEvent::TextDelta { kind, delta });
            return Ok(events);
        }

        let mut output = ToolParserOutput::default();
        match self.parser.parse_into(&delta, &mut output) {
            Ok(()) => self.process_parser_output(kind, output, &mut events)?,
            Err(error) => {
                warn!(error = %error.as_report(), "tool-call parsing failed");
                self.parser_failed = true;
                self.process_parser_output(kind, output, &mut events)?;
                self.open_call_index = None;
                push_text_delta(&mut events, kind, self.parser.reset());
            }
        }
        Ok(events)
    }

    /// Emits one parser output's items in their input order.
    ///
    /// The parser resumes text only after a call's end delimiter, so a text
    /// item closes the open call.
    fn process_parser_output(
        &mut self,
        kind: AssistantBlockKind,
        output: ToolParserOutput,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        for item in output.items {
            match item {
                ToolParserItem::Text(text) => {
                    self.open_call_index = None;
                    push_text_delta(events, kind, text);
                }
                ToolParserItem::Call(call) => self.process_tool_item(call, events)?,
            }
        }
        Ok(())
    }

    /// Converts one parser delta into tool-call start and argument events.
    fn process_tool_item(
        &mut self,
        item: ToolCallDelta,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        if let Some(name) = item.name
            && self.open_call_index != Some(item.tool_index)
        {
            self.open_call_index = Some(item.tool_index);
            events.push(AssistantEvent::ToolCallStart {
                id: generate_tool_call_id(),
                name,
            });
        }

        if item.arguments.is_empty() {
            return Ok(());
        }
        let Some(open_call_index) = self.open_call_index else {
            return Err(Error::ToolCallStreamInvariant {
                message: format!(
                    "received arguments for tool index {} before any tool-call start",
                    item.tool_index
                ),
            });
        };
        if open_call_index != item.tool_index {
            return Err(Error::ToolCallStreamInvariant {
                message: format!(
                    "received arguments for tool index {} while tool index {} is open",
                    item.tool_index, open_call_index
                ),
            });
        }
        events.push(AssistantEvent::ToolCallArgumentsDelta {
            delta: item.arguments,
        });
        Ok(())
    }

    /// Flushes the parser at end of generation.
    ///
    /// Buffered text, including an unfinished call with its start marker, is
    /// emitted as visible text. The parser fails to finalize only while a
    /// published call is still open; that error is logged and produces no
    /// events, because the call's arguments have already been streamed and
    /// only wrapper text remains buffered.
    fn finish(&mut self) -> Result<Vec<AssistantEvent>> {
        if self.parser_failed {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        match self.parser.finish() {
            Ok(output) => {
                self.process_parser_output(AssistantBlockKind::Text, output, &mut events)?
            }
            Err(error) => {
                warn!(error = %error.as_report(), "tool-call parser finalization failed");
                self.parser_failed = true;
            }
        }
        Ok(events)
    }
}

/// Appends a nonempty semantic text delta.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if !delta.is_empty() {
        events.push(AssistantEvent::TextDelta { kind, delta });
    }
}

#[try_stream]
/// Converts assistant text events into tool-call-aware events.
///
/// Without a parser the input stream is forwarded unchanged. Errors are
/// `Error::ToolCallStreamInvariant` from inconsistent parser output or errors
/// propagated from the upstream stage.
pub(super) async fn tool_event_stream(
    stream: impl futures::Stream<Item = Result<AssistantEvent>> + Send,
    parser: Option<impl ToolParser>,
    mut y: TryYielder<AssistantEvent, Error>,
) -> Result<()> {
    let Some(parser) = parser else {
        pin_mut!(stream);
        while let Some(event) = stream.next().await.transpose()? {
            y.yield_ok(event).await;
        }
        return Ok(());
    };

    pin_mut!(stream);
    let mut state = ToolState::new(parser);
    while let Some(event) = stream.next().await.transpose()? {
        match event {
            event @ (AssistantEvent::ToolCallStart { .. }
            | AssistantEvent::ToolCallArgumentsDelta { .. }) => y.yield_ok(event).await,
            AssistantEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                y.yield_ok(AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                    queued_at,
                    scheduled_at,
                })
                .await;
            }
            AssistantEvent::TextDelta { kind, delta } => {
                for next in state.process_text_delta(kind, delta)? {
                    y.yield_ok(next).await;
                }
            }
            AssistantEvent::SampleDelta {
                logprobs,
                token_ids,
            } => {
                y.yield_ok(AssistantEvent::SampleDelta {
                    logprobs,
                    token_ids,
                })
                .await;
            }
            AssistantEvent::Done {
                prompt_token_count,
                output_token_count,
                internal_token_count,
                finish_reason,
            } => {
                for next in state.finish()? {
                    y.yield_ok(next).await;
                }
                y.yield_ok(AssistantEvent::Done {
                    prompt_token_count,
                    output_token_count,
                    internal_token_count,
                    finish_reason,
                })
                .await;
            }
        }
    }
    Ok(())
}
