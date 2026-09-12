//! Converts Qwen3 assistant text into tool-call-aware updates.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::warn;

use crate::profile::tools::{Qwen3XmlToolParser, ToolCallDelta, ToolParserOutput};
use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::processor::AssistantEvent;
use crate::serving::chat::output::processor::generate_tool_call_id;
use crate::serving::chat::{Error, Result};

struct ToolState {
    parser: Qwen3XmlToolParser,
    parser_failed: bool,
    open_call_index: Option<usize>,
}

impl ToolState {
    /// Creates a Qwen tool-call stream parser.
    fn new(parser: Qwen3XmlToolParser) -> Self {
        Self {
            parser,
            parser_failed: false,
            open_call_index: None,
        }
    }

    /// Feeds visible text through incremental tool parsing with lossless text fallback.
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
                warn!(error = %error.as_report(), "Qwen3 tool parsing failed");
                self.parser_failed = true;
                self.process_parser_output(kind, output, &mut events)?;
                self.open_call_index = None;
                push_text_delta(&mut events, kind, self.parser.reset());
            }
        }
        Ok(events)
    }

    /// Processes the parser output.
    fn process_parser_output(
        &mut self,
        kind: AssistantBlockKind,
        output: ToolParserOutput,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        if self.open_call_index.is_none() {
            push_text_delta(events, kind, output.normal_text);
            self.process_tool_items(output.calls, events)?;
        } else {
            self.process_tool_items(output.calls, events)?;
            if !output.normal_text.is_empty() {
                self.open_call_index = None;
                push_text_delta(events, kind, output.normal_text);
            }
        }
        Ok(())
    }

    /// Converts parser deltas into ordered tool-call start and argument events.
    fn process_tool_items(
        &mut self,
        items: Vec<ToolCallDelta>,
        events: &mut Vec<AssistantEvent>,
    ) -> Result<()> {
        for item in items {
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
                continue;
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
        }
        Ok(())
    }

    /// Finishes incremental output processing.
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
                warn!(error = %error.as_report(), "Qwen3 tool parser finalization failed");
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
pub async fn tool_event_stream(
    stream: impl futures::Stream<Item = Result<AssistantEvent>> + Send,
    parser: Option<Qwen3XmlToolParser>,
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
