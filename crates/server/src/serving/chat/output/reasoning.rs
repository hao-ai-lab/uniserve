//! Converts decoded text into reasoning-aware assistant deltas.
//!
//! This is the first chat output stage. It maps decoded-text events onto
//! [`AssistantEvent`]s: text becomes `Reasoning` or `Text` deltas, token
//! metadata becomes `SampleDelta`, and the terminal update becomes `Done`.
//! Without a parser, all text is forwarded as `Text`.

use crate::serving::text::output::DecodedTextEvent;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};

use crate::profile::reasoning::{ReasoningDelta, ReasoningParser};
use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::processor::AssistantEvent;
use crate::serving::chat::{Error, Result};

/// Reasoning-stage state; a `None` parser disables reasoning parsing.
struct ReasoningState<P> {
    parser: Option<P>,
}

impl<P: ReasoningParser> ReasoningState<P> {
    /// Creates the stage state around an optional parser.
    fn new(parser: Option<P>) -> Self {
        Self { parser }
    }

    /// Processes one text delta through the reasoning parser.
    fn process_delta(&mut self, delta: String) -> Vec<AssistantEvent> {
        let Some(parser) = self.parser.as_mut() else {
            return vec![AssistantEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            }];
        };

        let mut events = Vec::new();
        push_reasoning_delta(&mut events, parser.push(&delta));
        events
    }

    /// Initializes reasoning state from the prompt token sequence.
    ///
    /// A reasoning delimiter token in the prompt suffix after the last other
    /// special token decides whether generation starts inside a reasoning
    /// section; without one, generation starts outside it.
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        let Some(parser) = self.parser.as_mut() else {
            return;
        };
        parser.initialize(prompt_token_ids);
    }

    /// Flushes text the parser held back as a possible partial delimiter.
    fn finish(&mut self) -> Vec<AssistantEvent> {
        let Some(parser) = self.parser.as_mut() else {
            return Vec::new();
        };
        let mut events = Vec::new();
        push_reasoning_delta(&mut events, parser.finish());
        events
    }
}

/// Appends a nonempty semantic text delta.
fn push_text_delta(events: &mut Vec<AssistantEvent>, kind: AssistantBlockKind, delta: String) {
    if !delta.is_empty() {
        events.push(AssistantEvent::TextDelta { kind, delta });
    }
}

/// Appends the non-empty reasoning and visible parts of `delta`, reasoning
/// first.
fn push_reasoning_delta(events: &mut Vec<AssistantEvent>, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        push_text_delta(events, AssistantBlockKind::Reasoning, reasoning);
    }
    if let Some(content) = delta.content {
        push_text_delta(events, AssistantBlockKind::Text, content);
    }
}

#[try_stream]
/// Converts decoded text events into a reasoning-aware event stream.
///
/// One decoded delta may span several delimiters, as when a block-diffusion
/// model commits a whole block of tokens at once; its reasoning text is
/// emitted before its visible text.
pub(super) async fn reasoning_event_stream(
    decoded_stream: impl futures::Stream<Item = crate::serving::text::Result<DecodedTextEvent>> + Send,
    parser: Option<impl ReasoningParser>,
    mut y: TryYielder<AssistantEvent, Error>,
) -> Result<()> {
    pin_mut!(decoded_stream);
    let mut state = ReasoningState::new(parser);

    while let Some(event) = decoded_stream.next().await.transpose()? {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                state.initialize(&prompt_token_ids);
                y.yield_ok(AssistantEvent::Start {
                    prompt_token_ids,
                    prompt_logprobs,
                    queued_at,
                    scheduled_at,
                })
                .await;
            }
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                // Parsed text precedes the token metadata of the same update,
                // and any flushed text precedes `Done`.
                for next in state.process_delta(delta) {
                    y.yield_ok(next).await;
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    y.yield_ok(AssistantEvent::SampleDelta {
                        logprobs,
                        token_ids,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    for next in state.finish() {
                        y.yield_ok(next).await;
                    }
                    y.yield_ok(AssistantEvent::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        internal_token_count: finished.internal_token_count,
                        finish_reason: finished.finish_reason,
                    })
                    .await;
                }
            }
        }
    }
    Ok(())
}
