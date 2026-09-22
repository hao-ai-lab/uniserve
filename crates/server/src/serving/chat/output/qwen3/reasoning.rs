//! Converts decoded Qwen3 text into reasoning-aware assistant deltas.

use crate::serving::text::output::DecodedTextEvent;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};

use crate::profile::reasoning::{Qwen3ReasoningParser, ReasoningDelta};
use crate::serving::chat::AssistantBlockKind;
use crate::serving::chat::output::processor::AssistantEvent;
use crate::serving::chat::{Error, Result};

struct ReasoningState {
    parser: Option<Qwen3ReasoningParser>,
}

impl ReasoningState {
    /// Creates a Qwen reasoning-stream parser.
    fn new(parser: Option<Qwen3ReasoningParser>) -> Self {
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
    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        let Some(parser) = self.parser.as_mut() else {
            return;
        };
        parser.initialize(prompt_token_ids);
    }

    /// Finishes incremental output processing.
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

/// Pushes the reasoning delta.
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
pub async fn reasoning_event_stream(
    decoded_stream: impl futures::Stream<Item = crate::serving::text::Result<DecodedTextEvent>> + Send,
    parser: Option<Qwen3ReasoningParser>,
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
