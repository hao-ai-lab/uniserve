//! Adapts Qwen3 decoded text updates into reasoning-aware assistant deltas.

use crate::serving::text::output::DecodedTextEvent;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::warn;

use super::ContentEvent;
use crate::serving::chat::output::Result;
use crate::serving::chat::output::error::Error;
use crate::serving::chat::output::event::AssistantBlockKind;
use crate::serving::chat::output::parser::reasoning::{Qwen3ReasoningParser, ReasoningDelta};
use crate::serving::chat::output::processor::DecodedTextEventStream;

struct ReasoningState {
    parser: Option<Qwen3ReasoningParser>,
    parser_failed: bool,
}

impl ReasoningState {
    fn new(parser: Option<Qwen3ReasoningParser>) -> Self {
        Self {
            parser,
            parser_failed: false,
        }
    }

    fn process_delta(&mut self, delta: String) -> Vec<ContentEvent> {
        let Some(parser) = self.parser.as_mut().filter(|_| !self.parser_failed) else {
            return vec![ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            }];
        };

        let mut events = Vec::new();
        match parser.push(&delta) {
            Ok(result) => push_reasoning_delta(&mut events, result),
            Err(error) => {
                warn!(error = %error.as_report(), "Qwen3 reasoning parsing failed");
                self.parser_failed = true;
                push_text_delta(&mut events, AssistantBlockKind::Text, delta);
            }
        }
        events
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) {
        let Some(parser) = self.parser.as_mut().filter(|_| !self.parser_failed) else {
            return;
        };
        if let Err(error) = parser.initialize(prompt_token_ids) {
            warn!(error = %error.as_report(), "Qwen3 reasoning parser initialization failed");
            self.parser_failed = true;
        }
    }

    fn finish(&mut self) -> Vec<ContentEvent> {
        let Some(parser) = self.parser.as_mut().filter(|_| !self.parser_failed) else {
            return Vec::new();
        };
        match parser.finish() {
            Ok(result) => {
                let mut events = Vec::new();
                push_reasoning_delta(&mut events, result);
                events
            }
            Err(error) => {
                warn!(error = %error.as_report(), "Qwen3 reasoning parser finalization failed");
                Vec::new()
            }
        }
    }
}

fn push_text_delta(events: &mut Vec<ContentEvent>, kind: AssistantBlockKind, delta: String) {
    if !delta.is_empty() {
        events.push(ContentEvent::TextDelta { kind, delta });
    }
}

fn push_reasoning_delta(events: &mut Vec<ContentEvent>, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        push_text_delta(events, AssistantBlockKind::Reasoning, reasoning);
    }
    if let Some(content) = delta.content {
        push_text_delta(events, AssistantBlockKind::Text, content);
    }
}

#[try_stream]
pub async fn reasoning_event_stream(
    decoded_stream: impl DecodedTextEventStream,
    parser: Option<Qwen3ReasoningParser>,
    mut y: TryYielder<ContentEvent, Error>,
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
                y.yield_ok(ContentEvent::Start {
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
                ..
            } => {
                for next in state.process_delta(delta) {
                    y.yield_ok(next).await;
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    y.yield_ok(ContentEvent::LogprobsDelta {
                        logprobs,
                        token_ids,
                    })
                    .await;
                }
                if let Some(finished) = finished {
                    for next in state.finish() {
                        y.yield_ok(next).await;
                    }
                    y.yield_ok(ContentEvent::Done {
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
