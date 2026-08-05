//! Adapts Qwen3 decoded text updates into reasoning-aware assistant deltas.

use crate::text::output::DecodedTextEvent;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::warn;

use super::ContentEvent;
use crate::chat::output::Result;
use crate::chat::output::error::Error;
use crate::chat::output::event::AssistantBlockKind;
use crate::chat::output::parser::reasoning::{Qwen3ReasoningParser, ReasoningDelta};
use crate::chat::output::processor::DecodedTextEventStream;

struct ReasoningState {
    parser: Qwen3ReasoningParser,
    parser_failed: bool,
}

impl ReasoningState {
    fn new(parser: Qwen3ReasoningParser) -> Self {
        Self {
            parser,
            parser_failed: false,
        }
    }

    fn process_delta(&mut self, delta: String) -> Vec<ContentEvent> {
        if self.parser_failed {
            return vec![ContentEvent::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            }];
        }

        let mut events = Vec::new();
        match self.parser.push(&delta) {
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
        if self.parser_failed {
            return;
        }
        if let Err(error) = self.parser.initialize(prompt_token_ids) {
            warn!(error = %error.as_report(), "Qwen3 reasoning parser initialization failed");
            self.parser_failed = true;
        }
    }

    fn finish(&mut self) -> Vec<ContentEvent> {
        if self.parser_failed {
            return Vec::new();
        }
        match self.parser.finish() {
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
    parser: Qwen3ReasoningParser,
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
                        kv_transfer_params: finished.kv_transfer_params,
                    })
                    .await;
                }
            }
        }
    }
    Ok(())
}
