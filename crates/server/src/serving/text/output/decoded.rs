//! Incremental conversion from engine events to decoded text events.

use std::sync::Arc;

use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer, IncrementalDecoder};
use asynk_strim_attr::{TryYielder, try_stream};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, trace};
use uniserve_core::{EngineCoreOutput, PositionLogprobs};
use uniserve_engine::EventRx;

use super::finish::{FinishReason, StopReason};
use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::serving::text::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    /// Whether tokenizer special tokens are omitted from decoded text.
    pub skip_special_tokens: bool,
    /// Whether the matched stop string remains in emitted output.
    pub include_stop_str_in_output: bool,
    /// Decoded strings that terminate generation.
    pub stop_strings: Option<Vec<String>>,
    /// Minimum number of generated tokens before stop strings can terminate output.
    pub min_tokens: u32,
}

impl Default for TextDecodeOptions {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            stop_strings: None,
            min_tokens: 0,
        }
    }
}

/// Terminal metadata carried on the final [`DecodedTextEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct Finished {
    /// Number of prompt tokens submitted to the engine.
    pub prompt_token_count: usize,
    /// Total number of generated tokens.
    pub output_token_count: usize,
    /// Number of generated tokens consumed by internal protocol sections.
    pub internal_token_count: usize,
    /// Terminal condition and optional concrete stop cause.
    pub finish_reason: FinishReason,
}

/// Internal decoded-text event emitted before higher-level assistant adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    /// The request has been scheduled and prompt metadata is available.
    Start {
        /// Prompt token identifiers submitted to the engine.
        prompt_token_ids: Arc<[u32]>,
        /// Per-position prompt log probabilities, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        /// Monotonic timestamp at which the request entered the serving queue.
        queued_at: Option<f64>,
        /// Monotonic timestamp at which engine execution began.
        scheduled_at: Option<f64>,
    },
    /// Newly decoded text and token metadata.
    TextDelta {
        /// Newly visible decoded text.
        delta: String,
        /// Token identifiers represented by this update.
        token_ids: Vec<u32>,
        /// Per-position candidate log probabilities, when requested.
        logprobs: Option<DecodedLogprobs>,
        /// Terminal metadata when this is the final update.
        finished: Option<Finished>,
    },
}

struct ArDecode {
    delta: String,
    stop: Option<(String, usize)>,
}

struct DecodeState<'a> {
    decoder: IncrementalDecoder<'a>,
    options: TextDecodeOptions,
    prompt_positions: Vec<PositionLogprobs>,
    queued_at: Option<f64>,
    scheduled_at: Option<f64>,
    started: bool,
    pending_token: Option<u32>,
    output_token_count: usize,
    accumulated_token_ids: Vec<u32>,
    accumulated_logprobs: Option<DecodedLogprobs>,
}

impl DecodeState<'_> {
    /// Emits start metadata once scheduling and any requested prompt scores are complete.
    async fn emit_start_if_ready(
        &mut self,
        request_id: &str,
        tokenizer: &HuggingFaceTokenizer,
        prompt_token_ids: &[u32],
        prompt_logprobs_requested: bool,
        expected_prompt_positions: usize,
        y: &mut TryYielder<DecodedTextEvent, Error>,
    ) -> Result<(), Error> {
        let prompt_ready =
            !prompt_logprobs_requested || self.prompt_positions.len() == expected_prompt_positions;
        if self.started || !prompt_ready || self.scheduled_at.is_none() {
            return Ok(());
        }
        let prompt_logprobs = prompt_logprobs_requested
            .then(|| {
                decode_prompt_logprobs(
                    request_id,
                    tokenizer,
                    prompt_token_ids,
                    &self.prompt_positions,
                    self.options.skip_special_tokens,
                )
            })
            .transpose()?;
        y.yield_ok(DecodedTextEvent::Start {
            prompt_token_ids: Arc::from(prompt_token_ids),
            prompt_logprobs,
            queued_at: self.queued_at,
            scheduled_at: self.scheduled_at,
        })
        .await;
        self.started = true;
        Ok(())
    }

    /// Decodes one committed token, applies stop-string holdback, and emits terminal metadata.
    #[allow(clippy::too_many_arguments)]
    async fn consume_token(
        &mut self,
        tokenizer: &HuggingFaceTokenizer,
        prompt_token_count: usize,
        token_id: u32,
        positions: Vec<PositionLogprobs>,
        intermediate: bool,
        raw_stream: &mut EventRx,
        y: &mut TryYielder<DecodedTextEvent, Error>,
    ) -> Result<bool, Error> {
        let decoded_logprobs = (!positions.is_empty())
            .then(|| decode_logprobs(tokenizer, &positions, self.options.skip_special_tokens))
            .transpose()?;
        let decoded = decode_one_token(
            &mut self.decoder,
            token_id,
            self.output_token_count,
            &mut self.options,
            intermediate,
        )?;
        self.output_token_count = self.output_token_count.saturating_add(1);
        if !intermediate {
            self.accumulated_token_ids.push(token_id);
            if let Some(logprobs) = decoded_logprobs.as_ref() {
                self.accumulated_logprobs
                    .get_or_insert_with(|| DecodedLogprobs {
                        positions: Vec::new(),
                    })
                    .positions
                    .extend_from_slice(&logprobs.positions);
            }
        }
        if let Some((stop_string, offset)) = decoded.stop {
            raw_stream.cancel_at_consumed_prefix(
                crate::engine_client::StreamCancelCause::StopStringMatched,
            );
            let truncate_to = Some(if self.options.include_stop_str_in_output {
                offset + stop_string.len()
            } else {
                offset
            });
            let (last_chunk, text) = self.decoder.flush(truncate_to)?;
            let (delta, token_ids, logprobs) = if intermediate {
                (
                    last_chunk.unwrap_or_default(),
                    vec![token_id],
                    decoded_logprobs,
                )
            } else {
                (
                    text,
                    std::mem::take(&mut self.accumulated_token_ids),
                    self.accumulated_logprobs.take(),
                )
            };
            y.yield_ok(DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished: Some(Finished {
                    prompt_token_count,
                    output_token_count: self.output_token_count,
                    internal_token_count: 0,
                    finish_reason: FinishReason::with_stop_reason(
                        uniserve_core::FinishReason::Stop,
                        Some(StopReason::String(stop_string)),
                    ),
                }),
            })
            .await;
            return Ok(true);
        }
        raw_stream.acknowledge_consumed_prefix();
        if intermediate {
            y.yield_ok(DecodedTextEvent::TextDelta {
                delta: decoded.delta,
                token_ids: vec![token_id],
                logprobs: decoded_logprobs,
                finished: None,
            })
            .await;
        }
        Ok(false)
    }
}

/// Advances incremental decoding and returns only bytes newly made visible by this token.
fn decode_one_token(
    decoder: &mut IncrementalDecoder<'_>,
    token_id: u32,
    output_token_count: usize,
    options: &mut TextDecodeOptions,
    intermediate: bool,
) -> Result<ArDecode, Error> {
    let new_bytes = decoder.push_token(token_id)?;
    let stop = if output_token_count + 1 > options.min_tokens as usize {
        if let Some(stops) = options.stop_strings.as_mut()
            && let Some((index, offset)) = matches_stop_string(stops, decoder.output(), new_bytes)
        {
            Some((stops.swap_remove(index), offset))
        } else {
            None
        }
    } else {
        None
    };
    let delta = if intermediate && stop.is_none() {
        decoder.next_chunk().unwrap_or_default()
    } else {
        String::new()
    };
    Ok(ArDecode { delta, stop })
}

/// Decodes one canonical generation event stream into text-runtime events.
#[allow(clippy::too_many_arguments)]
#[try_stream]
pub async fn decoded_text_event_stream(
    request_id: String,
    tokenizer: DynTokenizer,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    mut raw_stream: EventRx,
    decode_options: TextDecodeOptions,
    intermediate: bool,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> crate::serving::text::Result<()> {
    // The prompt seeds incremental decoding and supplies the expected prompt-
    // logprob cardinality before any generated token can be accepted.
    if prompt_token_ids.is_empty() {
        return Err(Error::EmptyPromptTokenIds {
            request_id: request_id.clone(),
        });
    }
    let prompt_token_count = prompt_token_ids.len();
    let decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        stop_string_holdback_bytes(&decode_options),
    );
    let expected_prompt_positions = prompt_token_count.saturating_sub(1);
    let mut state = DecodeState {
        decoder,
        options: decode_options,
        prompt_positions: Vec::new(),
        queued_at: None,
        scheduled_at: None,
        started: false,
        pending_token: None,
        output_token_count: 0,
        accumulated_token_ids: Vec::new(),
        accumulated_logprobs: None,
    };

    // Engine events form an ordered protocol: scheduling and optional prompt
    // metadata establish Start, then each token is paired with optional logprobs.
    while let Some(event) = raw_stream.next().await {
        match event {
            // Scheduling metadata may arrive before prompt logprobs; start is
            // emitted only after both prerequisites are complete.
            EngineCoreOutput::Scheduled {
                queued_at: queued,
                scheduled_at: scheduled,
            } => {
                state.queued_at = Some(queued);
                state.scheduled_at = Some(scheduled);
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        expected_prompt_positions,
                        &mut y,
                    )
                    .await?;
            }
            EngineCoreOutput::PromptLogprobs { positions } => {
                if !prompt_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned unrequested prompt logprobs".to_string(),
                    });
                }
                state.prompt_positions.extend(positions);
                if state.prompt_positions.len() > expected_prompt_positions {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned too many prompt logprob positions".to_string(),
                    });
                }
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        expected_prompt_positions,
                        &mut y,
                    )
                    .await?;
            }
            // Generated logprobs, when requested, defer decoding until the
            // matching TokenLogprobs event validates the token identity.
            EngineCoreOutput::TextToken { id, .. } => {
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        expected_prompt_positions,
                        &mut y,
                    )
                    .await?;
                if !state.started {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "generation began before prompt metadata was complete".to_string(),
                    });
                }
                if state.pending_token.is_some() {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine emitted a new token before resolving prior logprobs"
                            .to_string(),
                    });
                }
                if generated_logprobs_requested {
                    state.pending_token = Some(id);
                } else if state
                    .consume_token(
                        tokenizer.as_ref(),
                        prompt_token_count,
                        id,
                        Vec::new(),
                        intermediate,
                        &mut raw_stream,
                        &mut y,
                    )
                    .await?
                {
                    return Ok(());
                }
            }
            EngineCoreOutput::TokenLogprobs { id, candidates } => {
                let pending = state
                    .pending_token
                    .take()
                    .ok_or_else(|| Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned token logprobs without a pending token"
                            .to_string(),
                    })?;
                if pending != id || candidates.first().is_none_or(|entry| entry.token_id != id) {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "token logprobs do not match the emitted token".to_string(),
                    });
                }
                if state
                    .consume_token(
                        tokenizer.as_ref(),
                        prompt_token_count,
                        id,
                        vec![PositionLogprobs {
                            entries: candidates,
                        }],
                        intermediate,
                        &mut raw_stream,
                        &mut y,
                    )
                    .await?
                {
                    return Ok(());
                }
            }
            // A terminal event flushes decoder holdback only after every pending
            // token/logprob pair and all prompt metadata have resolved.
            EngineCoreOutput::Finished {
                reason,
                stop_reason,
                completion_tokens,
                ..
            } => {
                state
                    .emit_start_if_ready(
                        &request_id,
                        tokenizer.as_ref(),
                        &prompt_token_ids,
                        prompt_logprobs_requested,
                        expected_prompt_positions,
                        &mut y,
                    )
                    .await?;
                if !state.started || state.pending_token.is_some() {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "terminal event arrived before output metadata was complete"
                            .to_string(),
                    });
                }
                let finish_reason = FinishReason::with_stop_reason(reason, stop_reason);

                let (last_chunk, text) = state.decoder.flush(None)?;
                let full_text = tracing::enabled!(Level::TRACE).then(|| text.clone());
                let (delta, token_ids, logprobs) = if intermediate {
                    (last_chunk.unwrap_or_default(), Vec::new(), None)
                } else {
                    (
                        text,
                        std::mem::take(&mut state.accumulated_token_ids),
                        state.accumulated_logprobs.take(),
                    )
                };

                debug!(
                    ?finish_reason,
                    output_token_count = state.output_token_count,
                    "request finished with decoded text"
                );
                if let Some(full_text) = full_text {
                    trace!(full_text, "terminal decoded text");
                }

                y.yield_ok(DecodedTextEvent::TextDelta {
                    delta,
                    token_ids,
                    logprobs,
                    finished: Some(Finished {
                        prompt_token_count,
                        output_token_count: completion_tokens,
                        internal_token_count: completion_tokens
                            .saturating_sub(state.output_token_count),
                        finish_reason,
                    }),
                })
                .await;
                return Ok(());
            }
            EngineCoreOutput::Rejected { message } => {
                return Err(Error::Rejected {
                    request_id,
                    message,
                });
            }
            EngineCoreOutput::Error { message }
            | EngineCoreOutput::ArtifactUnavailable { message } => {
                return Err(Error::MalformedOutput {
                    request_id: request_id.clone(),
                    message,
                });
            }
            // Media lifecycle events cannot be represented by a text-only stream.
            EngineCoreOutput::ImageBegin { .. }
            | EngineCoreOutput::ImageStep { .. }
            | EngineCoreOutput::ImageCommit { .. }
            | EngineCoreOutput::ImageDone { .. }
            | EngineCoreOutput::Artifact(_)
            | EngineCoreOutput::MediaProgress { .. } => {
                return Err(Error::MalformedOutput {
                    request_id: request_id.clone(),
                    message: "text-only request received a non-text lifecycle event".to_string(),
                });
            }
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

/// Returns the suffix byte count retained to detect cross-chunk stop strings.
pub(crate) fn stop_string_holdback_bytes(options: &TextDecodeOptions) -> usize {
    if options.include_stop_str_in_output {
        return 0;
    }
    options
        .stop_strings
        .as_ref()
        .and_then(|stops| stops.iter().map(String::len).max())
        .unwrap_or(1)
        .saturating_sub(1)
}

/// Returns the earliest configured stop string ending at the current suffix.
pub(crate) fn matches_stop_string(
    stops: &[String],
    output: &str,
    new_bytes: usize,
) -> Option<(usize, usize)> {
    let output = output.as_bytes();
    let next_offset = (output.len() + 1).saturating_sub(new_bytes);
    stops
        .iter()
        .map(|stop| {
            (
                stop.as_bytes(),
                stop.len(),
                next_offset.saturating_sub(stop.len()),
            )
        })
        .enumerate()
        .find_map(|(index, (stop, len, start))| {
            output[start..]
                .windows(len)
                .rposition(|window| window == stop)
                .map(|position| (index, start + position))
        })
}
