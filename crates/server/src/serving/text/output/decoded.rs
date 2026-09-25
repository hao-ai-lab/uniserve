//! Incremental conversion from engine events to decoded text events.
//!
//! [`decoded_text_event_stream`] consumes one request's `EventRx`, checks the
//! engine event protocol, detokenizes generated tokens with the tokenizer's
//! `IncrementalDecoder`, decodes requested logprobs, and matches stop strings.
//! `serving::assembly` feeds its output to the chat output processor.
//!
//! The decoder also drives the engine's output acknowledgement. When the
//! request has stop strings, `EventRx` does not acknowledge tokens on receipt;
//! this module acknowledges each token only after checking that it completes
//! no stop string, and on a match stops generation at the consumed token
//! count with `EventRx::cancel_at_consumed_prefix`.

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
    /// Skips special tokens, excludes a matched stop string from output, and
    /// configures no stop strings or minimum length.
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
    ///
    /// Taken from the engine's `completion_tokens` when the engine finishes
    /// the request and from the decoder's own count when a stop string ends
    /// the stream.
    pub output_token_count: usize,
    /// Number of generated tokens consumed by internal protocol sections.
    ///
    /// Computed as the engine's `completion_tokens` minus the text tokens this
    /// decoder consumed; zero when a stop string ends the stream.
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
        /// Unix timestamp in seconds at which the request entered the engine
        /// queue.
        queued_at: Option<f64>,
        /// Unix timestamp in seconds at which the engine scheduler admitted
        /// the request.
        scheduled_at: Option<f64>,
    },
    /// Newly decoded text and token metadata.
    ///
    /// With intermediate output enabled, each non-terminal update covers one
    /// generated token (its text may be empty while the decoder holds bytes
    /// back). The terminal update carries the flushed remainder of the text
    /// and, when a stop string ended the stream, the matching token. Without
    /// intermediate output, a single terminal update carries the complete
    /// text, token IDs, and logprobs.
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

/// Result of decoding one generated token.
struct ArDecode {
    /// Newly visible text; empty unless intermediate output is enabled and no
    /// stop string matched.
    delta: String,
    /// Matched stop string and the byte offset of its start in
    /// `IncrementalDecoder::output`.
    stop: Option<(String, usize)>,
}

/// Per-request decoding state for [`decoded_text_event_stream`].
struct DecodeState<'a> {
    decoder: IncrementalDecoder<'a>,
    /// Decode options; matched stop strings are removed from `stop_strings`.
    options: TextDecodeOptions,
    /// Prompt logprob positions received so far.
    prompt_positions: Vec<PositionLogprobs>,
    queued_at: Option<f64>,
    scheduled_at: Option<f64>,
    /// Whether `DecodedTextEvent::Start` has been yielded.
    started: bool,
    /// Generated token awaiting its `TokenLogprobs` event when generated
    /// logprobs are requested.
    pending_token: Option<u32>,
    /// Generated text tokens consumed by the decoder.
    output_token_count: usize,
    /// Token IDs and logprobs held for the single terminal update when
    /// intermediate output is disabled.
    accumulated_token_ids: Vec<u32>,
    accumulated_logprobs: Option<DecodedLogprobs>,
}

impl DecodeState<'_> {
    /// Emits start metadata once scheduling and any requested prompt scores are complete.
    ///
    /// Idempotent: does nothing after `Start` has been yielded or while a
    /// prerequisite is missing. The `TextToken` and `Finished` handlers check
    /// `started` afterwards to reject output that arrives too early.
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
    ///
    /// Returns `true` when the token completed a stop string. In that case the
    /// engine has been told to stop at this token, the decoder has been
    /// flushed and truncated at the match, and the terminal `TextDelta` has
    /// been yielded, so the caller must end the stream. Otherwise the token is
    /// acknowledged to the engine and, with intermediate output, yielded as
    /// its own `TextDelta`.
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
            // Stop generation at exactly the tokens received so far, which
            // include this one.
            raw_stream.cancel_at_consumed_prefix(
                crate::engine_client::StreamCancelCause::StopStringMatched,
            );
            let truncate_to = Some(if self.options.include_stop_str_in_output {
                offset + stop_string.len()
            } else {
                offset
            });
            // `last_chunk` is the not-yet-emitted remainder of the truncated
            // text; `text` is the complete truncated output.
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
///
/// Stop strings are checked only from token number `min_tokens + 1` onward
/// (`output_token_count` counts the tokens before this one). A matched stop
/// string is removed from `options.stop_strings`.
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
///
/// On success the stream yields exactly one `DecodedTextEvent::Start`, then
/// `TextDelta` updates, the last of which carries `finished`. `intermediate`
/// selects one update per generated token instead of a single accumulated
/// terminal update. When `generated_logprobs_requested` is set, every
/// `TextToken` must be followed by a `TokenLogprobs` event for the same token
/// whose first candidate is that token.
///
/// # Errors
///
/// Fails with [`Error::EmptyPromptTokenIds`] for an empty prompt,
/// [`Error::Rejected`] when the engine rejects the request,
/// [`Error::MalformedOutput`] when engine events violate the protocol or the
/// engine reports an error or an unavailable artifact,
/// [`Error::StreamClosedBeforeTerminalOutput`] when the channel closes without
/// a terminal event, and a tokenizer error when decoding fails.
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
    // The first prompt token has no left context to score, so prompt logprobs
    // cover every position after it (see `DecodedPromptLogprobs`).
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
                // The event protocol lists the sampled token as the first candidate.
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

                // Release the decoder's holdback untruncated: a stop-string
                // match would have ended the stream in `consume_token`.
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
            EngineCoreOutput::Rejected { kind, message } => {
                return Err(Error::Rejected {
                    request_id,
                    kind,
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
///
/// Holding back one byte less than the longest stop string keeps any partial
/// match out of emitted text until the next token confirms or rules it out.
/// No holdback is needed when matched stop strings stay in the output. The
/// result is passed to the decoder as its `min_bytes_to_buffer`.
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

/// Returns the configured stop string that completes earliest in newly
/// decoded text.
///
/// Only matches that include at least one of the last `new_bytes` bytes of
/// `output` are considered, so earlier text is not rescanned and nothing
/// matches when `new_bytes` is zero. Each stop string is located at its first
/// occurrence in that range, and the one whose match ends earliest wins, with
/// ties going to the earlier entry in `stops`. The result is the one a
/// decoder that received the new bytes one at a time would report, and
/// truncating `output` at the match start removes every stop occurrence
/// (vLLM's `check_stop_strings` selects the same match). Returns the stop
/// string's index in `stops` and the byte offset of the match start in
/// `output`.
///
/// Matching compares bytes. A match of a complete UTF-8 stop string in UTF-8
/// text starts and ends on character boundaries, so the offset is a valid
/// truncation point.
pub(crate) fn matches_stop_string(
    stops: &[String],
    output: &str,
    new_bytes: usize,
) -> Option<(usize, usize)> {
    let output = output.as_bytes();
    // A match must end at or after this offset to include a new byte.
    let next_offset = (output.len() + 1).saturating_sub(new_bytes);
    stops
        .iter()
        .enumerate()
        .filter_map(|(index, stop)| {
            let stop = stop.as_bytes();
            let search_start = next_offset.saturating_sub(stop.len());
            output[search_start..]
                .windows(stop.len())
                .position(|window| window == stop)
                .map(|position| {
                    let match_start = search_start + position;
                    (index, match_start, match_start + stop.len())
                })
        })
        // `min_by_key` keeps the first of equal keys, so ties go to list order.
        .min_by_key(|&(_, _, match_end)| match_end)
        .map(|(index, match_start, _)| (index, match_start))
}

#[cfg(test)]
mod tests {
    use super::matches_stop_string;

    fn stops(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    /// Truncating at the returned offset must remove every occurrence of the
    /// stop string, so a new chunk holding it twice (Qwen3 encodes `"\n\n"`
    /// as one token) matches at its first occurrence.
    #[test]
    fn stop_string_matches_its_first_occurrence_in_new_text() {
        assert_eq!(
            matches_stop_string(&stops(&["\n"]), "Answer: 42\n\n", 2),
            Some((0, 10))
        );
    }

    /// Among stop strings matching in the same new chunk, the one that
    /// completes earliest wins, as if tokens arrived one byte at a time;
    /// list order breaks ties.
    #[test]
    fn stop_string_that_completes_earliest_wins() {
        assert_eq!(
            matches_stop_string(&stops(&["b", "a"]), "xab", 2),
            Some((1, 1))
        );
        assert_eq!(
            matches_stop_string(&stops(&["ab", "b"]), "xab", 2),
            Some((0, 1))
        );
    }
}
