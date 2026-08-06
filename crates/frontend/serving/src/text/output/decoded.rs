use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, trace};
use uniserve_engine_gateway::generation::{
    GenEvent, GenerationEventStream, GenerationFinishReason, GenerationPositionLogprobs,
    PublicCommit,
};
use uniserve_model_profile::tokenizer::{DynTokenizer, IncrementalDecoder};

use super::finish::{FinishReason, StopReason};
use super::logprobs::{
    DecodedLogprobs, DecodedPromptLogprobs, decode_logprobs, decode_prompt_logprobs,
};
use crate::text::error::Error;

/// Request-neutral options for incremental text decoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecodeOptions {
    pub skip_special_tokens: bool,
    pub include_stop_str_in_output: bool,
    pub stop_strings: Option<Vec<String>>,
    pub min_tokens: u32,
}

impl Default for TextDecodeOptions {
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
    pub prompt_token_count: usize,
    pub output_token_count: usize,
    pub internal_token_count: usize,
    pub finish_reason: FinishReason,
}

/// Internal decoded-text event emitted before higher-level assistant adaptation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTextEvent {
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
        queued_at: Option<f64>,
        scheduled_at: Option<f64>,
    },
    TextDelta {
        delta: String,
        token_ids: Vec<u32>,
        logprobs: Option<DecodedLogprobs>,
        public_commit: Option<PublicCommit>,
        finished: Option<Finished>,
    },
}

struct TokenDecode {
    delta: String,
    stop: Option<(String, usize)>,
}

fn decode_one_token(
    decoder: &mut IncrementalDecoder<'_>,
    token_id: u32,
    output_token_count: usize,
    options: &mut TextDecodeOptions,
    intermediate: bool,
) -> Result<TokenDecode, Error> {
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
    Ok(TokenDecode { delta, stop })
}

/// Decode one canonical generation event stream into text-runtime events.
#[try_stream]
pub async fn decoded_text_event_stream(
    request_id: String,
    tokenizer: DynTokenizer,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs_requested: bool,
    generated_logprobs_requested: bool,
    mut raw_stream: GenerationEventStream,
    mut decode_options: TextDecodeOptions,
    intermediate: bool,
    mut y: TryYielder<DecodedTextEvent, Error>,
) -> crate::text::Result<()> {
    if prompt_token_ids.is_empty() {
        return Err(Error::EmptyPromptTokenIds {
            request_id: request_id.clone(),
        });
    }
    let prompt_token_count = prompt_token_ids.len();
    let mut decoder = tokenizer.create_decode_stream(
        &prompt_token_ids,
        decode_options.skip_special_tokens,
        stop_string_holdback_bytes(&decode_options),
    );
    let expected_prompt_positions = prompt_token_count.saturating_sub(1);
    let mut prompt_positions: Vec<GenerationPositionLogprobs> = Vec::new();
    let mut queued_at = None;
    let mut scheduled_at = None;
    let mut started = false;
    let mut pending_token = None;
    let mut last_public_commit = None;
    let mut output_token_count = 0_usize;
    let mut accumulated_token_ids = Vec::new();
    let mut accumulated_logprobs: Option<DecodedLogprobs> = None;

    macro_rules! emit_start_if_ready {
        () => {{
            let prompt_ready =
                !prompt_logprobs_requested || prompt_positions.len() == expected_prompt_positions;
            if !started && prompt_ready && scheduled_at.is_some() {
                let prompt_logprobs = prompt_logprobs_requested
                    .then(|| {
                        decode_prompt_logprobs(
                            &request_id,
                            tokenizer.as_ref(),
                            &prompt_token_ids,
                            &prompt_positions,
                            decode_options.skip_special_tokens,
                        )
                    })
                    .transpose()?;
                y.yield_ok(DecodedTextEvent::Start {
                    prompt_token_ids: Arc::from(prompt_token_ids.clone()),
                    prompt_logprobs,
                    queued_at,
                    scheduled_at,
                })
                .await;
                started = true;
            }
        }};
    }

    macro_rules! consume_token {
        ($token_id:expr, $positions:expr, $public_commit:expr) => {{
            let token_id = $token_id;
            let public_commit: Option<PublicCommit> = $public_commit;
            if public_commit.is_some() {
                last_public_commit = public_commit.clone();
            }
            let positions: Vec<GenerationPositionLogprobs> = $positions;
            let decoded_logprobs = (!positions.is_empty())
                .then(|| {
                    decode_logprobs(
                        tokenizer.as_ref(),
                        &positions,
                        decode_options.skip_special_tokens,
                    )
                })
                .transpose()?;
            let decoded = decode_one_token(
                &mut decoder,
                token_id,
                output_token_count,
                &mut decode_options,
                intermediate,
            )?;
            output_token_count = output_token_count.saturating_add(1);
            if !intermediate {
                accumulated_token_ids.push(token_id);
                if let Some(logprobs) = decoded_logprobs.as_ref() {
                    accumulated_logprobs
                        .get_or_insert_with(|| DecodedLogprobs {
                            positions: Vec::new(),
                        })
                        .positions
                        .extend_from_slice(&logprobs.positions);
                }
            }
            if let Some((stop_string, offset)) = decoded.stop {
                raw_stream.cancel_at_consumed_prefix(
                    uniserve_engine_gateway::transport::StreamCancelCause::StopStringMatched,
                );
                let truncate_to = Some(if decode_options.include_stop_str_in_output {
                    offset + stop_string.len()
                } else {
                    offset
                });
                let (last_chunk, text) = decoder.flush(truncate_to)?;
                let (delta, token_ids, logprobs) = if intermediate {
                    (
                        last_chunk.unwrap_or_default(),
                        vec![token_id],
                        decoded_logprobs,
                    )
                } else {
                    (text, accumulated_token_ids, accumulated_logprobs)
                };
                y.yield_ok(DecodedTextEvent::TextDelta {
                    delta,
                    token_ids,
                    logprobs,
                    public_commit: public_commit.or_else(|| last_public_commit.clone()),
                    finished: Some(Finished {
                        prompt_token_count,
                        output_token_count,
                        internal_token_count: 0,
                        finish_reason: FinishReason::Stop(Some(StopReason::Text(stop_string))),
                    }),
                })
                .await;
                return Ok(());
            }
            raw_stream.acknowledge_text_prefix();
            if intermediate {
                y.yield_ok(DecodedTextEvent::TextDelta {
                    delta: decoded.delta,
                    token_ids: vec![token_id],
                    logprobs: decoded_logprobs,
                    public_commit,
                    finished: None,
                })
                .await;
            }
        }};
    }

    while let Some(event) = raw_stream.next().await {
        match event {
            GenEvent::Scheduled {
                queued_at: queued,
                scheduled_at: scheduled,
            } => {
                queued_at = Some(queued);
                scheduled_at = Some(scheduled);
                emit_start_if_ready!();
            }
            GenEvent::PromptLogprobs { positions } => {
                if !prompt_logprobs_requested {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned unrequested prompt logprobs".to_string(),
                    });
                }
                prompt_positions.extend(positions);
                if prompt_positions.len() > expected_prompt_positions {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine returned too many prompt logprob positions".to_string(),
                    });
                }
                emit_start_if_ready!();
            }
            GenEvent::TextToken {
                id, public_commit, ..
            } => {
                emit_start_if_ready!();
                if !started {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "generation began before prompt metadata was complete".to_string(),
                    });
                }
                if pending_token.is_some() {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "engine emitted a new token before resolving prior logprobs"
                            .to_string(),
                    });
                }
                if generated_logprobs_requested {
                    pending_token = Some((id, public_commit));
                } else {
                    consume_token!(id, Vec::new(), public_commit);
                }
            }
            GenEvent::TokenLogprobs { id, candidates } => {
                let (pending, public_commit) =
                    pending_token.take().ok_or_else(|| Error::MalformedOutput {
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
                consume_token!(
                    id,
                    vec![GenerationPositionLogprobs {
                        entries: candidates,
                    }],
                    public_commit
                );
            }
            GenEvent::Finished {
                reason,
                stop_reason,
                completion_tokens,
                ..
            } => {
                emit_start_if_ready!();
                if !started || pending_token.is_some() {
                    return Err(Error::MalformedOutput {
                        request_id: request_id.clone(),
                        message: "terminal event arrived before output metadata was complete"
                            .to_string(),
                    });
                }
                let finish_reason = text_finish_reason(reason, stop_reason);
                let (last_chunk, text) = decoder.flush(None)?;
                let full_text = tracing::enabled!(Level::TRACE).then(|| text.clone());
                let (delta, token_ids, logprobs) = if intermediate {
                    (last_chunk.unwrap_or_default(), Vec::new(), None)
                } else {
                    (text, accumulated_token_ids, accumulated_logprobs)
                };
                debug!(
                    ?finish_reason,
                    output_token_count, "request finished with decoded text"
                );
                if let Some(full_text) = full_text {
                    trace!(full_text, "terminal decoded text");
                }
                y.yield_ok(DecodedTextEvent::TextDelta {
                    delta,
                    token_ids,
                    logprobs,
                    public_commit: last_public_commit,
                    finished: Some(Finished {
                        prompt_token_count,
                        output_token_count: completion_tokens,
                        internal_token_count: completion_tokens.saturating_sub(output_token_count),
                        finish_reason,
                    }),
                })
                .await;
                return Ok(());
            }
            GenEvent::Rejected { message } | GenEvent::Error { message } => {
                return Err(Error::MalformedOutput {
                    request_id: request_id.clone(),
                    message,
                });
            }
            GenEvent::ImageBegin { .. }
            | GenEvent::ImageStep { .. }
            | GenEvent::ImageCommit { .. }
            | GenEvent::ImageDone { .. } => {
                return Err(Error::MalformedOutput {
                    request_id: request_id.clone(),
                    message: "text-only request received an image lifecycle event".to_string(),
                });
            }
        }
    }

    Err(Error::StreamClosedBeforeTerminalOutput { request_id })
}

fn text_finish_reason(reason: GenerationFinishReason, stop_reason: Option<String>) -> FinishReason {
    match reason {
        GenerationFinishReason::Eos | GenerationFinishReason::ImageDone => FinishReason::Stop(None),
        GenerationFinishReason::Stop => FinishReason::Stop(stop_reason.map(|reason| {
            reason
                .strip_prefix("token:")
                .and_then(|id| id.parse().ok())
                .map_or_else(|| StopReason::Text(reason), StopReason::TokenId)
        })),
        GenerationFinishReason::MaxTokens => FinishReason::Length,
        GenerationFinishReason::Cancelled => FinishReason::Cancelled,
        GenerationFinishReason::Aborted => FinishReason::Aborted,
        GenerationFinishReason::Repetition => FinishReason::Repetition,
        GenerationFinishReason::Error => FinishReason::Error,
    }
}

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
