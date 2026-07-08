use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use uniserve_openai_api::completions::prepare_completion_request;
use uniserve_openai_api::logprobs::{
    collected_logprobs_to_openai, decoded_logprobs_to_openai, decoded_prompt_logprobs_to_maps,
    text_len,
};
use uniserve_openai_types::{
    CompletionChoice, CompletionRequest, CompletionResponse, CompletionSseChunk,
    CompletionStreamChoice, CompletionStreamResponse, LogProbs, Usage,
};
use uniserve_serving::{FinishStatus, RequestMetadata, ServeError, ServeEvent, ServeRequest};
use uniserve_text::{CollectedTextOutput, DecodedLogprobs, FinishReason};

use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::utils::{resolve_request_context, unix_timestamp};
use uniserve_server_app::AppState;

/// Validate one completions request and proxy it into the shared `text`
/// stack.
pub(crate) async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<CompletionRequest>,
) -> Response {
    let stream = body.stream;
    let logprobs = body.logprobs;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    let prepared = match prepare_completion_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let request_span = tracing::info_span!(
        "completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let include_prompt_logprobs = prepared
        .text_request
        .sampling_params
        .prompt_logprobs
        .is_some();
    let log_request = state.enable_log_requests();

    let serve_request = match ServeRequest::from_text_request(
        prepared.text_request,
        RequestMetadata {
            protocol_adapter: Some("openai_completions".to_string()),
            route: Some("v1.completions".to_string()),
            ..RequestMetadata::default()
        },
    ) {
        Ok(request) => request,
        Err(error) => return serve_error_to_api(error).into_response(),
    };

    let serve_stream = match state
        .runtime()
        .serve(serve_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit completion request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if stream {
        let chunk_stream = completion_chunk_stream(
            serve_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            log_request,
            prepared.include_usage,
            prepared.echo,
            logprobs,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
        );
        let sse_stream = completion_sse_stream(chunk_stream).instrument(request_span);

        Sse::new(sse_stream).into_response()
    } else {
        let response = match collect_completion(
            serve_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.echo,
            logprobs,
            include_prompt_logprobs,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
        )
        .instrument(request_span.clone())
        .await
        {
            Ok(response) => response,
            Err(error) => return error.into_response(),
        };

        if log_request {
            let usage = response.usage.as_ref();
            info!(
                parent: &request_span,
                model = %response.model,
                prompt_tokens = usage.map_or(0, |u| u.prompt_tokens),
                output_tokens = usage.and_then(|u| u.completion_tokens).unwrap_or(0),
                finish_reason = response.choices.first().and_then(|c| c.finish_reason.as_deref()).unwrap_or("unknown"),
                "completion finished"
            );
        }

        Json(response).into_response()
    }
}

async fn collect_completion(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
    request_id: String,
    response_model: String,
    created: u64,
    echo: Option<String>,
    requested_logprobs: Option<u32>,
    include_prompt_logprobs: bool,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
) -> Result<CompletionResponse, ApiError> {
    let (collected, finish_status) = collect_serve_text_output(stream).await?;
    let stop_reason = finish_status_stop_reason(&finish_status);

    let prompt_char_count = echo
        .as_ref()
        .map(|prompt| text_len(prompt))
        .unwrap_or_default();
    let prompt_logprobs = if include_prompt_logprobs {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            server_error!(
                "completion response requested prompt_logprobs but generation returned none"
            )
        })?;
        Some(prompt_logprobs)
    } else {
        None
    };
    let logprobs = if requested_logprobs.is_some() {
        Some(collected_logprobs_to_openai(
            &collected,
            echo.is_some(),
            prompt_char_count,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs =
        prompt_logprobs.map(|lp| decoded_prompt_logprobs_to_maps(lp, return_tokens_as_token_ids));
    let text = match &echo {
        None => collected.text,
        Some(prompt) => format!("{prompt}{}", collected.text),
    };

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion".to_string(),
        created,
        model: response_model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs,
            finish_reason: Some(completion_finish_reason_to_openai(&finish_status)?.into()),
            stop_reason,
            prompt_logprobs,
            token_ids: return_token_ids.then(|| collected.token_ids.clone()),
            prompt_token_ids: return_token_ids.then(|| collected.prompt_token_ids.to_vec()),
        }],
        usage: Some(Usage::from_counts(
            collected.prompt_token_ids.len() as u32,
            collected.token_ids.len() as u32,
        )),
        system_fingerprint: None,
        kv_transfer_params: collected.kv_transfer_params,
    })
}

/// Convert one internal decoded-text stream into OpenAI completions chunks.
#[try_stream]
async fn completion_chunk_stream(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
    request_id: String,
    response_model: String,
    created: u64,
    log_request: bool,
    include_usage: bool,
    echo: Option<String>,
    requested_logprobs: Option<u32>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<CompletionSseChunk, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut visible_text_len = 0_u32;
    let mut first_chunk = true;
    let mut prompt_tokens = 0_usize;
    let mut output_tokens = 0_usize;

    while let Some(next) = stream.next().await {
        match next {
            Ok(ServeEvent::Accepted {
                prompt_token_ids, ..
            }) => {
                debug!("completion stream started");
                prompt_tokens = prompt_token_ids.len();
                if let Some(prompt) = echo.as_ref() {
                    visible_text_len = text_len(prompt);
                    let mut chunk =
                        delta_chunk(&request_id, &response_model, created, prompt.clone(), None);
                    if return_token_ids && first_chunk {
                        if let Some(choice) = chunk.choices.first_mut() {
                            choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                        }
                        first_chunk = false;
                    }
                    y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
                } else if return_token_ids {
                    // Emit a chunk with prompt_token_ids in the first streaming response
                    let mut chunk =
                        delta_chunk(&request_id, &response_model, created, String::new(), None);
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.prompt_token_ids = Some(prompt_token_ids.to_vec());
                    }
                    first_chunk = false;
                    y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
                }
            }
            Ok(ServeEvent::TextDelta {
                text,
                token_ids,
                logprobs,
                ..
            }) => {
                if text.is_empty() && token_ids.is_empty() && logprobs.is_none() {
                    continue;
                }
                let delta_text_len = text_len(&text);
                let logprobs = if requested_logprobs.is_some() {
                    let decoded_logprobs = logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "completion stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(decoded_logprobs_to_openai(
                        decoded_logprobs,
                        visible_text_len,
                        return_tokens_as_token_ids,
                    )?)
                } else {
                    None
                };
                let mut chunk = delta_chunk(&request_id, &response_model, created, text, logprobs);
                if return_token_ids && let Some(choice) = chunk.choices.first_mut() {
                    choice.token_ids = Some(token_ids.clone());
                }
                y.yield_ok(CompletionSseChunk::Chunk(chunk)).await;
                visible_text_len = visible_text_len.saturating_add(delta_text_len);
                output_tokens = output_tokens.saturating_add(token_ids.len());
            }
            Ok(ServeEvent::Usage {
                prompt_tokens: prompt,
                visible_output_tokens,
                ..
            }) => {
                prompt_tokens = prompt as usize;
                output_tokens = visible_output_tokens as usize;
            }
            Ok(ServeEvent::Finished { reason, .. }) => {
                if log_request {
                    info!(
                        stream = true,
                        model = %response_model,
                        prompt_tokens,
                        output_tokens,
                        finish_reason = finish_status_as_str(&reason),
                        "completion finished"
                    );
                }
                y.yield_ok(CompletionSseChunk::Chunk(final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    &reason,
                )?))
                .await;

                if include_usage {
                    y.yield_ok(CompletionSseChunk::Usage(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_counts(prompt_tokens as u32, output_tokens as u32),
                    )))
                    .await;
                }
            }
            Ok(_) => {}
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "completion stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    text: String,
    logprobs: Option<LogProbs>,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        text,
        logprobs,
        ..Default::default()
    });
    chunk
}

fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: &FinishStatus,
) -> Result<CompletionStreamResponse, ApiError> {
    let finish_reason = completion_finish_reason_to_openai(finish_reason)?;

    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(CompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        ..Default::default()
    });
    Ok(chunk)
}

fn completion_finish_reason_to_openai(
    finish_reason: &FinishStatus,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishStatus::Stop { .. } | FinishStatus::Repetition => Ok("stop"),
        FinishStatus::Length => Ok("length"),
        FinishStatus::Abort => Ok("abort"),
        FinishStatus::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

async fn collect_serve_text_output(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
) -> Result<(CollectedTextOutput, FinishStatus), ApiError> {
    pin_mut!(stream);
    let mut prompt_token_ids: Arc<[u32]> = Arc::from([]);
    let mut prompt_logprobs = None;
    let mut text = String::new();
    let mut token_ids = Vec::new();
    let mut logprobs: Option<DecodedLogprobs> = None;
    let mut finish_status = None;
    let mut kv_transfer_params = None;

    while let Some(next) = stream.next().await {
        match next {
            Ok(ServeEvent::Accepted {
                prompt_token_ids: ids,
                prompt_logprobs: start_prompt_logprobs,
                ..
            }) => {
                prompt_token_ids = ids.into();
                prompt_logprobs = start_prompt_logprobs;
            }
            Ok(ServeEvent::TextDelta {
                text: delta,
                token_ids: delta_token_ids,
                logprobs: delta_logprobs,
                ..
            }) => {
                text.push_str(&delta);
                token_ids.extend(delta_token_ids);
                if let Some(mut delta_logprobs) = delta_logprobs {
                    logprobs
                        .get_or_insert_with(|| DecodedLogprobs {
                            positions: Vec::new(),
                        })
                        .positions
                        .append(&mut delta_logprobs.positions);
                }
            }
            Ok(ServeEvent::Finished {
                reason,
                kv_transfer_params: params,
                ..
            }) => {
                kv_transfer_params = params;
                finish_status = Some(reason);
                break;
            }
            Ok(_) => {}
            Err(error) => {
                return Err(server_error!(
                    "completion stream failed: {}",
                    error.to_report_string()
                ));
            }
        }
    }

    let Some(finish_status) = finish_status else {
        return Err(server_error!(
            "completion stream closed before terminal finish event"
        ));
    };
    let finish_reason = finish_status_to_text_reason(&finish_status);

    Ok((
        CollectedTextOutput {
            text,
            prompt_token_ids,
            prompt_logprobs,
            logprobs,
            token_ids,
            finish_reason,
            kv_transfer_params,
        },
        finish_status,
    ))
}

fn finish_status_to_text_reason(status: &FinishStatus) -> FinishReason {
    match status {
        FinishStatus::Stop { .. } => FinishReason::Stop(None),
        FinishStatus::Length => FinishReason::Length,
        FinishStatus::Abort => FinishReason::Abort,
        FinishStatus::Error => FinishReason::Error,
        FinishStatus::Repetition => FinishReason::Repetition,
    }
}

fn finish_status_as_str(status: &FinishStatus) -> &'static str {
    match status {
        FinishStatus::Stop { .. } => "stop",
        FinishStatus::Length => "length",
        FinishStatus::Abort => "abort",
        FinishStatus::Error => "error",
        FinishStatus::Repetition => "repetition",
    }
}

fn finish_status_stop_reason(status: &FinishStatus) -> Option<Value> {
    match status {
        FinishStatus::Stop { stop_reason } => stop_reason.clone(),
        FinishStatus::Length
        | FinishStatus::Abort
        | FinishStatus::Error
        | FinishStatus::Repetition => None,
    }
}

fn serve_error_to_api(error: ServeError) -> ApiError {
    match error {
        ServeError::UnsupportedRuntimeExtension { key, .. } => ApiError::invalid_request(
            format!("Unsupported runtime extension `{key}`."),
            Some("uniserve_xargs"),
        ),
        ServeError::UnsupportedOutputCount { requested, .. } => ApiError::invalid_request(
            format!("Only one completion output is supported, got {requested}."),
            Some("n"),
        ),
        ServeError::Text(error) => {
            ApiError::server_error(format!("text runtime error: {}", error.to_report_string()))
        }
        ServeError::Chat(error) => {
            ApiError::server_error(format!("chat runtime error: {}", error.to_report_string()))
        }
        ServeError::Engine(message) => {
            ApiError::server_error(format!("engine runtime error: {message}"))
        }
    }
}

fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> CompletionStreamResponse {
    let mut chunk = CompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// Convert one chunk stream into OpenAI-style SSE events.

/// OpenAI-style streaming errors are encoded as ordinary `data: {"error":...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn completion_sse_stream(
    stream: impl Stream<Item = Result<CompletionSseChunk, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => y.yield_ok(to_sse_event(&chunk)).await,
            Err(error) => {
                y.yield_ok(to_error_sse_event(&error)).await;
                break;
            }
        }
    }

    y.yield_ok(done_sse_event()).await;
    Ok(())
}

/// Serialize one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &CompletionSseChunk) -> Event {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "completion emitting chunk");
    Event::default().data(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "completion emitting error");
    Event::default().data(payload)
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("completion emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use itertools::Itertools as _;
    use uniserve_serving::{FinishStatus, ServeEvent};
    use uniserve_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{CompletionSseChunk, completion_chunk_stream, final_chunk};

    #[test]
    fn final_chunk_maps_stop_finish_reason() {
        let chunk = final_chunk(
            "cmpl-1",
            "model",
            1,
            &FinishStatus::Stop { stop_reason: None },
        )
        .expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].text, "");
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, &FinishStatus::Length).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk =
            final_chunk("cmpl-1", "model", 1, &FinishStatus::Abort).expect("finish reason valid");
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("cmpl-1", "model", 1, &FinishStatus::Error).is_err());
    }

    #[tokio::test]
    async fn completion_chunk_stream_maps_streaming_logprobs() {
        let stream = stream::iter(vec![
            Ok(ServeEvent::Accepted {
                request_id: "cmpl-1".to_string(),
                prompt_token_count: 5,
                prompt_token_ids: vec![1, 2, 3, 4, 5],
                prompt_logprobs: None,
            }),
            Ok(ServeEvent::TextDelta {
                candidate_id: 0,
                text: "h".to_string(),
                token_ids: vec![b'h' as u32],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![
                            DecodedTokenLogprob {
                                token_id: 0,
                                token: "h".to_string(),
                                logprob: -0.1,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token_id: 0,
                                token: "H".to_string(),
                                logprob: -0.2,
                                rank: 1,
                            },
                        ],
                    }],
                }),
            }),
            Ok(ServeEvent::TextDelta {
                candidate_id: 0,
                text: String::new(),
                token_ids: vec![b'!' as u32],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![
                            DecodedTokenLogprob {
                                token_id: 0,
                                token: "!".to_string(),
                                logprob: -0.3,
                                rank: 1,
                            },
                            DecodedTokenLogprob {
                                token_id: 0,
                                token: "?".to_string(),
                                logprob: -0.4,
                                rank: 1,
                            },
                        ],
                    }],
                }),
            }),
            Ok(ServeEvent::Usage {
                prompt_tokens: 5,
                visible_output_tokens: 2,
                internal_tokens: 0,
                image_count: 0,
                image_steps: 0,
            }),
            Ok(ServeEvent::Finished {
                candidate_id: 0,
                reason: FinishStatus::Stop { stop_reason: None },
                finish_detail: None,
                kv_transfer_params: None,
            }),
        ]);

        let chunks = completion_chunk_stream(
            stream,
            "cmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            None,
            Some(1),
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await;

        let chunks: Vec<_> = chunks
            .into_iter()
            .try_collect()
            .expect("stream should succeed");

        match &chunks[0] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "h");
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").tokens,
                    vec!["h".to_string()]
                );
                assert_eq!(
                    chunk.choices[0]
                        .logprobs
                        .as_ref()
                        .expect("logprobs")
                        .text_offset,
                    vec![0]
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }

        match &chunks[1] {
            CompletionSseChunk::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].text, "");
                assert_eq!(
                    chunk.choices[0].logprobs.as_ref().expect("logprobs").tokens,
                    vec!["!".to_string()]
                );
                assert_eq!(
                    chunk.choices[0]
                        .logprobs
                        .as_ref()
                        .expect("logprobs")
                        .text_offset,
                    vec![1]
                );
            }
            CompletionSseChunk::Usage(_) => panic!("expected regular chunk"),
        }
    }
}
