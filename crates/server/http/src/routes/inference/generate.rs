mod convert;
mod types;
mod validate;

use std::collections::HashMap;
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
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use uniserve_serving::{FinishStatus, RequestMetadata, ServeError, ServeEvent, ServeRequest};
use uniserve_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs};

use self::convert::prepare_generate_request;
use self::types::{
    GenerateLogprob, GenerateRequest, GenerateResponse, GenerateResponseChoice,
    GenerateResponseStreamChoice, GenerateStreamResponse,
};
use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::utils::resolve_request_context;
use serde_json::Value;
use uniserve_openai_api::logprobs::clamp_logprob;
use uniserve_openai_types::{ChatLogProbs, ChatLogProbsContent, TopLogProb, Usage};
use uniserve_server_app::AppState;

/// Validate one token-in/token-out request and run it through the serving runtime.
pub(crate) async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<GenerateRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;
    let prepared = match prepare_generate_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let request_span = tracing::info_span!(
        "generate",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let log_request = state.enable_log_requests();
    let include_logprobs = prepared.include_logprobs;
    let include_prompt_logprobs = prepared.include_prompt_logprobs;
    let stream = prepared.stream;

    let mut serve_request = match ServeRequest::from_text_request(
        prepared.text_request,
        RequestMetadata {
            protocol_adapter: Some("native_generate".to_string()),
            route: Some("generate".to_string()),
            ..RequestMetadata::default()
        },
    ) {
        Ok(request) => request,
        Err(error) => return serve_error_to_api(error).into_response(),
    };
    serve_request.generation.intermediate = stream;

    let serve_stream = match state
        .runtime()
        .serve(serve_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit raw generate request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if stream {
        let chunk_stream = generate_chunk_stream(
            serve_stream,
            prepared.request_id,
            log_request,
            prepared.include_usage,
            prepared.include_continuous_usage,
            include_logprobs,
        );
        let sse_stream = generate_sse_stream(chunk_stream).instrument(request_span);

        return Sse::new(sse_stream).into_response();
    }

    let collected = match collect_generate_events(serve_stream)
        .instrument(request_span.clone())
        .await
    {
        Ok(collected) => collected,
        Err(error) => return error.into_response(),
    };

    if log_request {
        info!(
            parent: &request_span,
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            finish_reason = finish_status_as_str(&collected.finish_reason),
            "generate finished"
        );
    }

    let response = match collect_generate(
        collected,
        prepared.request_id,
        include_logprobs,
        include_prompt_logprobs,
    ) {
        Ok(response) => response,
        Err(error) => return error.into_response(),
    };

    Json(response).into_response()
}

#[try_stream]
async fn generate_chunk_stream(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>>,
    request_id: String,
    log_request: bool,
    include_usage: bool,
    include_continuous_usage: bool,
    include_logprobs: bool,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens: Option<u32> = None;
    let mut output_tokens = 0_u32;
    let mut emitted_token_count = 0_usize;
    let mut pending_chunk: Option<GenerateStreamResponse> = None;

    while let Some(next) = stream.next().await {
        match next {
            Ok(ServeEvent::Accepted {
                prompt_token_count, ..
            }) => {
                prompt_tokens = Some(prompt_token_count as u32);
            }
            Ok(ServeEvent::TextDelta {
                token_ids,
                logprobs,
                ..
            }) => {
                if let Some(chunk) = pending_chunk.take() {
                    y.yield_ok(chunk).await;
                }
                let usage_prompt_tokens = prompt_tokens.unwrap_or_default();
                let suffix_start = if token_ids.len() > emitted_token_count {
                    emitted_token_count
                } else {
                    0
                };
                let token_ids = token_ids[suffix_start..].to_vec();
                let logprobs = logprobs
                    .as_ref()
                    .map(|logprobs| decoded_logprobs_suffix(logprobs, suffix_start));
                output_tokens = output_tokens.saturating_add(token_ids.len() as u32);
                emitted_token_count = emitted_token_count.saturating_add(token_ids.len());

                if token_ids.is_empty() {
                    pending_chunk = Some(GenerateStreamResponse {
                        request_id: request_id.clone(),
                        choices: vec![GenerateResponseStreamChoice {
                            index: 0,
                            logprobs: None,
                            finish_reason: None,
                            token_ids,
                        }],
                        usage: include_continuous_usage
                            .then(|| Usage::from_counts(usage_prompt_tokens, output_tokens)),
                    });
                    continue;
                }

                let logprobs = if include_logprobs && !token_ids.is_empty() {
                    let logprobs = logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "raw generate stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(decoded_logprobs_to_openai_chat(logprobs)?)
                } else {
                    None
                };

                pending_chunk = Some(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: None,
                        token_ids,
                    }],
                    usage: include_continuous_usage
                        .then(|| Usage::from_counts(usage_prompt_tokens, output_tokens)),
                });
            }
            Ok(ServeEvent::Usage {
                prompt_tokens: prompt,
                visible_output_tokens,
                ..
            }) => {
                prompt_tokens = Some(prompt);
                output_tokens = visible_output_tokens;
                if include_continuous_usage && let Some(chunk) = pending_chunk.as_mut() {
                    chunk.usage = Some(Usage::from_counts(
                        prompt_tokens.unwrap_or_default(),
                        output_tokens,
                    ));
                }
            }
            Ok(ServeEvent::Finished { reason, .. }) => {
                if matches!(reason, FinishStatus::Error) {
                    bail_server_error!("Internal server error");
                }
                if log_request {
                    info!(
                        stream = true,
                        prompt_tokens = prompt_tokens.unwrap_or_default(),
                        output_tokens,
                        finish_reason = finish_status_as_str(&reason),
                        "generate finished"
                    );
                }
                if let Some(mut chunk) = pending_chunk.take() {
                    if let Some(choice) = chunk.choices.first_mut() {
                        choice.finish_reason = Some(finish_status_as_str(&reason).to_string());
                    }
                    if include_continuous_usage {
                        chunk.usage = Some(Usage::from_counts(
                            prompt_tokens.unwrap_or_default(),
                            output_tokens,
                        ));
                    }
                    y.yield_ok(chunk).await;
                } else {
                    y.yield_ok(GenerateStreamResponse {
                        request_id: request_id.clone(),
                        choices: vec![GenerateResponseStreamChoice {
                            index: 0,
                            logprobs: None,
                            finish_reason: Some(finish_status_as_str(&reason).to_string()),
                            token_ids: Vec::new(),
                        }],
                        usage: include_continuous_usage.then(|| {
                            Usage::from_counts(prompt_tokens.unwrap_or_default(), output_tokens)
                        }),
                    })
                    .await;
                }
            }
            Ok(_) => {}
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "raw generate stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }

    if let Some(chunk) = pending_chunk.take() {
        y.yield_ok(chunk).await;
    }

    if include_usage {
        y.yield_ok(GenerateStreamResponse {
            request_id,
            choices: Vec::new(),
            usage: Some(Usage::from_counts(
                prompt_tokens.unwrap_or_default(),
                output_tokens,
            )),
        })
        .await;
    }

    Ok(())
}

fn collect_generate(
    collected: CollectedGenerateOutput,
    request_id: String,
    include_logprobs: bool,
    include_prompt_logprobs: bool,
) -> Result<GenerateResponse, ApiError> {
    let logprobs = if include_logprobs {
        let logprobs = collected.logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested logprobs but generation returned none".to_string(),
            )
        })?;
        Some(decoded_logprobs_to_openai_chat(logprobs)?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            ApiError::server_error(
                "raw generate response requested prompt_logprobs but generation returned none"
                    .to_string(),
            )
        })?;
        Some(decoded_prompt_logprobs_to_maps(prompt_logprobs))
    } else {
        None
    };

    Ok(GenerateResponse {
        request_id,
        choices: vec![GenerateResponseChoice {
            index: 0,
            logprobs,
            finish_reason: Some(finish_status_as_str(&collected.finish_reason).to_string()),
            token_ids: collected.token_ids,
        }],
        prompt_logprobs,
        kv_transfer_params: collected.kv_transfer_params,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct CollectedGenerateOutput {
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: Option<DecodedPromptLogprobs>,
    token_ids: Vec<u32>,
    logprobs: Option<DecodedLogprobs>,
    finish_reason: FinishStatus,
    kv_transfer_params: Option<Value>,
}

async fn collect_generate_events(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
) -> Result<CollectedGenerateOutput, ApiError> {
    pin_mut!(stream);
    let mut prompt_token_ids = Vec::new();
    let mut prompt_logprobs = None;
    let mut token_ids = Vec::new();
    let mut logprobs: Option<DecodedLogprobs> = None;
    let mut finish_reason = None;
    let mut kv_transfer_params = None;

    while let Some(next) = stream.next().await {
        match next {
            Ok(ServeEvent::Accepted {
                prompt_token_ids: ids,
                prompt_logprobs: accepted_prompt_logprobs,
                ..
            }) => {
                prompt_token_ids = ids;
                prompt_logprobs = accepted_prompt_logprobs;
            }
            Ok(ServeEvent::TextDelta {
                token_ids: delta_token_ids,
                logprobs: delta_logprobs,
                ..
            }) => {
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
                finish_reason = Some(reason);
                kv_transfer_params = params;
                break;
            }
            Ok(_) => {}
            Err(error) => {
                return Err(server_error!(
                    "raw generate stream failed: {}",
                    error.to_report_string()
                ));
            }
        }
    }

    let Some(finish_reason) = finish_reason else {
        return Err(server_error!(
            "raw generate stream closed before terminal finish event"
        ));
    };

    Ok(CollectedGenerateOutput {
        prompt_token_ids,
        prompt_logprobs,
        token_ids,
        logprobs,
        finish_reason,
        kv_transfer_params,
    })
}

fn decoded_logprobs_to_openai_chat(logprobs: &DecodedLogprobs) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(position_to_chat_logprobs_content)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

fn decoded_logprobs_suffix(logprobs: &DecodedLogprobs, start: usize) -> DecodedLogprobs {
    DecodedLogprobs {
        positions: logprobs.positions.iter().skip(start).cloned().collect(),
    }
}

fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    std::iter::once(None)
        .chain(
            prompt_logprobs
                .scored_positions
                .iter()
                .map(|position| Some(position_to_logprob_map(position))),
        )
        .collect()
}

fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        ApiError::server_error(
            "raw generate logprobs position unexpectedly had no token candidates".to_string(),
        )
    })?;
    let token = format_token_id(chosen.token_id);

    Ok(ChatLogProbsContent {
        token: token.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token.as_bytes().to_vec()),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token_id(entry.token_id);
                TopLogProb {
                    token: token.clone(),
                    logprob: clamp_logprob(entry.logprob),
                    bytes: Some(token.into_bytes()),
                }
            })
            .collect(),
    })
}

fn position_to_logprob_map(position: &DecodedPositionLogprobs) -> HashMap<u32, GenerateLogprob> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                entry.token_id,
                GenerateLogprob {
                    logprob: clamp_logprob(entry.logprob),
                    rank: Some(entry.rank),
                    decoded_token: Some(format_token_id(entry.token_id)),
                },
            )
        })
        .collect()
}

fn format_token_id(token_id: u32) -> String {
    format!("token_id:{token_id}")
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

fn serve_error_to_api(error: ServeError) -> ApiError {
    match error {
        ServeError::UnsupportedRuntimeExtension { key, .. } => ApiError::invalid_request(
            format!("Unsupported runtime extension `{key}`."),
            Some("uniserve_xargs"),
        ),
        ServeError::UnsupportedOutputCount { requested, .. } => ApiError::invalid_request(
            format!("Only one generate output is supported, got {requested}."),
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

/// Convert one raw-generate chunk stream into SSE events.
#[try_stream]
async fn generate_sse_stream(
    stream: impl Stream<Item = Result<GenerateStreamResponse, ApiError>>,
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

fn to_sse_event(chunk: &GenerateStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize generate chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "generate emitting chunk");
    json_sse_event(payload)
}

fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "generate emitting error");
    json_sse_event(payload)
}

fn json_sse_event(payload: String) -> Event {
    Event::default().data(payload.replace('\r', "\\r").replace('\n', "\\n"))
}

fn done_sse_event() -> Event {
    trace!("generate emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use futures::{TryStreamExt as _, stream};
    use uniserve_text::DecodedTokenLogprob;

    use super::*;

    #[tokio::test]
    async fn generate_chunk_stream_uses_accepted_prompt_info_for_usage() {
        let stream = stream::iter(vec![
            Ok(ServeEvent::Accepted {
                request_id: "raw-stream".to_string(),
                prompt_token_count: 2,
                prompt_token_ids: vec![11, 22],
                prompt_logprobs: None,
            }),
            Ok(ServeEvent::TextDelta {
                candidate_id: 0,
                text: String::new(),
                token_ids: vec![33],
                logprobs: None,
            }),
            Ok(ServeEvent::Finished {
                candidate_id: 0,
                reason: FinishStatus::Stop { stop_reason: None },
                finish_detail: None,
                kv_transfer_params: None,
            }),
        ]);

        let chunks: Vec<_> =
            generate_chunk_stream(stream, "raw-stream".to_string(), false, true, true, false)
                .try_collect()
                .await
                .expect("collect chunks");

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].usage.as_ref().expect("chunk usage").prompt_tokens,
            2
        );
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            chunks[1].usage.as_ref().expect("final usage").prompt_tokens,
            2
        );
    }

    #[tokio::test]
    async fn collect_generate_events_aggregates_logprobs_and_finish_metadata() {
        let stream = stream::iter(vec![
            Ok(ServeEvent::Accepted {
                request_id: "raw".to_string(),
                prompt_token_count: 2,
                prompt_token_ids: vec![11, 22],
                prompt_logprobs: Some(DecodedPromptLogprobs {
                    first_token_id: 11,
                    first_token: "A".to_string(),
                    scored_positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 22,
                            token: "B".to_string(),
                            logprob: -0.5,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(ServeEvent::TextDelta {
                candidate_id: 0,
                text: "C".to_string(),
                token_ids: vec![33],
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 33,
                            token: "C".to_string(),
                            logprob: -0.25,
                            rank: 1,
                        }],
                    }],
                }),
            }),
            Ok(ServeEvent::Finished {
                candidate_id: 0,
                reason: FinishStatus::Length,
                finish_detail: None,
                kv_transfer_params: None,
            }),
        ]);

        let collected = collect_generate_events(stream)
            .await
            .expect("collect generate events");

        assert_eq!(collected.prompt_token_ids, vec![11, 22]);
        assert_eq!(collected.token_ids, vec![33]);
        assert_eq!(collected.finish_reason, FinishStatus::Length);
        assert_eq!(
            collected
                .prompt_logprobs
                .as_ref()
                .expect("prompt logprobs")
                .scored_positions[0]
                .entries[0]
                .token,
            "B"
        );
        assert_eq!(
            collected
                .logprobs
                .as_ref()
                .expect("sample logprobs")
                .positions[0]
                .entries[0]
                .token,
            "C"
        );
    }
}
