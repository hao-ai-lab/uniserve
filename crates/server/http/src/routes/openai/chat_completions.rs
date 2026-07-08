use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use tracing_futures::Instrument as _;
use uniserve_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatEventStream, ChatEventStreamTrait,
    CollectedAssistantMessage, FinishReason,
};
use uniserve_engine_client::protocol::StopReason;
use uniserve_engine_client::{GenEvent, GenMode};
use uniserve_native_api::events::{Detok, event_json};
use uniserve_native_api::schema::NativeInputImage;
use uniserve_native_api::{NativeGenerateBody, NativeImageBody, NativeRequestBuilder, mode_name};
use uniserve_openai_api::chat_completions::{prepare_chat_request, validate_request_compat};
use uniserve_openai_api::logprobs::{
    decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps,
};
use uniserve_openai_types::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatImageType, ChatLogProbs, ChatMessage, ChatMessageDelta, ChatModality, ContentPart,
    FunctionCallDelta, FunctionCallResponse, ImageUrl, MessageContent, ToolCall, ToolCallDelta,
    Usage,
};

use crate::error::{ApiError, bail_server_error, server_error};
use crate::routes::native::NativeTextOutputFilter;
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::utils::{resolve_request_context, unix_timestamp};
use uniserve_openai_api::lora::LoraModelResolution;
use uniserve_openai_api::utils::ResolvedRequestContext;
use uniserve_server_app::AppState;

/// Validate one chat completion request and proxy it into the shared
/// `chat` stack.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let stream = body.stream;
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(Some(&body.model)).await;

    if let Some(native_mode) = native_chat_mode(&state, &body) {
        return native_chat_completions(
            state,
            body,
            &lora_resolution,
            request_context,
            native_mode,
        )
        .await;
    }

    let prepared = match prepare_chat_request(body, &lora_resolution, request_context) {
        Ok(prepared) => prepared,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let request_span = tracing::info_span!(
        "chat_completions",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let created = unix_timestamp();
    let log_request = state.enable_log_requests();

    let chat_stream = match state
        .chat()
        .chat(prepared.chat_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return server_error!(
                "failed to submit chat request: {}",
                error.to_report_string()
            )
            .into_response();
        }
    };

    if stream {
        let chunk_stream = chat_completion_chunk_stream(
            chat_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            log_request,
            prepared.include_usage,
            prepared.requested_logprobs,
            prepared.include_reasoning,
            prepared.echo,
            prepared.return_token_ids,
            prepared.return_tokens_as_token_ids,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream).instrument(request_span);

        // Emit periodic SSE keep-alive comments so long prefills (no chunks yet)
        // do not get torn down by idle-timeout proxies between client and server.
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let response = match collect_chat_completion(
            chat_stream,
            prepared.request_id,
            prepared.response_model,
            created,
            prepared.requested_logprobs,
            prepared.include_prompt_logprobs,
            prepared.include_reasoning,
            prepared.echo,
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
                "chat completion finished"
            );
        }

        Json(response).into_response()
    }
}

async fn collect_chat_completion(
    stream: ChatEventStream,
    request_id: String,
    response_model: String,
    created: u64,
    requested_logprobs: bool,
    include_prompt_logprobs: bool,
    include_reasoning: bool,
    echo: Option<String>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
) -> Result<ChatCompletionResponse, ApiError> {
    let collected = stream.collect_message().await.map_err(|error| {
        server_error!(
            "failed to collect chat completion response: {}",
            error.to_report_string()
        )
    })?;
    let CollectedAssistantMessage {
        message,
        prompt_token_count,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        output_token_count,
        finish_reason,
        kv_transfer_params,
    } = collected;
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let saw_tool_calls = message.tool_calls().next().is_some();
    let reasoning = message.reasoning();
    // Output logprobs and token IDs cover the complete generated token stream.
    // When reasoning is hidden, omit them rather than leaking hidden reasoning
    // tokens through per-token metadata.
    let include_output_metadata = include_reasoning || reasoning.is_none();
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?.to_string();
    let tool_calls = message
        .tool_calls()
        .map(|call| ToolCall {
            id: call.id.clone(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: call.name.clone(),
                arguments: Some(call.arguments.clone()),
            },
        })
        .collect::<Vec<_>>();
    let logprobs = if requested_logprobs && include_output_metadata {
        Some(decoded_logprobs_to_openai_chat(
            logprobs.as_ref().ok_or_else(|| {
                server_error!("chat response requested logprobs but generation returned none")
            })?,
            return_tokens_as_token_ids,
        )?)
    } else {
        None
    };
    let prompt_logprobs = if include_prompt_logprobs {
        Some(decoded_prompt_logprobs_to_maps(
            prompt_logprobs.as_ref().ok_or_else(|| {
                server_error!(
                    "chat response requested prompt_logprobs but generation returned none"
                )
            })?,
            return_tokens_as_token_ids,
        ))
    } else {
        None
    };
    let usage = Usage::from_counts(prompt_token_count as u32, output_token_count as u32);

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: match &echo {
                    Some(prefix) => Some(format!("{prefix}{}", message.text())),
                    None => Some(message.text()).filter(|t| !t.is_empty()),
                },
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning: if include_reasoning { reasoning } else { None },
                images: None,
            },
            logprobs,
            finish_reason: Some(finish_reason),
            stop_reason,
            token_ids: (return_token_ids && include_output_metadata).then_some(token_ids),
        }],
        usage: Some(usage),
        system_fingerprint: None,
        prompt_logprobs,
        prompt_token_ids: return_token_ids.then(|| prompt_token_ids.to_vec()),
        kv_transfer_params,
    })
}

async fn native_chat_completions(
    state: Arc<AppState>,
    body: ChatCompletionRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    native_mode: GenMode,
) -> Response {
    if let Err(error) = validate_request_compat(&body, &lora_resolution.model_names) {
        return ApiError::from(error).into_response();
    }
    if let Err(error) = validate_native_chat_request(&body, native_mode) {
        return error.into_response();
    }

    let stream = body.stream;
    let include_usage = (body.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let request_id = format!("chatcmpl-{}", ctx.request_id);
    let response_model = lora_resolution
        .lora_request
        .as_ref()
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| {
            lora_resolution
                .model_names
                .first()
                .cloned()
                .unwrap_or_default()
        });
    let created = unix_timestamp();
    let native_body = match native_chat_body(&body, native_mode) {
        Ok(body) => body,
        Err(error) => return error.into_response(),
    };

    let tokenizer = state.chat().text().tokenizer();
    let native_request =
        match NativeRequestBuilder::new(Arc::clone(&tokenizer), state.native_profile())
            .build(&native_body)
        {
            Ok(request) => request,
            Err(error) => {
                return ApiError::invalid_request(error.message().to_string(), None)
                    .into_response();
            }
        };
    let prompt_ids = native_request.prompt_ids.clone();
    let text_filter = match NativeTextOutputFilter::new(
        state.native_profile().output_filter.clone(),
        Arc::clone(&tokenizer),
        &prompt_ids,
    ) {
        Ok(filter) => filter,
        Err(error) => return ApiError::server_error(error.to_string()).into_response(),
    };

    let native_stream = match state
        .chat()
        .uniserve_engine_client()
        .generate_native(native_request)
        .await
    {
        Ok(stream) => stream,
        Err(error) => return ApiError::server_error(error.to_string()).into_response(),
    };

    if stream {
        let chunk_stream = native_chat_completion_chunk_stream(
            native_stream,
            request_id,
            response_model,
            created,
            include_usage,
            tokenizer,
            text_filter,
        );
        let sse_stream = chat_completion_sse_stream(chunk_stream);
        Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        match collect_native_chat_completion(
            native_stream,
            request_id,
            response_model,
            created,
            tokenizer,
            text_filter,
        )
        .await
        {
            Ok(response) => Json(response).into_response(),
            Err(error) => error.into_response(),
        }
    }
}

#[try_stream]
async fn native_chat_completion_chunk_stream(
    mut stream: uniserve_engine_client::NativeEventStream,
    request_id: String,
    response_model: String,
    created: u64,
    include_usage: bool,
    tokenizer: uniserve_text::tokenizer::DynTokenizer,
    mut text_filter: NativeTextOutputFilter,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    y.yield_ok(start_chunk(&request_id, &response_model, created))
        .await;

    let mut detok = Detok::new(tokenizer);
    while let Some(event) = stream.next().await {
        match event {
            GenEvent::TextToken { .. } => {
                let payload = event_json(&event, &mut detok);
                let text = payload.get("text").and_then(Value::as_str).unwrap_or("");
                let text = text_filter.push(text);
                if !text.is_empty() {
                    y.yield_ok(block_delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        AssistantBlockKind::Text,
                        text,
                    ))
                    .await;
                }
            }
            GenEvent::ImageDone { pixels_png_b64, .. } => {
                y.yield_ok(image_delta_chunk(
                    &request_id,
                    &response_model,
                    created,
                    pixels_png_b64,
                ))
                .await;
            }
            GenEvent::Finished {
                reason,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                y.yield_ok(native_final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    &reason,
                )?)
                .await;
                if include_usage {
                    y.yield_ok(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_counts(prompt_tokens as u32, completion_tokens as u32),
                    ))
                    .await;
                }
                return Ok(());
            }
            GenEvent::Rejected { message } => {
                return Err(ApiError::invalid_request(message, None));
            }
            GenEvent::Error { message } => {
                bail_server_error!("{}", message);
            }
            GenEvent::Scheduled { .. }
            | GenEvent::TokenLogprobs { .. }
            | GenEvent::ImageBegin { .. }
            | GenEvent::ImageStep { .. } => {}
        }
    }

    Ok(())
}

async fn collect_native_chat_completion(
    mut stream: uniserve_engine_client::NativeEventStream,
    request_id: String,
    response_model: String,
    created: u64,
    tokenizer: uniserve_text::tokenizer::DynTokenizer,
    mut text_filter: NativeTextOutputFilter,
) -> Result<ChatCompletionResponse, ApiError> {
    let mut detok = Detok::new(tokenizer);
    let mut text = String::new();
    let mut images = Vec::new();
    let mut prompt_tokens = 0_u32;
    let mut completion_tokens = 0_u32;
    let mut finish_reason = "stop".to_string();

    while let Some(event) = stream.next().await {
        match event {
            GenEvent::TextToken { .. } => {
                let payload = event_json(&event, &mut detok);
                let delta = payload.get("text").and_then(Value::as_str).unwrap_or("");
                text.push_str(&text_filter.push(delta));
            }
            GenEvent::ImageDone { pixels_png_b64, .. } => {
                images.push(image_content_part(pixels_png_b64));
            }
            GenEvent::Finished {
                reason,
                prompt_tokens: p,
                completion_tokens: c,
                ..
            } => {
                prompt_tokens = p as u32;
                completion_tokens = c as u32;
                finish_reason = native_finish_reason_to_openai(&reason)?.to_string();
                break;
            }
            GenEvent::Rejected { message } => {
                return Err(ApiError::invalid_request(message, None));
            }
            GenEvent::Error { message } => {
                bail_server_error!("{}", message);
            }
            GenEvent::Scheduled { .. }
            | GenEvent::TokenLogprobs { .. }
            | GenEvent::ImageBegin { .. }
            | GenEvent::ImageStep { .. } => {}
        }
    }

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: Some(text).filter(|text| !text.is_empty()),
                tool_calls: None,
                reasoning: None,
                images: Some(images).filter(|images| !images.is_empty()),
            },
            logprobs: None,
            finish_reason: Some(finish_reason),
            stop_reason: None,
            token_ids: None,
        }],
        usage: Some(Usage::from_counts(prompt_tokens, completion_tokens)),
        system_fingerprint: None,
        prompt_logprobs: None,
        prompt_token_ids: None,
        kv_transfer_params: None,
    })
}

/// Convert one internal chat event stream into OpenAI chat-completion chunks.
#[try_stream]
async fn chat_completion_chunk_stream(
    mut stream: impl ChatEventStreamTrait + Unpin,
    request_id: String,
    response_model: String,
    created: u64,
    log_request: bool,
    include_usage: bool,
    requested_logprobs: bool,
    include_reasoning: bool,
    echo: Option<String>,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    let mut saw_tool_calls = false;
    // `LogprobsDelta` is emitted after all chat events for one decoded update.
    // If that update contains hidden reasoning, including delimiter-only block
    // starts or ends, omit its token metadata as well as its visible delta.
    let mut inside_hidden_reasoning = false;
    let mut suppress_current_update_metadata = false;

    // If the client requested logprobs or token_ids, we need to buffer chunks until
    // we receive the separate `LogprobsDelta` event, so that we can emit one
    // combined chunk with both the semantic delta and its per-update metadata.
    let mut pending_chunk =
        (requested_logprobs || return_token_ids).then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next {
            Ok(ChatEvent::Start {
                prompt_token_ids, ..
            }) => {
                let mut chunk = start_chunk(&request_id, &response_model, created);
                if return_token_ids {
                    chunk.prompt_token_ids = Some(prompt_token_ids.to_vec());
                }
                y.yield_ok(chunk).await;
                // When echo=true, emit the last assistant message content as a delta chunk.
                if let Some(echo_text) = &echo {
                    y.yield_ok(block_delta_chunk(
                        &request_id,
                        &response_model,
                        created,
                        AssistantBlockKind::Text,
                        echo_text.clone(),
                    ))
                    .await;
                }
            }
            Ok(ChatEvent::BlockDelta { kind, delta, .. }) => {
                let include_delta =
                    include_reasoning || !matches!(kind, AssistantBlockKind::Reasoning);
                if include_delta {
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, delta);
                    } else {
                        y.yield_ok(block_delta_chunk(
                            &request_id,
                            &response_model,
                            created,
                            kind,
                            delta,
                        ))
                        .await;
                    }
                } else {
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::LogprobsDelta {
                logprobs,
                token_ids,
            }) => {
                let include_metadata =
                    !suppress_current_update_metadata && !inside_hidden_reasoning;
                suppress_current_update_metadata = false;
                let openai_logprobs = if include_metadata {
                    logprobs
                        .as_ref()
                        .map(|lp| decoded_logprobs_to_openai_chat(lp, return_tokens_as_token_ids))
                        .transpose()?
                } else {
                    None
                };
                let openai_token_ids = include_metadata
                    .then_some(token_ids)
                    .and_then(|token_ids| return_token_ids.then_some(token_ids))
                    .filter(|t| !t.is_empty());
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.logprobs = openai_logprobs;
                    pending_chunk.token_ids = openai_token_ids;
                    if let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                    {
                        y.yield_ok(chunk).await;
                    }
                } else if let Some(logprobs) = openai_logprobs {
                    y.yield_ok(logprobs_only_chunk(
                        &request_id,
                        &response_model,
                        created,
                        logprobs,
                    ))
                    .await;
                }
            }
            Ok(ChatEvent::BlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
                if !include_reasoning && matches!(kind, AssistantBlockKind::Reasoning) {
                    inside_hidden_reasoning = true;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::BlockEnd { .. }) => {
                debug!("ending current block");
                if inside_hidden_reasoning {
                    inside_hidden_reasoning = false;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ChatEvent::ToolCallStart { index, id, name }) => {
                let tool_index = index as u32;
                saw_tool_calls = true;
                debug!(
                    tool_call_id = %id,
                    tool_call_name = %name,
                    "starting new tool call"
                );
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_start(tool_index, id, name);
                } else {
                    y.yield_ok(tool_call_start_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        id,
                        name,
                    ))
                    .await;
                }
            }
            Ok(ChatEvent::ToolCallArgumentsDelta { index, delta }) => {
                let tool_index = index as u32;
                if let Some(pending_chunk) = pending_chunk.as_mut() {
                    pending_chunk.push_tool_call_arguments(tool_index, delta);
                } else {
                    y.yield_ok(tool_call_arguments_chunk(
                        &request_id,
                        &response_model,
                        created,
                        tool_index,
                        delta,
                    ))
                    .await;
                }
            }
            Ok(ChatEvent::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(ChatEvent::Done {
                prompt_token_count,
                finish_reason,
                output_token_count,
                ..
            }) => {
                if log_request {
                    info!(
                        stream = true,
                        model = %response_model,
                        prompt_tokens = prompt_token_count,
                        output_tokens = output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "chat completion finished"
                    );
                }

                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    y.yield_ok(chunk).await;
                }

                match final_chunk(
                    &request_id,
                    &response_model,
                    created,
                    finish_reason,
                    saw_tool_calls,
                ) {
                    Ok(chunk) => y.yield_ok(chunk).await,
                    Err(error) => {
                        error!(
                            error = %error.to_error_response().error.message,
                            "invalid terminal finish reason"
                        );
                        return Err(error);
                    }
                }

                if include_usage {
                    y.yield_ok(usage_chunk(
                        &request_id,
                        &response_model,
                        created,
                        Usage::from_counts(prompt_token_count as u32, output_token_count as u32),
                    ))
                    .await;
                }

                return Ok(());
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "chat stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }
    Ok(())
}

fn usage_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    usage: Usage,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.usage = Some(usage);
    chunk
}

/// Derive the native generation mode for one chat request, if the request
/// belongs on the native engine path at all.

/// Explicit `image` output modality always selects a native mode; the
/// builder's profile check turns unsupported modes into clean 400s. Image
/// *inputs* with text-only output select native understanding only when the
/// model has no generic multimodal chat backend and the native profile
/// declares understanding support — otherwise the request stays on the
/// generic chat stack.
fn native_chat_mode(state: &AppState, request: &ChatCompletionRequest) -> Option<GenMode> {
    let has_image = request.modalities.contains(&ChatModality::Image);
    let has_text = request.modalities.contains(&ChatModality::Text);
    match (has_text, has_image) {
        (true, true) => Some(GenMode::AutoInterleave),
        (false, true) => Some(GenMode::Image),
        (true, false) => {
            let has_image_input = request.messages.iter().any(|message| {
                matches!(message, ChatMessage::User { content, .. } if content_has_image_parts(content))
            });
            (has_image_input
                && !state.chat().has_multimodal_backend()
                && state.native_profile().supports_mode(GenMode::InterleaveUnd))
            .then_some(GenMode::InterleaveUnd)
        }
        (false, false) => None,
    }
}

fn content_has_image_parts(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Parts(parts)
            if parts.iter().any(|part| matches!(part, ContentPart::ImageUrl { .. }))
    )
}

fn validate_native_chat_request(
    request: &ChatCompletionRequest,
    native_mode: GenMode,
) -> Result<(), ApiError> {
    if request.logprobs
        || request.prompt_logprobs.is_some()
        || request.return_token_ids == Some(true)
    {
        return Err(ApiError::invalid_request(
            "token logprobs and token ids are not available for native chat completions"
                .to_string(),
            None,
        ));
    }
    if request.tools.is_some() || request.tool_choice.is_some() {
        return Err(ApiError::invalid_request(
            "tools are not supported for native chat completions".to_string(),
            None,
        ));
    }
    if request.stop.is_some() {
        return Err(ApiError::invalid_request(
            "stop strings are not supported for native chat completions; use stop_token_ids"
                .to_string(),
            None,
        ));
    }
    if native_mode == GenMode::AutoInterleave && !request.stream {
        return Err(ApiError::invalid_request(
            "text+image chat completions require stream=true".to_string(),
            None,
        ));
    }
    if let Some(seed) = request.seed
        && seed < 0
    {
        return Err(ApiError::invalid_request(
            "seed must be non-negative for native chat completions".to_string(),
            Some("seed"),
        ));
    }
    if let Some(config) = &request.image_config {
        if let Some(image_type) = config.image_type
            && image_type != ChatImageType::Png
        {
            return Err(ApiError::invalid_request(
                "image_type must be png for image chat completions".to_string(),
                Some("image_config"),
            ));
        }
        // Both fields alias the profile's resolution-bucket table; the
        // builder resolves whichever is present and rejects unknown names.
        if config.aspect_ratio.is_some() && config.image_size.is_some() {
            return Err(ApiError::invalid_request(
                "aspect_ratio and image_size are mutually exclusive".to_string(),
                Some("image_config"),
            ));
        }
    }
    Ok(())
}

fn native_chat_body(
    request: &ChatCompletionRequest,
    native_mode: GenMode,
) -> Result<NativeGenerateBody, ApiError> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut input_images = Vec::new();

    for message in &request.messages {
        match message {
            ChatMessage::System { content, .. } | ChatMessage::Developer { content, .. } => {
                let (text, images) = message_content_text_and_images(content)?;
                if !images.is_empty() {
                    return Err(ApiError::invalid_request(
                        "system and developer messages cannot contain image_url parts for image chat completions".to_string(),
                        None,
                    ));
                }
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            ChatMessage::User { content, .. } => {
                let (text, images) = message_content_text_and_images(content)?;
                if !text.is_empty() {
                    prompt_parts.push(text);
                }
                input_images.extend(images);
            }
            ChatMessage::Assistant { .. }
            | ChatMessage::Tool { .. }
            | ChatMessage::Function { .. } => {
                return Err(ApiError::invalid_request(
                    "image chat completions accept system/developer/user messages only".to_string(),
                    None,
                ));
            }
        }
    }

    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .map(|tokens| tokens as usize);
    let seed = request.seed.map(|seed| seed as u64);

    Ok(NativeGenerateBody {
        prompt: prompt_parts.join("\n"),
        mode: Some(mode_name(native_mode).to_string()),
        system_prompt: (!system_parts.is_empty()).then(|| system_parts.join("\n")),
        assistant_prefix: None,
        negative_prompt: None,
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed,
        stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
        image_bias: None,
        image: Some(native_image_body(request.image_config.as_ref())),
        input_images,
        input_image_b64: None,
    })
}

fn message_content_text_and_images(
    content: &MessageContent,
) -> Result<(String, Vec<NativeInputImage>), ApiError> {
    match content {
        MessageContent::Text(text) => Ok((text.clone(), Vec::new())),
        MessageContent::Parts(parts) => {
            let mut text_parts = Vec::new();
            let mut images = Vec::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => text_parts.push(text.clone()),
                    ContentPart::ImageUrl { image_url, .. } => {
                        images.push(NativeInputImage {
                            b64: data_url_image_b64(&image_url.url)?,
                            position: None,
                            num_tokens: None,
                        });
                    }
                    ContentPart::VideoUrl { .. } => {
                        return Err(ApiError::invalid_request(
                            "video_url content parts are not supported for image chat completions"
                                .to_string(),
                            None,
                        ));
                    }
                }
            }
            Ok((text_parts.join("\n"), images))
        }
    }
}

fn data_url_image_b64(url: &str) -> Result<String, ApiError> {
    let (meta, payload) = url.split_once(',').ok_or_else(|| {
        ApiError::invalid_request(
            "image_url must be a data:image/*;base64 URL".to_string(),
            None,
        )
    })?;
    if !meta.starts_with("data:image/") || !meta.ends_with(";base64") || payload.is_empty() {
        return Err(ApiError::invalid_request(
            "image_url must be a data:image/*;base64 URL".to_string(),
            None,
        ));
    }
    Ok(payload.to_string())
}

fn native_image_body(config: Option<&uniserve_openai_types::ChatImageConfig>) -> NativeImageBody {
    let mut image = NativeImageBody::default();
    if let Some(config) = config {
        // `aspect_ratio` and `image_size` are aliases into the profile's
        // resolution-bucket table (validated as mutually exclusive).
        image.resolution = config
            .aspect_ratio
            .clone()
            .or_else(|| config.image_size.clone());
        image.width = positive_dimension(config.width);
        image.height = positive_dimension(config.height);
        image.steps = config.steps;
        image.cfg_text_scale = config.guidance_scale;
        image.cfg_img_scale = config.image_guidance_scale;
        image.cfg_renorm_type = config.cfg_norm.clone();
        image.cfg_interval = config.cfg_interval;
        image.cfg_renorm_min = None;
        image.timestep_shift = config.timestep_shift;
        image.seed = config.seed;
        image.max_images = config.num_images;
    }
    image
}

fn positive_dimension(value: Option<i32>) -> Option<u32> {
    value.and_then(|value| (value > 0).then_some(value as u32))
}

fn image_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    png_b64: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            images: Some(vec![image_content_part(png_b64)]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn image_content_part(png_b64: String) -> ContentPart {
    ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!("data:image/png;base64,{png_b64}"),
            detail: None,
        },
        uuid: None,
    }
}

fn native_final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    reason: &uniserve_engine_client::NativeFinishReason,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        finish_reason: Some(native_finish_reason_to_openai(reason)?.to_string()),
        ..Default::default()
    });
    Ok(chunk)
}

fn native_finish_reason_to_openai(
    reason: &uniserve_engine_client::NativeFinishReason,
) -> Result<&'static str, ApiError> {
    match reason {
        uniserve_engine_client::NativeFinishReason::Eos
        | uniserve_engine_client::NativeFinishReason::Stop
        | uniserve_engine_client::NativeFinishReason::ImageDone => Ok("stop"),
        uniserve_engine_client::NativeFinishReason::MaxTokens => Ok("length"),
        uniserve_engine_client::NativeFinishReason::Cancelled
        | uniserve_engine_client::NativeFinishReason::Aborted => Ok("abort"),
        uniserve_engine_client::NativeFinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

/// One in-flight chat-completions SSE chunk being assembled at the route layer.

/// `chat` emits semantic chat events first and `LogprobsDelta` separately,
/// because one decoded update may be rewritten into multiple chat events.
/// The OpenAI chat API, though, wants one streamed chunk to optionally carry
/// both the delta and its logprobs.

/// This small buffer accumulates the semantic delta first, then attaches the
/// following `LogprobsDelta` and flushes one combined chunk. It relies on the
/// current `chat` invariant that all semantic events from one decoded
/// update are emitted before that update's `LogprobsDelta`.
#[derive(Debug, Default)]
struct PendingChatChunk {
    /// The currently buffered OpenAI delta payload assembled from one or more
    /// chat semantic events belonging to the same decoded update.
    delta: ChatMessageDelta,
    /// The token-aligned logprobs for that same decoded update.
    logprobs: Option<ChatLogProbs>,
    /// Per-update output token IDs for the same decoded update.
    token_ids: Option<Vec<u32>>,
}

impl PendingChatChunk {
    /// Append one assistant text/reasoning block delta to the buffered OpenAI
    /// delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => append_delta_text(&mut self.delta.reasoning, delta),
            // Tool calls are expected to flow through the dedicated tool-call
            // chunks, never as block deltas. Drop a stray tool-call delta with a
            // loud log rather than panicking and tearing down the live stream.
            AssistantBlockKind::ToolCall => {
                error!("unexpected tool-call block delta on chunk path; dropping");
            }
        }
    }

    /// Append the OpenAI tool-call-start representation to the buffered delta.
    fn push_tool_call_start(&mut self, index: u32, id: String, name: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
                index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            });
    }

    /// Append one incremental tool-call arguments update to the buffered delta.
    fn push_tool_call_arguments(&mut self, index: u32, delta: String) {
        self.delta
            .tool_calls
            .get_or_insert_with(Vec::new)
            .push(ToolCallDelta {
                index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            });
    }

    /// Finalize the currently buffered SSE chunk, if it contains either a
    /// semantic delta or a logprobs payload.

    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk

    /// The logprobs-only case is intentional: token-level metadata in one
    /// decoded update is correlated with the same update boundary, not
    /// necessarily with a visible/chat-semantic delta.
    fn take_chunk(
        &mut self,
        request_id: &str,
        response_model: &str,
        created: u64,
    ) -> Option<ChatCompletionStreamResponse> {
        let has_delta = self.delta.content.is_some()
            || self.delta.reasoning.is_some()
            || self.delta.tool_calls.is_some();
        let logprobs = self.logprobs.take();
        let token_ids = self.token_ids.take();
        if !has_delta && logprobs.is_none() && token_ids.is_none() {
            return None;
        }

        let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
        chunk.choices.push(ChatCompletionStreamChoice {
            delta: self.take_delta(),
            logprobs,
            token_ids,
            ..Default::default()
        });
        Some(chunk)
    }

    /// Take the currently buffered OpenAI delta payload and leave this pending
    /// chunk empty for the next decoded update.
    fn take_delta(&mut self) -> ChatMessageDelta {
        ChatMessageDelta {
            role: self.delta.role.take(),
            content: self.delta.content.take(),
            tool_calls: self.delta.tool_calls.take(),
            reasoning: self.delta.reasoning.take(),
            images: self.delta.images.take(),
        }
    }
}

/// Append one text fragment to an optional OpenAI delta string field.
fn append_delta_text(slot: &mut Option<String>, delta: String) {
    match slot {
        Some(existing) => existing.push_str(&delta),
        None => *slot = Some(delta),
    }
}

/// Convert one chunk stream into OpenAI-style SSE events.

/// OpenAI-style streaming errors are encoded as ordinary `data: {"error":...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
async fn chat_completion_sse_stream(
    stream: impl Stream<Item = Result<ChatCompletionStreamResponse, ApiError>>,
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
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize chat completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting chunk");
    json_sse_event(payload)
}

/// Serialize one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting error");
    json_sse_event(payload)
}

fn json_sse_event(payload: String) -> Event {
    Event::default().data(payload.replace('\r', "\\r").replace('\n', "\\n"))
}

/// Build the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("chat completion emitting done");
    Event::default().data("[DONE]")
}

/// Build the initial assistant-role SSE chunk required by the OpenAI streaming
/// protocol.
fn start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            role: Some(AssistantRole),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

/// Build one content-delta SSE chunk from one internal assistant block delta.
fn block_delta_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    kind: AssistantBlockKind,
    delta: String,
) -> ChatCompletionStreamResponse {
    let delta = match kind {
        AssistantBlockKind::Text => ChatMessageDelta {
            content: Some(delta),
            ..Default::default()
        },
        AssistantBlockKind::Reasoning => ChatMessageDelta {
            reasoning: Some(delta),
            ..Default::default()
        },
        // Tool calls are expected to flow through the dedicated tool-call
        // chunks, never as block deltas. Emit an empty delta with a loud log
        // rather than panicking and tearing down the live stream.
        AssistantBlockKind::ToolCall => {
            error!("unexpected tool-call block delta on chunk path; emitting empty delta");
            ChatMessageDelta::default()
        }
    };

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta,
        ..Default::default()
    });
    chunk
}

fn tool_call_start_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    id: String,
    name: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: Some(id),
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallDelta {
                    name: Some(name),
                    arguments: None,
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn tool_call_arguments_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    tool_index: u32,
    delta: String,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        delta: ChatMessageDelta {
            tool_calls: Some(vec![ToolCallDelta {
                index: tool_index,
                id: None,
                tool_type: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: Some(delta),
                }),
            }]),
            ..Default::default()
        },
        ..Default::default()
    });
    chunk
}

fn logprobs_only_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    logprobs: ChatLogProbs,
) -> ChatCompletionStreamResponse {
    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        logprobs: Some(logprobs),
        ..Default::default()
    });
    chunk
}

/// Build the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
    saw_tool_calls: bool,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let stop_reason = finish_reason.as_stop_reason().map(stop_reason_to_json);
    let finish_reason = chat_finish_reason_to_openai(&finish_reason, saw_tool_calls)?;

    debug!(
        finish_reason = %finish_reason,
        stop_reason = ?stop_reason,
        "chat stream finished"
    );

    let mut chunk = ChatCompletionStreamResponse::new(request_id, response_model, created);
    chunk.choices.push(ChatCompletionStreamChoice {
        finish_reason: Some(finish_reason.to_string()),
        stop_reason,
        ..Default::default()
    });
    Ok(chunk)
}

fn chat_finish_reason_to_openai(
    finish_reason: &FinishReason,
    saw_tool_calls: bool,
) -> Result<&'static str, ApiError> {
    match finish_reason {
        FinishReason::Stop(_) if saw_tool_calls => Ok("tool_calls"),
        FinishReason::Stop(_) => Ok("stop"),
        FinishReason::Length => Ok("length"),
        FinishReason::Abort => Ok("abort"),
        FinishReason::Repetition => Ok("stop"),
        FinishReason::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

/// Convert one internal stop reason into the OpenAI-compatible `stop_reason`
/// JSON shape.
fn stop_reason_to_json(stop_reason: &StopReason) -> Value {
    serde_json::to_value(stop_reason).unwrap_or_else(|_| Value::String(format!("{stop_reason:?}")))
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use serde_json::json;
    use uniserve_chat::{
        AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatEvent, FinishReason,
    };
    use uniserve_engine_client::protocol::StopReason;
    use uniserve_text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};

    use super::{block_delta_chunk, chat_completion_chunk_stream, final_chunk};

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Text,
            "hello".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(chunk.choices[0].delta.reasoning, None);
    }

    #[test]
    fn reasoning_chunk_uses_reasoning_only_delta() {
        let chunk = block_delta_chunk(
            "chatcmpl-1",
            "model",
            1,
            AssistantBlockKind::Reasoning,
            "thinking".to_string(),
        );
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(
            chunk.choices[0].delta.reasoning.as_deref(),
            Some("thinking")
        );
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_stop_reason() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Stop(Some(StopReason::Text("stop".to_string()))),
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].stop_reason, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::Length, false)
            .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::Abort, false)
            .expect("abort is a valid finish reason");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Error, false).is_err());
    }

    #[test]
    fn final_chunk_maps_stop_to_tool_calls_when_tool_calls_were_streamed() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, FinishReason::stop_eos(), true)
            .expect("finish reason is valid");

        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_text_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Text,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Text,
                delta: "hi".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "hi".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            true,
            true,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("hi"));
        let logprobs = chunks[1].choices[0].logprobs.as_ref().expect("logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "hi");
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_reasoning_delta_with_logprobs() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 0,
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            true,
            true,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.reasoning.as_deref(),
            Some("think")
        );
        assert!(chunks[1].choices[0].logprobs.is_some());
    }

    #[tokio::test]
    async fn chunk_stream_omits_reasoning_delta_when_disabled() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 2,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            false,
            false,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[1].choices[0].delta.content.as_deref(),
            Some("answer")
        );
        assert!(chunks.iter().all(|chunk| {
            chunk
                .choices
                .iter()
                .all(|choice| choice.delta.reasoning.is_none())
        }));
    }

    #[tokio::test]
    async fn chunk_stream_omits_logprobs_for_suppressed_reasoning() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 11,
                            token: "think".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![11],
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 22,
                            token: "answer".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![22],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 2,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            true,
            false,
            None,
            true,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        let choice = &chunks[1].choices[0];
        assert_eq!(choice.delta.content.as_deref(), Some("answer"));
        assert_eq!(choice.token_ids.as_deref(), Some(&[22][..]));
        let logprobs = choice.logprobs.as_ref().expect("answer logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "answer");
        assert!(chunks.iter().all(|chunk| {
            chunk.choices.iter().all(|choice| {
                choice.delta.reasoning.is_none()
                    && choice.token_ids.as_deref() != Some(&[11][..])
                    && choice
                        .logprobs
                        .as_ref()
                        .and_then(|logprobs| logprobs.content.as_ref())
                        .is_none_or(|content| content.iter().all(|entry| entry.token != "think"))
            })
        }));
    }

    #[tokio::test]
    async fn chunk_stream_omits_logprobs_for_hidden_reasoning_delimiters() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::BlockStart {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 11,
                            token: "<think>".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![11],
            }),
            Ok(ChatEvent::BlockDelta {
                index: 0,
                kind: AssistantBlockKind::Reasoning,
                delta: "think".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 12,
                            token: "think".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![12],
            }),
            Ok(ChatEvent::BlockEnd {
                index: 0,
                block: AssistantContentBlock::Reasoning {
                    text: "think".to_string(),
                },
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 13,
                            token: "</think>".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![13],
            }),
            Ok(ChatEvent::BlockStart {
                index: 1,
                kind: AssistantBlockKind::Text,
            }),
            Ok(ChatEvent::BlockDelta {
                index: 1,
                kind: AssistantBlockKind::Text,
                delta: "answer".to_string(),
            }),
            Ok(ChatEvent::LogprobsDelta {
                logprobs: Some(DecodedLogprobs {
                    positions: vec![DecodedPositionLogprobs {
                        entries: vec![DecodedTokenLogprob {
                            token_id: 22,
                            token: "answer".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        }],
                    }],
                }),
                token_ids: vec![22],
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 4,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            true,
            false,
            None,
            true,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(chunks.len(), 3);
        let choice = &chunks[1].choices[0];
        assert_eq!(choice.delta.content.as_deref(), Some("answer"));
        assert_eq!(choice.token_ids.as_deref(), Some(&[22][..]));
        let logprobs = choice.logprobs.as_ref().expect("answer logprobs");
        let content = logprobs.content.as_ref().expect("logprobs content");
        assert_eq!(content[0].token, "answer");
        assert!(chunks.iter().all(|chunk| {
            chunk.choices.iter().all(|choice| {
                choice.delta.reasoning.is_none()
                    && !choice
                        .token_ids
                        .as_ref()
                        .is_some_and(|ids| matches!(ids.as_slice(), [11] | [12] | [13]))
                    && choice
                        .logprobs
                        .as_ref()
                        .and_then(|logprobs| logprobs.content.as_ref())
                        .is_none_or(|content| {
                            content.iter().all(|entry| {
                                !matches!(entry.token.as_str(), "<think>" | "think" | "</think>")
                            })
                        })
            })
        }));
    }

    #[tokio::test]
    async fn chunk_stream_preserves_tool_call_index_and_omits_id_from_arguments_delta() {
        let stream = stream::iter(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
            }),
            Ok(ChatEvent::ToolCallStart {
                index: 3,
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(ChatEvent::ToolCallArgumentsDelta {
                index: 3,
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(ChatEvent::ToolCallEnd {
                index: 3,
                call: AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },
            }),
            Ok(ChatEvent::Done {
                message: Default::default(),
                prompt_token_count: 1,
                output_token_count: 1,
                finish_reason: FinishReason::stop_eos(),
                kv_transfer_params: None,
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            stream,
            "chatcmpl-1".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            false,
            true,
            None,
            false,
            false,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("stream chunks");

        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            Some("call_1".to_string())
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            3
        );
        assert_eq!(
            chunks[2].choices[0].delta.tool_calls.as_ref().unwrap()[0].id,
            None
        );
    }
}
