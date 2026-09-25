//! Buffered and streaming OpenAI chat-completion response assembly.
//!
//! Both paths consume the `RequestOutput` stream that
//! `ServingRuntime::generate_chat` returns. `collect_chat_completion` folds it
//! into one `ChatCompletionResponse`; `chat_completion_chunk_stream` maps it to
//! `ChatCompletionStreamResponse` chunks, and `chat_completion_sse_stream`
//! frames those chunks as SSE events ending in `data: [DONE]`.
//!
//! One decoded update from the engine becomes several events: semantic events
//! (reasoning and text deltas, block and tool-call boundaries) followed by the
//! update's token IDs and logprobs on a `TextDelta`. On the Qwen3 chat output
//! path that metadata arrives as its own `TextDelta` with empty text; on the
//! other generation paths it rides on the same `TextDelta` as the update's
//! visible text, which may be empty. On the Qwen3 path, events flushed when
//! generation finishes follow the final update's metadata. The streaming path
//! relies on this order to attach metadata to the chunk of the same update and
//! to drop the metadata of updates that contain hidden reasoning.

use std::collections::HashMap;
use std::convert::Infallible;
use std::result::Result;

use crate::openai::types::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionResponse,
    ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatLogProbs, ChatMessageDelta,
    ContentPart, FunctionCallDelta, FunctionCallResponse, ImageUrl, ToolCall, ToolCallDelta, Usage,
};
use crate::serving::chat::{AssistantBlockKind, AssistantContentBlock, AssistantMessage};
use crate::serving::text::DecodedLogprobs;
use crate::serving::{FinishStatus, RequestOutput};
use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};

use crate::openai::ApiError;
use crate::openai::logprobs::{decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps};
use crate::openai::utils::completion_token_count;

/// Converts a terminal serving event into its OpenAI representation.
///
/// `Cancelled` and `Aborted` become a normal `Finished` with
/// `FinishStatus::Abort`, so the response carries `finish_reason: "abort"`.
/// Every other event passes through unchanged.
fn openai_terminal_event(event: RequestOutput) -> RequestOutput {
    match event {
        RequestOutput::Cancelled { .. } | RequestOutput::Aborted { .. } => {
            RequestOutput::Finished {
                reason: FinishStatus::Abort,
                finish_detail: None,
            }
        }
        event => event,
    }
}

macro_rules! server_error {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        ApiError::server_error(format!($fmt $(, $arg)*))
    };
}

macro_rules! bail_server_error {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        return Err(server_error!($fmt $(, $arg)*))
    };
}

/// Collects a chat event stream into one non-streaming response.
///
/// The flags mirror the fields of `ChatResponseContext`. `created` is the
/// response timestamp in UNIX seconds.
///
/// When reasoning is present but `include_reasoning` is false, the response
/// omits output logprobs and token IDs as well as the reasoning text, because
/// that metadata covers every generated token, reasoning tokens included.
///
/// # Errors
///
/// Returns `ApiError::rejected` for a `Rejected` event and a server error
/// when the stream yields an error or a `Failed` event, closes without a
/// terminal event, finishes with `FinishStatus::Error`, completes an image
/// without an inline PNG, lacks output logprobs the response must include or
/// prompt logprobs the request asked for, or fails to convert output
/// logprobs.
#[allow(clippy::too_many_arguments)]
pub async fn collect_chat_completion(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    request_id: String,
    response_model: String,
    created: u64,
    requested_logprobs: bool,
    include_prompt_logprobs: bool,
    include_reasoning: bool,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
) -> Result<ChatCompletionResponse, ApiError> {
    let collected = collect_chat_events(stream).await?;
    let CollectedChatOutput {
        message,
        prompt_token_count,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        output_token_count,
        images,
        image_count,
        image_steps,
        image_steps_per_image,
        finish_status,
    } = collected;

    let stop_reason = finish_status_stop_reason(&finish_status);
    let saw_tool_calls = message.tool_calls().next().is_some();
    let reasoning = message.reasoning();
    // Output logprobs and token IDs cover the complete generated token stream.
    // When reasoning is hidden, omit them rather than leaking hidden reasoning
    // tokens through per-token metadata.
    let include_output_metadata = include_reasoning || reasoning.is_none();
    let finish_reason = chat_finish_status_to_openai(&finish_status, saw_tool_calls)?.to_string();
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
    let usage = Usage::from_generation_counts(
        prompt_token_count as u32,
        output_token_count as u32,
        image_count,
        image_steps,
    )
    .with_image_steps_per_image(image_steps_per_image);

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: response_model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: AssistantRole,
                content: Some(message.text()).filter(|text| !text.is_empty()),
                tool_calls: Some(tool_calls).filter(|calls| !calls.is_empty()),
                reasoning: if include_reasoning { reasoning } else { None },
                images: Some(images).filter(|images| !images.is_empty()),
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
    })
}

/// Everything `collect_chat_events` accumulates from one request's stream.
#[derive(Debug, Clone, PartialEq)]
struct CollectedChatOutput {
    message: AssistantMessage,
    /// Prompt tokens from `Accepted`, replaced by the final `Usage` count.
    prompt_token_count: usize,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: Option<crate::serving::text::DecodedPromptLogprobs>,
    /// Output logprobs concatenated across `TextDelta` updates; `None` when no
    /// update carried any.
    logprobs: Option<DecodedLogprobs>,
    token_ids: Vec<u32>,
    /// Visible plus internal generated tokens, as OpenAI usage counts them.
    output_token_count: usize,
    images: Vec<ContentPart>,
    image_count: u32,
    image_steps: u32,
    /// `ImageStep` counts per completed image, in completion order. Images
    /// that began but never completed are not listed.
    image_steps_per_image: Vec<u32>,
    finish_status: FinishStatus,
}

/// Collects semantic serving events into one non-streaming chat response payload.
async fn collect_chat_events(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
) -> Result<CollectedChatOutput, ApiError> {
    pin_mut!(stream);
    let mut message = AssistantMessage::default();
    let mut prompt_token_count = 0_usize;
    let mut prompt_token_ids = Vec::new();
    let mut prompt_logprobs = None;
    let mut logprobs: Option<DecodedLogprobs> = None;
    let mut token_ids = Vec::new();
    let mut output_token_count = 0_usize;
    let mut images = Vec::new();
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;
    let mut image_step_counts = HashMap::<String, u32>::new();
    let mut completed_image_ids = Vec::<String>::new();
    let mut finish_status = None;
    // Structured processors finalize text through `OutputBlockEnd`, whose
    // block carries the complete content, so deltas inside an open block are
    // skipped. Deltas outside any block come from output processors that emit
    // no structured blocks; they are appended after the stream ends, reasoning
    // before text.
    let mut loose_text = String::new();
    let mut loose_reasoning = String::new();
    let mut structured_text = false;

    while let Some(next) = stream.next().await {
        match next.map(openai_terminal_event) {
            Ok(RequestOutput::Accepted {
                prompt_token_count: accepted_prompt_token_count,
                prompt_token_ids: accepted_prompt_token_ids,
                prompt_logprobs: accepted_prompt_logprobs,
                ..
            }) => {
                prompt_token_count = accepted_prompt_token_count;
                prompt_token_ids = accepted_prompt_token_ids;
                prompt_logprobs = accepted_prompt_logprobs;
            }
            Ok(RequestOutput::TextDelta {
                text,
                token_ids: delta_token_ids,
                logprobs: delta_logprobs,
                ..
            }) => {
                if !structured_text {
                    loose_text.push_str(&text);
                }
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
            Ok(RequestOutput::ReasoningDelta { text, .. }) => {
                if !structured_text {
                    loose_reasoning.push_str(&text);
                }
            }
            Ok(RequestOutput::OutputBlockStart { .. }) => structured_text = true,
            Ok(RequestOutput::OutputBlockEnd { block, .. }) => {
                structured_text = false;
                message.push_block(block);
            }
            Ok(RequestOutput::ImageBegin { image_id, .. }) => {
                image_step_counts.entry(image_id).or_default();
            }
            Ok(RequestOutput::ImageStep { image_id, .. }) => {
                *image_step_counts.entry(image_id).or_default() += 1;
            }
            Ok(RequestOutput::ImageDone {
                image_id,
                pixels_png_b64,
                ..
            }) => {
                let png_b64 = pixels_png_b64.ok_or_else(|| {
                    server_error!("image completion ended without an inline PNG artifact")
                })?;
                images.push(image_content_part(png_b64));
                image_step_counts.entry(image_id.clone()).or_default();
                completed_image_ids.push(image_id);
            }
            Ok(RequestOutput::ToolCallEnd {
                id,
                name,
                arguments,
                ..
            }) => {
                message.push_block(AssistantContentBlock::ToolCall(
                    crate::serving::chat::AssistantToolCall {
                        id,
                        name,
                        arguments,
                    },
                ));
            }
            Ok(RequestOutput::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                image_count: generated_images,
                image_steps: generated_steps,
                ..
            }) => {
                prompt_token_count = prompt_tokens as usize;
                output_token_count =
                    completion_token_count(visible_output_tokens, internal_tokens) as usize;
                image_count = generated_images;
                image_steps = generated_steps;
            }
            Ok(RequestOutput::Finished { reason, .. }) => {
                finish_status = Some(reason);
                break;
            }
            Ok(RequestOutput::Rejected { kind, message, .. }) => {
                return Err(ApiError::rejected(kind, message));
            }
            Ok(RequestOutput::Failed {
                request_id,
                message,
            }) => {
                // The failure detail is logged but not returned to the client.
                error!(%request_id, %message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            Ok(_) => {}
            Err(error) => {
                return Err(server_error!(
                    "chat completion stream failed: {}",
                    error.to_report_string()
                ));
            }
        }
    }

    let Some(finish_status) = finish_status else {
        return Err(server_error!(
            "chat completion stream closed before terminal finish event"
        ));
    };

    if !loose_reasoning.is_empty() {
        message.push_block(AssistantContentBlock::Reasoning {
            text: loose_reasoning,
        });
    }
    if !loose_text.is_empty() {
        message.push_block(AssistantContentBlock::Text { text: loose_text });
    }

    Ok(CollectedChatOutput {
        message,
        prompt_token_count,
        prompt_token_ids,
        prompt_logprobs,
        logprobs,
        token_ids,
        output_token_count,
        images,
        image_count,
        image_steps,
        image_steps_per_image: completed_image_ids
            .iter()
            .map(|image_id| image_step_counts.get(image_id).copied().unwrap_or_default())
            .collect(),
        finish_status,
    })
}

/// Converts one serving event stream into OpenAI chat-completion chunks.
///
/// The flags mirror the fields of `ChatResponseContext`; `log_request` logs
/// one summary line when the request finishes. The chunk sequence is an
/// assistant-role start chunk, content, reasoning, tool-call, logprob, and
/// image deltas, one chunk carrying `finish_reason`, and, with
/// `include_usage`, a final usage chunk.
///
/// When logprobs or token IDs are requested, the semantic deltas of one
/// decoded update are buffered in a `PendingChatChunk` and emitted together
/// with that update's metadata. With reasoning hidden, the metadata of an
/// update that touches hidden reasoning is dropped along with the reasoning
/// text.
///
/// # Errors
///
/// Yields `ApiError::rejected` for a `Rejected` event, and a server error for
/// a stream error, a `Failed` event, a `FinishStatus::Error` finish, an image
/// without an inline PNG, or a logprob conversion failure. The stream ends
/// after the first error.
#[allow(clippy::too_many_arguments)]
#[try_stream]
pub async fn chat_completion_chunk_stream(
    stream: impl Stream<Item = crate::serving::Result<RequestOutput>> + Send,
    request_id: String,
    response_model: String,
    created: u64,
    log_request: bool,
    include_usage: bool,
    requested_logprobs: bool,
    include_reasoning: bool,
    return_token_ids: bool,
    return_tokens_as_token_ids: bool,
    mut y: TryYielder<ChatCompletionStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut saw_tool_calls = false;
    let mut prompt_token_count = 0_usize;
    let mut output_token_count = 0_usize;
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;
    let mut image_step_counts = HashMap::<String, u32>::new();
    let mut completed_image_ids = Vec::<String>::new();
    // Token metadata arrives after all semantic deltas of one decoded update
    // (see the module docs). `inside_hidden_reasoning` stays set across
    // updates while a hidden reasoning block is open.
    // `suppress_current_update_metadata` is set by the reasoning arms below,
    // including for a delimiter-only block start or end, and is cleared when
    // the current update's metadata arrives.
    let mut inside_hidden_reasoning = false;
    let mut suppress_current_update_metadata = false;

    // With logprobs or token IDs requested, semantic deltas are buffered until
    // the update's metadata `TextDelta` arrives, so one chunk carries both.
    let mut pending_chunk =
        (requested_logprobs || return_token_ids).then(PendingChatChunk::default);

    while let Some(next) = stream.next().await {
        match next.map(openai_terminal_event) {
            Ok(RequestOutput::Accepted {
                prompt_token_count: accepted_prompt_token_count,
                prompt_token_ids,
                ..
            }) => {
                prompt_token_count = accepted_prompt_token_count;
                let mut chunk = start_chunk(&request_id, &response_model, created);
                if return_token_ids {
                    chunk.prompt_token_ids = Some(prompt_token_ids);
                }
                y.yield_ok(chunk).await;
            }
            Ok(RequestOutput::TextDelta {
                text,
                token_ids,
                logprobs,
                ..
            }) => {
                if !text.is_empty() {
                    let kind = AssistantBlockKind::Text;
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, text);
                    } else {
                        let chunk =
                            block_delta_chunk(&request_id, &response_model, created, kind, text);
                        y.yield_ok(chunk).await;
                    }
                }
                if !token_ids.is_empty() || logprobs.is_some() {
                    let include_metadata =
                        !suppress_current_update_metadata && !inside_hidden_reasoning;
                    suppress_current_update_metadata = false;
                    let openai_logprobs = if include_metadata {
                        logprobs
                            .as_ref()
                            .map(|lp| {
                                decoded_logprobs_to_openai_chat(lp, return_tokens_as_token_ids)
                            })
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
            }
            Ok(RequestOutput::ReasoningDelta { text: delta, .. }) => {
                let kind = AssistantBlockKind::Reasoning;
                let include_delta =
                    include_reasoning || !matches!(kind, AssistantBlockKind::Reasoning);
                if include_delta {
                    if let Some(pending_chunk) = pending_chunk.as_mut() {
                        pending_chunk.push_block_delta(kind, delta);
                    } else {
                        let chunk =
                            block_delta_chunk(&request_id, &response_model, created, kind, delta);
                        y.yield_ok(chunk).await;
                    }
                } else {
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::OutputBlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
                if !include_reasoning && matches!(kind, AssistantBlockKind::Reasoning) {
                    inside_hidden_reasoning = true;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::OutputBlockEnd { block, .. }) => {
                debug!("ending current block");
                if inside_hidden_reasoning || matches!(block.kind(), AssistantBlockKind::Reasoning)
                {
                    inside_hidden_reasoning = false;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(RequestOutput::ToolCallStart {
                index, id, name, ..
            }) => {
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
            Ok(RequestOutput::ToolCallArgumentsDelta { index, delta, .. }) => {
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
            Ok(RequestOutput::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(RequestOutput::ImageBegin { image_id, .. }) => {
                image_step_counts.entry(image_id).or_default();
            }
            Ok(RequestOutput::ImageStep { image_id, .. }) => {
                *image_step_counts.entry(image_id).or_default() += 1;
            }
            Ok(RequestOutput::ImageDone {
                image_id,
                pixels_png_b64,
                ..
            }) => {
                // Flush buffered deltas first so the image chunk keeps its
                // position after the text generated before it.
                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    y.yield_ok(chunk).await;
                }
                let png_b64 = pixels_png_b64.ok_or_else(|| {
                    server_error!("image completion ended without an inline PNG artifact")
                })?;
                let chunk = image_delta_chunk(&request_id, &response_model, created, png_b64);
                y.yield_ok(chunk).await;
                image_step_counts.entry(image_id.clone()).or_default();
                completed_image_ids.push(image_id);
            }
            Ok(RequestOutput::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                image_count: generated_images,
                image_steps: generated_steps,
                ..
            }) => {
                prompt_token_count = prompt_tokens as usize;
                output_token_count =
                    completion_token_count(visible_output_tokens, internal_tokens) as usize;
                image_count = generated_images;
                image_steps = generated_steps;
            }
            Ok(RequestOutput::Finished { reason, .. }) => {
                if log_request {
                    info!(
                        stream = true,
                        model = %response_model,
                        prompt_tokens = prompt_token_count,
                        output_tokens = output_token_count,
                        finish_reason = finish_status_as_str(&reason),
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
                    &reason,
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
                        Usage::from_generation_counts(
                            prompt_token_count as u32,
                            output_token_count as u32,
                            image_count,
                            image_steps,
                        )
                        .with_image_steps_per_image(
                            completed_image_ids
                                .iter()
                                .map(|image_id| {
                                    image_step_counts.get(image_id).copied().unwrap_or_default()
                                })
                                .collect(),
                        ),
                    ))
                    .await;
                }

                return Ok(());
            }
            Ok(RequestOutput::Rejected { kind, message, .. }) => {
                return Err(ApiError::rejected(kind, message));
            }
            Ok(RequestOutput::Failed {
                request_id,
                message,
            }) => {
                // As in `collect_chat_completion`, the failure detail is
                // logged but not returned to the client.
                error!(%request_id, %message, "chat completion failed");
                bail_server_error!("Internal server error");
            }
            Ok(_) => {}
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

/// Builds a streaming usage chunk.
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

/// Builds a streaming image-delta chunk.
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

/// Builds an image content part for a chat response.
fn image_content_part(png_b64: String) -> ContentPart {
    ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: format!("data:image/png;base64,{png_b64}"),
            detail: None,
        },
        uuid: None,
    }
}

/// One in-flight chat-completions SSE chunk being assembled at the route layer.
///
/// Serving output processing splits one decoded update into several semantic
/// events and delivers the update's token IDs and logprobs on a `TextDelta`
/// after them (see the module docs). The OpenAI chat API, though, wants one
/// streamed chunk to optionally carry both the delta and its logprobs.
///
/// This small buffer accumulates the semantic delta first, then attaches the
/// following metadata and flushes one combined chunk. It relies on the
/// output-processing invariant that all semantic events from one decoded
/// update are emitted before that update's metadata. Events flushed when
/// generation finishes are buffered after the final metadata and emitted by
/// the flush before the `finish_reason` chunk.
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
    /// Appends one assistant text/reasoning block delta to the buffered OpenAI
    /// delta payload.
    fn push_block_delta(&mut self, kind: AssistantBlockKind, delta: String) {
        match kind {
            AssistantBlockKind::Text => append_delta_text(&mut self.delta.content, delta),
            AssistantBlockKind::Reasoning => append_delta_text(&mut self.delta.reasoning, delta),
            // Callers pass only `Text` and `Reasoning`; tool calls use
            // `push_tool_call_start` and `push_tool_call_arguments`. A stray
            // tool-call kind is logged and dropped instead of panicking inside
            // a live stream.
            AssistantBlockKind::ToolCall => {
                error!("unexpected tool-call block delta on chunk path; dropping");
            }
        }
    }

    /// Appends the OpenAI tool-call-start representation to the buffered delta.
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

    /// Appends one incremental tool-call arguments update to the buffered delta.
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

    /// Finalizes the currently buffered SSE chunk when it contains either a
    /// semantic delta or a logprobs payload.
    ///
    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk
    ///
    /// Metadata-only chunks are intentional: token metadata belongs to a
    /// decoded update, and an update need not produce a visible delta, for
    /// example while the decoder or the output parsers hold back incomplete
    /// text.
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

    /// Takes the currently buffered OpenAI delta payload and leaves this pending
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

/// Appends one text fragment to an optional OpenAI delta string field.
fn append_delta_text(slot: &mut Option<String>, delta: String) {
    match slot {
        Some(existing) => existing.push_str(&delta),
        None => *slot = Some(delta),
    }
}

/// Converts one chunk stream into OpenAI-style SSE events.
///
/// OpenAI-style streaming errors are encoded as ordinary `data: {"error":...}`
/// events followed by `data: [DONE]`, so the transport stream itself stays
/// infallible even when generation fails after the HTTP response has started.
#[try_stream]
pub async fn chat_completion_sse_stream(
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

/// Serializes one OpenAI chunk payload into one SSE `data:` event.
fn to_sse_event(chunk: &ChatCompletionStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize chat completion chunk","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting chunk");
    json_sse_event(payload)
}

/// Serializes one OpenAI error payload into one SSE `data:` event.
fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response()).unwrap_or_else(|_| {
        r#"{"error":{"message":"failed to serialize error response","type":"server_error"}}"#
            .to_string()
    });
    trace!(payload, "chat completion emitting error");
    json_sse_event(payload)
}

/// Wraps a serialized JSON payload in a server-sent event.
///
/// Compact `serde_json` output already escapes line breaks inside strings;
/// the replacement guarantees that one payload stays on one `data:` line.
/// axum's `Event::data` splits a raw `\n` across several `data:` fields and
/// panics on a raw `\r`.
fn json_sse_event(payload: String) -> Event {
    Event::default().data(payload.replace('\r', "\\r").replace('\n', "\\n"))
}

/// Builds the terminal OpenAI SSE sentinel event.
fn done_sse_event() -> Event {
    trace!("chat completion emitting done");
    Event::default().data("[DONE]")
}

/// Builds the initial assistant-role SSE chunk required by OpenAI streaming.
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

/// Builds one content-delta SSE chunk from one internal assistant block delta.
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
        // Callers pass only `Text` and `Reasoning`; tool calls use
        // `tool_call_start_chunk` and `tool_call_arguments_chunk`. A stray
        // tool-call kind yields an empty delta and an error log instead of
        // panicking inside a live stream.
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

/// Builds the streaming chunk that opens one assistant function call.
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

/// Builds a streaming chunk containing incremental function arguments.
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

/// Builds a chunk containing only log-probability updates.
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

/// Builds the terminal SSE chunk carrying the OpenAI finish reason.
fn final_chunk(
    request_id: &str,
    response_model: &str,
    created: u64,
    finish_status: &FinishStatus,
    saw_tool_calls: bool,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let stop_reason = finish_status_stop_reason(finish_status);
    let finish_reason = chat_finish_status_to_openai(finish_status, saw_tool_calls)?;

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

/// Maps a finish status to the OpenAI `finish_reason` string.
///
/// A stop after any tool call reports `tool_calls`, and a repetition stop
/// reports `stop`. `FinishStatus::Error` returns a server error instead of a
/// reason.
fn chat_finish_status_to_openai(
    finish_status: &FinishStatus,
    saw_tool_calls: bool,
) -> Result<&'static str, ApiError> {
    match finish_status {
        FinishStatus::Stop { .. } if saw_tool_calls => Ok("tool_calls"),
        FinishStatus::Stop { .. } => Ok("stop"),
        FinishStatus::Length => Ok("length"),
        FinishStatus::Abort => Ok("abort"),
        FinishStatus::Repetition => Ok("stop"),
        FinishStatus::Error => {
            bail_server_error!("Internal server error");
        }
    }
}

/// Names a finish status for request logs, keeping the `repetition` and
/// `error` statuses that `chat_finish_status_to_openai` does not report as
/// such.
fn finish_status_as_str(status: &FinishStatus) -> &'static str {
    match status {
        FinishStatus::Stop { .. } => "stop",
        FinishStatus::Length => "length",
        FinishStatus::Abort => "abort",
        FinishStatus::Error => "error",
        FinishStatus::Repetition => "repetition",
    }
}

/// Returns the `stop_reason` response value: the matched stop token ID or
/// stop string, or `None` for an EOS stop, a stop without a recorded cause,
/// and every non-stop status.
fn finish_status_stop_reason(status: &FinishStatus) -> Option<Value> {
    match status {
        FinishStatus::Stop { cause } => cause.as_ref().and_then(|cause| match cause {
            crate::serving::StopCause::Eos => None,
            crate::serving::StopCause::TokenId(id) => Some(serde_json::json!(id)),
            crate::serving::StopCause::Text(text) => Some(serde_json::json!(text)),
        }),
        FinishStatus::Length
        | FinishStatus::Abort
        | FinishStatus::Error
        | FinishStatus::Repetition => None,
    }
}
