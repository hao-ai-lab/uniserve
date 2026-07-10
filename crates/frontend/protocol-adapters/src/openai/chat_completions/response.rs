use std::convert::Infallible;
use std::result::Result;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::{Stream, StreamExt as _, pin_mut};
use serde_json::Value;
use thiserror_ext::AsReport as _;
use tracing::{debug, error, info, trace};
use uniserve_core::GenerationConstraint;
use uniserve_openai_types::{
    AssistantRole, ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamChoice, ChatCompletionStreamResponse,
    ChatImageType, ChatLogProbs, ChatMessage, ChatMessageDelta, ChatModality, ContentPart,
    FunctionCallDelta, FunctionCallResponse, ImageUrl, MessageContent, ToolCall, ToolCallDelta,
    Usage,
};
use uniserve_serving::chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt as _,
};
use uniserve_serving::text::DecodedLogprobs;
use uniserve_serving::{
    CandidateId, FinishStatus, ImageGenerationPolicy, ModalityPolicy, ServeEvent,
};

use crate::openai::ApiError;
use crate::openai::chat_completions::convert::{PreparedRequest, prepare_chat_request};
use crate::openai::logprobs::{decoded_logprobs_to_openai_chat, decoded_prompt_logprobs_to_maps};
use crate::openai::lora::LoraModelResolution;
use crate::openai::utils::{ResolvedRequestContext, completion_token_count};

fn openai_terminal_event(event: ServeEvent) -> ServeEvent {
    match event {
        ServeEvent::Cancelled { .. } | ServeEvent::Aborted { .. } => ServeEvent::Finished {
            candidate_id: CandidateId::PRIMARY,
            reason: FinishStatus::Abort,
            finish_detail: None,
        },
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

pub async fn collect_chat_completion(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
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
    );

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
        kv_transfer_params: None,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct CollectedChatOutput {
    message: AssistantMessage,
    prompt_token_count: usize,
    prompt_token_ids: Vec<u32>,
    prompt_logprobs: Option<uniserve_serving::text::DecodedPromptLogprobs>,
    logprobs: Option<DecodedLogprobs>,
    token_ids: Vec<u32>,
    output_token_count: usize,
    images: Vec<ContentPart>,
    image_count: u32,
    image_steps: u32,
    finish_status: FinishStatus,
}

async fn collect_chat_events(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
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
    let mut finish_status = None;

    while let Some(next) = stream.next().await {
        match next.map(openai_terminal_event) {
            Ok(ServeEvent::Accepted {
                prompt_token_count: accepted_prompt_token_count,
                prompt_token_ids: accepted_prompt_token_ids,
                prompt_logprobs: accepted_prompt_logprobs,
                ..
            }) => {
                prompt_token_count = accepted_prompt_token_count;
                prompt_token_ids = accepted_prompt_token_ids;
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
            Ok(ServeEvent::OutputBlockEnd { block, .. }) => message.push_block(block),
            Ok(ServeEvent::ImageDone { pixels_png_b64, .. }) => {
                let png_b64 = pixels_png_b64.ok_or_else(|| {
                    server_error!("image completion ended without an inline PNG artifact")
                })?;
                images.push(image_content_part(png_b64));
            }
            Ok(ServeEvent::ToolCallEnd {
                id,
                name,
                arguments,
                ..
            }) => {
                message.push_block(AssistantContentBlock::ToolCall(
                    uniserve_serving::chat::AssistantToolCall {
                        id,
                        name,
                        arguments,
                    },
                ));
            }
            Ok(ServeEvent::Usage {
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
            Ok(ServeEvent::Finished { reason, .. }) => {
                finish_status = Some(reason);
                break;
            }
            Ok(ServeEvent::Rejected { message, .. }) => {
                return Err(ApiError::invalid_request(message, None));
            }
            Ok(ServeEvent::Failed {
                request_id,
                message,
            }) => {
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
        finish_status,
    })
}

/// Convert one serving event stream into OpenAI chat-completion chunks.
#[try_stream]
pub async fn chat_completion_chunk_stream(
    stream: impl Stream<Item = uniserve_serving::Result<ServeEvent>> + Send,
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
    pin_mut!(stream);
    let mut saw_tool_calls = false;
    let mut prompt_token_count = 0_usize;
    let mut output_token_count = 0_usize;
    let mut image_count = 0_u32;
    let mut image_steps = 0_u32;
    // Token metadata is emitted after all semantic deltas for one decoded update.
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
        match next.map(openai_terminal_event) {
            Ok(ServeEvent::Accepted {
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
            Ok(ServeEvent::TextDelta {
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
                        y.yield_ok(block_delta_chunk(
                            &request_id,
                            &response_model,
                            created,
                            kind,
                            text,
                        ))
                        .await;
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
            Ok(ServeEvent::ReasoningDelta { text: delta, .. }) => {
                let kind = AssistantBlockKind::Reasoning;
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
            Ok(ServeEvent::OutputBlockStart { kind, .. }) => {
                debug!(?kind, "starting new block");
                if !include_reasoning && matches!(kind, AssistantBlockKind::Reasoning) {
                    inside_hidden_reasoning = true;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ServeEvent::OutputBlockEnd { block, .. }) => {
                debug!("ending current block");
                if inside_hidden_reasoning || matches!(block.kind(), AssistantBlockKind::Reasoning)
                {
                    inside_hidden_reasoning = false;
                    suppress_current_update_metadata = true;
                }
            }
            Ok(ServeEvent::ToolCallStart {
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
            Ok(ServeEvent::ToolCallArgumentsDelta { index, delta, .. }) => {
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
            Ok(ServeEvent::ToolCallEnd { .. }) => {
                debug!("ending current tool call");
            }
            Ok(ServeEvent::ImageDone { pixels_png_b64, .. }) => {
                if let Some(pending_chunk) = pending_chunk.as_mut()
                    && let Some(chunk) =
                        pending_chunk.take_chunk(&request_id, &response_model, created)
                {
                    y.yield_ok(chunk).await;
                }
                let png_b64 = pixels_png_b64.ok_or_else(|| {
                    server_error!("image completion ended without an inline PNG artifact")
                })?;
                y.yield_ok(image_delta_chunk(
                    &request_id,
                    &response_model,
                    created,
                    png_b64,
                ))
                .await;
            }
            Ok(ServeEvent::Usage {
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
            Ok(ServeEvent::Finished { reason, .. }) => {
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
                        ),
                    ))
                    .await;
                }

                return Ok(());
            }
            Ok(ServeEvent::Rejected { message, .. }) => {
                return Err(ApiError::invalid_request(message, None));
            }
            Ok(ServeEvent::Failed { .. }) => {
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

pub fn native_chat_constraint(
    supports_und_only: bool,
    request: &ChatCompletionRequest,
) -> Option<GenerationConstraint> {
    let has_image = request.modalities.contains(&ChatModality::Image);
    let has_text = request.modalities.contains(&ChatModality::Text);
    match (has_text, has_image) {
        (true, true) => Some(GenerationConstraint::Default),
        (false, true) => Some(GenerationConstraint::GenOnly),
        (true, false) => {
            let has_image_input = request.messages.iter().any(chat_message_has_image_parts);
            (has_image_input && supports_und_only).then_some(GenerationConstraint::UndOnly)
        }
        (false, false) => None,
    }
}

fn chat_message_has_image_parts(message: &ChatMessage) -> bool {
    match message {
        ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. } => content_has_image_parts(content),
        ChatMessage::Assistant { content, .. } => {
            content.as_ref().is_some_and(content_has_image_parts)
        }
        ChatMessage::Function { .. } => false,
    }
}

fn content_has_image_parts(content: &MessageContent) -> bool {
    matches!(
        content,
        MessageContent::Parts(parts)
            if parts.iter().any(|part| matches!(part, ContentPart::ImageUrl { .. }))
    )
}

pub fn validate_native_chat_request(
    request: &ChatCompletionRequest,
    _native_constraint: GenerationConstraint,
) -> Result<(), ApiError> {
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

pub fn prepare_native_chat_request(
    request: ChatCompletionRequest,
    lora_resolution: &LoraModelResolution,
    request_context: ResolvedRequestContext,
    native_constraint: GenerationConstraint,
) -> Result<PreparedRequest, ApiError> {
    validate_native_chat_request(&request, native_constraint)?;
    let input_image = request.messages.iter().any(chat_message_has_image_parts);
    let image = native_image_policy(request.image_config.as_ref());
    let mut prepared = prepare_chat_request(request, lora_resolution, request_context)?;
    prepared.serve_request.generation.constraint = native_constraint;
    prepared.serve_request.generation.image = image;
    prepared.serve_request.modalities = ModalityPolicy {
        input_text: true,
        input_image,
        output_text: native_constraint != GenerationConstraint::GenOnly,
        output_image: native_constraint != GenerationConstraint::UndOnly,
    };
    Ok(prepared)
}

fn native_image_policy(
    config: Option<&uniserve_openai_types::ChatImageConfig>,
) -> ImageGenerationPolicy {
    let mut image = ImageGenerationPolicy::default();
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

/// One in-flight chat-completions SSE chunk being assembled at the route layer.
///
/// `chat` emits semantic chat events first and `LogprobsDelta` separately,
/// because one decoded update may be rewritten into multiple chat events.
/// The OpenAI chat API, though, wants one streamed chunk to optionally carry
/// both the delta and its logprobs.
///
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
    ///
    /// This may produce:
    /// - a combined delta + logprobs chunk
    /// - a delta-only chunk
    /// - a logprobs-only chunk
    ///
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
        FinishStatus::Stop { cause } => cause.as_ref().and_then(|cause| match cause {
            uniserve_serving::StopCause::Eos => None,
            uniserve_serving::StopCause::TokenId(id) => Some(serde_json::json!(id)),
            uniserve_serving::StopCause::Text(text) => Some(serde_json::json!(text)),
        }),
        FinishStatus::Length
        | FinishStatus::Abort
        | FinishStatus::Error
        | FinishStatus::Repetition => None,
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt as _, stream};
    use serde_json::json;
    use uniserve_serving::chat::{
        AssistantBlockKind, AssistantContentBlock, AssistantToolCall, ChatEvent, FinishReason,
    };
    use uniserve_serving::text::{DecodedLogprobs, DecodedPositionLogprobs, DecodedTokenLogprob};
    use uniserve_serving::{
        AdapterSelection, CandidateId, FinishStatus, ModelContext, ServeError, ServeEvent,
    };

    use super::{
        block_delta_chunk, chat_completion_chunk_stream, collect_chat_completion, final_chunk,
        prepare_native_chat_request,
    };
    use crate::openai::lora::LoraModelResolution;
    use crate::openai::utils::ResolvedRequestContext;

    #[test]
    fn native_chat_request_preserves_sampling_controls() {
        let request = serde_json::from_value(json!({
            "model": "model",
            "stream": true,
            "messages": [{"role": "user", "content": "draw"}],
            "min_tokens": 5,
            "min_p": 0.12,
            "frequency_penalty": 0.25,
            "presence_penalty": -0.5,
            "repetition_penalty": 1.1,
            "ignore_eos": true,
            "logit_bias": {"151670": 80.0},
            "allowed_token_ids": [7, 11],
            "bad_words": ["blocked"]
        }))
        .unwrap();

        let prepared = prepare_native_chat_request(
            request,
            &LoraModelResolution {
                model_names: vec!["model".to_string()],
                adapter: AdapterSelection::Base,
            },
            ResolvedRequestContext {
                request_id: "req".to_string(),
                ..ResolvedRequestContext::default()
            },
            uniserve_core::GenerationConstraint::Default,
        )
        .expect("sampling controls are valid");
        let generation = prepared.serve_request.generation;

        assert_eq!(generation.min_tokens, Some(5));
        assert_eq!(generation.min_p, Some(0.12));
        assert_eq!(generation.frequency_penalty, Some(0.25));
        assert_eq!(generation.presence_penalty, Some(-0.5));
        assert_eq!(generation.repetition_penalty, Some(1.1));
        assert!(generation.ignore_eos);
        assert_eq!(
            generation.logit_bias,
            Some(std::collections::HashMap::from([(151670, 80.0)]))
        );
        assert_eq!(generation.allowed_token_ids, Some(vec![7, 11]));
        assert_eq!(generation.bad_words, vec!["blocked"]);
        assert!(matches!(
            prepared.serve_request.model_context,
            ModelContext::Chat { .. }
        ));
    }

    #[test]
    fn native_chat_request_preserves_history_tools_and_image_order() {
        let request = serde_json::from_value(json!({
            "model": "model",
            "stream": true,
            "modalities": ["text", "image"],
            "messages": [
                {"role":"system","content":"policy"},
                {"role":"user","content":[
                    {"type":"text","text":"inspect"},
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,aQ=="}}
                ]},
                {"role":"assistant","content":"checking","tool_calls":[{
                    "id":"call-1","type":"function",
                    "function":{"name":"lookup","arguments":"{\"id\":1}"}
                }]},
                {"role":"tool","tool_call_id":"call-1","content":"result"},
                {"role":"user","content":"continue"}
            ],
            "tools":[{"type":"function","function":{
                "name":"lookup","description":"Lookup",
                "parameters":{"type":"object","properties":{"id":{"type":"integer"}}}
            }}],
            "tool_choice":"auto"
        }))
        .expect("OpenAI chat request");

        let prepared = prepare_native_chat_request(
            request,
            &LoraModelResolution {
                model_names: vec!["model".to_string()],
                adapter: AdapterSelection::Base,
            },
            ResolvedRequestContext {
                request_id: "history".to_string(),
                ..ResolvedRequestContext::default()
            },
            uniserve_core::GenerationConstraint::Default,
        )
        .expect("multimodal history");

        let ModelContext::Chat {
            messages,
            tools,
            tool_choice,
            ..
        } = prepared.serve_request.model_context
        else {
            panic!("chat context");
        };
        assert_eq!(messages.len(), 5);
        assert!(matches!(
            messages[2],
            uniserve_serving::chat::ChatMessage::Assistant { .. }
        ));
        assert!(matches!(
            messages[3],
            uniserve_serving::chat::ChatMessage::ToolResponse { .. }
        ));
        assert_eq!(tools.len(), 1);
        assert_eq!(tool_choice, uniserve_serving::chat::ChatToolChoice::Auto);
    }

    #[tokio::test]
    async fn multimodal_collection_preserves_reasoning_tools_images_and_usage() {
        let events = stream::iter(vec![
            Ok(ServeEvent::Accepted {
                request_id: "req".into(),
                profile_id: "profile".to_string(),
                dialect_id: "dialect".to_string(),
                compile_duration_us: 1,
                prompt_token_count: 3,
                prompt_token_ids: vec![1, 2, 3],
                prompt_logprobs: None,
            }),
            Ok(ServeEvent::OutputBlockEnd {
                candidate_id: CandidateId::PRIMARY,
                index: 0,
                block: AssistantContentBlock::Reasoning {
                    text: "plan".to_string(),
                },
            }),
            Ok(ServeEvent::ToolCallEnd {
                candidate_id: CandidateId::PRIMARY,
                index: 0,
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                arguments: r#"{"id":1}"#.to_string(),
            }),
            Ok(ServeEvent::ImageDone {
                candidate_id: CandidateId::PRIMARY,
                image_id: "0".to_string(),
                width: Some(64),
                height: Some(64),
                bytes: Some(3),
                sha256: Some("hash".to_string()),
                pixels_png_b64: Some("cG5n".to_string()),
                elapsed_us: 2,
            }),
            Ok(ServeEvent::Usage {
                prompt_tokens: 3,
                visible_output_tokens: 2,
                internal_tokens: 1,
                image_count: 1,
                image_steps: 4,
                cache: Default::default(),
                resources: Default::default(),
                timings: Default::default(),
            }),
            Ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: Some("image_done".to_string()),
            }),
        ]);

        let response = collect_chat_completion(
            events,
            "req".to_string(),
            "model".to_string(),
            1,
            false,
            false,
            true,
            None,
            false,
            false,
        )
        .await
        .expect("multimodal response");
        let choice = &response.choices[0];
        assert_eq!(choice.message.reasoning.as_deref(), Some("plan"));
        assert_eq!(
            choice.message.tool_calls.as_ref().unwrap()[0].function.name,
            "lookup"
        );
        assert!(matches!(
            choice.message.images.as_deref(),
            Some([uniserve_openai_types::ContentPart::ImageUrl { image_url, .. }])
                if image_url.url == "data:image/png;base64,cG5n"
        ));
        let usage = response.usage.expect("usage");
        assert_eq!(usage.completion_tokens, Some(3));
        assert_eq!(usage.image_count, Some(1));
        assert_eq!(usage.image_steps, Some(4));
    }

    #[tokio::test]
    async fn multimodal_stream_preserves_reasoning_tools_and_images() {
        let events = stream::iter(vec![
            Ok(ServeEvent::Accepted {
                request_id: "req".into(),
                profile_id: "profile".to_string(),
                dialect_id: "dialect".to_string(),
                compile_duration_us: 1,
                prompt_token_count: 1,
                prompt_token_ids: vec![1],
                prompt_logprobs: None,
            }),
            Ok(ServeEvent::ReasoningDelta {
                candidate_id: CandidateId::PRIMARY,
                text: "plan".to_string(),
            }),
            Ok(ServeEvent::ToolCallStart {
                candidate_id: CandidateId::PRIMARY,
                index: 0,
                id: "call_1".to_string(),
                name: "lookup".to_string(),
            }),
            Ok(ServeEvent::ToolCallArgumentsDelta {
                candidate_id: CandidateId::PRIMARY,
                index: 0,
                delta: r#"{"id":1}"#.to_string(),
            }),
            Ok(ServeEvent::ImageDone {
                candidate_id: CandidateId::PRIMARY,
                image_id: "0".to_string(),
                width: Some(64),
                height: Some(64),
                bytes: Some(3),
                sha256: Some("hash".to_string()),
                pixels_png_b64: Some("cG5n".to_string()),
                elapsed_us: 2,
            }),
            Ok(ServeEvent::Usage {
                prompt_tokens: 1,
                visible_output_tokens: 1,
                internal_tokens: 1,
                image_count: 1,
                image_steps: 4,
                cache: Default::default(),
                resources: Default::default(),
                timings: Default::default(),
            }),
            Ok(ServeEvent::Finished {
                candidate_id: CandidateId::PRIMARY,
                reason: FinishStatus::Stop { cause: None },
                finish_detail: Some("image_done".to_string()),
            }),
        ]);

        let chunks = chat_completion_chunk_stream(
            events,
            "req".to_string(),
            "model".to_string(),
            1,
            false,
            true,
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
        .expect("multimodal chunks");
        assert!(chunks.iter().any(|chunk| {
            chunk
                .choices
                .iter()
                .any(|choice| choice.delta.reasoning.as_deref() == Some("plan"))
        }));
        assert!(chunks.iter().any(|chunk| {
            chunk.choices.iter().any(|choice| {
                choice
                    .delta
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
            })
        }));
        assert!(chunks.iter().any(|chunk| {
            chunk
                .choices
                .iter()
                .any(|choice| choice.delta.images.is_some())
        }));
        let usage = chunks
            .iter()
            .find_map(|chunk| chunk.usage.as_ref())
            .unwrap();
        assert_eq!(usage.image_count, Some(1));
        assert_eq!(usage.image_steps, Some(4));
    }

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
            &FinishStatus::Stop {
                cause: Some(uniserve_serving::StopCause::Text("stop".to_string())),
            },
            false,
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].stop_reason, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, &FinishStatus::Length, false)
            .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_maps_abort_finish_reason() {
        let chunk = final_chunk("chatcmpl-1", "model", 1, &FinishStatus::Abort, false)
            .expect("abort is a valid finish reason");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("abort"));
        assert_eq!(chunk.choices[0].stop_reason, None);
    }

    #[test]
    fn final_chunk_rejects_error_finish_reason() {
        assert!(final_chunk("chatcmpl-1", "model", 1, &FinishStatus::Error, false).is_err());
    }

    #[test]
    fn final_chunk_maps_stop_to_tool_calls_when_tool_calls_were_streamed() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            &FinishStatus::Stop { cause: None },
            true,
        )
        .expect("finish reason is valid");

        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    fn serve_stream(
        events: Vec<uniserve_serving::chat::Result<ChatEvent>>,
    ) -> impl futures::Stream<Item = uniserve_serving::Result<ServeEvent>> {
        stream::iter(events.into_iter().flat_map(|event| {
            match event {
                Ok(event) => chat_event_to_serve_events(event)
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(error) => vec![Err(ServeError::Chat(error))],
            }
        }))
    }

    fn chat_event_to_serve_events(event: ChatEvent) -> Vec<ServeEvent> {
        match event {
            ChatEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
                queued_at,
                scheduled_at,
            } => {
                let prompt_token_ids = prompt_token_ids.to_vec();
                vec![
                    ServeEvent::Accepted {
                        request_id: "chatcmpl-test".into(),
                        profile_id: "test-profile".to_string(),
                        dialect_id: "chat".to_string(),
                        compile_duration_us: 7,
                        prompt_token_count: prompt_token_ids.len(),
                        prompt_token_ids,
                        prompt_logprobs,
                    },
                    ServeEvent::Scheduled {
                        request_id: "chatcmpl-test".into(),
                        queued_at,
                        scheduled_at,
                        cache: Default::default(),
                        resources: Default::default(),
                    },
                ]
            }
            ChatEvent::BlockStart { index, kind } => vec![ServeEvent::OutputBlockStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                kind,
            }],
            ChatEvent::BlockDelta { kind, delta, .. } => match kind {
                AssistantBlockKind::Text => vec![ServeEvent::TextDelta {
                    candidate_id: CandidateId::PRIMARY,
                    text: delta,
                    token_ids: Vec::new(),
                    logprobs: None,
                }],
                AssistantBlockKind::Reasoning => vec![ServeEvent::ReasoningDelta {
                    candidate_id: CandidateId::PRIMARY,
                    text: delta,
                }],
                AssistantBlockKind::ToolCall => vec![ServeEvent::InternalTextDelta {
                    candidate_id: CandidateId::PRIMARY,
                    text: delta,
                }],
            },
            ChatEvent::LogprobsDelta {
                logprobs,
                token_ids,
            } => vec![ServeEvent::TextDelta {
                candidate_id: CandidateId::PRIMARY,
                text: String::new(),
                token_ids,
                logprobs,
            }],
            ChatEvent::BlockEnd { index, block } => vec![ServeEvent::OutputBlockEnd {
                candidate_id: CandidateId::PRIMARY,
                index,
                block,
            }],
            ChatEvent::ToolCallStart { index, id, name } => vec![ServeEvent::ToolCallStart {
                candidate_id: CandidateId::PRIMARY,
                index,
                id,
                name,
            }],
            ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                vec![ServeEvent::ToolCallArgumentsDelta {
                    candidate_id: CandidateId::PRIMARY,
                    index,
                    delta,
                }]
            }
            ChatEvent::ToolCallEnd { index, call } => vec![ServeEvent::ToolCallEnd {
                candidate_id: CandidateId::PRIMARY,
                index,
                id: call.id,
                name: call.name,
                arguments: call.arguments,
            }],
            ChatEvent::Done {
                prompt_token_count,
                visible_output_token_count,
                internal_token_count,
                finish_reason,
                ..
            } => vec![
                ServeEvent::Usage {
                    prompt_tokens: prompt_token_count as u32,
                    visible_output_tokens: visible_output_token_count as u32,
                    internal_tokens: internal_token_count as u32,
                    image_count: 0,
                    image_steps: 0,
                    cache: Default::default(),
                    resources: Default::default(),
                    timings: Default::default(),
                },
                ServeEvent::Finished {
                    candidate_id: CandidateId::PRIMARY,
                    reason: FinishStatus::from(&finish_reason),
                    finish_detail: None,
                },
            ],
        }
    }

    #[tokio::test]
    async fn chunk_stream_coalesces_text_delta_with_logprobs() {
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 0,
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
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 0,
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
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 1,
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
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 1,
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
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 3,
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
        let stream = serve_stream(vec![
            Ok(ChatEvent::Start {
                prompt_token_ids: vec![].into(),
                prompt_logprobs: None,
                queued_at: None,
                scheduled_at: None,
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
                visible_output_token_count: 1,
                internal_token_count: 0,
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
