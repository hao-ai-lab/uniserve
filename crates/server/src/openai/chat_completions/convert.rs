use crate::profile::tools;
use crate::serving::chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as ServingChatMessage, ChatToolChoice, ReasoningEffort as ServingReasoningEffort,
};
use crate::serving::{
    CacheBounds, DecodeControls, GenerateReqInput, ImageGenControls, ModalitySelection,
    OutputDetail, PromptInput, SamplingConfig, SchedulingBounds, ServeRequestId, StopConfig,
};
use itertools::Itertools as _;

use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::types::{
    ChatCompletionRequest, ChatImageConfig, ChatMessage, ChatModality, ContentPart, MessageContent,
    ReasoningEffort, Tool, ToolCall, ToolChoice, ToolChoiceValue,
};
use crate::openai::utils::{ResolvedRequestContext, convert_logit_bias};

use super::validate;

/// Public response metadata retained after the wire request has been lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponseContext {
    pub request_id: String,
    pub response_model: String,
    pub include_usage: bool,
    pub requested_logprobs: bool,
    pub include_prompt_logprobs: bool,
    pub include_reasoning: bool,
    pub return_token_ids: bool,
    pub return_tokens_as_token_ids: bool,
}

/// Lower one validated wire request into the sole generate admission value and
/// response-only metadata.
pub fn lower_chat_request(
    request: ChatCompletionRequest,
    served_model_name: &str,
    context: ResolvedRequestContext,
) -> Result<(GenerateReqInput, ChatResponseContext), ApiError> {
    validate::validate_request_compat(&request, served_model_name)?;
    let response_model = served_model_name.to_owned();
    let request_id = format!("chatcmpl-{}", context.request_id);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let requested_logprobs = request.logprobs;
    let prompt_logprobs = request.prompt_logprobs;
    let include_prompt_logprobs = prompt_logprobs.is_some();
    let return_token_ids = request.return_token_ids.unwrap_or(false);
    let output = if requested_logprobs || include_prompt_logprobs {
        OutputDetail::Logprobs
    } else if return_token_ids {
        OutputDetail::Tokens
    } else {
        OutputDetail::VisibleText
    };
    let modalities = convert_modalities(&request.modalities);
    let image_gen = request.image_config.as_ref().map(convert_image_config);
    let messages = request
        .messages
        .into_iter()
        .map(convert_message)
        .try_collect()?;
    let prompt = PromptInput::Chat {
        messages,
        tools: convert_tools(request.tools)?,
        tool_choice: convert_tool_choice(request.tool_choice),
        reasoning_effort: request.reasoning_effort.map(convert_reasoning_effort),
    };
    let input = GenerateReqInput {
        request_id: ServeRequestId::from(request_id.clone()),
        stream: request.stream,
        prompt,
        images: Vec::new(),
        modalities,
        sampling: SamplingConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            min_p: request.min_p,
            seed: request.seed,
            max_tokens: request.max_completion_tokens,
            min_tokens: request.min_tokens,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            ignore_eos: request.ignore_eos,
        },
        stop: StopConfig {
            stop_token_ids: request.stop_token_ids.unwrap_or_default(),
            stop_strings: request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
            bad_words: request.bad_words.unwrap_or_default(),
            allowed_token_ids: request.allowed_token_ids,
            logit_bias: convert_logit_bias(request.logit_bias)?,
            logprobs: requested_logprobs.then_some(request.top_logprobs.unwrap_or(0)),
            prompt_logprobs,
            logprob_token_ids: None,
        },
        negative_text: None,
        image_gen,
        cache: CacheBounds {
            namespace: None,
            salt: request.cache_salt,
            bypass_read: false,
            no_store: false,
        },
        scheduling: SchedulingBounds {
            priority: request.priority.unwrap_or(0),
            trace_context: context.trace_context.into_iter().collect(),
        },
        output,
        decode: DecodeControls {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_string_in_output: request.include_stop_str_in_output,
        },
    };
    let response = ChatResponseContext {
        request_id,
        response_model,
        include_usage,
        requested_logprobs,
        include_prompt_logprobs,
        include_reasoning: request.include_reasoning,
        return_token_ids,
        return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
    };
    Ok((input, response))
}

fn convert_modalities(modalities: &[ChatModality]) -> ModalitySelection {
    match (
        modalities.contains(&ChatModality::Text),
        modalities.contains(&ChatModality::Image),
    ) {
        (true, false) => ModalitySelection::Text,
        (false, true) => ModalitySelection::Image,
        (true, true) | (false, false) => ModalitySelection::TextAndImage,
    }
}

fn convert_image_config(config: &ChatImageConfig) -> ImageGenControls {
    ImageGenControls {
        resolution: config.resolution.clone(),
        width: config.width,
        height: config.height,
        steps: config.steps,
        cfg_text_scale: config.guidance_scale,
        cfg_img_scale: config.image_guidance_scale,
        cfg_interval: config.cfg_interval,
        cfg_renorm_type: config.cfg_norm.clone(),
        cfg_renorm_min: config.cfg_renorm_min,
        timestep_shift: config.timestep_shift,
        seed: config.seed,
        max_images: config.num_images,
        prompts: Vec::new(),
        retain_images: None,
    }
}

fn convert_reasoning_effort(value: ReasoningEffort) -> ServingReasoningEffort {
    match value {
        ReasoningEffort::None => ServingReasoningEffort::None,
        ReasoningEffort::Minimal => ServingReasoningEffort::Minimal,
        ReasoningEffort::Low => ServingReasoningEffort::Low,
        ReasoningEffort::Medium => ServingReasoningEffort::Medium,
        ReasoningEffort::High => ServingReasoningEffort::High,
        ReasoningEffort::XHigh => ServingReasoningEffort::XHigh,
        ReasoningEffort::Max => ServingReasoningEffort::Max,
    }
}

fn convert_message(message: ChatMessage) -> Result<ServingChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => {
            Ok(ServingChatMessage::system(convert_content(content)?))
        }
        ChatMessage::User { content, .. } => {
            Ok(ServingChatMessage::user(convert_content(content)?))
        }
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } => {
            let mut blocks = Vec::new();
            if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
                blocks.push(AssistantContentBlock::Reasoning { text: reasoning });
            }
            if let Some(content) = content {
                blocks.extend(convert_assistant_content(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                blocks.extend(convert_assistant_tool_calls(tool_calls)?);
            }
            if blocks.is_empty() {
                bail_invalid_request!("Assistant messages must contain content or tool_calls.");
            }
            Ok(ServingChatMessage::assistant_blocks(blocks))
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => Ok(ServingChatMessage::tool_response(
            convert_content(content)?,
            tool_call_id,
        )),
        ChatMessage::Developer { content, tools, .. } => Ok(ServingChatMessage::developer(
            convert_content(content)?,
            convert_message_tools(tools)?,
        )),
    }
}

fn convert_content(content: MessageContent) -> Result<ChatContent, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(ChatContent::Text(text)),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(ChatContentPart::text(text)),
                ContentPart::ImageUrl { image_url, uuid } => Ok(ChatContentPart::ImageUrl {
                    image_url: image_url.url,
                    detail: image_url.detail,
                    uuid,
                }),
            })
            .try_collect()
            .map(ChatContent::Parts),
    }
}

fn convert_assistant_content(
    content: MessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(vec![AssistantContentBlock::Text { text }]),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AssistantContentBlock::Text { text }),
                ContentPart::ImageUrl { .. } => {
                    bail_invalid_request!("Assistant message content must be text.")
                }
            })
            .collect(),
    }
}

fn convert_assistant_tool_calls(
    tool_calls: Vec<ToolCall>,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    tool_calls
        .into_iter()
        .map(|tool_call| {
            if tool_call.tool_type != "function" {
                bail_invalid_request!("Only function tool calls are supported.");
            }
            Ok(AssistantContentBlock::ToolCall(AssistantToolCall {
                id: tool_call.id,
                name: tool_call.function.name,
                arguments: tool_call
                    .function
                    .arguments
                    .unwrap_or_else(|| "{}".to_string()),
            }))
        })
        .collect()
}

fn convert_tools(tools: Option<Vec<Tool>>) -> Result<Vec<tools::Tool>, ApiError> {
    tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| {
            let Tool::Function { function } = tool;
            Ok(tools::Tool {
                name: function.name,
                description: function.description,
                parameters: function.parameters,
                strict: None,
            })
        })
        .collect()
}

fn convert_message_tools(tools: Option<Vec<Tool>>) -> Result<Option<Vec<tools::Tool>>, ApiError> {
    let tools = convert_tools(tools)?;
    Ok((!tools.is_empty()).then_some(tools))
}

fn convert_tool_choice(tool_choice: Option<ToolChoice>) -> ChatToolChoice {
    match tool_choice.map(|choice| choice.0) {
        None | Some(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
        Some(ToolChoiceValue::None) => ChatToolChoice::None,
    }
}
