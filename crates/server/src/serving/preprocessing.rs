//! Chat API validation, message normalization, and model preprocessing.

use crate::profile::tools;
use crate::serving::chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as ServingChatMessage, ChatToolChoice, ReasoningEffort as ServingReasoningEffort,
};
use crate::serving::{
    DecodeControls, ImageGenControls, ModalitySelection, OutputDetail, PromptInput, SamplingConfig,
    ServeRequestId, StopConfig,
};
use itertools::Itertools as _;
use uniserve_core::CachePolicy;

use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::utils::convert_logit_bias;
use crate::openai::{
    ChatCompletionRequest, ChatImageConfig, ChatMessage, ChatModality, ContentPart, MessageContent,
    ReasoningEffort, Tool, ToolCall, ToolChoice, ToolChoiceValue,
};

impl crate::serving::InputProcessor {
    /// Validates and tokenizes a chat API request into its final engine input.
    pub fn preprocess_chat_request(
        &self,
        request_id: ServeRequestId,
        request: ChatCompletionRequest,
    ) -> Result<
        (
            uniserve_core::GenerationRequest,
            crate::serving::ResponseOptions,
        ),
        ApiError,
    > {
        crate::openai::chat_completions::validate_request_compat(
            &request,
            self.served_model_name(),
        )?;

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

        // Normalize public chat content and tool declarations into the serving
        // layer's closed prompt representation.
        let modalities = convert_modalities(&request.modalities);
        let image_gen = request.image_config.as_ref().map(convert_image_config);
        let messages = request
            .messages
            .into_iter()
            .map(convert_message)
            .try_collect()?;
        let prompt = PromptInput::Chat(crate::serving::chat::ChatRequest {
            messages,
            tools: convert_tools(request.tools)?,
            tool_choice: convert_tool_choice(request.tool_choice),
            chat_options: crate::serving::chat::ChatOptions {
                reasoning_effort: request.reasoning_effort.map(convert_reasoning_effort),
                ..Default::default()
            },
            decode_options: Default::default(),
        });

        self.preprocess_generation(
            request_id,
            prompt,
            Vec::new(),
            modalities,
            SamplingConfig {
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
            StopConfig {
                stop_token_ids: request.stop_token_ids.unwrap_or_default(),
                stop_strings: request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
                bad_words: request.bad_words.unwrap_or_default(),
                allowed_token_ids: request.allowed_token_ids,
                logit_bias: convert_logit_bias(request.logit_bias)?,
                logprobs: requested_logprobs.then_some(request.top_logprobs.unwrap_or(0)),
                prompt_logprobs,
                logprob_token_ids: None,
            },
            None,
            image_gen,
            CachePolicy {
                isolation_key: crate::serving::cache_isolation_key(
                    None,
                    request.cache_salt.as_deref(),
                ),
                ..CachePolicy::default()
            },
            request.priority.unwrap_or(0),
            output,
            DecodeControls {
                skip_special_tokens: request.skip_special_tokens,
                include_stop_string_in_output: request.include_stop_str_in_output,
            },
        )
        .map_err(crate::openai::serve_error_to_api)
    }
}

/// Converts requested response modalities into the serving selection.
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

/// Converts the image config.
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

/// Converts the reasoning effort.
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

/// Converts one wire chat message into the serving layer's structured message model.
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

/// Converts OpenAI message content into serving chat content.
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

/// Converts the assistant content.
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

/// Validates and converts assistant function calls into structured content blocks.
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

/// Converts OpenAI function tools into serving tool definitions.
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

/// Converts the message tools.
fn convert_message_tools(tools: Option<Vec<Tool>>) -> Result<Option<Vec<tools::Tool>>, ApiError> {
    let tools = convert_tools(tools)?;
    Ok((!tools.is_empty()).then_some(tools))
}

/// Converts the tool choice.
fn convert_tool_choice(tool_choice: Option<ToolChoice>) -> ChatToolChoice {
    match tool_choice.map(|choice| choice.0) {
        None | Some(ToolChoiceValue::Auto) => ChatToolChoice::Auto,
        Some(ToolChoiceValue::None) => ChatToolChoice::None,
    }
}

impl crate::serving::InputProcessor {
    /// Validates and tokenizes an image API request into its final engine input.
    pub fn preprocess_image_request(
        &self,
        request_id: ServeRequestId,
        request: crate::openai::ImageGenerationRequest,
    ) -> Result<
        (
            uniserve_core::GenerationRequest,
            crate::serving::ResponseOptions,
        ),
        ApiError,
    > {
        if request.prompt.trim().is_empty() {
            return Err(ApiError::invalid_request(
                "prompt must not be empty".to_string(),
                Some("prompt"),
            ));
        }
        if request.n != 1 {
            return Err(ApiError::invalid_request(
                "n must be 1 for the configured image-generation route".to_string(),
                Some("n"),
            ));
        }
        if let Some(model) = request.model.as_deref() {
            crate::openai::utils::check_model_served(model, self.served_model_name())?;
        }
        if request.steps == Some(0) {
            return Err(ApiError::invalid_request(
                "steps must be positive".to_string(),
                Some("steps"),
            ));
        }
        let (width, height) = request
            .size
            .as_deref()
            .map(parse_size)
            .transpose()?
            .map_or((None, None), |(width, height)| (Some(width), Some(height)));
        self.preprocess_generation(
            request_id,
            PromptInput::Text(request.prompt),
            Vec::new(),
            ModalitySelection::Image,
            SamplingConfig::default(),
            StopConfig::default(),
            request.negative_prompt,
            Some(ImageGenControls {
                width,
                height,
                steps: request.steps,
                cfg_text_scale: request.guidance_scale,
                cfg_img_scale: request.image_guidance_scale,
                cfg_interval: request.cfg_interval,
                cfg_renorm_type: request.cfg_norm,
                timestep_shift: request.timestep_shift,
                seed: request.seed,
                max_images: Some(1),
                ..ImageGenControls::default()
            }),
            CachePolicy::default(),
            0,
            OutputDetail::VisibleText,
            DecodeControls::default(),
        )
        .map_err(crate::openai::serve_error_to_api)
    }
}

/// Parses a positive `WIDTHxHEIGHT` image-size string.
fn parse_size(size: &str) -> Result<(u32, u32), ApiError> {
    let Some((width, height)) = size.split_once('x') else {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    };
    if height.contains('x') {
        return Err(ApiError::invalid_request(
            "size must use WIDTHxHEIGHT syntax".to_string(),
            Some("size"),
        ));
    }
    let width = width.parse::<u32>().ok().filter(|value| *value > 0);
    let height = height.parse::<u32>().ok().filter(|value| *value > 0);
    width.zip(height).ok_or_else(|| {
        ApiError::invalid_request(
            "size dimensions must be positive integers".to_string(),
            Some("size"),
        )
    })
}
