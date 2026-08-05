use itertools::Itertools as _;
use uniserve_openai_types::{
    ChatCompletionRequest, ChatImageConfig, ChatImageType, ChatMessage, ChatModality, ContentPart,
    MessageContent, ReasoningEffort as OpenAiReasoningEffort, Tool, ToolCall, ToolChoice,
    ToolChoiceValue,
};
use uniserve_serving::chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as UniserveChatMessage, ChatRole, ChatTool, ChatToolChoice, GenerationPromptMode,
    ReasoningEffort,
};
use uniserve_serving::{
    CacheBounds, DecodeControls, GenerateReqInput, ImageGenControls, ModalitySelection,
    OutputContract, PromptInput, SamplingConfig, SchedulingBounds, ServeRequestId, StopConfig,
};

use super::validate;
use crate::openai::error::{ApiError, bail_invalid_request};
use crate::openai::utils::{ResolvedRequestContext, convert_logit_bias};

/// One lowered chat request: the sole [`GenerateReqInput`] plus the public
/// response metadata carried by response assembly and every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedChatRequest {
    /// Stable OpenAI-style request ID, reused as the external chat request ID.
    pub request_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether the caller requested output logprobs on chat choices.
    pub requested_logprobs: bool,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Whether to include reasoning content in OpenAI responses.
    pub include_reasoning: bool,
    /// The single canonical admission value submitted to the serving runtime.
    pub input: GenerateReqInput,
    /// Last assistant-role message content to echo back when `echo=true`.
    pub echo: Option<String>,
    /// Whether to include token IDs alongside generated text.
    pub return_token_ids: bool,
    /// Whether to format logprob tokens as `token_id:{id}`.
    pub return_tokens_as_token_ids: bool,
}

/// Validate and lower one OpenAI chat completion request into exactly one
/// [`GenerateReqInput`] plus the response metadata needed to assemble the
/// OpenAI response.
///
/// `served_model_names` must be non-empty; the first entry is used as the
/// `model` field echoed back in responses.
pub fn prepare_chat_request(
    request: ChatCompletionRequest,
    served_model_names: &[String],
    ctx: ResolvedRequestContext,
) -> Result<PreparedChatRequest, ApiError> {
    validate::validate_request_compat(&request, served_model_names)?;
    if request
        .kv_transfer_params
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(ApiError::invalid_request(
            "`kv_transfer_params` is not supported by this runtime.".to_string(),
            Some("kv_transfer_params"),
        ));
    }
    if request
        .uniserve_xargs
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(ApiError::invalid_request(
            "`uniserve_xargs` is not supported; use typed request fields.".to_string(),
            Some("uniserve_xargs"),
        ));
    }

    let request_id = format!("chatcmpl-{}", ctx.request_id);
    let response_model = served_model_names.first().cloned().ok_or_else(|| {
        ApiError::server_error("chat completion has no served model configured".to_string())
    })?;
    let include_reasoning = request.include_reasoning;
    let echo = request
        .echo
        .then(|| extract_last_assistant_content(&request.messages))
        .flatten();

    let modalities = convert_modalities(&request.modalities)?;
    let image_gen = request
        .image_config
        .as_ref()
        .map(convert_image_config)
        .transpose()?;

    let messages: Vec<_> = request
        .messages
        .into_iter()
        .map(convert_message)
        .try_collect()?;
    let generation_prompt_mode = normalize_generation_prompt_mode(
        request.add_generation_prompt,
        request.continue_final_message,
        &messages,
    )?;

    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let requested_logprobs = request.logprobs;

    // Auto-enable prompt logprobs for non-streaming echo, matching the reference's
    // behavior.
    let top_logprobs = request.top_logprobs.unwrap_or(0);
    let prompt_logprobs = request
        .prompt_logprobs
        .or((request.echo && !request.stream).then_some(top_logprobs));
    let include_prompt_logprobs = prompt_logprobs.is_some();
    let logprobs = request.logprobs.then_some(top_logprobs);

    let return_token_ids = request.return_token_ids.unwrap_or(false);
    let output = if requested_logprobs || include_prompt_logprobs {
        OutputContract::Logprobs
    } else if return_token_ids {
        OutputContract::Tokens
    } else {
        OutputContract::VisibleText
    };

    let prompt = PromptInput::Chat {
        messages,
        tools: convert_tools(request.tools)?,
        tool_choice: convert_tool_choice(request.tool_choice.as_ref())?,
        generation_prompt_mode,
        reasoning_effort: request.reasoning_effort.map(convert_reasoning_effort),
    };

    let sampling = SamplingConfig {
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
    };
    let stop = StopConfig {
        stop_token_ids: request.stop_token_ids.unwrap_or_default(),
        stop_strings: request.stop.map(|stop| stop.into_vec()).unwrap_or_default(),
        bad_words: request.bad_words.unwrap_or_default(),
        allowed_token_ids: request.allowed_token_ids,
        logit_bias: convert_logit_bias(request.logit_bias)?,
        logprobs,
        prompt_logprobs,
        logprob_token_ids: None,
    };

    let input = GenerateReqInput {
        request_id: ServeRequestId::from(request_id.clone()),
        stream: request.stream,
        prompt,
        images: Vec::new(),
        modalities,
        sampling,
        stop,
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
            data_parallel_rank: ctx.data_parallel_rank,
            trace_context: ctx.trace_context.into_iter().collect(),
        },
        output,
        decode: DecodeControls {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_string_in_output: request.include_stop_str_in_output,
            add_special_tokens: request.add_special_tokens,
        },
    };

    Ok(PreparedChatRequest {
        request_id,
        response_model,
        include_usage,
        requested_logprobs,
        include_prompt_logprobs,
        include_reasoning,
        input,
        echo,
        return_token_ids,
        return_tokens_as_token_ids: request.return_tokens_as_token_ids.unwrap_or(false),
    })
}

/// Lower requested output modalities into the canonical [`ModalitySelection`].
fn convert_modalities(modalities: &[ChatModality]) -> Result<ModalitySelection, ApiError> {
    let mut output_text = false;
    let mut output_image = false;
    for modality in modalities {
        match modality {
            ChatModality::Text => output_text = true,
            ChatModality::Image => output_image = true,
            ChatModality::Audio => {
                bail_invalid_request!(param = "modalities", "audio output is not supported.")
            }
        }
    }
    // An empty modality list falls back to text output, matching the OpenAI
    // default of `modalities = ["text"]`.
    if !output_text && !output_image {
        return Ok(ModalitySelection::default());
    }
    Ok(ModalitySelection {
        output_text,
        output_image,
    })
}

/// Lower OpenAI chat image controls into the canonical [`ImageGenControls`].
fn convert_image_config(config: &ChatImageConfig) -> Result<ImageGenControls, ApiError> {
    if let Some(image_type) = config.image_type
        && image_type != ChatImageType::Png
    {
        bail_invalid_request!(
            param = "image_config",
            "image_type must be png for image chat completions."
        );
    }
    Ok(ImageGenControls {
        resolution: config.resolution.clone(),
        width: positive_dimension(config.width),
        height: positive_dimension(config.height),
        steps: config.steps,
        cfg_text_scale: config.guidance_scale,
        cfg_img_scale: config.image_guidance_scale,
        cfg_interval: config.cfg_interval,
        cfg_renorm_type: config.cfg_norm.clone(),
        cfg_renorm_min: None,
        timestep_shift: config.timestep_shift,
        seed: config.seed,
        max_images: config.num_images,
        prompts: Vec::new(),
        retain_images: None,
    })
}

fn positive_dimension(value: Option<i32>) -> Option<u32> {
    value.and_then(|value| (value > 0).then_some(value as u32))
}

fn convert_reasoning_effort(value: OpenAiReasoningEffort) -> ReasoningEffort {
    match value {
        OpenAiReasoningEffort::None => ReasoningEffort::None,
        OpenAiReasoningEffort::Minimal => ReasoningEffort::Minimal,
        OpenAiReasoningEffort::Low => ReasoningEffort::Low,
        OpenAiReasoningEffort::Medium => ReasoningEffort::Medium,
        OpenAiReasoningEffort::High => ReasoningEffort::High,
        OpenAiReasoningEffort::XHigh => ReasoningEffort::XHigh,
        OpenAiReasoningEffort::Max => ReasoningEffort::Max,
    }
}

fn normalize_generation_prompt_mode(
    add_generation_prompt: Option<bool>,
    continue_final_message: bool,
    messages: &[UniserveChatMessage],
) -> Result<GenerationPromptMode, ApiError> {
    if add_generation_prompt == Some(true) && continue_final_message {
        bail_invalid_request!(
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        );
    }

    let last_role = messages.last().map(UniserveChatMessage::role);
    match (add_generation_prompt, continue_final_message, last_role) {
        (Some(true), true, _) => unreachable!("rejected above"),
        (_, true, Some(ChatRole::Assistant)) => Ok(GenerationPromptMode::ContinueFinalAssistant),
        (_, true, _) => {
            bail_invalid_request!(
                "Cannot set `continue_final_message` to True when the last message is not from the assistant."
            );
        }
        (Some(false), false, _) => Ok(GenerationPromptMode::NoGenerationPrompt),
        (None | Some(true), false, _) => Ok(GenerationPromptMode::StartNewAssistant),
    }
}

/// Extract the text content of the last message if it has the assistant role.
fn extract_last_assistant_content(messages: &[ChatMessage]) -> Option<String> {
    let ChatMessage::Assistant { content, .. } = messages.last()? else {
        return None;
    };
    let text = match content.as_ref()? {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    (!text.is_empty()).then_some(text)
}

/// Lower one OpenAI chat message into the `chat` message shape.
fn convert_message(message: ChatMessage) -> Result<UniserveChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => {
            Ok(UniserveChatMessage::system(convert_content(content)?))
        }
        ChatMessage::User { content, .. } => {
            Ok(UniserveChatMessage::user(convert_content(content)?))
        }
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            name: _,
        } => {
            let mut blocks = Vec::new();
            if let Some(reasoning) = reasoning
                && !reasoning.is_empty()
            {
                blocks.push(AssistantContentBlock::Reasoning { text: reasoning });
            }
            if let Some(content) = content {
                blocks.extend(convert_assistant_text_blocks(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                blocks.extend(convert_assistant_tool_calls(tool_calls)?);
            }
            if blocks.is_empty() {
                bail_invalid_request!(
                    "Assistant messages must contain text, reasoning content, or tool_calls."
                );
            }

            Ok(UniserveChatMessage::assistant_blocks(blocks))
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => Ok(UniserveChatMessage::tool_response(
            convert_content(content)?,
            tool_call_id,
        )),
        ChatMessage::Function { .. } => {
            bail_invalid_request!("Function messages are not supported.")
        }
        ChatMessage::Developer {
            content,
            tools,
            name: _,
        } => Ok(UniserveChatMessage::developer(
            convert_content(content)?,
            convert_message_tools(tools)?,
        )),
    }
}

/// Convert the given OpenAI message content value into the internal format in
/// `chat`.
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
                _ => bail_invalid_request!("Only text and image_url content parts are supported."),
            })
            .try_collect()
            .map(ChatContent::Parts),
    }
}

/// Convert the given OpenAI assistant message content into the internal format
/// in `chat`.
fn convert_assistant_text_blocks(
    content: MessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(vec![AssistantContentBlock::Text { text }]),
        MessageContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(AssistantContentBlock::Text { text }),
                _ => bail_invalid_request!(
                    "Only text content parts are supported for assistant messages."
                ),
            })
            .try_collect(),
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

fn convert_tools(tools: Option<Vec<Tool>>) -> Result<Vec<ChatTool>, ApiError> {
    tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                bail_invalid_request!("Only function tools are supported.");
            }
            Ok(ChatTool {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            })
        })
        .collect()
}

fn convert_message_tools(tools: Option<Vec<Tool>>) -> Result<Option<Vec<ChatTool>>, ApiError> {
    let tools = convert_tools(tools)?;
    Ok((!tools.is_empty()).then_some(tools))
}

fn convert_tool_choice(tool_choice: Option<&ToolChoice>) -> Result<ChatToolChoice, ApiError> {
    match tool_choice {
        None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => Ok(ChatToolChoice::Auto),
        Some(ToolChoice::Value(ToolChoiceValue::None)) => Ok(ChatToolChoice::None),
        _ => bail_invalid_request!("tool_choice={:?} is not supported yet.", tool_choice),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use expect_test::expect;
    use llm_multimodal::ImageDetail;
    use serde_json::json;
    use uniserve_openai_types::{
        ChatCompletionRequest, ChatMessage, ContentPart, Function, FunctionCallResponse, ImageUrl,
        MessageContent, Tool, ToolCall, ToolChoice, ToolChoiceValue, VideoUrl,
    };
    use uniserve_serving::chat::{
        AssistantContentBlock, AssistantToolCall, ChatContentPart,
        ChatMessage as UniserveChatMessage, ChatTool as UniserveChatTool, ChatToolChoice,
        GenerationPromptMode,
    };
    use uniserve_serving::{GenerateReqInput, PromptInput, SamplingConfig, StopConfig};

    use super::{PreparedChatRequest, prepare_chat_request};
    use crate::openai::utils::ResolvedRequestContext;

    fn served(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn base_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "Qwen/Qwen1.5-0.5B-Chat".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hello".to_string()),
                name: None,
            }],
            stream: true,
            ..Default::default()
        }
    }

    fn prepared(request: ChatCompletionRequest) -> PreparedChatRequest {
        prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .expect("request is valid")
    }

    fn chat_context(
        input: &GenerateReqInput,
    ) -> (
        &[UniserveChatMessage],
        GenerationPromptMode,
        &[UniserveChatTool],
        &ChatToolChoice,
    ) {
        let PromptInput::Chat {
            messages,
            generation_prompt_mode,
            tools,
            tool_choice,
            ..
        } = &input.prompt
        else {
            panic!("OpenAI chat adapter must produce a chat prompt");
        };
        (messages, *generation_prompt_mode, tools, tool_choice)
    }

    fn assert_sampling_matches(sampling: &SamplingConfig, expected: &SamplingConfig) {
        assert_eq!(sampling, expected);
    }

    #[test]
    fn prepare_chat_request_maps_text_parts() {
        let mut request = base_request();
        request.messages = vec![ChatMessage::Assistant {
            content: Some(MessageContent::Parts(vec![ContentPart::Text {
                text: "hello".to_string(),
            }])),
            name: None,
            tool_calls: None,
            reasoning: None,
        }];
        request.add_generation_prompt = Some(false);
        request.continue_final_message = true;
        request.skip_special_tokens = false;
        let prepared = prepared(request);

        assert!(prepared.request_id.starts_with("chatcmpl-"));
        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::assistant_text("hello")]
        );
        assert_sampling_matches(&prepared.input.sampling, &SamplingConfig::default());
        assert_eq!(
            chat_context(&prepared.input).1,
            GenerationPromptMode::ContinueFinalAssistant
        );
        assert!(!prepared.input.decode.skip_special_tokens);
        assert_eq!(prepared.input.stop, StopConfig::default());
        assert!(chat_context(&prepared.input).2.is_empty());
        assert_eq!(*chat_context(&prepared.input).3, ChatToolChoice::Auto);
    }

    #[test]
    fn prepare_chat_request_keeps_optional_sampling_fields_unset() {
        let prepared = prepared(base_request());

        assert!(prepared.request_id.starts_with("chatcmpl-"));
        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::user("hello")]
        );
        assert_sampling_matches(&prepared.input.sampling, &SamplingConfig::default());
        assert_eq!(
            chat_context(&prepared.input).1,
            GenerationPromptMode::StartNewAssistant
        );
        assert!(prepared.input.decode.skip_special_tokens);
        assert!(chat_context(&prepared.input).2.is_empty());
        assert_eq!(*chat_context(&prepared.input).3, ChatToolChoice::Auto);
    }

    #[test]
    fn prepare_chat_request_echoes_first_served_model_name() {
        let prepared = prepared(base_request());
        assert_eq!(prepared.response_model, "Qwen/Qwen1.5-0.5B-Chat");
    }

    #[test]
    fn prepare_chat_request_preserves_include_reasoning_false() {
        let request = ChatCompletionRequest {
            include_reasoning: false,
            ..base_request()
        };

        let prepared = prepared(request);

        assert!(!prepared.include_reasoning);
    }

    #[test]
    fn prepare_chat_request_preserves_sampling_passthrough_fields() {
        let request = ChatCompletionRequest {
            seed: Some(42),
            min_p: Some(0.2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            ..base_request()
        };

        let prepared = prepared(request);
        let expected = SamplingConfig {
            seed: Some(42),
            min_p: Some(0.2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            ..SamplingConfig::default()
        };
        assert_sampling_matches(&prepared.input.sampling, &expected);
    }

    #[test]
    fn prepare_chat_request_maps_stop_and_sampling_controls() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "min_tokens": 5,
            "ignore_eos": true,
            "logit_bias": {"151670": 80.0},
            "allowed_token_ids": [7, 11],
            "bad_words": ["blocked"],
            "stop": ["END"],
            "stop_token_ids": [3, 4]
        }))
        .unwrap();

        let prepared = prepared(request);

        assert_eq!(prepared.input.sampling.min_tokens, Some(5));
        assert!(prepared.input.sampling.ignore_eos);
        assert_eq!(
            prepared.input.stop.logit_bias,
            Some(HashMap::from([(151670, 80.0)]))
        );
        assert_eq!(prepared.input.stop.allowed_token_ids, Some(vec![7, 11]));
        assert_eq!(prepared.input.stop.bad_words, vec!["blocked".to_string()]);
        assert_eq!(prepared.input.stop.stop_strings, vec!["END".to_string()]);
        assert_eq!(prepared.input.stop.stop_token_ids, vec![3, 4]);
    }

    #[test]
    fn prepare_chat_request_maps_image_output_modality_and_controls() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "modalities": ["text", "image"],
            "messages": [{"role": "user", "content": "draw a cat"}],
            "image_config": {
                "width": 640,
                "height": 480,
                "steps": 12,
                "guidance_scale": 4.0,
                "seed": 7
            }
        }))
        .unwrap();

        let prepared = prepared(request);

        assert!(prepared.input.modalities.output_text);
        assert!(prepared.input.modalities.output_image);
        let image_gen = prepared.input.image_gen.expect("image controls");
        assert_eq!(image_gen.width, Some(640));
        assert_eq!(image_gen.height, Some(480));
        assert_eq!(image_gen.steps, Some(12));
        assert_eq!(image_gen.cfg_text_scale, Some(4.0));
        assert_eq!(image_gen.seed, Some(7));
    }

    #[test]
    fn prepare_chat_request_rejects_audio_modality() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "modalities": ["audio"],
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();

        let error = prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .unwrap_err();
        expect!["audio output is not supported."]
            .assert_eq(&error.to_error_response().error.message);
    }

    #[test]
    fn prepare_chat_request_accepts_developer_messages() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Developer {
                content: MessageContent::Text("hello".to_string()),
                tools: Some(vec![Tool {
                    tool_type: "function".to_string(),
                    function: Function {
                        name: "get_weather".to_string(),
                        description: Some("Get weather".to_string()),
                        parameters: json!({
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        }),
                        strict: Some(true),
                    },
                }]),
                name: None,
            }],
            ..base_request()
        };

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::developer(
                "hello",
                Some(vec![UniserveChatTool {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }),
                    strict: Some(true),
                }]),
            )]
        );
    }

    #[test]
    fn prepare_chat_request_maps_image_url_content_parts() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "describe ".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.png".to_string(),
                            detail: Some(ImageDetail::Low),
                        },
                        uuid: Some("image-1".to_string()),
                    },
                    ContentPart::Text {
                        text: " briefly".to_string(),
                    },
                ]),
                name: None,
            }],
            ..base_request()
        };

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::user(vec![
                ChatContentPart::text("describe "),
                ChatContentPart::ImageUrl {
                    image_url: "https://example.com/image.png".to_string(),
                    detail: Some(ImageDetail::Low),
                    uuid: Some("image-1".to_string()),
                },
                ChatContentPart::text(" briefly"),
            ])]
        );
    }

    #[test]
    fn prepare_chat_request_maps_developer_image_url_content_parts() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Developer {
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                    uuid: None,
                }]),
                tools: None,
                name: None,
            }],
            ..base_request()
        };

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::developer(
                vec![ChatContentPart::image_url("https://example.com/image.png")],
                None,
            )]
        );
    }

    #[test]
    fn prepare_chat_request_rejects_video_content_parts() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::User {
                content: MessageContent::Parts(vec![ContentPart::VideoUrl {
                    video_url: VideoUrl {
                        url: "https://example.com/video.mp4".to_string(),
                    },
                }]),
                name: None,
            }],
            ..base_request()
        };

        let error = prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .unwrap_err();

        expect!["Only text and image_url content parts are supported."]
            .assert_eq(&error.to_error_response().error.message);
    }

    #[test]
    fn prepare_chat_request_rejects_assistant_image_url_content_parts() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Assistant {
                content: Some(MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                    uuid: None,
                }])),
                name: None,
                tool_calls: None,
                reasoning: None,
            }],
            ..base_request()
        };

        let error = prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .unwrap_err();

        expect!["Only text content parts are supported for assistant messages."]
            .assert_eq(&error.to_error_response().error.message);
    }

    #[test]
    fn prepare_chat_request_preserves_assistant_reasoning_history() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Assistant {
                content: Some(MessageContent::Text("answer".to_string())),
                name: None,
                tool_calls: None,
                reasoning: Some("inner".to_string()),
            }],
            add_generation_prompt: Some(false),
            ..base_request()
        };

        let prepared = prepared(request);
        assert_eq!(
            chat_context(&prepared.input).0,
            vec![UniserveChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "inner".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "answer".to_string(),
                },
            ])]
        );
        assert!(chat_context(&prepared.input).2.is_empty());
        assert_eq!(*chat_context(&prepared.input).3, ChatToolChoice::Auto);
    }

    #[test]
    fn prepare_chat_request_accepts_tools_and_tool_history() {
        let request = ChatCompletionRequest {
            messages: vec![
                ChatMessage::Assistant {
                    content: None,
                    name: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: "get_weather".to_string(),
                            arguments: Some(r#"{"city":"Paris"}"#.to_string()),
                        },
                    }]),
                    reasoning: None,
                },
                ChatMessage::Tool {
                    content: MessageContent::Text("Sunny".to_string()),
                    tool_call_id: "call_1".to_string(),
                },
            ],
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::None)),
            ..base_request()
        };

        let prepared = prepared(request);
        assert_eq!(
            chat_context(&prepared.input).0,
            vec![
                UniserveChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(
                    AssistantToolCall {
                        id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"Paris"}"#.to_string(),
                    },
                )]),
                UniserveChatMessage::tool_response("Sunny", "call_1"),
            ]
        );
        assert_eq!(
            chat_context(&prepared.input).2,
            vec![UniserveChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: None,
            }]
        );
        assert_eq!(*chat_context(&prepared.input).3, ChatToolChoice::None);
    }

    #[test]
    fn prepare_chat_request_lowers_logprobs_fields() {
        let request = ChatCompletionRequest {
            stream: false,
            logprobs: true,
            prompt_logprobs: Some(2),
            ..base_request()
        };

        let prepared = prepared(request);

        assert!(prepared.requested_logprobs);
        assert!(prepared.include_prompt_logprobs);
        assert_eq!(prepared.input.stop.logprobs, Some(0));
        assert_eq!(prepared.input.stop.prompt_logprobs, Some(2));
    }

    #[test]
    fn prepare_chat_request_keeps_prompt_logprobs_independent_from_echo() {
        let request = ChatCompletionRequest {
            logprobs: true,
            top_logprobs: Some(3),
            echo: true,
            ..base_request()
        };

        let prepared = prepared(request);

        assert_eq!(prepared.input.stop.logprobs, Some(3));
        assert_eq!(prepared.input.stop.prompt_logprobs, None);
        assert!(!prepared.include_prompt_logprobs);
    }

    #[test]
    fn prepare_chat_request_threads_data_parallel_rank() {
        let prepared = prepare_chat_request(
            base_request(),
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext {
                request_id: "req".to_string(),
                data_parallel_rank: Some(7),
                ..ResolvedRequestContext::default()
            },
        )
        .expect("request is valid");
        assert_eq!(prepared.input.scheduling.data_parallel_rank, Some(7));
    }

    #[test]
    fn prepare_chat_request_leaves_data_parallel_rank_none_when_absent() {
        let prepared = prepared(base_request());
        assert_eq!(prepared.input.scheduling.data_parallel_rank, None);
    }

    #[test]
    fn prepare_chat_request_maps_no_generation_prompt_mode() {
        let mut request = base_request();
        request.add_generation_prompt = Some(false);

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).1,
            GenerationPromptMode::NoGenerationPrompt
        );
    }

    #[test]
    fn prepare_chat_request_rejects_conflicting_explicit_generation_prompt_flags() {
        let mut request = base_request();
        request.add_generation_prompt = Some(true);
        request.continue_final_message = true;

        let error = prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .unwrap_err();

        expect!["Cannot set both `continue_final_message` and `add_generation_prompt` to True."]
            .assert_eq(&error.to_error_response().error.message);
    }

    #[test]
    fn prepare_chat_request_accepts_continue_final_message_with_implicit_add_generation_prompt() {
        let mut request = base_request();
        request.messages = vec![ChatMessage::Assistant {
            content: Some(MessageContent::Text("hello".to_string())),
            name: None,
            tool_calls: None,
            reasoning: None,
        }];
        request.continue_final_message = true;

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).1,
            GenerationPromptMode::ContinueFinalAssistant
        );
    }

    #[test]
    fn prepare_chat_request_rejects_continue_final_message_without_final_assistant() {
        let mut request = base_request();
        request.continue_final_message = true;

        let error = prepare_chat_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
        )
        .unwrap_err();

        expect!["Cannot set `continue_final_message` to True when the last message is not from the assistant."]
            .assert_eq(&error.to_error_response().error.message);
    }

    #[test]
    fn prepare_chat_request_allows_new_assistant_mode_after_final_assistant() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Assistant {
                content: Some(MessageContent::Text("hello".to_string())),
                name: None,
                tool_calls: None,
                reasoning: None,
            }],
            ..base_request()
        };

        let prepared = prepared(request);

        assert_eq!(
            chat_context(&prepared.input).1,
            GenerationPromptMode::StartNewAssistant
        );
    }
}
