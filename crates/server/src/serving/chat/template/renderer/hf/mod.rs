//! Hugging Face Jinja chat-template renderer.
//!
//! `ModelConfig::load` builds one [`HfChatRenderer`] from the checkpoint files
//! and the server's template options for every model except a diffusers
//! pipeline. For each chat prompt, `InputProcessor::preprocess_qwen3_input` or
//! the omni chat preprocessors (`render_sensenova_chat`, `render_bagel_chat`)
//! call [`HfChatRenderer::render`] and tokenize the rendered prompt (the omni
//! paths first resolve their image placeholders in it).
//!
//! Rendering converts each `ChatMessage` into a `TemplateMessage`, the
//! OpenAI-compatible JSON shape that Hugging Face templates inspect: assistant
//! messages split into visible `content`, `reasoning_content`, and
//! `tool_calls`, and message content takes the shape selected by
//! `ChatTemplateContentFormat` (configured, or detected in `format` under
//! `Auto`). Request tools reach the template only when
//! `ChatRequest::tool_parsing_enabled` holds.

use std::collections::HashMap;

use crate::profile::assets::{
    HfSpecialTokens, HfTokenizerConfig, ResolvedModelFiles, load_tokenizer_config,
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use thiserror_ext::AsReport as _;
use tracing::{info, trace, warn};

use self::format::{
    ChatTemplateContentFormat, ChatTemplateContentFormatOption as ContentFormatOption,
};
use self::template::{CompiledChatTemplate, TemplateContext};
use self::value::{TemplateValue, to_template_value};
use crate::serving::chat::Error;
use crate::serving::chat::Result;
use crate::serving::chat::template::{ChatTemplateLoadOptions, Tool};
use crate::serving::chat::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};

mod error;
mod format;
mod template;
mod tojson;
mod value;

pub use template::{load_chat_template, resolve_chat_template};

pub use self::format::ChatTemplateContentFormatOption;

#[derive(Debug, Clone)]
/// Image rendering settings for a renderer that accepts `image_url` parts.
///
/// Without this value, rendering fails with
/// `Error::UnsupportedMultimodalContent` on any `image_url` part. Production
/// loading passes `None`: the omni preprocessors replace each chat image with
/// a text placeholder (`replace_chat_images`) before rendering, so image parts
/// never reach the renderer there.
pub struct MultimodalRenderInfo {
    /// Text substituted for each image part when content is flattened to a
    /// string. With list-shaped content, image parts become `{"type": "image"}`
    /// items and this token is unused.
    pub placeholder_token: String,
}

/// Hugging Face chat-template renderer for one served model.
///
/// Holds at most one compiled template together with the default template
/// kwargs and tokenizer special tokens installed into every render.
pub struct HfChatRenderer {
    // `None` when the model provides no template; `render` then fails with
    // `Error::MissingChatTemplate`.
    default_template: Option<CompiledChatTemplate>,
    default_template_kwargs: HashMap<String, JsonValue>,
    special_tokens: Option<HfSpecialTokens>,
    multimodal: Option<MultimodalRenderInfo>,
}

impl HfChatRenderer {
    /// Creates a renderer from an optional template source.
    ///
    /// Resolves the content format (detecting it from the template under
    /// `Auto`) and compiles the template eagerly, so syntax errors surface
    /// here as `Error::ChatTemplate` rather than on the first request. A
    /// `None` template yields a renderer whose `render` always fails.
    pub fn new(
        template: Option<String>,
        default_template_kwargs: HashMap<String, JsonValue>,
        content_format: ContentFormatOption,
    ) -> Result<Self> {
        Ok(Self {
            default_template: template
                .map(|template| {
                    CompiledChatTemplate::new(template, content_format)
                        .map_err(|error| Error::ChatTemplate(error.to_report_string()))
                })
                .transpose()?,
            default_template_kwargs,
            special_tokens: None,
            multimodal: None,
        })
    }

    /// Attaches tokenizer special tokens to the renderer.
    pub fn with_special_tokens(mut self, special_tokens: Option<HfSpecialTokens>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Attaches image rendering settings; with `None`, any `image_url` part
    /// fails rendering.
    pub fn with_multimodal(mut self, multimodal: Option<MultimodalRenderInfo>) -> Self {
        self.multimodal = multimodal;
        self
    }

    /// Creates a renderer from the given model files and loading options.
    ///
    /// Template precedence, highest first: `options.chat_template` (a file
    /// path or inline Jinja, see `resolve_chat_template`), then a non-blank
    /// standalone template file (`files.chat_template_path`), then the
    /// `chat_template` entry of `tokenizer_config.json`. Special tokens always
    /// come from the tokenizer config; when it defines none, templates see
    /// them as undefined.
    ///
    /// # Errors
    ///
    /// Fails when the tokenizer config cannot be loaded, a configured or
    /// standalone template cannot be read or resolved, or the selected
    /// template does not compile.
    pub fn load(
        files: &ResolvedModelFiles,
        options: ChatTemplateLoadOptions,
        multimodal: Option<MultimodalRenderInfo>,
    ) -> Result<Self> {
        let HfTokenizerConfig {
            special_tokens,
            chat_template,
            ..
        } = load_tokenizer_config(files.tokenizer_config_path.as_deref())?;
        let mut template = chat_template;
        let special_tokens = (!special_tokens.is_empty()).then_some(special_tokens);

        if let Some(configured_template) = options.chat_template.as_deref() {
            template = Some(
                resolve_chat_template(configured_template)
                    .map_err(|error| Error::ChatTemplate(error.to_report_string()))?,
            );
            info!("using configured chat template override");
        } else if let Some(chat_template_path) = files.chat_template_path.as_deref() {
            // A standalone template file overrides the tokenizer config entry
            // only when its content is non-blank.
            let file_template = load_chat_template(chat_template_path)
                .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

            if file_template.as_ref().is_some_and(|t| !t.trim().is_empty()) {
                info!(
                    path = %chat_template_path.display(),
                    "loaded dedicated chat template file, overriding tokenizer_config chat_template"
                );
                template = file_template;
            } else {
                warn!(
                    path = %chat_template_path.display(),
                    "ignoring empty dedicated chat template file and falling back to tokenizer_config chat_template"
                );
            }
        }

        Ok(Self::new(
            template,
            options.default_chat_template_kwargs,
            options.chat_template_content_format,
        )?
        .with_special_tokens(special_tokens)
        .with_multimodal(multimodal))
    }

    /// Renders one chat request into prompt text for the model tokenizer.
    ///
    /// # Errors
    ///
    /// Returns `Error::MissingChatTemplate` when the renderer has no template,
    /// `Error::UnsupportedMultimodalContent` for an `image_url` part without
    /// [`MultimodalRenderInfo`], and `Error::ChatTemplate` when assistant tool
    /// arguments are not valid JSON or the template fails to render.
    pub fn render(&self, request: &ChatRequest) -> Result<String> {
        let template = self
            .default_template
            .as_ref()
            .ok_or(Error::MissingChatTemplate)?;

        self.apply_chat_template_inner(template, request)
    }

    /// Builds template values and renders them with request-specific control flags.
    fn apply_chat_template_inner(
        &self,
        effective_template: &CompiledChatTemplate,
        request: &ChatRequest,
    ) -> Result<String> {
        let messages = to_template_messages(
            &request.messages,
            effective_template.content_format(),
            self.multimodal.as_ref(),
        )?;
        let tools = request
            .tool_parsing_enabled()
            .then(|| to_template_tools(&request.tools));
        trace!(
            message_count = messages.len(),
            content_format = ?effective_template.content_format(),
            ?messages,
            ?tools,
            "applying chat template"
        );

        let prompt = effective_template
            .apply(TemplateContext {
                messages: &messages,
                add_generation_prompt: request.chat_options.add_generation_prompt(),
                continue_final_message: request.chat_options.continue_final_message(),
                tools: tools.as_deref(),
                documents: None,
                template_kwargs: Some(&self.default_template_kwargs),
                special_tokens: self.special_tokens.as_ref(),
                reasoning_effort: request.chat_options.reasoning_effort,
            })
            .map_err(|error| Error::ChatTemplate(error.to_report_string()))?;

        trace!(
            prompt_len = prompt.len(),
            prompt, "rendered chat template prompt"
        );

        Ok(prompt)
    }
}

/// Chat message in the JSON shape expected by Jinja chat templates.
///
/// `None` fields are omitted rather than serialized as null, so templates can
/// test them with `is defined`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateMessage {
    role: &'static str,
    content: TemplateContent,
    // Developer-role messages may provide message-local tools in the same shape
    // as top-level request tools.
    tools: Option<Vec<TemplateTool>>,
    // Assistant reasoning, kept apart from `content` so each template decides
    // whether to replay it.
    reasoning_content: Option<String>,
    // Function-call-capable templates commonly expect assistant tool calls
    // under this OpenAI-compatible field name.
    tool_calls: Option<Vec<TemplateToolCall>>,
    // Tool-role messages refer back to the assistant call they are answering.
    tool_call_id: Option<String>,
}

/// Chat content in the two shapes HF templates commonly expect: a plain string
/// or an OpenAI-style list of typed parts.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum TemplateContent {
    String(String),
    OpenAi(Vec<TemplateContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TemplateContentPart {
    Text { text: String },
    Image,
}

#[derive(Debug, Serialize)]
struct TemplateToolCall {
    id: String,
    r#type: &'static str, // always "function"
    function: TemplateToolFunction,
}

#[derive(Debug, Serialize)]
struct TemplateToolFunction {
    name: String,
    arguments: TemplateValue,
}

#[derive(Debug, Serialize)]
/// Template-facing function tool representation.
pub(super) struct TemplateTool {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: TemplateToolDefinition,
}

/// Function definition in the shape of an OpenAI request tool.
///
/// `None` fields are omitted rather than serialized as null, so a template
/// that renders the tool with `tojson` sees only the fields a client sets,
/// as Hugging Face `apply_chat_template` renders the client's tool objects.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct TemplateToolDefinition {
    name: String,
    description: Option<String>,
    parameters: TemplateValue,
    strict: Option<bool>,
}

/// Converts chat messages into the value shape expected by Jinja chat templates.
fn to_template_messages(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateMessage>> {
    messages
        .iter()
        .map(|message| to_template_message(message, content_format, multimodal))
        .collect()
}

/// Converts one structured message into the renderer's template value model.
fn to_template_message(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateMessage> {
    Ok(match message {
        ChatMessage::System { content } => TemplateMessage {
            role: "system",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Developer { content, tools } => TemplateMessage {
            role: "developer",
            content: to_template_content(content, content_format, multimodal)?,
            tools: tools.as_deref().map(to_template_tools),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::User { content } => TemplateMessage {
            role: "user",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        ChatMessage::Assistant { content } => {
            let text = content.text();
            let reasoning = content.reasoning();
            let tool_calls = to_template_tool_calls(content)?;
            let content =
                to_template_content(&ChatContent::Text(text), content_format, multimodal)?;
            TemplateMessage {
                role: "assistant",
                content,
                tools: None,
                reasoning_content: reasoning.clone(),
                tool_calls,
                tool_call_id: None,
            }
        }
        ChatMessage::ToolResponse {
            content,
            tool_call_id,
        } => TemplateMessage {
            role: "tool",
            content: to_template_content(content, content_format, multimodal)?,
            tools: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.clone()),
        },
    })
}

/// Parses assistant tool arguments and converts complete calls into template values.
fn to_template_tool_calls(
    content: &crate::serving::chat::AssistantMessage,
) -> Result<Option<Vec<TemplateToolCall>>> {
    let mut tool_calls = Vec::new();

    // Arguments reach the template as a parsed JSON value rather than JSON
    // text, so templates can index or iterate them (`arguments.items()`).
    for tool_call in content.tool_calls() {
        let arguments = serde_json::from_str(&tool_call.arguments).map_err(|error| {
            Error::ChatTemplate(format!(
                "assistant tool call `{}` has invalid JSON arguments: {}",
                tool_call.id,
                error.as_report()
            ))
        })?;
        let arguments = to_template_value(arguments);

        tool_calls.push(TemplateToolCall {
            id: tool_call.id.clone(),
            r#type: "function",
            function: TemplateToolFunction {
                name: tool_call.name.clone(),
                arguments,
            },
        });
    }

    Ok((!tool_calls.is_empty()).then_some(tool_calls))
}

/// Converts message content according to the template's resolved representation.
///
/// Under `Preserve`, plain text stays a string and part lists stay lists, so
/// templates that branch on `content is string` see the caller's shape.
fn to_template_content(
    content: &ChatContent,
    content_format: ChatTemplateContentFormat,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<TemplateContent> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => {
            TemplateContent::String(to_template_string_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::OpenAi => {
            TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
        }
        ChatTemplateContentFormat::Preserve => match content {
            ChatContent::Text(text) => TemplateContent::String(text.clone()),
            ChatContent::Parts(_) => {
                TemplateContent::OpenAi(to_template_openai_content(content, multimodal)?)
            }
        },
    })
}

/// Converts message content into an OpenAI-style list of typed parts.
fn to_template_openai_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<Vec<TemplateContentPart>> {
    match content {
        ChatContent::Text(text) => Ok(vec![TemplateContentPart::Text { text: text.clone() }]),
        ChatContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ChatContentPart::Text { text } => {
                    Ok(TemplateContentPart::Text { text: text.clone() })
                }
                // Image parts reach the template as `{"type": "image"}`; the
                // image payload itself is not exposed to the template.
                ChatContentPart::ImageUrl { .. } => {
                    multimodal.ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                    Ok(TemplateContentPart::Image)
                }
            })
            .collect(),
    }
}

/// Flattens content into a string while substituting multimodal placeholder tokens.
fn to_template_string_content(
    content: &ChatContent,
    multimodal: Option<&MultimodalRenderInfo>,
) -> Result<String> {
    match content {
        ChatContent::Text(text) => Ok(text.clone()),
        ChatContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::ImageUrl { .. } => {
                        let multimodal =
                            multimodal.ok_or(Error::UnsupportedMultimodalContent("image_url"))?;
                        out.push_str(&multimodal.placeholder_token);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// Converts request tools into template values.
fn to_template_tools(tools: &[Tool]) -> Vec<TemplateTool> {
    tools
        .iter()
        .map(|tool| TemplateTool {
            tool_type: "function",
            function: TemplateToolDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: to_template_value(tool.parameters.clone()),
                strict: tool.strict,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::profile::assets::{HfSpecialTokens, NamedSpecialToken};
    use expect_test::expect;
    use serde_json::Value;

    use super::{ChatTemplateContentFormatOption, HfChatRenderer, MultimodalRenderInfo};
    use crate::serving::chat::{AssistantContentBlock, Error, Result};
    use crate::serving::chat::{
        ChatContentPart, ChatMessage, ChatRequest, ChatRole, ChatToolChoice, GenerationPromptMode,
        ReasoningEffort, Tool,
    };

    const QWEN3_0_6B_TEMPLATE: &str = include_str!("../../../../../../tests/templates/qwen3.jinja");

    fn sample_request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            messages,
            ..ChatRequest::for_test()
        }
    }

    fn render(template: Option<&str>, request: &ChatRequest) -> Result<String> {
        HfChatRenderer::new(
            template.map(str::to_owned),
            HashMap::new(),
            ChatTemplateContentFormatOption::Auto,
        )?
        .render(request)
    }

    fn render_mm(
        template: &str,
        request: &ChatRequest,
        content_format: ChatTemplateContentFormatOption,
    ) -> Result<String> {
        HfChatRenderer::new(Some(template.to_string()), HashMap::new(), content_format)?
            .with_multimodal(Some(MultimodalRenderInfo {
                placeholder_token: "<image>".to_string(),
            }))
            .render(request)
    }

    fn image_request() -> ChatRequest {
        sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("a"),
            ChatContentPart::image_url("data:image/png;base64,test"),
            ChatContentPart::text("b"),
        ])])
    }

    #[test]
    fn string_content_format_replaces_image_with_placeholder_text() {
        let rendered = render_mm(
            "{{ messages[0].content }}",
            &image_request(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap();

        assert_eq!(rendered, "a<image>b");
    }

    #[test]
    fn openai_content_format_normalizes_image_url_for_template() {
        let rendered = render_mm(
            "{% for item in messages[0].content %}{% if item.type == 'image' %}<|image_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}",
            &image_request(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap();

        assert_eq!(rendered, "a<|image_pad|>b");
    }

    #[test]
    fn auto_content_format_preserves_mixed_system_and_user_shapes() {
        // The template both loops over content items and tests `content is
        // string`, so `Auto` selects `Preserve`: the system string still
        // supports `+` concatenation while the user parts stay a list.
        let request = sample_request(vec![
            ChatMessage::system("policy"),
            ChatMessage::user(vec![
                ChatContentPart::text("a"),
                ChatContentPart::image_url("data:image/png;base64,test"),
                ChatContentPart::text("b"),
            ]),
        ]);
        let rendered = render_mm(
            "{% if messages[0].role == 'system' %}{{ 'S:' + messages[0].content + '|' }}{% endif %}{% for message in messages[1:] %}{% if message.content is string %}{{ message.content }}{% else %}{% for item in message.content %}{% if item.type == 'image' %}<image>{% else %}{{ item.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
            &request,
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        assert_eq!(rendered, "S:policy|a<image>b");
    }

    #[test]
    fn chat_template_supports_pycompat_templates() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "<think>hello")]);

        let rendered = render(
            Some(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "think");
    }

    #[test]
    fn chat_template_passes_continue_final_message_to_template() {
        let mut request = sample_request(vec![ChatMessage::text(
            ChatRole::Assistant,
            "The capital of",
        )]);

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "new"
        );

        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "continue"
        );
    }

    #[test]
    fn chat_template_flattens_text_parts_for_string_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_exposes_developer_tools() {
        let request = sample_request(vec![ChatMessage::developer(
            "policy",
            Some(vec![Tool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                strict: Some(true),
            }]),
        )]);

        let rendered = render(
            Some("{{ messages[0].role }}|{{ messages[0].content }}|{{ messages[0].tools[0].function.name }}|{{ messages[0].tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "developer|policy|get_weather|city");
    }

    #[test]
    fn chat_template_keeps_string_text_for_openai_detected_templates() {
        // The template tests `content is string` without looping over content
        // items, so `Auto` selects `String` and plain text arrives as a string.
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn chat_template_emits_openai_text_blocks_for_structured_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text("world"),
        ])]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello|world|");
    }

    #[test]
    fn chat_template_requires_a_template() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let error = render(None, &request).unwrap_err();

        assert!(matches!(error, Error::MissingChatTemplate));
    }

    #[test]
    fn chat_template_injects_special_tokens_into_context() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<bos>".to_string())),
            ..Default::default()
        };

        let rendered = HfChatRenderer::new(
            Some("{{ bos_token }}|{{ bos_token is defined }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap()
        .with_special_tokens(Some(special_tokens))
        .render(&request)
        .unwrap();

        // Chat templates follow Python Jinja2, which prints booleans as
        // Python literals.
        assert_eq!(rendered, "<bos>|True");
    }

    #[test]
    fn chat_template_exposes_assistant_reasoning_separately() {
        let request = sample_request(vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "inner".to_string(),
            },
            AssistantContentBlock::Text {
                text: "outer".to_string(),
            },
        ])]);

        let rendered = render(
            Some("{{ messages[0].reasoning_content }}|{{ messages[0].content }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "inner|outer");
    }

    #[test]
    fn chat_template_forces_string_content_format_when_configured() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some(
                "{%- if messages[0].content is string -%}{{ messages[0].content }}{%- else -%}{%- for item in messages[0].content %}{{ item.text }}|{%- endfor -%}{%- endif -%}".to_string(),
            ),
            HashMap::new(),
            ChatTemplateContentFormatOption::String,
        )
        .unwrap()
        .render(&request)
        .unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_forces_openai_content_format_when_configured() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = HfChatRenderer::new(
            Some("{{ messages[0].content[0].text }}{{ messages[0].content[1].text }}".to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap()
        .render(&request)
        .unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_exposes_default_template_kwargs() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let renderer = HfChatRenderer::new(
            Some("{{ enable_thinking }}|{{ default_only }}".to_string()),
            HashMap::from([
                ("enable_thinking".to_string(), Value::Bool(true)),
                ("default_only".to_string(), Value::String("x".to_string())),
            ]),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap();

        assert_eq!(rendered, "True|x");
    }

    #[test]
    fn chat_template_reasoning_effort_overrides_default_template_kwargs() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.chat_options.reasoning_effort = Some(ReasoningEffort::Max);

        let renderer = HfChatRenderer::new(
            Some("{{ reasoning_effort }}".to_string()),
            HashMap::from([(
                "reasoning_effort".to_string(),
                Value::String("medium".to_string()),
            )]),
            ChatTemplateContentFormatOption::Auto,
        )
        .unwrap();

        let rendered = renderer.render(&request).unwrap();

        assert_eq!(rendered, "max");
    }

    #[test]
    fn qwen3_template_omits_reasoning_for_prior_assistant_messages() {
        let request = sample_request(vec![
            ChatMessage::text(
                ChatRole::User,
                "Hi. Tell me about the capital of France in short",
            ),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "\nOkay, the user is asking... I think that's all.\n".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "Paris is the capital of France.".to_string(),
                },
            ]),
            ChatMessage::text(ChatRole::User, "Tell me about Paris more."),
        ]);

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            Hi. Tell me about the capital of France in short<|im_end|>
            <|im_start|>assistant
            Paris is the capital of France.<|im_end|>
            <|im_start|>user
            Tell me about Paris more.<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_renders_single_user_prompt() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_renders_system_user_prompt() {
        let request = sample_request(vec![
            ChatMessage::text(ChatRole::System, "be brief"),
            ChatMessage::text(ChatRole::User, "hello"),
        ]);

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>system
            be brief<|im_end|>
            <|im_start|>user
            hello<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_respects_forced_openai_content_format() {
        // The Qwen3 template emits content only when it is a string. A forced
        // `OpenAi` format still wins over detection, so the part list renders
        // as empty user content.
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        let rendered = HfChatRenderer::new(
            Some(QWEN3_0_6B_TEMPLATE.to_string()),
            HashMap::new(),
            ChatTemplateContentFormatOption::OpenAi,
        )
        .unwrap()
        .render(&request)
        .unwrap();

        expect![[r#"
            <|im_start|>user
            <|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_keeps_reasoning_after_the_last_user_query() {
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "What is 1 + 1?"),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "need simple arithmetic".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "2".to_string(),
                },
            ]),
        ]);
        request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            What is 1 + 1?<|im_end|>
            <|im_start|>assistant
            <think>
            need simple arithmetic
            </think>

            2<|im_end|>
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn chat_template_exposes_tools_to_templates_when_auto_enabled() {
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
            strict: None,
        }];
        request.tool_choice = ChatToolChoice::Auto;

        let rendered = render(
            Some("{{ tools[0].function.name }}|{{ tools[0].function.parameters.required[0] }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|city");
    }

    /// Tools reach the template in the shape of the client's tool objects:
    /// optional fields the tool leaves unset are absent rather than null.
    #[test]
    fn chat_template_omits_unset_optional_tool_fields() {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
        });
        let mut request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        request.tools = vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: parameters.clone(),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: Some("Get time".to_string()),
                parameters: parameters.clone(),
                strict: Some(true),
            },
        ];
        request.tool_choice = ChatToolChoice::Auto;

        let rendered = render(Some("{{ tools | tojson }}"), &request).unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&rendered).unwrap(),
            serde_json::json!([
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": parameters},
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": parameters,
                        "strict": true,
                    },
                },
            ])
        );
    }

    #[test]
    fn chat_template_exposes_assistant_tool_calls_and_tool_messages() {
        let request = sample_request(vec![
            ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(
                crate::serving::chat::template::AssistantToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                },
            )]),
            ChatMessage::tool_response("Sunny", "call_1"),
        ]);

        let rendered = render(
            Some(
                "{{ messages[0].tool_calls[0].function.name }}|{{ messages[0].tool_calls[0].function.arguments.city }}|{{ messages[1].tool_call_id }}|{{ messages[1].content }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "get_weather|Paris|call_1|Sunny");
    }

    #[test]
    fn chat_template_tool_call_argument_items_method_is_not_shadowed_by_field() {
        // An `items` key in tool arguments must not shadow the dict `items()`
        // method (see `TemplateMap`), and iteration follows the key order of
        // the arguments JSON.
        let request = sample_request(vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::ToolCall(crate::serving::chat::template::AssistantToolCall {
                id: "call_1".to_string(),
                name: "add".to_string(),
                arguments: r#"{"items":"operands","x":2,"y":1.0}"#.to_string(),
            }),
        ])]);

        let rendered = render(
            Some(
                "{%- set arguments = messages[0].tool_calls[0].function.arguments -%}
{%- for key, value in arguments.items() -%}{{ key }}={{ value }};{%- endfor -%}
|{{ arguments['items'] }}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "items=operands;x=2;y=1.0;|operands");
    }
}
