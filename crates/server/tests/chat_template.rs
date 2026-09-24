//! Public Hugging Face chat-template rendering behavior.
//!
//! The integration cases cover content-shape detection under
//! `ChatTemplateContentFormatOption::Auto`, byte-exact rendering of prior
//! assistant turns through the Qwen3 template, and key-order-preserving
//! tool-argument formatting with `serde_json`-normalized numbers.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;

use uniserve_server::serving::chat::template::renderer::hf::{
    ChatTemplateContentFormatOption, HfChatRenderer,
};
use uniserve_server::serving::chat::template::{AssistantContentBlock, AssistantToolCall};
use uniserve_server::serving::chat::{
    ChatContentPart, ChatMessage, ChatRequest, ChatRole, GenerationPromptMode,
};

const QWEN3_TEMPLATE: &str = include_str!("templates/qwen3.jinja");

fn hf_render(
    template: &str,
    format: ChatTemplateContentFormatOption,
    request: &ChatRequest,
) -> String {
    HfChatRenderer::new(Some(template.to_owned()), HashMap::new(), format)
        .expect("template should compile")
        .render(request)
        .expect("render should succeed")
}

fn base_request(messages: Vec<ChatMessage>) -> ChatRequest {
    ChatRequest {
        messages,
        ..ChatRequest::for_test()
    }
}

/// One user message with two text parts, rendered without a generation prompt.
///
/// Flattened to a string, the content reads `"ab"`; in the OpenAI format it is
/// a list of two text-part objects.
fn multipart_user() -> ChatRequest {
    let mut request = base_request(vec![ChatMessage::user(vec![
        ChatContentPart::text("a"),
        ChatContentPart::text("b"),
    ])]);
    request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
    request
}

#[test]
fn auto_detection_treats_alias_loop_template_as_string() {
    // The template only aliases content as `parts` and tests the alias with
    // `is string`. Without a loop over a message's `content`, `Auto` selects
    // the string format, so the parts arrive flattened and the string branch
    // renders.
    let template = "{%- for message in messages -%}{%- set parts = message.content -%}\
{%- if parts is string -%}STR:{{ parts }}{%- else -%}LIST{%- endif -%}{%- endfor -%}";
    let rendered = hf_render(
        template,
        ChatTemplateContentFormatOption::Auto,
        &multipart_user(),
    );
    assert_eq!(rendered, "STR:ab");
}

#[test]
fn auto_detection_treats_length_and_index_access_template_as_string() {
    // Length / index access on content must not be mistaken for a content loop.
    let template = "{%- for message in messages -%}len={{ message.content|length }};\
first={{ message.content[0] }}{%- endfor -%}";
    let rendered = hf_render(
        template,
        ChatTemplateContentFormatOption::Auto,
        &multipart_user(),
    );
    // String content "ab": length 2, index 0 is the character 'a'.
    assert_eq!(rendered, "len=2;first=a");
}

#[test]
fn auto_detection_treats_direct_content_loop_template_as_openai() {
    let template = "{%- for message in messages -%}\
{%- for part in message.content -%}{{ part.text }}|{%- endfor -%}{%- endfor -%}";
    let rendered = hf_render(
        template,
        ChatTemplateContentFormatOption::Auto,
        &multipart_user(),
    );
    // Each part surfaces as a structured `{text: ...}` object.
    assert_eq!(rendered, "a|b|");
}

fn qwen_history() -> ChatRequest {
    base_request(vec![
        ChatMessage::text(ChatRole::User, "What is the capital of France?"),
        ChatMessage::assistant_text("The capital of France is Paris."),
        ChatMessage::text(ChatRole::User, "And of Italy?"),
    ])
}

#[test]
fn qwen3_preserves_prior_assistant_completion_text_byte_identically() {
    let request = qwen_history();
    let rendered = hf_render(
        QWEN3_TEMPLATE,
        ChatTemplateContentFormatOption::Auto,
        &request,
    );

    // The prior assistant turn's visible completion text must survive
    // intact inside the rendered prompt, framed by the assistant turn markers.
    assert!(
        rendered.contains("<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>"),
        "assistant completion text should round-trip byte-for-byte, got:\n{rendered}"
    );
}

/// Deliberately non-alphabetical key order plus mixed numeric spellings.
///
/// Key order survives because the workspace builds `serde_json` with
/// `preserve_order` and the renderer exposes the parsed tool-call arguments to
/// templates as insertion-ordered maps.
const MIXED_ARGS: &str = r#"{"zulu":2,"alpha":1.00,"mike":"hi","delta":[3,4]}"#;

fn assistant_tool_call_history(arguments: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::user("Run the tool."),
        ChatMessage::assistant_blocks(vec![AssistantContentBlock::ToolCall(AssistantToolCall {
            id: "call-1".to_string(),
            name: "do_thing".to_string(),
            arguments: arguments.to_string(),
        })]),
        ChatMessage::tool_response("{\"ok\":true}", "call-1"),
    ]
}

#[test]
fn configured_template_preserves_tool_argument_order_via_items_iteration() {
    // The HF path exposes tool-call arguments as a structured map. Iterating
    // `.items()` must yield the original (non-alphabetical) key order, and the
    // numeric spellings must match serde_json's normalized output.
    let request = base_request(assistant_tool_call_history(MIXED_ARGS));
    let template = "{%- set args = messages[1].tool_calls[0].function.arguments -%}\
{%- for key, value in args.items() -%}{{ key }}={{ value }};{%- endfor -%}";
    let rendered = hf_render(template, ChatTemplateContentFormatOption::Auto, &request);

    // Key order preserved; integer stays 2; `1.00` -> 1.0; array stringified.
    assert_eq!(rendered, "zulu=2;alpha=1.0;mike=hi;delta=[3, 4];");
}

#[test]
fn configured_template_tojson_preserves_key_order_and_number_precision() {
    // Rendering the arguments back through HF `tojson` must keep insertion order
    // (no implicit sort) and the normalized number spelling.
    let request = base_request(assistant_tool_call_history(MIXED_ARGS));
    let template =
        "{{ messages[1].tool_calls[0].function.arguments|tojson(separators=[',', ':']) }}";
    let rendered = hf_render(template, ChatTemplateContentFormatOption::Auto, &request);

    assert_eq!(
        rendered,
        r#"{"zulu":2,"alpha":1.0,"mike":"hi","delta":[3,4]}"#
    );
}
