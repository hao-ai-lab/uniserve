//! Hugging Face chat-template rendering behavior.
//!
//! These integration tests drive the public [`HfChatRenderer`] API against the
//! committed offline chat-template fixtures under `tests/templates`.
//! They cover three observable renderer behaviors:
//!
//! 1. The String-vs-OpenAI content-format detector (used by `HfChatRenderer`
//!    with `Auto`) classifies real committed templates correctly, observed
//!    through the rendered output rather than the private AST detector.
//! 2. Re-rendering identical multi-turn chat is deterministic and preserves
//!    assistant completion text byte-for-byte.
//! 3. Tool-call argument key order and JSON number precision survive each
//!    configured argument formatting.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;

use uniserve_serving::chat::template::renderer::hf::{
    ChatTemplateContentFormatOption, HfChatRenderer,
};
use uniserve_serving::chat::template::request::{
    ChatContentPart, ChatMessage, ChatRequest, ChatRole, GenerationPromptMode,
};
use uniserve_serving::chat::template::{AssistantContentBlock, AssistantToolCall};

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

// ---------------------------------------------------------------------------
// Behavior 1: content-format detector over real committed templates.
//
// The private AST detector is exercised through `Auto` + the observable
// rendering difference: a String-format template flattens multi-part content
// into one concatenated string, while an OpenAI-format template iterates the
// content list (so a probe template seeing the list emits one marker per part).
// ---------------------------------------------------------------------------

/// Minimal probe whose output distinguishes how the detector treated content:
/// it both flattens (`{{ content }}`) and, when content is a list, iterates it.
/// Under `String` the content is a plain string so the for-loop iterates the
/// characters' container as a string and the `is string` test is true; we use a
/// dedicated probe that branches on `is string` to read the detected format.
const DETECT_PROBE: &str = "{%- for message in messages -%}\
{%- if message.content is string -%}STR:{{ message.content }}\
{%- else -%}LIST{%- for part in message.content -%}:{{ part.text }}{%- endfor -%}\
{%- endif -%}{%- endfor -%}";

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
    // `{% set parts = message.content %}{% for item in parts %}` aliases content
    // first, which the detector must NOT treat as a direct content loop.
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

#[test]
fn detector_probe_reports_string_for_string_format() {
    // Sanity anchor for the probe template itself under an explicit String force.
    let rendered = hf_render(
        DETECT_PROBE,
        ChatTemplateContentFormatOption::String,
        &multipart_user(),
    );
    assert_eq!(rendered, "STR:ab");
}

#[test]
fn detector_probe_reports_list_for_openai_format() {
    let rendered = hf_render(
        DETECT_PROBE,
        ChatTemplateContentFormatOption::OpenAi,
        &multipart_user(),
    );
    assert_eq!(rendered, "LIST:a:b");
}

// ---------------------------------------------------------------------------
// Behavior 2: re-rendering identical history is deterministic and reproduces
// the prior assistant completion text byte-for-byte.
// ---------------------------------------------------------------------------

fn qwen_history() -> ChatRequest {
    base_request(vec![
        ChatMessage::text(ChatRole::User, "What is the capital of France?"),
        ChatMessage::assistant_text("The capital of France is Paris."),
        ChatMessage::text(ChatRole::User, "And of Italy?"),
    ])
}

#[test]
fn qwen_family_render_is_deterministic_across_repeated_renders() {
    let renderer = HfChatRenderer::new(
        Some(QWEN3_TEMPLATE.to_owned()),
        HashMap::new(),
        ChatTemplateContentFormatOption::Auto,
    )
    .unwrap();
    let request = qwen_history();

    let first = renderer.render(&request).unwrap();
    let second = renderer.render(&request).unwrap();

    assert_eq!(
        first, second,
        "repeated Qwen renders must be byte-identical"
    );
}

#[test]
fn qwen_family_preserves_prior_assistant_completion_text_byte_identically() {
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

// ---------------------------------------------------------------------------
// Behavior 3: tool-call argument key order and number precision survive each
// configured JSON formatting.
//
// `serde_json` is built with `preserve_order`, so object key insertion order is
// preserved end-to-end; floats with a trailing fractional zero normalize to a
// single `.0` and integers stay integers.
// ---------------------------------------------------------------------------

/// Deliberately non-alphabetical key order plus mixed numeric spellings.
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
fn hf_family_tool_call_arguments_preserve_key_order_via_items_iteration() {
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
fn hf_family_tojson_preserves_key_order_and_number_precision() {
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
