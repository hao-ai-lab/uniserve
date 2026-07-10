//! Shared DeepSeek DSML renderer support.
//!
//! The DeepSeek V3.2 and V4 renderers emit the same DSML tool-call wire format
//! and share the same JSON/assistant helper logic; the only legitimate
//! difference is the tool-call wrapper token (`function_calls` for V3.2 vs
//! `tool_calls` for V4) and per-version message-ordering/thinking policy.
//!
//! This module holds the single source of truth for the DSML grammar markers,
//! the JSON serialization format, and the assistant/content write helpers so a
//! fix to argument encoding or JSON formatting is made once. The wrapper-token
//! pair is parameterized through [`DsmlWrapper`].
//!
//! The DSML grammar markers here mirror the read side in the `tool-parser`
//! crate (`deepseek_dsml::DsmlTokens`). The two crates have no dependency edge
//! between them, so the marker spellings cannot be deduplicated across the
//! read/write boundary in code; the `chat` crate's round-trip test
//! (`tests/roundtrip.rs`) guards that the two sides agree by rendering via this
//! module and parsing via the tool-parser.

use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::chat::template::error::{Error, Result};
use crate::chat::template::request::{ChatContent, ChatMessage, ChatTool};
use crate::chat::template::{AssistantContentBlock, AssistantToolCall};

/// DSML attribute/markup token shared by both DeepSeek renderers.
const DSML_TOKEN: &str = "｜DSML｜";

pub(super) const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub(super) const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub(super) const THINKING_START_TOKEN: &str = "<think>";
pub(super) const THINKING_END_TOKEN: &str = "</think>";

/// DeepSeek uses `"chat"` vs `"thinking"` mode names. Keep the split explicit
/// so the per-renderer branches stay easy to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ThinkingMode {
    Chat,
    Thinking,
}

/// The tool-call block wrapper that differs between DeepSeek versions.
///
/// This is the write-side counterpart to the tool-parser's
/// `deepseek_dsml::DsmlTokens`: V3.2 wraps tool calls in
/// `<｜DSML｜function_calls>` while V4 uses `<｜DSML｜tool_calls>`.
#[derive(Debug, Clone, Copy)]
pub(super) struct DsmlWrapper {
    /// Opening tool-calls block marker, e.g. `<｜DSML｜function_calls>`.
    start: &'static str,
    /// Closing tool-calls block marker, e.g. `</｜DSML｜function_calls>`.
    end: &'static str,
}

impl DsmlWrapper {
    /// DeepSeek V3.2 wraps tool calls in `function_calls`.
    pub(super) const V32: Self = Self {
        start: "<｜DSML｜function_calls>",
        end: "</｜DSML｜function_calls>",
    };
    /// DeepSeek V4 wraps tool calls in `tool_calls`.
    pub(super) const V4: Self = Self {
        start: "<｜DSML｜tool_calls>",
        end: "</｜DSML｜tool_calls>",
    };
}

/// Tool schema shape rendered inside the prompt's tool block.
#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct RenderedToolSchema<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: &'a Value,
    strict: Option<bool>,
}

/// Render the whole tool-call block for one assistant turn.
///
/// Emits the version-specific wrapper around one or more invoke blocks.
pub(super) fn render_tool_calls<'a>(
    out: &mut String,
    wrapper: DsmlWrapper,
    model_label: &str,
    tool_calls: impl Iterator<Item = &'a AssistantToolCall>,
) -> Result<()> {
    out.push('\n');
    out.push('\n');
    out.push_str(wrapper.start);
    out.push('\n');
    for (index, tool_call) in tool_calls.enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_tool_call(out, model_label, tool_call)?;
    }
    out.push('\n');
    out.push_str(wrapper.end);
    Ok(())
}

/// Render one assistant tool call in DSML XML-like format.
fn render_tool_call(
    out: &mut String,
    model_label: &str,
    tool_call: &AssistantToolCall,
) -> Result<()> {
    writeln!(out, "<{DSML_TOKEN}invoke name=\"{}\">", tool_call.name)
        .map_err(|error| Error::ChatTemplate(format!("failed to render DSML invoke: {error}")))?;
    encode_arguments_to_dsml(out, model_label, tool_call)?;
    write!(out, "\n</{DSML_TOKEN}invoke>")
        .map_err(|error| Error::ChatTemplate(format!("failed to render DSML invoke: {error}")))?;
    Ok(())
}

/// Convert one assistant tool-call arguments object into DSML parameter form.
///
/// String values are emitted raw with `string="true"`, while all other JSON
/// values are rendered with JSON syntax and `string="false"`.
fn encode_arguments_to_dsml(
    out: &mut String,
    model_label: &str,
    tool_call: &AssistantToolCall,
) -> Result<()> {
    let arguments: Value = serde_json::from_str(&tool_call.arguments).map_err(|error| {
        Error::ChatTemplate(format!(
            "assistant tool call has invalid JSON arguments for {model_label}: {error}"
        ))
    })?;
    let Some(arguments) = arguments.as_object() else {
        return Err(Error::ChatTemplate(format!(
            "assistant tool call arguments for {model_label} must be a JSON object"
        )));
    };

    let mut wrote_parameter = false;
    for (key, value) in arguments {
        if wrote_parameter {
            out.push('\n');
        }

        let is_string = matches!(value, Value::String(_));
        write!(
            out,
            "<{DSML_TOKEN}parameter name=\"{key}\" string=\"{}\">",
            if is_string { "true" } else { "false" }
        )
        .map_err(|error| {
            Error::ChatTemplate(format!("failed to render DSML parameter: {error}"))
        })?;

        match value {
            Value::String(value) => out.push_str(value),
            value => out.push_str(&json_dumps(model_label, value)?),
        }

        write!(out, "</{DSML_TOKEN}parameter>").map_err(|error| {
            Error::ChatTemplate(format!("failed to render DSML parameter: {error}"))
        })?;
        wrote_parameter = true;
    }

    Ok(())
}

/// Serialize one typed tool schema into the JSON shape embedded in the prompt.
pub(super) fn render_tool_schema(
    out: &mut String,
    model_label: &str,
    tool: &ChatTool,
) -> Result<()> {
    out.push_str(&json_dumps(
        model_label,
        &RenderedToolSchema {
            name: &tool.name,
            description: tool.description.as_deref(),
            parameters: &tool.parameters,
            strict: tool.strict,
        },
    )?);
    Ok(())
}

/// Write chat content directly into the destination buffer without flattening
/// it into an intermediate `String`.
pub(super) fn write_chat_content(out: &mut String, content: &ChatContent) -> Result<()> {
    match content {
        ChatContent::Text(text) => out.push_str(text),
        ChatContent::Parts(parts) => {
            for part in parts {
                out.push_str(part.as_text()?);
            }
        }
    }
    Ok(())
}

/// Write all reasoning blocks in encounter order.
pub(super) fn write_assistant_reasoning(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Reasoning { text } = block {
            out.push_str(text);
        }
    }
}

/// Write all visible assistant text blocks in encounter order.
pub(super) fn write_assistant_text(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Text { text } = block {
            out.push_str(text);
        }
    }
}

/// Return the contiguous tool-response block containing `actual_index`.
pub(super) fn tool_response_block_bounds(
    messages: &[ChatMessage],
    actual_index: usize,
) -> (usize, usize) {
    let mut block_start = actual_index;
    while block_start > 0 && matches!(messages[block_start - 1], ChatMessage::ToolResponse { .. }) {
        block_start -= 1;
    }

    let mut block_end = actual_index + 1;
    while block_end < messages.len()
        && matches!(messages[block_end], ChatMessage::ToolResponse { .. })
    {
        block_end += 1;
    }

    (block_start, block_end)
}

/// Compact JSON serialization used by both DeepSeek renderers for exact prompt
/// text (Python `json.dumps`-style spacing after commas and colons, no ASCII
/// escaping).
fn json_dumps<T: Serialize>(model_label: &str, value: &T) -> Result<String> {
    JsonFormat::new()
        .comma(", ")
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to configure {model_label} JSON comma separator: {error}"
            ))
        })?
        .colon(": ")
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to configure {model_label} JSON colon separator: {error}"
            ))
        })?
        .ascii(false)
        .format_to_string(value)
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to serialize {model_label} JSON payload: {error}"
            ))
        })
}
