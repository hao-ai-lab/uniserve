//! Shared parser core for JSON tool calls wrapped by text markers.

pub use qwen::Qwen3XmlToolParser;

mod qwen;

use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::{AddContext, ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::literal;

use super::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len, take_json_object,
};
use super::{Result, ToolCallDelta, ToolParserOutput};

type JsonToolInput<'i> = Partial<&'i str>;

/// Upper bound on the per-stream parser buffer, in bytes.
///
/// While inside an unclosed tool-call header or JSON arguments object the
/// buffer retains every byte that has not yet formed a complete event. A stream
/// that opens a marker and then never closes it (e.g. an endless function name
/// string or an unbalanced `{`) would otherwise grow the buffer without bound.
/// Capping it converts that into a bounded parse error instead of unbounded
/// memory growth. The limit is generous relative to any realistic tool call.
const MAX_BUFFER_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy)]
struct JsonToolCallConfig {
    parser_name: &'static str,
    start_marker: &'static str,
    end_marker: &'static str,
    marker_whitespace: &'static str,
    name_key: &'static str,
    /// Candidate JSON keys naming the arguments payload, tried in order.
    /// Most parsers use a single key like `["arguments"]`, but some accept
    /// multiple (e.g. InternLM2 accepts `parameters` or `arguments`).
    arguments_key: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallMode {
    Text,
    Header,
    Arguments { json_scan: JsonObjectScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallEvent {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallEnd,
}

/// Tool parser core for marker-wrapped JSON tool calls.
#[derive(Debug)]
struct JsonToolCallParser {
    config: JsonToolCallConfig,
    buffer: String,
    mode: JsonToolCallMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl JsonToolCallParser {
    /// Create a marker-wrapped JSON tool-call parser.
    fn new(config: JsonToolCallConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            mode: JsonToolCallMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        let config = self.config;

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_json_tool_call_event(input, &mut self.mode, config)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        // Any bytes still buffered here belong to an event that has not yet
        // completed. Bound that retention so an unterminated tool call cannot
        // grow the buffer without limit.
        if self.buffer.len() > MAX_BUFFER_BYTES {
            return Err(parsing_failed!(
                "{} buffer exceeded {} bytes without completing a tool call",
                self.config.parser_name,
                MAX_BUFFER_BYTES
            ));
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            JsonToolCallMode::Text => output.normal_text.push_str(&self.buffer),
            JsonToolCallMode::Header | JsonToolCallMode::Arguments { .. } => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.config.parser_name
                ));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    /// Apply one parsed JSON tool-call event to parser state and output.
    fn apply_event(
        &mut self,
        event: JsonToolCallEvent,
        output: &mut ToolParserOutput,
    ) -> Result<()> {
        match event {
            JsonToolCallEvent::Text { len: consumed_len } => {
                output.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            JsonToolCallEvent::ToolCallStart => self.mode = JsonToolCallMode::Header,
            JsonToolCallEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = JsonToolCallMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            JsonToolCallEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "{} arguments without an active tool call",
                        self.config.parser_name
                    ));
                };
                output.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            JsonToolCallEvent::ToolCallEnd => {
                self.active_tool_index = None;
                self.mode = JsonToolCallMode::Text;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = JsonToolCallMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parse a JSON tool-call event for the current parser mode.
fn parse_next_json_tool_call_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut JsonToolCallMode,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    match mode {
        JsonToolCallMode::Text => parse_text_event(input, config),
        JsonToolCallMode::Header => tool_call_header_event(input, config),
        JsonToolCallMode::Arguments { json_scan } => {
            parse_arguments_event(input, json_scan, config)
        }
    }
}

/// Parse a text-mode JSON tool-call event.
fn parse_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    alt((
        |input: &mut JsonToolInput<'_>| tool_call_start_event(input, config),
        |input: &mut JsonToolInput<'_>| safe_text_event(input, config),
    ))
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call start marker.
fn tool_call_start_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: literal(config.start_marker),
        _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
    )
    .value(JsonToolCallEvent::ToolCallStart)
    .parse_next(input)
}

/// Parse a marker-wrapped JSON tool-call header before the raw arguments
/// payload.
fn tool_call_header_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let (function_name,) = seq!(
        _: ws0,
        _: literal("{"),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_key(input, config.name_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
        json_str,
        _: ws0,
        _: literal(","),
        _: ws0,
        _: |input: &mut JsonToolInput<'_>| json_arguments_key(input, config.arguments_key),
        _: ws0,
        _: literal(":"),
        _: ws0,
    )
    .context(StrContext::Label(config.parser_name))
    .parse_next(input)?;

    Ok(JsonToolCallEvent::ToolCallHeader { function_name })
}

/// Parse a configured JSON object key.
fn json_key(input: &mut JsonToolInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parse a JSON object key accepting any of `candidates`.
///
/// The full quoted key is consumed and compared against the candidate list,
/// so this works correctly under partial input regardless of key lengths.
///
/// On mismatch, each candidate is attached as its own `Expected` context so the
/// error enumerates every valid key ("expected `a`, expected `b`"). Because
/// `StrContextValue::StringLiteral` carries a single `&'static str`, the
/// contexts are added in a loop over `candidates` rather than through chained
/// `.context(...)` calls, which keeps the diagnostics complete for any number
/// of candidates.
fn json_arguments_key(
    input: &mut JsonToolInput<'_>,
    candidates: &'static [&'static str],
) -> ModalResult<()> {
    let start = input.checkpoint();
    json_str
        .verify(|key: &String| candidates.contains(&key.as_str()))
        .void()
        .parse_next(input)
        .map_err(|err| {
            err.map(|context_error| {
                candidates
                    .iter()
                    .fold(context_error, |context_error, candidate| {
                        context_error.add_context(
                            &*input,
                            &start,
                            StrContext::Expected(StrContextValue::StringLiteral(candidate)),
                        )
                    })
            })
        })
}

/// Parse one event inside a marker-wrapped JSON tool-call arguments payload.
fn parse_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    if json_scan.complete() {
        tool_call_close_event(input, config)
    } else {
        argument_delta_event(input, json_scan)
    }
}

/// Parse a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<JsonToolCallEvent> {
    take_json_object(input, json_scan).map(|len| JsonToolCallEvent::Arguments { len })
}

/// Parse a marker-wrapped JSON tool-call close marker.
fn tool_call_close_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let _ = literal("}").parse_next(input)?;

    tool_call_end_event(input, config)
}

/// Parse a marker-wrapped JSON tool-call end marker.
fn tool_call_end_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    seq!(
        _: |input: &mut JsonToolInput<'_>| marker_whitespace(input, config),
        _: literal(config.end_marker),
    )
    .value(JsonToolCallEvent::ToolCallEnd)
    .parse_next(input)
}

/// Parse configured whitespace around a marker-wrapped JSON tool call.
fn marker_whitespace(input: &mut JsonToolInput<'_>, config: JsonToolCallConfig) -> ModalResult<()> {
    literal(config.marker_whitespace).void().parse_next(input)
}

/// Parse a safe text run before the next marker-wrapped JSON tool call.
fn safe_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    safe_text_len(input, config.start_marker).map(|len| JsonToolCallEvent::Text { len })
}
