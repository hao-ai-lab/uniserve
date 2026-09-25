//! Incremental parser for JSON tool calls enclosed by text markers.
//!
//! The parser is a three-mode state machine over a buffered stream:
//!
//! - `Text` emits plain text up to the start marker and holds back a trailing
//!   partial marker. The start marker followed by the configured marker
//!   whitespace switches to `Header`.
//! - `Header` parses `{"<name key>": "<name>", "<arguments key>":` with
//!   optional JSON whitespace, emits the call's first delta (name, empty
//!   arguments), and switches to `Arguments`.
//! - `Arguments` streams the raw arguments object lexically as argument
//!   deltas. After the object closes, it expects the wrapper's `}` after
//!   optional JSON whitespace, then the marker whitespace and end marker, and
//!   returns to `Text`.
//!
//! Any other header shape, such as reordered or extra keys, is a parse error.

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
/// The buffer retains every byte that has not yet formed a complete event.
/// Argument bytes leave it as soon as they arrive, but input that cannot yet
/// complete an event, such as an unterminated tool-call header (for example an
/// endless function-name string), keeps accumulating. Exceeding the cap fails
/// `parse_into` with a parse error instead of growing memory without bound.
const MAX_BUFFER_BYTES: usize = 1 << 20;

/// Marker and key vocabulary of one marker-wrapped JSON tool-call format.
#[derive(Debug, Clone, Copy)]
struct JsonToolCallConfig {
    /// Name used in parse error messages.
    parser_name: &'static str,
    start_marker: &'static str,
    end_marker: &'static str,
    /// Exact text required after `start_marker` and before `end_marker`. It is
    /// matched literally, not as optional whitespace.
    marker_whitespace: &'static str,
    /// JSON key of the function name, which must be the header's first key.
    name_key: &'static str,
    /// Candidate JSON keys naming the arguments payload, which must follow the
    /// name. The header accepts any one of them.
    arguments_key: &'static [&'static str],
}

/// Parser mode; see the module documentation for the transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonToolCallMode {
    Text,
    Header,
    Arguments { json_scan: JsonObjectScanState },
}

/// One parsed unit of the buffered stream.
///
/// `len` fields count bytes at the head of the parser buffer that
/// `JsonToolCallParser::apply_event` copies into the output before the buffer
/// drains them.
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
    /// Input not yet consumed by a complete event.
    buffer: String,
    mode: JsonToolCallMode,
    /// Tool index that argument deltas extend; set by a header, cleared by the
    /// end marker.
    active_tool_index: Option<usize>,
    /// Number of tool calls started since the last reset; also the next
    /// call's `tool_index`.
    emitted_tool_count: usize,
}

impl JsonToolCallParser {
    /// Creates a marker-wrapped JSON tool-call parser.
    fn new(config: JsonToolCallConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            mode: JsonToolCallMode::Text,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Advances incremental parsing and appends every complete event to `output`.
    ///
    /// On error, events parsed before the failure stay in `output` and the
    /// buffer keeps the input that no completed event consumed.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        let config = self.config;

        // An event's bytes are applied before they are drained, because text and
        // argument events copy their payload out of the buffer head.
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

    /// Finalizes buffered input or rejects an incomplete tool call.
    ///
    /// In text mode the buffer, including any held-back partial marker, becomes
    /// plain text and the parser resets. While a tool call is open (from its
    /// start marker until its end marker is parsed) this fails and leaves the
    /// parser state unchanged.
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

    /// Applies one parsed JSON tool-call event to parser state and output.
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

    /// Resets the incremental parser state and returns the unconsumed buffer.
    ///
    /// Tool indices restart at 0 afterwards.
    fn reset(&mut self) -> String {
        self.mode = JsonToolCallMode::Text;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Parses a JSON tool-call event for the current parser mode.
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

/// Parses a text-mode JSON tool-call event.
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

/// Parses a marker-wrapped JSON tool-call start marker.
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

/// Parses a marker-wrapped JSON tool-call header before the raw arguments
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

/// Parses a configured JSON object key.
fn json_key(input: &mut JsonToolInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parses a JSON object key accepting any of `candidates`.
///
/// The full quoted key is parsed with `json_str` and then compared against the
/// candidate list, so partial input stays `Incomplete` until the closing quote
/// is buffered, whatever the candidates' lengths.
///
/// On mismatch, each candidate is attached as its own `Expected` context so the
/// error enumerates every valid key ("expected `a`, expected `b`").
/// `StrContextValue::StringLiteral` carries a single `&'static str`, so the
/// contexts are folded over `candidates` rather than chained with `.context`.
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

/// Parses one event inside a marker-wrapped JSON tool-call arguments payload.
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

/// Parses a raw JSON arguments delta.
fn argument_delta_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<JsonToolCallEvent> {
    take_json_object(input, json_scan).map(|len| JsonToolCallEvent::Arguments { len })
}

/// Parses the wrapper object's `}` and the end marker after the arguments.
///
/// JSON whitespace may separate the arguments object from this `}`, as it may
/// separate every other token of the wrapper object.
fn tool_call_close_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    let _ = (ws0, literal("}")).parse_next(input)?;

    tool_call_end_event(input, config)
}

/// Parses a marker-wrapped JSON tool-call end marker.
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

/// Parses configured whitespace around a marker-wrapped JSON tool call.
fn marker_whitespace(input: &mut JsonToolInput<'_>, config: JsonToolCallConfig) -> ModalResult<()> {
    literal(config.marker_whitespace).void().parse_next(input)
}

/// Parses a safe text run before the next marker-wrapped JSON tool call.
fn safe_text_event(
    input: &mut JsonToolInput<'_>,
    config: JsonToolCallConfig,
) -> ModalResult<JsonToolCallEvent> {
    safe_text_len(input, config.start_marker).map(|len| JsonToolCallEvent::Text { len })
}
