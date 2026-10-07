//! Incremental parser for Gemma-4 tool calls.
//!
//! Gemma-4 writes each tool call as
//! `<|tool_call>call:NAME{ARGUMENTS}<tool_call|>`, where `<|tool_call>` and
//! `<tool_call|>` are tokenizer special tokens and consecutive calls follow
//! each other directly. ARGUMENTS uses Gemma-4's native value encoding, the
//! one its chat template renders for tool calls, rather than JSON:
//!
//! - object members are `key:value` pairs separated by `,`, with keys
//!   written verbatim;
//! - a string is raw text between two `<|"|>` special tokens, without
//!   escapes;
//! - numbers, `true`, `false`, and `null` are bare JSON literals;
//! - objects `{...}` and arrays `[...]` nest as in JSON.
//!
//! For example,
//! `<|tool_call>call:get_weather{days:3,location:<|"|>Tokyo<|"|>}<tool_call|>`
//! calls `get_weather` with the JSON arguments
//! `{"days":3,"location":"Tokyo"}`. Whitespace between tokens is accepted,
//! and a key may also be written as a `<|"|>` string.
//!
//! The parser alternates between two modes over a buffered stream:
//!
//! - Text mode emits text up to `<|tool_call>` and holds back a trailing
//!   partial marker.
//! - Call mode buffers the call up to its `<tool_call|>` marker, which counts
//!   only outside strings, then translates the whole call into JSON and
//!   publishes it as one `ToolCallDelta` carrying the name and the complete
//!   arguments.
//!
//! A call is published only after it parses completely, so a malformed call
//! never yields a partial tool call. Its `<|tool_call>` marker is returned as
//! plain text and text mode resumes right after it, so the rest of the
//! attempted call becomes text while a later well-formed call still parses.
//! The call header is checked as it arrives, so text that cannot begin
//! `call:NAME{` is released without waiting for an end marker, and a
//! `<|tool_call>` outside strings ends an attempted call that never closed.

use tracing::warn;
use winnow::ascii::multispace0;
use winnow::combinator::{delimited, dispatch, eof, peek, preceded, separated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::token::{any, take_until, take_while};

use super::utils::{MAX_BUFFER_BYTES, partial_prefix_len};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};

/// Special token that opens one tool call.
const TOOL_CALL_START: &str = "<|tool_call>";

/// Special token that closes one tool call.
const TOOL_CALL_END: &str = "<tool_call|>";

/// Special token on both sides of a string value.
const STRING_DELIMITER: &str = "<|\"|>";

/// Text between [`TOOL_CALL_START`] and the function name.
const CALL_PREFIX: &str = "call:";

/// Deepest object and array nesting accepted in call arguments, counting the
/// arguments object itself. It is the deepest nesting that `serde_json`
/// deserializes under its default recursion limit of 128, so the arguments of
/// every published call also deserialize, and it bounds the parser's
/// recursion.
const MAX_NESTING_DEPTH: usize = 127;

/// Tool parser for Gemma-4 native tool calls; see the module documentation
/// for the grammar and the streaming behavior.
///
/// Function names are not checked against the declared tools.
#[derive(Debug, Default)]
pub struct Gemma4ToolParser {
    /// Input not yet represented in output: in text mode a held-back partial
    /// `<|tool_call>` marker, in call mode the call text after the marker.
    buffer: String,
    mode: Mode,
    /// Number of calls published since the last reset; also the next call's
    /// `tool_index`.
    published_calls: usize,
}

/// Parser mode; see the module documentation for the transitions.
#[derive(Debug, Default)]
enum Mode {
    #[default]
    Text,
    /// Inside an attempted call whose `<|tool_call>` marker has been consumed.
    Call(CallScan),
}

impl Gemma4ToolParser {
    /// Creates a parser in text mode with no published calls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Emits buffered text up to the next `<|tool_call>` marker.
    ///
    /// Returns `true` after consuming a marker and entering call mode, and
    /// `false` once the buffer holds at most a partial marker.
    fn emit_text_until_call(&mut self, output: &mut ToolParserOutput) -> bool {
        if let Some(start) = self.buffer.find(TOOL_CALL_START) {
            output.push_text(&self.buffer[..start]);
            self.buffer.drain(..start + TOOL_CALL_START.len());
            self.mode = Mode::Call(CallScan::default());
            return true;
        }

        let held_len = partial_prefix_len(&self.buffer, TOOL_CALL_START);
        let text_len = self.buffer.len() - held_len;
        output.push_text(&self.buffer[..text_len]);
        self.buffer.drain(..text_len);
        false
    }

    /// Abandons the attempted call: its `<|tool_call>` marker becomes plain
    /// text, and text mode rescans the buffered call text after it.
    fn reject_call(&mut self, reason: String, output: &mut ToolParserOutput) {
        warn!(%reason, "emitting malformed Gemma-4 tool call as text");
        output.push_text(TOOL_CALL_START);
        self.mode = Mode::Text;
    }
}

impl ToolParser for Gemma4ToolParser {
    /// Parses one text chunk into an existing output accumulator.
    ///
    /// Malformed calls become plain text instead of failing. Fails with
    /// `ToolParserError::ParsingFailed` only when an unterminated call grows
    /// the pending buffer past its size cap; `reset` then returns that call,
    /// `<|tool_call>` marker included.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        loop {
            let scan = match &mut self.mode {
                Mode::Text => {
                    if !self.emit_text_until_call(output) {
                        break;
                    }
                    continue;
                }
                Mode::Call(scan) => scan.advance(&self.buffer),
            };

            match scan {
                CallScanOutcome::Pending => break,
                CallScanOutcome::Rejected { reason } => self.reject_call(reason, output),
                CallScanOutcome::Complete { body_len } => {
                    match parse_call(&self.buffer[..body_len]) {
                        Ok(call) => {
                            output.push_call(ToolCallDelta {
                                tool_index: self.published_calls,
                                name: Some(call.name),
                                arguments: call.arguments,
                            });
                            self.published_calls += 1;
                            self.buffer.drain(..body_len + TOOL_CALL_END.len());
                            self.mode = Mode::Text;
                        }
                        Err(reason) => self.reject_call(reason, output),
                    }
                }
            }
        }

        if self.buffer.len() > MAX_BUFFER_BYTES {
            return Err(parsing_failed!(
                "Gemma-4 tool call exceeded {} bytes without its end marker",
                MAX_BUFFER_BYTES
            ));
        }

        Ok(())
    }

    /// Flushes buffered input as plain text and resets the parser.
    ///
    /// A call without its end marker, such as one cut off by the output
    /// length limit, is returned whole, `<|tool_call>` marker included. Never
    /// fails, because no published call is ever left open.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        output.push_text(&self.reset());
        Ok(output)
    }

    /// Resets parser state and returns the buffered input that no emitted
    /// output represents: a held-back partial marker, or the unfinished call
    /// with its `<|tool_call>` marker. Tool indices restart at 0 afterwards.
    fn reset(&mut self) -> String {
        let mut unpublished = std::mem::take(&mut self.buffer);
        if matches!(self.mode, Mode::Call(_)) {
            unpublished.insert_str(0, TOOL_CALL_START);
        }

        self.mode = Mode::Text;
        self.published_calls = 0;
        unpublished
    }
}

/// Incremental lexical scan of one call's text after `<|tool_call>`.
///
/// Each chunk resumes the scan where the previous one stopped, so every byte
/// is examined once, except that a trailing partial marker is examined again
/// when the next chunk arrives.
#[derive(Debug, Default)]
struct CallScan {
    phase: CallPhase,
    /// Byte offset in the call text up to which the scan has advanced.
    offset: usize,
}

/// Part of the call the scan is in.
#[derive(Debug, Default, Clone, Copy)]
enum CallPhase {
    /// Matching the `call:` prefix.
    #[default]
    Prefix,
    /// Reading the function name up to the `{` that opens the arguments.
    Name,
    /// Looking for the end marker inside the arguments. `in_string` records
    /// whether the scan is between `<|"|>` delimiters, where markers are
    /// string text.
    Arguments { in_string: bool },
}

/// Result of advancing a [`CallScan`] over the buffered call text.
#[derive(Debug)]
enum CallScanOutcome {
    /// The call text is a valid prefix of a call; more input is needed.
    Pending,
    /// The call text cannot become a call.
    Rejected { reason: String },
    /// The end marker starts at byte `body_len` of the call text.
    Complete { body_len: usize },
}

impl CallScan {
    /// Advances the scan over `body`, the call text buffered so far.
    fn advance(&mut self, body: &str) -> CallScanOutcome {
        loop {
            match self.phase {
                CallPhase::Prefix => {
                    // Compare bytes: the prefix is ASCII, while `body` may end
                    // inside a multi-byte character.
                    let compared = body.len().min(CALL_PREFIX.len());
                    if body.as_bytes()[..compared] != CALL_PREFIX.as_bytes()[..compared] {
                        return rejected("call text does not start with `call:`");
                    }
                    if compared < CALL_PREFIX.len() {
                        return CallScanOutcome::Pending;
                    }
                    self.offset = CALL_PREFIX.len();
                    self.phase = CallPhase::Name;
                }
                CallPhase::Name => {
                    let rest = &body[self.offset..];
                    match rest.char_indices().find(|&(_, c)| !is_name_char(c)) {
                        None => {
                            self.offset = body.len();
                            return CallScanOutcome::Pending;
                        }
                        Some((index, '{')) if self.offset + index > CALL_PREFIX.len() => {
                            self.offset += index + 1;
                            self.phase = CallPhase::Arguments { in_string: false };
                        }
                        Some(_) => return rejected("function name is not followed by `{`"),
                    }
                }
                CallPhase::Arguments { in_string } => {
                    // Every marker starts with `<`, which is ASCII, so the
                    // offsets below stay on character boundaries.
                    let Some(index) = body[self.offset..].find('<') else {
                        self.offset = body.len();
                        return CallScanOutcome::Pending;
                    };
                    let marker_start = self.offset + index;
                    let tail = &body[marker_start..];

                    if tail.starts_with(STRING_DELIMITER) {
                        self.phase = CallPhase::Arguments {
                            in_string: !in_string,
                        };
                        self.offset = marker_start + STRING_DELIMITER.len();
                    } else if in_string {
                        if STRING_DELIMITER.starts_with(tail) {
                            // A delimiter may be split across chunks.
                            self.offset = marker_start;
                            return CallScanOutcome::Pending;
                        }
                        self.offset = marker_start + 1;
                    } else if tail.starts_with(TOOL_CALL_END) {
                        return CallScanOutcome::Complete {
                            body_len: marker_start,
                        };
                    } else if tail.starts_with(TOOL_CALL_START) {
                        return rejected("a new tool call starts before the end marker");
                    } else if [STRING_DELIMITER, TOOL_CALL_END, TOOL_CALL_START]
                        .iter()
                        .any(|marker| marker.starts_with(tail))
                    {
                        // A marker may be split across chunks.
                        self.offset = marker_start;
                        return CallScanOutcome::Pending;
                    } else {
                        self.offset = marker_start + 1;
                    }
                }
            }
        }
    }
}

/// Builds a rejection with a static reason.
fn rejected(reason: &str) -> CallScanOutcome {
    CallScanOutcome::Rejected {
        reason: reason.to_string(),
    }
}

/// Returns whether `c` may appear in a function name: the OpenAI function
/// name alphabet (letters, digits, `_`, and `-`) extended with `.` and
/// non-ASCII letters and digits.
fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.')
}

/// Returns whether `c` may appear in a bare object key. The template writes
/// keys verbatim, so a key may contain inner spaces, but not the structural
/// characters, `"`, or `<`, which starts every special-token marker.
fn is_bare_key_char(c: char) -> bool {
    !matches!(c, ':' | ',' | '{' | '}' | '[' | ']' | '"' | '<')
}

/// Returns whether `c` may appear in a bare scalar value, which runs until
/// whitespace or a character that ends a value.
fn is_scalar_char(c: char) -> bool {
    !c.is_whitespace() && !matches!(c, ',' | '}' | ']')
}

/// One complete call translated to JSON arguments.
#[derive(Debug)]
struct ParsedCall {
    name: String,
    /// Compact JSON object text of the arguments, members in written order.
    arguments: String,
}

/// Parses one complete call, the text between `<|tool_call>` and
/// `<tool_call|>`, and translates its arguments to JSON.
///
/// Returns the parse error description when the text is not a call.
fn parse_call(body: &str) -> std::result::Result<ParsedCall, String> {
    let mut input = body;
    call.parse_next(&mut input)
        .map_err(|error| format!("{error} at byte {}", body.len() - input.len()))
}

/// `call:NAME{...}`, optionally followed by whitespace.
fn call(input: &mut &str) -> ModalResult<ParsedCall> {
    let name = preceded(CALL_PREFIX, take_while(1.., is_name_char))
        .context(StrContext::Label("function name"))
        .parse_next(input)?;
    let arguments = object(input, 0)?;
    (multispace0, eof)
        .context(StrContext::Expected(StrContextValue::StringLiteral(
            TOOL_CALL_END,
        )))
        .parse_next(input)?;

    Ok(ParsedCall {
        name: name.to_string(),
        arguments,
    })
}

/// Any value, dispatched on its first character.
fn value(input: &mut &str, depth: usize) -> ModalResult<String> {
    dispatch! {peek(any);
        '{' => |input: &mut &str| object(input, depth),
        '[' => |input: &mut &str| array(input, depth),
        '<' => string,
        _ => scalar,
    }
    .parse_next(input)
}

/// `{key:value,...}` translated to a JSON object.
fn object(input: &mut &str, depth: usize) -> ModalResult<String> {
    let depth = nested(depth)?;
    ('{', multispace0).parse_next(input)?;
    let members: Vec<String> = separated(
        0..,
        |input: &mut &str| member(input, depth),
        (multispace0, ',', multispace0),
    )
    .parse_next(input)?;
    (multispace0, '}')
        .context(StrContext::Label("object"))
        .parse_next(input)?;

    Ok(format!("{{{}}}", members.join(",")))
}

/// `key:value` translated to a JSON object member.
fn member(input: &mut &str, depth: usize) -> ModalResult<String> {
    // Whitespace before a key is consumed by the preceding `{` or `,`, so
    // only trailing whitespace before `:` remains to trim from a bare key.
    let key = dispatch! {peek(any);
        '<' => string,
        _ => take_while(1.., is_bare_key_char)
            .verify(|key: &str| !key.trim_end().is_empty())
            .map(|key: &str| json_string(key.trim_end())),
    }
    .parse_next(input)?;
    (multispace0, ':', multispace0).parse_next(input)?;
    let value = value(input, depth)?;

    Ok(format!("{key}:{value}"))
}

/// `[value,...]` translated to a JSON array.
fn array(input: &mut &str, depth: usize) -> ModalResult<String> {
    let depth = nested(depth)?;
    ('[', multispace0).parse_next(input)?;
    let items: Vec<String> = separated(
        0..,
        |input: &mut &str| value(input, depth),
        (multispace0, ',', multispace0),
    )
    .parse_next(input)?;
    (multispace0, ']')
        .context(StrContext::Label("array"))
        .parse_next(input)?;

    Ok(format!("[{}]", items.join(",")))
}

/// Returns the depth of a container opened inside one at `depth`, or a
/// non-backtracking error past [`MAX_NESTING_DEPTH`].
fn nested(depth: usize) -> ModalResult<usize> {
    if depth >= MAX_NESTING_DEPTH {
        let mut error = ContextError::new();
        error.push(StrContext::Label("nesting depth"));
        return Err(ErrMode::Cut(error));
    }
    Ok(depth + 1)
}

/// `<|"|>text<|"|>` translated to a JSON string.
fn string(input: &mut &str) -> ModalResult<String> {
    delimited(
        STRING_DELIMITER,
        take_until(0.., STRING_DELIMITER),
        STRING_DELIMITER,
    )
    .map(json_string)
    .context(StrContext::Label("string"))
    .parse_next(input)
}

/// A bare `true`, `false`, `null`, or JSON number, copied verbatim.
fn scalar(input: &mut &str) -> ModalResult<String> {
    take_while(1.., is_scalar_char)
        .verify(|text: &str| matches!(text, "true" | "false" | "null") || is_json_number(text))
        .map(str::to_string)
        .context(StrContext::Label("scalar"))
        .parse_next(input)
}

/// Encodes raw text as a JSON string literal.
fn json_string(text: &str) -> String {
    serde_json::Value::String(text.to_string()).to_string()
}

/// Returns whether `text` is exactly one number in JSON syntax:
/// `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
fn is_json_number(text: &str) -> bool {
    fn digits_len(text: &str) -> usize {
        text.bytes().take_while(u8::is_ascii_digit).count()
    }

    let mut rest = text.strip_prefix('-').unwrap_or(text);
    let integer_len = digits_len(rest);
    if integer_len == 0 || (integer_len > 1 && rest.starts_with('0')) {
        return false;
    }
    rest = &rest[integer_len..];

    if let Some(fraction) = rest.strip_prefix('.') {
        let fraction_len = digits_len(fraction);
        if fraction_len == 0 {
            return false;
        }
        rest = &fraction[fraction_len..];
    }

    if let Some(exponent) = rest.strip_prefix(['e', 'E']) {
        let exponent = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        let exponent_len = digits_len(exponent);
        if exponent_len == 0 {
            return false;
        }
        rest = &exponent[exponent_len..];
    }

    rest.is_empty()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{Gemma4ToolParser, MAX_NESTING_DEPTH};
    use crate::profile::tools::test_utils::{collect_stream, split_by_chars};
    use crate::profile::tools::utils::MAX_BUFFER_BYTES;
    use crate::profile::tools::{ToolParser, ToolParserItem, ToolParserOutput};

    /// Parser output with each call's arguments parsed as JSON, so comparisons
    /// do not depend on JSON formatting.
    #[derive(Debug, PartialEq)]
    enum Item {
        Text(String),
        Call {
            tool_index: usize,
            name: String,
            arguments: Value,
        },
    }

    fn text(text: &str) -> Item {
        Item::Text(text.to_string())
    }

    fn call(tool_index: usize, name: &str, arguments: Value) -> Item {
        Item::Call {
            tool_index,
            name: name.to_string(),
            arguments,
        }
    }

    fn items(output: ToolParserOutput) -> Vec<Item> {
        output
            .items
            .into_iter()
            .map(|item| match item {
                ToolParserItem::Text(text) => Item::Text(text),
                ToolParserItem::Call(call) => Item::Call {
                    tool_index: call.tool_index,
                    name: call.name.expect("a published call carries its name"),
                    arguments: serde_json::from_str(&call.arguments)
                        .expect("published arguments are JSON"),
                },
            })
            .collect()
    }

    /// Asserts that `input` parses to `expected` when streamed one, two,
    /// three, and seven characters at a time, and as one chunk.
    fn assert_parses(input: &str, expected: &[Item]) {
        let whole = input.chars().count().max(1);
        for chunk_chars in [1, 2, 3, 7, whole] {
            let mut parser = Gemma4ToolParser::new();
            let output = collect_stream(&mut parser, &split_by_chars(input, chunk_chars));
            assert_eq!(
                items(output),
                expected,
                "chunks of {chunk_chars} characters"
            );
        }
    }

    /// Every native value type translates to the JSON value the template
    /// would have rendered it from: strings keep quotes, backslashes, braces,
    /// newlines, `<`, and non-ASCII text literally, and a key is written
    /// verbatim or as a `<|"|>` string.
    #[test]
    fn gemma4_translates_native_arguments_to_json() {
        let input = concat!(
            "<|tool_call>call:plan_trip{",
            "budget:1250.5,",
            "flags:[true,false,null],",
            r#"meta:{quote:<|"|>He said "hi" {x:1}, \ and <b>"#,
            "\n",
            r#"done<|"|>,ratio:-2.5e-3,none:{},list:[]},"#,
            r#"<|"|>first name<|"|>:<|"|>Zoë<|"|>,last name:<|"|>Durand<|"|>,"#,
            r#"stops:[{city:<|"|>Paris<|"|>,nights:2},{city:<|"|>Lyon<|"|>,nights:1}]"#,
            "}<tool_call|>",
        );

        assert_parses(
            input,
            &[call(
                0,
                "plan_trip",
                json!({
                    "budget": 1250.5,
                    "flags": [true, false, null],
                    "meta": {
                        "quote": "He said \"hi\" {x:1}, \\ and <b>\ndone",
                        "ratio": -2.5e-3,
                        "none": {},
                        "list": []
                    },
                    "first name": "Zoë",
                    "last name": "Durand",
                    "stops": [{"city": "Paris", "nights": 2}, {"city": "Lyon", "nights": 1}]
                }),
            )],
        );
    }

    /// Text around and between consecutive calls keeps its position, calls
    /// take consecutive indices, and whitespace between argument tokens is
    /// insignificant.
    #[test]
    fn gemma4_keeps_text_and_calls_in_order() {
        let input = concat!(
            "Let me check.",
            r#"<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>"#,
            "<|tool_call>call:add{ x : 1 , y : [ 2 , 3 ] }<tool_call|>",
            "Done.",
        );

        assert_parses(
            input,
            &[
                text("Let me check."),
                call(0, "get_weather", json!({"location": "Tokyo"})),
                call(1, "add", json!({"x": 1, "y": [2, 3]})),
                text("Done."),
            ],
        );
    }

    /// Markers inside a string value are string text.
    #[test]
    fn gemma4_markers_inside_strings_are_argument_text() {
        let input = concat!(
            r#"<|tool_call>call:run{code:<|"|>emit("<tool_call|>");emit("<|tool_call>")<|"|>}"#,
            "<tool_call|>",
        );

        assert_parses(
            input,
            &[call(
                0,
                "run",
                json!({"code": r#"emit("<tool_call|>");emit("<|tool_call>")"#}),
            )],
        );
    }

    /// A malformed call is returned verbatim as text, publishes nothing, and
    /// leaves a following well-formed call parseable.
    #[test]
    fn gemma4_returns_malformed_calls_as_text() {
        let valid = r#"<|tool_call>call:add{x:<|"|>ok<|"|>}<tool_call|>"#;
        for malformed in [
            // An unquoted string value.
            "<|tool_call>call:get_weather{location:Tokyo}<tool_call|>",
            // Numbers outside JSON number syntax.
            "<|tool_call>call:add{x:01}<tool_call|>",
            "<|tool_call>call:add{x:1.}<tool_call|>",
            "<|tool_call>call:add{x:NaN}<tool_call|>",
            // A trailing separator.
            "<|tool_call>call:add{x:1,}<tool_call|>",
            // A missing `call:` prefix.
            "<|tool_call>add{x:1}<tool_call|>",
            // An empty function name.
            "<|tool_call>call:{x:1}<tool_call|>",
            // A call that never closes before the next one starts.
            "<|tool_call>call:add{x:1}",
            // Text after the arguments object.
            "<|tool_call>call:add{x:1}}<tool_call|>",
        ] {
            assert_parses(
                &format!("{malformed}{valid}"),
                &[text(malformed), call(0, "add", json!({"x": "ok"}))],
            );
        }
    }

    /// Text that cannot begin a call header streams out with its chunk
    /// instead of waiting for an end marker.
    #[test]
    fn gemma4_releases_text_after_marker_without_call_header() {
        let mut parser = Gemma4ToolParser::new();

        let output = parser.parse_chunk("<|tool_call>Hello").unwrap();

        assert_eq!(items(output), [text("<|tool_call>Hello")]);
    }

    /// A call cut off before its end marker, as by the output length limit,
    /// is returned whole as text at end of stream.
    #[test]
    fn gemma4_finish_returns_unterminated_call_as_text() {
        let mut parser = Gemma4ToolParser::new();
        let input = r#"Checking.<|tool_call>call:get_weather{location:<|"|>Tok"#;

        let streamed = parser.parse_chunk(input).unwrap();
        let finished = parser.finish().unwrap();

        assert_eq!(items(streamed), [text("Checking.")]);
        assert_eq!(items(finished), [text(&input["Checking.".len()..])]);
    }

    /// Arguments nested as deeply as `serde_json` deserializes are published;
    /// one level deeper, the call is text.
    #[test]
    fn gemma4_bounds_argument_nesting() {
        let nested_call = |depth: usize| {
            // The arguments object is the outermost level.
            let arrays = depth - 1;
            format!(
                "<|tool_call>call:f{{x:{}{}}}<tool_call|>",
                "[".repeat(arrays),
                "]".repeat(arrays)
            )
        };

        let deepest = nested_call(MAX_NESTING_DEPTH);
        let output = Gemma4ToolParser::new().parse_complete(&deepest).unwrap();
        let arguments = &output.calls().next().expect("call is published").arguments;
        assert!(serde_json::from_str::<Value>(arguments).is_ok());

        let too_deep = nested_call(MAX_NESTING_DEPTH + 1);
        let output = Gemma4ToolParser::new().parse_complete(&too_deep).unwrap();
        assert_eq!(items(output), [text(&too_deep)]);
    }

    /// An unterminated call fails parsing once the buffer passes its cap,
    /// and `reset` returns the whole attempted call.
    #[test]
    fn gemma4_bounds_unterminated_call_buffer() {
        let mut parser = Gemma4ToolParser::new();
        let input = format!(
            "<|tool_call>call:f{{s:<|\"|>{}",
            "a".repeat(MAX_BUFFER_BYTES)
        );

        let mut output = ToolParserOutput::default();
        let result = parser.parse_into(&input, &mut output);

        assert!(result.is_err());
        assert!(output.items.is_empty());
        assert_eq!(parser.reset(), input);
    }
}
