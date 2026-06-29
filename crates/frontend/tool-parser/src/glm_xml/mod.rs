use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until, take_while};

use super::parameters::ToolSchemas;
use super::utils::{ValueTrim, normalize_xml_value, parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParserOutput};
use crate::Tool;

mod glm45_moe;
mod glm47_moe;

pub use glm45_moe::Glm45MoeToolParser;
pub use glm47_moe::Glm47MoeToolParser;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

type GlmInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmMode {
    Text,
    ToolCall,
    AfterToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Separator {
 /// GLM-4.5/4.6 format: function name must end at a newline before
 /// arguments.
    Newline,
 /// GLM-4.7 format: function name may end at whitespace or directly before
 /// `<arg_key>`.
    Flexible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GlmEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    IgnoredRest,
}

/// Tool parser core for GLM XML-style tool calls.
struct GlmXmlToolParser {
    buffer: String,
    mode: GlmMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    separator: Separator,
 /// Number of leading buffer bytes already scanned for [`TOOL_CALL_END`]
 /// while in [`GlmMode::ToolCall`].

 /// The buffered tool-call body is re-fed to the parser on every chunk, so
 /// without this watermark each chunk re-scans the whole accumulated body
 /// for the closing marker (`O(n^2)` over a body delivered in `n` chunks).
 /// We instead only search the freshly appended tail, keeping it in `[..)`
 /// of `tool_call_scan_offset`, and reset it whenever the buffer is drained
 /// or the parser is reset.
    tool_call_scan_offset: usize,
}

impl GlmXmlToolParser {
 /// Create a GLM XML tool parser with a function-name separator.
    fn new(tools: &[Tool], separator: Separator) -> Self {
        Self {
            buffer: String::new(),
            mode: GlmMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            separator,
            tool_call_scan_offset: 0,
        }
    }

 /// Apply one parsed GLM event to parser state and output.
    fn apply_event(&mut self, event: GlmEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            GlmEvent::Text { len: consumed_len } => {
                output.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            GlmEvent::ToolCallStart => self.mode = GlmMode::ToolCall,
            GlmEvent::ToolCall { name, raw_params } => {
                self.mode = GlmMode::AfterToolCall;
                let arguments = self
                    .tool_parameters
                    .convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.calls.push(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            GlmEvent::IgnoredRest => {}
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = GlmMode::Text;
        self.emitted_tool_count = 0;
        self.tool_call_scan_offset = 0;
        std::mem::take(&mut self.buffer)
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        loop {
 // While buffering an incomplete tool-call body, cheaply gate on the
 // closing marker so we only scan the freshly appended tail instead
 // of re-running the body parser over the whole accumulated buffer
 // on every chunk.
            if self.mode == GlmMode::ToolCall && !self.tool_call_end_buffered() {
                break;
            }

            let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
                parse_next_glm_event(input, self.mode, self.separator)
            })?
            else {
                break;
            };

            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
 // The remaining buffer shifted; restart the closing-marker scan.
            self.tool_call_scan_offset = 0;
        }

        Ok(())
    }

 /// Returns whether the buffered tool-call body already contains the closing
 /// [`TOOL_CALL_END`] marker, advancing the scan watermark over the bytes
 /// that have been confirmed not to contain it.

 /// `tool_call_scan_offset` records how many leading buffer bytes were
 /// already searched on a previous chunk. Only the freshly appended tail is
 /// examined here, re-checking the last `TOOL_CALL_END.len - 1` bytes of
 /// the already-scanned region so a marker straddling the watermark is
 /// not missed.
    fn tool_call_end_buffered(&mut self) -> bool {
 // Overlap the search so a marker that straddles the watermark is found.
        let search_from = self
            .tool_call_scan_offset
            .saturating_sub(TOOL_CALL_END.len() - 1);
        let search_from = floor_char_boundary(&self.buffer, search_from);

        if self.buffer[search_from..].contains(TOOL_CALL_END) {
            return true;
        }

 // No full marker yet: the entire current buffer has now been searched.
        self.tool_call_scan_offset = self.buffer.len();
        false
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        if !self.buffer.is_empty() {
            match self.mode {
                GlmMode::Text => output.normal_text.push_str(&self.buffer),
                GlmMode::ToolCall => return Err(parsing_failed!("incomplete GLM MoE tool call")),
                GlmMode::AfterToolCall => {}
            }
        }
        let _ = self.reset();
        Ok(output)
    }
}

/// Parse a GLM event for the current parser mode.
fn parse_next_glm_event(
    input: &mut GlmInput<'_>,
    mode: GlmMode,
    separator: Separator,
) -> ModalResult<GlmEvent> {
    match mode {
        GlmMode::Text => parse_text_event(input),
        GlmMode::ToolCall => tool_call_event(input, separator),
        GlmMode::AfterToolCall => after_tool_call_event(input),
    }
}

/// Parse a text-mode GLM event.
fn parse_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a GLM tool-call start marker.
fn tool_call_start_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    literal(TOOL_CALL_START)
        .value(GlmEvent::ToolCallStart)
        .parse_next(input)
}

/// Parse a safe text run before the next GLM marker.
fn safe_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| GlmEvent::Text { len })
}

/// Parse text after a completed GLM tool call.
fn after_tool_call_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    ws0.void().parse_next(input)?;
    alt((tool_call_start_event, ignored_rest_event)).parse_next(input)
}

/// Parse a trailing rest after GLM tool calls.
fn ignored_rest_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    rest.value(GlmEvent::IgnoredRest).parse_next(input)
}

/// Return the largest byte index `<= index` that is a char boundary of `text`.

/// A stable-Rust stand-in for the unstable `str::floor_char_boundary`, used to
/// slice the buffer at an arbitrary scan watermark without splitting a
/// multi-byte UTF-8 sequence (tool-call argument values may be non-ASCII).
fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Parse a complete GLM tool call.
fn tool_call_event(input: &mut GlmInput<'_>, separator: Separator) -> ModalResult<GlmEvent> {
    let (body,) = seq!(
        take_until(0.., TOOL_CALL_END),
        _: literal(TOOL_CALL_END),
    )
    .parse_next(input)?;

    parse_tool_call_body(body, separator)
}

/// Parse a GLM tool-call body.
fn parse_tool_call_body(body: &str, separator: Separator) -> ModalResult<GlmEvent> {
    let mut input = body;
    let (name, raw_params) = match separator {
        Separator::Newline => seq!(
            _: ws0,
            parse_newline_separated_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
        Separator::Flexible => seq!(
            _: ws0,
            parse_flexible_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
    };

    Ok(GlmEvent::ToolCall {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a GLM-4.5 newline-separated function name.
fn parse_newline_separated_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(take_until(1.., "\n"), "\n")
        .map(str::trim)
        .parse_next(input)
}

/// Parse a GLM-4.7 whitespace-or-tag-separated function name.
fn parse_flexible_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(
        take_while(1.., |ch: char| !ch.is_whitespace() && ch != '<'),
        ws0,
    )
    .parse_next(input)
}

/// Parse GLM argument key-value pairs.
fn parse_parameters(input: &mut &str) -> ModalResult<Vec<(String, String)>> {
    repeat(0.., terminated(parse_parameter, ws0)).parse_next(input)
}

/// Parse a GLM argument key-value pair.
fn parse_parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (key, value) = seq!(
        _: literal(ARG_KEY_START),
        take_until(1.., ARG_KEY_END),
        _: literal(ARG_KEY_END),
        _: ws0,
        _: literal(ARG_VALUE_START),
        take_until(0.., ARG_VALUE_END).map(|value: &str| normalize_xml_value(value, ValueTrim::Whitespace)),
        _: literal(ARG_VALUE_END),
    )
    .parse_next(input)?;

    Ok((key.trim().to_string(), value))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::Glm45MoeToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParser, ToolParserTestExt as _};

    fn glm45_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!("<arg_key>{name}</arg_key>\n<arg_value>{value}</arg_value>")
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>{function_name}\n{params}\n</tool_call>")
    }

    #[test]
    fn glm45_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text, "Hello, world!");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn glm45_parse_complete_extracts_single_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = format!(
            "Let me search for that.\n{}",
            glm45_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2024-12-25")]
            )
        );

        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text, "Let me search for that.\n");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({"city": "Beijing", "date": "2024-12-25"})
        );
    }

    #[test]
    fn glm45_streaming_extracts_multiple_tool_calls() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = format!(
            "{}\n{}",
            glm45_tool_call("get_weather", &[("city", "Shanghai")]),
            glm45_tool_call("add", &[("x", "1"), ("y", "2")])
        );

        let chunks = split_by_chars(&output, 11);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 2);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls[1].name.as_deref(), Some("add"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[1].arguments).unwrap(),
            json!({"x": 1, "y": 2})
        );
    }

    #[test]
    fn glm45_parse_complete_unescapes_literal_closing_tags_in_arg_value() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&glm45_tool_call(
                "get_weather",
                &[
                    ("city", "Paris &lt;/arg_value&gt;&lt;/tool_call&gt;"),
                    ("date", "2026-05-08"),
                ],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({
                "city": "Paris </arg_value></tool_call>",
                "date": "2026-05-08",
            })
        );
    }

    #[test]
    fn glm45_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &["hello ", "world"]);

        assert_eq!(output.normal_text, "hello world");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn glm45_streaming_preserves_prefix_text() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let output = collect_stream(
            &mut parser,
            &[
                "Prefix ",
                &glm45_tool_call("get_weather", &[("city", "Hangzhou")]),
            ],
        );

        assert_eq!(output.normal_text, "Prefix ");
        assert_eq!(output.calls.len(), 1);
    }

    #[test]
    fn glm45_streaming_handles_start_token_split_across_chunks() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "hello <tool",
                "_call>get_weather\n",
                "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            ],
        );

        assert_eq!(output.normal_text, "hello ");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn glm45_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let output = parser
            .parse_chunk("<tool_call>get_weather\n<arg_key>city</arg_key>")
            .unwrap();

        assert_eq!(output.normal_text, "");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn glm45_finish_fails_incomplete_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        parser
            .parse_chunk("<tool_call>get_weather\n<arg_key>city</arg_key>")
            .unwrap();
        let error = parser.finish().unwrap_err();

        assert!(
            error
                .as_report()
                .to_string()
                .contains("incomplete GLM MoE tool call")
        );
    }

    #[test]
    fn glm45_malformed_tool_call_fails_fast() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let error = parser.parse_chunk("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>").unwrap_err();

        assert!(
            error
                .as_report()
                .to_string()
                .contains("tool parser parsing failed")
        );
    }

    #[test]
    fn glm45_streaming_incremental_scan_handles_unicode_split_char_by_char() {
 // Stream a tool-call body one character at a time, including a
 // multi-byte argument value, to exercise the incremental closing-marker
 // scan (and its UTF-8 char-boundary clamping) across many chunks.
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let call = glm45_tool_call("get_weather", &[("city", "北京 — 東京 — 🌧")]);

        let chunks = split_by_chars(&call, 1);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({"city": "北京 — 東京 — 🌧"})
        );
    }

    #[test]
    fn glm45_streaming_ignores_trailing_text_after_tool_calls() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let output = collect_stream(
            &mut parser,
            &[&format!(
                "{}<|endoftext|>",
                glm45_tool_call("get_weather", &[("city", "Paris")])
            )],
        );

        assert_eq!(output.normal_text, "");
        assert_eq!(output.calls.len(), 1);
    }
}
