//! Qwen XML marker configuration for the JSON tool-call parser.

use super::{JsonToolCallConfig, JsonToolCallParser};
use crate::profile::tools::{Result, Tool, ToolParser, ToolParserOutput};

/// The newlines belong to the delimiters, as in the format the Qwen3 chat
/// template instructs and SGLang's Qwen detector matches, so a bare
/// `<tool_call>` tag in prose stays plain text.
const QWEN_XML_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Qwen XML",
    start_delimiter: "<tool_call>\n",
    end_delimiter: "\n</tool_call>",
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Qwen XML-wrapped JSON tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// {"name": "get_weather", "arguments": {"location":"Tokyo"}}
/// </tool_call>
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3XmlToolParser {
    inner: JsonToolCallParser,
}

impl Qwen3XmlToolParser {
    /// Creates a Qwen XML tool parser.
    ///
    /// The declared tools are not read: parsed function names are not checked
    /// against them.
    pub fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(QWEN_XML_CONFIG),
        }
    }
}

impl ToolParser for Qwen3XmlToolParser {
    /// Parses one text chunk into an existing output accumulator.
    ///
    /// Fails with `ToolParserError::ParsingFailed` on input outside the
    /// grammar or when the pending buffer exceeds its size cap. Events parsed
    /// before the failure remain in `output`, and `reset` returns the input
    /// that `output` does not represent.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    /// Flushes buffered parser state at end of generation.
    ///
    /// Buffered input outside a published call, including an unfinished call
    /// header together with its `<tool_call>` line, is returned as plain text
    /// and the parser resets. Fails with `ToolParserError::ParsingFailed`,
    /// leaving the state unchanged, while a published call is still open; its
    /// arguments have already been emitted.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    /// Resets parser state and returns the buffered input that no emitted
    /// output represents.
    ///
    /// When a call header fails to parse, this is the whole attempted call,
    /// `<tool_call>` line included. When a published call later fails, it is
    /// the input after the arguments already emitted for that call.
    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::Qwen3XmlToolParser;
    use crate::profile::tools::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::profile::tools::{ToolCallDelta, ToolParser, ToolParserItem, ToolParserOutput};

    /// Collects the tool-call updates of `output` in order.
    fn calls(output: &ToolParserOutput) -> Vec<&ToolCallDelta> {
        output.calls().collect()
    }

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(
            "<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": {arguments}}}\n</tool_call>"
        )
    }

    #[test]
    fn qwen_xml_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(calls(&output).is_empty());
    }

    #[test]
    fn qwen_xml_parse_complete_extracts_raw_json_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": "3" }"#;
        let output = parser
            .parse_complete(&format!(
                "Let me check.\n{}",
                build_tool_call("get_weather", arguments)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check.\n");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].tool_index, 0);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_does_not_validate_or_normalize_arguments() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let output = parser
            .parse_complete(&build_tool_call("get_weather", arguments))
            .unwrap();

        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    /// Argument bytes stream out chunk by chunk: the fifth chunk closes the
    /// arguments object and the last one only closes the wrapper and the tag,
    /// so exactly three argument deltas follow the header.
    #[test]
    fn qwen_xml_streaming_emits_argument_deltas() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let chunks = [
            "<tool_call>",
            "\n{\"name\": \"get_weather\", \"arguments\": ",
            "{\"location\":",
            "\"Beijing\"",
            "}",
            "}\n</tool_call>",
        ];

        let mut output = ToolParserOutput::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            observed_arguments.extend(
                next.calls()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            calls(&output.coalesce_calls())[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn qwen_xml_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "hello ");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn qwen_xml_keeps_end_marker_literal_inside_json_string() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let arguments = r#"{"text":"literal </tool_call> inside"}"#;
        let output = parser
            .parse_complete(&build_tool_call("echo", arguments))
            .unwrap();

        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].arguments, arguments);
    }

    #[test]
    fn qwen_xml_decodes_escaped_function_name() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                r#"<tool_call>
{"name":"say_\"hi","arguments":{}}
</tool_call>"#,
            )
            .unwrap();

        assert_eq!(calls(&output)[0].name.as_deref(), Some("say_\"hi"));
    }

    /// Without the newline after `<tool_call>` the block is not a tool call,
    /// and the whole input is plain text.
    #[test]
    fn qwen_xml_requires_newline_after_tool_call_start() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let input = r#"<tool_call>{"name":"get_weather","arguments":{}}
</tool_call>"#;

        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.normal_text(), input);
        assert!(calls(&output).is_empty());
    }

    /// A `<tool_call>` tag that does not start a tool call streams out as
    /// text with its chunk instead of holding back all later text.
    #[test]
    fn qwen_xml_streams_text_after_tag_without_newline() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let first = parser.parse_chunk("Use the <tool_call> tag.").unwrap();
        let second = parser.parse_chunk(" More.").unwrap();

        assert_eq!(first.normal_text(), "Use the <tool_call> tag.");
        assert_eq!(second.normal_text(), " More.");
    }

    /// A tag that does not start a tool call leaves later well-formed calls
    /// in the same response parseable.
    #[test]
    fn qwen_xml_parses_tool_call_after_tag_without_newline() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "<tool_call> x\n{}",
                build_tool_call("add", r#"{"x":1}"#)
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "<tool_call> x\n");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("add"));
        assert_eq!(calls(&output)[0].arguments, r#"{"x":1}"#);
    }

    /// JSON whitespace between the arguments object and the wrapper's closing
    /// brace is insignificant, as it is anywhere else inside the wrapper.
    #[test]
    fn qwen_xml_accepts_whitespace_before_wrapper_close() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1} \n}\n</tool_call>",
            )
            .unwrap();

        assert_eq!(output.normal_text(), "");
        assert_eq!(calls(&output).len(), 1);
        assert_eq!(calls(&output)[0].name.as_deref(), Some("add"));
        assert_eq!(calls(&output)[0].arguments, r#"{"x": 1}"#);
    }

    #[test]
    fn qwen_xml_requires_newline_before_tool_call_end() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let error = parser
            .parse_complete(
                r#"<tool_call>
{"name":"get_weather","arguments":{}}</tool_call>"#,
            )
            .unwrap_err();

        assert!(
            error
                .to_report_string()
                .starts_with("tool parser parsing failed:")
        );
    }

    #[test]
    fn qwen_xml_streaming_extracts_multiple_tool_calls() {
        let input = format!(
            "{}{}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let chunks = split_by_chars(&input, 7);
        let mut parser = Qwen3XmlToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(
            output.items,
            [
                ToolParserItem::Call(ToolCallDelta {
                    tool_index: 0,
                    name: Some("get_weather".to_string()),
                    arguments: r#"{"location":"Shanghai"}"#.to_string(),
                }),
                ToolParserItem::Call(ToolCallDelta {
                    tool_index: 1,
                    name: Some("add".to_string()),
                    arguments: r#"{"x":1,"y":2}"#.to_string(),
                }),
            ]
        );
    }

    #[test]
    fn qwen_xml_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        parser
            .parse_chunk(
                r#"<tool_call>
{"name":"get_weather","arguments":{"location""#,
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Qwen XML tool call"]
            .assert_eq(&error.to_report_string());
    }

    /// An unfinished header has published no call, so finalization returns
    /// it as plain text together with its start delimiter.
    #[test]
    fn qwen_xml_finish_returns_incomplete_header_as_text() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let input = "Checking.<tool_call>\n{\"name\": \"get_wea";

        let streamed = parser.parse_chunk(input).unwrap();
        let finished = parser.finish().unwrap();

        assert_eq!(
            streamed.normal_text() + finished.normal_text().as_str(),
            input
        );
        assert!(calls(&streamed).is_empty());
        assert!(calls(&finished).is_empty());
    }

    /// A header that fails to parse publishes no call, and `reset` returns
    /// the whole attempted call, start delimiter included, even when it
    /// arrived over several chunks.
    #[test]
    fn qwen_xml_reset_after_header_error_returns_whole_call() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        parser
            .parse_into("Hi <tool_call>\n{\"na", &mut output)
            .unwrap();
        let result = parser.parse_into("me\": \"add\", \"args\": {}}\n</tool_call>", &mut output);

        assert!(result.is_err());
        assert_eq!(output.normal_text(), "Hi ");
        assert!(calls(&output).is_empty());
        assert_eq!(
            parser.reset(),
            "<tool_call>\n{\"name\": \"add\", \"args\": {}}\n</tool_call>"
        );
    }

    /// Arguments that are not a JSON object fail the header, so no call is
    /// published with unusable arguments and `reset` returns the whole call.
    #[test]
    fn qwen_xml_rejects_non_object_arguments_before_publishing() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        let input = "<tool_call>\n{\"name\": \"add\", \"arguments\": \"{}\"}\n</tool_call>";

        let result = parser.parse_into(input, &mut output);

        assert!(result.is_err());
        assert!(calls(&output).is_empty());
        assert_eq!(parser.reset(), input);
    }

    #[test]
    fn qwen_xml_malformed_field_order_fails_fast() {
        let mut parser = Qwen3XmlToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(
                r#"<tool_call>
{"arguments":{},"name":"get_weather"}
</tool_call>"#,
            )
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Qwen XML
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }
}
