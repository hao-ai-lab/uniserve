use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::ToolParserTestExt as _;

struct DefaultParser;

impl ToolParser for DefaultParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    fn parse_into(&mut self, _chunk: &str, _output: &mut ToolParserOutput) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        Ok(ToolParserOutput::default())
    }

    fn reset(&mut self) -> String {
        String::new()
    }
}

#[test]
fn tool_parser_does_not_preserve_special_tokens_by_default() {
    let parser = DefaultParser;

    assert!(!parser.preserve_special_tokens());
}

#[test]
fn default_parse_complete_delegates_through_parse_chunk_and_finish() {
    struct StreamingParser;

    impl ToolParser for StreamingParser {
        fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            output.normal_text.push_str("prefix ");
            output.calls.extend([
                ToolCallDelta {
                    tool_index: 0,
                    name: Some("weather".to_string()),
                    arguments: "{\"location\":".to_string(),
                },
                ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: "\"Paris\"".to_string(),
                },
                ToolCallDelta {
                    tool_index: 1,
                    name: Some("time".to_string()),
                    arguments: "{\"timezone\":".to_string(),
                },
            ]);
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(ToolParserOutput {
                normal_text: "suffix".to_string(),
                calls: vec![
                    ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "}".to_string(),
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: None,
                        arguments: "\"UTC\"}".to_string(),
                    },
                ],
            })
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    let mut parser = StreamingParser;
    let output = parser.parse_complete("ignored").unwrap();
    assert_eq!(output.normal_text, "prefix suffix");
    assert_eq!(
        output.calls,
        vec![
            ToolCallDelta {
                tool_index: 0,
                name: Some("weather".to_string()),
                arguments: "{\"location\":\"Paris\"}".to_string(),
            },
            ToolCallDelta {
                tool_index: 1,
                name: Some("time".to_string()),
                arguments: "{\"timezone\":\"UTC\"}".to_string(),
            },
        ]
    );
}

/// Cross-parser streaming-invariant tests.
///
/// These exercise behaviors that must hold uniformly across the concrete
/// `ToolParser` implementations, through the public API only:
///   * incremental streaming at any chunk boundary == parsing whole,
///   * an end-of-call/section sentinel that appears literally inside a JSON
///     string value (even split across chunks) does not terminate early,
///   * truncated-mid-call output errors as incomplete on `finish()` and
///     `reset()` hands back the exact uncommitted buffer,
///   * an EOS mid-parameter surfaces as incomplete and never leaks bytes into
///     a committed tool call.
mod streaming_invariants {
    use crate::test_utils::{split_by_chars, test_tools};
    use crate::{
        DeepSeekV3ToolParser, HermesToolParser, KimiK2ToolParser, Qwen3CoderToolParser, Tool,
        ToolParser, ToolParserOutput, ToolParserTestExt as _,
    };

    /// A `(name, arguments)` projection of one coalesced tool call, used so
    /// assertions compare stable identity fields rather than full Debug dumps.
    type CallShape = (Option<String>, String);

    /// One representative parser plus a model output it should parse.
    struct ParserCase {
        /// Human-readable parser label, surfaced in assertion messages.
        label: &'static str,
        /// Builds a fresh boxed parser for this family.
        build: fn(&[Tool]) -> Box<dyn ToolParser>,
        /// A complete model output containing one or more tool calls.
        whole: String,
        /// Expected plain text outside any tool call.
        expected_normal_text: String,
        /// Expected coalesced `(name, arguments)` for each call, in order.
        expected_calls: Vec<CallShape>,
    }

    fn hermes(tools: &[Tool]) -> Box<dyn ToolParser> {
        HermesToolParser::create(tools).unwrap()
    }

    fn kimi_k2(tools: &[Tool]) -> Box<dyn ToolParser> {
        KimiK2ToolParser::create(tools).unwrap()
    }

    fn deepseek_v3(tools: &[Tool]) -> Box<dyn ToolParser> {
        DeepSeekV3ToolParser::create(tools).unwrap()
    }

    fn qwen_coder(tools: &[Tool]) -> Box<dyn ToolParser> {
        Qwen3CoderToolParser::create(tools).unwrap()
    }

    fn call(name: &str, arguments: &str) -> CallShape {
        (Some(name.to_string()), arguments.to_string())
    }

    fn shape(output: &ToolParserOutput) -> Vec<CallShape> {
        output
            .calls
            .iter()
            .map(|delta| (delta.name.clone(), delta.arguments.clone()))
            .collect()
    }

    /// Representative parsers that carry OpenAI-style JSON argument text
    /// verbatim. These three differ in their wrapping markers but share the
    /// raw-JSON-arguments contract, so a single expected-arguments string lines
    /// up across all of them.
    fn json_argument_cases() -> Vec<ParserCase> {
        vec![
            ParserCase {
                label: "hermes",
                build: hermes,
                whole: concat!(
                    "before <tool_call>",
                    r#"{"name":"get_weather","arguments":{"location":"Tokyo","days":"3"}}"#,
                    "</tool_call> after"
                )
                .to_string(),
                expected_normal_text: "before  after".to_string(),
                expected_calls: vec![call(
                    "get_weather",
                    r#"{"location":"Tokyo","days":"3"}"#,
                )],
            },
            ParserCase {
                label: "kimi_k2",
                build: kimi_k2,
                // Kimi K2 discards any text after the tool-calls section ends,
                // so the trailing " after" is intentionally absent from the
                // expected normal text.
                whole: concat!(
                    "before <|tool_calls_section_begin|>",
                    "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
                    r#"{"location":"Tokyo","days":"3"}"#,
                    "<|tool_call_end|><|tool_calls_section_end|> after"
                )
                .to_string(),
                expected_normal_text: "before ".to_string(),
                expected_calls: vec![call(
                    "get_weather",
                    r#"{"location":"Tokyo","days":"3"}"#,
                )],
            },
            ParserCase {
                label: "deepseek_v3",
                build: deepseek_v3,
                // DeepSeek V3 likewise discards text after the tool-calls
                // section ends.
                whole: concat!(
                    "before <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather",
                    "\n```json\n",
                    r#"{"location":"Tokyo","days":"3"}"#,
                    "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜> after"
                )
                .to_string(),
                expected_normal_text: "before ".to_string(),
                expected_calls: vec![call(
                    "get_weather",
                    r#"{"location":"Tokyo","days":"3"}"#,
                )],
            },
        ]
    }

    /// Every representative family, including the XML/atomic-block Qwen Coder
    /// parser whose arguments are schema-converted JSON rather than verbatim.
    fn all_cases() -> Vec<ParserCase> {
        let mut cases = json_argument_cases();
        cases.push(ParserCase {
            label: "qwen_coder",
            build: qwen_coder,
            whole: concat!(
                "before ",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=location>Tokyo</parameter>\n",
                "</function>\n</tool_call>",
                " after"
            )
            .to_string(),
            expected_normal_text: "before  after".to_string(),
            expected_calls: vec![call("get_weather", r#"{"location":"Tokyo"}"#)],
        });
        cases
    }

    /// Feed `chunks` through `parser`, requiring every step to succeed, and
    /// return the coalesced output.
    fn stream(parser: &mut dyn ToolParser, chunks: &[&str]) -> ToolParserOutput {
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output.coalesce_calls()
    }

    /// All 2-way partitions of `text` at UTF-8 char boundaries, as `(head, tail)`.
    ///
    /// Combined with the per-char and pseudo-random partitions below, this
    /// exercises "any chunk boundary" without depending on a single split.
    fn binary_partitions(text: &str) -> Vec<(&str, &str)> {
        let mut splits = Vec::new();
        for (index, _) in text.char_indices() {
            splits.push((&text[..index], &text[index..]));
        }
        splits.push((text, ""));
        splits
    }

    /// A deterministic pseudo-random multi-way partition of `text` at char
    /// boundaries, seeded by `seed` (a tiny xorshift so the test never touches
    /// wall-clock or a real RNG and is perfectly reproducible).
    fn seeded_partition(text: &str, mut seed: u64) -> Vec<&str> {
        let boundaries: Vec<usize> = text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
            .collect();

        let mut chunks = Vec::new();
        let mut start = 0usize; // index into `boundaries`
        while start + 1 < boundaries.len() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let remaining = boundaries.len() - 1 - start;
            let step = 1 + (seed as usize % remaining.max(1));
            let end = (start + step).min(boundaries.len() - 1);
            chunks.push(&text[boundaries[start]..boundaries[end]]);
            start = end;
        }
        chunks
    }

    #[test]
    fn streaming_at_any_chunk_boundary_matches_whole_parse() {
        let tools = test_tools();
        for case in all_cases() {
            let whole_output = {
                let mut parser = (case.build)(&tools);
                parser.parse_complete(&case.whole).unwrap()
            };

            // Sanity: the fixture must actually exercise a tool call.
            assert_eq!(
                shape(&whole_output),
                case.expected_calls,
                "{}: whole-parse calls mismatch",
                case.label
            );
            assert_eq!(
                whole_output.normal_text, case.expected_normal_text,
                "{}: whole-parse normal_text mismatch",
                case.label
            );

            // Every 2-way char-boundary partition reproduces the whole parse.
            for (head, tail) in binary_partitions(&case.whole) {
                let mut parser = (case.build)(&tools);
                let streamed = stream(parser.as_mut(), &[head, tail]);
                assert_eq!(
                    streamed.normal_text, whole_output.normal_text,
                    "{}: normal_text diverged for split at {:?}|{:?}",
                    case.label, head, tail
                );
                assert_eq!(
                    shape(&streamed),
                    shape(&whole_output),
                    "{}: calls diverged for split at {:?}|{:?}",
                    case.label, head, tail
                );
            }

            // One-char-per-chunk partition (the finest possible boundary).
            {
                let mut parser = (case.build)(&tools);
                let chunks = split_by_chars(&case.whole, 1);
                let streamed = stream(parser.as_mut(), &chunks);
                assert_eq!(
                    streamed.normal_text, whole_output.normal_text,
                    "{}: normal_text diverged for per-char split",
                    case.label
                );
                assert_eq!(
                    shape(&streamed),
                    shape(&whole_output),
                    "{}: calls diverged for per-char split",
                    case.label
                );
            }

            // A handful of seeded pseudo-random multi-way partitions.
            for seed in [0x1234_5678u64, 0xDEAD_BEEF, 0x0F0F_0F0F, 1, 999_983] {
                let mut parser = (case.build)(&tools);
                let chunks = seeded_partition(&case.whole, seed);
                let streamed = stream(parser.as_mut(), &chunks);
                assert_eq!(
                    streamed.normal_text, whole_output.normal_text,
                    "{}: normal_text diverged for seeded split {seed:#x}",
                    case.label
                );
                assert_eq!(
                    shape(&streamed),
                    shape(&whole_output),
                    "{}: calls diverged for seeded split {seed:#x}",
                    case.label
                );
            }
        }
    }

    #[test]
    fn end_of_call_sentinel_inside_json_string_does_not_terminate_early() {
        let tools = test_tools();
        // Each JSON-argument parser is fed its own end-of-call/section sentinel
        // verbatim inside a JSON string value. The sentinel must be preserved
        // in the arguments and must not close the call.
        let sentinels = [
            ("hermes", "</tool_call>"),
            ("kimi_k2", "<|tool_call_end|>"),
            ("deepseek_v3", "\n```<｜tool▁call▁end｜>"),
        ];

        for case in json_argument_cases() {
            let sentinel = sentinels
                .iter()
                .find(|(label, _)| *label == case.label)
                .map(|(_, sentinel)| *sentinel)
                .unwrap();

            let arguments = format!(r#"{{"text":"literal {sentinel} inside"}}"#);
            let whole = case.whole.replacen(
                r#"{"location":"Tokyo","days":"3"}"#,
                &arguments,
                1,
            );

            // Whole parse keeps the sentinel inside the single call's arguments.
            let mut parser = (case.build)(&tools);
            let output = parser.parse_complete(&whole).unwrap();
            assert_eq!(
                shape(&output),
                vec![call("get_weather", &arguments)],
                "{}: sentinel-in-string whole parse",
                case.label
            );

            // The same holds when the embedded sentinel is split across chunks
            // at every char boundary inside the arguments object.
            for (head, tail) in binary_partitions(&whole) {
                let mut parser = (case.build)(&tools);
                let streamed = stream(parser.as_mut(), &[head, tail]);
                assert_eq!(
                    shape(&streamed),
                    vec![call("get_weather", &arguments)],
                    "{}: sentinel-in-string diverged for split {:?}|{:?}",
                    case.label, head, tail
                );
            }
        }
    }

    #[test]
    fn truncated_mid_call_finish_errors_incomplete_for_every_parser() {
        let tools = test_tools();
        // A prefix that opens a call/header but stops before the call closes.
        // Each family must surface the unterminated call as an "incomplete"
        // error from finish(), rather than fabricating a closed call.
        let truncations = [
            (
                "hermes",
                hermes as fn(&[Tool]) -> Box<dyn ToolParser>,
                r#"<tool_call>{"name":"get_weather","arguments":{"location""#,
            ),
            (
                "kimi_k2",
                kimi_k2 as fn(&[Tool]) -> Box<dyn ToolParser>,
                concat!(
                    "<|tool_calls_section_begin|><|tool_call_begin|>",
                    "functions.get_weather:0<|tool_call_argument_begin|>",
                    r#"{"location""#
                ),
            ),
            (
                "deepseek_v3",
                deepseek_v3 as fn(&[Tool]) -> Box<dyn ToolParser>,
                concat!(
                    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather",
                    "\n```json\n",
                    r#"{"location""#
                ),
            ),
            (
                "qwen_coder",
                qwen_coder as fn(&[Tool]) -> Box<dyn ToolParser>,
                "<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>",
            ),
        ];

        for (label, build, prefix) in truncations {
            let mut parser = build(&tools);
            // Streaming the prefix succeeds (raw-JSON families may stream a name
            // and partial argument fragments here), but no committed delta may
            // ever carry a closed arguments object: the call is not finalized.
            let progressive = parser.parse_chunk(prefix).unwrap();
            let coalesced = progressive.coalesce_calls();
            for delta in &coalesced.calls {
                assert!(
                    !is_closed_json_object(&delta.arguments),
                    "{label}: a truncated call must not yield a closed arguments object, \
                     got {:?}",
                    delta.arguments
                );
            }

            // finish() reports the call as incomplete rather than fabricating a
            // partial/garbled finalized call.
            let error = parser.finish().unwrap_err();
            assert!(
                error.to_string().contains("incomplete"),
                "{label}: expected incomplete error, got: {error}"
            );
        }
    }

    #[test]
    fn atomic_parser_emits_no_partial_call_before_finish() {
        // The Qwen Coder parser buffers a whole `<tool_call>` block before
        // emitting anything, so a truncated block produces no committed delta
        // at all (no garbled partial call leaks through streaming).
        let tools = test_tools();
        let mut parser = qwen_coder(&tools);
        let output = parser
            .parse_chunk("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        assert!(output.normal_text.is_empty());
        assert!(output.calls.is_empty());

        let error = parser.finish().unwrap_err();
        assert!(
            error.to_string().contains("incomplete"),
            "expected incomplete error, got: {error}"
        );
    }

    /// Returns whether `arguments` is a single balanced top-level JSON object,
    /// i.e. a finalized (closed) arguments payload. Used to detect whether a
    /// truncated call leaked a complete object. This walks string/brace state
    /// itself so it does not depend on serde accepting the (possibly malformed)
    /// argument text.
    fn is_closed_json_object(arguments: &str) -> bool {
        let bytes = arguments.as_bytes();
        if bytes.first() != Some(&b'{') {
            return false;
        }
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escape = false;
        let mut closed_at = None;
        for (index, &byte) in bytes.iter().enumerate() {
            if in_string {
                if escape {
                    escape = false;
                } else if byte == b'\\' {
                    escape = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        closed_at = Some(index);
                        break;
                    }
                }
                _ => {}
            }
        }
        // Closed only if the matching brace is the final byte.
        closed_at == Some(bytes.len() - 1)
    }

    #[test]
    fn reset_after_truncated_atomic_call_recovers_exact_uncommitted_buffer() {
        // The Qwen Coder parser buffers an entire `<tool_call>` block until it
        // closes, so a truncated block leaves real uncommitted bytes. The
        // already-consumed start marker is dropped from the buffer, so reset()
        // returns exactly the un-consumed tail of the fed input.
        let tools = test_tools();
        let prefix = "<tool_call>\n<function=get_weather>\n<parameter=location>SF";

        let mut parser = qwen_coder(&tools);
        let committed = parser.parse_chunk(prefix).unwrap();
        assert!(
            committed.calls.is_empty(),
            "atomic parser should not commit a partial call"
        );

        // reset() hands back the exact uncommitted buffer: the input minus the
        // consumed `<tool_call>` start marker, and a contiguous suffix of it.
        let buffered = parser.reset();
        assert_eq!(
            buffered, "\n<function=get_weather>\n<parameter=location>SF",
            "reset() must return the exact uncommitted buffer"
        );
        assert!(
            prefix.ends_with(&buffered),
            "reset() buffer {buffered:?} must be a suffix of {prefix:?}"
        );

        // After reset() the parser is empty: finishing now yields no output.
        let after_reset = parser.finish().unwrap();
        assert_eq!(after_reset, ToolParserOutput::default());
    }

    #[test]
    fn reset_returns_held_back_partial_marker_as_uncommitted_buffer() {
        // In text mode the JSON-argument parsers hold back a trailing fragment
        // that might still grow into a start marker, committing only the safe
        // prefix as normal text. reset() then returns exactly that held-back
        // marker fragment (a suffix of the input), with nothing lost.
        let tools = test_tools();
        let cases = [
            (
                "hermes",
                hermes as fn(&[Tool]) -> Box<dyn ToolParser>,
                "plain text <tool",
                "plain text ",
                "<tool",
            ),
            (
                "kimi_k2",
                kimi_k2 as fn(&[Tool]) -> Box<dyn ToolParser>,
                "plain text <|tool_calls",
                "plain text ",
                "<|tool_calls",
            ),
        ];

        for (label, build, fed, expected_committed_text, expected_buffer) in cases {
            let mut parser = build(&tools);
            let output = parser.parse_chunk(fed).unwrap();

            assert_eq!(
                output.normal_text, expected_committed_text,
                "{label}: only the safe prefix should be committed as text"
            );
            assert!(
                output.calls.is_empty(),
                "{label}: no call should be committed yet"
            );

            let buffered = parser.reset();
            assert_eq!(
                buffered, expected_buffer,
                "{label}: reset() must return the held-back marker fragment"
            );
            assert!(
                fed.ends_with(&buffered),
                "{label}: reset() buffer must be a suffix of the fed input"
            );
        }
    }

    #[test]
    fn eos_mid_parameter_is_incomplete_and_never_seals_a_call() {
        let tools = test_tools();
        // The stream ends (EOS) partway through a parameter value. Each parser
        // must report the call as incomplete from finish() and must never seal
        // the half-written parameter into a finalized/closed call. Streaming
        // families may surface the partial bytes as in-flight argument deltas,
        // but those deltas must never form a closed arguments object, and
        // finish() must fail so no consumer treats the call as finalized.
        let leak_probe = "PARTIAL_VALUE_DO_NOT_SEAL";
        let truncations = [
            (
                "hermes",
                hermes as fn(&[Tool]) -> Box<dyn ToolParser>,
                format!(
                    r#"<tool_call>{{"name":"get_weather","arguments":{{"location":"{leak_probe}"#
                ),
            ),
            (
                "kimi_k2",
                kimi_k2 as fn(&[Tool]) -> Box<dyn ToolParser>,
                format!(
                    concat!(
                        "<|tool_calls_section_begin|><|tool_call_begin|>",
                        "functions.get_weather:0<|tool_call_argument_begin|>",
                        r#"{{"location":"{}"#
                    ),
                    leak_probe
                ),
            ),
            (
                "deepseek_v3",
                deepseek_v3 as fn(&[Tool]) -> Box<dyn ToolParser>,
                format!(
                    concat!(
                        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather",
                        "\n```json\n",
                        r#"{{"location":"{}"#
                    ),
                    leak_probe
                ),
            ),
            (
                "qwen_coder",
                qwen_coder as fn(&[Tool]) -> Box<dyn ToolParser>,
                format!(
                    "<tool_call>\n<function=get_weather>\n<parameter=location>{leak_probe}"
                ),
            ),
        ];

        for (label, build, prefix) in truncations {
            let mut parser = build(&tools);
            let progressive = parser.parse_chunk(&prefix).unwrap().coalesce_calls();

            // The half-written parameter is never sealed into a closed call.
            for delta in &progressive.calls {
                assert!(
                    !is_closed_json_object(&delta.arguments),
                    "{label}: a mid-parameter EOS must not seal a closed arguments \
                     object, got {:?}",
                    delta.arguments
                );
            }

            // finish() reports incomplete, so no consumer finalizes the call.
            let error = parser.finish().unwrap_err();
            assert!(
                error.to_string().contains("incomplete"),
                "{label}: EOS mid-parameter should be incomplete, got: {error}"
            );
        }
    }

    #[test]
    fn eos_mid_parameter_keeps_atomic_parser_value_unemitted_but_recoverable() {
        // For the atomic Qwen Coder parser, a mid-parameter EOS emits no call
        // delta at all, yet the half-written value stays recoverable through
        // reset() so a caller can fall back to surfacing it as normal text.
        let tools = test_tools();
        let leak_probe = "PARTIAL_VALUE_DO_NOT_SEAL";
        let prefix =
            format!("<tool_call>\n<function=get_weather>\n<parameter=location>{leak_probe}");

        let mut parser = qwen_coder(&tools);
        let progressive = parser.parse_chunk(&prefix).unwrap();
        assert!(
            progressive.calls.is_empty(),
            "atomic parser must not emit the partial parameter as a delta"
        );

        assert!(parser.finish().is_err());

        let buffered = parser.reset();
        assert!(
            buffered.contains(leak_probe),
            "the half-written parameter should remain recoverable via reset(), got {buffered:?}"
        );
    }
}
