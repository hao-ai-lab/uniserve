#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(dead_code)]

use uniserve_tool_parser::ToolParser;
use uniserve_tool_parser::test_utils::collect_stream;

pub(super) fn feed_parser(parser: &mut dyn ToolParser, chunks: &[&str]) -> (String, usize) {
    let result = collect_stream(parser, chunks);
    (result.normal_text, result.calls.len())
}
