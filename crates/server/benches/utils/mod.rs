#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(dead_code)]
//! Shared benchmark drivers for streaming output parsers.

use uniserve_server::profile::tools::Qwen3XmlToolParser;
use uniserve_server::profile::tools::test_utils::collect_stream;

/// Feeds chunks into a parser and returns visible text plus parsed-call count.
///
/// The stream is finished after the last chunk, which resets `parser` for the
/// next stream. A parser error panics inside `collect_stream`.
pub(super) fn feed_parser(parser: &mut Qwen3XmlToolParser, chunks: &[&str]) -> (String, usize) {
    let result = collect_stream(parser, chunks);
    (result.normal_text(), result.calls().count())
}
