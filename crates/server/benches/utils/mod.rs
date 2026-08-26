#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(dead_code)]

use uniserve_server::profile::tools::Qwen3XmlToolParser;
use uniserve_server::profile::tools::test_utils::collect_stream;

pub(super) fn feed_parser(parser: &mut Qwen3XmlToolParser, chunks: &[&str]) -> (String, usize) {
    let result = collect_stream(parser, chunks);
    (result.normal_text, result.calls.len())
}
