#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use uniserve_model_profile::tools::test_utils::{split_by_chars, test_tools};
use uniserve_model_profile::tools::{KimiK2ToolParser, Tool, ToolParser};

mod utils;
use utils::feed_parser;

const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>{\"location\":\"Hangzhou\",\"days\":3}",
        "<|tool_call_end|>",
        "<|tool_call_begin|>functions.get_weather:1",
        "<|tool_call_argument_begin|>{\"location\":\"San Francisco\",\"days\":2}",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
    )
    .to_string()
}

fn mixed_chunks() -> Vec<&'static str> {
    vec![
        "I will check two cities before answering.\n",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>",
        "{\"location\":",
        "\"Hangzhou\",",
        "\"days\":3}",
        "<|tool_call_end|>",
        "<|tool_call_begin|>functions.get_weather:1",
        "<|tool_call_argument_begin|>",
        "{\"location\":",
        "\"San Francisco\",",
        "\"days\":2}",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
    ]
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Kimi K2 tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    KimiK2ToolParser::create(tools).expect("Kimi K2 parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[Tool],
    text: &str,
    chunks: &[&str],
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("native_reuse_parser", |b| {
        let mut parser = native_parser(tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            || native_parser(tools),
            |mut parser| {
                let result = feed_parser(&mut *parser, black_box(chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_native_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_kimi_k2(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let mixed_chunks = mixed_chunks();
    let long_normal_text = long_normal_text_fixture();
    let long_normal_chunks = split_by_chars(&long_normal_text, CHUNK_CHARS);

    run_stream_group(
        c,
        "kimi_k2/mixed_text_tool_call",
        &tools,
        &mixed_text,
        &mixed_chunks,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "kimi_k2/long_normal_text",
        &tools,
        &long_normal_text,
        &long_normal_chunks,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_kimi_k2);
criterion_main!(benches);
