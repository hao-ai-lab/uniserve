#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hf_hub::api::sync::ApiBuilder;
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;

const MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
const SAMPLE_TEXT: &str =
    "<|im_start|>user\nSummarize this request.\n<|im_end|>\n<|im_start|>assistant\n";

struct BenchFixture {
    tokenizer: HuggingFaceTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    fn load() -> Self {
        let path = ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .expect("build hf-hub api")
            .model(MODEL_ID.to_string())
            .get("tokenizer.json")
            .expect("fetch tokenizer.json from hf-hub");
        let tokenizer = HuggingFaceTokenizer::new(&path).expect("load configured tokenizer");
        let text = SAMPLE_TEXT.repeat(32);
        let token_ids = tokenizer.encode(&text, false).expect("encode sample text");
        Self {
            tokenizer,
            text,
            token_ids,
        }
    }
}

fn bench_encode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tokenizer_encode");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));
    group.bench_function("configured", |b| {
        b.iter(|| {
            fixture
                .tokenizer
                .encode(black_box(&fixture.text), black_box(false))
                .expect("encode sample text")
        })
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tokenizer_decode");
    group.throughput(Throughput::Elements(fixture.token_ids.len() as u64));
    group.bench_function("configured", |b| {
        b.iter(|| {
            fixture
                .tokenizer
                .decode(black_box(&fixture.token_ids), black_box(false))
                .expect("decode sample tokens")
        })
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
