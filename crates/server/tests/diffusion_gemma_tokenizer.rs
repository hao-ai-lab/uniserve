#![allow(clippy::expect_used, clippy::unwrap_used)]
//! The DiffusionGemma (Gemma-4) tokenizer encodes and decodes as Hugging Face does.
//!
//! Gemma-4 tokenizes SentencePiece-style: a `Replace` normalizer turns spaces
//! into `▁`, BPE falls back to byte tokens (`<0xEA>`), and decoding reverses
//! both. The fixture `tests/python/fixtures/diffusion_gemma_tokenizer.json`
//! holds the Hugging Face ids and decoded texts for plain, multilingual,
//! byte-fallback, special-token, and chat-template strings.
//!
//! The test reads the checkpoint directory named by
//! `UNISERVE_DIFFUSION_GEMMA_MODEL` and runs only on request:
//! `cargo test -p uniserve-server --test diffusion_gemma_tokenizer -- --include-ignored`.

use std::path::Path;

use serde::Deserialize;
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    token_ids: Vec<u32>,
    decoded: String,
    decoded_without_special: String,
}

#[test]
#[ignore = "requires the DiffusionGemma checkpoint named by UNISERVE_DIFFUSION_GEMMA_MODEL"]
fn gemma_tokenization_round_trips_as_hugging_face() {
    let directory = std::env::var("UNISERVE_DIFFUSION_GEMMA_MODEL")
        .expect("UNISERVE_DIFFUSION_GEMMA_MODEL must name the checkpoint directory");
    let tokenizer =
        HuggingFaceTokenizer::new(&Path::new(&directory).join("tokenizer.json")).unwrap();
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/python/fixtures/diffusion_gemma_tokenizer.json"
    ))
    .unwrap();

    for case in fixture.cases {
        let text = case.text.as_str();
        assert_eq!(
            tokenizer.encode(text, false).unwrap(),
            case.token_ids,
            "{text:?}"
        );
        assert_eq!(
            tokenizer.decode(&case.token_ids, false).unwrap(),
            case.decoded,
            "{text:?}"
        );
        assert_eq!(
            tokenizer.decode(&case.token_ids, true).unwrap(),
            case.decoded_without_special,
            "{text:?}"
        );
    }
}
