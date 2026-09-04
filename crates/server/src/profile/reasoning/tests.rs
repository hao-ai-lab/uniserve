//! Incremental reasoning-parser behavior across delimiter boundaries.

use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

use super::{DelimitedReasoningParser, Qwen3ReasoningParser};

fn reasoning_tokenizer() -> DynTokenizer {
    let model = BPE::builder()
        .vocab_and_merges(
            [
                ("<unk>".to_string(), 0),
                ("a".to_string(), 1),
                ("b".to_string(), 2),
            ],
            Vec::new(),
        )
        .unk_token("<unk>".to_string())
        .build()
        .expect("build tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
        AddedToken::from("<|im_end|>", true),
    ]);
    let directory = tempdir().expect("create tokenizer directory");
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).expect("save tokenizer");
    Arc::new(HuggingFaceTokenizer::new(&path).expect("load configured tokenizer"))
}

fn token_id(tokenizer: &DynTokenizer, token: &str) -> u32 {
    tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("tokenizer must contain {token}"))
}

#[test]
fn delimited_parser_splits_reasoning_across_chunk_boundaries() {
    let tokenizer = reasoning_tokenizer();
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert!(parser.push("<thi").is_empty());
    let delta = parser.push("nk>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_parser_flushes_partial_end_marker() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[start]);

    let delta = parser.push("unfinished</thi");
    assert_eq!(delta.reasoning.as_deref(), Some("unfinished"));
    assert_eq!(parser.finish().reasoning.as_deref(), Some("</thi"));
}

#[test]
fn qwen3_without_prompt_boundary_waits_for_reasoning_start() {
    let tokenizer = reasoning_tokenizer();
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn qwen3_prompt_boundaries_select_the_initial_output_region() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let end = token_id(&tokenizer, "</think>");

    let mut content = Qwen3ReasoningParser::new(Arc::clone(&tokenizer)).unwrap();
    content.initialize(&[end]).unwrap();
    assert_eq!(
        content.push("answer").unwrap().content.as_deref(),
        Some("answer")
    );

    let mut reasoning = Qwen3ReasoningParser::new(tokenizer).unwrap();
    reasoning.initialize(&[start]).unwrap();
    let delta = reasoning.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn qwen3_prompt_scan_stops_at_the_last_special_token() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let prompt_end = token_id(&tokenizer, "<|im_end|>");
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[start, prompt_end]).unwrap();
    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}
