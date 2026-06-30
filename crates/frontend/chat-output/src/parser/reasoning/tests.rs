use std::sync::Arc;

use uniserve_tokenizer::Tokenizer;

use super::{ReasoningParserFactory, names};
use crate::Error;

struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> uniserve_tokenizer::Result<Vec<u32>> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> uniserve_tokenizer::Result<String> {
        Ok(token_ids
            .iter()
            .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
            .collect())
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

#[test]
fn factory_contains_and_lists_registered_parsers() {
    let factory = ReasoningParserFactory::new();
    assert!(factory.contains(names::QWEN3));
    assert!(factory.contains(names::DEEPSEEK_V4));
    assert!(factory.list().contains(&names::QWEN3.to_string()));
    assert!(factory.list().contains(&names::DEEPSEEK_V4.to_string()));
}

#[test]
fn factory_resolves_deepseek_v4_to_qwen3_alias() {
    let factory = ReasoningParserFactory::new();
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V4"),
        Some(names::DEEPSEEK_V4)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek_v4"),
        Some(names::DEEPSEEK_V4)
    );
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let tokenizer = Arc::new(FakeTokenizer);
    let factory = ReasoningParserFactory::new();
    let error = match factory.create("missing", tokenizer) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("choose from"));
}

#[test]
fn unknown_reasoning_parser_error_names_rejected_parser_and_lists_available() {
 // Mirrors the tool-side contract: rejecting an unknown explicit reasoning
 // parser name yields a ParserUnavailableByName error tagged with the
 // "reasoning" kind, carrying the rejected name and the available parser
 // names. We assert the variant plus a couple of stable built-in names
 // instead of snapshotting the whole registry list.
    let tokenizer = Arc::new(FakeTokenizer);
    let factory = ReasoningParserFactory::new();
    let error = match factory.create("nope-not-a-parser", tokenizer) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };

    let Error::ParserUnavailableByName {
        kind,
        name,
        available_names,
    } = &error
    else {
        panic!("expected ParserUnavailableByName, got {error:?}");
    };
    assert_eq!(*kind, "reasoning");
    assert_eq!(name, "nope-not-a-parser");
    assert!(
        available_names.iter().any(|n| n == names::QWEN3),
        "available_names should include the qwen3 reasoning parser: {available_names:?}",
    );
    assert!(
        available_names.iter().any(|n| n == names::DEEPSEEK_R1),
        "available_names should include the deepseek_r1 reasoning parser: {available_names:?}",
    );

    let rendered = error.to_string();
    assert!(
        rendered.contains("nope-not-a-parser"),
        "error message should name the rejected parser: {rendered}",
    );
    assert!(
        rendered.contains("choose from"),
        "error message should list available parsers: {rendered}",
    );
    assert!(
        rendered.contains(names::QWEN3),
        "error message should mention a stable available parser: {rendered}",
    );
}
