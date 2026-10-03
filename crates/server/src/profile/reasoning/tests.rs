//! Incremental reasoning-parser behavior across delimiter boundaries.

use std::sync::Arc;

use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};

use super::{
    DelimitedReasoningParser, Gemma4ReasoningParser, Qwen3ReasoningParser, ReasoningParser,
};

/// Builds a tokenizer whose only special tokens are the reasoning delimiters
/// and `<|im_end|>`, which stands in for a chat-turn boundary.
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

    // A partial start delimiter is held back until the next delta completes it.
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

    // At end of stream the incomplete delimiter is text of the current region.
    assert_eq!(parser.finish().reasoning.as_deref(), Some("</thi"));
}

#[test]
fn qwen3_without_prompt_boundary_waits_for_reasoning_start() {
    let tokenizer = reasoning_tokenizer();
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn qwen3_prompt_boundaries_select_the_initial_output_region() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let end = token_id(&tokenizer, "</think>");

    let mut content = Qwen3ReasoningParser::new(Arc::clone(&tokenizer)).unwrap();
    content.initialize(&[end]);
    assert_eq!(content.push("answer").content.as_deref(), Some("answer"));

    let mut reasoning = Qwen3ReasoningParser::new(tokenizer).unwrap();
    reasoning.initialize(&[start]);
    let delta = reasoning.push("reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

/// The backward prompt scan stops at `<|im_end|>` before reaching the earlier
/// `<think>`, so the parser keeps its default content region.
#[test]
fn qwen3_prompt_scan_stops_at_the_last_special_token() {
    let tokenizer = reasoning_tokenizer();
    let start = token_id(&tokenizer, "<think>");
    let prompt_end = token_id(&tokenizer, "<|im_end|>");
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[start, prompt_end]);
    let delta = parser.push("answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

/// Gemma-4 checkpoint token ids used in prompts below: `<|turn>`,
/// `<|channel>`, and `<tool_response|>` are special tokens, while `model`,
/// `thought`, and the newline are ordinary tokens.
const GEMMA4_TURN_START: u32 = 105;
const GEMMA4_CHANNEL_START: u32 = 100;
const GEMMA4_TOOL_RESPONSE_END: u32 = 51;
const GEMMA4_MODEL: u32 = 4368;
const GEMMA4_THOUGHT: u32 = 45518;
const GEMMA4_NEWLINE: u32 = 107;

/// Special tokens of the Gemma-4 checkpoint tokenizer with their ids.
const GEMMA4_SPECIAL_TOKENS: [(u32, &str); 11] = [
    (3, "<unk>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (GEMMA4_TOOL_RESPONSE_END, "<tool_response|>"),
    (52, "<|\"|>"),
    (98, "<|think|>"),
    (GEMMA4_CHANNEL_START, "<|channel>"),
    (101, "<channel|>"),
    (GEMMA4_TURN_START, "<|turn>"),
    (106, "<turn|>"),
];

/// Builds a tokenizer whose ids 0 through 107 cover the checkpoint's
/// special tokens and its newline token at their checkpoint ids; the other
/// ids hold ordinary placeholder tokens. Ids beyond the vocabulary, such as
/// `model` and `thought`, are ordinary tokens to the prompt scan.
fn gemma4_tokenizer() -> DynTokenizer {
    let mut vocab: Vocab = (0..=GEMMA4_NEWLINE)
        .map(|id| (format!("<placeholder_{id}>"), id))
        .collect();
    for (id, token) in GEMMA4_SPECIAL_TOKENS
        .into_iter()
        .chain([(GEMMA4_NEWLINE, "\n")])
    {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    let special_tokens: Vec<AddedToken> = GEMMA4_SPECIAL_TOKENS
        .into_iter()
        .map(|(_, token)| AddedToken::from(token, true))
        .collect();
    tokenizer.add_special_tokens(&special_tokens);
    let directory = tempdir().expect("create tokenizer directory");
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).expect("save tokenizer");
    Arc::new(HuggingFaceTokenizer::new(&path).expect("load configured tokenizer"))
}

/// Pushes every chunk, finishes the stream, and returns the concatenated
/// reasoning and content text.
fn collect_reasoning(parser: &mut impl ReasoningParser, chunks: &[&str]) -> (String, String) {
    let mut deltas: Vec<_> = chunks.iter().map(|chunk| parser.push(chunk)).collect();
    deltas.push(parser.finish());

    let mut reasoning = String::new();
    let mut content = String::new();
    for delta in deltas {
        reasoning.push_str(delta.reasoning.as_deref().unwrap_or_default());
        content.push_str(delta.content.as_deref().unwrap_or_default());
    }
    (reasoning, content)
}

/// A new model turn starts in content; the thought channel the model opens
/// is reasoning without its `<|channel>thought\n` header, however the text is
/// split, including one character at a time and one delta for the whole
/// reply.
#[test]
fn gemma4_splits_thought_channel_from_content() {
    let reply = "<|channel>thought\nStep 1.\nStep 2.<channel|>The answer is 4.";
    let characters: Vec<String> = reply.chars().map(String::from).collect();
    let characters: Vec<&str> = characters.iter().map(String::as_str).collect();

    for chunks in [characters, vec![reply]] {
        let mut parser = Gemma4ReasoningParser::new(gemma4_tokenizer()).unwrap();
        parser.initialize(&[GEMMA4_TURN_START, GEMMA4_MODEL, GEMMA4_NEWLINE]);

        assert_eq!(
            collect_reasoning(&mut parser, &chunks),
            (
                "Step 1.\nStep 2.".to_string(),
                "The answer is 4.".to_string()
            )
        );
    }
}

/// A thought channel whose name lacks its newline still opens reasoning and
/// its name is stripped, however the text is split, so neither the channel
/// markup nor its name reaches the reasoning or the content. A channel name
/// that the stream ends on is stripped too.
#[test]
fn gemma4_strips_a_channel_name_without_its_newline() {
    for (reply, expected) in [
        ("<|channel>thought<channel|>Paris.", ("", "Paris.")),
        (
            "<|channel>thoughtIt is sunny.<channel|>Sunny today.",
            ("It is sunny.", "Sunny today."),
        ),
        ("<|channel>thought", ("", "")),
    ] {
        let characters: Vec<String> = reply.chars().map(String::from).collect();
        let characters: Vec<&str> = characters.iter().map(String::as_str).collect();
        for chunks in [characters, vec![reply]] {
            let mut parser = Gemma4ReasoningParser::new(gemma4_tokenizer()).unwrap();
            parser.initialize(&[GEMMA4_TURN_START, GEMMA4_MODEL, GEMMA4_NEWLINE]);

            assert_eq!(
                collect_reasoning(&mut parser, &chunks),
                (expected.0.to_string(), expected.1.to_string()),
                "{reply:?} in {} chunks",
                chunks.len()
            );
        }
    }
}

/// The empty thought channel that opens a reply without thinking carries no
/// reasoning, and its delimiters do not reach the content.
#[test]
fn gemma4_empty_thought_channel_yields_only_content() {
    let mut parser = Gemma4ReasoningParser::new(gemma4_tokenizer()).unwrap();
    parser.initialize(&[GEMMA4_TURN_START, GEMMA4_MODEL, GEMMA4_NEWLINE]);

    let (reasoning, content) =
        collect_reasoning(&mut parser, &["<|channel>thought\n<channel|>Paris."]);

    assert_eq!(reasoning, "");
    assert_eq!(content, "Paris.");
}

/// A prompt that ends with an open thought channel after a tool response
/// starts generation inside reasoning.
#[test]
fn gemma4_prompt_with_open_thought_channel_starts_in_reasoning() {
    let mut parser = Gemma4ReasoningParser::new(gemma4_tokenizer()).unwrap();
    parser.initialize(&[
        GEMMA4_TOOL_RESPONSE_END,
        GEMMA4_CHANNEL_START,
        GEMMA4_THOUGHT,
        GEMMA4_NEWLINE,
    ]);

    let (reasoning, content) =
        collect_reasoning(&mut parser, &["It is sunny.", "<channel|>", "Sunny today."]);

    assert_eq!(reasoning, "It is sunny.");
    assert_eq!(content, "Sunny today.");
}
