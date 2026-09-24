//! Deterministic tokenizer fixtures shared by serving tests.

use std::sync::Arc;

use crate::profile::tokenizer::{DynTokenizer, HuggingFaceTokenizer};
use tempfile::tempdir;
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

/// Builds a deterministic byte-oriented tokenizer for serving tests.
///
/// Every printable ASCII character encodes to the token whose ID equals its
/// code point, so tests write IDs as `b'a' as u32`. ID 0 is the `<unk>`
/// token, and IDs 1 through 6 (control characters) are replaced by the chat,
/// vision, and thinking special tokens, which encode atomically. The BPE model
/// has no merges, so encoding is one token per character.
pub(crate) fn configured_tokenizer() -> DynTokenizer {
    let mut vocab = (0_u32..=127)
        .map(|id| {
            let token = if id == 0 {
                "<unk>".to_string()
            } else {
                char::from_u32(id).expect("ASCII code point").to_string()
            };
            (token, id)
        })
        .collect::<Vocab>();
    for (id, token) in [
        (1, "<|im_start|>"),
        (2, "<|im_end|>"),
        (3, "<|vision_start|>"),
        (4, "<|vision_end|>"),
        (5, "<think>"),
        (6, "</think>"),
        (97, "a"),
        (98, "b"),
    ] {
        vocab.retain(|_, existing_id| *existing_id != id);
        vocab.insert(token.to_string(), id);
    }
    let model = BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .unk_token("<unk>".to_string())
        .build()
        .expect("build configured tokenizer model");
    let mut tokenizer = TokenizerBuilder::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|vision_end|>", true),
        AddedToken::from("<think>", true),
        AddedToken::from("</think>", true),
    ]);
    // `HuggingFaceTokenizer::new` loads from a file path, so the tokenizer is
    // saved to a temporary file that is deleted when `directory` drops at the
    // end of this function.
    let directory = tempdir().expect("create tokenizer directory");
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).expect("save tokenizer");
    Arc::new(HuggingFaceTokenizer::new(&path).expect("load configured tokenizer"))
}
