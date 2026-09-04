//! Hugging Face tokenizer loading and the unified tokenizer implementation.

use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use thiserror_ext::AsReport as _;
use tracing::info;

use crate::profile::tokenizer::Result;
use crate::profile::tokenizer::byte_level_decode::decode_byte_level as decode_tokens_byte_level;
use crate::profile::tokenizer::incremental::IncrementalDecoder;

/// Returns whether a tokenizer decoder is exclusively byte-level.
fn is_byte_level_only(decoder: &FastokensDecoder) -> bool {
    /// Returns the count byte level.
    fn count_byte_level(decoder: &FastokensDecoder) -> usize {
        match decoder {
            FastokensDecoder::ByteLevel(_) => 1,
            FastokensDecoder::Sequence(steps) => steps.iter().map(count_byte_level).sum(),
        }
    }
    count_byte_level(decoder) == 1
}
/// Decodes tokens with the byte-level fast tokenizer path.
fn decode_fastokens_byte_level(
    tokenizer: &FastokensTokenizer,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> Result<String> {
    let tokens: Vec<&str> = token_ids
        .iter()
        .filter(|&&id| !(skip_special_tokens && tokenizer.is_special_token(id)))
        .map(|&id| {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| tokenizer_error!("decoding failed: unknown token ID: {id}"))
        })
        .collect::<Result<_>>()?;
    Ok(decode_tokens_byte_level(tokens))
}

/// Load-bound tokenizer for the configured Hugging Face tokenizer.json format.
pub struct HuggingFaceTokenizer {
    tokenizer: Box<FastokensTokenizer>,
    byte_level: bool,
    special_token_ids: Arc<[u32]>,
}

impl HuggingFaceTokenizer {
    /// Loads tokenizer data and derives special-token lookup tables.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading configured Hugging Face tokenizer");
        let tokenizer = FastokensTokenizer::from_file(path)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        let mut special_token_ids: Vec<u32> = tokenizer
            .added_tokens()
            .into_iter()
            .flat_map(|tokens| tokens.iter())
            .filter(|token| token.special)
            .map(|token| token.id)
            .collect();
        special_token_ids.sort_unstable();
        special_token_ids.dedup();
        let byte_level = tokenizer.decoder().is_some_and(is_byte_level_only);
        Ok(Self {
            tokenizer: Box::new(tokenizer),
            byte_level,
            special_token_ids: Arc::from(special_token_ids),
        })
    }
    /// Encodes one prompt string into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))
    }

    /// Decodes token identifiers with optional special-token filtering.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if self.byte_level {
            decode_fastokens_byte_level(&self.tokenizer, token_ids, skip_special_tokens)
        } else {
            self.tokenizer
                .decode(token_ids, skip_special_tokens)
                .map_err(|error| tokenizer_error!("decoding failed: {}", error.as_report()))
        }
    }

    /// Resolves one vocabulary token to its identifier.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Resolves one vocabulary identifier to its token text.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id).map(ToOwned::to_owned)
    }

    /// Returns whether an identifier belongs to the special-token set.
    pub fn is_special_id(&self, token_id: u32) -> bool {
        self.special_token_ids.binary_search(&token_id).is_ok()
    }

    /// Creates a stateful incremental decoder primed with the given prompt tokens.
    pub fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> IncrementalDecoder<'_> {
        IncrementalDecoder::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        )
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer as TokenizerBuilder};

    use super::HuggingFaceTokenizer;

    fn tiny_bpe_tokenizer() -> TokenizerBuilder {
        let model = BPE::builder()
            .vocab_and_merges(
                [
                    ("<unk>".to_string(), 0),
                    ("h".to_string(), 1),
                    ("e".to_string(), 2),
                    ("l".to_string(), 3),
                    ("o".to_string(), 4),
                    ("he".to_string(), 5),
                    ("ll".to_string(), 6),
                    ("hell".to_string(), 7),
                    ("hello".to_string(), 8),
                ],
                vec![
                    ("h".to_string(), "e".to_string()),
                    ("l".to_string(), "l".to_string()),
                    ("he".to_string(), "ll".to_string()),
                    ("hell".to_string(), "o".to_string()),
                ],
            )
            .unk_token("<unk>".to_string())
            .build()
            .expect("build bpe tokenizer");
        TokenizerBuilder::new(model)
    }

    fn save_tokenizer(tokenizer: &TokenizerBuilder) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempdir().expect("create tokenizer directory");
        let path = directory.path().join("tokenizer.json");
        tokenizer.save(&path, false).expect("save tokenizer");
        (directory, path)
    }

    #[test]
    fn configured_tokenizer_encodes_decodes_and_identifies_special_tokens() {
        let mut tokenizer = tiny_bpe_tokenizer();
        tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
        let (_directory, path) = save_tokenizer(&tokenizer);
        let configured = HuggingFaceTokenizer::new(&path).expect("load configured tokenizer");

        let encoded = configured.encode("hello", false).expect("encode text");
        assert_eq!(configured.decode(&encoded, false).unwrap(), "hello");
        let special_id = configured
            .token_to_id("<|im_end|>")
            .expect("resolve special token");
        assert!(configured.is_special_id(special_id));
    }

    const BYTE_LEVEL_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "single_word": false,
             "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                          "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                    "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": {
                "<|endoftext|>": 0,
                "H": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7,
                "Ġ": 8, "!": 9, "｜": 10
            },
            "merges": []
        }
    }"#;

    fn byte_level_tokenizers() -> (
        tempfile::TempDir,
        HuggingFaceTokenizer,
        fastokens::Tokenizer,
    ) {
        let directory = tempdir().expect("create tokenizer directory");
        let path = directory.path().join("tokenizer.json");
        std::fs::write(&path, BYTE_LEVEL_TOKENIZER).expect("write tokenizer");
        let configured = HuggingFaceTokenizer::new(&path).expect("load configured tokenizer");
        let value = serde_json::from_str(BYTE_LEVEL_TOKENIZER).expect("parse tokenizer");
        let provider = fastokens::Tokenizer::from_json(value).expect("load provider tokenizer");
        (directory, configured, provider)
    }

    #[test]
    fn configured_byte_level_decode_matches_provider() {
        let (_directory, configured, provider) = byte_level_tokenizers();
        let cases: &[&[u32]] = &[
            &[],
            &[1, 2, 3, 3, 4],
            &[1, 2, 3, 3, 4, 8, 5, 4, 6, 3, 7],
            &[0, 1, 2, 3, 3, 4, 0, 9, 0],
            &[10, 1, 2, 3, 3, 4, 10],
        ];
        for ids in cases {
            for &skip_special_tokens in &[false, true] {
                let expected = provider
                    .decode(ids, skip_special_tokens)
                    .expect("provider decode");
                let actual = configured
                    .decode(ids, skip_special_tokens)
                    .expect("configured decode");
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn configured_byte_level_decode_rejects_unknown_token_id() {
        let (_directory, configured, _provider) = byte_level_tokenizers();
        let error = configured
            .decode(&[999], false)
            .expect_err("unknown token ID must fail");
        assert!(error.to_string().contains("999"));
    }
}
