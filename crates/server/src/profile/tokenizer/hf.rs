//! Hugging Face tokenizer loading and the unified tokenizer implementation.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use fastokens::Tokenizer as FastokensTokenizer;
use fastokens::decoders::Decoder as FastokensDecoder;
use serde_json::{Value, json};
use thiserror_ext::AsReport as _;
use tracing::info;

use crate::profile::tokenizer::Result;
use crate::profile::tokenizer::byte_level_decode::decode_byte_level as decode_tokens_byte_level;
use crate::profile::tokenizer::incremental::IncrementalDecoder;

/// Returns whether a tokenizer decoder consists of exactly one `ByteLevel`
/// step, possibly nested inside `Sequence` steps.
///
/// The match is exhaustive over `fastokens::decoders::Decoder`, so a
/// `fastokens` upgrade that adds decoder kinds fails to compile here until each
/// new kind is classified for the byte-level fast path.
fn is_byte_level_only(decoder: &FastokensDecoder) -> bool {
    /// Counts `ByteLevel` steps across nested decoder sequences.
    fn count_byte_level(decoder: &FastokensDecoder) -> usize {
        match decoder {
            FastokensDecoder::ByteLevel(_) => 1,
            FastokensDecoder::Sequence(steps) => steps.iter().map(count_byte_level).sum(),
        }
    }
    count_byte_level(decoder) == 1
}

/// Decodes token IDs by looking up each vocabulary piece and unescaping the
/// pieces with `decode_byte_level`.
///
/// The result matches `fastokens` decoding with the same byte-level decoder.
/// IDs missing from the vocabulary contribute no text, as in Hugging Face
/// `tokenizers` and vLLM: a model's vocabulary dimension can exceed its
/// tokenizer vocabulary (Qwen3 and BAGEL checkpoints pad it), so sampling or
/// a logprob candidate can produce an ID in that gap, and one such ID must
/// not fail the whole response.
fn decode_fastokens_byte_level(
    tokenizer: &FastokensTokenizer,
    token_ids: &[u32],
    skip_special_tokens: bool,
) -> String {
    let tokens: Vec<&str> = token_ids
        .iter()
        .filter(|&&id| !(skip_special_tokens && tokenizer.is_special_token(id)))
        .filter_map(|&id| tokenizer.id_to_token(id))
        .collect();
    decode_tokens_byte_level(tokens)
}

/// Tokenizer loaded from a Hugging Face `tokenizer.json` file.
///
/// One instance is shared through `DynTokenizer` by every request of a
/// resolved model; all methods take `&self`.
pub struct HuggingFaceTokenizer {
    tokenizer: Box<FastokensTokenizer>,
    /// Whether the decoder is a single `ByteLevel` step, which routes `decode`
    /// through `decode_fastokens_byte_level`.
    byte_level: bool,
    /// IDs of added tokens flagged `special`, sorted and deduplicated for the
    /// binary search in `is_special_id`.
    special_token_ids: Arc<[u32]>,
}

impl HuggingFaceTokenizer {
    /// Loads tokenizer data and derives special-token lookup tables.
    ///
    /// Returns a tokenizer error when `fastokens` cannot load `path`.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading configured Hugging Face tokenizer");
        let tokenizer = FastokensTokenizer::from_file(path)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_fastokens(tokenizer))
    }

    /// Loads a tokenizer the way `transformers` loads its directory: the
    /// `tokenizer.json` at `path` plus the tokens the `tokenizer_config.json`
    /// at `config` declares and the definition lacks.
    ///
    /// A checkpoint may declare special tokens only in its configuration;
    /// MiniMax-H3's `<d>` dialogue marker is one. `transformers` registers
    /// them when it loads the tokenizer, so a prompt containing one encodes
    /// it as a single token; [`register_declared_tokens`] states the rule.
    ///
    /// Returns a tokenizer error when either file cannot be read or parsed,
    /// or when `fastokens` cannot build the extended definition.
    pub fn with_config(path: &Path, config: &Path) -> Result<Self> {
        info!(
            path = %path.display(),
            config = %config.display(),
            "loading configured Hugging Face tokenizer"
        );
        let mut definition = read_json(path)?;
        register_declared_tokens(&mut definition, &read_json(config)?)?;
        let tokenizer = FastokensTokenizer::from_json(definition)
            .map_err(|error| tokenizer_error!("failed to load tokenizer: {}", error.as_report()))?;
        Ok(Self::from_fastokens(tokenizer))
    }

    /// Derives the special-token lookup tables of a loaded tokenizer.
    fn from_fastokens(tokenizer: FastokensTokenizer) -> Self {
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

        Self {
            tokenizer: Box::new(tokenizer),
            byte_level,
            special_token_ids: Arc::from(special_token_ids),
        }
    }

    /// Encodes one prompt string into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(|error| tokenizer_error!("encoding failed: {}", error.as_report()))
    }

    /// Decodes token identifiers with optional special-token filtering.
    ///
    /// IDs outside the vocabulary contribute no text. A tokenizer whose
    /// decoder is a single byte-level step decodes through
    /// `decode_fastokens_byte_level`, which cannot fail. Any other tokenizer
    /// decodes through `fastokens`, which fails only when its decoder does.
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        if self.byte_level {
            Ok(decode_fastokens_byte_level(
                &self.tokenizer,
                token_ids,
                skip_special_tokens,
            ))
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
    ///
    /// `min_bytes_to_buffer` is the number of trailing output bytes that
    /// `IncrementalDecoder::next_chunk` withholds; the serving output stages
    /// pass the stop-string holdback computed by `stop_string_holdback_bytes`.
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

/// Reads one JSON tokenizer file.
fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        tokenizer_error!("failed to read {}: {}", path.display(), error.as_report())
    })?;
    serde_json::from_str(&text).map_err(|error| {
        tokenizer_error!("failed to parse {}: {}", path.display(), error.as_report())
    })
}

/// The named special tokens `transformers` registers first, in its order
/// (`SPECIAL_TOKENS_ATTRIBUTES`).
const NAMED_SPECIAL_TOKENS: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// One token a `tokenizer_config.json` declares, with its matching flags.
#[derive(Debug)]
struct DeclaredToken {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl DeclaredToken {
    /// Reads a token written as its text, which `transformers` makes a
    /// special token, or as an `AddedToken` object. An unset `normalized`
    /// defaults to `!special`, as `AddedToken` does; `special` forces the
    /// special flag of a named token.
    fn parse(value: &Value, special: bool) -> Option<Self> {
        if let Some(content) = value.as_str() {
            return Some(Self {
                content: content.to_owned(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            });
        }
        let content = value.get("content")?.as_str()?.to_owned();
        let flag = |name: &str| value.get(name).and_then(Value::as_bool);
        let special = special || flag("special").unwrap_or(false);
        Some(Self {
            content,
            single_word: flag("single_word").unwrap_or(false),
            lstrip: flag("lstrip").unwrap_or(false),
            rstrip: flag("rstrip").unwrap_or(false),
            normalized: flag("normalized").unwrap_or(!special),
            special,
        })
    }
}

/// Lists the tokens a `tokenizer_config.json` declares, in the order
/// `transformers` (5.x) registers them: the `added_tokens_decoder` entries
/// by id; the named special tokens ([`NAMED_SPECIAL_TOKENS`], then any other
/// `*_token` key holding a token, then the entries of an object-valued
/// `extra_special_tokens`); and the listed extra special tokens
/// (`extra_special_tokens`, or the older `additional_special_tokens`).
fn declared_tokens(config: &Value) -> Vec<DeclaredToken> {
    let mut tokens = Vec::new();

    if let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    {
        let mut entries: Vec<(u64, &Value)> = decoder
            .iter()
            .filter_map(|(id, token)| Some((id.parse().ok()?, token)))
            .collect();
        entries.sort_by_key(|&(id, _)| id);
        tokens.extend(
            entries
                .into_iter()
                .filter_map(|(_, token)| DeclaredToken::parse(token, false)),
        );
    }

    let named = NAMED_SPECIAL_TOKENS
        .iter()
        .filter_map(|&key| config.get(key));
    let model_specific = config.as_object().into_iter().flat_map(|fields| {
        fields
            .iter()
            .filter(|(key, _)| {
                key.ends_with("_token") && !NAMED_SPECIAL_TOKENS.contains(&key.as_str())
            })
            .map(|(_, value)| value)
    });
    let extras = config.get("extra_special_tokens");
    let named_extras = extras
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|fields| fields.values());
    tokens.extend(
        named
            .chain(model_specific)
            .chain(named_extras)
            .filter_map(|token| DeclaredToken::parse(token, true)),
    );

    // `additional_special_tokens` stands in only for an absent
    // `extra_special_tokens`.
    let listed = extras
        .or_else(|| config.get("additional_special_tokens"))
        .and_then(Value::as_array);
    tokens.extend(
        listed
            .into_iter()
            .flatten()
            .filter_map(|token| DeclaredToken::parse(token, false)),
    );
    tokens
}

/// The id a `tokenizer.json` model vocabulary gives `content`: an object
/// from text to id (BPE, WordPiece) or a list of `[text, score]` rows whose
/// position is the id (Unigram).
fn vocabulary_id(vocabulary: &Value, content: &str) -> Option<u64> {
    match vocabulary {
        Value::Object(entries) => entries.get(content)?.as_u64(),
        Value::Array(rows) => rows
            .iter()
            .position(|row| row.get(0).and_then(Value::as_str) == Some(content))
            .map(|position| position as u64),
        _ => None,
    }
}

/// Registers the tokens a `tokenizer_config.json` declares in the
/// `added_tokens` of a `tokenizer.json` definition, as `transformers` does
/// when it loads the pair.
///
/// The tokens of [`declared_tokens`] are registered in order unless their
/// text is already an added token. As in `tokenizers`, a text the model
/// vocabulary holds keeps its vocabulary id, and any other text takes the
/// next id after both the model vocabulary and every added token.
fn register_declared_tokens(definition: &mut Value, config: &Value) -> Result<()> {
    let vocabulary = &definition["model"]["vocab"];
    let vocabulary_size = match vocabulary {
        Value::Object(entries) => entries.len(),
        Value::Array(rows) => rows.len(),
        _ => return Err(tokenizer_error!("the tokenizer model has no vocabulary")),
    } as u64;
    // Resolve the vocabulary ids before the added tokens are borrowed
    // mutably.
    let declared: Vec<(DeclaredToken, Option<u64>)> = declared_tokens(config)
        .into_iter()
        .map(|token| {
            let id = vocabulary_id(vocabulary, &token.content);
            (token, id)
        })
        .collect();

    let added = definition
        .get_mut("added_tokens")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| tokenizer_error!("the tokenizer has no added_tokens list"))?;
    let mut registered: HashSet<String> = added
        .iter()
        .filter_map(|token| token["content"].as_str().map(ToOwned::to_owned))
        .collect();
    let mut next_id = added
        .iter()
        .filter_map(|token| token["id"].as_u64())
        .map(|id| id + 1)
        .fold(vocabulary_size, u64::max);
    for (token, vocabulary_id) in declared {
        if !registered.insert(token.content.clone()) {
            continue;
        }
        let id = match vocabulary_id {
            Some(id) => id,
            None => {
                let id = next_id;
                next_id += 1;
                id
            }
        };
        added.push(json!({
            "id": id,
            "content": token.content,
            "single_word": token.single_word,
            "lstrip": token.lstrip,
            "rstrip": token.rstrip,
            "normalized": token.normalized,
            "special": token.special,
        }));
    }
    Ok(())
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

        // The tiny BPE tokenizer has no decoder, so this decodes through
        // `fastokens` rather than the byte-level fast path.
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

    /// The byte-level fast path must reproduce `fastokens` decoding, with and
    /// without special-token skipping. The cases cover the empty input, `Ġ`
    /// (an escaped space), the special `<|endoftext|>` at several positions,
    /// `｜`, which lies outside GPT-2's byte alphabet and passes through
    /// unchanged, and IDs 999 and 1000, which the vocabulary lacks.
    #[test]
    fn configured_byte_level_decode_matches_provider() {
        let (_directory, configured, provider) = byte_level_tokenizers();
        let cases: &[&[u32]] = &[
            &[],
            &[1, 2, 3, 3, 4],
            &[1, 2, 3, 3, 4, 8, 5, 4, 6, 3, 7],
            &[0, 1, 2, 3, 3, 4, 0, 9, 0],
            &[10, 1, 2, 3, 3, 4, 10],
            &[999, 1, 2, 1000, 3, 3, 4, 999],
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

    /// Models sample over a vocabulary dimension that can exceed the tokenizer
    /// vocabulary, so an ID in that gap can reach decoding. Like Hugging Face
    /// `tokenizers`, decoding skips it instead of failing.
    #[test]
    fn configured_byte_level_decode_skips_unknown_token_ids() {
        let (_directory, configured, _provider) = byte_level_tokenizers();
        assert_eq!(configured.decode(&[999], false).unwrap(), "");
        assert_eq!(configured.decode(&[1, 999, 2], false).unwrap(), "He");
    }

    /// The special tokens a `tokenizer_config.json` declares but the
    /// definition lacks are registered as `transformers` registers them:
    /// once each, a vocabulary text at its vocabulary id, any other text
    /// after the largest id in declaration order (named tokens before the
    /// extra ones). The expected ids are those `transformers` 5.14 assigns
    /// when it loads the same two files.
    #[test]
    fn configured_special_tokens_are_registered_like_transformers() {
        let directory = tempdir().expect("create tokenizer directory");
        let path = directory.path().join("tokenizer.json");
        std::fs::write(&path, BYTE_LEVEL_TOKENIZER).expect("write tokenizer");
        let config = directory.path().join("tokenizer_config.json");
        std::fs::write(
            &config,
            r#"{
                "added_tokens_decoder": {
                    "0": {"content": "<|endoftext|>", "lstrip": false, "normalized": false,
                          "rstrip": false, "single_word": false, "special": true}
                },
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "add_bos_token": false,
                "additional_special_tokens": ["<|endoftext|>", "<d>", "</d>", "<d>", "H"]
            }"#,
        )
        .expect("write tokenizer config");
        let configured =
            HuggingFaceTokenizer::with_config(&path, &config).expect("load configured tokenizer");

        for (token, id) in [("<|pad|>", 11), ("<d>", 12), ("</d>", 13), ("H", 1)] {
            assert_eq!(configured.token_to_id(token), Some(id), "{token}");
            assert!(configured.is_special_id(id), "{token}");
        }
        assert_eq!(
            configured
                .encode("Hello<d>world</d>!<|pad|>", false)
                .unwrap(),
            [1, 2, 3, 3, 4, 12, 5, 4, 6, 3, 7, 13, 9, 11]
        );
    }
}
