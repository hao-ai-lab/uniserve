use std::path::Path;
use std::sync::Arc;

use tekken::Tekkenizer;
use tracing::info;

use crate::tokenizer::grammar_vocab::SPECIAL_TOKEN_MARKER;
use crate::tokenizer::{Result, Tokenizer};

/// Mistral Tekken tokenizer from a `tekken.json` file.
pub struct TekkenTokenizer {
    inner: Tekkenizer,
    grammar_token_bytes: Arc<[Vec<u8>]>,
}

impl TekkenTokenizer {
    /// Load a Mistral Tekken tokenizer from a `tekken.json` file.
    pub fn new(path: &Path) -> Result<Self> {
        info!(path = %path.display(), "loading tokenizer with Mistral Tekken");

        let inner = Tekkenizer::from_file(path).map_err(|error| {
            tokenizer_error!(
                "failed to load tekken tokenizer from {}: {error}",
                path.display()
            )
        })?;
        let grammar_token_bytes = (0..inner.vocab().len())
            .map(|id| {
                let id = id as u32;
                let mut bytes = inner
                    .id_to_byte_piece(id, tekken::SpecialTokenPolicy::Keep)
                    .map_err(|error| {
                        tokenizer_error!("failed to decode token {id} for grammar use: {error}")
                    })?;
                if inner.is_special_token(id) {
                    bytes.insert(0, SPECIAL_TOKEN_MARKER);
                }
                Ok(bytes)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner,
            grammar_token_bytes: Arc::from(grammar_token_bytes),
        })
    }
}

impl Tokenizer for TekkenTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens, false)
            .map_err(|error| tokenizer_error!("encoding failed: {error}"))
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let policy = if skip_special_tokens {
            tekken::SpecialTokenPolicy::Ignore
        } else {
            tekken::SpecialTokenPolicy::Keep
        };
        self.inner
            .decode(token_ids, policy)
            .map_err(|error| tokenizer_error!("decoding failed: {error}"))
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // tekken-rs exposes `get_control_token` for special tokens, a direct map
        // lookup. Try that first, then fall back to encoding the literal.
        self.inner.get_control_token(token).ok().or_else(|| {
            // Encode without special tokens; a vocabulary token must encode to a
            // single piece. A multi-token result means `token` is not a single
            // vocabulary entry, so report it as unknown rather than returning the
            // first sub-token (which would misclassify the string).
            let ids = self.inner.encode(token, false, false).ok()?;
            let [id] = ids[..] else { return None };
            // Confirm the candidate id round-trips back to the exact requested
            // string. Without this check a string that merely BPE-collapses to one
            // token (e.g. differing whitespace/normalization) could resolve to an
            // id whose piece is not `token`.
            if self
                .inner
                .id_to_piece(id)
                .is_ok_and(|piece| piece.as_str() == token)
            {
                Some(id)
            } else {
                None
            }
        })
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_piece(id).ok()
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        self.inner.is_special_token(token_id)
    }

    fn grammar_token_bytes(&self) -> Result<Arc<[Vec<u8>]>> {
        Ok(Arc::clone(&self.grammar_token_bytes))
    }
}
