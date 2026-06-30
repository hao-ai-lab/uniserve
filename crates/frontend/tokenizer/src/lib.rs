#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::sync::Arc;

use crate::incremental::DecodeStream;

mod byte_level_decode;
#[macro_use]
mod error;
mod hf;
mod incremental;
mod tekken;
mod tiktoken;

pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::IncrementalDecoder;
pub use tekken::TekkenTokenizer;
pub use tiktoken::TiktokenTokenizer;

pub trait Tokenizer: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Convert one token string into a token ID, returning `None` if the token
    /// is not in the tokenizer vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Convert one token ID into the tokenizer's raw token string, returning
    /// `None` when the backend cannot resolve the ID.
    fn id_to_token(&self, _id: u32) -> Option<String> {
        // The default keeps existing tokenizer backends source-compatible; new
        // backends should override this when exact token strings are available.
        // Callers cannot distinguish "unknown ID" from "backend has no mapping",
        // so any backend with real token strings must override this.
        None
    }

    /// Return whether the given token ID is special.

    /// The default returns `false` to keep existing tokenizer backends
    /// source-compatible; any backend that knows its special-token set must
    /// override this, since callers treat the default as "not special" rather
    /// than "unknown" (e.g. reasoning-boundary detection and special-token
    /// skipping silently no-op on the default).
    fn is_special_id(&self, _token_id: u32) -> bool {
        false
    }

    /// Create a stateful incremental decoder primed with the given prompt
    /// tokens.

    /// The prompt tokens provide left context for the first generated token;
    /// the decoder does not re-emit prompt text.
    fn create_decode_stream(
        &self,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Box<dyn IncrementalDecoder + '_> {
        Box::new(DecodeStream::new(
            self,
            prompt_token_ids,
            skip_special_tokens,
            min_bytes_to_buffer,
        ))
    }
}

pub type DynTokenizer = Arc<dyn Tokenizer>;
