#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
//! Tokenizer abstraction, Hugging Face implementation, and incremental decoding.
//!
//! `HuggingFaceTokenizer` loads a model's `tokenizer.json` through `fastokens`
//! (with `HuggingFaceTokenizer::with_config`, as a diffusers pipeline's
//! tokenizer component is loaded, also the special tokens its
//! `tokenizer_config.json` declares) and serves both directions of the text
//! path: `serving::model`, `serving::omni`, and the `profile::omni` prompt
//! builders encode prompts into token IDs (`serving::sampling` also encodes
//! bad words), and the output stages in `serving::text::output` and
//! `serving::assembly` turn generated IDs back into streamed text through
//! `IncrementalDecoder`.

use std::sync::Arc;

mod byte_level_decode;
#[macro_use]
mod error;
mod hf;
mod incremental;

pub use error::{Result, TokenizerError};
pub use hf::HuggingFaceTokenizer;
pub use incremental::IncrementalDecoder;

/// The configured tokenizer instance shared by one resolved model.
pub type DynTokenizer = Arc<HuggingFaceTokenizer>;
