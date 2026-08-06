#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
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
