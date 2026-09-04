//! Errors raised while locating and reading model assets.

use thiserror::Error;

#[derive(Debug, Error)]
/// Model-asset discovery and deserialization failure.
pub enum Error {
    #[error("failed to read model asset `{path}`")]
    /// A local model asset could not be read.
    Io {
        /// Path of the unreadable asset.
        path: std::path::PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model asset `{path}` as JSON")]
    /// A model asset contains invalid JSON.
    Json {
        /// Path of the malformed asset.
        path: std::path::PathBuf,
        /// Underlying JSON parser error.
        #[source]
        source: serde_json::Error,
    },
    #[error("model `{model}` is missing required asset `{file}`")]
    /// A required model file cannot be resolved locally or remotely.
    MissingFile {
        /// Model identifier or local directory.
        model: String,
        /// Required filename.
        file: &'static str,
    },
    #[error("model metadata is missing required field `{field}`")]
    /// Required model metadata is absent.
    MissingField {
        /// Required metadata field name.
        field: &'static str,
    },
    #[error("model type mismatch: expected {expected}, found {actual}")]
    /// Model metadata declares a different architecture family.
    ModelTypeMismatch {
        /// Architecture family required by the selected profile.
        expected: &'static str,
        /// Architecture family declared by the model.
        actual: String,
    },
    #[error("failed to resolve remote model `{model}`: {message}")]
    /// Hugging Face Hub model resolution failed.
    Remote {
        /// Remote model identifier.
        model: String,
        /// Human-readable remote resolution failure.
        message: String,
    },
    #[error("invalid model asset: {0}")]
    /// Model asset content violates a profile invariant.
    Invalid(String),
}

/// Result type returned by model-asset operations.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Constructs an invalid-asset error with contextual text.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
