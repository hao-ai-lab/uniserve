use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to read model asset `{path}`")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse model asset `{path}` as JSON")]
    Json {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("model `{model}` is missing required asset `{file}`")]
    MissingFile { model: String, file: &'static str },
    #[error("model metadata is missing required field `{field}`")]
    MissingField { field: &'static str },
    #[error("model type mismatch: expected {expected}, found {actual}")]
    ModelTypeMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("failed to resolve remote model `{model}`: {message}")]
    Remote { model: String, message: String },
    #[error("invalid model asset: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}
