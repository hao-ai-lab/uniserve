use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("model asset error: {0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}
