use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("chat template is required but none was configured")]
    MissingChatTemplate,
    #[error("chat template error: {0}")]
    ChatTemplate(String),
    #[error("multimodal input is not supported by this chat renderer")]
    UnsupportedMultimodalRenderer,
    #[error("unsupported multimodal content: {0}")]
    UnsupportedMultimodalContent(&'static str),
    #[error(transparent)]
    ModelAssets(#[from] uniserve_model_profile::assets::Error),
    #[error(transparent)]
    Protocol(#[from] crate::chat::protocol::Error),
    #[error(transparent)]
    Text(#[from] crate::text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
