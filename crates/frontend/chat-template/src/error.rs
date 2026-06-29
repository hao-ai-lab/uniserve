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
    ModelAssets(#[from] uniserve_model_assets::Error),
    #[error(transparent)]
    Protocol(#[from] uniserve_chat_protocol::Error),
    #[error(transparent)]
    Text(#[from] uniserve_text::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
