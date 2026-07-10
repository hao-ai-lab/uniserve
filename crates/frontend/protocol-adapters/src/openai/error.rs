use thiserror_ext::AsReport as _;
use uniserve_openai_types::{ErrorDetail, ErrorResponse};
use uniserve_serving::ServeError;

/// Error categories raised while validating or lowering OpenAI-compatible
/// requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiError {
    /// The JSON request is syntactically valid but asks for unsupported or
    /// invalid behavior.
    InvalidRequest {
        message: String,
        param: Option<&'static str>,
    },
    /// The requested model name does not match any served model or adapter.
    ModelNotFound { model: String },
    /// An internal conversion invariant failed.
    ServerError { message: String },
}

pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    pub fn invalid_request(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param,
        }
    }

    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    pub fn server_error(message: impl Into<String>) -> Self {
        Self::ServerError {
            message: message.into(),
        }
    }

    /// Convert this error into the standard OpenAI-compatible JSON error
    /// payload.
    pub fn to_error_response(&self) -> ErrorResponse {
        let error = match self {
            Self::InvalidRequest { message, param } => ErrorDetail {
                message: message.clone(),
                error_type: "invalid_request_error".to_string(),
                param: param.map(|p| p.to_string()),
                code: Some("invalid_request_error".to_string()),
            },
            Self::ModelNotFound { model } => ErrorDetail {
                message: format!("The model `{model}` does not exist."),
                error_type: "invalid_request_error".to_string(),
                param: Some("model".to_string()),
                code: Some("model_not_found".to_string()),
            },
            Self::ServerError { message } => ErrorDetail {
                message: message.clone(),
                error_type: "server_error".to_string(),
                param: None,
                code: Some("server_error".to_string()),
            },
        };

        ErrorResponse { error }
    }
}

/// Map one canonical serving error into the OpenAI error vocabulary.
pub fn serve_error_to_api(error: ServeError) -> ApiError {
    match error {
        ServeError::UnsupportedOutputCount { requested, .. } => ApiError::invalid_request(
            format!("Only one output is supported, got {requested}."),
            Some("n"),
        ),
        error @ (ServeError::UnsupportedCapability { .. }
        | ServeError::TokenizerMismatch { .. }
        | ServeError::ContextLengthExceeded { .. }
        | ServeError::ContextCapacityExceeded { .. }
        | ServeError::DuplicateRequestId { .. }) => {
            ApiError::invalid_request(error.to_string(), None)
        }
        ServeError::ForeignExecutionPlan { .. } => {
            ApiError::server_error("execution plan/runtime identity mismatch".to_string())
        }
        ServeError::Compilation { message, .. } => ApiError::invalid_request(message, None),
        ServeError::Text(error) if text_error_is_invalid_request(&error) => {
            ApiError::invalid_request(error.to_string(), None)
        }
        ServeError::Text(error) => {
            ApiError::server_error(format!("text runtime error: {}", error.to_report_string()))
        }
        ServeError::Chat(error) if chat_error_is_invalid_request(&error) => {
            ApiError::invalid_request(error.to_string(), None)
        }
        ServeError::Chat(error) => {
            ApiError::server_error(format!("chat runtime error: {}", error.to_report_string()))
        }
        ServeError::Engine(message) => {
            ApiError::server_error(format!("engine runtime error: {message}"))
        }
        ServeError::OutputProcessing { message, .. } => {
            ApiError::server_error(format!("output processing error: {message}"))
        }
    }
}

fn text_error_is_invalid_request(error: &uniserve_serving::text::Error) -> bool {
    use uniserve_serving::text::Error;

    matches!(
        error,
        Error::StructuredOutput(_)
            | Error::InvalidSampling(_)
            | Error::InvalidLogprobCount { .. }
            | Error::MinTokensExceedsMaximum { .. }
            | Error::EmptyPromptTokenIds { .. }
            | Error::AdapterIdOutOfRange { .. }
            | Error::InvalidGenerationRequest(_)
            | Error::PromptTooLong { .. }
    )
}

fn chat_error_is_invalid_request(error: &uniserve_serving::chat::Error) -> bool {
    use uniserve_serving::chat::Error;

    match error {
        Error::EmptyMessages
        | Error::ContinueFinalAssistantWithoutFinalAssistant
        | Error::UnsupportedMultimodalRenderer
        | Error::UnsupportedMultimodalContent(_)
        | Error::ParserUnavailableForModel { .. }
        | Error::ParserDisabled { .. }
        | Error::ParserUnavailableByName { .. }
        | Error::PromptTooLong { .. } => true,
        Error::Text(error) => text_error_is_invalid_request(error),
        Error::MissingChatTemplate
        | Error::ChatTemplate(_)
        | Error::Multimodal(_)
        | Error::ParserInitialization { .. }
        | Error::HarmonyParserOverrideUnsupported { .. }
        | Error::HarmonyOutputParsing { .. }
        | Error::StreamClosedBeforeTerminalOutput { .. }
        | Error::ToolCallStreamInvariant { .. }
        | Error::ModelAssets(_) => false,
    }
}

macro_rules! bail_invalid_request {
    (param = $param:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        {
            return Err($crate::openai::error::ApiError::invalid_request(
                format!($fmt $(, $arg)*),
                Some($param),
            ))
        }
    };
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        {
            return Err($crate::openai::error::ApiError::invalid_request(
                format!($fmt $(, $arg)*),
                None,
            ))
        }
    };
}

pub(crate) use bail_invalid_request;

macro_rules! server_error {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::openai::error::ApiError::server_error(format!($fmt $(, $arg)*))
    };
}

pub(crate) use server_error;
