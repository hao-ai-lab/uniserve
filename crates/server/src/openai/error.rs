use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::openai::types::{ErrorDetail, ErrorResponse};
use crate::serving::ServeError;

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
    /// The requested model name does not match any served model.
    ModelNotFound { model: String },
    /// An internal conversion invariant failed.
    ServerError { message: String },
    /// The request body could not be parsed as valid JSON.
    JsonParseError { message: String },
    /// A valid control operation conflicts with the current runtime state.
    Conflict { message: String },
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

    pub fn json_parse_error(message: impl Into<String>) -> Self {
        Self::JsonParseError {
            message: message.into(),
        }
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::Conflict {
            message: message.into(),
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } | Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Conflict { .. } => StatusCode::CONFLICT,
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
            Self::JsonParseError { message } => ErrorDetail {
                message: message.clone(),
                error_type: "invalid_request_error".to_string(),
                param: None,
                code: Some("json_parse_error".to_string()),
            },
            Self::Conflict { message } => ErrorDetail {
                message: message.clone(),
                error_type: "conflict_error".to_string(),
                param: None,
                code: Some("runtime_state_conflict".to_string()),
            },
        };

        ErrorResponse { error }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
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
        | ServeError::ContextLengthExceeded { .. }
        | ServeError::ContextCapacityExceeded { .. }
        | ServeError::DuplicateRequestId { .. }
        | ServeError::Tokenize { .. }) => ApiError::invalid_request(error.to_string(), None),
        ServeError::ModelResolution(source) => {
            ApiError::server_error(format!("model resolution error: {source}"))
        }
        ServeError::Engine(source) => {
            ApiError::server_error(format!("engine runtime error: {source}"))
        }
        ServeError::OutputProcessing { source, .. } => {
            ApiError::server_error(format!("output processing error: {source}"))
        }
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
