use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror_ext::{Construct, Macro};
use uniserve_openai_types::{ErrorDetail, ErrorResponse};

/// Small OpenAI-style error family used by the minimal HTTP layer.
#[derive(Debug, Construct, Macro)]
pub enum ApiError {
    /// The request is syntactically valid OpenAI JSON but asks for unsupported
    /// behavior.
    InvalidRequest {
        message: String,
        param: Option<&'static str>,
    },
    /// The requested model name does not match the single configured model.
    ModelNotFound { model: String },
    /// The request body could not be parsed as valid JSON.
    JsonParseError { message: String },
    /// An unexpected internal failure happened before streaming started.
    ServerError { message: String },
    /// A valid control operation cannot be applied to the current runtime state.
    Conflict { message: String },
}

impl ApiError {
    /// Return the HTTP status code associated with this API error.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
        }
    }

    /// Convert this error into the standard OpenAI-compatible JSON error
    /// payload.
    ///
    /// The shared variants (`InvalidRequest`/`ModelNotFound`/`ServerError`)
    /// delegate to [`uniserve_protocol_adapters::openai::ApiError`] so the OpenAI error JSON
    /// shape lives in exactly one place. Only `JsonParseError`, which has no
    /// upstream counterpart, is mapped locally.
    pub fn to_error_response(&self) -> ErrorResponse {
        match self {
            Self::InvalidRequest { message, param } => {
                uniserve_protocol_adapters::openai::ApiError::InvalidRequest {
                    message: message.clone(),
                    param: *param,
                }
                .to_error_response()
            }
            Self::ModelNotFound { model } => {
                uniserve_protocol_adapters::openai::ApiError::ModelNotFound {
                    model: model.clone(),
                }
                .to_error_response()
            }
            Self::ServerError { message } => {
                uniserve_protocol_adapters::openai::ApiError::ServerError {
                    message: message.clone(),
                }
                .to_error_response()
            }
            Self::Conflict { message } => ErrorResponse {
                error: ErrorDetail {
                    message: message.clone(),
                    error_type: "conflict_error".to_string(),
                    param: None,
                    code: Some("runtime_state_conflict".to_string()),
                },
            },
            Self::JsonParseError { message } => ErrorResponse {
                error: ErrorDetail {
                    message: message.clone(),
                    error_type: "invalid_request_error".to_string(),
                    param: None,
                    code: Some("json_parse_error".to_string()),
                },
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
    }
}

impl From<uniserve_protocol_adapters::openai::ApiError> for ApiError {
    fn from(error: uniserve_protocol_adapters::openai::ApiError) -> Self {
        match error {
            uniserve_protocol_adapters::openai::ApiError::InvalidRequest { message, param } => {
                Self::InvalidRequest { message, param }
            }
            uniserve_protocol_adapters::openai::ApiError::ModelNotFound { model } => {
                Self::ModelNotFound { model }
            }
            uniserve_protocol_adapters::openai::ApiError::ServerError { message } => {
                Self::ServerError { message }
            }
        }
    }
}
