//! OpenAI-compatible error categories and HTTP response conversion.
//!
//! [`ApiError`] is the error vocabulary of the OpenAI surface. Each variant
//! fixes an HTTP status ([`ApiError::status_code`]) and an OpenAI error body
//! with a stable `type` and `code` ([`ApiError::to_error_response`]).
//! Serving-layer failures enter through [`serve_error_to_api`] and engine
//! admission rejections through [`ApiError::rejected`].

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
        /// Human-readable validation failure.
        message: String,
        /// Request parameter associated with the failure, when known.
        param: Option<&'static str>,
    },
    /// The requested model name does not match any served model.
    ModelNotFound {
        /// Requested model name.
        model: String,
    },
    /// The server could not produce a result, for example because an
    /// internal invariant, model resolution, the engine, or output processing
    /// failed, or because generation ended without a usable output.
    ServerError {
        /// Human-readable internal failure.
        message: String,
    },
    /// The request body could not be parsed as valid JSON.
    JsonParseError {
        /// Parser failure message.
        message: String,
    },
    /// A valid call conflicts with the current runtime state, for example a
    /// request for the content of a video job that has none (queued, running,
    /// or failed).
    Conflict {
        /// Human-readable conflict description.
        message: String,
    },
    /// The engine's waiting queue is full; the same request may succeed later.
    Overloaded {
        /// Human-readable admission failure.
        message: String,
    },
}

/// Result type for OpenAI request validation and conversion.
pub type Result<T> = std::result::Result<T, ApiError>;

impl ApiError {
    /// Constructs an invalid-request error for an optional parameter.
    pub fn invalid_request(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param,
        }
    }

    /// Constructs an error for a model name not served by this configuration.
    pub fn model_not_found(model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model: model.into(),
        }
    }

    /// Constructs an internal server error.
    pub fn server_error(message: impl Into<String>) -> Self {
        Self::ServerError {
            message: message.into(),
        }
    }

    /// Constructs an error for malformed JSON input.
    pub fn json_parse_error(message: impl Into<String>) -> Self {
        Self::JsonParseError {
            message: message.into(),
        }
    }

    /// Constructs a request-conflict error.
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::Conflict {
            message: message.into(),
        }
    }

    /// Maps an engine admission rejection to its caller-facing category.
    ///
    /// `Invalid` becomes a 400 invalid request that fails the same way on
    /// retry; `Overloaded` becomes a 503 that the same request may pass once
    /// the engine's waiting queue drains.
    pub fn rejected(kind: uniserve_core::RejectionKind, message: impl Into<String>) -> Self {
        match kind {
            uniserve_core::RejectionKind::Invalid => Self::invalid_request(message, None),
            uniserve_core::RejectionKind::Overloaded => Self::Overloaded {
                message: message.into(),
            },
        }
    }

    /// Returns the HTTP status associated with this error category.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } | Self::JsonParseError { .. } => StatusCode::BAD_REQUEST,
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::ServerError { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::Overloaded { .. } => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    /// Returns the stable machine-readable `code` of this error category, as
    /// the error body reports it.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } => "invalid_request_error",
            Self::ModelNotFound { .. } => "model_not_found",
            Self::ServerError { .. } => "server_error",
            Self::JsonParseError { .. } => "json_parse_error",
            Self::Conflict { .. } => "runtime_state_conflict",
            Self::Overloaded { .. } => "server_overloaded",
        }
    }

    /// Converts this error into the standard OpenAI-compatible JSON error
    /// payload, whose `code` is [`ApiError::code`].
    pub fn to_error_response(&self) -> ErrorResponse {
        let (message, error_type, param) = match self {
            Self::InvalidRequest { message, param } => {
                (message.clone(), "invalid_request_error", *param)
            }
            Self::ModelNotFound { model } => (
                format!("The model `{model}` does not exist."),
                "invalid_request_error",
                Some("model"),
            ),
            Self::ServerError { message } => (message.clone(), "server_error", None),
            Self::JsonParseError { message } => (message.clone(), "invalid_request_error", None),
            Self::Conflict { message } => (message.clone(), "conflict_error", None),
            Self::Overloaded { message } => (message.clone(), "server_error", None),
        };

        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                param: param.map(str::to_string),
                code: Some(self.code().to_string()),
            },
        }
    }
}

impl IntoResponse for ApiError {
    /// Converts the error into an HTTP response.
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.to_error_response())).into_response()
    }
}

/// Maps one canonical serving error into the OpenAI error vocabulary.
///
/// Unsupported output counts or features, context limits, duplicate request
/// IDs, and tokenization failures become invalid requests; model-resolution,
/// engine, and output-processing failures become server errors. A
/// tokenization failure's message appends its cause (the violated field or
/// limit), which `ServeError::Tokenize` carries as its error source rather
/// than in its own `Display`.
pub fn serve_error_to_api(error: ServeError) -> ApiError {
    match error {
        ServeError::UnsupportedOutputCount { requested, .. } => ApiError::invalid_request(
            format!("Only one output is supported, got {requested}."),
            Some("n"),
        ),
        error @ (ServeError::UnsupportedFeature { .. }
        | ServeError::ContextLengthExceeded { .. }
        | ServeError::ContextCapacityExceeded { .. }
        | ServeError::DuplicateRequestId { .. }) => {
            ApiError::invalid_request(error.to_string(), None)
        }
        ServeError::Tokenize { ref source, .. } => {
            ApiError::invalid_request(format!("{error}: {source}"), None)
        }
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

/// Returns early with an [`ApiError::InvalidRequest`] built from a format
/// string, optionally tagged with `param = <&'static str>`.
///
/// The expansion is a `return` statement, so it is usable only inside a
/// function whose error type is [`ApiError`].
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

/// Builds (without returning) an [`ApiError::ServerError`] from a format
/// string.
macro_rules! server_error {
    ($fmt:literal $(, $arg:expr)* $(,)?) => {
        $crate::openai::error::ApiError::server_error(format!($fmt $(, $arg)*))
    };
}

pub(crate) use server_error;

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::RejectionKind;

    #[test]
    fn an_overloaded_engine_is_retryable_and_an_invalid_request_is_not() {
        let overloaded = ApiError::rejected(RejectionKind::Overloaded, "queue full");
        assert_eq!(overloaded.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            overloaded.to_error_response().error.code.as_deref(),
            Some("server_overloaded")
        );

        let invalid = ApiError::rejected(RejectionKind::Invalid, "unsupported");
        assert_eq!(invalid.status_code(), StatusCode::BAD_REQUEST);
    }

    /// A preprocessing rejection must tell the caller which limit or field it
    /// violated, not only that the request was refused.
    #[test]
    fn a_tokenize_rejection_reports_its_cause() {
        let cause = "video duration must be finite, positive, and at most 15 seconds";
        let error = serve_error_to_api(ServeError::Tokenize {
            request_id: crate::serving::ServeRequestId::new("vid-1"),
            source: crate::serving::TokenizeError::Invalid(cause.to_string()),
        });

        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
        let message = error.to_error_response().error.message;
        assert!(message.contains("vid-1"), "{message}");
        assert!(message.contains(cause), "{message}");
    }
}
