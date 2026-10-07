//! System One error bodies and their HTTP statuses.
//!
//! The endpoint follows the FastAPI conventions of the official service: a
//! request-validation failure is `422` with `{"detail": [ValidationError]}`,
//! and every other failure carries `{"detail": "<message>"}`.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::{Value, json};

/// One element of a validation error location: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum LocItem {
    /// Object field name, map key, or union-member label.
    Key(String),
    /// Array position.
    Index(usize),
}

impl From<&str> for LocItem {
    fn from(value: &str) -> Self {
        Self::Key(value.to_owned())
    }
}

impl From<usize> for LocItem {
    fn from(value: usize) -> Self {
        Self::Index(value)
    }
}

/// One request-validation failure in FastAPI's `ValidationError` shape.
///
/// Fields serialize in FastAPI's order `type`, `loc`, `msg`, `input`, `ctx`;
/// `input` and `ctx` are omitted when absent, while an explicit `null` input
/// is kept.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ValidationIssue {
    /// Machine-readable pydantic error type, such as `missing` or `too_long`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Path to the invalid value, starting with `body`.
    pub loc: Vec<LocItem>,
    /// Human-readable explanation.
    pub msg: String,
    /// The value that failed validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Error-type parameters, such as the violated length bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Value>,
}

impl ValidationIssue {
    /// Builds a pydantic `value_error` raised by a custom rule.
    ///
    /// The message gains pydantic's `Value error, ` prefix, and the context is
    /// `{"error": {}}`, which is how FastAPI encodes the raised exception.
    pub(crate) fn value_error(loc: Vec<LocItem>, message: &str, input: Option<Value>) -> Self {
        Self {
            kind: "value_error",
            loc,
            msg: format!("Value error, {message}"),
            input,
            ctx: Some(json!({ "error": {} })),
        }
    }
}

/// A failed System One request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SystemOneError {
    /// The request body, or a limit the encoded request exceeds, failed
    /// validation (`422`).
    #[error("request validation failed with {} issue(s)", .0.len())]
    Validation(Vec<ValidationIssue>),
    /// The requested model is not the served model (`404`).
    #[error("model `{requested}` is not served; this server serves `{served}`")]
    ModelNotFound {
        /// Model name the request asked for.
        requested: String,
        /// The `--served-model-name` of this server.
        served: String,
    },
    /// The server is at its concurrent-request capacity (`503`).
    #[error("{0}")]
    Overloaded(String),
    /// Inference or an internal invariant failed (`500`).
    #[error("{0}")]
    Server(String),
}

impl SystemOneError {
    /// Returns the HTTP status of this failure.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Validation(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::Overloaded(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Server(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Returns the JSON response body of this failure.
    pub fn body(&self) -> Value {
        match self {
            Self::Validation(issues) => json!({ "detail": issues }),
            Self::ModelNotFound { requested, served } => json!({
                "detail": format!(
                    "The model '{requested}' does not exist; this server serves '{served}'."
                ),
            }),
            Self::Overloaded(message) | Self::Server(message) => json!({ "detail": message }),
        }
    }
}

impl IntoResponse for SystemOneError {
    fn into_response(self) -> Response {
        (self.status_code(), Json(self.body())).into_response()
    }
}
