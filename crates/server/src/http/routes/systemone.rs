//! HTTP handler for the System One decision readout (`POST /v1/systemone`).

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::json;

use crate::AppState;
use crate::http::middleware::RequestId;
use crate::serving::ServeRequestId;

/// Answers one System One request.
///
/// The body is read as bytes so that malformed JSON is reported in the
/// endpoint's own `422` shape. A model without the readout answers `404`, as
/// for a route it does not serve.
pub(crate) async fn systemone(
    State(state): State<Arc<AppState>>,
    Extension(RequestId(base_id)): Extension<RequestId>,
    body: Bytes,
) -> Response {
    let runtime = state.runtime();
    if !runtime.serves_readout() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "detail": "Not Found" })),
        )
            .into_response();
    }
    let request_id = ServeRequestId::new(format!("systemone-{base_id}"));
    match runtime.systemone(&request_id, &body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.into_response(),
    }
}
