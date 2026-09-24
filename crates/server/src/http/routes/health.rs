//! Liveness endpoint backed by engine health.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;

use crate::AppState;

/// Returns service availability according to engine health.
///
/// Answers `200 OK` until the engine core is reported dead
/// (`EngineClient::is_healthy`), then `503 Service Unavailable`. The route is
/// exempt from the API key and excluded from HTTP metrics.
pub(super) async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.engine().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
