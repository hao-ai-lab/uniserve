//! Liveness endpoint backed by engine health.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;

use crate::AppState;

/// Returns service availability according to engine health.
pub(super) async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.engine().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
