use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;

use crate::AppState;

pub(super) async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.engine_status().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
