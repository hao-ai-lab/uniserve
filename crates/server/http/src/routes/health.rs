use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;

use uniserve_server_app::AppState;

pub(super) async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.chat().uniserve_engine_client().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}
