use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use uniserve_server_app::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata.
pub(super) async fn version(State(state): State<Arc<AppState>>) -> Json<VersionResponse> {
    let version = state
        .uniserve_engine_client()
        .uniserve_version()
        .to_string();

    Json(VersionResponse {
        version,
        rust_frontend_version: env!("CARGO_PKG_VERSION"),
    })
}
