use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::AppState;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata.
pub(super) async fn version(State(state): State<Arc<AppState>>) -> Json<VersionResponse> {
    Json(VersionResponse {
        version: state.engine().uniserve_version().to_owned(),
        rust_frontend_version: env!("CARGO_PKG_VERSION"),
    })
}
