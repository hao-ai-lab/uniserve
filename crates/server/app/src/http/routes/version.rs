use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct VersionResponse {
    version: String,
    rust_frontend_version: &'static str,
}

/// Get engine and Rust frontend version metadata.
pub(super) async fn version(
    State(state): State<Arc<AppState>>,
) -> Result<Json<VersionResponse>, ApiError> {
    let version = state
        .engine_control()
        .version()
        .map_err(|error| utility_call_error("version", error))?;

    Ok(Json(VersionResponse {
        version,
        rust_frontend_version: env!("CARGO_PKG_VERSION"),
    }))
}
