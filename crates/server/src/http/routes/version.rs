//! Version metadata endpoint.

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
/// Build version returned by the version endpoint.
pub(crate) struct VersionResponse {
    version: &'static str,
}

/// Returns the UniServe distribution version.
pub(super) async fn version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION"),
    })
}
