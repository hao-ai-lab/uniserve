use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::AppState;
use crate::http::error::ApiError;
use crate::http::utils::utility_call_error;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResetPrefixCacheParams {
    #[serde(default)]
    reset_running_requests: bool,
    #[serde(default)]
    reset_external: bool,
}

/// Reset the local prefix cache and optionally the connector-managed external
/// cache.
pub(super) async fn reset_prefix_cache(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ResetPrefixCacheParams>,
) -> Result<StatusCode, ApiError> {
    if params.reset_external {
        return Err(ApiError::InvalidRequest {
            message: "no external prefix-cache connector is configured".to_string(),
            param: Some("reset_external"),
        });
    }
    let reset = state
        .engine_control()
        .reset_prefix_cache(params.reset_running_requests, params.reset_external)
        .await
        .map_err(|error| utility_call_error("reset_prefix_cache", error))?;
    if !reset {
        return Err(ApiError::Conflict {
            message:
                "prefix cache reset could not complete under the current running-request policy"
                    .to_string(),
        });
    }

    Ok(StatusCode::OK)
}

/// Reset the multi-modal cache.
pub(super) async fn reset_mm_cache(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_control()
        .reset_mm_cache()
        .await
        .map_err(|error| utility_call_error("reset_mm_cache", error))?;

    Ok(StatusCode::OK)
}

/// Reset the encoder cache.
pub(super) async fn reset_encoder_cache(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode, ApiError> {
    state
        .engine_control()
        .reset_encoder_cache()
        .await
        .map_err(|error| utility_call_error("reset_encoder_cache", error))?;

    Ok(StatusCode::OK)
}
