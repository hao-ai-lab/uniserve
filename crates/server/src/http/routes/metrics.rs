use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport;
use uniserve_observability::METRICS;

use crate::AppState;

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

pub(super) async fn scrape(State(state): State<Arc<AppState>>) -> Response {
    let identity = state.runtime().model().served_identity();
    METRICS.serving.set_request_states(
        state.served_model_name(),
        identity.description.id(),
        state.runtime().metrics_snapshot().state_counts(),
    );
    match METRICS.render() {
        Ok(body) => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static(OPENMETRICS_CONTENT_TYPE),
            )],
            body,
        )
            .into_response(),

        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render metrics: {}", error.as_report()),
        )
            .into_response(),
    }
}
use std::sync::Arc;

use axum::extract::State;
