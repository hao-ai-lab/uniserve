mod health;
mod metrics;
pub(crate) mod openai;
mod version;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{Next, from_fn, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::http::middleware;

fn authorize(headers: &HeaderMap, expected: &str) -> bool {
    let Some(value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some(token) = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
    else {
        return false;
    };
    constant_time_eq(token.trim().as_bytes(), expected.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0_u8;
    for (left, right) in left.iter().zip(right.iter()) {
        difference |= left ^ right;
    }
    difference == 0
}

fn unauthorized_response() -> Response {
    let body = json!({
        "error": {
            "message": "Missing or invalid API key. Provide `Authorization: Bearer <key>`.",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    });
    (StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

fn timeout_response(timeout: Duration) -> Response {
    let body = json!({
        "error": {
            "message": format!("Request exceeded the {}s server timeout.", timeout.as_secs()),
            "type": "timeout",
            "code": "request_timeout",
        }
    });
    (StatusCode::GATEWAY_TIMEOUT, Json(body)).into_response()
}

const AUTH_EXEMPT_PATHS: &[&str] = &["/health", "/metrics"];

async fn require_api_key(api_key: Option<Arc<String>>, request: Request, next: Next) -> Response {
    if AUTH_EXEMPT_PATHS.contains(&request.uri().path())
        || api_key
            .as_deref()
            .is_none_or(|expected| authorize(request.headers(), expected))
    {
        next.run(request).await
    } else {
        unauthorized_response()
    }
}

async fn enforce_timeout(timeout: Duration, request: Request, next: Next) -> Response {
    match tokio::time::timeout(timeout, next.run(request)).await {
        Ok(response) => response,
        Err(_) => timeout_response(timeout),
    }
}

/// Build the complete configured public router.
pub fn build_router(state: Arc<AppState>) -> Router {
    let enable_request_id_headers = state.enable_request_id_headers();
    let request_timeout = state.request_timeout();
    let api_key = state.api_key().map(|key| Arc::new(key.to_string()));

    let mut router = Router::new()
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/version", get(version::version))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/images/generations", post(openai::images_generations))
        .with_state(Arc::clone(&state));

    if let Some(timeout) = request_timeout {
        router = router.layer(from_fn(move |request: Request, next: Next| {
            enforce_timeout(timeout, request, next)
        }));
    }

    let mut router = router
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http())
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024));

    if enable_request_id_headers {
        router = router.layer(from_fn(middleware::set_request_id_header));
    }
    if api_key.is_some() {
        router = router.layer(from_fn(move |request: Request, next: Next| {
            require_api_key(api_key.clone(), request, next)
        }));
    }
    router
}
