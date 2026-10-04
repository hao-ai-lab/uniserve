//! Route assembly, authentication, request timeouts, and request-size
//! enforcement.

mod health;
mod metrics;
pub(crate) mod openai;
mod systemone;
mod version;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Request};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{Next, from_fn, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::http::middleware;

/// Authorizes an HTTP request against the configured API key.
///
/// Accepts `Authorization: Bearer <key>` with the scheme spelled `Bearer` or
/// `bearer` and surrounding whitespace trimmed from the token. A missing
/// header, a non-visible-ASCII header value, or any other scheme is refused.
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

/// Compares byte strings without exiting early on the first differing byte.
///
/// Inputs of different lengths return `false` immediately, so timing can
/// reveal the expected key's length but not its contents.
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

/// Builds an unauthorized HTTP response.
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

/// Builds a request-timeout HTTP response.
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

/// Paths served without an API key, compared exactly against the request URI
/// path. `/version` and `/v1/models` still require the key.
const AUTH_EXEMPT_PATHS: &[&str] = &["/health", "/metrics"];

/// Validates the configured API key for an HTTP request.
///
/// Requests to `AUTH_EXEMPT_PATHS`, and every request when no key is
/// configured, pass through; others without a matching bearer token get
/// `401 Unauthorized`.
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

/// Bounds the time until the inner service returns a response head, answering
/// `504 Gateway Timeout` when it expires.
///
/// The response body is not covered: a streamed chat completion and a video
/// content download are bounded only until their headers are returned, not
/// while their bodies are sent. Handlers that generate before responding, such
/// as non-streaming chat completions and `/v1/videos/sync`, are bounded through
/// generation. On expiry the handler future is dropped.
async fn enforce_timeout(timeout: Duration, request: Request, next: Next) -> Response {
    match tokio::time::timeout(timeout, next.run(request)).await {
        Ok(response) => response,
        Err(_) => timeout_response(timeout),
    }
}

/// Builds the complete configured public router.
///
/// Each `layer` call wraps everything added before it, so a request passes the
/// layers in this order: API-key check (when a key is configured), request-ID
/// resolution (which also sets the `X-Request-Id` response header when
/// enabled), body-size limit, tracing, HTTP metrics, load tracking, request
/// timeout (when configured), then the handler. Consequences of this order:
/// `401` responses carry no `X-Request-Id` and are not counted in HTTP
/// metrics, while `503` load-shedding and `504` timeout responses are.
pub fn build_router(state: Arc<AppState>) -> Router {
    let enable_request_id_headers = state.enable_request_id_headers();
    let request_timeout = state.request_timeout();
    let api_key = state.api_key().map(|key| Arc::new(key.to_string()));
    let video_body_limit = state.video_body_limit();

    let mut router = Router::new()
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/version", get(version::version))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/images/generations", post(openai::images_generations))
        // The video submission routes take condition media inline, so their
        // body limit follows the configured media total.
        .route(
            "/v1/videos/sync",
            post(openai::videos_sync).layer(DefaultBodyLimit::max(video_body_limit)),
        )
        .route("/v1/capabilities", get(openai::capabilities))
        .route("/v1/systemone", post(systemone::systemone))
        .route(
            "/v1/videos",
            get(openai::videos_list).post(openai::videos_create).layer(
                // Applies to the listing too, which reads no body.
                DefaultBodyLimit::max(video_body_limit),
            ),
        )
        .route(
            "/v1/videos/{id}",
            get(openai::videos_get).delete(openai::videos_delete),
        )
        .route("/v1/videos/{id}/content", get(openai::videos_content))
        .with_state(Arc::clone(&state));

    if let Some(timeout) = request_timeout {
        router = router.layer(from_fn(move |request: Request, next: Next| {
            enforce_timeout(timeout, request, next)
        }));
    }

    // The request-ID layer is unconditional: every generation handler reads
    // the `RequestId` it resolves.
    let mut router = router
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http())
        // Sets the limit that axum's body extractors (`Json`, `Multipart`)
        // apply; it does not cap bodies read by other means.
        .layer(DefaultBodyLimit::max(crate::http::BODY_LIMIT))
        .layer(from_fn(move |request: Request, next: Next| {
            middleware::resolve_request_id(enable_request_id_headers, request, next)
        }));

    if api_key.is_some() {
        router = router.layer(from_fn(move |request: Request, next: Next| {
            require_api_key(api_key.clone(), request, next)
        }));
    }
    router
}
