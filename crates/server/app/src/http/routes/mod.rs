mod cache;
mod collective_rpc;
mod health;
mod inference;
mod load;
mod lora;
mod metrics;
pub(crate) mod openai;
mod server_info;
mod sleep;
mod version;

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{Next, from_fn, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::AppState;
use crate::http::middleware;

/// Whether the `Authorization` header carries the expected bearer token.
///
/// Matching is constant-time over the byte contents so a caller cannot infer
/// the key from response-time differences.
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

/// Length-aware constant-time byte comparison (no early return on mismatch).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Build the `401 Unauthorized` response returned when auth is enabled and the
/// bearer token is missing or wrong.
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

/// Build the `504 Gateway Timeout` response returned when a request exceeds the
/// configured per-request timeout.
fn timeout_response(timeout: Duration) -> Response {
    let body = json!({
        "error": {
            "message": format!(
                "Request exceeded the {}s server timeout.",
                timeout.as_secs()
            ),
            "type": "timeout",
            "code": "request_timeout",
        }
    });
    (StatusCode::GATEWAY_TIMEOUT, Json(body)).into_response()
}

/// Operational endpoints that stay reachable without the API key even when
/// auth is enabled, so liveness probes and Prometheus scraping keep working.
/// These expose no request data and no dev/admin actions.
const AUTH_EXEMPT_PATHS: &[&str] = &["/health", "/metrics"];

/// Sensitive management routes protected by the admin API key when configured.
const ADMIN_AUTH_PATHS: &[&str] = &[
    "/collective_rpc",
    "/is_sleeping",
    "/reset_encoder_cache",
    "/reset_mm_cache",
    "/reset_prefix_cache",
    "/server_info",
    "/sleep",
    "/v1/load_lora_adapter",
    "/v1/unload_lora_adapter",
    "/wake_up",
];

/// Bearer-token auth middleware. Rejects requests whose `Authorization` header
/// does not carry the configured key with `401 Unauthorized`. Operational
/// endpoints in [`AUTH_EXEMPT_PATHS`] stay open for probes.
async fn require_api_key(
    api_key: Option<Arc<String>>,
    admin_api_key: Option<Arc<String>>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    if AUTH_EXEMPT_PATHS.contains(&path) {
        return next.run(req).await;
    }

    let expected = if ADMIN_AUTH_PATHS.contains(&path) {
        admin_api_key.as_deref().or(api_key.as_deref())
    } else {
        api_key.as_deref()
    };

    if expected.is_none_or(|expected| authorize(req.headers(), expected)) {
        next.run(req).await
    } else {
        unauthorized_response()
    }
}

/// Per-request timeout middleware. Aborts handlers that run longer than
/// `timeout` and returns `504 Gateway Timeout`.
async fn enforce_timeout(timeout: Duration, req: Request, next: Next) -> Response {
    match tokio::time::timeout(timeout, next.run(req)).await {
        Ok(response) => response,
        Err(_elapsed) => timeout_response(timeout),
    }
}

/// Build the minimal OpenAI-compatible router for one configured model.
pub fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_options(
        Arc::clone(&state),
        state.server_dev_mode(),
        state.runtime_lora_updating_enabled(),
    )
}

#[cfg(test)]
fn build_router_with_dev_mode(state: Arc<AppState>, dev_mode_enabled: bool) -> Router {
    build_router_with_dev_mode_and_lora(state, dev_mode_enabled, false)
}

#[cfg(test)]
fn build_router_with_dev_mode_and_lora(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
    runtime_lora_updating_enabled: bool,
) -> Router {
    build_router_with_options(state, dev_mode_enabled, runtime_lora_updating_enabled)
}

fn build_router_with_options(
    state: Arc<AppState>,
    dev_mode_enabled: bool,
    runtime_lora_updating_enabled: bool,
) -> Router {
    let mut router = Router::new()
        // Health & monitoring
        .route("/health", get(health::health))
        .route("/metrics", get(metrics::scrape))
        .route("/load", get(load::load))
        .route("/version", get(version::version))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(openai::list_models))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route(
            "/v1/chat/completions/plan",
            post(openai::chat_completion_plan),
        )
        // inference endpoints
        .route("/inference/v1/generate", post(inference::generate))
        .route(
            "/inference/v1/native/generate",
            post(inference::native_generate),
        )
        .route("/v1/images/generations", post(openai::images_generations))
        .route(
            "/v1/images/generations/plan",
            post(openai::image_generation_plan),
        );

    if runtime_lora_updating_enabled {
        router = router
            .route("/v1/load_lora_adapter", post(lora::load_lora_adapter))
            .route("/v1/unload_lora_adapter", post(lora::unload_lora_adapter));
    }

    if dev_mode_enabled {
        // Development-only
        router = router
            .route("/reset_prefix_cache", post(cache::reset_prefix_cache))
            .route("/reset_mm_cache", post(cache::reset_mm_cache))
            .route("/reset_encoder_cache", post(cache::reset_encoder_cache))
            .route("/collective_rpc", post(collective_rpc::collective_rpc))
            .route("/sleep", post(sleep::sleep))
            .route("/wake_up", post(sleep::wake_up))
            .route("/is_sleeping", get(sleep::is_sleeping))
            .route("/server_info", get(server_info::server_info))
    }

    let enable_request_id_headers = state.enable_request_id_headers();
    let request_timeout = state.request_timeout();
    let api_key = state.api_key().map(|key| Arc::new(key.to_string()));
    let admin_api_key = state.admin_api_key().map(|key| Arc::new(key.to_string()));
    let mut router = router.with_state(Arc::clone(&state));

    // Per-request wall-clock timeout (opt-in). Applied closest to the route
    // handlers so it bounds the actual work, not the surrounding bookkeeping
    // layers.
    if let Some(timeout) = request_timeout {
        router = router.layer(from_fn(move |req: Request, next: Next| {
            enforce_timeout(timeout, req, next)
        }));
    }

    let mut router = router
        .layer(from_fn_with_state(state, middleware::track_server_load))
        .layer(from_fn(middleware::track_http_metrics))
        .layer(TraceLayer::new_for_http())
        // Native image-context requests carry base64 photos in the body;
        // raise the default 2 MB cap so real images fit.
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024));

    if enable_request_id_headers {
        router = router.layer(from_fn(middleware::set_request_id_header));
    }

    // Bearer-token auth (opt-in). Applied as the outermost layer so an
    // unauthenticated request is rejected before any handler, body read, load
    // tracking, or dev/admin route is reached.
    if api_key.is_some() || admin_api_key.is_some() {
        router = router.layer(from_fn(move |req: Request, next: Next| {
            require_api_key(api_key.clone(), admin_api_key.clone(), req, next)
        }));
    }

    router
}

#[cfg(test)]
mod middleware_config_tests {
    use axum::http::header::AUTHORIZATION;
    use axum::http::{HeaderMap, HeaderValue};

    use super::{authorize, constant_time_eq};

    #[test]
    fn constant_time_eq_matches_byte_equality() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    fn headers_with_auth(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_str(value).unwrap());
        headers
    }

    #[test]
    fn authorize_accepts_matching_bearer_token() {
        assert!(authorize(&headers_with_auth("Bearer s3cr3t"), "s3cr3t"));
        // Scheme is case-insensitive; surrounding whitespace is trimmed.
        assert!(authorize(&headers_with_auth("bearer s3cr3t"), "s3cr3t"));
        assert!(authorize(&headers_with_auth("Bearer  s3cr3t "), "s3cr3t"));
    }

    #[test]
    fn authorize_rejects_missing_or_wrong_token() {
        assert!(!authorize(&HeaderMap::new(), "s3cr3t"));
        assert!(!authorize(&headers_with_auth("Bearer wrong"), "s3cr3t"));
        assert!(!authorize(&headers_with_auth("s3cr3t"), "s3cr3t"));
        assert!(!authorize(&headers_with_auth("Basic s3cr3t"), "s3cr3t"));
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod http_client_tests;
