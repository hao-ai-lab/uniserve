mod cache;
mod collective_rpc;
mod health;
mod inference;
mod load;
mod lora;
mod metrics;
mod native;
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
use tracing::warn;

use crate::middleware;
use uniserve_server_app::AppState;

/// Environment variable that, when set to a non-empty value, requires every
/// request to present `Authorization: Bearer <value>`.

/// Auth is opt-in: unset or empty disables it, leaving requests fully open for
/// trusted-mesh deployments. The value is matched in constant time to avoid
/// leaking key length / prefix via timing.
const API_KEY_ENV: &str = "UNISERVE_API_KEY";

/// Environment variable controlling a per-request wall-clock timeout, in
/// seconds. Unset, empty, `0`, or unparseable values disable the timeout,
/// leaving requests unbounded.
const REQUEST_TIMEOUT_SECONDS_ENV: &str = "UNISERVE_REQUEST_TIMEOUT_SECONDS";

fn server_dev_mode_enabled() -> bool {
    uniserve_config::env_bool("UNISERVE_SERVER_DEV_MODE")
        .ok()
        .flatten()
        .unwrap_or(false)
}

fn runtime_lora_updating_enabled() -> bool {
    uniserve_config::env_bool("UNISERVE_ALLOW_RUNTIME_LORA_UPDATING")
        .ok()
        .flatten()
        .unwrap_or(false)
}

/// Resolve the configured API key, if any. Returns `None` when auth is disabled
/// (absent or whitespace-only value).
fn configured_api_key() -> Option<String> {
    let raw = std::env::var(API_KEY_ENV).ok()?;
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Parse the configured per-request timeout from a raw environment value.

/// Returns `None` (timeout disabled) for an absent, empty, `0`, or unparseable
/// value; an unparseable value is logged so the misconfiguration is visible.
fn parse_request_timeout(raw: Option<String>) -> Option<Duration> {
    let trimmed = raw?.trim().to_string();
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.parse::<u64>() {
        Ok(0) => None,
        Ok(seconds) => Some(Duration::from_secs(seconds)),
        Err(_) => {
            warn!(
                env = REQUEST_TIMEOUT_SECONDS_ENV,
                value = trimmed,
                "ignoring invalid request timeout; expected a non-negative integer"
            );
            None
        }
    }
}

/// Whether the `Authorization` header carries the expected bearer token.

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

/// Bearer-token auth middleware. Rejects requests whose `Authorization` header
/// does not carry the configured key with `401 Unauthorized`, except for the
/// unauthenticated operational endpoints in [`AUTH_EXEMPT_PATHS`].
async fn require_api_key(api_key: Arc<String>, req: Request, next: Next) -> Response {
    if AUTH_EXEMPT_PATHS.contains(&req.uri().path()) || authorize(req.headers(), api_key.as_str()) {
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
        state,
        server_dev_mode_enabled(),
        runtime_lora_updating_enabled(),
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
        // inference endpoints
        .route("/inference/v1/generate", post(inference::generate))
        // UniServe native generation surface (no OpenAI analog)
        .route("/generate", post(native::generate))
        .route("/v1/images/generations", post(native::images_generations));

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
    let mut router = router.with_state(Arc::clone(&state));

    // Per-request wall-clock timeout (opt-in). Applied closest to the route
    // handlers so it bounds the actual work, not the surrounding bookkeeping
    // layers. Unset/zero leaves requests unbounded, preserving prior behavior.
    if let Some(timeout) = parse_request_timeout(std::env::var(REQUEST_TIMEOUT_SECONDS_ENV).ok()) {
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
    // tracking, or dev/admin route is reached. Unset/empty leaves the surface
    // fully open for trusted-mesh deployments fronted by a TLS+auth gateway.
    if let Some(api_key) = configured_api_key() {
        let api_key = Arc::new(api_key);
        router = router.layer(from_fn(move |req: Request, next: Next| {
            require_api_key(Arc::clone(&api_key), req, next)
        }));
    }

    router
}

#[cfg(test)]
mod middleware_config_tests {
    use std::time::Duration;

    use axum::http::header::AUTHORIZATION;
    use axum::http::{HeaderMap, HeaderValue};

    use super::{authorize, constant_time_eq, parse_request_timeout};

    #[test]
    fn parse_request_timeout_disabled_for_absent_or_zero() {
        assert_eq!(parse_request_timeout(None), None);
        assert_eq!(parse_request_timeout(Some(String::new())), None);
        assert_eq!(parse_request_timeout(Some("   ".to_string())), None);
        assert_eq!(parse_request_timeout(Some("0".to_string())), None);
    }

    #[test]
    fn parse_request_timeout_accepts_positive_seconds() {
        assert_eq!(
            parse_request_timeout(Some("30".to_string())),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_request_timeout(Some(" 120 ".to_string())),
            Some(Duration::from_secs(120))
        );
    }

    #[test]
    fn parse_request_timeout_ignores_invalid() {
        assert_eq!(parse_request_timeout(Some("-1".to_string())), None);
        assert_eq!(parse_request_timeout(Some("abc".to_string())), None);
    }

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
