use axum::http::HeaderMap;
use thiserror_ext::AsReport;
pub(crate) use uniserve_protocol_adapters::openai::ResolvedRequestContext;
use uuid::Uuid;

use crate::http::error::ApiError;

/// Return the current Unix timestamp in seconds for OpenAI response objects.
///
/// delegates to the single shared epoch helper (the integer-seconds
/// companion) so this matches the fractional-second timestamps used elsewhere.
pub(crate) fn unix_timestamp() -> u64 {
    uniserve_core::now_unix_secs_u64()
}

/// Construct an API error when engine status is unavailable.
pub(crate) fn engine_status_error(error: impl AsReport) -> ApiError {
    ApiError::server_error(format!(
        "engine status is unavailable: {}",
        error.as_report()
    ))
}

/// Extract the external request ID and tracing metadata from HTTP headers.
pub(crate) fn resolve_request_context(headers: &HeaderMap) -> ResolvedRequestContext {
    // Extract request id from header.
    let request_id_header = headers
        .get("X-Request-Id")
        .and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header);
    let trace_context = ["traceparent", "tracestate", "baggage"]
        .into_iter()
        .filter_map(|name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();

    ResolvedRequestContext {
        request_id,
        trace_context,
    }
}

/// Maximum accepted length (in bytes) of a client-supplied request ID. IDs
/// longer than this are rejected and a fresh ID is generated instead, so a
/// client cannot use the correlation key as an unbounded-memory or log-injection
/// vector.
const MAX_REQUEST_ID_LEN: usize = 128;

/// Resolve the base external request ID before API-specific prefixes such as
/// `chatcmpl-`.
///
/// A client-supplied ID (from the `X-Request-Id` header or the request body) is
/// only honored when it passes [`is_acceptable_request_id`]; otherwise it is
/// ignored and a fresh server-generated ID is returned. This prevents an
/// untrusted, unsanitized value from flowing through to the engine as a
/// correlation key.
pub(crate) fn resolve_base_request_id(request_id_header: Option<&str>) -> String {
    request_id_header
        .filter(|id| is_acceptable_request_id(id))
        .map(ToOwned::to_owned)
        .unwrap_or_else(generate_request_id)
}

/// Whether a client-supplied request ID is safe to use verbatim as an engine
/// correlation key. Rejects empty, over-long, and non-printable values (control
/// characters, including newlines, would otherwise enable log injection).
fn is_acceptable_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id.chars().all(|c| !c.is_control() && !c.is_whitespace())
}

/// Generate a fresh server-side request ID. Uses a full (untruncated) UUIDv4 so
/// the IDs stay collision-resistant under high request volume.
fn generate_request_id() -> String {
    Uuid::new_v4().simple().to_string()
}
