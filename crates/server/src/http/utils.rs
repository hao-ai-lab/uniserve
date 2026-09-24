//! Transport-level request metadata and timestamp helpers.

use axum::http::HeaderMap;
use uuid::Uuid;

/// Returns the current Unix timestamp in whole seconds for API response objects.
pub(crate) fn unix_timestamp() -> u64 {
    uniserve_core::now_unix_secs_u64()
}

/// Resolves the caller's request ID from HTTP headers, generating one when absent or invalid.
///
/// Route handlers prefix the result per API (`chatcmpl-`, `img-`, `vid-`) and
/// use it as the serving-runtime request ID, which the engine client registers
/// as the request's external ID. Reusing the ID of a request that is still live
/// therefore fails that registration as a duplicate. A header value that is not
/// visible ASCII is treated as absent.
///
/// This check is independent of the `X-Request-Id` response header, which the
/// request-ID middleware validates with a stricter character set.
pub(crate) fn resolve_request_id(headers: &HeaderMap) -> String {
    let request_id_header = headers
        .get("X-Request-Id")
        .and_then(|value| value.to_str().ok());
    resolve_base_request_id(request_id_header)
}

/// Maximum accepted length (in bytes) of a client-supplied request ID. IDs
/// longer than this are rejected and a fresh ID is generated instead, so a
/// client cannot use the correlation key as an unbounded-storage or
/// log-injection vector.
const MAX_REQUEST_ID_LEN: usize = 128;

/// Resolves the base external request ID before API-specific prefixes such as
/// `chatcmpl-`.
///
/// A client-supplied ID from the `X-Request-Id` header is only honored when it
/// passes [`is_acceptable_request_id`]; otherwise it is ignored and a fresh
/// server-generated ID is returned. This prevents an untrusted, unsanitized
/// value from flowing through to the engine as a correlation key.
pub(crate) fn resolve_base_request_id(request_id_header: Option<&str>) -> String {
    request_id_header
        .filter(|id| is_acceptable_request_id(id))
        .map(ToOwned::to_owned)
        .unwrap_or_else(generate_request_id)
}

/// Returns whether a client-supplied request ID is safe to use as an engine
/// correlation key. Rejects empty and over-long values and any value containing
/// a control or whitespace character (control characters, including newlines,
/// would otherwise enable log injection).
fn is_acceptable_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id.chars().all(|c| !c.is_control() && !c.is_whitespace())
}

/// Generates a fresh server-side request ID. Uses a full (untruncated) UUIDv4 so
/// the IDs stay collision-resistant under high request volume.
fn generate_request_id() -> String {
    Uuid::new_v4().simple().to_string()
}
