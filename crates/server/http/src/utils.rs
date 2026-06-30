use axum::http::HeaderMap;
use thiserror_ext::AsReport;
pub(crate) use uniserve_openai_api::ResolvedRequestContext;
use uuid::Uuid;

use crate::error::ApiError;

/// Return the current Unix timestamp in seconds for OpenAI response objects.

/// delegates to the single shared epoch helper (the integer-seconds
/// companion) so this matches the fractional-second timestamps used elsewhere.
pub(crate) fn unix_timestamp() -> u64 {
    uniserve_core::now_unix_secs_u64()
}

/// Construct an API error for a failed utility call to the engine core.
pub(crate) fn utility_call_error(method: &str, error: impl AsReport) -> ApiError {
    ApiError::server_error(format!("failed to call {method}: {}", error.as_report()))
}

/// Extract common request metadata from HTTP headers: the external request ID
/// and the optional data-parallel rank used for engine routing.
pub(crate) fn resolve_request_context(
    headers: &HeaderMap,
    request_id: Option<&str>,
) -> ResolvedRequestContext {
    // `None` when the header is absent. A present-but-unparseable value is
    // logged at WARN (rather than silently swallowed) before falling back to
    // `None` so an operator can spot a misconfigured client.
    let data_parallel_rank = headers
        .get("X-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| match s.parse::<u32>() {
            Ok(rank) => Some(rank),
            Err(error) => {
                tracing::warn!(
                    value = s,
                    %error,
                    "ignoring unparseable X-data-parallel-rank header"
                );
                None
            }
        });

    // Extract request id from header.
    let request_id_header = headers
        .get("X-Request-Id")
        .and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header, request_id);

    ResolvedRequestContext {
        request_id,
        data_parallel_rank,
    }
}

/// Maximum accepted length (in bytes) of a client-supplied request ID. IDs
/// longer than this are rejected and a fresh ID is generated instead, so a
/// client cannot use the correlation key as an unbounded-memory or log-injection
/// vector.
const MAX_REQUEST_ID_LEN: usize = 128;

/// Resolve the base external request ID before API-specific prefixes such as
/// `chatcmpl-`.

/// A client-supplied ID (from the `X-Request-Id` header or the request body) is
/// only honored when it passes [`is_acceptable_request_id`]; otherwise it is
/// ignored and a fresh server-generated ID is returned. This prevents an
/// untrusted, unsanitized value from flowing through to the engine as a
/// correlation key.
pub(crate) fn resolve_base_request_id(
    request_id_header: Option<&str>,
    request_id: Option<&str>,
) -> String {
    request_id_header
        .or(request_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_well_formed_client_id() {
        assert_eq!(
            resolve_base_request_id(Some("req-abc123"), None),
            "req-abc123"
        );
        // Header takes precedence over the body-supplied id.
        assert_eq!(
            resolve_base_request_id(Some("from-header"), Some("from-body")),
            "from-header"
        );
        // Falls back to the body-supplied id when the header is absent.
        assert_eq!(
            resolve_base_request_id(None, Some("from-body")),
            "from-body"
        );
    }

    #[test]
    fn rejects_empty_overlong_and_control_chars() {
        // Empty supplied id -> generated fallback (not the empty string).
        assert!(!resolve_base_request_id(Some(""), None).is_empty());
        assert!(is_acceptable_request_id("ok-id"));
        assert!(!is_acceptable_request_id(""));
        // Control characters (newline) would otherwise enable log injection.
        assert!(!is_acceptable_request_id("inject\nme"));
        assert!(!is_acceptable_request_id("space here"));
        // Over-long ids are rejected.
        let too_long = "a".repeat(MAX_REQUEST_ID_LEN + 1);
        assert!(!is_acceptable_request_id(&too_long));
        assert!(is_acceptable_request_id(&"a".repeat(MAX_REQUEST_ID_LEN)));
    }

    #[test]
    fn generated_id_is_full_uuid_not_truncated() {
        let id = generate_request_id();
        // A simple UUIDv4 renders as 32 hex characters; the previous 8-char
        // truncation is what this guards against (collision risk at scale).
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn rejected_client_id_falls_back_to_generated() {
        let id = resolve_base_request_id(Some("bad\nid"), None);
        assert_eq!(id.len(), 32);
    }
}
