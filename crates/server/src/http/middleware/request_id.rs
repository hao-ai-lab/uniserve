use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::header::HeaderName;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Maximum length of a client-supplied request id we are willing to echo back.
///
/// A generated `uuid4` simple form is 32 chars; legitimate correlation ids are
/// short. Capping the length bounds how much attacker-controlled data we
/// reflect into the response and downstream logs.
const MAX_REQUEST_ID_LEN: usize = 128;

/// Echo the request's `X-Request-Id` on the response, or generate a fresh
/// `uuid4` hex if the request did not provide a usable one.
///
/// The incoming value is attacker-controlled, so we only echo it when it is a
/// short, non-empty token of safe visible-ASCII characters (`A-Za-z0-9` plus
/// the separators `-`, `_`, `.`). Anything outside that — empty, over-long, or
/// containing opaque/non-visible bytes that a `HeaderValue` is otherwise
/// permitted to hold — is discarded in favour of a freshly generated id rather
/// than reflected unvalidated.
pub(crate) async fn set_request_id_header(req: Request, next: Next) -> Response {
    let incoming = req
        .headers()
        .get(&X_REQUEST_ID)
        .filter(|value| is_safe_request_id(value))
        .cloned();
    let mut response = next.run(req).await;
    let value = incoming.unwrap_or_else(generate_request_id);
    response.headers_mut().insert(X_REQUEST_ID, value);
    response
}

/// Returns `true` if a client-supplied `X-Request-Id` is safe to echo verbatim.
fn is_safe_request_id(value: &HeaderValue) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_REQUEST_ID_LEN
        && bytes
            .iter()
            .all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Generate a fresh `uuid4` hex request id.
fn generate_request_id() -> HeaderValue {
    HeaderValue::from_str(&Uuid::new_v4().simple().to_string())
        .unwrap_or_else(|_| HeaderValue::from_static("00000000000000000000000000000000"))
}

#[cfg(test)]
mod tests {
    use super::{MAX_REQUEST_ID_LEN, generate_request_id, is_safe_request_id};
    use axum::http::HeaderValue;

    #[test]
    fn accepts_uuid_and_common_correlation_ids() {
        assert!(is_safe_request_id(&HeaderValue::from_static(
            "0123456789abcdef0123456789abcdef"
        )));
        assert!(is_safe_request_id(&HeaderValue::from_static(
            "req-123_abc.def"
        )));
        // A freshly generated id must itself pass validation.
        assert!(is_safe_request_id(&generate_request_id()));
    }

    #[test]
    fn rejects_empty_overlong_and_unsafe_values() {
        assert!(!is_safe_request_id(&HeaderValue::from_static("")));

        let too_long = "a".repeat(MAX_REQUEST_ID_LEN + 1);
        assert!(!is_safe_request_id(
            &HeaderValue::from_str(&too_long).unwrap()
        ));

        // Spaces and other visible-but-unsafe chars are rejected.
        assert!(!is_safe_request_id(&HeaderValue::from_static("has space")));
        assert!(!is_safe_request_id(&HeaderValue::from_static("a/b")));
        assert!(!is_safe_request_id(&HeaderValue::from_static("<script>")));

        // Opaque non-visible bytes a HeaderValue may legally hold are rejected.
        assert!(!is_safe_request_id(
            &HeaderValue::from_bytes(&[0x80, 0x81]).unwrap()
        ));
    }
}
