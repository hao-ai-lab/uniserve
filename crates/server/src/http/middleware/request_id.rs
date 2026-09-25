//! Request-ID resolution, propagation, and the `X-Request-Id` response header.
//!
//! [`resolve_request_id`] decides each request's ID exactly once, before the
//! handler runs. Handlers read it as `Extension<RequestId>` to build their
//! serving-runtime request ID, and the same value is echoed in the
//! `X-Request-Id` response header when that header is enabled, so the header
//! always names the request the server ran.

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::header::HeaderName;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Maximum accepted length, in bytes, of a client-supplied request ID.
///
/// A generated ID (a `uuid4` in simple form) is 32 characters and legitimate
/// correlation IDs are short. The cap bounds how much client-controlled data
/// enters the runtime request ID, logs, and the echoed response header.
const MAX_REQUEST_ID_LEN: usize = 128;

/// The request's base ID, inserted into the request extensions by
/// [`resolve_request_id`].
///
/// Route handlers prefix it per API (`chatcmpl-`, `img-`, `vid-`) and use the
/// result as the serving-runtime request ID, which the engine client registers
/// as the request's external ID. Reusing the ID of a request that is still
/// live therefore fails that registration as a duplicate.
#[derive(Clone, Debug)]
pub(crate) struct RequestId(pub(crate) String);

/// Resolves the request's ID and, when `echo` is set, returns it in the
/// `X-Request-Id` response header.
///
/// The client's `X-Request-Id` is used when it is acceptable (see
/// `is_acceptable_request_id`); otherwise a fresh ID is generated. The ID is
/// inserted as [`RequestId`] before the inner service runs. `routes::build_router`
/// installs this layer on every route, with `echo` set from
/// `AppState::enable_request_id_headers`; a response that never reached a
/// handler, such as a load-shedding `503`, still carries the header.
pub(crate) async fn resolve_request_id(echo: bool, mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|value| value.to_str().ok())
        .filter(|id| is_acceptable_request_id(id))
        .map_or_else(generate_request_id, ToOwned::to_owned);
    request.extensions_mut().insert(RequestId(id.clone()));

    let mut response = next.run(request).await;

    // An accepted or generated ID is visible ASCII without whitespace, which
    // is always a valid header value.
    if echo && let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(X_REQUEST_ID, value);
    }
    response
}

/// Returns whether a client-supplied request ID (already visible ASCII) is
/// used as the request's ID.
///
/// Rejects empty and over-long values and any value containing whitespace or
/// a control character, which would otherwise allow log injection.
fn is_acceptable_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_REQUEST_ID_LEN
        && id.chars().all(|c| !c.is_control() && !c.is_whitespace())
}

/// Generates a fresh request ID: a full (untruncated) `uuid4` in simple form,
/// so IDs stay collision-resistant under high request volume.
fn generate_request_id() -> String {
    Uuid::new_v4().simple().to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::http::test_support::{SERVED_MODEL, post_json, send, sim_state};
    use crate::profile::ModelParameters;

    /// Sends a two-token chat completion with `request_id` as its
    /// `X-Request-Id` and returns the response header and the body's
    /// completion `id`.
    async fn chat(request_id: Option<&str>) -> (String, String) {
        let state = sim_state(ModelParameters::Qwen3).with_request_id_headers(true);
        let router = crate::http::build_router(Arc::new(state));
        let mut request = post_json(
            "/v1/chat/completions",
            &serde_json::json!({
                "model": SERVED_MODEL,
                "messages": [{"role": "user", "content": "prompt"}],
                "max_completion_tokens": 2,
                "temperature": 0,
                "allowed_token_ids": [u32::from(b'a')],
            }),
        );
        if let Some(request_id) = request_id {
            request
                .headers_mut()
                .insert("x-request-id", request_id.parse().unwrap());
        }

        let (status, headers, body) = send(&router, request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let header = headers["x-request-id"].to_str().unwrap().to_owned();
        (header, body["id"].as_str().unwrap().to_owned())
    }

    /// The response header names the request the server ran: the completion
    /// `id` is the header value behind the `chatcmpl-` prefix.
    #[tokio::test]
    async fn the_response_header_carries_the_runtime_request_id() {
        let (header, id) = chat(None).await;
        assert_eq!(id, format!("chatcmpl-{header}"));

        let (header, id) = chat(Some("client/42")).await;
        assert_eq!(header, "client/42");
        assert_eq!(id, "chatcmpl-client/42");
    }

    /// A client ID the server refuses (empty, containing whitespace, or longer
    /// than 128 bytes) is replaced by one generated ID, used for both the
    /// header and the request.
    #[tokio::test]
    async fn a_refused_client_id_is_replaced_by_one_generated_id() {
        for refused in [String::new(), "has space".to_owned(), "a".repeat(129)] {
            let (header, id) = chat(Some(&refused)).await;
            assert_ne!(header, refused);
            assert_eq!(id, format!("chatcmpl-{header}"));
        }
    }
}
