use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};

use axum::Json;
use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request, State};
use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http_body::{Frame, SizeHint};
use serde_json::json;

use uniserve_server_app::AppState;

/// Endpoints that will be tracked for server load.
///
/// Derived from the Python frontend's actual `@load_aware_call` coverage. This
/// includes alias paths that delegate into decorated handlers, such as
/// `/v1/rerank` and `/v2/rerank`.
const TRACKED_HANDLERS: &[&str] = &[
    "/v1/responses",
    "/v1/responses/{response_id}",
    "/v1/responses/{response_id}/cancel",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/embeddings",
    "/pooling",
    "/classify",
    "/score",
    "/v1/score",
    "/rerank",
    "/v1/rerank",
    "/v2/rerank",
    "/inference/v1/generate",
];

/// `Retry-After` hint (in seconds) advertised when shedding load.
const RETRY_AFTER_SECONDS: &str = "1";

/// Build the 503 load-shedding response returned when the in-flight limit is
/// reached.
fn overloaded_response(limit: u64) -> Response {
    let body = json!({
        "error": {
            "message": format!(
                "Server overloaded: {limit} concurrent requests already in flight. \
                 Retry after a short delay."
            ),
            "type": "service_unavailable",
            "code": "server_overloaded",
        }
    });
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(RETRY_AFTER, HeaderValue::from_static(RETRY_AFTER_SECONDS))],
        Json(body),
    )
        .into_response()
}

/// Track frontend-local in-flight inference requests for the `/load` endpoint.
///
/// When an admission limit is configured, tracked requests are shed with `503
/// Service Unavailable` (plus a `Retry-After` hint) once that many requests are
/// already in flight, so a fixed-capacity engine browns out gracefully instead
/// of accepting unbounded work.
pub(crate) async fn track_server_load(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let handler = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "none", |path| path.as_str());

    if !TRACKED_HANDLERS.contains(&handler) {
        return next.run(req).await;
    }

    // Admission control: shed load before admitting a new request when the
    // configured in-flight limit has been reached. The check-then-increment is
    // intentionally not a single atomic compare-and-set; a brief overshoot of a
    // request or two under contention is acceptable for load shedding and keeps
    // the shared `AppState` counter API unchanged.
    if let Some(limit) = state.max_concurrent_requests()
        && state.server_load() >= limit
    {
        return overloaded_response(limit);
    }

    state.increment_server_load();
    let guard = ServerLoadGuard {
        state: Arc::downgrade(&state),
    };
    let response = next.run(req).await;

    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(LoadTrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A guard that decrements the server load when dropped.
struct ServerLoadGuard {
    state: Weak<AppState>,
}

impl Drop for ServerLoadGuard {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state.decrement_server_load();
        }
    }
}

/// A wrapper around response bodies that tracks server load by holding a
/// `ServerLoadGuard`, which will decrement the load when the body is fully
/// consumed and dropped.
struct LoadTrackedBody {
    inner: Body,
    _guard: ServerLoadGuard,
}

// Simply delegate all `HttpBody` methods to the inner body.
impl HttpBody for LoadTrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overloaded_response_is_503_with_retry_after() {
        let response = overloaded_response(8);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some(RETRY_AFTER_SECONDS)
        );
    }
}
