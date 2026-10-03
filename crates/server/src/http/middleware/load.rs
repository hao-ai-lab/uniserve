//! Middleware that tracks active requests and streaming response bodies.
//!
//! The in-flight count lives in `AppState` (`server_load`). Only the generation
//! routes in `TRACKED_HANDLERS` are counted and subject to the optional
//! `max_concurrent_requests` limit. The video routes are not: both video
//! submission routes are bounded by the job slots in
//! `crate::video_jobs::VideoJobs` instead.

use std::sync::{Arc, Weak};

use axum::Json;
use axum::body::Body;

use super::GuardedBody;
use axum::extract::{MatchedPath, Request, State};
use axum::http::header::RETRY_AFTER;
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::AppState;

/// Generation endpoints counted as in-flight server work.
///
/// Entries are matched against the route template (`MatchedPath`), so they must
/// equal the paths registered in `routes::build_router`.
const TRACKED_HANDLERS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/images/generations",
    SYSTEMONE_PATH,
];

/// The System One readout route, whose errors follow FastAPI's
/// `{"detail": ...}` shape rather than OpenAI's.
const SYSTEMONE_PATH: &str = "/v1/systemone";

/// `Retry-After` hint (in seconds) advertised when shedding load.
const RETRY_AFTER_SECONDS: &str = "1";

/// Builds the 503 load-shedding response returned when the in-flight limit is
/// reached, in the error shape of the route `handler` matched.
fn overloaded_response(limit: u64, handler: &str) -> Response {
    let message = format!(
        "Server overloaded: {limit} concurrent requests already in flight. \
         Retry after a short delay."
    );
    let body = if handler == SYSTEMONE_PATH {
        json!({ "detail": message })
    } else {
        json!({
            "error": {
                "message": message,
                "type": "service_unavailable",
                "code": "server_overloaded",
            }
        })
    };
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(RETRY_AFTER, HeaderValue::from_static(RETRY_AFTER_SECONDS))],
        Json(body),
    )
        .into_response()
}

/// Tracks frontend-local in-flight inference requests. When an admission limit
/// is configured, tracked requests are shed with `503 Service Unavailable`
/// (plus a `Retry-After` hint) once that many requests are already in flight,
/// so a fixed-capacity engine browns out gracefully instead of accepting
/// unbounded work.
///
/// A request counts as in flight from admission until its response body is
/// dropped, so a streamed chat completion holds its slot for the whole stream.
/// Requests to other routes pass through uncounted.
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

    // The limit is a soft bound: the load check and the increment below are
    // separate relaxed operations rather than one compare-and-set, so requests
    // racing past the check can briefly push the count above the limit. That
    // brief overshoot is accepted for load shedding.
    if let Some(limit) = state.max_concurrent_requests()
        && state.server_load() >= limit
    {
        return overloaded_response(limit, handler);
    }

    // The guard is created right after the increment so that the decrement
    // also runs if this future is dropped before the handler returns.
    state.increment_server_load();
    let guard = ServerLoadGuard {
        state: Arc::downgrade(&state),
    };
    let response = next.run(req).await;

    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(GuardedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A guard that decrements the server load when dropped.
///
/// It holds a `Weak` reference, so an undelivered response body does not keep
/// `AppState` alive or count toward the strong references `AppState::shutdown`
/// waits on. Once the state is gone the decrement is skipped.
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
