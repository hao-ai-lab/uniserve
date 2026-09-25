//! HTTP request identification, load tracking, and metrics middleware.
//!
//! `routes::build_router` installs these as `axum::middleware` function layers
//! around every route; the request-ID layer sets its response header only when
//! that header is enabled. Load tracking and metrics must observe the whole
//! response, including a streamed body that outlives the handler, so both wrap
//! the response body in [`GuardedBody`] and do their accounting when that body
//! is dropped.

mod load;
mod metrics;
mod request_id;

pub(crate) use load::track_server_load;
pub(crate) use metrics::track_http_metrics;
pub(crate) use request_id::{RequestId, resolve_request_id};

use axum::body::{Body, Bytes, HttpBody};
use http_body::{Frame, SizeHint};
use std::pin::Pin;
use std::task::{Context, Poll};

/// Keeps request accounting alive until its response body is dropped.
///
/// The body forwards every frame and hint to `inner` unchanged; `_guard` exists
/// only for its `Drop`, which runs when the server finishes sending the body or
/// drops it because the connection closed. For an SSE response that is the end
/// of the stream, not the moment the handler returned.
struct GuardedBody<G> {
    inner: Body,
    _guard: G,
}

impl<G: Unpin> HttpBody for GuardedBody<G> {
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
