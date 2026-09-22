//! HTTP request identification, load tracking, and metrics middleware.

mod load;
mod metrics;
mod request_id;

pub(crate) use load::track_server_load;
pub(crate) use metrics::track_http_metrics;
pub(crate) use request_id::set_request_id_header;

use axum::body::{Body, Bytes, HttpBody};
use http_body::{Frame, SizeHint};
use std::pin::Pin;
use std::task::{Context, Poll};

/// Keeps request accounting alive until its response body is dropped.
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
