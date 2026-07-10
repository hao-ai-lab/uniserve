use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
use http_body::{Frame, SizeHint};
use uniserve_observability::{HttpHandlerLabels, HttpRequestLabels, METRICS};

/// Endpoints that will be excluded from HTTP metrics tracking.
///
const EXCLUDED_HANDLERS: &[&str] = &[
    "/metrics",
    "/health",
    "/load",
    "/ping",
    "/version",
    "/server_info",
    // Rust frontend extra:
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/reset_encoder_cache",
    "/collective_rpc",
    "/sleep",
    "/wake_up",
    "/is_sleeping",
];

/// Record API-server HTTP metrics with Python-compatible
/// (`PrometheusFastApiInstrumentator` style) family names and labels.
pub(crate) async fn track_http_metrics(req: Request, next: Next) -> Response {
    // Resolve the handler from a borrowed `&str` first so excluded requests
    // (the bypass path) never allocate the method/handler strings.
    let handler = req
        .extensions()
        .get::<MatchedPath>()
        .map_or("none", |path| path.as_str());

    if EXCLUDED_HANDLERS.contains(&handler) {
        return next.run(req).await;
    }

    // Only allocate the owned label strings for tracked requests. They are
    // moved into the body guard so they outlive the handler return.
    let method = req.method().as_str().to_string();
    let handler = handler.to_string();
    let started_at = Instant::now();

    let response = next.run(req).await;

    // Status is available as soon as the handler returns the response head;
    // duration, however, must be measured at body completion (see below).
    let status = status_group(response.status().as_u16());
    let guard = MetricsGuard {
        started_at,
        method,
        handler,
        status,
    };

    // Wrap the body so the guard's `Drop` fires when the response body is
    // fully sent. For streaming (SSE) responses the handler returns the body
    // immediately, so observing at handler-return would record a near-zero
    // duration; deferring to body-end captures the true request duration.
    // Mirrors the `LoadTrackedBody` pattern in `load.rs`.
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(MetricsTrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// Records HTTP metrics when dropped, i.e. when the response body has been
/// fully sent (or aborted). Carries the labels and start time captured at
/// request entry.
struct MetricsGuard {
    started_at: Instant,
    method: String,
    handler: String,
    status: &'static str,
}

impl Drop for MetricsGuard {
    fn drop(&mut self) {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        let metrics = &METRICS.api_server;

        metrics
            .http_requests
            .get_or_create(&HttpRequestLabels {
                method: self.method.clone(),
                status: self.status,
                handler: self.handler.clone(),
            })
            .inc();

        metrics
            .http_request_duration_seconds
            .get_or_create(&HttpHandlerLabels {
                method: self.method.clone(),
                handler: self.handler.clone(),
            })
            .observe(elapsed);

        metrics.http_request_duration_highr_seconds.observe(elapsed);
    }
}

/// A wrapper around response bodies that records HTTP metrics by holding a
/// `MetricsGuard`, which observes the request duration when the body is fully
/// consumed and dropped.
struct MetricsTrackedBody {
    inner: Body,
    _guard: MetricsGuard,
}

// Simply delegate all `HttpBody` methods to the inner body.
impl HttpBody for MetricsTrackedBody {
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

fn status_group(status: u16) -> &'static str {
    match status / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "unknown",
    }
}
