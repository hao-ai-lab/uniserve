//! Process-wide Prometheus registry and UniServe metric families.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

mod api_server;
mod request;
mod scheduler;
mod serving;

pub use api_server::{ApiServerMetrics, HttpHandlerLabels, HttpRequestLabels};
pub use request::{FinishedReasonLabels, PromptTokenSourceLabels, RequestMetrics};
pub use scheduler::{
    EngineBackendLabels, EngineComponentLabels, EngineDomainKindLabels, EngineDomainLabels,
    EngineLabels, EngineModeLabels, EnginePathLabels, SchedulerMetrics, WaitingReasonLabels,
};
pub use serving::{ServingMetrics, ServingRequestLabels};

// `prometheus-client` appends `_total` while encoding counters, so registration
// names use the base metric name.
/// Unsigned Prometheus counter used by metric families.
pub type U64Counter = Counter<u64, AtomicU64>;
/// Unsigned Prometheus gauge used by metric families.
pub type U64Gauge = Gauge<u64, AtomicU64>;
/// Floating-point Prometheus gauge used by metric families.
pub type F64Gauge = Gauge<f64, AtomicU64>;
/// Histogram family keyed by engine identity.
pub(crate) type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;

/// Shared Prometheus registry for frontend metrics.
pub struct Metrics {
    registry: Registry,
    /// Scheduler, cache, and worker execution metric families.
    pub scheduler: SchedulerMetrics,
    /// Request latency, token, and completion metric families.
    pub request: RequestMetrics,
    /// HTTP transport metric families.
    pub api_server: ApiServerMetrics,
    /// Serving-runtime lifecycle metric families.
    pub serving: ServingMetrics,
}

impl Metrics {
    /// Constructs a metrics registry with every UniServe family registered.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let scheduler = SchedulerMetrics::register(&mut registry);
        let request = RequestMetrics::register(&mut registry);
        let api_server = ApiServerMetrics::register(&mut registry);
        let serving = ServingMetrics::register(&mut registry);

        Self {
            registry,
            scheduler,
            request,
            api_server,
            serving,
        }
    }

    /// Renders the current metrics registry into Prometheus/OpenMetrics text
    /// format.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.registry)?;
        Ok(output)
    }

    /// Returns the registry owned by this metrics object.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for Metrics {
    /// Constructs the fully registered default metrics collection.
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global metrics registry shared by the frontend crates.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
