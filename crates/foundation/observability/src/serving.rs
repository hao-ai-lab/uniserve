//! Serving-runtime request lifecycle metrics.
//!
//! The server's OpenMetrics route copies the serving runtime's lifecycle
//! snapshot into these gauges on every scrape (`set_request_states`), just
//! before rendering. Every state is a gauge whose value each scrape overwrites
//! from that snapshot: `active` is a current count, while every other state
//! the server exports is a cumulative total.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use uniserve_observability_derive::MetricFamily;

use crate::U64Gauge;

/// Labels identifying a serving request lifecycle series.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ServingRequestLabels {
    /// Served model identity.
    pub model_name: String,
    /// Model description identity.
    pub description: String,
    /// Request lifecycle state name, as supplied by the caller of
    /// `ServingMetrics::set_request_states`.
    pub state: &'static str,
}

/// Public serving-runtime lifecycle metrics.
#[derive(MetricFamily)]
pub struct ServingMetrics {
    #[metric(
        name = "uniserve:serving_requests",
        help = "Current active requests and cumulative serving-runtime lifecycle state counts."
    )]
    requests: Family<ServingRequestLabels, U64Gauge>,
}

impl ServingMetrics {
    /// Sets every supplied lifecycle-state gauge for a model description.
    ///
    /// Each `(state, value)` pair overwrites one series; states absent from
    /// `states` keep whatever value an earlier call set.
    pub fn set_request_states(
        &self,
        model_name: &str,
        description: &str,
        states: impl IntoIterator<Item = (&'static str, u64)>,
    ) {
        for (state, value) in states {
            self.requests
                .get_or_create(&ServingRequestLabels {
                    model_name: model_name.to_string(),
                    description: description.to_string(),
                    state,
                })
                .set(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Metrics;

    #[test]
    fn lifecycle_states_are_exposed_in_openmetrics_text() {
        let metrics = Metrics::new();
        metrics.serving.set_request_states(
            "served-model",
            "description",
            [("active", 2), ("finished", 7)],
        );

        let rendered = metrics.render().unwrap();
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:serving_requests{")
                && line.contains("model_name=\"served-model\"")
                && line.contains("state=\"active\"")
                && line.ends_with(" 2")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:serving_requests{")
                && line.contains("state=\"finished\"")
                && line.ends_with(" 7")
        }));
    }
}
