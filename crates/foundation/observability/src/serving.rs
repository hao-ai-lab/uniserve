use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use uniserve_observability_derive::MetricFamily;

use crate::U64Gauge;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ServingRequestLabels {
    pub model_name: String,
    pub description: String,
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
