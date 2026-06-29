//! Offline, exact latency-distribution summaries for benchmark reports.
//!
//! This is the *offline/exact-percentile* counterpart to the live, bucketed
//! Prometheus histograms in `uniserve-observability` (request latency, TTFT,
//! ITL — see `foundation/observability/src/request.rs`). The two are
//! deliberately separate tools and share no contract: the observability crate
//! emits live, pre-bucketed counters for monitoring, whereas this module sorts
//! a captured sample and computes exact percentiles for after-the-fact bench
//! reporting. Use this when a benchmark needs precise p50/p90/p95/p99 over a
//! finite, in-memory sample rather than streaming bucket counts.

#[derive(Debug, Clone, PartialEq)]
pub struct PercentileSummary {
    pub count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

pub fn summarize(values: &[f64]) -> Option<PercentileSummary> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    Some(PercentileSummary {
        count: sorted.len(),
        mean,
        p50: percentile(&sorted, 50.0),
        p90: percentile(&sorted, 90.0),
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
        max: sorted[sorted.len() - 1],
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0) * (sorted.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = rank - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_uses_sorted_percentiles() {
        let s = summarize(&[4.0, 1.0, 2.0, 3.0]).expect("summary");
        assert_eq!(s.count, 4);
        assert_eq!(s.p50, 2.5);
        assert_eq!(s.max, 4.0);
    }
}
