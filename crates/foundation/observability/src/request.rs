use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use uniserve_observability_derive::MetricFamily;

use crate::{EngineLabels, HistogramFamily, U64Counter};

const TTFT_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
const ITL_BUCKETS: [f64; 19] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
const REQUEST_LATENCY_BUCKETS: [f64; 21] = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
const REQUEST_PARAMS_N_BUCKETS: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];

fn build_1_2_5_buckets(max_value: u32) -> Vec<f64> {
    let mut buckets = Vec::new();
    let mut exponent = 0;
    loop {
        for mantissa in [1_u32, 2, 5] {
            let value = mantissa * 10_u32.pow(exponent);
            if value <= max_value {
                buckets.push(value as f64);
            } else {
                if buckets.last().copied() != Some(max_value as f64) {
                    buckets.push(max_value as f64);
                }
                return buckets;
            }
        }
        exponent += 1;
    }
}

fn time_to_first_token_histogram() -> Histogram {
    Histogram::new(TTFT_BUCKETS.iter().copied())
}

fn inter_token_latency_histogram() -> Histogram {
    Histogram::new(ITL_BUCKETS.iter().copied())
}

fn request_time_per_output_token_histogram() -> Histogram {
    Histogram::new(ITL_BUCKETS.iter().copied())
}

fn request_latency_histogram() -> Histogram {
    Histogram::new(REQUEST_LATENCY_BUCKETS.iter().copied())
}

fn request_token_count_histogram() -> Histogram {
    // Histogram upper bound is intentionally static; request-level context
    // limits are enforced before metrics are recorded.
    Histogram::new(build_1_2_5_buckets(131_072))
}

fn request_params_n_histogram() -> Histogram {
    Histogram::new(REQUEST_PARAMS_N_BUCKETS.iter().copied())
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FinishedReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub finished_reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PromptTokenSourceLabels {
    pub model_name: String,
    pub engine: u32,
    pub source: &'static str,
}

pub(crate) type FinishedReasonCounterFamily = Family<FinishedReasonLabels, U64Counter>;
pub(crate) type PromptTokenSourceCounterFamily = Family<PromptTokenSourceLabels, U64Counter>;

/// Request-lifecycle Prometheus families exported from the `llm` layer.
#[derive(MetricFamily)]
pub struct RequestMetrics {
    // Request-derived counters.
    #[metric(
        name = "uniserve:num_preemptions",
        help = "Cumulative number of preemption events."
    )]
    pub num_preemptions: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:prompt_tokens",
        help = "Number of prefill tokens processed."
    )]
    pub prompt_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:prompt_tokens_by_source",
        help = "Number of prompt tokens by source."
    )]
    pub prompt_tokens_by_source: PromptTokenSourceCounterFamily,
    #[metric(
        name = "uniserve:prompt_tokens_cached",
        help = "Number of prompt tokens with prefix cache hits."
    )]
    pub prompt_tokens_cached: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:generation_tokens",
        help = "Number of generation tokens processed."
    )]
    pub generation_tokens: Family<EngineLabels, U64Counter>,

    // We intentionally don't support iteration-level histograms for now, since it seems to make
    // more sense if the engine maintains these metrics and frontend simply forwards.

    // pub iteration_tokens_total: HistogramFamily,

    // Request lifecycle counters and histograms.
    #[metric(
        name = "uniserve:request_success",
        help = "Count of successfully processed requests."
    )]
    pub request_success: FinishedReasonCounterFamily,
    #[metric(
        name = "uniserve:request_prompt_tokens",
        help = "Number of prefill tokens processed.",
        init = Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram)
    )]
    pub request_prompt_tokens: HistogramFamily,
    #[metric(
        name = "uniserve:request_generation_tokens",
        help = "Number of generation tokens processed.",
        init = Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram)
    )]
    pub request_generation_tokens: HistogramFamily,
    #[metric(
        name = "uniserve:request_max_num_generation_tokens",
        help = "Histogram of maximum number of requested generation tokens.",
        init = Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram)
    )]
    pub request_max_num_generation_tokens: HistogramFamily,
    #[metric(
        name = "uniserve:request_params_max_tokens",
        help = "Histogram of the max_tokens request parameter.",
        init = Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram)
    )]
    pub request_params_max_tokens: HistogramFamily,
    #[metric(
        name = "uniserve:request_params_n",
        help = "Histogram of the n request parameter.",
        init = Family::new_with_constructor(request_params_n_histogram as fn() -> Histogram)
    )]
    pub request_params_n: HistogramFamily,
    #[metric(
        name = "uniserve:request_prefill_kv_computed_tokens",
        help = "Histogram of new KV tokens computed during prefill (excluding cached tokens).",
        init = Family::new_with_constructor(request_token_count_histogram as fn() -> Histogram)
    )]
    pub request_prefill_kv_computed_tokens: HistogramFamily,
    #[metric(
        name = "uniserve:time_to_first_token_seconds",
        help = "Histogram of time to first token in seconds.",
        init = Family::new_with_constructor(time_to_first_token_histogram as fn() -> Histogram)
    )]
    pub time_to_first_token_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:inter_token_latency_seconds",
        help = "Histogram of inter-token latency in seconds.",
        init = Family::new_with_constructor(inter_token_latency_histogram as fn() -> Histogram)
    )]
    pub inter_token_latency_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:e2e_request_latency_seconds",
        help = "Histogram of e2e request latency in seconds.",
        init = Family::new_with_constructor(request_latency_histogram as fn() -> Histogram)
    )]
    pub e2e_request_latency_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:request_queue_time_seconds",
        help = "Histogram of time spent in WAITING phase for request.",
        init = Family::new_with_constructor(request_latency_histogram as fn() -> Histogram)
    )]
    pub request_queue_time_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:request_prefill_time_seconds",
        help = "Histogram of time spent in PREFILL phase for request.",
        init = Family::new_with_constructor(request_latency_histogram as fn() -> Histogram)
    )]
    pub request_prefill_time_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:request_decode_time_seconds",
        help = "Histogram of time spent in DECODE phase for request.",
        init = Family::new_with_constructor(request_latency_histogram as fn() -> Histogram)
    )]
    pub request_decode_time_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:request_inference_time_seconds",
        help = "Histogram of time spent in RUNNING phase for request.",
        init = Family::new_with_constructor(request_latency_histogram as fn() -> Histogram)
    )]
    pub request_inference_time_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:request_time_per_output_token_seconds",
        help = "Histogram of time_per_output_token_seconds per request.",
        init = Family::new_with_constructor(request_time_per_output_token_histogram as fn() -> Histogram)
    )]
    pub request_time_per_output_token_seconds: HistogramFamily,
}
