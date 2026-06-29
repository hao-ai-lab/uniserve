use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use uniserve_observability_derive::MetricFamily;

use crate::{F64Gauge, HistogramFamily, U64Counter, U64Gauge};

const KV_CACHE_RESIDENCY_BUCKETS: [f64; 21] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
    120.0, 300.0, 600.0, 1200.0, 1800.0,
];

fn kv_block_lifetime_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_idle_before_evict_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

fn kv_block_reuse_gap_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePositionLabels {
    pub model_name: String,
    pub engine: u32,
    pub position: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePathLabels {
    pub model_name: String,
    pub engine: u32,
    pub path: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineModeLabels {
    pub model_name: String,
    pub engine: u32,
    pub mode: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineComponentLabels {
    pub model_name: String,
    pub engine: u32,
    pub component: String,
}

/// label set for per-attention-backend counters (backend = the kernel
/// family that served the launch, e.g. `flashinfer`/`triton`).
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineBackendLabels {
    pub model_name: String,
    pub engine: u32,
    pub backend: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WaitingReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub reason: &'static str,
}

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
#[derive(MetricFamily)]
pub struct SchedulerMetrics {
 // Scheduler state gauges.
    #[metric(
        name = "uniserve:num_requests_running",
        help = "Number of requests in model execution batches"
    )]
    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    #[metric(
        name = "uniserve:num_requests_waiting",
        help = "Number of requests waiting to be processed"
    )]
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    #[metric(
        name = "uniserve:num_requests_waiting_by_reason",
        help = "Number of waiting requests by reason. \
             Reason labels: 'capacity' = waiting for scheduling capacity; \
             'deferred' = deferred by transient constraints (LoRA budget, KV transfer, \
             blocked status). Sum of all reasons equals uniserve:num_requests_waiting."
    )]
    pub scheduler_waiting_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    #[metric(
        name = "uniserve:kv_cache_usage_perc",
        help = "KV-cache usage. 1 means 100 percent usage"
    )]
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,
    #[metric(
        name = "uniserve:num_requests_admitted",
        help = "Number of requests admitted into running batches."
    )]
    pub scheduler_admitted: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:request_queue_wait_us",
        help = "Total queue wait in microseconds for admitted requests."
    )]
    pub scheduler_queue_wait_us: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:request_queue_wait_max_us",
        help = "Maximum queue wait in microseconds observed by the scheduler."
    )]
    pub scheduler_queue_wait_max_us: Family<EngineLabels, U64Gauge>,

 // Prefix-cache counters, including the connector-backed external cache path.
    #[metric(
        name = "uniserve:prefix_cache_queries",
        help = "Prefix cache queries, in terms of number of queried tokens"
    )]
    pub prefix_cache_queries: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:prefix_cache_hits",
        help = "Prefix cache hits, in terms of number of cached tokens."
    )]
    pub prefix_cache_hits: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:external_prefix_cache_queries",
        help = "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens."
    )]
    pub external_prefix_cache_queries: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:external_prefix_cache_hits",
        help = "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens."
    )]
    pub external_prefix_cache_hits: Family<EngineLabels, U64Counter>,

 // Speculative decoding counters.
    #[metric(
        name = "uniserve:spec_decode_num_drafts",
        help = "Number of spec decoding drafts."
    )]
    pub spec_decode_num_drafts: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:spec_decode_num_draft_tokens",
        help = "Number of draft tokens."
    )]
    pub spec_decode_num_draft_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:spec_decode_num_accepted_tokens",
        help = "Number of accepted tokens."
    )]
    pub spec_decode_num_accepted_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:spec_decode_num_accepted_tokens_per_pos",
        help = "Accepted tokens per draft position."
    )]
    pub spec_decode_num_accepted_tokens_per_pos: Family<EnginePositionLabels, U64Counter>,

 // Worker-local forward/kernel counters.
    #[metric(
        name = "uniserve:worker_attention_launches",
        help = "Worker attention launches."
    )]
    pub worker_attention_launches: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_attention_us",
        help = "Worker attention time in microseconds."
    )]
    pub worker_attention_us: Family<EngineLabels, U64Counter>,
 /// per-attention-backend launch counts.
    #[metric(
        name = "uniserve:worker_attention_backend_counts",
        help = "Worker attention launches by backend kernel family."
    )]
    pub worker_attention_backend_counts: Family<EngineBackendLabels, U64Counter>,
 /// per-runtime-mode CUDA-graph dispatch counts.
    #[metric(
        name = "uniserve:worker_cuda_graph_runtime_mode_counts",
        help = "Worker forward dispatches by CUDA-graph runtime mode."
    )]
    pub worker_cuda_graph_runtime_mode_counts: Family<EngineModeLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_forward_mode_counts",
        help = "Worker forward operations by forward mode."
    )]
    pub worker_forward_mode_counts: Family<EngineModeLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_forward_mode_tokens",
        help = "Worker forward tokens by forward mode."
    )]
    pub worker_forward_mode_tokens: Family<EngineModeLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_forward_mode_us",
        help = "Worker forward wall time by forward mode in microseconds."
    )]
    pub worker_forward_mode_us: Family<EngineModeLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_forward_component_us",
        help = "Worker forward wall time by component in microseconds."
    )]
    pub worker_forward_component_us: Family<EngineComponentLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_captures",
        help = "Worker CUDA graph captures."
    )]
    pub worker_cuda_graph_captures: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_replays",
        help = "Worker CUDA graph replays."
    )]
    pub worker_cuda_graph_replays: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_misses",
        help = "Worker CUDA graph bucket misses."
    )]
    pub worker_cuda_graph_misses: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_fallbacks",
        help = "Worker CUDA graph fallbacks to eager execution."
    )]
    pub worker_cuda_graph_fallbacks: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_unpadded_tokens",
        help = "Worker CUDA graph real tokens before padding."
    )]
    pub worker_cuda_graph_unpadded_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_cuda_graph_padded_tokens",
        help = "Worker CUDA graph padding tokens."
    )]
    pub worker_cuda_graph_padded_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_text_decode_token_relay_hits",
        help = "Worker decode token relay hits."
    )]
    pub worker_text_decode_token_relay_hits: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_text_decode_token_relay_misses",
        help = "Worker decode token relay misses."
    )]
    pub worker_text_decode_token_relay_misses: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_text_decode_position_relay_hits",
        help = "Worker decode position relay hits."
    )]
    pub worker_text_decode_position_relay_hits: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_text_decode_position_relay_misses",
        help = "Worker decode position relay misses."
    )]
    pub worker_text_decode_position_relay_misses: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_calls",
        help = "Worker FlashInfer decode plan calls."
    )]
    pub worker_flashinfer_decode_plan_calls: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_reuses",
        help = "Worker FlashInfer decode plan reuses."
    )]
    pub worker_flashinfer_decode_plan_reuses: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_rows",
        help = "Worker FlashInfer decode planned rows."
    )]
    pub worker_flashinfer_decode_plan_rows: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_indices",
        help = "Worker FlashInfer decode planned page indices."
    )]
    pub worker_flashinfer_decode_plan_indices: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_graph_plan_calls",
        help = "Worker FlashInfer CUDA-graph decode plan calls."
    )]
    pub worker_flashinfer_decode_graph_plan_calls: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_flashinfer_decode_graph_plan_reuses",
        help = "Worker FlashInfer CUDA-graph decode plan reuses."
    )]
    pub worker_flashinfer_decode_graph_plan_reuses: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_rows",
        help = "Worker speculative verifier rows."
    )]
    pub worker_spec_verify_rows: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_draft_tokens",
        help = "Worker speculative verifier draft tokens."
    )]
    pub worker_spec_verify_draft_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_accepted_tokens",
        help = "Worker speculative verifier accepted draft tokens."
    )]
    pub worker_spec_verify_accepted_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_rejected_tokens",
        help = "Worker speculative verifier rejected draft tokens."
    )]
    pub worker_spec_verify_rejected_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_committed_tokens",
        help = "Worker speculative verifier committed tokens."
    )]
    pub worker_spec_verify_committed_tokens: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:worker_spec_verify_path_counts",
        help = "Worker speculative verifier rows by verifier path."
    )]
    pub worker_spec_verify_path_counts: Family<EnginePathLabels, U64Counter>,

 // directly-measured batch latency, cumulative microseconds and the
 // resolved-batch count so dashboards can compute per-batch averages.
    #[metric(
        name = "uniserve:worker_exec_us",
        help = "Cumulative worker-reported batch compute time in microseconds."
    )]
    pub worker_exec_us: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:batch_roundtrip_us",
        help = "Cumulative host-observed batch submit-to-result round-trip time in microseconds."
    )]
    pub batch_roundtrip_us: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:batch_timing_count",
        help = "Number of resolved batches contributing to the batch-latency counters \
             (the denominator for per-batch averages of worker_exec_us / batch_roundtrip_us)."
    )]
    pub batch_timing_count: Family<EngineLabels, U64Counter>,

 // Per-engine performance / MFU counters.
    #[metric(
        name = "uniserve:estimated_flops_per_gpu",
        help = "Estimated number of floating point operations per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_flops_per_gpu: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:estimated_read_bytes_per_gpu",
        help = "Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_read_bytes_per_gpu: Family<EngineLabels, U64Counter>,
    #[metric(
        name = "uniserve:estimated_write_bytes_per_gpu",
        help = "Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_write_bytes_per_gpu: Family<EngineLabels, U64Counter>,

 // Sampled KV-cache residency histograms.
    #[metric(
        name = "uniserve:kv_block_lifetime_seconds",
        help = "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_lifetime_histogram as fn() -> Histogram)
    )]
    pub kv_block_lifetime_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:kv_block_idle_before_evict_seconds",
        help = "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_idle_before_evict_histogram as fn() -> Histogram)
    )]
    pub kv_block_idle_before_evict_seconds: HistogramFamily,
    #[metric(
        name = "uniserve:kv_block_reuse_gap_seconds",
        help = "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_reuse_gap_histogram as fn() -> Histogram)
    )]
    pub kv_block_reuse_gap_seconds: HistogramFamily,
}

#[cfg(test)]
mod tests {
    use crate::{EngineLabels, Metrics};

    #[test]
    fn perf_counters_render_with_a_single_total_suffix() {
        let metrics = Metrics::new();
        let labels = EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        };

        metrics
            .scheduler
            .estimated_flops_per_gpu
            .get_or_create(&labels)
            .inc();
        metrics
            .scheduler
            .estimated_read_bytes_per_gpu
            .get_or_create(&labels)
            .inc();
        metrics
            .scheduler
            .estimated_write_bytes_per_gpu
            .get_or_create(&labels)
            .inc();
        metrics
            .scheduler
            .scheduler_admitted
            .get_or_create(&labels)
            .inc();
        metrics
            .scheduler
            .scheduler_queue_wait_us
            .get_or_create(&labels)
            .inc();

        let rendered = metrics.render().unwrap();
        assert!(rendered.contains(
            "uniserve:estimated_flops_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(rendered.contains(
            "uniserve:estimated_read_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(rendered.contains(
            "uniserve:estimated_write_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(
            rendered.contains(
                "uniserve:num_requests_admitted_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(
            rendered.contains(
                "uniserve:request_queue_wait_us_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(!rendered.contains("uniserve:estimated_flops_per_gpu_total_total"));
        assert!(!rendered.contains("uniserve:estimated_read_bytes_per_gpu_total_total"));
        assert!(!rendered.contains("uniserve:estimated_write_bytes_per_gpu_total_total"));
        assert!(!rendered.contains("uniserve:num_requests_admitted_total_total"));
        assert!(!rendered.contains("uniserve:request_queue_wait_us_total_total"));
    }
}
