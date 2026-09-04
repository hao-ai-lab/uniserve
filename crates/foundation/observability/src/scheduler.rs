//! Scheduler, cache, and worker execution metrics.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use uniserve_observability_derive::MetricFamily;

use crate::{F64Gauge, HistogramFamily, U64Counter, U64Gauge};

const KV_CACHE_RESIDENCY_BUCKETS: [f64; 21] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
    120.0, 300.0, 600.0, 1200.0, 1800.0,
];

/// Builds the histogram used for KV-block allocation lifetime.
fn kv_block_lifetime_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

/// Builds the histogram used for idle time immediately before KV eviction.
fn kv_block_idle_before_evict_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

/// Builds the histogram used for gaps between accesses to one KV block.
fn kv_block_reuse_gap_histogram() -> Histogram {
    Histogram::new(KV_CACHE_RESIDENCY_BUCKETS.iter().copied())
}

/// Labels identifying one model and engine instance.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
}

/// Labels identifying an engine execution path.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePathLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Execution or verification path.
    pub path: String,
}

/// Labels identifying a worker runtime mode.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineModeLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Worker runtime mode.
    pub mode: String,
}

/// Labels identifying an instrumented engine component.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineComponentLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Instrumented worker component.
    pub component: String,
}

/// Labels identifying a scheduler execution domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineDomainLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Scheduler execution domain.
    pub domain: String,
}

/// Labels identifying an event kind within a scheduler domain.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineDomainKindLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Scheduler execution domain.
    pub domain: String,
    /// Event or timing phase within the domain.
    pub kind: String,
}

/// Labels identifying the attention backend that served a kernel launch.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineBackendLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Attention kernel backend family.
    pub backend: String,
}

/// Labels identifying a reason that work remains queued.
#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WaitingReasonLabels {
    /// Served model identity.
    pub model_name: String,
    /// Engine instance index.
    pub engine: u32,
    /// Stable reason that work remains queued.
    pub reason: &'static str,
}

/// Scheduler/batch-scoped Prometheus families exported from `SchedulerStats`.
#[derive(MetricFamily)]
pub struct SchedulerMetrics {
    // Scheduler state gauges.
    /// Requests currently participating in model execution.
    #[metric(
        name = "uniserve:num_requests_running",
        help = "Number of requests in model execution batches"
    )]
    pub scheduler_running: Family<EngineLabels, U64Gauge>,
    /// Requests waiting for scheduler admission.
    #[metric(
        name = "uniserve:num_requests_waiting",
        help = "Number of requests waiting to be processed"
    )]
    pub scheduler_waiting: Family<EngineLabels, U64Gauge>,
    /// Waiting requests grouped by the limiting condition.
    #[metric(
        name = "uniserve:num_requests_waiting_by_reason",
        help = "Number of waiting requests by reason. \
             Reason labels: 'capacity' = waiting for scheduling capacity; \
             'deferred' = deferred by transient constraints (KV transfer, \
             blocked status). Sum of all reasons equals uniserve:num_requests_waiting."
    )]
    pub scheduler_waiting_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    /// Fraction of KV-cache capacity currently occupied.
    #[metric(
        name = "uniserve:kv_cache_usage_perc",
        help = "KV-cache usage. 1 means 100 percent usage"
    )]
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,
    /// Requests admitted into resident execution.
    #[metric(
        name = "uniserve:num_requests_admitted",
        help = "Number of requests admitted into running batches."
    )]
    pub scheduler_admitted: Family<EngineLabels, U64Counter>,
    /// Cumulative admission queue time in microseconds.
    #[metric(
        name = "uniserve:request_queue_wait_us",
        help = "Total queue wait in microseconds for admitted requests."
    )]
    pub scheduler_queue_wait_us: Family<EngineLabels, U64Counter>,
    /// Maximum observed admission queue time in microseconds.
    #[metric(
        name = "uniserve:request_queue_wait_max_us",
        help = "Maximum queue wait in microseconds observed by the scheduler."
    )]
    pub scheduler_queue_wait_max_us: Family<EngineLabels, U64Gauge>,
    /// Active operation credits grouped by execution domain.
    #[metric(
        name = "uniserve:scheduler_domain_active_credits",
        help = "Current scheduler operation credits in use by execution domain."
    )]
    pub scheduler_domain_active_credits: Family<EngineDomainLabels, U64Gauge>,
    /// Peak active operation credits grouped by execution domain.
    #[metric(
        name = "uniserve:scheduler_domain_peak_credits",
        help = "Maximum scheduler operation credits observed in use by execution domain."
    )]
    pub scheduler_domain_peak_credits: Family<EngineDomainLabels, U64Gauge>,
    /// Operation lifecycle events grouped by domain and outcome.
    #[metric(
        name = "uniserve:scheduler_domain_operations",
        help = "Scheduler operation events by execution domain and kind: launched, completed, predicated, or error."
    )]
    pub scheduler_domain_operations: Family<EngineDomainKindLabels, U64Counter>,
    /// Non-blocking resource-pressure events grouped by domain.
    #[metric(
        name = "uniserve:scheduler_domain_backpressure",
        help = "Nonblocking scheduler resource-pressure events by execution domain."
    )]
    pub scheduler_domain_backpressure: Family<EngineDomainLabels, U64Counter>,
    /// Reclaimed operation credits grouped by domain.
    #[metric(
        name = "uniserve:scheduler_domain_reclaimed_credits",
        help = "Scheduler operation credits reclaimed by execution domain."
    )]
    pub scheduler_domain_reclaimed_credits: Family<EngineDomainLabels, U64Counter>,
    /// Completed physical runs grouped by domain.
    #[metric(
        name = "uniserve:scheduler_domain_completed_runs",
        help = "Completed physical runs by execution domain."
    )]
    pub scheduler_domain_completed_runs: Family<EngineDomainLabels, U64Counter>,
    /// Cumulative domain time in microseconds grouped by lifecycle phase.
    #[metric(
        name = "uniserve:scheduler_domain_time_us",
        help = "Cumulative execution-domain time in microseconds by phase: queue, launch, device, completion, or co_resident."
    )]
    pub scheduler_domain_time_us: Family<EngineDomainKindLabels, U64Counter>,
    /// Runs executed concurrently with another domain.
    #[metric(
        name = "uniserve:scheduler_domain_co_resident_runs",
        help = "Physical runs served in a qualified tensorized mixed call by execution domain."
    )]
    pub scheduler_domain_co_resident_runs: Family<EngineDomainLabels, U64Counter>,

    // Prefix-cache counters, including the connector-backed external cache path.
    /// Prefix-cache query tokens.
    #[metric(
        name = "uniserve:prefix_cache_queries",
        help = "Prefix cache queries, in terms of number of queried tokens"
    )]
    pub prefix_cache_queries: Family<EngineLabels, U64Counter>,
    /// Prefix-cache hit tokens.
    #[metric(
        name = "uniserve:prefix_cache_hits",
        help = "Prefix cache hits, in terms of number of cached tokens."
    )]
    pub prefix_cache_hits: Family<EngineLabels, U64Counter>,
    // Worker-local forward/kernel counters.
    /// Attention kernel launches.
    #[metric(
        name = "uniserve:worker_attention_launches",
        help = "Worker attention launches."
    )]
    pub worker_attention_launches: Family<EngineLabels, U64Counter>,
    /// Cumulative attention kernel time in microseconds.
    #[metric(
        name = "uniserve:worker_attention_us",
        help = "Worker attention time in microseconds."
    )]
    pub worker_attention_us: Family<EngineLabels, U64Counter>,
    /// Attention launches grouped by kernel backend.
    #[metric(
        name = "uniserve:worker_attention_backend_counts",
        help = "Worker attention launches by backend kernel family."
    )]
    pub worker_attention_backend_counts: Family<EngineBackendLabels, U64Counter>,
    /// CUDA graph dispatches grouped by runtime mode.
    #[metric(
        name = "uniserve:worker_cuda_graph_runtime_mode_counts",
        help = "Worker forward dispatches by CUDA-graph runtime mode."
    )]
    pub worker_cuda_graph_runtime_mode_counts: Family<EngineModeLabels, U64Counter>,
    /// Forward operations grouped by runtime mode.
    #[metric(
        name = "uniserve:worker_forward_mode_counts",
        help = "Worker forward operations by forward mode."
    )]
    pub worker_forward_mode_counts: Family<EngineModeLabels, U64Counter>,
    /// Forward tokens grouped by runtime mode.
    #[metric(
        name = "uniserve:worker_forward_mode_tokens",
        help = "Worker forward tokens by forward mode."
    )]
    pub worker_forward_mode_tokens: Family<EngineModeLabels, U64Counter>,
    /// Forward wall time grouped by runtime mode.
    #[metric(
        name = "uniserve:worker_forward_mode_us",
        help = "Worker forward wall time by forward mode in microseconds."
    )]
    pub worker_forward_mode_us: Family<EngineModeLabels, U64Counter>,
    /// Forward wall time grouped by worker component.
    #[metric(
        name = "uniserve:worker_forward_component_us",
        help = "Worker forward wall time by component in microseconds."
    )]
    pub worker_forward_component_us: Family<EngineComponentLabels, U64Counter>,
    /// CUDA graph captures.
    #[metric(
        name = "uniserve:worker_cuda_graph_captures",
        help = "Worker CUDA graph captures."
    )]
    pub worker_cuda_graph_captures: Family<EngineLabels, U64Counter>,
    /// CUDA graph replays.
    #[metric(
        name = "uniserve:worker_cuda_graph_replays",
        help = "Worker CUDA graph replays."
    )]
    pub worker_cuda_graph_replays: Family<EngineLabels, U64Counter>,
    /// CUDA graph bucket misses.
    #[metric(
        name = "uniserve:worker_cuda_graph_misses",
        help = "Worker CUDA graph bucket misses."
    )]
    pub worker_cuda_graph_misses: Family<EngineLabels, U64Counter>,
    /// Executions falling back from CUDA graph replay.
    #[metric(
        name = "uniserve:worker_cuda_graph_fallbacks",
        help = "Worker CUDA graph fallbacks to eager execution."
    )]
    pub worker_cuda_graph_fallbacks: Family<EngineLabels, U64Counter>,
    /// Real tokens represented by CUDA graph executions.
    #[metric(
        name = "uniserve:worker_cuda_graph_unpadded_tokens",
        help = "Worker CUDA graph real tokens before padding."
    )]
    pub worker_cuda_graph_unpadded_tokens: Family<EngineLabels, U64Counter>,
    /// Padding tokens represented by CUDA graph executions.
    #[metric(
        name = "uniserve:worker_cuda_graph_padded_tokens",
        help = "Worker CUDA graph padding tokens."
    )]
    pub worker_cuda_graph_padded_tokens: Family<EngineLabels, U64Counter>,
    /// Decode tokens served from the token relay.
    #[metric(
        name = "uniserve:worker_text_decode_token_relay_hits",
        help = "Worker decode token relay hits."
    )]
    pub worker_text_decode_token_relay_hits: Family<EngineLabels, U64Counter>,
    /// Decode token relay misses.
    #[metric(
        name = "uniserve:worker_text_decode_token_relay_misses",
        help = "Worker decode token relay misses."
    )]
    pub worker_text_decode_token_relay_misses: Family<EngineLabels, U64Counter>,
    /// Decode positions served from the position relay.
    #[metric(
        name = "uniserve:worker_text_decode_position_relay_hits",
        help = "Worker decode position relay hits."
    )]
    pub worker_text_decode_position_relay_hits: Family<EngineLabels, U64Counter>,
    /// Decode position relay misses.
    #[metric(
        name = "uniserve:worker_text_decode_position_relay_misses",
        help = "Worker decode position relay misses."
    )]
    pub worker_text_decode_position_relay_misses: Family<EngineLabels, U64Counter>,
    /// FlashInfer decode plan constructions.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_calls",
        help = "Worker FlashInfer decode plan calls."
    )]
    pub worker_flashinfer_decode_plan_calls: Family<EngineLabels, U64Counter>,
    /// FlashInfer decode plan reuses.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_reuses",
        help = "Worker FlashInfer decode plan reuses."
    )]
    pub worker_flashinfer_decode_plan_reuses: Family<EngineLabels, U64Counter>,
    /// Rows represented by FlashInfer decode plans.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_rows",
        help = "Worker FlashInfer decode planned rows."
    )]
    pub worker_flashinfer_decode_plan_rows: Family<EngineLabels, U64Counter>,
    /// Page indices represented by FlashInfer decode plans.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_plan_indices",
        help = "Worker FlashInfer decode planned page indices."
    )]
    pub worker_flashinfer_decode_plan_indices: Family<EngineLabels, U64Counter>,
    /// FlashInfer graph-decode plan constructions.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_graph_plan_calls",
        help = "Worker FlashInfer CUDA-graph decode plan calls."
    )]
    pub worker_flashinfer_decode_graph_plan_calls: Family<EngineLabels, U64Counter>,
    /// FlashInfer graph-decode plan reuses.
    #[metric(
        name = "uniserve:worker_flashinfer_decode_graph_plan_reuses",
        help = "Worker FlashInfer CUDA-graph decode plan reuses."
    )]
    pub worker_flashinfer_decode_graph_plan_reuses: Family<EngineLabels, U64Counter>,
    /// Rows processed by speculative verification.
    #[metric(
        name = "uniserve:worker_spec_verify_rows",
        help = "Worker speculative verifier rows."
    )]
    pub worker_spec_verify_rows: Family<EngineLabels, U64Counter>,
    /// Draft tokens presented for speculative verification.
    #[metric(
        name = "uniserve:worker_spec_verify_draft_tokens",
        help = "Worker speculative verifier draft tokens."
    )]
    pub worker_spec_verify_draft_tokens: Family<EngineLabels, U64Counter>,
    /// Draft tokens accepted by speculative verification.
    #[metric(
        name = "uniserve:worker_spec_verify_accepted_tokens",
        help = "Worker speculative verifier accepted draft tokens."
    )]
    pub worker_spec_verify_accepted_tokens: Family<EngineLabels, U64Counter>,
    /// Draft tokens rejected by speculative verification.
    #[metric(
        name = "uniserve:worker_spec_verify_rejected_tokens",
        help = "Worker speculative verifier rejected draft tokens."
    )]
    pub worker_spec_verify_rejected_tokens: Family<EngineLabels, U64Counter>,
    /// Tokens committed after speculative verification.
    #[metric(
        name = "uniserve:worker_spec_verify_committed_tokens",
        help = "Worker speculative verifier committed tokens."
    )]
    pub worker_spec_verify_committed_tokens: Family<EngineLabels, U64Counter>,
    /// Speculative verification rows grouped by resolution path.
    #[metric(
        name = "uniserve:worker_spec_verify_path_counts",
        help = "Worker speculative verifier rows by verifier path."
    )]
    pub worker_spec_verify_path_counts: Family<EnginePathLabels, U64Counter>,

    // Batch latency uses cumulative microseconds plus a normalization count.
    /// Cumulative worker-reported batch execution time in microseconds.
    #[metric(
        name = "uniserve:worker_exec_us",
        help = "Cumulative worker-reported batch compute time in microseconds."
    )]
    pub worker_exec_us: Family<EngineLabels, U64Counter>,
    /// Cumulative host-observed submit-to-result time in microseconds.
    #[metric(
        name = "uniserve:batch_roundtrip_us",
        help = "Cumulative host-observed batch submit-to-result round-trip time in microseconds."
    )]
    pub batch_roundtrip_us: Family<EngineLabels, U64Counter>,
    /// Resolved batches contributing to cumulative latency counters.
    #[metric(
        name = "uniserve:batch_timing_count",
        help = "Number of resolved batches contributing to the batch-latency counters \
             (the denominator for per-batch averages of worker_exec_us / batch_roundtrip_us)."
    )]
    pub batch_timing_count: Family<EngineLabels, U64Counter>,

    // Per-engine performance / MFU counters.
    /// Estimated floating-point operations per GPU.
    #[metric(
        name = "uniserve:estimated_flops_per_gpu",
        help = "Estimated number of floating point operations per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_flops_per_gpu: Family<EngineLabels, U64Counter>,
    /// Estimated memory bytes read per GPU.
    #[metric(
        name = "uniserve:estimated_read_bytes_per_gpu",
        help = "Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_read_bytes_per_gpu: Family<EngineLabels, U64Counter>,
    /// Estimated memory bytes written per GPU.
    #[metric(
        name = "uniserve:estimated_write_bytes_per_gpu",
        help = "Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations)."
    )]
    pub estimated_write_bytes_per_gpu: Family<EngineLabels, U64Counter>,

    // Sampled KV-cache residency histograms.
    /// KV block lifetime from allocation to eviction.
    #[metric(
        name = "uniserve:kv_block_lifetime_seconds",
        help = "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_lifetime_histogram as fn() -> Histogram)
    )]
    pub kv_block_lifetime_seconds: HistogramFamily,
    /// KV block idle time immediately before eviction.
    #[metric(
        name = "uniserve:kv_block_idle_before_evict_seconds",
        help = "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_idle_before_evict_histogram as fn() -> Histogram)
    )]
    pub kv_block_idle_before_evict_seconds: HistogramFamily,
    /// Time between consecutive accesses to a KV block.
    #[metric(
        name = "uniserve:kv_block_reuse_gap_seconds",
        help = "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).",
        init = Family::new_with_constructor(kv_block_reuse_gap_histogram as fn() -> Histogram)
    )]
    pub kv_block_reuse_gap_seconds: HistogramFamily,
}
