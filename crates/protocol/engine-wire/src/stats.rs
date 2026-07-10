use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::OpaqueValue;

/// Stores cache hit statistics.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// Whether the cache was reset.
    pub reset: bool,
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Stores prefix cache hit statistics.
/// - `reset`: Whether `reset_prefix_cache` was invoked.
/// - `queries`: Refers to the number of tokens that were queried.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters and reset flag.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The number of requests preempted before this stats update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
}

/// Single KV cache block eviction sample.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    /// Lifetime from allocation to eviction.
    pub lifetime_seconds: f64,
    /// Idle time observed before eviction.
    pub idle_seconds: f64,
    /// Time gaps between consecutive accesses before eviction.
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Per-step iteration decoding stats from scheduler.
///
/// Each scheduler step, statistics on spec decoding performance are aggregated
/// across requests by the scheduler and returned to the frontend in
/// `EngineCoreOutputs -> SchedulerStats`.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpecDecodingStats {
    /// Configured speculative token count for this scheduler.
    pub num_spec_tokens: u64,
    /// Number of drafted speculative decoding attempts.
    pub num_drafts: u64,
    /// Number of drafted tokens.
    pub num_draft_tokens: u64,
    /// Number of accepted drafted tokens.
    pub num_accepted_tokens: u64,
    /// Accepted drafted tokens counted by draft position.
    pub num_accepted_tokens_per_pos: Vec<u64>,
}

/// Breakdown of a scheduled prefill computation.
///
/// Python models this as a plain `@dataclass`, so it is serialized by msgspec
/// as a map (named fields) rather than in the array-like form used by
/// `EngineCoreOutput` itself.
///
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillStats {
    /// Total number of tokens to be prefilled.
    #[serde(default)]
    pub num_prompt_tokens: u32,
    /// Tokens to be prefilled locally (actual compute work).
    #[serde(default)]
    pub num_computed_tokens: u32,
    /// Tokens to be prefilled without actual compute work.
    #[serde(default)]
    pub num_cached_tokens: u32,
    /// Tokens to be prefilled from local prefix cache.
    #[serde(default)]
    pub num_local_cached_tokens: u32,
    /// Tokens to be prefilled from external KV transfer.
    #[serde(default)]
    pub num_external_cached_tokens: u32,
}

/// Stats for debugging the metrics calculation.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DebugPerfStats {
    /// Time spent calculating these stats.
    pub calc_duration: f64,
    /// Number of prefill requests included in the sampled batch.
    pub num_prefill_requests: u64,
    /// Number of decode requests included in the sampled batch.
    pub num_decode_requests: u64,
    /// Optional execution-context breakdown used for debugging.
    pub context_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component FLOPs breakdown.
    pub num_flops_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-read breakdown.
    pub num_read_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-write breakdown.
    pub num_write_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfStats {
    /// Estimated floating point operations per GPU.
    pub num_flops_per_gpu: u64,
    /// Estimated bytes read from memory per GPU.
    pub num_read_bytes_per_gpu: u64,
    /// Estimated bytes written to memory per GPU.
    pub num_write_bytes_per_gpu: u64,
    /// Optional debug-only perf derivation details.
    pub debug_stats: Option<DebugPerfStats>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CudagraphStat {
    /// Number of real tokens in the captured batch before padding.
    pub num_unpadded_tokens: u64,
    /// Number of padded tokens in the captured batch.
    pub num_padded_tokens: u64,
    /// Number of padding positions added for capture/runtime shape alignment.
    pub num_paddings: u64,
    /// Runtime mode string associated with this CUDA graph sample.
    pub runtime_mode: String,
}

/// Worker-local forward/kernel counters folded into scheduler stats.
///
/// These are per-update deltas when carried in `SchedulerStats`, not lifetime
/// totals. Field names intentionally mirror the Python worker metrics service.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerForwardStats {
    #[serde(default)]
    pub mode_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub mode_tokens: BTreeMap<String, u64>,
    #[serde(default)]
    pub mode_us: BTreeMap<String, u64>,
    #[serde(default)]
    pub component_us: BTreeMap<String, u64>,
    #[serde(default)]
    pub attention_launches: u64,
    #[serde(default)]
    pub attention_us: u64,
    #[serde(default)]
    pub attention_backend_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub cuda_graph_captures: u64,
    #[serde(default)]
    pub cuda_graph_replays: u64,
    #[serde(default)]
    pub cuda_graph_misses: u64,
    #[serde(default)]
    pub cuda_graph_fallbacks: u64,
    #[serde(default)]
    pub cuda_graph_unpadded_tokens: u64,
    #[serde(default)]
    pub cuda_graph_padded_tokens: u64,
    #[serde(default)]
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub text_decode_token_relay_hits: u64,
    #[serde(default)]
    pub text_decode_token_relay_misses: u64,
    #[serde(default)]
    pub text_decode_position_relay_hits: u64,
    #[serde(default)]
    pub text_decode_position_relay_misses: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_calls: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_reuses: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_rows: u64,
    #[serde(default)]
    pub flashinfer_decode_plan_indices: u64,
    #[serde(default)]
    pub flashinfer_decode_graph_plan_calls: u64,
    #[serde(default)]
    pub flashinfer_decode_graph_plan_reuses: u64,
    #[serde(default)]
    pub spec_verify_rows: u64,
    #[serde(default)]
    pub spec_verify_draft_tokens: u64,
    #[serde(default)]
    pub spec_verify_accepted_tokens: u64,
    #[serde(default)]
    pub spec_verify_rejected_tokens: u64,
    #[serde(default)]
    pub spec_verify_committed_tokens: u64,
    #[serde(default)]
    pub spec_verify_path_counts: BTreeMap<String, u64>,
}

impl WorkerForwardStats {
    pub fn is_empty(&self) -> bool {
        self.mode_counts.is_empty()
            && self.mode_tokens.is_empty()
            && self.mode_us.is_empty()
            && self.component_us.is_empty()
            && self.attention_launches == 0
            && self.attention_us == 0
            && self.attention_backend_counts.is_empty()
            && self.cuda_graph_captures == 0
            && self.cuda_graph_replays == 0
            && self.cuda_graph_misses == 0
            && self.cuda_graph_fallbacks == 0
            && self.cuda_graph_unpadded_tokens == 0
            && self.cuda_graph_padded_tokens == 0
            && self.cuda_graph_runtime_mode_counts.is_empty()
            && self.text_decode_token_relay_hits == 0
            && self.text_decode_token_relay_misses == 0
            && self.text_decode_position_relay_hits == 0
            && self.text_decode_position_relay_misses == 0
            && self.flashinfer_decode_plan_calls == 0
            && self.flashinfer_decode_plan_reuses == 0
            && self.flashinfer_decode_plan_rows == 0
            && self.flashinfer_decode_plan_indices == 0
            && self.flashinfer_decode_graph_plan_calls == 0
            && self.flashinfer_decode_graph_plan_reuses == 0
            && self.spec_verify_rows == 0
            && self.spec_verify_draft_tokens == 0
            && self.spec_verify_accepted_tokens == 0
            && self.spec_verify_rejected_tokens == 0
            && self.spec_verify_committed_tokens == 0
            && self.spec_verify_path_counts.is_empty()
    }
}

/// Stats associated with the scheduler.
///
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Length of the "waiting" request queue.
    pub num_waiting_reqs: u64,
    /// Length of the "skipped waiting" queue.
    #[serde(default)]
    pub num_skipped_waiting_reqs: u64,
    /// Internal DP load-balancing step counter.
    pub step_counter: u64,
    /// Internal DP load-balancing wave number.
    pub current_wave: u64,
    /// KV-cache usage. `1.0` means 100% usage.
    pub kv_cache_usage: f64,
    /// Requests admitted since the previous stats snapshot.
    #[serde(default)]
    pub num_admitted_reqs: u64,
    /// Average queue wait for requests admitted since the previous snapshot.
    #[serde(default)]
    pub avg_queue_wait_us: u64,
    /// Total queue wait for requests admitted since the previous snapshot.
    #[serde(default)]
    pub queue_wait_us_total: u64,
    /// Maximum queue wait observed by the scheduler so far.
    #[serde(default)]
    pub max_queue_wait_us: u64,
    /// Local prefix cache statistics.
    pub prefix_cache_stats: PrefixCacheStats,
    /// External connector prefix cache statistics, when configured.
    pub connector_prefix_cache_stats: Option<PrefixCacheStats>,
    /// Sampled KV cache eviction events for residency metrics.
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    /// Speculative decoding scheduler stats, when enabled.
    pub spec_decoding_stats: Option<SpecDecodingStats>,
    /// Connector-specific KV transfer stats, kept opaque for now.
    pub kv_connector_stats: Option<BTreeMap<String, OpaqueValue>>,
    /// Waiting request counts per LoRA adapter.
    pub waiting_lora_adapters: BTreeMap<String, u64>,
    /// Running request counts per LoRA adapter.
    pub running_lora_adapters: BTreeMap<String, u64>,
    /// CUDA graph runtime stats when graph metrics are enabled.
    pub cudagraph_stats: Option<CudagraphStat>,
    /// Estimated MFU/performance stats, when enabled.
    pub perf_stats: Option<PerfStats>,
    /// Worker-local forward/kernel counters since the previous stats snapshot.
    #[serde(default)]
    pub worker_forward_stats: Option<WorkerForwardStats>,
    /// directly-measured batch latency since the previous snapshot, in
    /// microseconds. `worker_exec_us` is the worker compute time the worker
    /// reports; `batch_roundtrip_us` is the host-observed submit->result
    /// round-trip; `batch_count` is the number of resolved batches over the
    /// interval so the two sums can be normalized to per-batch averages. These
    /// also reaches Prometheus.
    #[serde(default)]
    pub worker_exec_us: u64,
    #[serde(default)]
    pub batch_roundtrip_us: u64,
    #[serde(default)]
    pub batch_count: u64,
}
