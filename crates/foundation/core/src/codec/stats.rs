//! Serializable scheduler, cache, and worker performance snapshots.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Cache query and hit counters for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BaseCacheStats {
    /// The number of requests in this update.
    pub requests: u64,
    /// The number of queries in these requests.
    pub queries: u64,
    /// The number of hits in these requests.
    pub hits: u64,
}

/// Prefix-cache counters, where query fields count tokens.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Embedded base cache counters.
    #[serde(flatten)]
    pub base: BaseCacheStats,
    /// The number of requests preempted before this stats update.
    pub preempted_requests: u64,
    /// The `queries` number for preempted requests.
    pub preempted_queries: u64,
    /// The `hits` number for preempted requests.
    pub preempted_hits: u64,
}

/// Timing sample captured when one KV-cache block is evicted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KvCacheEvictionEvent {
    /// Lifetime from allocation to eviction.
    pub lifetime_seconds: f64,
    /// Idle time observed before eviction.
    pub idle_seconds: f64,
    /// Time gaps between consecutive accesses before eviction.
    pub reuse_gaps_seconds: Vec<f64>,
}

/// Token-source breakdown for scheduled prefill work.
///
/// The value serializes as a map with stable field names.
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

/// Inputs and timing used to inspect performance-estimate calculation.
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

/// Estimated compute and memory traffic for one worker update.
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

/// Shape and runtime-mode metadata for one CUDA graph execution.
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
/// Values are per-update deltas when carried in [`SchedulerStats`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerForwardStats {
    /// Forward executions grouped by runtime mode.
    #[serde(default)]
    pub mode_counts: BTreeMap<String, u64>,
    /// Tokens processed by runtime mode.
    #[serde(default)]
    pub mode_tokens: BTreeMap<String, u64>,
    /// Forward time in microseconds by runtime mode.
    #[serde(default)]
    pub mode_us: BTreeMap<String, u64>,
    /// Component execution time in microseconds.
    #[serde(default)]
    pub component_us: BTreeMap<String, u64>,
    /// Attention kernel launch count.
    #[serde(default)]
    pub attention_launches: u64,
    /// Cumulative attention kernel time in microseconds.
    #[serde(default)]
    pub attention_us: u64,
    /// Attention launches grouped by backend.
    #[serde(default)]
    pub attention_backend_counts: BTreeMap<String, u64>,
    /// CUDA graph capture count.
    #[serde(default)]
    pub cuda_graph_captures: u64,
    /// CUDA graph replay count.
    #[serde(default)]
    pub cuda_graph_replays: u64,
    /// CUDA graph lookup misses.
    #[serde(default)]
    pub cuda_graph_misses: u64,
    /// Executions that fell back from graph replay.
    #[serde(default)]
    pub cuda_graph_fallbacks: u64,
    /// Real tokens represented by CUDA graph executions.
    #[serde(default)]
    pub cuda_graph_unpadded_tokens: u64,
    /// Padded token slots represented by CUDA graph executions.
    #[serde(default)]
    pub cuda_graph_padded_tokens: u64,
    /// CUDA graph dispatches grouped by runtime mode.
    #[serde(default)]
    pub cuda_graph_runtime_mode_counts: BTreeMap<String, u64>,
    /// Decode tokens served from the token relay.
    #[serde(default)]
    pub text_decode_token_relay_hits: u64,
    /// Decode token relay misses.
    #[serde(default)]
    pub text_decode_token_relay_misses: u64,
    /// Decode positions served from the position relay.
    #[serde(default)]
    pub text_decode_position_relay_hits: u64,
    /// Decode position relay misses.
    #[serde(default)]
    pub text_decode_position_relay_misses: u64,
    /// FlashInfer decode plan construction count.
    #[serde(default)]
    pub flashinfer_decode_plan_calls: u64,
    /// FlashInfer decode plan reuse count.
    #[serde(default)]
    pub flashinfer_decode_plan_reuses: u64,
    /// Rows represented by FlashInfer decode plans.
    #[serde(default)]
    pub flashinfer_decode_plan_rows: u64,
    /// Indices represented by FlashInfer decode plans.
    #[serde(default)]
    pub flashinfer_decode_plan_indices: u64,
    /// FlashInfer graph-decode plan construction count.
    #[serde(default)]
    pub flashinfer_decode_graph_plan_calls: u64,
    /// FlashInfer graph-decode plan reuse count.
    #[serde(default)]
    pub flashinfer_decode_graph_plan_reuses: u64,
    /// Rows processed by speculative verification.
    #[serde(default)]
    pub spec_verify_rows: u64,
    /// Draft tokens presented for speculative verification.
    #[serde(default)]
    pub spec_verify_draft_tokens: u64,
    /// Draft tokens accepted by speculative verification.
    #[serde(default)]
    pub spec_verify_accepted_tokens: u64,
    /// Draft tokens rejected by speculative verification.
    #[serde(default)]
    pub spec_verify_rejected_tokens: u64,
    /// Tokens committed after speculative verification.
    #[serde(default)]
    pub spec_verify_committed_tokens: u64,
    /// Speculative verification rows grouped by resolution path.
    #[serde(default)]
    pub spec_verify_path_counts: BTreeMap<String, u64>,
}

impl WorkerForwardStats {
    /// Returns whether every worker counter and breakdown is empty or zero.
    pub fn is_empty(&self) -> bool {
        // Map-backed breakdowns and scalar counters form one aggregate delta;
        // any populated component makes the snapshot observable.
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

/// Per-domain scheduler accounting for one stats update.
///
/// Operation, run, pressure, reclaim, and time fields are interval
/// deltas. Credit fields are gauges. Device and co-residency time describe the
/// full interval visible to the named domain; values from co-resident domains
/// therefore must not be summed to estimate aggregate GPU busy time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSchedulerStats {
    /// Stable execution-domain name.
    pub domain: String,
    /// Credits currently occupied by in-flight work.
    #[serde(default)]
    pub active_credits: u64,
    /// Highest active-credit count in the interval.
    #[serde(default)]
    pub peak_credits: u64,
    /// Operations launched in the interval.
    #[serde(default)]
    pub launched_operations: u64,
    /// Operations completed in the interval.
    #[serde(default)]
    pub completed_operations: u64,
    /// Operations skipped by a false predicate.
    #[serde(default)]
    pub predicated_operations: u64,
    /// Operations completed with an error.
    #[serde(default)]
    pub error_operations: u64,
    /// Submission attempts rejected by executor backpressure.
    #[serde(default)]
    pub backpressure_events: u64,
    /// Scheduling credits returned after completion.
    #[serde(default)]
    pub reclaimed_credits: u64,
    /// Physical runs completed in the interval.
    #[serde(default)]
    pub completed_runs: u64,
    /// Runs overlapping another execution domain.
    #[serde(default)]
    pub co_resident_runs: u64,
    /// Time spent queued before launch, in microseconds.
    #[serde(default)]
    pub queue_us: u64,
    /// Host launch overhead in microseconds.
    #[serde(default)]
    pub launch_us: u64,
    /// Device execution time in microseconds.
    #[serde(default)]
    pub device_us: u64,
    /// Host completion processing time in microseconds.
    #[serde(default)]
    pub completion_us: u64,
    /// Device time overlapping another execution domain, in microseconds.
    #[serde(default)]
    pub co_resident_us: u64,
}

/// Serializable scheduler snapshot for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of requests in model execution batches.
    pub num_running_reqs: u64,
    /// Length of the "waiting" request queue.
    pub num_waiting_reqs: u64,
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
    /// Sampled KV cache eviction events for residency metrics.
    pub kv_cache_eviction_events: Vec<KvCacheEvictionEvent>,
    /// CUDA graph runtime stats when graph metrics are enabled.
    pub cudagraph_stats: Option<CudagraphStat>,
    /// Estimated MFU/performance stats, when enabled.
    pub perf_stats: Option<PerfStats>,
    /// Worker-local forward/kernel counters since the previous stats snapshot.
    #[serde(default)]
    pub worker_forward_stats: Option<WorkerForwardStats>,
    /// Worker-reported execution time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub worker_exec_us: u64,
    /// Host-observed submit-to-result time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub batch_roundtrip_us: u64,
    /// Resolved batch count used to normalize interval latency totals.
    #[serde(default)]
    pub batch_count: u64,
    /// Exact prefill, decode, and flow accounting for this update.
    #[serde(default)]
    pub domain_stats: Vec<DomainSchedulerStats>,
}
