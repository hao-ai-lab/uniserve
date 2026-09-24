//! Serializable scheduler, cache, and worker performance snapshots.
//!
//! The engine's `SchedulerStatsReporter` builds one [`SchedulerStats`] per
//! reporting interval from its cumulative scheduler counters, and the server's
//! `record_scheduler_stats` publishes it as Prometheus metrics: interval deltas
//! are added to counters, while gauges and lifetime values are set. Each
//! field's documentation says which kind it is when that is not an interval
//! delta. [`ForwardStats`] also travels from workers to the engine in batch
//! results that report it.

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
///
/// The engine's `SchedulerStatsReporter` fills only `base.queries` and
/// `base.hits`; it reports the request and preemption counters as zero.
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

/// Worker-local forward/kernel counters folded into scheduler stats.
///
/// In a worker batch result (`BatchOutput::forward_stats` in the worker
/// protocol) the values cover the work reported in that result, and the engine
/// scheduler adds them into its cumulative `WorkerStats` counters. Values are
/// per-update deltas when carried in [`SchedulerStats`].
///
/// The same counters are mirrored elsewhere, including the `ForwardStats`
/// table of the worker flatbuffers schema, the worker-ipc codec, the Python
/// extension's `forward_stats_from_py`, the Python worker's `ForwardStats`,
/// the engine's `WorkerStats` and `SchedulerStatsReporter`, and the server's
/// `record_scheduler_stats`. A field added here also needs an entry in
/// [`ForwardStats::is_empty`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ForwardStats {
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

impl ForwardStats {
    /// Returns whether every worker counter and breakdown is empty or zero.
    ///
    /// The engine reports an interval's worker counters as `None` when their
    /// delta is empty, so a field missing from this check is dropped from any
    /// interval in which only that field changed.
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

/// Per-domain scheduler accounting for one stats update.
///
/// The engine aggregates call kinds into the `prefill`, `decode`, and `flow`
/// domains. `active_credits` is a gauge and `peak_credits` a lifetime maximum;
/// every other count and time field is an interval delta. The worker-reported
/// time fields (`launch_us`, `device_us`, `completion_us`) add, once per batch
/// result report that contains calls of the domain, the maximum over those
/// calls of each `TimingCounters` field they draw on. Batches of different
/// domains can execute concurrently on separate worker execution lanes, so
/// time fields must not be summed across domains to estimate aggregate GPU
/// busy time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainSchedulerStats {
    /// Stable execution-domain name.
    pub domain: String,
    /// Credits currently occupied by in-flight work.
    #[serde(default)]
    pub active_credits: u64,
    /// Highest active-credit count since the scheduler started.
    #[serde(default)]
    pub peak_credits: u64,
    /// Calls launched in the interval.
    #[serde(default)]
    pub launched_calls: u64,
    /// Call results in the interval, including predicated and error results.
    #[serde(default)]
    pub completed_calls: u64,
    /// Calls skipped by a false predicate.
    #[serde(default)]
    pub predicated_calls: u64,
    /// Calls that returned an error status or whose credit was reclaimed
    /// after an execution failure.
    #[serde(default)]
    pub error_calls: u64,
    /// Submission attempts rejected by executor backpressure.
    #[serde(default)]
    pub backpressure_events: u64,
    /// Scheduling credits returned on completion or execution failure.
    #[serde(default)]
    pub reclaimed_credits: u64,
    /// Batch result reports in the interval that contained calls of this
    /// domain.
    #[serde(default)]
    pub completed_batches: u64,
    /// Scheduler time from planning a call to registering it in flight, in
    /// microseconds. Calls of diffusion media requests contribute zero.
    #[serde(default)]
    pub queue_us: u64,
    /// Worker-reported wait before execution (`TimingCounters::queued_us`),
    /// in microseconds.
    #[serde(default)]
    pub launch_us: u64,
    /// Device execution time in microseconds.
    #[serde(default)]
    pub device_us: u64,
    /// Worker completion copy plus host processing time
    /// (`TimingCounters::copy_us` and `host_us`), in microseconds.
    #[serde(default)]
    pub completion_us: u64,
}

/// Serializable scheduler snapshot for one reporting interval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Running token and media requests (gauge).
    pub num_running_reqs: u64,
    /// Token and media requests waiting for admission (gauge).
    pub num_waiting_reqs: u64,
    /// Generation and command-only batches the scheduler has finalized since
    /// it started (a lifetime value, not a delta). Batches built by the media
    /// scheduling pass for diffusion media requests, including its retirement
    /// batch, are not counted.
    pub step_counter: u64,
    /// Data-parallel wave number. The engine tracks no waves and reports zero.
    pub current_wave: u64,
    /// Fraction of usable KV blocks not free (gauge). `1.0` means 100% usage.
    pub kv_cache_usage: f64,
    /// Requests admitted since the previous stats snapshot.
    #[serde(default)]
    pub num_admitted_reqs: u64,
    /// Average queue wait for requests admitted since the previous snapshot,
    /// in microseconds; zero when none was admitted.
    #[serde(default)]
    pub avg_queue_wait_us: u64,
    /// Total queue wait for requests admitted since the previous snapshot, in
    /// microseconds.
    #[serde(default)]
    pub queue_wait_us_total: u64,
    /// Maximum queue wait observed by the scheduler so far, in microseconds
    /// (a lifetime value, not a delta).
    #[serde(default)]
    pub max_queue_wait_us: u64,
    /// Local prefix-cache query and hit deltas, in tokens.
    pub prefix_cache_stats: PrefixCacheStats,
    /// Worker-local forward/kernel counters since the previous stats snapshot,
    /// or `None` when none of them changed.
    #[serde(default)]
    pub worker_forward_stats: Option<ForwardStats>,
    /// Worker-reported execution time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub worker_exec_us: u64,
    /// Host-observed submit-to-result time accumulated in the interval, in microseconds.
    #[serde(default)]
    pub batch_roundtrip_us: u64,
    /// Completed batches represented by `worker_exec_us` and
    /// `batch_roundtrip_us` in this interval.
    #[serde(default)]
    pub batch_count: u64,
    /// Per-domain accounting, in `prefill`, `decode`, `flow` order.
    #[serde(default)]
    pub domain_stats: Vec<DomainSchedulerStats>,
}
