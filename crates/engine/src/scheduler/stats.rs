//! Atomic counters and gauges shared with the scheduler statistics reporter.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize};

/// General loop counters that do not belong to a more specific group.
#[derive(Default)]
pub struct GeneralStats {
    /// Largest number of calls observed in one submitted batch.
    pub peak_ops: AtomicUsize,
    /// Number of scheduler loop iterations completed.
    pub steps: AtomicU64,
    /// Current number of admitted, nonterminal requests.
    pub running: AtomicUsize,
    /// Current number of requests waiting for admission.
    pub pending: AtomicUsize,
    /// Current number of submitted calls awaiting completion.
    pub in_flight: AtomicUsize,
}

/// KV block-cache observability.
#[derive(Default)]
pub struct KvCacheStats {
    /// Physical KV blocks currently available for allocation.
    pub free_blocks: AtomicUsize,
    /// Total physical KV blocks managed by the scheduler.
    pub num_blocks: AtomicUsize,
    /// Cumulative KV blocks evicted from the prefix cache.
    pub blocks_evicted: AtomicU64,
    /// Cumulative KV blocks inserted into the prefix cache.
    pub blocks_stored: AtomicU64,
    /// Current number of reusable prefix-cache blocks.
    pub cached_blocks: AtomicUsize,
}

/// Prefix-cache hit-rate counters.
#[derive(Default)]
pub struct PrefixStats {
    /// Cumulative prefix-cache lookups.
    pub queries: AtomicU64,
    /// Cumulative prefix-cache lookups that matched at least one block.
    pub hits: AtomicU64,
    /// Cumulative prompt tokens reused from matched prefix blocks.
    pub hit_tokens: AtomicU64,
}

/// Encoder-output (multimodal) cache counters.
#[derive(Default)]
pub struct EncoderStats {
    /// Cumulative encoder-cache lookups.
    pub cache_queries: AtomicU64,
    /// Cumulative encoder-cache hits.
    pub cache_hits: AtomicU64,
    /// Current number of resident encoder products.
    pub cached: AtomicUsize,
}

/// Batch-timing and queueing/admission latency counters.
#[derive(Default)]
pub struct TimingStats {
    /// Most recent worker-reported batch execution time, in microseconds.
    pub last_worker_exec_us: AtomicU64,
    /// Cumulative worker execution time, in microseconds.
    pub worker_exec_us_total: AtomicU64,
    /// Cumulative submission-to-completion latency, in microseconds.
    pub batch_roundtrip_us_total: AtomicU64,
    /// Number of completed batches represented by timing totals.
    pub batch_timing_count: AtomicU64,
    /// Number of requests with recorded admission queue latency.
    pub queue_wait_count: AtomicU64,
    /// Cumulative admission queue latency, in microseconds.
    pub queue_wait_us_total: AtomicU64,
    /// Maximum observed admission queue latency, in microseconds.
    pub queue_wait_us_max: AtomicU64,
}

/// Cumulative accounting for one physical execution domain.
#[derive(Default)]
pub struct DomainStats {
    /// Credits currently held by in-flight calls.
    pub active_credits: AtomicUsize,
    /// Maximum number of concurrently held credits.
    pub peak_credits: AtomicUsize,
    /// Cumulative calls submitted to the domain.
    pub launched_calls: AtomicU64,
    /// Cumulative calls completed by the domain.
    pub completed_calls: AtomicU64,
    /// Cumulative calls skipped by a false predicate.
    pub predicated_calls: AtomicU64,
    /// Cumulative calls completed with an error.
    pub error_calls: AtomicU64,
    /// Cumulative submissions rejected by executor backpressure.
    pub backpressure_events: AtomicU64,
    /// Cumulative credits recovered after terminal execution failure.
    pub reclaimed_credits: AtomicU64,
    /// Cumulative batches completed in the domain.
    pub completed_batches: AtomicU64,
    /// Cumulative worker queue time, in microseconds.
    pub queue_us: AtomicU64,
    /// Cumulative worker launch time, in microseconds.
    pub launch_us: AtomicU64,
    /// Cumulative device execution time, in microseconds.
    pub device_us: AtomicU64,
    /// Cumulative completion processing time, in microseconds.
    pub completion_us: AtomicU64,
}

/// Domain-indexed scheduler accounting shared with the stats reporter.
#[derive(Default)]
pub struct ExecutionDomainStats {
    /// Counters for prefill and encoder execution.
    pub prefill: DomainStats,
    /// Counters for autoregressive decode execution.
    pub decode: DomainStats,
    /// Counters for diffusion flow execution.
    pub flow: DomainStats,
}

impl ExecutionDomainStats {
    /// Public metrics aggregate concrete call kinds into three stable labels.
    /// This index is used only for counters, never for execution or lane routing.
    pub(super) const fn index(computation: uniserve_worker_ipc::CallKind) -> usize {
        use uniserve_worker_ipc::{CallKind, ForwardMode, MediaCall};
        match computation {
            CallKind::Forward(ForwardMode::Decode | ForwardMode::Verify) => 1,
            CallKind::Forward(_)
            | CallKind::Transfer(_)
            | CallKind::Media(
                MediaCall::TextEncoding | MediaCall::VisionEncoding | MediaCall::LatentEncoding,
            ) => 0,
            CallKind::Media(_) => 2,
        }
    }

    /// Stable public metric labels and their accumulated counters.
    pub(super) fn groups(&self) -> [(&'static str, &DomainStats); 3] {
        [
            ("prefill", &self.prefill),
            ("decode", &self.decode),
            ("flow", &self.flow),
        ]
    }

    /// Returns the public metrics counters for an actual computation.
    pub fn get(&self, computation: uniserve_worker_ipc::CallKind) -> &DomainStats {
        self.groups()[Self::index(computation)].1
    }
}

/// Worker-local forward/kernel counters, aggregated as completed batches
/// resolve so OpenMetrics can explain scheduler vs worker/kernel latency.
#[derive(Default)]
pub struct WorkerStats {
    /// Forward invocation counts keyed by worker runtime mode.
    pub forward_mode_counts: Mutex<BTreeMap<String, u64>>,
    /// Forward token counts keyed by worker runtime mode.
    pub forward_mode_tokens: Mutex<BTreeMap<String, u64>>,
    /// Forward execution time keyed by worker runtime mode, in microseconds.
    pub forward_mode_us: Mutex<BTreeMap<String, u64>>,
    /// Forward execution time keyed by worker component, in microseconds.
    pub forward_component_us: Mutex<BTreeMap<String, u64>>,
    /// Cumulative attention kernel launches.
    pub attention_launches: AtomicU64,
    /// Cumulative attention execution time, in microseconds.
    pub attention_us: AtomicU64,
    /// Attention launch counts keyed by backend.
    pub attention_backend_counts: Mutex<BTreeMap<String, u64>>,
    /// Cumulative CUDA graph captures.
    pub cuda_graph_captures: AtomicU64,
    /// Cumulative CUDA graph replays.
    pub cuda_graph_replays: AtomicU64,
    /// Cumulative requests without a matching captured graph.
    pub cuda_graph_misses: AtomicU64,
    /// Cumulative graph dispatches that fell back to eager execution.
    pub cuda_graph_fallbacks: AtomicU64,
    /// Cumulative logical tokens submitted to CUDA graphs.
    pub cuda_graph_unpadded_tokens: AtomicU64,
    /// Cumulative padded tokens executed by CUDA graphs.
    pub cuda_graph_padded_tokens: AtomicU64,
    /// CUDA graph dispatch counts keyed by worker runtime mode.
    pub cuda_graph_runtime_mode_counts: Mutex<BTreeMap<String, u64>>,
    /// Cumulative decode rows that reused a relayed token.
    pub text_decode_token_relay_hits: AtomicU64,
    /// Cumulative decode rows that required a host-supplied token.
    pub text_decode_token_relay_misses: AtomicU64,
    /// Cumulative decode rows that reused a relayed position.
    pub text_decode_position_relay_hits: AtomicU64,
    /// Cumulative decode rows that required a host-supplied position.
    pub text_decode_position_relay_misses: AtomicU64,
    /// Cumulative FlashInfer decode-plan constructions.
    pub flashinfer_decode_plan_calls: AtomicU64,
    /// Cumulative FlashInfer decode-plan cache reuses.
    pub flashinfer_decode_plan_reuses: AtomicU64,
    /// Cumulative rows included in FlashInfer decode plans.
    pub flashinfer_decode_plan_rows: AtomicU64,
    /// Cumulative page-table indices included in FlashInfer decode plans.
    pub flashinfer_decode_plan_indices: AtomicU64,
    /// Cumulative graph-compatible FlashInfer decode-plan constructions.
    pub flashinfer_decode_graph_plan_calls: AtomicU64,
    /// Cumulative graph-compatible FlashInfer decode-plan reuses.
    pub flashinfer_decode_graph_plan_reuses: AtomicU64,
    /// Cumulative rows processed by speculative verification.
    pub spec_verify_rows: AtomicU64,
    /// Cumulative draft tokens submitted for verification.
    pub spec_verify_draft_tokens: AtomicU64,
    /// Cumulative draft tokens accepted by verification.
    pub spec_verify_accepted_tokens: AtomicU64,
    /// Cumulative draft tokens rejected by verification.
    pub spec_verify_rejected_tokens: AtomicU64,
    /// Cumulative target tokens committed after verification.
    pub spec_verify_committed_tokens: AtomicU64,
    /// Speculative verification counts keyed by execution path.
    pub spec_verify_path_counts: Mutex<BTreeMap<String, u64>>,
}

/// Live scheduler stats, shared with the frontend for `/stats` observability.
#[derive(Default)]
pub struct SchedulerStats {
    /// General scheduler loop and queue counters.
    pub general: GeneralStats,
    /// Paged KV allocation and prefix-cache counters.
    pub kv_cache: KvCacheStats,
    /// Prefix-cache lookup counters.
    pub prefix: PrefixStats,
    /// Encoder-output cache counters.
    pub encoder: EncoderStats,
    /// Batch and admission timing counters.
    pub timing: TimingStats,
    /// Per-execution-domain accounting.
    pub domains: ExecutionDomainStats,
    /// Worker forward and kernel accounting.
    pub worker: WorkerStats,
}
