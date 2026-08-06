//! Shared scheduler-stats reporter: the single mapping from the
//! scheduler's live [`SchedStats`] counters onto the wire [`SchedulerStats`]
//! shape, used identically by the in-process server and the headless engine
//! process so the aggregator is defined once instead of copy-pasted per
//! transport.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use crate::SchedStats;
use uniserve_engine_wire::stats::{
    BaseCacheStats, PrefixCacheStats, SchedulerStats, WorkerForwardStats,
};

/// Converts the scheduler's cumulative counters into per-update deltas for the
/// wire shape (whose prefix-cache counters are increments, not totals).
#[derive(Debug, Default)]
pub struct SchedStatsReporter {
    last_prefix_queries: u64,
    last_prefix_hit_tokens: u64,
    last_queue_wait_count: u64,
    last_queue_wait_us_total: u64,
    last_worker_forward_stats: WorkerForwardStats,
    // cumulative batch-timing counters, delta'd into per-update sums.
    last_worker_exec_us_total: u64,
    last_batch_roundtrip_us_total: u64,
    last_batch_timing_count: u64,
}

impl SchedStatsReporter {
    /// Snapshot the live counters into one wire `SchedulerStats` update.
    ///
    /// `block_size` converts block-granular prefix-cache query counts into the
    /// token-granular counts the wire shape documents.
    pub fn snapshot(&mut self, stats: &SchedStats, block_size: u32) -> SchedulerStats {
        let num_blocks = stats.kv_cache.num_blocks.load(Ordering::Relaxed);
        let free_blocks = stats.kv_cache.free_blocks.load(Ordering::Relaxed);
        let kv_cache_usage = if num_blocks > 0 {
            1.0 - (free_blocks as f64 / num_blocks as f64)
        } else {
            0.0
        };

        let prefix_queries = stats.prefix.queries.load(Ordering::Relaxed);
        let prefix_hit_tokens = stats.prefix.hit_tokens.load(Ordering::Relaxed);
        let delta_queries = prefix_queries.saturating_sub(self.last_prefix_queries);
        let delta_hit_tokens = prefix_hit_tokens.saturating_sub(self.last_prefix_hit_tokens);
        self.last_prefix_queries = prefix_queries;
        self.last_prefix_hit_tokens = prefix_hit_tokens;

        let queue_wait_count = stats.timing.queue_wait_count.load(Ordering::Relaxed);
        let queue_wait_us_total = stats.timing.queue_wait_us_total.load(Ordering::Relaxed);
        let delta_queue_wait_count = queue_wait_count.saturating_sub(self.last_queue_wait_count);
        let delta_queue_wait_us_total =
            queue_wait_us_total.saturating_sub(self.last_queue_wait_us_total);
        self.last_queue_wait_count = queue_wait_count;
        self.last_queue_wait_us_total = queue_wait_us_total;
        let avg_queue_wait_us = delta_queue_wait_us_total
            .checked_div(delta_queue_wait_count)
            .unwrap_or(0);

        // per-update deltas of the directly-measured batch timing.
        let worker_exec_us_total = stats.timing.worker_exec_us_total.load(Ordering::Relaxed);
        let batch_roundtrip_us_total = stats
            .timing
            .batch_roundtrip_us_total
            .load(Ordering::Relaxed);
        let batch_timing_count = stats.timing.batch_timing_count.load(Ordering::Relaxed);
        let delta_worker_exec_us =
            worker_exec_us_total.saturating_sub(self.last_worker_exec_us_total);
        let delta_batch_roundtrip_us =
            batch_roundtrip_us_total.saturating_sub(self.last_batch_roundtrip_us_total);
        let delta_batch_count = batch_timing_count.saturating_sub(self.last_batch_timing_count);
        self.last_worker_exec_us_total = worker_exec_us_total;
        self.last_batch_roundtrip_us_total = batch_roundtrip_us_total;
        self.last_batch_timing_count = batch_timing_count;

        SchedulerStats {
            num_running_reqs: stats.general.running.load(Ordering::Relaxed) as u64,
            num_waiting_reqs: stats.general.pending.load(Ordering::Relaxed) as u64,
            step_counter: stats.general.steps.load(Ordering::Relaxed),
            current_wave: 0,
            kv_cache_usage,
            num_admitted_reqs: delta_queue_wait_count,
            avg_queue_wait_us,
            queue_wait_us_total: delta_queue_wait_us_total,
            max_queue_wait_us: stats.timing.queue_wait_us_max.load(Ordering::Relaxed),
            prefix_cache_stats: PrefixCacheStats {
                base: BaseCacheStats {
                    requests: 0,
                    queries: delta_queries * block_size as u64,
                    hits: delta_hit_tokens,
                },
                preempted_requests: 0,
                ..Default::default()
            },
            worker_forward_stats: self.worker_forward_stats(stats),
            worker_exec_us: delta_worker_exec_us,
            batch_roundtrip_us: delta_batch_roundtrip_us,
            batch_count: delta_batch_count,
            ..Default::default()
        }
    }

    fn worker_forward_stats(&mut self, stats: &SchedStats) -> Option<WorkerForwardStats> {
        let current = worker_forward_stats_snapshot(stats);
        let delta = delta_worker_forward_stats(&current, &self.last_worker_forward_stats);
        self.last_worker_forward_stats = current;
        (!delta.is_empty()).then_some(delta)
    }
}

fn worker_forward_stats_snapshot(stats: &SchedStats) -> WorkerForwardStats {
    let path_counts = stats
        .worker
        .spec_verify_path_counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let spec_verify_path_counts = path_counts.clone();
    let mode_counts = stats
        .worker
        .forward_mode_counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let mode_tokens = stats
        .worker
        .forward_mode_tokens
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let mode_us = stats
        .worker
        .forward_mode_us
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let component_us = stats
        .worker
        .forward_component_us
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    // surface the two maps the worker computes (attention backend /
    // cuda-graph runtime mode) so they reach the wire stats and Prometheus.
    let attention_backend_counts = stats
        .worker
        .attention_backend_counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let cuda_graph_runtime_mode_counts = stats
        .worker
        .cuda_graph_runtime_mode_counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    WorkerForwardStats {
        mode_counts,
        mode_tokens,
        mode_us,
        component_us,
        attention_launches: stats.worker.attention_launches.load(Ordering::Relaxed),
        attention_us: stats.worker.attention_us.load(Ordering::Relaxed),
        attention_backend_counts,
        cuda_graph_runtime_mode_counts,
        cuda_graph_captures: stats.worker.cuda_graph_captures.load(Ordering::Relaxed),
        cuda_graph_replays: stats.worker.cuda_graph_replays.load(Ordering::Relaxed),
        cuda_graph_misses: stats.worker.cuda_graph_misses.load(Ordering::Relaxed),
        cuda_graph_fallbacks: stats.worker.cuda_graph_fallbacks.load(Ordering::Relaxed),
        cuda_graph_unpadded_tokens: stats
            .worker
            .cuda_graph_unpadded_tokens
            .load(Ordering::Relaxed),
        cuda_graph_padded_tokens: stats
            .worker
            .cuda_graph_padded_tokens
            .load(Ordering::Relaxed),
        text_decode_token_relay_hits: stats
            .worker
            .text_decode_token_relay_hits
            .load(Ordering::Relaxed),
        text_decode_token_relay_misses: stats
            .worker
            .text_decode_token_relay_misses
            .load(Ordering::Relaxed),
        text_decode_position_relay_hits: stats
            .worker
            .text_decode_position_relay_hits
            .load(Ordering::Relaxed),
        text_decode_position_relay_misses: stats
            .worker
            .text_decode_position_relay_misses
            .load(Ordering::Relaxed),
        flashinfer_decode_plan_calls: stats
            .worker
            .flashinfer_decode_plan_calls
            .load(Ordering::Relaxed),
        flashinfer_decode_plan_reuses: stats
            .worker
            .flashinfer_decode_plan_reuses
            .load(Ordering::Relaxed),
        flashinfer_decode_plan_rows: stats
            .worker
            .flashinfer_decode_plan_rows
            .load(Ordering::Relaxed),
        flashinfer_decode_plan_indices: stats
            .worker
            .flashinfer_decode_plan_indices
            .load(Ordering::Relaxed),
        flashinfer_decode_graph_plan_calls: stats
            .worker
            .flashinfer_decode_graph_plan_calls
            .load(Ordering::Relaxed),
        flashinfer_decode_graph_plan_reuses: stats
            .worker
            .flashinfer_decode_graph_plan_reuses
            .load(Ordering::Relaxed),
        spec_verify_rows: stats.worker.spec_verify_rows.load(Ordering::Relaxed),
        spec_verify_draft_tokens: stats
            .worker
            .spec_verify_draft_tokens
            .load(Ordering::Relaxed),
        spec_verify_accepted_tokens: stats
            .worker
            .spec_verify_accepted_tokens
            .load(Ordering::Relaxed),
        spec_verify_rejected_tokens: stats
            .worker
            .spec_verify_rejected_tokens
            .load(Ordering::Relaxed),
        spec_verify_committed_tokens: stats
            .worker
            .spec_verify_committed_tokens
            .load(Ordering::Relaxed),
        spec_verify_path_counts,
    }
}

fn delta_worker_forward_stats(
    current: &WorkerForwardStats,
    previous: &WorkerForwardStats,
) -> WorkerForwardStats {
    WorkerForwardStats {
        mode_counts: delta_map(&current.mode_counts, &previous.mode_counts),
        mode_tokens: delta_map(&current.mode_tokens, &previous.mode_tokens),
        mode_us: delta_map(&current.mode_us, &previous.mode_us),
        component_us: delta_map(&current.component_us, &previous.component_us),
        attention_launches: current
            .attention_launches
            .saturating_sub(previous.attention_launches),
        attention_us: current.attention_us.saturating_sub(previous.attention_us),
        // per-update deltas for the two newly-surfaced maps.
        attention_backend_counts: delta_map(
            &current.attention_backend_counts,
            &previous.attention_backend_counts,
        ),
        cuda_graph_runtime_mode_counts: delta_map(
            &current.cuda_graph_runtime_mode_counts,
            &previous.cuda_graph_runtime_mode_counts,
        ),
        cuda_graph_captures: current
            .cuda_graph_captures
            .saturating_sub(previous.cuda_graph_captures),
        cuda_graph_replays: current
            .cuda_graph_replays
            .saturating_sub(previous.cuda_graph_replays),
        cuda_graph_misses: current
            .cuda_graph_misses
            .saturating_sub(previous.cuda_graph_misses),
        cuda_graph_fallbacks: current
            .cuda_graph_fallbacks
            .saturating_sub(previous.cuda_graph_fallbacks),
        cuda_graph_unpadded_tokens: current
            .cuda_graph_unpadded_tokens
            .saturating_sub(previous.cuda_graph_unpadded_tokens),
        cuda_graph_padded_tokens: current
            .cuda_graph_padded_tokens
            .saturating_sub(previous.cuda_graph_padded_tokens),
        text_decode_token_relay_hits: current
            .text_decode_token_relay_hits
            .saturating_sub(previous.text_decode_token_relay_hits),
        text_decode_token_relay_misses: current
            .text_decode_token_relay_misses
            .saturating_sub(previous.text_decode_token_relay_misses),
        text_decode_position_relay_hits: current
            .text_decode_position_relay_hits
            .saturating_sub(previous.text_decode_position_relay_hits),
        text_decode_position_relay_misses: current
            .text_decode_position_relay_misses
            .saturating_sub(previous.text_decode_position_relay_misses),
        flashinfer_decode_plan_calls: current
            .flashinfer_decode_plan_calls
            .saturating_sub(previous.flashinfer_decode_plan_calls),
        flashinfer_decode_plan_reuses: current
            .flashinfer_decode_plan_reuses
            .saturating_sub(previous.flashinfer_decode_plan_reuses),
        flashinfer_decode_plan_rows: current
            .flashinfer_decode_plan_rows
            .saturating_sub(previous.flashinfer_decode_plan_rows),
        flashinfer_decode_plan_indices: current
            .flashinfer_decode_plan_indices
            .saturating_sub(previous.flashinfer_decode_plan_indices),
        flashinfer_decode_graph_plan_calls: current
            .flashinfer_decode_graph_plan_calls
            .saturating_sub(previous.flashinfer_decode_graph_plan_calls),
        flashinfer_decode_graph_plan_reuses: current
            .flashinfer_decode_graph_plan_reuses
            .saturating_sub(previous.flashinfer_decode_graph_plan_reuses),
        spec_verify_rows: current
            .spec_verify_rows
            .saturating_sub(previous.spec_verify_rows),
        spec_verify_draft_tokens: current
            .spec_verify_draft_tokens
            .saturating_sub(previous.spec_verify_draft_tokens),
        spec_verify_accepted_tokens: current
            .spec_verify_accepted_tokens
            .saturating_sub(previous.spec_verify_accepted_tokens),
        spec_verify_rejected_tokens: current
            .spec_verify_rejected_tokens
            .saturating_sub(previous.spec_verify_rejected_tokens),
        spec_verify_committed_tokens: current
            .spec_verify_committed_tokens
            .saturating_sub(previous.spec_verify_committed_tokens),
        spec_verify_path_counts: delta_map(
            &current.spec_verify_path_counts,
            &previous.spec_verify_path_counts,
        ),
    }
}

fn delta_map(
    current: &BTreeMap<String, u64>,
    previous: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    current
        .iter()
        .filter_map(|(key, value)| {
            let delta = value.saturating_sub(*previous.get(key).unwrap_or(&0));
            (delta > 0).then(|| (key.clone(), delta))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_deltas_and_usage() {
        let stats = SchedStats::default();
        stats.kv_cache.num_blocks.store(100, Ordering::Relaxed);
        stats.kv_cache.free_blocks.store(75, Ordering::Relaxed);
        stats.general.running.store(3, Ordering::Relaxed);
        stats.general.pending.store(2, Ordering::Relaxed);
        stats.prefix.queries.store(10, Ordering::Relaxed);
        stats.prefix.hit_tokens.store(512, Ordering::Relaxed);
        stats.timing.queue_wait_count.store(3, Ordering::Relaxed);
        stats
            .timing
            .queue_wait_us_total
            .store(15_003, Ordering::Relaxed);
        stats
            .timing
            .queue_wait_us_max
            .store(9_000, Ordering::Relaxed);
        stats
            .worker
            .flashinfer_decode_plan_calls
            .store(5, Ordering::Relaxed);
        stats
            .worker
            .flashinfer_decode_plan_reuses
            .store(120, Ordering::Relaxed);
        stats
            .worker
            .forward_component_us
            .lock()
            .unwrap()
            .insert("text_model_forward".into(), 42);
        stats.worker.spec_verify_rows.store(4, Ordering::Relaxed);
        stats
            .worker
            .spec_verify_accepted_tokens
            .store(7, Ordering::Relaxed);
        stats
            .worker
            .spec_verify_path_counts
            .lock()
            .unwrap()
            .insert("greedy_device".into(), 4);

        let mut reporter = SchedStatsReporter::default();
        let wire = reporter.snapshot(&stats, 256);
        assert_eq!(wire.num_running_reqs, 3);
        assert_eq!(wire.num_waiting_reqs, 2);
        assert!((wire.kv_cache_usage - 0.25).abs() < 1e-9);
        assert_eq!(wire.num_admitted_reqs, 3);
        assert_eq!(wire.avg_queue_wait_us, 5_001);
        assert_eq!(wire.queue_wait_us_total, 15_003);
        assert_eq!(wire.max_queue_wait_us, 9_000);
        assert_eq!(wire.prefix_cache_stats.base.queries, 10 * 256);
        assert_eq!(wire.prefix_cache_stats.base.hits, 512);
        let worker = wire.worker_forward_stats.expect("worker stats present");
        assert_eq!(worker.component_us.get("text_model_forward"), Some(&42));
        assert_eq!(worker.flashinfer_decode_plan_calls, 5);
        assert_eq!(worker.flashinfer_decode_plan_reuses, 120);
        assert_eq!(worker.spec_verify_rows, 4);
        assert_eq!(worker.spec_verify_accepted_tokens, 7);
        assert_eq!(
            worker.spec_verify_path_counts.get("greedy_device"),
            Some(&4)
        );

        // Second snapshot with unchanged counters reports zero deltas.
        let wire2 = reporter.snapshot(&stats, 256);
        assert_eq!(wire2.num_admitted_reqs, 0);
        assert_eq!(wire2.avg_queue_wait_us, 0);
        assert_eq!(wire2.queue_wait_us_total, 0);
        assert_eq!(wire2.max_queue_wait_us, 9_000);
        assert_eq!(wire2.prefix_cache_stats.base.queries, 0);
        assert_eq!(wire2.prefix_cache_stats.base.hits, 0);
        assert!(wire2.worker_forward_stats.is_none());
    }

    /// the two maps the worker computes (attention backend / cuda-graph
    /// runtime mode) must survive the SchedStats -> wire snapshot/delta instead
    /// of being silently dropped before they can reach Prometheus.
    /// the directly-measured per-batch worker compute time and host
    /// round-trip latency must surface as per-update deltas in the wire stats so
    /// they reach Prometheus.
    #[test]
    fn snapshot_surfaces_batch_timing() {
        let stats = SchedStats::default();
        stats
            .timing
            .worker_exec_us_total
            .store(1_200, Ordering::Relaxed);
        stats
            .timing
            .batch_roundtrip_us_total
            .store(1_500, Ordering::Relaxed);
        stats.timing.batch_timing_count.store(3, Ordering::Relaxed);

        let mut reporter = SchedStatsReporter::default();
        let wire = reporter.snapshot(&stats, 256);
        assert_eq!(wire.worker_exec_us, 1_200);
        assert_eq!(wire.batch_roundtrip_us, 1_500);
        assert_eq!(wire.batch_count, 3);

        // Counters unchanged -> zero deltas on the next snapshot.
        let wire2 = reporter.snapshot(&stats, 256);
        assert_eq!(wire2.worker_exec_us, 0);
        assert_eq!(wire2.batch_roundtrip_us, 0);
        assert_eq!(wire2.batch_count, 0);
    }

    #[test]
    fn snapshot_surfaces_attention_backend_and_cuda_graph_runtime_mode_counts() {
        let stats = SchedStats::default();
        stats
            .worker
            .attention_backend_counts
            .lock()
            .unwrap()
            .insert("flashinfer".into(), 9);
        stats
            .worker
            .cuda_graph_runtime_mode_counts
            .lock()
            .unwrap()
            .insert("graph".into(), 5);

        let mut reporter = SchedStatsReporter::default();
        let wire = reporter.snapshot(&stats, 256);
        let worker = wire.worker_forward_stats.expect("worker stats present");
        assert_eq!(worker.attention_backend_counts.get("flashinfer"), Some(&9));
        assert_eq!(worker.cuda_graph_runtime_mode_counts.get("graph"), Some(&5));

        // Second snapshot with unchanged counters reports zero deltas (so both
        // maps are part of the delta/is_empty bookkeeping, not always-present).
        let wire2 = reporter.snapshot(&stats, 256);
        assert!(wire2.worker_forward_stats.is_none());
    }
}
