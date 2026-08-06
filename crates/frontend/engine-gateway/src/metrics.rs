//! Prometheus recording for engine-reported scheduler statistics.

use uniserve_observability::{
    EngineBackendLabels, EngineComponentLabels, EngineLabels, EngineModeLabels, EnginePathLabels,
    SchedulerMetrics, WaitingReasonLabels,
};

use crate::protocol::stats::SchedulerStats;

const WAITING_REASON_CAPACITY: &str = "capacity";

/// Record the scheduler-stats-backed metrics for one engine at one point in
/// time.
pub fn record_scheduler_stats(
    metrics: &SchedulerMetrics,
    model_name: impl Into<String>,
    engine: u32,
    stats: &SchedulerStats,
) {
    let model_name = model_name.into();
    let labels = EngineLabels {
        model_name: model_name.clone(),
        engine,
    };

    // Scheduler state gauges.
    metrics
        .scheduler_running
        .get_or_create(&labels)
        .set(stats.num_running_reqs);
    metrics
        .scheduler_waiting
        .get_or_create(&labels)
        .set(stats.num_waiting_reqs);
    metrics
        .scheduler_waiting_by_reason
        .get_or_create(&WaitingReasonLabels {
            model_name: model_name.clone(),
            engine,
            reason: WAITING_REASON_CAPACITY,
        })
        .set(stats.num_waiting_reqs);
    metrics
        .kv_cache_usage
        .get_or_create(&labels)
        .set(stats.kv_cache_usage);
    metrics
        .scheduler_admitted
        .get_or_create(&labels)
        .inc_by(stats.num_admitted_reqs);
    metrics
        .scheduler_queue_wait_us
        .get_or_create(&labels)
        .inc_by(stats.queue_wait_us_total);
    metrics
        .scheduler_queue_wait_max_us
        .get_or_create(&labels)
        .set(stats.max_queue_wait_us);

    // Prefix-cache counters, including the connector-backed external cache path.
    metrics
        .prefix_cache_queries
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.queries);
    metrics
        .prefix_cache_hits
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.hits);

    // Worker-local forward/kernel counters.
    if let Some(worker_stats) = &stats.worker_forward_stats {
        for (mode, count) in &worker_stats.mode_counts {
            metrics
                .worker_forward_mode_counts
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*count);
        }
        for (mode, tokens) in &worker_stats.mode_tokens {
            metrics
                .worker_forward_mode_tokens
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*tokens);
        }
        for (mode, us) in &worker_stats.mode_us {
            metrics
                .worker_forward_mode_us
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*us);
        }
        for (component, us) in &worker_stats.component_us {
            metrics
                .worker_forward_component_us
                .get_or_create(&EngineComponentLabels {
                    model_name: model_name.clone(),
                    engine,
                    component: component.clone(),
                })
                .inc_by(*us);
        }
        // the two maps the scheduler folds all
        // the way through to Prometheus.
        for (backend, count) in &worker_stats.attention_backend_counts {
            metrics
                .worker_attention_backend_counts
                .get_or_create(&EngineBackendLabels {
                    model_name: model_name.clone(),
                    engine,
                    backend: backend.clone(),
                })
                .inc_by(*count);
        }
        for (mode, count) in &worker_stats.cuda_graph_runtime_mode_counts {
            metrics
                .worker_cuda_graph_runtime_mode_counts
                .get_or_create(&EngineModeLabels {
                    model_name: model_name.clone(),
                    engine,
                    mode: mode.clone(),
                })
                .inc_by(*count);
        }
        metrics
            .worker_attention_launches
            .get_or_create(&labels)
            .inc_by(worker_stats.attention_launches);
        metrics
            .worker_attention_us
            .get_or_create(&labels)
            .inc_by(worker_stats.attention_us);
        metrics
            .worker_cuda_graph_captures
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_captures);
        metrics
            .worker_cuda_graph_replays
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_replays);
        metrics
            .worker_cuda_graph_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_misses);
        metrics
            .worker_cuda_graph_fallbacks
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_fallbacks);
        metrics
            .worker_cuda_graph_unpadded_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_unpadded_tokens);
        metrics
            .worker_cuda_graph_padded_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.cuda_graph_padded_tokens);
        metrics
            .worker_text_decode_token_relay_hits
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_token_relay_hits);
        metrics
            .worker_text_decode_token_relay_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_token_relay_misses);
        metrics
            .worker_text_decode_position_relay_hits
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_position_relay_hits);
        metrics
            .worker_text_decode_position_relay_misses
            .get_or_create(&labels)
            .inc_by(worker_stats.text_decode_position_relay_misses);
        metrics
            .worker_flashinfer_decode_plan_calls
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_calls);
        metrics
            .worker_flashinfer_decode_plan_reuses
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_reuses);
        metrics
            .worker_flashinfer_decode_plan_rows
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_rows);
        metrics
            .worker_flashinfer_decode_plan_indices
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_plan_indices);
        metrics
            .worker_flashinfer_decode_graph_plan_calls
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_graph_plan_calls);
        metrics
            .worker_flashinfer_decode_graph_plan_reuses
            .get_or_create(&labels)
            .inc_by(worker_stats.flashinfer_decode_graph_plan_reuses);
        metrics
            .worker_spec_verify_rows
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_rows);
        metrics
            .worker_spec_verify_draft_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_draft_tokens);
        metrics
            .worker_spec_verify_accepted_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_accepted_tokens);
        metrics
            .worker_spec_verify_rejected_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_rejected_tokens);
        metrics
            .worker_spec_verify_committed_tokens
            .get_or_create(&labels)
            .inc_by(worker_stats.spec_verify_committed_tokens);
        for (path, count) in &worker_stats.spec_verify_path_counts {
            metrics
                .worker_spec_verify_path_counts
                .get_or_create(&EnginePathLabels {
                    model_name: model_name.clone(),
                    engine,
                    path: path.clone(),
                })
                .inc_by(*count);
        }
    }

    // directly-measured batch latency (worker compute + host
    // round-trip), now surfaced to Prometheus instead of the JSON trace only.

    // the worker/scheduler latency counters use cumulative *microseconds*
    // (the `_us` suffix is the unit contract carried on the wire), deliberately
    // distinct from the per-request second-valued histograms in
    // `uniserve_observability::request`. They are not a competing unit system:
    // a dashboard reconciles them as `seconds = <_us counter> / 1e6`. The
    // microsecond integer counter is kept (rather than a lossy us->s cast at
    // ingest) so sub-microsecond cumulative precision is preserved.
    metrics
        .worker_exec_us
        .get_or_create(&labels)
        .inc_by(stats.worker_exec_us);
    metrics
        .batch_roundtrip_us
        .get_or_create(&labels)
        .inc_by(stats.batch_roundtrip_us);
    metrics
        .batch_timing_count
        .get_or_create(&labels)
        .inc_by(stats.batch_count);

    // Per-engine performance / MFU counters.
    if let Some(perf_stats) = &stats.perf_stats
        && (perf_stats.num_flops_per_gpu != 0
            || perf_stats.num_read_bytes_per_gpu != 0
            || perf_stats.num_write_bytes_per_gpu != 0)
    {
        metrics
            .estimated_flops_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_flops_per_gpu);
        metrics
            .estimated_read_bytes_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_read_bytes_per_gpu);
        metrics
            .estimated_write_bytes_per_gpu
            .get_or_create(&labels)
            .inc_by(perf_stats.num_write_bytes_per_gpu);
    }

    // Sampled KV-cache residency histograms.
    if !stats.kv_cache_eviction_events.is_empty() {
        let kv_block_lifetime_seconds = metrics.kv_block_lifetime_seconds.get_or_create(&labels);
        let kv_block_idle_before_evict_seconds = metrics
            .kv_block_idle_before_evict_seconds
            .get_or_create(&labels);
        let kv_block_reuse_gap_seconds = metrics.kv_block_reuse_gap_seconds.get_or_create(&labels);

        for event in &stats.kv_cache_eviction_events {
            kv_block_lifetime_seconds.observe(event.lifetime_seconds);
            kv_block_idle_before_evict_seconds.observe(event.idle_seconds);
            for reuse_gap_seconds in &event.reuse_gaps_seconds {
                kv_block_reuse_gap_seconds.observe(*reuse_gap_seconds);
            }
        }
    }
}
