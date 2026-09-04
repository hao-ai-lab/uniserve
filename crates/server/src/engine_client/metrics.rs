//! Prometheus publication of engine-reported scheduler statistics.

use uniserve_observability::{
    EngineBackendLabels, EngineComponentLabels, EngineDomainKindLabels, EngineDomainLabels,
    EngineLabels, EngineModeLabels, EnginePathLabels, SchedulerMetrics, WaitingReasonLabels,
};

use uniserve_core::codec::stats::SchedulerStats;

const WAITING_REASON_CAPACITY: &str = "capacity";

/// Records the scheduler-stats-backed metrics for one engine at one point in
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
    for domain in &stats.domain_stats {
        let domain_labels = EngineDomainLabels {
            model_name: model_name.clone(),
            engine,
            domain: domain.domain.clone(),
        };
        metrics
            .scheduler_domain_active_credits
            .get_or_create(&domain_labels)
            .set(domain.active_credits);
        metrics
            .scheduler_domain_peak_credits
            .get_or_create(&domain_labels)
            .set(domain.peak_credits);
        for (kind, value) in [
            ("launched", domain.launched_operations),
            ("completed", domain.completed_operations),
            ("predicated", domain.predicated_operations),
            ("error", domain.error_operations),
        ] {
            metrics
                .scheduler_domain_operations
                .get_or_create(&EngineDomainKindLabels {
                    model_name: model_name.clone(),
                    engine,
                    domain: domain.domain.clone(),
                    kind: kind.to_string(),
                })
                .inc_by(value);
        }
        metrics
            .scheduler_domain_backpressure
            .get_or_create(&domain_labels)
            .inc_by(domain.backpressure_events);
        metrics
            .scheduler_domain_reclaimed_credits
            .get_or_create(&domain_labels)
            .inc_by(domain.reclaimed_credits);
        metrics
            .scheduler_domain_completed_runs
            .get_or_create(&domain_labels)
            .inc_by(domain.completed_runs);
        for (kind, value) in [
            ("queue", domain.queue_us),
            ("launch", domain.launch_us),
            ("device", domain.device_us),
            ("completion", domain.completion_us),
            ("co_resident", domain.co_resident_us),
        ] {
            metrics
                .scheduler_domain_time_us
                .get_or_create(&EngineDomainKindLabels {
                    model_name: model_name.clone(),
                    engine,
                    domain: domain.domain.clone(),
                    kind: kind.to_string(),
                })
                .inc_by(value);
        }
        metrics
            .scheduler_domain_co_resident_runs
            .get_or_create(&domain_labels)
            .inc_by(domain.co_resident_runs);
    }

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

    // Worker and round-trip latency remain cumulative integer microseconds to
    // preserve precision. Dashboards convert these counters to seconds when
    // combining them with per-request second-valued histograms.
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

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::codec::stats::DomainSchedulerStats;
    use uniserve_observability::Metrics;

    #[test]
    fn domain_accounting_reaches_openmetrics() {
        let metrics = Metrics::new();
        let stats = SchedulerStats {
            domain_stats: vec![DomainSchedulerStats {
                domain: "decode".to_string(),
                active_credits: 2,
                peak_credits: 5,
                launched_operations: 7,
                completed_operations: 6,
                error_operations: 1,
                backpressure_events: 3,
                reclaimed_credits: 6,
                completed_runs: 4,
                co_resident_runs: 2,
                queue_us: 11,
                launch_us: 13,
                device_us: 17,
                completion_us: 19,
                co_resident_us: 31,
                ..Default::default()
            }],
            ..Default::default()
        };
        record_scheduler_stats(&metrics.scheduler, "model", 0, &stats);
        let rendered = metrics.render().expect("metrics render");
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_active_credits")
                && line.contains("domain=\"decode\"")
                && line.ends_with(" 2")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_operations_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"error\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_time_us_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"co_resident\"")
                && line.ends_with(" 31")
        }));
    }
}
