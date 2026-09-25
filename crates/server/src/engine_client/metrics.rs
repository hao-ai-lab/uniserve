//! Prometheus publication of engine-reported scheduler statistics and of
//! completed request lifecycles.
//!
//! `EngineClient` (in `in_process`) runs a once-per-second task that asks the
//! engine's `SchedulerStatsReporter` for a `SchedulerStats` snapshot and
//! passes it to [`record_scheduler_stats`] as engine `0`. The reporter already
//! turns cumulative scheduler counters into per-interval increments; this
//! module only maps fields onto the `SchedulerMetrics` families.
//!
//! When a request the engine accepted completes, the request registry maps
//! its final statistics onto the `RequestMetrics` families: its outcome
//! through `record_request_outcome`, and its token usage through
//! `record_request_usage` when its output reported final usage.

use uniserve_observability::{
    EngineBackendLabels, EngineComponentLabels, EngineDomainKindLabels, EngineDomainLabels,
    EngineLabels, EngineModeLabels, EnginePathLabels, FinishedReasonLabels, RequestMetrics,
    SchedulerMetrics, WaitingReasonLabels,
};

use crate::serving::{FinishStatus, RequestStatsSnapshot};
use uniserve_core::codec::stats::SchedulerStats;

const WAITING_REASON_CAPACITY: &str = "capacity";

/// Records the scheduler-stats-backed metrics for one engine at one point in
/// time.
///
/// Interval-delta fields of `stats` are added to Prometheus counters, while
/// point-in-time values (running and waiting requests, KV usage, active
/// credits) and lifetime maxima (`max_queue_wait_us`, `peak_credits`) are set
/// on gauges. `stats` must therefore be an interval snapshot: passing a
/// cumulative one, or the same snapshot twice, double-counts every counter.
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

    // Request-level scheduler state, admissions, and queue wait.
    metrics
        .scheduler_running
        .get_or_create(&labels)
        .set(stats.num_running_reqs);
    metrics
        .scheduler_waiting
        .get_or_create(&labels)
        .set(stats.num_waiting_reqs);
    // `SchedulerStats` carries no per-reason breakdown, so every waiting
    // request is published under the `capacity` reason and this family equals
    // `scheduler_waiting`.
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

    // Per-execution-domain accounting: credit gauges, then call, pressure,
    // and phase-time counters keyed by the domain name.
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
            ("launched", domain.launched_calls),
            ("completed", domain.completed_calls),
            ("predicated", domain.predicated_calls),
            ("error", domain.error_calls),
        ] {
            metrics
                .scheduler_domain_calls
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
            .scheduler_domain_completed_batches
            .get_or_create(&domain_labels)
            .inc_by(domain.completed_batches);
        for (kind, value) in [
            ("queue", domain.queue_us),
            ("launch", domain.launch_us),
            ("device", domain.device_us),
            ("completion", domain.completion_us),
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
    }

    // Prefix-cache query and hit counters, both in tokens.
    metrics
        .prefix_cache_queries
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.queries);
    metrics
        .prefix_cache_hits
        .get_or_create(&labels)
        .inc_by(stats.prefix_cache_stats.base.hits);

    // Worker-local forward/kernel counters. The reporter sends `None` when no
    // worker counter changed in the interval, and then none of these families
    // is touched.
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

    // Worker execution and host-observed round-trip time, in integer
    // microseconds; `batch_timing_count` counts the completed batches they
    // cover and is the denominator for per-batch averages.
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
}

/// Returns the `finished_reason` label value of a finish status.
pub(crate) fn finished_reason_name(status: &FinishStatus) -> &'static str {
    match status {
        FinishStatus::Stop { .. } => "stop",
        FinishStatus::Length => "length",
        FinishStatus::Abort => "abort",
        FinishStatus::Error => "error",
        FinishStatus::Repetition => "repetition",
    }
}

/// Records the outcome of one completed request into the request metric
/// families.
///
/// Counts the request under `finished_reason` and observes its end-to-end
/// latency, measured from the start of its lifecycle, and its queue time when
/// the engine reported scheduling timestamps.
pub(crate) fn record_request_outcome(
    metrics: &RequestMetrics,
    labels: &EngineLabels,
    finished_reason: &'static str,
    stats: &RequestStatsSnapshot,
) {
    metrics
        .request_success
        .get_or_create(&FinishedReasonLabels {
            model_name: labels.model_name.clone(),
            engine: labels.engine,
            finished_reason,
        })
        .inc();
    metrics
        .e2e_request_latency_seconds
        .get_or_create(labels)
        .observe(seconds(stats.timings.total_us));
    if let Some(queue_us) = stats.timings.queue_us {
        metrics
            .request_queue_time_seconds
            .get_or_create(labels)
            .observe(seconds(queue_us));
    }
}

/// Records the final usage of one completed request into the request metric
/// families.
///
/// `stats` must carry the totals of the request's `Usage` event. Its prompt
/// and generated tokens are added to the cumulative token counters and
/// observed in the per-request token histograms. Time to first token is
/// measured from the start of the request's lifecycle to its first visible
/// output, and observed only for a request that generated tokens and produced
/// visible output.
pub(crate) fn record_request_usage(
    metrics: &RequestMetrics,
    labels: &EngineLabels,
    stats: &RequestStatsSnapshot,
) {
    // Hidden protocol tokens are generated tokens too; together with the
    // visible ones they make up the engine's completion tokens.
    let prompt_tokens = u64::from(stats.prompt_tokens);
    let generation_tokens =
        u64::from(stats.visible_output_tokens) + u64::from(stats.internal_tokens);

    metrics
        .prompt_tokens
        .get_or_create(labels)
        .inc_by(prompt_tokens);
    metrics
        .generation_tokens
        .get_or_create(labels)
        .inc_by(generation_tokens);

    metrics
        .request_prompt_tokens
        .get_or_create(labels)
        .observe(prompt_tokens as f64);
    metrics
        .request_generation_tokens
        .get_or_create(labels)
        .observe(generation_tokens as f64);
    // A request samples one sequence, so its largest sequence generation is
    // its own.
    metrics
        .request_max_num_generation_tokens
        .get_or_create(labels)
        .observe(generation_tokens as f64);

    if generation_tokens > 0
        && let Some(first_output_us) = stats.timings.first_visible_output_us
    {
        metrics
            .time_to_first_token_seconds
            .get_or_create(labels)
            .observe(seconds(first_output_us));
    }
}

/// Converts a microsecond duration to the seconds the histograms observe.
fn seconds(microseconds: u64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::codec::stats::DomainSchedulerStats;
    use uniserve_observability::Metrics;

    // Per-domain scheduler statistics reach the rendered exposition under
    // their `domain` label, with call and phase-time breakdowns under `kind`.
    #[test]
    fn domain_accounting_reaches_openmetrics() {
        let metrics = Metrics::new();
        let stats = SchedulerStats {
            domain_stats: vec![DomainSchedulerStats {
                domain: "decode".to_string(),
                active_credits: 2,
                peak_credits: 5,
                launched_calls: 7,
                completed_calls: 6,
                error_calls: 1,
                backpressure_events: 3,
                reclaimed_credits: 6,
                completed_batches: 4,
                queue_us: 11,
                launch_us: 13,
                device_us: 17,
                completion_us: 19,
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
            line.starts_with("uniserve:scheduler_domain_calls_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"error\"")
                && line.ends_with(" 1")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_time_us_total")
                && line.contains("domain=\"decode\"")
                && line.contains("kind=\"device\"")
                && line.ends_with(" 17")
        }));
        assert!(rendered.lines().any(|line| {
            line.starts_with("uniserve:scheduler_domain_completed_batches")
                && line.contains("domain=\"decode\"")
                && line.ends_with(" 4")
        }));
    }
}
