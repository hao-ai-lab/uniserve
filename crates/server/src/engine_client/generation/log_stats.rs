//! Periodic logging of generation throughput and scheduler occupancy.
//!
//! The logger only reads the process-wide `METRICS` registry and updates no
//! metric value, although resolving its handles creates any of its series
//! that do not exist yet. Scheduler gauges and prefix-cache counters are
//! written by the engine client's scheduler-stats export task
//! (`EngineClient::from_core` through
//! `engine_client::metrics::record_scheduler_stats`).
//! `ServingRuntime::new` starts the logger when `Config::log_stats` is set.

use std::fmt::Write;
use std::time::{Duration, Instant};

use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info};
use uniserve_observability::{
    EngineLabels, F64Gauge, METRICS, PromptTokenSourceLabels, U64Counter, U64Gauge,
};

/// Period between log lines. Throughput is averaged over the measured time
/// between samples, which can exceed this period when a tick is delayed.
const LOG_STATS_INTERVAL: Duration = Duration::from_secs(10);

/// Cached, cloned metric handles for one engine. Each clone shares the same
/// underlying `Arc<Atomic*>` as the prometheus `Family` entry, so reads go
/// straight to the atomic with no lock.
struct EngineMetrics {
    // Counters for throughput deltas.
    prompt_tokens_computed: U64Counter,
    generation_tokens: U64Counter,
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,

    // Gauges for instantaneous scheduler state.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    kv_cache_usage: F64Gauge,
}

/// Cumulative counter values summed across engines at one sample, used to
/// compute per-interval deltas.
struct CounterSnapshot {
    prompt_tokens: u64,
    generation_tokens: u64,
    prefix_cache_queries: u64,
    prefix_cache_hits: u64,
}

/// Periodic logger for throughput and scheduler state.
///
/// Spawns a background task that logs throughput and scheduler state at a fixed
/// interval. When idle (both current and previous throughputs are zero), logs
/// at DEBUG level. When load drops to zero, emits one final INFO-level line
/// before going quiet. Dropping the logger aborts the task.
pub(crate) struct StatsLogger {
    _task: AbortOnDropHandle<()>,
}

impl StatsLogger {
    /// Starts the background stats logging task for engines
    /// `0..engine_count`.
    ///
    /// Must be called within a Tokio runtime, since it spawns onto the current
    /// one.
    pub(crate) fn start(model_name: String, engine_count: usize) -> Self {
        let task = AbortOnDropHandle::new(tokio::spawn(async move {
            run_stats_logger(model_name, engine_count).await;
        }));
        Self { _task: task }
    }
}

/// Resolves and clones all metric handles once, so each interval reads the
/// atomics without locking the metric families.
fn resolve_engine_metrics(model_name: &str, engine_count: usize) -> Vec<EngineMetrics> {
    let m = &METRICS;
    (0..engine_count as u32)
        .map(|engine| {
            let el = EngineLabels {
                model_name: model_name.to_string(),
                engine,
            };
            let pt = PromptTokenSourceLabels {
                model_name: model_name.to_string(),
                engine,
                source: "local_compute",
            };
            EngineMetrics {
                // Prompt throughput reads only the `local_compute` source series;
                // prompt tokens recorded under any other source do not count.
                prompt_tokens_computed: m.request.prompt_tokens_by_source.get_or_create_owned(&pt),
                generation_tokens: m.request.generation_tokens.get_or_create_owned(&el),
                prefix_cache_queries: m.scheduler.prefix_cache_queries.get_or_create_owned(&el),
                prefix_cache_hits: m.scheduler.prefix_cache_hits.get_or_create_owned(&el),
                scheduler_running: m.scheduler.scheduler_running.get_or_create_owned(&el),
                scheduler_waiting: m.scheduler.scheduler_waiting.get_or_create_owned(&el),
                kv_cache_usage: m.scheduler.kv_cache_usage.get_or_create_owned(&el),
            }
        })
        .collect()
}

/// Samples cumulative engine metrics and periodically logs interval throughput and occupancy.
///
/// Runs until its task is aborted.
async fn run_stats_logger(model_name: String, engine_count: usize) {
    let engines = resolve_engine_metrics(&model_name, engine_count);

    let mut interval = tokio::time::interval(LOG_STATS_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires immediately; skip it so the first log is after one full
    // interval.
    interval.tick().await;

    let mut prev = read_counters(&engines);
    let mut last_log_time = Instant::now();
    let mut last_prompt_throughput: f64 = 0.0;
    let mut last_generation_throughput: f64 = 0.0;

    let mut msg = String::new();
    loop {
        interval.tick().await;

        let now = Instant::now();
        let elapsed = now.duration_since(last_log_time).as_secs_f64();
        if elapsed <= 0.0 {
            continue;
        }

        let curr = read_counters(&engines);

        let prompt_throughput =
            curr.prompt_tokens.wrapping_sub(prev.prompt_tokens) as f64 / elapsed;
        let generation_throughput =
            curr.generation_tokens.wrapping_sub(prev.generation_tokens) as f64 / elapsed;

        // Idle = both current and previous throughputs are zero, so the first
        // zero interval after load still logs at INFO.
        let is_idle = prompt_throughput == 0.0
            && generation_throughput == 0.0
            && last_prompt_throughput == 0.0
            && last_generation_throughput == 0.0;

        // Read scheduler gauges (aggregate across engines).
        let (num_running, num_waiting, kv_cache_usage) = read_scheduler_gauges(&engines);

        // Compute prefix cache hit rate over this interval.
        let delta_queries = curr
            .prefix_cache_queries
            .wrapping_sub(prev.prefix_cache_queries);
        let prefix_cache_hit_rate = if delta_queries > 0 {
            let delta_hits = curr.prefix_cache_hits.wrapping_sub(prev.prefix_cache_hits);
            delta_hits as f64 / delta_queries as f64 * 100.0
        } else {
            0.0
        };

        // Build the log line.
        msg.clear();
        if write!(
            msg,
            "Avg prompt tput: {prompt_throughput:.1} toks/s, \
             Avg generation tput: {generation_throughput:.1} toks/s, \
             Reqs Running: {num_running}, \
             Waiting: {num_waiting}, \
             GPU KV cache used: {:.1}%, \
             Prefix cache hit rate: {prefix_cache_hit_rate:.1}%",
            kv_cache_usage * 100.0,
        )
        .is_err()
        {
            msg.clear();
        }

        if is_idle {
            debug!("{msg}");
        } else {
            info!("{msg}");
        }

        last_prompt_throughput = prompt_throughput;
        last_generation_throughput = generation_throughput;
        last_log_time = now;
        prev = curr;
    }
}

/// Reads the current cumulative counter values for throughput computation.
fn read_counters(engines: &[EngineMetrics]) -> CounterSnapshot {
    let mut snap = CounterSnapshot {
        prompt_tokens: 0,
        generation_tokens: 0,
        prefix_cache_queries: 0,
        prefix_cache_hits: 0,
    };
    for e in engines {
        snap.prompt_tokens += e.prompt_tokens_computed.get();
        snap.generation_tokens += e.generation_tokens.get();
        snap.prefix_cache_queries += e.prefix_cache_queries.get();
        snap.prefix_cache_hits += e.prefix_cache_hits.get();
    }
    snap
}

/// Reads the current scheduler gauge values, aggregated across engines.
///
/// Running and waiting counts are summed; KV cache usage is the mean of the
/// per-engine usage fractions (0.0 to 1.0), or 0.0 with no engines.
fn read_scheduler_gauges(engines: &[EngineMetrics]) -> (u64, u64, f64) {
    let mut num_running = 0u64;
    let mut num_waiting = 0u64;
    let mut kv_cache_usage_sum = 0.0f64;

    for e in engines {
        num_running += e.scheduler_running.get();
        num_waiting += e.scheduler_waiting.get();
        kv_cache_usage_sum += e.kv_cache_usage.get();
    }

    let kv_cache_usage = if !engines.is_empty() {
        kv_cache_usage_sum / engines.len() as f64
    } else {
        0.0
    };

    (num_running, num_waiting, kv_cache_usage)
}
