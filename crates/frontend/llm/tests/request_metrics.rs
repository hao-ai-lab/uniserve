//! Behavioral tests for `RequestMetricsTracker`, asserted purely against the
//! Prometheus text emitted by `METRICS.render()`.
//!
//! `RequestMetricsTracker` is a `pub(crate)` type, so these tests exercise it
//! through the only public path that drives it: `Llm::generate(..)` feeding a
//! mock engine, with the resulting `GenerateOutputStream` polled to completion.
//!
//! Determinism: the engine-side timestamps that feed the time-derived metrics
//! (queue time, prefill time, inter-token latency, decode time, and
//! time-per-output-token) are injected via the mock engine's per-batch
//! `timestamp` and per-event `timestamp` fields, so they never touch the wall
//! clock. TTFT and end-to-end latency are derived from `received_at`
//! (wall-clock), so for those we only assert the deterministic observation
//! *count*, never a value. Each test uses a unique model name so its label set
//! is isolated within the process-global `METRICS` registry.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;

use futures::StreamExt as _;
use uniserve_engine_client::EngineCoreClient;
use uniserve_engine_client::protocol::stats::PrefillStats;
use uniserve_engine_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs,
};
use uniserve_engine_client::test_utils::spawn_mock_engine_task;
use uniserve_llm::{FinishReason, GenerateRequest, Llm};
use uniserve_observability::METRICS;
use uuid::Uuid;

/// Unique per-test model name so each test owns an isolated label set in the
/// process-global metrics registry.
fn unique_model_name(prefix: &str) -> String {
    format!("{prefix}-{}", Uuid::new_v4().simple())
}

/// Build a minimal engine output for one request, with the given new tokens,
/// finish reason, optional lifecycle events, and optional prefill stats.
fn engine_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason: None,
        events,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats,
        routed_experts: None,
        num_nans_in_logits: 0,
        native: None,
    }
}

fn queued_at(timestamp: f64) -> EngineCoreEvent {
    EngineCoreEvent {
        r#type: EngineCoreEventType::Queued,
        timestamp,
    }
}

fn scheduled_at(timestamp: f64) -> EngineCoreEvent {
    EngineCoreEvent {
        r#type: EngineCoreEventType::Scheduled,
        timestamp,
    }
}

fn preempted_at(timestamp: f64) -> EngineCoreEvent {
    EngineCoreEvent {
        r#type: EngineCoreEventType::Preempted,
        timestamp,
    }
}

fn generate_request(request_id: &str, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: vec![11, 22],
        sampling_params: uniserve_engine_client::protocol::EngineCoreSamplingParams {
            max_tokens,
            ..uniserve_engine_client::protocol::EngineCoreSamplingParams::for_test()
        },
        mm_features: None,
        // Leave arrival_time unset so the engine client stamps it; the
        // time-derived metrics asserted here never depend on arrival_time.
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    }
}

/// Drive a single request to completion through the mock engine, returning the
/// rendered Prometheus text after the request has finished. `batches` is the
/// sequence of output batches the mock engine emits; the final batch must carry
/// the terminal finish reason and mark the request finished.
async fn render_after_request(
    model_name: &str,
    request_id: &str,
    max_tokens: u32,
    batches: Vec<EngineCoreOutputs>,
) -> String {
    let (client, mock) = EngineCoreClient::connect_mock(model_name.to_string());

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(mock, move |mut mock| {
        Box::pin(async move {
            let request = mock.recv_request().await;
            for mut batch in batches {
                // Rewrite each output's request_id to the engine-assigned id and
                // ensure the finished set references it once terminal.
                let is_terminal = batch.outputs.iter().any(|o| o.finish_reason.is_some());
                for output in &mut batch.outputs {
                    output.request_id = request.request_id.clone();
                }
                if is_terminal {
                    batch.finished_requests = Some(BTreeSet::from([request.request_id.clone()]));
                }
                mock.send_outputs(batch);
            }
        })
    });

    let llm = Llm::new(client);
    let mut stream = llm
        .generate(generate_request(request_id, max_tokens))
        .await
        .unwrap();
    while stream.next().await.transpose().unwrap().is_some() {}

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    let rendered = METRICS.render().unwrap();
    llm.shutdown().await.unwrap();
    rendered
}

/// A two-batch lifecycle: prefill batch carries Queued+Scheduled events and the
/// prefill stats; the second batch is the terminal decode batch.
///
/// queue time = scheduled_ts - queued_ts, prefill time = first_token_ts -
/// scheduled_ts, decode time = last_token_ts - first_token_ts, where the
/// per-batch timestamps are the injected `batch_timestamp` values.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_time_is_scheduled_minus_queued_timestamp() {
    let model = unique_model_name("queue-time");
    let rendered = render_after_request(
        &model,
        "req-queue",
        8,
        vec![
            EngineCoreOutputs {
                engine_index: 1,
                timestamp: 100.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(90.0), scheduled_at(95.5)]),
                    Some(PrefillStats {
                        num_prompt_tokens: 2,
                        num_computed_tokens: 2,
                        ..Default::default()
                    }),
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 1,
                timestamp: 101.0,
                outputs: vec![engine_output(
                    "",
                    vec![2],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    // queue time = 95.5 - 90.0 = 5.5 seconds, observed exactly once.
    assert!(
        rendered.contains(&format!(
            "uniserve:request_queue_time_seconds_sum{{model_name=\"{model}\",engine=\"1\"}} 5.5"
        )),
        "expected queue_time_sum 5.5 in:\n{rendered}"
    );
    assert!(rendered.contains(&format!(
        "uniserve:request_queue_time_seconds_count{{model_name=\"{model}\",engine=\"1\"}} 1"
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefill_time_is_first_token_minus_scheduled_timestamp() {
    let model = unique_model_name("prefill-time");
    let rendered = render_after_request(
        &model,
        "req-prefill",
        8,
        vec![
            // first_token_ts = batch_timestamp of the prefill batch = 200.0
            EngineCoreOutputs {
                engine_index: 2,
                timestamp: 200.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(180.0), scheduled_at(190.0)]),
                    Some(PrefillStats {
                        num_prompt_tokens: 2,
                        num_computed_tokens: 2,
                        ..Default::default()
                    }),
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 2,
                timestamp: 205.0,
                outputs: vec![engine_output(
                    "",
                    vec![2],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    // prefill time = first_token_ts(200.0) - scheduled_ts(190.0) = 10.0.
    assert!(
        rendered.contains(&format!(
            "uniserve:request_prefill_time_seconds_sum{{model_name=\"{model}\",engine=\"2\"}} 10.0"
        )),
        "expected prefill_time_sum 10.0 in:\n{rendered}"
    );
    assert!(rendered.contains(&format!(
        "uniserve:request_prefill_time_seconds_count{{model_name=\"{model}\",engine=\"2\"}} 1"
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inter_token_latency_sums_decode_step_gaps() {
    let model = unique_model_name("itl");
    let rendered = render_after_request(
        &model,
        "req-itl",
        8,
        vec![
            // prefill batch -> first_token_ts = 10.0, no ITL recorded yet.
            EngineCoreOutputs {
                engine_index: 3,
                timestamp: 10.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(1.0), scheduled_at(2.0)]),
                    None,
                )],
                ..Default::default()
            },
            // first decode batch -> ITL gap = 12.0 - 10.0 = 2.0.
            EngineCoreOutputs {
                engine_index: 3,
                timestamp: 12.0,
                outputs: vec![engine_output("", vec![2], None, None, None)],
                ..Default::default()
            },
            // second decode batch (terminal) -> ITL gap = 15.0 - 12.0 = 3.0.
            EngineCoreOutputs {
                engine_index: 3,
                timestamp: 15.0,
                outputs: vec![engine_output(
                    "",
                    vec![3],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    // Two inter-token gaps observed: 2.0 and 3.0; sum = 5.0.
    assert!(rendered.contains(&format!(
        "uniserve:inter_token_latency_seconds_count{{model_name=\"{model}\",engine=\"3\"}} 2"
    )));
    assert!(
        rendered.contains(&format!(
            "uniserve:inter_token_latency_seconds_sum{{model_name=\"{model}\",engine=\"3\"}} 5.0"
        )),
        "expected inter_token_latency_sum 5.0 in:\n{rendered}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decode_time_is_last_minus_first_token_timestamp() {
    let model = unique_model_name("decode-time");
    let rendered = render_after_request(
        &model,
        "req-decode",
        8,
        vec![
            // first_token_ts = 50.0
            EngineCoreOutputs {
                engine_index: 6,
                timestamp: 50.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(40.0), scheduled_at(45.0)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 6,
                timestamp: 53.0,
                outputs: vec![engine_output("", vec![2], None, None, None)],
                ..Default::default()
            },
            // last_token_ts = 58.0 -> decode time = 58.0 - 50.0 = 8.0
            EngineCoreOutputs {
                engine_index: 6,
                timestamp: 58.0,
                outputs: vec![engine_output(
                    "",
                    vec![3],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    assert!(
        rendered.contains(&format!(
            "uniserve:request_decode_time_seconds_sum{{model_name=\"{model}\",engine=\"6\"}} 8.0"
        )),
        "expected decode_time_sum 8.0 in:\n{rendered}"
    );
    assert!(rendered.contains(&format!(
        "uniserve:request_decode_time_seconds_count{{model_name=\"{model}\",engine=\"6\"}} 1"
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn time_per_output_token_divides_decode_span_by_generated_minus_one() {
    let model = unique_model_name("tpot");
    // 3 generation tokens total (1 prefill + 2 decode steps), decode span =
    // last_token_ts(20.0) - first_token_ts(10.0) = 10.0, divided by
    // (num_generation_tokens - 1) = 2 -> 5.0 per output token.
    let rendered = render_after_request(
        &model,
        "req-tpot",
        8,
        vec![
            EngineCoreOutputs {
                engine_index: 7,
                timestamp: 10.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(1.0), scheduled_at(2.0)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 7,
                timestamp: 15.0,
                outputs: vec![engine_output("", vec![2], None, None, None)],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 7,
                timestamp: 20.0,
                outputs: vec![engine_output(
                    "",
                    vec![3],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    assert!(
        rendered.contains(&format!(
            "uniserve:request_time_per_output_token_seconds_sum{{model_name=\"{model}\",engine=\"7\"}} 5.0"
        )),
        "expected tpot_sum 5.0 in:\n{rendered}"
    );
    assert!(rendered.contains(&format!(
        "uniserve:request_time_per_output_token_seconds_count{{model_name=\"{model}\",engine=\"7\"}} 1"
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generation_token_counter_sums_new_tokens_across_batches() {
    let model = unique_model_name("gen-tokens");
    let rendered = render_after_request(
        &model,
        "req-gen",
        16,
        vec![
            EngineCoreOutputs {
                engine_index: 8,
                timestamp: 1.0,
                outputs: vec![engine_output(
                    "",
                    vec![1, 2], // 2 tokens
                    None,
                    Some(vec![queued_at(0.1), scheduled_at(0.2)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 8,
                timestamp: 2.0,
                outputs: vec![engine_output("", vec![3, 4, 5], None, None, None)], // 3 tokens
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 8,
                timestamp: 3.0,
                outputs: vec![engine_output(
                    "",
                    vec![6], // 1 token
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    // Cumulative generation_tokens counter = 2 + 3 + 1 = 6.
    assert!(
        rendered.contains(&format!(
            "uniserve:generation_tokens_total{{model_name=\"{model}\",engine=\"8\"}} 6"
        )),
        "expected generation_tokens_total 6 in:\n{rendered}"
    );
    // Per-request generation token histogram observes the same total (6) once.
    assert!(rendered.contains(&format!(
        "uniserve:request_generation_tokens_sum{{model_name=\"{model}\",engine=\"8\"}} 6.0"
    )));
    assert!(rendered.contains(&format!(
        "uniserve:request_generation_tokens_count{{model_name=\"{model}\",engine=\"8\"}} 1"
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preemption_counter_increments_once_per_preempted_event() {
    let model = unique_model_name("preempt");
    let rendered = render_after_request(
        &model,
        "req-preempt",
        16,
        vec![
            EngineCoreOutputs {
                engine_index: 9,
                timestamp: 1.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(0.1), scheduled_at(0.2)]),
                    None,
                )],
                ..Default::default()
            },
            // Two preemption events delivered across two decode batches.
            EngineCoreOutputs {
                engine_index: 9,
                timestamp: 2.0,
                outputs: vec![engine_output(
                    "",
                    vec![2],
                    None,
                    Some(vec![preempted_at(1.5)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 9,
                timestamp: 3.0,
                outputs: vec![engine_output(
                    "",
                    vec![3],
                    Some(EngineCoreFinishReason::Length),
                    Some(vec![preempted_at(2.5)]),
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    assert!(
        rendered.contains(&format!(
            "uniserve:num_preemptions_total{{model_name=\"{model}\",engine=\"9\"}} 2"
        )),
        "expected num_preemptions_total 2 in:\n{rendered}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ttft_observation_recorded_exactly_once_on_first_output() {
    let model = unique_model_name("ttft");
    // TTFT value depends on wall-clock `received_at`, so we only assert the
    // deterministic observation count: it is recorded exactly once, on the
    // first (prefill) output, regardless of how many decode batches follow.
    let rendered = render_after_request(
        &model,
        "req-ttft",
        16,
        vec![
            EngineCoreOutputs {
                engine_index: 10,
                timestamp: 1.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(0.1), scheduled_at(0.2)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 10,
                timestamp: 2.0,
                outputs: vec![engine_output("", vec![2], None, None, None)],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 10,
                timestamp: 3.0,
                outputs: vec![engine_output(
                    "",
                    vec![3],
                    Some(EngineCoreFinishReason::Length),
                    None,
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    assert!(
        rendered.contains(&format!(
            "uniserve:time_to_first_token_seconds_count{{model_name=\"{model}\",engine=\"10\"}} 1"
        )),
        "expected exactly one TTFT observation in:\n{rendered}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finish_reason_label_tracks_terminal_reason() {
    let model = unique_model_name("finish-reason");
    let rendered = render_after_request(
        &model,
        "req-finish",
        16,
        vec![EngineCoreOutputs {
            engine_index: 11,
            timestamp: 1.0,
            outputs: vec![engine_output(
                "",
                vec![1],
                Some(EngineCoreFinishReason::Stop),
                Some(vec![queued_at(0.1), scheduled_at(0.2)]),
                None,
            )],
            ..Default::default()
        }],
    )
    .await;

    // A Stop finish reason renders the success counter with finished_reason="stop".
    assert!(
        rendered.contains(&format!(
            "uniserve:request_success_total{{model_name=\"{model}\",engine=\"11\",finished_reason=\"stop\"}} 1"
        )),
        "expected request_success stop label in:\n{rendered}"
    );
    assert_eq!(
        FinishReason::Length.as_str(),
        "length",
        "sanity: finish reason string mapping is stable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queue_time_uses_first_scheduled_timestamp_when_rescheduled() {
    let model = unique_model_name("queue-resched");
    // The Scheduled timestamp is latched on first occurrence; a later Scheduled
    // event (e.g. after preemption) must not move queue time.
    let rendered = render_after_request(
        &model,
        "req-resched",
        16,
        vec![
            EngineCoreOutputs {
                engine_index: 12,
                timestamp: 100.0,
                outputs: vec![engine_output(
                    "",
                    vec![1],
                    None,
                    Some(vec![queued_at(80.0), scheduled_at(90.0)]),
                    None,
                )],
                ..Default::default()
            },
            EngineCoreOutputs {
                engine_index: 12,
                timestamp: 110.0,
                outputs: vec![engine_output(
                    "",
                    vec![2],
                    Some(EngineCoreFinishReason::Length),
                    // A second Scheduled event with a later timestamp.
                    Some(vec![scheduled_at(105.0)]),
                    None,
                )],
                ..Default::default()
            },
        ],
    )
    .await;

    // queue time stays 90.0 - 80.0 = 10.0, not 105.0 - 80.0.
    assert!(
        rendered.contains(&format!(
            "uniserve:request_queue_time_seconds_sum{{model_name=\"{model}\",engine=\"12\"}} 10.0"
        )),
        "expected queue_time_sum 10.0 (first Scheduled latched) in:\n{rendered}"
    );
}
