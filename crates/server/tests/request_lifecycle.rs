//! Frontend identity, cancellation, and drain across real preprocessing and output.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use futures::{StreamExt, poll};
use tokenizers::models::bpe::{BPE, Vocab};
use tokenizers::{AddedToken, Tokenizer};
use uniserve_engine::{EngineConfig, SimEngine, SimExecutor};
use uniserve_server::engine_client::EngineClient;
use uniserve_server::profile::tokenizer::HuggingFaceTokenizer;
use uniserve_server::profile::{ModelConfig, ModelParameters, SamplingDefaults};
use uniserve_server::serving::chat::{ChatTemplateContentFormatOption, HfChatRenderer};
use uniserve_server::serving::media::{ImageFetchPolicy, ImageFetcher};
use uniserve_server::serving::{
    FinishStatus, InputProcessor, RequestLifecycleState, RequestOutput, ServedSamplingControl,
    ServingRuntime, StopCause, TextPromptRequest,
};

/// Builds a serving runtime over a default `SimEngine`.
fn runtime() -> ServingRuntime {
    runtime_with_worker(SimEngine::new())
}

/// Builds a Qwen3 serving runtime whose engine drives `worker` in process
/// through `SimExecutor`.
fn runtime_with_worker(worker: SimEngine) -> ServingRuntime {
    runtime_for_engine_model(worker, "sim-model")
}

/// Builds a Qwen3 serving runtime whose engine, named `engine_model`, drives
/// `worker` in process through `SimExecutor`.
///
/// The engine's model name labels the process-wide request metrics, so a test
/// that reads them names its own engine. The tokenizer maps each code point
/// below 128 to one token whose ID is the code point; `<|im_end|>` is added
/// as ID 128, which the model config names as its end-of-sequence token. The
/// chat template renders only the first message's content.
fn runtime_for_engine_model(worker: SimEngine, engine_model: &str) -> ServingRuntime {
    let vocabulary = (0_u32..128)
        .map(|id| (char::from_u32(id).unwrap().to_string(), id))
        .collect::<Vocab>();
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocabulary, Vec::new())
            .build()
            .unwrap(),
    );
    tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();
    let tokenizer = Arc::new(HuggingFaceTokenizer::new(&path).unwrap());

    let client = Arc::new(
        EngineClient::connect_with_executor(
            EngineConfig::sim(engine_model),
            Box::new(SimExecutor::new(worker)),
        )
        .unwrap(),
    );
    let processor = InputProcessor::new(
        ModelConfig {
            served_name: "sim-model".to_string(),
            parameters: ModelParameters::Qwen3,
            sampling_defaults: SamplingDefaults::default(),
            max_model_tokens: Some(4096),
            primary_eos_token_id: Some(128),
            eos_token_ids: BTreeSet::from([128]),
        },
        tokenizer,
        Some(
            HfChatRenderer::new(
                Some("{{ messages[0].content }}".to_string()),
                HashMap::new(),
                ChatTemplateContentFormatOption::String,
            )
            .unwrap(),
        ),
        uniserve_server::serving::WorkerCapabilities {
            limits: client.generation_limits(),
            sampling_controls: ServedSamplingControl::ALL.to_vec(),
            max_model_tokens: 4096,
        },
        None,
        false,
    )
    .unwrap();

    let images = ImageFetcher::new(ImageFetchPolicy::default()).unwrap();
    ServingRuntime::new(processor, client, images, false)
}

/// Greedy text request that may sample only `a`, so it deterministically
/// stops on the `aa` stop string after two output tokens.
fn request(id: &str) -> TextPromptRequest {
    let mut request = TextPromptRequest::new(id, "prompt");
    request.sampling.max_tokens = Some(8);
    request.sampling.temperature = Some(0.0);
    request.stop.allowed_token_ids = Some(vec![u32::from(b'a')]);
    request.stop.stop_strings = vec!["aa".to_string()];
    request
}

/// A request identifier is refused while a request holding it is live and
/// becomes reusable once the finished request leaves the active registry,
/// which `drain_request` waits for. Dropping the first, already exhausted
/// stream afterwards must not affect the second request that reuses the
/// identifier.
#[tokio::test]
async fn stop_completion_releases_identity_without_dropping_the_exhausted_stream() {
    let runtime = runtime();
    let mut first = runtime.generate_text(request("same-id")).await.unwrap();
    assert!(matches!(
        runtime.generate_text(request("same-id")).await,
        Err(uniserve_server::openai::ApiError::InvalidRequest { .. })
    ));

    let mut text = String::new();
    let mut output_tokens = None;
    loop {
        match first.next().await.unwrap().unwrap() {
            RequestOutput::TextDelta { text: delta, .. } => text.push_str(&delta),
            RequestOutput::Usage {
                visible_output_tokens,
                internal_tokens,
                ..
            } => {
                output_tokens = Some(visible_output_tokens + internal_tokens);
            }
            RequestOutput::Finished { reason, .. } => {
                assert_eq!(
                    reason,
                    FinishStatus::Stop {
                        cause: Some(StopCause::Text("aa".to_string()))
                    }
                );
                break;
            }
            _ => {}
        }
    }

    // The matched stop string is withheld from the visible text, but both
    // sampled tokens count toward usage.
    assert_eq!(text, "");
    assert_eq!(output_tokens, Some(2));
    assert_eq!(
        runtime.drain_request("same-id").await.unwrap().state,
        RequestLifecycleState::Finished
    );

    let mut second = runtime.generate_text(request("same-id")).await.unwrap();
    // A retained exhausted stream must not cancel a reused external identity.
    drop(first);
    while let Some(event) = second.next().await {
        if let RequestOutput::Finished { .. } = event.unwrap() {
            break;
        }
    }
    assert_eq!(runtime.metrics_snapshot().finished, 2);
    runtime.shutdown().await.unwrap();
}

/// Cancel, abort, and a dropped `generate_text` future each end a request
/// whose preprocessing has not started. A control yields a single terminal
/// event, abort takes precedence over an earlier or later cancel, and a
/// disconnect is recorded as a cancellation.
#[test]
fn controls_and_disconnection_release_requests_during_preprocessing() {
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    executor.block_on(async {
        let runtime = runtime();

        // Occupy the real blocking pool so preprocessing remains queued. This
        // controls the execution environment without replacing the tokenizer.
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let occupied = tokio::task::spawn_blocking(move || {
            let _ = ready_tx.send(());
            let _ = release_rx.recv();
        });
        ready_rx.await.unwrap();

        for (id, expected) in [
            ("cancel", RequestLifecycleState::Cancelled),
            ("abort", RequestLifecycleState::Aborted),
            ("disconnect", RequestLifecycleState::Cancelled),
        ] {
            // One poll registers the request before it waits for the blocking
            // pool; the registered identifier already refuses a duplicate.
            let mut pending = Box::pin(runtime.generate_text(request(id)));
            assert!(poll!(pending.as_mut()).is_pending());
            assert_eq!(
                runtime.request_stats(id).unwrap().state,
                RequestLifecycleState::Compiling
            );
            assert!(matches!(
                runtime.generate_text(request(id)).await,
                Err(uniserve_server::openai::ApiError::InvalidRequest { .. })
            ));
            if id == "disconnect" {
                drop(pending);
            } else {
                runtime.cancel(id).await.unwrap();
                if id == "abort" {
                    runtime.abort(id).await.unwrap();
                    runtime.cancel(id).await.unwrap();
                }
                let mut stream = pending.await.unwrap();
                let terminal = stream.next().await.unwrap().unwrap();
                assert!(match expected {
                    RequestLifecycleState::Cancelled =>
                        matches!(terminal, RequestOutput::Cancelled { .. }),
                    RequestLifecycleState::Aborted =>
                        matches!(terminal, RequestOutput::Aborted { .. }),
                    _ => unreachable!(),
                });
                assert!(stream.next().await.is_none());
            }
            assert_eq!(runtime.drain_request(id).await.unwrap().state, expected);
        }

        assert_eq!(runtime.metrics_snapshot().cancelled, 2);
        assert_eq!(runtime.metrics_snapshot().aborted, 1);
        release_tx.send(()).unwrap();
        occupied.await.unwrap();
        runtime.shutdown().await.unwrap();
    });
}

/// A chat prompt that exceeds the total capacity of a two-unit KV cache is
/// returned as a stream and then rejected by engine admission. Collecting the
/// completion must report that rejection as HTTP 400 and record the request
/// as `Rejected`.
#[tokio::test]
async fn chat_admission_rejection_remains_a_bad_request() {
    let mut worker = SimEngine::new();
    // Unit zero is the sentinel: two allocatable 64-token pages remain.
    worker.set_num_units(3);
    let runtime = runtime_with_worker(worker);
    let request = serde_json::from_value(serde_json::json!({
        "model": "sim-model",
        "messages": [{"role": "user", "content": "a".repeat(1024)}],
        "max_completion_tokens": 8
    }))
    .unwrap();

    let stream = runtime
        .generate_chat("rejected-chat".into(), request)
        .await
        .unwrap();
    let error = uniserve_server::openai::chat_completions::collect_chat_completion(
        stream,
        "rejected-chat".to_string(),
        "sim-model".to_string(),
        0,
        false,
        false,
        false,
        false,
        false,
    )
    .await
    .unwrap_err();

    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(
        runtime.request_stats("rejected-chat").unwrap().state,
        RequestLifecycleState::Rejected
    );
    runtime.shutdown().await.unwrap();
}

/// `compile_us` is measured from the start of `generate_text`, so it includes
/// time spent queued for the blocking pool; the reported total latency must
/// cover that wait as well.
#[test]
fn successful_total_includes_time_waiting_for_preprocessing() {
    let executor = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    executor.block_on(async {
        let runtime = runtime();

        // Hold the only blocking thread so preprocessing waits in the queue
        // for a measurable interval before it runs.
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let occupied = tokio::task::spawn_blocking(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx.await.unwrap();
        let mut pending = Box::pin(runtime.generate_text(request("timed")));
        assert!(poll!(pending.as_mut()).is_pending());
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        release_tx.send(()).unwrap();
        occupied.await.unwrap();

        let mut stream = pending.await.unwrap();
        let mut timing = None;
        while let Some(event) = stream.next().await {
            if let RequestOutput::Usage { timings, .. } = event.unwrap() {
                timing = Some(timings);
            }
        }
        let timing = timing.unwrap();
        assert!(timing.total_us >= timing.compile_us, "{timing:?}");
        runtime.shutdown().await.unwrap();
    });
}

/// A completed request is recorded once in the request metrics under its
/// engine's model name: the token counters and token histograms carry the
/// counts of its usage report, its finish status is counted, and each latency
/// histogram whose input it carries holds one observation.
#[tokio::test]
async fn completed_request_is_recorded_in_request_metrics() {
    let engine_model = "request-metrics-model";
    let runtime = runtime_for_engine_model(SimEngine::new(), engine_model);
    let mut request = TextPromptRequest::new("metrics", "prompt");
    request.sampling.max_tokens = Some(3);
    request.sampling.temperature = Some(0.0);
    request.stop.allowed_token_ids = Some(vec![u32::from(b'a')]);

    let mut stream = runtime.generate_text(request).await.unwrap();
    let mut usage = None;
    let mut finish = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            RequestOutput::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                ..
            } => usage = Some((prompt_tokens, visible_output_tokens + internal_tokens)),
            RequestOutput::Finished { reason, .. } => finish = Some(reason),
            _ => {}
        }
    }
    assert_eq!(finish, Some(FinishStatus::Length));
    let (prompt_tokens, generation_tokens) = usage.unwrap();
    assert_eq!(generation_tokens, 3);

    let rendered = uniserve_observability::METRICS.render().unwrap();
    let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
    for expected in [
        format!("uniserve:request_success_total{{{labels},finished_reason=\"length\"}} 1"),
        format!("uniserve:prompt_tokens_total{{{labels}}} {prompt_tokens}"),
        format!("uniserve:generation_tokens_total{{{labels}}} {generation_tokens}"),
        format!("uniserve:request_prompt_tokens_count{{{labels}}} 1"),
        format!("uniserve:request_generation_tokens_count{{{labels}}} 1"),
        format!("uniserve:request_max_num_generation_tokens_count{{{labels}}} 1"),
        format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
        format!("uniserve:request_queue_time_seconds_count{{{labels}}} 1"),
        format!("uniserve:time_to_first_token_seconds_count{{{labels}}} 1"),
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing `{expected}` in:\n{rendered}"
        );
    }
    runtime.shutdown().await.unwrap();
}

/// A client that disconnects mid-generation is counted as an aborted request
/// with its latency, while its partial output, which never reached a usage
/// report, adds nothing to the token metrics.
#[tokio::test]
async fn disconnected_request_is_recorded_as_aborted_without_token_usage() {
    let engine_model = "disconnect-metrics-model";
    let mut worker = SimEngine::new();
    worker.set_text_len(1_000_000);
    let runtime = runtime_for_engine_model(worker, engine_model);
    let mut request = TextPromptRequest::new("disconnect", "prompt");
    request.sampling.max_tokens = Some(1_000);
    request.sampling.temperature = Some(0.0);
    request.stop.allowed_token_ids = Some(vec![u32::from(b'a')]);

    let mut stream = runtime.generate_text(request).await.unwrap();
    loop {
        match stream.next().await.unwrap().unwrap() {
            RequestOutput::TextDelta { text, .. } if !text.is_empty() => break,
            _ => {}
        }
    }
    drop(stream);

    let rendered = uniserve_observability::METRICS.render().unwrap();
    let labels = format!("model_name=\"{engine_model}\",engine=\"0\"");
    for expected in [
        format!("uniserve:request_success_total{{{labels},finished_reason=\"abort\"}} 1"),
        format!("uniserve:e2e_request_latency_seconds_count{{{labels}}} 1"),
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing `{expected}` in:\n{rendered}"
        );
    }
    for family in [
        "uniserve:generation_tokens_total",
        "uniserve:request_generation_tokens_count",
    ] {
        let series = format!("{family}{{{labels}}}");
        assert!(
            !rendered.lines().any(|line| line.starts_with(&series)),
            "unexpected `{series}` in:\n{rendered}"
        );
    }
    runtime.shutdown().await.unwrap();
}
