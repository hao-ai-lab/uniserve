//! Kill-the-worker smoke test: a stub worker process dies abruptly mid-serve;
//! the executor's worker monitor converts the death into an engine-fatal error,
//! the scheduler loop exits, the engine proc emits ENGINE_CORE_DEAD, and the
//! frontend client fails the in-flight stream and latches unhealthy.
//!
//! Run with: UNISERVE_WORKER_STUB=1 UNISERVE_STUB_DIE_AFTER=3 \
//! cargo run -p uniserve-engine-process --example worker_death_smoke

use std::time::Duration;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use uniserve_engine_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};
use uniserve_engine_client::{EngineCoreClient, TransportMode, ZmqClientConfig};
use uniserve_engine_process::{EngineProcConfig, run_engine_proc};
use uniserve_engine_runtime::{EngineBackend, EngineCoreConfig};

#[tokio::main(worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    assert!(
        std::env::var("UNISERVE_WORKER_STUB").is_ok()
            && std::env::var("UNISERVE_STUB_DIE_AFTER").is_ok(),
        "run with UNISERVE_WORKER_STUB=1 UNISERVE_STUB_DIE_AFTER=3"
    );

    let handshake = format!("ipc:///tmp/uniserve-death-{}.sock", std::process::id());
    let shutdown = CancellationToken::new();
    let mut core = EngineCoreConfig::sim("stub-model");
    core.backend = EngineBackend::Worker; // the real ring + stub python worker
    let proc_task = tokio::spawn(run_engine_proc(
        EngineProcConfig {
            handshake_address: handshake.clone(),
            engine_index: 0,
            init_timeout: Duration::from_secs(30),
            core,
        },
        shutdown.clone(),
    ));

    let client = EngineCoreClient::connect_zmq(ZmqClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: handshake,
            advertised_host: "127.0.0.1".into(),
            engine_count: 1,
            ready_timeout: Duration::from_secs(60),
            local_input_address: None,
            local_output_address: None,
        },
        model_name: "stub-model".into(),
        client_index: 0,
        native_controls: None,
    })
    .await?;
    println!("connected; submitting a request the worker will die under...");

    let request = EngineCoreRequest {
        request_id: "req-death".to_string(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        sampling_params: Some(EngineCoreSamplingParams {
            temperature: 0.0,
            max_tokens: 64,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    };
    let mut stream = client.call(request).await?;

 // The in-flight request fails either as a terminal Error output (the
 // request-level path) or as a stream error once the dead sentinel closes
 // the registries — both are acceptable failure shapes.
    let mut saw_error = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let item = tokio::time::timeout_at(deadline, stream.next()).await;
        match item {
            Ok(Some(Ok(out))) => {
                if out.output.finish_reason
                    == Some(uniserve_engine_client::protocol::EngineCoreFinishReason::Error)
                {
                    println!("request resolved with finish_reason=Error as expected");
                    saw_error = true;
                    break;
                }
            }
            Ok(Some(Err(e))) => {
                println!("stream failed as expected: {e}");
                saw_error = true;
                break;
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    assert!(
        saw_error,
        "in-flight request must fail when the worker dies"
    );

 // The ENGINE_CORE_DEAD sentinel arrives asynchronously; poll the latch.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while client.is_healthy() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "client must latch unhealthy after the dead sentinel"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    println!(
        "health error: {:?}",
        client.health_error().map(|e| e.to_string())
    );

 // The engine proc itself exits with an error (managed mode would reap it).
    let proc_result = proc_task.await?;
    assert!(
        proc_result.is_err(),
        "engine proc must exit fatally, got {proc_result:?}"
    );
    println!("engine proc exited fatally as expected");

    client.shutdown().await?;
    println!("\nWORKER DEATH SMOKE: PASS");
    Ok(())
}
