#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Managed-mode integration test: spawn the `uniserve engine --sim` binary as a
//! supervised subprocess, drive it through the ZMQ client over real sockets, and
//! tear it down through the supervisor (SIGTERM → process-group reap).

use std::time::Duration;

use futures::StreamExt;
use uniserve_engine_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use uniserve_engine_client::{EngineCoreClient, TransportMode, ZmqClientConfig};
use uniserve_managed_engine::{ManagedEngineConfig, ManagedEngineHandle, allocate_handshake_port};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn managed_sim_engine_serves_and_shuts_down() {
    let host = "127.0.0.1".to_string();
    let port = allocate_handshake_port(&host).expect("allocate handshake port");

    let engine = ManagedEngineHandle::spawn(ManagedEngineConfig {
        binary: env!("CARGO_BIN_EXE_uniserve").to_string(),
        model: "sim-model".to_string(),
        handshake_host: host.clone(),
        handshake_port: port,
        engine_index: 0,
        engine_args: vec!["--sim".to_string()],
    })
    .await
    .expect("spawn managed sim engine");

    let client = EngineCoreClient::connect_zmq(ZmqClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: format!("tcp://{host}:{port}"),
            advertised_host: host.clone(),
            engine_count: 1,
            ready_timeout: Duration::from_secs(60),
            local_input_address: None,
            local_output_address: None,
        },
        model_name: "sim-model".to_string(),
        client_index: 0,
        native_controls: None,
    })
    .await
    .expect("connect to managed engine");

    // The subprocess engine answered the handshake with post-load truth.
    assert_eq!(client.engine_count(), 1);
    assert!(client.total_num_gpu_blocks() > 0);

    // One text generation through the real subprocess.
    let request = EngineCoreRequest {
        request_id: "req-managed".to_string(),
        prompt_token_ids: Some(vec![1, 2, 3, 4]),
        sampling_params: Some(EngineCoreSamplingParams {
            temperature: 0.0,
            max_tokens: 64,
            ..EngineCoreSamplingParams::for_test()
        }),
        ..Default::default()
    };
    let mut stream = client.call(request).await.expect("submit request");
    let mut tokens = 0usize;
    let mut finish = None;
    while let Some(item) = stream.next().await {
        let output = item.expect("stream item").output;
        tokens += output.new_token_ids.len();
        if let Some(reason) = output.finish_reason {
            finish = Some(reason);
            break;
        }
    }
    assert!(tokens >= 1);
    assert_eq!(finish, Some(EngineCoreFinishReason::Stop));

    client.shutdown().await.expect("shutdown client");

    // Supervised teardown: SIGTERM the engine's process group and reap it.
    engine
        .shutdown(Duration::from_secs(10))
        .await
        .expect("shutdown managed engine");
    let status = engine.try_wait().await.expect("poll managed engine");
    assert!(
        status.is_some(),
        "managed engine should have exited after shutdown"
    );
}
