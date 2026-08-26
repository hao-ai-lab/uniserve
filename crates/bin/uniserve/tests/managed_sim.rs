#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Managed-mode integration test: spawn the `uniserve engine --sim` binary as a
//! supervised subprocess, drive it through the ZMQ client over real sockets, and
//! tear it down through the supervisor (SIGTERM → process-group reap).

use std::time::Duration;

use uniserve_core::{
    ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
    GenerationRequest, GenerationResourceBounds, ImageParams, RequestId, SamplingParams,
    UndVisibility,
};
use uniserve_engine_gateway::transport::{
    EngineCoreClient, GenEvent, GenerationFinishReason, GenerationSubmission, TransportMode,
    ZmqClientConfig,
};
use uniserve_managed_engine::{ManagedEngineConfig, ManagedEngineHandle, allocate_handshake_port};

fn text_generation_request() -> GenerationRequest {
    let constraint = GenerationConstraint::UndOnly;
    let policy = GenerationPolicyDescriptor::default();
    GenerationRequest {
        request_id: RequestId(0),
        context: vec![ContextSegment::UndTokens {
            token_ids: vec![1, 2, 3, 4],
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        image: ImageParams::default(),
        max_und_tokens: 64,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 4,
            max_kv_tokens: 68,
            ..GenerationResourceBounds::default()
        },
    }
}

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
        generation_controls: None,
        media_spool: None,
    })
    .await
    .expect("connect to managed engine");

    // The subprocess engine answered the handshake with post-load truth.
    assert_eq!(client.engine_count(), 1);
    assert!(client.total_num_gpu_blocks() > 0);

    // One text generation through the real subprocess.
    let mut stream = client
        .submit_generation(GenerationSubmission::new(
            "req-managed",
            text_generation_request(),
        ))
        .await
        .expect("submit request");
    let mut tokens = 0usize;
    let mut finish = None;
    while let Some(event) = stream.next().await {
        match event {
            GenEvent::TextToken { .. } => tokens += 1,
            GenEvent::Finished { reason, .. } => {
                finish = Some(reason);
                break;
            }
            GenEvent::Rejected { message } => panic!("request rejected: {message}"),
            GenEvent::Error { message } => panic!("engine error: {message}"),
            _ => {}
        }
    }
    assert!(tokens >= 1);
    assert_eq!(finish, Some(GenerationFinishReason::Eos));

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
