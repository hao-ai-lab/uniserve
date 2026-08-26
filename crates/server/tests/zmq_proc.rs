#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Socket-mode end-to-end tests: the same scenarios as `sim_bridge.rs`, driven
//! through the full out-of-process topology: the ZMQ client talks to a real
//! headless engine process hosting the GPU-free sim engine.

use std::time::Duration;

use tokio_util::sync::CancellationToken;
use uniserve_core::codec::generation::GenerationControlTokens;
use uniserve_core::{
    ContextSegment, FeedbackNextToken, FeedbackSource, FinishReason, GenEvent,
    GeneratedImageFeedbackRecipe, GenerationBehaviorDescriptor, GenerationConstraint,
    GenerationPolicyDescriptor, GenerationRequest, GenerationResourceBounds,
    GenerationRuntimeCapabilities, ImageIngestRecipe, ImageKvEffect, ImageParams, RequestId,
    SamplingParams, TriggerPolicyDescriptor, UndVisibility,
};
use uniserve_engine::EngineCoreConfig;
use uniserve_engine::process::{EngineProcConfig, run_engine_proc};
use uniserve_server::engine_client::{
    EngineClient, GenerationSubmission, TransportMode, ZmqClientConfig,
};

/// Unique ipc:// endpoint for one test.
fn ipc_endpoint(tag: &str) -> String {
    format!(
        "ipc:///tmp/uniserve-test-{}-{}-{}.sock",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    )
}

/// Spawn one sim-backed engine proc dialing `handshake_address`.
fn spawn_sim_proc(
    handshake_address: &str,
    engine_index: u32,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<anyhow::Result<()>> {
    let cfg = EngineProcConfig {
        handshake_address: handshake_address.to_string(),
        engine_index,
        init_timeout: Duration::from_secs(10),
        core: EngineCoreConfig::sim("sim-model"),
    };
    tokio::spawn(run_engine_proc(cfg, shutdown))
}

fn client_config(handshake_address: &str, engine_count: usize) -> ZmqClientConfig {
    ZmqClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: handshake_address.to_string(),
            advertised_host: "127.0.0.1".to_string(),
            engine_count,
            ready_timeout: Duration::from_secs(20),
            local_input_address: None,
            local_output_address: None,
        },
        model_name: "sim-model".to_string(),
        client_index: 0,
        generation_controls: None,
        media_spool: None,
    }
}

fn generation_request(
    constraint: GenerationConstraint,
    sampling: SamplingParams,
    image: ImageParams,
    max_und_tokens: usize,
    trigger_token_id: u32,
) -> GenerationRequest {
    let policy = GenerationPolicyDescriptor {
        trigger: TriggerPolicyDescriptor::Token {
            token_id: trigger_token_id,
        },
        gen_only_start: uniserve_core::GenOnlyStartPolicyDescriptor::Immediate,
        feedback: Some(GeneratedImageFeedbackRecipe {
            source: FeedbackSource::DeviceProduct,
            next_und_token: FeedbackNextToken::EndOfImage,
            ingest: ImageIngestRecipe::vit_only(2, ImageKvEffect::WorkerDefined),
            sample_continuation: true,
        }),
        ..GenerationPolicyDescriptor::default()
    };
    GenerationRequest {
        request_id: RequestId(0),
        context: vec![ContextSegment::UndTokens {
            token_ids: vec![1, 2, 3],
            visibility: UndVisibility::Internal,
        }],
        negative_context: Vec::new(),
        constraint,
        behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
        sampling,
        image,
        max_und_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        policy,
        resources: GenerationResourceBounds {
            context_tokens: 3,
            max_kv_tokens: 3usize.saturating_add(max_und_tokens),
            ..GenerationResourceBounds::default()
        },
    }
}

fn declare_resources(
    mut request: GenerationRequest,
    capabilities: &GenerationRuntimeCapabilities,
) -> GenerationRequest {
    request.resources = GenerationResourceBounds::conservative(
        &request.context,
        &request.negative_context,
        &request.behavior,
        &request.policy,
        &request.image,
        request.max_und_tokens,
        &request.cache,
        capabilities,
    )
    .expect("request resources must fit the connected runtime");
    request
}

fn text_submission(
    request_id: impl Into<String>,
    max_tokens: u32,
    data_parallel_rank: Option<u32>,
) -> GenerationSubmission {
    let sampling = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let mut submission = GenerationSubmission::new(
        request_id.into(),
        generation_request(
            GenerationConstraint::UndOnly,
            sampling,
            ImageParams::default(),
            max_tokens as usize,
            1000,
        ),
    );
    submission.data_parallel_rank = data_parallel_rank;
    submission
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn socket_mode_text_generation() {
    let handshake = ipc_endpoint("text");
    let shutdown = CancellationToken::new();
    let proc_task = spawn_sim_proc(&handshake, 0, shutdown.clone());

    let client = EngineClient::connect_zmq(client_config(&handshake, 1))
        .await
        .expect("connect zmq client");

    // The handshake delivered real post-load truth from WorkerCapabilities.
    assert_eq!(client.engine_count(), 1);
    assert_eq!(client.ready_responses().len(), 1);
    assert!(client.total_num_gpu_blocks() > 0);

    let mut stream = client
        .submit_generation(text_submission("req-1", 64, None))
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

    assert!(
        tokens >= 1,
        "expected at least one generated token, got {tokens}"
    );
    assert_eq!(
        finish,
        Some(FinishReason::Eos),
        "expected the request to terminate on the synthetic EOS"
    );

    shutdown.cancel();
    client.shutdown().await.expect("shutdown client");
    let _ = proc_task.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn socket_mode_image_generation() {
    let handshake = ipc_endpoint("image");
    let shutdown = CancellationToken::new();
    let proc_task = spawn_sim_proc(&handshake, 0, shutdown.clone());

    let client = EngineClient::connect_zmq(client_config(&handshake, 1))
        .await
        .expect("connect zmq client");

    let request = declare_resources(
        generation_request(
            GenerationConstraint::GenOnly,
            SamplingParams::default(),
            ImageParams {
                steps: 4,
                ..ImageParams::default()
            },
            0,
            1000,
        ),
        &client.generation_capabilities(),
    );

    let mut stream = client
        .submit_generation(GenerationSubmission::new("req-image", request))
        .await
        .expect("submit generation request");

    let mut begins = 0usize;
    let mut steps = 0usize;
    let mut commits = 0usize;
    let mut dones = 0usize;
    let mut png = String::new();
    let mut finished = false;
    while let Some(ev) = stream.next().await {
        match ev {
            GenEvent::ImageBegin { .. } => begins += 1,
            GenEvent::ImageStep { .. } => steps += 1,
            GenEvent::ImageCommit { .. } => commits += 1,
            GenEvent::ImageDone { pixels_png_b64, .. } => {
                dones += 1;
                png = pixels_png_b64;
            }
            GenEvent::Finished { images, .. } => {
                assert_eq!(images, 1, "expected exactly one image");
                finished = true;
                break;
            }
            GenEvent::Rejected { message } => panic!("request rejected: {message}"),
            GenEvent::Error { message } => panic!("engine error: {message}"),
            _ => {}
        }
    }

    assert_eq!(begins, 1, "expected one ImageBegin");
    assert!(
        (1..=4).contains(&steps),
        "expected per-step diffusion progress, got {steps}"
    );
    assert_eq!(commits, 1, "expected one ImageCommit");
    assert_eq!(dones, 1, "expected one ImageDone");
    assert!(!png.is_empty(), "expected PNG bytes to cross the wire");
    assert!(finished, "expected a terminal Finished event");

    shutdown.cancel();
    client.shutdown().await.expect("shutdown client");
    let _ = proc_task.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn socket_mode_default_generation() {
    let handshake = ipc_endpoint("default-generation");
    let shutdown = CancellationToken::new();
    let proc_task = spawn_sim_proc(&handshake, 0, shutdown.clone());

    let mut cfg = client_config(&handshake, 1);
    cfg.generation_controls = Some(GenerationControlTokens {
        ..Default::default()
    });
    let client = EngineClient::connect_zmq(cfg)
        .await
        .expect("connect zmq client");

    let request = declare_resources(
        generation_request(
            GenerationConstraint::Default,
            SamplingParams {
                logit_bias: vec![(2222, 1000.0)],
                ..SamplingParams::default()
            },
            ImageParams {
                steps: 2,
                max_images: 2,
                ..ImageParams::default()
            },
            64,
            2222,
        ),
        &client.generation_capabilities(),
    );

    let mut stream = client
        .submit_generation(GenerationSubmission::new("req-mixed", request))
        .await
        .expect("submit generation request");

    let mut text_tokens = 0usize;
    let mut commits = 0usize;
    let mut dones = 0usize;
    let mut finished_images = 0usize;
    let mut finished = false;
    while let Some(ev) = stream.next().await {
        match ev {
            GenEvent::TextToken { .. } => text_tokens += 1,
            GenEvent::ImageCommit { .. } => commits += 1,
            GenEvent::ImageDone { .. } => dones += 1,
            GenEvent::Finished { images, .. } => {
                finished_images = images;
                finished = true;
                break;
            }
            GenEvent::Rejected { message } => panic!("request rejected: {message}"),
            GenEvent::Error { message } => panic!("engine error: {message}"),
            _ => {}
        }
    }

    assert!(finished, "expected a terminal Finished event");
    assert_eq!(
        commits, dones,
        "every completed image must commit exactly once"
    );
    assert!(
        text_tokens >= 1,
        "expected mixed-output text, got {text_tokens} tokens"
    );
    assert_eq!(dones, 2, "expected two generated images, got {dones}");
    assert_eq!(finished_images, 2);

    shutdown.cancel();
    client.shutdown().await.expect("shutdown client");
    let _ = proc_task.await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn socket_mode_two_engines_serve_default_and_ranked_requests() {
    let handshake = ipc_endpoint("dp");
    let shutdown = CancellationToken::new();
    let proc0 = spawn_sim_proc(&handshake, 0, shutdown.clone());
    let proc1 = spawn_sim_proc(&handshake, 1, shutdown.clone());

    let client = EngineClient::connect_zmq(client_config(&handshake, 2))
        .await
        .expect("connect zmq client to two engines");
    assert_eq!(client.engine_count(), 2);
    // The 2-byte little-endian engine-index identities are both registered.
    let mut identities = client.engine_identities();
    identities.sort();
    assert_eq!(identities, vec![&[0u8, 0u8][..], &[1u8, 0u8][..]]);

    let request = |id: &str, rank: Option<u32>| text_submission(id, 32, rank);

    // Default routing keeps concurrent requests independently consumable.
    let mut streams = Vec::new();
    for i in 0..6 {
        streams.push(
            client
                .submit_generation(request(&format!("req-{i}"), None))
                .await
                .expect("submit"),
        );
    }
    for mut stream in streams {
        let mut finished = false;
        while let Some(event) = stream.next().await {
            if matches!(event, GenEvent::Finished { .. }) {
                finished = true;
                break;
            }
            if let GenEvent::Rejected { message } | GenEvent::Error { message } = event {
                panic!("generation failed: {message}");
            }
        }
        assert!(finished, "default-routed request must finish");
    }

    // Every configured data-parallel rank accepts the canonical request contract.
    for rank in [1u32, 0, 1] {
        let mut stream = client
            .submit_generation(request(
                &format!("req-rank-{rank}-{}", rand_tag()),
                Some(rank),
            ))
            .await
            .expect("submit ranked");
        let mut finished = false;
        while let Some(event) = stream.next().await {
            if matches!(event, GenEvent::Finished { .. }) {
                finished = true;
                break;
            }
            if let GenEvent::Rejected { message } | GenEvent::Error { message } = event {
                panic!("generation failed: {message}");
            }
        }
        assert!(finished, "ranked request must finish");
    }

    shutdown.cancel();
    client.shutdown().await.expect("shutdown client");
    let _ = proc0.await;
    let _ = proc1.await;
}

fn rand_tag() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
}

/// ENGINE_CORE_DEAD semantics: a minimal engine-side peer completes the
/// handshake, accepts one request, then emits the dead sentinel — the in-flight
/// stream must resolve with an error, the persistent health latch must stick,
/// and new requests must fail fast.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn engine_dead_sentinel_latches_health() {
    use uniserve_core::ModelDtype;
    use uniserve_core::codec::handshake::{
        EngineCoreReadyResponse, HandshakeInitMessage, ReadyMessage,
    };
    use uniserve_core::codec::{ENGINE_CORE_DEAD_SENTINEL, decode_msgpack, encode_msgpack};
    use zeromq::prelude::{Socket, SocketRecv, SocketSend};
    use zeromq::util::PeerIdentity;
    use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

    let handshake_addr = ipc_endpoint("dead");

    // Engine-side peer: handshake, register, accept one request, die.
    let engine_handshake = handshake_addr.clone();
    let engine_task = tokio::spawn(async move {
        let identity =
            PeerIdentity::try_from(bytes::Bytes::copy_from_slice(&0u16.to_le_bytes())).unwrap();
        let mut options = SocketOptions::default();
        options.peer_identity(identity.clone());
        let mut handshake = DealerSocket::with_options(options);
        // The frontend binds after we spawn; retry the dial.
        for _ in 0..100 {
            if handshake.connect(&engine_handshake).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let hello = ReadyMessage {
            status: Some("HELLO".into()),
            local: Some(true),
            headless: Some(true),
            parallel_config_hash: None,
        };
        handshake
            .send(ZmqMessage::from(encode_msgpack(&hello).unwrap()))
            .await
            .unwrap();
        let init: HandshakeInitMessage =
            decode_msgpack(handshake.recv().await.unwrap().into_vec()[0].as_ref()).unwrap();

        let mut input_options = SocketOptions::default();
        input_options.peer_identity(identity);
        let mut input = DealerSocket::with_options(input_options);
        input.connect(&init.addresses.inputs[0]).await.unwrap();
        let ready = EngineCoreReadyResponse {
            max_model_len: 4096,
            num_gpu_blocks: 64,
            dp_stats_address: None,
            dtype: ModelDtype::BFloat16,
            uniserve_version: "test".into(),
            generation_capabilities: uniserve_core::GenerationRuntimeCapabilities::default(),
        };
        input
            .send(ZmqMessage::from(encode_msgpack(&ready).unwrap()))
            .await
            .unwrap();
        let mut output = PushSocket::new();
        output.connect(&init.addresses.outputs[0]).await.unwrap();
        let ready_status = ReadyMessage {
            status: Some("READY".into()),
            ..hello
        };
        handshake
            .send(ZmqMessage::from(encode_msgpack(&ready_status).unwrap()))
            .await
            .unwrap();

        // Accept one generation request, then declare the engine dead.
        let _ = input.recv().await.unwrap();
        output
            .send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec()))
            .await
            .unwrap();
        // Keep sockets alive briefly so the sentinel flushes.
        tokio::time::sleep(Duration::from_secs(1)).await;
    });

    let client = EngineClient::connect_zmq(client_config(&handshake_addr, 1))
        .await
        .expect("connect to fake engine");
    assert!(client.is_healthy());

    let mut stream = client
        .submit_generation(text_submission("req-dead", 8, None))
        .await
        .expect("submit request");

    // The in-flight stream resolves with an error once the sentinel lands.
    let mut saw_error = false;
    while let Some(event) = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("stream must resolve after engine death")
    {
        if matches!(event, GenEvent::Error { .. }) {
            saw_error = true;
            break;
        }
    }
    assert!(saw_error, "in-flight stream must fail on engine death");

    // The health latch sticks and new work fails fast.
    assert!(
        !client.is_healthy(),
        "health latch must record engine death"
    );
    assert!(client.health_error().is_some());
    let request = text_submission("req-after-death", 1, None);
    assert!(
        client.submit_generation(request).await.is_err(),
        "new requests must fail fast after engine death"
    );

    client.shutdown().await.expect("shutdown client");
    let _ = engine_task.await;
}
