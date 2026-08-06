//! Headless engine process hosting an [`EngineCore`] behind the ZMQ wire
//! protocol.
//!
//! Startup composes the two handshakes exactly as the plan requires: the
//! frontend handshake (dial `HELLO` → receive `INIT` with the data-plane
//! addresses and resolved generation control tokens) and the
//! engine↔worker handshake (`EngineCore::new` waits on the worker's
//! `get_caps`, i.e. model load), with `get_caps` completing **before** `READY`
//! is sent — READY carries the post-load truth reported by the runtime.
//!
//! After startup the process runs three explicit tasks: an input task decodes
//! `[request-type, payload]` frames from the DEALER socket into scheduler
//! commands, per-request tasks forward canonical generation events, and an
//! output task batches them onto the PUSH socket.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod stats;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uniserve_core::RequestId;
use uniserve_engine_wire::handshake::{HandshakeInitMessage, ReadyMessage};
use uniserve_engine_wire::{
    EngineRequest, GenEvent, GenerationEventBatch, GenerationRequestEnvelope,
    RoutedGenerationEvent, decode_msgpack, encode_msgpack,
};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use uniserve_engine_runtime::{EngineCore, EngineCoreConfig};
use uniserve_engine_wire::generation::GenerationControlTokens;
use uniserve_engine_wire::handshake::EngineCoreReadyResponse;
use uniserve_sim::{SimEngine, SimExecutor};

/// Configuration for one headless engine process.
#[derive(Debug, Clone)]
pub struct EngineProcConfig {
    /// Frontend handshake endpoint this engine dials (`tcp://…` / `ipc://…`).
    pub handshake_address: String,
    /// Engine index; becomes the 2-byte little-endian socket identity.
    pub engine_index: u32,
    /// Maximum time to wait for the frontend's INIT after HELLO.
    pub init_timeout: Duration,
    /// Engine runtime construction (worker spawn, model, scheduler).
    pub core: EngineCoreConfig,
}

fn now_secs() -> f64 {
    // single shared epoch helper so this matches the frontend's and
    // scheduler's wall-clock timestamps.
    uniserve_core::now_unix_secs()
}

fn engine_identity(engine_index: u32) -> Result<PeerIdentity> {
    let bytes = Bytes::copy_from_slice(&(engine_index as u16).to_le_bytes());
    PeerIdentity::try_from(bytes)
        .map_err(|e| anyhow::anyhow!("invalid engine identity for index {engine_index}: {e}"))
}

fn status_message(status: &str) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(true),
        headless: Some(true),
        parallel_config_hash: None,
    }
}

fn apply_generation_controls(config: &mut EngineCoreConfig, ctrl: &GenerationControlTokens) {
    config.bos = ctrl.bos;
    config.end_of_image = ctrl.end_of_image;
    if config.backend != uniserve_engine_runtime::EngineBackend::Sim {
        config.eos = ctrl.eos.clone();
    }
}

fn ready_response(core: &EngineCore) -> EngineCoreReadyResponse {
    let caps = core.caps();
    EngineCoreReadyResponse {
        max_model_len: core.max_model_len() as u64,
        num_gpu_blocks: caps.num_blocks as u64,
        dp_stats_address: None,
        dtype: core.model_dtype(),
        uniserve_version: env!("CARGO_PKG_VERSION").to_string(),
        generation_capabilities: core.generation_capabilities(),
    }
}

/// One message bound for the output PUSH socket.
enum OutMsg {
    Event(Box<RoutedGenerationEvent>),
    /// The engine died: emit the ENGINE_CORE_DEAD sentinel and stop.
    Dead,
}

const OUTPUT_BUFFER_CAPACITY: usize = 256;

type ActiveRequests = HashMap<String, RequestId>;
type SharedActiveRequests = Arc<Mutex<ActiveRequests>>;

fn lock_active(active: &Mutex<ActiveRequests>) -> std::sync::MutexGuard<'_, ActiveRequests> {
    match active.lock() {
        Ok(guard) => guard,
        Err(error) => {
            warn!("active request map lock poisoned");
            error.into_inner()
        }
    }
}

fn remove_active_request(active: &Mutex<ActiveRequests>, request_id: &str, rid: RequestId) {
    let mut active = lock_active(active);
    if active.get(request_id) == Some(&rid) {
        active.remove(request_id);
    }
}

fn insert_active_request(active: &Mutex<ActiveRequests>, request_id: &str, rid: RequestId) -> bool {
    let mut active = lock_active(active);
    if active.contains_key(request_id) {
        return false;
    }
    active.insert(request_id.to_string(), rid);
    true
}

/// Dial the frontend handshake endpoint, send HELLO, and wait for INIT —
/// retrying the whole dial while the frontend has not bound the endpoint yet.
async fn dial_handshake(
    handshake_address: &str,
    identity: PeerIdentity,
    init_timeout: Duration,
) -> Result<(DealerSocket, HandshakeInitMessage)> {
    const RETRY_DELAY: Duration = Duration::from_millis(250);
    let deadline = tokio::time::Instant::now() + init_timeout;

    loop {
        let attempt: Result<(DealerSocket, HandshakeInitMessage)> = async {
            let mut options = SocketOptions::default();
            options.peer_identity(identity.clone());
            let mut handshake = DealerSocket::with_options(options);
            handshake
                .connect(handshake_address)
                .await
                .with_context(|| format!("connecting handshake {handshake_address}"))?;
            handshake
                .send(ZmqMessage::from(encode_msgpack(&status_message("HELLO"))?))
                .await?;

            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            // Bound each wait so a HELLO sent into the void (endpoint bound by
            // nobody, or bound after our connect raced it) re-dials.
            let wait = remaining
                .min(Duration::from_secs(5))
                .max(Duration::from_millis(100));
            let init_frames = timeout(wait, handshake.recv())
                .await
                .map_err(|_| anyhow::anyhow!("no INIT within {wait:?}"))??
                .into_vec();
            if init_frames.len() != 1 {
                bail!("expected one INIT frame, got {}", init_frames.len());
            }
            let init: HandshakeInitMessage = decode_msgpack(init_frames[0].as_ref())?;
            Ok((handshake, init))
        }
        .await;

        match attempt {
            Ok(ok) => {
                info!(handshake = %handshake_address, "received INIT");
                return Ok(ok);
            }
            Err(e) if tokio::time::Instant::now() + RETRY_DELAY < deadline => {
                debug!(error = %e, "handshake dial failed; retrying");
                tokio::time::sleep(RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(e.context(format!(
                    "handshake with {handshake_address} not completed within {init_timeout:?}"
                )));
            }
        }
    }
}

/// Run the headless engine: handshake, host the engine core, serve the wire
/// protocol until `shutdown` is cancelled or the frontend goes away.
pub async fn run_engine_proc(cfg: EngineProcConfig, shutdown: CancellationToken) -> Result<()> {
    let mut core_cfg = cfg.core.clone();
    let identity = engine_identity(cfg.engine_index)?;

    // ---- 1+2. HELLO → INIT on the handshake socket. ----

    // In managed mode the supervisor spawns this process *before* the frontend
    // binds the handshake endpoint (the frontend binds it inside its engine
    // connect), so the dial retries with backoff until INIT arrives or the
    // init timeout elapses.
    let (mut handshake, init) = tokio::select! {
        r = dial_handshake(&cfg.handshake_address, identity.clone(), cfg.init_timeout) => r?,
        _ = shutdown.cancelled() => return Ok(()),
    };
    let input_address = init
        .addresses
        .inputs
        .first()
        .cloned()
        .context("INIT carried no input address")?;
    let output_address = init
        .addresses
        .outputs
        .first()
        .cloned()
        .context("INIT carried no output address")?;
    if let Some(ctrl) = &init.generation_controls {
        apply_generation_controls(&mut core_cfg, ctrl);
    }

    // ---- 3. Build the runtime: spawns the worker/sim and waits on the
    // worker's get_caps (model load can take minutes for the real worker). ----
    info!(model = %core_cfg.model, "building engine core (worker handshake / model load)");
    let core = tokio::task::spawn_blocking(move || {
        if core_cfg.backend == uniserve_engine_runtime::EngineBackend::Sim {
            EngineCore::with_executor(
                core_cfg,
                Box::new(SimExecutor::new(Box::new(SimEngine::new()))),
            )
        } else {
            EngineCore::new(core_cfg)
        }
    })
    .await
    .context("engine runtime build task panicked")??;
    let core = Arc::new(core);

    // ---- 4. Register on the data plane: input DEALER (identity + ready
    // response payload) and output PUSH. ----
    let mut input_options = SocketOptions::default();
    input_options.peer_identity(identity);
    let mut input = DealerSocket::with_options(input_options);
    input
        .connect(&input_address)
        .await
        .with_context(|| format!("connecting input {input_address}"))?;
    input
        .send(ZmqMessage::from(encode_msgpack(&ready_response(&core))?))
        .await?;

    let mut output = PushSocket::new();
    output
        .connect(&output_address)
        .await
        .with_context(|| format!("connecting output {output_address}"))?;

    // ---- 5. READY: post-load truth has been registered; open the gate. ----
    handshake
        .send(ZmqMessage::from(encode_msgpack(&status_message("READY"))?))
        .await?;
    info!(engine_index = cfg.engine_index, "engine ready");

    // ---- 6. Serve. ----
    let (out_tx, out_rx) = mpsc::channel::<OutMsg>(OUTPUT_BUFFER_CAPACITY);
    let output_task = tokio::spawn(run_output_loop(
        cfg.engine_index,
        output,
        out_rx,
        Arc::clone(core.stats()),
        core.caps().block_size,
    ));

    // Engine-dead monitor: when the scheduler loop dies (worker failure), push
    // the ENGINE_CORE_DEAD sentinel so the frontend fails fast, then bring this
    // process down (the managed-mode supervisor
    // observes the exit).
    let dead_watch = {
        let core = Arc::clone(&core);
        let out_tx = out_tx.clone();
        let shutdown = shutdown.clone();
        async move {
            loop {
                if core.is_dead() {
                    warn!("engine core died; emitting ENGINE_CORE_DEAD");
                    let _ = out_tx.send(OutMsg::Dead).await;
                    // Give the output loop a beat to flush the sentinel.
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    return;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(200)) => {}
                    _ = shutdown.cancelled() => std::future::pending::<()>().await,
                }
            }
        }
    };

    let active: SharedActiveRequests = Arc::new(Mutex::new(HashMap::new()));
    let serve_result = tokio::select! {
        r = run_input_loop(&core, &active, &out_tx, &mut input, &shutdown) => r,
        _ = dead_watch => Err(anyhow::anyhow!("engine core died (worker failure)")),
    };

    // Tear down: stop accepting, stop the engine, flush the output task.
    drop(out_tx);
    let _ = output_task.await;
    let core = Arc::try_unwrap(core);
    match core {
        Ok(core) => tokio::task::spawn_blocking(move || core.shutdown())
            .await
            .context("engine shutdown task panicked")?,
        Err(core_arc) => {
            // Event-forwarding tasks still hold clones; shut down through the shared ref.
            core_arc.shutdown();
        }
    }
    serve_result
}

/// Decode and dispatch input frames until shutdown or socket close.
async fn run_input_loop(
    core: &Arc<EngineCore>,
    active: &SharedActiveRequests,
    out_tx: &mpsc::Sender<OutMsg>,
    input: &mut DealerSocket,
    shutdown: &CancellationToken,
) -> Result<()> {
    loop {
        let message = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),
            m = input.recv() => m.context("input socket closed")?,
        };
        let frames = message.into_vec();
        if frames.len() != 2 {
            warn!(
                frame_count = frames.len(),
                "malformed input message (ignored)"
            );
            continue;
        }
        let request = match EngineRequest::decode_frames(frames[0].as_ref(), frames[1].as_ref()) {
            None => {
                warn!(type_frame = ?frames[0].as_ref(), "unknown request type (ignored)");
                continue;
            }
            Some(Err(e)) => {
                warn!(error = %e, "failed to decode request (ignored)");
                continue;
            }
            Some(Ok(request)) => request,
        };

        match request {
            EngineRequest::Submit(req) => handle_submit(core, active, out_tx, *req).await,
            EngineRequest::Abort(ids) => {
                let handle = core.handle();
                let active = lock_active(active);
                for id in &ids {
                    if let Some(rid) = active.get(id) {
                        handle.abort(*rid);
                    }
                }
            }
            EngineRequest::Cancel(ids) => {
                let handle = core.handle();
                let active = lock_active(active);
                for id in &ids {
                    if let Some(rid) = active.get(id) {
                        handle.cancel(*rid);
                    }
                }
            }
            EngineRequest::CancelAt(requests) => {
                let handle = core.handle();
                let active = lock_active(active);
                for request in requests {
                    if let Some(rid) = active.get(&request.external_request_id)
                        && let Ok(output_token_count) = usize::try_from(request.output_token_count)
                    {
                        handle.cancel_at(*rid, output_token_count);
                    }
                }
            }
            EngineRequest::AcknowledgeAt(requests) => {
                let handle = core.handle();
                let active = lock_active(active);
                for request in requests {
                    if let Some(rid) = active.get(&request.external_request_id)
                        && let Ok(output_token_count) = usize::try_from(request.output_token_count)
                    {
                        handle.acknowledge_at(*rid, output_token_count);
                    }
                }
            }
            EngineRequest::StopAt(requests) => {
                let handle = core.handle();
                let active = lock_active(active);
                for request in requests {
                    if let Some(rid) = active.get(&request.external_request_id)
                        && let Ok(output_token_count) = usize::try_from(request.output_token_count)
                    {
                        handle.stop_at(*rid, output_token_count);
                    }
                }
            }
        }
    }
}

/// Register and submit one canonical request, then forward its event stream.
async fn handle_submit(
    core: &Arc<EngineCore>,
    active: &SharedActiveRequests,
    out_tx: &mpsc::Sender<OutMsg>,
    mut envelope: GenerationRequestEnvelope,
) {
    let request_id = envelope.external_request_id.clone();
    if let Err(error) = envelope.validate() {
        warn!(request_id, %error, "generation request rejected");
        let _ = out_tx
            .send(OutMsg::Event(Box::new(RoutedGenerationEvent {
                external_request_id: request_id,
                event: GenEvent::Rejected {
                    message: error.to_string(),
                },
            })))
            .await;
        return;
    }
    let rid = core.next_request_id();
    if !insert_active_request(active, &request_id, rid) {
        let _ = out_tx
            .send(OutMsg::Event(Box::new(RoutedGenerationEvent {
                external_request_id: request_id,
                event: GenEvent::Rejected {
                    message: "request id is already active".to_string(),
                },
            })))
            .await;
        return;
    }
    envelope.request.request_id = rid;
    let mut event_rx = match core.submit(envelope.request) {
        Ok(event_rx) => event_rx,
        Err(error) => {
            warn!(request_id, %error, "submit failed");
            remove_active_request(active, &request_id, rid);
            let _ = out_tx
                .send(OutMsg::Event(Box::new(RoutedGenerationEvent {
                    external_request_id: request_id,
                    event: GenEvent::Error {
                        message: error.to_string(),
                    },
                })))
                .await;
            return;
        }
    };
    event_rx.delegate_cancellation();

    let out_tx = out_tx.clone();
    let active = Arc::clone(active);
    tokio::spawn(async move {
        let mut reached_terminal = false;
        while let Some(event) = event_rx.recv().await {
            reached_terminal = matches!(
                event,
                GenEvent::Finished { .. } | GenEvent::Rejected { .. } | GenEvent::Error { .. }
            );
            if out_tx
                .send(OutMsg::Event(Box::new(RoutedGenerationEvent {
                    external_request_id: request_id.clone(),
                    event,
                })))
                .await
                .is_err()
            {
                break;
            }
            if reached_terminal {
                break;
            }
        }
        if !reached_terminal {
            let _ = out_tx
                .send(OutMsg::Event(Box::new(RoutedGenerationEvent {
                    external_request_id: request_id.clone(),
                    event: GenEvent::Error {
                        message: "generation event stream closed before a terminal event"
                            .to_string(),
                    },
                })))
                .await;
        }
        remove_active_request(&active, &request_id, rid);
    });
}

/// Batch canonical generation events onto the PUSH socket.
async fn run_output_loop(
    engine_index: u32,
    mut socket: PushSocket,
    mut rx: mpsc::Receiver<OutMsg>,
    stats: std::sync::Arc<uniserve_scheduler::SchedStats>,
    block_size: u32,
) {
    const MAX_BATCH: usize = 128;
    let mut reporter = stats::SchedStatsReporter::default();
    while let Some(msg) = rx.recv().await {
        let batch = match msg {
            OutMsg::Dead => {
                let _ = socket
                    .send(ZmqMessage::from(
                        uniserve_engine_wire::ENGINE_CORE_DEAD_SENTINEL.to_vec(),
                    ))
                    .await;
                return;
            }
            OutMsg::Event(first) => {
                let mut events = vec![*first];
                let mut dead_after = false;
                while events.len() < MAX_BATCH {
                    match rx.try_recv() {
                        Ok(OutMsg::Event(event)) => events.push(*event),
                        Ok(OutMsg::Dead) => {
                            dead_after = true;
                            break;
                        }
                        Err(_) => break,
                    }
                }
                if dead_after {
                    let _ = socket
                        .send(ZmqMessage::from(
                            uniserve_engine_wire::ENGINE_CORE_DEAD_SENTINEL.to_vec(),
                        ))
                        .await;
                    return;
                }
                GenerationEventBatch {
                    engine_index,
                    events,
                    // Live scheduler load rides every request batch: it feeds the frontend's
                    // routing score and Prometheus metrics.
                    scheduler_stats: Some(Box::new(reporter.snapshot(&stats, block_size))),
                    emitted_at: now_secs(),
                }
            }
        };

        let bytes = match encode_msgpack(&batch) {
            Ok(bytes) => bytes,
            Err(error) => {
                warn!(%error, "failed to encode generation event batch");
                continue;
            }
        };
        if let Err(error) = socket.send(ZmqMessage::from(bytes)).await {
            warn!(%error, "output socket send failed; stopping output loop");
            return;
        }
    }
}
