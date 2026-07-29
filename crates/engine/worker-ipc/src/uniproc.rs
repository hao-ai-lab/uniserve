//! Single-worker executor over the iceoryx2 request-response service.
//!
//! The host sends FlatBuffers descriptors and receives small scalar/image
//! results. Worker-resident tensors, KV pages, and latents never cross this
//! boundary.

use std::collections::{HashMap, VecDeque};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::Serialize;
use uniserve_core::CommandWaker;
use uniserve_executor::{ControlAck, ControlOp, Executor, WorkerExecError};
use uniserve_worker_ipc_core::{
    ClientEndpoint, Frame, Pending, event_driven_enabled, service_name,
};
use uniserve_worker_wire::{
    Batch, CompletionReport, EngineCaps, ResponseKind, WorkerRequest, WorkerResponse,
};

use crate::death_watch::DeathWatcher;

fn enqueue_ready(ready: &mut VecDeque<CompletionReport>, report: CompletionReport) {
    ready.push_back(report);
}

/// Deadline for the initial worker connect / caps handshake, where the worker may still
/// be loading a large model and the IPC server may not yet be connected.
const WORKER_CONNECT_TIMEOUT: Duration = Duration::from_secs(300);
/// Per-call backpressure deadline for steady-state sends (batch submit / control). The
/// IPC server is already connected by this point, so a missing server connection means
/// the worker has dropped off and we should fail fast rather than block for the full
/// startup grace period.
const WORKER_SEND_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum time `shutdown` will spend draining in-flight responses from a still-alive
/// worker before falling through to the graceful-shutdown request and kill fallback. A
/// hung (alive but unresponsive) worker must not be able to block shutdown forever.
const WORKER_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
const STARTUP_LOG_INTERVAL: Duration = Duration::from_secs(30);
const RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(1);
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// Explicit Python worker launch configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WorkerLaunchConfig {
    pub stub: bool,
    pub model_dtype: String,
    pub kv_cache_dtype: Option<String>,
    pub kv_memory_fraction: String,
    pub transformers_trust_remote_code: bool,
    pub transformers_attn_implementation: String,
    pub disable_model_arch: Vec<String>,
    pub mesh: Option<String>,
    pub tp_backend: Option<String>,
    pub mooncake_device: String,
    pub mooncake_protocol: String,
    pub torch_compile: bool,
    pub torch_compile_backend: String,
    pub torch_compile_mode: Option<String>,
    pub torch_compile_fullgraph: bool,
    pub torch_compile_dynamic: Option<String>,
    pub cuda_graph: bool,
    pub cuda_graph_warmup: bool,
    pub cuda_graph_warmup_batches: Option<String>,
    pub prefill_cuda_graph: bool,
    pub prefill_cuda_graph_warmup: bool,
    pub prefill_cuda_graph_warmup_tokens: Option<String>,
    pub mixed_text_max_tokens: u32,
    pub varlen_prefill: bool,
    pub flashinfer_workspace_size: u64,
    pub flashinfer_use_tensor_core: Option<String>,
    pub flashinfer_decode_backend: String,
    pub flashinfer_prefill_backend: String,
    pub flashinfer_decode_split_tile_size: Option<u32>,
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    pub flashinfer_disable_split_kv: bool,
    pub flashinfer_fast_decode_plan: bool,
    pub snapshot_dir: Option<String>,
    pub restore_snapshots: bool,
}

impl Default for WorkerLaunchConfig {
    fn default() -> Self {
        Self {
            stub: false,
            model_dtype: "bfloat16".to_string(),
            kv_cache_dtype: None,
            kv_memory_fraction: "0.70".to_string(),
            transformers_trust_remote_code: false,
            transformers_attn_implementation: "uniserve".to_string(),
            disable_model_arch: Vec::new(),
            mesh: None,
            tp_backend: None,
            mooncake_device: String::new(),
            mooncake_protocol: "rdma".to_string(),
            torch_compile: false,
            torch_compile_backend: "inductor".to_string(),
            torch_compile_mode: None,
            torch_compile_fullgraph: false,
            torch_compile_dynamic: None,
            cuda_graph: true,
            cuda_graph_warmup: true,
            cuda_graph_warmup_batches: None,
            prefill_cuda_graph: false,
            prefill_cuda_graph_warmup: false,
            prefill_cuda_graph_warmup_tokens: None,
            mixed_text_max_tokens: 8192,
            varlen_prefill: true,
            flashinfer_workspace_size: 512 * 1024 * 1024,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: "fa2".to_string(),
            flashinfer_prefill_backend: "auto".to_string(),
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_disable_split_kv: false,
            flashinfer_fast_decode_plan: true,
            snapshot_dir: None,
            restore_snapshots: false,
        }
    }
}

impl WorkerLaunchConfig {
    fn append_worker_args(&self, cmd: &mut Command) {
        if self.stub {
            cmd.arg("--no-model").arg("--allow-stub");
        }
        cmd.arg("--model-dtype").arg(&self.model_dtype);
        if let Some(value) = &self.kv_cache_dtype {
            cmd.arg("--kv-cache-dtype").arg(value);
        }
        cmd.arg("--kv-memory-fraction")
            .arg(&self.kv_memory_fraction);
        if self.transformers_trust_remote_code {
            cmd.arg("--transformers-trust-remote-code");
        }
        cmd.arg("--transformers-attn-implementation")
            .arg(&self.transformers_attn_implementation);
        for arch in &self.disable_model_arch {
            cmd.arg("--disable-model-arch").arg(arch);
        }
        if let Some(value) = &self.mesh {
            cmd.arg("--mesh").arg(value);
        }
        if let Some(value) = &self.tp_backend {
            cmd.arg("--tp-backend").arg(value);
        }
        if !self.mooncake_device.is_empty() {
            cmd.arg("--mooncake-device").arg(&self.mooncake_device);
        }
        cmd.arg("--mooncake-protocol").arg(&self.mooncake_protocol);
        if self.torch_compile {
            cmd.arg("--torch-compile");
        }
        cmd.arg("--torch-compile-backend")
            .arg(&self.torch_compile_backend);
        if let Some(value) = &self.torch_compile_mode {
            cmd.arg("--torch-compile-mode").arg(value);
        }
        if self.torch_compile_fullgraph {
            cmd.arg("--torch-compile-fullgraph");
        }
        if let Some(value) = &self.torch_compile_dynamic {
            cmd.arg("--torch-compile-dynamic").arg(value);
        }
        if !self.cuda_graph {
            cmd.arg("--no-cuda-graph");
        }
        if !self.cuda_graph_warmup {
            cmd.arg("--no-cuda-graph-warmup");
        }
        if let Some(value) = &self.cuda_graph_warmup_batches {
            cmd.arg("--cuda-graph-warmup-batches").arg(value);
        }
        if self.prefill_cuda_graph {
            cmd.arg("--prefill-cuda-graph");
        }
        if self.prefill_cuda_graph_warmup {
            cmd.arg("--prefill-cuda-graph-warmup");
        }
        if let Some(value) = &self.prefill_cuda_graph_warmup_tokens {
            cmd.arg("--prefill-cuda-graph-warmup-tokens").arg(value);
        }
        cmd.arg("--mixed-text-max-tokens")
            .arg(self.mixed_text_max_tokens.to_string());
        if !self.varlen_prefill {
            cmd.arg("--no-varlen-prefill");
        }
        cmd.arg("--flashinfer-workspace-size")
            .arg(self.flashinfer_workspace_size.to_string());
        if let Some(value) = &self.flashinfer_use_tensor_core {
            cmd.arg("--flashinfer-use-tensor-core").arg(value);
        }
        cmd.arg("--flashinfer-decode-backend")
            .arg(&self.flashinfer_decode_backend);
        cmd.arg("--flashinfer-prefill-backend")
            .arg(&self.flashinfer_prefill_backend);
        if let Some(value) = self.flashinfer_decode_split_tile_size {
            cmd.arg("--flashinfer-decode-split-tile-size")
                .arg(value.to_string());
        }
        if let Some(value) = self.flashinfer_prefill_split_tile_size {
            cmd.arg("--flashinfer-prefill-split-tile-size")
                .arg(value.to_string());
        }
        if self.flashinfer_disable_split_kv {
            cmd.arg("--flashinfer-disable-split-kv");
        }
        if !self.flashinfer_fast_decode_plan {
            cmd.arg("--no-flashinfer-fast-decode-plan");
        }
        if let Some(value) = &self.snapshot_dir {
            cmd.arg("--snapshot-dir").arg(value);
        }
        if self.restore_snapshots {
            cmd.arg("--restore-snapshots");
        }
    }
}

/// Single-process worker executor over iceoryx2 IPC.
pub struct UniprocExecutor {
    client: ClientEndpoint,
    caps: EngineCaps,
    child: Child,
    depth: usize,
    rank: u32,
    tp_size: u32,
    pending: HashMap<u64, PendingRecord>,
    ready: VecDeque<CompletionReport>,
    acks: HashMap<u64, ControlAck>,
    awaited: Option<u64>,
    next_call_id: u64,
    shutdown_sent: bool,
    /// Edge-triggered worker-death watcher: fires the scheduler park's death
    /// wake when the child exits. `None` when polling or when `pidfd` could not
    /// be opened (falls back to the bounded liveness probe).
    death_watcher: Option<DeathWatcher>,
}

struct PendingRecord {
    kind: OutstandingKind,
    pending: Pending,
}

enum OutstandingKind {
    Batch { step_id: u64 },
    Control,
}

impl UniprocExecutor {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        python: &str,
        model_dir: &str,
        device: &str,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
    ) -> anyhow::Result<Self> {
        Self::spawn_with_config(
            python,
            model_dir,
            device,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        Self::spawn_ranked_with_config(
            python,
            model_dir,
            device,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            0,
            1,
            None,
            worker_config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ranked(
        python: &str,
        model_dir: &str,
        device: &str,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        tp_rank: u32,
        tp_size: u32,
        tp_init_method: Option<&str>,
    ) -> anyhow::Result<Self> {
        Self::spawn_ranked_with_config(
            python,
            model_dir,
            device,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            tp_rank,
            tp_size,
            tp_init_method,
            &WorkerLaunchConfig::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_ranked_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        tp_rank: u32,
        tp_size: u32,
        tp_init_method: Option<&str>,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        let mut me = Self::spawn_ranked_deferred_with_config(
            python,
            model_dir,
            device,
            pipeline_depth,
            req_slot_cap,
            resp_slot_cap,
            kv_token_capacity,
            block_size,
            attention_backend,
            tp_rank,
            tp_size,
            tp_init_method,
            None,
            None,
            false,
            worker_config,
        )?;
        me.finish_startup()?;
        Ok(me)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn spawn_ranked_deferred_with_config(
        python: &str,
        model_dir: &str,
        device: &str,
        pipeline_depth: usize,
        req_slot_cap: usize,
        resp_slot_cap: usize,
        kv_token_capacity: Option<u64>,
        block_size: u32,
        attention_backend: &str,
        tp_rank: u32,
        tp_size: u32,
        tp_init_method: Option<&str>,
        worker_kind: Option<&str>,
        transfer_backend: Option<&str>,
        defer_sampling: bool,
        worker_config: &WorkerLaunchConfig,
    ) -> anyhow::Result<Self> {
        let depth = effective_worker_pipeline_depth(pipeline_depth, tp_size);
        let max_payload = req_slot_cap.max(resp_slot_cap).max(1);
        let service = service_name(&format!("{}_{}_{}", std::process::id(), tp_rank, nano_id()));
        // The host is authoritative for the boundary mode: resolve it once and
        // force the worker to match (env var on the child) so the two ends never
        // disagree on whether to run event-driven or polled.
        let event_driven = event_driven_enabled();

        let mut cmd = Command::new(python);
        cmd.arg("-m")
            .arg("uniserve_worker.main")
            .arg("--service-name")
            .arg(&service)
            .arg("--pipeline-depth")
            .arg(depth.to_string())
            .arg("--ipc-payload-cap")
            .arg(max_payload.to_string())
            .arg("--ipc-max-inflight")
            .arg(depth.to_string())
            .arg("--model")
            .arg(model_dir)
            .arg("--device")
            .arg(device)
            .arg("--attention-backend")
            .arg(attention_backend)
            .arg("--block-size")
            .arg(block_size.to_string())
            .arg("--tp-rank")
            .arg(tp_rank.to_string())
            .arg("--tp-size")
            .arg(tp_size.to_string());
        // Staged topology: tell the worker which pipeline stage it serves.
        // Omitted for the default `full` worker so the command line stays
        // byte-identical to the non-disaggregated path.
        if let Some(kind) = worker_kind {
            cmd.arg("--worker-kind").arg(kind);
        }
        // Data-plane Tier-2 backend for this stage's tensor handoffs. The
        // default (in-process) is omitted so the non-disaggregated worker command
        // line stays byte-identical.
        if let Some(backend) = transfer_backend.filter(|b| *b != "inproc") {
            cmd.arg("--transfer-backend").arg(backend);
        }
        // Sampler-stage split: a model pool peeled from its sampler publishes
        // logits + defers sampling.
        if defer_sampling {
            cmd.arg("--defer-sampling");
        }
        cmd.env("RANK", tp_rank.to_string())
            .env("WORLD_SIZE", tp_size.to_string())
            .env("LOCAL_RANK", tp_rank.to_string())
            .env("LOCAL_WORLD_SIZE", tp_size.to_string())
            .env(
                uniserve_worker_ipc_core::EVENT_DRIVEN_ENV,
                if event_driven { "1" } else { "0" },
            );
        // Serving batches change shape continuously, and fixed-size cached
        // segments strand device memory that later shapes cannot use.
        // Expandable segments let the allocator resize its mapping instead, so
        // the worker keeps serving instead of exhausting the device on a
        // workload whose geometry keeps moving.
        if std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none() {
            cmd.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
        }
        if tp_size > 1
            && let Some(init_method) = tp_init_method
        {
            cmd.arg("--tp-init-method").arg(init_method);
        }
        if let Some(c) = kv_token_capacity {
            cmd.arg("--kv-token-capacity").arg(c.to_string());
        }
        worker_config.append_worker_args(&mut cmd);
        if let Ok(cwd) = std::env::current_dir() {
            let pp = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env("PYTHONPATH", format!("{}:{}", cwd.display(), pp));
        }
        let child = cmd.spawn().context("spawning python worker")?;

        let client = ClientEndpoint::connect_with(&service, max_payload, depth, event_driven)
            .context("connecting to worker IPC service")?;
        // Wire the edge-triggered death watcher onto the park's death wake. Only
        // meaningful on the event-driven path (the wake feeds the park
        // listener); on the polling path the scheduler probes liveness on its
        // own timer.
        let death_watcher = client
            .death_wake()
            .and_then(|wake| DeathWatcher::spawn(child.id(), wake));
        Ok(Self {
            client,
            caps: EngineCaps::default(),
            child,
            depth,
            rank: tp_rank,
            tp_size,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            acks: HashMap::new(),
            awaited: None,
            next_call_id: 1,
            shutdown_sent: false,
            death_watcher,
        })
    }

    pub(crate) fn finish_startup(&mut self) -> anyhow::Result<()> {
        tracing::info!(
            tp_rank = self.rank,
            tp_size = self.tp_size,
            "waiting for worker to load model + report caps..."
        );
        let call_id = self.alloc_call_id();
        let mut req = WorkerRequest::get_capabilities();
        req.call_id = Some(call_id);
        let pending =
            self.send_request_with_timeout(&req, "caps handshake", WORKER_CONNECT_TIMEOUT)?;
        let resp = self.wait_pending_response(&pending, "caps handshake")?;
        let wr = resp.decode_response()?;
        let caps = match wr.kind {
            ResponseKind::Capabilities => wr
                .capabilities
                .ok_or_else(|| anyhow::anyhow!("capabilities response is missing its payload"))?,
            ResponseKind::Error => bail!(
                "worker error during caps: {}",
                wr.message.unwrap_or_default()
            ),
            kind => bail!("unexpected capabilities response kind: {kind:?}"),
        };
        let host_depth = self.depth as u32;
        anyhow::ensure!(
            caps.pipeline_depth == host_depth,
            "worker pipeline_depth {} does not match launched depth {}",
            caps.pipeline_depth,
            host_depth
        );
        anyhow::ensure!(
            caps.rank.tp_rank == self.rank && caps.rank.tp_size == self.tp_size,
            "worker TP rank/size ({}/{}) does not match launched topology ({}/{})",
            caps.rank.tp_rank,
            caps.rank.tp_size,
            self.rank,
            self.tp_size
        );
        self.caps = caps;
        tracing::info!(?self.caps, "worker ready");
        Ok(())
    }

    fn alloc_call_id(&mut self) -> u64 {
        let id = self.next_call_id;
        self.next_call_id += 1;
        id
    }

    fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if let Some(status) = self.child.try_wait()? {
            bail!("worker process exited during {context}: {status}");
        }
        Ok(())
    }

    fn send_request_checked(
        &mut self,
        req: &WorkerRequest,
        context: &str,
    ) -> anyhow::Result<Pending> {
        self.send_request_with_timeout(req, context, WORKER_SEND_TIMEOUT)
    }

    fn send_request_with_timeout(
        &mut self,
        req: &WorkerRequest,
        context: &str,
        timeout: Duration,
    ) -> anyhow::Result<Pending> {
        let deadline = Instant::now() + timeout;
        loop {
            let pending = self.client.send_request_attempt(req)?;
            if pending.number_of_server_connections() > 0 {
                return Ok(pending);
            }
            drop(pending);
            self.check_worker(context)?;
            if Instant::now() >= deadline {
                bail!("worker IPC service had no connected server during {context}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn wait_pending_response(&mut self, pending: &Pending, context: &str) -> anyhow::Result<Frame> {
        let started = Instant::now();
        let mut last_log = started;
        loop {
            if let Some(frame) = self
                .client
                .recv_response_timeout(pending, WORKER_CHECK_INTERVAL)?
            {
                return Ok(frame);
            }
            self.check_worker(context)?;
            if last_log.elapsed() >= STARTUP_LOG_INTERVAL {
                tracing::info!(
                    elapsed_secs = started.elapsed().as_secs(),
                    "worker still busy during {context}"
                );
                last_log = Instant::now();
            }
        }
    }

    fn drain_ready(&mut self) -> anyhow::Result<usize> {
        let ids = self.pending.keys().copied().collect::<Vec<_>>();
        let mut drained = 0usize;
        for call_id in ids {
            let Some(record) = self.pending.get(&call_id) else {
                continue;
            };
            let Some(frame) = self.client.try_recv_response(&record.pending)? else {
                continue;
            };
            let record = self.pending.remove(&call_id).ok_or_else(|| {
                anyhow::anyhow!("pending record {call_id} disappeared while routing response")
            })?;
            self.route(call_id, record.kind, frame)?;
            drained += 1;
        }
        Ok(drained)
    }

    fn route(&mut self, call_id: u64, kind: OutstandingKind, frame: Frame) -> anyhow::Result<()> {
        if frame.header.call_id != 0 && frame.header.call_id != call_id {
            bail!(
                "worker response call id mismatch: expected {call_id}, got {}",
                frame.header.call_id
            );
        }
        let wr = frame.decode_response()?;
        if let Some(echoed) = wr.call_id
            && echoed != call_id
        {
            bail!("worker response echoed call id {echoed}, expected {call_id}");
        }
        match kind {
            OutstandingKind::Batch { step_id } => self.route_batch(step_id, wr),
            OutstandingKind::Control => {
                let (ok, snapshot) = match wr.kind {
                    ResponseKind::Ok => (true, None),
                    ResponseKind::Snapshot => (
                        true,
                        Some(wr.snapshot.clone().ok_or_else(|| {
                            anyhow::anyhow!("snapshot response is missing its payload")
                        })?),
                    ),
                    ResponseKind::Error => (false, None),
                    kind => bail!("unexpected control response kind: {kind:?}"),
                };
                if !ok {
                    tracing::error!(
                        call_id,
                        "worker control call error: {}",
                        wr.message.clone().unwrap_or_default()
                    );
                }
                if self.awaited == Some(call_id) {
                    self.acks.insert(
                        call_id,
                        ControlAck {
                            rank: self.rank,
                            ok,
                            message: wr.message,
                            snapshot,
                        },
                    );
                }
                Ok(())
            }
        }
    }

    fn route_batch(&mut self, step_id: u64, wr: WorkerResponse) -> anyhow::Result<()> {
        match wr.kind {
            ResponseKind::Result => {
                let r = wr
                    .completion_report
                    .ok_or_else(|| anyhow::anyhow!("completion report missing"))?;
                if r.step_id != step_id {
                    bail!(
                        "worker result step id mismatch: expected {step_id}, got {}",
                        r.step_id
                    );
                }
                enqueue_ready(&mut self.ready, r);
                Ok(())
            }
            ResponseKind::Error => {
                let fatal = wr
                    .fatal
                    .ok_or_else(|| anyhow::anyhow!("error response missing fatality"))?;
                let retryable = wr
                    .retryable
                    .ok_or_else(|| anyhow::anyhow!("error response missing retryability"))?;
                Err(WorkerExecError {
                    fatal,
                    retryable,
                    code: wr.code.clone(),
                    message: wr.message.clone().unwrap_or_default(),
                    phase: wr.phase.clone(),
                    route: wr.route.clone(),
                    operations: wr.operations.clone(),
                }
                .into())
            }
            kind => bail!("unexpected execute response kind: {kind:?}"),
        }
    }

    fn wait_for_one_response(&mut self) -> anyhow::Result<()> {
        if self.pending.is_empty() {
            bail!("no pending worker requests to wait for");
        }
        loop {
            let drained = self.drain_ready()?;
            if drained > 0 {
                return Ok(());
            }
            self.check_worker("worker response wait")?;
            // Park on the worker's result wake instead of polling on a fixed
            // 1ms interval: the worker fires EVT_RESULT the instant it responds,
            // so the submitter wakes in ~event latency rather than waiting out
            // the poll. `wait_wake` degrades to a bounded sleep on a
            // non-event-driven endpoint, so the safety-net slice still bounds
            // latency if a notification is missed.
            self.client.wait_wake(RESPONSE_POLL_INTERVAL)?;
        }
    }

    fn ensure_slot(&mut self) -> anyhow::Result<()> {
        self.drain_ready()?;
        while self.pending.len() >= self.depth {
            self.wait_for_one_response()?;
        }
        Ok(())
    }

    fn submit_control_request(&mut self, req: &WorkerRequest, call_id: u64) -> anyhow::Result<()> {
        if self.shutdown_sent {
            // The worker is being torn down; drop the control op rather than send onto a
            // closing IPC channel. Log so this is observable instead of a silent no-op.
            tracing::debug!(
                call_id,
                "dropping control request: executor already shutting down"
            );
            return Ok(());
        }
        // Control ops deliberately share the pipeline-depth slot budget with batches: the
        // worker is launched with --ipc-max-inflight = depth, so total outstanding requests
        // (batch + control) must not exceed `depth` or we would overflow the IPC ring.
        self.ensure_slot()?;
        let pending = self.send_request_checked(req, "control request")?;
        self.pending.insert(
            call_id,
            PendingRecord {
                kind: OutstandingKind::Control,
                pending,
            },
        );
        Ok(())
    }

    fn batch_pending_count(&self) -> usize {
        self.pending
            .values()
            .filter(|record| matches!(record.kind, OutstandingKind::Batch { .. }))
            .count()
    }
}

impl Executor for UniprocExecutor {
    fn caps(&self) -> EngineCaps {
        self.caps.clone()
    }

    fn pipeline_depth(&self) -> usize {
        self.depth
    }

    fn in_flight(&self) -> usize {
        self.batch_pending_count() + self.ready.len()
    }

    fn can_submit(&self) -> bool {
        self.pending.len() < self.depth
    }

    fn event_driven(&self) -> bool {
        self.client.is_event_driven()
    }

    fn command_waker(&self) -> CommandWaker {
        match self.client.command_wake() {
            Some(sender) => CommandWaker::new(move || sender.wake()),
            None => CommandWaker::noop(),
        }
    }

    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        // Park over {result, command, death} without consuming anything: the
        // scheduler drains results, drains commands, and probes liveness after
        // we return, so a spurious wake is harmless. The wait carries its own
        // safety-net slice, so a missed notification degrades to poll latency.
        self.client.wait_wake(timeout)?;
        Ok(())
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()> {
        self.drain_ready()?;
        if !self.can_submit() {
            self.ensure_slot()?;
        }
        let step_id = batch.step_id;
        let call_id = self.alloc_call_id();
        let mut req = WorkerRequest::execute(batch);
        req.call_id = Some(call_id);
        let pending = self.send_request_checked(&req, "batch submit")?;
        self.pending.insert(
            call_id,
            PendingRecord {
                kind: OutstandingKind::Batch { step_id },
                pending,
            },
        );
        Ok(())
    }

    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>> {
        self.drain_ready()?;
        Ok(self.ready.pop_front())
    }

    fn check_liveness(&mut self) -> anyhow::Result<()> {
        // Non-blocking reap of the worker child so death is observed even when
        // no requests are in flight.
        self.check_worker("idle liveness check")
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<CompletionReport>> {
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            self.drain_ready()?;
            return Ok(self.ready.pop_front());
        };
        loop {
            self.drain_ready()?;
            if let Some(r) = self.ready.pop_front() {
                return Ok(Some(r));
            }
            if self.batch_pending_count() == 0 {
                return Ok(None);
            }
            self.check_worker("worker result timed wait")?;
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            // Event-driven wake on EVT_RESULT; bounded by the remaining deadline
            // so the timeout contract is preserved.
            self.client
                .wait_wake((deadline - now).min(RESPONSE_POLL_INTERVAL))?;
        }
    }

    fn next_result(&mut self) -> anyhow::Result<CompletionReport> {
        loop {
            self.drain_ready()?;
            if let Some(r) = self.ready.pop_front() {
                return Ok(r);
            }
            if self.batch_pending_count() == 0 {
                bail!("next_result called with no in-flight batches");
            }
            self.wait_for_one_response()?;
        }
    }

    /// Fire-and-forget control op. The returned `u64` MUST be treated as opaque: callers
    /// should discard it and use [`Executor::control_wait`] when they need to correlate an
    /// ack. In this single-worker transport the value happens to be the genuine wire
    /// call_id the worker echoes, but the multiproc/disagg transports return a private
    /// counter that matches no worker request, so no caller may assume these semantics.
    /// `0` is returned for empty copy or product-release controls that are never sent.
    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64> {
        match &op {
            ControlOp::CopyKv(copies) if copies.is_empty() => return Ok(0),
            ControlOp::ReleaseProducts(handles) if handles.is_empty() => return Ok(0),
            _ => {}
        }
        let call_id = self.alloc_call_id();
        self.submit_control_request(&op.to_request(call_id), call_id)?;
        Ok(call_id)
    }

    fn control_wait(
        &mut self,
        op: ControlOp,
        _targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>> {
        let call_id = self.alloc_call_id();
        self.awaited = Some(call_id);
        self.submit_control_request(&op.to_request(call_id), call_id)?;
        loop {
            self.drain_ready()?;
            if let Some(ack) = self.acks.remove(&call_id) {
                self.awaited = None;
                return Ok(vec![ack]);
            }
            self.wait_for_one_response()?;
        }
    }

    fn shutdown(&mut self) {
        if self.shutdown_sent {
            return;
        }
        self.shutdown_sent = true;
        // Stop the death watcher before we intentionally tear the worker down,
        // so its exit does not fire a spurious death wake during shutdown.
        let _ = self.death_watcher.take();
        let exited = matches!(self.child.try_wait(), Ok(Some(_)));
        if !exited {
            // Drain in-flight responses, but do NOT block indefinitely on a worker that is
            // alive yet hung: bound the drain with a deadline so we always fall through to
            // the graceful shutdown request and, ultimately, the kill fallback below.
            let drain_deadline = Instant::now() + WORKER_DRAIN_TIMEOUT;
            while !self.pending.is_empty() {
                if Instant::now() >= drain_deadline {
                    tracing::warn!(
                        pending = self.pending.len(),
                        "worker did not drain in-flight responses before shutdown deadline"
                    );
                    break;
                }
                match self.drain_ready() {
                    Ok(0) => {
                        if self.check_worker("shutdown drain").is_err() {
                            break;
                        }
                        std::thread::sleep(RESPONSE_POLL_INTERVAL);
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            if matches!(self.child.try_wait(), Ok(None)) {
                let call_id = self.alloc_call_id();
                let mut req = WorkerRequest::shutdown();
                req.call_id = Some(call_id);
                if let Ok(pending) = self.send_request_checked(&req, "shutdown") {
                    let _ = self
                        .client
                        .recv_response_timeout(&pending, Duration::from_secs(5));
                }
            }
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
    }
}

impl Drop for UniprocExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub(crate) fn nano_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    // Build an id from a wall-clock timestamp in the high bits plus a process-wide
    // monotonic counter in the low 16 bits. subsec_nanos alone wraps every second and
    // would collide for workers spawned within the same wall-clock second; the counter
    // makes ids produced within any 65536-call window distinct regardless of clock
    // resolution or non-monotonicity, while the timestamp separates ids across windows.
    // The full service name also includes pid + tp_rank, so any residual aliasing in the
    // shifted timestamp bits cannot produce a real cross-worker collision.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    (nanos << 16) | (seq & 0xffff)
}

fn effective_worker_pipeline_depth(requested: usize, tp_size: u32) -> usize {
    let requested = requested.max(1);
    if tp_size > 1 {
        // Tensor-parallel ranks must enter collectives in the same order. A
        // per-rank Python pipeline can let one rank finalize an older response
        // while another rank starts the next forward, which is illegal for NCCL.
        1
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::{effective_worker_pipeline_depth, enqueue_ready, nano_id};
    use std::collections::{HashSet, VecDeque};
    use uniserve_worker_wire::{CompletionReport, RegistrationAck};

    fn report(step_id: u64) -> CompletionReport {
        CompletionReport {
            step_id,
            completions: Vec::new(),
            products: Vec::new(),
            registration: RegistrationAck::default(),
            worker_exec_us: None,
            forward_stats: None,
        }
    }

    #[test]
    fn nano_id_is_unique_within_a_tight_loop() {
        // subsec_nanos alone would collide here because the loop runs faster than a
        // second; the monotonic counter must make every id distinct.
        let n = 10_000;
        let ids: HashSet<u64> = (0..n).map(|_| nano_id()).collect();
        assert_eq!(
            ids.len(),
            n,
            "nano_id produced a collision within a tight loop"
        );
    }

    #[test]
    fn tensor_parallel_workers_do_not_pipeline_across_collectives() {
        assert_eq!(effective_worker_pipeline_depth(4, 1), 4);
        assert_eq!(effective_worker_pipeline_depth(0, 1), 1);
        assert_eq!(effective_worker_pipeline_depth(4, 2), 1);
        assert_eq!(effective_worker_pipeline_depth(4, 8), 1);
    }

    #[test]
    fn ready_reports_preserve_worker_readiness_order() {
        let mut ready = VecDeque::new();
        enqueue_ready(&mut ready, report(9));
        enqueue_ready(&mut ready, report(4));

        assert_eq!(ready.pop_front().map(|value| value.step_id), Some(9));
        assert_eq!(ready.pop_front().map(|value| value.step_id), Some(4));
    }
}
