//! Single-worker executor over an iceoryx2 request-response service.
//!
//! The host exchanges FlatBuffers descriptors and bounded result values while
//! tensors, KV pages, and latent storage remain worker-resident.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use crate::executor::{
    Batch, BatchResult, Executor, ExecutorInfo, ExecutorSubmitError, LogicalResultTracker,
    PhysicalExecutor, PhysicalSubmitError, PoolId, WorkerExecError, lower_batch,
};
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use uniserve_core::CommandWaker;
use uniserve_worker_ipc::{ClientEndpoint, Frame, Pending, service_name};
use uniserve_worker_ipc::{
    Domain, OpId, RequestKey, Run as PhysicalRun, RunResult, WorkerInfo, WorkerRequest,
    WorkerResponse,
};

use crate::worker::WorkerProcessArgs;
use crate::worker::death_watch::DeathWatcher;

/// Enqueues a completed worker result for delivery.
fn enqueue_ready(ready: &mut VecDeque<RunResult>, report: RunResult) {
    ready.push_back(report);
}

/// Deadline for the initial worker connect / info handshake, where the worker may still
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
const WORKER_CHECK_INTERVAL: Duration = Duration::from_millis(500);

/// One configured model-execution lane passed in [`WorkerProcessArgs`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaneConfig {
    /// Stable lane identity within the worker process.
    pub lane_id: String,
    /// Streaming-multiprocessor budget assigned to the lane.
    pub sm_budget: u32,
    /// Execution domains routed to the lane.
    pub domains: Vec<Domain>,
    /// Optional lane-local KV capacity in tokens.
    pub kv_capacity_tokens: Option<u64>,
    /// Optional lane-local latent capacity in allocation units.
    pub latent_capacity_units: Option<u64>,
    /// Optional operation-count limit per batch.
    pub max_batch_operations: Option<u32>,
    /// Optional token-count limit per batch.
    pub max_batch_tokens: Option<u32>,
    /// Optional unresolved-run limit.
    pub max_inflight: Option<u32>,
}

impl std::str::FromStr for LaneConfig {
    type Err = String;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let lane: Self = serde_json::from_str(value)
            .map_err(|error| format!("invalid execution lane JSON: {error}"))?;
        if lane.lane_id.is_empty()
            || lane.sm_budget == 0
            || lane.domains.is_empty()
            || lane.domains.iter().copied().collect::<HashSet<_>>().len() != lane.domains.len()
        {
            return Err("execution lane identity, SM budget, and domains must be valid".into());
        }
        Ok(lane)
    }
}

impl LaneConfig {
    /// Serializes the lane as the worker command-line JSON value.
    pub fn worker_arg(&self) -> String {
        serde_json::json!({
            "lane_id": self.lane_id,
            "sm_budget": self.sm_budget,
            "domains": self.domains,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "latent_capacity_units": self.latent_capacity_units,
            "max_batch_operations": self.max_batch_operations,
            "max_batch_tokens": self.max_batch_tokens,
            "max_inflight": self.max_inflight,
        })
        .to_string()
    }
}

/// FlashInfer implementation selected in [`WorkerProcessArgs`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashInferBackend {
    /// Lets the worker select a compatible backend.
    #[default]
    Auto,
    /// Uses FlashAttention 2 kernels.
    Fa2,
    /// Uses FlashAttention 3 kernels.
    Fa3,
}

impl FlashInferBackend {
    /// Returns the stable command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fa2 => "fa2",
            Self::Fa3 => "fa3",
        }
    }
}

impl std::str::FromStr for FlashInferBackend {
    type Err = FlashInferBackendParseError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "fa2" => Ok(Self::Fa2),
            "fa3" => Ok(Self::Fa3),
            _ => Err(FlashInferBackendParseError(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported FlashInfer backend {0:?}")]
/// Error returned for an unsupported FlashInfer backend name.
pub struct FlashInferBackendParseError(String);

impl Default for WorkerProcessArgs {
    /// Returns worker launch settings suitable for a single local rank.
    fn default() -> Self {
        Self {
            python: "python3".into(),
            model: String::new(),
            device: "cuda".into(),
            world_size: 1,
            pipeline_depth: 2,
            req_slot_cap: 1 << 20,
            resp_slot_cap: 8 << 20,
            kv_token_capacity: None,
            block_size: 64,
            max_batch_operations: 128,
            max_batch_tokens: 16_384,
            attention_backend: uniserve_worker_ipc::AttentionBackend::Auto,
            supported_ops: uniserve_worker_ipc::OpKind::ALL.to_vec(),
            transfer_backend: crate::executor::TransferBackend::Inproc,
            stub: false,
            load_format: "auto".to_string(),
            download_dir: None,
            load_threads: None,
            checksum_manifest: None,
            model_dtype: uniserve_core::ModelDtype::BFloat16,
            quantization_config: serde_json::json!({}),
            kv_cache_dtype: None,
            kv_memory_fraction: 0.70,
            mesh: None,
            tp_backend: None,
            lanes: Vec::new(),
            cuda_graph: true,
            decode_graph_batch_sizes: None,
            prefill_cuda_graph: false,
            prefill_graph_token_sizes: None,
            flow_graph_batch_sizes: None,
            flow_graph_shapes: None,
            flashinfer_workspace_size: 512 * 1024 * 1024,
            flashinfer_use_tensor_core: None,
            flashinfer_decode_backend: FlashInferBackend::Fa2,
            flashinfer_prefill_backend: FlashInferBackend::Auto,
            flashinfer_decode_split_tile_size: None,
            flashinfer_prefill_split_tile_size: None,
            flashinfer_disable_split_kv: false,
            flashinfer_fast_decode_plan: true,
            max_model_len: 8192,
            max_video_seconds: 15.0,
        }
    }
}

impl WorkerProcessArgs {
    /// Appends model-loading, memory, graph, and sampler options to a worker command.
    fn append_worker_args(&self, cmd: &mut Command) {
        if self.stub {
            cmd.arg("--no-model").arg("--allow-stub");
        }
        if self.load_format != "auto" {
            cmd.arg("--load-format").arg(&self.load_format);
        }
        if let Some(value) = &self.download_dir {
            cmd.arg("--download-dir").arg(value);
        }
        if let Some(value) = self.load_threads {
            cmd.arg("--load-threads").arg(value.to_string());
        }
        if let Some(value) = &self.checksum_manifest {
            cmd.arg("--checksum-manifest").arg(value);
        }
        cmd.arg("--model-dtype").arg(self.model_dtype.as_str());
        cmd.arg("--quantization-config")
            .arg(self.quantization_config.to_string());
        if let Some(value) = &self.kv_cache_dtype {
            cmd.arg("--kv-cache-dtype").arg(value.as_str());
        }
        cmd.arg("--kv-memory-fraction")
            .arg(self.kv_memory_fraction.to_string());
        if let Some(value) = &self.mesh {
            cmd.arg("--mesh").arg(value);
        }
        if let Some(value) = &self.tp_backend {
            cmd.arg("--tp-backend").arg(value);
        }
        for lane in &self.lanes {
            cmd.arg("--lane").arg(lane.worker_arg());
        }
        if !self.cuda_graph {
            cmd.arg("--no-cuda-graph");
        }
        if let Some(value) = &self.decode_graph_batch_sizes {
            cmd.arg("--decode-graph-batch-sizes").arg(value);
        }
        if self.prefill_cuda_graph {
            cmd.arg("--prefill-cuda-graph");
        }
        if let Some(value) = &self.prefill_graph_token_sizes {
            cmd.arg("--prefill-graph-token-sizes").arg(value);
        }
        if let Some(value) = &self.flow_graph_batch_sizes {
            cmd.arg("--flow-graph-batch-sizes").arg(value);
        }
        if let Some(value) = &self.flow_graph_shapes {
            cmd.arg("--flow-graph-shapes").arg(value);
        }
        cmd.arg("--flashinfer-workspace-size")
            .arg(self.flashinfer_workspace_size.to_string());
        if let Some(value) = &self.flashinfer_use_tensor_core {
            cmd.arg("--flashinfer-use-tensor-core").arg(value);
        }
        cmd.arg("--flashinfer-decode-backend")
            .arg(self.flashinfer_decode_backend.as_str());
        cmd.arg("--flashinfer-prefill-backend")
            .arg(self.flashinfer_prefill_backend.as_str());
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
        cmd.arg("--max-model-len")
            .arg(self.max_model_len.to_string());
        cmd.arg("--max-video-seconds")
            .arg(self.max_video_seconds.to_string());
    }
}

/// Single-process worker executor over iceoryx2 IPC.
pub struct UniprocExecutor {
    client: ClientEndpoint,
    info: WorkerInfo,
    executor_info: ExecutorInfo,
    child: Child,
    depth: usize,
    rank: u32,
    tp_size: u32,
    pending: HashMap<u64, PendingRecord>,
    ready: VecDeque<RunResult>,
    next_call_id: u64,
    next_collective_seq: u64,
    logical_results: LogicalResultTracker,
    command_wake_pending: bool,
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

/// Releases the consumed request.
fn release_consumed_request(record: PendingRecord) -> OutstandingKind {
    let PendingRecord { kind, pending } = record;
    drop(pending);
    kind
}

enum OutstandingKind {
    Batch {
        run_id: u64,
        remaining_operations: HashSet<(RequestKey, OpId)>,
    },
}

impl UniprocExecutor {
    /// Spawns one worker process and completes its capability handshake.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid world size, process or IPC setup failure,
    /// or an invalid worker capability response.
    pub fn spawn(args: WorkerProcessArgs) -> anyhow::Result<Self> {
        anyhow::ensure!(
            args.world_size == 1,
            "uniproc worker world size must be one"
        );
        let mut me = Self::spawn_rank_deferred(&args, &args.device, 0, 1, None)?;
        me.finish_startup()?;
        Ok(me)
    }

    /// Spawns one rank and connects its IPC client without waiting for model readiness.
    pub(crate) fn spawn_rank_deferred(
        args: &WorkerProcessArgs,
        device: &str,
        tp_rank: u32,
        tp_size: u32,
        tp_init_method: Option<&str>,
    ) -> anyhow::Result<Self> {
        let depth = args.pipeline_depth.max(1);
        let max_payload = args.req_slot_cap.max(args.resp_slot_cap).max(1);
        let service = service_name(&format!("{}_{}_{}", std::process::id(), tp_rank, nano_id()));
        let mut cmd = Command::new(&args.python);
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
            .arg(&args.model)
            .arg("--device")
            .arg(device)
            .arg("--attention-backend")
            .arg(args.attention_backend.as_name())
            .arg("--block-size")
            .arg(args.block_size.to_string())
            .arg("--max-batch-operations")
            .arg(args.max_batch_operations.to_string())
            .arg("--max-batch-tokens")
            .arg(args.max_batch_tokens.to_string())
            .arg("--tp-rank")
            .arg(tp_rank.to_string())
            .arg("--tp-size")
            .arg(tp_size.to_string());
        if args.supported_ops != uniserve_worker_ipc::OpKind::ALL {
            cmd.arg("--supported-ops").arg(
                args.supported_ops
                    .iter()
                    .flat_map(|operation| operation.run_kinds())
                    .map(|operation| operation.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        // In-process transfer is the worker default and needs no command-line override.
        if args.transfer_backend != crate::executor::TransferBackend::Inproc {
            cmd.arg("--transfer-backend")
                .arg(args.transfer_backend.as_str());
        }
        cmd.env("RANK", tp_rank.to_string())
            .env("WORLD_SIZE", tp_size.to_string())
            .env("LOCAL_RANK", tp_rank.to_string())
            .env("LOCAL_WORLD_SIZE", tp_size.to_string());
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
        if let Some(c) = args.kv_token_capacity {
            cmd.arg("--kv-token-capacity").arg(c.to_string());
        }
        args.append_worker_args(&mut cmd);
        if let Ok(cwd) = std::env::current_dir() {
            let pp = std::env::var("PYTHONPATH").unwrap_or_default();
            cmd.env("PYTHONPATH", format!("{}:{}", cwd.display(), pp));
        }
        let child = cmd.spawn().context("spawning python worker")?;

        let client = ClientEndpoint::connect(&service, max_payload, depth)
            .context("connecting to worker IPC service")?;
        let death_watcher = DeathWatcher::spawn(child.id(), client.death_wake());
        Ok(Self {
            client,
            info: WorkerInfo::default(),
            executor_info: ExecutorInfo::single(
                PoolId(format!("rank-{tp_rank}")),
                WorkerInfo::default(),
            ),
            child,
            depth,
            rank: tp_rank,
            tp_size,
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_call_id: 1,
            next_collective_seq: 1,
            logical_results: LogicalResultTracker::default(),
            command_wake_pending: false,
            shutdown_sent: false,
            death_watcher,
        })
    }

    /// Completes the worker information handshake and validates rank capabilities.
    pub(crate) fn finish_startup(&mut self) -> anyhow::Result<()> {
        tracing::info!(
            tp_rank = self.rank,
            tp_size = self.tp_size,
            "waiting for worker to load model + report info..."
        );
        let call_id = self.alloc_call_id();
        let mut req = WorkerRequest::info();
        req.set_call_id(Some(call_id));
        let pending =
            self.send_request_with_timeout(&req, "info handshake", WORKER_CONNECT_TIMEOUT)?;
        let resp = self.wait_pending_response(&pending, "info handshake")?;
        let wr = resp.decode_response()?;
        let info = match wr {
            WorkerResponse::Info { info, .. } => info,
            WorkerResponse::Error { error, .. } => {
                bail!("worker error during info: {}", error.message)
            }
            other => bail!("unexpected worker info response kind: {:?}", other.kind()),
        };
        info.validate()
            .context("worker reported invalid worker info during startup")?;
        let host_depth = self.depth as u32;
        anyhow::ensure!(
            info.queue_depth == host_depth,
            "worker pipeline_depth {} does not match launched depth {}",
            info.queue_depth,
            host_depth
        );
        anyhow::ensure!(
            info.rank.tp_rank == self.rank && info.rank.tp_size == self.tp_size,
            "worker TP rank/size ({}/{}) does not match launched topology ({}/{})",
            info.rank.tp_rank,
            info.rank.tp_size,
            self.rank,
            self.tp_size
        );
        self.executor_info =
            ExecutorInfo::single(PoolId(format!("rank-{}", self.rank)), info.clone());
        self.info = info;
        tracing::info!(?self.info, "worker ready");
        Ok(())
    }

    /// Allocates the next worker call identifier.
    fn alloc_call_id(&mut self) -> u64 {
        let id = self.next_call_id;
        self.next_call_id += 1;
        id
    }

    /// Checks the worker.
    fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if let Some(status) = self.child.try_wait()? {
            bail!("worker process exited during {context}: {status}");
        }
        Ok(())
    }

    /// Sends the request checked.
    fn send_request_checked(
        &mut self,
        req: &WorkerRequest,
        context: &str,
    ) -> anyhow::Result<Pending> {
        self.send_request_with_timeout(req, context, WORKER_SEND_TIMEOUT)
    }

    /// Waits for an IPC server connection and submits a request before the deadline.
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

    /// Waits for one startup response while checking child liveness and reporting progress.
    fn wait_pending_response(&mut self, pending: &Pending, context: &str) -> anyhow::Result<Frame> {
        let started = Instant::now();
        let mut last_log = started;
        let mut last_worker_check = started;
        loop {
            if let Some(frame) = self.client.try_recv_response(pending)? {
                return Ok(frame);
            }
            if last_worker_check.elapsed() >= WORKER_CHECK_INTERVAL {
                self.check_worker(context)?;
                last_worker_check = Instant::now();
            }
            if last_log.elapsed() >= STARTUP_LOG_INTERVAL {
                tracing::info!(
                    elapsed_secs = started.elapsed().as_secs(),
                    "worker still busy during {context}"
                );
                last_log = Instant::now();
            }
            let until_check = WORKER_CHECK_INTERVAL.saturating_sub(last_worker_check.elapsed());
            let until_log = STARTUP_LOG_INTERVAL.saturating_sub(last_log.elapsed());
            self.client.wait_wake(until_check.min(until_log))?;
        }
    }

    /// Drains ready IPC responses and routes them to physical-run completion state.
    fn drain_ready(&mut self) -> anyhow::Result<(usize, uniserve_worker_ipc::WakeEvents)> {
        let wakes = self.client.drain_wakes()?;
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
            // Consuming a partial execute response ends this physical IPC
            // request. Release its iceoryx active-request slot before routing
            // can submit the continuation poll for the remaining lanes.
            let kind = release_consumed_request(record);
            self.route(call_id, kind, frame)?;
            drained += 1;
        }
        Ok((drained, wakes))
    }

    /// Validates response correlation and routes one decoded worker response.
    fn route(&mut self, call_id: u64, kind: OutstandingKind, frame: Frame) -> anyhow::Result<()> {
        if frame.header.call_id != 0 && frame.header.call_id != call_id {
            bail!(
                "worker response call id mismatch: expected {call_id}, got {}",
                frame.header.call_id
            );
        }
        let wr = frame.decode_response()?;
        if let Some(echoed) = wr.call_id()
            && echoed != call_id
        {
            bail!("worker response echoed call id {echoed}, expected {call_id}");
        }
        match kind {
            OutstandingKind::Batch {
                run_id,
                remaining_operations,
            } => self.route_batch(run_id, remaining_operations, wr),
        }
    }

    /// Accumulates a partial run response or schedules polling for remaining operations.
    fn route_batch(
        &mut self,
        run_id: u64,
        mut remaining_operations: HashSet<(RequestKey, OpId)>,
        wr: WorkerResponse,
    ) -> anyhow::Result<()> {
        match wr {
            WorkerResponse::Result { result: r, .. } => {
                if r.run_id != run_id {
                    bail!(
                        "worker result step id mismatch: expected {run_id}, got {}",
                        r.run_id
                    );
                }
                for output in &r.completions {
                    anyhow::ensure!(
                        remaining_operations.remove(&(output.request_key, output.op_id)),
                        "worker returned a duplicate or unknown operation for step {run_id}"
                    );
                }
                anyhow::ensure!(
                    r.done == remaining_operations.is_empty(),
                    "worker run completion flag disagrees with remaining physical work"
                );
                anyhow::ensure!(
                    !r.completions.is_empty() || remaining_operations.is_empty(),
                    "worker returned an empty partial completion for step {run_id}"
                );
                enqueue_ready(&mut self.ready, r);
                if !remaining_operations.is_empty() {
                    self.submit_completion_poll(run_id, remaining_operations)?;
                }
                Ok(())
            }
            WorkerResponse::Error { error, .. } => Err(WorkerExecError {
                run_id: Some(run_id),
                fatal: error.fatal,
                retryable: error.retryable,
                code: error.code,
                message: error.message,
                phase: error.phase,
                route: error.route,
                operations: error.operations,
            }
            .into()),
            other => bail!("unexpected execute response kind: {:?}", other.kind()),
        }
    }

    /// Submits a continuation poll for the unresolved operations of one run.
    fn submit_completion_poll(
        &mut self,
        run_id: u64,
        remaining_operations: HashSet<(RequestKey, OpId)>,
    ) -> anyhow::Result<()> {
        let call_id = self.alloc_call_id();
        let mut request = WorkerRequest::poll(run_id);
        request.set_call_id(Some(call_id));
        let pending = self.send_request_checked(&request, "completion poll")?;
        self.pending.insert(
            call_id,
            PendingRecord {
                kind: OutstandingKind::Batch {
                    run_id,
                    remaining_operations,
                },
                pending,
            },
        );
        Ok(())
    }

    /// Returns a waker for interrupting the worker IPC wait after command enqueue.
    pub fn command_waker(&self) -> CommandWaker {
        let sender = self.client.command_wake();
        CommandWaker::new(move || sender.wake())
    }

    /// Returns the file descriptor that signals worker progress.
    pub(crate) fn progress_fd(&self) -> i32 {
        self.client.wake_file_descriptor()
    }
}

impl PhysicalExecutor for UniprocExecutor {
    /// Returns metadata for the physical worker.
    fn physical_info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Submits one physical run and records its outstanding operation identities.
    fn submit_run(&mut self, batch: PhysicalRun) -> Result<(), PhysicalSubmitError> {
        self.drain_ready().map_err(PhysicalSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(PhysicalSubmitError::WouldBlock(batch));
        }
        let run_id = batch.run_id;
        let remaining_operations = batch
            .operations
            .iter()
            .map(|operation| (operation.request_key, operation.op_id))
            .collect::<HashSet<_>>();
        let call_id = self.alloc_call_id();
        let mut req = WorkerRequest::submit(batch);
        req.set_call_id(Some(call_id));
        let pending = self
            .send_request_checked(&req, "batch submit")
            .map_err(PhysicalSubmitError::Failed)?;
        self.pending.insert(
            call_id,
            PendingRecord {
                kind: OutstandingKind::Batch {
                    run_id,
                    remaining_operations,
                },
                pending,
            },
        );
        Ok(())
    }

    /// Drives IPC progress until a result, command wake, worker death, or timeout.
    fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
        if self.command_wake_pending {
            return Ok(None);
        }
        let (_, mut wakes) = self.drain_ready()?;
        self.command_wake_pending |= wakes.command;
        self.check_worker("executor poll")?;
        if self.command_wake_pending || wakes.death {
            return Ok(None);
        }
        if let Some(result) = self.ready.pop_front() {
            return Ok(Some(result));
        }
        if timeout.is_zero() {
            return Ok(None);
        }
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            wakes = self
                .client
                .wait_wake((deadline - now).min(WORKER_CHECK_INTERVAL))?;
            let (_, queued_wakes) = self.drain_ready()?;
            wakes.command |= queued_wakes.command;
            wakes.death |= queued_wakes.death;
            self.command_wake_pending |= wakes.command;
            self.check_worker("executor poll")?;
            if self.command_wake_pending || wakes.death {
                return Ok(None);
            }
            if let Some(result) = self.ready.pop_front() {
                return Ok(Some(result));
            }
        }
    }

    /// Consumes the pending command-wake notification.
    fn take_command_wake(&mut self) -> bool {
        std::mem::take(&mut self.command_wake_pending)
    }

    /// Drains outstanding calls, requests graceful shutdown, and bounds forced termination.
    fn close_physical(&mut self) -> anyhow::Result<()> {
        if self.shutdown_sent {
            return Ok(());
        }
        self.shutdown_sent = true;
        // Stop the death watcher before we intentionally tear the worker down,
        // so its exit does not fire a spurious death wake during shutdown.
        let _ = self.death_watcher.take();
        let exited = matches!(self.child.try_wait(), Ok(Some(_)));
        if !exited {
            // Bound response draining so shutdown can advance to graceful termination
            // and, if necessary, forced process cleanup.
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
                    Ok((0, _)) => {
                        if self.check_worker("shutdown drain").is_err() {
                            break;
                        }
                        let remaining = drain_deadline.saturating_duration_since(Instant::now());
                        let _ = self.client.wait_wake(remaining.min(WORKER_CHECK_INTERVAL));
                    }
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            if matches!(self.child.try_wait(), Ok(None)) {
                let call_id = self.alloc_call_id();
                let mut req = WorkerRequest::close();
                req.set_call_id(Some(call_id));
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
        Ok(())
    }
}

impl Executor for UniprocExecutor {
    /// Returns the worker metadata.
    fn info(&self) -> &ExecutorInfo {
        &self.executor_info
    }

    /// Lowers and submits a logical batch while preserving result-tracker ownership.
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError> {
        if self.pending.len() >= self.depth {
            return Err(ExecutorSubmitError::WouldBlock(batch));
        }
        let run = lower_batch(&batch, &mut self.next_collective_seq)
            .map_err(ExecutorSubmitError::Failed)?;
        self.logical_results
            .register(&batch)
            .map_err(ExecutorSubmitError::Failed)?;
        match self.submit_run(run) {
            Ok(()) => Ok(()),
            Err(PhysicalSubmitError::WouldBlock(_)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::WouldBlock(batch))
            }
            Err(PhysicalSubmitError::Failed(error)) => {
                self.logical_results.unregister(batch.id);
                Err(ExecutorSubmitError::Failed(error))
            }
        }
    }

    /// Polls for the next completed worker operation.
    fn poll(&mut self, timeout: Duration) -> anyhow::Result<Option<BatchResult>> {
        if self.take_command_wake() {
            return Ok(None);
        }
        let report = self.poll_run(timeout)?;
        if report.is_none() {
            self.take_command_wake();
        }
        report
            .map(|report| self.logical_results.apply(report))
            .transpose()
    }

    /// Closes the component and releases its resources.
    fn close(&mut self) -> anyhow::Result<()> {
        self.close_physical()
    }
}

impl Drop for UniprocExecutor {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        let _ = self.close_physical();
    }
}

/// Returns a process-local identifier suitable for worker IPC service names.
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
