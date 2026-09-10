//! Single-worker executor over an iceoryx2 request-response service.
//!
//! The host exchanges FlatBuffers descriptors and bounded result values while
//! tensors, KV pages, and latent storage remain worker-resident.

use std::collections::{HashMap, HashSet, VecDeque};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use super::RunSubmitError;
use crate::executor::WorkerExecError;
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
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
            worker_id: "worker".into(),
            python: "python3".into(),
            model: String::new(),
            ranks: crate::WorkerConfig::model("cuda", 1, 2).ranks,
            entries: crate::WorkerConfig::model("cuda", 1, 2).entries,
            pipeline_depth: 2,
            req_slot_cap: 1 << 20,
            resp_slot_cap: 8 << 20,
            kv_token_capacity: None,
            block_size: 64,
            max_batch_operations: 128,
            max_batch_tokens: 16_384,
            attention_backend: uniserve_worker_ipc::AttentionBackend::Auto,
            supported_ops: uniserve_worker_ipc::OpCode::ALL.to_vec(),
            transfer: Default::default(),
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
            distributed_backend: None,
            lanes: Vec::new(),
            graph_policy: "auto".into(),
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
        if let Some(value) = &self.distributed_backend {
            cmd.arg("--distributed-backend").arg(value);
        }
        for lane in &self.lanes {
            cmd.arg("--lane").arg(lane.worker_arg());
        }
        cmd.args(["--graph-policy", &self.graph_policy]);
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
pub(super) struct RankProcess {
    client: ClientEndpoint,
    info: WorkerInfo,
    startup_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    child: Child,
    /// Keep the shared store directory until every process in the group exits.
    _rendezvous: Option<std::sync::Arc<tempfile::TempDir>>,
    depth: usize,
    rank: u32,
    world_size: u32,
    expected_components: std::collections::BTreeMap<String, uniserve_core::EntryConfig>,
    pending: HashMap<u64, PendingRecord>,
    ready: VecDeque<RunResult>,
    next_call_id: u64,
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

impl RankProcess {
    /// Spawns one rank and connects its IPC client without waiting for model readiness.
    pub(crate) fn spawn_rank_deferred(
        args: &WorkerProcessArgs,
        device: &str,
        rank: u32,
        world_size: u32,
        rendezvous: Option<std::sync::Arc<tempfile::TempDir>>,
        components: &std::collections::BTreeMap<String, crate::executor::EntryConfig>,
        startup_abort: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<Self> {
        let depth = args.pipeline_depth.max(1);
        let max_payload = args.req_slot_cap.max(args.resp_slot_cap).max(1);
        let service = service_name(&format!("{}_{}_{}", std::process::id(), rank, nano_id()));
        let mut cmd = Command::new(&args.python);
        cmd.arg("-m")
            .arg("uniserve_worker.main")
            .arg("--worker-id")
            .arg(&args.worker_id)
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
            .arg("--rank")
            .arg(rank.to_string())
            .arg("--world-size")
            .arg(world_size.to_string());
        cmd.arg("--local-rank")
            .arg(rank.to_string())
            .arg("--entries")
            .arg(serde_json::to_string(&components)?);
        if args.supported_ops != uniserve_worker_ipc::OpCode::ALL {
            cmd.arg("--supported-ops").arg(
                args.supported_ops
                    .iter()
                    .map(|operation| operation.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        // Resolve mechanism ownership from the physical rank's incident edges.
        let (backends, publications) = args.transfer.rank_backends(&args.worker_id, rank);
        let names = |backends: &std::collections::BTreeSet<crate::executor::TransferBackend>| {
            backends
                .iter()
                .map(|backend| backend.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        cmd.arg("--transfer-backends").arg(names(&backends));
        cmd.arg("--publish-backends").arg(names(&publications));
        cmd.env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("LOCAL_RANK", rank.to_string())
            .env("LOCAL_WORLD_SIZE", world_size.to_string());
        // CUDA IPC exports cudaMalloc allocations. Expandable VMM segments
        // cannot supply its memory handles; other ranks retain expandable
        // allocation to accommodate varying serving shapes.
        if std::env::var_os("PYTORCH_ALLOC_CONF").is_none()
            && std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none()
        {
            let allocation = if publications.contains(&crate::executor::TransferBackend::CudaIpc) {
                "expandable_segments:False"
            } else {
                "expandable_segments:True"
            };
            cmd.env("PYTORCH_CUDA_ALLOC_CONF", allocation);
        }
        if let Some(directory) = &rendezvous {
            let store = directory.path().join("store");
            cmd.arg("--distributed-init-method")
                .arg(format!("file://{}", store.display()));
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
        let death_watcher =
            DeathWatcher::spawn(child.id(), client.death_wake(), startup_abort.clone());
        Ok(Self {
            client,
            info: WorkerInfo::default(),
            startup_cancel: Some(startup_abort),
            child,
            _rendezvous: rendezvous,
            depth,
            rank,
            world_size,
            expected_components: components.clone(),
            pending: HashMap::new(),
            ready: VecDeque::new(),
            next_call_id: 1,
            command_wake_pending: false,
            shutdown_sent: false,
            death_watcher,
        })
    }

    /// Publish capabilities only after model resources and warmup are ready.
    pub(crate) fn finish_startup(&mut self) -> anyhow::Result<()> {
        let call_id = self.alloc_call_id();
        let mut request = WorkerRequest::info();
        request.set_call_id(Some(call_id));
        let pending =
            self.send_request_with_timeout(&request, "Worker startup", WORKER_CONNECT_TIMEOUT)?;
        let response = self
            .wait_pending_response(&pending, "Worker startup")?
            .decode_response()?;
        let info = match response {
            WorkerResponse::Info { info, .. } => info,
            WorkerResponse::Error { error, .. } => {
                bail!("Worker startup failed: {}", error.message)
            }
            other => bail!("unexpected startup response: {:?}", other.kind()),
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
            info.endpoint.rank == self.rank && info.world_size == self.world_size,
            "worker process rank/world_size ({}/{}) does not match launched topology ({}/{})",
            info.endpoint.rank,
            info.world_size,
            self.rank,
            self.world_size
        );
        anyhow::ensure!(
            info.components
                .iter()
                .map(|entry| (entry.name.clone(), entry.config.clone()))
                .collect::<std::collections::BTreeMap<_, _>>()
                == self.expected_components,
            "worker resolved component configuration disagrees with configuration"
        );
        anyhow::ensure!(
            info.configuration_id.len() == 64,
            "worker omitted resolved configuration identity"
        );
        self.check_worker("Worker startup")?;
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

    /// Cancel unfinished rank startup when a required peer exits.
    pub(crate) fn set_startup_cancel(
        &mut self,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) {
        self.startup_cancel = cancel;
    }

    /// Terminates a failed or cancelled rank and waits for process-owned resources to retire.
    pub(crate) fn terminate(&mut self) {
        self.shutdown_sent = true;
        self.death_watcher.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.pending.clear();
        self.ready.clear();
    }

    /// Checks child liveness and cancellation of an unfinished startup.
    pub(super) fn check_worker(&mut self, context: &str) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            bail!("worker startup cancelled during {context}");
        }
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
                for product in &r.products {
                    if let uniserve_worker_ipc::InlineValue::Transfer(handle) = &product.value {
                        anyhow::ensure!(
                            handle
                                .locators()
                                .all(|locator| locator.source == self.info.endpoint),
                            "worker published a product from an unbound rank incarnation"
                        );
                    }
                }
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
                    !r.done || remaining_operations.is_empty(),
                    "worker run completion flag disagrees with remaining physical work"
                );
                anyhow::ensure!(
                    !r.completions.is_empty() || r.done,
                    "worker returned an empty partial completion for step {run_id}"
                );
                let done = r.done;
                enqueue_ready(&mut self.ready, r);
                if !done {
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

    /// Returns the file descriptor that signals worker progress.
    pub(crate) fn progress_fd(&self) -> i32 {
        self.client.wake_file_descriptor()
    }
}

impl RankProcess {
    /// Returns metadata for the physical worker.
    pub(super) fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Submits one physical run and records its outstanding operation identities.
    pub(super) fn submit_run(&mut self, batch: PhysicalRun) -> Result<(), RunSubmitError> {
        self.drain_ready().map_err(RunSubmitError::Failed)?;
        if self.pending.len() >= self.depth {
            return Err(RunSubmitError::WouldBlock(batch));
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
            .map_err(RunSubmitError::Failed)?;
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
    pub(super) fn poll_run(&mut self, timeout: Duration) -> anyhow::Result<Option<RunResult>> {
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
    pub(crate) fn take_command_wake(&mut self) -> bool {
        std::mem::take(&mut self.command_wake_pending)
    }

    /// Drains outstanding calls, requests graceful shutdown, and bounds forced termination.
    pub(super) fn close(&mut self) -> anyhow::Result<()> {
        if self
            .startup_cancel
            .as_ref()
            .is_some_and(|cancel| cancel.load(std::sync::atomic::Ordering::Acquire))
        {
            self.terminate();
            return Ok(());
        }
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

impl Drop for RankProcess {
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        let _ = self.close();
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
    // The full service name also includes pid + rank, so any residual aliasing in the
    // shifted timestamp bits cannot produce a real cross-worker collision.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    (nanos << 16) | (seq & 0xffff)
}
