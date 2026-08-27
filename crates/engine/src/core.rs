//! The reified `EngineCore`: scheduler + executor + worker lifecycle, behind a
//! transport-free surface (queues + an [`EngineHandle`]). The HTTP process
//! hosts one in-process core.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::executor::{Executor, TransferBackend, TransferSpec, WorkerKind, WorkersSpec};
use crate::handle::{EngineHandle, EventRx, SubmitError};
use crate::scheduler::{
    ControlTokens, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedStats,
    Scheduler, SchedulingPolicy,
};
use crate::worker::{
    MultiprocExecutor, StagedExecutor, UniprocExecutor, WorkerLaunchConfig, WorkerSpawnSpec,
};
use anyhow::Context as _;
use uniserve_core::{GenerationRequest, GenerationRuntimeCapabilities, ModelDtype, RequestId};
use uniserve_worker_ipc::{AttentionBackend, WorkerCapabilities};

/// Place a staged pool on GPU `gpu`. A plain `cuda`/`gpu`
/// device becomes `cuda:{gpu}` (distinct GPU per pool); an explicit device
/// (`cuda:1`, `cpu`) is left as the operator set it.
fn assign_pool_device(device: &str, gpu: usize) -> String {
    match device.trim() {
        "cuda" | "gpu" => format!("cuda:{gpu}"),
        other => other.to_string(),
    }
}

/// Configuration for one in-process engine core.
#[derive(Debug, Clone)]
pub struct EngineCoreConfig {
    /// Model directory / identifier passed to the worker and used for metrics.
    pub model: String,
    /// Compute device for the worker (e.g. `cuda`, `cpu`).
    pub device: String,
    /// KV block size in tokens (the page size).
    pub block_size: u32,
    /// How many op-batches the scheduler keeps in flight against the worker.
    pub pipeline_depth: usize,
    /// Maximum number of ops assembled into a single forward batch.
    pub max_batch: usize,
    /// Per-step scheduling token budget (vLLM's `max_num_batched_tokens`).
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests (vLLM's `max_num_seqs`).
    pub max_num_seqs: usize,
    /// Per-request ceiling for one prefill chunk (SGLang's chunked prefill size).
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as one mixed extend+decode forward. `0` disables mixing.
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy for admitting/scheduling requests.
    pub scheduler_policy: SchedulingPolicy,
    /// Maximum model context length reported to the frontend.
    pub max_model_len: u32,
    /// Optional explicit KV token capacity override for the worker.
    pub kv_token_capacity: Option<u64>,
    /// Attention backend preference forwarded to the Python worker.
    pub attention_backend: AttentionBackend,
    /// Python interpreter used to launch the worker.
    pub worker_python: std::path::PathBuf,
    /// Parsed worker topology.
    pub workers: WorkersSpec,
    /// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`. Participating worker
    /// pools receive the selected transport; unconfigured edges use local
    /// worker-resident products.
    pub transfer: TransferSpec,
    /// Explicit Python worker launch/runtime configuration.
    pub worker_launch: WorkerLaunchConfig,
    /// Request-ring slot capacity in bytes.
    pub req_slot_cap: usize,
    /// Response-ring slot capacity in bytes.
    pub resp_slot_cap: usize,
    /// Control-token ids resolved from the tokenizer for EOS and feedback continuation.
    pub bos: u32,
    pub eos: Vec<u32>,
    pub end_of_image: u32,
}

impl EngineCoreConfig {
    /// A minimal config for the GPU-free sim backend (used by tests).
    ///
    /// Pair this with [`EngineCore::with_executor`]: [`EngineCore::new`] cannot
    /// build a `Sim` backend because it has no spawnable worker process.
    pub fn sim(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: "cpu".into(),
            block_size: 64,
            pipeline_depth: 2,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: 8192,
            kv_token_capacity: None,
            attention_backend: AttentionBackend::Auto,
            worker_python: "python3".into(),
            workers: WorkersSpec::single_full(1),
            transfer: TransferSpec::default(),
            worker_launch: WorkerLaunchConfig::default(),
            req_slot_cap: 1 << 20,
            resp_slot_cap: 8 << 20,
            // `SimEngine` fabricates this fake EOS id after `text_len` tokens; the
            // scheduler must recognize it to finish a sim request.
            bos: 0,
            eos: vec![151645],
            end_of_image: 0,
        }
    }

    fn control_tokens(&self) -> ControlTokens {
        ControlTokens {
            bos: self.bos,
            eos: self.eos.clone(),
            end_of_image: self.end_of_image,
        }
    }
}

/// The engine core: scheduler thread, executor, and capabilities as one
/// transport-free object.
pub struct EngineCore {
    handle: EngineHandle,
    caps: WorkerCapabilities,
    stats: Arc<SchedStats>,
    model_name: String,
    model_dtype: ModelDtype,
    max_model_len: u32,
    /// Engine-dead latch: set when the scheduler loop exits fatally (worker
    /// death) or panics.
    dead: Arc<AtomicBool>,
    next_id: AtomicU64,
    sched_thread: Mutex<Option<JoinHandle<()>>>,
}

impl EngineCore {
    /// Build the scheduler, spawn the forward-only worker, and start the
    /// scheduler owner thread.
    ///
    /// Blocks until the worker has loaded the model and answered the
    /// capability handshake — for the real worker this can take minutes.
    ///
    pub fn new(config: EngineCoreConfig) -> anyhow::Result<Self> {
        let executor: Box<dyn Executor> = if config.workers.is_single_full() {
            Self::spawn_full_pool(&config, config.workers.pools[0].tp)?
        } else {
            Self::spawn_staged(&config, &config.workers)?
        };
        Self::assemble(config, executor)
    }

    /// Spawn the single Full pool. `tp == 1`
    /// is a `UniprocExecutor`; `tp > 1` a `MultiprocExecutor`. No `--worker-kind`
    /// is passed because the worker starts in Full mode by default.
    fn spawn_full_pool(config: &EngineCoreConfig, tp: usize) -> anyhow::Result<Box<dyn Executor>> {
        let kv_token_capacity = config.kv_token_capacity;
        let max_batch_operations =
            u32::try_from(config.max_batch.max(1).min(config.max_num_seqs.max(1)))
                .context("max_batch exceeds the worker capability representation")?;
        let max_batch_tokens = u32::try_from(config.max_num_batched_tokens)
            .context("max_num_batched_tokens exceeds the worker capability representation")?;
        let spec = WorkerSpawnSpec {
            python: config.worker_python.clone(),
            model: config.model.clone(),
            device: config.device.clone(),
            world_size: tp,
            pipeline_depth: config.pipeline_depth,
            req_slot_cap: config.req_slot_cap,
            resp_slot_cap: config.resp_slot_cap,
            kv_token_capacity,
            block_size: config.block_size,
            max_batch_operations,
            max_batch_tokens,
            attention_backend: config.attention_backend.clone(),
            worker_kind: None,
            transfer_backend: TransferBackend::Inproc,
            launch: config.worker_launch.clone(),
        };
        if tp > 1 {
            let workers = MultiprocExecutor::spawn(spec)
                .context("failed to spawn forward-only worker ranks")?;
            Ok(Box::new(workers))
        } else {
            let worker =
                UniprocExecutor::spawn(spec).context("failed to spawn forward-only worker")?;
            Ok(Box::new(worker))
        }
    }

    /// Compose a `StagedExecutor` over the heterogeneous pools of a staged topology.
    /// Each pool instance is a (tp-sized) `MultiprocExecutor` spawned with its
    /// `--worker-kind`; the router fans the scheduler's batch across them by
    /// exact operation type and merges the results.
    fn spawn_staged(
        config: &EngineCoreConfig,
        workers: &WorkersSpec,
    ) -> anyhow::Result<Box<dyn Executor>> {
        let kv_token_capacity = config.kv_token_capacity;
        let max_batch_operations =
            u32::try_from(config.max_batch.max(1).min(config.max_num_seqs.max(1)))
                .context("max_batch exceeds the worker capability representation")?;
        let max_batch_tokens = u32::try_from(config.max_num_batched_tokens)
            .context("max_num_batched_tokens exceeds the worker capability representation")?;
        // Per-edge transfer backends (validated up front so a typo fails at
        // startup). A pool's worker uses the backend of its incoming edge for
        // the worker-side data plane (read-driven fetch).
        // A pool uses the transport of any edge it participates in — as producer
        // (src) OR consumer (dst) — so both ends of an edge agree on the backend.
        let backend_for = |kind: WorkerKind| -> Option<TransferBackend> {
            config
                .transfer
                .edges
                .iter()
                .find(|((src, dst), _)| *src == kind || *dst == kind)
                .map(|(_, backend)| *backend)
        };
        // The Und/Gen stage split places each pool on its own GPU so the two
        // towers run in parallel and the conditioning KV crosses GPU↔GPU over
        // CUDA IPC. Other staged topologies keep the shared device.
        let is_tower = workers
            .pools
            .iter()
            .any(|p| matches!(p.kind, WorkerKind::Und | WorkerKind::Gen));
        let mut next_gpu = 0usize;
        let mut pools: Vec<(WorkerKind, Box<dyn Executor>)> =
            Vec::with_capacity(workers.total_pools());
        for pool in &workers.pools {
            let backend = backend_for(pool.kind).or_else(|| {
                if is_tower && matches!(pool.kind, WorkerKind::Und | WorkerKind::Gen) {
                    Some(TransferBackend::CudaIpc)
                } else {
                    None
                }
            });
            for instance in 0..pool.count {
                let pool_device = if is_tower {
                    assign_pool_device(&config.device, next_gpu)
                } else {
                    config.device.clone()
                };
                next_gpu += pool.tp.max(1);
                let pool_worker_launch = config.worker_launch.clone();
                let exec = MultiprocExecutor::spawn(WorkerSpawnSpec {
                    python: config.worker_python.clone(),
                    model: config.model.clone(),
                    device: pool_device,
                    world_size: pool.tp.max(1),
                    pipeline_depth: config.pipeline_depth,
                    req_slot_cap: config.req_slot_cap,
                    resp_slot_cap: config.resp_slot_cap,
                    kv_token_capacity,
                    block_size: config.block_size,
                    max_batch_operations,
                    max_batch_tokens,
                    attention_backend: config.attention_backend.clone(),
                    worker_kind: Some(pool.kind),
                    transfer_backend: backend.unwrap_or_default(),
                    launch: pool_worker_launch,
                })
                .with_context(|| {
                    format!(
                        "failed to spawn staged pool {}#{instance} (tp={})",
                        pool.kind.as_str(),
                        pool.tp
                    )
                })?;
                pools.push((pool.kind, Box::new(exec)));
            }
        }
        Ok(Box::new(StagedExecutor::try_new(pools)?))
    }

    /// Build the engine core from an executor supplied by a higher composition layer.
    pub fn with_executor(
        config: EngineCoreConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor)
    }

    fn assemble(config: EngineCoreConfig, executor: Box<dyn Executor>) -> anyhow::Result<Self> {
        let ctrl = config.control_tokens();
        // Capture the command waker before the executor moves into the
        // scheduler: when the executor is event-driven this fires its park's
        // command notifier; otherwise it is the no-op waker.
        let waker = executor.command_waker();
        let sched = Scheduler::with_config(
            executor,
            ctrl,
            crate::scheduler::SchedulerConfig {
                max_batch: config.max_batch,
                max_num_batched_tokens: config.max_num_batched_tokens.max(1),
                max_num_seqs: config.max_num_seqs.max(1),
                long_prefill_threshold: config.long_prefill_threshold.max(1),
                mixed_prefill_tokens: config.mixed_prefill_tokens,
                policy: config.scheduler_policy,
                ..Default::default()
            },
        );
        let caps = sched.caps().clone();
        let model_dtype = caps.model_dtype;
        let stats = sched.stats_handle();

        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let dead = Arc::new(AtomicBool::new(false));
        let sched_thread = {
            let dead = Arc::clone(&dead);
            std::thread::Builder::new()
                .name("scheduler".into())
                .spawn(move || {
                    let fatal = sched.run(cmd_rx);
                    if fatal {
                        dead.store(true, Ordering::SeqCst);
                    }
                })
                .context("failed to spawn scheduler thread")?
        };
        let handle = EngineHandle::with_waker(cmd_tx, waker);

        Ok(Self {
            handle,
            caps,
            stats,
            model_name: config.model,
            model_dtype,
            max_model_len: config.max_model_len,
            dead,
            next_id: AtomicU64::new(1),
            sched_thread: Mutex::new(Some(sched_thread)),
        })
    }

    /// Cloneable command-channel front door over the scheduler.
    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    /// Worker-reported capabilities (the post-load truth).
    pub fn caps(&self) -> &WorkerCapabilities {
        &self.caps
    }

    /// Serving-facing projection of post-load worker limits.
    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        self.caps.generation_runtime_capabilities()
    }

    pub fn supports_token_sampling(&self) -> bool {
        self.caps.supported_work.iter().any(|mode| {
            matches!(
                mode,
                uniserve_worker_ipc::ForwardMode::TokenDecode
                    | uniserve_worker_ipc::ForwardMode::TokenVerify
                    | uniserve_worker_ipc::ForwardMode::Draft
            )
        })
    }

    /// Live scheduler stats, shared with the scheduler thread.
    pub fn stats(&self) -> &Arc<SchedStats> {
        &self.stats
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn model_dtype(&self) -> ModelDtype {
        self.model_dtype
    }

    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    /// Allocate the next internal scheduler request id.
    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Whether the engine died (worker/executor failure). Sticky.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    /// Submit one translated request to the scheduler.
    pub fn submit(&self, request: GenerationRequest) -> Result<EventRx, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.handle.submit(request)
    }

    /// Shut down the scheduler (which tears down the executor/worker) and join
    /// its thread. Idempotent.
    pub fn shutdown(&self) {
        self.handle.shutdown();
        let thread = match self.sched_thread.lock() {
            Ok(mut guard) => guard.take(),
            Err(error) => {
                tracing::warn!("scheduler thread lock poisoned during shutdown");
                error.into_inner().take()
            }
        };
        if let Some(thread) = thread
            && thread.join().is_err()
        {
            tracing::warn!("scheduler thread panicked during shutdown");
        }
    }
}

impl Drop for EngineCore {
    fn drop(&mut self) {
        self.shutdown();
    }
}
