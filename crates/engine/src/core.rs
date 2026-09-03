//! In-process engine composition: worker lifecycle, scheduler ownership, and the
//! transport-free [`EngineHandle`] surface used by the server.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::executor::{Executor, PoolConfig, TransferBackend, TransportMap, WorkerTopology};
use crate::handle::{EngineHandle, EventRx, SubmitError};
use crate::runtime::{ControlTokens, EngineLoop, RuntimeProfile};
use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedStats, SchedulerConfig,
    SchedulingPolicy,
};
use crate::worker::{MultiprocExecutor, StagedExecutor, UniprocExecutor, WorkerProcessArgs};
use anyhow::Context as _;
use uniserve_core::{
    CommandWaker, GenerationLimits, ModelDtype, Request, RequestId, RuntimeFamily,
};
use uniserve_worker_ipc::WorkerInfo;

/// Configuration for one in-process engine core.
#[derive(Debug, Clone)]
pub struct EngineCoreConfig {
    /// Request runtime selected for this deployment.
    pub runtime_family: RuntimeFamily,
    /// Model-family semantics resolved by the serving profile.
    pub runtime_profile: RuntimeProfile,
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
    /// Parsed worker topology.
    pub workers: WorkerTopology,
    /// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`. Participating worker
    /// pools receive the selected transport; unconfigured edges use local
    /// worker-resident products.
    pub transfer: TransportMap,
    /// Complete process arguments; staged pools override placement and role.
    pub worker_process: WorkerProcessArgs,
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
        let worker_process = WorkerProcessArgs {
            model: model.into(),
            device: "cpu".into(),
            block_size: 64,
            pipeline_depth: 2,
            max_batch_operations: DEFAULT_MAX_BATCH as u32,
            max_batch_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS as u32,
            ..WorkerProcessArgs::default()
        };
        Self {
            runtime_family: RuntimeFamily::Umm,
            runtime_profile: RuntimeProfile::umm(
                ModelDtype::BFloat16,
                crate::runtime::sim_umm_generation_limits(),
            ),
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: 8192,
            workers: WorkerTopology::single_full(1),
            transfer: TransportMap::default(),
            worker_process,
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

/// The engine core: scheduler thread, executor, and worker info as one
/// transport-free object.
pub struct EngineCore {
    handle: EngineHandle,
    info: WorkerInfo,
    stats: Arc<SchedStats>,
    model_name: String,
    model_dtype: ModelDtype,
    generation_limits: GenerationLimits,
    max_model_len: u32,
    runtime_family: RuntimeFamily,
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
    /// worker-info handshake — for the real worker this can take minutes.
    ///
    pub fn new(config: EngineCoreConfig) -> anyhow::Result<Self> {
        let workers = config.workers.clone().with_process_defaults(
            &config.worker_process.device,
            config.worker_process.pipeline_depth,
        );
        let (executor, command_waker) = if workers.is_single_full() {
            Self::spawn_full_pool(&config, &workers.pools[0])?
        } else {
            Self::spawn_staged(&config, &workers)?
        };
        Self::assemble(config, executor, command_waker)
    }

    /// Spawn the single Full pool. `tp == 1` uses a `UniprocExecutor`; `tp > 1`
    /// uses a `MultiprocExecutor`.
    fn spawn_full_pool(
        config: &EngineCoreConfig,
        pool: &PoolConfig,
    ) -> anyhow::Result<(Box<dyn Executor>, CommandWaker)> {
        let args = WorkerProcessArgs {
            device: pool.device.clone(),
            world_size: pool.tensor_parallel_size,
            pipeline_depth: pool.queue_depth,
            supported_ops: pool.supported_ops.clone(),
            transfer_backend: TransferBackend::Inproc,
            ..config.worker_process.clone()
        };
        if pool.tensor_parallel_size > 1 {
            let workers = MultiprocExecutor::spawn(args)
                .context("failed to spawn forward-only worker ranks")?;
            let waker = workers.command_waker();
            Ok((Box::new(workers), waker))
        } else {
            let worker =
                UniprocExecutor::spawn(args).context("failed to spawn forward-only worker")?;
            let waker = worker.command_waker();
            Ok((Box::new(worker), waker))
        }
    }

    /// Compose a staged executor over explicitly configured physical pools.
    fn spawn_staged(
        config: &EngineCoreConfig,
        workers: &WorkerTopology,
    ) -> anyhow::Result<(Box<dyn Executor>, CommandWaker)> {
        for edge in &config.transfer.edges {
            anyhow::ensure!(
                workers.pools.iter().any(|pool| pool.id == edge.source_pool),
                "transfer edge source pool {} is not configured",
                edge.source_pool
            );
            anyhow::ensure!(
                workers
                    .pools
                    .iter()
                    .any(|pool| pool.id == edge.destination_pool),
                "transfer edge destination pool {} is not configured",
                edge.destination_pool
            );
        }
        let mut pools: Vec<(PoolConfig, Box<dyn crate::executor::PhysicalExecutor>)> =
            Vec::with_capacity(workers.total_pools());
        let mut command_wakers = Vec::with_capacity(workers.total_pools());
        let mut progress_fds = Vec::new();
        for pool in &workers.pools {
            let incident = config
                .transfer
                .edges
                .iter()
                .filter(|edge| edge.source_pool == pool.id || edge.destination_pool == pool.id)
                .map(|edge| edge.transport)
                .collect::<std::collections::BTreeSet<_>>();
            anyhow::ensure!(
                incident.len() <= 1,
                "pool {} has transfer edges with incompatible transports",
                pool.id
            );
            let exec = MultiprocExecutor::spawn(WorkerProcessArgs {
                device: pool.device.clone(),
                world_size: pool.tensor_parallel_size.max(1),
                pipeline_depth: pool.queue_depth.max(1),
                supported_ops: pool.supported_ops.clone(),
                transfer_backend: incident.first().copied().unwrap_or_default(),
                ..config.worker_process.clone()
            })
            .with_context(|| {
                format!(
                    "failed to spawn staged pool {} (tp={})",
                    pool.id, pool.tensor_parallel_size
                )
            })?;
            command_wakers.push(exec.command_waker());
            progress_fds.extend_from_slice(exec.progress_fds());
            pools.push((pool.clone(), Box::new(exec)));
        }
        let executor = StagedExecutor::try_new_with_signals(pools, command_wakers, progress_fds)?;
        let waker = executor.command_waker();
        Ok((Box::new(executor), waker))
    }

    /// Build the engine core from an executor supplied by a higher composition layer.
    pub fn with_executor(
        config: EngineCoreConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, CommandWaker::noop())
    }

    pub fn with_executor_and_waker(
        config: EngineCoreConfig,
        executor: Box<dyn Executor>,
        command_waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, command_waker)
    }

    fn assemble(
        config: EngineCoreConfig,
        executor: Box<dyn Executor>,
        waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        let ctrl = config.control_tokens();
        let sched = EngineLoop::with_runtime_profile(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch: config.max_batch,
                max_num_batched_tokens: config.max_num_batched_tokens.max(1),
                max_num_seqs: config.max_num_seqs.max(1),
                long_prefill_threshold: config.long_prefill_threshold.max(1),
                mixed_prefill_tokens: config.mixed_prefill_tokens,
                policy: config.scheduler_policy,
                ..Default::default()
            },
            config.runtime_family,
            config.runtime_profile,
        );
        let info = sched.info().clone();
        let model_dtype = sched.runtime_profile().model_dtype;
        let generation_limits = sched.runtime_profile().generation_limits.clone();
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
            info,
            stats,
            model_name: config.worker_process.model,
            model_dtype,
            generation_limits,
            max_model_len: config.max_model_len,
            runtime_family: config.runtime_family,
            dead,
            next_id: AtomicU64::new(1),
            sched_thread: Mutex::new(Some(sched_thread)),
        })
    }

    /// Cloneable command-channel front door over the scheduler.
    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    /// Worker-reported worker info (the post-load truth).
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Serving-facing projection of post-load worker limits.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
    }

    pub fn supports_token_sampling(&self) -> bool {
        self.info.supported_ops.iter().any(|mode| {
            matches!(
                mode,
                uniserve_worker_ipc::OpKind::ArDecode | uniserve_worker_ipc::OpKind::ArVerify
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

    pub const fn runtime_family(&self) -> RuntimeFamily {
        self.runtime_family
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
    pub fn submit(&self, request: Request) -> Result<EventRx, SubmitError> {
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
