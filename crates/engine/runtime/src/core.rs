//! The reified `EngineCore`: scheduler + executor + worker lifecycle, behind a
//! transport-free surface (queues + an [`EngineHandle`]), so the same engine
//! can be hosted in-process (the frontend's `InprocClient` analog) or inside a
//! headless `uniserve engine` process ([`crate::proc`]).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::Context as _;
use uniserve_core::{GenerationRequest, GenerationRuntimeCapabilities, ModelDtype, RequestId};
use uniserve_engine_api::{EngineHandle, EventRx};
use uniserve_executor::{Executor, TransferSpec, WorkerKind, WorkersSpec};
use uniserve_scheduler::{
    ControlTokens, DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH,
    DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedStats,
    Scheduler, SchedulingPolicy,
};
use uniserve_worker_ipc::{MultiprocExecutor, StageRouter, WorkerLaunchConfig};
use uniserve_worker_wire::EngineCaps;

/// Which forward-only worker the engine drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineBackend {
    /// GPU-free CPU simulation engine (`sim`): no Python, no GPU. Used for
    /// fast, deterministic end-to-end tests of the full serving stack.
    Sim,
    /// The real GPU path: spawn the Python forward-only worker
    /// (`python -m uniserve_worker.main`) and drive it over the shared-memory
    /// descriptor ring.
    Worker,
}

fn default_worker_ranks() -> usize {
    1
}

/// Place a staged pool on GPU `gpu` for tower disaggregation. A plain `cuda`/`gpu`
/// device becomes `cuda:{gpu}` (distinct GPU per pool); an explicit device
/// (`cuda:1`, `cpu`) is left as the operator set it.
fn assign_pool_device(device: &str, gpu: usize) -> String {
    match device.trim() {
        "cuda" | "gpu" => format!("cuda:{gpu}"),
        other => other.to_string(),
    }
}

/// Configuration for one engine core (in-process or headless).
#[derive(Debug, Clone)]
pub struct EngineCoreConfig {
    /// Model directory / identifier passed to the worker and used for metrics.
    pub model: String,
    /// Compute device for the worker (e.g. `cuda`, `cpu`).
    pub device: String,
    /// Which forward-only worker to drive.
    pub backend: EngineBackend,
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
    pub attention_backend: String,
    /// Python interpreter used to launch the worker.
    pub worker_python: String,
    /// Number of worker rank processes (1 = UniprocExecutor; >1 spawns a
    /// MultiprocExecutor with one ring per rank). Used as the tp size of the
    /// single Full pool when `workers` is unset (the non-disaggregated default).
    pub worker_ranks: usize,
    /// Staged-worker topology, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// `None` (or `full:1`) is the non-disaggregated default: a single Full pool
    /// driven directly, with no `StageRouter`. A multi-stage spec composes pools
    /// behind a `StageRouter`.
    pub workers: Option<String>,
    /// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
    /// `encoder->prefill=cuda_ipc,prefill->decode=mooncake`. Passed through to the
    /// `TensorMover`; unconfigured edges use the in-process backend.
    pub transfer: Option<String>,
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
            backend: EngineBackend::Sim,
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
            attention_backend: "auto".into(),
            worker_python: "python3".into(),
            worker_ranks: default_worker_ranks(),
            workers: None,
            transfer: None,
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

    fn effective_kv_token_capacity(&self) -> Option<u64> {
        self.kv_token_capacity
    }
}

/// The engine core: scheduler thread, executor, and capabilities as one
/// transport-free object.
pub struct EngineCore {
    handle: EngineHandle,
    caps: EngineCaps,
    stats: Arc<SchedStats>,
    model_name: String,
    model_dtype: ModelDtype,
    max_model_len: u32,
    generated_image_commit: uniserve_core::GeneratedImageCommitCapabilities,
    sleeping: Arc<AtomicBool>,
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
    /// Only [`EngineBackend::Worker`] is constructible here: the `Sim` backend
    /// has no spawnable process, so a `Sim` config must instead supply its
    /// `SimEngine` executor through [`EngineCore::with_executor`] (which is
    /// backend-agnostic). Passing a `Sim` config to `new` is rejected rather
    /// than silently producing a half-built engine.
    pub fn new(config: EngineCoreConfig) -> anyhow::Result<Self> {
        EngineCoreBuilder::new()
            .config(config)
            .spawn_executor()?
            .build()
    }

    /// Spawn the non-disaggregated single Full pool (the default path). `tp == 1`
    /// is a `UniprocExecutor`; `tp > 1` a `MultiprocExecutor`. No `--worker-kind`
    /// is passed because the worker starts in Full mode by default.
    fn spawn_full_pool(config: &EngineCoreConfig, tp: usize) -> anyhow::Result<Box<dyn Executor>> {
        let kv_token_capacity = config.effective_kv_token_capacity();
        if tp > 1 {
            let workers = MultiprocExecutor::spawn_with_config(
                &config.worker_python,
                &config.model,
                &config.device,
                tp,
                config.pipeline_depth,
                config.req_slot_cap,
                config.resp_slot_cap,
                kv_token_capacity,
                config.block_size,
                &config.attention_backend,
                &config.worker_launch,
            )
            .context("failed to spawn forward-only worker ranks")?;
            Ok(Box::new(workers))
        } else {
            let worker = MultiprocExecutor::spawn_with_config(
                &config.worker_python,
                &config.model,
                &config.device,
                1,
                config.pipeline_depth,
                config.req_slot_cap,
                config.resp_slot_cap,
                kv_token_capacity,
                config.block_size,
                &config.attention_backend,
                &config.worker_launch,
            )
            .context("failed to spawn forward-only worker")?;
            Ok(Box::new(worker))
        }
    }

    /// Compose a `StageRouter` over the heterogeneous pools of a staged topology.
    /// Each pool instance is a (tp-sized) `MultiprocExecutor` spawned with its
    /// `--worker-kind`; the router fans the scheduler's batch across them by
    /// exact operation type and merges the results.
    #[allow(clippy::too_many_arguments)]
    fn spawn_staged(
        config: &EngineCoreConfig,
        workers: &WorkersSpec,
    ) -> anyhow::Result<Box<dyn Executor>> {
        let kv_token_capacity = config.effective_kv_token_capacity();
        // Per-edge transfer backends (validated up front so a typo fails at
        // startup). A pool's worker uses the backend of its incoming edge for
        // the worker-side data plane (read-driven fetch).
        let transfer = match config.transfer.as_deref() {
            Some(spec) => Some(TransferSpec::parse(spec).context("invalid --transfer spec")?),
            None => None,
        };
        // A pool uses the transport of any edge it participates in — as producer
        // (src) OR consumer (dst) — so both ends of an edge agree on the backend.
        let backend_for = |kind: WorkerKind| -> Option<String> {
            let transfer = transfer.as_ref()?;
            transfer
                .edges
                .iter()
                .find(|((src, dst), _)| *src == kind || *dst == kind)
                .map(|(_, backend)| backend.clone())
        };
        // When the topology peels a Sampler, the model pools (Full/Prefill/Decode)
        // publish logits + defer sampling.
        let has_sampler = workers.pools.iter().any(|p| p.kind == WorkerKind::Sampler);
        // Tower disaggregation (und/gen) places each pool on its own GPU so the two
        // towers run in parallel and the conditioning KV crosses GPU↔GPU over
        // cuda_ipc. Non-tower staged topologies keep the shared device (the model-
        // free sampler/postprocess pools don't need a dedicated GPU).
        let is_tower = workers
            .pools
            .iter()
            .any(|p| matches!(p.kind, WorkerKind::Und | WorkerKind::Gen));
        let mut next_gpu = 0usize;
        let mut pools: Vec<(WorkerKind, Box<dyn Executor>)> =
            Vec::with_capacity(workers.total_pools());
        for pool in &workers.pools {
            // The decode↔sampler edge is inter-process, so it needs a real
            // cross-process transport; default to same-node shm when the topology
            // peels a sampler and no explicit --transfer backend was given.
            let backend = backend_for(pool.kind).or_else(|| {
                if is_tower && matches!(pool.kind, WorkerKind::Und | WorkerKind::Gen) {
                    Some("cuda_ipc".to_string())
                } else {
                    has_sampler.then(|| "shm".to_string())
                }
            });
            let defer_sampling = has_sampler
                && matches!(
                    pool.kind,
                    WorkerKind::Full | WorkerKind::Prefill | WorkerKind::Decode
                );
            for instance in 0..pool.count {
                let pool_device = if is_tower {
                    assign_pool_device(&config.device, next_gpu)
                } else {
                    config.device.clone()
                };
                next_gpu += pool.tp.max(1);
                let mut pool_worker_launch = config.worker_launch.clone();
                if let Some(root) = &config.worker_launch.snapshot_dir {
                    pool_worker_launch.snapshot_dir = Some(
                        std::path::PathBuf::from(root)
                            .join("pools")
                            .join(pool.kind.as_str())
                            .join(instance.to_string())
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
                let exec = MultiprocExecutor::spawn_staged_with_config(
                    &config.worker_python,
                    &config.model,
                    &pool_device,
                    pool.tp.max(1),
                    config.pipeline_depth,
                    config.req_slot_cap,
                    config.resp_slot_cap,
                    kv_token_capacity,
                    config.block_size,
                    &config.attention_backend,
                    pool.kind.as_str(),
                    backend.as_deref(),
                    defer_sampling,
                    &pool_worker_launch,
                )
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
        Ok(Box::new(StageRouter::try_new(pools)?))
    }

    /// Build the engine core from an executor supplied by a higher composition
    /// layer. This is the backend-agnostic entry point and the only way to
    /// construct an [`EngineBackend::Sim`] core (whose `SimEngine` executor is
    /// caller-supplied rather than spawned).
    pub fn with_executor(
        config: EngineCoreConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        EngineCoreBuilder::new()
            .config(config)
            .executor(executor)
            .build()
    }

    fn assemble(config: EngineCoreConfig, executor: Box<dyn Executor>) -> anyhow::Result<Self> {
        let ctrl = config.control_tokens();
        let generated_image_commit = executor.generated_image_commit_capabilities();
        // Capture the command waker before the executor moves into the
        // scheduler: when the executor is event-driven this fires its park's
        // command notifier; otherwise it is the no-op waker.
        let waker = executor.command_waker();
        let sched = Scheduler::with_config(
            executor,
            ctrl,
            uniserve_scheduler::SchedulerConfig {
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
        let model_dtype = ModelDtype::parse(&caps.model_dtype).ok_or_else(|| {
            anyhow::anyhow!(
                "executor reported non-canonical model dtype {:?}",
                caps.model_dtype
            )
        })?;
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
            generated_image_commit,
            sleeping: Arc::new(AtomicBool::new(false)),
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
    pub fn caps(&self) -> &EngineCaps {
        &self.caps
    }

    /// Serving-facing projection of post-load worker limits.
    pub fn generation_capabilities(&self) -> GenerationRuntimeCapabilities {
        use uniserve_worker_wire::WorkVariant;
        let caps = &self.caps;
        let supports = |variant: WorkVariant| caps.supported_work.contains(&variant);
        GenerationRuntimeCapabilities {
            supports_understanding: supports(WorkVariant::TokenExtend)
                && supports(WorkVariant::TokenDecode),
            supports_vision_encode: supports(WorkVariant::EncodeVision),
            supports_latent_encode: supports(WorkVariant::EncodeLatent),
            supports_image_generation: supports(WorkVariant::GenFlow)
                && supports(WorkVariant::Materialize),
            supports_commit_writeback: supports(WorkVariant::TransferKvPublish)
                || supports(WorkVariant::TransferKvInstall),
            max_latent_units: u64::from(caps.max_latent_size),
            latent_downsample: caps.latent_downsample,
            max_vae_grid_tokens: if caps.max_vae_grid_tokens > 0 {
                caps.max_vae_grid_tokens
            } else {
                caps.max_latent_size
            },
            max_vit_grid_tokens: caps.max_vit_grid_tokens,
            commit_marker_tokens: caps.commit_marker_tokens,
            max_cfg_branches: caps.max_cfg_branches,
            scratch_capacity_tokens: caps.scratch_capacity_tokens,
            scratch_block_size: caps.block_size,
            encoder_cache_entries: caps.encoder_cache_budget,
            generated_image_commit: self.generated_image_commit,
        }
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
    pub fn submit(&self, request: GenerationRequest) -> anyhow::Result<EventRx> {
        if self.is_dead() {
            anyhow::bail!("engine core is dead (worker failure)");
        }
        self.handle
            .submit(request)
            .map_err(|message| anyhow::anyhow!(message))
    }

    // ---- control surface (the utility-call implementations) ----

    pub fn reset_prefix_cache(
        &self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> anyhow::Result<bool> {
        if reset_connector {
            anyhow::bail!("no external prefix-cache connector is configured");
        }
        self.handle
            .reset_prefix_cache(reset_running_requests)
            .map_err(anyhow::Error::msg)
    }

    pub fn reset_encoder_cache(&self) {
        self.handle.reset_encoder_cache();
    }

    pub fn is_sleeping(&self) -> bool {
        self.sleeping.load(Ordering::Relaxed)
    }

    pub fn sleep(&self) {
        self.sleeping.store(true, Ordering::Relaxed);
        self.handle.set_sleeping(true);
    }

    pub fn wake_up(&self) {
        self.sleeping.store(false, Ordering::Relaxed);
        self.handle.set_sleeping(false);
    }

    pub fn add_lora(&self, lora_id: u32, path: String) -> bool {
        self.handle.load_lora(lora_id, path);
        true
    }

    pub fn remove_lora(&self, lora_id: u32) -> bool {
        self.handle.unload_lora(lora_id);
        true
    }

    /// Execute one control method on every worker rank, returning per-rank
    /// `(rank, ok, message)` acks (the collective_rpc surface).
    pub fn collective_rpc(&self, method: &str) -> Result<Vec<(u32, bool, Option<String>)>, String> {
        self.handle.collective_rpc(method)
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

/// Initial typestate: no config supplied yet.
pub struct NeedsConfig;

/// A config is set; an executor must still be selected (spawned or supplied).
pub struct NeedsExecutor {
    config: EngineCoreConfig,
}

/// Config and executor are both set; the engine can be built.
pub struct Ready {
    config: EngineCoreConfig,
    executor: Box<dyn Executor>,
}

/// Typestate builder for [`EngineCore`]: the construction protocol (config →
/// executor selection → scheduler init + thread spawn) is encoded in the type
/// parameter `S`, so calling the steps out of order does not compile.
pub struct EngineCoreBuilder<S> {
    state: S,
}

impl EngineCoreBuilder<NeedsConfig> {
    pub fn new() -> Self {
        Self { state: NeedsConfig }
    }

    pub fn config(self, config: EngineCoreConfig) -> EngineCoreBuilder<NeedsExecutor> {
        EngineCoreBuilder {
            state: NeedsExecutor { config },
        }
    }
}

impl Default for EngineCoreBuilder<NeedsConfig> {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineCoreBuilder<NeedsExecutor> {
    /// Supply a caller-built executor (the backend-agnostic path; the only way
    /// to carry an [`EngineBackend::Sim`] core, whose `SimEngine` executor is
    /// caller-supplied rather than spawned).
    pub fn executor(self, executor: Box<dyn Executor>) -> EngineCoreBuilder<Ready> {
        EngineCoreBuilder {
            state: Ready {
                config: self.state.config,
                executor,
            },
        }
    }

    /// Spawn the forward-only worker executor from the config (the real GPU
    /// path). Only [`EngineBackend::Worker`] is spawnable; a `Sim` config must
    /// instead reach `Ready` through [`EngineCoreBuilder::executor`].
    pub fn spawn_executor(self) -> anyhow::Result<EngineCoreBuilder<Ready>> {
        let config = self.state.config;
        if config.backend == EngineBackend::Sim {
            anyhow::bail!(
                "sim backend has no spawnable worker; construct it via \
                 EngineCore::with_executor with a caller-supplied SimEngine executor"
            );
        }
        // Resolve the staged topology. `None` or the trivial single-Full pool
        // takes the non-disaggregated path below, byte-identical to before;
        // anything else composes a StageRouter over heterogeneous pools.
        let workers = match config.workers.as_deref() {
            Some(spec) => WorkersSpec::parse(spec).context("invalid --workers spec")?,
            None => WorkersSpec::single_full(config.worker_ranks),
        };
        let executor: Box<dyn Executor> = if workers.is_single_full() {
            EngineCore::spawn_full_pool(&config, workers.pools[0].tp.max(config.worker_ranks))?
        } else {
            EngineCore::spawn_staged(&config, &workers)?
        };
        Ok(EngineCoreBuilder {
            state: Ready { config, executor },
        })
    }
}

impl EngineCoreBuilder<Ready> {
    /// Build the scheduler, capture caps/stats, and start the scheduler owner
    /// thread.
    pub fn build(self) -> anyhow::Result<EngineCore> {
        EngineCore::assemble(self.state.config, self.state.executor)
    }
}
