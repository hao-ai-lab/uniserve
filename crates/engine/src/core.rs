//! In-process composition of worker executors, scheduler ownership, and handles.
//!
//! [`EngineCore`] owns the scheduler thread and worker lifecycle while exposing
//! a transport-independent [`EngineHandle`] to request producers.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::executor::{Executor, TransportMap, WorkerId};
use crate::handle::{EngineHandle, EventRx, SubmitError};
use crate::runtime::{ControlTokens, EngineLoop, RuntimeProfile};
use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedStats, SchedulerConfig,
    SchedulingPolicy,
};
use crate::worker::{Worker, WorkerExecutor, WorkerProcessArgs};
use anyhow::Context as _;
use uniserve_core::{
    CommandWaker, ComponentDistribution, EntryConfig, GenerationLimits, ModelDtype, ParallelConfig,
    Request, RequestId, RuntimeFamily, SequenceParallel,
};
use uniserve_worker_ipc::WorkerInfo;

/// One cooperative member's physical execution endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRank {
    pub node: String,
    pub device: String,
}

/// Static computation entries and their ordered physical rank membership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub id: WorkerId,
    pub ranks: Vec<WorkerRank>,
    pub entries: BTreeMap<String, EntryConfig>,
    pub queue_depth: usize,
}

impl WorkerConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        WorkerId::new(self.id.0.clone())?;
        anyhow::ensure!(
            self.queue_depth > 0,
            "worker {} queue depth must be positive",
            self.id
        );
        Self::validate_members(&self.ranks, &self.entries)?;
        Ok(())
    }

    pub fn validate_members(
        ranks: &[WorkerRank],
        entries: &BTreeMap<String, EntryConfig>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!ranks.is_empty(), "worker requires rank members");
        let mut devices = BTreeSet::new();
        for rank in ranks {
            anyhow::ensure!(
                !rank.node.is_empty() && !rank.device.is_empty(),
                "rank requires node and device"
            );
            anyhow::ensure!(
                devices.insert((&rank.node, &rank.device)) || rank.device == "cpu",
                "worker repeats a physical device"
            );
        }
        anyhow::ensure!(!entries.is_empty(), "worker requires computation entries");
        for (name, entry) in entries {
            anyhow::ensure!(!name.is_empty(), "entry name must not be empty");
            anyhow::ensure!(
                !entry.ranks.is_empty() && entry.ranks.iter().all(|&rank| rank < ranks.len()),
                "entry {name} contains invalid members"
            );
            anyhow::ensure!(
                entry.ranks.iter().collect::<BTreeSet<_>>().len() == entry.ranks.len(),
                "entry {name} repeats members"
            );
            let degree = entry.parallel_config.world_size()?;
            if entry.distribution.is_some() {
                anyhow::ensure!(
                    degree == 1,
                    "distributed temporal units require local entry geometry"
                );
            } else {
                anyhow::ensure!(
                    degree == entry.ranks.len(),
                    "entry {name} parallel degree {degree} disagrees with {} members",
                    entry.ranks.len()
                );
            }
            anyhow::ensure!(
                entry.units_per_rank > 0,
                "entry {name} units_per_rank must be positive"
            );
        }
        Ok(())
    }

    /// Validates unique Worker identities and one static owner for each entry.
    pub fn validate_all(workers: &[Self]) -> anyhow::Result<()> {
        anyhow::ensure!(!workers.is_empty(), "engine requires workers");
        let mut ids = BTreeSet::new();
        let mut entries = BTreeSet::new();
        for worker in workers {
            worker.validate()?;
            anyhow::ensure!(ids.insert(&worker.id), "Worker identity is repeated");
            for entry in worker.entries.keys() {
                anyhow::ensure!(
                    entries.insert(entry),
                    "entry {entry} has multiple Worker owners"
                );
            }
        }
        Ok(())
    }

    /// Expands the single-instance CLI shorthand into explicit device membership.
    pub fn model(device: &str, rank_count: usize, queue_depth: usize) -> Self {
        let ranks = (0..rank_count)
            .map(|rank| WorkerRank {
                node: "localhost".into(),
                device: if matches!(device, "cuda" | "gpu") {
                    format!("cuda:{rank}")
                } else {
                    device.into()
                },
            })
            .collect();
        Self {
            id: WorkerId("model".into()),
            ranks,
            entries: BTreeMap::from([(
                "model".into(),
                EntryConfig::parallel(
                    (0..rank_count).collect(),
                    ParallelConfig {
                        tensor_parallel_size: rank_count,
                        ..Default::default()
                    },
                ),
            )]),
            queue_depth,
        }
    }

    /// Resolves the explicit device budget into the established H3 Ulysses params.
    pub fn h3(device: &str, rank_count: usize, queue_depth: usize) -> Self {
        let mut worker = Self::model(device, rank_count, queue_depth);
        let ranks: Vec<_> = (0..rank_count).collect();
        let denoiser = ParallelConfig {
            sequence_parallel: SequenceParallel::Ulysses {
                ulysses_degree: ranks.len(),
            },
            ..ParallelConfig::default()
        };
        let encoder = ParallelConfig {
            tensor_parallel_size: ranks.len(),
            ..ParallelConfig::default()
        };
        worker.entries = BTreeMap::from([
            (
                "denoiser".into(),
                EntryConfig::parallel(ranks.clone(), denoiser),
            ),
            (
                "text_encoder".into(),
                EntryConfig::parallel(ranks.clone(), encoder),
            ),
            (
                "video_decoder".into(),
                EntryConfig {
                    ranks,
                    parallel_config: ParallelConfig::default(),
                    distribution: Some(ComponentDistribution::TemporalUnits),
                    units_per_rank: 1,
                },
            ),
            (
                "audio_decoder".into(),
                EntryConfig::parallel(vec![0], ParallelConfig::default()),
            ),
            (
                "output".into(),
                EntryConfig::parallel(vec![0], ParallelConfig::default()),
            ),
        ]);
        worker
    }
}

/// Configuration for one in-process engine core.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Request runtime selected for this configuration.
    pub runtime_family: RuntimeFamily,
    /// Model-family semantics resolved by the serving profile.
    pub runtime_profile: RuntimeProfile,
    /// Maximum number of ops assembled into a single forward batch.
    pub max_batch: usize,
    /// Maximum number of tokens scheduled in one engine step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently running requests.
    pub max_num_seqs: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as one mixed extend+decode forward. `0` disables mixing.
    pub mixed_prefill_tokens: usize,
    /// Waiting queue policy for admitting/scheduling requests.
    pub scheduler_policy: SchedulingPolicy,
    /// Maximum model context length reported to the frontend.
    pub max_model_len: u32,
    /// Static computation entries and physical rank membership.
    pub workers: Vec<WorkerConfig>,
    /// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_ipc`. Participating worker
    /// ranks receive the selected transport. Intra-Worker defaults are resolved
    /// once from rank node/device coordinates before process launch.
    pub transfer: TransportMap,
    /// Rank launch defaults refined by each Worker configuration.
    pub worker_process: WorkerProcessArgs,
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
}

impl EngineConfig {
    /// Builds a minimal configuration for the CPU simulation backend.
    ///
    /// Pair this with [`EngineCore::with_executor`]: [`EngineCore::new`] cannot
    /// build a `Sim` backend because it has no spawnable worker process.
    pub fn sim(model: impl Into<String>) -> Self {
        let worker_process = WorkerProcessArgs {
            model: model.into(),
            ranks: WorkerConfig::model("cpu", 1, 2).ranks,
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
            workers: vec![WorkerConfig::model("cpu", 1, 2)],
            transfer: TransportMap::default(),
            worker_process,
            // `SimEngine` fabricates this fake EOS id after `text_len` tokens; the
            // scheduler must recognize it to finish a sim request.
            bos: 0,
            eos: vec![151645],
            end_of_image: 0,
        }
    }

    /// Returns the control tokens for a scheduler request.
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
    /// Builds the scheduler, spawn the forward-only worker, and start the
    /// scheduler owner thread.
    ///
    /// Blocks until the worker has loaded the model and answered the
    /// worker-info handshake, including model initialization.
    pub fn new(mut config: EngineConfig) -> anyhow::Result<Self> {
        config.transfer = config.transfer.with_worker_defaults(&config.workers)?;
        let instances = config
            .workers
            .iter()
            .map(|worker| worker.id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        for edge in &config.transfer.edges {
            anyhow::ensure!(
                instances.contains(&edge.source_worker)
                    && instances.contains(&edge.destination_worker),
                "transfer edge names an unconfigured worker"
            );
        }
        let mut bindings = Vec::new();
        let mut arguments = Vec::new();
        for worker in &config.workers {
            arguments.push(WorkerProcessArgs {
                worker_id: worker.id.to_string(),
                ranks: worker.ranks.clone(),
                entries: worker.entries.clone(),
                pipeline_depth: worker.queue_depth,
                transfer: config.transfer.clone(),
                ..config.worker_process.clone()
            });
            bindings.push(worker.id.clone());
        }
        let workers = bindings
            .into_iter()
            .zip(Worker::spawn_all(arguments)?)
            .collect();
        let executor = WorkerExecutor::try_new(workers, config.transfer.clone())?;
        let waker = executor.command_waker();
        Self::assemble(config, Box::new(executor), waker)
    }

    /// Builds the engine core from an executor supplied by a higher composition layer.
    pub fn with_executor(
        config: EngineConfig,
        executor: Box<dyn Executor>,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, CommandWaker::noop())
    }

    /// Builds the engine core from an executor with an event-driven command waker.
    pub fn with_executor_and_waker(
        config: EngineConfig,
        executor: Box<dyn Executor>,
        command_waker: CommandWaker,
    ) -> anyhow::Result<Self> {
        Self::assemble(config, executor, command_waker)
    }

    /// Assembles scheduler state and transport channels around an initialized executor.
    fn assemble(
        config: EngineConfig,
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

    /// Returns a cloneable command-channel front door over the scheduler.
    pub fn handle(&self) -> EngineHandle {
        self.handle.clone()
    }

    /// Returns the worker-reported runtime capabilities.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the generation limits resolved against worker capabilities.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
    }

    /// Returns whether the worker can sample autoregressive tokens.
    pub fn supports_token_sampling(&self) -> bool {
        self.info.supported_ops.iter().any(|mode| {
            matches!(
                mode,
                uniserve_worker_ipc::OpCode::ArDecode | uniserve_worker_ipc::OpCode::ArVerify
            )
        })
    }

    /// Returns live scheduler statistics shared with the scheduler thread.
    pub fn stats(&self) -> &Arc<SchedStats> {
        &self.stats
    }

    /// Returns the served model identifier.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the model parameter data type.
    pub fn model_dtype(&self) -> ModelDtype {
        self.model_dtype
    }

    /// Returns the configured maximum model context length.
    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    /// Returns the request runtime family.
    pub const fn runtime_family(&self) -> RuntimeFamily {
        self.runtime_family
    }

    /// Allocates the next internal scheduler request id.
    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns whether the engine has encountered a terminal worker or executor failure.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    /// Submits one translated request to the scheduler.
    pub fn submit(&self, request: Request) -> Result<EventRx, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.handle.submit(request)
    }

    /// Shuts down the scheduler, tears down its executor, and joins
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
    /// Releases resources owned by this value.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_components_validate_their_own_parallel_members() -> anyhow::Result<()> {
        let mut workers = Vec::new();
        for (name, degree) in [("text_encoder", 1), ("denoiser", 4), ("video_decoder", 2)] {
            let mut worker = WorkerConfig::model("cuda", degree, 2);
            worker.id = WorkerId(name.into());
            let entry = worker.entries.remove("model").unwrap();
            worker.entries.insert(name.into(), entry);
            workers.push(worker);
        }
        WorkerConfig::validate_all(&workers)?;
        workers[1].ranks.pop();
        assert!(WorkerConfig::validate_all(&workers).is_err());
        Ok(())
    }

    #[test]
    fn entry_geometry_uses_unique_ordered_rank_members() {
        let mut worker = WorkerConfig::h3("cuda", 4, 2);
        assert!(worker.validate().is_ok());
        worker.entries.get_mut("denoiser").unwrap().ranks.swap(0, 3);
        assert!(worker.validate().is_ok());
        worker.entries.get_mut("denoiser").unwrap().ranks[1] = 3;
        assert!(worker.validate().is_err());
    }

    #[test]
    fn static_bindings_reject_duplicate_identities_and_entry_owners() {
        let worker = WorkerConfig::model("cuda", 1, 2);
        let mut other = worker.clone();
        assert!(WorkerConfig::validate_all(&[worker.clone(), other.clone()]).is_err());
        other.id = WorkerId("other".into());
        assert!(WorkerConfig::validate_all(&[worker.clone(), other.clone()]).is_err());
        let entry = other.entries.remove("model").unwrap();
        other.entries.insert("text_encoder".into(), entry);
        assert!(WorkerConfig::validate_all(&[worker, other]).is_ok());
    }
}
