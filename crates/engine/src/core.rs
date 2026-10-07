//! In-process composition of worker executors, scheduler ownership, and handles.
//!
//! [`EngineCore`] owns the scheduler thread and worker lifecycle while exposing
//! a transport-independent [`EngineHandle`] to request producers.
//!
//! [`WorkerConfig`] is the static deployment description: each WorkerGroup has
//! an ordered list of physical [`WorkerRank`]s and named components whose
//! `ranks` index into that list. [`EngineConfig`] adds scheduler budgets,
//! transfer bindings, and rank launch defaults. `EngineCore::new` turns that
//! configuration into launched worker processes; `EngineCore::with_executor`
//! accepts an executor built elsewhere, such as the CPU simulation backend.

use std::collections::{BTreeMap, BTreeSet};
#[cfg(any(feature = "testing", test))]
use uniserve_worker_ipc::DEFAULT_COMPONENT;
use uniserve_worker_ipc::ForwardMode;

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::executor::{Executor, TransferConfig, WorkerId};
use crate::handle::{EngineHandle, EventRx, SubmitError};
#[cfg(feature = "testing")]
use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS,
};
use crate::scheduler::{MAX_FLOW_PREFIX_ROWS, MAX_NUM_SEQS, flow_prefix_rows};
use crate::scheduler::{Scheduler, SpecialTokenIds};
use crate::scheduler::{SchedulerConfig, SchedulerStats, SchedulingPolicy};
use crate::worker::{WorkerExecutor, WorkerGroup, WorkerProcessArgs};
use anyhow::Context as _;
#[cfg(test)]
use uniserve_core::ComponentDistribution;
use uniserve_core::{
    CommandWaker, ComponentConfig, GenerationLimits, ModelDtype, ParallelConfig, Request,
    RequestId, RuntimeFamily,
};
use uniserve_worker_ipc::WorkerInfo;

/// One cooperative member's physical execution endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRank {
    /// Host identity. The engine process spawns the ranks whose node equals
    /// its own `WorkerProcessArgs::host`; a rank on another node is started by
    /// that host's launcher.
    pub node: String,
    /// Device string, such as `cuda:0` for a GPU rank or `cpu` for a host rank.
    /// On the edges `TransferConfig::with_worker_defaults` derives, device
    /// products use CUDA VMM only between distinct ranks whose devices both
    /// start with `cuda:`.
    pub device: String,
}

/// The numerical work a worker group owns in a deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    /// Model capabilities, request state, and (when present) attention/KV.
    #[default]
    Model,
    /// Routed experts shared by the deployment's model replicas.
    Experts,
}

/// Static computation components and their ordered physical rank membership.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerConfig {
    pub id: WorkerId,
    #[serde(default)]
    pub role: WorkerRole,
    /// Ordered physical members; a rank's position is its rank number within
    /// the group.
    pub ranks: Vec<WorkerRank>,
    /// Components keyed by name, each listing indices into `ranks`.
    pub components: BTreeMap<String, ComponentConfig>,
    /// Maximum number of physical runs in flight per rank, launched as
    /// `WorkerProcessArgs::queue_depth`.
    pub queue_depth: usize,
    /// Per-rank share of device storage available to this worker process.
    ///
    /// When set, `EngineCore::new` launches the group's ranks with it as
    /// `WorkerProcessArgs::kv_storage_fraction` instead of the engine-wide
    /// default. Deployment files spell the field `memory_fraction`.
    ///
    /// Distinct WorkerGroups may share one physical GPU. Their configured
    /// shares must leave enough aggregate headroom for the device runtime;
    /// validation checks each group alone, never the sum across groups, and
    /// accepts the range [`WorkerConfig::validate_storage_fraction`] states.
    #[serde(default)]
    #[serde(rename = "memory_fraction")]
    pub storage_fraction: Option<f64>,
}

impl WorkerConfig {
    /// Validates one WorkerGroup in isolation.
    ///
    /// Fails when the identity is not a valid `WorkerId`, `queue_depth` is
    /// zero, `validate_storage_fraction` refuses a set `storage_fraction`, or
    /// `validate_members` rejects the ranks and components.
    pub fn validate(&self) -> anyhow::Result<()> {
        WorkerId::new(self.id.0.clone())?;
        anyhow::ensure!(
            self.queue_depth > 0,
            "worker {} queue depth must be positive",
            self.id
        );
        if let Some(fraction) = self.storage_fraction {
            Self::validate_storage_fraction(fraction)
                .map_err(|error| anyhow::anyhow!("worker {}: {error}", self.id))?;
        }
        Self::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() == (self.role == WorkerRole::Experts),
            "model workers require components; expert workers have no request components"
        );
        Ok(())
    }

    /// Checks a per-process device storage fraction: a group's
    /// `storage_fraction` or the engine-wide
    /// `WorkerProcessArgs::kv_storage_fraction` default.
    ///
    /// Each rank grants itself this share of every device's total storage,
    /// less what its process already holds and never more than the device
    /// has free, so any finite value in (0, 1] is meaningful, 1 granting the
    /// whole device. The worker refuses any other value at startup; checking
    /// here refuses it before a rank is launched.
    pub fn validate_storage_fraction(fraction: f64) -> anyhow::Result<()> {
        anyhow::ensure!(
            fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
            "storage fraction {fraction} must be finite and in (0, 1]"
        );
        Ok(())
    }

    /// Validates a group's rank list and its component membership.
    ///
    /// Ranks must be nonempty, name a node and a device, and not repeat a
    /// `(node, device)` pair unless the device is `cpu`. Components must be
    /// valid for their worker role; each needs a nonempty name, a nonempty list of distinct
    /// in-range rank indices, a `parallel_config` whose
    /// `ParallelConfig::world_size` succeeds, and a positive `units_per_rank`.
    /// A component without a distribution must have a parallel world size
    /// equal to its rank count; one with a distribution must have world size
    /// one.
    ///
    /// `WorkerGroup::spawn_all` calls this again for each group before
    /// starting its ranks.
    pub fn validate_members(
        ranks: &[WorkerRank],
        components: &BTreeMap<String, ComponentConfig>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!ranks.is_empty(), "worker requires rank members");
        let mut devices = BTreeSet::new();
        for rank in ranks {
            anyhow::ensure!(
                !rank.node.is_empty() && !rank.device.is_empty(),
                "rank requires node and device"
            );
            // Within a group a `(node, device)` pair names one rank, except
            // `cpu`, which several ranks on one node may share.
            anyhow::ensure!(
                devices.insert((&rank.node, &rank.device)) || rank.device == "cpu",
                "worker repeats a physical device"
            );
        }
        for (name, entry) in components {
            anyhow::ensure!(!name.is_empty(), "component name must not be empty");
            anyhow::ensure!(
                !entry.ranks.is_empty() && entry.ranks.iter().all(|&rank| rank < ranks.len()),
                "component {name} contains invalid members"
            );
            entry.validate()?;
        }
        Ok(())
    }

    /// Validates unique WorkerGroup identities and every local placement.
    ///
    /// Component names may repeat across WorkerGroups. Repeated names are
    /// replicas of the same numerical component; the scheduler binds each
    /// request to one of them and keeps that affinity for the request lifetime.
    ///
    /// Physical devices may repeat across groups; only `validate_members`
    /// checks device reuse, and it does so within one group.
    pub fn validate_all(workers: &[Self]) -> anyhow::Result<()> {
        anyhow::ensure!(!workers.is_empty(), "engine requires workers");
        let mut ids = BTreeSet::new();
        for worker in workers {
            worker.validate()?;
            anyhow::ensure!(ids.insert(&worker.id), "WorkerGroup identity is repeated");
        }
        Ok(())
    }

    /// One component tensor-parallel over every rank.
    ///
    /// The `uniserve` CLI uses this placement when no `--workers` configuration
    /// is given, and the default engine settings and the simulation use it. A
    /// deployment with several components, or with other partitions, writes
    /// its own configuration.
    pub fn single_component(name: &str, rank_count: usize) -> BTreeMap<String, ComponentConfig> {
        BTreeMap::from([(
            name.into(),
            ComponentConfig::parallel(
                (0..rank_count).collect(),
                ParallelConfig {
                    tensor_parallel_size: rank_count,
                    ..Default::default()
                },
            ),
        )])
    }

    /// Expands the single-instance CLI shorthand into explicit device membership
    /// across the named hosts, carrying the given components.
    ///
    /// Ranks are assigned in blocks, so the lowest ranks stay on the first
    /// host, and each host numbers its devices from zero. A placement on one
    /// host is the same arithmetic with one block. A `device` of `cuda` or
    /// `gpu` becomes `cuda:<local index>`; any other string is used verbatim
    /// for every rank.
    ///
    /// `components` must already be resolved for `rank_count`; this assigns
    /// rank identity and nothing else. The group is named `model` and has no
    /// storage fraction of its own. Nothing is validated here, and an empty
    /// `hosts` yields no ranks.
    pub fn placed(
        hosts: &[String],
        device: &str,
        rank_count: usize,
        queue_depth: usize,
        components: BTreeMap<String, ComponentConfig>,
    ) -> Self {
        // A host takes its share of the ranks, and the first hosts take the
        // remainder, so a count that does not divide evenly still places every
        // rank and keeps each host's block contiguous.
        let host_count = hosts.len().max(1);
        let share = rank_count / host_count;
        let remainder = rank_count % host_count;
        let mut placement = Vec::with_capacity(rank_count);
        for (index, host) in hosts.iter().enumerate() {
            let count = share + usize::from(index < remainder);
            for local in 0..count {
                placement.push(WorkerRank {
                    node: host.clone(),
                    device: if matches!(device, "cuda" | "gpu") {
                        format!("cuda:{local}")
                    } else {
                        device.into()
                    },
                });
            }
        }
        Self {
            id: WorkerId("model".into()),
            role: WorkerRole::Model,
            ranks: placement,
            components,
            queue_depth,
            storage_fraction: None,
        }
    }

    /// Expands the data-parallel CLI shorthand into `replicas` groups of
    /// `ranks_per_replica` ranks, each carrying `components`.
    ///
    /// All `replicas * ranks_per_replica` ranks are placed at once as
    /// [`WorkerConfig::placed`] places them, in blocks over `hosts`, and
    /// consecutive runs of `ranks_per_replica` form the replicas, so
    /// replicas fill the head's host first. One replica is the group
    /// `model`; replica `i` of several is `model-<i>`.
    pub fn replicated(
        hosts: &[String],
        device: &str,
        replicas: usize,
        ranks_per_replica: usize,
        queue_depth: usize,
        components: BTreeMap<String, ComponentConfig>,
    ) -> Vec<Self> {
        let placed = Self::placed(
            hosts,
            device,
            replicas * ranks_per_replica,
            queue_depth,
            components,
        );
        if replicas <= 1 {
            return vec![placed];
        }
        placed
            .ranks
            .chunks(ranks_per_replica.max(1))
            .enumerate()
            .map(|(index, ranks)| Self {
                id: WorkerId(format!("model-{index}")),
                role: WorkerRole::Model,
                ranks: ranks.to_vec(),
                components: placed.components.clone(),
                queue_depth,
                storage_fraction: None,
            })
            .collect()
    }
}

/// Configuration for the in-process engine cores of one deployment: one core,
/// or one per data-parallel replica.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Request runtime selected for this configuration.
    pub runtime_family: RuntimeFamily,
    /// Model generation requirements, intersected with loaded worker capacities.
    pub generation_limits: GenerationLimits,
    /// Maximum number of calls assembled into a single forward batch.
    pub max_batch: usize,
    /// Maximum number of tokens scheduled in one engine step.
    pub max_num_batched_tokens: usize,
    /// Maximum number of concurrently running requests. `EngineCore::new`
    /// sizes every worker's request pool from it.
    pub max_num_seqs: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub long_prefill_threshold: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as one mixed extend+decode forward. `0` disables mixing.
    pub mixed_prefill_tokens: usize,
    /// Reuse and retain prompt KV prefixes across requests. Disabling this
    /// does not remove the KV storage needed within an active request.
    pub prefix_cache: bool,
    /// Waiting queue policy for admitting/scheduling requests.
    pub scheduler_policy: SchedulingPolicy,
    /// Maximum model context length reported to the frontend.
    pub max_model_len: u32,
    /// Static computation components and physical rank membership.
    pub workers: Vec<WorkerConfig>,
    /// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
    /// `encoder->prefill=shm,prefill->decode=cuda_vmm`. Participating worker
    /// ranks receive the selected transport. Before process launch,
    /// `EngineCore::new` binds a default for every ordered rank pair, within
    /// and across WorkerGroups, that no configured edge covers, derived from
    /// rank node/device coordinates.
    pub transfer: TransferConfig,
    /// Number of independent replicas `workers` forms, as equal consecutive
    /// blocks of groups; each replica is served by its own engine core
    /// ([`EngineCore::replicas`]). One means the workers form one engine.
    pub data_parallel_size: usize,
    /// Whether the data-parallel replicas shard the model's routed experts,
    /// and how they exchange tokens: each replica is then one rank, keeps
    /// its share of every expert layer and exchanges tokens with the others
    /// at those layers, while attention and every other layer stay
    /// data-parallel. `None` keeps every expert on every replica.
    pub expert_parallel: Option<crate::worker::ExpertExchange>,
    /// Rank launch defaults refined by each WorkerGroup configuration.
    ///
    /// Every constructor also reads the model identifier (`model`) and
    /// `model_dtype` from here.
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
    #[cfg(feature = "testing")]
    pub fn sim(model: impl Into<String>) -> Self {
        let worker_process = WorkerProcessArgs {
            model: model.into(),
            ranks: WorkerConfig::placed(
                &["localhost".to_owned()],
                "cpu",
                1,
                2,
                WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            )
            .ranks,
            block_size: Some(64),
            queue_depth: 2,
            max_batch_calls: DEFAULT_MAX_BATCH as u32,
            max_batch_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS as u32,
            ..WorkerProcessArgs::default()
        };
        Self {
            runtime_family: RuntimeFamily::Umm,
            generation_limits: crate::scheduler::unbounded_umm_generation_limits(),
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
            prefix_cache: true,
            scheduler_policy: SchedulingPolicy::Fcfs,
            max_model_len: 8192,
            workers: vec![WorkerConfig::placed(
                &["localhost".to_owned()],
                "cpu",
                1,
                2,
                WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
            )],
            transfer: TransferConfig::default(),
            data_parallel_size: 1,
            expert_parallel: None,
            worker_process,
            bos: 0,
            // `SimEngine` fabricates this fake EOS id after `text_len` tokens; the
            // scheduler must recognize it to finish a sim request.
            eos: vec![151645],
            end_of_image: 0,
        }
    }

    /// Returns the control tokens for a scheduler request.
    fn control_tokens(&self) -> SpecialTokenIds {
        SpecialTokenIds {
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
    stats: Arc<SchedulerStats>,
    model_name: String,
    model_dtype: ModelDtype,
    generation_limits: GenerationLimits,
    max_model_len: u32,
    runtime_family: RuntimeFamily,
    /// Engine-dead latch: set after `Scheduler::run` returns a fatal exit
    /// (executor or worker failure). A scheduler thread panic does not set it.
    dead: Arc<AtomicBool>,
    next_id: AtomicU64,
    sched_thread: Mutex<Option<JoinHandle<()>>>,
}

impl EngineCore {
    /// Launches every configured WorkerGroup, builds the scheduler, and starts
    /// the scheduler owner thread.
    ///
    /// Every group is launched with a request pool of `max_num_seqs` rows
    /// plus the largest reserve a runtime keeps outside that limit.
    ///
    /// Blocks until every rank has loaded the model and reported its
    /// capabilities. Fails when `config.data_parallel_size` is not one (see
    /// [`EngineCore::replicas`]), `WorkerConfig::validate_all` rejects the
    /// workers, a configured transfer edge names an unconfigured worker,
    /// `WorkerGroup::spawn_all` fails to launch a group,
    /// `WorkerExecutor::try_new` refuses the launched groups, the workers of
    /// a runtime with a KV cache hold too few request rows for
    /// `max_num_seqs` running requests, or the scheduler or its thread
    /// cannot be created.
    pub fn new(config: EngineConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.data_parallel_size == 1,
            "one engine core serves one replica; EngineCore::replicas launches \
             data-parallel replicas"
        );
        let mut cores = Self::replicas(config)?;
        Ok(cores.remove(0))
    }

    /// Launches `config.data_parallel_size` independent replicas of the
    /// deployment and returns one engine core per replica.
    ///
    /// `config.workers` lists the replicas as equal consecutive blocks of
    /// WorkerGroups. Each replica gets its own scheduler thread, executor, KV
    /// pool, prefix cache and request rows over its own groups, the way
    /// SGLang runs one scheduler per data-parallel rank behind its
    /// `DataParallelController` and vLLM one engine core per data-parallel
    /// rank. Every group of every replica launches through one
    /// `WorkerGroup::spawn_all`, so the replicas load their models at the same
    /// time and one launcher per remote host serves all of them.
    ///
    /// Replicas exchange no products: every configured transfer edge must stay
    /// inside one replica, and each replica binds its default edges over its
    /// own groups only.
    ///
    /// Fails when the size is zero or does not divide the groups, when the
    /// replicas are not the same deployment on different ranks (group count,
    /// rank counts, components, queue depth and storage fraction must match
    /// position by position), when a configured edge crosses replicas or
    /// names an unconfigured worker, or for any reason `EngineCore::new`
    /// names.
    pub fn replicas(config: EngineConfig) -> anyhow::Result<Vec<Self>> {
        let size = config.data_parallel_size;
        WorkerConfig::validate_all(&config.workers)?;
        let (model_workers, expert_workers): (Vec<_>, Vec<_>) = config
            .workers
            .iter()
            .cloned()
            .partition(|worker| worker.role == WorkerRole::Model);
        anyhow::ensure!(
            size > 0 && !model_workers.is_empty() && model_workers.len().is_multiple_of(size),
            "{} worker groups do not form {size} equal data-parallel replicas",
            model_workers.len()
        );
        let replicas: Vec<&[WorkerConfig]> =
            model_workers.chunks(model_workers.len() / size).collect();
        for replica in &replicas[1..] {
            for (group, first) in replica.iter().zip(replicas[0]) {
                anyhow::ensure!(
                    group.ranks.len() == first.ranks.len()
                        && group.components == first.components
                        && group.queue_depth == first.queue_depth
                        && group.storage_fraction == first.storage_fraction,
                    "data-parallel replicas must place the same groups: worker {} \
                     differs from worker {}",
                    group.id,
                    first.id
                );
            }
        }

        let disaggregated = !expert_workers.is_empty();
        anyhow::ensure!(
            (1..=4).contains(&config.worker_process.expert_microbatches)
                && (config.worker_process.expert_microbatches == 1 || disaggregated),
            "one to four expert microbatches require disaggregated placement when greater than one"
        );
        if disaggregated {
            anyhow::ensure!(
                matches!(
                    config.expert_parallel,
                    Some(
                        crate::worker::ExpertExchange::DeepEp
                            | crate::worker::ExpertExchange::MegaMoe
                    )
                ) && replicas.iter().all(|replica| replica.len() == 1),
                "disaggregated experts require deepep or megamoe and one model group per replica"
            );
        } else {
            anyhow::ensure!(
                config.expert_parallel.is_none()
                    || (config.expert_parallel != Some(crate::worker::ExpertExchange::DeepEp)
                        && size > 1
                        && replicas
                            .iter()
                            .all(|replica| { replica.len() == 1 && replica[0].ranks.len() == 1 })),
                "colocated expert parallelism requires two or more one-rank replicas; \
                 deepep requires dedicated expert workers"
            );
        }
        let attention_ranks = if disaggregated {
            model_workers
                .iter()
                .map(|worker| worker.ranks.len())
                .sum::<usize>()
        } else {
            0
        };
        let expert_world_size = if disaggregated {
            attention_ranks
                + expert_workers
                    .iter()
                    .map(|worker| worker.ranks.len())
                    .sum::<usize>()
        } else {
            size
        };

        // Each replica resolves its transfer edges over its own groups. The
        // edges `with_worker_defaults` adds only name the replica's groups, so
        // checking the configured ones covers every edge.
        let mut transfers = Vec::with_capacity(size);
        let mut assigned = 0;
        for replica in &replicas {
            let members = replica
                .iter()
                .map(|worker| worker.id.clone())
                .collect::<BTreeSet<_>>();
            let edges = config
                .transfer
                .edges
                .iter()
                .filter(|edge| members.contains(&edge.source_worker))
                .cloned()
                .collect::<Vec<_>>();
            for edge in &edges {
                anyhow::ensure!(
                    members.contains(&edge.destination_worker),
                    "transfer edge {} -> {} leaves its source's data-parallel replica",
                    edge.source_worker,
                    edge.destination_worker
                );
            }
            assigned += edges.len();
            transfers.push(
                TransferConfig {
                    edges,
                    ..config.transfer.clone()
                }
                .with_worker_defaults(replica)?,
            );
        }
        anyhow::ensure!(
            assigned == config.transfer.edges.len(),
            "transfer edge names an unconfigured worker"
        );

        // Every group holds a request row for each running request the
        // scheduler may keep resident, plus the rows a runtime keeps outside
        // that limit. Only the loaded capabilities decide that reserve
        // (`flow_prefix_rows`), so the launch sizes for the largest one.
        let max_num_seqs = config.max_num_seqs.clamp(1, MAX_NUM_SEQS);
        let max_request_pool_size = u32::try_from(max_num_seqs + MAX_FLOW_PREFIX_ROWS)
            .context("max_num_seqs exceeds the worker request-pool field")?;

        // Each group launches from the shared defaults with its own identity,
        // placement, components, queue depth, and storage fraction, plus its
        // replica's resolved transfer edges and the components of every group
        // of its replica (`peers`), from which the group resolves the ranks,
        // possibly in other groups, that read its products.
        let mut arguments = Vec::with_capacity(config.workers.len());
        let mut union_rank = 0;
        for (replica, transfer) in replicas.iter().zip(&transfers) {
            let peers = replica
                .iter()
                .map(|worker| (worker.id.to_string(), worker.components.clone()))
                .collect::<BTreeMap<_, _>>();
            for worker in *replica {
                arguments.push(WorkerProcessArgs {
                    worker_id: worker.id.to_string(),
                    role: worker.role,
                    ranks: worker.ranks.clone(),
                    components: worker.components.clone(),
                    peers: peers.clone(),
                    queue_depth: worker.queue_depth,
                    max_request_pool_size,
                    kv_storage_fraction: worker
                        .storage_fraction
                        .unwrap_or(config.worker_process.kv_storage_fraction),
                    transfer: transfer.clone(),
                    expert_parallel: config.expert_parallel.map(|exchange| {
                        crate::worker::ExpertParallelPlacement {
                            rank: union_rank as u32,
                            size: expert_world_size as u32,
                            attention_ranks: attention_ranks as u32,
                            address: None,
                            exchange,
                        }
                    }),
                    ..config.worker_process.clone()
                });
                union_rank += worker.ranks.len();
            }
        }
        for worker in &expert_workers {
            arguments.push(WorkerProcessArgs {
                worker_id: worker.id.to_string(),
                role: worker.role,
                ranks: worker.ranks.clone(),
                components: worker.components.clone(),
                peers: BTreeMap::new(),
                queue_depth: worker.queue_depth,
                max_request_pool_size,
                kv_storage_fraction: worker
                    .storage_fraction
                    .unwrap_or(config.worker_process.kv_storage_fraction),
                transfer: transfers[0].clone(),
                expert_parallel: config.expert_parallel.map(|exchange| {
                    crate::worker::ExpertParallelPlacement {
                        rank: union_rank as u32,
                        size: expert_world_size as u32,
                        attention_ranks: attention_ranks as u32,
                        address: None,
                        exchange,
                    }
                }),
                ..config.worker_process.clone()
            });
            union_rank += worker.ranks.len();
        }

        // `spawn_all` returns groups in argument order, which pairs each group
        // with its identity and, block by block, with its replica.
        let mut groups = model_workers
            .iter()
            .chain(&expert_workers)
            .map(|worker| worker.id.clone())
            .zip(WorkerGroup::spawn_all(arguments)?)
            .collect::<Vec<_>>();
        // Shared expert groups have one lifecycle owner. They advertise no
        // request capabilities, so they never become a scheduling replica.
        let mut shared = groups.split_off(model_workers.len());
        let mut groups = groups.into_iter();
        let mut cores = Vec::with_capacity(size);
        for (index, (replica, transfer)) in replicas.iter().zip(transfers).enumerate() {
            let mut workers = groups.by_ref().take(replica.len()).collect::<Vec<_>>();
            if index == 0 {
                workers.append(&mut shared);
            }
            let executor = WorkerExecutor::try_new(workers, transfer.clone())?;

            // A token runtime serves `max_num_seqs` running requests only while
            // its workers hold that many request rows beyond the reserve; the
            // scheduler would otherwise lower the limit. Workers without a KV
            // cache fit their request pools to device storage, and the
            // scheduler bounds their requests by the rows they report.
            let info = executor.info().runtime_info()?;
            let request_rows = info.request_slots as usize;
            let needed_rows = max_num_seqs + flow_prefix_rows(&info);
            anyhow::ensure!(
                !info.uses_kv() || request_rows >= needed_rows,
                "the workers hold {request_rows} request rows, but {max_num_seqs} running \
                 requests need {needed_rows}"
            );

            let waker = executor.command_waker();
            let mut placement = replica.to_vec();
            if index == 0 {
                placement.extend(expert_workers.iter().cloned());
            }
            let replica_config = EngineConfig {
                workers: placement,
                transfer,
                data_parallel_size: 1,
                expert_parallel: None,
                ..config.clone()
            };
            cores.push(Self::assemble(replica_config, Box::new(executor), waker)?);
        }
        Ok(cores)
    }

    /// Builds the engine core from an executor supplied by a higher composition layer.
    ///
    /// The command waker is a no-op, so a command does not interrupt the
    /// scheduler's idle park; `with_executor_and_waker` supplies a real one.
    /// Neither constructor uses `config.workers` or `config.transfer`.
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
        let mut sched = Scheduler::with_model_limits(
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
            config.worker_process.model_dtype,
            config.generation_limits,
        )?;
        sched.set_prefix_cache(config.prefix_cache);
        let info = sched.info().clone();
        let model_dtype = config.worker_process.model_dtype;
        let generation_limits = sched.generation_limits().clone();
        let stats = sched.stats_handle();

        // The scheduler thread owns the `Scheduler` and its executor; request
        // producers reach it only through the command channel.
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
    ///
    /// With several WorkerGroups this is the executor's merged view
    /// (`ExecutorInfo::runtime_info`), not any single group's report.
    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    /// Returns the generation limits resolved against worker capabilities.
    pub fn generation_limits(&self) -> GenerationLimits {
        self.generation_limits.clone()
    }

    /// Returns whether the reported capabilities include a decode or verify
    /// forward; with several WorkerGroups, whether any group serves one.
    ///
    /// The server advertises request sampling controls only when this holds
    /// (`EngineClient::served_sampling_controls`).
    pub fn supports_token_sampling(&self) -> bool {
        self.info.supported_calls.iter().any(|mode| {
            matches!(
                mode,
                uniserve_worker_ipc::CallKind::Forward(ForwardMode::Decode)
                    | uniserve_worker_ipc::CallKind::Forward(ForwardMode::Verify)
            )
        })
    }

    /// Returns live scheduler statistics shared with the scheduler thread.
    pub fn stats(&self) -> &Arc<SchedulerStats> {
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
    ///
    /// Ids start at 1 and are unique for the lifetime of this core; `Relaxed`
    /// suffices because only uniqueness is required.
    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns whether the engine has encountered a terminal worker or executor failure.
    pub fn is_dead(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }

    /// Submits one translated request to the scheduler.
    ///
    /// Returns `SubmitError::Dead` once the dead latch is set and
    /// `SubmitError::Closed` when the command channel no longer accepts
    /// commands. Submitting through a cloned `handle()` skips the latch check
    /// and can only report `Closed`.
    pub fn submit(&self, request: Request) -> Result<EventRx, SubmitError> {
        if self.is_dead() {
            return Err(SubmitError::Dead);
        }
        self.handle.submit(request)
    }

    /// Shuts down the scheduler, tears down its executor, and joins
    /// its thread. Idempotent.
    ///
    /// Blocks until the scheduler thread exits; on shutdown `Scheduler::run`
    /// first finishes queued and running requests with `Aborted` and closes
    /// the executor. A poisoned thread lock or a panicked scheduler thread is
    /// logged, not propagated.
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
    /// Runs `EngineCore::shutdown`, blocking until the scheduler thread exits.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entries covering the shapes a placement must validate: one component
    /// parallel over every rank, one dividing its output into units, and one
    /// placed alone.
    fn mixed_components(rank_count: usize) -> BTreeMap<String, ComponentConfig> {
        let members: Vec<_> = (0..rank_count).collect();
        let mut components = WorkerConfig::single_component("parallel", rank_count);
        components.insert(
            "divided".into(),
            ComponentConfig {
                ranks: members,
                parallel_config: ParallelConfig::default(),
                distribution: Some(ComponentDistribution::TemporalUnits),
                units_per_rank: 1,
            },
        );
        components.insert(
            "alone".into(),
            ComponentConfig::parallel(vec![0], ParallelConfig::default()),
        );
        components
    }

    /// Groups of different sizes validate independently, and a group whose
    /// component spans more ranks than the group holds is refused.
    #[test]
    fn static_components_validate_their_own_parallel_members() -> anyhow::Result<()> {
        let mut workers = Vec::new();
        for (name, degree) in [("first", 1), ("second", 4), ("third", 2)] {
            let mut worker = WorkerConfig::placed(
                &["localhost".to_owned()],
                "cuda",
                degree,
                2,
                WorkerConfig::single_component(name, degree),
            );
            worker.id = WorkerId(name.into());
            workers.push(worker);
        }
        WorkerConfig::validate_all(&workers)?;

        workers[1].ranks.pop();
        assert!(WorkerConfig::validate_all(&workers).is_err());
        Ok(())
    }

    #[test]
    fn the_shorthand_blocks_ranks_across_the_hosts_it_is_given() {
        // Eight devices over two four-device hosts, the head's host first as
        // the CLI lists them. `WorkerGroup` refuses a muxer off the head's
        // host, so a deployment that places its muxer on rank zero needs the
        // lowest ranks on the first host, and each host must number its own
        // devices from zero.
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let worker = WorkerConfig::placed(&hosts, "cuda", 8, 2, mixed_components(8));

        let placement: Vec<_> = worker
            .ranks
            .iter()
            .map(|rank| (rank.node.as_str(), rank.device.as_str()))
            .collect();
        assert_eq!(
            placement,
            vec![
                ("rank-0", "cuda:0"),
                ("rank-0", "cuda:1"),
                ("rank-0", "cuda:2"),
                ("rank-0", "cuda:3"),
                ("rank-1", "cuda:0"),
                ("rank-1", "cuda:1"),
                ("rank-1", "cuda:2"),
                ("rank-1", "cuda:3"),
            ]
        );
        assert!(worker.validate().is_ok());

        // A component the model spans over the instance keeps all eight ranks,
        // and one it places alone stays on rank zero, which is the head's host.
        for entry in ["parallel", "divided"] {
            assert_eq!(
                worker.components[entry].ranks.len(),
                8,
                "{entry} spans the instance"
            );
        }
        assert_eq!(worker.components["alone"].ranks, vec![0]);
    }

    /// Four two-rank replicas over two four-device hosts fill the head's host
    /// first, and each replica keeps its ranks on one host.
    #[test]
    fn the_data_parallel_shorthand_places_replicas_in_host_blocks() {
        let hosts = ["rank-0".to_owned(), "rank-1".to_owned()];
        let replicas = WorkerConfig::replicated(
            &hosts,
            "cuda",
            4,
            2,
            2,
            WorkerConfig::single_component("model", 2),
        );

        let placement: Vec<_> = replicas
            .iter()
            .map(|worker| {
                (
                    worker.id.to_string(),
                    worker
                        .ranks
                        .iter()
                        .map(|rank| format!("{}/{}", rank.node, rank.device))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            placement,
            vec![
                (
                    "model-0".into(),
                    vec!["rank-0/cuda:0".into(), "rank-0/cuda:1".into()]
                ),
                (
                    "model-1".into(),
                    vec!["rank-0/cuda:2".into(), "rank-0/cuda:3".into()]
                ),
                (
                    "model-2".into(),
                    vec!["rank-1/cuda:0".into(), "rank-1/cuda:1".into()]
                ),
                (
                    "model-3".into(),
                    vec!["rank-1/cuda:2".into(), "rank-1/cuda:3".into()]
                ),
            ]
        );
        assert!(WorkerConfig::validate_all(&replicas).is_ok());

        // One replica keeps the single-group identity.
        let single = WorkerConfig::replicated(
            &hosts,
            "cuda",
            1,
            2,
            2,
            WorkerConfig::single_component("model", 2),
        );
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].id.to_string(), "model");
    }

    /// Replicas must be the same deployment on different ranks, and the size
    /// must divide the groups; both are refused before any rank launches.
    #[test]
    fn data_parallel_replicas_refuse_unequal_deployments() {
        let hosts = ["localhost".to_owned()];
        let mut config = EngineConfig {
            workers: WorkerConfig::replicated(
                &hosts,
                "cpu",
                2,
                1,
                2,
                WorkerConfig::single_component("model", 1),
            ),
            data_parallel_size: 3,
            ..EngineConfig::sim("sim-model")
        };
        assert!(EngineCore::replicas(config.clone()).is_err());

        config.data_parallel_size = 2;
        config.workers[1].queue_depth = 3;
        let error = EngineCore::replicas(config)
            .err()
            .expect("unequal replicas are refused");
        assert!(error.to_string().contains("data-parallel replicas"));
    }

    /// Experts shard across two or more replicas of one rank each; any other
    /// placement is refused before a rank launches.
    #[test]
    fn expert_parallelism_needs_several_one_rank_replicas() {
        let hosts = ["localhost".to_owned()];
        for (replicas, ranks) in [(1, 1), (2, 2)] {
            let config = EngineConfig {
                workers: WorkerConfig::replicated(
                    &hosts,
                    "cpu",
                    replicas,
                    ranks,
                    2,
                    WorkerConfig::single_component("model", ranks),
                ),
                data_parallel_size: replicas,
                expert_parallel: Some(crate::worker::ExpertExchange::AllToAll),
                ..EngineConfig::sim("sim-model")
            };
            let error = EngineCore::replicas(config)
                .err()
                .expect("the placement is refused");
            assert!(error.to_string().contains("expert parallelism"), "{error}");
        }
    }

    #[test]
    fn a_rank_count_that_does_not_divide_its_hosts_still_places_every_rank() {
        // The first hosts take the remainder, which keeps each host's block
        // contiguous and rank zero on the head's host.
        let hosts = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let worker = WorkerConfig::placed(
            &hosts,
            "cuda",
            8,
            2,
            WorkerConfig::single_component(DEFAULT_COMPONENT, 8),
        );

        let nodes: Vec<_> = worker.ranks.iter().map(|rank| rank.node.as_str()).collect();
        assert_eq!(nodes, vec!["a", "a", "a", "b", "b", "b", "c", "c"]);
        assert_eq!(worker.ranks.len(), 8);
    }

    /// Reordering a component's ranks is a valid placement; repeating a rank
    /// is not.
    #[test]
    fn entry_geometry_uses_unique_ordered_rank_members() {
        let mut worker =
            WorkerConfig::placed(&["localhost".to_owned()], "cuda", 4, 2, mixed_components(4));
        assert!(worker.validate().is_ok());

        worker
            .components
            .get_mut("parallel")
            .unwrap()
            .ranks
            .swap(0, 3);
        assert!(worker.validate().is_ok());

        worker.components.get_mut("parallel").unwrap().ranks[1] = 3;
        assert!(worker.validate().is_err());
    }

    #[test]
    fn static_bindings_reject_duplicate_identities_and_allow_component_replicas() {
        let worker = WorkerConfig::placed(
            &["localhost".to_owned()],
            "cuda",
            1,
            2,
            WorkerConfig::single_component(DEFAULT_COMPONENT, 1),
        );
        let mut other = worker.clone();
        assert!(WorkerConfig::validate_all(&[worker.clone(), other.clone()]).is_err());

        // Distinct identities on the same device validate, whether the groups
        // replicate one component name or hold different ones.
        other.id = WorkerId("other".into());
        assert!(WorkerConfig::validate_all(&[worker.clone(), other.clone()]).is_ok());

        let entry = other.components.remove(DEFAULT_COMPONENT).unwrap();
        other.components.insert("divided".into(), entry);
        assert!(WorkerConfig::validate_all(&[worker, other]).is_ok());
    }
}
