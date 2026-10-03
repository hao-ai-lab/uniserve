//! Logical execution batches, physical execution, and executor contracts.
//!
//! The scheduler submits logical calls through [`Executor`]. Physical
//! executors lower those calls into worker protocol batches while retaining
//! the request and product identities needed to correlate completions. The
//! implementations are `WorkerExecutor` (`crate::worker::executor`), which
//! drives worker processes, and `SimExecutor` (`crate::sim`).
//!
//! Besides the trait, this module owns the values that cross that boundary:
//! [`ExecutorInfo`], which merges the pools' reported capabilities into the
//! single capacity view the scheduler plans against; the result types a poll
//! returns; and [`TransferConfig`], the directed transfer edges between worker
//! ranks and the placement they resolve against, from which the head derives
//! each rank's launch descriptor fields (transfer mechanisms, acknowledgment
//! slot, same-host slots, and whether products cross hosts) and the consumer
//! slots stated on each producing call.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub use uniserve_core::{ComponentConfig, ComponentDistribution, ParallelConfig, SequenceParallel};

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
#[cfg(test)]
use uniserve_worker_ipc::VideoDenoiserInfo;
use uniserve_worker_ipc::{ForwardMode, MediaCall};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use uniserve_worker_ipc::{
    BatchCommand, BatchOutput, CallId, CallKind, RequestKey, TensorPublication, WorkerInfo,
};

mod batch;

pub use batch::{ExecutionBatch, RequestPlacement};

/// Concrete pool information reported through the execution boundary.
#[derive(Debug, Clone)]
pub struct ExecutorInfo {
    /// Physical pool identities and their reported capabilities.
    pub workers: Vec<(WorkerId, WorkerInfo)>,
    /// For each worker that decodes video, the encoder replicas whose rank
    /// dealing keeps every media unit on the host that decoded it. A decoder
    /// absent from the map places no host constraint on its encoder.
    ///
    /// `WorkerExecutor::try_new` fills this from the placement; `single` and
    /// `from_workers` leave it empty. Scheduler admission restricts a video
    /// encoding call's candidates to the set of the worker chosen to decode.
    pub video_codecs: BTreeMap<WorkerId, BTreeSet<WorkerId>>,
}

impl ExecutorInfo {
    /// Constructs capability information for one physical pool.
    pub fn single(id: WorkerId, info: WorkerInfo) -> Self {
        Self {
            workers: vec![(id, info)],
            video_codecs: BTreeMap::new(),
        }
    }

    /// Validates and constructs capability information for multiple pools.
    ///
    /// Fails when `pools` is empty, repeats a pool id, contains a record that
    /// fails `WorkerInfo::validate`, mixes models or checkpoint identities, or
    /// holds replicas of one component that expose different numerical
    /// outputs.
    pub fn from_workers(pools: Vec<(WorkerId, WorkerInfo)>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !pools.is_empty(),
            "executor info must contain at least one pool"
        );
        let mut ids = std::collections::HashSet::new();
        let mut components: BTreeMap<&str, &uniserve_worker_ipc::ComponentInfo> = BTreeMap::new();
        let model_name = &pools[0].1.model_name;
        let checkpoint = &pools[0].1.checkpoint_identity;
        for (id, info) in &pools {
            anyhow::ensure!(ids.insert(id), "executor info repeats pool id {id}");
            info.validate()?;
            anyhow::ensure!(
                &info.model_name == model_name && &info.checkpoint_identity == checkpoint,
                "worker {id} loaded a different model or checkpoint"
            );
            for component in &info.components {
                if let Some(other) = components.get(component.name.as_str()) {
                    anyhow::ensure!(
                        other.outputs == component.outputs,
                        "replicas of component {} expose different numerical outputs",
                        component.name
                    );
                } else {
                    components.insert(&component.name, component);
                }
            }
        }
        Ok(Self {
            workers: pools,
            video_codecs: BTreeMap::new(),
        })
    }

    /// Resolves which component serves each media call across every worker.
    ///
    /// The video graph may span workers: a model worker serves decoding and a
    /// host worker serves encoding and muxing. Each call is served by one
    /// component, which several workers may replicate, and a deployment that
    /// muxes serves the whole video graph between its workers.
    pub fn media_routing(&self) -> anyhow::Result<BTreeMap<MediaCall, String>> {
        // The first worker seen serving a call is kept only to name it if
        // another worker routes that call to a different component.
        let mut routing: BTreeMap<MediaCall, (String, &WorkerId)> = BTreeMap::new();
        for (id, info) in &self.workers {
            for (call, component) in &info.media_components {
                if let Some((other, owner)) = routing.get(call) {
                    anyhow::ensure!(
                        other == component,
                        "media call {call:?} is served by component {other} of worker {owner} \
                         and by component {component} of worker {id}"
                    );
                    continue;
                }
                routing.insert(*call, (component.clone(), id));
            }
        }
        if routing.contains_key(&MediaCall::Muxing) {
            let missing = crate::scheduler::graph::task_calls(uniserve_core::VideoTask::T2va)
                .into_iter()
                .filter(|call| !routing.contains_key(call))
                .map(|call| format!("{call:?}"))
                .collect::<Vec<_>>();
            anyhow::ensure!(
                missing.is_empty(),
                "a deployment that muxes video serves no component for {}",
                missing.join(", ")
            );
        }
        Ok(routing
            .into_iter()
            .map(|(call, (component, _))| (call, component))
            .collect())
    }

    /// Returns the sole worker capability record.
    ///
    /// # Panics
    ///
    /// Panics unless the executor contains exactly one physical pool.
    pub fn single_worker(&self) -> &WorkerInfo {
        assert_eq!(
            self.workers.len(),
            1,
            "executor does not contain exactly one physical pool"
        );
        &self.workers[0].1
    }

    /// Derives the runtime's immutable capacity view from concrete pools.
    /// The returned value is not part of executor identity and is never
    /// reported as a physical worker.
    ///
    /// A single pool's record is returned unchanged. With several pools, the
    /// record is cloned from the first pool that serves a KV forward call (or
    /// the first pool when none does), so fields the merge does not set, such
    /// as the model identity and endpoint, come from that pool. Call support and
    /// media routing are unions, queue depth is a sum, and most batch, slot,
    /// and storage limits take the smallest nonzero value any pool reports.
    /// The exceptions are the KV cache, whose layout the KV pools must share
    /// and whose block count is their smallest and per-token footprint their
    /// largest; request slots of a deployment that muxes video,
    /// the narrowest per-component sum of replica slots; and the diffusion
    /// step count and latent page geometry, taken from the first denoising
    /// pool and zero without one.
    ///
    /// # Errors
    ///
    /// Fails when there are no pools. With several pools, it also fails when
    /// media routing conflicts or is incomplete, when a deployment that muxes
    /// video reports no diffusion step count, when denoising pools disagree on the
    /// step count, when KV pools disagree on cache layout or a routed KV pool
    /// has no cache, when summed capacities overflow `u32`, or when the merged
    /// record fails `WorkerInfo::validate`.
    pub fn runtime_info(&self) -> anyhow::Result<WorkerInfo> {
        anyhow::ensure!(
            !self.workers.is_empty(),
            "executor exposes no physical pools"
        );
        if self.workers.len() == 1 {
            let info = self.workers[0].1.clone();
            check_video_tasks(&info)?;
            return Ok(info);
        }

        // Route-specific capacities contribute only when a pool implements the
        // corresponding call family. `routed` picks the first such pool, and
        // `kv_indices` holds the index of the first pool serving each KV
        // forward mode (prefill, decode, verify), deduplicated and in pool
        // order.
        let routed = |variant: CallKind| {
            self.workers
                .iter()
                .find(|(_, info)| info.supported_calls.contains(&variant))
                .map(|(_, info)| info)
        };
        let mut kv_indices = [
            CallKind::Forward(ForwardMode::Prefill),
            CallKind::Forward(ForwardMode::Decode),
            CallKind::Forward(ForwardMode::Verify),
        ]
        .into_iter()
        .filter_map(|variant| {
            self.workers
                .iter()
                .position(|(_, info)| info.supported_calls.contains(&variant))
        })
        .collect::<Vec<_>>();
        kv_indices.sort_unstable();
        kv_indices.dedup();

        // Fields not merged below keep the seed pool's values.
        let seed_index = kv_indices.first().copied().unwrap_or(0);
        let mut merged = self.workers[seed_index].1.clone();
        merged.media_components = self.media_routing()?;
        let denoising = routed(CallKind::Media(MediaCall::Denoising));
        merged.num_inference_steps = denoising.map_or(0, |info| info.num_inference_steps);
        // Every worker describes the deployment's placed denoiser alike; the
        // denoising worker's description is the deployment's.
        merged.video_denoiser = denoising.and_then(|info| info.video_denoiser.clone());
        // A deployment that assembles video denoises over a fixed ladder,
        // which the denoising worker reports.
        anyhow::ensure!(
            !merged.media_components.contains_key(&MediaCall::Muxing)
                || merged.num_inference_steps > 0,
            "a deployment that muxes video reports no diffusion step count"
        );
        anyhow::ensure!(
            self.workers.iter().all(|(_, info)| {
                !info
                    .supported_calls
                    .contains(&CallKind::Media(MediaCall::Denoising))
                    || info.num_inference_steps == merged.num_inference_steps
            }),
            "workers disagree on diffusion steps"
        );
        anyhow::ensure!(
            self.workers
                .iter()
                .all(|(_, info)| info.video_denoiser.is_none()
                    || info.video_denoiser == merged.video_denoiser),
            "workers disagree on the deployment's video denoiser"
        );
        // Every KV stage must agree on its groups' retention policies and page
        // shapes. Capacity is the narrowest pool because a request may
        // traverse all routed KV stages with one set of unit ids, and the
        // per-unit footprint is the largest any of them reports
        // (`KvCacheInfo::merge`).
        if let Some(first_index) = kv_indices.first().copied() {
            let mut kv_cache = self.workers[first_index]
                .1
                .kv_cache
                .clone()
                .context("executor routes KV work to a pool without a KV cache")?;
            for index in kv_indices.iter().copied().skip(1) {
                let other = self.workers[index]
                    .1
                    .kv_cache
                    .as_ref()
                    .context("executor routes KV work to a pool without a KV cache")?;
                kv_cache = kv_cache
                    .merge(other)
                    .context("executor KV pools expose incompatible cache layouts")?;
            }
            merged.kv_cache = Some(kv_cache);
        } else {
            merged.kv_cache = None;
        }

        // A call is supported when any pool serves it. Each pool contributes
        // its own queue depth, counted as at least one. Unless noted at the
        // field, the remaining limits take the smallest nonzero value, and are
        // zero only when every pool reports zero.
        merged.supported_calls = CallKind::ALL
            .into_iter()
            .filter(|variant| routed(*variant).is_some())
            .collect();
        merged.queue_depth = self
            .workers
            .iter()
            .map(|(_, info)| info.queue_depth.max(1))
            .try_fold(0u32, |total, depth| total.checked_add(depth))
            .context("worker capacity exceeds the engine window")?;
        merged.max_batch_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_batch_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_batch_tokens = self
            .workers
            .iter()
            .map(|(_, info)| info.max_batch_tokens)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        // A prefill call may reach any pool, so the narrowest prefill graph
        // capacity binds; a pool without one leaves the batch bound.
        merged.max_prefill_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_prefill_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0)
            .min(merged.max_batch_calls);
        // Likewise, the narrowest decode graph capacity binds every decode
        // call.
        merged.max_decode_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_decode_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0)
            .min(merged.max_batch_calls);
        merged.request_slots = if merged.media_components.contains_key(&MediaCall::Muxing) {
            // A replicated media route owns an independent request-row bank in
            // every WorkerGroup. End-to-end concurrency is the narrowest sum
            // of replica capacities among the graph's components. Each
            // component is keyed to the first call it serves; its replicas are
            // the pools that support that call and bind the component.
            let mut components = BTreeMap::new();
            for (call, component) in &merged.media_components {
                components.entry(component.as_str()).or_insert(*call);
            }
            components
                .into_iter()
                .map(|(component, call)| {
                    self.workers
                        .iter()
                        .filter(|(_, info)| {
                            info.supported_calls.contains(&CallKind::Media(call))
                                && info
                                    .components
                                    .iter()
                                    .any(|binding| binding.name == component)
                        })
                        .map(|(_, info)| info.request_slots)
                        .try_fold(0u32, |total, slots| total.checked_add(slots))
                        .context("media replica request capacity exceeds protocol range")
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .min()
                .unwrap_or(0)
        } else {
            self.workers
                .iter()
                .map(|(_, info)| info.request_slots)
                .filter(|limit| *limit > 0)
                .min()
                .unwrap_or(0)
        };
        merged.max_unresolved_calls = self
            .workers
            .iter()
            .map(|(_, info)| info.max_unresolved_calls)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        // Latent page geometry is the denoising pool's, zero without one.
        let flow = routed(CallKind::Media(MediaCall::Denoising));
        merged.latent_page_units = flow.map_or(0, |info| info.latent_page_units);
        merged.latent_pages = flow.map_or(0, |info| info.latent_pages);
        merged.buffer_pool_bytes = self
            .workers
            .iter()
            .map(|(_, info)| info.buffer_pool_bytes)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);
        merged.encoder_cache_entries = self
            .workers
            .iter()
            .map(|(_, info)| info.encoder_cache_entries)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);
        merged.encoder_entry_bytes = self
            .workers
            .iter()
            .map(|(_, info)| info.encoder_entry_bytes)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);

        merged.validate()?;
        check_video_tasks(&merged)?;
        Ok(merged)
    }
}

/// Checks that the deployment places a component for every call of every
/// task its video denoiser serves.
///
/// A conditioned task reads its media on a host component and encodes its
/// conditions before denoising, so a deployment serving one must place the
/// media reader and both condition encoders as well as the generation calls.
fn check_video_tasks(info: &WorkerInfo) -> anyhow::Result<()> {
    let Some(denoiser) = info.video_denoiser.as_ref() else {
        return Ok(());
    };
    for name in &denoiser.tasks {
        let Some(task) = uniserve_core::VideoTask::from_name(name) else {
            anyhow::bail!("the video denoiser serves unknown task {name:?}");
        };
        let missing = crate::scheduler::graph::task_calls(task)
            .into_iter()
            .filter(|call| !info.media_components.contains_key(call))
            .map(|call| format!("{call:?}"))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing.is_empty(),
            "the deployment serves {name} but places no component for {}",
            missing.join(", ")
        );
    }
    Ok(())
}

/// One call result returned from an executor-owned batch.
#[derive(Debug, Clone)]
pub struct CallResult {
    /// Validated completion values; media storage is carried by `media` below.
    pub output: uniserve_worker_ipc::RequestOutput,
    /// Claimed immutable output storage, or its request-local acquisition error.
    pub media: Result<Option<Arc<uniserve_core::SharedMedia>>, String>,
}

/// A decoded physical result that owns media before routing or rank validation.
#[derive(Debug)]
pub struct WorkerResult {
    pub batch_id: u64,
    /// Every participating rank has retired this batch, including command-only ranks.
    pub done: bool,
    pub results: Vec<CallResult>,
    /// Tensor publications remain owned by the executor's transfer consumers.
    pub products: Vec<TensorPublication>,
    /// Worker-reported execution duration in microseconds, when reported.
    pub worker_exec_us: Option<u64>,
    /// Model-forward statistics, when reported.
    pub forward_stats: Option<uniserve_worker_ipc::ForwardStats>,
}

impl WorkerResult {
    /// Claims all media before any fallible correlation or aggregation step.
    /// Acquisition failure belongs to the call; independent results remain usable.
    ///
    /// The returned value always sets `done`; `WorkerGroup::try_join`, which
    /// joins rank reports, decides `done` for the joined result itself.
    pub(crate) fn receive(report: BatchOutput) -> Self {
        let results = report
            .completions
            .into_iter()
            .map(|mut output| {
                let media = output
                    .media_output
                    .take()
                    .map(|media| {
                        let uniserve_worker_ipc::ArtifactHandle::PosixShm { name } = media.handle;
                        // SAFETY: publication transfers immutable storage after the writer closes.
                        // The mapping owns the bytes even if this result is rejected downstream.
                        unsafe { uniserve_core::SharedMedia::open(&name, media.bytes) }
                            .map(Arc::new)
                    })
                    .transpose();
                CallResult { output, media }
            })
            .collect();
        Self {
            batch_id: report.batch_id,
            done: true,
            results,
            products: report.products,
            worker_exec_us: report.worker_exec_us,
            forward_stats: report.forward_stats,
        }
    }
}

/// Distinguishes applied control from physical retirement and unacknowledged failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    /// Every target pool acknowledged the command.
    Applied,
    /// The targeted request was tainted by a worker failure, so the command
    /// settles without an acknowledgment. The scheduler treats it as settled,
    /// like `Applied`.
    Retired,
    /// The command was not acknowledged and its physical ownership remains;
    /// the scheduler queues a failed `Finish` or `Free` again.
    Failed,
}

/// Terminal outcome of one logical lifecycle command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    /// Position among lifecycle commands, excluding request admissions.
    pub command_index: u32,
    /// Request targeted by the command.
    pub request_key: RequestKey,
    /// A failed command retains physical ownership until a subsequent release succeeds.
    pub outcome: CommandOutcome,
}

/// One independently ready subset of a logical batch.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Logical batch identity assigned at submission.
    pub batch_id: u64,
    /// Call completions this join reports as ready.
    pub results: Vec<CallResult>,
    /// Receipts for the batch's lifecycle commands; empty unless `done`.
    pub command_results: Vec<CommandResult>,
    /// Whether this join completes the batch.
    pub done: bool,
    /// Per-worker execution durations in microseconds.
    pub worker_exec_us: Vec<u64>,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<uniserve_worker_ipc::ForwardStats>,
}

/// Converts one physical result into the scheduler-facing [`BatchResult`].
///
/// `done` comes from the caller, not from `report.done`. A terminal result
/// carries one `Applied` receipt per lifecycle command in `commands`,
/// numbered with `Start` admissions skipped, so `commands` may be given with
/// or without them; callers that track other outcomes overwrite the receipts.
/// A partial result carries no receipts. `report.products` is not carried
/// into the result: publications stay with the executor, and
/// `WorkerExecutor` records them before converting.
pub(crate) fn logical_result(
    report: WorkerResult,
    done: bool,
    commands: &[BatchCommand],
) -> BatchResult {
    let mut worker_exec_us = Vec::new();
    let mut forward_stats = Vec::new();
    if let Some(value) = report.worker_exec_us {
        worker_exec_us.push(value);
    }
    if let Some(value) = report.forward_stats {
        forward_stats.push(value);
    }
    BatchResult {
        batch_id: report.batch_id,
        results: report.results,
        command_results: if done {
            commands
                .iter()
                .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                .enumerate()
                .map(|(index, command)| CommandResult {
                    command_index: index as u32,
                    request_key: command.request_key(),
                    outcome: CommandOutcome::Applied,
                })
                .collect()
        } else {
            Vec::new()
        },
        done,
        worker_exec_us,
        forward_stats,
    }
}

/// Dynamic error returned while polling or administering an executor.
pub type ExecutorError = anyhow::Error;

/// Backpressure and terminal failures returned by logical batch submission.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorSubmitError {
    #[error("executor queue is full")]
    /// Returns ownership of a batch rejected by bounded queue capacity.
    WouldBlock(ExecutionBatch),
    #[error(transparent)]
    /// Reports a terminal submission failure.
    Failed(#[from] anyhow::Error),
}

/// Stable identity of one explicitly configured physical worker pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(pub String);

impl WorkerId {
    /// Validates and constructs a stable pool identifier.
    ///
    /// An identifier is nonempty ASCII alphanumerics, `-`, and `_`. Only this
    /// constructor checks that; the public field admits any string.
    pub fn new(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        if value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            anyhow::bail!("invalid worker id {value:?}");
        }
        Ok(Self(value))
    }
}

impl std::fmt::Display for WorkerId {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Transport available for product movement between worker pools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferBackend {
    /// Keeps products within one worker process.
    #[default]
    Local,
    /// Publishes products through POSIX shared storage.
    Shm,
    /// Publishes device products through CUDA VMM handles.
    CudaVmm,
    /// Carries a host product's bytes on the rank channel's data path.
    ///
    /// Shared storage reaches only one host. A product on this edge travels in
    /// the producing rank's result and in the consuming rank's batch, so it
    /// crosses hosts wherever the rank channel does.
    Channel,
}

impl TransferBackend {
    /// Returns the stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Shm => "shm",
            Self::CudaVmm => "cuda_vmm",
            Self::Channel => "channel",
        }
    }
}

impl std::fmt::Display for TransferBackend {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TransferBackend {
    type Err = TransferConfigError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "local" => Ok(Self::Local),
            "shm" => Ok(Self::Shm),
            "cuda_vmm" => Ok(Self::CudaVmm),
            "channel" => Ok(Self::Channel),
            _ => Err(TransferConfigError::message(format!(
                "unsupported transfer backend {value:?}"
            ))),
        }
    }
}

/// One explicit directed transfer edge between configured pool identities.
///
/// Products move by one device mechanism and one host mechanism, and a
/// product's location decides which carries it: a device product travels on
/// the edge's device mechanism, a host product on its host mechanism. An edge
/// may name only one of the two, in which case products of the other
/// location have no way across it: `TransferConfig::bind_inputs` refuses them
/// where they are bound, and `WorkerExecutor` fails the requests reading them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferEdge {
    /// Pool that produces the transferred product.
    pub source_worker: WorkerId,
    /// One source member, or all members when omitted.
    pub source_rank: Option<u32>,
    /// Pool that consumes the transferred product.
    pub destination_worker: WorkerId,
    /// One consumer member, or all members when omitted.
    pub destination_rank: Option<u32>,
    /// Mechanism carrying device products across this edge.
    pub device: Option<TransferBackend>,
    /// Mechanism carrying host products across this edge.
    pub host: Option<TransferBackend>,
}

impl TransferEdge {
    /// The mechanisms this edge carries products on, device first.
    pub fn mechanisms(&self) -> impl Iterator<Item = TransferBackend> + '_ {
        self.device.into_iter().chain(self.host)
    }

    /// Whether this edge carries products on `backend`.
    pub fn carries(&self, backend: TransferBackend) -> bool {
        self.device == Some(backend) || self.host == Some(backend)
    }
}

/// Per-edge data-plane transfer selection (`--transfer`) and the placement it
/// is resolved against.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Explicit directed transfer edges.
    pub edges: Vec<TransferEdge>,
    /// Rank count of each worker an edge may name.
    ///
    /// An edge that omits a rank names every rank of its worker, so counting
    /// a product's consumers needs the membership the placement gave each.
    #[serde(default)]
    pub worker_ranks: std::collections::BTreeMap<String, u32>,
    /// Host of each rank of each worker an edge may name.
    ///
    /// A publication's readiness mechanism depends on where it is read: an
    /// interprocess event reaches another process on this host and no further.
    /// Only the placement knows where a consumer runs.
    #[serde(default)]
    pub worker_hosts: std::collections::BTreeMap<String, Vec<String>>,
}

impl TransferConfig {
    /// Records the placement's rank counts and hosts, and binds a default edge
    /// for every ordered pair of ranks, within and across workers, that no
    /// configured edge covers.
    ///
    /// Each rank is a separate process. Its self-edge uses local storage for
    /// both locations. Between two ranks, device products move over CUDA VMM
    /// where both hold a CUDA device, which reaches another host where both
    /// devices export a fabric handle; host products move over shared storage
    /// on one host and over the rank channel across hosts, because a
    /// shared-storage segment is named in one host's namespace. Explicit
    /// bindings take precedence. `WorkerExecutor::try_new` validates each
    /// edge against the initialized endpoints and backends before the
    /// executor accepts work, and refuses by name an edge that would have to
    /// cross hosts without fabric handles.
    ///
    /// # Errors
    ///
    /// Fails when `WorkerConfig::validate_all` rejects `workers`.
    pub fn with_worker_defaults(mut self, workers: &[crate::WorkerConfig]) -> anyhow::Result<Self> {
        crate::WorkerConfig::validate_all(workers)?;
        for worker in workers {
            self.worker_ranks
                .insert(worker.id.0.clone(), worker.ranks.len() as u32);
            self.worker_hosts.insert(
                worker.id.0.clone(),
                worker.ranks.iter().map(|rank| rank.node.clone()).collect(),
            );
        }
        // Products flow between the components of one worker and between
        // workers, a model worker's decoded media units to a host worker's
        // encoders, so every ordered pair of ranks gets an edge derived from
        // its coordinates unless the configuration named one.
        for worker in workers {
            for peer in workers {
                for (source_rank, source) in worker.ranks.iter().enumerate() {
                    for (destination_rank, destination) in peer.ranks.iter().enumerate() {
                        let source_rank = source_rank as u32;
                        let destination_rank = destination_rank as u32;
                        if self.edges.iter().any(|edge| {
                            edge.source_worker == worker.id
                                && edge.destination_worker == peer.id
                                && edge.source_rank.is_none_or(|rank| rank == source_rank)
                                && edge
                                    .destination_rank
                                    .is_none_or(|rank| rank == destination_rank)
                        }) {
                            continue;
                        }
                        let (device, host) =
                            if worker.id == peer.id && source_rank == destination_rank {
                                (Some(TransferBackend::Local), Some(TransferBackend::Local))
                            } else {
                                // A fabric handle reaches another host; where the
                                // devices export a process descriptor instead, the
                                // physical edge check refuses this edge by name.
                                let device = (source.device.starts_with("cuda:")
                                    && destination.device.starts_with("cuda:"))
                                .then_some(TransferBackend::CudaVmm);
                                // A shared-storage segment is named in one host's
                                // namespace, so a host product that leaves its host
                                // travels on the rank channel's data path.
                                let host = if source.node == destination.node {
                                    TransferBackend::Shm
                                } else {
                                    TransferBackend::Channel
                                };
                                (device, Some(host))
                            };
                        self.edges.push(TransferEdge {
                            source_worker: worker.id.clone(),
                            source_rank: Some(source_rank),
                            destination_worker: peer.id.clone(),
                            destination_rank: Some(destination_rank),
                            device,
                            host,
                        });
                    }
                }
            }
        }
        Ok(self)
    }

    /// Resolve only mechanisms incident to one physical rank, with local access
    /// available for products that stay in its own address space.
    ///
    /// Returns `(transfer, publish)`: `transfer` is every mechanism on an edge
    /// the rank produces or consumes on, plus `Local`; `publish` is the
    /// mechanisms on edges the rank produces on, or only `Local` when it
    /// produces on none. The launch descriptor carries them as
    /// `transfer_backends` and `publish_backends`.
    pub fn rank_backends(
        &self,
        worker: &str,
        rank: u32,
    ) -> (
        std::collections::BTreeSet<TransferBackend>,
        std::collections::BTreeSet<TransferBackend>,
    ) {
        let mut backends = std::collections::BTreeSet::from([TransferBackend::Local]);
        let mut publications = std::collections::BTreeSet::new();
        for edge in &self.edges {
            if edge.source_worker.0 == worker
                && edge.source_rank.is_none_or(|source| source == rank)
            {
                backends.extend(edge.mechanisms());
                publications.extend(edge.mechanisms());
            }
            if edge.destination_worker.0 == worker
                && edge
                    .destination_rank
                    .is_none_or(|destination| destination == rank)
            {
                backends.extend(edge.mechanisms());
            }
        }
        if publications.is_empty() {
            publications.insert(TransferBackend::Local);
        }
        (backends, publications)
    }

    /// Returns the instance-wide acknowledgment slot of one rank.
    ///
    /// Each published chunk or segment carries one acknowledgment word per
    /// instance rank. A consumer claims the word at its own slot when it
    /// begins reading and acknowledges it once its reads are done, so the
    /// producer can tell a consumer still reading from one that has finished
    /// or never began. Slots are dense across the instance. A worker absent
    /// from `worker_ranks` yields a slot past every placed worker's run.
    pub fn acknowledgment_slot(&self, worker: &str, rank: u32) -> u32 {
        // Workers are keyed in a BTreeMap, so their order is the same in every
        // process that derives a slot. Each worker owns a contiguous run of
        // slots, and a rank takes its offset within that run.
        let mut base = 0;
        for (name, ranks) in &self.worker_ranks {
            if name == worker {
                return base + rank;
            }
            base += ranks;
        }
        base + rank
    }

    /// Acknowledgment slots of every rank placed on the same host as one
    /// rank, across workers, this rank's own included.
    ///
    /// A host product is published over the mechanism that reaches the
    /// consumers its producing call names: shared storage for a consumer on
    /// the host, the rank channel for one elsewhere. The producing rank is
    /// told which slots share its host so it can tell the two apart; an
    /// unplaced rank shares a host with no one.
    pub fn host_slots(&self, worker: &str, rank: u32) -> Vec<u32> {
        let Some(host) = self
            .worker_hosts
            .get(worker)
            .and_then(|hosts| hosts.get(rank as usize))
        else {
            return Vec::new();
        };
        let mut slots = Vec::new();
        for (name, hosts) in &self.worker_hosts {
            for (member, placed) in hosts.iter().enumerate() {
                if placed == host {
                    slots.push(self.acknowledgment_slot(name, member as u32));
                }
            }
        }
        slots
    }

    /// Whether any rank that reads this rank's products is on another host.
    ///
    /// The consumers are the destinations of the edges leaving this rank,
    /// excluding the rank itself. The answer is `true` when this rank or any
    /// consumer has no recorded host, because a shared host cannot then be
    /// established. A rank cannot derive this itself: the mapping lives in the
    /// transfer edges and the placement, which only the head holds.
    ///
    /// Readiness is a producer synchronize only where it has to be. Within a
    /// host an interprocess event carries it at no cost to the producing
    /// stream, and stalling the producer for a consumer that could have waited
    /// on the device is a bubble the placement does not require.
    pub fn products_cross_hosts(&self, worker: &str, rank: u32) -> bool {
        let host = |name: &str, rank: u32| -> Option<&String> {
            self.worker_hosts
                .get(name)
                .and_then(|hosts| hosts.get(rank as usize))
        };
        let Some(source_host) = host(worker, rank) else {
            // An unplaced producer cannot be shown to stay on one host.
            return true;
        };
        for edge in &self.edges {
            if edge.source_worker.0 != worker
                || !edge.source_rank.is_none_or(|source| source == rank)
            {
                continue;
            }
            let destination = &edge.destination_worker.0;
            let members = || match edge.destination_rank {
                Some(consumer) => consumer..consumer + 1,
                None => 0..self.worker_ranks.get(destination).copied().unwrap_or(0),
            };
            for consumer in members() {
                if destination == worker && consumer == rank {
                    continue;
                }
                match host(destination, consumer) {
                    Some(consumer_host) if consumer_host == source_host => {}
                    // A consumer elsewhere, or one the placement does not
                    // name, cannot be assumed to share this host.
                    _ => return true,
                }
            }
        }
        false
    }

    /// Acknowledgment slots of the ranks that read this rank's products.
    ///
    /// A retired product's chunk or segment is released only once no named
    /// consumer's word shows it still reading, so the producer needs the
    /// consumers' identities and not merely their number. The consumers are the
    /// destinations of the edges leaving this rank, fixed when the placement
    /// is; an edge that omits the destination rank names every rank of that
    /// worker. A producing rank cannot derive them: it knows which component
    /// it belongs to, not which component reads what it publishes, and a
    /// product is consumed in a later batch than the one that produced it.
    /// `WorkerGroup::consumer_slots` uses this for calls outside the video
    /// graph and states the slots on each producing call.
    ///
    /// The result is sorted and excludes this rank's own slot.
    pub fn product_consumers(&self, worker: &str, rank: u32) -> Vec<u32> {
        let mut slots = std::collections::BTreeSet::new();
        for edge in &self.edges {
            if edge.source_worker.0 != worker
                || !edge.source_rank.is_none_or(|source| source == rank)
            {
                continue;
            }
            // A self-edge publishes to this rank's own address space, which
            // needs no acknowledgment from another process.
            if edge.destination_worker.0 == worker && edge.destination_rank == Some(rank) {
                continue;
            }
            let destination = &edge.destination_worker.0;
            match edge.destination_rank {
                Some(consumer) => {
                    slots.insert(self.acknowledgment_slot(destination, consumer));
                }
                // An edge without a rank names every rank of its worker.
                None => {
                    let members = self.worker_ranks.get(destination).copied().unwrap_or(0);
                    for consumer in 0..members {
                        if destination == worker && consumer == rank {
                            continue;
                        }
                        slots.insert(self.acknowledgment_slot(destination, consumer));
                    }
                }
            }
        }
        slots.into_iter().collect()
    }

    /// Select explicitly bound locations for a rank before admitting device work.
    ///
    /// Each input tensor keeps only the published locations that reach
    /// `destination`: a location survives when the first edge matching its
    /// source and `destination` carries the mechanism it was published on, or,
    /// with no matching edge, when it is a local location on `destination`
    /// itself. `WorkerGroup` calls this for each rank's projected batch.
    ///
    /// # Errors
    ///
    /// Fails with every product that has a tensor left with no location. An
    /// edge that names only one of the two mechanisms refuses products of the
    /// other location here, and so does a product published over none of the
    /// mechanisms its edge carries. Filtering is not rolled back: every
    /// tensor keeps its filtered locations.
    pub(crate) fn bind_inputs(
        &self,
        products: &mut [uniserve_worker_ipc::TensorPublication],
        kv_inputs: &mut [uniserve_worker_ipc::KvTransfer],
        destination: &uniserve_worker_ipc::WorkerEndpoint,
    ) -> Result<(), UnroutableInputs> {
        let mut unroutable = Vec::new();
        for payload in products.iter_mut() {
            let buffer = payload.product.buffer_id();
            let mut bound = true;
            for tensor in payload.value.tensors_mut() {
                bound &= self.bind_locations(tensor, destination);
            }
            if !bound {
                unroutable.push(buffer);
            }
        }
        for publication in kv_inputs.iter_mut() {
            let mut bound = true;
            for tensor in publication.tensors_mut() {
                bound &= self.bind_locations(tensor, destination);
            }
            if !bound {
                unroutable.push(publication.source);
            }
        }
        if unroutable.is_empty() {
            return Ok(());
        }
        Err(UnroutableInputs {
            destination: format!("{}:{}", destination.worker_id, destination.rank),
            buffers: unroutable,
        })
    }

    /// Keeps the locations of one tensor that reach `destination`, as
    /// `bind_inputs` selects them, and reports whether any remain.
    fn bind_locations(
        &self,
        tensor: &mut uniserve_worker_ipc::TensorTransfer,
        destination: &uniserve_worker_ipc::WorkerEndpoint,
    ) -> bool {
        use uniserve_worker_ipc::TransferTransport;

        // A product's location chose the mechanism it was published on, so a
        // location is kept where the edge carries that mechanism for either
        // location.
        tensor.locations.retain(|location| {
            let published = match &location.transport {
                TransferTransport::Local { .. } => TransferBackend::Local,
                TransferTransport::PosixShm { .. } => TransferBackend::Shm,
                TransferTransport::CudaVmm { .. } => TransferBackend::CudaVmm,
                TransferTransport::Channel { .. } => TransferBackend::Channel,
            };
            self.edges
                .iter()
                .find(|edge| {
                    edge.source_worker.0 == location.source.worker_id
                        && edge
                            .source_rank
                            .is_none_or(|rank| rank == location.source.rank)
                        && edge.destination_worker.0 == destination.worker_id
                        && edge
                            .destination_rank
                            .is_none_or(|rank| rank == destination.rank)
                })
                .map_or_else(
                    || &location.source == destination && published == TransferBackend::Local,
                    |edge| edge.carries(published),
                )
        });
        !tensor.locations.is_empty()
    }

    /// Parses `source[:rank]->destination[:rank]=mechanisms` directed bindings.
    ///
    /// `mechanisms` names the edge's device mechanism, its host mechanism, or
    /// both joined by `+` with the device mechanism first, as in
    /// `cuda_vmm+shm`. `local` serves both locations. Entries are separated by
    /// commas, and an omitted rank covers every rank of that worker.
    ///
    /// The result holds no placement: `with_worker_defaults` fills
    /// `worker_ranks` and `worker_hosts` and binds the edges left unnamed.
    ///
    /// # Errors
    ///
    /// Fails on a malformed entry, an invalid worker id or rank, an unknown
    /// or conflicting mechanism, or an edge that overlaps an earlier one.
    pub fn parse(s: &str) -> Result<Self, TransferConfigError> {
        let mut edges = Vec::new();

        let endpoint = |text: &str| -> Result<(WorkerId, Option<u32>), TransferConfigError> {
            let (worker, rank) = match text.trim().split_once(':') {
                Some((worker, rank)) => (
                    worker,
                    Some(rank.parse::<u32>().map_err(|_| {
                        TransferConfigError::message("transfer rank must be a nonnegative integer")
                    })?),
                ),
                None => (text.trim(), None),
            };
            let worker = WorkerId::new(worker).map_err(|error| {
                TransferConfigError::message(format!("invalid transfer worker: {error}"))
            })?;
            Ok((worker, rank))
        };

        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            // Parse and validate both endpoint identities before accepting the
            // transport so errors remain attributable to one edge.
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                TransferConfigError::message(format!(
                    "transfer entry {entry:?} must be edge=backend"
                ))
            })?;
            let (src, dst) = edge.split_once("->").ok_or_else(|| {
                TransferConfigError::message(format!("transfer edge {edge:?} must be src->dst"))
            })?;
            let (src, source_rank) = endpoint(src)?;
            let (dst, destination_rank) = endpoint(dst)?;
            let (device, host) = Self::parse_mechanisms(backend.trim())?;

            // Two edges between the same workers overlap when their ranks
            // can name one pair; an omitted rank overlaps every rank.
            if edges.iter().any(|existing: &TransferEdge| {
                existing.source_worker == src
                    && existing.destination_worker == dst
                    && (existing.source_rank.is_none()
                        || source_rank.is_none()
                        || existing.source_rank == source_rank)
                    && (existing.destination_rank.is_none()
                        || destination_rank.is_none()
                        || existing.destination_rank == destination_rank)
            }) {
                return Err(TransferConfigError::message(format!(
                    "duplicate transfer edge {edge:?}"
                )));
            }

            edges.push(TransferEdge {
                source_worker: src,
                source_rank,
                destination_worker: dst,
                destination_rank,
                device,
                host,
            });
        }

        Ok(Self {
            edges,
            worker_ranks: std::collections::BTreeMap::new(),
            worker_hosts: std::collections::BTreeMap::new(),
        })
    }

    /// Resolves an edge's mechanism text into its device and host mechanisms.
    fn parse_mechanisms(
        text: &str,
    ) -> Result<(Option<TransferBackend>, Option<TransferBackend>), TransferConfigError> {
        let backends = text
            .split('+')
            .map(|name| TransferBackend::from_str(name.trim()))
            .collect::<Result<Vec<_>, _>>()?;
        if backends.contains(&TransferBackend::Local) {
            // Local storage serves both locations and combines with nothing.
            if backends.len() > 1 {
                return Err(TransferConfigError::message(format!(
                    "transfer mechanisms {text:?} combine local with another mechanism"
                )));
            }
            return Ok((Some(TransferBackend::Local), Some(TransferBackend::Local)));
        }
        let mut device = None;
        let mut host = None;
        for backend in backends {
            let slot = match backend {
                TransferBackend::CudaVmm => &mut device,
                TransferBackend::Shm | TransferBackend::Channel => &mut host,
                TransferBackend::Local => unreachable!("local mechanisms return above"),
            };
            if slot.replace(backend).is_some() {
                return Err(TransferConfigError::message(format!(
                    "transfer mechanisms {text:?} name two mechanisms for one location"
                )));
            }
        }
        Ok((device, host))
    }
}

impl FromStr for TransferConfig {
    type Err = TransferConfigError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// Transferred inputs that a consuming rank cannot read.
///
/// Every published location of these products travels on a mechanism that no
/// configured edge to the consumer carries, so the requests reading them can
/// never run there. `WorkerExecutor` fails those requests alone: the refusal
/// says nothing about the rest of the batch or the health of either instance.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("input products have no location on a configured transfer edge to {destination}")]
pub(crate) struct UnroutableInputs {
    /// The consuming rank, as `worker:rank`.
    pub(crate) destination: String,
    /// The products left without a location the consumer can read.
    pub(crate) buffers: Vec<uniserve_worker_ipc::BufferId>,
}

/// Error returned for an invalid transfer-transport mapping.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TransferConfigError(String);

impl TransferConfigError {
    /// Returns the human-readable error message.
    fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// A worker-reported execution error classified for scheduler failure policy.
///
/// The typed taxonomy and execution context cross the IPC together so failure
/// policy and diagnostics use the same call identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker execute error: {message}")]
pub struct WorkerExecError {
    /// Batch whose response carried this error. Composite executors use it to
    /// join the same terminal outcome across ranks before returning the
    /// failure to the scheduler.
    pub batch_id: Option<u64>,
    /// Whether the error invalidates the worker process or executor.
    pub fatal: bool,
    /// Stable worker-defined error code, when classified.
    pub code: Option<String>,
    /// Human-readable failure description.
    pub message: String,
    /// Execution phase in which the failure occurred, when known.
    pub phase: Option<String>,
    /// Call kind of the failed batch or numerical call, when the failure
    /// belongs to a batch with calls (see `WorkerResponseError::route`).
    pub route: Option<String>,
    /// Calls affected by the worker failure.
    pub calls: Vec<uniserve_worker_ipc::ErrorCallIdentity>,
}

/// Terminal work abandoned by one Worker, with the exact invalidated ownership.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkerFailure {
    /// Logical instance reporting the failure.
    pub worker_id: WorkerId,
    /// Invalidated incarnations; empty when the instance retains its allocations.
    pub endpoints: Vec<uniserve_worker_ipc::WorkerEndpoint>,
    /// Request epochs affected by the failed work or invalidated allocations.
    pub requests: Vec<RequestKey>,
    /// Accepted logical calls that will no longer produce a device completion.
    pub retired: Vec<(u64, RequestKey, CallId)>,
    /// Buffers whose published locations no longer cover their complete logical value.
    pub buffers: Vec<uniserve_worker_ipc::BufferId>,
    /// Classified execution failure, if the ranks returned one before retirement.
    pub execution: Option<WorkerExecError>,
    /// Human-readable failure description.
    pub message: String,
}

/// The asynchronous, pipelined boundary the scheduler drives.
///
/// One submitted batch may come back from `poll` as several
/// [`BatchResult`]s: each carries completions not reported by an earlier one,
/// and the last has `done` set and carries the lifecycle command receipts.
/// A failure may instead surface as an error from `poll`.
pub trait Executor: Send {
    /// Returns the executor's concrete pool capabilities.
    fn info(&self) -> &ExecutorInfo;
    /// Whether `worker`'s complete instance is ready to accept execution.
    fn is_ready(&self, worker: &WorkerId) -> bool;
    /// Whether this instance has an unclaimed destination submission slot.
    fn has_capacity(&self, worker: &WorkerId) -> bool;
    /// Whether every physical owner of a lifecycle command can accept its submission.
    fn command_has_capacity(&self, command: &BatchCommand) -> bool;
    /// Submits one logical batch without blocking for capacity.
    fn submit(&mut self, batch: ExecutionBatch) -> Result<(), ExecutorSubmitError>;
    /// Waits up to `timeout` for one partial or terminal batch result.
    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, ExecutorError>;
    /// Closes the executor and its physical workers.
    fn close(&mut self) -> Result<(), ExecutorError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media_info(routes: &[(MediaCall, &str)], steps: u32) -> WorkerInfo {
        WorkerInfo {
            media_components: routes
                .iter()
                .map(|(call, component)| (*call, (*component).to_owned()))
                .collect(),
            supported_calls: routes
                .iter()
                .map(|(call, _)| CallKind::Media(*call))
                .collect(),
            num_inference_steps: steps,
            ..WorkerInfo::default()
        }
    }

    fn media_pool(
        id: &str,
        routes: &[(MediaCall, &str)],
        steps: u32,
        request_slots: u32,
    ) -> (WorkerId, WorkerInfo) {
        let mut info = media_info(routes, steps);
        info.endpoint.worker_id = id.to_owned();
        info.request_slots = request_slots;
        info.components = routes
            .iter()
            .map(|(_, component)| *component)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|name| uniserve_worker_ipc::ComponentInfo {
                name: name.to_owned(),
                config: ComponentConfig::parallel(vec![0], ParallelConfig::default()),
                outputs: Vec::new(),
            })
            .collect();
        (WorkerId(id.to_owned()), info)
    }

    fn media_replica(id: &str, request_slots: u32) -> (WorkerId, WorkerInfo) {
        media_pool(
            id,
            &[
                (MediaCall::TextEncoding, "text_encoder"),
                (MediaCall::LatentPreparation, "denoiser"),
                (MediaCall::Denoising, "denoiser"),
                (MediaCall::VideoDecoding, "video_decoder"),
                (MediaCall::VideoEncoding, "video_codec"),
                (MediaCall::AudioDecoding, "audio_decoder"),
                (MediaCall::AudioEncoding, "muxer"),
                (MediaCall::Muxing, "muxer"),
            ],
            4,
            request_slots,
        )
    }

    #[test]
    fn replicated_media_routes_add_independent_request_capacity() -> anyhow::Result<()> {
        // Each pool serves the whole video graph, so every component's
        // replica capacity is the sum over both pools.
        let info = ExecutorInfo::from_workers(vec![
            media_replica("replica-0", 2),
            media_replica("replica-1", 3),
        ])?;

        assert_eq!(info.runtime_info()?.request_slots, 5);
        Ok(())
    }

    #[test]
    fn shared_media_stages_bound_aggregate_replica_capacity() -> anyhow::Result<()> {
        let flow = [
            (MediaCall::LatentPreparation, "denoiser"),
            (MediaCall::Denoising, "denoiser"),
            (MediaCall::VideoDecoding, "video_decoder"),
            (MediaCall::AudioDecoding, "audio_decoder"),
        ];
        let info = ExecutorInfo::from_workers(vec![
            media_pool("text", &[(MediaCall::TextEncoding, "text_encoder")], 0, 8),
            media_pool("flow-0", &flow, 4, 2),
            media_pool("flow-1", &flow, 4, 2),
            media_pool(
                "host",
                &[
                    (MediaCall::VideoEncoding, "video_codec"),
                    (MediaCall::AudioEncoding, "muxer"),
                    (MediaCall::Muxing, "muxer"),
                ],
                0,
                16,
            ),
        ])?;

        // The two flow banks contribute four routes. The wider shared text
        // and host stages must not multiply that end-to-end capacity.
        assert_eq!(info.runtime_info()?.request_slots, 4);
        Ok(())
    }

    #[test]
    fn media_routing_is_the_union_over_workers() {
        // A model worker decodes and a host worker encodes and muxes; the
        // deployment's routing names every video call once.
        let model = media_info(
            &[
                (MediaCall::TextEncoding, "text_encoder"),
                (MediaCall::LatentPreparation, "denoiser"),
                (MediaCall::Denoising, "denoiser"),
                (MediaCall::VideoDecoding, "video_decoder"),
                (MediaCall::AudioDecoding, "audio_decoder"),
            ],
            4,
        );
        let host = media_info(
            &[
                (MediaCall::VideoEncoding, "video_codec"),
                (MediaCall::AudioEncoding, "muxer"),
                (MediaCall::Muxing, "muxer"),
            ],
            0,
        );
        let info = ExecutorInfo {
            workers: vec![
                (WorkerId("model".to_owned()), model),
                (WorkerId("host".to_owned()), host),
            ],
            video_codecs: BTreeMap::new(),
        };
        let routing = info.media_routing().expect("the union is complete");
        assert_eq!(routing.len(), 8);
        assert_eq!(routing[&MediaCall::VideoEncoding], "video_codec");
        assert_eq!(routing[&MediaCall::VideoDecoding], "video_decoder");
    }

    /// A deployment whose denoiser serves a conditioned task must place the
    /// media reader and both condition encoders; one serving `t2va` alone
    /// needs only the generation calls.
    #[test]
    fn a_conditioned_task_requires_its_condition_components() {
        let generation = [
            (MediaCall::TextEncoding, "text_encoder"),
            (MediaCall::LatentPreparation, "denoiser"),
            (MediaCall::Denoising, "denoiser"),
            (MediaCall::VideoDecoding, "video_decoder"),
            (MediaCall::AudioDecoding, "audio_decoder"),
            (MediaCall::VideoEncoding, "video_codec"),
            (MediaCall::AudioEncoding, "muxer"),
            (MediaCall::Muxing, "muxer"),
        ];
        let serving = |tasks: &[&str], routes: &[(MediaCall, &str)]| {
            let mut info = media_info(routes, 4);
            info.video_denoiser = Some(VideoDenoiserInfo {
                tasks: tasks.iter().map(|task| (*task).to_owned()).collect(),
                schedule_points: 5,
                video_shift: 12.0,
                audio_shift: 3.0,
                canvases: Vec::new(),
                max_sequence_rows: None,
                condition_tiles: None,
            });
            ExecutorInfo {
                workers: vec![(WorkerId("model".to_owned()), info)],
                video_codecs: BTreeMap::new(),
            }
        };

        assert!(serving(&["t2va"], &generation).runtime_info().is_ok());
        let message = serving(&["t2va", "fl2va"], &generation)
            .runtime_info()
            .expect_err("fl2va reads and encodes conditions")
            .to_string();
        assert!(
            message.contains("fl2va") && message.contains("MediaReading"),
            "{message}"
        );

        let conditioned = [
            (MediaCall::MediaReading, "media_reader"),
            (MediaCall::VisionEncoding, "text_encoder"),
            (MediaCall::LatentEncoding, "latent_encoder"),
        ];
        let routes = [generation.as_slice(), conditioned.as_slice()].concat();
        assert!(serving(&["t2va", "fl2va"], &routes).runtime_info().is_ok());
    }

    #[test]
    fn media_routing_refuses_an_incomplete_video_graph_and_a_repeated_call() {
        let model = media_info(&[(MediaCall::VideoDecoding, "video_decoder")], 4);
        let host = media_info(&[(MediaCall::Muxing, "muxer")], 0);
        let info = ExecutorInfo {
            workers: vec![
                (WorkerId("model".to_owned()), model.clone()),
                (WorkerId("host".to_owned()), host),
            ],
            video_codecs: BTreeMap::new(),
        };
        let message = info
            .media_routing()
            .expect_err("muxing needs the graph")
            .to_string();
        assert!(message.contains("VideoEncoding"), "{message}");

        let other = media_info(&[(MediaCall::VideoDecoding, "decoder")], 4);
        let info = ExecutorInfo {
            workers: vec![
                (WorkerId("model".to_owned()), model),
                (WorkerId("other".to_owned()), other),
            ],
            video_codecs: BTreeMap::new(),
        };
        let message = info
            .media_routing()
            .expect_err("one owner per call")
            .to_string();
        assert!(message.contains("VideoDecoding"), "{message}");
    }

    #[test]
    fn a_producer_watches_the_slots_its_consumers_write() {
        // A product retires only when the slots the producer watches are the
        // same slots its consumers write. Both come from this derivation, so
        // a disagreement would hang retirement or reuse live storage.
        let mut transfer = TransferConfig {
            edges: vec![
                TransferEdge {
                    source_worker: WorkerId("encoder".to_owned()),
                    source_rank: Some(0),
                    destination_worker: WorkerId("decode".to_owned()),
                    destination_rank: None,
                    device: Some(TransferBackend::CudaVmm),
                    host: Some(TransferBackend::Shm),
                },
                TransferEdge {
                    source_worker: WorkerId("encoder".to_owned()),
                    source_rank: Some(0),
                    destination_worker: WorkerId("encoder".to_owned()),
                    destination_rank: Some(0),
                    device: Some(TransferBackend::Local),
                    host: Some(TransferBackend::Local),
                },
            ],
            worker_ranks: std::collections::BTreeMap::new(),
            worker_hosts: std::collections::BTreeMap::new(),
        };
        transfer.worker_ranks.insert("decode".to_owned(), 2);
        transfer.worker_ranks.insert("encoder".to_owned(), 3);

        // Slots are dense across the instance: workers take contiguous runs in
        // the order the placement keys them.
        assert_eq!(transfer.acknowledgment_slot("decode", 0), 0);
        assert_eq!(transfer.acknowledgment_slot("decode", 1), 1);
        assert_eq!(transfer.acknowledgment_slot("encoder", 0), 2);
        assert_eq!(transfer.acknowledgment_slot("encoder", 2), 4);

        // The unranked edge names every decode rank; the self-edge publishes
        // into the producer's own address space and acknowledges nothing.
        assert_eq!(transfer.product_consumers("encoder", 0), vec![0, 1]);
        // A rank with no outgoing edge has no consumer to wait for.
        assert_eq!(transfer.product_consumers("encoder", 1), Vec::<u32>::new());
    }

    #[test]
    fn a_channel_edge_is_a_transfer_backend() {
        // Host products cross hosts on the rank channel, so the mechanism has
        // to be nameable in a placement's transfer bindings.
        let transfer = TransferConfig::parse("video_decoder->muxer=channel").unwrap();
        assert_eq!(transfer.edges[0].host, Some(TransferBackend::Channel));
        assert_eq!(transfer.edges[0].device, None);
        assert_eq!(TransferBackend::Channel.as_str(), "channel");
    }

    #[test]
    fn an_edge_names_a_device_mechanism_and_a_host_mechanism() {
        // A product's location decides which mechanism carries it, so one
        // edge binds both, and each location has at most one.
        let both = TransferConfig::parse("decoder->muxer=cuda_vmm+shm").unwrap();
        assert_eq!(both.edges[0].device, Some(TransferBackend::CudaVmm));
        assert_eq!(both.edges[0].host, Some(TransferBackend::Shm));
        assert!(both.edges[0].carries(TransferBackend::Shm));
        assert!(!both.edges[0].carries(TransferBackend::Channel));

        let local = TransferConfig::parse("decoder->decoder=local").unwrap();
        assert_eq!(local.edges[0].device, Some(TransferBackend::Local));
        assert_eq!(local.edges[0].host, Some(TransferBackend::Local));

        assert!(TransferConfig::parse("a->b=shm+channel").is_err());
        assert!(TransferConfig::parse("a->b=local+shm").is_err());
        assert!(TransferConfig::parse("a->b=cuda_vmm+cuda_vmm").is_err());
    }

    #[test]
    fn a_device_only_edge_refuses_a_product_published_as_host_bytes() {
        use uniserve_worker_ipc::{
            CallId, DType, DimBound, Locator, RequestKey, ShapeBound, TensorPublication, TensorRef,
            TensorTransfer, TransferHandle, TransferTransport, WorkerEndpoint as Endpoint,
        };

        // A device product that did not fit its producer's pool is published
        // in the producer's own address space and as shared-storage bytes; an
        // edge that carries only CUDA VMM takes neither to the decoder.
        let transfer = TransferConfig::parse("denoiser->decoder=cuda_vmm").unwrap();
        let endpoint = |worker: &str| Endpoint {
            worker_id: worker.to_owned(),
            rank: 0,
            node: "host".to_owned(),
            address_space: format!("{worker}-process"),
            incarnation: format!("{worker}-incarnation"),
        };
        let producer = endpoint("denoiser");
        let locator = |transport| Locator {
            source: producer.clone(),
            transport,
            nbytes: 4,
            dtype: "float32".to_owned(),
            shape: vec![1],
            offset: vec![0],
            device: "cpu".to_owned(),
        };
        let product = TensorRef {
            request_key: RequestKey::new(1, uniserve_core::RequestId(7), 1),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
            dtype: DType::F32,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(1)],
            },
        };
        let mut products = vec![TensorPublication {
            product: product.clone(),
            value: TransferHandle::DeviceProduct {
                height: 0,
                width: 0,
                value_range: String::new(),
                tensor: TensorTransfer {
                    shape: vec![1],
                    locations: vec![
                        locator(TransferTransport::Local {
                            endpoint: producer.incarnation.clone(),
                            key: 1,
                        }),
                        locator(TransferTransport::PosixShm {
                            endpoint: producer.incarnation.clone(),
                            name: "segment".to_owned(),
                        }),
                    ],
                },
            },
        }];

        let refusal = transfer
            .bind_inputs(&mut products, &mut [], &endpoint("decoder"))
            .unwrap_err();
        assert_eq!(refusal.buffers, vec![product.buffer_id()]);
        assert_eq!(refusal.destination, "decoder:0");
    }

    #[test]
    fn transport_map_parses_edges() {
        let t = TransferConfig::parse("encoder->prefill=cuda_vmm,prefill->decode=shm").unwrap();
        assert_eq!(
            t.edges[0],
            TransferEdge {
                source_worker: WorkerId("encoder".to_owned()),
                source_rank: None,
                destination_worker: WorkerId("prefill".to_owned()),
                destination_rank: None,
                device: Some(TransferBackend::CudaVmm),
                host: None,
            }
        );
        assert_eq!(t.edges[1].host, Some(TransferBackend::Shm));
        assert!(TransferConfig::parse("bad-entry").is_err());
        assert!(TransferConfig::parse("prefill->decode=tcp").is_err());
        assert!(TransferConfig::parse("prefill->decode=shm,prefill->decode=cuda_vmm").is_err());
        let split =
            TransferConfig::parse("encoder:0->denoiser:0=shm,encoder:0->denoiser:1=cuda_vmm")
                .unwrap();
        assert_eq!(split.edges[1].source_rank, Some(0));
        assert_eq!(split.edges[1].destination_rank, Some(1));
        assert!(TransferConfig::parse("encoder:x->denoiser=shm").is_err());
        assert!(
            TransferConfig::parse("encoder->denoiser=shm,encoder:0->denoiser:1=cuda_vmm").is_err()
        );
    }

    #[test]
    fn host_slots_name_every_rank_on_the_producing_ranks_host() {
        let mut transfer = TransferConfig::default();
        transfer.worker_ranks.insert("host".to_owned(), 2);
        transfer.worker_ranks.insert("model".to_owned(), 4);
        transfer
            .worker_hosts
            .insert("host".to_owned(), vec!["a".to_owned(), "b".to_owned()]);
        transfer.worker_hosts.insert(
            "model".to_owned(),
            vec![
                "a".to_owned(),
                "a".to_owned(),
                "b".to_owned(),
                "b".to_owned(),
            ],
        );
        // Model rank 3 is on host b with host rank 1 and model rank 2.
        let slots = transfer.host_slots("model", 3);
        let expected = [
            transfer.acknowledgment_slot("host", 1),
            transfer.acknowledgment_slot("model", 2),
            transfer.acknowledgment_slot("model", 3),
        ];
        assert_eq!(slots, expected);
        // An unplaced rank shares a host with no one.
        assert!(transfer.host_slots("model", 7).is_empty());
        assert!(TransferConfig::default().host_slots("model", 0).is_empty());
    }
}
