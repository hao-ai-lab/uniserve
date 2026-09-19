//! Logical execution batches, physical execution, and executor contracts.
//!
//! The scheduler submits logical calls through [`Executor`]. Physical
//! executors lower those calls into worker protocol batches while retaining
//! the request and product identities needed to correlate completions.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub use uniserve_core::{ComponentConfig, ComponentDistribution, ParallelConfig, SequenceParallel};

use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use uniserve_worker_ipc::{ForwardMode, MediaCall};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use uniserve_worker_ipc::{
    Batch, BatchCommand, BatchOutput, BlockTable, BufferAllocation, CachePageAllocation, Call,
    CallId, CallKind, DecodeRange, ForwardBatch, LatentParams, NewRequest, RequestKey,
    TensorPublication, WorkerInfo,
};

/// Physical placement selected by the scheduler for a computation.
///
/// The computation itself is the shared IPC `Call`. These fields describe
/// its worker and allocations, which are gathered into physical batch arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestPlacement {
    /// Worker routing stays outside the computation sent across IPC.
    pub worker: WorkerId,
    /// KV tables and newly acquired pages used by this computation.
    pub block_tables: Vec<BlockTable>,
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Rows use a local call index until gathered into the physical batch.
    pub forward: ForwardBatch,
    pub latent: Option<LatentParams>,
    pub decode: Option<DecodeRange>,
    /// Persistent output spans; request retirement retains ownership of readers.
    pub buffers: Vec<BufferAllocation>,
}

/// One logical executor submission. Its rank projections are derived only inside an executor.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBatch {
    /// Logical batch identity used to correlate partial completions.
    pub id: u64,
    /// Shared call kinds paired with physical placement in scheduler order.
    pub requests: Vec<(Call, RequestPlacement)>,
    /// Ordered lifecycle and resource commands.
    pub commands: Vec<BatchCommand>,
    /// Published transfer descriptors supplied by an external storage owner.
    pub input_transfers: Vec<TensorPublication>,
    /// External cache publications consumed by explicit KV installation.
    pub kv_inputs: Vec<uniserve_worker_ipc::KvTransfer>,
}

impl ExecutionBatch {
    /// Constructs a logical executor submission.
    pub fn new(
        id: u64,
        requests: Vec<(Call, RequestPlacement)>,
        commands: Vec<BatchCommand>,
        input_transfers: Vec<TensorPublication>,
    ) -> Self {
        Self {
            id,
            requests,
            commands,
            input_transfers,
            kv_inputs: Vec::new(),
        }
    }

    /// Iterates over request admissions carried by batch commands.
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request),
            _ => None,
        })
    }

    /// Removes unstarted work for terminated epochs while preserving independent calls.
    /// Their resource descriptions stay attached to the removed calls. Close commands
    /// retain their physical retirement and reader obligations.
    pub(crate) fn retire_requests(
        &mut self,
        requests: &std::collections::HashSet<RequestKey>,
    ) -> Vec<(Call, RequestPlacement)> {
        let (retired, active): (Vec<_>, Vec<_>) = std::mem::take(&mut self.requests)
            .into_iter()
            .partition(|(op, _)| requests.contains(&op.request_key));
        self.requests = active;
        self.commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });
        let inputs = self
            .requests
            .iter()
            .flat_map(|(op, _)| op.tensor_inputs().chain(op.predicate.iter()))
            .collect::<std::collections::HashSet<_>>();
        self.input_transfers
            .retain(|payload| inputs.contains(&payload.product));
        self.kv_inputs.retain(|publication| {
            self.requests
                .iter()
                .any(|(call, _)| call.kv_input == Some(publication.source))
        });
        retired
    }

    /// Validates call identities, execution ownership, and command payloads.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.requests.is_empty() || !self.commands.is_empty(),
            "logical batch must carry at least one call or command"
        );

        let mut requests = std::collections::HashSet::with_capacity(self.requests.len());
        let mut identities = std::collections::HashSet::with_capacity(self.requests.len());
        for (call, placement) in &self.requests {
            call.validate()?;
            anyhow::ensure!(
                call.call_id.batch_id == self.id,
                "computation identity belongs to another logical batch"
            );
            requests.insert(call.request_key);
            WorkerId::new(placement.worker.0.clone())?;
            anyhow::ensure!(!call.component.is_empty(), "call requires a component");
            anyhow::ensure!(
                identities.insert(call.call_id),
                "logical batch repeats an call identity"
            );
            placement.forward.validate(1)?;
            for table in &placement.block_tables {
                table.validate()?;
            }
            for pages in &placement.new_cache_pages {
                pages.validate()?;
            }
            if let Some(latent) = &placement.latent {
                latent.validate()?;
                anyhow::ensure!(
                    (latent.request_key, latent.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's latent execution"
                );
            }
            if let Some(decode) = &placement.decode {
                decode.validate()?;
                anyhow::ensure!(
                    (decode.request_key, decode.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's decode execution"
                );
            }
            for buffer in &placement.buffers {
                buffer.validate()?;
                anyhow::ensure!(
                    call.buffer_outputs()
                        .any(|output| output.buffer_id() == buffer.buffer),
                    "logical call carries a buffer execution for another output"
                );
            }
        }

        let mut admitted = std::collections::HashSet::new();
        for admission in self.admissions() {
            admission.validate()?;
            anyhow::ensure!(
                admitted.insert(admission.request_key),
                "logical batch repeats a request start"
            );
            anyhow::ensure!(
                requests.contains(&admission.request_key),
                "logical batch starts a request without an call"
            );
        }

        for command in &self.commands {
            command.validate()?;
        }

        for transfer in &self.input_transfers {
            transfer.validate()?;
        }
        for publication in &self.kv_inputs {
            publication.validate()?;
        }
        Ok(())
    }
}

/// Concrete pool information reported through the execution boundary.
#[derive(Debug, Clone)]
pub struct ExecutorInfo {
    /// Physical pool identities and their reported capabilities.
    pub workers: Vec<(WorkerId, WorkerInfo)>,
}

impl ExecutorInfo {
    /// Constructs capability information for one physical pool.
    pub fn single(id: WorkerId, info: WorkerInfo) -> Self {
        Self {
            workers: vec![(id, info)],
        }
    }

    /// Validates and constructs capability information for multiple pools.
    pub fn from_workers(pools: Vec<(WorkerId, WorkerInfo)>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !pools.is_empty(),
            "executor info must contain at least one pool"
        );
        let mut ids = std::collections::HashSet::new();
        for (id, info) in &pools {
            anyhow::ensure!(ids.insert(id), "executor info repeats pool id {id}");
            info.validate()?;
        }
        Ok(Self { workers: pools })
    }

    /// Resolves which component serves each media call across every worker.
    ///
    /// The video graph may span workers: a model worker serves decoding and a
    /// host worker serves encoding and muxing. Each call is served by one
    /// component, which several workers may replicate, and a deployment that
    /// muxes serves the whole video graph between its workers.
    pub fn media_routing(&self) -> anyhow::Result<BTreeMap<MediaCall, String>> {
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
            let missing = MediaCall::VIDEO
                .iter()
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
    pub fn runtime_info(&self) -> anyhow::Result<WorkerInfo> {
        anyhow::ensure!(
            !self.workers.is_empty(),
            "executor exposes no physical pools"
        );
        if self.workers.len() == 1 {
            return Ok(self.workers[0].1.clone());
        }

        // Route-specific capacities contribute only when a pool implements the
        // corresponding call family.
        let routed = |variant: CallKind| {
            self.workers
                .iter()
                .find(|(_, info)| info.supported_ops.contains(&variant))
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
                .position(|(_, info)| info.supported_ops.contains(&variant))
        })
        .collect::<Vec<_>>();
        kv_indices.sort_unstable();
        kv_indices.dedup();

        let seed_index = kv_indices.first().copied().unwrap_or(0);
        let mut merged = self.workers[seed_index].1.clone();
        merged.media_components = self.media_routing()?;
        merged.num_inference_steps = routed(CallKind::Media(MediaCall::Denoising))
            .map_or(0, |info| info.num_inference_steps);
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
                    .supported_ops
                    .contains(&CallKind::Media(MediaCall::Denoising))
                    || info.num_inference_steps == merged.num_inference_steps
            }),
            "workers disagree on diffusion steps"
        );
        // Every KV stage must agree on layout. Capacity is the narrowest pool
        // because a lineage may traverse all routed KV stages.
        if let Some(first_index) = kv_indices.first().copied() {
            let first = &self.workers[first_index].1;
            let first_kv = first
                .kv_cache
                .as_ref()
                .context("executor routes KV work to a pool without a KV cache")?;
            for index in kv_indices.iter().copied().skip(1) {
                let other = &self.workers[index].1;
                let other_kv = other
                    .kv_cache
                    .as_ref()
                    .context("executor routes KV work to a pool without a KV cache")?;
                anyhow::ensure!(
                    other_kv.block_size == first_kv.block_size,
                    "executor KV pools disagree on block size"
                );
                anyhow::ensure!(
                    other_kv.total_layers == first_kv.total_layers
                        && other_kv.total_kv_heads == first_kv.total_kv_heads
                        && other_kv.head_dim == first_kv.head_dim
                        && other_kv.dtype == first_kv.dtype
                        && other_kv.groups == first_kv.groups,
                    "executor KV pools expose incompatible cache layouts"
                );
            }
            let mut kv_cache = first_kv.clone();
            kv_cache.num_blocks = kv_indices
                .iter()
                .filter_map(|index| self.workers[*index].1.kv_cache.as_ref())
                .map(|config| config.num_blocks)
                .min()
                .unwrap_or(first_kv.num_blocks);
            kv_cache.bytes_per_token = kv_indices
                .iter()
                .filter_map(|index| self.workers[*index].1.kv_cache.as_ref())
                .map(|config| config.bytes_per_token)
                .max()
                .unwrap_or(first_kv.bytes_per_token);
            merged.kv_cache = Some(kv_cache);
        } else {
            merged.kv_cache = None;
        }

        // Aggregate global limits conservatively across all physical pools.
        merged.supported_ops = CallKind::ALL
            .into_iter()
            .filter(|variant| routed(*variant).is_some())
            .collect();
        merged.queue_depth = self
            .workers
            .iter()
            .map(|(_, info)| info.queue_depth.max(1))
            .try_fold(0u32, |total, depth| total.checked_add(depth))
            .context("worker capacity exceeds the engine window")?;
        merged.max_batch_ops = self
            .workers
            .iter()
            .map(|(_, info)| info.max_batch_ops)
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
        merged.request_slots = self
            .workers
            .iter()
            .map(|(_, info)| info.request_slots)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_unresolved_ops = self
            .workers
            .iter()
            .map(|(_, info)| info.max_unresolved_ops)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
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
        Ok(merged)
    }
}

/// One call result returned from an executor-owned batch.
#[derive(Debug, Clone)]
pub struct OpResult {
    /// Validated completion values; media storage is carried by `media` below.
    pub output: uniserve_worker_ipc::RequestOutput,
    /// Claimed immutable output storage, or its request-local acquisition error.
    pub media: Result<Option<Arc<uniserve_core::SharedMedia>>, String>,
}

/// A decoded physical result that owns media before routing or rank validation.
#[derive(Debug)]
pub struct WorkerResult {
    pub batch_id: u64,
    pub results: Vec<OpResult>,
    /// Tensor publications remain owned by the executor's transfer consumers.
    pub products: Vec<TensorPublication>,
    pub registration: uniserve_worker_ipc::RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<uniserve_worker_ipc::ForwardStats>,
}

impl WorkerResult {
    /// Claims all media before any fallible correlation or aggregation step.
    /// Acquisition failure belongs to the call; independent results remain usable.
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
                OpResult { output, media }
            })
            .collect();
        Self {
            batch_id: report.batch_id,
            results,
            products: report.products,
            registration: report.registration,
            worker_exec_us: report.worker_exec_us,
            forward_stats: report.forward_stats,
        }
    }
}

/// Distinguishes applied control from physical retirement and unacknowledged failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    Applied,
    Retired,
    Failed,
}

/// Terminal outcome of one logical lifecycle command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    /// Position among lifecycle commands, excluding request admissions.
    pub command_index: u32,
    /// Request lineage targeted by the command.
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
    pub results: Vec<OpResult>,
    /// Control commands acknowledged by all target pools.
    pub command_results: Vec<CommandResult>,
    /// Whether this join completes the batch.
    pub done: bool,
    /// Per-worker execution durations in microseconds.
    pub worker_exec_us: Vec<u64>,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<uniserve_worker_ipc::ForwardStats>,
}

/// Resolves and validates one logical completion against its submitted call.
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
        command_results: done
            .then(|| {
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
            })
            .unwrap_or_default(),
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
    /// Publishes products through POSIX shared memory.
    Shm,
    /// Publishes device products through CUDA VMM handles.
    CudaVmm,
    /// Carries a host product's bytes on the rank channel's data path.
    ///
    /// Shared memory reaches only one host. A product on this edge travels in
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
/// location have no way across it and are refused where they are bound.
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

/// Per-edge local data-plane transfer selection (`--transfer`).
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
    /// Binds missing intra-Worker edges from the configured physical endpoints.
    ///
    /// Each rank is a separate process. Its self-edge uses local storage for
    /// both locations. Between two ranks, device products move over CUDA VMM
    /// where both hold a CUDA device, which reaches another host where both
    /// devices export a fabric handle; host products move over shared memory
    /// on one host and over the rank channel across hosts, because a
    /// shared-memory segment is named in one host's namespace. Explicit bindings take
    /// precedence. Initialized endpoint and backend capabilities are validated
    /// before the executor accepts work, and an edge that would have to cross
    /// hosts without fabric handles is refused by name.
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
                                // A shared-memory segment is named in one host's
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

    /// Counts the ranks that read one rank's device products.
    ///
    /// A product's consumers are the destinations of the edges leaving its
    /// producing rank, which is what component membership resolves to, and
    /// they are fixed when the placement is. A producing rank cannot derive
    /// this: it knows which component it belongs to, not which component reads
    /// what it publishes, and a product is consumed in a later batch than the
    /// one that produced it. So the head states it at launch.
    ///
    /// A rank omitted from an edge names every rank of that worker.
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
    /// consumers its producing call names: shared memory for a consumer on
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
    /// A product retires when every consumer has written its word in the
    /// chunk's header, so the producer needs the consumers' identities and not
    /// merely their number. A rank cannot derive either this or that: the
    /// mapping lives in the transfer edges, which only the head holds.
    pub fn products_cross_hosts(&self, worker: &str, rank: u32) -> bool {
        // Readiness is a producer synchronize only where it has to be. Within
        // a host an interprocess event carries it at no cost to the producing
        // stream, and stalling the producer for a consumer that could have
        // waited on the device is a bubble the placement does not require.
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

    /// Acknowledgment slots of the ranks that read this rank's device products.
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
    pub(crate) fn bind_inputs(
        &self,
        products: &mut [uniserve_worker_ipc::TensorPublication],
        kv_inputs: &mut [uniserve_worker_ipc::KvTransfer],
        destination: &uniserve_worker_ipc::WorkerEndpoint,
    ) -> anyhow::Result<()> {
        use uniserve_worker_ipc::{TransferHandle, TransferTransport};

        let tensors = products
            .iter_mut()
            .map(|payload| &mut payload.value)
            .flat_map(TransferHandle::tensors_mut)
            .chain(
                kv_inputs
                    .iter_mut()
                    .flat_map(|publication| &mut publication.tensors),
            );
        for tensor in tensors {
            // A product's location chose the mechanism it was published on,
            // so a location is kept where the edge carries that mechanism for
            // either location.
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
            anyhow::ensure!(
                !tensor.locations.is_empty(),
                "product has no location on a configured edge to {}:{}",
                destination.worker_id,
                destination.rank,
            );
        }
        Ok(())
    }

    /// Parses `source[:rank]->destination[:rank]=mechanisms` directed bindings.
    ///
    /// `mechanisms` names the edge's device mechanism, its host mechanism, or
    /// both joined by `+` with the device mechanism first, as in
    /// `cuda_vmm+shm`. `local` serves both locations.
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
    /// Physical route or worker pool associated with the failure.
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

/// Lowers a logical batch into a validated wire batch.
pub(crate) fn physical_batch(
    batch_id: u64,
    collective_seq: u64,
    requests: Vec<(Call, RequestPlacement)>,
    commands: Vec<BatchCommand>,
    input_products: Vec<TensorPublication>,
    kv_inputs: Vec<uniserve_worker_ipc::KvTransfer>,
) -> anyhow::Result<Batch> {
    let mut block_tables = Vec::new();
    let mut new_cache_pages = Vec::new();
    let mut forward = ForwardBatch::default();
    let mut latent_params = Vec::new();
    let mut decode_ranges = Vec::new();
    let mut buffer_allocations = Vec::new();
    let mut calls = Vec::with_capacity(requests.len());
    for (call_index, (call, placement)) in requests.into_iter().enumerate() {
        block_tables.extend(placement.block_tables);
        new_cache_pages.extend(placement.new_cache_pages);
        forward.append(placement.forward, call_index as u32);
        latent_params.extend(placement.latent);
        decode_ranges.extend(placement.decode);
        buffer_allocations.extend(placement.buffers);
        calls.push(call);
    }
    let batch = Batch {
        batch_id,
        collective_seq,
        calls,
        block_tables,
        new_cache_pages,
        forward,
        latent_params,
        decode_ranges,
        buffer_allocations,
        commands,
        input_products,
        kv_inputs,
    };
    batch.validate()?;
    Ok(batch)
}

/// The asynchronous, pipelined boundary the scheduler drives.
pub trait Executor: Send {
    /// Returns the executor's concrete pool capabilities.
    fn info(&self) -> &ExecutorInfo;
    /// Whether the complete instance is ready to accept execution.
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
        let mut info = WorkerInfo::default();
        info.media_components = routes
            .iter()
            .map(|(call, component)| (*call, (*component).to_owned()))
            .collect();
        info.supported_ops = routes
            .iter()
            .map(|(call, _)| CallKind::Media(*call))
            .collect();
        info.num_inference_steps = steps;
        info
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
                (MediaCall::VideoEncoding, "video_encoder"),
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
        };
        let routing = info.media_routing().expect("the union is complete");
        assert_eq!(routing.len(), MediaCall::VIDEO.len());
        assert_eq!(routing[&MediaCall::VideoEncoding], "video_encoder");
        assert_eq!(routing[&MediaCall::VideoDecoding], "video_decoder");
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
