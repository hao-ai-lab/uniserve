//! Logical execution batches, physical execution, and executor contracts.
//!
//! The scheduler submits logical operations through [`Executor`]. Physical
//! executors lower those operations into worker protocol batches while retaining
//! the request and product identities needed to correlate completions.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub use uniserve_core::{ComponentConfig, ComponentDistribution, ParallelConfig, SequenceParallel};

use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use uniserve_worker_ipc::{ForwardMode, PipelineStage};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use uniserve_worker_ipc::{
    BatchCommand, BatchOutput, BlockTable, BufferAllocation, CachePageAllocation, Computation,
    ComputationId, DecodeRange, ForwardBatch, LatentParams, NewRequest, RequestKey, ScheduleBatch,
    ScheduledRequest, TensorPublication, WorkerInfo,
};

/// Physical placement selected by the scheduler for a computation.
///
/// The computation itself is the shared IPC `ScheduledRequest`. These fields describe
/// its worker and allocations, which are gathered into physical batch arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestPlacement {
    /// Worker routing stays outside the computation sent across IPC.
    pub worker: WorkerId,
    /// KV tables and newly acquired pages used by this computation.
    pub block_tables: Vec<BlockTable>,
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Rows use a local operation index until gathered into the physical batch.
    pub forward: ForwardBatch,
    pub latent: Option<LatentParams>,
    pub decode: Option<DecodeRange>,
    /// Persistent output spans; request retirement retains ownership of readers.
    pub buffers: Vec<BufferAllocation>,
}

/// One logical executor submission. Physical runs are derived only inside an executor.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionBatch {
    /// Logical batch identity used to correlate partial completions.
    pub id: u64,
    /// Shared computations paired with physical placement in scheduler order.
    pub requests: Vec<(ScheduledRequest, RequestPlacement)>,
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
        requests: Vec<(ScheduledRequest, RequestPlacement)>,
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

    /// Removes unstarted work for terminated epochs while preserving independent operations.
    /// Their resource descriptions stay attached to the removed operations. Close commands
    /// retain their physical retirement and reader obligations.
    pub(crate) fn retire_requests(
        &mut self,
        requests: &std::collections::HashSet<RequestKey>,
    ) -> Vec<(ScheduledRequest, RequestPlacement)> {
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
                .any(|(operation, _)| operation.kv_input == Some(publication.source))
        });
        retired
    }

    /// Validates operation identities, execution ownership, and command payloads.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.requests.is_empty() || !self.commands.is_empty(),
            "logical batch must carry at least one operation or command"
        );

        let mut requests = std::collections::HashSet::with_capacity(self.requests.len());
        let mut identities = std::collections::HashSet::with_capacity(self.requests.len());
        for (operation, placement) in &self.requests {
            operation.validate()?;
            anyhow::ensure!(
                operation.op_id.batch_id == self.id,
                "computation identity belongs to another logical batch"
            );
            requests.insert(operation.request_key);
            WorkerId::new(placement.worker.0.clone())?;
            anyhow::ensure!(
                !operation.entry.is_empty(),
                "operation requires a computation entry"
            );
            anyhow::ensure!(
                identities.insert(operation.op_id),
                "logical batch repeats an operation identity"
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
                    (latent.request_key, latent.op_id) == (operation.request_key, operation.op_id),
                    "logical operation carries another operation's latent execution"
                );
            }
            if let Some(decode) = &placement.decode {
                decode.validate()?;
                anyhow::ensure!(
                    (decode.request_key, decode.op_id) == (operation.request_key, operation.op_id),
                    "logical operation carries another operation's decode execution"
                );
            }
            for buffer in &placement.buffers {
                buffer.validate()?;
                anyhow::ensure!(
                    operation
                        .buffer_outputs()
                        .any(|output| output.buffer_id() == buffer.buffer),
                    "logical operation carries a buffer execution for another output"
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
                "logical batch starts a request without an operation"
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
        // corresponding operation family.
        let routed = |variant: Computation| {
            self.workers
                .iter()
                .find(|(_, info)| info.supported_ops.contains(&variant))
                .map(|(_, info)| info)
        };
        let mut kv_indices = [
            Computation::Forward(ForwardMode::Prefill),
            Computation::Forward(ForwardMode::Decode),
            Computation::Forward(ForwardMode::Verify),
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
        merged.pipeline_components = routed(Computation::Pipeline(PipelineStage::Denoising))
            .map(|info| info.pipeline_components.clone())
            .unwrap_or_default();
        merged.num_inference_steps = routed(Computation::Pipeline(PipelineStage::Denoising))
            .map_or(0, |info| info.num_inference_steps);
        anyhow::ensure!(
            self.workers.iter().all(|(_, info)| {
                !info
                    .supported_ops
                    .contains(&Computation::Pipeline(PipelineStage::Denoising))
                    || (info.pipeline_components == merged.pipeline_components
                        && info.num_inference_steps == merged.num_inference_steps)
            }),
            "workers disagree on pipeline components or diffusion steps"
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
        merged.supported_ops = Computation::ALL
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
        let flow = routed(Computation::Pipeline(PipelineStage::Denoising));
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

/// One operation result returned from an executor-owned physical run.
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
    pub run_id: u64,
    pub results: Vec<OpResult>,
    /// Tensor publications remain owned by the executor's transfer consumers.
    pub products: Vec<TensorPublication>,
    pub registration: uniserve_worker_ipc::RegistrationAck,
    pub worker_exec_us: Option<u64>,
    pub forward_stats: Option<uniserve_worker_ipc::ForwardStats>,
    pub done: bool,
}

impl WorkerResult {
    /// Claims all media before any fallible correlation or aggregation step.
    /// Acquisition failure belongs to the operation; independent results remain usable.
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
            run_id: report.run_id,
            results,
            products: report.products,
            registration: report.registration,
            worker_exec_us: report.worker_exec_us,
            forward_stats: report.forward_stats,
            done: report.done,
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
    /// Operation completions ready in this result fragment.
    pub results: Vec<OpResult>,
    /// Control commands acknowledged by all target pools.
    pub command_results: Vec<CommandResult>,
    /// Whether this fragment terminates the logical batch.
    pub done: bool,
    /// Per-worker execution durations in microseconds.
    pub worker_exec_us: Vec<u64>,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<uniserve_worker_ipc::ForwardStats>,
}

/// Resolves and validates one logical completion against its submitted operation.
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
    /// Publishes device products through CUDA IPC handles.
    CudaIpc,
}

impl TransferBackend {
    /// Returns the stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Shm => "shm",
            Self::CudaIpc => "cuda_ipc",
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
            "cuda_ipc" => Ok(Self::CudaIpc),
            _ => Err(TransferConfigError::message(format!(
                "unsupported transfer backend {value:?}"
            ))),
        }
    }
}

/// One explicit directed transfer edge between configured pool identities.
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
    /// Data-plane mechanism used for the edge.
    pub transport: TransferBackend,
}

/// Per-edge local data-plane transfer selection (`--transfer`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Explicit directed transfer edges.
    pub edges: Vec<TransferEdge>,
}

impl TransferConfig {
    /// Binds missing intra-Worker edges from the configured physical endpoints.
    ///
    /// Each rank is a separate process. Its self-edge uses local storage; CUDA
    /// peers on one node use CUDA IPC, and host-accessible pairs use shared
    /// memory. Explicit bindings take precedence. Initialized endpoint and
    /// backend capabilities are validated before the executor accepts work.
    pub fn with_worker_defaults(mut self, workers: &[crate::WorkerConfig]) -> anyhow::Result<Self> {
        crate::WorkerConfig::validate_all(workers)?;
        for worker in workers {
            for (source_rank, source) in worker.ranks.iter().enumerate() {
                for (destination_rank, destination) in worker.ranks.iter().enumerate() {
                    let source_rank = source_rank as u32;
                    let destination_rank = destination_rank as u32;
                    if self.edges.iter().any(|edge| {
                        edge.source_worker == worker.id
                            && edge.destination_worker == worker.id
                            && edge.source_rank.is_none_or(|rank| rank == source_rank)
                            && edge
                                .destination_rank
                                .is_none_or(|rank| rank == destination_rank)
                    }) {
                        continue;
                    }
                    anyhow::ensure!(
                        source.node == destination.node,
                        "cross-node Worker edges require an explicit supported transport"
                    );
                    let transport = if source_rank == destination_rank {
                        TransferBackend::Local
                    } else if source.device.starts_with("cuda:")
                        && destination.device.starts_with("cuda:")
                    {
                        TransferBackend::CudaIpc
                    } else {
                        TransferBackend::Shm
                    };
                    self.edges.push(TransferEdge {
                        source_worker: worker.id.clone(),
                        source_rank: Some(source_rank),
                        destination_worker: worker.id.clone(),
                        destination_rank: Some(destination_rank),
                        transport,
                    });
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
                backends.insert(edge.transport);
                publications.insert(edge.transport);
            }
            if edge.destination_worker.0 == worker
                && edge
                    .destination_rank
                    .is_none_or(|destination| destination == rank)
            {
                backends.insert(edge.transport);
            }
        }
        if publications.is_empty() {
            publications.insert(TransferBackend::Local);
        }
        (backends, publications)
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
            tensor.locations.retain(|location| {
                let selected = self
                    .edges
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
                    .map(|edge| edge.transport)
                    .or_else(|| {
                        (&location.source == destination).then_some(TransferBackend::Local)
                    });
                matches!(
                    (selected, &location.transport),
                    (
                        Some(TransferBackend::Local),
                        TransferTransport::Local { .. }
                    ) | (
                        Some(TransferBackend::Shm),
                        TransferTransport::PosixShm { .. }
                    ) | (
                        Some(TransferBackend::CudaIpc),
                        TransferTransport::CudaIpc { .. }
                    )
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

    /// Parses `source[:rank]->destination[:rank]=backend` directed bindings.
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
            let backend = TransferBackend::from_str(backend.trim())?;

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
                transport: backend,
            });
        }

        Ok(Self { edges })
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
/// policy and diagnostics use the same operation identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker execute error: {message}")]
pub struct WorkerExecError {
    /// Physical submission whose response carried this error. Composite
    /// executors use it to join the same terminal outcome across ranks before
    /// returning the failure to the scheduler.
    pub run_id: Option<u64>,
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
    /// Operations affected by the worker failure.
    pub operations: Vec<uniserve_worker_ipc::ErrorOperationIdentity>,
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
    /// Accepted logical operations that will no longer produce a device completion.
    pub retired: Vec<(u64, RequestKey, ComputationId)>,
    /// Buffers whose published locations no longer cover their complete logical value.
    pub buffers: Vec<uniserve_worker_ipc::BufferId>,
    /// Classified execution failure, if the ranks returned one before retirement.
    pub execution: Option<WorkerExecError>,
    /// Human-readable failure description.
    pub message: String,
}

/// Lowers a logical batch into a validated physical worker run.
pub(crate) fn physical_run(
    batch_id: u64,
    run_id: u64,
    collective_seq: u64,
    requests: Vec<(ScheduledRequest, RequestPlacement)>,
    commands: Vec<BatchCommand>,
    input_products: Vec<TensorPublication>,
    kv_inputs: Vec<uniserve_worker_ipc::KvTransfer>,
) -> anyhow::Result<ScheduleBatch> {
    let mut block_tables = Vec::new();
    let mut new_cache_pages = Vec::new();
    let mut forward = ForwardBatch::default();
    let mut latent_params = Vec::new();
    let mut decode_ranges = Vec::new();
    let mut buffer_allocations = Vec::new();
    let mut operations = Vec::with_capacity(requests.len());
    for (operation_index, (operation, placement)) in requests.into_iter().enumerate() {
        block_tables.extend(placement.block_tables);
        new_cache_pages.extend(placement.new_cache_pages);
        forward.append(placement.forward, operation_index as u32);
        latent_params.extend(placement.latent);
        decode_ranges.extend(placement.decode);
        buffer_allocations.extend(placement.buffers);
        operations.push(operation);
    }
    let run = ScheduleBatch {
        batch_id,
        run_id,
        collective_seq,
        operations,
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
    run.validate()?;
    Ok(run)
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

    #[test]
    fn transport_map_parses_edges() {
        let t = TransferConfig::parse("encoder->prefill=cuda_ipc,prefill->decode=shm").unwrap();
        assert_eq!(
            t.edges[0],
            TransferEdge {
                source_worker: WorkerId("encoder".to_owned()),
                source_rank: None,
                destination_worker: WorkerId("prefill".to_owned()),
                destination_rank: None,
                transport: TransferBackend::CudaIpc,
            }
        );
        assert_eq!(t.edges[1].transport, TransferBackend::Shm);
        assert!(TransferConfig::parse("bad-entry").is_err());
        assert!(TransferConfig::parse("prefill->decode=tcp").is_err());
        assert!(TransferConfig::parse("prefill->decode=shm,prefill->decode=cuda_ipc").is_err());
        let split =
            TransferConfig::parse("encoder:0->denoiser:0=shm,encoder:0->denoiser:1=cuda_ipc")
                .unwrap();
        assert_eq!(split.edges[1].source_rank, Some(0));
        assert_eq!(split.edges[1].destination_rank, Some(1));
        assert!(TransferConfig::parse("encoder:x->denoiser=shm").is_err());
        assert!(
            TransferConfig::parse("encoder->denoiser=shm,encoder:0->denoiser:1=cuda_ipc").is_err()
        );
    }
}
