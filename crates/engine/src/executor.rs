//! Logical execution batches, physical execution, and executor contracts.
//!
//! The scheduler submits logical operations through [`Executor`]. Physical
//! executors lower those operations into worker protocol batches while retaining
//! the request and product identities needed to correlate completions.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
pub use uniserve_core::{ComponentDistribution, EntryConfig, ParallelConfig, SequenceParallel};

use std::str::FromStr;
use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use uniserve_worker_ipc::{
    BatchCommand, BlockTable, BufferAllocation, CachePageAllocation, DecodeRange, LatentParams,
    NewRequest, OpCode, OpId, OpPayload, Operation, ProductPayload, RequestKey, RowGeometry,
    Run as PhysicalRun, RunResult, WorkerInfo,
};

/// One logical operation and its scheduler-authoritative physical execution.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    /// Operation identity within the request lineage.
    pub id: OpId,
    /// Request lineage that owns the operation.
    pub request: RequestKey,
    /// Expected committed parent checkpoint.
    pub parent: Option<uniserve_worker_ipc::Checkpoint>,
    /// Selected Worker instance and computation entry.
    pub target: (WorkerId, String),
    /// Typed operation parameters.
    pub payload: OpPayload,
    /// KV page tables visible to the operation.
    pub block_tables: Vec<BlockTable>,
    /// KV pages allocated for this operation.
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Per-operation row geometry in the physical forward.
    pub forward_rows: Vec<RowGeometry>,
    /// Latent arena execution for trajectory operations.
    pub latent: Option<LatentParams>,
    /// Output execution for media decoding.
    pub decode: Option<DecodeRange>,
    /// Persistent output-buffer allocations.
    pub buffers: Vec<BufferAllocation>,
}

impl Op {
    pub const fn kind(&self) -> OpCode {
        self.payload.code
    }

    /// Binds a logical operation to its Worker entry before assigning physical resources.
    pub fn new(operation: Operation, target: (WorkerId, String)) -> Self {
        Self {
            target,
            id: operation.op_id,
            request: operation.request_key,
            parent: operation.parent,
            payload: operation.payload,
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward_rows: Vec::new(),
            latent: None,
            decode: None,
            buffers: Vec::new(),
        }
    }

    /// Returns the owning request lineage.
    pub const fn request_key(&self) -> RequestKey {
        self.request
    }

    /// Returns the operation identity.
    pub const fn id(&self) -> OpId {
        self.id
    }

    /// Returns the parsed command as an executable operation.
    pub(crate) fn into_operation(self) -> Operation {
        Operation {
            request_key: self.request,
            op_id: self.id,
            parent: self.parent,
            entry: self.target.1,
            payload: self.payload,
        }
    }
}

/// One logical executor submission. Physical runs are derived only inside an executor.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// Logical batch identity used to correlate partial completions.
    pub id: u64,
    /// Operations in scheduler submission order.
    pub ops: Vec<Op>,
    /// Ordered lifecycle and resource commands.
    pub commands: Vec<BatchCommand>,
    /// Host-resident input product payloads.
    pub inline: Vec<ProductPayload>,
}

impl Batch {
    /// Constructs a logical executor submission.
    pub fn new(
        id: u64,
        ops: Vec<Op>,
        commands: Vec<BatchCommand>,
        inline: Vec<ProductPayload>,
    ) -> Self {
        Self {
            id,
            ops,
            commands,
            inline,
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
    /// become physical retirement; unexecuted commits cannot select semantic state.
    pub(crate) fn retire_requests(
        &mut self,
        requests: &std::collections::HashSet<RequestKey>,
    ) -> Vec<Op> {
        let (retired, active): (Vec<_>, Vec<_>) = std::mem::take(&mut self.ops)
            .into_iter()
            .partition(|op| requests.contains(&op.request));
        self.ops = active;
        self.commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            match command {
                BatchCommand::Start { .. } | BatchCommand::Commit { .. } => false,
                BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                    ..
                } => {
                    *command = BatchCommand::Retire {
                        request_key: *request_key,
                        retained_buffers: std::mem::take(retained_buffers),
                    };
                    true
                }
                BatchCommand::Retire { .. } | BatchCommand::Free { .. } => true,
            }
        });
        let inputs = self
            .ops
            .iter()
            .flat_map(|op| op.payload.inputs.iter().chain(op.payload.predicate.iter()))
            .collect::<std::collections::HashSet<_>>();
        self.inline
            .retain(|payload| inputs.contains(&payload.product));
        retired
    }

    /// Validates operation identities, execution ownership, and command payloads.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.ops.is_empty() || !self.commands.is_empty(),
            "logical batch must carry at least one operation or command"
        );

        let mut requests = std::collections::HashSet::with_capacity(self.ops.len());
        let mut identities = std::collections::HashSet::with_capacity(self.ops.len());
        for op in &self.ops {
            let operation = op.clone().into_operation();
            operation.validate()?;
            requests.insert(op.request_key());
            WorkerId::new(op.target.0.0.clone())?;
            anyhow::ensure!(
                !op.target.1.is_empty(),
                "operation requires a computation entry"
            );
            anyhow::ensure!(
                identities.insert((op.request_key(), op.id())),
                "logical batch repeats an operation identity"
            );
            for table in &op.block_tables {
                table.validate()?;
            }
            for pages in &op.new_cache_pages {
                pages.validate()?;
            }
            if let Some(latent) = &op.latent {
                latent.validate()?;
                anyhow::ensure!(
                    (latent.request_key, latent.op_id) == (op.request_key(), op.id()),
                    "logical operation carries another operation's latent execution"
                );
            }
            if let Some(decode) = &op.decode {
                decode.validate()?;
                anyhow::ensure!(
                    (decode.request_key, decode.op_id) == (op.request_key(), op.id()),
                    "logical operation carries another operation's decode execution"
                );
            }
            for buffer in &op.buffers {
                buffer.validate()?;
                anyhow::ensure!(
                    operation
                        .outputs()
                        .iter()
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

        for inline in &self.inline {
            inline.validate()?;
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
        let routed = |variant: OpCode| {
            self.workers
                .iter()
                .find(|(_, info)| info.supported_ops.contains(&variant))
                .map(|(_, info)| info)
        };
        let mut kv_indices = [OpCode::ArExtend, OpCode::ArDecode, OpCode::ArVerify]
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
        merged.media_plan = routed(OpCode::DiffusionStep).and_then(|info| info.media_plan.clone());
        anyhow::ensure!(
            self.workers.iter().all(|(_, info)| !info
                .supported_ops
                .contains(&OpCode::DiffusionStep)
                || info.media_plan == merged.media_plan),
            "workers disagree on the numerical media plan"
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
        merged.supported_ops = OpCode::ALL
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
        let flow = routed(OpCode::DiffusionStep);
        merged.latent_page_units = flow.map_or(0, |info| info.latent_page_units);
        merged.latent_pages = flow.map_or(0, |info| info.latent_pages);
        merged.buffer_pool_bytes = self
            .workers
            .iter()
            .map(|(_, info)| info.buffer_pool_bytes)
            .filter(|capacity| *capacity > 0)
            .min()
            .unwrap_or(0);
        merged.validate()?;
        Ok(merged)
    }
}

/// One operation result returned from an executor-owned physical run.
#[derive(Debug, Clone, PartialEq)]
pub struct OpResult {
    /// Validated completion record returned by the worker.
    pub output: uniserve_worker_ipc::ModelOutput,
    /// Products published by the completed operation.
    pub products: Vec<ProductPayload>,
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
#[derive(Debug, Clone, PartialEq)]
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
    pub forward_stats: Vec<uniserve_worker_ipc::WorkerForwardStats>,
}

/// Resolves and validates one logical completion against its submitted operation.
pub(crate) fn logical_result(
    report: RunResult,
    done: bool,
    commands: &[BatchCommand],
) -> BatchResult {
    let mut results = Vec::new();
    let mut worker_exec_us = Vec::new();
    let mut forward_stats = Vec::new();
    if let Some(value) = report.worker_exec_us {
        worker_exec_us.push(value);
    }
    if let Some(value) = report.forward_stats {
        forward_stats.push(value);
    }
    for output in report.completions {
        let products = report
            .products
            .iter()
            .filter(|payload| {
                payload.product.request_key == output.request_key
                    && payload.product.producer_op_id == output.op_id
            })
            .cloned()
            .collect();
        results.push(OpResult { output, products });
    }
    BatchResult {
        batch_id: report.batch_id,
        results,
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

struct PendingLogicalResult {
    remaining: std::collections::HashMap<(RequestKey, OpId), OpCode>,
    commands: Vec<BatchCommand>,
    command_outcomes: std::collections::HashMap<u32, CommandOutcome>,
}

/// Executor-local reconciliation from physical partial reports to logical batch results.
#[derive(Default)]
pub(crate) struct LogicalResultTracker {
    pending: std::collections::HashMap<u64, PendingLogicalResult>,
}

impl LogicalResultTracker {
    /// Lifecycle commands awaiting this batch's physical acknowledgement.
    pub(crate) fn commands(&self, batch_id: u64) -> &[BatchCommand] {
        self.pending
            .get(&batch_id)
            .map_or(&[], |pending| &pending.commands)
    }

    /// Registers the operation executor.
    pub(crate) fn register(&mut self, batch: &Batch) -> anyhow::Result<()> {
        let state = PendingLogicalResult {
            remaining: batch
                .ops
                .iter()
                .map(|op| ((op.request_key(), op.id()), op.kind()))
                .collect(),
            commands: batch.commands.clone(),
            command_outcomes: std::collections::HashMap::new(),
        };
        anyhow::ensure!(
            self.pending.insert(batch.id, state).is_none(),
            "logical batch {} is already registered",
            batch.id
        );
        Ok(())
    }

    /// Unregisters the operation executor.
    pub(crate) fn unregister(&mut self, batch_id: u64) {
        self.pending.remove(&batch_id);
    }

    /// Retires work that cannot return after endpoint loss or cancellation before dispatch.
    /// This is host reconciliation, not a fabricated device completion.
    pub(crate) fn retire(
        &mut self,
        batch_id: u64,
        request: RequestKey,
        op: OpId,
    ) -> anyhow::Result<()> {
        let state = self
            .pending
            .get_mut(&batch_id)
            .context("retired operation has no logical batch")?;
        anyhow::ensure!(
            state.remaining.remove(&(request, op)).is_some(),
            "retired operation is unknown or already completed"
        );
        Ok(())
    }

    /// Record affected controls without replacing a prior failed acknowledgement.
    pub(crate) fn retire_commands(
        &mut self,
        batch_id: u64,
        requests: &std::collections::HashSet<RequestKey>,
    ) {
        self.set_command_outcome(batch_id, requests, CommandOutcome::Retired);
    }

    /// A rejected command has not retired storage even though its run is terminal.
    pub(crate) fn fail_commands(
        &mut self,
        batch_id: u64,
        requests: &std::collections::HashSet<RequestKey>,
    ) {
        self.set_command_outcome(batch_id, requests, CommandOutcome::Failed);
    }

    fn set_command_outcome(
        &mut self,
        batch_id: u64,
        requests: &std::collections::HashSet<RequestKey>,
        outcome: CommandOutcome,
    ) {
        if let Some(state) = self.pending.get_mut(&batch_id) {
            for (index, command) in state
                .commands
                .iter()
                .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                .enumerate()
            {
                if requests.contains(&command.request_key()) {
                    let current = state
                        .command_outcomes
                        .entry(index as u32)
                        .or_insert(outcome);
                    if outcome == CommandOutcome::Failed {
                        *current = outcome;
                    }
                }
            }
        }
    }

    /// Failed releases still own their physical routes and allocations.
    pub(crate) fn failed_commands(&self, batch_id: u64) -> Vec<BatchCommand> {
        self.pending
            .get(&batch_id)
            .into_iter()
            .flat_map(|state| {
                state
                    .commands
                    .iter()
                    .filter(|command| !matches!(command, BatchCommand::Start { .. }))
                    .enumerate()
                    .filter_map(|(index, command)| {
                        (state.command_outcomes.get(&(index as u32))
                            == Some(&CommandOutcome::Failed))
                        .then(|| command.clone())
                    })
            })
            .collect()
    }

    /// Validates one physical result and merges it into its pending logical batch.
    pub(crate) fn apply(&mut self, report: RunResult) -> anyhow::Result<BatchResult> {
        report.validate()?;
        let state = self.pending.get_mut(&report.batch_id).ok_or_else(|| {
            anyhow::anyhow!(
                "physical result names unknown logical batch {}",
                report.batch_id
            )
        })?;
        for output in report.completions() {
            let kind = state
                .remaining
                .remove(&(output.request_key, output.op_id))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "physical result repeats or invents operation {:?}/{} in logical batch {}",
                        output.request_key,
                        output.op_id.0,
                        report.batch_id
                    )
                })?;
            anyhow::ensure!(
                output.payload.family_matches(kind),
                "physical result payload family disagrees with operation {:?}/{}",
                output.request_key,
                output.op_id.0
            );
        }
        anyhow::ensure!(
            !report.done || state.remaining.is_empty(),
            "physical run ended before every logical operation completed"
        );
        let done = report.done && state.remaining.is_empty();
        let commands = state.commands.clone();
        let mut result = logical_result(report, done, &commands);
        for command in &mut result.command_results {
            command.outcome = state
                .command_outcomes
                .get(&command.command_index)
                .copied()
                .unwrap_or(CommandOutcome::Applied);
        }
        if done {
            self.pending.remove(&result.batch_id);
        }
        Ok(result)
    }
}

/// Dynamic error returned while polling or administering an executor.
pub type ExecutorError = anyhow::Error;

/// Backpressure and terminal failures returned by logical batch submission.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorSubmitError {
    #[error("executor queue is full")]
    /// Returns ownership of a batch rejected by bounded queue capacity.
    WouldBlock(Batch),
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
    type Err = TransportMapError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "local" => Ok(Self::Local),
            "shm" => Ok(Self::Shm),
            "cuda_ipc" => Ok(Self::CudaIpc),
            _ => Err(TransportMapError::message(format!(
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
pub struct TransportMap {
    /// Explicit directed transfer edges.
    pub edges: Vec<TransferEdge>,
}

impl TransportMap {
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
        products: &mut [uniserve_worker_ipc::ProductPayload],
        destination: &uniserve_worker_ipc::WorkerEndpoint,
    ) -> anyhow::Result<()> {
        use uniserve_worker_ipc::{InlineValue, TransferHandle, TransferTransport};

        for payload in products {
            let InlineValue::Transfer(handle) = &mut payload.value else {
                continue;
            };
            let tensors = match handle {
                TransferHandle::Encoder { tensor, .. }
                | TransferHandle::DeviceProduct { tensor, .. }
                | TransferHandle::Latent { tensor, .. } => std::slice::from_mut(tensor),
                TransferHandle::Kv { tensors, .. } => tensors.as_mut_slice(),
            };
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
        }
        Ok(())
    }

    /// Parses `source[:rank]->destination[:rank]=backend` directed bindings.
    pub fn parse(s: &str) -> Result<Self, TransportMapError> {
        let mut edges = Vec::new();

        let endpoint = |text: &str| -> Result<(WorkerId, Option<u32>), TransportMapError> {
            let (worker, rank) = match text.trim().split_once(':') {
                Some((worker, rank)) => (
                    worker,
                    Some(rank.parse::<u32>().map_err(|_| {
                        TransportMapError::message("transfer rank must be a nonnegative integer")
                    })?),
                ),
                None => (text.trim(), None),
            };
            let worker = WorkerId::new(worker).map_err(|error| {
                TransportMapError::message(format!("invalid transfer worker: {error}"))
            })?;
            Ok((worker, rank))
        };

        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            // Parse and validate both endpoint identities before accepting the
            // transport so errors remain attributable to one edge.
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                TransportMapError::message(format!("transfer entry {entry:?} must be edge=backend"))
            })?;
            let (src, dst) = edge.split_once("->").ok_or_else(|| {
                TransportMapError::message(format!("transfer edge {edge:?} must be src->dst"))
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
                return Err(TransportMapError::message(format!(
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

impl FromStr for TransportMap {
    type Err = TransportMapError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// Error returned for an invalid transfer-transport mapping.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TransportMapError(String);

impl TransportMapError {
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
    /// Whether retrying the same op could succeed (e.g. transient OOM).
    /// Defaults to `false` when the worker did not classify the error.
    pub retryable: bool,
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
    pub retired: Vec<(u64, RequestKey, OpId)>,
    /// Products whose published locations no longer cover their complete logical value.
    pub products: Vec<uniserve_worker_ipc::ProductRef>,
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
    ops: Vec<Op>,
    commands: Vec<BatchCommand>,
    inline: Vec<ProductPayload>,
) -> anyhow::Result<PhysicalRun> {
    let mut block_tables = Vec::new();
    let mut new_cache_pages = Vec::new();
    let mut forward_rows = Vec::new();
    let mut latent_params = Vec::new();
    let mut decode_ranges = Vec::new();
    let mut buffer_allocations = Vec::new();
    let mut operations = Vec::with_capacity(ops.len());
    for (operation_index, op) in ops.into_iter().enumerate() {
        let Op {
            id,
            request,
            parent,
            payload,
            block_tables: op_block_tables,
            new_cache_pages: op_new_cache_pages,
            forward_rows: op_forward_rows,
            latent,
            decode,
            buffers,
            target,
        } = op;
        block_tables.extend(op_block_tables);
        new_cache_pages.extend(op_new_cache_pages);
        forward_rows.extend(op_forward_rows.into_iter().map(|mut row| {
            row.operation_index = operation_index as u32;
            row
        }));
        latent_params.extend(latent);
        decode_ranges.extend(decode);
        buffer_allocations.extend(buffers);
        operations.push(Operation {
            request_key: request,
            op_id: id,
            parent,
            entry: target.1,
            payload,
        });
    }
    let run = PhysicalRun {
        batch_id,
        run_id,
        collective_seq,
        operations,
        block_tables,
        new_cache_pages,
        forward_rows,
        latent_params,
        decode_ranges,
        buffer_allocations,
        commands,
        input_products: inline,
    };
    run.validate()?;
    Ok(run)
}

/// Lowers one logical batch into the physical worker framing without changing its work shape.
pub(crate) fn lower_batch(
    batch: &Batch,
    next_collective_seq: &mut u64,
) -> anyhow::Result<PhysicalRun> {
    batch.validate()?;
    let collective_seq = (*next_collective_seq).max(1);
    *next_collective_seq = collective_seq.saturating_add(1);
    physical_run(
        batch.id,
        batch.id,
        collective_seq,
        batch.ops.clone(),
        batch.commands.clone(),
        batch.inline.clone(),
    )
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
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError>;
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
        let t = TransportMap::parse("encoder->prefill=cuda_ipc,prefill->decode=shm").unwrap();
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
        assert!(TransportMap::parse("bad-entry").is_err());
        assert!(TransportMap::parse("prefill->decode=tcp").is_err());
        assert!(TransportMap::parse("prefill->decode=shm,prefill->decode=cuda_ipc").is_err());
        let split = TransportMap::parse("encoder:0->denoiser:0=shm,encoder:0->denoiser:1=cuda_ipc")
            .unwrap();
        assert_eq!(split.edges[1].source_rank, Some(0));
        assert_eq!(split.edges[1].destination_rank, Some(1));
        assert!(TransportMap::parse("encoder:x->denoiser=shm").is_err());
        assert!(
            TransportMap::parse("encoder->denoiser=shm,encoder:0->denoiser:1=cuda_ipc").is_err()
        );
    }
}
