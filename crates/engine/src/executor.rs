//! Logical execution batches, physical placement, and executor contracts.
//!
//! The scheduler submits logical operations through [`Executor`]. Physical
//! executors lower those operations into worker protocol batches while retaining
//! the request and product identities needed to correlate completions.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::str::FromStr;
use std::time::Duration;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use uniserve_worker_ipc::{
    BatchCommand, BlockTable, BufferPlacement, CachePageAllocation, DecodePlacement,
    LatentPlacement, NewRequest, OpId, OpKind, OpPayload, Operation, ProductPayload, RequestKey,
    RowGeometry, Run as PhysicalRun, RunKind, RunResult, WorkerInfo,
};

/// Immutable worker placement for one logical operation.
#[derive(Debug, Clone, PartialEq)]
pub struct OpPlacement {
    /// KV page tables visible to the operation.
    pub block_tables: Vec<BlockTable>,
    /// KV pages allocated for this operation.
    pub new_cache_pages: Vec<CachePageAllocation>,
    /// Per-operation row geometry in the physical forward.
    pub forward_rows: Vec<RowGeometry>,
    /// Latent arena placement for trajectory operations.
    pub latent: Option<LatentPlacement>,
    /// Output placement for media decoding.
    pub decode: Option<DecodePlacement>,
    /// Persistent output-buffer placements.
    pub buffers: Vec<BufferPlacement>,
}

impl OpPlacement {
    /// Constructs an operation placement with no physical resources.
    pub fn empty() -> Self {
        Self {
            block_tables: Vec::new(),
            new_cache_pages: Vec::new(),
            forward_rows: Vec::new(),
            latent: None,
            decode: None,
            buffers: Vec::new(),
        }
    }
}

/// One logical operation and its scheduler-authoritative physical placement.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    /// Operation identity within the request lineage.
    pub id: OpId,
    /// Request lineage that owns the operation.
    pub request: RequestKey,
    /// Expected committed parent checkpoint.
    pub parent: Option<uniserve_worker_ipc::Checkpoint>,
    /// Coarse operation family used for routing.
    pub kind: OpKind,
    /// Typed operation parameters.
    pub payload: OpPayload,
    /// Scheduler-authoritative physical placement.
    pub placement: OpPlacement,
    run_kind: RunKind,
}

impl Op {
    /// Combines a worker operation with its scheduler placement.
    pub fn new(operation: Operation, placement: OpPlacement) -> Self {
        Self {
            id: operation.op_id,
            request: operation.request_key,
            parent: Some(operation.parent),
            kind: operation.kind.op_kind(),
            payload: operation.payload,
            placement,
            run_kind: operation.kind,
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
            parent: self
                .parent
                .unwrap_or_else(|| uniserve_worker_ipc::Checkpoint::admission_root(OpId(0))),
            kind: self.run_kind,
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

    /// Validates operation identities, placement ownership, and command payloads.
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
            anyhow::ensure!(
                requests.insert(op.request_key()),
                "logical batch carries multiple operations for one request"
            );
            anyhow::ensure!(
                identities.insert((op.request_key(), op.id())),
                "logical batch repeats an operation identity"
            );
            for table in &op.placement.block_tables {
                table.validate()?;
            }
            for pages in &op.placement.new_cache_pages {
                pages.validate()?;
            }
            if let Some(latent) = &op.placement.latent {
                latent.validate()?;
                anyhow::ensure!(
                    (latent.request_key, latent.op_id) == (op.request_key(), op.id()),
                    "logical operation carries another operation's latent placement"
                );
            }
            if let Some(decode) = &op.placement.decode {
                decode.validate()?;
                anyhow::ensure!(
                    (decode.request_key, decode.op_id) == (op.request_key(), op.id()),
                    "logical operation carries another operation's decode placement"
                );
            }
            for buffer in &op.placement.buffers {
                buffer.validate()?;
                anyhow::ensure!(
                    operation
                        .outputs()
                        .iter()
                        .any(|output| output.buffer_id() == buffer.buffer),
                    "logical operation carries a buffer placement for another output"
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
    pub pools: Vec<(PoolId, WorkerInfo)>,
}

impl ExecutorInfo {
    /// Constructs capability information for one physical pool.
    pub fn single(id: PoolId, info: WorkerInfo) -> Self {
        Self {
            pools: vec![(id, info)],
        }
    }

    /// Validates and constructs capability information for multiple pools.
    pub fn from_pools(pools: Vec<(PoolId, WorkerInfo)>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !pools.is_empty(),
            "executor info must contain at least one pool"
        );
        let mut ids = std::collections::HashSet::new();
        for (id, info) in &pools {
            anyhow::ensure!(ids.insert(id), "executor info repeats pool id {id}");
            info.validate()?;
        }
        Ok(Self { pools })
    }

    /// Returns the sole worker capability record.
    ///
    /// # Panics
    ///
    /// Panics unless the executor contains exactly one physical pool.
    pub fn single_pool(&self) -> &WorkerInfo {
        assert_eq!(
            self.pools.len(),
            1,
            "executor does not contain exactly one physical pool"
        );
        &self.pools[0].1
    }

    /// Derives the runtime's immutable capacity view from concrete pools.
    /// The returned value is not part of executor identity and is never
    /// reported as a physical worker.
    pub fn runtime_info(&self) -> anyhow::Result<WorkerInfo> {
        anyhow::ensure!(!self.pools.is_empty(), "executor exposes no physical pools");
        if self.pools.len() == 1 {
            return Ok(self.pools[0].1.clone());
        }

        // Route-specific capacities contribute only when a pool implements the
        // corresponding operation family.
        let routed = |variant: OpKind| {
            self.pools
                .iter()
                .find(|(_, info)| info.supported_ops.contains(&variant))
                .map(|(_, info)| info)
        };
        let mut kv_indices = [OpKind::ArExtend, OpKind::ArDecode, OpKind::ArVerify]
            .into_iter()
            .filter_map(|variant| {
                self.pools
                    .iter()
                    .position(|(_, info)| info.supported_ops.contains(&variant))
            })
            .collect::<Vec<_>>();
        kv_indices.sort_unstable();
        kv_indices.dedup();

        let seed_index = kv_indices.first().copied().unwrap_or(0);
        let mut merged = self.pools[seed_index].1.clone();
        let identity = (&self.pools[0].1.model_name, self.pools[0].1.weight_version);
        anyhow::ensure!(
            self.pools.iter().all(|(_, info)| {
                info.model_name == *identity.0 && info.weight_version == identity.1
            }),
            "executor pools expose different model names or weight versions"
        );

        // Every KV stage must agree on layout. Capacity is the narrowest pool
        // because a lineage may traverse all routed KV stages.
        if let Some(first_index) = kv_indices.first().copied() {
            let first = &self.pools[first_index].1;
            let first_kv = first
                .kv_cache
                .as_ref()
                .context("executor routes KV work to a pool without a KV cache")?;
            for index in kv_indices.iter().copied().skip(1) {
                let other = &self.pools[index].1;
                let other_kv = other
                    .kv_cache
                    .as_ref()
                    .context("executor routes KV work to a pool without a KV cache")?;
                anyhow::ensure!(
                    other_kv.block_size == first_kv.block_size,
                    "executor KV pools disagree on block size"
                );
                anyhow::ensure!(
                    other_kv.num_layers == first_kv.num_layers
                        && other_kv.num_kv_heads == first_kv.num_kv_heads
                        && other_kv.head_dim == first_kv.head_dim
                        && other_kv.dtype == first_kv.dtype
                        && other_kv.groups == first_kv.groups,
                    "executor KV pools expose incompatible cache layouts"
                );
            }
            let mut kv_cache = first_kv.clone();
            kv_cache.num_blocks = kv_indices
                .iter()
                .filter_map(|index| self.pools[*index].1.kv_cache.as_ref())
                .map(|config| config.num_blocks)
                .min()
                .unwrap_or(first_kv.num_blocks);
            kv_cache.bytes_per_token = kv_indices
                .iter()
                .filter_map(|index| self.pools[*index].1.kv_cache.as_ref())
                .map(|config| config.bytes_per_token)
                .max()
                .unwrap_or(first_kv.bytes_per_token);
            merged.kv_cache = Some(kv_cache);
        } else {
            merged.kv_cache = None;
        }

        // Aggregate global limits conservatively across all physical pools.
        merged.supported_ops = OpKind::ALL
            .into_iter()
            .filter(|variant| routed(*variant).is_some())
            .collect();
        merged.queue_depth = self
            .pools
            .iter()
            .map(|(_, info)| info.queue_depth.max(1))
            .min()
            .unwrap_or(1);
        merged.max_batch_ops = self
            .pools
            .iter()
            .map(|(_, info)| info.max_batch_ops)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_batch_tokens = self
            .pools
            .iter()
            .map(|(_, info)| info.max_batch_tokens)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.request_slots = self
            .pools
            .iter()
            .map(|(_, info)| info.request_slots)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        merged.max_unresolved_ops = self
            .pools
            .iter()
            .map(|(_, info)| info.max_unresolved_ops)
            .filter(|limit| *limit > 0)
            .min()
            .unwrap_or(0);
        let flow = routed(OpKind::DiffusionStep);
        merged.latent_page_units = flow.map_or(0, |info| info.latent_page_units);
        merged.latent_pages = flow.map_or(0, |info| info.latent_pages);
        merged.buffer_pool_bytes = self
            .pools
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

/// Acknowledgement of one logical batch command after every target pool applied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    /// Position of the acknowledged command in the logical batch.
    pub command_index: u32,
    /// Request lineage targeted by the command.
    pub request_key: RequestKey,
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
                    .enumerate()
                    .map(|(index, command)| CommandResult {
                        command_index: index as u32,
                        request_key: command.request_key(),
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
    remaining: std::collections::HashMap<(RequestKey, OpId), RunKind>,
    commands: Vec<BatchCommand>,
}

/// Executor-local reconciliation from physical partial reports to logical batch results.
#[derive(Default)]
pub(crate) struct LogicalResultTracker {
    pending: std::collections::HashMap<u64, PendingLogicalResult>,
}

impl LogicalResultTracker {
    /// Registers the operation executor.
    pub(crate) fn register(&mut self, batch: &Batch) -> anyhow::Result<()> {
        let state = PendingLogicalResult {
            remaining: batch
                .ops
                .iter()
                .map(|op| ((op.request_key(), op.id()), op.run_kind))
                .collect(),
            commands: batch.commands.clone(),
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
        if done {
            self.pending.remove(&report.batch_id);
        }
        Ok(logical_result(report, done, &commands))
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

#[doc(hidden)]
/// Backpressure and terminal failures returned by a physical executor.
#[derive(Debug, thiserror::Error)]
pub enum PhysicalSubmitError {
    #[error("physical executor queue is full")]
    WouldBlock(PhysicalRun),
    #[error(transparent)]
    Failed(anyhow::Error),
}

/// Internal worker-pool boundary used after logical routing and lowering.
#[doc(hidden)]
pub trait PhysicalExecutor: Send {
    /// Returns physical pool capabilities.
    fn physical_info(&self) -> &ExecutorInfo;
    /// Submits one fully lowered worker run.
    fn submit_run(&mut self, run: PhysicalRun) -> Result<(), PhysicalSubmitError>;
    /// Waits up to `timeout` for one partial or terminal run result.
    fn poll_run(&mut self, timeout: Duration) -> Result<Option<RunResult>, ExecutorError>;
    /// Consumes a pending command wake notification.
    fn take_command_wake(&mut self) -> bool {
        false
    }
    /// Closes physical workers and releases their transport resources.
    fn close_physical(&mut self) -> Result<(), ExecutorError>;
}

/// Stable identity of one explicitly configured physical worker pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PoolId(pub String);

impl PoolId {
    /// Validates and constructs a stable pool identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, WorkerTopologyError> {
        let value = value.into();
        if value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(WorkerTopologyError::message(format!(
                "invalid pool id {value:?}"
            )));
        }
        Ok(Self(value))
    }
}

impl std::fmt::Display for PoolId {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One concrete physical pool. Routing consumes only this explicit operation
/// set; profile names are discarded during CLI expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Stable identity used by routes and transfer edges.
    pub id: PoolId,
    /// Device selector forwarded to worker processes.
    pub device: String,
    /// Number of tensor-parallel worker ranks.
    pub tensor_parallel_size: usize,
    /// Logical operation families accepted by this pool.
    pub supported_ops: Vec<OpKind>,
    /// Maximum number of unresolved physical runs.
    pub queue_depth: usize,
}

/// The fully expanded worker topology consumed by engine construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerTopology {
    /// Concrete physical pools in configuration order.
    pub pools: Vec<PoolConfig>,
}

impl WorkerTopology {
    /// Builds the default single-pool topology with the given tensor-parallel size.
    pub fn single_full(tp: usize) -> Self {
        Self {
            pools: vec![PoolConfig {
                id: PoolId("full".to_owned()),
                device: "cuda".to_owned(),
                tensor_parallel_size: tp.max(1),
                supported_ops: OpKind::ALL.to_vec(),
                queue_depth: 0,
            }],
        }
    }

    /// Applies process-level defaults while producing concrete pool values.
    pub fn with_process_defaults(mut self, device: &str, queue_depth: usize) -> Self {
        for pool in &mut self.pools {
            if pool.device == "cuda" {
                pool.device = device.to_owned();
            }
            if pool.queue_depth == 0 {
                pool.queue_depth = queue_depth.max(1);
            }
        }
        self
    }

    /// Expands CLI convenience profiles such as
    /// `prefill:1:tp=4,decode:1:tp=4` into concrete pool configuration.
    pub fn parse(s: &str) -> Result<Self, WorkerTopologyError> {
        // JSON topology input already names every concrete pool.
        if s.trim_start().starts_with('[') {
            let pools: Vec<PoolConfig> = serde_json::from_str(s).map_err(|error| {
                WorkerTopologyError::message(format!("invalid explicit worker topology: {error}"))
            })?;
            let topology = Self { pools };
            topology.validate()?;
            return Ok(topology);
        }

        // Compact entries select a capability profile and then override its
        // instance count, placement, parallelism, and queue depth.
        let mut pools = Vec::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let mut parts = entry.split(':');
            let profile = parts.next().unwrap_or("").trim();
            let supported_ops = match profile {
                "full" => OpKind::ALL.to_vec(),
                "prefill" => vec![OpKind::ArExtend],
                "decode" => vec![
                    OpKind::ArDecode,
                    OpKind::ArVerify,
                    OpKind::DiffusionPrepare,
                    OpKind::DiffusionStep,
                    OpKind::DiffusionDecode,
                ],
                _ => {
                    return Err(WorkerTopologyError::message(format!(
                        "unknown worker profile {profile:?}"
                    )));
                }
            };
            let mut count = 1usize;
            let mut tp = 1usize;
            let mut device = "cuda".to_owned();
            let mut queue_depth = 0usize;

            for part in parts {
                let part = part.trim();
                if let Some(tp_str) = part.strip_prefix("tp=") {
                    tp = tp_str
                        .parse::<usize>()
                        .map_err(|_| {
                            WorkerTopologyError::message(format!(
                                "invalid tensor-parallel size in entry {entry:?}"
                            ))
                        })?
                        .max(1);
                } else if let Some(value) = part.strip_prefix("device=") {
                    if value.is_empty() {
                        return Err(WorkerTopologyError::message(format!(
                            "empty device in entry {entry:?}"
                        )));
                    }
                    device = value.to_owned();
                } else if let Some(value) = part.strip_prefix("depth=") {
                    queue_depth = value.parse::<usize>().map_err(|_| {
                        WorkerTopologyError::message(format!(
                            "invalid queue depth in entry {entry:?}"
                        ))
                    })?;
                } else {
                    count = part
                        .parse::<usize>()
                        .map_err(|_| {
                            WorkerTopologyError::message(format!(
                                "invalid worker count in entry {entry:?}"
                            ))
                        })?
                        .max(1);
                }
            }

            // Expand repeated profile instances into independently routable pools.
            for instance in 0..count {
                let name = if count == 1 {
                    profile.to_owned()
                } else {
                    format!("{profile}-{instance}")
                };
                pools.push(PoolConfig {
                    id: PoolId::new(name)?,
                    device: device.clone(),
                    tensor_parallel_size: tp,
                    supported_ops: supported_ops.clone(),
                    queue_depth,
                });
            }
        }

        let topology = Self { pools };
        topology.validate()?;
        Ok(topology)
    }

    /// Validates pool identities, devices, parallelism, and routing declarations.
    pub fn validate(&self) -> Result<(), WorkerTopologyError> {
        if self.pools.is_empty() {
            return Err(WorkerTopologyError::message(
                "workers must list at least one pool",
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for pool in &self.pools {
            PoolId::new(pool.id.0.clone())?;
            if !ids.insert(&pool.id) {
                return Err(WorkerTopologyError::message(format!(
                    "worker topology repeats pool id {}",
                    pool.id
                )));
            }
            if pool.device.is_empty() || pool.tensor_parallel_size == 0 {
                return Err(WorkerTopologyError::message(format!(
                    "pool {} has an invalid device or tensor-parallel size",
                    pool.id
                )));
            }
            if pool.supported_ops.is_empty()
                || pool
                    .supported_ops
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != pool.supported_ops.len()
            {
                return Err(WorkerTopologyError::message(format!(
                    "pool {} must declare a unique, non-empty operation set",
                    pool.id
                )));
            }
        }
        Ok(())
    }

    /// Returns whether this is one pool capable of every logical operation.
    pub fn is_single_full(&self) -> bool {
        self.pools.len() == 1
            && OpKind::ALL
                .iter()
                .all(|operation| self.pools[0].supported_ops.contains(operation))
    }

    /// Returns the total concrete pool count.
    pub fn total_pools(&self) -> usize {
        self.pools.len()
    }
}

impl FromStr for WorkerTopology {
    type Err = WorkerTopologyError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// Error returned for an invalid worker topology string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct WorkerTopologyError(String);

impl WorkerTopologyError {
    /// Returns the human-readable error message.
    fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Transport available for product movement between worker pools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferBackend {
    /// Keeps products within one worker process.
    #[default]
    Inproc,
    /// Publishes products through POSIX shared memory.
    Shm,
    /// Publishes device products through CUDA IPC handles.
    CudaIpc,
}

impl TransferBackend {
    /// Returns the stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inproc => "inproc",
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
            "inproc" => Ok(Self::Inproc),
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
    pub source_pool: PoolId,
    /// Pool that consumes the transferred product.
    pub destination_pool: PoolId,
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
    /// Parses a comma-separated list of `source->destination=backend` edges.
    pub fn parse(s: &str) -> Result<Self, TransportMapError> {
        let mut edges = Vec::new();

        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            // Parse and validate both endpoint identities before accepting the
            // transport so errors remain attributable to one edge.
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                TransportMapError::message(format!("transfer entry {entry:?} must be edge=backend"))
            })?;
            let (src, dst) = edge.split_once("->").ok_or_else(|| {
                TransportMapError::message(format!("transfer edge {edge:?} must be src->dst"))
            })?;
            let src = PoolId::new(src.trim()).map_err(|error| {
                TransportMapError::message(format!("invalid source pool: {error}"))
            })?;
            let dst = PoolId::new(dst.trim()).map_err(|error| {
                TransportMapError::message(format!("invalid destination pool: {error}"))
            })?;
            let backend = TransferBackend::from_str(backend.trim())?;

            if backend == TransferBackend::Inproc {
                return Err(TransportMapError::message(
                    "inproc is the implicit backend and cannot be assigned to an edge",
                ));
            }
            if edges.iter().any(|existing: &TransferEdge| {
                existing.source_pool == src && existing.destination_pool == dst
            }) {
                return Err(TransportMapError::message(format!(
                    "duplicate transfer edge {edge:?}"
                )));
            }

            edges.push(TransferEdge {
                source_pool: src,
                destination_pool: dst,
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

/// A worker process was replaced without request snapshots, so every request
/// assigned to that executor must terminate explicitly before new work begins.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkerLossError {
    /// Human-readable worker-loss description.
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
    let mut latent_placements = Vec::new();
    let mut decode_placements = Vec::new();
    let mut buffer_placements = Vec::new();
    let mut operations = Vec::with_capacity(ops.len());
    for (operation_index, op) in ops.into_iter().enumerate() {
        let Op {
            id,
            request,
            parent,
            payload,
            placement,
            run_kind,
            ..
        } = op;
        block_tables.extend(placement.block_tables);
        new_cache_pages.extend(placement.new_cache_pages);
        forward_rows.extend(placement.forward_rows.into_iter().map(|mut row| {
            row.operation_index = operation_index as u32;
            row
        }));
        latent_placements.extend(placement.latent);
        decode_placements.extend(placement.decode);
        buffer_placements.extend(placement.buffers);
        operations.push(Operation {
            request_key: request,
            op_id: id,
            parent: parent
                .unwrap_or_else(|| uniserve_worker_ipc::Checkpoint::admission_root(OpId(0))),
            kind: run_kind,
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
        latent_placements,
        decode_placements,
        buffer_placements,
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
    use uniserve_worker_ipc::OpKind;

    #[test]
    fn worker_topology_parses_pool_layouts() {
        let single = WorkerTopology::single_full(4);
        assert!(single.is_single_full());
        assert_eq!(
            single.with_process_defaults("cuda:1", 3).pools[0].queue_depth,
            3
        );
        let epd = WorkerTopology::parse("prefill:1:tp=4,decode:1:tp=4").unwrap();
        assert_eq!(epd.pools.len(), 2);
        assert_eq!(epd.pools[0].id, PoolId("prefill".to_owned()));
        assert_eq!(epd.pools[0].tensor_parallel_size, 4);
        assert_eq!(epd.pools[0].supported_ops, vec![OpKind::ArExtend]);
        assert_eq!(epd.pools[1].id, PoolId("decode".to_owned()));
        assert_eq!(epd.pools[1].tensor_parallel_size, 4);
        assert!(epd.pools[1].supported_ops.contains(&OpKind::ArDecode));
        assert_eq!(epd.total_pools(), 2);
        assert!(!epd.is_single_full());
        assert!(WorkerTopology::parse("full:1").unwrap().is_single_full());
        assert!(WorkerTopology::parse("bogus:1").is_err());
        assert!(WorkerTopology::parse("").is_err());
    }

    #[test]
    fn transport_map_parses_edges() {
        let t = TransportMap::parse("encoder->prefill=cuda_ipc,prefill->decode=shm").unwrap();
        assert_eq!(
            t.edges[0],
            TransferEdge {
                source_pool: PoolId("encoder".to_owned()),
                destination_pool: PoolId("prefill".to_owned()),
                transport: TransferBackend::CudaIpc,
            }
        );
        assert_eq!(t.edges[1].transport, TransferBackend::Shm);
        assert!(TransportMap::parse("bad-entry").is_err());
        assert!(TransportMap::parse("prefill->decode=tcp").is_err());
        assert!(TransportMap::parse("prefill->decode=shm,prefill->decode=cuda_ipc").is_err());
    }
}
