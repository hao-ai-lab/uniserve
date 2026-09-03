//! Executor types shared by scheduler, worker IPC, and local engines.

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
    pub block_tables: Vec<BlockTable>,
    pub new_cache_pages: Vec<CachePageAllocation>,
    pub forward_rows: Vec<RowGeometry>,
    pub latent: Option<LatentPlacement>,
    pub decode: Option<DecodePlacement>,
    pub buffers: Vec<BufferPlacement>,
}

impl OpPlacement {
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
    pub id: OpId,
    pub request: RequestKey,
    pub parent: Option<uniserve_worker_ipc::Checkpoint>,
    pub kind: OpKind,
    pub payload: OpPayload,
    pub placement: OpPlacement,
    run_kind: RunKind,
}

impl Op {
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

    pub const fn request_key(&self) -> RequestKey {
        self.request
    }

    pub const fn id(&self) -> OpId {
        self.id
    }

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
    pub id: u64,
    pub ops: Vec<Op>,
    pub commands: Vec<BatchCommand>,
    pub inline: Vec<ProductPayload>,
}

impl Batch {
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

    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request),
            _ => None,
        })
    }

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
    pub pools: Vec<(PoolId, WorkerInfo)>,
}

impl ExecutorInfo {
    pub fn single(id: PoolId, info: WorkerInfo) -> Self {
        Self {
            pools: vec![(id, info)],
        }
    }

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

    pub fn single_pool(&self) -> &WorkerInfo {
        assert_eq!(
            self.pools.len(),
            1,
            "executor does not contain exactly one physical pool"
        );
        &self.pools[0].1
    }

    /// Derive the runtime's immutable capacity view from concrete pools.
    /// The returned value is not part of executor identity and is never
    /// reported as a physical worker.
    pub fn runtime_info(&self) -> anyhow::Result<WorkerInfo> {
        anyhow::ensure!(!self.pools.is_empty(), "executor exposes no physical pools");
        if self.pools.len() == 1 {
            return Ok(self.pools[0].1.clone());
        }
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
    pub output: uniserve_worker_ipc::ModelOutput,
    pub products: Vec<ProductPayload>,
}

/// Acknowledgement of one logical batch command after every target pool applied it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub command_index: u32,
    pub request_key: RequestKey,
}

/// One independently ready subset of a logical batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchResult {
    pub batch_id: u64,
    pub results: Vec<OpResult>,
    pub command_results: Vec<CommandResult>,
    pub done: bool,
    pub worker_exec_us: Vec<u64>,
    pub forward_stats: Vec<uniserve_worker_ipc::WorkerForwardStats>,
}

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

    pub(crate) fn unregister(&mut self, batch_id: u64) {
        self.pending.remove(&batch_id);
    }

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

pub type ExecutorError = anyhow::Error;

#[derive(Debug, thiserror::Error)]
pub enum ExecutorSubmitError {
    #[error("executor queue is full")]
    WouldBlock(Batch),
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

#[doc(hidden)]
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
    fn physical_info(&self) -> &ExecutorInfo;
    fn submit_run(&mut self, run: PhysicalRun) -> Result<(), PhysicalSubmitError>;
    fn poll_run(&mut self, timeout: Duration) -> Result<Option<RunResult>, ExecutorError>;
    fn take_command_wake(&mut self) -> bool {
        false
    }
    fn close_physical(&mut self) -> Result<(), ExecutorError>;
}

/// Stable identity of one explicitly configured physical worker pool.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PoolId(pub String);

impl PoolId {
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
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One concrete physical pool. Routing consumes only this explicit operation
/// set; profile names are discarded during CLI expansion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolConfig {
    pub id: PoolId,
    pub device: String,
    pub tensor_parallel_size: usize,
    pub supported_ops: Vec<OpKind>,
    pub queue_depth: usize,
}

/// The fully expanded worker topology consumed by engine construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerTopology {
    pub pools: Vec<PoolConfig>,
}

impl WorkerTopology {
    /// The default single-pool topology with the given tensor-parallel size.
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

    /// Apply process-level defaults while producing concrete pool values.
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

    /// Expand CLI convenience profiles such as
    /// `prefill:1:tp=4,decode:1:tp=4` into concrete pool configuration.
    pub fn parse(s: &str) -> Result<Self, WorkerTopologyError> {
        if s.trim_start().starts_with('[') {
            let pools: Vec<PoolConfig> = serde_json::from_str(s).map_err(|error| {
                WorkerTopologyError::message(format!("invalid explicit worker topology: {error}"))
            })?;
            let topology = Self { pools };
            topology.validate()?;
            return Ok(topology);
        }
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

    /// Whether this is one pool capable of every logical operation.
    pub fn is_single_full(&self) -> bool {
        self.pools.len() == 1
            && OpKind::ALL
                .iter()
                .all(|operation| self.pools[0].supported_ops.contains(operation))
    }

    /// Total concrete pool count.
    pub fn total_pools(&self) -> usize {
        self.pools.len()
    }
}

impl FromStr for WorkerTopology {
    type Err = WorkerTopologyError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct WorkerTopologyError(String);

impl WorkerTopologyError {
    fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferBackend {
    #[default]
    Inproc,
    Shm,
    CudaIpc,
}

impl TransferBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inproc => "inproc",
            Self::Shm => "shm",
            Self::CudaIpc => "cuda_ipc",
        }
    }
}

impl std::fmt::Display for TransferBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TransferBackend {
    type Err = TransportMapError;

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
    pub source_pool: PoolId,
    pub destination_pool: PoolId,
    pub transport: TransferBackend,
}

/// Per-edge local data-plane transfer selection (`--transfer`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportMap {
    pub edges: Vec<TransferEdge>,
}

impl TransportMap {
    pub fn parse(s: &str) -> Result<Self, TransportMapError> {
        let mut edges = Vec::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
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

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TransportMapError(String);

impl TransportMapError {
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
    pub fatal: bool,
    /// Whether retrying the same op could succeed (e.g. transient OOM).
    /// Defaults to `false` when the worker did not classify the error.
    pub retryable: bool,
    pub code: Option<String>,
    pub message: String,
    pub phase: Option<String>,
    pub route: Option<String>,
    pub operations: Vec<uniserve_worker_ipc::ErrorOperationIdentity>,
}

/// A worker process was replaced without request snapshots, so every request
/// assigned to that executor must terminate explicitly before new work begins.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkerLossError {
    pub message: String,
}

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

/// Lower one logical batch into the physical worker framing without changing its work shape.
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
    fn info(&self) -> &ExecutorInfo;
    fn submit(&mut self, batch: Batch) -> Result<(), ExecutorSubmitError>;
    fn poll(&mut self, timeout: Duration) -> Result<Option<BatchResult>, ExecutorError>;
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
