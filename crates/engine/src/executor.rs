//! Executor contracts shared by scheduler, worker IPC, and local engines.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::str::FromStr;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use uniserve_core::{CommandWaker, RequestId};
use uniserve_worker_ipc::{
    Batch, CompletionReport, ForwardMode, Operation, RequestKind, WorkerInfo, WorkerRequest,
};

/// Which pipeline stage a worker pool serves.
///
/// A pool is fully determined by the typed operations it accepts, the worker
/// implementation that executes those operations, and its device profile. The
/// control plane routes on the closed operation union and its nested mode.
///
/// `Full` holds the whole model and runs every model op. The other kinds are
/// model-backed stages with explicit weight-materialization scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKind {
    /// Whole model; runs ALL model ops in one mixed-batch forward.
    Full,
    /// Vision encode only — `vit_encode`/`vae_encode`; embedding handoff.
    Encoder,
    /// Prefill phase only — `prefill_und`; KV handoff to a Decode pool.
    Prefill,
    /// Decode/generation phase.
    Decode,
    /// Understanding tower — text + vision-encode + sampling. The und half of
    /// the local MoT understanding/generation stage split; routes
    /// every non-generation model op so a `--workers und:1,gen:1` topology
    /// composes the und/gen split through the general `StagedExecutor::new` path.
    Und,
    /// Generation tower — image denoise/commit + frame encode. The gen half of
    /// the Und/Gen stage split.
    Gen,
}

const FULL_WORK: &[ForwardMode] = &ForwardMode::ALL;
const ENCODER_WORK: &[ForwardMode] = &[ForwardMode::EncodeVision, ForwardMode::EncodeLatent];
const PREFILL_WORK: &[ForwardMode] = &[ForwardMode::TokenExtend];
const DECODE_WORK: &[ForwardMode] = &[
    ForwardMode::TokenDecode,
    ForwardMode::TokenVerify,
    ForwardMode::GenTransition,
    ForwardMode::GenFlow,
    ForwardMode::GenDecode,
    ForwardMode::Materialize,
    ForwardMode::TransferKvPublish,
    ForwardMode::TransferKvInstall,
];
const UND_WORK: &[ForwardMode] = &[
    ForwardMode::TokenExtend,
    ForwardMode::TokenDecode,
    ForwardMode::TokenVerify,
    ForwardMode::EncodeVision,
    ForwardMode::EncodeLatent,
    ForwardMode::TransferKvPublish,
    ForwardMode::TransferKvInstall,
];
const GEN_WORK: &[ForwardMode] = &[
    ForwardMode::GenTransition,
    ForwardMode::GenFlow,
    ForwardMode::GenDecode,
    ForwardMode::Materialize,
];

impl WorkerKind {
    /// Wire/config name (matches the Python `--worker-kind` vocabulary).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Encoder => "encoder",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Und => "und",
            Self::Gen => "gen",
        }
    }

    /// Parse a `--worker-kind`/`--workers` token.
    pub fn from_token(s: &str) -> Option<Self> {
        Some(match s {
            "full" => Self::Full,
            "encoder" => Self::Encoder,
            "prefill" => Self::Prefill,
            "decode" => Self::Decode,
            "und" => Self::Und,
            "gen" => Self::Gen,
            _ => return None,
        })
    }

    /// Exact work variants accepted by this worker role.
    pub fn supported_work(self) -> &'static [ForwardMode] {
        match self {
            Self::Full => FULL_WORK,
            Self::Encoder => ENCODER_WORK,
            Self::Prefill => PREFILL_WORK,
            Self::Decode => DECODE_WORK,
            Self::Und => UND_WORK,
            Self::Gen => GEN_WORK,
        }
    }

    /// Whether this role accepts this operation's work variant.
    pub fn handles(self, operation: &Operation) -> bool {
        self.supported_work().contains(&operation.work)
    }
}

/// One pool in a staged topology: `count` instances of `kind`, each with the
/// given tensor-parallel size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pool {
    pub kind: WorkerKind,
    /// Number of data-parallel pool instances of this kind (e.g. `encoder:2`).
    pub count: usize,
    /// Tensor-parallel rank count within each pool instance (`tp=N`).
    pub tp: usize,
}

/// The `--workers` topology: an ordered list of pool specs. `full:1` (one Full
/// pool, tp = `--worker-ranks`) is the default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerTopology {
    pub pools: Vec<Pool>,
}

impl WorkerTopology {
    /// The default single-Full-pool topology with the given tp size.
    pub fn single_full(tp: usize) -> Self {
        Self {
            pools: vec![Pool {
                kind: WorkerKind::Full,
                count: 1,
                tp: tp.max(1),
            }],
        }
    }

    /// Parse `--workers`, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4`.
    /// Each entry is `kind[:count[:tp=N]]`; `count` and `tp` default to 1.
    pub fn parse(s: &str) -> Result<Self, WorkerTopologyError> {
        let mut pools = Vec::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let mut parts = entry.split(':');
            let kind_str = parts.next().unwrap_or("").trim();
            let kind = WorkerKind::from_token(kind_str).ok_or_else(|| {
                WorkerTopologyError::message(format!("unknown worker kind {kind_str:?}"))
            })?;
            let mut count = 1usize;
            let mut tp = 1usize;
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
            pools.push(Pool { kind, count, tp });
        }
        if pools.is_empty() {
            return Err(WorkerTopologyError::message(
                "workers must list at least one pool",
            ));
        }
        Ok(Self { pools })
    }

    /// Whether this is the trivial single-Full-pool topology (the default, which
    /// composes a plain executor rather than a `StagedExecutor`).
    pub fn is_single_full(&self) -> bool {
        self.pools.len() == 1 && self.pools[0].kind == WorkerKind::Full && self.pools[0].count == 1
    }

    /// Total pool instances (sum of `count` across entries).
    pub fn total_pools(&self) -> usize {
        self.pools.iter().map(|p| p.count).sum()
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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

/// Per-edge local data-plane transfer selection (`--transfer`), e.g.
/// `encoder->prefill=shm,prefill->decode=cuda_ipc`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportMap {
    /// (producer kind, consumer kind) → backend name.
    pub edges: std::collections::BTreeMap<(WorkerKind, WorkerKind), TransferBackend>,
}

impl TransportMap {
    pub fn parse(s: &str) -> Result<Self, TransportMapError> {
        let mut edges = std::collections::BTreeMap::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                TransportMapError::message(format!("transfer entry {entry:?} must be edge=backend"))
            })?;
            let (src, dst) = edge.split_once("->").ok_or_else(|| {
                TransportMapError::message(format!("transfer edge {edge:?} must be src->dst"))
            })?;
            let src = WorkerKind::from_token(src.trim()).ok_or_else(|| {
                TransportMapError::message(format!(
                    "unknown worker kind {:?} in transfer edge",
                    src.trim()
                ))
            })?;
            let dst = WorkerKind::from_token(dst.trim()).ok_or_else(|| {
                TransportMapError::message(format!(
                    "unknown worker kind {:?} in transfer edge",
                    dst.trim()
                ))
            })?;
            let backend = TransferBackend::from_str(backend.trim())?;
            if backend == TransferBackend::Inproc {
                return Err(TransportMapError::message(
                    "inproc is the implicit backend and cannot be assigned to an edge",
                ));
            }
            if edges.insert((src, dst), backend).is_some() {
                return Err(TransportMapError::message(format!(
                    "duplicate transfer edge {edge:?}"
                )));
            }
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

/// One typed control operation carried over the executor control plane.
#[derive(Debug, Clone)]
pub enum ControlOp {
    DropSession(RequestId),
    ReleaseProducts(Vec<u64>),
}

impl ControlOp {
    pub const fn request_kind(&self) -> RequestKind {
        match self {
            Self::DropSession(_) => RequestKind::DropSession,
            Self::ReleaseProducts(_) => RequestKind::ReleaseProducts,
        }
    }

    pub fn method(&self) -> &'static str {
        match self {
            Self::DropSession(_) => "drop_session",
            Self::ReleaseProducts(_) => "release_products",
        }
    }

    pub fn to_request(&self, call_id: u64) -> WorkerRequest {
        let mut req = match self {
            Self::DropSession(id) => WorkerRequest::drop_session(*id),
            Self::ReleaseProducts(handles) => WorkerRequest::release_products(handles.clone()),
        };
        req.set_call_id(Some(call_id));
        req
    }
}

/// One rank's acknowledgment of a control call.
#[derive(Debug, Clone)]
pub struct ControlAck {
    pub rank: u32,
    pub result: Result<(), String>,
}

/// A worker-reported execution error classified for scheduler failure policy.
///
/// The typed taxonomy and execution context cross the wire together so failure
/// policy and diagnostics use the same operation identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("worker execute error: {message}")]
pub struct WorkerExecError {
    /// Physical submission whose response carried this error. Composite
    /// executors use it to join the same terminal outcome across ranks before
    /// returning the failure to the scheduler.
    pub step_id: Option<u64>,
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

/// A worker process was replaced without session snapshots, so every session
/// assigned to that executor must terminate explicitly before new work begins.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct WorkerLossError {
    pub message: String,
}

/// The asynchronous, pipelined boundary the scheduler drives.
pub trait Executor: Send {
    fn caps(&self) -> &WorkerInfo;
    fn pipeline_depth(&self) -> usize;
    fn in_flight(&self) -> usize;

    fn can_submit(&self) -> bool {
        self.in_flight() < self.pipeline_depth()
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()>;
    fn poll(&mut self) -> anyhow::Result<Option<CompletionReport>>;

    /// Whether the executor can preserve an exact device product from the
    /// producer through registration of the consumer. A local staged executor
    /// may satisfy this contract by retaining the consumer until the producing
    /// pool publishes its bounded transfer descriptor.
    fn device_products_reachable(&self, _producer: ForwardMode, _consumer: ForwardMode) -> bool {
        true
    }

    /// Liveness probe independent of in-flight work. Returns `Err` if the
    /// backing worker has died so the scheduler can latch fatal even while idle.
    /// Sim/local executors with no separate worker process always return `Ok`.
    fn check_liveness(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Cloneable waker the command ingress fires after enqueuing a command.
    /// In-process test executors may retain the default when they never drive
    /// the threaded scheduler loop.
    fn command_waker(&self) -> CommandWaker {
        CommandWaker::noop()
    }

    /// Native progress descriptors included in a composite executor's single
    /// park. Production process executors return one descriptor per worker.
    fn wake_file_descriptors(&self) -> Vec<i32> {
        Vec::new()
    }

    /// Block until a result may be ready, a command may have arrived, the worker
    /// died, or `timeout` elapses, without consuming a result.
    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        std::thread::park_timeout(timeout);
        Ok(())
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<CompletionReport>> {
        let Some(deadline) = Instant::now().checked_add(timeout) else {
            return self.poll();
        };
        loop {
            if let Some(result) = self.poll()? {
                return Ok(Some(result));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            self.park_for_event(deadline.saturating_duration_since(now))?;
        }
    }
    fn next_result(&mut self) -> anyhow::Result<CompletionReport>;
    fn control(&mut self, op: ControlOp) -> anyhow::Result<u64>;
    fn control_wait(
        &mut self,
        op: ControlOp,
        targets: Option<&[u32]>,
    ) -> anyhow::Result<Vec<ControlAck>>;
    fn shutdown(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{Digest, RequestId};
    use uniserve_worker_ipc::{Bounds, ForwardMode, OpId, RequestKey, RouteId, VersionRef};

    #[test]
    fn worker_kind_round_trips_and_maps_work() {
        for kind in [
            WorkerKind::Full,
            WorkerKind::Encoder,
            WorkerKind::Prefill,
            WorkerKind::Decode,
            WorkerKind::Und,
            WorkerKind::Gen,
        ] {
            assert_eq!(WorkerKind::from_token(kind.as_str()), Some(kind));
            assert!(!kind.supported_work().is_empty());
        }
        let extend = op(ForwardMode::TokenExtend);
        let decode = op(ForwardMode::TokenDecode);
        let transition = op(ForwardMode::GenTransition);
        let materialize = op(ForwardMode::Materialize);

        assert!(WorkerKind::Full.handles(&extend));
        assert!(WorkerKind::Prefill.handles(&extend));
        assert!(!WorkerKind::Prefill.handles(&decode));
        assert!(WorkerKind::Decode.handles(&decode));
        assert!(WorkerKind::Decode.handles(&transition));
        assert!(WorkerKind::Decode.handles(&materialize));
        assert!(WorkerKind::Und.handles(&decode));
        assert!(!WorkerKind::Und.handles(&materialize));
        assert!(WorkerKind::Gen.handles(&materialize));
        assert!(!WorkerKind::Gen.handles(&decode));
        assert_eq!(WorkerKind::from_token("nope"), None);
    }

    fn op(work: ForwardMode) -> Operation {
        let request_key = RequestKey::new(1, RequestId(1), 1);
        Operation {
            request_key,
            op_id: OpId(1),
            parent: VersionRef::admission_root(request_key, OpId(1), Digest::zero()),
            work,
            route: RouteId(0),
            domain: work.domain(),
            advances_state: false,
            bounds: Bounds::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            predicate: None,
            rng: None,
            control_seq: 0,
            plan_digest: Digest::zero(),
        }
        .sealed()
    }

    #[test]
    fn worker_topology_parses_pool_layouts() {
        assert!(WorkerTopology::single_full(4).is_single_full());
        let epd = WorkerTopology::parse("encoder:2,prefill:1:tp=4,decode:1:tp=4").unwrap();
        assert_eq!(epd.pools.len(), 3);
        assert_eq!(
            epd.pools[0],
            Pool {
                kind: WorkerKind::Encoder,
                count: 2,
                tp: 1
            }
        );
        assert_eq!(
            epd.pools[1],
            Pool {
                kind: WorkerKind::Prefill,
                count: 1,
                tp: 4
            }
        );
        assert_eq!(
            epd.pools[2],
            Pool {
                kind: WorkerKind::Decode,
                count: 1,
                tp: 4
            }
        );
        assert_eq!(epd.total_pools(), 4);
        assert!(!epd.is_single_full());
        assert!(WorkerTopology::parse("full:1").unwrap().is_single_full());
        assert!(WorkerTopology::parse("bogus:1").is_err());
        assert!(WorkerTopology::parse("").is_err());
        // The Und/Gen topology composes through the general staged path.
        let und_gen = WorkerTopology::parse("und:1,gen:1").unwrap();
        assert_eq!(und_gen.pools.len(), 2);
        assert_eq!(und_gen.pools[0].kind, WorkerKind::Und);
        assert_eq!(und_gen.pools[1].kind, WorkerKind::Gen);
        assert!(!und_gen.is_single_full());
    }

    #[test]
    fn transport_map_parses_edges() {
        let t = TransportMap::parse("encoder->prefill=cuda_ipc,prefill->decode=shm").unwrap();
        assert_eq!(
            t.edges.get(&(WorkerKind::Encoder, WorkerKind::Prefill)),
            Some(&TransferBackend::CudaIpc)
        );
        assert_eq!(
            t.edges.get(&(WorkerKind::Prefill, WorkerKind::Decode)),
            Some(&TransferBackend::Shm)
        );
        assert!(TransportMap::parse("bad-entry").is_err());
        assert!(TransportMap::parse("prefill->decode=tcp").is_err());
        assert!(TransportMap::parse("prefill->decode=shm,prefill->decode=cuda_ipc").is_err());
    }
}
