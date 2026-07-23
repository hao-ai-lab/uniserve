//! Executor contracts shared by scheduler, worker IPC, and local engines.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::time::{Duration, Instant};

use uniserve_core::{BlockId, CommandWaker, GeneratedImageCommitCapabilities, RequestId};
use uniserve_worker_wire::{
    Batch, EngineCaps, ExecutionResult, Operation, OperationType, RequestKind, SnapshotRef,
    WorkerRequest,
};

/// Synchronous model-engine seam used by deterministic local implementations.
pub trait ModelEngine: Send {
    fn caps(&self) -> EngineCaps;
    fn execute(&mut self, batch: Batch) -> anyhow::Result<ExecutionResult>;
    fn drop_session(&mut self, id: RequestId) -> anyhow::Result<()>;
}

/// Which pipeline stage a worker pool serves.
///
/// A pool is fully determined by the typed operations it accepts, the worker
/// implementation that executes those operations, and its device profile. The
/// control plane routes on the closed operation union and its nested mode.
///
/// `Full` is the non-disaggregated default: it holds the whole model and runs
/// every model op in one mixed-batch forward. The other kinds are stages peeled
/// off `Full` (encoder / prefill / decode / sampler / post-process).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WorkerKind {
    /// Whole model; runs ALL model ops in one mixed-batch forward.
    Full,
    /// Vision encode only — `vit_encode`/`vae_encode`; embedding handoff.
    Encoder,
    /// Prefill phase only — `prefill_und`; KV handoff to a Decode pool.
    Prefill,
    /// Decode/generation phase — `decode_und`/`target_verify_und`/`denoise_gen`/
    /// `commit_gen`.
    Decode,
    /// Sampler only — `sample`; logits→token, CPU data-parallel.
    Sampler,
    /// Post-process only — `encode_frame`; FFmpeg, CPU.
    PostProcess,
    /// Understanding tower — text + vision-encode + sampling. The und half of
    /// the MoT understanding/generation disaggregation (`two_role`); routes
    /// every non-generation model op so a `--workers und:1,gen:1` topology
    /// composes the und/gen split through the general `StageRouter::new` path.
    Und,
    /// Generation tower — image denoise/commit + frame encode. The gen half of
    /// the und/gen disaggregation.
    Gen,
}

const FULL_OPERATIONS: &[OperationType] = &OperationType::ALL;
const ENCODER_OPERATIONS: &[OperationType] =
    &[OperationType::EncodeVision, OperationType::EncodeLatent];
const PREFILL_OPERATIONS: &[OperationType] = &[OperationType::SequenceExtend];
const DECODE_OPERATIONS: &[OperationType] = &[
    OperationType::SequenceDecode,
    OperationType::SequenceVerify,
    OperationType::Flow,
    OperationType::MaterializeImage,
    OperationType::TransferKv,
];
const SAMPLER_OPERATIONS: &[OperationType] = &[OperationType::SequenceSample];
const POSTPROCESS_OPERATIONS: &[OperationType] = &[OperationType::MaterializeFrame];
const UND_OPERATIONS: &[OperationType] = &[
    OperationType::SequenceExtend,
    OperationType::SequenceDecode,
    OperationType::SequenceVerify,
    OperationType::SequenceSample,
    OperationType::EncodeVision,
    OperationType::EncodeLatent,
    OperationType::TransferKv,
];
const GEN_OPERATIONS: &[OperationType] = &[
    OperationType::Flow,
    OperationType::MaterializeImage,
    OperationType::MaterializeFrame,
];

impl WorkerKind {
    /// Wire/config name (matches the Python `--worker-kind` vocabulary).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Encoder => "encoder",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Sampler => "sampler",
            Self::PostProcess => "postprocess",
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
            "sampler" => Self::Sampler,
            "postprocess" => Self::PostProcess,
            "und" => Self::Und,
            "gen" => Self::Gen,
            _ => return None,
        })
    }

    /// Exact operation types accepted by this worker role.
    pub fn supported_operation_types(self) -> &'static [OperationType] {
        match self {
            Self::Full => FULL_OPERATIONS,
            Self::Encoder => ENCODER_OPERATIONS,
            Self::Prefill => PREFILL_OPERATIONS,
            Self::Decode => DECODE_OPERATIONS,
            Self::Sampler => SAMPLER_OPERATIONS,
            Self::PostProcess => POSTPROCESS_OPERATIONS,
            Self::Und => UND_OPERATIONS,
            Self::Gen => GEN_OPERATIONS,
        }
    }

    /// Whether this role accepts this exact typed operation.
    pub fn handles(self, operation: &Operation) -> bool {
        self.supported_operation_types()
            .contains(&operation.operation_type())
    }
}

/// One pool in a staged topology: `count` instances of `kind`, each with the
/// given tensor-parallel size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSpec {
    pub kind: WorkerKind,
    /// Number of data-parallel pool instances of this kind (e.g. `encoder:2`).
    pub count: usize,
    /// Tensor-parallel rank count within each pool instance (`tp=N`).
    pub tp: usize,
}

/// The `--workers` topology: an ordered list of pool specs. `full:1` (one Full
/// pool, tp = `--worker-ranks`) is the non-disaggregated default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkersSpec {
    pub pools: Vec<PoolSpec>,
}

impl WorkersSpec {
    /// The default single-Full-pool topology with the given tp size.
    pub fn single_full(tp: usize) -> Self {
        Self {
            pools: vec![PoolSpec {
                kind: WorkerKind::Full,
                count: 1,
                tp: tp.max(1),
            }],
        }
    }

    /// Parse `--workers`, e.g. `encoder:2,prefill:1:tp=4,decode:1:tp=4,sampler:4`.
    /// Each entry is `kind[:count[:tp=N]]`; `count` and `tp` default to 1.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut pools = Vec::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let mut parts = entry.split(':');
            let kind_str = parts.next().unwrap_or("").trim();
            let kind = WorkerKind::from_token(kind_str)
                .ok_or_else(|| anyhow::anyhow!("unknown worker kind {kind_str:?} in --workers"))?;
            let mut count = 1usize;
            let mut tp = 1usize;
            for part in parts {
                let part = part.trim();
                if let Some(tp_str) = part.strip_prefix("tp=") {
                    tp = tp_str
                        .parse::<usize>()
                        .map_err(|_| anyhow::anyhow!("invalid tp in --workers entry {entry:?}"))?
                        .max(1);
                } else {
                    count = part
                        .parse::<usize>()
                        .map_err(|_| anyhow::anyhow!("invalid count in --workers entry {entry:?}"))?
                        .max(1);
                }
            }
            pools.push(PoolSpec { kind, count, tp });
        }
        anyhow::ensure!(!pools.is_empty(), "--workers must list at least one pool");
        Ok(Self { pools })
    }

    /// Whether this is the trivial single-Full-pool topology (the default, which
    /// composes a plain executor rather than a `StageRouter`).
    pub fn is_single_full(&self) -> bool {
        self.pools.len() == 1 && self.pools[0].kind == WorkerKind::Full && self.pools[0].count == 1
    }

    /// Total pool instances (sum of `count` across entries).
    pub fn total_pools(&self) -> usize {
        self.pools.iter().map(|p| p.count).sum()
    }
}

/// Per-edge data-plane transfer backend selection (`--transfer`), e.g.
/// `encoder->prefill=cuda_ipc,prefill->decode=mooncake`. The backend string is
/// passed through to [`make_transfer_agent`](uniserve_worker_ipc_core); the
/// control plane never interprets it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransferSpec {
    /// (producer kind, consumer kind) → backend name.
    pub edges: std::collections::BTreeMap<(WorkerKind, WorkerKind), String>,
}

impl TransferSpec {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        let mut edges = std::collections::BTreeMap::new();
        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("--transfer entry {entry:?} must be edge=backend")
            })?;
            let (src, dst) = edge
                .split_once("->")
                .ok_or_else(|| anyhow::anyhow!("--transfer edge {edge:?} must be src->dst"))?;
            let src = WorkerKind::from_token(src.trim())
                .ok_or_else(|| anyhow::anyhow!("unknown src kind in --transfer {edge:?}"))?;
            let dst = WorkerKind::from_token(dst.trim())
                .ok_or_else(|| anyhow::anyhow!("unknown dst kind in --transfer {edge:?}"))?;
            edges.insert((src, dst), backend.trim().to_string());
        }
        Ok(Self { edges })
    }

    /// Backend for an edge, or `"inproc"` (the in-process zero-transfer default).
    pub fn backend_for(&self, src: WorkerKind, dst: WorkerKind) -> &str {
        self.edges
            .get(&(src, dst))
            .map(String::as_str)
            .unwrap_or("inproc")
    }
}

/// Semantic class of a data-plane tensor. The Host routes on it (e.g.
/// `Embedding`→Prefill, `Logits`→Sampler).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorKind {
    Embedding,
    KvPages,
    Logits,
    Image,
    VideoFrame,
}

/// A control-plane reference to a data-plane tensor. The Host routes by
/// `id`+`kind` and forwards `locator` verbatim, never parsing it — only the
/// consumer worker's `TransferAgent` interprets the locator. The locator is a
/// small (tens of bytes) opaque descriptor encoding `(segment, offset, length,
/// device, dtype, shape, rkey, …)`, so carrying it on the control plane stays
/// descriptor-only (not a tensor payload). With the in-process direct-reference
/// in-process direct-reference data plane the locator is empty and `id` is the
/// existing worker-local handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHandle {
    /// Globally unique id; the control plane routes / dedups / lifetimes by it.
    pub id: u64,
    /// Tensor semantic class; the control plane routes by it.
    pub kind: TensorKind,
    /// Opaque locator produced by the producer worker's `TransferAgent`. The
    /// Host passes it through unparsed; empty in the in-process backend.
    pub locator: Vec<u8>,
}

impl TensorHandle {
    /// A locator-less handle (in-process data plane / degenerate worker-local
    /// reference such as today's `encoder_handle`).
    pub fn local(id: u64, kind: TensorKind) -> Self {
        Self {
            id,
            kind,
            locator: Vec::new(),
        }
    }
}

/// One typed control operation carried over the executor control plane.
#[derive(Debug, Clone)]
pub enum ControlOp {
    DropSession(RequestId),
    CopyKv(Vec<(BlockId, BlockId)>),
    ReleaseProducts(Vec<u64>),
    LoadAdapter { adapter_id: u32, path: String },
    UnloadAdapter { adapter_id: u32 },
    ResetPrefixCache,
    SnapshotSession(RequestId),
    RestoreSession(SnapshotRef),
}

impl ControlOp {
    pub const fn request_kind(&self) -> RequestKind {
        match self {
            Self::DropSession(_) => RequestKind::DropSession,
            Self::CopyKv(_) => RequestKind::CopyKv,
            Self::ReleaseProducts(_) => RequestKind::ReleaseProducts,
            Self::LoadAdapter { .. } => RequestKind::LoadAdapter,
            Self::UnloadAdapter { .. } => RequestKind::UnloadAdapter,
            Self::ResetPrefixCache => RequestKind::ResetPrefixCache,
            Self::SnapshotSession(_) => RequestKind::SnapshotSession,
            Self::RestoreSession(_) => RequestKind::RestoreSession,
        }
    }

    pub fn method(&self) -> &'static str {
        match self {
            Self::DropSession(_) => "drop_session",
            Self::CopyKv(_) => "copy_kv",
            Self::ReleaseProducts(_) => "release_products",
            Self::LoadAdapter { .. } => "load_adapter",
            Self::UnloadAdapter { .. } => "unload_adapter",
            Self::ResetPrefixCache => "reset_prefix_cache",
            Self::SnapshotSession(_) => "snapshot_session",
            Self::RestoreSession(_) => "restore_session",
        }
    }

    /// Reconstruct a payload-free control op from its wire method name.
    ///
    /// This is the `collective_rpc` surface: a method name with no payload, so
    /// only the parameter-free variants are constructible here. Payload-carrying
    /// variants cannot be rebuilt from a bare method name and return `None`.
    pub fn from_method(method: &str) -> Option<Self> {
        match method {
            "reset_prefix_cache" => Some(Self::ResetPrefixCache),
            "drop_session" | "copy_kv" | "release_products" | "load_adapter" | "unload_adapter"
            | "snapshot_session" | "restore_session" => None,
            _ => None,
        }
    }

    pub fn to_request(&self, call_id: u64) -> WorkerRequest {
        let mut req = match self {
            Self::DropSession(id) => WorkerRequest::drop_session(*id),
            Self::CopyKv(copies) => WorkerRequest::copy_kv(copies.clone()),
            Self::ReleaseProducts(handles) => WorkerRequest::release_products(handles.clone()),
            Self::LoadAdapter { adapter_id, path } => {
                WorkerRequest::load_adapter(*adapter_id, path.clone())
            }
            Self::UnloadAdapter { adapter_id } => WorkerRequest::unload_adapter(*adapter_id),
            Self::ResetPrefixCache => WorkerRequest::reset_prefix_cache(),
            Self::SnapshotSession(id) => WorkerRequest::snapshot_session(*id),
            Self::RestoreSession(snapshot) => WorkerRequest::restore_session(snapshot.clone()),
        };
        req.call_id = Some(call_id);
        req
    }
}

/// One rank's acknowledgment of a control call.
#[derive(Debug, Clone)]
pub struct ControlAck {
    pub rank: u32,
    pub ok: bool,
    pub message: Option<String>,
    pub snapshot: Option<SnapshotRef>,
}

/// A worker-reported execution error classified for scheduler failure policy.
///
/// The typed taxonomy and execution context cross the wire together so failure
/// policy and diagnostics use the same operation identity.
#[derive(Debug, Clone)]
pub struct WorkerExecError {
    pub fatal: bool,
    /// Whether retrying the same op could succeed (e.g. transient OOM).
    /// Defaults to `false` when the worker did not classify the error.
    pub retryable: bool,
    pub code: Option<String>,
    pub message: String,
    pub phase: Option<String>,
    pub route: Option<String>,
    pub operations: Vec<uniserve_worker_wire::ErrorOperationIdentity>,
}

impl std::fmt::Display for WorkerExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "worker execute error [{}{}{}; phase={}; route={}; operations={}]: {}",
            self.code.as_deref().unwrap_or("unclassified"),
            if self.fatal { ", fatal" } else { ", non-fatal" },
            if self.retryable { ", retryable" } else { "" },
            self.phase.as_deref().unwrap_or("unknown"),
            self.route.as_deref().unwrap_or("unknown"),
            self.operations.len(),
            self.message,
        )
    }
}

impl std::error::Error for WorkerExecError {}

/// A worker process was replaced without session snapshots, so every session
/// assigned to that executor must terminate explicitly before new work begins.
#[derive(Debug)]
pub struct WorkerLossError {
    pub message: String,
}

impl std::fmt::Display for WorkerLossError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for WorkerLossError {}

/// The asynchronous, pipelined boundary the scheduler drives.
pub trait Executor: Send {
    fn caps(&self) -> EngineCaps;
    fn pipeline_depth(&self) -> usize;
    fn in_flight(&self) -> usize;

    fn can_submit(&self) -> bool {
        self.in_flight() < self.pipeline_depth()
    }

    /// Generated-image commit shapes executable on this topology.
    /// Non-disaggregated pools append image KV inside `commit_gen`.
    fn generated_image_commit_capabilities(&self) -> GeneratedImageCommitCapabilities {
        GeneratedImageCommitCapabilities {
            inline: true,
            separate_writeback: false,
        }
    }

    fn submit(&mut self, batch: Batch) -> anyhow::Result<()>;
    fn poll(&mut self) -> anyhow::Result<Option<ExecutionResult>>;

    /// Data-plane causality gate: whether every data-plane tensor the request's
    /// next op depends on is reachable on the worker that would run it. The
    /// run it. The scheduler consults this in `next_op()` before emitting an op
    /// for `req_id`.
    ///
    /// Non-disaggregated executors (Uniproc/Multiproc) have no inter-stage
    /// transfer, so the default is always `true`. The `StageRouter` overrides it
    /// to report cross-stage transfer readiness from its `TensorMover`.
    fn stage_ready(&self, _req_id: RequestId) -> bool {
        true
    }

    /// Liveness probe independent of in-flight work. Returns `Err` if the
    /// backing worker has died so the scheduler can latch fatal even while idle.
    /// Sim/local executors with no separate worker process always return `Ok`.
    fn check_liveness(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Whether this executor drives an event-driven boundary: a single park
    /// over {result-ready, command, worker-death} instead of fixed-interval poll.
    /// When `true`, the scheduler parks via [`Executor::park_for_event`] and
    /// uses [`Executor::command_waker`] to wake on new commands.
    fn event_driven(&self) -> bool {
        false
    }

    /// Cloneable waker the command ingress fires after enqueuing a command.
    /// Default is a no-op for polling executors.
    fn command_waker(&self) -> CommandWaker {
        CommandWaker::noop()
    }

    /// Block until a result may be ready, a command may have arrived, the worker
    /// died, or `timeout` elapses — without consuming any result. Only called
    /// when [`Executor::event_driven`] is `true`.
    fn park_for_event(&mut self, timeout: Duration) -> anyhow::Result<()> {
        std::thread::sleep(timeout.min(Duration::from_millis(1)));
        Ok(())
    }

    fn wait_result_timeout(
        &mut self,
        timeout: Duration,
    ) -> anyhow::Result<Option<ExecutionResult>> {
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
            std::thread::sleep((deadline - now).min(Duration::from_millis(1)));
        }
    }
    fn next_result(&mut self) -> anyhow::Result<ExecutionResult>;
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
    use uniserve_worker_wire::{MaterializeKind, SequenceMode};

    #[test]
    fn payload_free_control_round_trips() {
        let control = ControlOp::from_method("reset_prefix_cache").unwrap();
        assert_eq!(control.method(), "reset_prefix_cache");
        assert!(ControlOp::from_method("drop_session").is_none());
    }

    #[test]
    fn from_method_rejects_unknown() {
        assert!(ControlOp::from_method("definitely_not_a_method").is_none());
    }

    #[test]
    fn worker_kind_round_trips_and_maps_ops() {
        for kind in [
            WorkerKind::Full,
            WorkerKind::Encoder,
            WorkerKind::Prefill,
            WorkerKind::Decode,
            WorkerKind::Sampler,
            WorkerKind::PostProcess,
            WorkerKind::Und,
            WorkerKind::Gen,
        ] {
            assert_eq!(WorkerKind::from_token(kind.as_str()), Some(kind));
            assert!(!kind.supported_operation_types().is_empty());
        }
        let extend = sequence(SequenceMode::Extend);
        let decode = sequence(SequenceMode::Decode);
        let sample = sequence(SequenceMode::Sample);
        let image = materialize(MaterializeKind::Image);
        let frame = materialize(MaterializeKind::Frame);

        assert!(WorkerKind::Full.handles(&extend));
        assert!(WorkerKind::Prefill.handles(&extend));
        assert!(!WorkerKind::Prefill.handles(&decode));
        assert!(WorkerKind::Decode.handles(&decode));
        assert!(WorkerKind::Sampler.handles(&sample));
        assert!(WorkerKind::Decode.handles(&image));
        assert!(WorkerKind::PostProcess.handles(&frame));
        assert!(WorkerKind::Und.handles(&decode));
        assert!(!WorkerKind::Und.handles(&image));
        assert!(WorkerKind::Gen.handles(&image));
        assert!(!WorkerKind::Gen.handles(&decode));
        assert_eq!(WorkerKind::from_token("nope"), None);
    }

    fn sequence(mode: SequenceMode) -> Operation {
        use uniserve_worker_wire::{
            KvLeaseDelta, PublishedProduct, SequenceInput, SequenceOperation, TokenInput,
            TokenPolicy, TokenSource,
        };
        Operation::Sequence(SequenceOperation {
            mode,
            lease: KvLeaseDelta::default(),
            position: (0, 1),
            policy: TokenPolicy::default(),
            input: if mode == SequenceMode::Sample {
                SequenceInput::PublishedLogits(PublishedProduct {
                    handle: 1,
                    locator: String::new(),
                })
            } else {
                SequenceInput::Tokens(TokenInput {
                    token_ids: vec![1],
                    source: TokenSource::Wire,
                    draft_token_ids: Vec::new(),
                    burst_tokens: 1,
                    stop_token_ids: Vec::new(),
                    stop_terminal: false,
                    return_all_logits: false,
                })
            },
        })
    }

    fn materialize(kind: MaterializeKind) -> Operation {
        use uniserve_worker_wire::{
            KvLeaseDelta, MaterializeInput, MaterializeOperation, PublishedProduct, TokenPolicy,
        };
        Operation::Materialize(MaterializeOperation {
            kind,
            lease: KvLeaseDelta::default(),
            position: 0,
            conditioning_position: 0,
            policy: TokenPolicy::default(),
            input: MaterializeInput::Published(PublishedProduct {
                handle: 1,
                locator: String::new(),
            }),
        })
    }

    #[test]
    fn workers_spec_parses_topologies() {
        assert!(WorkersSpec::single_full(4).is_single_full());
        let epd = WorkersSpec::parse("encoder:2,prefill:1:tp=4,decode:1:tp=4,sampler:4").unwrap();
        assert_eq!(epd.pools.len(), 4);
        assert_eq!(
            epd.pools[0],
            PoolSpec {
                kind: WorkerKind::Encoder,
                count: 2,
                tp: 1
            }
        );
        assert_eq!(
            epd.pools[1],
            PoolSpec {
                kind: WorkerKind::Prefill,
                count: 1,
                tp: 4
            }
        );
        assert_eq!(
            epd.pools[3],
            PoolSpec {
                kind: WorkerKind::Sampler,
                count: 4,
                tp: 1
            }
        );
        assert_eq!(epd.total_pools(), 8);
        assert!(!epd.is_single_full());
        assert!(WorkersSpec::parse("full:1").unwrap().is_single_full());
        assert!(WorkersSpec::parse("bogus:1").is_err());
        assert!(WorkersSpec::parse("").is_err());
        // The und/gen MoT disaggregation topology composes through the general
        // staged path (StageRouter::new over Und/Gen pools).
        let und_gen = WorkersSpec::parse("und:1,gen:1").unwrap();
        assert_eq!(und_gen.pools.len(), 2);
        assert_eq!(und_gen.pools[0].kind, WorkerKind::Und);
        assert_eq!(und_gen.pools[1].kind, WorkerKind::Gen);
        assert!(!und_gen.is_single_full());
    }

    #[test]
    fn transfer_spec_parses_edges_and_defaults_inproc() {
        let t = TransferSpec::parse(
            "encoder->prefill=cuda_ipc,prefill->decode=mooncake,decode->sampler=shm",
        )
        .unwrap();
        assert_eq!(
            t.backend_for(WorkerKind::Encoder, WorkerKind::Prefill),
            "cuda_ipc"
        );
        assert_eq!(
            t.backend_for(WorkerKind::Prefill, WorkerKind::Decode),
            "mooncake"
        );
        // Unconfigured edge falls back to the in-process backend.
        assert_eq!(
            t.backend_for(WorkerKind::Sampler, WorkerKind::Full),
            "inproc"
        );
        assert!(TransferSpec::parse("bad-entry").is_err());
    }
}
