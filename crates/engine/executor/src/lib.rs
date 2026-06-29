//! Executor contracts shared by scheduler, worker IPC, and local engines.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::time::{Duration, Instant};

use uniserve_core::{BlockId, CommandWaker, RequestId};
use uniserve_worker_wire::{EngineCaps, ForwardBatch, ForwardResult, OpKind, WorkerRequest};

/// Synchronous model-engine seam used by deterministic local implementations.
pub trait ModelEngine: Send {
    fn caps(&self) -> EngineCaps;
    fn execute(&mut self, batch: ForwardBatch) -> anyhow::Result<ForwardResult>;
    fn drop_request(&mut self, id: RequestId) -> anyhow::Result<()>;
}

/// Which pipeline stage a worker pool serves.
///
/// A pool is fully determined by three things: the `OpKind` subset it declares
/// (via `caps().supported_ops`, given here by [`WorkerKind::supported_ops`]),
/// the driver that handles that subset, and its device profile. The control
/// plane ([`StageRouter`](../../worker_ipc/struct.StageRouter.html)) routes ops
/// by `OpKind`→pool; `WorkerKind` is the pool's declared role.
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

// Static op-subsets, promoted to `'static` so `supported_ops` can return a
// borrowed slice without allocation.
const FULL_OPS: &[OpKind] = &[
    OpKind::PrefillUnd,
    OpKind::DecodeUnd,
    OpKind::TargetVerifyUnd,
    OpKind::DenoiseGen,
    OpKind::CommitGen,
    OpKind::CommitWriteback,
    OpKind::VaeEncode,
    OpKind::VitEncode,
];
const ENCODER_OPS: &[OpKind] = &[OpKind::VitEncode, OpKind::VaeEncode];
const PREFILL_OPS: &[OpKind] = &[OpKind::PrefillUnd];
const DECODE_OPS: &[OpKind] = &[
    OpKind::DecodeUnd,
    OpKind::TargetVerifyUnd,
    OpKind::DenoiseGen,
    OpKind::CommitGen,
    OpKind::CommitWriteback,
];
const SAMPLER_OPS: &[OpKind] = &[OpKind::Sample];
const POSTPROCESS_OPS: &[OpKind] = &[OpKind::EncodeFrame];
// Understanding/Generation tower split (the `two_role` routing): every model op
// except image generation is "understanding"; denoise/commit/frame-encode is
// "generation".
const UND_OPS: &[OpKind] = &[
    OpKind::PrefillUnd,
    OpKind::DecodeUnd,
    OpKind::TargetVerifyUnd,
    OpKind::VitEncode,
    OpKind::VaeEncode,
    OpKind::Sample,
    OpKind::CommitWriteback,
];
const GEN_OPS: &[OpKind] = &[OpKind::DenoiseGen, OpKind::CommitGen, OpKind::EncodeFrame];

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

    /// The `OpKind` subset this kind handles — the StageRouter's routing key.
    pub fn supported_ops(self) -> &'static [OpKind] {
        match self {
            Self::Full => FULL_OPS,
            Self::Encoder => ENCODER_OPS,
            Self::Prefill => PREFILL_OPS,
            Self::Decode => DECODE_OPS,
            Self::Sampler => SAMPLER_OPS,
            Self::PostProcess => POSTPROCESS_OPS,
            Self::Und => UND_OPS,
            Self::Gen => GEN_OPS,
        }
    }

    /// Whether this kind handles a given op kind.
    pub fn handles(self, op: OpKind) -> bool {
        self.supported_ops().contains(&op)
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
        self.pools.len() == 1
            && self.pools[0].kind == WorkerKind::Full
            && self.pools[0].count == 1
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
            let (edge, backend) = entry
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--transfer entry {entry:?} must be edge=backend"))?;
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
    DropRequest(RequestId),
    CopyBlocks(Vec<(BlockId, BlockId)>),
    FreeEncoder(Vec<u64>),
    LoadLora { lora_id: u32, path: String },
    UnloadLora { lora_id: u32 },
    ResetPrefixCache,
    Sleep,
    WakeUp,
}

impl ControlOp {
    pub fn method(&self) -> &'static str {
        match self {
            Self::DropRequest(_) => "drop_request",
            Self::CopyBlocks(_) => "copy_blocks",
            Self::FreeEncoder(_) => "free_encoder",
            Self::LoadLora { .. } => "load_lora",
            Self::UnloadLora { .. } => "unload_lora",
            Self::ResetPrefixCache => "reset_prefix_cache",
            Self::Sleep => "sleep",
            Self::WakeUp => "wake_up",
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
            "sleep" => Some(Self::Sleep),
            "wake_up" => Some(Self::WakeUp),
            // Payload-carrying ops require their arguments.
            "drop_request" | "copy_blocks" | "free_encoder" | "load_lora"
            | "unload_lora" => None,
            _ => None,
        }
    }

    pub fn to_request(&self, call_id: u64) -> WorkerRequest {
        let mut req = match self {
            Self::DropRequest(id) => WorkerRequest::drop_request(*id),
            Self::CopyBlocks(copies) => WorkerRequest::copy_blocks(copies.clone()),
            Self::FreeEncoder(handles) => WorkerRequest::free_encoder(handles.clone()),
            Self::LoadLora { lora_id, path } => WorkerRequest::load_lora(*lora_id, path.clone()),
            Self::UnloadLora { lora_id } => WorkerRequest::unload_lora(*lora_id),
            Self::ResetPrefixCache => WorkerRequest::reset_prefix_cache(),
            Self::Sleep => WorkerRequest::sleep(),
            Self::WakeUp => WorkerRequest::wake_up(),
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
}

/// A worker-reported execution error classified for scheduler failure policy.
///
/// The typed taxonomy crosses the wire as `(code, retryable, fatal)`. All three
/// fields are carried here so the scheduler can consult error class and
/// retryability, not only the `fatal` flag.
#[derive(Debug, Clone)]
pub struct WorkerExecError {
    pub fatal: bool,
    /// Whether retrying the same op could succeed (e.g. transient OOM).
    /// Defaults to `false` when the worker did not classify the error.
    pub retryable: bool,
    pub code: Option<String>,
    pub message: String,
}

impl std::fmt::Display for WorkerExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "worker execute error [{}{}{}]: {}",
            self.code.as_deref().unwrap_or("unclassified"),
            if self.fatal { ", fatal" } else { ", non-fatal" },
            if self.retryable { ", retryable" } else { "" },
            self.message,
        )
    }
}

impl std::error::Error for WorkerExecError {}

/// The asynchronous, pipelined boundary the scheduler drives.
pub trait Executor: Send {
    fn caps(&self) -> EngineCaps;
    fn pipeline_depth(&self) -> usize;
    fn in_flight(&self) -> usize;

    fn can_submit(&self) -> bool {
        self.in_flight() < self.pipeline_depth()
    }

    fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()>;
    fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>>;

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

    fn wait_result_timeout(&mut self, timeout: Duration) -> anyhow::Result<Option<ForwardResult>> {
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
    fn next_result(&mut self) -> anyhow::Result<ForwardResult>;
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

 /// Whether a variant is reachable via the name-only `collective_rpc`
 /// surface (`from_method`). The exhaustive `match` over `ControlOp` is the
 /// drift guard: adding a variant forces a deliberate classification here,
 /// which keeps `method` and `from_method` in lockstep.
    fn payload_free(op: &ControlOp) -> bool {
        match op {
            ControlOp::ResetPrefixCache | ControlOp::Sleep | ControlOp::WakeUp => true,
            ControlOp::DropRequest(_)
            | ControlOp::CopyBlocks(_)
            | ControlOp::FreeEncoder(_)
            | ControlOp::LoadLora { .. }
            | ControlOp::UnloadLora { .. } => false,
        }
    }

    #[test]
    fn from_method_round_trips() {
 // One instance of every variant. The exhaustive match in
 // `payload_free` guarantees this list stays complete.
        let variants = [
            ControlOp::DropRequest(RequestId(0)),
            ControlOp::CopyBlocks(vec![(BlockId(0), BlockId(1))]),
            ControlOp::FreeEncoder(vec![0]),
            ControlOp::LoadLora {
                lora_id: 0,
                path: String::new(),
            },
            ControlOp::UnloadLora { lora_id: 0 },
            ControlOp::ResetPrefixCache,
            ControlOp::Sleep,
            ControlOp::WakeUp,
        ];

        for op in &variants {
            let method = op.method();
            let rebuilt = ControlOp::from_method(method);
            if payload_free(op) {
                assert!(
                    matches!(&rebuilt, Some(r) if r.method() == method),
                    "payload-free op `{method}` must round-trip through from_method"
                );
            } else {
                assert!(
                    rebuilt.is_none(),
                    "payload-carrying op `{method}` must not be constructible from a bare method name"
                );
            }
        }
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
            assert!(!kind.supported_ops().is_empty());
        }
        assert!(WorkerKind::Full.handles(OpKind::DenoiseGen));
        assert!(WorkerKind::Encoder.handles(OpKind::VitEncode));
        assert!(!WorkerKind::Encoder.handles(OpKind::DecodeUnd));
        assert!(WorkerKind::Sampler.handles(OpKind::Sample));
        assert!(WorkerKind::PostProcess.handles(OpKind::EncodeFrame));
        // Und/Gen tower split: understanding ops on und, generation ops on gen,
        // disjoint and exhaustive over the model ops (the `two_role` routing).
        assert!(WorkerKind::Und.handles(OpKind::PrefillUnd));
        assert!(WorkerKind::Und.handles(OpKind::DecodeUnd));
        assert!(WorkerKind::Und.handles(OpKind::VitEncode));
        assert!(WorkerKind::Und.handles(OpKind::Sample));
        assert!(!WorkerKind::Und.handles(OpKind::DenoiseGen));
        assert!(WorkerKind::Gen.handles(OpKind::DenoiseGen));
        assert!(WorkerKind::Gen.handles(OpKind::CommitGen));
        assert!(!WorkerKind::Gen.handles(OpKind::DecodeUnd));
        assert_eq!(WorkerKind::from_token("nope"), None);
    }

    /// Drift guard for the canonical worker-kind vocabulary
    /// (`crates/protocol/vocab/worker_kinds.toml`): the Rust `WorkerKind` enum and
    /// its `supported_ops` arrays must equal the schema exactly — same tokens,
    /// same variant identifiers, and the same OpKind-subset (by wire name) for
    /// each kind. Adding a kind or moving an op requires updating the schema
    /// AND every language, or this fails.
    #[test]
    fn worker_kind_vocabulary_matches_canonical_schema() {
        const SCHEMA: &str = include_str!("../../../protocol/vocab/worker_kinds.toml");
        let schema: toml::Value = toml::from_str(SCHEMA).unwrap();

        // OpKind -> wire string (snake_case). An exhaustive match so a new op
        // kind fails to compile here. Mirrors the serde snake_case form pinned in
        // worker-wire's own op-kind drift guard.
        fn op_wire(op: OpKind) -> &'static str {
            match op {
                OpKind::PrefillUnd => "prefill_und",
                OpKind::DecodeUnd => "decode_und",
                OpKind::TargetVerifyUnd => "target_verify_und",
                OpKind::DenoiseGen => "denoise_gen",
                OpKind::CommitGen => "commit_gen",
                OpKind::CommitWriteback => "commit_writeback",
                OpKind::VaeEncode => "vae_encode",
                OpKind::VitEncode => "vit_encode",
                OpKind::Sample => "sample",
                OpKind::EncodeFrame => "encode_frame",
            }
        }

        // Every `WorkerKind` variant, enumerated via an exhaustive match so a new
        // variant fails to compile here until it is added to this list and the
        // schema.
        fn variant_ident(k: WorkerKind) -> &'static str {
            match k {
                WorkerKind::Full => "Full",
                WorkerKind::Encoder => "Encoder",
                WorkerKind::Prefill => "Prefill",
                WorkerKind::Decode => "Decode",
                WorkerKind::Sampler => "Sampler",
                WorkerKind::PostProcess => "PostProcess",
                WorkerKind::Und => "Und",
                WorkerKind::Gen => "Gen",
            }
        }
        let variants = [
            WorkerKind::Full,
            WorkerKind::Encoder,
            WorkerKind::Prefill,
            WorkerKind::Decode,
            WorkerKind::Sampler,
            WorkerKind::PostProcess,
            WorkerKind::Und,
            WorkerKind::Gen,
        ];

        let kinds = schema["kind"].as_array().expect("schema [[kind]] array");

        // Token set + (token -> entry) lookup.
        let schema_tokens: std::collections::BTreeSet<String> = kinds
            .iter()
            .map(|k| k["token"].as_str().unwrap().to_string())
            .collect();
        let rust_tokens: std::collections::BTreeSet<String> =
            variants.iter().map(|k| k.as_str().to_string()).collect();
        assert_eq!(
            rust_tokens, schema_tokens,
            "WorkerKind token set drifted from worker_kinds.toml"
        );

        for k in variants {
            let token = k.as_str();
            let entry = kinds
                .iter()
                .find(|e| e["token"].as_str() == Some(token))
                .unwrap_or_else(|| panic!("no schema entry for token {token:?}"));

            // Variant identifier pinned.
            assert_eq!(
                entry["rust"].as_str(),
                Some(variant_ident(k)),
                "worker_kinds.toml `rust` column for {token:?} drifted"
            );

            // supported_ops set (compared as a set; order is not significant).
            let schema_ops: std::collections::BTreeSet<String> = entry["supported_ops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            let rust_ops: std::collections::BTreeSet<String> = k
                .supported_ops()
                .iter()
                .map(|op| op_wire(*op).to_string())
                .collect();
            assert_eq!(
                rust_ops, schema_ops,
                "WorkerKind::{token} supported_ops drifted from worker_kinds.toml"
            );
        }
    }

    #[test]
    fn workers_spec_parses_topologies() {
        assert!(WorkersSpec::single_full(4).is_single_full());
        let epd = WorkersSpec::parse("encoder:2,prefill:1:tp=4,decode:1:tp=4,sampler:4").unwrap();
        assert_eq!(epd.pools.len(), 4);
        assert_eq!(epd.pools[0], PoolSpec { kind: WorkerKind::Encoder, count: 2, tp: 1 });
        assert_eq!(epd.pools[1], PoolSpec { kind: WorkerKind::Prefill, count: 1, tp: 4 });
        assert_eq!(epd.pools[3], PoolSpec { kind: WorkerKind::Sampler, count: 4, tp: 1 });
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
        assert_eq!(t.backend_for(WorkerKind::Encoder, WorkerKind::Prefill), "cuda_ipc");
        assert_eq!(t.backend_for(WorkerKind::Prefill, WorkerKind::Decode), "mooncake");
        // Unconfigured edge falls back to the in-process backend.
        assert_eq!(t.backend_for(WorkerKind::Sampler, WorkerKind::Full), "inproc");
        assert!(TransferSpec::parse("bad-entry").is_err());
    }
}
