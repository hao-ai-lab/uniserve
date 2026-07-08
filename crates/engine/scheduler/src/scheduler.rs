//! Scheduler control loop: one owner thread drives the whole loop (single-owner,
//! no locks on engine state) — drain commands, advance each running request's
//! interleave FSM, admit pending requests against the block and scratch budget,
//! assemble a `ForwardBatch`, submit it asynchronously through the `Executor`,
//! and resolve completed `ForwardResult`s into `GenEvent`s and FSM transitions.
//! `ForwardBatch` assembly is lane-aware for text prefill/decode and may still
//! mix compatible non-text ops; workers preserve one result per submitted op.
//!
//! Scheduling follows vLLM-style budgeted, chunked-prefill, preempting scheduling:
//! - waiting requests live in a [`RequestQueue`] (FCFS deque or priority
//!   ordering) and admission consumes the queue head;
//! - scheduling is bounded by the `max_num_batched_tokens` / `max_num_seqs`
//!   pair with vLLM's clip rule (`min(num_new_tokens, token_budget)`);
//! - worst-case reservation is a per-request admission attribute: image and
//!   interleave requests allocate their worst-case KV at admission and are never
//!   preempted; text requests are budgeted and preemptible.
//!
//! The worker contract is a stateful diff: a request's static state crosses once
//! as [`NewRequestData`]; per-step ops carry only deltas (new block ids, new
//! tokens, per-step masks). Preemption resets the diff state (`drop_request` plus
//! re-registration on resumption).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;

pub const MAX_SPEC_DECODE_POS_STATS: usize = 16;
pub const DEFAULT_MAX_BATCH: usize = 128;
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
pub const DEFAULT_MAX_NUM_SEQS: usize = 128;
pub const DEFAULT_LONG_PREFILL_THRESHOLD: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
/// Default per-step budget of text prefill tokens that may ride along inside a
/// decode batch (a mixed extend+decode forward shares the decode step's weight
/// sweep instead of paying its own). `0` disables mixing — matching SGLang's
/// `enable_mixed_chunk` default: its reference serving configuration keeps
/// prefill and decode in separate batches.
pub const DEFAULT_MIXED_PREFILL_TOKENS: usize = 0;
pub const DEFAULT_DENOISE_STEP_BURST: u16 = 1;
pub const DEFAULT_DECODE_TOKEN_BURST: u16 = 1;
/// Default admission backpressure bound: maximum waiting requests buffered
/// before new submits are rejected at enqueue.
pub const DEFAULT_MAX_NUM_WAITING: usize = 4096;

/// General loop counters that do not belong to a more specific group.
#[derive(Default)]
pub struct GeneralStats {
    pub peak_ops: AtomicUsize,
    pub steps: AtomicU64,
    pub running: AtomicUsize,
    pub pending: AtomicUsize,
    /// Requests gated on grammar compilation (vLLM's skipped_waiting).
    pub skipped_waiting: AtomicUsize,
    pub preemptions: AtomicU64,
    pub in_flight: AtomicUsize,
}

/// KV block-cache observability.
#[derive(Default)]
pub struct KvCacheStats {
    pub free_blocks: AtomicUsize,
    pub num_blocks: AtomicUsize,
    pub blocks_evicted: AtomicU64,
    pub blocks_stored: AtomicU64,
    pub cached_blocks: AtomicUsize,
}

/// Prefix-cache hit-rate counters.
#[derive(Default)]
pub struct PrefixStats {
    pub queries: AtomicU64,
    pub hits: AtomicU64,
    pub hit_tokens: AtomicU64,
}

/// Encoder-output (multimodal) cache counters.
#[derive(Default)]
pub struct EncoderStats {
    pub cache_queries: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cached: AtomicUsize,
}

/// Resource-plane observability: active per-request leases (0 when idle) and
/// any resource-invariant violations (leases retained after a request left).
#[derive(Default)]
pub struct ResourceStats {
    pub active: AtomicUsize,
    pub invariant_violations: AtomicU64,
}

/// Batch-timing and queueing/admission latency counters.
#[derive(Default)]
pub struct TimingStats {
    /// last worker-reported batch compute time (us).
    pub last_worker_exec_us: AtomicU64,
    /// Cumulative batch-timing counters for worker compute and round-trip latency.
    pub worker_exec_us_total: AtomicU64,
    pub batch_roundtrip_us_total: AtomicU64,
    pub batch_timing_count: AtomicU64,
    /// Queueing/admission latency, recorded when a pending request is admitted.
    pub queue_wait_count: AtomicU64,
    pub queue_wait_us_total: AtomicU64,
    pub queue_wait_us_max: AtomicU64,
}

/// Speculative-decode accounting (drafter is default-off).
#[derive(Default)]
pub struct SpecDecodeStats {
    pub max_draft_tokens: AtomicUsize,
    pub num_drafts: AtomicU64,
    pub num_draft_tokens: AtomicU64,
    pub num_accepted_tokens: AtomicU64,
    pub num_accepted_tokens_per_pos: [AtomicU64; MAX_SPEC_DECODE_POS_STATS],
}

/// Worker-local forward/kernel counters, aggregated as completed batches
/// resolve so OpenMetrics can explain scheduler vs worker/kernel latency.
#[derive(Default)]
pub struct WorkerStats {
    pub forward_mode_counts: Mutex<BTreeMap<String, u64>>,
    pub forward_mode_tokens: Mutex<BTreeMap<String, u64>>,
    pub forward_mode_us: Mutex<BTreeMap<String, u64>>,
    pub forward_component_us: Mutex<BTreeMap<String, u64>>,
    pub attention_launches: AtomicU64,
    pub attention_us: AtomicU64,
    /// Per-backend attention launch counts, aggregated from worker forward stats.
    pub attention_backend_counts: Mutex<BTreeMap<String, u64>>,
    pub cuda_graph_captures: AtomicU64,
    pub cuda_graph_replays: AtomicU64,
    pub cuda_graph_misses: AtomicU64,
    pub cuda_graph_fallbacks: AtomicU64,
    pub cuda_graph_unpadded_tokens: AtomicU64,
    pub cuda_graph_padded_tokens: AtomicU64,
    /// Per-runtime-mode CUDA-graph dispatch counts from worker forward stats.
    pub cuda_graph_runtime_mode_counts: Mutex<BTreeMap<String, u64>>,
    pub text_decode_token_relay_hits: AtomicU64,
    pub text_decode_token_relay_misses: AtomicU64,
    pub text_decode_position_relay_hits: AtomicU64,
    pub text_decode_position_relay_misses: AtomicU64,
    pub flashinfer_decode_plan_calls: AtomicU64,
    pub flashinfer_decode_plan_reuses: AtomicU64,
    pub flashinfer_decode_plan_rows: AtomicU64,
    pub flashinfer_decode_plan_indices: AtomicU64,
    pub flashinfer_decode_graph_plan_calls: AtomicU64,
    pub flashinfer_decode_graph_plan_reuses: AtomicU64,
    pub spec_verify_rows: AtomicU64,
    pub spec_verify_draft_tokens: AtomicU64,
    pub spec_verify_accepted_tokens: AtomicU64,
    pub spec_verify_rejected_tokens: AtomicU64,
    pub spec_verify_committed_tokens: AtomicU64,
    pub spec_verify_path_counts: Mutex<BTreeMap<String, u64>>,
}

/// Live scheduler stats, shared with the frontend for `/stats` observability.
#[derive(Default)]
pub struct SchedStats {
    pub general: GeneralStats,
    pub kv_cache: KvCacheStats,
    pub prefix: PrefixStats,
    pub encoder: EncoderStats,
    pub resources: ResourceStats,
    pub timing: TimingStats,
    pub spec_decode: SpecDecodeStats,
    pub worker: WorkerStats,
}

use crossbeam_channel::Receiver;
use uniserve_core::{BlockId, CfgParams, Modality};
use uniserve_core::{GenMode, HashAlgo, RequestId};
use uniserve_engine_api::{Command, FinishReason, GenEvent, GenerateRequest};
use uniserve_kv::{BlockManager, EncoderCacheManager};
use uniserve_program::InferenceProgram;
use uniserve_worker_wire::{
    EngineCaps, ForwardBatch, ForwardOp, ForwardResult, NewRequestData, OpKind, ResourceClass,
    TokenSource, WorkerForwardStats,
};

use crate::grammar::{GrammarCompiler, GrammarMatcher};
use crate::queue::{FcfsRequestQueue, PriorityRequestQueue, RequestQueue};
#[cfg(test)]
use crate::spec_decode::ngram_draft_one;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use uniserve_executor::{ControlOp, Executor, WorkerExecError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssemblyLane {
    Prefill,
    Decode,
    Other,
}

/// default per-step multimodal encode budget when the worker caps do not pin it.
const DEFAULT_MM_ENCODE_BUDGET: usize = 4;
/// default encoder-output cache capacity when the worker reports no
/// `encoder_cache_budget` in its caps.
const DEFAULT_ENCODER_CACHE_BUDGET: usize = 256;
/// default bound on the recent-output window carried for penalties.
const DEFAULT_PENALTY_WINDOW: usize = 256;
const SCHEDULER_WAIT_SLICE: Duration = Duration::from_millis(1);
/// maximum time the fully-idle scheduler blocks on the command channel
/// before waking to probe worker liveness. Bounds how long a worker that dies
/// while the engine is idle stays undetected (the next request would otherwise
/// be the first to notice). Small enough for prompt death detection, large
/// enough that the idle engine is effectively asleep.
const IDLE_LIVENESS_POLL: Duration = Duration::from_millis(500);
const DECODE_LOOKAHEAD_ENV: &str = "UNISERVE_DECODE_LOOKAHEAD";
const DENOISE_STEP_BURST_ENV: &str = "UNISERVE_DENOISE_STEP_BURST";
const DECODE_TOKEN_BURST_ENV: &str = "UNISERVE_DECODE_TOKEN_BURST";

fn image_done_event(image_id: u32, pixels_png_b64: String) -> GenEvent {
    let (height, width, bytes, sha256) =
        image_png_metadata(&pixels_png_b64).unwrap_or_else(|| (0, 0, 0, String::new()));
    GenEvent::ImageDone {
        image_id,
        height,
        width,
        bytes,
        sha256,
        pixels_png_b64,
    }
}

fn image_png_metadata(pixels_png_b64: &str) -> Option<(u32, u32, u64, String)> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(pixels_png_b64.as_bytes())
        .ok()?;
    let width = png_u32_be(&bytes, 16)?;
    let height = png_u32_be(&bytes, 20)?;
    if bytes.get(0..8)? != b"\x89PNG\r\n\x1a\n" || bytes.get(12..16)? != b"IHDR" {
        return None;
    }
    let sha256 = Sha256::digest(&bytes);
    Some((height, width, bytes.len() as u64, format!("{sha256:x}")))
}

fn png_u32_be(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    Some(u32::from_be_bytes(slice))
}

#[derive(Clone)]
pub struct ControlTokens {
    pub bos: u32,
    pub eos: Vec<u32>,
    pub start_of_image: u32,
    pub end_of_image: u32,
    /// Token-id subsequence for the `<image_start>` trigger. Empty disables
    /// the literal trigger.
    #[allow(dead_code)]
    pub image_start_ids: Vec<u32>,
}

impl Default for ControlTokens {
    fn default() -> Self {
        Self {
            bos: 151644,
            eos: vec![151645, 151643],
            start_of_image: 151652,
            end_of_image: 151653,
            image_start_ids: Vec::new(),
        }
    }
}

/// Waiting-queue ordering policy (vLLM's `scheduler_config.policy`): FCFS or
/// priority. Both are budgeted, chunked-prefill, preempting schedulers;
/// [`ReqState::reserve_worstcase`] toggles worst-case reservation per request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingPolicy {
    #[default]
    Fcfs,
    Priority,
}

/// First-class scheduling budgets (vLLM's `max_num_batched_tokens` /
/// `max_num_seqs` pair), decoupled from the engine's `max_batch` op cap.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// Waiting-queue ordering policy.
    pub policy: SchedulingPolicy,
    /// Maximum ops assembled into a single forward batch (engine constraint).
    pub max_batch: usize,
    /// Per-step token budget across all scheduled ops; prefill chunks are
    /// clipped to the remaining budget (the binding constraint).
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests.
    pub max_num_seqs: usize,
    /// Additional per-request cap on one prefill chunk, so a single giant
    /// prompt cannot monopolize a step even within the budget. `usize::MAX`
    /// disables it (the budget binds).
    pub long_prefill_threshold: usize,
    /// Admission backpressure — maximum waiting requests (pending +
    /// grammar-gated) buffered before new submits are rejected at enqueue.
    /// `usize::MAX` disables the cap (unbounded).
    pub max_num_waiting: usize,
    /// Per-step budget of text prefill tokens allowed to join a decode batch
    /// as a mixed extend+decode forward. `0` keeps prefill and decode in
    /// separate batches.
    pub mixed_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            max_num_waiting: DEFAULT_MAX_NUM_WAITING,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Phase {
    /// Encode staged input images (VitEncode) before prefill.
    Encode,
    Prefill,
    DecodeUnd,
    DenoiseGen,
    CommitGen,
    CommitWriteback,
}

pub struct ReqState {
    pub req: GenerateRequest,
    pub(crate) phase: Phase,
    pub(crate) pos: u32,
    pub(crate) next_token: u32,
    pub(crate) n_generated: usize,
    pub(crate) worstcase_blocks: usize,
    /// Reservation as a per-request admission attribute: image-only and
    /// image-understanding requests allocate their full worst-case KV at
    /// admission; auto-interleave allocates its bounded retained-image envelope
    /// up front and grows text KV incrementally. Reserving requests are exempt
    /// from preemption.
    pub(crate) reserve_worstcase: bool,
    pub(crate) image_id: u32,
    pub(crate) images_done: usize,
    pub(crate) text_since_image: usize,
    /// AutoInterleave sampled an image boundary but cannot reserve the image
    /// span yet. The request stays resident and must not decode more text until
    /// the reservation succeeds.
    pub(crate) auto_image_pending: bool,
    pub(crate) cond_pos: u32,
    pub(crate) steps_done: u16,
    pub queued_at: f64,
    pub(crate) cancelled: bool,
    /// The cancel was a server-side abort, not a client cancel.
    pub(crate) aborted: bool,
    /// Chunked prefill: how many prompt tokens have been prefilled so far.
    pub(crate) prompt_cursor: u32,
    /// Preemption: this request was preempted and must recompute from `prompt_cursor`.
    pub(crate) preempted: bool,
    /// Per-full-prompt-block prefix hashes (computed at admission).
    pub(crate) block_hashes: Vec<u64>,
    /// How many leading prompt blocks were reused from the prefix cache.
    pub(crate) prefix_cached_blocks: usize,
    /// Whether this request's blocks have been cached after prefill.
    pub(crate) blocks_cached: bool,
    /// Text tokens emitted so far (needed to recompute KV after preemption
    /// without re-streaming them to the client).
    pub(crate) generated_ids: Vec<u32>,
    /// Preemption-recompute: when set, prefill rebuilds KV over this
    /// sequence (`prompt ++ generated`) instead of the bare prompt.
    pub(crate) recompute_ids: Option<Vec<u32>>,
    /// Index of the next staged image to encode.
    pub(crate) mm_cursor: usize,
    /// Encoder-cache hashes this request has pinned (released on finish).
    pub(crate) mm_acquired: Vec<u64>,
    /// Structured-output matcher (compiled grammar and progress).
    pub(crate) grammar: Option<GrammarMatcher>,
    // ---- stateful-diff contract ----
    /// Whether the worker holds this request's control record (NewRequestData
    /// delivered). Reset on preemption (`drop_request` clears the record).
    pub(crate) worker_registered: bool,
    /// How many of this request's blocks have crossed the wire; per-op deltas
    /// are `blocks_for(id)[blocks_sent..]`.
    pub(crate) blocks_sent: usize,
    // ---- image-understanding interleave (ThinkMorph) ----
    /// KV write head (= worker `self.lengths`), distinct from `pos` (rope) once
    /// image blocks (N+2 KV positions, 1 rope position) are spliced in.
    pub(crate) kvlen: u32,
    /// Dual-encode sub-step for the current input image (0 = VAE next, 1 = ViT next).
    pub(crate) iu_encode_step: u8,
    /// Generated-image dimensions (the VAE-resized input dims), from the worker.
    pub(crate) gen_h: u32,
    pub(crate) gen_w: u32,
    /// Tokens of the current text round (for the literal `<image_start>` trigger).
    pub(crate) round_tokens: Vec<u32>,
    /// True while feeding the closing `<|im_end|>` into KV before a round transition.
    pub(crate) iu_closing: bool,
    /// The typed op-graph compiled from this
    /// request. Host-internal IR alongside the FSM; travels + drops with the
    /// request, so no manual cleanup. The FSM stays authoritative for execution.
    pub(crate) program: InferenceProgram,
    /// This request's lifecycle trace.
    pub(crate) trace: crate::trace::RequestTrace,
    /// Worker-side image latent units currently resident for this request. This
    /// mirrors the Python runner's ResourceRuntime, which acquires the latent on
    /// the first denoise op and releases it at commit.
    pub(crate) worker_image_latent_units: u64,
}

pub struct Scheduler {
    executor: Box<dyn Executor>,
    caps: EngineCaps,
    bm: BlockManager,
    ctrl: ControlTokens,
    config: SchedulerConfig,
    /// Total allocatable blocks (the empty pool's free count) — the structural
    /// admission/rejection bound.
    usable_blocks: usize,
    /// Automatic prefix caching: toggle, hash config, lookup, and reuse.
    prefix_cache: crate::prefix_cache::PrefixCacheCoordinator,
    /// When sleeping, admission is paused (engine idled).
    sleeping: bool,
    /// Pluggable host-side logits-processor pipeline.
    logits_pipeline: Vec<Box<dyn crate::logits::LogitsProcessor>>,
    /// Cap on the recent-output window carried for penalties .
    penalty_window: usize,
    /// Encoder-output cache (hashed, LRU, budgeted).
    enc_cache: EncoderCacheManager,
    /// Max image-encode ops admitted per step.
    mm_encode_budget: usize,
    running: HashMap<RequestId, ReqState>,
    order: Vec<RequestId>, // stable iteration order
    pending: Box<dyn RequestQueue>,
    /// The structured-output gate: requests whose grammar is still compiling
    /// (vLLM's skipped_waiting). Admission drains the compiler each pass and
    /// only then queues these.
    skipped_waiting: HashMap<RequestId, ReqState>,
    grammar_compiler: GrammarCompiler,
    reserved_blocks: usize,
    step_id: u64,
    /// Per-request ops currently in flight (submitted, not yet resolved). A
    /// request may have multiple queued decode ops when FutureMap-style
    /// lookahead is safe. Each [`InflightOp`] carries the op's `op_id`, so a
    /// resolving result is matched to its exact op by `op_id` rather than the
    /// FIFO submission order; the FIFO
    /// front is only a fallback when the worker did not echo an op_id.
    inflight_ops: HashMap<RequestId, VecDeque<InflightOp>>,
    /// Decode lookahead toggle. The fast path is additionally gated per request
    /// to plain text generation whose next logits processors do not depend on
    /// an unknown sampled token.
    decode_lookahead: bool,
    /// Sequential denoise timesteps to execute per denoise op. The worker runs
    /// the exact same Euler steps and reports the cumulative step cursor.
    denoise_step_burst: u16,
    /// Sequential greedy text decode tokens to execute per decode op when the
    /// request has token-independent sampling/masking constraints.
    decode_token_burst: u16,
    /// Default-off n-gram drafter and per-position acceptance accounting. The
    /// worker target-verifies the drafts and resolve commits only the accepted
    /// prefix.
    spec_decode: crate::spec_decode::SpecDecodeAccounting,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    /// Per-request lease accounting and invariant checks (observe-only).
    ledger: crate::resources::ResourceLedger,
    /// Explainable policy decisions and per-op latency history.
    decisions: crate::policy::DecisionLog,
    latency: crate::policy::LatencyHistory,
    /// Submit timestamp per in-flight batch (for batch round-trip traces).
    batch_started: HashMap<u64, Instant>,
    /// Monotonic op/program ids and archived lifecycle traces.
    next_op_id: u64,
    next_program_id: u64,
    completed_traces: VecDeque<crate::trace::RequestTrace>,
    trace_sink: Option<crate::bench_trace::SchedulerTraceSink>,
    pub peak_ops_in_batch: usize,
    pub stats: Arc<SchedStats>,
}

/// A self-describing health snapshot.
#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub running: usize,
    pub pending: usize,
    pub in_flight: usize,
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub reserved_blocks: usize,
    pub active_leases: usize,
    pub resource_invariant_violations: u64,
    pub completed_traces: usize,
    pub last_worker_exec_us: u64,
    pub queue_wait_count: u64,
    pub queue_wait_us_total: u64,
    pub queue_wait_us_max: u64,
    pub fatal: bool,
    pub supported_ops: Vec<String>,
    pub op_latency_us: Vec<(String, u64)>,
}

fn now() -> f64 {
    // route through the single shared epoch helper so every
    // component's wall-clock timestamps match. It never panics on the hot loop:
    // a wall clock set before the UNIX epoch (or stepped backward) clamps to 0
    // instead of unwrapping the `Result`.
    uniserve_core::now_unix_secs()
}

fn add_worker_forward_map(target: &Mutex<BTreeMap<String, u64>>, delta: &BTreeMap<String, u64>) {
    if delta.is_empty() {
        return;
    }
    let mut target = target
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (key, value) in delta {
        *target.entry(key.clone()).or_default() += *value;
    }
}

fn make_queue(policy: SchedulingPolicy) -> Box<dyn RequestQueue> {
    match policy {
        SchedulingPolicy::Fcfs => Box::new(FcfsRequestQueue::default()),
        SchedulingPolicy::Priority => Box::new(PriorityRequestQueue::default()),
    }
}

/// Stable finish-reason label for lifecycle traces.
fn finish_reason_str(r: &FinishReason) -> &'static str {
    match r {
        FinishReason::Eos => "eos",
        FinishReason::Stop => "stop",
        FinishReason::MaxTokens => "max_tokens",
        FinishReason::ImageDone => "image_done",
        FinishReason::Cancelled => "cancelled",
        FinishReason::Aborted => "aborted",
        FinishReason::Error => "error",
    }
}

/// Stable op-kind label for latency-history keys.
fn opkind_str(k: OpKind) -> &'static str {
    match k {
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

fn phase_str(phase: Phase) -> &'static str {
    match phase {
        Phase::Encode => "encode",
        Phase::Prefill => "prefill",
        Phase::DecodeUnd => "decode_und",
        Phase::DenoiseGen => "denoise_gen",
        Phase::CommitGen => "commit_gen",
        Phase::CommitWriteback => "commit_writeback",
    }
}

fn mode_str(mode: GenMode) -> &'static str {
    match mode {
        GenMode::Text => "text",
        GenMode::Image => "image",
        GenMode::AutoInterleave => "auto_interleave",
        GenMode::InterleaveUnd => "interleave_und",
    }
}

fn assembly_lane_for_kind(kind: OpKind) -> AssemblyLane {
    match kind {
        OpKind::PrefillUnd | OpKind::VitEncode | OpKind::VaeEncode => AssemblyLane::Prefill,
        OpKind::DecodeUnd | OpKind::TargetVerifyUnd => AssemblyLane::Decode,
        OpKind::DenoiseGen
        | OpKind::CommitGen
        | OpKind::CommitWriteback
        | OpKind::Sample
        | OpKind::EncodeFrame => AssemblyLane::Other,
    }
}

fn policy_str(policy: SchedulingPolicy) -> &'static str {
    match policy {
        SchedulingPolicy::Fcfs => "fcfs",
        SchedulingPolicy::Priority => "priority",
    }
}

/// One submitted-but-unresolved op tracked per request. Bundles everything the
/// resolve path needs so completion can be correlated by `op_id` rather
/// than by parallel FIFO queues.
struct InflightOp {
    kind: OpKind,
    /// The op's wire `op_id`, echoed back on its result. `None` for transports
    /// that do not stamp it (then resolution falls back to FIFO order).
    op_id: Option<u64>,
    /// Speculative draft token ids attached to this op (empty when none).
    spec_tokens: Vec<u32>,
    /// Sequential text decode tokens requested by this op. Values above one
    /// make dependent decode lookahead unsafe until the op resolves.
    decode_token_count: u16,
    /// Submit timestamp, for the op's host round-trip latency history.
    started: Instant,
}

/// Remove the in-flight op whose `op_id` matches the worker-echoed `op_id`,
/// falling back to the FIFO front when the worker did not echo one (or it is
/// unknown). This is the op_id-correlated completion of: each op
/// resolves its exact submission even if a request's ops complete out of order
/// (decode-priority lanes, disagg fan-in), instead of assuming submission order.
fn take_inflight_by_op_id(
    queue: &mut VecDeque<InflightOp>,
    op_id: Option<u64>,
) -> Option<InflightOp> {
    if let Some(target) = op_id
        && let Some(pos) = queue.iter().position(|op| op.op_id == Some(target))
    {
        return queue.remove(pos);
    }
    queue.pop_front()
}

impl Scheduler {
    pub fn new(executor: Box<dyn Executor>, ctrl: ControlTokens, max_batch: usize) -> Self {
        Self::with_config(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch,
                ..Default::default()
            },
        )
    }

    pub fn with_policy(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
        max_batch: usize,
        policy: SchedulingPolicy,
    ) -> Self {
        Self::with_config(
            executor,
            ctrl,
            SchedulerConfig {
                max_batch,
                policy,
                ..Default::default()
            },
        )
    }

    pub fn with_config(
        executor: Box<dyn Executor>,
        ctrl: ControlTokens,
        mut config: SchedulerConfig,
    ) -> Self {
        let caps = executor.caps();
        let max_batch_ops = caps.execution_constraints.max_batch_ops as usize;
        if max_batch_ops > 0 {
            config.max_batch = config.max_batch.min(max_batch_ops.max(1));
        }
        // build from the worker's reported KV-cache groups (hybrid layouts);
        // empty == the single full-attention group BAGEL uses today.
        let bm = if caps.groups.is_empty() {
            BlockManager::new(
                caps.num_blocks as usize,
                caps.block_size as usize,
                caps.scratch_capacity_tokens,
            )
        } else {
            let specs: Vec<(uniserve_core::KvGroupKind, u32, u32)> = caps
                .groups
                .iter()
                .map(|g| (g.kind, g.block_offset, g.num_blocks))
                .collect();
            BlockManager::with_groups(
                caps.num_blocks as usize,
                caps.block_size as usize,
                caps.scratch_capacity_tokens,
                &specs,
            )
        };
        let usable_blocks = bm.free_blocks();
        let stats = Arc::new(SchedStats::default());
        stats
            .kv_cache
            .num_blocks
            .store(caps.num_blocks as usize, Ordering::Relaxed);
        let caps_encoder_budget = caps.encoder_cache_budget as usize;
        let spec_decode = crate::spec_decode::SpecDecodeAccounting::new(Arc::clone(&stats));
        let spec_ngram_max_tokens = spec_decode.max_ngram_tokens();
        let decode_lookahead = decode_lookahead_from_env();
        let denoise_step_burst = denoise_step_burst_from_env();
        let decode_token_burst = decode_token_burst_from_env();
        let mut trace_sink = crate::bench_trace::SchedulerTraceSink::from_env();
        if let Some(sink) = trace_sink.as_mut() {
            sink.record(&json!({
                "event": "run_started",
                "at_s": now(),
                "pid": std::process::id(),
                "scheduler": {
                    "policy": policy_str(config.policy),
                    "max_batch": config.max_batch,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "max_num_seqs": config.max_num_seqs,
                    "long_prefill_threshold": config.long_prefill_threshold,
                    "mixed_prefill_tokens": config.mixed_prefill_tokens,
                    "spec_ngram_max_tokens": spec_ngram_max_tokens,
                    "decode_lookahead": decode_lookahead,
                    "denoise_step_burst": denoise_step_burst,
                    "decode_token_burst": decode_token_burst,
                },
                "caps": {
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "supported_ops": &caps.supported_ops,
                    "max_batch_ops": caps.execution_constraints.max_batch_ops,
                    "pipeline_depth": caps.pipeline_depth,
                    "max_latent_size": caps.max_latent_size,
                    "latent_downsample": caps.latent_downsample,
                    "max_vae_grid_tokens": caps.max_vae_grid_tokens,
                    "max_vit_grid_tokens": caps.max_vit_grid_tokens,
                    "commit_marker_tokens": caps.commit_marker_tokens,
                    "gen_rope_advance": caps.gen_rope_advance,
                    "max_cfg_branches": caps.max_cfg_branches,
                    "resource_classes": &caps.resource_classes,
                },
            }));
        }
        Self {
            executor,
            caps,
            bm,
            ctrl,
            pending: make_queue(config.policy),
            config,
            usable_blocks,
            prefix_cache: crate::prefix_cache::PrefixCacheCoordinator::new(),
            sleeping: false,
            logits_pipeline: crate::logits::default_pipeline(),
            penalty_window: DEFAULT_PENALTY_WINDOW,
            enc_cache: EncoderCacheManager::new(if caps_encoder_budget > 0 {
                caps_encoder_budget
            } else {
                DEFAULT_ENCODER_CACHE_BUDGET
            }),
            mm_encode_budget: DEFAULT_MM_ENCODE_BUDGET,
            running: HashMap::new(),
            order: Vec::new(),
            skipped_waiting: HashMap::new(),
            grammar_compiler: GrammarCompiler::new(),
            reserved_blocks: 0,
            step_id: 0,
            inflight_ops: HashMap::new(),
            decode_lookahead,
            denoise_step_burst,
            decode_token_burst,
            spec_decode,
            fatal: false,
            ledger: crate::resources::ResourceLedger::new(),
            decisions: crate::policy::DecisionLog::default(),
            latency: crate::policy::LatencyHistory::new(),
            batch_started: HashMap::new(),
            next_op_id: 1,
            next_program_id: 1,
            completed_traces: VecDeque::new(),
            trace_sink,
            peak_ops_in_batch: 0,
            stats,
        }
    }

    pub fn policy(&self) -> SchedulingPolicy {
        self.config.policy
    }
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
    pub fn set_prefix_cache(&mut self, on: bool) {
        self.prefix_cache.set_enabled(on);
    }
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        self.prefix_cache.set_hash_algo(algo);
    }
    /// Register an extra logits processor — no other scheduler code changes.
    pub fn with_logits_processor(mut self, p: Box<dyn crate::logits::LogitsProcessor>) -> Self {
        self.logits_pipeline.push(p);
        self
    }
    pub fn set_penalty_window(&mut self, n: usize) {
        self.penalty_window = n.max(1);
    }
    /// Configure the per-step multimodal encode budget. `0` disables encoding.
    pub fn set_mm_encode_budget(&mut self, n: usize) {
        self.mm_encode_budget = n;
    }
    pub fn set_token_budget(&mut self, tokens: usize) {
        self.config.max_num_batched_tokens = tokens.max(1);
    }
    pub fn set_long_prefill_threshold(&mut self, n: usize) {
        self.config.long_prefill_threshold = n.max(1);
    }
    pub fn set_max_num_seqs(&mut self, n: usize) {
        self.config.max_num_seqs = n.max(1);
    }
    /// Cap the waiting-queue depth (pending + grammar-gated) for admission
    /// backpressure. `usize::MAX` disables the cap.
    pub fn set_max_num_waiting(&mut self, n: usize) {
        self.config.max_num_waiting = n.max(1);
    }
    pub fn caps(&self) -> &EngineCaps {
        &self.caps
    }
    pub fn stats_handle(&self) -> Arc<SchedStats> {
        self.stats.clone()
    }

    /// Record one explainable policy decision; `free_blocks`
    /// is sampled from the block manager at the decision point.
    fn record_decision(
        &mut self,
        request: RequestId,
        reason: crate::policy::PolicyReason,
        needed_blocks: usize,
    ) {
        let free_blocks = self.bm.free_blocks();
        self.decisions.record(crate::policy::PolicyDecision {
            request,
            reason,
            free_blocks,
            needed_blocks,
        });
    }

    /// Structured scheduling facts the policy weighs. A
    /// read-only snapshot — a pluggable policy could consume this without
    /// reaching into scheduler internals.
    pub fn policy_snapshot(&self) -> crate::policy::PolicySnapshot {
        let pq = self.stats.prefix.queries.load(Ordering::Relaxed);
        let ph = self.stats.prefix.hits.load(Ordering::Relaxed);
        let mq = self.stats.encoder.cache_queries.load(Ordering::Relaxed);
        let mh = self.stats.encoder.cache_hits.load(Ordering::Relaxed);
        crate::policy::PolicySnapshot {
            waiting: self.pending.len(),
            running: self.running.len(),
            skipped_waiting: self.skipped_waiting.len(),
            in_flight: self.executor.in_flight(),
            free_blocks: self.bm.free_blocks(),
            total_blocks: self.usable_blocks,
            reserved_blocks: self.reserved_blocks,
            active_leases: self.ledger.total_active(),
            cached_blocks: self.bm.cached_blocks(),
            prefix_hit_rate: if pq > 0 { ph as f32 / pq as f32 } else { 0.0 },
            mm_cache_hit_rate: if mq > 0 { mh as f32 / mq as f32 } else { 0.0 },
            op_latency_us: self.latency.as_pairs(),
        }
    }

    /// Drain the recent explainable policy decisions.
    pub fn take_policy_decisions(&mut self) -> Vec<crate::policy::PolicyDecision> {
        self.decisions.drain()
    }

    /// Round-trip latency EWMA for one op kind (microseconds), if observed.
    pub fn op_latency_us(&self, kind: &str) -> Option<u64> {
        self.latency.get(kind)
    }

    /// The typed program compiled for a running request.
    pub fn program_for(&self, id: RequestId) -> Option<&InferenceProgram> {
        self.running.get(&id).map(|st| &st.program)
    }

    /// The in-flight lifecycle trace of a running request.
    pub fn request_trace(&self, id: RequestId) -> Option<&crate::trace::RequestTrace> {
        self.running.get(&id).map(|st| &st.trace)
    }

    /// Drain archived lifecycle traces of completed requests — the
    /// reconstructable record after a request has finished.
    pub fn take_completed_traces(&mut self) -> Vec<crate::trace::RequestTrace> {
        self.completed_traces.drain(..).collect()
    }

    /// A health snapshot the engine can expose: queue +
    /// resource pressure + policy/latency + backend caps + liveness.
    pub fn health_snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            running: self.running.len(),
            pending: self.pending.len(),
            in_flight: self.executor.in_flight(),
            free_blocks: self.bm.free_blocks(),
            total_blocks: self.usable_blocks,
            reserved_blocks: self.reserved_blocks,
            active_leases: self.ledger.total_active(),
            resource_invariant_violations: self.ledger.stats.invariant_violations,
            completed_traces: self.completed_traces.len(),
            last_worker_exec_us: self
                .stats
                .timing
                .last_worker_exec_us
                .load(Ordering::Relaxed),
            queue_wait_count: self.stats.timing.queue_wait_count.load(Ordering::Relaxed),
            queue_wait_us_total: self
                .stats
                .timing
                .queue_wait_us_total
                .load(Ordering::Relaxed),
            queue_wait_us_max: self.stats.timing.queue_wait_us_max.load(Ordering::Relaxed),
            fatal: self.fatal,
            supported_ops: self.caps.supported_ops.clone(),
            op_latency_us: self.latency.as_pairs(),
        }
    }

    /// The owner thread: block on the command channel when fully idle, else
    /// spin the schedule-ahead loop. Returns `true` if the engine died
    /// (executor/worker failure) rather than shutting down gracefully.
    pub fn run(mut self, rx: Receiver<Command>) -> bool {
        // Event-driven executors (the iceoryx2 worker) park on a single wait
        // over {result, command, worker-death} instead of the fixed-interval
        // poll; the command ingress fires the executor's command waker, so a
        // freshly enqueued command interrupts the park (no missed-command
        // window). Polling executors (sim/local, which wake natively on their
        // result channel) keep the crossbeam select below
        // Event-driven executors park on {result, command, death}.
        let event_driven = self.executor.event_driven();
        loop {
            // drain pending commands (non-blocking)
            let mut shutdown = false;
            loop {
                match rx.try_recv() {
                    Ok(cmd) => {
                        if self.handle_command(cmd) {
                            shutdown = true;
                            break;
                        }
                    }
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        shutdown = true;
                        break;
                    }
                }
            }
            if shutdown {
                break;
            }

            let progressed = self.step_nonblocking();
            if self.fatal {
                tracing::error!("engine fatal: executor/worker died; stopping the control loop");
                self.abort_all_requests();
                self.executor.shutdown();
                return true;
            }

            if event_driven {
                // One park point. On wake, loop back to drain commands + step.
                // The park itself never observes shutdown — commands are drained
                // at the loop head — so it only needs to surface fatal death.
                if !progressed {
                    self.park_event_driven();
                    if self.fatal {
                        tracing::error!(
                            "engine fatal: executor/worker died during park; stopping the control loop"
                        );
                        self.abort_all_requests();
                        self.executor.shutdown();
                        return true;
                    }
                }
                continue;
            }

            if !progressed && self.executor.in_flight() > 0 {
                if self.wait_for_inflight_or_command(&rx) {
                    break;
                }
                if self.fatal {
                    tracing::error!(
                        "engine fatal: executor/worker died during wait; stopping the control loop"
                    );
                    self.abort_all_requests();
                    self.executor.shutdown();
                    return true;
                }
                continue;
            }

            if !progressed
                && self.running.is_empty()
                && self.pending.is_empty()
                && self.executor.in_flight() == 0
            {
                // even fully idle, wake periodically to probe worker
                // liveness so a worker that dies with nothing in flight is
                // detected promptly (the engine latches fatal and the frontend
                // stops routing here) instead of only being noticed when the
                // next request arrives. When gated on grammar compilation we use
                // the shorter slice so the compiler is polled responsively.
                let wait = if self.skipped_waiting.is_empty() {
                    IDLE_LIVENESS_POLL
                } else {
                    std::time::Duration::from_millis(1)
                };
                match rx.recv_timeout(wait) {
                    Ok(cmd) => {
                        if self.handle_command(cmd) {
                            break;
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        // No command arrived; verify the worker is still alive.
                        if let Err(e) = self.executor.check_liveness() {
                            self.on_executor_error(e);
                            if self.fatal {
                                tracing::error!(
                                    "engine fatal: worker died while idle; stopping the control loop"
                                );
                                self.abort_all_requests();
                                self.executor.shutdown();
                                return true;
                            }
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        // Graceful shutdown: report Aborted to everything still queued or
        // running before tearing the executor down (staged drain happened at
        // the HTTP layer; nothing in flight should just see a closed channel).
        self.abort_all_requests();
        self.executor.shutdown();
        false
    }

    /// Event-driven park: block on the
    /// executor's single wait over {result, command, worker-death} until any
    /// source fires or a state-dependent safety-net timeout elapses, then probe
    /// liveness so an idle worker death is caught even when no result surfaces
    /// it. No fixed-interval poll: the timeout is purely a backstop for a missed
    /// notification (which would only cost that much latency, never correctness)
    /// and, while grammar compilation is pending, a short slice so the compiler
    /// is drained promptly (that completion is not an event source).
    fn park_event_driven(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        let timeout = if !self.skipped_waiting.is_empty() {
            Duration::from_millis(1)
        } else if self.executor.in_flight() > 0 {
            SCHEDULER_WAIT_SLICE
        } else {
            IDLE_LIVENESS_POLL
        };
        if let Err(e) = self.executor.park_for_event(timeout) {
            self.on_executor_error(e);
            return;
        }
        // The death watcher wakes the park instantly on child exit; confirm and
        // latch it here (also the only death signal when fully idle, where no
        // result drain would otherwise surface it).
        if let Err(e) = self.executor.check_liveness() {
            self.on_executor_error(e);
        }
        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        self.publish_cache_stats();
    }

    fn wait_for_inflight_or_command(&mut self, rx: &Receiver<Command>) -> bool {
        crossbeam_channel::select! {
            recv(rx) -> msg => match msg {
                Ok(cmd) => {
                    if self.handle_command(cmd) {
                        return true;
                    }
                }
                Err(_) => return true,
            },
            default(SCHEDULER_WAIT_SLICE) => {
                match self.executor.wait_result_timeout(Duration::ZERO) {
                    Ok(Some(r)) => self.apply_result(r),
                    Ok(None) => {}
                    Err(e) => self.on_executor_error(e),
                }
            }
        }
        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        self.publish_cache_stats();
        false
    }

    /// Abort every queued/gated/running request with a terminal event.
    fn abort_all_requests(&mut self) {
        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(st) = self.pending.pop_request() {
                ids.push(st.req.request_id);
                let _ = st.req.event_tx.send(GenEvent::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.req.prompt_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            ids
        };
        let _ = queued;
        for (_, st) in self.skipped_waiting.drain() {
            let _ = st.req.event_tx.send(GenEvent::Finished {
                reason: FinishReason::Aborted,
                stop_reason: None,
                prompt_tokens: st.req.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
        let running: Vec<RequestId> = self.running.keys().copied().collect();
        for id in running {
            self.finish(id, FinishReason::Aborted);
        }
    }

    /// Apply one command; returns true on shutdown.
    fn handle_command(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Submit(req) => self.enqueue(*req),
            Command::Cancel(id) => self.mark_cancelled(id, false),
            Command::Abort(id) => self.mark_cancelled(id, true),
            Command::ResetPrefixCache => self.reset_prefix_cache(),
            Command::ResetEncoderCache => self.reset_encoder_cache(),
            Command::LoadLora { id, path } => {
                self.gated_control(ControlOp::LoadLora { lora_id: id, path });
            }
            Command::UnloadLora { id } => {
                self.gated_control(ControlOp::UnloadLora { lora_id: id });
            }
            Command::SetSleeping(s) => self.set_sleeping(s),
            Command::CollectiveRpc { method, reply } => {
                let _ = reply.send(self.collective_rpc(&method));
            }
            Command::Shutdown => return true,
        }
        false
    }

    /// Test-only direct enqueue (the production path is `run` over the channel).
    pub fn submit_for_test(&mut self, req: GenerateRequest) {
        self.enqueue(req);
    }

    fn mark_cancelled(&mut self, id: RequestId, abort: bool) {
        if let Some(st) = self.running.get_mut(&id) {
            st.cancelled = true;
            st.aborted = abort;
        }
        // a request still gated on grammar compilation can be cancelled too
        if let Some(st) = self.skipped_waiting.remove(&id) {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.trace_request_finished(
                id,
                &reason,
                None,
                st.req.prompt_ids.len(),
                0,
                0,
                "skipped_waiting",
            );
            let _ = st.req.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.req.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
            return;
        }
        // also drop from the waiting queue if not yet admitted (reporting the reason)
        if let Some(st) = self.pending.remove_request(id) {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.trace_request_finished(
                id,
                &reason,
                None,
                st.req.prompt_ids.len(),
                0,
                0,
                "pending",
            );
            let _ = st.req.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.req.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
    }

    /// Suppress controls not declared in `supported_controls`. An empty
    /// declared set is treated as unspecified. `drop_request` is never gated.
    fn control_allowed(&self, op: &ControlOp) -> bool {
        if matches!(op, ControlOp::DropRequest(_)) {
            return true;
        }
        let declared = &self.caps.supported_controls;
        declared.is_empty() || declared.iter().any(|c| c == op.method())
    }

    /// Dispatch a control only if the worker declares support; otherwise drop it
    /// (logged) instead of fire-and-forgetting into an UnsupportedControl error.
    fn gated_control(&mut self, op: ControlOp) {
        if self.control_allowed(&op) {
            let _ = self.executor.control(op);
        } else {
            tracing::debug!(
                control = op.method(),
                "skipping control absent from worker supported_controls"
            );
        }
    }

    /// Clear the prefix cache.
    fn reset_prefix_cache(&mut self) {
        self.bm.reset_prefix_cache();
        self.gated_control(ControlOp::ResetPrefixCache);
        self.publish_cache_stats();
    }

    /// Pause or resume admission.
    fn set_sleeping(&mut self, sleeping: bool) {
        self.sleeping = sleeping;
        let op = if sleeping {
            ControlOp::Sleep
        } else {
            ControlOp::WakeUp
        };
        self.gated_control(op);
    }
    pub fn is_sleeping(&self) -> bool {
        self.sleeping
    }

    fn trace_record(&mut self, record: serde_json::Value) {
        if let Some(sink) = self.trace_sink.as_mut() {
            sink.record(&record);
        }
    }

    fn trace_request_queued(&mut self, st: &ReqState, queue: &'static str) {
        self.trace_record(json!({
            "event": "request_queued",
            "at_s": st.queued_at,
            "request_id": st.req.request_id.0,
            "trace_id": st.trace.trace_id.0,
            "program_id": st.program.program_id.0,
            "queue": queue,
            "mode": mode_str(st.req.mode),
            "initial_phase": phase_str(st.phase),
            "prompt_tokens": st.req.prompt_ids.len(),
            "max_tokens": st.req.max_tokens,
            "priority": st.req.priority,
            "reserve_worstcase": st.reserve_worstcase,
            "worstcase_blocks": st.worstcase_blocks,
            "image": {
                "steps": st.req.image.steps,
                "max_images": st.req.image.max_images,
                "height": st.req.image.height,
                "width": st.req.image.width,
                "retain_images": st.req.image.retain_images,
            },
            "pending": self.pending.len(),
            "running": self.running.len(),
            "skipped_waiting": self.skipped_waiting.len(),
        }));
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_request_finished(
        &mut self,
        id: RequestId,
        reason: &FinishReason,
        stop_reason: Option<&str>,
        prompt_tokens: usize,
        completion_tokens: usize,
        images: usize,
        queue: &'static str,
    ) {
        self.trace_record(json!({
            "event": "request_finished",
            "at_s": now(),
            "request_id": id.0,
            "reason": finish_reason_str(reason),
            "stop_reason": stop_reason,
            "queue": queue,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "images": images,
            "pending": self.pending.len(),
            "running": self.running.len(),
            "in_flight": self.executor.in_flight(),
        }));
    }

    fn enqueue(&mut self, req: GenerateRequest) {
        // admission backpressure. Shed load instead of letting the waiting
        // queues (pending + grammar-gated) grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending.len() + self.skipped_waiting.len();
        if waiting >= self.config.max_num_waiting {
            self.record_decision(
                req.request_id,
                crate::policy::PolicyReason::RejectedTooLarge,
                0,
            );
            self.trace_record(json!({
                "event": "request_rejected",
                "at_s": now(),
                "request_id": req.request_id.0,
                "reason": "queue_full",
                "waiting": waiting,
                "max_num_waiting": self.config.max_num_waiting,
                "mode": mode_str(req.mode),
                "prompt_tokens": req.prompt_ids.len(),
            }));
            let _ = req.event_tx.send(GenEvent::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let prompt_len = req.prompt_ids.len();
        let worst = self.worstcase(&req, prompt_len);
        // Non-text multimodal requests are not safely preemptible once they
        // start producing image/text state. Reserve their configured bounded KV
        // envelope at admission so excess concurrency queues instead of
        // exhausting KV after all requests have become resident.
        let reserve_worstcase = matches!(
            req.mode,
            GenMode::Image | GenMode::AutoInterleave | GenMode::InterleaveUnd
        );
        // a request with staged images encodes them before prefill.
        // Understanding interleave prefills SYS first, then encodes the
        // image at its in-prompt position, so it starts in Prefill.
        let phase0 = if req.mm_items.is_empty() || req.mode == GenMode::InterleaveUnd {
            Phase::Prefill
        } else {
            Phase::Encode
        };
        // compile the request into its typed program (host-internal
        // IR). The FSM stays authoritative; the program rides along for typed
        // representation, lifecycle tracking, and policy hints.
        let mut program = uniserve_program::compile(&req);
        let program_id = uniserve_core::ProgramId(self.next_program_id);
        self.next_program_id += 1;
        program.program_id = program_id;
        // a lifecycle trace keyed by trace/program/request ids.
        let trace = crate::trace::RequestTrace::new(
            req.request_id,
            uniserve_core::TraceId(req.request_id.0),
            program_id,
        );
        let st = ReqState {
            phase: phase0,
            pos: 0,
            next_token: 0,
            n_generated: 0,
            worstcase_blocks: worst,
            reserve_worstcase,
            image_id: 0,
            images_done: 0,
            text_since_image: 0,
            auto_image_pending: false,
            cond_pos: 0,
            steps_done: 0,
            queued_at: now(),
            cancelled: false,
            aborted: false,
            prompt_cursor: 0,
            preempted: false,
            block_hashes: Vec::new(),
            prefix_cached_blocks: 0,
            blocks_cached: false,
            generated_ids: Vec::new(),
            recompute_ids: None,
            mm_cursor: 0,
            mm_acquired: Vec::new(),
            grammar: None,
            worker_registered: false,
            blocks_sent: 0,
            kvlen: 0,
            iu_encode_step: 0,
            gen_h: 0,
            gen_w: 0,
            round_tokens: Vec::new(),
            iu_closing: false,
            program,
            trace,
            worker_image_latent_units: 0,
            req,
        };
        // Structured-output gate: a request with a grammar
        // waits in skipped_waiting until its compilation finishes; admission
        // collects ready grammars each pass.
        if let Some(spec) = st.req.grammar.clone() {
            let id = st.req.request_id;
            self.grammar_compiler.submit(id, spec);
            self.trace_request_queued(&st, "skipped_waiting");
            self.skipped_waiting.insert(id, st);
            return;
        }
        self.trace_request_queued(&st, "pending");
        self.pending.add_request(st);
    }

    fn num_vae(&self, ip: &uniserve_core::ImageParams) -> u64 {
        let dl = self.caps.latent_downsample as u64;
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.caps.max_vae_grid_tokens > 0 {
            self.caps.max_vae_grid_tokens as usize
        } else {
            self.caps.max_latent_size as usize
        }
    }

    fn cap_max_vit_grid_tokens(&self) -> usize {
        self.caps.max_vit_grid_tokens as usize
    }

    fn cap_commit_marker_tokens(&self) -> usize {
        self.caps.commit_marker_tokens.max(1) as usize
    }

    fn cap_gen_rope_advance(&self) -> u32 {
        self.caps.gen_rope_advance.max(1)
    }

    fn cap_max_cfg_branches(&self) -> u64 {
        self.caps.max_cfg_branches.max(1) as u64
    }

    fn worker_tracks_image_latent(&self) -> bool {
        self.caps.max_latent_size > 0
            && self
                .caps
                .resource_classes
                .contains(&ResourceClass::ImageLatent)
    }

    fn worker_image_latent_used(&self) -> u64 {
        self.running
            .values()
            .map(|st| st.worker_image_latent_units)
            .sum()
    }

    fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.caps.latent_downsample as u64).max(1);
        let (height, width) =
            if st.req.mode == GenMode::InterleaveUnd && st.gen_h > 0 && st.gen_w > 0 {
                (st.gen_h, st.gen_w)
            } else {
                (st.req.image.height, st.req.image.width)
            };
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    fn denoise_worker_image_latent_additional(&self, id: RequestId) -> Option<u64> {
        if !self.worker_tracks_image_latent() {
            return Some(0);
        }
        let st = self.running.get(&id)?;
        if st.worker_image_latent_units > 0 {
            Some(0)
        } else {
            Some(self.worker_image_latent_units_for(st).max(1))
        }
    }

    fn can_schedule_denoise(&self, id: RequestId) -> bool {
        let Some(additional) = self.denoise_worker_image_latent_additional(id) else {
            return false;
        };
        if additional == 0 {
            return true;
        }
        self.worker_image_latent_used().saturating_add(additional)
            <= self.caps.max_latent_size as u64
    }

    fn reserve_worker_image_latent_for_denoise(&mut self, id: RequestId) {
        let Some(additional) = self.denoise_worker_image_latent_additional(id) else {
            return;
        };
        if additional == 0 {
            return;
        }
        debug_assert!(
            self.worker_image_latent_used().saturating_add(additional)
                <= self.caps.max_latent_size as u64,
            "denoise latent reservation must be checked before op assembly"
        );
        if let Some(st) = self.running.get_mut(&id) {
            st.worker_image_latent_units = additional;
        }
    }

    fn generated_image_span(&self, ip: &uniserve_core::ImageParams) -> usize {
        if ip.retain_images {
            self.num_vae(ip) as usize + self.cap_commit_marker_tokens()
        } else {
            0
        }
    }

    fn auto_interleave_remaining_image_span(&self, st: &ReqState) -> usize {
        if st.req.mode != GenMode::AutoInterleave {
            return 0;
        }
        let remaining_images = (st.req.image.max_images as usize).saturating_sub(st.images_done);
        self.generated_image_span(&st.req.image)
            .saturating_mul(remaining_images)
    }

    fn decode_capacity_target(
        &self,
        id: RequestId,
        pos: usize,
        decode_len: usize,
        spec_len: usize,
    ) -> usize {
        let base = pos.saturating_add(decode_len).saturating_add(spec_len);
        let slack = self
            .running
            .get(&id)
            .map(|st| self.auto_interleave_remaining_image_span(st))
            .unwrap_or(0);
        base.saturating_add(slack)
    }

    fn worstcase(&self, req: &GenerateRequest, prompt_len: usize) -> usize {
        let bs = self.caps.block_size as usize;
        let toks = match req.mode {
            GenMode::Text => prompt_len + req.max_tokens,
            GenMode::Image => {
                prompt_len + self.generated_image_span(&req.image) * req.image.max_images as usize
            }
            GenMode::AutoInterleave => {
                // Text generation can stop naturally far before `max_tokens` on
                // chat-style auto-interleave requests. Reserve the finite image
                // envelope up front, then let decode grow KV under the normal
                // per-step capacity gates.
                prompt_len + self.generated_image_span(&req.image) * req.image.max_images as usize
            }
            // Exact image token counts come back from the worker's encode ops;
            // reserve generously up front for the input image + each generated
            // reasoning image. The VAE half of the dual-encode budgets only
            // when the worker implements it.
            GenMode::InterleaveUnd => {
                let vae_tokens = if self.supports_vae_encode() {
                    self.cap_max_vae_grid_tokens()
                } else {
                    0
                };
                let per_image = vae_tokens
                    + self.cap_max_vit_grid_tokens()
                    + 2 * self.cap_commit_marker_tokens();
                prompt_len + req.max_tokens + per_image * (1 + req.image.max_images as usize)
            }
        };
        toks.div_ceil(bs)
    }

    /// num_vae for an explicit (h, w) — the generated reasoning image size the
    /// worker reports, distinct from the request's `ImageParams`.
    fn num_vae_hw(&self, h: u32, w: u32) -> u64 {
        let dl = self.caps.latent_downsample as u64;
        (h as u64 / dl) * (w as u64 / dl)
    }

    fn image_prompt_for(st: &ReqState) -> Option<String> {
        st.req.image.image_prompts.get(st.images_done).cloned()
    }

    /// One loop iteration of the schedule-ahead loop. Returns true if any work
    /// was submitted or any result resolved.
    pub fn step(&mut self) -> bool {
        let mut progressed = self.step_nonblocking();
        if !progressed && self.executor.in_flight() > 0 {
            match self.executor.next_result() {
                Ok(r) => {
                    self.apply_result(r);
                    progressed = true;
                }
                Err(e) => {
                    self.on_executor_error(e);
                    progressed = true;
                }
            }
            self.stats
                .general
                .in_flight
                .store(self.executor.in_flight(), Ordering::Relaxed);
            self.publish_cache_stats();
        }
        progressed
    }

    /// Nonblocking schedule-ahead tick used by the owner-thread reactor. It
    /// drains ready results, reaps cancellations, and fills available executor
    /// slots, but leaves any blocking result wait to `run`.
    fn step_nonblocking(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.step").entered();
        // 1. resolve any completed batches (non-blocking).
        let mut progressed = self.drain_results();

        // 2. reap cancellations before assembling.
        self.reap_cancellations();

        // 3. submit as many batches as pipeline capacity allows.
        while self.executor.can_submit() {
            self.admit();
            let (new_reqs, ops) = self.assemble();
            if ops.is_empty() {
                break;
            }
            self.submit_batch(new_reqs, ops);
            progressed = true;
        }

        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        self.publish_cache_stats();
        progressed
    }

    /// Surface cache observability (events drained into counters).
    fn publish_cache_stats(&mut self) {
        self.stats
            .general
            .running
            .store(self.running.len(), Ordering::Relaxed);
        self.stats
            .general
            .pending
            .store(self.pending.len(), Ordering::Relaxed);
        self.stats
            .general
            .skipped_waiting
            .store(self.skipped_waiting.len(), Ordering::Relaxed);
        self.stats
            .kv_cache
            .free_blocks
            .store(self.bm.free_blocks(), Ordering::Relaxed);
        self.stats
            .general
            .in_flight
            .store(self.executor.in_flight(), Ordering::Relaxed);
        // drain the manager's event ring so it doesn't grow unbounded; the
        // counters below already aggregate it, but draining keeps memory bounded.
        let _ = self.bm.drain_events();
        self.stats
            .kv_cache
            .blocks_evicted
            .store(self.bm.stats.evictions, Ordering::Relaxed);
        self.stats
            .kv_cache
            .blocks_stored
            .store(self.bm.stats.blocks_stored, Ordering::Relaxed);
        self.stats
            .kv_cache
            .cached_blocks
            .store(self.bm.cached_blocks(), Ordering::Relaxed);
        self.stats
            .encoder
            .cache_queries
            .store(self.enc_cache.stats.queries, Ordering::Relaxed);
        self.stats
            .encoder
            .cache_hits
            .store(self.enc_cache.stats.hits, Ordering::Relaxed);
        self.stats
            .encoder
            .cached
            .store(self.enc_cache.len(), Ordering::Relaxed);
        // resource-plane observability — active leases (0 when idle) and
        // any retained-after-release invariant violations.
        self.stats
            .resources
            .active
            .store(self.ledger.total_active(), Ordering::Relaxed);
        self.stats
            .resources
            .invariant_violations
            .store(self.ledger.stats.invariant_violations, Ordering::Relaxed);
    }

    /// Drain every ready result and resolve it. Returns true if any resolved.
    fn drain_results(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.drain_results").entered();
        let mut any = false;
        loop {
            match self.executor.poll() {
                Ok(Some(r)) => {
                    self.apply_result(r);
                    any = true;
                }
                Ok(None) => break,
                Err(e) => {
                    self.on_executor_error(e);
                    any = true;
                    break;
                }
            }
        }
        any
    }

    /// Whether any request currently has a prefill op in flight. Prompt work
    /// coalesces behind it: while one prefill batch runs, newly arrived
    /// prompts wait (decode fills the slot) and merge into the next prefill
    /// batch, so bursts cost one sweep instead of one sweep per arrival.
    fn any_prefill_inflight(&self) -> bool {
        self.inflight_ops
            .values()
            .any(|items| items.iter().any(|op| op.kind == OpKind::PrefillUnd))
    }

    fn has_inflight(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|items| !items.is_empty())
    }

    fn inflight_decode_count(&self, id: RequestId) -> usize {
        self.inflight_ops
            .get(&id)
            .map(|items| {
                items
                    .iter()
                    .filter(|op| op.kind == OpKind::DecodeUnd)
                    .count()
            })
            .unwrap_or(0)
    }

    fn inflight_has_spec_tokens(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|items| items.iter().any(|op| !op.spec_tokens.is_empty()))
    }

    fn inflight_has_multi_decode(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|items| items.iter().any(|op| op.decode_token_count > 1))
    }

    /// Whether any in-flight op for `id` is something other than a plain decode
    /// (a prefill chunk, an image op, …). Used to gate decode-lookahead.
    fn inflight_has_non_decode(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|items| items.iter().any(|op| op.kind != OpKind::DecodeUnd))
    }

    fn register_inflight(&mut self, op: &ForwardOp, started: Instant) {
        self.inflight_ops
            .entry(op.req_id)
            .or_default()
            .push_back(InflightOp {
                kind: op.kind,
                op_id: op.op_id,
                spec_tokens: op.spec_token_ids.clone().unwrap_or_default(),
                decode_token_count: op.decode_token_count.unwrap_or(1).max(1),
                started,
            });
    }

    /// Resolve one in-flight op for `id`, matching the worker's echoed `op_id`
    /// and falling back to FIFO order when it is absent/unknown.
    fn pop_inflight(
        &mut self,
        id: RequestId,
        op_id: Option<u64>,
    ) -> (Option<OpKind>, Vec<u32>, Option<Instant>) {
        let Some(queue) = self.inflight_ops.get_mut(&id) else {
            return (None, Vec::new(), None);
        };
        let resolved = take_inflight_by_op_id(queue, op_id);
        if queue.is_empty() {
            self.inflight_ops.remove(&id);
        }
        match resolved {
            Some(op) => (Some(op.kind), op.spec_tokens, Some(op.started)),
            None => (None, Vec::new(), None),
        }
    }

    /// Failure policy after an executor/worker error.
    /// A typed non-fatal [`WorkerExecError`] fails the in-flight requests but
    /// keeps the engine alive to serve subsequent requests; anything else (a
    /// fatal worker error, ring/transport death) latches the engine fatal.

    /// The typed taxonomy's `code` and `retryable` drive
    /// real policy rather than being log-only. The worker's `fatal` bit is the
    /// baseline, but the host *escalates* host-bug classes to fatal even when
    /// the worker marked them non-fatal (a `SCHEDULER_BUG`/`INVARIANT_VIOLATION`
    /// means the control plane can no longer be trusted), and it picks the log
    /// severity by class so a benign `USER_INPUT_ERROR` does not spam warnings.
    fn on_executor_error(&mut self, e: anyhow::Error) {
        let exec = e.downcast_ref::<WorkerExecError>();
        let worker_fatal = exec.map(|w| w.fatal).unwrap_or(true);
        let code = exec.and_then(|w| w.code.as_deref());
        let retryable = exec.map(|w| w.retryable).unwrap_or(false);
        // Host-bug classes are latched fatal regardless of the worker's bit:
        // continuing to schedule against a violated invariant is unsafe.
        let host_escalates_fatal =
            matches!(code, Some("SchedulerBug") | Some("InvariantViolation"));
        let fatal = worker_fatal || host_escalates_fatal;
        if fatal {
            tracing::error!(
                ?code,
                retryable,
                "fatal executor error (engine will stop): {e}"
            );
            self.fatal = true;
        } else if matches!(code, Some("UserInputError")) {
            // A malformed request is the client's fault, not a worker problem:
            // fail just that request at info level instead of warn-spam.
            tracing::info!(?code, "request rejected by worker (failing in-flight): {e}");
        } else {
            // Non-fatal worker errors keep the worker up. `retryable` is
            // surfaced so a recoverable class (OOM/transient) is visible to
            // operators; an automatic requeue path lands with the drafter/retry
            // budget work and is intentionally not attempted here.
            tracing::warn!(
                ?code,
                retryable,
                "non-fatal worker error (failing in-flight, worker stays up): {e}"
            );
        }
        self.fail_all_inflight(&format!("{e}"));
    }

    fn apply_result(&mut self, result: ForwardResult) {
        // worker-reported batch compute time (one value per batch).
        let result_step_id = result.step_id;
        let batch_roundtrip_us = self
            .batch_started
            .remove(&result_step_id)
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);
        let worker_us = result.worker_exec_us.unwrap_or(0);
        if let Some(w) = result.worker_exec_us {
            self.stats
                .timing
                .last_worker_exec_us
                .store(w, Ordering::Relaxed);
        }
        // fold the per-batch worker compute time and host round-trip
        // latency into cumulative counters so they surface in the structured
        // wire stats / Prometheus, mirroring what the JSON trace already records.
        self.stats
            .timing
            .worker_exec_us_total
            .fetch_add(worker_us, Ordering::Relaxed);
        self.stats
            .timing
            .batch_roundtrip_us_total
            .fetch_add(batch_roundtrip_us, Ordering::Relaxed);
        self.stats
            .timing
            .batch_timing_count
            .fetch_add(1, Ordering::Relaxed);
        let forward_stats_trace = result
            .forward_stats
            .as_ref()
            .map(worker_forward_stats_trace);
        self.record_worker_forward_stats(result.forward_stats.as_ref());
        let mut resolved_ops = Vec::with_capacity(result.per_seq.len());
        let mut progress_ops = Vec::with_capacity(result.per_seq.len());
        let mut to_resolve = Vec::with_capacity(result.per_seq.len());
        for (seq_index, sr) in result.per_seq.into_iter().enumerate() {
            let id = sr.req_id;
            // op_id-correlated completion: resolve the exact op the
            // worker echoed, not merely the FIFO-oldest one.
            let (kind, draft_token_ids, started) = self.pop_inflight(id, sr.op_id);
            let draft_tokens = draft_token_ids.len();
            // fold this op's host-side round-trip latency into the history.
            let roundtrip_us = started
                .map(|start| start.elapsed().as_micros() as u64)
                .unwrap_or(0);
            if let Some(k) = kind {
                self.latency.observe(opkind_str(k), roundtrip_us);
            }
            // record the op-resolved lifecycle event (op_id echoed by the
            // worker, host round-trip + worker compute time).
            let op_id = sr.op_id;
            let sampled_token_ids_len = sr.sampled_token_ids.as_ref().map_or(0, Vec::len);
            let sampled_token_ids_last = sr
                .sampled_token_ids
                .as_ref()
                .and_then(|ids| ids.last().copied());
            resolved_ops.push(json!({
                "request_id": id.0,
                "op_id": op_id,
                "op_kind": kind.map(opkind_str),
                "roundtrip_us": roundtrip_us,
                "sampled_token": sr.sampled_token_id.is_some(),
                "sampled_token_ids_len": sampled_token_ids_len,
                "sampled_token_ids_last": sampled_token_ids_last,
                "denoise_done": sr.denoise_done,
                "num_steps_done": sr.num_steps_done,
                "image_done": sr.image_png_b64.is_some(),
                "image_hw": sr.image_hw,
                "num_tokens": sr.num_tokens,
                "num_draft_tokens": draft_tokens,
                "num_accepted_tokens": sr.num_accepted_tokens,
                "encoder_handle": sr.encoder_handle,
            }));
            if draft_tokens > 0 {
                self.spec_decode
                    .record_acceptance(draft_tokens, sr.num_accepted_tokens);
            }
            if let Some(st) = self.running.get_mut(&id) {
                let mut ev = crate::trace::TraceEvent::at(crate::trace::TraceEventKind::OpResolved);
                ev.op_id = op_id;
                ev.op_kind = kind.map(opkind_str);
                ev.roundtrip_us = roundtrip_us;
                ev.worker_us = worker_us;
                st.trace.push(ev);
            }
            if let Some(kind) = kind {
                let priority = match kind {
                    OpKind::DenoiseGen | OpKind::CommitGen | OpKind::CommitWriteback => 0,
                    _ => 1,
                };
                to_resolve.push((priority, seq_index, id, kind, sr, draft_token_ids));
            }
        }
        to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
        for (_priority, _seq_index, id, kind, sr, draft_token_ids) in to_resolve {
            if self.running.contains_key(&id) {
                self.resolve(id, kind, sr, draft_token_ids);
            }
            if let Some(st) = self.running.get(&id) {
                progress_ops.push(json!({
                    "request_id": id.0,
                    "phase": phase_str(st.phase),
                    "generated_tokens": st.n_generated,
                    "images_done": st.images_done,
                    "image_id": st.image_id,
                    "steps_done": st.steps_done,
                    "pos": st.pos,
                    "kvlen": st.kvlen,
                    "next_token": st.next_token,
                    "text_since_image": st.text_since_image,
                    "auto_image_pending": st.auto_image_pending,
                    "iu_closing": st.iu_closing,
                }));
            }
        }
        self.trace_record(json!({
            "event": "batch_resolved",
            "at_s": now(),
            "step_id": result_step_id,
            "worker_exec_us": worker_us,
            "host_roundtrip_us": batch_roundtrip_us,
            "forward_stats": forward_stats_trace,
            "batch_size": resolved_ops.len(),
            "ops": resolved_ops,
            "progress": progress_ops,
            "running": self.running.len(),
            "pending": self.pending.len(),
            "in_flight": self.executor.in_flight(),
        }));
    }

    fn record_worker_forward_stats(&self, stats: Option<&WorkerForwardStats>) {
        let Some(stats) = stats else {
            return;
        };
        add_worker_forward_map(&self.stats.worker.forward_mode_counts, &stats.mode_counts);
        add_worker_forward_map(&self.stats.worker.forward_mode_tokens, &stats.mode_tokens);
        add_worker_forward_map(&self.stats.worker.forward_mode_us, &stats.mode_us);
        add_worker_forward_map(&self.stats.worker.forward_component_us, &stats.component_us);
        // Fold worker forward stats maps into scheduler stats.
        add_worker_forward_map(
            &self.stats.worker.attention_backend_counts,
            &stats.attention_backend_counts,
        );
        add_worker_forward_map(
            &self.stats.worker.cuda_graph_runtime_mode_counts,
            &stats.cuda_graph_runtime_mode_counts,
        );
        self.stats
            .worker
            .attention_launches
            .fetch_add(stats.attention_launches, Ordering::Relaxed);
        self.stats
            .worker
            .attention_us
            .fetch_add(stats.attention_us, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_captures
            .fetch_add(stats.cuda_graph_captures, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_replays
            .fetch_add(stats.cuda_graph_replays, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_misses
            .fetch_add(stats.cuda_graph_misses, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_fallbacks
            .fetch_add(stats.cuda_graph_fallbacks, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_unpadded_tokens
            .fetch_add(stats.cuda_graph_unpadded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_padded_tokens
            .fetch_add(stats.cuda_graph_padded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_hits
            .fetch_add(stats.text_decode_token_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_misses
            .fetch_add(stats.text_decode_token_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_hits
            .fetch_add(stats.text_decode_position_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_misses
            .fetch_add(stats.text_decode_position_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_calls
            .fetch_add(stats.flashinfer_decode_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_reuses
            .fetch_add(stats.flashinfer_decode_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_rows
            .fetch_add(stats.flashinfer_decode_plan_rows, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_indices
            .fetch_add(stats.flashinfer_decode_plan_indices, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_calls
            .fetch_add(stats.flashinfer_decode_graph_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_reuses
            .fetch_add(stats.flashinfer_decode_graph_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rows
            .fetch_add(stats.spec_verify_rows, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_draft_tokens
            .fetch_add(stats.spec_verify_draft_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_accepted_tokens
            .fetch_add(stats.spec_verify_accepted_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rejected_tokens
            .fetch_add(stats.spec_verify_rejected_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_committed_tokens
            .fetch_add(stats.spec_verify_committed_tokens, Ordering::Relaxed);
        if !stats.spec_verify_path_counts.is_empty() {
            let mut path_counts = self
                .stats
                .worker
                .spec_verify_path_counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (path, count) in &stats.spec_verify_path_counts {
                *path_counts.entry(path.clone()).or_default() += *count;
            }
        }
    }

    fn fail_all_inflight(&mut self, msg: &str) {
        let ids: Vec<RequestId> = self.inflight_ops.keys().copied().collect();
        self.inflight_ops.clear();
        // the submitted batches whose results will now never return are
        // failed here, so drop their pending submit-timestamps too — otherwise
        // `batch_started` accumulates orphaned entries for every failed batch.
        self.batch_started.clear();
        for id in ids {
            if self.running.contains_key(&id) {
                self.emit(
                    id,
                    GenEvent::Error {
                        message: msg.to_string(),
                    },
                );
                self.finish(id, FinishReason::Error);
            }
        }
    }

    fn reap_cancellations(&mut self) {
        // never release a cancelled request's KV while one of its ops is
        // still executing on the worker. `finish` frees the physical blocks
        // immediately (and DropRequest is fire-and-forget), so reaping an
        // in-flight request would let the blocks be re-allocated to another
        // request before the cancelled forward completes — corrupting that
        // request's KV cache on a pipelined worker. Defer to a later step: once
        // the op resolves, `inflight_kinds` drains and the next reap finishes it.
        // This mirrors the `!has_inflight` guard preemption already uses.
        let cancelled: Vec<(RequestId, bool)> = self
            .running
            .iter()
            .filter(|(id, s)| s.cancelled && !self.has_inflight(**id))
            .map(|(k, s)| (*k, s.aborted))
            .collect();
        for (id, aborted) in cancelled {
            let reason = if aborted {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.finish(id, reason);
        }
    }

    /// Admission: consume the waiting-queue head while budgets allow (vLLM's
    /// posture — head-of-line, `max_num_seqs`-capped). Reserving requests
    /// allocate their full worst-case KV here, which is what makes them
    /// Preemption-exempt by construction.
    fn admit(&mut self) {
        if self.sleeping {
            return;
        }
        // Open the structured-output gate for grammars that finished compiling.
        for (id, compiled) in self.grammar_compiler.drain_ready() {
            if let Some(mut st) = self.skipped_waiting.remove(&id) {
                st.grammar = Some(GrammarMatcher::new(compiled));
                self.pending.add_request(st);
            }
        }
        let bs = self.caps.block_size as usize;
        loop {
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }
            let Some(head) = self.pending.peek_request() else {
                break;
            };

            if head.reserve_worstcase {
                let need = head.worstcase_blocks;
                let scratch_ok = match head.req.mode {
                    GenMode::Image | GenMode::AutoInterleave => self.bm.can_reserve_scratch(
                        self.num_vae(&head.req.image) * self.cap_max_cfg_branches(),
                    ),
                    GenMode::InterleaveUnd => self
                        .bm
                        .can_reserve_scratch(self.cap_max_vae_grid_tokens() as u64),
                    GenMode::Text => true,
                };
                if need > self.usable_blocks {
                    let st = self.pending.pop_request().unwrap();
                    self.record_decision(
                        st.req.request_id,
                        crate::policy::PolicyReason::RejectedTooLarge,
                        need,
                    );
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": need,
                        "usable_blocks": self.usable_blocks,
                        "mode": mode_str(st.req.mode),
                        "prompt_tokens": st.req.prompt_ids.len(),
                    }));
                    let _ = st.req.event_tx.send(GenEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.bm.free_blocks() >= need && scratch_ok {
                    let st = self.pending.pop_request().unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    // Physically allocate the worst case now: nothing can take
                    // these blocks, so this request can never fail mid-flight.
                    self.bm.ensure_capacity(id, need * bs);
                    self.reserved_blocks += need;
                    self.record_decision(id, crate::policy::PolicyReason::Admitted, need);
                    continue;
                }
            } else {
                let n = head.req.prompt_ids.len();
                let text_usable_blocks = self.bm.group_capacity(0);
                let (cached_prefix_blocks, cached_prefix_free_blocks) = self
                    .prefix_cache
                    .cached_blocks_for_admission(head, &self.bm, bs);
                let cached_prefix_blocks = cached_prefix_blocks.min(n.div_ceil(bs));
                let cached_prefix_tokens = cached_prefix_blocks.saturating_mul(bs);
                let uncached_remaining = n.saturating_sub(cached_prefix_tokens);
                let first_uncached_chunk = uncached_remaining
                    .min(self.config.long_prefill_threshold)
                    .min(self.config.max_num_batched_tokens)
                    .max(1);
                let first_chunk_blocks = cached_prefix_tokens
                    .saturating_add(first_uncached_chunk)
                    .div_ceil(bs)
                    .saturating_sub(cached_prefix_blocks);
                if n > text_usable_blocks * bs {
                    let st = self.pending.pop_request().unwrap();
                    self.record_decision(
                        st.req.request_id,
                        crate::policy::PolicyReason::RejectedTooLarge,
                        first_chunk_blocks,
                    );
                    self.trace_record(json!({
                        "event": "request_rejected",
                        "at_s": now(),
                        "request_id": st.req.request_id.0,
                        "reason": "too_large",
                        "needed_blocks": first_chunk_blocks,
                        "usable_blocks": text_usable_blocks,
                        "mode": mode_str(st.req.mode),
                        "prompt_tokens": st.req.prompt_ids.len(),
                    }));
                    let _ = st.req.event_tx.send(GenEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                let free_blocks_after_prefix_acquire = self
                    .bm
                    .free_blocks_in_group(0)
                    .saturating_sub(cached_prefix_free_blocks);
                if free_blocks_after_prefix_acquire >= first_chunk_blocks {
                    let st = self.pending.pop_request().unwrap();
                    let id = st.req.request_id;
                    self.admit_running(st);
                    self.record_decision(
                        id,
                        crate::policy::PolicyReason::Admitted,
                        first_chunk_blocks,
                    );
                    continue;
                }
            }

            // No room now. Under Priority, a higher-priority arrival preempts a
            // lower-priority budgeted running request to be admitted; retry if
            // a victim was freed, else stop (the queue is ordered, so later
            // entries can't fit either).
            let head_priority = self.pending.peek_request().map(|s| s.req.priority);
            if let Some(prio) = head_priority
                && self.preempt_for_admission(prio)
            {
                continue;
            }
            break;
        }
    }

    fn admit_running(&mut self, mut st: ReqState) {
        let id = st.req.request_id;
        let q = st.queued_at;
        let scheduled_at = now();
        let queue_wait_us = ((scheduled_at - q).max(0.0) * 1_000_000.0) as u64;
        self.stats
            .timing
            .queue_wait_count
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_total
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_max
            .fetch_max(queue_wait_us, Ordering::Relaxed);
        // lease KvBlock capacity in BLOCKS, the same unit the worker's
        // ResourceRuntime accounts (`len(new_blocks)` against `num_blocks`).
        // Capacity is counted in blocks, not tokens multiplied by `block_size`.
        // `resources.rs` (ResourceLease::capacity) is the unit authority: KvBlock is blocks.
        let kv_capacity = st.worstcase_blocks as u64;
        let mode = mode_str(st.req.mode);
        let phase = phase_str(st.phase);
        let prompt_tokens = st.req.prompt_ids.len();
        let max_tokens = st.req.max_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.reserve_worstcase;
        let worstcase_blocks = st.worstcase_blocks;
        self.emit_st(
            &mut st,
            GenEvent::Scheduled {
                queued_at: q,
                scheduled_at,
            },
        );
        st.trace.push(crate::trace::TraceEvent::at(
            crate::trace::TraceEventKind::Admitted,
        ));
        self.running.insert(id, st);
        self.order.push(id);
        self.trace_record(json!({
            "event": "request_admitted",
            "at_s": scheduled_at,
            "request_id": id.0,
            "queued_at_s": q,
            "queue_wait_s": scheduled_at - q,
            "mode": mode,
            "phase": phase,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "priority": priority,
            "reserve_worstcase": reserve_worstcase,
            "worstcase_blocks": worstcase_blocks,
            "running": self.running.len(),
            "pending": self.pending.len(),
            "free_blocks": self.bm.free_blocks(),
            "reserved_blocks": self.reserved_blocks,
        }));
        // lease this request's KV residency (released at finish/drop/preempt).
        self.ledger.issue(
            id,
            uniserve_worker_wire::ResourceClass::KvBlock,
            kv_capacity,
            uniserve_worker_wire::LeasePolicy::PerRequest,
        );
        let bs = self.caps.block_size as usize;
        if let Some(st) = self.running.get_mut(&id) {
            self.prefix_cache.lookup(st, &mut self.bm, &self.stats, bs);
        }
    }

    /// Assemble the per-step batch: walk the priority order, ask each request for
    /// at most one op, clip prefill chunks to the remaining token budget, and
    /// pair first-dispatch requests with their `NewRequestData` record.
    fn assemble(&mut self) -> (Vec<NewRequestData>, Vec<ForwardOp>) {
        let ids = self.assembly_order();
        let lane = self.select_assembly_lane(&ids);
        let (new_reqs, ops) = self.assemble_pass(&ids, lane);
        if ops.is_empty() && lane == Some(AssemblyLane::Prefill) {
            // Prefill runs first *if possible* (SGLang's order). When no
            // prefill op could actually be built (e.g. blocked on KV memory
            // it may not preempt), fall through to the decode lane instead of
            // idling — otherwise a starved waiting prompt would stall ready
            // decodes forever.
            return self.assemble_pass(&ids, Some(AssemblyLane::Decode));
        }
        (new_reqs, ops)
    }

    fn assemble_pass(
        &mut self,
        ids: &[RequestId],
        lane: Option<AssemblyLane>,
    ) -> (Vec<NewRequestData>, Vec<ForwardOp>) {
        let mut new_reqs: Vec<NewRequestData> = Vec::new();
        let mut ops: Vec<ForwardOp> = Vec::new();
        let mut selected: HashSet<RequestId> = HashSet::new();
        // vLLM's per-step token budget with the clip rule: the budget, not the
        // chunk threshold, is the binding constraint.
        let mut budget: usize = self.config.max_num_batched_tokens;
        // per-step multimodal encode budget.
        let mut encodes_left = self.mm_encode_budget;
        // Text prefill tokens may ride along inside a decode batch (mixed
        // extend+decode forward): the prompt work then shares the decode
        // step's weight sweep instead of paying a full sweep of its own.
        // Mixed prefill rows are appended after the decode rows so the batch
        // keeps an extend row last (graph token-bucket padding extends the
        // last row).
        let mut mixed_left: usize = if lane == Some(AssemblyLane::Decode) {
            self.config.mixed_prefill_tokens
        } else {
            0
        };
        let mut mixed_ops: Vec<ForwardOp> = Vec::new();
        for id in ids.iter().copied() {
            if ops.len() + mixed_ops.len() >= self.config.max_batch {
                break;
            }
            if budget == 0 {
                break;
            }
            if self.has_inflight(id)
                && !self.can_decode_lookahead(id)
                && !self.can_prefill_decode_lookahead(id)
            {
                continue;
            }
            let cancelled = self.running.get(&id).map(|s| s.cancelled).unwrap_or(true);
            if cancelled {
                continue;
            }
            let next_kind = self.peek_next_kind(id);
            let mut mixed_prefill = false;
            if let (Some(target), Some(kind)) = (lane, next_kind)
                && {
                    let candidate_lane = assembly_lane_for_kind(kind);
                    matches!(candidate_lane, AssemblyLane::Prefill | AssemblyLane::Decode)
                        && candidate_lane != target
                }
            {
                mixed_prefill = target == AssemblyLane::Decode
                    && kind == OpKind::PrefillUnd
                    && mixed_left > 0
                    && self
                        .running
                        .get(&id)
                        .is_some_and(|st| st.req.mode == GenMode::Text);
                if !mixed_prefill {
                    continue;
                }
            }
            if next_kind == Some(OpKind::DenoiseGen) && !self.can_schedule_denoise(id) {
                continue;
            }
            // Build an op; on a block-budget miss, preempt a budgeted victim
            // and retry — else skip this request for the step.
            let mut tries = 0usize;
            loop {
                let op_budget = if mixed_prefill {
                    budget.min(mixed_left)
                } else {
                    budget
                };
                match self.next_op(id, op_budget) {
                    Some(mut op) => {
                        // bound encode work per step; defer if over budget.
                        if op.kind == OpKind::VitEncode || op.kind == OpKind::VaeEncode {
                            if encodes_left == 0 {
                                break;
                            }
                            encodes_left -= 1;
                        }
                        if mixed_prefill {
                            mixed_left = mixed_left.saturating_sub(op_token_cost(&op));
                        }
                        budget = budget.saturating_sub(op_token_cost(&op));
                        // Stateful-diff contract: a request's static state and
                        // initial block allocation cross once, ahead of its
                        // first op; the op then carries only deltas.
                        if let Some(st) = self.running.get_mut(&id)
                            && !st.worker_registered
                        {
                            st.worker_registered = true;
                            let neg = (!st.req.neg_prompt_ids.is_empty())
                                .then(|| st.req.neg_prompt_ids.clone());
                            new_reqs.push(NewRequestData {
                                sampling: Some(st.req.sampling.clone()),
                                image: Some(st.req.image.clone()),
                                neg_token_ids: neg,
                                lora_id: st.req.lora_id,
                                block_ids: std::mem::take(&mut op.new_block_ids),
                                group_id: 0,
                                ..NewRequestData::new(id)
                            });
                        }
                        if op.kind == OpKind::DenoiseGen {
                            self.reserve_worker_image_latent_for_denoise(id);
                        }
                        selected.insert(id);
                        if mixed_prefill {
                            mixed_ops.push(op);
                        } else {
                            ops.push(op);
                        }
                        break;
                    }
                    None => {
                        if self
                            .running
                            .get(&id)
                            .map(|st| st.auto_image_pending)
                            .unwrap_or(false)
                        {
                            break;
                        }
                        // A mixed rider must not preempt running decodes to
                        // make room for itself; it waits for a prefill-lane
                        // step instead.
                        if mixed_prefill {
                            break;
                        }
                        if tries < self.running.len() && self.preempt_one(id, &selected) {
                            tries += 1;
                            continue;
                        }
                        break;
                    }
                }
            }
        }
        ops.extend(mixed_ops);
        (new_reqs, ops)
    }

    fn select_assembly_lane(&self, ids: &[RequestId]) -> Option<AssemblyLane> {
        let mut decode_ready = false;
        let mut decode_ready_without_lookahead = false;
        let mut prefill_ready = false;
        for id in ids.iter().copied() {
            if self.running.get(&id).map(|s| s.cancelled).unwrap_or(true) {
                continue;
            }
            let Some(kind) = self.peek_next_kind(id) else {
                continue;
            };
            match assembly_lane_for_kind(kind) {
                AssemblyLane::Decode => {
                    if self.has_inflight(id) {
                        if self.can_decode_lookahead(id) || self.can_prefill_decode_lookahead(id) {
                            decode_ready = true;
                        }
                    } else {
                        decode_ready = true;
                        decode_ready_without_lookahead = true;
                    }
                }
                AssemblyLane::Prefill => {
                    if !self.has_inflight(id) {
                        prefill_ready = true;
                    }
                }
                AssemblyLane::Other => {}
            }
        }
        if decode_ready_without_lookahead {
            Some(AssemblyLane::Decode)
        } else if decode_ready && self.config.mixed_prefill_tokens > 0 {
            // With mixed batching, prompt work rides inside the decode batch
            // (sharing its weight sweep) instead of claiming a sweep of its
            // own; the decode lane wins even when decodes are only
            // lookahead-ready.
            Some(AssemblyLane::Decode)
        } else if prefill_ready && !self.any_prefill_inflight() {
            Some(AssemblyLane::Prefill)
        } else if decode_ready {
            Some(AssemblyLane::Decode)
        } else if prefill_ready {
            // a prefill batch is already in flight and no decode can fill the
            // slot: run the waiting prompts anyway rather than idling.
            Some(AssemblyLane::Prefill)
        } else {
            None
        }
    }

    fn assembly_order(&self) -> Vec<RequestId> {
        let mut ids: Vec<(usize, RequestId)> = self.order.iter().copied().enumerate().collect();
        ids.sort_by_key(|(idx, id)| (self.assembly_priority(*id), *idx));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    fn assembly_priority(&self, id: RequestId) -> u8 {
        match self.peek_next_kind(id) {
            Some(OpKind::VitEncode | OpKind::VaeEncode | OpKind::PrefillUnd) => 0,
            Some(
                OpKind::DecodeUnd
                | OpKind::TargetVerifyUnd
                | OpKind::CommitGen
                | OpKind::CommitWriteback,
            ) => 1,
            Some(OpKind::DenoiseGen) => 2,
            // Sample/EncodeFrame are StageRouter-injected stage ops, never produced
            // by this scheduler's per-request FSM; group them with "no op".
            Some(OpKind::Sample | OpKind::EncodeFrame) | None => 3,
        }
    }

    fn can_decode_lookahead(&self, id: RequestId) -> bool {
        if !self.decode_lookahead || !self.has_inflight(id) {
            return false;
        }
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        if st.cancelled
            || st.req.mode != GenMode::Text
            || st.phase != Phase::DecodeUnd
            || st.grammar.is_some()
        {
            return false;
        }
        let depth = self.inflight_decode_count(id);
        if depth == 0
            || self.inflight_has_non_decode(id)
            || self.inflight_has_spec_tokens(id)
            || self.inflight_has_multi_decode(id)
        {
            return false;
        }
        let sp = &st.req.sampling;
        let greedy = sp.temperature <= 0.0;
        let penalties = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        greedy
            && sp.ignore_eos
            && st.req.stop_token_ids.is_empty()
            && st.n_generated >= sp.min_tokens
            && st.n_generated.saturating_add(depth) < st.req.max_tokens
            && sp.n_logprobs == 0
            && sp.bad_words_ids.is_empty()
            && !penalties
    }

    /// Cross-boundary async submission (the analog of SGLang's overlap
    /// scheduler's future-token map): once a request's FINAL prefill chunk is
    /// in flight, its first decode op may be submitted immediately with
    /// `token_source = last_sampled`, reading the prefill's sampled token from
    /// the worker's device-side relay. The first decode step then starts right
    /// after the prefill step instead of waiting one extra pipeline slot for
    /// the prefill result's host round-trip.
    fn can_prefill_decode_lookahead(&self, id: RequestId) -> bool {
        if !self.decode_lookahead {
            return false;
        }
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        if st.cancelled
            || st.req.mode != GenMode::Text
            || st.phase != Phase::Prefill
            || st.grammar.is_some()
        {
            return false;
        }
        // The whole prompt must already be cursored into the in-flight chunk
        // (cursor advances at build time), i.e. the final prefill is running.
        if (st.prompt_cursor as usize) < Self::effective_prompt(st).len() {
            return false;
        }
        let Some(items) = self.inflight_ops.get(&id) else {
            return false;
        };
        if items.len() != 1 || items.iter().any(|op| op.kind != OpKind::PrefillUnd) {
            return false;
        }
        let sp = &st.req.sampling;
        let greedy = sp.temperature <= 0.0;
        let penalties = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        greedy
            && sp.ignore_eos
            && st.req.stop_token_ids.is_empty()
            && st.n_generated >= sp.min_tokens
            // prefill samples one token, the lookahead decode a second.
            && st.n_generated.saturating_add(2) <= st.req.max_tokens
            && sp.n_logprobs == 0
            && sp.bad_words_ids.is_empty()
            && !penalties
    }

    fn supports_spec_decode(&self) -> bool {
        self.caps
            .supported_ops
            .iter()
            .any(|kind| kind == opkind_str(OpKind::TargetVerifyUnd))
    }

    /// Whether the worker implements the VAE half of the understanding
    /// dual-encode. Models without a VAE (e.g. patch-space denoisers) declare
    /// only `vit_encode`; their input images single-encode and budget no VAE
    /// grid tokens.
    fn supports_vae_encode(&self) -> bool {
        self.caps
            .supported_ops
            .iter()
            .any(|kind| kind == opkind_str(OpKind::VaeEncode))
    }

    fn decode_burst_plan(
        &self,
        id: RequestId,
        use_last_sampled: bool,
        pos: usize,
        budget: usize,
        allowed: Option<&[u32]>,
    ) -> (u16, Option<Vec<u32>>) {
        let Some(st) = self.running.get(&id) else {
            return (1, None);
        };
        if self.decode_token_burst <= 1
            || use_last_sampled
            || budget <= 1
            || st.phase != Phase::DecodeUnd
            || st.req.mode == GenMode::InterleaveUnd
            || st.grammar.is_some()
            || allowed.is_some()
        {
            return (1, None);
        }
        let sp = &st.req.sampling;
        let penalties = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        if sp.temperature > 0.0
            || sp.n_logprobs != 0
            || !sp.bad_words_ids.is_empty()
            || penalties
            || st.n_generated < sp.min_tokens
        {
            return (1, None);
        }

        let remaining = st.req.max_tokens.saturating_sub(st.n_generated);
        let count = self
            .decode_token_burst
            .min(remaining.min(u16::MAX as usize).max(1) as u16)
            .min(budget.min(u16::MAX as usize) as u16)
            .min(
                self.decode_burst_kv_cap(id, pos)
                    .min(u16::MAX as usize)
                    .max(1) as u16,
            )
            .max(1);
        if count <= 1 {
            return (1, None);
        }

        let mut stop_ids = Vec::new();
        if !sp.ignore_eos {
            stop_ids.extend(self.ctrl.eos.iter().copied());
        }
        stop_ids.extend(st.req.stop_token_ids.iter().copied());
        if st.req.mode == GenMode::AutoInterleave
            && st.images_done < st.req.image.max_images as usize
        {
            if self.ctrl.start_of_image != 0 {
                stop_ids.push(self.ctrl.start_of_image);
            }
            // Literal multi-token image triggers are host-detected by suffix.
            // Stopping on any member is conservative: it may shorten a burst,
            // but it cannot decode past a trigger that the host would need to
            // observe before scheduling the image phase.
            stop_ids.extend(self.ctrl.image_start_ids.iter().copied());
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        (count, (!stop_ids.is_empty()).then_some(stop_ids))
    }

    fn decode_burst_kv_cap(&self, id: RequestId, pos: usize) -> usize {
        let Some(st) = self.running.get(&id) else {
            return 1;
        };
        if st.req.mode != GenMode::AutoInterleave {
            return usize::MAX;
        }
        let bs = self.caps.block_size as usize;
        let image_slack = self.auto_interleave_remaining_image_span(st);
        let allocated_token_cap = self.bm.blocks_for(id).len().saturating_mul(bs);
        let in_allocated = allocated_token_cap.saturating_sub(pos.saturating_add(image_slack));
        let decode_waiters = self
            .running
            .values()
            .filter(|candidate| {
                candidate.req.mode == GenMode::AutoInterleave
                    && candidate.phase == Phase::DecodeUnd
                    && !candidate.auto_image_pending
                    && !candidate.cancelled
            })
            .count()
            .max(1);
        let shared_free_blocks = self.bm.free_blocks() / decode_waiters;
        in_allocated.saturating_add(shared_free_blocks.saturating_mul(bs))
    }

    fn peek_next_kind(&self, id: RequestId) -> Option<OpKind> {
        let st = self.running.get(&id)?;
        if st.auto_image_pending {
            return None;
        }
        Some(match st.phase {
            Phase::Encode
                if st.req.mode == GenMode::InterleaveUnd
                    && st.iu_encode_step == 0
                    && self.supports_vae_encode() =>
            {
                OpKind::VaeEncode
            }
            Phase::Encode => OpKind::VitEncode,
            Phase::Prefill if self.can_prefill_decode_lookahead(id) => OpKind::DecodeUnd,
            Phase::Prefill => OpKind::PrefillUnd,
            Phase::DecodeUnd => OpKind::DecodeUnd,
            Phase::DenoiseGen => OpKind::DenoiseGen,
            Phase::CommitGen => OpKind::CommitGen,
            Phase::CommitWriteback => OpKind::CommitWriteback,
        })
    }

    fn preemptible(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|st| {
            if st.reserve_worstcase {
                return false;
            }
            // AutoInterleave replay is too expensive under high concurrency: even
            // prompt-only victims can quickly become prompt++generated victims and
            // churn the same long prefix forever. Keep these requests resident.
            if st.req.mode == GenMode::AutoInterleave {
                return false;
            }
            true
        })
    }

    /// Preemption under decode-growth pressure: free a budgeted victim so
    /// `protect` can grow. FCFS evicts the last-admitted budgeted running
    /// request; Priority evicts a strictly lower-priority request. Never
    /// touches a request with an op in flight (its KV is being written), the
    /// protected requester, a higher/equal-priority request under Priority, or
    /// a worst-case-reserving request (their blocks are pre-allocated and
    /// eviction would discard diffusion work).
    fn preempt_one(&mut self, protect: RequestId, selected: &HashSet<RequestId>) -> bool {
        let protect_priority = self.running.get(&protect).map(|st| st.req.priority);
        let candidates: Vec<RequestId> = self
            .order
            .iter()
            .copied()
            .filter(|x| *x != protect && !selected.contains(x) && !self.has_inflight(*x))
            .filter(|x| self.preemptible(*x))
            .filter(|x| {
                self.config.policy != SchedulingPolicy::Priority
                    || protect_priority
                        .map(|priority| self.running[x].req.priority > priority)
                        .unwrap_or(false)
            })
            .collect();
        if candidates.is_empty() {
            return false;
        }
        let victim = match self.config.policy {
            SchedulingPolicy::Priority => *candidates
                .iter()
                .max_by_key(|x| self.running[x].req.priority)
                .unwrap(),
            SchedulingPolicy::Fcfs => *candidates.last().unwrap(), // most-recently-admitted
        };
        self.do_preempt(victim);
        true
    }

    /// Preemption under admission pressure (Priority policy): free a strictly
    /// lower-priority budgeted running request so a higher-priority pending
    /// request can be admitted. Returns true if a victim was preempted.
    fn preempt_for_admission(&mut self, requester_priority: i32) -> bool {
        if self.config.policy != SchedulingPolicy::Priority {
            return false;
        }
        let victim = self
            .order
            .iter()
            .copied()
            .filter(|x| !self.has_inflight(*x))
            .filter(|x| self.preemptible(*x))
            .filter(|x| self.running[x].req.priority > requester_priority)
            .max_by_key(|x| self.running[x].req.priority);
        match victim {
            Some(v) => {
                self.do_preempt(v);
                true
            }
            None => false,
        }
    }

    /// Release a victim's blocks, reset it for recompute (KV rebuilt over
    /// `prompt ++ generated`), and re-queue it so it restarts promptly. The
    /// worker's control record is dropped; resumption re-registers.
    fn do_preempt(&mut self, victim: RequestId) {
        let st = match self.running.remove(&victim) {
            Some(s) => s,
            None => return,
        };
        debug_assert!(
            !st.reserve_worstcase,
            "reserving requests are preemption-exempt"
        );
        self.order.retain(|x| *x != victim);
        self.bm.release(victim);
        // a preempted request is re-queued and re-admitted (which re-issues
        // its lease), so release its current leases now to avoid a stale double.
        self.ledger.release_request(victim);
        if let Err(error) = self
            .executor
            .control_wait(ControlOp::DropRequest(victim), None)
        {
            tracing::warn!(
                request_id = victim.0,
                %error,
                "preemption drop_request control failed"
            );
        }
        let mut st = st;
        st.worker_image_latent_units = 0;
        let mut recompute = st.req.prompt_ids.clone();
        recompute.extend_from_slice(&st.generated_ids);
        st.recompute_ids = Some(recompute);
        st.phase = Phase::Prefill;
        st.pos = 0;
        st.prompt_cursor = 0;
        st.prefix_cached_blocks = 0;
        st.blocks_cached = false;
        st.image_id = 0;
        st.images_done = 0;
        st.text_since_image = 0;
        st.cond_pos = 0;
        st.steps_done = 0;
        st.preempted = true;
        st.worker_registered = false;
        st.blocks_sent = 0;
        st.trace.push(crate::trace::TraceEvent::at(
            crate::trace::TraceEventKind::Preempted,
        ));
        self.trace_record(json!({
            "event": "request_preempted",
            "at_s": now(),
            "request_id": victim.0,
            "generated_tokens": st.generated_ids.len(),
            "prompt_tokens": st.req.prompt_ids.len(),
            "reset_phase": phase_str(st.phase),
            "pending_before_requeue": self.pending.len(),
            "running": self.running.len(),
            "free_blocks": self.bm.free_blocks(),
        }));
        self.pending.prepend_request(st);
        self.stats
            .general
            .preemptions
            .fetch_add(1, Ordering::Relaxed);
        self.record_decision(victim, crate::policy::PolicyReason::Preempted, 0);
    }

    fn submit_batch(&mut self, new_reqs: Vec<NewRequestData>, mut ops: Vec<ForwardOp>) {
        let _span = tracing::trace_span!("scheduler.submit_batch", ops = ops.len()).entered();
        self.step_id += 1;
        let step = self.step_id;
        // /10: stamp each op's submit time (round-trip latency) + assign its
        // op_id (lifecycle correlation), and record an OpSubmitted trace event.
        let submit_at = Instant::now();
        for op in &mut ops {
            let oid = self.next_op_id;
            self.next_op_id += 1;
            op.op_id = Some(oid);
            self.register_inflight(op, submit_at);
            let opk = opkind_str(op.kind);
            if let Some(st) = self.running.get_mut(&op.req_id) {
                // the FSM may only emit op kinds
                // the request's typed program declares (proven across the sim suite).
                // A violation indicates a program/FSM drift bug; surface it in all
                // builds (release included) rather than silently no-op'ing, while
                // still failing hard under test via the debug_assert.
                if !st.program.wire_kinds().contains(opk) {
                    tracing::warn!(
                        "program/FSM drift: FSM emitted op {opk} not declared by the \
                         typed program for {:?}",
                        op.req_id,
                    );
                    debug_assert!(
                        false,
                        "FSM emitted op {opk} not declared by the program for {:?}",
                        op.req_id,
                    );
                }
                let mut ev =
                    crate::trace::TraceEvent::at(crate::trace::TraceEventKind::OpSubmitted);
                ev.op_id = Some(oid);
                ev.op_kind = Some(opk);
                ev.step_id = step;
                st.trace.push(ev);
            }
        }
        self.peak_ops_in_batch = self.peak_ops_in_batch.max(ops.len());
        self.stats
            .general
            .peak_ops
            .fetch_max(ops.len(), Ordering::Relaxed);
        self.stats.general.steps.fetch_add(1, Ordering::Relaxed);
        self.stats
            .general
            .running
            .store(self.running.len(), Ordering::Relaxed);
        self.stats
            .general
            .pending
            .store(self.pending.len(), Ordering::Relaxed);
        self.stats
            .general
            .skipped_waiting
            .store(self.skipped_waiting.len(), Ordering::Relaxed);
        self.stats
            .kv_cache
            .free_blocks
            .store(self.bm.free_blocks(), Ordering::Relaxed);
        let mixed = ops
            .first()
            .is_some_and(|first| ops.iter().any(|op| op.kind != first.kind));
        let op_kinds: Vec<&'static str> = ops.iter().map(|op| opkind_str(op.kind)).collect();
        let req_ids: Vec<u64> = ops.iter().map(|op| op.req_id.0).collect();
        let trace_ops: Vec<_> = ops
            .iter()
            .map(|op| {
                let phase = self.running.get(&op.req_id).map(|st| phase_str(st.phase));
                json!({
                    "request_id": op.req_id.0,
                    "op_id": op.op_id,
                    "op_kind": opkind_str(op.kind),
                    "phase": phase,
                    "modality": format!("{:?}", op.modality),
                    "pos_range": op.pos_range,
                    "token_ids_len": op.token_ids.as_ref().map(|ids| ids.len()).unwrap_or(0),
                    "token_source": format!("{:?}", op.token_source),
                    "spec_token_ids_len": op.spec_token_ids.as_ref().map(|ids| ids.len()).unwrap_or(0),
                    "new_block_ids_len": op.new_block_ids.len(),
                    "token_cost": op_token_cost(op),
                    "timestep_idx": op.timestep_idx,
                    "denoise_step_count": op.denoise_step_count,
                    "decode_token_count": op.decode_token_count,
                    "decode_stop_token_ids_len": op.decode_stop_token_ids.as_ref().map(|ids| ids.len()).unwrap_or(0),
                    "cond_pos": op.cond_pos,
                    "image_in": op.image_in,
                    "mm_hash": op.mm_hash,
                })
            })
            .collect();
        let new_req_ids: Vec<u64> = new_reqs.iter().map(|req| req.req_id.0).collect();
        self.batch_started.insert(step, submit_at);
        self.trace_record(json!({
            "event": "batch_submitted",
            "at_s": now(),
            "step_id": step,
            "batch_size": ops.len(),
            "mixed": mixed,
            "op_kinds": op_kinds.clone(),
            "request_ids": req_ids.clone(),
            "new_request_ids": new_req_ids,
            "ops": trace_ops,
            "scheduler": {
                "policy": policy_str(self.config.policy),
                "max_batch": self.config.max_batch,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
            },
            "running": self.running.len(),
            "pending": self.pending.len(),
            "skipped_waiting": self.skipped_waiting.len(),
            "in_flight_before_submit": self.executor.in_flight(),
            "free_blocks": self.bm.free_blocks(),
            "reserved_blocks": self.reserved_blocks,
            "worker_image_latent_active": self.worker_image_latent_used(),
            "worker_image_latent_capacity": self.caps.max_latent_size,
        }));
        if mixed {
            tracing::debug!(
                step_id = self.step_id,
                ?op_kinds,
                ?req_ids,
                "submitting mixed forward batch"
            );
        }
        let batch = ForwardBatch {
            step_id: self.step_id,
            new_reqs,
            ops,
        };
        let spec_draft_counts: Vec<usize> = batch
            .ops
            .iter()
            .filter_map(|op| op.spec_token_ids.as_ref().map(Vec::len))
            .filter(|count| *count > 0)
            .collect();
        if let Err(e) = self.executor.submit(batch) {
            self.batch_started.remove(&step);
            self.trace_record(json!({
                "event": "batch_submit_failed",
                "at_s": now(),
                "step_id": step,
                "error": format!("{e}"),
            }));
            tracing::error!("executor submit failed: {e}");
            self.fatal = true;
            self.fail_all_inflight(&format!("{e}"));
        } else {
            for count in spec_draft_counts {
                self.stats
                    .spec_decode
                    .num_drafts
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .spec_decode
                    .num_draft_tokens
                    .fetch_add(count as u64, Ordering::Relaxed);
                self.stats
                    .spec_decode
                    .max_draft_tokens
                    .fetch_max(count, Ordering::Relaxed);
            }
        }
    }

    /// Effective prefill sequence: `prompt ++ generated` while recomputing after
    /// Preemption, else the bare prompt.
    fn effective_prompt(st: &ReqState) -> &[u32] {
        st.recompute_ids.as_deref().unwrap_or(&st.req.prompt_ids)
    }

    fn prefilled_auto_image_start(&self, id: RequestId) -> bool {
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        st.req.mode == GenMode::AutoInterleave
            && self.ctrl.start_of_image != 0
            && st.images_done < st.req.image.max_images as usize
            && Self::effective_prompt(st).last().copied() == Some(self.ctrl.start_of_image)
    }

    /// The block-id delta since the last op for this request (the stateful-diff
    /// contract): everything `blocks_for` holds beyond what already crossed.
    fn take_new_blocks(&mut self, id: RequestId) -> Vec<BlockId> {
        let all = self.bm.blocks_for(id);
        let Some(st) = self.running.get_mut(&id) else {
            return Vec::new();
        };
        let sent = st.blocks_sent.min(all.len());
        let new = all[sent..].to_vec();
        st.blocks_sent = all.len();
        new
    }

    /// Build the next op for a running request, given the remaining per-step token
    /// `budget`. Returns `None` only when the block budget can't be satisfied
    /// (the caller may preempt a budgeted victim and retry).
    /// Data-plane causality gate: whether the request's next op may be dispatched
    /// given that its input tensors must be reachable on the worker that will run
    /// that will run it. Delegates to the executor, which is the `StageRouter`
    /// under a disaggregated topology and
    /// reports readiness from its `TensorMover`. Non-disaggregated executors
    /// return `true`, so this is a no-op gate in the single-pool default.
    fn stage_ready(&self, id: RequestId) -> bool {
        self.executor.stage_ready(id)
    }

    fn next_op(&mut self, id: RequestId, budget: usize) -> Option<ForwardOp> {
        if !self.stage_ready(id) {
            return None;
        }
        if self.running.get(&id)?.req.mode == GenMode::InterleaveUnd {
            return self.next_op_iu(id, budget);
        }
        if self.running.get(&id)?.auto_image_pending {
            self.begin_image(id);
            if self.running.get(&id)?.auto_image_pending {
                return None;
            }
        }
        let phase = self.running.get(&id)?.phase;
        // Final-prefill-in-flight requests build their first decode op early
        // (cross-boundary lookahead); the FSM phase itself advances at resolve.
        let phase = if phase == Phase::Prefill && self.can_prefill_decode_lookahead(id) {
            Phase::DecodeUnd
        } else {
            phase
        };
        match phase {
            Phase::Encode => {
                // skip any images already in the encoder cache (pinning them),
                // then emit a VitEncode for the first uncached image. The embedding
                // never crosses the wire — the op carries the content hash + a
                // staged-image handle, the result returns an encoder handle.
                let (mm_items, mut cursor) = {
                    let st = self.running.get(&id)?;
                    (st.req.mm_items.clone(), st.mm_cursor)
                };
                while cursor < mm_items.len() {
                    let hash = mm_items[cursor].hash;
                    if self.enc_cache.lookup(hash).is_some() {
                        self.enc_cache.acquire(hash);
                        if let Some(st) = self.running.get_mut(&id) {
                            st.mm_acquired.push(hash);
                            st.mm_cursor = cursor + 1;
                        }
                        cursor += 1;
                    } else {
                        break;
                    }
                }
                if cursor >= mm_items.len() {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.phase = Phase::Prefill;
                    }
                    return self.next_op(id, budget);
                }
                let item = mm_items[cursor].clone();
                let staged = ((id.0) << 20) | (cursor as u64); // StagedImageId
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::VitEncode,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (item.position, item.position + item.num_tokens),
                    mm_hash: Some(item.hash),
                    image_in: Some(staged),
                    ..Default::default()
                })
            }
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let prompt = Self::effective_prompt(st).to_vec();
                let n = prompt.len();
                let cursor = st.prompt_cursor as usize;
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.long_prefill_threshold)
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1)).min(n);
                if !self.bm.ensure_capacity(id, end) {
                    return None;
                }
                let chunk: Vec<u32> = prompt[cursor..end].to_vec();
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                // Advance the cursor at build time: a request with an op in flight
                // is never re-scheduled, so this is the authoritative chunk end and
                // `resolve` simply checks cursor vs prompt length (no re-derivation).
                if let Some(st) = self.running.get_mut(&id) {
                    st.prompt_cursor = end as u32;
                    st.pos = end as u32;
                }
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::PrefillUnd,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (cursor as u32, end as u32),
                    token_ids: Some(chunk),
                    recent_tokens: recent,
                    allowed_tokens: allowed,
                    suppress_tokens: suppress,
                    ..Default::default()
                })
            }
            Phase::DecodeUnd => {
                let st = self.running.get(&id)?;
                let prefill_lookahead = st.phase == Phase::Prefill;
                let lookahead_depth = self.inflight_decode_count(id);
                let use_last_sampled = lookahead_depth > 0 || prefill_lookahead;
                if use_last_sampled
                    && !self.can_decode_lookahead(id)
                    && !self.can_prefill_decode_lookahead(id)
                {
                    return None;
                }
                let pos = st.pos + lookahead_depth as u32;
                let tok = if use_last_sampled { 0 } else { st.next_token };
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                let (decode_token_count, decode_stop_token_ids) = self.decode_burst_plan(
                    id,
                    use_last_sampled,
                    pos as usize,
                    budget,
                    allowed.as_deref(),
                );
                let spec_token_ids =
                    if !use_last_sampled && budget > 1 && self.supports_spec_decode() {
                        self.running.get(&id).and_then(|st| {
                            self.spec_decode.draft_tokens(
                                st,
                                tok,
                                allowed.as_deref(),
                                suppress.as_deref(),
                            )
                        })
                    } else {
                        None
                    };
                let spec_len = spec_token_ids.as_ref().map_or(0, Vec::len);
                let decode_len = if spec_len == 0 {
                    decode_token_count.max(1) as usize
                } else {
                    1
                };
                let capacity_target =
                    self.decode_capacity_target(id, pos as usize, decode_len, spec_len);
                if !self.bm.ensure_capacity(id, capacity_target) {
                    return None;
                }
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::DecodeUnd,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (pos, pos + 1),
                    token_ids: Some(vec![tok]),
                    token_source: if use_last_sampled {
                        TokenSource::LastSampled
                    } else {
                        TokenSource::Wire
                    },
                    spec_token_ids,
                    decode_token_count: (decode_len > 1).then_some(decode_len as u16),
                    decode_stop_token_ids,
                    recent_tokens: recent,
                    allowed_tokens: allowed,
                    suppress_tokens: suppress,
                    ..Default::default()
                })
            }
            Phase::DenoiseGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                let nvae = self.num_vae(&st.req.image) as usize;
                if st.req.image.retain_images
                    && !self.bm.ensure_capacity(
                        id,
                        cond_pos as usize + nvae + self.cap_commit_marker_tokens(),
                    )
                {
                    return None;
                }
                let timestep = st.steps_done;
                let remaining = st.req.image.steps.saturating_sub(timestep).max(1);
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                // pure text->image path runs single-branch CFG.
                let cfg = cfg_params(&st.req.image, 1);
                let image_prompt = Self::image_prompt_for(st);
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::DenoiseGen,
                    modality: Modality::Gen,
                    new_block_ids: Vec::new(),
                    pos_range: (cond_pos, cond_pos + 1),
                    timestep_idx: Some(timestep),
                    denoise_step_count: Some(denoise_step_count),
                    cond_pos: Some(cond_pos),
                    cfg: Some(cfg),
                    image_prompt,
                    ..Default::default()
                })
            }
            Phase::CommitGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::CommitGen,
                    modality: Modality::Gen,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (cond_pos, cond_pos + 1),
                    cond_pos: Some(cond_pos),
                    recent_tokens: recent,
                    allowed_tokens: allowed,
                    suppress_tokens: suppress,
                    ..Default::default()
                })
            }
            Phase::CommitWriteback => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::CommitWriteback,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (cond_pos, cond_pos + 1),
                    cond_pos: Some(cond_pos),
                    recent_tokens: recent,
                    allowed_tokens: allowed,
                    suppress_tokens: suppress,
                    ..Default::default()
                })
            }
        }
    }

    // ====================== image-understanding FSM ======================
    // ThinkMorph interleave: prefill [SYS], dual-encode the input image (VAE clean
    // ⊕ ViT) at its in-prompt position, prefill [question], decode text; on the
    // literal `<image_start>` in a finished round, 3-branch-CFG denoise a reasoning
    // image, dual re-encode it back, and continue. KV length (`kvlen`) and rope
    // (`pos`) diverge once an image block (N+2 KV positions, 1 rope position) is
    // spliced in.

    /// Build the next op for an InterleaveUnd request. Block addressing uses the
    /// full worst-case mapping (exact image token counts come from the worker),
    /// which the reservation allocated at admission.
    /// Burst plan for understanding-mode text decode. Rounds always close on
    /// <|im_end|>, so eos is unconditionally a burst stop; the KV capacity is
    /// pre-reserved worst-case at admission, so no per-token block gating is
    /// needed beyond the step budget.
    fn iu_decode_burst_plan(&self, id: RequestId, budget: usize) -> (u16, Option<Vec<u32>>) {
        let Some(st) = self.running.get(&id) else {
            return (1, None);
        };
        let sp = &st.req.sampling;
        let penalties = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        if self.decode_token_burst <= 1
            || budget <= 1
            || st.iu_closing
            || st.grammar.is_some()
            || sp.temperature > 0.0
            || sp.n_logprobs != 0
            || !sp.bad_words_ids.is_empty()
            || penalties
            || st.n_generated < sp.min_tokens
        {
            return (1, None);
        }
        let remaining = st.req.max_tokens.saturating_sub(st.n_generated);
        let count = self
            .decode_token_burst
            .min(remaining.min(u16::MAX as usize).max(1) as u16)
            .min(budget.min(u16::MAX as usize) as u16)
            .max(1);
        if count <= 1 {
            return (1, None);
        }
        let mut stop_ids: Vec<u32> = self.ctrl.eos.clone();
        stop_ids.extend(st.req.stop_token_ids.iter().copied());
        stop_ids.sort_unstable();
        stop_ids.dedup();
        (count, Some(stop_ids))
    }

    fn next_op_iu(&mut self, id: RequestId, budget: usize) -> Option<ForwardOp> {
        let (phase, wc) = {
            let st = self.running.get(&id)?;
            (st.phase, st.worstcase_blocks)
        };
        self.bm
            .ensure_capacity(id, wc * self.caps.block_size as usize);
        match phase {
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let prompt = st.req.prompt_ids.clone();
                let n = prompt.len();
                let cursor = st.prompt_cursor as usize;
                let next_img = st
                    .req
                    .mm_items
                    .get(st.mm_cursor)
                    .map(|m| m.position as usize);
                // at an unencoded image position -> switch to Encode
                if next_img == Some(cursor) {
                    if let Some(s) = self.running.get_mut(&id) {
                        s.phase = Phase::Encode;
                        s.iu_encode_step = 0;
                    }
                    return self.next_op_iu(id, budget);
                }
                // chunk to the next image boundary, clipped by the step budget.
                let boundary = match next_img {
                    Some(p) if p > cursor => p,
                    _ => n,
                };
                let end = boundary
                    .min(cursor + budget.max(1))
                    .min(n)
                    .max(cursor + 1)
                    .min(n);
                let chunk = prompt[cursor..end].to_vec();
                let pos = st.pos;
                let dn = (end - cursor) as u32;
                if let Some(s) = self.running.get_mut(&id) {
                    s.prompt_cursor = end as u32;
                    s.pos += dn;
                    s.kvlen += dn;
                }
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::PrefillUnd,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (pos, pos + dn),
                    token_ids: Some(chunk),
                    ..Default::default()
                })
            }
            Phase::Encode => {
                let st = self.running.get(&id)?;
                let item = st.req.mm_items.get(st.mm_cursor)?.clone();
                let pos = st.pos;
                let step = st.iu_encode_step;
                let (kind, modality) = if step == 0 && self.supports_vae_encode() {
                    (OpKind::VaeEncode, Modality::Gen)
                } else {
                    (OpKind::VitEncode, Modality::Und)
                };
                if let Some(s) = self.running.get_mut(&id) {
                    s.pos += 1;
                } // one rope per block
                Some(ForwardOp {
                    req_id: id,
                    kind,
                    modality,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (pos, pos + 1),
                    cond_pos: Some(pos),
                    image_b64: Some(item.b64.clone()),
                    mm_hash: Some(item.hash),
                    ..Default::default()
                })
            }
            Phase::DecodeUnd => {
                let st = self.running.get(&id)?;
                let (pos, tok) = (st.pos, st.next_token);
                let (burst_count, burst_stop_ids) = self.iu_decode_burst_plan(id, budget);
                if let Some(s) = self.running.get_mut(&id) {
                    s.pos += 1;
                    s.kvlen += 1;
                }
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::DecodeUnd,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (pos, pos + 1),
                    token_ids: Some(vec![tok]),
                    decode_token_count: (burst_count > 1).then_some(burst_count),
                    decode_stop_token_ids: burst_stop_ids,
                    ..Default::default()
                })
            }
            Phase::DenoiseGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                let timestep = st.steps_done;
                let remaining = st.req.image.steps.saturating_sub(timestep).max(1);
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let cfg = cfg_params(
                    &st.req.image,
                    self.caps.max_cfg_branches.max(1).min(u8::MAX as u32) as u8,
                );
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::DenoiseGen,
                    modality: Modality::Gen,
                    new_block_ids: Vec::new(),
                    pos_range: (cond_pos, cond_pos + 1),
                    timestep_idx: Some(timestep),
                    denoise_step_count: Some(denoise_step_count),
                    cond_pos: Some(cond_pos),
                    cfg: Some(cfg),
                    ..Default::default()
                })
            }
            Phase::CommitGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::CommitGen,
                    modality: Modality::Gen,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (cond_pos, cond_pos + 1),
                    cond_pos: Some(cond_pos),
                    ..Default::default()
                })
            }
            Phase::CommitWriteback => {
                let st = self.running.get(&id)?;
                let cond_pos = st.cond_pos;
                Some(ForwardOp {
                    req_id: id,
                    kind: OpKind::CommitWriteback,
                    modality: Modality::Und,
                    new_block_ids: self.take_new_blocks(id),
                    pos_range: (cond_pos, cond_pos + 1),
                    cond_pos: Some(cond_pos),
                    ..Default::default()
                })
            }
        }
    }

    fn resolve_iu(&mut self, id: RequestId, kind: OpKind, sr: uniserve_worker_wire::SeqResult) {
        match kind {
            OpKind::PrefillUnd => {
                self.bm.activate(id);
                let (cursor, n) = {
                    let st = self.running.get(&id).unwrap();
                    (st.prompt_cursor as usize, st.req.prompt_ids.len())
                };
                if cursor >= n {
                    // prompt fully prefilled -> open the assistant turn with <bos>.
                    if let Some(st) = self.running.get_mut(&id) {
                        st.next_token = self.ctrl.bos;
                        st.round_tokens.clear();
                        st.phase = Phase::DecodeUnd;
                    }
                }
                // else: more prefill (the next op continues, or switches to Encode).
            }
            OpKind::VaeEncode | OpKind::VitEncode => {
                let added = sr.num_tokens.unwrap_or(0);
                if let Some(st) = self.running.get_mut(&id) {
                    st.kvlen += added;
                    if kind == OpKind::VaeEncode {
                        if let Some((h, w)) = sr.image_hw {
                            st.gen_h = h;
                            st.gen_w = w;
                            st.req.image.height = h;
                            st.req.image.width = w;
                        }
                        st.iu_encode_step = 1; // ViT next
                    } else {
                        st.mm_cursor += 1;
                        st.iu_encode_step = 0;
                        st.phase = Phase::Prefill; // continue with the question
                    }
                }
            }
            OpKind::DecodeUnd => {
                self.bm.activate(id);
                let closing = self.running.get(&id).map(|s| s.iu_closing).unwrap_or(false);
                if closing {
                    // the <|im_end|> is now committed; decide image vs finish.
                    if let Some(s) = self.running.get_mut(&id) {
                        s.iu_closing = false;
                    }
                    return self.close_iu_round(id);
                }
                if let Some(tokens) = sr.sampled_token_ids.as_ref().filter(|t| !t.is_empty()) {
                    // Worker-side decode burst: every committed token arrives at
                    // once, and when the burst stopped on <|im_end|> the worker's
                    // speculative stop-token forward already fed it into KV — the
                    // round closes here directly instead of via an iu_closing op.
                    let tokens = tokens.clone();
                    let count = tokens.len() as u32;
                    let stopped_on_eos =
                        tokens.last().is_some_and(|tok| self.ctrl.eos.contains(tok));
                    // The op's build advanced pos/kvlen by 1; the worker appended
                    // one KV row per launched forward (`count` without a stop,
                    // `count + 1` including the speculative stop-token feed).
                    let extra = if stopped_on_eos { count } else { count - 1 };
                    if let Some(s) = self.running.get_mut(&id) {
                        s.pos += extra;
                        s.kvlen += extra;
                    }
                    for &tok in &tokens {
                        if self.ctrl.eos.contains(&tok) {
                            break;
                        }
                        self.emit_text(id, tok, None);
                        let (n_gen, max_tokens) = {
                            let st = self.running.get_mut(&id).unwrap();
                            st.round_tokens.push(tok);
                            st.n_generated += 1;
                            st.next_token = tok;
                            (st.n_generated, st.req.max_tokens)
                        };
                        if n_gen >= max_tokens {
                            return self.finish(id, FinishReason::MaxTokens);
                        }
                    }
                    if stopped_on_eos {
                        return self.close_iu_round(id);
                    }
                    return;
                }
                let tok = sr.sampled_token_id.unwrap_or(self.ctrl.eos[0]);
                if self.ctrl.eos.contains(&tok) {
                    // close the round: feed <|im_end|> into KV, transition on its resolve.
                    if let Some(s) = self.running.get_mut(&id) {
                        s.next_token = self.ctrl.eos[0];
                        s.iu_closing = true;
                    }
                    return;
                }
                self.emit_text(id, tok, sr.sampled_logprob);
                let (n_gen, max_tokens) = {
                    let st = self.running.get_mut(&id).unwrap();
                    st.round_tokens.push(tok);
                    st.n_generated += 1;
                    st.next_token = tok;
                    (st.n_generated, st.req.max_tokens)
                };
                if n_gen >= max_tokens {
                    self.finish(id, FinishReason::MaxTokens);
                }
            }
            OpKind::DenoiseGen => {
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = st.steps_done;
                    st.steps_done = sr.num_steps_done.unwrap_or(st.steps_done + 1);
                    (st.image_id, st.gen_h, st.gen_w, st.req.image.steps, prev)
                };
                let sd = self.running.get(&id).map(|s| s.steps_done).unwrap_or(0);
                if prev_sd == 0 && sd >= 1 {
                    self.emit(
                        id,
                        GenEvent::ImageBegin {
                            image_id,
                            height: h,
                            width: w,
                            steps,
                        },
                    );
                }
                for step in prev_sd.saturating_add(1)..=sd {
                    self.emit(id, GenEvent::ImageStep { image_id, step });
                }
                if sr.denoise_done
                    && let Some(st) = self.running.get_mut(&id)
                {
                    st.phase = Phase::CommitGen;
                }
            }
            OpKind::CommitGen | OpKind::CommitWriteback => {
                if kind == OpKind::CommitGen && sr.locator.is_some() {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.phase = Phase::CommitWriteback;
                    }
                    if let Some(b64) = sr.image_png_b64 {
                        let image_id = self.running.get(&id).map(|st| st.image_id).unwrap_or(0);
                        self.emit(id, image_done_event(image_id, b64));
                    }
                    return;
                }
                self.bm.activate(id);
                self.bm.release_scratch(id);
                // the image committed — release its latent + scratch leases
                // (the request continues; its KV lease persists until finish).
                self.ledger
                    .release_class(id, uniserve_worker_wire::ResourceClass::ImageLatent);
                self.ledger
                    .release_class(id, uniserve_worker_wire::ResourceClass::Scratch);
                let (image_id, cond_pos) = {
                    let st = self.running.get(&id).unwrap();
                    (st.image_id, st.cond_pos)
                };
                if let Some(b64) = sr.image_png_b64 {
                    self.emit(id, image_done_event(image_id, b64));
                }
                let added = sr.num_tokens.unwrap_or(0);
                let gen_rope_advance = self.cap_gen_rope_advance();
                if let Some(st) = self.running.get_mut(&id) {
                    st.worker_image_latent_units = 0;
                    st.kvlen += added;
                    st.pos = cond_pos + gen_rope_advance;
                    st.images_done += 1;
                    // resume text: a fresh assistant turn (<bos>), like native gen_text.
                    st.next_token = self.ctrl.bos;
                    st.round_tokens.clear();
                    st.phase = Phase::DecodeUnd;
                }
            }
            _ => {}
        }
    }

    /// Decide image-vs-finish once a round's <|im_end|> KV is committed —
    /// shared by the legacy iu_closing op resolve and the burst path (whose
    /// speculative stop-token forward already fed <|im_end|>).
    fn close_iu_round(&mut self, id: RequestId) {
        let (triggered, images_done, max_images, n_gen, max_tokens) = {
            let Some(st) = self.running.get(&id) else {
                return;
            };
            (
                contains_subseq(&st.round_tokens, &self.ctrl.image_start_ids),
                st.images_done,
                st.req.image.max_images as usize,
                st.n_generated,
                st.req.max_tokens,
            )
        };
        if triggered && images_done < max_images && n_gen < max_tokens {
            return self.begin_image_iu(id);
        }
        self.finish(
            id,
            if n_gen >= max_tokens {
                FinishReason::MaxTokens
            } else {
                FinishReason::Eos
            },
        )
    }

    fn begin_image_iu(&mut self, id: RequestId) {
        let nvae = {
            let st = self.running.get(&id).unwrap();
            self.num_vae_hw(st.gen_h, st.gen_w)
        };
        self.bm.reserve_scratch(id, nvae);
        // physical scratch reserved in latent tokens above; the observe-only
        // ledger Scratch lease is in CFG branch slots to match the worker.
        let scratch_branches = self.cap_max_cfg_branches();
        // lease the reasoning-image latent + scratch (released at commit).
        self.ledger.issue(
            id,
            uniserve_worker_wire::ResourceClass::ImageLatent,
            nvae,
            uniserve_worker_wire::LeasePolicy::Pinned,
        );
        self.ledger.issue(
            id,
            uniserve_worker_wire::ResourceClass::Scratch,
            scratch_branches,
            uniserve_worker_wire::LeasePolicy::PerRequest,
        );
        if let Some(st) = self.running.get_mut(&id) {
            st.worker_image_latent_units = 0;
            st.cond_pos = st.pos; // rope after the thinking text + <|im_end|>
            st.steps_done = 0;
            st.image_id += 1;
            st.phase = Phase::DenoiseGen;
        }
    }

    /// bounded recent-output window for penalties (only carried when a
    /// penalty is active, to keep the op small — risk 4).
    fn recent_tokens(&self, id: RequestId) -> Option<Vec<u32>> {
        let st = self.running.get(&id)?;
        let sp = &st.req.sampling;
        let active = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        if !active || st.generated_ids.is_empty() {
            return None;
        }
        let n = st.generated_ids.len();
        let start = n.saturating_sub(self.penalty_window);
        Some(st.generated_ids[start..].to_vec())
    }

    /// run the host-side logits-processor pipeline to compute the op's
    /// allowed/suppress masks (min-tokens, bad-words, allowed-tokens).
    fn token_masks(&self, id: RequestId) -> (Option<Vec<u32>>, Option<Vec<u32>>) {
        let st = match self.running.get(&id) {
            Some(s) => s,
            None => return (None, None),
        };
        let ctx = crate::logits::ProcCtx {
            n_generated: st.n_generated,
            eos: &self.ctrl.eos,
            generated: &st.generated_ids,
            sampling: &st.req.sampling,
        };
        let (mut allowed, mut suppress) = crate::logits::run_pipeline(&self.logits_pipeline, &ctx);
        // Image-budget enforcement (host-side mask): once an interleave
        // request has drawn its max_images, the image-start token is
        // suppressed, so an eager or positively-biased model must return to
        // text/EOS instead of emitting un-actionable image triggers into its
        // own context forever. Suppression beats logit bias (bias skips
        // -inf'd logits on the worker).
        if st.req.mode == GenMode::AutoInterleave
            && st.images_done >= st.req.image.max_images as usize
            && self.ctrl.start_of_image != 0
        {
            suppress
                .get_or_insert_with(Vec::new)
                .push(self.ctrl.start_of_image);
        }
        // If the model's text turn is naturally complete while image budget
        // remains, resolve treats EOS as a clean image boundary instead of
        // making the model invent filler text.
        // Structured outputs: the grammar's per-step mask intersects whatever
        // the pipeline allows. A completed grammar restricts to the stop set so
        // the request terminates on the next step.
        if let Some(matcher) = &st.grammar {
            let g_allowed = if matcher.is_complete() {
                let mut stops = self.ctrl.eos.clone();
                stops.extend(st.req.stop_token_ids.iter().copied());
                stops.sort();
                stops.dedup();
                stops
            } else {
                matcher.allowed_next()
            };
            allowed = Some(match allowed {
                Some(a) => a.into_iter().filter(|t| g_allowed.contains(t)).collect(),
                None => g_allowed,
            });
        }
        (allowed, suppress)
    }

    fn resolve_spec_decode_text(
        &mut self,
        id: RequestId,
        sr: uniserve_worker_wire::SeqResult,
        draft_token_ids: Vec<u32>,
    ) {
        self.bm.activate(id);
        let accepted = (sr.num_accepted_tokens.unwrap_or(0) as usize).min(draft_token_ids.len());
        let sampled = sr.sampled_token_id.unwrap_or(self.ctrl.eos[0]);
        if let Some(st) = self.running.get_mut(&id) {
            st.pos += 1 + accepted as u32;
            st.phase = Phase::DecodeUnd;
        }
        let mut outputs: Vec<(u32, Option<f32>, bool)> = draft_token_ids
            .iter()
            .take(accepted)
            .map(|token| (*token, None, false))
            .collect();
        outputs.push((sampled, sr.sampled_logprob, true));

        for (tok, logprob, is_sampled) in outputs {
            let (ignore_eos, min_tokens, max_tokens, n_generated) = match self.running.get_mut(&id)
            {
                Some(st) => (
                    st.req.sampling.ignore_eos,
                    st.req.sampling.min_tokens,
                    st.req.max_tokens,
                    {
                        st.n_generated += 1;
                        st.n_generated
                    },
                ),
                None => return,
            };
            let under_floor = n_generated < min_tokens;
            let stop_tok_hit = !under_floor
                && self
                    .running
                    .get(&id)
                    .map(|s| s.req.stop_token_ids.contains(&tok))
                    .unwrap_or(false);
            if stop_tok_hit {
                return self.finish_with(id, FinishReason::Stop, Some(format!("token:{tok}")));
            }
            let hit_eos = self.ctrl.eos.contains(&tok) && !ignore_eos && !under_floor;
            let hit_max = n_generated >= max_tokens;
            if hit_eos || hit_max {
                if !self.ctrl.eos.contains(&tok) {
                    self.emit_text(id, tok, logprob);
                }
                return self.finish(
                    id,
                    if hit_max {
                        FinishReason::MaxTokens
                    } else {
                        FinishReason::Eos
                    },
                );
            }
            self.emit_text(id, tok, logprob);
            if is_sampled
                && let Some(top) = sr.top_logprobs.clone()
                && !top.is_empty()
            {
                self.emit(id, GenEvent::TokenLogprobs { id: tok, top });
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.next_token = tok;
                if let Some(matcher) = &mut st.grammar {
                    matcher.advance(tok);
                }
            }
        }
    }

    fn resolve_decode_text(&mut self, id: RequestId, sr: uniserve_worker_wire::SeqResult) {
        self.bm.activate(id);
        let tokens = sr
            .sampled_token_ids
            .clone()
            .filter(|ids| !ids.is_empty())
            .unwrap_or_else(|| vec![sr.sampled_token_id.unwrap_or(self.ctrl.eos[0])]);
        if let Some(st) = self.running.get_mut(&id) {
            st.pos += tokens.len() as u32;
        }
        for (idx, tok) in tokens.iter().copied().enumerate() {
            let is_last = idx + 1 == tokens.len();
            let logprob = is_last.then_some(sr.sampled_logprob).flatten();
            let (mode, max_tokens, ignore_eos, min_tokens, images_done, max_images) = {
                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.n_generated += 1;
                (
                    st.req.mode,
                    st.req.max_tokens,
                    st.req.sampling.ignore_eos,
                    st.req.sampling.min_tokens,
                    st.images_done,
                    st.req.image.max_images as usize,
                )
            };
            if tok == self.ctrl.start_of_image
                && mode == GenMode::AutoInterleave
                && images_done < max_images
            {
                self.begin_image(id);
                return;
            }
            let n_gen = self.running.get(&id).map(|s| s.n_generated).unwrap_or(0);
            let under_floor = n_gen < min_tokens;
            let stop_tok_hit = !under_floor
                && self
                    .running
                    .get(&id)
                    .map(|s| s.req.stop_token_ids.contains(&tok))
                    .unwrap_or(false);
            if stop_tok_hit {
                return self.finish_with(id, FinishReason::Stop, Some(format!("token:{tok}")));
            }
            let hit_eos = self.ctrl.eos.contains(&tok) && !ignore_eos && !under_floor;
            let hit_max = n_gen >= max_tokens;
            if hit_eos || hit_max {
                if !self.ctrl.eos.contains(&tok) {
                    self.emit_text(id, tok, logprob);
                }
                return self.finish(
                    id,
                    if hit_max {
                        FinishReason::MaxTokens
                    } else {
                        FinishReason::Eos
                    },
                );
            }
            self.emit_text(id, tok, logprob);
            if is_last
                && let Some(top) = sr.top_logprobs.clone()
                && !top.is_empty()
            {
                self.emit(id, GenEvent::TokenLogprobs { id: tok, top });
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.next_token = tok;
                st.phase = Phase::DecodeUnd;
                if let Some(matcher) = &mut st.grammar {
                    matcher.advance(tok);
                }
            }
            if mode == GenMode::AutoInterleave
                && images_done < max_images
                && !self.ctrl.image_start_ids.is_empty()
            {
                let triggered = self
                    .running
                    .get(&id)
                    .map(|s| ends_with(&s.generated_ids, &self.ctrl.image_start_ids))
                    .unwrap_or(false);
                if triggered {
                    self.begin_image(id);
                    return;
                }
            }
        }
    }

    fn resolve(
        &mut self,
        id: RequestId,
        kind: OpKind,
        sr: uniserve_worker_wire::SeqResult,
        draft_token_ids: Vec<u32>,
    ) {
        if self.running.get(&id).map(|s| s.req.mode) == Some(GenMode::InterleaveUnd) {
            return self.resolve_iu(id, kind, sr);
        }
        if kind == OpKind::DecodeUnd && !draft_token_ids.is_empty() {
            return self.resolve_spec_decode_text(id, sr, draft_token_ids);
        }
        if kind == OpKind::DecodeUnd {
            return self.resolve_decode_text(id, sr);
        }
        match kind {
            OpKind::PrefillUnd | OpKind::DecodeUnd => {
                self.bm.activate(id);
                // chunked prefill: a prefill op may only have consumed part of
                // the prompt; if so, advance the cursor and stay in Prefill.
                if kind == OpKind::PrefillUnd {
                    // The cursor was advanced to the chunk end at build time; if it
                    // hasn't reached the (effective) prompt end, more chunks remain.
                    let (cursor, prompt_len) = {
                        let st = self.running.get(&id).unwrap();
                        (st.prompt_cursor as usize, Self::effective_prompt(st).len())
                    };
                    if cursor < prompt_len {
                        return; // still prefilling; ignore the (partial) sampled token
                    }
                }
                let (mode, max_tokens, ignore_eos, min_tokens) = {
                    let st = self.running.get_mut(&id).unwrap();
                    if kind == OpKind::PrefillUnd {
                        // prompt (possibly prompt++generated on recompute) is done.
                        st.pos = Self::effective_prompt(st).len() as u32;
                        st.prompt_cursor = st.pos;
                        // recompute finished: future growth is plain decode again.
                        st.recompute_ids = None;
                    } else {
                        st.pos += 1;
                        st.n_generated += 1;
                    }
                    (
                        st.req.mode,
                        st.req.max_tokens,
                        st.req.sampling.ignore_eos,
                        st.req.sampling.min_tokens,
                    )
                };
                // the prompt is fully prefilled now — publish its full
                // blocks to the prefix cache for later requests to reuse.
                if kind == OpKind::PrefillUnd {
                    let bs = self.caps.block_size as usize;
                    if let Some(st) = self.running.get_mut(&id) {
                        self.prefix_cache.cache_blocks(st, &mut self.bm, bs);
                    }
                }
                // Native callers may deliberately force the first image by
                // ending an assistant prefix with the model's image-start
                // control token. Treat that prefilled boundary the same as a
                // sampled boundary so AutoInterleave is not left to sampling
                // luck after an explicit prefix.
                if kind == OpKind::PrefillUnd && self.prefilled_auto_image_start(id) {
                    self.begin_image(id);
                    return;
                }
                // pure t2i: ignore the sampled token, jump straight to denoise
                if mode == GenMode::Image && kind == OpKind::PrefillUnd {
                    self.begin_image(id);
                    return;
                }
                let tok = sr.sampled_token_id.unwrap_or(self.ctrl.eos[0]);
                let logprob = sr.sampled_logprob;
                let (images_done, max_images) = {
                    let st = self.running.get(&id).unwrap();
                    (st.images_done, st.req.image.max_images as usize)
                };
                // the model requested an image inline; honor it while the
                // request is still under its image budget.
                if tok == self.ctrl.start_of_image
                    && mode == GenMode::AutoInterleave
                    && images_done < max_images
                {
                    self.begin_image(id);
                    return;
                }
                let n_gen = self.running.get(&id).map(|s| s.n_generated).unwrap_or(0);
                // min_tokens floor: suppress EOS/stop termination until met.
                let under_floor = n_gen < min_tokens;
                // explicit stop-token termination (distinct from model EOS).
                let stop_tok_hit = !under_floor
                    && self
                        .running
                        .get(&id)
                        .map(|s| s.req.stop_token_ids.contains(&tok))
                        .unwrap_or(false);
                if stop_tok_hit {
                    return self.finish_with(id, FinishReason::Stop, Some(format!("token:{tok}")));
                }
                let hit_eos = self.ctrl.eos.contains(&tok) && !ignore_eos && !under_floor;
                let hit_max = n_gen >= max_tokens;
                if hit_eos || hit_max {
                    if !self.ctrl.eos.contains(&tok) {
                        self.emit_text(id, tok, logprob);
                    }
                    return self.finish(
                        id,
                        if hit_max {
                            FinishReason::MaxTokens
                        } else {
                            FinishReason::Eos
                        },
                    );
                }
                self.emit_text(id, tok, logprob);
                if let Some(top) = sr.top_logprobs.clone()
                    && !top.is_empty()
                {
                    self.emit(id, GenEvent::TokenLogprobs { id: tok, top });
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.next_token = tok;
                    st.phase = Phase::DecodeUnd;
                    if let Some(matcher) = &mut st.grammar {
                        matcher.advance(tok);
                    }
                }
                // ThinkMorph's literal visual-thinking trigger in generation
                // mode: the fine-tune may signal the next image with the
                // literal "<image_start>" text instead of BAGEL's
                // <|vision_start|> token. Honor it the moment the trigger core
                // completes (the understanding FSM matches the same ids at
                // round close).
                if mode == GenMode::AutoInterleave
                    && images_done < max_images
                    && !self.ctrl.image_start_ids.is_empty()
                {
                    let triggered = self
                        .running
                        .get(&id)
                        .map(|s| ends_with(&s.generated_ids, &self.ctrl.image_start_ids))
                        .unwrap_or(false);
                    if triggered {
                        self.begin_image(id);
                        return;
                    }
                }
            }
            OpKind::DenoiseGen => {
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = st.steps_done;
                    st.steps_done = sr.num_steps_done.unwrap_or(st.steps_done + 1);
                    (
                        st.image_id,
                        st.req.image.height,
                        st.req.image.width,
                        st.req.image.steps,
                        prev,
                    )
                };
                let sd = self.running.get(&id).map(|s| s.steps_done).unwrap_or(0);
                if prev_sd == 0 && sd >= 1 {
                    self.emit(
                        id,
                        GenEvent::ImageBegin {
                            image_id,
                            height: h,
                            width: w,
                            steps,
                        },
                    );
                }
                for step in prev_sd.saturating_add(1)..=sd {
                    self.emit(id, GenEvent::ImageStep { image_id, step });
                }
                if sr.denoise_done
                    && let Some(st) = self.running.get_mut(&id)
                {
                    st.phase = Phase::CommitGen;
                }
            }
            OpKind::CommitGen | OpKind::CommitWriteback => {
                if kind == OpKind::CommitGen && sr.locator.is_some() {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.phase = Phase::CommitWriteback;
                    }
                    if let Some(b64) = sr.image_png_b64 {
                        let image_id = self.running.get(&id).map(|st| st.image_id).unwrap_or(0);
                        self.emit(id, image_done_event(image_id, b64));
                    }
                    return;
                }
                self.bm.activate(id);
                self.bm.release_scratch(id);
                // the image committed — release its latent + scratch leases
                // (the request continues; its KV lease persists until finish).
                self.ledger
                    .release_class(id, uniserve_worker_wire::ResourceClass::ImageLatent);
                self.ledger
                    .release_class(id, uniserve_worker_wire::ResourceClass::Scratch);
                let (image_id, mode, cond_pos) = {
                    let st = self.running.get(&id).unwrap();
                    (st.image_id, st.req.mode, st.cond_pos)
                };
                if let Some(b64) = sr.image_png_b64 {
                    self.emit(id, image_done_event(image_id, b64));
                }
                // Advance past the committed image's latent span (num_vae + the two
                // boundary tokens), matching the worst-case footprint sizing.
                let span = self
                    .running
                    .get(&id)
                    .map(|s| self.generated_image_span(&s.req.image) as u32)
                    .unwrap_or(1);
                if let Some(st) = self.running.get_mut(&id) {
                    st.worker_image_latent_units = 0;
                    st.images_done += 1;
                    st.text_since_image = 0;
                    st.pos = cond_pos + span;
                }
                // pure t2i finishes after one image; AutoInterleave round-trips
                // back to text (text → image → text → image …). Termination then
                // happens in DecodeUnd on a genuine terminal condition (EOS or
                // max_tokens) or from a commit-side EOS reported by the worker.
                if mode == GenMode::AutoInterleave {
                    if let Some(tok) = sr.sampled_token_id {
                        let (images_done, max_images, n_gen, max_tokens, hit_eos, start_image) = {
                            let st = self.running.get(&id).unwrap();
                            (
                                st.images_done,
                                st.req.image.max_images as usize,
                                st.n_generated,
                                st.req.max_tokens,
                                self.ctrl.eos.contains(&tok),
                                tok == self.ctrl.start_of_image,
                            )
                        };
                        if start_image && images_done < max_images {
                            self.begin_image(id);
                            return;
                        }
                        if hit_eos {
                            return self.finish(id, FinishReason::Eos);
                        }
                        self.emit_text(id, tok, sr.sampled_logprob);
                        if let Some(st) = self.running.get_mut(&id) {
                            st.n_generated += 1;
                            st.next_token = tok;
                            st.phase = Phase::DecodeUnd;
                        }
                        if n_gen + 1 >= max_tokens {
                            return self.finish(id, FinishReason::MaxTokens);
                        }
                        return;
                    }
                    if let Some(st) = self.running.get_mut(&id) {
                        st.phase = Phase::DecodeUnd;
                        st.next_token = self.ctrl.end_of_image;
                    }
                } else {
                    self.finish(id, FinishReason::ImageDone);
                }
            }
            OpKind::VitEncode | OpKind::VaeEncode => {
                // store the worker-side encoder handle in the cache, pin it
                // for this request, and advance to the next image (or prefill).
                let handle = sr.encoder_handle.unwrap_or(0);
                let (hash, total) = {
                    let st = self.running.get(&id).unwrap();
                    let cur = st.mm_cursor.min(st.req.mm_items.len().saturating_sub(1));
                    (
                        st.req.mm_items.get(cur).map(|m| m.hash).unwrap_or(0),
                        st.req.mm_items.len(),
                    )
                };
                if let Some(freed) = self.enc_cache.insert(hash, handle) {
                    self.gated_control(ControlOp::FreeEncoder(vec![freed]));
                }
                self.enc_cache.acquire(hash);
                if let Some(st) = self.running.get_mut(&id) {
                    st.mm_acquired.push(hash);
                    st.mm_cursor += 1;
                    if st.mm_cursor >= total {
                        st.phase = Phase::Prefill;
                    }
                }
            }
            _ => {}
        }
    }

    fn record_auto_image_trigger_for_replay(&mut self, id: RequestId) {
        let start = self.ctrl.start_of_image;
        if start == 0 {
            return;
        }
        if let Some(st) = self.running.get_mut(&id)
            && st.req.mode == GenMode::AutoInterleave
            && st.n_generated > st.generated_ids.len()
            && st.generated_ids.last().copied() != Some(start)
        {
            st.generated_ids.push(start);
        }
    }

    fn promote_auto_interleave_reservation(&mut self, id: RequestId) -> bool {
        let Some((is_auto, pos, remaining_image_span)) = self.running.get(&id).map(|st| {
            (
                st.req.mode == GenMode::AutoInterleave,
                st.pos as usize,
                self.auto_interleave_remaining_image_span(st),
            )
        }) else {
            return false;
        };
        if !is_auto {
            return true;
        }
        let bs = self.caps.block_size as usize;
        let target_tokens = pos.saturating_add(remaining_image_span);
        let need_blocks = target_tokens.div_ceil(bs);
        if need_blocks > self.usable_blocks {
            self.trace_record(json!({
                "event": "auto_interleave_reservation_rejected",
                "at_s": now(),
                "request_id": id.0,
                "needed_blocks": need_blocks,
                "usable_blocks": self.usable_blocks,
            }));
            self.finish(id, FinishReason::Error);
            return false;
        }

        if !self.bm.ensure_capacity(id, target_tokens) {
            self.trace_record(json!({
                "event": "auto_interleave_reservation_deferred",
                "at_s": now(),
                "request_id": id.0,
                "needed_blocks": need_blocks,
                "free_blocks": self.bm.free_blocks(),
                "running": self.running.len(),
                "pending": self.pending.len(),
            }));
            if let Some(st) = self.running.get_mut(&id) {
                st.auto_image_pending = true;
            }
            return false;
        }

        if let Some(st) = self.running.get_mut(&id) {
            st.auto_image_pending = false;
        }
        self.trace_record(json!({
            "event": "auto_interleave_image_capacity_ready",
            "at_s": now(),
            "request_id": id.0,
            "needed_blocks": need_blocks,
            "target_tokens": target_tokens,
            "free_blocks": self.bm.free_blocks(),
            "reserved_blocks": self.reserved_blocks,
        }));
        true
    }

    fn begin_image(&mut self, id: RequestId) {
        if let Some(st) = self.running.get_mut(&id)
            && st.req.mode == GenMode::AutoInterleave
        {
            st.auto_image_pending = true;
        }
        self.record_auto_image_trigger_for_replay(id);
        if !self.promote_auto_interleave_reservation(id) {
            return;
        }
        let nvae = {
            let st = self.running.get(&id).unwrap();
            self.num_vae(&st.req.image) * self.cap_max_cfg_branches()
        };
        self.bm.reserve_scratch(id, nvae);
        // the physical scratch pool is reserved in latent tokens above
        // (`reserve_scratch`), but the observe-only *ledger* Scratch lease is in
        // CFG BRANCH SLOTS — the same unit the worker's ResourceRuntime accounts
        // (`_scratch_units`) — so the host lease magnitude and worker `used`
        // agree.
        let scratch_branches = {
            let st = self.running.get(&id).unwrap();
            cfg_params(&st.req.image, 1).branch_count as u64
        };
        // lease the denoise latent (pinned — evicting discards diffusion
        // work) + the CFG scratch; both released when the image commits.
        self.ledger.issue(
            id,
            uniserve_worker_wire::ResourceClass::ImageLatent,
            nvae,
            uniserve_worker_wire::LeasePolicy::Pinned,
        );
        self.ledger.issue(
            id,
            uniserve_worker_wire::ResourceClass::Scratch,
            scratch_branches,
            uniserve_worker_wire::LeasePolicy::PerRequest,
        );
        if let Some(st) = self.running.get_mut(&id) {
            st.worker_image_latent_units = 0;
            st.cond_pos = st.pos;
            st.steps_done = 0;
            st.image_id += 1;
            st.text_since_image = 0;
            st.phase = Phase::DenoiseGen;
        }
    }

    fn emit(&mut self, id: RequestId, ev: GenEvent) {
        if let Some(st) = self.running.get_mut(&id)
            && st.req.event_tx.send(ev).is_err()
        {
            st.cancelled = true; // receiver dropped == cancel
        }
    }
    /// Emit a text token and record it for preemption-recompute.
    fn emit_text(&mut self, id: RequestId, tok: u32, logprob: Option<f32>) {
        if let Some(st) = self.running.get_mut(&id) {
            st.generated_ids.push(tok);
            st.text_since_image = st.text_since_image.saturating_add(1);
        }
        self.emit(id, GenEvent::TextToken { id: tok, logprob });
    }
    fn emit_st(&self, st: &mut ReqState, ev: GenEvent) {
        if st.req.event_tx.send(ev).is_err() {
            st.cancelled = true;
        }
    }

    fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    fn finish_with(&mut self, id: RequestId, reason: FinishReason, stop_reason: Option<String>) {
        if let Some(mut st) = self.running.remove(&id) {
            self.order.retain(|x| *x != id);
            if st.reserve_worstcase {
                self.reserved_blocks = self.reserved_blocks.saturating_sub(st.worstcase_blocks);
            }
            // release this request's encoder-cache references (evictable now).
            for h in &st.mm_acquired {
                self.enc_cache.release(*h);
            }
            // close + archive the lifecycle trace (reconstructable post-finish).
            let mut ev = crate::trace::TraceEvent::at(crate::trace::TraceEventKind::Finished);
            ev.finish_reason = Some(finish_reason_str(&reason));
            st.trace.push(ev);
            if self.completed_traces.len() >= 64 {
                self.completed_traces.pop_front();
            }
            self.completed_traces.push_back(st.trace.clone());
            self.trace_request_finished(
                id,
                &reason,
                stop_reason.as_deref(),
                st.req.prompt_ids.len(),
                st.n_generated,
                st.images_done,
                "running",
            );
            let _ = st.req.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens: st.req.prompt_ids.len(),
                completion_tokens: st.n_generated,
                images: st.images_done,
            });
        }
        self.bm.release(id);
        // release every lease this request held and assert it leaked none.
        self.ledger.release_request(id);
        self.ledger.assert_released(id);
        let _ = self.executor.control(ControlOp::DropRequest(id));
    }

    /// Clear the encoder cache (`/reset_encoder_cache` / `/reset_mm_cache`)
    /// and report freed handles to the worker.
    fn reset_encoder_cache(&mut self) {
        let freed = self.enc_cache.clear();
        if !freed.is_empty() {
            self.gated_control(ControlOp::FreeEncoder(freed));
        }
    }

    /// The collective_rpc surface: execute one payload-free
    /// control method on every worker rank and await per-rank acks. Executes
    /// inline in the control loop, like vLLM's utility execution in the engine
    /// busy loop.
    fn collective_rpc(&mut self, method: &str) -> Result<Vec<(u32, bool, Option<String>)>, String> {
        let op = ControlOp::from_method(method)
            .ok_or_else(|| format!("unsupported collective_rpc method `{method}`"))?;
        let acks = self
            .executor
            .control_wait(op, None)
            .map_err(|e| e.to_string())?;
        Ok(acks
            .into_iter()
            .map(|a| (a.rank, a.ok, a.message))
            .collect())
    }
}

fn worker_forward_stats_trace(stats: &WorkerForwardStats) -> serde_json::Value {
    json!({
        "mode_counts": stats.mode_counts,
        "mode_tokens": stats.mode_tokens,
        "mode_us": stats.mode_us,
        "component_us": stats.component_us,
        "attention_launches": stats.attention_launches,
        "attention_us": stats.attention_us,
        "attention_backend_counts": stats.attention_backend_counts,
        "cuda_graph_captures": stats.cuda_graph_captures,
        "cuda_graph_replays": stats.cuda_graph_replays,
        "cuda_graph_misses": stats.cuda_graph_misses,
        "cuda_graph_fallbacks": stats.cuda_graph_fallbacks,
        "cuda_graph_unpadded_tokens": stats.cuda_graph_unpadded_tokens,
        "cuda_graph_padded_tokens": stats.cuda_graph_padded_tokens,
        "cuda_graph_runtime_mode_counts": stats.cuda_graph_runtime_mode_counts,
        "text_decode_token_relay_hits": stats.text_decode_token_relay_hits,
        "text_decode_token_relay_misses": stats.text_decode_token_relay_misses,
        "text_decode_position_relay_hits": stats.text_decode_position_relay_hits,
        "text_decode_position_relay_misses": stats.text_decode_position_relay_misses,
        "flashinfer_decode_plan_calls": stats.flashinfer_decode_plan_calls,
        "flashinfer_decode_plan_reuses": stats.flashinfer_decode_plan_reuses,
        "flashinfer_decode_plan_rows": stats.flashinfer_decode_plan_rows,
        "flashinfer_decode_plan_indices": stats.flashinfer_decode_plan_indices,
        "flashinfer_decode_graph_plan_calls": stats.flashinfer_decode_graph_plan_calls,
        "flashinfer_decode_graph_plan_reuses": stats.flashinfer_decode_graph_plan_reuses,
        "spec_verify_rows": stats.spec_verify_rows,
        "spec_verify_draft_tokens": stats.spec_verify_draft_tokens,
        "spec_verify_accepted_tokens": stats.spec_verify_accepted_tokens,
        "spec_verify_rejected_tokens": stats.spec_verify_rejected_tokens,
        "spec_verify_committed_tokens": stats.spec_verify_committed_tokens,
        "spec_verify_path_counts": stats.spec_verify_path_counts,
    })
}

/// Does `hay` contain `needle` as a contiguous subsequence? Used to
/// detect the literal `<image_start>` trigger (its stable "image_start" core) in
/// a finished text round. Empty needle never matches (literal trigger disabled).
fn contains_subseq(hay: &[u32], needle: &[u32]) -> bool {
    if needle.is_empty() || needle.len() > hay.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Does `hay` end with `needle`? (The auto-interleave literal trigger fires on
/// the token that completes the trigger core.) Empty needle never matches.
fn ends_with(hay: &[u32], needle: &[u32]) -> bool {
    !needle.is_empty() && hay.len() >= needle.len() && hay[hay.len() - needle.len()..] == *needle
}

fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    let divisor = divisor.max(1);
    value.div_ceil(divisor)
}

/// Tokens a single op contributes toward the per-step scheduling budget.
/// Prefill contributes its chunk width; decode one; image ops a nominal one.
fn op_token_cost(op: &ForwardOp) -> usize {
    match op.kind {
        OpKind::PrefillUnd => (op.pos_range.1 - op.pos_range.0) as usize,
        OpKind::DecodeUnd => {
            op.decode_token_count.unwrap_or(1).max(1) as usize
                + op.spec_token_ids.as_ref().map_or(0, Vec::len)
        }
        OpKind::DenoiseGen => op.denoise_step_count.unwrap_or(1).max(1) as usize,
        _ => 1,
    }
}

fn decode_lookahead_from_env() -> bool {
    env::var(DECODE_LOOKAHEAD_ENV)
        .map(|raw| {
            !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn denoise_step_burst_from_env() -> u16 {
    env::var(DENOISE_STEP_BURST_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u16>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DENOISE_STEP_BURST)
}

fn decode_token_burst_from_env() -> u16 {
    env::var(DECODE_TOKEN_BURST_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u16>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DECODE_TOKEN_BURST)
}

/// Build the wire [`CfgParams`] for a denoise op.
///
/// The exact text/image CFG branch set is derived by the worker-side CFG plan.
/// The scheduler only carries the per-path branch bound needed by generic
/// denoise accounting.
fn cfg_params(image: &uniserve_core::ImageParams, branch_count: u8) -> CfgParams {
    CfgParams {
        branch_count: branch_count.max(1),
        text_scale: image.cfg_text_scale,
        img_scale: image.cfg_img_scale,
        renorm_type: image.cfg_renorm_type.clone(),
        renorm_min: image.cfg_renorm_min,
        interval: image.cfg_interval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inflight(kind: OpKind, op_id: Option<u64>) -> InflightOp {
        InflightOp {
            kind,
            op_id,
            spec_tokens: Vec::new(),
            decode_token_count: 1,
            started: Instant::now(),
        }
    }

    // a result resolves the exact op the worker echoed by op_id, even when
    // a request's ops complete out of submission order, and falls back to FIFO
    // only when no op_id is echoed.
    #[test]
    fn take_inflight_resolves_by_op_id_out_of_order() {
        let mut q = VecDeque::from(vec![
            inflight(OpKind::PrefillUnd, Some(10)),
            inflight(OpKind::DecodeUnd, Some(11)),
            inflight(OpKind::DecodeUnd, Some(12)),
        ]);
        // Resolve the middle op first (out of order): it must be removed, not the
        // FIFO front.
        let got = take_inflight_by_op_id(&mut q, Some(11)).expect("op 11 present");
        assert_eq!(got.op_id, Some(11));
        assert_eq!(q.len(), 2);
        assert_eq!(q.front().unwrap().op_id, Some(10));
        // Then the last, then the first — order is driven by op_id, not position.
        assert_eq!(
            take_inflight_by_op_id(&mut q, Some(12)).unwrap().op_id,
            Some(12)
        );
        assert_eq!(
            take_inflight_by_op_id(&mut q, Some(10)).unwrap().op_id,
            Some(10)
        );
        assert!(q.is_empty());
    }

    #[test]
    fn take_inflight_falls_back_to_fifo_without_op_id() {
        let mut q = VecDeque::from(vec![
            inflight(OpKind::DecodeUnd, Some(1)),
            inflight(OpKind::DecodeUnd, Some(2)),
        ]);
        // No echoed op_id (sim-style transport before stamping, or a drift):
        // resolve in FIFO order.
        assert_eq!(take_inflight_by_op_id(&mut q, None).unwrap().op_id, Some(1));
        // An unknown op_id also degrades to FIFO rather than dropping the result.
        assert_eq!(
            take_inflight_by_op_id(&mut q, Some(999)).unwrap().op_id,
            Some(2)
        );
        assert!(take_inflight_by_op_id(&mut q, None).is_none());
    }

    #[test]
    fn decode_op_token_cost_includes_speculative_drafts() {
        let plain = ForwardOp {
            kind: OpKind::DecodeUnd,
            ..Default::default()
        };
        let drafted = ForwardOp {
            kind: OpKind::DecodeUnd,
            spec_token_ids: Some(vec![11, 12, 13]),
            ..Default::default()
        };
        let prefill = ForwardOp {
            kind: OpKind::PrefillUnd,
            pos_range: (4, 9),
            spec_token_ids: Some(vec![99]),
            ..Default::default()
        };
        let burst_decode = ForwardOp {
            kind: OpKind::DecodeUnd,
            decode_token_count: Some(16),
            ..Default::default()
        };
        let denoise = ForwardOp {
            kind: OpKind::DenoiseGen,
            denoise_step_count: Some(8),
            ..Default::default()
        };

        assert_eq!(op_token_cost(&plain), 1);
        assert_eq!(op_token_cost(&drafted), 4);
        assert_eq!(op_token_cost(&prefill), 5);
        assert_eq!(op_token_cost(&burst_decode), 16);
        assert_eq!(op_token_cost(&denoise), 8);
    }

    #[test]
    fn ngram_draft_prefers_longest_recent_suffix() {
        let seq = vec![1, 2, 3, 4, 2, 3, 5, 2, 3];
        assert_eq!(ngram_draft_one(&seq, 4, None, None), Some(5));
    }

    #[test]
    fn ngram_draft_respects_allowed_and_suppressed_masks() {
        let seq = vec![9, 8, 7, 9, 8];
        assert_eq!(ngram_draft_one(&seq, 4, Some(&[7]), None), Some(7));
        assert_eq!(ngram_draft_one(&seq, 4, Some(&[6]), None), None);
        assert_eq!(ngram_draft_one(&seq, 4, None, Some(&[7])), None);
    }

    #[test]
    fn cfg_params_uses_declared_branch_count() {
        let mut img = uniserve_core::ImageParams {
            cfg_text_scale: 4.0,
            cfg_img_scale: 4.0,
            ..Default::default()
        };
        assert_eq!(cfg_params(&img, 1).branch_count, 1);
        assert_eq!(cfg_params(&img, 3).branch_count, 3);
        assert_eq!(cfg_params(&img, 0).branch_count, 1);
        img.cfg_text_scale = 1.0;
        img.cfg_img_scale = 1.0;
        assert_eq!(cfg_params(&img, 3).branch_count, 3);
    }

    /// A no-op executor: every submit succeeds, no result ever returns, and
    /// controls are accepted. Enough to unit-test admission/backpressure/reap.
    #[derive(Default)]
    struct NullExecutor {
        caps: EngineCaps,
        in_flight: usize,
    }

    impl Executor for NullExecutor {
        fn caps(&self) -> EngineCaps {
            self.caps.clone()
        }
        fn pipeline_depth(&self) -> usize {
            1
        }
        fn in_flight(&self) -> usize {
            self.in_flight
        }
        fn submit(&mut self, batch: ForwardBatch) -> anyhow::Result<()> {
            let _ = batch;
            self.in_flight += 1;
            Ok(())
        }
        fn poll(&mut self) -> anyhow::Result<Option<ForwardResult>> {
            Ok(None)
        }
        fn next_result(&mut self) -> anyhow::Result<ForwardResult> {
            anyhow::bail!("NullExecutor never returns a result")
        }
        fn control(&mut self, _op: ControlOp) -> anyhow::Result<u64> {
            Ok(0)
        }
        fn control_wait(
            &mut self,
            _op: ControlOp,
            _targets: Option<&[u32]>,
        ) -> anyhow::Result<Vec<uniserve_executor::ControlAck>> {
            Ok(Vec::new())
        }
    }

    fn test_request(id: u64, prompt_len: usize) -> GenerateRequest {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        GenerateRequest::new(
            RequestId(id),
            vec![1u32; prompt_len],
            uniserve_core::SamplingParams::default(),
            uniserve_core::ImageParams::default(),
            GenMode::Text,
            16,
            tx,
        )
    }

    fn test_scheduler() -> Scheduler {
        Scheduler::new(
            Box::new(NullExecutor::default()),
            ControlTokens::default(),
            DEFAULT_MAX_BATCH,
        )
    }

    #[test]
    fn explicit_vae_grid_cap_does_not_expand_to_latent_capacity() {
        let caps = EngineCaps {
            max_latent_size: 65_536,
            max_vae_grid_tokens: 4_096,
            ..Default::default()
        };
        let sched = Scheduler::new(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            DEFAULT_MAX_BATCH,
        );

        assert_eq!(sched.cap_max_vae_grid_tokens(), 4_096);
    }

    #[test]
    fn vae_grid_cap_falls_back_to_legacy_latent_size() {
        let caps = EngineCaps {
            max_latent_size: 65_536,
            max_vae_grid_tokens: 0,
            ..Default::default()
        };
        let sched = Scheduler::new(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            DEFAULT_MAX_BATCH,
        );

        assert_eq!(sched.cap_max_vae_grid_tokens(), 65_536);
    }

    #[test]
    fn assemble_never_preempts_request_already_staged_in_current_batch() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 4, // block 0 is padding, so only 3 usable KV blocks.
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 16,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                ..Default::default()
            },
        );
        let mut receivers = Vec::new();
        for id in 1..=4 {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req = test_request(id, 4);
            req.event_tx = tx;
            receivers.push(rx);
            sched.submit_for_test(req);
        }
        sched.admit();
        assert_eq!(sched.running.len(), 4);
        assert_eq!(sched.order.len(), 4);
        assert_eq!(sched.config.max_batch, 4);
        assert_eq!(sched.config.max_num_batched_tokens, 16);
        assert_eq!(sched.caps.block_size, 4);
        assert_eq!(sched.bm.free_blocks(), 3);

        let (_new_reqs, ops) = sched.assemble();
        let op_ids: Vec<RequestId> = ops.iter().map(|op| op.req_id).collect();

        assert_eq!(
            op_ids,
            vec![RequestId(1), RequestId(2), RequestId(3)],
            "request 4 must wait instead of preempting a request already staged in this batch"
        );
        assert!(
            sched.running.contains_key(&RequestId(3)),
            "staged request 3 must not be dropped before its op is submitted"
        );
        assert!(
            sched.running.contains_key(&RequestId(4)),
            "protected requester remains running and retries next scheduler step"
        );
        assert_eq!(
            sched.pending.len(),
            0,
            "no staged request should be requeued by preemption"
        );
    }

    #[test]
    fn assemble_keeps_ready_text_decode_separate_from_prefill() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 64,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                // mixing disabled: the decode lane must stay pure.
                mixed_prefill_tokens: 0,
                ..Default::default()
            },
        );
        let mut receivers = Vec::new();
        for id in 1..=2 {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req = test_request(id, 4);
            req.sampling.ignore_eos = true;
            req.event_tx = tx;
            receivers.push(rx);
            sched.submit_for_test(req);
        }
        sched.admit();
        assert_eq!(sched.running.len(), 2);
        assert_eq!(sched.order.len(), 2);
        let ids = sched.assembly_order();
        assert_eq!(ids, vec![RequestId(1), RequestId(2)]);
        assert_eq!(
            sched.select_assembly_lane(&ids),
            Some(AssemblyLane::Prefill)
        );
        let (_new_reqs, prefill_ops) = sched.assemble();
        assert_eq!(
            prefill_ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::PrefillUnd, OpKind::PrefillUnd]
        );
        for op in prefill_ops {
            sched.resolve(
                op.req_id,
                op.kind,
                uniserve_worker_wire::SeqResult {
                    req_id: op.req_id,
                    sampled_token_id: Some(11 + op.req_id.0 as u32),
                    ..Default::default()
                },
                Vec::new(),
            );
        }
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(3, 4);
        req.sampling.ignore_eos = true;
        req.event_tx = tx;
        receivers.push(rx);
        sched.submit_for_test(req);
        sched.admit();
        let ids = sched.assembly_order();
        assert_eq!(sched.select_assembly_lane(&ids), Some(AssemblyLane::Decode));

        let (_new_reqs, ops) = sched.assemble();

        assert_eq!(
            ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::DecodeUnd, OpKind::DecodeUnd]
        );
        assert!(
            ops.iter().all(|op| op.req_id != RequestId(3)),
            "ready decode lane must not mix in a text prefill op: {ops:?}"
        );
    }

    #[test]
    fn assemble_coalesces_prompts_while_prefill_inflight() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 64,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                mixed_prefill_tokens: 0,
                ..Default::default()
            },
        );
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(1, 4);
        req.sampling.ignore_eos = true;
        req.event_tx = tx1;
        sched.submit_for_test(req);
        sched.admit();
        let (_new_reqs, ops) = sched.assemble();
        assert_eq!(ops[0].kind, OpKind::PrefillUnd);
        sched.register_inflight(&ops[0], Instant::now());

        // A second prompt arrives while request 1's prefill is in flight:
        // decode work (request 1's lookahead decode) fills the slot and the
        // prompt coalesces into the next prefill batch.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(2, 4);
        req.sampling.ignore_eos = true;
        req.event_tx = tx2;
        sched.submit_for_test(req);
        sched.admit();
        let (_new_reqs, ops2) = sched.assemble();
        assert_eq!(
            ops2.iter().map(|op| (op.kind, op.req_id)).collect::<Vec<_>>(),
            vec![(OpKind::DecodeUnd, RequestId(1))]
        );
        sched.register_inflight(&ops2[0], Instant::now());

        // With no decode left to run, the waiting prompt proceeds even though
        // a prefill batch is still in flight (never idle the pipeline).
        let (_new_reqs, ops3) = sched.assemble();
        assert_eq!(
            ops3.iter().map(|op| (op.kind, op.req_id)).collect::<Vec<_>>(),
            vec![(OpKind::PrefillUnd, RequestId(2))]
        );
    }

    #[test]
    fn assemble_submits_first_decode_while_final_prefill_inflight() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 64,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                mixed_prefill_tokens: 0,
                ..Default::default()
            },
        );
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(1, 4);
        req.sampling.ignore_eos = true;
        req.event_tx = tx;
        sched.submit_for_test(req);
        sched.admit();

        // step 1: the whole prompt goes out as the final prefill chunk.
        let (_new_reqs, ops) = sched.assemble();
        assert_eq!(
            ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::PrefillUnd]
        );
        sched.register_inflight(&ops[0], Instant::now());

        // step 2: with the final prefill still in flight, the first decode op
        // is submitted early, reading its token from the device relay.
        let (_new_reqs, ops2) = sched.assemble();
        assert_eq!(
            ops2.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::DecodeUnd]
        );
        assert_eq!(ops2[0].pos_range, (4, 5));
        assert_eq!(ops2[0].token_source, TokenSource::LastSampled);

        // both in flight: nothing further until a result resolves.
        sched.register_inflight(&ops2[0], Instant::now());
        let (_new_reqs, ops3) = sched.assemble();
        assert!(ops3.is_empty(), "unexpected extra ops: {ops3:?}");
    }

    #[test]
    fn assemble_mixes_small_text_prefill_into_decode_batch() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 64,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                // small budget so the rider's prompt is chunk-clipped to it.
                mixed_prefill_tokens: 2,
                ..Default::default()
            },
        );
        let mut receivers = Vec::new();
        for id in 1..=2 {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req = test_request(id, 4);
            req.sampling.ignore_eos = true;
            req.event_tx = tx;
            receivers.push(rx);
            sched.submit_for_test(req);
        }
        sched.admit();
        let (_new_reqs, prefill_ops) = sched.assemble();
        assert_eq!(prefill_ops.len(), 2);
        for op in prefill_ops {
            sched.resolve(
                op.req_id,
                op.kind,
                uniserve_worker_wire::SeqResult {
                    req_id: op.req_id,
                    sampled_token_id: Some(11 + op.req_id.0 as u32),
                    ..Default::default()
                },
                Vec::new(),
            );
        }
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(3, 4);
        req.sampling.ignore_eos = true;
        req.event_tx = tx;
        receivers.push(rx);
        sched.submit_for_test(req);
        sched.admit();
        let ids = sched.assembly_order();
        assert_eq!(sched.select_assembly_lane(&ids), Some(AssemblyLane::Decode));

        let (_new_reqs, ops) = sched.assemble();

        // Both ready decodes run, and the new request's prefill rides along —
        // appended last, its chunk clipped to the mixed budget.
        assert_eq!(
            ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::DecodeUnd, OpKind::DecodeUnd, OpKind::PrefillUnd]
        );
        let rider = ops.last().unwrap();
        assert_eq!(rider.req_id, RequestId(3));
        assert_eq!(op_token_cost(rider), 2, "rider chunk must clip to the mixed budget");
    }

    #[test]
    fn auto_interleave_ensures_current_image_capacity_before_first_image() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            latent_downsample: 16,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 16,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                ..Default::default()
            },
        );
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(1, 4);
        req.mode = GenMode::AutoInterleave;
        req.max_tokens = 8;
        req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        };
        req.event_tx = tx;
        sched.submit_for_test(req);
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.phase = Phase::DecodeUnd;
            st.pos = 4;
            st.n_generated = 1;
            st.generated_ids.clear();
        }
        sched.begin_image(id);

        let (
            reserve_worstcase,
            auto_image_pending,
            phase,
            generated_ids,
            reserved_blocks,
            allocated_blocks,
            worstcase_blocks,
            image_boundary_blocks,
        ) = {
            let st = sched.running.get(&id).expect("request remains running");
            let image_boundary_blocks = (st.pos as usize
                + sched.generated_image_span(&st.req.image))
            .div_ceil(sched.caps.block_size as usize);
            (
                st.reserve_worstcase,
                st.auto_image_pending,
                st.phase,
                st.generated_ids.clone(),
                sched.reserved_blocks,
                sched.bm.blocks_for(id).len(),
                st.worstcase_blocks,
                image_boundary_blocks,
            )
        };
        assert!(reserve_worstcase);
        assert!(!auto_image_pending);
        assert_eq!(reserved_blocks, worstcase_blocks);
        assert_eq!(allocated_blocks, worstcase_blocks);
        assert!(allocated_blocks >= image_boundary_blocks);
        assert_eq!(generated_ids, vec![sched.ctrl.start_of_image]);
        assert_eq!(phase, Phase::DenoiseGen);
    }

    #[test]
    fn auto_interleave_decode_preserves_remaining_image_capacity() {
        let make_scheduler = |num_blocks| {
            Scheduler::with_config(
                Box::new(NullExecutor {
                    caps: EngineCaps {
                        block_size: 4,
                        num_blocks,
                        latent_downsample: 16,
                        ..Default::default()
                    },
                    in_flight: 0,
                }),
                ControlTokens::default(),
                SchedulerConfig {
                    max_batch: 4,
                    max_num_batched_tokens: 16,
                    max_num_seqs: 4,
                    long_prefill_threshold: 16,
                    ..Default::default()
                },
            )
        };
        let make_request = || {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req = test_request(1, 10);
            req.mode = GenMode::AutoInterleave;
            req.max_tokens = 16;
            req.image = uniserve_core::ImageParams {
                height: 32,
                width: 32,
                max_images: 1,
                retain_images: true,
                ..Default::default()
            };
            req.event_tx = tx;
            req
        };

        let mut full = make_scheduler(5);
        full.submit_for_test(make_request());
        full.admit();
        let id = RequestId(1);
        if let Some(st) = full.running.get_mut(&id) {
            st.phase = Phase::DecodeUnd;
            st.pos = st.req.prompt_ids.len() as u32;
        }
        assert_eq!(full.bm.free_blocks(), 0);
        assert!(
            full.next_op(id, 16).is_none(),
            "decode must wait instead of consuming the retained-image envelope"
        );
        assert_eq!(full.bm.blocks_for(id).len(), 4);

        let mut roomy = make_scheduler(6);
        roomy.decode_token_burst = 8;
        roomy.submit_for_test(make_request());
        roomy.admit();
        if let Some(st) = roomy.running.get_mut(&id) {
            st.phase = Phase::DecodeUnd;
            st.pos = st.req.prompt_ids.len() as u32;
        }
        let op = roomy
            .next_op(id, 16)
            .expect("one free block allows text growth while preserving image capacity");
        assert_eq!(op.kind, OpKind::DecodeUnd);
        assert_eq!(op.decode_token_count, Some(4));
        assert_eq!(roomy.bm.blocks_for(id).len(), 5);
        let st = roomy.running.get(&id).unwrap();
        let target = st.pos as usize
            + op.decode_token_count.unwrap() as usize
            + roomy.auto_interleave_remaining_image_span(st);
        assert!(roomy.bm.blocks_for(id).len() * roomy.caps.block_size as usize >= target);
    }

    fn drain_text_tokens(rx: &mut tokio::sync::mpsc::UnboundedReceiver<GenEvent>) -> Vec<u32> {
        let mut tokens = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let GenEvent::TextToken { id, .. } = ev {
                tokens.push(id);
            }
        }
        tokens
    }

    #[test]
    fn spec_decode_commits_accepted_and_sampled_tokens() {
        let mut sched = test_scheduler();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(1, 4);
        req.event_tx = tx;
        sched.submit_for_test(req);
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.phase = Phase::DecodeUnd;
            st.pos = 4;
            st.next_token = 10;
        }
        sched.resolve_spec_decode_text(
            id,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(13),
                num_accepted_tokens: Some(2),
                ..Default::default()
            },
            vec![11, 12],
        );

        let st = sched.running.get(&id).expect("request still running");
        assert_eq!(st.pos, 7);
        assert_eq!(st.n_generated, 3);
        assert_eq!(st.generated_ids, vec![11, 12, 13]);
        assert_eq!(st.next_token, 13);
        assert_eq!(drain_text_tokens(&mut rx), vec![11, 12, 13]);
    }

    // the waiting queue is bounded — submits past `max_num_waiting` are
    // rejected at enqueue instead of growing the pending queue without bound.
    #[test]
    fn enqueue_rejects_when_waiting_queue_is_full() {
        let mut sched = test_scheduler();
        sched.set_max_num_waiting(2);
        // The first two fit the waiting queue (admission is not run here).
        sched.submit_for_test(test_request(1, 4));
        sched.submit_for_test(test_request(2, 4));
        assert_eq!(sched.pending.len(), 2);
        // The third overflows and must be rejected (not queued).
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut over = test_request(3, 4);
        over.event_tx = tx;
        sched.submit_for_test(over);
        assert_eq!(
            sched.pending.len(),
            2,
            "overflow request must not be queued"
        );
        match rx.try_recv() {
            Ok(GenEvent::Rejected { .. }) => {}
            other => panic!("expected Rejected event, got {other:?}"),
        }
    }

    // a cancelled request with an op still in flight must NOT be finished
    // (its KV freed) until the op resolves; reaping defers it instead.
    #[test]
    fn reap_defers_cancelled_request_with_inflight_op() {
        let mut sched = test_scheduler();
        sched.submit_for_test(test_request(1, 4));
        sched.admit();
        let id = RequestId(1);
        assert!(sched.running.contains_key(&id));
        // Simulate an op in flight for this request.
        let op = ForwardOp {
            req_id: id,
            kind: OpKind::PrefillUnd,
            ..Default::default()
        };
        sched.register_inflight(&op, Instant::now());
        assert!(sched.has_inflight(id));
        // Cancel, then reap: the request must survive while its op is in flight.
        sched.mark_cancelled(id, false);
        sched.reap_cancellations();
        assert!(
            sched.running.contains_key(&id),
            "cancelled request with an in-flight op must not be reaped yet"
        );
        // Once the op drains, the next reap finishes it.
        let _ = sched.pop_inflight(id, None);
        assert!(!sched.has_inflight(id));
        sched.reap_cancellations();
        assert!(
            !sched.running.contains_key(&id),
            "cancelled request must be reaped after its op resolves"
        );
    }

    // An understanding-mode decode burst commits every returned token; a burst
    // that stopped on <|im_end|> closes the round directly (its KV was already
    // fed by the worker's speculative stop-token forward — no iu_closing op).
    #[test]
    fn understanding_decode_burst_commits_tokens_and_closes_round_on_eos() {
        let mut sched = test_scheduler();
        sched.ctrl.eos = vec![99];
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut req = test_request(1, 4);
        req.mode = GenMode::InterleaveUnd;
        req.max_tokens = 64;
        req.event_tx = tx;
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.phase = Phase::DecodeUnd;
            st.pos = 10;
            st.kvlen = 10;
        }

        // Full burst, no stop: every token commits, pos/kvlen advance by k-1
        // beyond the op build's +1 (which this direct resolve call skips).
        sched.resolve_iu(
            id,
            OpKind::DecodeUnd,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(6),
                sampled_token_ids: Some(vec![5, 6]),
                ..Default::default()
            },
        );
        {
            let st = sched.running.get(&id).expect("still running");
            assert_eq!(st.n_generated, 2);
            assert_eq!(st.round_tokens, vec![5, 6]);
            assert_eq!(st.next_token, 6);
            assert_eq!(st.pos, 11);
            assert_eq!(st.kvlen, 11);
        }
        assert_eq!(drain_text_tokens(&mut rx), vec![5, 6]);

        // Burst tail hits eos: the committed tokens emit, the round closes
        // without an iu_closing hop, and (no image trigger) the request ends.
        sched.resolve_iu(
            id,
            OpKind::DecodeUnd,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(99),
                sampled_token_ids: Some(vec![7, 99]),
                ..Default::default()
            },
        );
        assert!(
            !sched.running.contains_key(&id),
            "eos-terminated burst must finish the request"
        );
        assert_eq!(drain_text_tokens(&mut rx), vec![7]);
    }

    // Understanding-interleave input images dual-encode (VAE then ViT) only
    // when the worker declares `vae_encode`; a VAE-less model single-encodes
    // straight through ViT at the image's in-prompt position.
    #[test]
    fn understanding_encode_respects_vae_capability() {
        for (has_vae, expected_first) in [(false, OpKind::VitEncode), (true, OpKind::VaeEncode)] {
            let mut sched = test_scheduler();
            if has_vae {
                sched.caps.supported_ops.push("vae_encode".into());
            }
            let mut req = test_request(1, 5);
            req.mode = GenMode::InterleaveUnd;
            req.mm_items = vec![uniserve_engine_api::MmItem {
                hash: 7,
                position: 2,
                num_tokens: 0,
                b64: "aGVsbG8=".into(),
            }];
            sched.submit_for_test(req);
            sched.admit();
            let id = RequestId(1);

            // Prefill chunks to the image boundary...
            let op = sched.next_op(id, 64).expect("prefill op");
            assert_eq!(op.kind, OpKind::PrefillUnd);
            assert_eq!(op.pos_range, (0, 2));
            // ...then the encode fires at the in-prompt marker gap.
            let op = sched.next_op(id, 64).expect("encode op");
            assert_eq!(op.kind, expected_first, "has_vae={has_vae}");
            assert_eq!(op.cond_pos, Some(2));
            assert_eq!(op.image_b64.as_deref(), Some("aGVsbG8="));
        }
    }

    // AutoInterleave stays resident. Replaying prompt++generated under load is a
    // performance cliff.
    #[test]
    fn auto_interleave_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        req.mode = GenMode::AutoInterleave;
        sched.submit_for_test(req);
        sched.admit();
        assert!(!sched.preemptible(RequestId(1)));
    }

    #[test]
    fn auto_interleave_with_generated_output_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        req.mode = GenMode::AutoInterleave;
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.generated_ids.push(11);
        }
        assert!(!sched.preemptible(id));
    }

    #[test]
    fn auto_interleave_with_committed_image_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        req.mode = GenMode::AutoInterleave;
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.images_done = 1;
        }
        assert!(!sched.preemptible(id));
    }
}
