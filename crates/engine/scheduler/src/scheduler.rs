//! Scheduler control loop: one owner thread drives the whole loop (single-owner,
//! no locks on engine state) — drain commands, advance each running request's
//! generation lifecycle, admit pending requests against the block and scratch budget,
//! assemble a `ForwardBatch`, submit it asynchronously through the `Executor`,
//! and resolve completed `ForwardResult`s into `GenEvent`s and cursor transitions.
//! `ForwardBatch` assembly is lane-aware for text prefill/decode and may still
//! mix compatible non-text ops; workers preserve one result per submitted op.
//!
//! Scheduling follows vLLM-style budgeted, chunked-prefill, preempting scheduling:
//! - waiting requests live in a [`RequestQueue`] (FCFS deque or priority
//!   ordering) and admission consumes the queue head;
//! - scheduling is bounded by the `max_num_batched_tokens` / `max_num_seqs`
//!   pair with vLLM's clip rule (`min(num_new_tokens, token_budget)`);
//! - worst-case reservation is a per-request admission attribute: default and
//!   image-output requests allocate their worst-case KV at admission and are
//!   never preempted; text requests are budgeted and preemptible.
//!
//! The worker contract is a stateful diff: a request's static state crosses once
//! as [`NewRequestData`]; per-step ops carry only deltas (new block ids, new
//! tokens, per-step masks). Preemption resets the diff state (`drop_request` plus
//! re-registration on resumption).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::generation::{
    CursorApplyError, CursorProjection, EncoderCachePin, GenerationCursor,
    GenerationPhase as Phase, GenerationPlanner, PlannedTransition, SchedulerContext,
    TransitionIntent, TransitionValidationError,
};

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
use uniserve_core::{BlockId, CfgParams, ImageIngestStep, encoder_cache_key};
use uniserve_core::{GenerationRequest, GenerationRuntimeCapabilities};
use uniserve_core::{HashAlgo, RequestId};
use uniserve_engine_api::{Command, EventTx, FinishReason, GenEvent, GenerationSubmission};
use uniserve_kv::{BlockManager, EncoderCacheManager};
use uniserve_worker_wire::{
    EngineCaps, ForwardBatch, ForwardOp, ForwardResult, NewRequestData, OpKind, ResourceClass,
    TokenSource, WorkerForwardStats,
};

use crate::grammar::{GrammarCompiler, GrammarMatcher, grammar_allowed_tokens};
use crate::image_artifact::validate_png_artifact;
use crate::queue::{FcfsRequestQueue, PriorityRequestQueue, RequestQueue};
#[cfg(test)]
use crate::spec_decode::ngram_draft_one;
use serde_json::json;
use uniserve_executor::{ControlOp, Executor, WorkerExecError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssemblyLane {
    Prefill,
    Decode,
    Other,
}

/// default per-step multimodal encode budget when the worker caps do not pin it.
const DEFAULT_MM_ENCODE_BUDGET: usize = 4;
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

fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<GenEvent> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(GenEvent::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
        sha256: metadata.sha256,
        pixels_png_b64,
    })
}

fn transient_encoder_worker_key(cache_key: u64, request_id: RequestId) -> u64 {
    let mut value = cache_key ^ request_id.0.rotate_left(23) ^ 0xa076_1d64_78bd_642f_u64;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)).max(1)
}

#[derive(Clone)]
pub struct ControlTokens {
    pub bos: u32,
    pub eos: Vec<u32>,
    pub end_of_image: u32,
}

impl Default for ControlTokens {
    fn default() -> Self {
        Self {
            bos: 151644,
            eos: vec![151645, 151643],
            end_of_image: 151653,
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

pub struct ReqState {
    pub req: GenerationRequest,
    pub(crate) context: SchedulerContext,
    pub(crate) event_tx: EventTx,
    pub(crate) cursor: GenerationCursor,
    pub queued_at: f64,
    pub(crate) cancelled: bool,
    /// The cancel was a server-side abort, not a client cancel.
    pub(crate) aborted: bool,
    /// Structured-output matcher (compiled grammar and progress).
    pub(crate) grammar: Option<GrammarMatcher>,
    /// This request's lifecycle trace.
    pub(crate) trace: crate::trace::RequestTrace,
}

impl ReqState {
    fn has_context_images(&self) -> bool {
        !self.context.images.is_empty()
    }

    fn continues_after_gen_commit(&self) -> bool {
        self.req.behavior.continue_after_gen_commit
    }

    fn can_open_gen_branch(&self) -> bool {
        self.req.behavior.gen_output
            && self.image_gen.images_done < self.req.image.max_images as usize
    }

    fn starts_gen_after_context(&self) -> bool {
        self.req.behavior.start_gen_after_context && self.can_open_gen_branch()
    }

    pub(crate) fn is_replayable_text(&self) -> bool {
        self.replay.replayability == crate::generation::Replayability::Replayable
    }

    pub(crate) fn effective_prompt(&self) -> &[u32] {
        self.replay
            .recompute_ids
            .as_deref()
            .unwrap_or(&self.context.prompt_ids)
    }

    fn pending_image_step(&self) -> Option<ImageIngestStep> {
        self.context
            .images
            .get(self.ingest.mm_cursor)?
            .ingest
            .steps
            .get(self.ingest.pending_image_step)
            .copied()
    }
}

impl Deref for ReqState {
    type Target = GenerationCursor;

    fn deref(&self) -> &Self::Target {
        &self.cursor
    }
}

impl DerefMut for ReqState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cursor
    }
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
    /// An acknowledged reset waits for submitted ops to resolve before it can
    /// preempt request state and invalidate cache ownership atomically.
    pending_prefix_reset: Option<PendingPrefixCacheReset>,
    /// Pluggable host-side logits-processor pipeline.
    logits_pipeline: Vec<Box<dyn crate::logits::LogitsProcessor>>,
    /// Cap on the recent-output window carried for penalties .
    penalty_window: usize,
    /// Encoder-output cache (hashed, LRU, budgeted).
    enc_cache: EncoderCacheManager,
    /// Encoder-cache entries reserved by admitted image requests.
    reserved_encoder_entries: usize,
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
    /// Finite resident-request cohort whose prompt and image-ingest work is
    /// drained before its first decode service. Membership is frozen when a
    /// ready decode would otherwise overlap another resident prompt, so later
    /// arrivals cannot extend the cohort indefinitely.
    prompt_cohort: Option<HashSet<RequestId>>,
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
    planner: GenerationPlanner,
    /// Submit timestamp per in-flight batch (for batch round-trip traces).
    batch_started: HashMap<u64, Instant>,
    /// Monotonic op ids and archived lifecycle traces.
    next_op_id: u64,
    completed_traces: VecDeque<crate::trace::RequestTrace>,
    trace_sink: Option<crate::bench_trace::SchedulerTraceSink>,
    pub peak_ops_in_batch: usize,
    pub stats: Arc<SchedStats>,
}

struct PendingPrefixCacheReset {
    reply: uniserve_engine_api::PrefixCacheResetReply,
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
    pub supported_ops: Vec<OpKind>,
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
        FinishReason::Repetition => "repetition",
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

fn ranked_logprobs(
    entries: Vec<uniserve_worker_wire::TokenLogprob>,
) -> Vec<uniserve_engine_api::TokenLogprob> {
    entries
        .into_iter()
        .map(|entry| uniserve_engine_api::TokenLogprob {
            token_id: entry.0,
            logprob: entry.1,
            rank: entry.2,
        })
        .collect()
}

fn transition_validation_error_str(error: &TransitionValidationError) -> &'static str {
    match error {
        TransitionValidationError::OpIdMismatch { .. } => "op_id_mismatch",
        TransitionValidationError::OpKindMismatch { .. } => "op_kind_mismatch",
        TransitionValidationError::MissingDenoiseStep => "missing_denoise_step",
        TransitionValidationError::DenoiseStepMismatch { .. } => "denoise_step_mismatch",
        TransitionValidationError::MissingEncoderHandle => "missing_encoder_handle",
        TransitionValidationError::EncoderHandleMismatch { .. } => "encoder_handle_mismatch",
        TransitionValidationError::MissingImageArtifact => "missing_image_artifact",
        TransitionValidationError::InvalidImageArtifact => "invalid_image_artifact",
        TransitionValidationError::ImageArtifactDimensionsMismatch { .. } => {
            "image_artifact_dimensions_mismatch"
        }
        TransitionValidationError::MissingImageLocator => "missing_image_locator",
        TransitionValidationError::MissingImageDimensions => "missing_image_dimensions",
        TransitionValidationError::ImageDimensionsMismatch { .. } => "image_dimensions_mismatch",
        TransitionValidationError::MissingImageKvTokens => "missing_image_kv_tokens",
        TransitionValidationError::ImageKvMismatch { .. } => "image_kv_mismatch",
        TransitionValidationError::UnexpectedSampledToken { .. } => "unexpected_sampled_token",
        TransitionValidationError::MissingSampledToken { .. } => "missing_sampled_token",
        TransitionValidationError::AcceptedDraftCountExceeded { .. } => {
            "accepted_draft_count_exceeded"
        }
        TransitionValidationError::TextTokenCountInconsistent { .. } => {
            "text_token_count_inconsistent"
        }
        TransitionValidationError::TextTokenCountMismatch { .. } => "text_token_count_mismatch",
        TransitionValidationError::SampledTokenNotAllowed { .. } => "sampled_token_not_allowed",
        TransitionValidationError::UnexpectedPromptLogprobs { .. } => "unexpected_prompt_logprobs",
        TransitionValidationError::PromptLogprobCountMismatch { .. } => {
            "prompt_logprob_count_mismatch"
        }
        TransitionValidationError::EmptyPromptLogprobPosition { .. } => {
            "empty_prompt_logprob_position"
        }
        TransitionValidationError::PromptLogprobTokenMismatch { .. } => {
            "prompt_logprob_token_mismatch"
        }
        TransitionValidationError::InvalidPromptLogprobCandidates { .. } => {
            "invalid_prompt_logprob_candidates"
        }
        TransitionValidationError::UnexpectedGeneratedLogprobs { .. } => {
            "unexpected_generated_logprobs"
        }
        TransitionValidationError::MissingGeneratedLogprobs { .. } => "missing_generated_logprobs",
        TransitionValidationError::GeneratedLogprobTokenMismatch { .. } => {
            "generated_logprob_token_mismatch"
        }
        TransitionValidationError::InvalidGeneratedLogprobCandidates => {
            "invalid_generated_logprob_candidates"
        }
    }
}

fn cursor_apply_error_str(error: &CursorApplyError) -> &'static str {
    match error {
        CursorApplyError::MissingOperationId => "missing_operation_id",
        CursorApplyError::DuplicateOperation { .. } => "duplicate_operation",
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
        Phase::FeedbackIngest => "feedback_ingest",
    }
}

fn behavior_str(request: &GenerationRequest) -> &'static str {
    if request.behavior.continue_after_gen_commit {
        "und_gen_continuation"
    } else if request.behavior.gen_output {
        "gen_commit_terminal"
    } else {
        "und_decode"
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
    transition: PlannedTransition,
    /// The op's wire `op_id`, echoed back on its result.
    op_id: Option<u64>,
    /// Speculative draft token ids attached to this op (empty when none).
    spec_tokens: Vec<u32>,
    /// Sequential text decode tokens requested by this op. Projected cursors
    /// account for the full count while the device relay chains the next op.
    decode_token_count: u16,
    /// Submit timestamp, for the op's host round-trip latency history.
    started: Instant,
}

/// Remove the in-flight op whose `op_id` matches the worker-echoed `op_id`.
/// Unknown or absent operation identities leave the queue untouched so a
/// worker result can never resolve a different submission.
fn take_inflight_by_op_id(
    queue: &mut VecDeque<InflightOp>,
    op_id: Option<u64>,
) -> Option<InflightOp> {
    let target = op_id?;
    let pos = queue.iter().position(|op| op.op_id == Some(target))?;
    queue.remove(pos)
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
        // Empty means one full-attention group.
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
            pending_prefix_reset: None,
            logits_pipeline: crate::logits::default_pipeline(),
            penalty_window: DEFAULT_PENALTY_WINDOW,
            enc_cache: EncoderCacheManager::new(caps_encoder_budget),
            reserved_encoder_entries: 0,
            mm_encode_budget: DEFAULT_MM_ENCODE_BUDGET,
            running: HashMap::new(),
            order: Vec::new(),
            skipped_waiting: HashMap::new(),
            grammar_compiler: GrammarCompiler::new(),
            reserved_blocks: 0,
            step_id: 0,
            inflight_ops: HashMap::new(),
            prompt_cohort: None,
            decode_lookahead,
            denoise_step_burst,
            decode_token_burst,
            spec_decode,
            fatal: false,
            ledger: crate::resources::ResourceLedger::new(),
            decisions: crate::policy::DecisionLog::default(),
            latency: crate::policy::LatencyHistory::new(),
            planner: GenerationPlanner::new(),
            batch_started: HashMap::new(),
            next_op_id: 1,
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

            if !progressed && self.executor.in_flight() == 0 {
                // Nothing in flight and nothing schedulable this pass — park on
                // the command channel instead of spinning. This covers both the
                // fully idle case and the backpressured case (requests resident
                // or queued but no op currently buildable): any state change
                // arrives as a command or as freed capacity from a command
                // (cancel/finish), and the timeout doubles as the worker
                // liveness probe so a worker that dies with nothing in flight
                // is detected promptly. When gated on grammar compilation we use
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
        if let Some(pending) = self.pending_prefix_reset.take() {
            let _ = pending.reply.send(Err(
                "scheduler stopped before the prefix-cache reset completed".to_string(),
            ));
        }
        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(st) = self.pending.pop_request() {
                ids.push(st.req.request_id);
                let _ = st.event_tx.send(GenEvent::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.context.prompt_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                    kv_transfer_params: None,
                });
            }
            ids
        };
        let _ = queued;
        for (_, st) in self.skipped_waiting.drain() {
            let _ = st.event_tx.send(GenEvent::Finished {
                reason: FinishReason::Aborted,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
                kv_transfer_params: None,
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
            Command::Submit(submission) => self.enqueue(*submission),
            Command::Cancel(id) => self.mark_cancelled(id, false),
            Command::Abort(id) => self.mark_cancelled(id, true),
            Command::ResetPrefixCache {
                reset_running_requests,
                reply,
            } => self.begin_prefix_cache_reset(reset_running_requests, reply),
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

    /// Test harness direct enqueue (the production path is `run` over the channel).
    #[doc(hidden)]
    pub fn submit_for_test(&mut self, request: GenerationRequest) -> uniserve_engine_api::EventRx {
        let (event_tx, event_rx) = uniserve_engine_api::event_channel();
        self.enqueue(GenerationSubmission::new(request, event_tx));
        event_rx
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
                st.context.prompt_ids.len(),
                0,
                0,
                "skipped_waiting",
            );
            let _ = st.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
                kv_transfer_params: None,
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
                st.context.prompt_ids.len(),
                0,
                0,
                "pending",
            );
            let _ = st.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
                kv_transfer_params: None,
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

    fn begin_prefix_cache_reset(
        &mut self,
        reset_running_requests: bool,
        reply: uniserve_engine_api::PrefixCacheResetReply,
    ) {
        if self.pending_prefix_reset.is_some() {
            let _ = reply.send(Ok(false));
            return;
        }
        if !reset_running_requests && !self.running.is_empty() {
            let _ = reply.send(Ok(false));
            return;
        }
        self.pending_prefix_reset = Some(PendingPrefixCacheReset { reply });
        self.progress_prefix_cache_reset();
    }

    /// Complete a reset once no worker op can still write request-owned KV.
    fn progress_prefix_cache_reset(&mut self) -> bool {
        if self.pending_prefix_reset.is_none() || !self.inflight_ops.is_empty() {
            return false;
        }
        let pending = self
            .pending_prefix_reset
            .take()
            .expect("prefix reset presence checked above");
        if self
            .running
            .values()
            .any(|state| !state.is_replayable_text())
        {
            let _ = pending.reply.send(Ok(false));
            return true;
        }
        if self.control_allowed(&ControlOp::ResetPrefixCache)
            && let Err(error) = self
                .executor
                .control_wait(ControlOp::ResetPrefixCache, None)
        {
            let _ = pending.reply.send(Err(error.to_string()));
            return true;
        }

        let running = self.order.clone();
        for request_id in running {
            self.requeue_for_recompute(request_id);
        }
        self.bm.reset_prefix_cache();
        self.publish_cache_stats();
        let _ = pending.reply.send(Ok(true));
        true
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
            "queue": queue,
            "generation": behavior_str(&st.req),
            "initial_phase": phase_str(st.lifecycle.phase),
            "prompt_tokens": st.context.prompt_ids.len(),
            "max_tokens": st.req.max_und_tokens,
            "priority": st.req.priority,
            "reserve_worstcase": st.resources.reserve_worstcase,
            "worstcase_blocks": st.resources.worstcase_blocks,
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

    fn enqueue(&mut self, submission: GenerationSubmission) {
        let GenerationSubmission {
            request: req,
            event_tx,
        } = submission;
        let context = match SchedulerContext::lower(&req) {
            Ok(context) => context,
            Err(error) => {
                let _ = event_tx.send(GenEvent::Rejected {
                    message: format!("invalid generation request: {error:?}"),
                });
                return;
            }
        };
        if let Some(operation) = self.missing_required_operation(&req) {
            let _ = event_tx.send(GenEvent::Rejected {
                message: format!(
                    "generation request requires worker operation `{operation}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.generation_runtime_capabilities()) {
            let _ = event_tx.send(GenEvent::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
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
                "behavior": {
                    "und_decode": req.behavior.und_decode,
                    "und_tokens": format!("{:?}", req.behavior.und_tokens),
                    "gen_output": req.behavior.gen_output,
                    "generated_image_feedback": req.behavior.generated_image_feedback,
                },
                "prompt_tokens": context.prompt_ids.len(),
            }));
            let _ = event_tx.send(GenEvent::Rejected {
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let worst = req
            .resources
            .max_kv_tokens
            .div_ceil(self.caps.block_size as usize);
        // Non-text multimodal requests are not safely preemptible once they
        // start producing image/text state. Reserve their configured bounded KV
        // envelope at admission so excess concurrency queues instead of
        // exhausting KV after all requests have become resident.
        let reserve_worstcase = !context.images.is_empty() || req.behavior.gen_output;
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let phase0 = Phase::Prefill;
        // A lifecycle trace keyed by trace/request/op ids.
        let trace = crate::trace::RequestTrace::new(
            req.request_id,
            uniserve_core::TraceId(req.request_id.0),
        );
        let st = ReqState {
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            event_tx,
            queued_at: now(),
            cancelled: false,
            aborted: false,
            grammar: None,
            trace,
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
        let dl = u64::from(self.caps.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    fn denoise_host_scratch_tokens(&self, st: &ReqState) -> u64 {
        uniserve_core::denoise_scratch_tokens(
            self.num_vae(&st.req.image),
            u64::from(self.caps.commit_marker_tokens),
            u64::from(st.image_gen.cond_pos),
            st.context.negative_prompt_ids.len() as u64,
            u64::from(cfg_branch_count(&st.req.image)),
            u64::from(self.caps.block_size),
        )
    }

    fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.caps.max_vae_grid_tokens > 0 {
            self.caps.max_vae_grid_tokens as usize
        } else {
            self.caps.max_latent_size as usize
        }
    }

    fn cap_commit_marker_tokens(&self) -> usize {
        self.caps.commit_marker_tokens.max(1) as usize
    }

    fn generation_runtime_capabilities(&self) -> GenerationRuntimeCapabilities {
        let mut supported_ops = self.caps.supported_ops.clone();
        supported_ops.sort();
        supported_ops.dedup();
        GenerationRuntimeCapabilities {
            supported_ops,
            max_latent_units: u64::from(self.caps.max_latent_size),
            latent_downsample: self.caps.latent_downsample,
            max_vae_grid_tokens: self.cap_max_vae_grid_tokens() as u32,
            max_vit_grid_tokens: self.caps.max_vit_grid_tokens,
            commit_marker_tokens: self.caps.commit_marker_tokens,
            max_cfg_branches: self.caps.max_cfg_branches,
            scratch_capacity_tokens: self.caps.scratch_capacity_tokens,
            scratch_block_size: self.caps.block_size,
            encoder_cache_entries: self.caps.encoder_cache_budget,
            generated_image_commit: self.executor.generated_image_commit_capabilities(),
        }
    }

    fn missing_required_operation(&self, request: &GenerationRequest) -> Option<OpKind> {
        let context_steps = request.context.iter().flat_map(|segment| match segment {
            uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
            uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
        });
        request
            .behavior
            .required_operations(&request.policy, context_steps)
            .into_iter()
            .find(|operation| !self.caps.supported_ops.contains(operation))
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
            .map(|st| st.resources.worker_image_latent_units)
            .sum()
    }

    fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.caps.latent_downsample as u64).max(1);
        let (height, width) = if st.has_context_images()
            && st.image_gen.image_hw.0 > 0
            && st.image_gen.image_hw.1 > 0
        {
            (st.image_gen.image_hw.0, st.image_gen.image_hw.1)
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
        if st.resources.worker_image_latent_units > 0 {
            Some(0)
        } else {
            Some(self.worker_image_latent_units_for(st).max(1))
        }
    }

    fn can_schedule_denoise(&self, id: RequestId) -> bool {
        let Some(additional) = self.denoise_worker_image_latent_additional(id) else {
            return false;
        };
        let worker_capacity_ok = additional == 0
            || self.worker_image_latent_used().saturating_add(additional)
                <= self.caps.max_latent_size as u64;
        let host_scratch_ok = self.running.get(&id).is_some_and(|st| {
            st.resources.host_scratch_tokens > 0
                || self
                    .bm
                    .can_reserve_scratch(self.denoise_host_scratch_tokens(st))
        });
        worker_capacity_ok && host_scratch_ok
    }

    fn reserve_transition_resources(&mut self, transition: &PlannedTransition) -> bool {
        let id = transition.op.req_id;
        let resources = &transition.resources;
        let cached_encoder_key = match transition.delta {
            crate::generation::TransitionDelta::IngestImageStep {
                encoder_cache_key: Some(key),
                cache_hit: true,
                ..
            } => Some(key),
            _ => None,
        };
        if let Some(key) = cached_encoder_key {
            let Some(handle) = self.enc_cache.acquire(key) else {
                return false;
            };
            if transition.image_in != Some(handle) {
                if let Some(freed) = self.enc_cache.release(key, handle) {
                    self.gated_control(ControlOp::FreeEncoder(vec![freed]));
                }
                return false;
            }
            let Some(st) = self.running.get_mut(&id) else {
                if let Some(freed) = self.enc_cache.release(key, handle) {
                    self.gated_control(ControlOp::FreeEncoder(vec![freed]));
                }
                return false;
            };
            st.ingest
                .acquired_encoder_pins
                .push(EncoderCachePin { key, handle });
        }
        let needs_host_scratch = self
            .running
            .get(&id)
            .is_some_and(|st| st.resources.host_scratch_tokens == 0)
            && resources.host_scratch_tokens > 0;
        if needs_host_scratch && !self.bm.reserve_scratch(id, resources.host_scratch_tokens) {
            if let Some(key) = cached_encoder_key {
                let pin = self
                    .running
                    .get_mut(&id)
                    .and_then(|st| st.ingest.acquired_encoder_pins.pop());
                if let Some(pin) = pin {
                    debug_assert_eq!(pin.key, key);
                    if let Some(freed) = self.enc_cache.release(pin.key, pin.handle) {
                        self.gated_control(ControlOp::FreeEncoder(vec![freed]));
                    }
                }
            }
            return false;
        }
        if resources.latent_units > 0 {
            self.ledger.ensure(
                id,
                uniserve_worker_wire::ResourceClass::ImageLatent,
                resources.latent_units,
                uniserve_worker_wire::LeasePolicy::Pinned,
            );
        }
        if resources.scratch_units > 0 {
            self.ledger.ensure(
                id,
                uniserve_worker_wire::ResourceClass::Scratch,
                resources.scratch_units,
                uniserve_worker_wire::LeasePolicy::PerRequest,
            );
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.resources.worker_image_latent_units = st
                .resources
                .worker_image_latent_units
                .max(resources.latent_units);
            st.resources.scratch_units = st.resources.scratch_units.max(resources.scratch_units);
            st.resources.host_scratch_tokens = st
                .resources
                .host_scratch_tokens
                .max(resources.host_scratch_tokens);
        }
        true
    }

    fn release_transition_resources(&mut self, id: RequestId, transition: &PlannedTransition) {
        for class in &transition.resources.release_on_apply {
            self.ledger.release_class(id, *class);
            match class {
                uniserve_worker_wire::ResourceClass::ImageLatent => {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.resources.worker_image_latent_units = 0;
                    }
                }
                uniserve_worker_wire::ResourceClass::Scratch => {
                    self.bm.release_scratch(id);
                    if let Some(st) = self.running.get_mut(&id) {
                        st.resources.scratch_units = 0;
                        st.resources.host_scratch_tokens = 0;
                    }
                }
                _ => {}
            }
        }
    }

    fn decode_capacity_target(
        &self,
        _id: RequestId,
        pos: usize,
        decode_len: usize,
        spec_len: usize,
    ) -> usize {
        pos.saturating_add(decode_len).saturating_add(spec_len)
    }

    fn image_prompt_for(st: &ReqState) -> Option<String> {
        st.req
            .image
            .image_prompts
            .get(st.image_gen.images_done)
            .cloned()
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

        // A reset transaction drains already-submitted work and admits no new
        // operations until request ownership has been rewound or rejected.
        progressed |= self.progress_prefix_cache_reset();
        if self.pending_prefix_reset.is_some() {
            return progressed;
        }

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
        self.inflight_ops.values().any(|items| {
            items
                .iter()
                .any(|op| op.transition.kind == OpKind::PrefillUnd)
        })
    }

    fn any_denoise_inflight(&self) -> bool {
        self.inflight_ops.values().any(|items| {
            items
                .iter()
                .any(|op| op.transition.kind == OpKind::DenoiseGen)
        })
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
                    .filter(|op| op.transition.kind == OpKind::DecodeUnd)
                    .count()
            })
            .unwrap_or(0)
    }

    fn inflight_has_spec_tokens(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|items| items.iter().any(|op| !op.spec_tokens.is_empty()))
    }

    fn inflight_generated_token_count(&self, id: RequestId) -> usize {
        self.inflight_ops.get(&id).map_or(0, |items| {
            items
                .iter()
                .map(|op| match op.transition.kind {
                    OpKind::DecodeUnd => usize::from(op.decode_token_count.max(1)),
                    OpKind::PrefillUnd => 1,
                    _ => 0,
                })
                .sum()
        })
    }

    fn projected_cursor(&self, id: RequestId) -> Option<CursorProjection> {
        let st = self.running.get(&id)?;
        let transitions = self
            .inflight_ops
            .get(&id)
            .into_iter()
            .flat_map(|items| items.iter().map(|op| &op.transition));
        Some(st.cursor.project(transitions))
    }

    /// Whether any in-flight op for `id` is something other than a plain decode
    /// (a prefill chunk, an image op, …). Used to gate decode-lookahead.
    fn inflight_has_non_decode(&self, id: RequestId) -> bool {
        self.inflight_ops.get(&id).is_some_and(|items| {
            items
                .iter()
                .any(|op| op.transition.kind != OpKind::DecodeUnd)
        })
    }

    fn register_inflight(&mut self, transition: PlannedTransition, started: Instant) {
        let request_id = transition.op.req_id;
        let op_id = transition.op.op_id;
        let spec_tokens = transition.op.spec_token_ids.clone().unwrap_or_default();
        let decode_token_count = transition.op.decode_token_count.unwrap_or(1).max(1);
        self.inflight_ops
            .entry(request_id)
            .or_default()
            .push_back(InflightOp {
                transition,
                op_id,
                spec_tokens,
                decode_token_count,
                started,
            });
    }

    /// Resolve one in-flight op for `id` by the worker's echoed `op_id`.
    fn pop_inflight(
        &mut self,
        id: RequestId,
        op_id: Option<u64>,
    ) -> (Option<PlannedTransition>, Vec<u32>, Option<Instant>) {
        let Some(queue) = self.inflight_ops.get_mut(&id) else {
            return (None, Vec::new(), None);
        };
        let resolved = take_inflight_by_op_id(queue, op_id);
        if queue.is_empty() {
            self.inflight_ops.remove(&id);
        }
        match resolved {
            Some(op) => (Some(op.transition), op.spec_tokens, Some(op.started)),
            None => (None, Vec::new(), None),
        }
    }

    /// Failure policy after an executor/worker error.
    /// A typed non-fatal [`WorkerExecError`] fails the in-flight requests but
    /// keeps the engine alive to serve subsequent requests; anything else (a
    /// fatal worker error, ring/transport death) latches the engine fatal.
    ///
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
            let (transition, draft_token_ids, started) = self.pop_inflight(id, sr.op_id);
            if transition.is_none() {
                self.trace_record(json!({
                    "event": "unknown_result_op_id",
                    "at_s": now(),
                    "request_id": id.0,
                    "op_id": sr.op_id,
                }));
                if self.running.contains_key(&id) {
                    self.finish(id, FinishReason::Error);
                }
                continue;
            }
            let kind = transition.as_ref().map(|transition| transition.kind);
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
            let transition_delta = transition
                .as_ref()
                .map(|transition| transition.delta.as_str());
            let transition_new_blocks = transition
                .as_ref()
                .map(|transition| transition.resources.new_blocks);
            let transition_kv_target = transition
                .as_ref()
                .and_then(|transition| transition.resources.kv_target_tokens);
            let transition_scratch_units = transition
                .as_ref()
                .map(|transition| transition.resources.scratch_units);
            let transition_latent_units = transition
                .as_ref()
                .map(|transition| transition.resources.latent_units);
            let transition_encoder_pins = transition
                .as_ref()
                .map(|transition| transition.resources.encoder_pins.len());
            let transition_replayability = transition
                .as_ref()
                .map(|transition| transition.resources.replayability_after_apply.as_str());
            resolved_ops.push(json!({
                "request_id": id.0,
                "op_id": op_id,
                "op_kind": kind.map(opkind_str),
                "transition_delta": transition_delta,
                "transition_new_blocks": transition_new_blocks,
                "transition_kv_target": transition_kv_target,
                "transition_scratch_units": transition_scratch_units,
                "transition_latent_units": transition_latent_units,
                "transition_encoder_pins": transition_encoder_pins,
                "transition_replayability": transition_replayability,
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
            if let Some(transition) = transition {
                if let Err(error) = transition.validate_result(&sr) {
                    self.trace_record(json!({
                        "event": "transition_validation_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": transition.op_id,
                        "op_kind": opkind_str(transition.kind),
                        "error": transition_validation_error_str(&error),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                }
                let cursor_result = self
                    .running
                    .get_mut(&id)
                    .map(|state| state.cursor.apply_transition(&transition, &sr));
                if let Some(Err(error)) = cursor_result {
                    self.trace_record(json!({
                        "event": "cursor_transition_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": transition.op_id,
                        "op_kind": opkind_str(transition.kind),
                        "error": cursor_apply_error_str(&error),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                }
                self.release_transition_resources(id, &transition);
                let kind = transition.kind;
                let priority = match kind {
                    OpKind::DenoiseGen | OpKind::CommitGen | OpKind::CommitWriteback => 0,
                    _ => 1,
                };
                to_resolve.push((priority, seq_index, id, transition, sr, draft_token_ids));
            }
        }
        to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
        for (_priority, _seq_index, id, transition, sr, draft_token_ids) in to_resolve {
            if self.running.contains_key(&id) {
                self.resolve(id, transition, sr, draft_token_ids);
            }
            if let Some(st) = self.running.get(&id) {
                progress_ops.push(json!({
                    "request_id": id.0,
                    "phase": phase_str(st.lifecycle.phase),
                    "generated_tokens": st.und.tokens_emitted,
                    "images_done": st.image_gen.images_done,
                    "image_id": st.image_gen.image_id,
                    "steps_done": st.image_gen.steps_done,
                    "pos": st.und.logical_pos,
                    "kvlen": st.und.physical_kv_len,
                    "next_token": st.und.next_token,
                    "text_since_image": st.und.text_since_image,
                    "gen_branch_pending": st.image_gen.branch_pending,
                    "context_round_closing": st.ingest.round_closing,
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
            let Some(mut st) = self.skipped_waiting.remove(&id) else {
                continue;
            };
            match compiled.and_then(|grammar| self.grammar_compiler.create_matcher(grammar)) {
                Ok(matcher) => {
                    st.grammar = Some(matcher);
                    self.pending.add_request(st);
                }
                Err(message) => {
                    tracing::error!(request_id = id.0, %message, "grammar admission failed");
                    let _ = st.event_tx.send(GenEvent::Rejected { message });
                }
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

            if head.resources.reserve_worstcase {
                let need = head.resources.worstcase_blocks;
                let encoder_entries = head.req.resources.encoder_cache_keys.len();
                let encoder_ok = self
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.enc_cache.budget();
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
                        "generation": behavior_str(&st.req),
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.event_tx.send(GenEvent::Rejected {
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.bm.free_blocks() >= need && encoder_ok {
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
                let n = head.context.prompt_ids.len();
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
                        "generation": behavior_str(&st.req),
                        "prompt_tokens": st.context.prompt_ids.len(),
                    }));
                    let _ = st.event_tx.send(GenEvent::Rejected {
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
        let kv_capacity = st.resources.worstcase_blocks as u64;
        let generation = behavior_str(&st.req);
        let phase = phase_str(st.lifecycle.phase);
        let prompt_tokens = st.context.prompt_ids.len();
        let max_tokens = st.req.max_und_tokens;
        let priority = st.req.priority;
        let reserve_worstcase = st.resources.reserve_worstcase;
        let worstcase_blocks = st.resources.worstcase_blocks;
        let encoder_entries = st.req.resources.encoder_cache_keys.len();
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
        self.reserved_encoder_entries = self
            .reserved_encoder_entries
            .saturating_add(encoder_entries);
        self.trace_record(json!({
            "event": "request_admitted",
            "at_s": scheduled_at,
            "request_id": id.0,
            "queued_at_s": q,
            "queue_wait_s": scheduled_at - q,
            "generation": generation,
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
            "reserved_encoder_entries": self.reserved_encoder_entries,
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
    fn assemble(&mut self) -> (Vec<NewRequestData>, Vec<PlannedTransition>) {
        let ids = self.assembly_order();
        self.refresh_prompt_cohort(&ids);
        if let Some(cohort) = self.prompt_cohort.as_ref() {
            let cohort_ids = ids
                .iter()
                .copied()
                .filter(|id| cohort.contains(id))
                .collect::<Vec<_>>();
            return self.assemble_pass(&cohort_ids, Some(AssemblyLane::Prefill));
        }
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

    fn request_has_prompt_work(&self, id: RequestId) -> bool {
        if self.running.get(&id).is_none_or(|st| st.cancelled) {
            return false;
        }
        let prompt_inflight = self.inflight_ops.get(&id).is_some_and(|items| {
            items
                .iter()
                .any(|op| assembly_lane_for_kind(op.transition.kind) == AssemblyLane::Prefill)
        });
        prompt_inflight
            || self
                .peek_next_kind(id)
                .is_some_and(|kind| assembly_lane_for_kind(kind) == AssemblyLane::Prefill)
    }

    fn request_has_ready_decode(&self, id: RequestId) -> bool {
        if self.running.get(&id).is_none_or(|st| st.cancelled)
            || self
                .peek_next_kind(id)
                .is_none_or(|kind| assembly_lane_for_kind(kind) != AssemblyLane::Decode)
        {
            return false;
        }
        !self.has_inflight(id)
            || self.can_decode_lookahead(id)
            || self.can_prefill_decode_lookahead(id)
    }

    fn refresh_prompt_cohort(&mut self, ids: &[RequestId]) {
        let active = self.prompt_cohort.as_ref().is_some_and(|cohort| {
            cohort
                .iter()
                .copied()
                .any(|id| self.request_has_prompt_work(id))
        });
        if active {
            return;
        }
        if self.prompt_cohort.take().is_some() {
            // The first scheduling turn after a cohort drains remains available
            // to ready decodes before another finite cohort may open.
            return;
        }

        let prompt_ids = ids
            .iter()
            .copied()
            .filter(|id| self.request_has_prompt_work(*id))
            .collect::<HashSet<_>>();
        if prompt_ids.len() < 2 {
            return;
        }
        let has_cross_request_conflict = ids.iter().copied().any(|decode_id| {
            self.request_has_ready_decode(decode_id)
                && prompt_ids.iter().any(|prompt_id| *prompt_id != decode_id)
        });
        if has_cross_request_conflict {
            self.prompt_cohort = Some(prompt_ids);
        }
    }

    fn assemble_pass(
        &mut self,
        ids: &[RequestId],
        lane: Option<AssemblyLane>,
    ) -> (Vec<NewRequestData>, Vec<PlannedTransition>) {
        let mut new_reqs: Vec<NewRequestData> = Vec::new();
        let mut ops: Vec<PlannedTransition> = Vec::new();
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
        let mut mixed_ops: Vec<PlannedTransition> = Vec::new();
        let denoise_occupies_decode_pipeline =
            lane == Some(AssemblyLane::Decode) && self.any_denoise_inflight();
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
            // When a decode pass admits text prefill rows, the worker receives a
            // single mixed forward. There is no `supports_mixed_op_kinds` gate;
            // co-batched rows shift each other's numerics only through inherent
            // batched-kernel FP non-invariance, not structural corruption.
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
                    && self.running.get(&id).is_some_and(|st| {
                        st.is_replayable_text() && !st.req.sampling.prompt_logprobs_requested()
                    });
                if !mixed_prefill {
                    continue;
                }
            }
            if next_kind == Some(OpKind::DenoiseGen) {
                if denoise_occupies_decode_pipeline || !self.can_schedule_denoise(id) {
                    continue;
                }
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
                match self.next_transition(id, op_budget) {
                    Some(mut op) => {
                        // bound encode work per step; defer if over budget.
                        if op.kind == OpKind::VitEncode || op.kind == OpKind::VaeEncode {
                            if encodes_left == 0 {
                                break;
                            }
                            encodes_left -= 1;
                        }
                        if mixed_prefill {
                            mixed_left = mixed_left.saturating_sub(planned_op_token_cost(&op));
                        }
                        budget = budget.saturating_sub(planned_op_token_cost(&op));
                        // Stateful-diff contract: a request's static state and
                        // initial block allocation cross once, ahead of its
                        // first op; the op then carries only deltas.
                        if let Some(st) = self.running.get_mut(&id)
                            && !st.resources.worker_registered
                        {
                            st.resources.worker_registered = true;
                            let neg = (!st.context.negative_prompt_ids.is_empty())
                                .then(|| st.context.negative_prompt_ids.clone());
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
                        if !self.reserve_transition_resources(&op) {
                            tracing::error!(
                                request_id = id.0,
                                "planned resource reservation failed after readiness check"
                            );
                            self.finish(id, FinishReason::Error);
                            break;
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
                            .map(|st| st.image_gen.branch_pending)
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
            // by this scheduler's per-request planner; group them with "no op".
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
            || !st.is_replayable_text()
            || st.req.behavior.gen_output
            || st.lifecycle.phase != Phase::DecodeUnd
            || st.grammar.is_some()
        {
            return false;
        }
        let depth = self.inflight_decode_count(id);
        if depth == 0 || self.inflight_has_non_decode(id) || self.inflight_has_spec_tokens(id) {
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
            && st.und.tokens_emitted >= sp.min_tokens
            && st
                .und
                .tokens_emitted
                .saturating_add(self.inflight_generated_token_count(id))
                < st.req.max_und_tokens
            && !sp.generated_logprobs_requested()
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
            || !st.is_replayable_text()
            || st.req.behavior.gen_output
            || st.lifecycle.phase != Phase::Prefill
            || st.grammar.is_some()
        {
            return false;
        }
        // The whole prompt must already be covered by the in-flight prefill
        // transition, i.e. the final prefill is running.
        if self
            .projected_cursor(id)
            .is_none_or(|cursor| (cursor.prompt_cursor as usize) < st.effective_prompt().len())
        {
            return false;
        }
        let Some(items) = self.inflight_ops.get(&id) else {
            return false;
        };
        if items.len() != 1
            || items
                .iter()
                .any(|op| op.transition.kind != OpKind::PrefillUnd)
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
            && st.und.tokens_emitted >= sp.min_tokens
            && st
                .und
                .tokens_emitted
                .saturating_add(self.inflight_generated_token_count(id))
                < st.req.max_und_tokens
            && !sp.generated_logprobs_requested()
            && sp.bad_words_ids.is_empty()
            && !penalties
    }

    fn supports_spec_decode(&self) -> bool {
        self.caps.supported_ops.contains(&OpKind::TargetVerifyUnd)
    }

    fn decode_burst_plan(
        &self,
        id: RequestId,
        pos: usize,
        budget: usize,
        allowed: Option<&[u32]>,
    ) -> (u16, Option<Vec<u32>>, bool) {
        let Some(st) = self.running.get(&id) else {
            return (1, None, false);
        };
        if self.decode_token_burst <= 1 || budget <= 1 || st.grammar.is_some() || allowed.is_some()
        {
            return (1, None, false);
        }
        let sp = &st.req.sampling;
        let penalties = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        if sp.temperature > 0.0
            || sp.generated_logprobs_requested()
            || !sp.bad_words_ids.is_empty()
            || penalties
            || st.und.tokens_emitted < sp.min_tokens
        {
            return (1, None, false);
        }

        let remaining = st
            .req
            .max_und_tokens
            .saturating_sub(st.und.tokens_emitted)
            .saturating_sub(self.inflight_generated_token_count(id));
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
            return (1, None, false);
        }

        let mut stop_ids = Vec::new();
        if !sp.ignore_eos && st.req.policy.termination.eos_finishes {
            stop_ids.extend(self.ctrl.eos.iter().copied());
        }
        if st.req.policy.termination.stop_finishes {
            stop_ids.extend(st.req.stop_token_ids.iter().copied());
        }
        stop_ids.extend(st.req.policy.trigger.round_close_token_ids());
        if st.can_open_gen_branch() {
            // Multi-token triggers are host-detected by suffix.
            // Stopping on any member is conservative: it may shorten a burst,
            // but it cannot decode past a trigger that the host would need to
            // observe before scheduling the image phase.
            if let Some(trigger) = st.req.policy.trigger.generated_suffix() {
                stop_ids.extend(trigger.iter().copied());
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        let terminal = !stop_ids.is_empty()
            && !st.can_open_gen_branch()
            && st.req.policy.termination.eos_finishes
            && st.req.policy.termination.stop_finishes;
        (count, (!stop_ids.is_empty()).then_some(stop_ids), terminal)
    }

    fn decode_burst_kv_cap(&self, id: RequestId, pos: usize) -> usize {
        let Some(st) = self.running.get(&id) else {
            return 1;
        };
        if !st.continues_after_gen_commit() {
            return usize::MAX;
        }
        let bs = self.caps.block_size as usize;
        let allocated_token_cap = self.bm.blocks_for(id).len().saturating_mul(bs);
        let in_allocated = allocated_token_cap.saturating_sub(pos);
        let decode_waiters = self
            .running
            .values()
            .filter(|candidate| {
                candidate.continues_after_gen_commit()
                    && candidate.lifecycle.phase == Phase::DecodeUnd
                    && !candidate.image_gen.branch_pending
                    && !candidate.cancelled
            })
            .count()
            .max(1);
        let shared_free_blocks = self.bm.free_blocks() / decode_waiters;
        in_allocated.saturating_add(shared_free_blocks.saturating_mul(bs))
    }

    fn peek_next_kind(&self, id: RequestId) -> Option<OpKind> {
        let st = self.running.get(&id)?;
        if st.image_gen.branch_pending {
            return None;
        }
        if st.has_context_images()
            && st.lifecycle.phase == Phase::Prefill
            && self.projected_cursor(id).is_some_and(|cursor| {
                st.context
                    .images
                    .get(st.ingest.mm_cursor)
                    .is_some_and(|item| item.position as usize == cursor.prompt_cursor as usize)
            })
        {
            return st.pending_image_step().map(|step| match step {
                ImageIngestStep::VaeEncode => OpKind::VaeEncode,
                ImageIngestStep::VitEncode => OpKind::VitEncode,
            });
        }
        Some(match st.lifecycle.phase {
            Phase::Encode => match st.pending_image_step()? {
                ImageIngestStep::VaeEncode => OpKind::VaeEncode,
                ImageIngestStep::VitEncode => OpKind::VitEncode,
            },
            Phase::Prefill if self.can_prefill_decode_lookahead(id) => OpKind::DecodeUnd,
            Phase::Prefill => OpKind::PrefillUnd,
            Phase::DecodeUnd => OpKind::DecodeUnd,
            Phase::DenoiseGen => OpKind::DenoiseGen,
            Phase::CommitGen => OpKind::CommitGen,
            Phase::CommitWriteback => OpKind::CommitWriteback,
            Phase::FeedbackIngest => {
                let feedback = st.req.policy.feedback.as_ref()?;
                let uniserve_core::FeedbackWriteback::Reingest { ingest } = &feedback.writeback
                else {
                    return None;
                };
                match ingest.steps.get(st.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => OpKind::VaeEncode,
                    ImageIngestStep::VitEncode => OpKind::VitEncode,
                }
            }
        })
    }

    fn preemptible(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|st| {
            if st.resources.reserve_worstcase || !st.is_replayable_text() {
                return false;
            }
            // generated branch replay is too expensive under high concurrency: even
            // prompt-only victims can quickly become prompt++generated victims and
            // churn the same long prefix forever. Keep these requests resident.
            if st.continues_after_gen_commit() {
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
        self.requeue_for_recompute(victim);
    }

    fn requeue_for_recompute(&mut self, victim: RequestId) {
        let mut st = match self.running.remove(&victim) {
            Some(s) => s,
            None => return,
        };
        self.order.retain(|x| *x != victim);
        self.reserved_encoder_entries = self
            .reserved_encoder_entries
            .saturating_sub(st.req.resources.encoder_cache_keys.len());
        if st.resources.reserve_worstcase {
            self.reserved_blocks = self
                .reserved_blocks
                .saturating_sub(st.resources.worstcase_blocks);
        }
        let mut free_encoder_handles = std::mem::take(&mut st.ingest.transient_encoder_handles);
        for pin in &st.ingest.acquired_encoder_pins {
            if let Some(handle) = self.enc_cache.release(pin.key, pin.handle) {
                free_encoder_handles.push(handle);
            }
        }
        if !free_encoder_handles.is_empty() {
            self.gated_control(ControlOp::FreeEncoder(free_encoder_handles));
        }
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
        let mut recompute = st.context.prompt_ids.clone();
        recompute.extend_from_slice(&st.replay.generated_ids);
        let generated_ids = std::mem::take(&mut st.replay.generated_ids);
        let generated_count = st.und.tokens_emitted;
        let text_since_image = st.und.text_since_image;
        let prompt_logprobs_emitted = st.ingest.prompt_logprobs_emitted;
        let worstcase_blocks = st.resources.worstcase_blocks;
        let reserve_worstcase = st.resources.reserve_worstcase;
        st.cursor = GenerationCursor::new(Phase::Prefill, worstcase_blocks, reserve_worstcase);
        st.replay.generated_ids = generated_ids;
        st.replay.recompute_ids = Some(recompute);
        st.replay.preempted = true;
        st.und.tokens_emitted = generated_count;
        st.und.text_since_image = text_since_image;
        st.ingest.prompt_logprobs_emitted = prompt_logprobs_emitted;
        st.trace.push(crate::trace::TraceEvent::at(
            crate::trace::TraceEventKind::Preempted,
        ));
        self.trace_record(json!({
            "event": "request_preempted",
            "at_s": now(),
            "request_id": victim.0,
            "generated_tokens": st.replay.generated_ids.len(),
            "prompt_tokens": st.context.prompt_ids.len(),
            "reset_phase": phase_str(st.lifecycle.phase),
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

    fn submit_batch(
        &mut self,
        new_reqs: Vec<NewRequestData>,
        mut transitions: Vec<PlannedTransition>,
    ) {
        let _span =
            tracing::trace_span!("scheduler.submit_batch", ops = transitions.len()).entered();
        self.step_id += 1;
        let step = self.step_id;
        // /10: stamp each op's submit time (round-trip latency) + assign its
        // op_id (lifecycle correlation), and record an OpSubmitted trace event.
        let submit_at = Instant::now();
        for transition in &mut transitions {
            let oid = self.next_op_id;
            self.next_op_id += 1;
            transition.assign_op_id(oid);
            let request_id = transition.op.req_id;
            let opk = opkind_str(transition.op.kind);
            self.register_inflight(transition.clone(), submit_at);
            if let Some(st) = self.running.get_mut(&request_id) {
                let mut ev =
                    crate::trace::TraceEvent::at(crate::trace::TraceEventKind::OpSubmitted);
                ev.op_id = Some(oid);
                ev.op_kind = Some(opk);
                ev.step_id = step;
                st.trace.push(ev);
            }
        }
        self.peak_ops_in_batch = self.peak_ops_in_batch.max(transitions.len());
        self.stats
            .general
            .peak_ops
            .fetch_max(transitions.len(), Ordering::Relaxed);
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
        let mixed = transitions
            .first()
            .is_some_and(|first| transitions.iter().any(|op| op.kind != first.kind));
        let op_kinds: Vec<&'static str> =
            transitions.iter().map(|op| opkind_str(op.kind)).collect();
        let req_ids: Vec<u64> = transitions.iter().map(|op| op.req_id.0).collect();
        let trace_ops: Vec<_> = transitions
            .iter()
            .map(|op| {
                let phase = self.running.get(&op.req_id).map(|st| phase_str(st.lifecycle.phase));
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
                    "token_cost": planned_op_token_cost(op),
                    "timestep_idx": op.timestep_idx,
                    "denoise_step_count": op.denoise_step_count,
                    "decode_token_count": op.decode_token_count,
                    "decode_stop_token_ids_len": op.decode_stop_token_ids.as_ref().map(|ids| ids.len()).unwrap_or(0),
                    "decode_stop_terminal": op.decode_stop_terminal,
                    "cond_pos": op.cond_pos,
                    "image_in": op.image_in,
                    "mm_hash": op.mm_hash,
                    "transition": op.delta.as_str(),
                    "resources": {
                        "new_blocks": op.resources.new_blocks,
                        "kv_target_tokens": op.resources.kv_target_tokens,
                        "scratch_units": op.resources.scratch_units,
                        "latent_units": op.resources.latent_units,
                        "encoder_pins": op.resources.encoder_pins,
                        "replayability_after_apply": op.resources.replayability_after_apply.as_str(),
                    },
                    "visibility": {
                        "und_tokens": format!("{:?}", op.visibility.und_tokens),
                        "generated_image": op.visibility.generated_image,
                    },
                })
            })
            .collect();
        let new_req_ids: Vec<u64> = new_reqs.iter().map(|req| req.req_id.0).collect();
        self.batch_started.insert(step, submit_at);
        self.trace_record(json!({
            "event": "batch_submitted",
            "at_s": now(),
            "step_id": step,
            "batch_size": transitions.len(),
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
        let wire_ops = transitions
            .iter()
            .map(|transition| transition.op.clone())
            .collect();
        let batch = ForwardBatch {
            step_id: self.step_id,
            new_reqs,
            ops: wire_ops,
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

    fn generated_trigger_matches(st: &ReqState) -> bool {
        st.req
            .policy
            .trigger
            .matches_generated(&st.replay.generated_ids)
    }

    fn direct_trigger_matches(st: &ReqState, token_id: u32) -> bool {
        st.req.policy.trigger.direct_token() == Some(token_id)
    }

    fn feedback_next_token(&self, id: RequestId) -> Option<u32> {
        let next = self
            .running
            .get(&id)?
            .req
            .policy
            .feedback
            .as_ref()?
            .next_und_token;
        match next {
            uniserve_core::FeedbackNextToken::None => None,
            uniserve_core::FeedbackNextToken::Bos => Some(self.ctrl.bos),
            uniserve_core::FeedbackNextToken::EndOfImage => Some(self.ctrl.end_of_image),
            uniserve_core::FeedbackNextToken::Token { token_id } => Some(token_id),
        }
    }

    fn commit_requires_writeback(st: &ReqState) -> bool {
        st.req.behavior.generated_image_feedback
            && st.req.policy.feedback.as_ref().is_some_and(|feedback| {
                feedback.commit == uniserve_core::CommitRecipe::CommitGenThenWriteback
                    && matches!(
                        feedback.writeback,
                        uniserve_core::FeedbackWriteback::DirectKv
                    )
            })
    }

    fn reingests_generated_image(st: &ReqState) -> bool {
        st.req.behavior.generated_image_feedback
            && st.req.policy.feedback.as_ref().is_some_and(|feedback| {
                matches!(
                    feedback.writeback,
                    uniserve_core::FeedbackWriteback::Reingest { .. }
                )
            })
    }

    fn prefilled_gen_trigger(&self, id: RequestId) -> bool {
        let Some(st) = self.running.get(&id) else {
            return false;
        };
        st.can_open_gen_branch()
            && st
                .req
                .policy
                .trigger
                .matches_generated(st.effective_prompt())
    }

    /// The block-id delta since the last op for this request (the stateful-diff
    /// contract): everything `blocks_for` holds beyond what already crossed.
    fn take_new_blocks(&mut self, id: RequestId) -> Vec<BlockId> {
        let all = self.bm.blocks_for(id);
        let Some(st) = self.running.get_mut(&id) else {
            return Vec::new();
        };
        let sent = st.resources.blocks_sent.min(all.len());
        let new = all[sent..].to_vec();
        st.resources.blocks_sent = all.len();
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

    fn plan_intent(
        &mut self,
        id: RequestId,
        cursor: CursorProjection,
        intent: TransitionIntent,
    ) -> Option<PlannedTransition> {
        let planned = {
            let request = &self.running.get(&id)?.req;
            self.planner.plan(request, cursor, intent)
        };
        match planned {
            Ok(transition) => Some(transition),
            Err(error) => {
                tracing::error!(request_id = id.0, ?error, "generation planning failed");
                self.finish(id, FinishReason::Error);
                None
            }
        }
    }

    fn next_transition(&mut self, id: RequestId, budget: usize) -> Option<PlannedTransition> {
        if !self.stage_ready(id) {
            return None;
        }
        let projection = self.projected_cursor(id)?;
        let context_pending = self.running.get(&id).is_some_and(|st| {
            projection.prompt_cursor < st.context.prompt_ids.len() as u32
                || st.ingest.mm_cursor < st.context.images.len()
        });
        if context_pending {
            return self.next_context_ingest_transition(id, budget, projection);
        }
        if self.running.get(&id)?.image_gen.branch_pending
            && !self.promote_gen_branch_reservation(id)
        {
            return None;
        }
        let phase = self.running.get(&id)?.lifecycle.phase;
        // Final-prefill-in-flight requests build their first decode op early
        // (cross-boundary lookahead); the committed phase advances at resolve.
        let phase = if phase == Phase::Prefill && self.can_prefill_decode_lookahead(id) {
            Phase::DecodeUnd
        } else {
            phase
        };
        match phase {
            Phase::Encode => None,
            Phase::Prefill => {
                let st = self.running.get(&id)?;
                let prompt = st.effective_prompt().to_vec();
                let n = prompt.len();
                let projection = self.projected_cursor(id)?;
                let cursor = projection.prompt_cursor as usize;
                let (segment_index, segment_end) =
                    st.context.token_segment_at(cursor).unwrap_or((0, n));
                // Chunked prefill with the clip rule: the chunk is bounded by
                // the remaining step budget and the long-prefill threshold.
                let chunk_cap = (n - cursor)
                    .min(self.config.long_prefill_threshold)
                    .min(budget.max(1));
                let end = (cursor + chunk_cap.max(1))
                    .min(n)
                    .min(segment_end.max(cursor + 1));
                if !self.bm.ensure_capacity(id, end) {
                    return None;
                }
                let chunk: Vec<u32> = prompt[cursor..end].to_vec();
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::IngestText {
                        segment_index,
                        prompt_start: cursor as u32,
                        token_ids: chunk,
                        new_blocks,
                        recent_tokens: recent,
                        allowed_tokens: allowed,
                        suppress_tokens: suppress,
                    },
                )
            }
            Phase::DecodeUnd => {
                let st = self.running.get(&id)?;
                let prefill_lookahead = st.lifecycle.phase == Phase::Prefill;
                let lookahead_depth = self.inflight_decode_count(id);
                let use_last_sampled = lookahead_depth > 0 || prefill_lookahead;
                if use_last_sampled
                    && !self.can_decode_lookahead(id)
                    && !self.can_prefill_decode_lookahead(id)
                {
                    return None;
                }
                let projection = self.projected_cursor(id)?;
                let pos = projection.logical_pos;
                let tok = if use_last_sampled {
                    0
                } else {
                    st.und.next_token
                };
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                let (decode_token_count, decode_stop_token_ids, decode_stop_terminal) =
                    self.decode_burst_plan(id, pos as usize, budget, allowed.as_deref());
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
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DecodeUnd {
                        position: pos,
                        token_id: tok,
                        token_source: if use_last_sampled {
                            TokenSource::LastSampled
                        } else {
                            TokenSource::Wire
                        },
                        new_blocks,
                        spec_token_ids,
                        token_count: decode_len as u16,
                        stop_token_ids: decode_stop_token_ids,
                        stop_terminal: decode_stop_terminal,
                        recent_tokens: recent,
                        allowed_tokens: allowed,
                        suppress_tokens: suppress,
                    },
                )
            }
            Phase::DenoiseGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.image_gen.cond_pos;
                let nvae = self.num_vae(&st.req.image) as usize;
                if st.req.image.retain_images
                    && !self.bm.ensure_capacity(
                        id,
                        cond_pos as usize + nvae + self.cap_commit_marker_tokens(),
                    )
                {
                    return None;
                }
                let timestep = st.image_gen.steps_done;
                let remaining = st.req.image.steps.saturating_sub(timestep).max(1);
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let cfg = cfg_params(&st.req.image, cfg_branch_count(&st.req.image));
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                let scratch_units = u64::from(cfg.branch_count);
                let host_scratch_tokens = self.denoise_host_scratch_tokens(st);
                let image_prompt = Self::image_prompt_for(st);
                let image_id = st.image_gen.image_id;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DenoiseGen {
                        image_id,
                        position: cond_pos,
                        start_step: timestep,
                        step_count: denoise_step_count,
                        cfg,
                        image_prompt,
                        latent_units,
                        scratch_units,
                        host_scratch_tokens,
                    },
                )
            }
            Phase::CommitGen => {
                let st = self.running.get(&id)?;
                let cond_pos = st.image_gen.cond_pos;
                let image_id = st.image_gen.image_id;
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                let projection = self.projected_cursor(id)?;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::CommitGen {
                        image_id,
                        position: cond_pos,
                        new_blocks,
                        recent_tokens: recent,
                        allowed_tokens: allowed,
                        suppress_tokens: suppress,
                    },
                )
            }
            Phase::CommitWriteback => {
                let st = self.running.get(&id)?;
                let cond_pos = st.image_gen.cond_pos;
                let image_id = st.image_gen.image_id;
                let locator = st.feedback.locator.clone();
                let recent = self.recent_tokens(id);
                let (allowed, suppress) = self.token_masks(id);
                let Some(locator) = locator else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };
                let projection = self.projected_cursor(id)?;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::Feedback {
                        image_id,
                        position: cond_pos,
                        locator,
                        new_blocks,
                        recent_tokens: recent,
                        allowed_tokens: allowed,
                        suppress_tokens: suppress,
                    },
                )
            }
            Phase::FeedbackIngest => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let uniserve_core::FeedbackWriteback::Reingest { ingest } = &feedback.writeback
                else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };
                let step_index = st.feedback.ingest_step;
                let step = ingest.steps.get(step_index).copied()?;
                let is_final_step = step_index + 1 == ingest.steps.len();
                let image_id = st.image_gen.image_id;
                let feedback_key =
                    encoder_cache_key(id.0 ^ u64::from(image_id).rotate_left(17), step_index, step);
                let worker_hash = transient_encoder_worker_key(feedback_key, id);
                let image_b64 = st.feedback.image_b64.clone();
                let staged_image = st.feedback.staged_image;
                let logical_positions = ingest.logical_positions;
                let physical_kv_tokens = ingest.kv_effect(step_index)?;
                let Some(image_b64) = image_b64 else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };
                let projection = self.projected_cursor(id)?;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::FeedbackIngest {
                        image_id,
                        step_index,
                        step,
                        worker_hash,
                        is_final_step,
                        position: projection.logical_pos,
                        logical_positions,
                        physical_kv_tokens,
                        image_b64,
                        staged_image,
                        new_blocks,
                    },
                )
            }
        }
    }

    fn next_context_ingest_transition(
        &mut self,
        id: RequestId,
        budget: usize,
        projection: CursorProjection,
    ) -> Option<PlannedTransition> {
        let cursor = projection.prompt_cursor as usize;
        let image = self
            .running
            .get(&id)?
            .context
            .images
            .get(self.running.get(&id)?.ingest.mm_cursor)
            .cloned();
        if let Some(image) = image.as_ref()
            && image.position as usize == cursor
        {
            let state = self.running.get(&id)?;
            let step_index = state.ingest.pending_image_step;
            let step = image.ingest.steps.get(step_index).copied()?;
            let is_final_step = step_index + 1 == image.ingest.steps.len();
            let staged_image = state.ingest.staged_image;
            let cache_read = state.req.cache.read;
            let cache_write = state.req.cache.write;
            let cache_key = encoder_cache_key(image.hash, step_index, step);
            let cached = if cache_read {
                self.enc_cache.lookup_output(cache_key)
            } else {
                None
            };
            let cache_hit = cached.is_some();
            let persistent_cache_key = (cache_hit || cache_write).then_some(cache_key);
            let worker_hash = if cache_hit || cache_write {
                cache_key
            } else {
                transient_encoder_worker_key(cache_key, id)
            };
            let encoder_input = cached.map(|output| output.handle).or(staged_image);
            let new_blocks = self.take_new_blocks(id);
            return self.plan_intent(
                id,
                projection,
                TransitionIntent::IngestImage {
                    segment_index: image.segment_index,
                    step_index,
                    step,
                    is_final_step,
                    position: projection.logical_pos,
                    logical_positions: image.ingest.logical_positions,
                    physical_kv_tokens: image.ingest.kv_effect(step_index)?,
                    worker_hash,
                    encoder_cache_key: persistent_cache_key,
                    cache_hit,
                    image_b64: image.b64.clone(),
                    staged_image: encoder_input,
                    new_blocks,
                },
            );
        }

        let (prompt, segment_index, segment_end, next_image) = {
            let st = self.running.get(&id)?;
            let prompt = st.context.prompt_ids.clone();
            let (segment_index, segment_end) = st
                .context
                .token_segment_at(cursor)
                .unwrap_or((0, prompt.len()));
            let next_image = image
                .as_ref()
                .map_or(prompt.len(), |image| image.position as usize);
            (prompt, segment_index, segment_end, next_image)
        };
        if cursor >= prompt.len() {
            return None;
        }
        let end = cursor
            .saturating_add(budget.max(1).min(self.config.long_prefill_threshold))
            .min(prompt.len())
            .min(segment_end)
            .min(next_image.max(cursor + 1));
        if !self
            .bm
            .ensure_capacity(id, projection.physical_kv_len as usize + end - cursor)
        {
            return None;
        }
        let recent = self.recent_tokens(id);
        let (allowed, suppress) = self.token_masks(id);
        let new_blocks = self.take_new_blocks(id);
        self.plan_intent(
            id,
            projection,
            TransitionIntent::IngestText {
                segment_index,
                prompt_start: cursor as u32,
                token_ids: prompt[cursor..end].to_vec(),
                new_blocks,
                recent_tokens: recent,
                allowed_tokens: allowed,
                suppress_tokens: suppress,
            },
        )
    }

    /// Decide branch-vs-finish once a declared round-close token is committed.
    fn close_context_round(&mut self, id: RequestId, close_token: u32) {
        let (triggered, can_open_gen_branch, n_gen, max_tokens, eos_finishes) = {
            let Some(st) = self.running.get(&id) else {
                return;
            };
            (
                st.req
                    .policy
                    .trigger
                    .matches_round_close(&st.und.round_tokens, close_token),
                st.can_open_gen_branch(),
                st.und.tokens_emitted,
                st.req.max_und_tokens,
                st.req.policy.termination.eos_finishes,
            )
        };
        if triggered && can_open_gen_branch && n_gen < max_tokens {
            return self.begin_image(id);
        }
        if n_gen < max_tokens && !eos_finishes {
            if let Some(st) = self.running.get_mut(&id) {
                st.und.round_tokens.clear();
                st.und.next_token = close_token;
                st.lifecycle.phase = Phase::DecodeUnd;
            }
            return;
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

    /// bounded recent-output window for penalties (only carried when a
    /// penalty is active, to keep the op small — risk 4).
    fn recent_tokens(&self, id: RequestId) -> Option<Vec<u32>> {
        let st = self.running.get(&id)?;
        let sp = &st.req.sampling;
        let active = sp.repetition_penalty != 1.0
            || sp.frequency_penalty != 0.0
            || sp.presence_penalty != 0.0;
        if !active || st.replay.generated_ids.is_empty() {
            return None;
        }
        let n = st.replay.generated_ids.len();
        let start = n.saturating_sub(self.penalty_window);
        Some(st.replay.generated_ids[start..].to_vec())
    }

    /// run the host-side logits-processor pipeline to compute the op's
    /// allowed/suppress masks (min-tokens, bad-words, allowed-tokens).
    fn token_masks(&mut self, id: RequestId) -> (Option<Vec<u32>>, Option<Vec<u32>>) {
        let st = match self.running.get(&id) {
            Some(s) => s,
            None => return (None, None),
        };
        let ctx = crate::logits::ProcCtx {
            n_generated: st.und.tokens_emitted,
            eos: &self.ctrl.eos,
            generated: &st.replay.generated_ids,
            sampling: &st.req.sampling,
        };
        let (mut allowed, mut suppress) = crate::logits::run_pipeline(&self.logits_pipeline, &ctx);
        // Image-budget enforcement: once a request has produced max_images, a
        // direct branch-opening token is
        // suppressed, so an eager or positively-biased model must return to
        // text/EOS instead of emitting an un-actionable trigger into its
        // own context forever. Suppression beats logit bias (bias skips
        // -inf'd logits on the worker).
        if st.req.behavior.gen_output
            && st.image_gen.images_done >= st.req.image.max_images as usize
            && let Some(trigger) = st.req.policy.trigger.direct_token()
        {
            suppress.get_or_insert_with(Vec::new).push(trigger);
        }
        // If the model's text turn is naturally complete while image budget
        // remains, resolve treats EOS as a clean image boundary instead of
        // making the model invent filler text.
        // Structured outputs: the grammar's per-step mask intersects whatever
        // the pipeline allows. At a terminal trie node both termination and
        // continuations into longer alternatives remain valid.
        let grammar_stops = {
            let mut stops = self.ctrl.eos.clone();
            stops.extend(st.req.stop_token_ids.iter().copied());
            stops.sort_unstable();
            stops.dedup();
            stops
        };
        let grammar_mask = self
            .running
            .get_mut(&id)
            .and_then(|state| state.grammar.as_mut())
            .map(|matcher| matcher.next_mask());
        if let Some(mask) = grammar_mask {
            let (complete, g_allowed) = match mask {
                Ok(mask) => mask,
                Err(error) => {
                    tracing::error!(request_id = id.0, %error, "grammar mask computation failed");
                    return (Some(Vec::new()), suppress);
                }
            };
            let g_allowed = grammar_allowed_tokens(complete, g_allowed, &grammar_stops);
            allowed = Some(match allowed {
                Some(a) => a.into_iter().filter(|t| g_allowed.contains(t)).collect(),
                None => g_allowed,
            });
        }
        (allowed, suppress)
    }

    fn advance_grammar(&mut self, id: RequestId, token_id: u32) -> bool {
        let error = self
            .running
            .get_mut(&id)
            .and_then(|state| state.grammar.as_mut())
            .and_then(|matcher| matcher.advance(token_id).err());
        if let Some(error) = error {
            tracing::error!(request_id = id.0, token_id, %error, "grammar state rejected a committed token");
            self.finish(id, FinishReason::Error);
            return false;
        }
        true
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
            st.lifecycle.phase = Phase::DecodeUnd;
        }
        let mut outputs: Vec<(u32, Option<f32>, bool)> = draft_token_ids
            .iter()
            .take(accepted)
            .map(|token| (*token, None, false))
            .collect();
        outputs.push((sampled, sr.sampled_logprob, true));

        for (tok, logprob, is_sampled) in outputs {
            let Some(st) = self.running.get_mut(&id) else {
                return;
            };
            st.und.tokens_emitted += 1;
            let top_logprobs = is_sampled.then(|| sr.top_logprobs.clone()).flatten();
            if self.emit_or_finish_und_token(id, tok, logprob, top_logprobs, is_sampled) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.und.next_token = tok;
            }
            if !self.advance_grammar(id, tok) {
                return;
            }
        }
    }

    fn resolve_decode_text(&mut self, id: RequestId, sr: uniserve_worker_wire::SeqResult) {
        self.bm.activate(id);
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.ingest.round_closing)
        {
            let close_token = self
                .running
                .get(&id)
                .map(|state| state.und.next_token)
                .unwrap_or_default();
            if let Some(state) = self.running.get_mut(&id) {
                state.ingest.round_closing = false;
            }
            return self.close_context_round(id, close_token);
        }
        let burst_result = sr
            .sampled_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty());
        let tokens = sr
            .sampled_token_ids
            .clone()
            .filter(|ids| !ids.is_empty())
            .unwrap_or_else(|| vec![sr.sampled_token_id.unwrap_or(self.ctrl.eos[0])]);
        for (idx, tok) in tokens.iter().copied().enumerate() {
            let is_last = idx + 1 == tokens.len();
            let logprob = is_last.then_some(sr.sampled_logprob).flatten();
            let is_round_close = self
                .running
                .get(&id)
                .is_some_and(|st| st.req.policy.trigger.round_close_token_ids().contains(&tok));
            if is_round_close {
                if let Some(st) = self.running.get_mut(&id) {
                    st.und.tokens_emitted += 1;
                }
                if burst_result {
                    return self.close_context_round(id, tok);
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.und.next_token = tok;
                    st.ingest.round_closing = true;
                }
                return;
            }
            let (can_open_gen_branch, images_done, max_images) = {
                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.und.tokens_emitted += 1;
                (
                    st.can_open_gen_branch(),
                    st.image_gen.images_done,
                    st.req.image.max_images as usize,
                )
            };
            let direct_trigger = self
                .running
                .get(&id)
                .is_some_and(|st| Self::direct_trigger_matches(st, tok));
            if direct_trigger && can_open_gen_branch && images_done < max_images {
                self.begin_image(id);
                return;
            }
            let top_logprobs = is_last.then(|| sr.top_logprobs.clone()).flatten();
            if self.emit_or_finish_und_token(id, tok, logprob, top_logprobs, is_last) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.und.next_token = tok;
                st.lifecycle.phase = Phase::DecodeUnd;
                st.und.round_tokens.push(tok);
            }
            if !self.advance_grammar(id, tok) {
                return;
            }
            if can_open_gen_branch
                && images_done < max_images
                && self
                    .running
                    .get(&id)
                    .is_some_and(Self::generated_trigger_matches)
            {
                self.begin_image(id);
                return;
            }
        }
    }

    fn resolve(
        &mut self,
        id: RequestId,
        transition: PlannedTransition,
        mut sr: uniserve_worker_wire::SeqResult,
        draft_token_ids: Vec<u32>,
    ) {
        let kind = transition.kind;
        if let Some(positions) = sr.prompt_logprobs.take() {
            self.resolve_prompt_logprobs(id, positions);
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
                    let (cursor, prompt_len) = {
                        let st = self.running.get(&id).unwrap();
                        (
                            st.ingest.prompt_cursor as usize,
                            st.effective_prompt().len(),
                        )
                    };
                    if cursor < prompt_len {
                        return; // still prefilling; ignore the (partial) sampled token
                    }
                    if self.running.get(&id).is_some_and(|st| {
                        st.ingest.mm_cursor < st.context.images.len()
                            || st.ingest.prompt_cursor < st.context.prompt_ids.len() as u32
                    }) {
                        return;
                    }
                }
                let (starts_gen_after_context, can_open_gen_branch) = {
                    let st = self.running.get_mut(&id).unwrap();
                    if kind == OpKind::PrefillUnd {
                        // prompt (possibly prompt++generated on recompute) is done.
                        // recompute finished: future growth is plain decode again.
                        st.replay.recompute_ids = None;
                    }
                    st.und.tokens_emitted += 1;
                    (st.starts_gen_after_context(), st.can_open_gen_branch())
                };
                if kind == OpKind::PrefillUnd
                    && self.running.get(&id).is_some_and(|state| {
                        state.req.sampling.prompt_logprobs_requested()
                            && state.ingest.prompt_logprobs_emitted
                                != state.context.prompt_ids.len().saturating_sub(1)
                    })
                {
                    tracing::error!(
                        request_id = id.0,
                        "prompt logprob scoring ended before every prompt position was resolved"
                    );
                    return self.finish(id, FinishReason::Error);
                }
                // the prompt is fully prefilled now — publish its full
                // blocks to the prefix cache for later requests to reuse.
                if kind == OpKind::PrefillUnd {
                    let bs = self.caps.block_size as usize;
                    if let Some(st) = self.running.get_mut(&id) {
                        self.prefix_cache.cache_blocks(st, &mut self.bm, bs);
                    }
                }
                // A dialect-lowered prefix may already end at a branch trigger.
                // Treat that boundary exactly like a sampled trigger.
                if kind == OpKind::PrefillUnd && self.prefilled_gen_trigger(id) {
                    self.begin_image(id);
                    return;
                }
                // Immediate Gen-only profiles skip Und decode after context prep.
                if starts_gen_after_context && kind == OpKind::PrefillUnd {
                    self.begin_image(id);
                    return;
                }
                let tok = sr.sampled_token_id.unwrap_or(self.ctrl.eos[0]);
                let logprob = sr.sampled_logprob;
                let (images_done, max_images) = {
                    let st = self.running.get(&id).unwrap();
                    (st.image_gen.images_done, st.req.image.max_images as usize)
                };
                // the model requested an image inline; honor it while the
                // request is still under its image budget.
                let direct_trigger = self
                    .running
                    .get(&id)
                    .is_some_and(|st| Self::direct_trigger_matches(st, tok));
                if direct_trigger && can_open_gen_branch && images_done < max_images {
                    self.begin_image(id);
                    return;
                }
                if self.emit_or_finish_und_token(id, tok, logprob, sr.top_logprobs.clone(), true) {
                    return;
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.und.next_token = tok;
                    st.lifecycle.phase = Phase::DecodeUnd;
                }
                if !self.advance_grammar(id, tok) {
                    return;
                }
                if can_open_gen_branch
                    && images_done < max_images
                    && self
                        .running
                        .get(&id)
                        .is_some_and(Self::generated_trigger_matches)
                {
                    self.begin_image(id);
                }
            }
            OpKind::DenoiseGen => {
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = match transition.delta {
                        crate::generation::TransitionDelta::DenoiseGen { start_step, .. } => {
                            start_step
                        }
                        _ => st.image_gen.steps_done,
                    };
                    (
                        st.image_gen.image_id,
                        st.req.image.height,
                        st.req.image.width,
                        st.req.image.steps,
                        prev,
                    )
                };
                let sd = self
                    .running
                    .get(&id)
                    .map(|s| s.image_gen.steps_done)
                    .unwrap_or(0);
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
                    st.lifecycle.phase = Phase::CommitGen;
                }
            }
            OpKind::CommitGen | OpKind::CommitWriteback => {
                if kind == OpKind::CommitWriteback
                    && let Some(st) = self.running.get_mut(&id)
                {
                    sr.sampled_token_id = sr.sampled_token_id.or(st.feedback.sampled_token.take());
                    sr.sampled_logprob = sr.sampled_logprob.or(st.feedback.sampled_logprob.take());
                    sr.top_logprobs = sr.top_logprobs.or(st.feedback.top_logprobs.take());
                }
                if kind == OpKind::CommitGen {
                    let image_id = self.running.get(&id).map_or(0, |st| st.image_gen.image_id);
                    self.emit(id, GenEvent::ImageCommit { image_id });
                }
                if kind == OpKind::CommitGen
                    && self
                        .running
                        .get(&id)
                        .is_some_and(Self::reingests_generated_image)
                {
                    let Some(image_b64) = sr.image_png_b64 else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let image_id = self.running.get(&id).map_or(0, |st| st.image_gen.image_id);
                    let Some(event) = image_done_event(image_id, image_b64.clone()) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    self.emit(id, event);
                    if let Some(st) = self.running.get_mut(&id) {
                        st.feedback.image_b64 = Some(image_b64);
                        st.feedback.ingest_step = 0;
                        st.feedback.staged_image = None;
                        st.lifecycle.phase = Phase::FeedbackIngest;
                    }
                    return;
                }
                if kind == OpKind::CommitGen
                    && self
                        .running
                        .get(&id)
                        .is_some_and(Self::commit_requires_writeback)
                {
                    let Some(locator) = sr.locator.clone().filter(|value| !value.is_empty()) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.feedback.locator = Some(locator);
                        st.feedback.sampled_token = sr.sampled_token_id;
                        st.feedback.sampled_logprob = sr.sampled_logprob;
                        st.feedback.top_logprobs = sr.top_logprobs;
                        st.lifecycle.phase = Phase::CommitWriteback;
                    }
                    if let Some(b64) = sr.image_png_b64 {
                        let image_id = self
                            .running
                            .get(&id)
                            .map(|st| st.image_gen.image_id)
                            .unwrap_or(0);
                        let Some(event) = image_done_event(image_id, b64) else {
                            return self.finish(id, FinishReason::Error);
                        };
                        self.emit(id, event);
                    }
                    return;
                }
                self.bm.activate(id);
                let (image_id, continues_after_gen_commit) = {
                    let st = self.running.get(&id).unwrap();
                    (st.image_gen.image_id, st.continues_after_gen_commit())
                };
                if let Some(b64) = sr.image_png_b64 {
                    let Some(event) = image_done_event(image_id, b64) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    self.emit(id, event);
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.image_gen.images_done += 1;
                    st.und.text_since_image = 0;
                }
                // pure t2i finishes after one image; generated branch round-trips
                // back to text (text → image → text → image …). Termination then
                // happens in DecodeUnd on a genuine terminal condition (EOS or
                // max_tokens) or from a commit-side EOS reported by the worker.
                if continues_after_gen_commit {
                    if let Some(tok) = sr.sampled_token_id {
                        let (can_open_gen_branch, images_done, max_images) = {
                            let st = self.running.get_mut(&id).unwrap();
                            st.und.tokens_emitted += 1;
                            (
                                st.can_open_gen_branch(),
                                st.image_gen.images_done,
                                st.req.image.max_images as usize,
                            )
                        };
                        if self
                            .running
                            .get(&id)
                            .is_some_and(|state| Self::direct_trigger_matches(state, tok))
                            && can_open_gen_branch
                            && images_done < max_images
                        {
                            self.begin_image(id);
                            return;
                        }
                        if self.emit_or_finish_und_token(
                            id,
                            tok,
                            sr.sampled_logprob,
                            sr.top_logprobs.clone(),
                            true,
                        ) {
                            return;
                        }
                        if let Some(st) = self.running.get_mut(&id) {
                            st.und.next_token = tok;
                            st.lifecycle.phase = Phase::DecodeUnd;
                            st.und.round_tokens.push(tok);
                        }
                        if !self.advance_grammar(id, tok) {
                            return;
                        }
                        if can_open_gen_branch
                            && images_done < max_images
                            && self
                                .running
                                .get(&id)
                                .is_some_and(Self::generated_trigger_matches)
                        {
                            self.begin_image(id);
                        }
                        return;
                    }
                    let next_token = self.feedback_next_token(id);
                    let Some(next_token) = next_token else {
                        return self.finish(id, FinishReason::Error);
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.lifecycle.phase = Phase::DecodeUnd;
                        st.und.next_token = next_token;
                    }
                } else {
                    self.finish(id, FinishReason::ImageDone);
                }
            }
            OpKind::VitEncode | OpKind::VaeEncode => match &transition.delta {
                crate::generation::TransitionDelta::IngestImageStep {
                    is_final_step,
                    encoder_cache_key,
                    cache_hit,
                    ..
                } => {
                    let Some(result_handle) = sr.encoder_handle.filter(|handle| *handle != 0)
                    else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let mut active_handle = result_handle;
                    let mut free_handles = Vec::new();
                    if !cache_hit {
                        if let Some(cache_key) = encoder_cache_key {
                            if let Some(freed) = self.enc_cache.insert_output(
                                *cache_key,
                                result_handle,
                                sr.num_tokens.unwrap_or_default(),
                            ) {
                                free_handles.push(freed);
                            }
                            let Some(handle) = self.enc_cache.acquire(*cache_key) else {
                                return self.finish(id, FinishReason::Error);
                            };
                            active_handle = handle;
                            if let Some(st) = self.running.get_mut(&id) {
                                st.ingest.acquired_encoder_pins.push(EncoderCachePin {
                                    key: *cache_key,
                                    handle,
                                });
                            }
                        } else if let Some(st) = self.running.get_mut(&id) {
                            st.ingest.transient_encoder_handles.push(result_handle);
                        }
                    }
                    let bos = self.ctrl.bos;
                    if let Some(st) = self.running.get_mut(&id) {
                        if let Some((height, width)) = sr.image_hw {
                            st.image_gen.image_hw = (height, width);
                            st.req.image.height = height;
                            st.req.image.width = width;
                        }
                        if *is_final_step {
                            st.ingest.staged_image = None;
                            free_handles.append(&mut st.ingest.transient_encoder_handles);
                            if st.ingest.mm_cursor >= st.context.images.len()
                                && st.ingest.prompt_cursor >= st.context.prompt_ids.len() as u32
                            {
                                st.und.next_token = bos;
                                st.und.round_tokens.clear();
                                st.lifecycle.phase = Phase::DecodeUnd;
                            }
                        } else {
                            st.ingest.staged_image = Some(active_handle);
                        }
                    }
                    if !free_handles.is_empty() {
                        self.gated_control(ControlOp::FreeEncoder(free_handles));
                    }
                }
                crate::generation::TransitionDelta::FeedbackIngestStep {
                    is_final_step, ..
                } => {
                    let Some(handle) = sr.encoder_handle.filter(|handle| *handle != 0) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let mut free_handles = Vec::new();
                    if let Some(st) = self.running.get_mut(&id) {
                        st.ingest.transient_encoder_handles.push(handle);
                        if *is_final_step {
                            st.feedback.staged_image = None;
                            free_handles.append(&mut st.ingest.transient_encoder_handles);
                        } else {
                            st.feedback.staged_image = Some(handle);
                        }
                    }
                    if !free_handles.is_empty() {
                        self.gated_control(ControlOp::FreeEncoder(free_handles));
                    }
                    if !is_final_step {
                        return;
                    }
                    let Some(next_token) = self.feedback_next_token(id) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.image_gen.images_done += 1;
                        st.und.text_since_image = 0;
                        st.und.next_token = next_token;
                        st.und.round_tokens.clear();
                        st.lifecycle.phase = Phase::DecodeUnd;
                    }
                }
                _ => self.finish(id, FinishReason::Error),
            },
            _ => {}
        }
    }

    fn resolve_prompt_logprobs(
        &mut self,
        id: RequestId,
        positions: Vec<Vec<uniserve_worker_wire::TokenLogprob>>,
    ) {
        let total = positions.len();
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let processed_before = state.ingest.prompt_logprobs_processed;
        let emitted_before = state.ingest.prompt_logprobs_emitted;
        let expected_total = state.context.prompt_ids.len().saturating_sub(1);
        state.ingest.prompt_logprobs_processed = processed_before.saturating_add(total);
        let skip = emitted_before.saturating_sub(processed_before);
        let remaining = expected_total.saturating_sub(emitted_before);
        let selected: Vec<_> = positions.into_iter().skip(skip).take(remaining).collect();
        state.ingest.prompt_logprobs_emitted = emitted_before.saturating_add(selected.len());
        if selected.is_empty() {
            return;
        }
        self.emit(
            id,
            GenEvent::PromptLogprobs {
                positions: selected
                    .into_iter()
                    .map(|entries| uniserve_engine_api::PositionLogprobs {
                        entries: ranked_logprobs(entries),
                    })
                    .collect(),
            },
        );
    }

    fn record_gen_trigger_for_replay(&mut self, id: RequestId) {
        let Some(start) = self
            .running
            .get(&id)
            .and_then(|st| st.req.policy.trigger.direct_token())
        else {
            return;
        };
        if let Some(st) = self.running.get_mut(&id)
            && st.continues_after_gen_commit()
            && st.und.tokens_emitted > st.replay.generated_ids.len()
            && st.replay.generated_ids.last().copied() != Some(start)
        {
            st.replay.generated_ids.push(start);
        }
    }

    fn promote_gen_branch_reservation(&mut self, id: RequestId) -> bool {
        let Some((required_blocks, reserves_envelope)) = self.running.get(&id).map(|st| {
            (
                st.resources.worstcase_blocks,
                st.resources.reserve_worstcase,
            )
        }) else {
            return false;
        };
        let allocated_blocks = self.bm.blocks_for(id).len();
        if !reserves_envelope || allocated_blocks < required_blocks {
            self.trace_record(json!({
                "event": "gen_branch_reservation_rejected",
                "at_s": now(),
                "request_id": id.0,
                "required_blocks": required_blocks,
                "allocated_blocks": allocated_blocks,
            }));
            self.finish(id, FinishReason::Error);
            return false;
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.image_gen.branch_pending = false;
        }
        self.trace_record(json!({
            "event": "gen_branch_capacity_ready",
            "at_s": now(),
            "request_id": id.0,
            "required_blocks": required_blocks,
            "allocated_blocks": allocated_blocks,
            "free_blocks": self.bm.free_blocks(),
            "reserved_blocks": self.reserved_blocks,
        }));
        true
    }

    fn begin_image(&mut self, id: RequestId) {
        self.record_gen_trigger_for_replay(id);
        if let Some(st) = self.running.get_mut(&id) {
            st.image_gen.cond_pos = st.und.logical_pos;
            st.image_gen.steps_done = 0;
            st.image_gen.image_id += 1;
            st.und.text_since_image = 0;
            st.lifecycle.phase = Phase::DenoiseGen;
            st.image_gen.branch_pending = true;
        }
        self.promote_gen_branch_reservation(id);
    }

    fn emit(&mut self, id: RequestId, ev: GenEvent) {
        if let Some(st) = self.running.get_mut(&id)
            && st.event_tx.send(ev).is_err()
        {
            st.cancelled = true; // receiver dropped == cancel
        }
    }
    /// Emit a text token and record it for preemption-recompute.
    fn emit_text(&mut self, id: RequestId, tok: u32, logprob: Option<f32>) -> bool {
        let action = if let Some(st) = self.running.get_mut(&id) {
            st.replay.generated_ids.push(tok);
            st.und.text_since_image = st.und.text_since_image.saturating_add(1);
            st.req.behavior.und_tokens
        } else {
            return false;
        };
        match action {
            uniserve_core::UndTokenAction::Emit => {
                self.emit(id, GenEvent::TextToken { id: tok, logprob });
                true
            }
            uniserve_core::UndTokenAction::KeepInternal => false,
            uniserve_core::UndTokenAction::Reject => {
                self.finish(id, FinishReason::Error);
                false
            }
        }
    }

    fn emit_sampled_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<uniserve_worker_wire::TokenLogprob>>,
    ) {
        if self.emit_text(id, tok, logprob)
            && let Some(top_logprobs) = top_logprobs.filter(|entries| !entries.is_empty())
        {
            self.emit(
                id,
                GenEvent::TokenLogprobs {
                    id: tok,
                    candidates: ranked_logprobs(top_logprobs),
                },
            );
        }
    }

    fn emit_terminal_stop_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<uniserve_worker_wire::TokenLogprob>>,
    ) {
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.req.policy.termination.emit_stop_token)
        {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        }
    }

    fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<uniserve_worker_wire::TokenLogprob>>,
        sampled: bool,
    ) -> bool {
        let Some(state) = self.running.get(&id) else {
            return true;
        };
        let generated = state.und.tokens_emitted;
        let under_floor = generated < state.req.sampling.min_tokens;
        let stop_hit = state.req.policy.termination.stop_finishes
            && !under_floor
            && state.req.stop_token_ids.contains(&token_id);
        let eos_hit = state.req.policy.termination.eos_finishes
            && self.ctrl.eos.contains(&token_id)
            && !state.req.sampling.ignore_eos
            && !under_floor;
        let max_hit = generated >= state.req.max_und_tokens;

        if stop_hit {
            self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs);
            self.finish_with(id, FinishReason::Stop, Some(format!("token:{token_id}")));
            return true;
        }
        if eos_hit || max_hit {
            if !self.ctrl.eos.contains(&token_id) {
                if sampled {
                    self.emit_sampled_text(id, token_id, logprob, top_logprobs);
                } else {
                    self.emit_text(id, token_id, logprob);
                }
            }
            self.finish(
                id,
                if max_hit {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Eos
                },
            );
            return true;
        }
        if sampled {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        } else {
            self.emit_text(id, token_id, logprob);
        }
        !self.running.contains_key(&id)
    }

    fn emit_st(&self, st: &mut ReqState, ev: GenEvent) {
        if st.event_tx.send(ev).is_err() {
            st.cancelled = true;
        }
    }

    fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    fn finish_with(&mut self, id: RequestId, reason: FinishReason, stop_reason: Option<String>) {
        if reason == FinishReason::Error
            && let Some(state) = self.running.get(&id)
        {
            tracing::error!(
                request_id = id.0,
                phase = phase_str(state.lifecycle.phase),
                generated_tokens = state.und.tokens_emitted,
                images_done = state.image_gen.images_done,
                image_id = state.image_gen.image_id,
                denoise_steps_done = state.image_gen.steps_done,
                logical_position = state.und.logical_pos,
                physical_kv_len = state.und.physical_kv_len,
                "scheduler request terminated with an internal error"
            );
        }
        if let Some(mut st) = self.running.remove(&id) {
            self.order.retain(|x| *x != id);
            self.reserved_encoder_entries = self
                .reserved_encoder_entries
                .saturating_sub(st.req.resources.encoder_cache_keys.len());
            if st.resources.reserve_worstcase {
                self.reserved_blocks = self
                    .reserved_blocks
                    .saturating_sub(st.resources.worstcase_blocks);
            }
            let mut free_encoder_handles = std::mem::take(&mut st.ingest.transient_encoder_handles);
            for pin in &st.ingest.acquired_encoder_pins {
                if let Some(handle) = self.enc_cache.release(pin.key, pin.handle) {
                    free_encoder_handles.push(handle);
                }
            }
            if !free_encoder_handles.is_empty() {
                self.gated_control(ControlOp::FreeEncoder(free_encoder_handles));
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
                st.context.prompt_ids.len(),
                st.und.tokens_emitted,
                st.image_gen.images_done,
                "running",
            );
            let _ = st.event_tx.send(GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: st.und.tokens_emitted,
                images: st.image_gen.images_done,
                kv_transfer_params: None,
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

/// Physical transformer-token work a planned op contributes to one scheduler
/// step. Denoise executes every latent token once per CFG branch at every
/// timestep, so its cost must use the compiled latent geometry rather than the
/// scalar wire-op count.
fn planned_op_token_cost(transition: &PlannedTransition) -> usize {
    if transition.kind != OpKind::DenoiseGen {
        return op_token_cost(&transition.op);
    }
    let latent_tokens = usize::try_from(transition.resources.latent_units)
        .unwrap_or(usize::MAX)
        .max(1);
    let cfg_branches = transition
        .cfg
        .as_ref()
        .map_or(1, |cfg| usize::from(cfg.branch_count.max(1)));
    let timesteps = usize::from(transition.denoise_step_count.unwrap_or(1).max(1));
    latent_tokens
        .saturating_mul(cfg_branches)
        .saturating_mul(timesteps)
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

fn cfg_branch_count(image: &uniserve_core::ImageParams) -> u8 {
    image.cfg_branch_count()
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;

    use super::*;

    fn test_png_b64(width: u32, height: u32) -> String {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("PNG header");
            writer
                .write_image_data(&vec![0; (width * height) as usize])
                .expect("PNG pixels");
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn request_with_generation_behavior(
        request: &mut GenerationRequest,
        constraint: uniserve_core::GenerationConstraint,
    ) {
        request.constraint = constraint;
        request.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 42 };
        request.policy.feedback.get_or_insert({
            uniserve_core::GeneratedImageFeedbackRecipe {
                commit: uniserve_core::CommitRecipe::CommitGen,
                writeback: uniserve_core::FeedbackWriteback::DirectKv,
                next_und_token: uniserve_core::FeedbackNextToken::EndOfImage,
                logical_positions: 2,
                physical_kv_tokens: uniserve_core::ImageKvEffect::Bounded { max_tokens: 64 },
            }
        });
        request.behavior =
            uniserve_core::GenerationBehaviorDescriptor::resolve(constraint, &request.policy);
        request.resources.generated_feedback_makes_non_replayable =
            request.behavior.generated_image_feedback;
    }

    fn inflight(kind: OpKind, op_id: Option<u64>) -> InflightOp {
        let request = test_request(0, 1);
        let cursor = CursorProjection {
            phase: if kind == OpKind::PrefillUnd {
                Phase::Prefill
            } else {
                Phase::DecodeUnd
            },
            prompt_cursor: 0,
            logical_pos: 0,
            physical_kv_len: 0,
            replayability: crate::generation::Replayability::Replayable,
        };
        let intent = if kind == OpKind::PrefillUnd {
            TransitionIntent::IngestText {
                segment_index: 0,
                prompt_start: 0,
                token_ids: vec![1],
                new_blocks: Vec::new(),
                recent_tokens: None,
                allowed_tokens: None,
                suppress_tokens: None,
            }
        } else {
            TransitionIntent::DecodeUnd {
                position: 0,
                token_id: 1,
                token_source: TokenSource::Wire,
                new_blocks: Vec::new(),
                spec_token_ids: None,
                token_count: 1,
                stop_token_ids: None,
                stop_terminal: true,
                recent_tokens: None,
                allowed_tokens: None,
                suppress_tokens: None,
            }
        };
        let mut transition = GenerationPlanner::new()
            .plan(&request, cursor, intent)
            .expect("plan test transition");
        if let Some(op_id) = op_id {
            transition.assign_op_id(op_id);
        }
        InflightOp {
            transition,
            op_id,
            spec_tokens: Vec::new(),
            decode_token_count: 1,
            started: Instant::now(),
        }
    }

    // A result resolves the exact op the worker echoed even when a request's
    // operations complete out of submission order.
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
    fn take_inflight_rejects_absent_and_unknown_op_ids() {
        let mut q = VecDeque::from(vec![
            inflight(OpKind::DecodeUnd, Some(1)),
            inflight(OpKind::DecodeUnd, Some(2)),
        ]);

        assert!(take_inflight_by_op_id(&mut q, None).is_none());
        assert!(take_inflight_by_op_id(&mut q, Some(999)).is_none());
        assert_eq!(q.len(), 2);
        assert_eq!(q.front().and_then(|op| op.op_id), Some(1));
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
    fn denoise_step_budget_counts_cfg_latent_transformer_tokens() {
        let mut sched = test_scheduler();
        let request_id = RequestId(1);
        let mut request = test_request(request_id.0, 4);
        request_with_generation_behavior(
            &mut request,
            uniserve_core::GenerationConstraint::Default,
        );
        request.image = uniserve_core::ImageParams {
            height: 1_024,
            width: 1_024,
            steps: 50,
            cfg_img_scale: 1.5,
            max_images: 1,
            retain_images: false,
            ..Default::default()
        };
        compile_resources(&sched, &mut request);
        let _events = sched.submit_for_test(request);
        sched.admit();
        {
            let state = sched
                .running
                .get_mut(&request_id)
                .expect("generation request admitted");
            state.ingest.prompt_cursor = 4;
            state.und.logical_pos = 4;
            state.und.physical_kv_len = 4;
        }
        sched.begin_image(request_id);

        let transition = sched
            .next_transition(request_id, 8_192)
            .expect("denoise transition");

        assert_eq!(transition.kind, OpKind::DenoiseGen);
        assert_eq!(planned_op_token_cost(&transition), 12_288);
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

    #[test]
    fn cfg_branch_count_tracks_active_guidance_axes() {
        let mut img = uniserve_core::ImageParams {
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            ..Default::default()
        };
        assert_eq!(cfg_branch_count(&img), 2);
        img.cfg_text_scale = 1.0;
        img.cfg_img_scale = 1.0;
        assert_eq!(cfg_branch_count(&img), 1);
        img.cfg_text_scale = 3.0;
        img.cfg_img_scale = 3.0;
        assert_eq!(cfg_branch_count(&img), 2);
        img.cfg_text_scale = 4.0;
        img.cfg_img_scale = 1.5;
        assert_eq!(cfg_branch_count(&img), 3);
    }

    #[test]
    fn denoise_scratch_accounts_for_branch_prefixes_markers_and_block_rounding() {
        assert_eq!(
            uniserve_core::denoise_scratch_tokens(4096, 2, 45, 0, 1, 64),
            4160
        );
        assert_eq!(
            uniserve_core::denoise_scratch_tokens(4096, 2, 45, 0, 2, 64),
            8320
        );
        assert_eq!(
            uniserve_core::denoise_scratch_tokens(4096, 2, 45, 80, 3, 64),
            12_608
        );
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
        fn generated_image_commit_capabilities(
            &self,
        ) -> uniserve_core::GeneratedImageCommitCapabilities {
            uniserve_core::GeneratedImageCommitCapabilities {
                inline: true,
                separate_writeback: true,
            }
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

    fn test_request(id: u64, prompt_len: usize) -> GenerationRequest {
        let constraint = uniserve_core::GenerationConstraint::UndOnly;
        let policy = uniserve_core::GenerationPolicyDescriptor::default();
        GenerationRequest {
            request_id: RequestId(id),
            context: vec![uniserve_core::ContextSegment::UndTokens {
                token_ids: vec![1u32; prompt_len],
                visibility: uniserve_core::UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: uniserve_core::GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: uniserve_core::SamplingParams::default(),
            image: uniserve_core::ImageParams::default(),
            max_und_tokens: 16,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            lora_id: None,
            grammar: None,
            cache: Default::default(),
            policy,
            resources: uniserve_core::GenerationResourceBounds {
                context_tokens: prompt_len,
                max_kv_tokens: prompt_len + 16,
                ..Default::default()
            },
        }
    }

    fn test_scheduler() -> Scheduler {
        let mut caps = EngineCaps::default();
        caps.supported_ops
            .extend([OpKind::VitEncode, OpKind::CommitWriteback]);
        caps.max_latent_size = 65_536;
        caps.max_vae_grid_tokens = 65_536;
        caps.max_vit_grid_tokens = 65_536;
        caps.encoder_cache_budget = 256;
        Scheduler::new(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            DEFAULT_MAX_BATCH,
        )
    }

    fn compile_resources(scheduler: &Scheduler, request: &mut GenerationRequest) {
        request.resources = uniserve_core::GenerationResourceBounds::conservative(
            &request.context,
            &request.negative_context,
            &request.behavior,
            &request.policy,
            &request.image,
            request.max_und_tokens,
            &request.cache,
            &scheduler.generation_runtime_capabilities(),
        )
        .expect("test request resources");
    }

    #[test]
    fn prefix_reset_without_replay_policy_preserves_active_requests() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        compile_resources(&scheduler, &mut request);
        scheduler.submit_for_test(request);
        scheduler.admit();
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();

        scheduler.begin_prefix_cache_reset(false, reply_tx);

        assert!(!reply_rx.recv().unwrap().unwrap());
        assert!(scheduler.running.contains_key(&RequestId(1)));
        assert!(scheduler.pending_prefix_reset.is_none());
    }

    #[test]
    fn prefix_reset_requeues_replayable_request_with_generated_context() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        compile_resources(&scheduler, &mut request);
        scheduler.submit_for_test(request);
        scheduler.admit();
        let state = scheduler.running.get_mut(&RequestId(1)).unwrap();
        state.replay.generated_ids = vec![7, 8];
        state.und.tokens_emitted = 2;
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();

        scheduler.begin_prefix_cache_reset(true, reply_tx);

        assert!(reply_rx.recv().unwrap().unwrap());
        assert!(scheduler.running.is_empty());
        let state = scheduler.pending.pop_request().expect("request requeued");
        assert_eq!(state.effective_prompt(), &[1, 1, 1, 1, 7, 8]);
        assert_eq!(state.und.tokens_emitted, 2);
        assert!(state.replay.preempted);
        assert_eq!(state.ingest.prompt_cursor, 0);
    }

    #[test]
    fn prefix_reset_rejects_non_replayable_running_state() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        compile_resources(&scheduler, &mut request);
        scheduler.submit_for_test(request);
        scheduler.admit();
        scheduler
            .running
            .get_mut(&RequestId(1))
            .unwrap()
            .replay
            .replayability = crate::generation::Replayability::NotReplayable;
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();

        scheduler.begin_prefix_cache_reset(true, reply_tx);

        assert!(!reply_rx.recv().unwrap().unwrap());
        assert!(scheduler.running.contains_key(&RequestId(1)));
    }

    #[test]
    fn prefix_reset_waits_for_inflight_transition_resolution() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        compile_resources(&scheduler, &mut request);
        scheduler.submit_for_test(request);
        scheduler.admit();
        scheduler.inflight_ops.insert(
            RequestId(1),
            VecDeque::from([inflight(OpKind::PrefillUnd, Some(1))]),
        );
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();

        scheduler.begin_prefix_cache_reset(true, reply_tx);

        assert!(reply_rx.try_recv().is_err());
        assert!(scheduler.pending_prefix_reset.is_some());
        scheduler.inflight_ops.clear();
        assert!(scheduler.progress_prefix_cache_reset());
        assert!(reply_rx.recv().unwrap().unwrap());
    }

    #[test]
    fn shutdown_resolves_a_pending_prefix_reset_reply() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        compile_resources(&scheduler, &mut request);
        scheduler.submit_for_test(request);
        scheduler.admit();
        scheduler.inflight_ops.insert(
            RequestId(1),
            VecDeque::from([inflight(OpKind::PrefillUnd, Some(1))]),
        );
        let (reply_tx, reply_rx) = std::sync::mpsc::channel();
        scheduler.begin_prefix_cache_reset(true, reply_tx);

        scheduler.abort_all_requests();

        let error = reply_rx.recv().unwrap().expect_err("reset must fail");
        assert!(error.contains("scheduler stopped"));
    }

    #[test]
    fn prompt_logprobs_are_not_reemitted_after_preemption_recompute() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        request.sampling.return_prompt_logprobs = true;
        compile_resources(&scheduler, &mut request);
        let mut events = scheduler.submit_for_test(request);
        scheduler.admit();
        let state = scheduler.running.get_mut(&RequestId(1)).unwrap();
        state.ingest.prompt_logprobs_processed = 2;
        state.ingest.prompt_logprobs_emitted = 2;

        scheduler.do_preempt(RequestId(1));
        scheduler.admit();
        scheduler.resolve_prompt_logprobs(
            RequestId(1),
            vec![
                vec![uniserve_worker_wire::TokenLogprob(1, -0.1, 1)],
                vec![uniserve_worker_wire::TokenLogprob(1, -0.2, 1)],
                vec![uniserve_worker_wire::TokenLogprob(1, -0.3, 1)],
            ],
        );

        let prompt_events = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                GenEvent::PromptLogprobs { positions } => Some(positions),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(prompt_events.len(), 1);
        assert_eq!(prompt_events[0].len(), 1);
        assert_eq!(prompt_events[0][0].entries[0].logprob, -0.3);
    }

    #[test]
    fn max_token_output_emits_logprobs_before_terminal_event() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        request.max_und_tokens = 1;
        request.sampling.return_logprobs = true;
        request.sampling.n_logprobs = 1;
        compile_resources(&scheduler, &mut request);
        let mut events = scheduler.submit_for_test(request);
        scheduler.admit();

        scheduler.resolve_decode_text(
            RequestId(1),
            uniserve_worker_wire::SeqResult {
                req_id: RequestId(1),
                sampled_token_id: Some(7),
                sampled_logprob: Some(-0.25),
                top_logprobs: Some(vec![uniserve_worker_wire::TokenLogprob(7, -0.25, 1)]),
                ..Default::default()
            },
        );

        let kinds = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                GenEvent::TextToken { .. } => Some("text"),
                GenEvent::TokenLogprobs { .. } => Some("logprobs"),
                GenEvent::Finished { .. } => Some("finished"),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(kinds, ["text", "logprobs", "finished"]);
    }

    #[test]
    fn final_prefill_token_obeys_the_output_token_cap() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        request.max_und_tokens = 1;
        compile_resources(&scheduler, &mut request);
        let mut events = scheduler.submit_for_test(request);
        scheduler.admit();
        let (_, mut transitions) = scheduler.assemble();
        assert_eq!(transitions.len(), 1);
        let transition = transitions.pop().expect("prefill transition");
        assert_eq!(transition.kind, OpKind::PrefillUnd);

        apply_and_resolve(
            &mut scheduler,
            RequestId(1),
            transition,
            uniserve_worker_wire::SeqResult {
                req_id: RequestId(1),
                sampled_token_id: Some(7),
                ..Default::default()
            },
            Vec::new(),
        );

        assert!(!scheduler.running.contains_key(&RequestId(1)));
        let output = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                GenEvent::TextToken { id, .. } => Some(Ok(id)),
                GenEvent::Finished { reason, .. } => Some(Err(reason)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(output, [Ok(7), Err(FinishReason::MaxTokens)]);
    }

    #[test]
    fn post_commit_stop_token_counts_toward_the_minimum() {
        let mut scheduler = test_scheduler();
        let mut request = test_request(1, 4);
        request_with_generation_behavior(
            &mut request,
            uniserve_core::GenerationConstraint::Default,
        );
        request.sampling.min_tokens = 1;
        request.stop_token_ids = vec![77];
        request.image.height = 64;
        request.image.width = 64;
        request.image.max_images = 1;
        compile_resources(&scheduler, &mut request);
        let mut events = scheduler.submit_for_test(request);
        scheduler.admit();
        let id = RequestId(1);
        if let Some(state) = scheduler.running.get_mut(&id) {
            state.ingest.prompt_cursor = 4;
            state.und.logical_pos = 4;
            state.und.physical_kv_len = 4;
            state.lifecycle.phase = Phase::CommitGen;
            state.image_gen.cond_pos = 4;
            state.image_gen.image_id = 1;
        }
        let transition = scheduler.next_transition(id, 8).expect("commit transition");
        assert_eq!(transition.kind, OpKind::CommitGen);

        apply_and_resolve(
            &mut scheduler,
            id,
            transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(77),
                image_hw: Some((64, 64)),
                image_png_b64: Some(test_png_b64(64, 64)),
                num_tokens: Some(1),
                ..Default::default()
            },
            Vec::new(),
        );

        assert!(!scheduler.running.contains_key(&id));
        assert!(
            std::iter::from_fn(|| events.try_recv().ok()).any(|event| matches!(
                event,
                GenEvent::Finished {
                    reason: FinishReason::Stop,
                    ..
                }
            ))
        );
    }

    fn add_context_image(request: &mut GenerationRequest, position: usize, dual_encode: bool) {
        let tokens = request
            .context
            .iter()
            .flat_map(|segment| match segment {
                uniserve_core::ContextSegment::UndTokens { token_ids, .. } => token_ids.clone(),
                uniserve_core::ContextSegment::Image { .. } => Vec::new(),
            })
            .collect::<Vec<_>>();
        let position = position.min(tokens.len());
        let ingest = if dual_encode {
            uniserve_core::ImageIngestRecipe::vae_then_vit(
                1,
                uniserve_core::ImageKvEffect::WorkerDefined,
                uniserve_core::ImageKvEffect::WorkerDefined,
            )
        } else {
            uniserve_core::ImageIngestRecipe::vit_only(
                1,
                uniserve_core::ImageKvEffect::WorkerDefined,
            )
        };
        let encoder_cache_keys = ingest.encoder_cache_keys(7);
        request.context = vec![
            uniserve_core::ContextSegment::UndTokens {
                token_ids: tokens[..position].to_vec(),
                visibility: uniserve_core::UndVisibility::Internal,
            },
            uniserve_core::ContextSegment::Image {
                image: uniserve_core::ImageSegment {
                    hash: 7,
                    b64: "aGVsbG8=".to_string(),
                    placement: uniserve_core::SegmentPlacement::AtToken {
                        position: position as u32,
                    },
                },
                ingest,
            },
            uniserve_core::ContextSegment::UndTokens {
                token_ids: tokens[position..].to_vec(),
                visibility: uniserve_core::UndVisibility::Internal,
            },
        ];
        request.resources.encoder_cache_keys = encoder_cache_keys;
    }

    fn apply_and_resolve(
        scheduler: &mut Scheduler,
        id: RequestId,
        mut transition: PlannedTransition,
        mut result: uniserve_worker_wire::SeqResult,
        draft_token_ids: Vec<u32>,
    ) {
        let op_id = transition.op_id.unwrap_or_else(|| {
            let op_id = scheduler.next_op_id;
            scheduler.next_op_id = scheduler.next_op_id.saturating_add(1);
            op_id
        });
        transition.assign_op_id(op_id);
        result.op_id = Some(op_id);
        result.op_kind = Some(transition.kind);
        transition
            .validate_result(&result)
            .expect("test result validates against transition");
        scheduler
            .running
            .get_mut(&id)
            .expect("request running")
            .cursor
            .apply_transition(&transition, &result)
            .expect("apply test transition");
        scheduler.resolve(id, transition, result, draft_token_ids);
    }

    #[test]
    fn plain_decode_burst_marks_stop_tokens_terminal() {
        let mut sched = test_scheduler();
        sched.decode_token_burst = 8;
        let mut req = test_request(1, 5);
        req.stop_token_ids = vec![77];
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 5;
            st.und.logical_pos = 5;
            st.und.next_token = 11;
        }

        let op = sched.next_transition(id, 8).expect("decode op");

        assert_eq!(op.kind, OpKind::DecodeUnd);
        assert!(op.decode_token_count.unwrap_or(1) > 1);
        assert!(
            op.decode_stop_token_ids
                .as_ref()
                .is_some_and(|ids| ids.contains(&77))
        );
        assert!(op.decode_stop_terminal);
    }

    #[test]
    fn termination_descriptor_controls_eos_and_explicit_stop_tokens() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 5);
        req.max_und_tokens = 2;
        req.stop_token_ids = vec![77];
        req.policy.termination.eos_finishes = false;
        req.policy.termination.stop_finishes = false;
        req.behavior =
            uniserve_core::GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);
        let mut events = sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 5;
            st.und.logical_pos = 5;
            st.und.physical_kv_len = 5;
        }

        sched.resolve_decode_text(
            id,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(77),
                ..Default::default()
            },
        );
        assert!(sched.running.contains_key(&id));

        let eos = sched.ctrl.eos[0];
        sched.resolve_decode_text(
            id,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(eos),
                ..Default::default()
            },
        );
        assert!(!sched.running.contains_key(&id));
        let mut tokens = Vec::new();
        let mut terminal = None;
        while let Ok(event) = events.try_recv() {
            match event {
                GenEvent::TextToken { id, .. } => tokens.push(id),
                GenEvent::Finished { reason, .. } => terminal = Some(reason),
                _ => {}
            }
        }
        assert_eq!(tokens, vec![77]);
        assert_eq!(terminal, Some(FinishReason::MaxTokens));
    }

    #[test]
    fn termination_descriptor_controls_terminal_stop_token_emission() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 5);
        req.stop_token_ids = vec![77];
        req.policy.termination.emit_stop_token = true;
        req.behavior =
            uniserve_core::GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);
        let mut events = sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(state) = sched.running.get_mut(&id) {
            state.lifecycle.phase = Phase::DecodeUnd;
            state.ingest.prompt_cursor = 5;
            state.und.logical_pos = 5;
            state.und.physical_kv_len = 5;
        }

        sched.resolve_decode_text(
            id,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(77),
                ..Default::default()
            },
        );

        assert!(!sched.running.contains_key(&id));
        let emitted = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                GenEvent::TextToken { id, .. } => Some(format!("token:{id}")),
                GenEvent::Finished { reason, .. } => Some(format!("finish:{reason:?}")),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(emitted, vec!["token:77", "finish:Stop"]);
    }

    #[test]
    fn gen_branch_decode_burst_keeps_transition_stops_nonterminal() {
        let mut sched = test_scheduler();
        sched.decode_token_burst = 8;
        let mut req = test_request(1, 5);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        req.image = uniserve_core::ImageParams {
            max_images: 1,
            ..Default::default()
        };
        compile_resources(&sched, &mut req);
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 5;
            st.und.logical_pos = 5;
            st.und.next_token = 11;
            st.image_gen.images_done = 0;
        }

        let op = sched.next_transition(id, 8).expect("decode op");

        assert_eq!(op.kind, OpKind::DecodeUnd);
        assert!(op.decode_token_count.unwrap_or(1) > 1);
        assert!(
            op.decode_stop_token_ids
                .as_ref()
                .is_some_and(|ids| ids.contains(&42))
        );
        assert!(!op.decode_stop_terminal);
    }

    #[test]
    fn planning_text_prefill_does_not_advance_committed_cursor() {
        let mut sched = test_scheduler();
        let req = test_request(1, 5);
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);

        let op = sched.next_transition(id, 3).expect("prefill op");

        assert_eq!(op.kind, OpKind::PrefillUnd);
        assert_eq!(op.pos_range, (0, 3));
        let st = sched.running.get(&id).expect("request running");
        assert_eq!(st.ingest.prompt_cursor, 0);
        assert_eq!(st.und.logical_pos, 0);

        apply_and_resolve(
            &mut sched,
            id,
            op,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(9),
                ..Default::default()
            },
            Vec::new(),
        );
        let st = sched.running.get(&id).expect("request running");
        assert_eq!(st.ingest.prompt_cursor, 3);
        assert_eq!(st.und.logical_pos, 3);
    }

    #[test]
    fn planning_context_prefill_does_not_advance_committed_cursor() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 5);
        add_context_image(&mut req, 2, false);
        compile_resources(&sched, &mut req);
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);

        let op = sched.next_transition(id, 64).expect("context prefill op");

        assert_eq!(op.kind, OpKind::PrefillUnd);
        assert_eq!(op.pos_range, (0, 2));
        let st = sched.running.get(&id).expect("request running");
        assert_eq!(st.ingest.prompt_cursor, 0);
        assert_eq!(st.und.logical_pos, 0);
        assert_eq!(st.und.physical_kv_len, 0);

        apply_and_resolve(
            &mut sched,
            id,
            op,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(9),
                ..Default::default()
            },
            Vec::new(),
        );
        let st = sched.running.get(&id).expect("request running");
        assert_eq!(st.ingest.prompt_cursor, 2);
        assert_eq!(st.und.logical_pos, 2);
        assert_eq!(st.und.physical_kv_len, 2);
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
    fn vae_grid_cap_uses_latent_size_when_unspecified() {
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
            let req = test_request(id, 4);
            receivers.push(sched.submit_for_test(req));
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
            let mut req = test_request(id, 4);
            req.sampling.ignore_eos = true;
            receivers.push(sched.submit_for_test(req));
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
            let request_id = op.req_id;
            apply_and_resolve(
                &mut sched,
                request_id,
                op,
                uniserve_worker_wire::SeqResult {
                    req_id: request_id,
                    sampled_token_id: Some(11 + request_id.0 as u32),
                    ..Default::default()
                },
                Vec::new(),
            );
        }
        let mut req = test_request(3, 4);
        req.sampling.ignore_eos = true;
        receivers.push(sched.submit_for_test(req));
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
    fn assemble_drains_a_resident_prompt_cohort_before_decode() {
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
        let mut req = test_request(1, 4);
        req.sampling.ignore_eos = true;
        let _rx1 = sched.submit_for_test(req);
        sched.admit();
        let (_new_reqs, ops) = sched.assemble();
        assert_eq!(ops[0].kind, OpKind::PrefillUnd);
        sched.register_inflight(ops[0].clone(), Instant::now());

        // A second resident prompt joins the same admission cohort while the
        // first prompt's final prefill is in flight.
        let mut req = test_request(2, 4);
        req.sampling.ignore_eos = true;
        let _rx2 = sched.submit_for_test(req);
        sched.admit();
        let (_new_reqs, ops2) = sched.assemble();
        assert_eq!(
            ops2.iter()
                .map(|op| (op.kind, op.req_id))
                .collect::<Vec<_>>(),
            vec![(OpKind::PrefillUnd, RequestId(2))]
        );
        sched.register_inflight(ops2[0].clone(), Instant::now());

        // A later arrival cannot extend the frozen cohort. While both cohort
        // prefills are in flight, no first token is emitted and request 3 waits.
        let mut req = test_request(3, 4);
        req.sampling.ignore_eos = true;
        let _rx3 = sched.submit_for_test(req);
        sched.admit();
        let (_new_reqs, ops3) = sched.assemble();
        assert!(
            ops3.is_empty(),
            "unexpected ops while cohort drains: {ops3:?}"
        );

        for (id, op) in [
            (RequestId(1), ops[0].clone()),
            (RequestId(2), ops2[0].clone()),
        ] {
            sched.inflight_ops.remove(&id);
            apply_and_resolve(
                &mut sched,
                id,
                op,
                uniserve_worker_wire::SeqResult {
                    req_id: id,
                    sampled_token_id: Some(10 + id.0 as u32),
                    ..Default::default()
                },
                Vec::new(),
            );
        }

        // Completing the cohort guarantees a decode turn; the later prompt did
        // not extend the cohort that was already in flight.
        let (_new_reqs, ops4) = sched.assemble();
        assert_eq!(
            ops4.iter()
                .map(|op| (op.kind, op.req_id))
                .collect::<Vec<_>>(),
            vec![
                (OpKind::DecodeUnd, RequestId(1)),
                (OpKind::DecodeUnd, RequestId(2)),
            ]
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
        let mut req = test_request(1, 4);
        req.sampling.ignore_eos = true;
        let _rx = sched.submit_for_test(req);
        sched.admit();

        // step 1: the whole prompt goes out as the final prefill chunk.
        let (_new_reqs, ops) = sched.assemble();
        assert_eq!(
            ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
            vec![OpKind::PrefillUnd]
        );
        sched.register_inflight(ops[0].clone(), Instant::now());

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
        sched.register_inflight(ops2[0].clone(), Instant::now());
        let (_new_reqs, ops3) = sched.assemble();
        assert!(ops3.is_empty(), "unexpected extra ops: {ops3:?}");
    }

    #[test]
    fn assemble_preserves_decode_bursts_across_the_device_relay() {
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
        sched.decode_lookahead = true;
        sched.decode_token_burst = 8;
        let mut req = test_request(1, 4);
        req.max_und_tokens = 32;
        req.sampling.ignore_eos = true;
        compile_resources(&sched, &mut req);
        let _rx = sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 4;
            st.und.logical_pos = 4;
            st.und.physical_kv_len = 4;
            st.und.tokens_emitted = 1;
            st.und.next_token = 11;
        }

        let (_new_reqs, first) = sched.assemble();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].token_source, TokenSource::Wire);
        assert_eq!(first[0].decode_token_count, Some(8));
        sched.register_inflight(first[0].clone(), Instant::now());

        let (_new_reqs, relayed) = sched.assemble();
        assert_eq!(
            relayed.len(),
            1,
            "an in-flight burst must keep the relay pipeline full"
        );
        assert_eq!(relayed[0].token_source, TokenSource::LastSampled);
        assert_eq!(relayed[0].decode_token_count, Some(8));
        assert_eq!(relayed[0].pos_range, (12, 13));
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
            let mut req = test_request(id, 4);
            req.sampling.ignore_eos = true;
            receivers.push(sched.submit_for_test(req));
        }
        sched.admit();
        let (_new_reqs, prefill_ops) = sched.assemble();
        assert_eq!(prefill_ops.len(), 2);
        for op in prefill_ops {
            let request_id = op.req_id;
            apply_and_resolve(
                &mut sched,
                request_id,
                op,
                uniserve_worker_wire::SeqResult {
                    req_id: request_id,
                    sampled_token_id: Some(11 + request_id.0 as u32),
                    ..Default::default()
                },
                Vec::new(),
            );
        }
        let mut req = test_request(3, 4);
        req.sampling.ignore_eos = true;
        receivers.push(sched.submit_for_test(req));
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
        assert_eq!(
            op_token_cost(rider),
            2,
            "rider chunk must clip to the mixed budget"
        );
    }

    #[test]
    fn gen_branch_ensures_current_image_capacity_before_first_image() {
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
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        req.max_und_tokens = 8;
        req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut req);
        let _rx = sched.submit_for_test(req);
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = 4;
            st.und.tokens_emitted = 1;
            st.replay.generated_ids.clear();
        }
        sched.begin_image(id);

        let (
            reserve_worstcase,
            gen_branch_pending,
            phase,
            generated_ids,
            reserved_blocks,
            allocated_blocks,
            worstcase_blocks,
            image_boundary_blocks,
        ) = {
            let st = sched.running.get(&id).expect("request remains running");
            let image_boundary_blocks = st
                .req
                .resources
                .max_kv_tokens
                .div_ceil(sched.caps.block_size as usize);
            (
                st.resources.reserve_worstcase,
                st.image_gen.branch_pending,
                st.lifecycle.phase,
                st.replay.generated_ids.clone(),
                sched.reserved_blocks,
                sched.bm.blocks_for(id).len(),
                st.resources.worstcase_blocks,
                image_boundary_blocks,
            )
        };
        assert!(reserve_worstcase);
        assert!(!gen_branch_pending);
        assert_eq!(reserved_blocks, worstcase_blocks);
        assert_eq!(allocated_blocks, worstcase_blocks);
        assert!(allocated_blocks >= image_boundary_blocks);
        assert_eq!(generated_ids, vec![42]);
        assert_eq!(phase, Phase::DenoiseGen);
    }

    #[test]
    fn deferred_gen_reservation_retry_does_not_advance_image_cursor() {
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
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut req);
        let _rx = sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DenoiseGen;
            st.ingest.prompt_cursor = 4;
            st.und.logical_pos = 4;
            st.und.physical_kv_len = 4;
            st.image_gen.cond_pos = 4;
            st.image_gen.image_id = 1;
            st.image_gen.steps_done = 0;
            st.image_gen.branch_pending = true;
        }

        let transition = sched.next_transition(id, 8).expect("denoise transition");
        let st = sched.running.get(&id).expect("request remains running");
        assert_eq!(transition.kind, OpKind::DenoiseGen);
        assert_eq!(st.lifecycle.phase, Phase::DenoiseGen);
        assert_eq!(st.image_gen.cond_pos, 4);
        assert_eq!(st.image_gen.image_id, 1);
        assert_eq!(st.image_gen.steps_done, 0);
        assert!(!st.image_gen.branch_pending);
    }

    #[test]
    fn gen_branch_burst_image_trigger_advances_speculative_feed() {
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
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        req.max_und_tokens = 8;
        req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut req);
        let _rx = sched.submit_for_test(req);
        sched.decode_token_burst = 3;
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 4;
            st.und.logical_pos = 10;
            st.und.physical_kv_len = 10;
            st.und.next_token = 77;
            st.und.tokens_emitted = 0;
            st.replay.generated_ids.clear();
        }
        let transition = sched.next_transition(id, 3).expect("decode transition");
        let image_trigger = 42;
        apply_and_resolve(
            &mut sched,
            id,
            transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(image_trigger),
                sampled_token_ids: Some(vec![5, image_trigger]),
                ..Default::default()
            },
            Vec::new(),
        );

        let st = sched.running.get(&id).expect("request enters image phase");
        assert_eq!(st.lifecycle.phase, Phase::DenoiseGen);
        assert_eq!(st.und.logical_pos, 13);
        assert_eq!(st.image_gen.cond_pos, 13);
        assert_eq!(st.replay.generated_ids, vec![5, image_trigger]);
    }

    #[test]
    fn gen_branch_burst_literal_image_trigger_advances_speculative_feed() {
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 64,
            latent_downsample: 16,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens {
                ..Default::default()
            },
            SchedulerConfig {
                max_batch: 4,
                max_num_batched_tokens: 16,
                max_num_seqs: 4,
                long_prefill_threshold: 16,
                ..Default::default()
            },
        );
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        req.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Suffix {
            token_ids: vec![21, 22],
        };
        req.behavior =
            uniserve_core::GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);
        req.max_und_tokens = 8;
        req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 1,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut req);
        let _rx = sched.submit_for_test(req);
        sched.decode_token_burst = 3;
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 4;
            st.und.logical_pos = 10;
            st.und.physical_kv_len = 10;
            st.und.next_token = 77;
            st.und.tokens_emitted = 0;
            st.replay.generated_ids.clear();
        }
        let transition = sched.next_transition(id, 3).expect("decode transition");
        apply_and_resolve(
            &mut sched,
            id,
            transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(22),
                sampled_token_ids: Some(vec![21, 22]),
                ..Default::default()
            },
            Vec::new(),
        );

        let st = sched.running.get(&id).expect("request enters image phase");
        assert_eq!(st.lifecycle.phase, Phase::DenoiseGen);
        assert_eq!(st.und.logical_pos, 13);
        assert_eq!(st.image_gen.cond_pos, 13);
        assert_eq!(st.replay.generated_ids, vec![21, 22]);
    }

    #[test]
    fn gen_branch_admission_reserves_full_worstcase_and_never_deadlocks() {
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
        let make_request = |id| {
            let mut req = test_request(id, 10);
            request_with_generation_behavior(
                &mut req,
                uniserve_core::GenerationConstraint::Default,
            );
            req.max_und_tokens = 16;
            req.image = uniserve_core::ImageParams {
                height: 32,
                width: 32,
                max_images: 1,
                retain_images: true,
                ..Default::default()
            };
            req.policy
                .feedback
                .as_mut()
                .expect("feedback recipe")
                .physical_kv_tokens = uniserve_core::ImageKvEffect::Exact { tokens: 4 };
            req
        };

        let mut sched = make_scheduler(12);
        let mut first = make_request(1);
        let mut second = make_request(2);
        compile_resources(&sched, &mut first);
        compile_resources(&sched, &mut second);
        sched.submit_for_test(first);
        sched.submit_for_test(second);
        sched.admit();

        let id = RequestId(1);
        let st = sched.running.get(&id).expect("first request admitted");
        assert!(st.resources.reserve_worstcase);
        assert_eq!(st.resources.worstcase_blocks, 8);
        assert_eq!(
            sched.bm.blocks_for(id).len(),
            8,
            "full worst case is physically allocated at admission"
        );
        assert!(
            !sched.running.contains_key(&RequestId(2)),
            "second request queues: its worst case does not fit alongside the first"
        );
        assert_eq!(sched.pending.len(), 1);

        // The admitted request can decode through its entire text budget from
        // its own allocation, with the retained-image envelope intact — no
        // dependency on any other request freeing blocks.
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = st.context.prompt_ids.len() as u32;
            st.ingest.prompt_cursor = st.context.prompt_ids.len() as u32;
        }
        let op = sched
            .next_transition(id, 16)
            .expect("decode proceeds from the preallocated envelope");
        assert_eq!(op.kind, OpKind::DecodeUnd);
        assert_eq!(
            sched.bm.blocks_for(id).len(),
            8,
            "decode does not allocate beyond the admission worst case"
        );
        let st = sched.running.get(&id).unwrap();
        let target = st.req.resources.max_kv_tokens;
        assert!(sched.bm.blocks_for(id).len() * sched.caps.block_size as usize >= target);

        // Finishing the resident request frees its envelope and unblocks the
        // queued one — the FIFO drains instead of deadlocking.
        sched.finish(id, FinishReason::Eos);
        sched.admit();
        assert!(
            sched.running.contains_key(&RequestId(2)),
            "queued request admits once the envelope is released"
        );
        assert_eq!(sched.pending.len(), 0);
    }

    #[test]
    fn assemble_interleaves_packed_mixed_and_decode_service() {
        // A ready text-decode op and image-denoise op share one packed forward.
        // While that denoise op is in flight, the next pipeline slot services
        // decode lookahead without admitting another denoise request.
        let caps = EngineCaps {
            block_size: 4,
            num_blocks: 256,
            latent_downsample: 16,
            ..Default::default()
        };
        let mut sched = Scheduler::with_config(
            Box::new(NullExecutor { caps, in_flight: 0 }),
            ControlTokens::default(),
            SchedulerConfig {
                max_batch: 8,
                max_num_batched_tokens: 24,
                max_num_seqs: 8,
                long_prefill_threshold: 16,
                ..Default::default()
            },
        );

        // Request 1: a continuation-capable request at an image boundary.
        let mut gen_req = test_request(1, 4);
        request_with_generation_behavior(
            &mut gen_req,
            uniserve_core::GenerationConstraint::Default,
        );
        gen_req.max_und_tokens = 64;
        gen_req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 2,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut gen_req);
        let _rx1 = sched.submit_for_test(gen_req);
        // Request 2: a plain text-decode request -> DecodeUnd.
        let mut und_req = test_request(2, 4);
        und_req.max_und_tokens = 64;
        und_req.sampling.ignore_eos = true;
        compile_resources(&sched, &mut und_req);
        let _rx2 = sched.submit_for_test(und_req);
        // Request 3: another image-denoise request. The token budget admits
        // one denoise operation per batch, matching large-latent workloads.
        let mut second_gen_req = test_request(3, 4);
        request_with_generation_behavior(
            &mut second_gen_req,
            uniserve_core::GenerationConstraint::Default,
        );
        second_gen_req.max_und_tokens = 64;
        second_gen_req.image = uniserve_core::ImageParams {
            height: 64,
            width: 64,
            max_images: 2,
            retain_images: true,
            ..Default::default()
        };
        compile_resources(&sched, &mut second_gen_req);
        let _rx3 = sched.submit_for_test(second_gen_req);
        sched.admit();
        sched.decode_lookahead = true;
        sched.decode_token_burst = 8;

        if let Some(st) = sched.running.get_mut(&RequestId(1)) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = 4;
            st.ingest.prompt_cursor = 4;
            st.und.tokens_emitted = 1;
            st.replay.generated_ids.clear();
        }
        sched.begin_image(RequestId(1));
        if let Some(st) = sched.running.get_mut(&RequestId(3)) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = 4;
            st.ingest.prompt_cursor = 4;
            st.und.tokens_emitted = 1;
            st.replay.generated_ids.clear();
        }
        sched.begin_image(RequestId(3));
        assert_eq!(
            sched.running.get(&RequestId(1)).map(|s| s.lifecycle.phase),
            Some(Phase::DenoiseGen),
            "request 1 must be denoising"
        );
        assert_eq!(
            sched.running.get(&RequestId(3)).map(|s| s.lifecycle.phase),
            Some(Phase::DenoiseGen),
            "request 3 must be denoising"
        );
        if let Some(st) = sched.running.get_mut(&RequestId(2)) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = 4;
            st.ingest.prompt_cursor = 4;
            st.und.next_token = 7;
        }

        let (_new, ops) = sched.assemble();
        let has_gen = ops.iter().any(|o| o.kind == OpKind::DenoiseGen);
        let has_und_decode = ops.iter().any(|o| o.kind == OpKind::DecodeUnd);
        assert!(
            has_gen && has_und_decode,
            "assemble must co-batch text-decode with image-denoise in one forward: {:?}",
            ops.iter().map(|o| o.kind).collect::<Vec<_>>()
        );
        assert_eq!(
            ops.iter()
                .filter(|op| op.kind == OpKind::DenoiseGen)
                .count(),
            1,
            "the token budget should admit one denoise operation"
        );
        for op in ops {
            sched.register_inflight(op, Instant::now());
        }

        let (_new, lookahead) = sched.assemble();
        assert!(
            lookahead.iter().any(|op| op.kind == OpKind::DecodeUnd),
            "decode lookahead must keep the pipeline full"
        );
        assert!(
            lookahead.iter().all(|op| op.kind != OpKind::DenoiseGen),
            "a second denoise operation must not occupy the decode service slot: {:?}",
            lookahead.iter().map(|op| op.kind).collect::<Vec<_>>()
        );
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
        let req = test_request(1, 4);
        let mut rx = sched.submit_for_test(req);
        sched.admit();

        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.und.logical_pos = 4;
            st.ingest.prompt_cursor = 4;
            st.und.next_token = 10;
        }
        let transition = sched
            .plan_intent(
                id,
                sched.projected_cursor(id).expect("cursor projection"),
                TransitionIntent::DecodeUnd {
                    position: 4,
                    token_id: 10,
                    token_source: TokenSource::Wire,
                    new_blocks: Vec::new(),
                    spec_token_ids: Some(vec![11, 12]),
                    token_count: 1,
                    stop_token_ids: None,
                    stop_terminal: true,
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("speculative decode transition");
        apply_and_resolve(
            &mut sched,
            id,
            transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(13),
                num_accepted_tokens: Some(2),
                ..Default::default()
            },
            vec![11, 12],
        );

        let st = sched.running.get(&id).expect("request still running");
        assert_eq!(st.und.logical_pos, 7);
        assert_eq!(st.und.tokens_emitted, 3);
        assert_eq!(st.replay.generated_ids, vec![11, 12, 13]);
        assert_eq!(st.und.next_token, 13);
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
        let over = test_request(3, 4);
        let mut rx = sched.submit_for_test(over);
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
        let mut op = sched.next_transition(id, 4).expect("prefill transition");
        op.assign_op_id(1);
        sched.register_inflight(op, Instant::now());
        assert!(sched.has_inflight(id));
        // Cancel, then reap: the request must survive while its op is in flight.
        sched.mark_cancelled(id, false);
        sched.reap_cancellations();
        assert!(
            sched.running.contains_key(&id),
            "cancelled request with an in-flight op must not be reaped yet"
        );
        // Once the op drains, the next reap finishes it.
        let _ = sched.pop_inflight(id, Some(1));
        assert!(!sched.has_inflight(id));
        sched.reap_cancellations();
        assert!(
            !sched.running.contains_key(&id),
            "cancelled request must be reaped after its op resolves"
        );
    }

    #[test]
    fn cancellation_defers_cleanup_for_every_transition_class() {
        #[derive(Clone, Copy, Debug)]
        enum Case {
            TextIngest,
            ImageIngest,
            UndDecode,
            GenDenoise,
            GenCommit,
            Feedback,
        }

        for (index, case) in [
            Case::TextIngest,
            Case::ImageIngest,
            Case::UndDecode,
            Case::GenDenoise,
            Case::GenCommit,
            Case::Feedback,
        ]
        .into_iter()
        .enumerate()
        {
            let mut sched = test_scheduler();
            let id = RequestId(index as u64 + 1);
            let mut request = test_request(id.0, 4);
            if matches!(case, Case::ImageIngest) {
                add_context_image(&mut request, 2, false);
            }
            if matches!(case, Case::GenDenoise | Case::GenCommit | Case::Feedback) {
                if matches!(case, Case::Feedback) {
                    request.policy.feedback = Some(uniserve_core::GeneratedImageFeedbackRecipe {
                        commit: uniserve_core::CommitRecipe::CommitGenThenWriteback,
                        writeback: uniserve_core::FeedbackWriteback::DirectKv,
                        next_und_token: uniserve_core::FeedbackNextToken::EndOfImage,
                        logical_positions: 1,
                        physical_kv_tokens: uniserve_core::ImageKvEffect::Exact { tokens: 1 },
                    });
                }
                request_with_generation_behavior(
                    &mut request,
                    uniserve_core::GenerationConstraint::Default,
                );
            }
            compile_resources(&sched, &mut request);
            let mut events = sched.submit_for_test(request);
            sched.admit();

            if let Some(state) = sched.running.get_mut(&id) {
                match case {
                    Case::TextIngest => {}
                    Case::ImageIngest => {
                        state.ingest.prompt_cursor = 2;
                        state.und.logical_pos = 2;
                        state.und.physical_kv_len = 2;
                    }
                    Case::UndDecode => {
                        state.ingest.prompt_cursor = 4;
                        state.und.logical_pos = 4;
                        state.und.physical_kv_len = 4;
                        state.lifecycle.phase = Phase::DecodeUnd;
                    }
                    Case::GenDenoise => {
                        state.ingest.prompt_cursor = 4;
                        state.und.logical_pos = 4;
                        state.und.physical_kv_len = 4;
                        state.image_gen.cond_pos = 4;
                        state.image_gen.image_id = 1;
                        state.lifecycle.phase = Phase::DenoiseGen;
                    }
                    Case::GenCommit => {
                        state.ingest.prompt_cursor = 4;
                        state.und.logical_pos = 4;
                        state.und.physical_kv_len = 4;
                        state.image_gen.cond_pos = 4;
                        state.image_gen.image_id = 1;
                        state.lifecycle.phase = Phase::CommitGen;
                    }
                    Case::Feedback => {
                        state.ingest.prompt_cursor = 4;
                        state.und.logical_pos = 4;
                        state.und.physical_kv_len = 4;
                        state.image_gen.cond_pos = 4;
                        state.image_gen.image_id = 1;
                        state.feedback.locator = Some("locator".to_string());
                        state.lifecycle.phase = Phase::CommitWriteback;
                    }
                }
            }

            let mut transition = sched
                .next_transition(id, 8)
                .unwrap_or_else(|| panic!("{case:?} must produce a transition"));
            transition.assign_op_id(index as u64 + 1);
            sched.register_inflight(transition, Instant::now());
            sched.mark_cancelled(id, false);
            sched.reap_cancellations();
            assert!(
                sched.running.contains_key(&id),
                "{case:?} cancellation released state while its op was in flight"
            );

            let _ = sched.pop_inflight(id, Some(index as u64 + 1));
            sched.reap_cancellations();
            assert!(
                !sched.running.contains_key(&id),
                "{case:?} cancellation did not clean up after result ownership ended"
            );
            let terminal = std::iter::from_fn(|| events.try_recv().ok()).find_map(|event| {
                if let GenEvent::Finished { reason, .. } = event {
                    Some(reason)
                } else {
                    None
                }
            });
            assert_eq!(terminal, Some(FinishReason::Cancelled), "{case:?}");
            assert_eq!(sched.ledger.total_active(), 0, "{case:?} leaked leases");
        }
    }

    // A context-image decode burst commits every returned token; a burst
    // that stopped on <|im_end|> closes the round directly (its KV was already
    // fed by the worker's speculative stop-token forward; no context_round_closing op).
    #[test]
    fn context_decode_burst_commits_tokens_and_closes_round_on_eos() {
        let mut sched = test_scheduler();
        sched.ctrl.eos = vec![99];
        let mut req = test_request(1, 4);
        add_context_image(&mut req, 2, false);
        req.policy.trigger = uniserve_core::TriggerPolicyDescriptor::RoundCloseThenSuffix {
            close_token_ids: vec![99],
            trigger_token_ids: vec![21, 22],
        };
        req.behavior =
            uniserve_core::GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);
        req.max_und_tokens = 64;
        compile_resources(&sched, &mut req);
        let mut rx = sched.submit_for_test(req);
        sched.decode_token_burst = 2;
        sched.admit();
        let id = RequestId(1);
        if let Some(st) = sched.running.get_mut(&id) {
            st.lifecycle.phase = Phase::DecodeUnd;
            st.ingest.prompt_cursor = 4;
            st.ingest.mm_cursor = 1;
            st.und.logical_pos = 10;
            st.und.physical_kv_len = 10;
        }

        let decode_transition = sched.next_transition(id, 2).expect("decode transition");

        // Full burst, no stop: every token commits and resolution applies the
        // full position/KV advance.
        apply_and_resolve(
            &mut sched,
            id,
            decode_transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(6),
                sampled_token_ids: Some(vec![5, 6]),
                ..Default::default()
            },
            Vec::new(),
        );
        {
            let st = sched.running.get(&id).expect("still running");
            assert_eq!(st.und.tokens_emitted, 2);
            assert_eq!(st.und.round_tokens, vec![5, 6]);
            assert_eq!(st.und.next_token, 6);
            assert_eq!(st.und.logical_pos, 12);
            assert_eq!(st.und.physical_kv_len, 12);
        }
        assert_eq!(drain_text_tokens(&mut rx), vec![5, 6]);

        // Burst tail hits eos: the committed tokens emit, the round closes
        // without a context_round_closing hop, and (no image trigger) the request ends.
        let decode_transition = sched.next_transition(id, 2).expect("decode transition");
        apply_and_resolve(
            &mut sched,
            id,
            decode_transition,
            uniserve_worker_wire::SeqResult {
                req_id: id,
                sampled_token_id: Some(99),
                sampled_token_ids: Some(vec![7, 99]),
                ..Default::default()
            },
            Vec::new(),
        );
        assert!(
            !sched.running.contains_key(&id),
            "eos-terminated burst must finish the request"
        );
        assert_eq!(drain_text_tokens(&mut rx), vec![7]);
    }

    // Context ingest follows the profile recipe exactly; capability validation
    // rejects a recipe the worker cannot execute.
    #[test]
    fn context_encode_respects_vae_capability() {
        for (has_vae, expected_first) in [(false, OpKind::VitEncode), (true, OpKind::VaeEncode)] {
            let mut sched = test_scheduler();
            if has_vae {
                sched.caps.supported_ops.push(OpKind::VaeEncode);
            }
            let mut req = test_request(1, 5);
            add_context_image(&mut req, 2, has_vae);
            compile_resources(&sched, &mut req);
            sched.submit_for_test(req);
            sched.admit();
            let id = RequestId(1);

            // Prefill chunks to the image boundary...
            let op = sched.next_transition(id, 64).expect("prefill op");
            assert_eq!(op.kind, OpKind::PrefillUnd);
            assert_eq!(op.pos_range, (0, 2));
            apply_and_resolve(
                &mut sched,
                id,
                op,
                uniserve_worker_wire::SeqResult {
                    req_id: id,
                    sampled_token_id: Some(9),
                    ..Default::default()
                },
                Vec::new(),
            );
            // ...then the encode fires at the in-prompt marker gap.
            let op = sched.next_transition(id, 64).expect("encode op");
            assert_eq!(op.kind, expected_first, "has_vae={has_vae}");
            assert_eq!(op.cond_pos, Some(2));
            assert_eq!(op.image_b64.as_deref(), Some("aGVsbG8="));
        }
    }

    #[test]
    fn encoder_cache_hit_plans_worker_attach_and_no_store_uses_transient_residency() {
        let mut cached = test_scheduler();
        let mut cached_request = test_request(1, 4);
        add_context_image(&mut cached_request, 2, false);
        compile_resources(&cached, &mut cached_request);
        let cache_key = cached_request.resources.encoder_cache_keys[0];
        cached.enc_cache.insert_output(cache_key, 77, 1);
        cached.submit_for_test(cached_request);
        cached.admit();
        let id = RequestId(1);
        if let Some(state) = cached.running.get_mut(&id) {
            state.ingest.prompt_cursor = 2;
            state.und.logical_pos = 2;
            state.und.physical_kv_len = 2;
        }

        let hit = cached.next_transition(id, 64).expect("cached attach");
        assert_eq!(hit.kind, OpKind::VitEncode);
        assert_eq!(hit.image_b64, None);
        assert_eq!(hit.image_in, Some(77));
        assert_eq!(hit.mm_hash, Some(cache_key));
        assert!(matches!(
            hit.delta,
            crate::generation::TransitionDelta::IngestImageStep {
                encoder_cache_key: Some(key),
                cache_hit: true,
                ..
            } if key == cache_key
        ));
        assert!(cached.reserve_transition_resources(&hit));
        assert_eq!(
            cached
                .running
                .get(&id)
                .expect("cached request")
                .ingest
                .acquired_encoder_pins,
            vec![EncoderCachePin {
                key: cache_key,
                handle: 77,
            }]
        );
        cached.finish(id, FinishReason::Cancelled);

        let mut transient = test_scheduler();
        let mut transient_request = test_request(2, 4);
        add_context_image(&mut transient_request, 2, false);
        transient_request.cache.read = false;
        transient_request.cache.write = false;
        compile_resources(&transient, &mut transient_request);
        transient.submit_for_test(transient_request);
        transient.admit();
        let id = RequestId(2);
        if let Some(state) = transient.running.get_mut(&id) {
            state.ingest.prompt_cursor = 2;
            state.und.logical_pos = 2;
            state.und.physical_kv_len = 2;
        }

        let miss = transient.next_transition(id, 64).expect("transient encode");
        assert_eq!(miss.kind, OpKind::VitEncode);
        assert!(miss.image_b64.is_some());
        assert_ne!(miss.mm_hash, Some(cache_key));
        assert!(matches!(
            miss.delta,
            crate::generation::TransitionDelta::IngestImageStep {
                encoder_cache_key: None,
                cache_hit: false,
                ..
            }
        ));
    }

    // generated branch stays resident. Replaying prompt++generated under load is a
    // performance cliff.
    #[test]
    fn gen_branch_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        compile_resources(&sched, &mut req);
        sched.submit_for_test(req);
        sched.admit();
        assert!(sched.running.contains_key(&RequestId(1)));
        assert!(!sched.preemptible(RequestId(1)));
    }

    #[test]
    fn gen_branch_with_generated_output_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        compile_resources(&sched, &mut req);
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        assert!(sched.running.contains_key(&id));
        if let Some(st) = sched.running.get_mut(&id) {
            st.replay.generated_ids.push(11);
        }
        assert!(!sched.preemptible(id));
    }

    #[test]
    fn gen_branch_with_committed_image_is_not_preemptible() {
        let mut sched = test_scheduler();
        let mut req = test_request(1, 4);
        request_with_generation_behavior(&mut req, uniserve_core::GenerationConstraint::Default);
        compile_resources(&sched, &mut req);
        sched.submit_for_test(req);
        sched.admit();
        let id = RequestId(1);
        assert!(sched.running.contains_key(&id));
        if let Some(st) = sched.running.get_mut(&id) {
            st.image_gen.images_done = 1;
        }
        assert!(!sched.preemptible(id));
    }
}
