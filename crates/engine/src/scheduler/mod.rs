//! Scheduler control loop: one owner thread drives the whole loop (single-owner,
//! no locks on engine state) — drain commands, advance each running request's
//! generation lifecycle, admit pending requests against the block budget,
//! assemble a `Batch`, submit it asynchronously through the `Executor`, and
//! resolve completed reports into `GenerationEvent`s and cursor transitions.
//! Batch assembly chooses text prefill, text decode, or media work and may still
//! mix compatible non-text ops; workers preserve one result per submitted op.
//!
//! Scheduling uses budgeted, chunked-prefill admission with exact resident state:
//! - waiting requests live in a [`RequestQueue`] (FCFS deque or priority
//!   ordering) and admission consumes the queue head;
//! - scheduling is bounded by the `max_num_batched_tokens` / `max_num_seqs`
//!   pair with vLLM's clip rule (`min(num_new_tokens, token_budget)`);
//! - worst-case reservation is a per-request admission attribute; requests whose
//!   exact physical state cannot be relocated remain resident and apply queue
//!   backpressure when capacity is exhausted.
//!
//! Worker updates are stateful: a request's static state crosses once
//! as [`NewRequest`]; per-step ops carry only deltas (new block ids, new
//! tokens, and per-step masks).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod admission;
mod batching;
pub(crate) mod bench_trace;
mod control;
mod execution;
pub(crate) mod generation;
pub(crate) mod image_artifact;
mod inflight;
mod kv_budget;
mod logits;
mod output;
pub(crate) mod queue;
mod runtime;
mod stats;
pub(crate) mod stats_report;

pub(crate) use crate::executor::{ControlOp, Executor};
pub(crate) use queue::RequestQueue;
pub use stats::{
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats,
    SchedStats, TimingStats, WorkerStats,
};
pub use stats_report::SchedStatsReporter;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use std::env;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::scheduler::generation::{
    EncoderCachePin, GenerationCursor, GenerationPhase as Phase, NextOp, SchedulerApply,
    SchedulerContext, TransitionIntent, plan,
};

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
pub(crate) const DEFAULT_DENOISE_STEP_BURST: u16 = 1;
/// Default admission backpressure bound: maximum waiting requests buffered
/// before new submits are rejected at enqueue.
pub(crate) const DEFAULT_MAX_NUM_WAITING: usize = 4096;
pub(crate) const MAX_NUM_WAITING: usize = 65_536;
pub(crate) const MAX_NUM_SEQS: usize = 65_536;
const MAX_INFLIGHT_TRANSFERS: usize = 256;

use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EventRx, EventSendError, EventTx, MediaEventTx, event_channel,
};
use crate::kv::{BlockPool, BlockTable, EncoderCacheManager, KvCacheCoordinator};
use crossbeam_channel::Receiver;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{BlockId, CfgParams, ImageIngestStep, encoder_cache_key};
use uniserve_core::{
    FinishReason, GenerationEvent, GenerationRequest, MediaEvent, MediaRequest, PositionLogprobs,
    TokenLogprob,
};
use uniserve_core::{HashAlgo, RequestId};
use uniserve_worker_ipc::{
    AttentionRegime, Batch, BatchPartition, BlockTable as IpcBlockTable, Bounds,
    CachePageAllocation, CloseReason, CompletionReport, Control, DType, DecodeKind,
    DecodePlacement, DimBound, Disposition, ForwardMode, GenAdmission, LatentPlacement,
    MediaAdmission, MediaPlan, MediaProfileId, ModelOutput, NewRequest, OpId, OpStatus, Operation,
    Point, PointRange, ProductKind, ProductPayload, ProductRef, RequestKey, ResourceClass, RouteId,
    RowGeometry, SamplingState, ShapeBound, StorageClass, TimingCounters, UndAdmission, VersionRef,
    WorkerForwardStats, WorkerInfo,
};

use crate::executor::{WorkerExecError, WorkerLossError};
use crate::scheduler::image_artifact::validate_png_artifact;
use inflight::{
    InflightApply, InflightOp, InflightWindow, PendingCompletion, PendingFinish,
    SubmittedPartitionAccounting,
};
use kv_budget::{KvBudget, worker_kv_state};
use output::{OutputSender, RequestOutput};
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchKind {
    Prefill,
    Decode,
    Media,
}

type RouteDomainOperations = Vec<(
    uniserve_worker_ipc::RouteId,
    Vec<(uniserve_worker_ipc::Domain, Vec<Operation>)>,
)>;

/// maximum time the fully-idle scheduler blocks on the command channel
/// before waking to probe worker liveness. Bounds how long a worker that dies
/// while the engine is idle stays undetected (the next request would otherwise
/// be the first to notice). Small enough for prompt death detection, large
/// enough that the idle engine is effectively asleep.
const IDLE_LIVENESS_POLL: Duration = Duration::from_millis(500);
/// Physical submissions in the execution window that may carry prompt work at
/// once. One credit serializes prompt admission behind the previous prompt
/// batch's host-visible resolution, keeping decode continuity — and with it
/// inter-token latency and total throughput — at its strongest; additional
/// credits trade decode continuity for prompt admission latency.
const PREFILL_WINDOW_CREDITS: usize = 1;
const DENOISE_STEP_BURST_ENV: &str = "UNISERVE_DENOISE_STEP_BURST";
/// Diagnostic: give a denoise step a batch of its own instead of letting text
/// rows ride along in the same forward.
const FLOW_EXCLUSIVE_BATCH_ENV: &str = "UNISERVE_FLOW_EXCLUSIVE_BATCH";

fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<GenerationEvent> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(GenerationEvent::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
        sha256: metadata.sha256,
        pixels_png_b64,
    })
}

/// Locate the completion product a given operation produced for `kind`.
fn find_product(
    products: &[ProductPayload],
    op_id: OpId,
    kind: ProductKind,
) -> Option<&ProductPayload> {
    products
        .iter()
        .find(|payload| payload.product.producer_op_id == op_id && payload.product.kind == kind)
}

/// A host-side view of one operation's completion, assembled from its record and
/// the report's decoded output products. It carries the committed tokens and
/// logprob values a token operation samples, the KV product an operation
/// publishes, the materialized image, and the accounting the resolver reads.
struct SequenceView {
    committed_tokens: Vec<u32>,
    sampled_logprob: Option<f32>,
    top_logprobs: Vec<RankedToken>,
    prompt_logprobs: Vec<Vec<RankedToken>>,
    image_png: Option<String>,
    encode_generation: Option<u32>,
    kv_visible_len: u32,
    flow_done: bool,
}

impl SequenceView {
    fn from_report(record: &ModelOutput, products: &[ProductPayload]) -> Self {
        let logprobs = find_product(products, record.op_id, ProductKind::Logprob)
            .and_then(|payload| LogprobBlob::decode(&payload.bytes).ok())
            .unwrap_or_default();
        let image_png = find_product(products, record.op_id, ProductKind::Artifact)
            .and_then(|payload| String::from_utf8(payload.bytes.clone()).ok());
        Self {
            committed_tokens: record.committed_tokens.clone(),
            sampled_logprob: logprobs.sampled_logprob,
            top_logprobs: logprobs.top_logprobs,
            prompt_logprobs: logprobs.prompt_logprobs,
            image_png,
            encode_generation: record.product_generations.first().copied(),
            kv_visible_len: record.logical_lengths.kv_visible_len,
            flow_done: record.finish_flags.eos
                || record.finish_flags.length
                || record.finish_flags.stop,
        }
    }
}

fn token_prefix_versions(
    operation: Option<&Operation>,
    record: &ModelOutput,
    _parent: Option<&VersionRef>,
) -> Vec<VersionRef> {
    if operation.is_none() {
        return Vec::new();
    }
    let count = record.committed_tokens.len();
    if record.status != OpStatus::Ok || count == 0 || record.selected_point as usize != count {
        return Vec::new();
    }
    (1..=count)
        .map(|point_index| VersionRef {
            request_key: record.request_key,
            producer_op_id: record.op_id,
            point: Point::Fixed {
                point_index: point_index as u32,
            },
        })
        .collect()
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
/// priority. Both are budgeted, chunked-prefill schedulers;
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
    /// Waiting-queue backpressure — maximum requests buffered before new
    /// submits are rejected at enqueue.
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

pub(crate) struct ReqState {
    pub req: GenerationRequest,
    pub(crate) finish_token_ids: Vec<u32>,
    /// The cached sequence mappings. Each table owns its physical page
    /// references and therefore has exactly the request's lifetime.
    pub(crate) block_tables: Vec<BlockTable>,
    flow_prefix: Option<FlowPrefixState>,
    /// Stable scheduler-assigned row used by every worker-side request-indexed
    /// state owner for this request epoch. Zero denotes a pending request that
    /// has not entered the running set.
    pub(crate) request_pool_idx: u32,
    /// Scheduler-owned lifecycle generation and latest host-resolved worker version.
    pub(crate) epoch: u64,
    pub(crate) version: u64,
    /// Latest host-resolved producer used to build the next exact fixed parent.
    pub(crate) resolved_producer_op_id: u64,
    /// Latest committed point.
    pub(crate) committed_version: u64,
    pub(crate) committed_producer_op_id: u64,
    /// Last ordered semantic control emitted for this request.
    pub(crate) control_seq: u64,
    /// Monotonic public-event bound carried by the latest semantic commit.
    pub(crate) public_event_limit: u64,
    /// Fixed semantic cutoffs indexed by public text-token count.
    pub(crate) token_cutoffs: BTreeMap<usize, VersionRef>,
    /// Ordered semantic commits held for the frontend decoder's exact-prefix
    /// decision, oldest first. A stop-string request may register bounded
    /// provisional descendants before their predecessors are decoded, so more
    /// than one commit can await a decision at once; each carries the exact
    /// chained parent so the worker applies them in order once acknowledged.
    /// The frontend can retract every descendant beyond a matched stop by an
    /// exact cutoff, dropping the un-acknowledged tail without committing it.
    pub(crate) pending_commits: VecDeque<PendingSemanticCommit>,
    pub(crate) cancel_cutoff: Option<VersionRef>,
    /// Exact worker-local selected-point product for the latest resolved state,
    /// together with the work variant that owns its physical pool. The scheduler
    /// retains this logical ownership until it submits a reachable consumer.
    pub(crate) latest_device_version: Option<ResidentDeviceVersion>,
    pub(crate) context: SchedulerContext,
    output: RequestOutput,
    pub(crate) cursor: GenerationCursor,
    pub queued_at: f64,
    pub(crate) terminal_intent: TerminalIntent,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum TerminalIntent {
    #[default]
    None,
    Cancel,
    Abort,
    StopMatched,
}

impl TerminalIntent {
    pub(crate) const fn is_terminal(self) -> bool {
        !matches!(self, Self::None)
    }
}

struct FlowPrefixState {
    request_pool_idx: u32,
    block_tables: Vec<BlockTable>,
    new_pages: Vec<(u32, Vec<BlockId>)>,
    materialized: bool,
}

struct RetiringSession {
    request_key: RequestKey,
    request_pool_idx: u32,
    _block_tables: Vec<BlockTable>,
    flow_prefix: Option<FlowPrefixState>,
}

#[derive(Clone)]
pub(crate) struct PendingSemanticCommit {
    token_count: Option<usize>,
    expected_parent: VersionRef,
    selected: VersionRef,
    public_event_limit: u64,
}

#[derive(Clone)]
pub(crate) struct ResidentDeviceVersion {
    version: VersionRef,
    token: ProductRef,
    producer: ForwardMode,
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
            && self.cursor.image_gen.images_done < self.req.image.max_images as usize
    }

    fn starts_gen_after_context(&self) -> bool {
        self.req.behavior.start_gen_after_context && self.can_open_gen_branch()
    }

    pub(crate) fn is_replayable_text(&self) -> bool {
        self.cursor.replay.replayability == crate::scheduler::generation::Replayability::Replayable
    }

    pub(crate) fn effective_prompt(&self) -> &[u32] {
        &self.context.prompt_ids
    }

    fn pending_image_step(&self) -> Option<ImageIngestStep> {
        self.context
            .images
            .get(self.cursor.ingest.mm_cursor)?
            .ingest
            .steps
            .get(self.cursor.ingest.pending_image_step)
            .copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MediaCursor {
    prepared: bool,
    denoise_step: u32,
    video_unit: u32,
    audio_done: bool,
    materialized: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediaQuantum {
    Transition,
    Flow { step: u32 },
    Video { start_unit: u32, unit_count: u32 },
    Audio,
    Materialize,
}

fn next_media_quantum(cursor: MediaCursor, plan: MediaPlan) -> Option<MediaQuantum> {
    const VIDEO_UNITS_PER_ROUND: u32 = 4;
    if !cursor.prepared {
        Some(MediaQuantum::Transition)
    } else if cursor.denoise_step < plan.denoise_steps {
        Some(MediaQuantum::Flow {
            step: cursor.denoise_step,
        })
    } else if cursor.video_unit < plan.video_decode_units {
        Some(MediaQuantum::Video {
            start_unit: cursor.video_unit,
            unit_count: VIDEO_UNITS_PER_ROUND.min(plan.video_decode_units - cursor.video_unit),
        })
    } else if !cursor.audio_done {
        Some(MediaQuantum::Audio)
    } else if !cursor.materialized {
        Some(MediaQuantum::Materialize)
    } else {
        None
    }
}

fn advance_media_cursor(mut cursor: MediaCursor, quantum: MediaQuantum) -> MediaCursor {
    match quantum {
        MediaQuantum::Transition => cursor.prepared = true,
        MediaQuantum::Flow { step } => cursor.denoise_step = step.saturating_add(1),
        MediaQuantum::Video {
            start_unit,
            unit_count,
        } => cursor.video_unit = start_unit.saturating_add(unit_count),
        MediaQuantum::Audio => cursor.audio_done = true,
        MediaQuantum::Materialize => cursor.materialized = true,
    }
    cursor
}

fn media_work(quantum: MediaQuantum) -> ForwardMode {
    match quantum {
        MediaQuantum::Transition => ForwardMode::GenTransition,
        MediaQuantum::Flow { .. } => ForwardMode::GenFlow,
        MediaQuantum::Video { .. } | MediaQuantum::Audio => ForwardMode::GenDecode,
        MediaQuantum::Materialize => ForwardMode::Materialize,
    }
}

struct MediaFlowState {
    request: MediaRequest,
    event_tx: MediaEventTx,
    request_pool_idx: u32,
    admission: NewRequest,
    admission_state: MediaAdmissionState,
    committed: MediaCursor,
    projected: MediaCursor,
    fixed_parent: VersionRef,
    projected_parent: VersionRef,
    terminal_intent: MediaTerminalIntent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaAdmissionState {
    Unsubmitted,
    InFlight,
    Registered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MediaTerminalIntent {
    None,
    Cancel,
    Failure(String),
}

impl MediaTerminalIntent {
    const fn is_terminal(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn cancel(&mut self) {
        if matches!(self, Self::None) {
            *self = Self::Cancel;
        }
    }
}

struct RetiringMedia {
    request_key: RequestKey,
    request_pool_idx: u32,
}

struct PendingMedia {
    request: MediaRequest,
    event_tx: MediaEventTx,
}

pub struct Scheduler {
    executor: Box<dyn Executor>,
    info: WorkerInfo,
    /// Resident resource ownership and reservation accounting.
    kv_budget: KvBudget,
    ctrl: ControlTokens,
    config: SchedulerConfig,
    /// Pluggable host-side logits-processor pipeline.
    logits_pipeline: Vec<crate::scheduler::logits::BuiltinLogitsProcessor>,
    /// Active generation and standalone media request state.
    running: HashMap<RequestId, ReqState>,
    running_media: HashMap<RequestId, MediaFlowState>,
    /// Terminal requests retain only their bounded public journal.
    output: OutputSender,
    /// Worker-visible sessions whose close transaction has been submitted but
    /// not yet acknowledged. Their worker page holdings and logical leases remain
    /// owned until the close report establishes the worker-side retirement
    /// ordering point.
    retiring_sessions: HashMap<RequestId, RetiringSession>,
    order: Vec<RequestId>, // stable cross-request iteration order
    pending_media: VecDeque<PendingMedia>,
    retiring_media: HashMap<RequestId, RetiringMedia>,
    prefer_media: bool,
    pending: RequestQueue,
    /// Submitted operations, ordered completions, and per-batch timing state.
    inflight: InflightWindow,
    /// Sequential denoise timesteps to execute per denoise op. The worker runs
    /// the exact same Euler steps and reports the cumulative step cursor.
    denoise_step_burst: u16,
    /// Diagnostic: keep denoise steps out of batches that carry text rows.
    flow_exclusive_batch: bool,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    latent_dtype: Option<DType>,
    /// Ordered semantic controls waiting to cross the worker boundary.
    pending_controls: VecDeque<Control>,
    /// The scheduler-authority identity stamped into every request key.
    authority_id: u64,
    /// Monotonic operation and request epochs.
    next_op_id: u64,
    next_product_generation: u64,
    next_collective_seq: u64,
    next_epoch: u64,
    trace_sink: Option<crate::scheduler::bench_trace::SchedulerTraceSink>,
    pub peak_ops_in_batch: usize,
    pub stats: Arc<SchedStats>,
}

fn now() -> f64 {
    // route through the single shared epoch helper so every
    // component's wall-clock timestamps match. It never panics on the hot loop:
    // a wall clock set before the UNIX epoch (or stepped backward) clamps to 0
    // instead of unwrapping the `Result`.
    uniserve_core::now_unix_secs()
}

fn worker_float_dtype(
    value: Option<uniserve_core::ModelDtype>,
) -> Option<uniserve_worker_ipc::DType> {
    match value {
        Some(uniserve_core::ModelDtype::Float16) => Some(uniserve_worker_ipc::DType::F16),
        Some(uniserve_core::ModelDtype::BFloat16) => Some(uniserve_worker_ipc::DType::BF16),
        Some(uniserve_core::ModelDtype::Float32) => Some(uniserve_worker_ipc::DType::F32),
        None => None,
    }
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

fn close_reason(reason: &FinishReason) -> CloseReason {
    match reason {
        FinishReason::Cancelled | FinishReason::Aborted => CloseReason::Cancelled,
        FinishReason::Error => CloseReason::Error,
        _ => CloseReason::Completed,
    }
}

/// Tokens that stop a device-relay successor before host semantic resolution.
///
/// Terminal stop/EOS tokens end the request. A direct Gen trigger instead ends
/// the current Und continuation window: the sampled trigger remains
/// host-visible and opens the image branch, while an already registered text
/// successor resolves as a predicate no-op.
fn finish_token_ids(request: &GenerationRequest, eos: &[u32]) -> Vec<u32> {
    let mut finish_token_ids = request.stop_token_ids.clone();
    if !request.sampling.ignore_eos {
        finish_token_ids.extend(eos.iter().copied());
    }
    if request.behavior.gen_output
        && let Some(trigger) = request.policy.trigger.direct_token()
    {
        finish_token_ids.push(trigger);
    }
    finish_token_ids.sort_unstable();
    finish_token_ids.dedup();
    finish_token_ids
}

fn ranked_logprobs(entries: Vec<RankedToken>) -> Vec<TokenLogprob> {
    entries
        .into_iter()
        .map(|entry| TokenLogprob {
            token_id: entry.token_id,
            logprob: entry.logprob,
            rank: entry.rank,
        })
        .collect()
}

fn batch_kind(operation_variant: ForwardMode) -> BatchKind {
    match operation_variant {
        ForwardMode::TokenExtend | ForwardMode::EncodeVision | ForwardMode::EncodeLatent => {
            BatchKind::Prefill
        }
        ForwardMode::TokenDecode | ForwardMode::TokenVerify => BatchKind::Decode,
        ForwardMode::Draft
        | ForwardMode::GenFlow
        | ForwardMode::GenDecode
        | ForwardMode::GenTransition
        | ForwardMode::Materialize
        | ForwardMode::TransferProduct
        | ForwardMode::TransferKvPublish
        | ForwardMode::TransferKvInstall => BatchKind::Media,
    }
}

fn completion_priority(operation_variant: ForwardMode) -> u8 {
    match operation_variant {
        ForwardMode::GenFlow | ForwardMode::Materialize | ForwardMode::TransferKvInstall => 0,
        _ => 1,
    }
}

const OUTPUT_JOURNAL_CAPACITY: usize = EVENT_BUFFER_CAPACITY;
const OUTPUT_TERMINAL_RESERVE: usize = 2;

#[derive(Clone, Copy)]
struct KvLengths {
    input: u32,
    visible: u32,
}

fn transition_kv_lengths(delta: &TransitionIntent) -> Option<KvLengths> {
    let (prefix, input) = match delta {
        TransitionIntent::IngestText {
            start,
            end,
            physical_start,
            ..
        } => (*physical_start, end.saturating_sub(*start)),
        TransitionIntent::IngestImageState {
            physical_start,
            physical_kv_tokens,
            ..
        }
        | TransitionIntent::FeedbackState {
            physical_start,
            physical_kv_tokens,
            ..
        } => {
            let input = match physical_kv_tokens {
                uniserve_core::ImageKvEffect::Exact { tokens } => *tokens,
                uniserve_core::ImageKvEffect::Bounded { max_tokens } => *max_tokens,
                uniserve_core::ImageKvEffect::WorkerDefined => return None,
            };
            (*physical_start, input)
        }
        TransitionIntent::DecodeUnd {
            physical_position, ..
        }
        | TransitionIntent::CloseKv {
            physical_position, ..
        } => (*physical_position, 1),
        TransitionIntent::PublishKv {
            physical_kv_len, ..
        }
        | TransitionIntent::TransitionGen {
            physical_kv_len, ..
        }
        | TransitionIntent::DenoiseGen {
            physical_kv_len, ..
        } => (*physical_kv_len, 0),
        TransitionIntent::EncodeImageStep { .. }
        | TransitionIntent::CommitGen { .. }
        | TransitionIntent::EncodeFeedbackStep { .. } => return None,
    };
    Some(KvLengths {
        input,
        visible: prefix,
    })
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

/// Physical transformer-token work a planned op contributes to one scheduler
/// step. Denoise executes every latent token once per CFG branch at every
/// timestep, so its cost multiplies the compiled latent geometry rather than the
/// scalar per-step token cost.
fn planned_op_token_cost(transition: &NextOp) -> usize {
    if transition.operation_variant != ForwardMode::GenFlow {
        return transition.token_cost;
    }
    let latent_tokens = usize::try_from(transition.resources.latent_units)
        .unwrap_or(usize::MAX)
        .max(1);
    let cfg_branches = transition.resources.cfg_branches.max(1);
    let timesteps = transition.token_cost.max(1);
    latent_tokens
        .saturating_mul(cfg_branches)
        .saturating_mul(timesteps)
}

fn tensorized_mixed_runner_work(variant: ForwardMode) -> bool {
    matches!(variant, ForwardMode::TokenDecode | ForwardMode::GenFlow)
}

fn flow_matches_mixed_bucket(
    operation: &Operation,
    forward_rows: &HashMap<(RequestKey, OpId), Vec<RowGeometry>>,
    latent_placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    bucket: &uniserve_worker_ipc::GraphBucket,
) -> bool {
    if operation.work != ForwardMode::GenFlow {
        return false;
    }
    let identity = (operation.request_key, operation.op_id);
    latent_placements.get(&identity).is_some_and(|placement| {
        placement.height == bucket.height
            && placement.width == bucket.width
            && forward_rows
                .get(&identity)
                .is_some_and(|rows| rows.len() >= bucket.cfg_branches as usize)
    })
}

fn extract_mixed_group(
    candidates: &mut [(uniserve_worker_ipc::Domain, Vec<Operation>)],
    buckets: &[uniserve_worker_ipc::GraphBucket],
    forward_rows: &HashMap<(RequestKey, OpId), Vec<RowGeometry>>,
    latent_placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
) -> Option<Vec<(uniserve_worker_ipc::Domain, Vec<Operation>)>> {
    let decode_index = candidates
        .iter()
        .position(|(domain, _)| *domain == uniserve_worker_ipc::Domain::Decode)?;
    let flow_index = candidates
        .iter()
        .position(|(domain, _)| *domain == uniserve_worker_ipc::Domain::Flow)?;
    let decode_available = candidates[decode_index].1.len();
    let flow_operations = &candidates[flow_index].1;
    let bucket = buckets
        .iter()
        .filter(|bucket| bucket.decode_rows as usize <= decode_available)
        .filter(|bucket| {
            flow_operations
                .iter()
                .filter(|operation| {
                    flow_matches_mixed_bucket(operation, forward_rows, latent_placements, bucket)
                })
                .count()
                >= bucket.flow_rows as usize
        })
        .max_by_key(|bucket| {
            (
                u64::from(bucket.decode_rows) + u64::from(bucket.flow_rows),
                bucket.decode_rows,
                bucket.flow_rows,
            )
        })?
        .clone();

    let decode = candidates[decode_index]
        .1
        .drain(..bucket.decode_rows as usize)
        .collect::<Vec<_>>();
    let mut flow = Vec::with_capacity(bucket.flow_rows as usize);
    let mut remaining_flow = Vec::with_capacity(candidates[flow_index].1.len());
    for operation in std::mem::take(&mut candidates[flow_index].1) {
        if flow.len() < bucket.flow_rows as usize
            && flow_matches_mixed_bucket(&operation, forward_rows, latent_placements, &bucket)
        {
            flow.push(operation);
        } else {
            remaining_flow.push(operation);
        }
    }
    candidates[flow_index].1 = remaining_flow;
    let mut selected = vec![
        (uniserve_worker_ipc::Domain::Decode, decode),
        (uniserve_worker_ipc::Domain::Flow, flow),
    ];
    if flow_index < decode_index {
        selected.swap(0, 1);
    }
    Some(selected)
}

fn partition_attention(operations: &[Operation]) -> AttentionRegime {
    let regimes = operations
        .iter()
        .map(|operation| match operation.work {
            ForwardMode::TokenExtend
            | ForwardMode::TokenDecode
            | ForwardMode::TokenVerify
            | ForwardMode::Draft => AttentionRegime::Causal,
            ForwardMode::GenFlow => AttentionRegime::Hybrid,
            ForwardMode::GenTransition
            | ForwardMode::GenDecode
            | ForwardMode::EncodeVision
            | ForwardMode::EncodeLatent
            | ForwardMode::TransferProduct
            | ForwardMode::TransferKvPublish
            | ForwardMode::TransferKvInstall
            | ForwardMode::Materialize => AttentionRegime::None,
        })
        .collect::<HashSet<_>>();
    if regimes.len() == 1 {
        *regimes.iter().next().unwrap_or(&AttentionRegime::None)
    } else {
        AttentionRegime::Hybrid
    }
}

fn transition_output_bound(transition: &NextOp) -> usize {
    match transition.operation_variant {
        ForwardMode::TokenVerify => transition
            .validation
            .expected_text_tokens
            .map_or(4, |range| {
                usize::try_from(range.max)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(2)
                    .saturating_add(2)
            }),
        ForwardMode::TokenExtend | ForwardMode::TokenDecode => 4,
        ForwardMode::GenFlow => transition.token_cost.saturating_add(2),
        ForwardMode::GenDecode => 2,
        ForwardMode::Materialize => 3,
        _ => 2,
    }
}

fn operation_trace(operation: &Operation, apply: &SchedulerApply) -> serde_json::Value {
    let parent_kind = match operation.parent.point {
        Point::Fixed { .. } => "fixed",
        Point::Device { .. } => "device",
    };
    json!({
        "work": operation.work.as_str(),
        "domain": format!("{:?}", operation.domain),
        "parent_kind": parent_kind,
        "predicated": operation.predicate.is_some(),
        "inputs": operation.inputs.len(),
        "outputs": operation.outputs.len(),
        "max_tokens": operation.bounds.max_tokens,
        "max_kv_pages": operation.bounds.max_kv_pages,
        "output_event_bound": apply.output_event_bound,
    })
}

fn denoise_step_burst_from_env() -> u16 {
    env::var(DENOISE_STEP_BURST_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u16>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DENOISE_STEP_BURST)
}

/// Build the IPC [`CfgParams`] for a denoise op.
///
/// The exact text/image CFG branch set is derived by the worker-side CFG plan.
/// The scheduler only carries the per-path branch bound needed by generic
/// denoise accounting.
fn cfg_params(image: &uniserve_core::ImageParams, branch_count: u8) -> CfgParams {
    CfgParams {
        branch_count: branch_count.max(1),
        text_scale: image.cfg_text_scale,
        img_scale: image.cfg_img_scale,
        renorm_type: image.cfg_renorm_type.to_string(),
        renorm_min: image.cfg_renorm_min,
        interval: image.cfg_interval,
    }
}

fn cfg_branch_count(image: &uniserve_core::ImageParams) -> u8 {
    image.cfg_branch_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{
        ContextSegment, GenerationBehaviorDescriptor, GenerationConstraint,
        GenerationPolicyDescriptor, GenerationResourceBounds, ImageParams, SamplingParams,
        UndVisibility,
    };

    fn request(id: u64, tokens: usize) -> GenerationRequest {
        let policy = GenerationPolicyDescriptor::default();
        let constraint = GenerationConstraint::UndOnly;
        GenerationRequest {
            request_id: RequestId(id),
            context: vec![ContextSegment::UndTokens {
                token_ids: vec![0; tokens],
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: tokens,
                max_kv_tokens: tokens + 32,
                ..GenerationResourceBounds::default()
            },
        }
    }

    #[test]
    fn direct_gen_trigger_stops_a_registered_text_successor() {
        let mut req = request(1, 8);
        req.constraint = GenerationConstraint::Default;
        req.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 4_242 };
        req.behavior = GenerationBehaviorDescriptor::resolve(req.constraint, &req.policy);

        let stops = finish_token_ids(&req, &[151_643, 151_645]);

        assert_eq!(stops, vec![4_242, 151_643, 151_645]);
    }
}
