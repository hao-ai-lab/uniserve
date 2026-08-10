//! Scheduler control loop: one owner thread drives the whole loop (single-owner,
//! no locks on engine state) — drain commands, advance each running request's
//! generation lifecycle, admit pending requests against the block and scratch budget,
//! assemble a `ForwardBatch`, submit it asynchronously through the `Executor`,
//! and resolve completed `ForwardResult`s into `GenEvent`s and cursor transitions.
//! `ForwardBatch` assembly is lane-aware for text prefill/decode and may still
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
//! The worker contract is a stateful diff: a request's static state crosses once
//! as [`NewRequestData`]; per-step ops carry only deltas (new block ids, new
//! tokens, and per-step masks).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::cpu_continuation::{CpuContinuationPool, CpuMasks, CpuTask, CpuTaskKey};
use crate::generation::{
    CursorApplyError, CursorProjection, EncoderCachePin, GenerationCursor,
    GenerationPhase as Phase, GenerationPlanner, PlannedTransition, SchedulerApply,
    SchedulerContext, TransitionDelta, TransitionIntent, TransitionValidationError,
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
pub const DEFAULT_DENOISE_STEP_BURST: u16 = 1;
/// Default admission backpressure bound: maximum waiting requests buffered
/// before new submits are rejected at enqueue.
pub const DEFAULT_MAX_NUM_WAITING: usize = 4096;
pub const MAX_NUM_WAITING: usize = 65_536;
pub const MAX_NUM_SEQS: usize = 65_536;
const MAX_INFLIGHT_TRANSFERS: usize = 256;

/// General loop counters that do not belong to a more specific group.
#[derive(Default)]
pub struct GeneralStats {
    pub peak_ops: AtomicUsize,
    pub steps: AtomicU64,
    pub running: AtomicUsize,
    pub pending: AtomicUsize,
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
    pub timing: TimingStats,
    pub worker: WorkerStats,
}

use crossbeam_channel::Receiver;
use uniserve_core::GenerationRequest;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{BlockId, CfgParams, ImageIngestStep, encoder_cache_key};
use uniserve_core::{HashAlgo, RequestId};
use uniserve_engine_api::{
    Command, EventTx, FinishReason, GenEvent, GenerationSubmission, PublicCommit, PublicModality,
    SemanticRoot,
};
use uniserve_kv::{BlockManager, EncoderCacheManager};
use uniserve_worker_wire::{
    Admission, AttentionRegime, Batch, BatchPartition, CloseReason, CompletionRecord,
    CompletionReport, Control, Disposition, ExecutionCapability, GenAdmission, KvAdmission,
    KvBranchPlacement, KvPlacement, LatentPlacement, OpId, OpStatus, Operation, Point, ProductKind,
    ProductPayload, ProductRef, RequestKey, ResourceClass, SamplingState, UndAdmission, VersionRef,
    WorkVariant, WorkerCapabilities, WorkerForwardStats,
};

use crate::image_artifact::validate_png_artifact;
use crate::queue::{FcfsRequestQueue, PriorityRequestQueue, RequestQueue};
use serde_json::json;
use uniserve_executor::{ControlOp, Executor, WorkerExecError, WorkerLossError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssemblyLane {
    Prefill,
    Decode,
    Other,
}

type RouteDomainOperations = Vec<(
    uniserve_worker_wire::RouteId,
    Vec<(uniserve_worker_wire::Domain, Vec<Operation>)>,
)>;

const SCHEDULER_WAIT_SLICE: Duration = Duration::from_millis(1);
/// maximum time the fully-idle scheduler blocks on the command channel
/// before waking to probe worker liveness. Bounds how long a worker that dies
/// while the engine is idle stays undetected (the next request would otherwise
/// be the first to notice). Small enough for prompt death detection, large
/// enough that the idle engine is effectively asleep.
const IDLE_LIVENESS_POLL: Duration = Duration::from_millis(500);
const DENOISE_STEP_BURST_ENV: &str = "UNISERVE_DENOISE_STEP_BURST";
/// Diagnostic: give a denoise step a batch of its own instead of letting text
/// rows ride along in the same forward.
const FLOW_EXCLUSIVE_BATCH_ENV: &str = "UNISERVE_FLOW_EXCLUSIVE_BATCH";

fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<GenEvent> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(GenEvent::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
        sha256: metadata.sha256,
        pixels_png_b64,
        public_commit: None,
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
    fn from_report(record: &CompletionRecord, products: &[ProductPayload]) -> Self {
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
    record: &CompletionRecord,
    parent: Option<&VersionRef>,
) -> Vec<VersionRef> {
    let Some(operation) = operation else {
        return Vec::new();
    };
    let Some(Point::Fixed {
        semantic_digest: parent_semantic,
        ..
    }) = parent.map(|version| &version.point)
    else {
        return Vec::new();
    };
    let count = record.committed_tokens.len();
    if record.status != OpStatus::Ok || count == 0 || record.selected_point as usize != count {
        return Vec::new();
    }
    (1..=count)
        .map(|point_index| {
            let digest = if point_index == count {
                record.semantic_digest.clone()
            } else {
                let mut prefix = record.clone();
                prefix.selected_point = point_index as u32;
                prefix.logical_lengths.token_len = record
                    .logical_lengths
                    .token_len
                    .saturating_sub((count - point_index) as u32);
                prefix.logical_lengths.kv_visible_len = record
                    .logical_lengths
                    .kv_visible_len
                    .saturating_sub((count - point_index) as u32);
                prefix.token_span.len = point_index as u32;
                prefix.committed_tokens.truncate(point_index);
                prefix.finish_flags = Default::default();
                prefix.compute_semantic_digest(parent_semantic, &operation.plan_digest)
            };
            VersionRef {
                request_key: record.request_key,
                producer_op_id: record.op_id,
                point: Point::Fixed {
                    point_index: point_index as u32,
                    semantic_digest: digest,
                },
            }
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
    /// Admission backpressure — maximum waiting requests buffered before new
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

pub struct ReqState {
    pub req: GenerationRequest,
    /// Stable scheduler-assigned row used by every worker-side request-indexed
    /// state owner for this request epoch. Zero denotes a pending request that
    /// has not entered the running set.
    pub(crate) request_pool_idx: u32,
    /// Scheduler-owned lifecycle generation and latest host-resolved worker version.
    pub(crate) epoch: u64,
    pub(crate) version: u64,
    pub(crate) admission_digest: Option<String>,
    /// Latest host-resolved lineage used to build the next exact fixed parent.
    pub(crate) resolved_semantic: String,
    pub(crate) resolved_producer_op_id: u64,
    /// Latest semantically committed lineage.
    pub(crate) committed_version: u64,
    pub(crate) committed_semantic: String,
    pub(crate) committed_producer_op_id: u64,
    /// Last ordered semantic control emitted for this request.
    pub(crate) control_seq: u64,
    /// Number of public events accepted by the request's output journal.
    pub(crate) public_event_seq: u64,
    /// Monotonic public-event bound carried by the latest semantic commit.
    pub(crate) public_event_limit: u64,
    pub(crate) public_token_seq: usize,
    /// Latest frontend-decoder token prefix accepted for semantic commit.
    pub(crate) semantic_token_seq: usize,
    /// Ordered public events waiting for immediate-consumer channel capacity.
    pub(crate) output_journal: VecDeque<GenEvent>,
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
    pub(crate) event_tx: EventTx,
    pub(crate) cursor: GenerationCursor,
    pub queued_at: f64,
    pub(crate) cancelled: bool,
    /// The cancel was a server-side abort, not a client cancel.
    pub(crate) aborted: bool,
    /// The frontend decoder selected an exact terminal stop prefix.
    pub(crate) stop_matched: bool,
    /// At most one CPU continuation may run for this lineage.
    pub(crate) cpu_pending: Option<CpuTaskKey>,
    pub(crate) cpu_masks: Option<CpuMasks>,
    pub(crate) cpu_generation: u64,
    /// This request's lifecycle trace.
    pub(crate) trace: crate::trace::RequestTrace,
}

struct RequestSlotPool {
    free: Vec<u32>,
    live: Vec<bool>,
}

struct LatentPagePool {
    page_units: u32,
    free: Vec<u32>,
    owners: Vec<Option<RequestId>>,
    allocations: HashMap<RequestId, Vec<u32>>,
}

struct KvScratchPagePool {
    page_size: u32,
    free: Vec<BlockId>,
    allocations: HashMap<RequestId, Vec<BlockId>>,
    pending_zero: HashMap<RequestId, Vec<BlockId>>,
}

impl KvScratchPagePool {
    fn new(page_offset: u32, capacity_tokens: u64, page_size: u32) -> Self {
        let page_size = page_size.max(1);
        let page_count = capacity_tokens.div_ceil(u64::from(page_size));
        let end = u64::from(page_offset)
            .checked_add(page_count)
            .and_then(|value| u32::try_from(value).ok())
            .expect("KV scratch page range exceeds u32");
        Self {
            page_size,
            free: (page_offset..end).rev().map(BlockId).collect(),
            allocations: HashMap::new(),
            pending_zero: HashMap::new(),
        }
    }

    fn pages_needed(&self, tokens: u64) -> Option<usize> {
        usize::try_from(tokens.div_ceil(u64::from(self.page_size))).ok()
    }

    fn can_reserve(&self, request_id: RequestId, tokens: u64) -> bool {
        let Some(needed) = self.pages_needed(tokens) else {
            return false;
        };
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        needed.saturating_sub(held) <= self.free.len()
    }

    fn reserve(&mut self, request_id: RequestId, tokens: u64) -> bool {
        if !self.can_reserve(request_id, tokens) {
            return false;
        }
        let needed = self.pages_needed(tokens).unwrap_or_default();
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        let mut pages = Vec::with_capacity(needed.saturating_sub(held));
        for _ in held..needed {
            pages.push(self.free.pop().expect("KV scratch free-page invariant"));
        }
        self.pending_zero
            .entry(request_id)
            .or_default()
            .extend(pages.iter().copied());
        self.allocations
            .entry(request_id)
            .or_default()
            .extend(pages);
        true
    }

    fn pages_for(&self, request_id: RequestId) -> &[BlockId] {
        self.allocations
            .get(&request_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn take_pages_to_zero(&mut self, request_id: RequestId) -> Vec<BlockId> {
        self.pending_zero.remove(&request_id).unwrap_or_default()
    }

    fn release(&mut self, request_id: RequestId) {
        self.pending_zero.remove(&request_id);
        if let Some(pages) = self.allocations.remove(&request_id) {
            self.free.extend(pages.into_iter().rev());
        }
    }
}

impl LatentPagePool {
    fn new(num_pages: u32, page_units: u32) -> Self {
        Self {
            page_units,
            free: (1..num_pages).rev().collect(),
            owners: vec![None; num_pages as usize],
            allocations: HashMap::new(),
        }
    }

    fn pages_needed(&self, units: u64) -> Option<usize> {
        if units == 0 {
            return Some(0);
        }
        let page_units = u64::from(self.page_units);
        if page_units == 0 {
            return None;
        }
        usize::try_from(units.div_ceil(page_units)).ok()
    }

    fn can_reserve(&self, request_id: RequestId, units: u64) -> bool {
        let Some(needed) = self.pages_needed(units) else {
            return false;
        };
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        needed.saturating_sub(held) <= self.free.len()
    }

    fn reserve(&mut self, request_id: RequestId, units: u64) -> bool {
        if !self.can_reserve(request_id, units) {
            return false;
        }
        let needed = self.pages_needed(units).unwrap_or_default();
        let held = self.allocations.get(&request_id).map_or(0, Vec::len);
        let mut pages = Vec::with_capacity(needed.saturating_sub(held));
        for _ in held..needed {
            let page = self.free.pop().expect("latent free-page invariant");
            self.owners[page as usize] = Some(request_id);
            pages.push(page);
        }
        self.allocations
            .entry(request_id)
            .or_default()
            .extend(pages);
        true
    }

    fn pages_for(&self, request_id: RequestId) -> &[u32] {
        self.allocations
            .get(&request_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    fn release(&mut self, request_id: RequestId) {
        if let Some(pages) = self.allocations.remove(&request_id) {
            for page in pages.into_iter().rev() {
                self.owners[page as usize] = None;
                self.free.push(page);
            }
        }
    }

    fn used_pages(&self) -> usize {
        self.owners
            .iter()
            .skip(1)
            .filter(|owner| owner.is_some())
            .count()
    }
}

impl RequestSlotPool {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.clamp(1, u32::MAX as usize);
        Self {
            free: (1..=capacity as u32).rev().collect(),
            live: vec![false; capacity + 1],
        }
    }

    fn capacity(&self) -> usize {
        self.live.len().saturating_sub(1)
    }

    fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    fn acquire(&mut self) -> Option<u32> {
        let index = self.free.pop()?;
        self.live[index as usize] = true;
        Some(index)
    }

    fn release(&mut self, index: u32) -> Result<(), &'static str> {
        let Some(live) = self.live.get_mut(index as usize) else {
            return Err("request-pool index is outside scheduler capacity");
        };
        if index == 0 || !*live {
            return Err("request-pool index is not live");
        }
        *live = false;
        self.free.push(index);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RetiringSession {
    request_key: RequestKey,
    request_pool_idx: u32,
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
    producer: WorkVariant,
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
        &self.context.prompt_ids
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
    caps: WorkerCapabilities,
    bm: BlockManager,
    ctrl: ControlTokens,
    config: SchedulerConfig,
    /// Total allocatable blocks (the empty pool's free count) — the structural
    /// admission/rejection bound.
    usable_blocks: usize,
    /// Automatic prefix caching: toggle, hash config, lookup, and reuse.
    prefix_cache: crate::prefix_cache::PrefixCacheCoordinator,
    /// Pluggable host-side logits-processor pipeline.
    logits_pipeline: Vec<Arc<dyn crate::logits::LogitsProcessor>>,
    custom_logits_processors: usize,
    /// Encoder-output cache (hashed, LRU, budgeted).
    enc_cache: EncoderCacheManager,
    /// Encoder-cache entries reserved by admitted image requests.
    reserved_encoder_entries: usize,
    request_slots: RequestSlotPool,
    latent_pages: LatentPagePool,
    kv_scratch_pages: KvScratchPagePool,
    running: HashMap<RequestId, ReqState>,
    /// Terminal requests retain only their bounded public journal.
    completed_outputs: HashMap<RequestId, RetiredOutput>,
    /// Worker-visible sessions whose close transaction has been submitted but
    /// not yet acknowledged. Their worker page holdings and logical leases remain
    /// owned until the close report establishes the worker-side retirement
    /// ordering point.
    retiring_sessions: HashMap<RequestId, RetiringSession>,
    order: Vec<RequestId>, // stable iteration order
    pending: Box<dyn RequestQueue>,
    cpu_continuations: CpuContinuationPool,
    cpu_task_timeout: Duration,
    /// Request-local continuation deadlines, bounded by the CPU task pool. This
    /// keeps the scheduler hot path independent of the number of active requests.
    cpu_deadlines: HashMap<CpuTaskKey, Instant>,
    reserved_blocks: usize,
    transfer_capacity: usize,
    inflight_transfers: usize,
    step_id: u64,
    /// Ordered submitted-but-unresolved operations for each request. The front
    /// owns the committed lease; later entries are bounded projected successors
    /// whose versions and cursor deltas are resolved strictly in order.
    inflight_ops: HashMap<RequestId, VecDeque<InflightOp>>,
    /// Worker-ready completions waiting for their request-local predecessors.
    /// Independent batch partitions may publish a successor before the host
    /// observes its predecessor, while semantic cursor application remains
    /// strictly request-ordered.
    pending_completions: HashMap<RequestId, BTreeMap<u64, PendingCompletion>>,
    /// Terminal outcomes whose projected successors still own worker-visible
    /// request state. The scheduler retains all leases until those successors
    /// drain, then publishes the terminal event exactly once.
    pending_finishes: HashMap<RequestId, PendingFinish>,
    /// Finite set of resident prompts coalesced behind active decode work.
    /// Membership is frozen until every member's prompt work drains, so later
    /// arrivals cannot indefinitely delay the cohort's first decode service.
    prompt_cohort: Option<HashSet<RequestId>>,
    /// Sequential denoise timesteps to execute per denoise op. The worker runs
    /// the exact same Euler steps and reports the cumulative step cursor.
    denoise_step_burst: u16,
    /// Diagnostic: keep denoise steps out of batches that carry text rows.
    flow_exclusive_batch: bool,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    /// Explainable policy decisions and per-op latency history.
    decisions: crate::policy::DecisionLog,
    latency: crate::policy::LatencyHistory,
    planner: GenerationPlanner,
    /// Submit timestamp per in-flight batch (for batch round-trip traces).
    batch_started: HashMap<u64, Instant>,
    /// Partition ids still expected for each submitted batch.
    batch_partitions: HashMap<u64, HashSet<u32>>,
    /// Ordered semantic controls waiting to cross the worker boundary.
    pending_controls: VecDeque<Control>,
    /// Controls attached to an in-flight batch, retained until its ack report.
    control_batches: HashMap<u64, Vec<Control>>,
    /// The scheduler-authority identity stamped into every request key.
    authority_id: u64,
    /// Monotonic op ids and archived lifecycle traces.
    next_op_id: u64,
    next_completion_seq: u64,
    next_product_generation: u64,
    next_collective_seq: u64,
    next_epoch: u64,
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
    pub completed_traces: usize,
    pub last_worker_exec_us: u64,
    pub queue_wait_count: u64,
    pub queue_wait_us_total: u64,
    pub queue_wait_us_max: u64,
    pub fatal: bool,
    pub supported_work: Vec<WorkVariant>,
    pub op_latency_us: Vec<(String, u64)>,
}

/// Aggregate delay across one lifecycle phase span over completed operations.
#[derive(Debug, Clone)]
pub struct PhaseSpanDelay {
    pub from: crate::trace::LifecyclePhase,
    pub to: crate::trace::LifecyclePhase,
    pub count: u64,
    pub sum_us: u64,
    pub max_us: u64,
}

/// The bounded operation-window and lifecycle observability surface.
#[derive(Debug, Clone)]
pub struct ResourceWindowMetrics {
    pub max_operations: usize,
    pub active_operations: usize,
    pub max_unresolved_window: u32,
    pub peak_ops_in_batch: usize,
    pub phase_delays: Vec<PhaseSpanDelay>,
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
fn canonical_continuation_stop_token_ids(request: &GenerationRequest, eos: &[u32]) -> Vec<u32> {
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

fn ranked_logprobs(entries: Vec<RankedToken>) -> Vec<uniserve_engine_api::TokenLogprob> {
    entries
        .into_iter()
        .map(|entry| uniserve_engine_api::TokenLogprob {
            token_id: entry.token_id,
            logprob: entry.logprob,
            rank: entry.rank,
        })
        .collect()
}

fn transition_validation_error_str(error: &TransitionValidationError) -> &'static str {
    match error {
        TransitionValidationError::OperationFailed => "operation_failed",
        TransitionValidationError::OpIdMismatch { .. } => "op_id_mismatch",
        TransitionValidationError::SessionMismatch { .. } => "session_mismatch",
        TransitionValidationError::VersionMismatch { .. } => "version_mismatch",
        TransitionValidationError::DenoiseStepMismatch { .. } => "denoise_step_mismatch",
        TransitionValidationError::MissingEncoderHandle => "missing_encoder_handle",
        TransitionValidationError::MissingLatentGeneration => "missing_latent_generation",
        TransitionValidationError::MissingImageArtifact => "missing_image_artifact",
        TransitionValidationError::InvalidImageArtifact => "invalid_image_artifact",
        TransitionValidationError::MissingKvPublication => "missing_kv_publication",
        TransitionValidationError::MissingImageDimensions => "missing_image_dimensions",
        TransitionValidationError::ImageDimensionsMismatch { .. } => "image_dimensions_mismatch",
        TransitionValidationError::ImageKvMismatch { .. } => "image_kv_mismatch",
        TransitionValidationError::UnexpectedSampledToken { .. } => "unexpected_sampled_token",
        TransitionValidationError::MissingSampledToken { .. } => "missing_sampled_token",
        TransitionValidationError::AcceptedDraftCountExceeded { .. } => {
            "accepted_draft_count_exceeded"
        }
        TransitionValidationError::VerifiedDraftPrefixMismatch => "verified_draft_prefix_mismatch",
        TransitionValidationError::TextTokenCountMismatch { .. } => "text_token_count_mismatch",
        TransitionValidationError::SampledTokenOutsideAllowedSet { .. } => {
            "sampled_token_outside_allowed_set"
        }
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
        CursorApplyError::MissingLatentProduct => "missing_latent_product",
        CursorApplyError::DuplicateOperation { .. } => "duplicate_operation",
    }
}

fn phase_str(phase: Phase) -> &'static str {
    match phase {
        Phase::Encode => "encode",
        Phase::IngestState => "ingest_state",
        Phase::Prefill => "prefill",
        Phase::DecodeUnd => "decode_und",
        Phase::CloseKv => "close_kv",
        Phase::PublishKv => "publish_kv",
        Phase::TransitionGen => "transition_gen",
        Phase::DenoiseGen => "denoise_gen",
        Phase::CommitGen => "commit_gen",
        Phase::FeedbackEncode => "feedback_encode",
        Phase::FeedbackState => "feedback_state",
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

fn assembly_lane(operation_variant: WorkVariant) -> AssemblyLane {
    match operation_variant {
        WorkVariant::TokenExtend | WorkVariant::EncodeVision | WorkVariant::EncodeLatent => {
            AssemblyLane::Prefill
        }
        WorkVariant::TokenDecode | WorkVariant::TokenVerify => AssemblyLane::Decode,
        WorkVariant::Draft
        | WorkVariant::GenFlow
        | WorkVariant::GenTransition
        | WorkVariant::Materialize
        | WorkVariant::TransferProduct
        | WorkVariant::TransferKvPublish
        | WorkVariant::TransferKvInstall => AssemblyLane::Other,
    }
}

fn completion_priority(operation_variant: WorkVariant) -> u8 {
    match operation_variant {
        WorkVariant::GenFlow | WorkVariant::Materialize | WorkVariant::TransferKvInstall => 0,
        _ => 1,
    }
}

fn policy_str(policy: SchedulingPolicy) -> &'static str {
    match policy {
        SchedulingPolicy::Fcfs => "fcfs",
        SchedulingPolicy::Priority => "priority",
    }
}

/// One submitted-but-unresolved operation tracked in a request's ordered queue.
/// Its apply record is paired by `(operation.request_key, operation.op_id)`.
struct InflightOp {
    operation: Operation,
    apply: SchedulerApply,
    /// Submit timestamp, for the op's host round-trip latency history.
    started: Instant,
}

struct PendingCompletion {
    record: CompletionRecord,
    products: Arc<[ProductPayload]>,
    arrival_seq: u64,
}

struct PendingFinish {
    reason: FinishReason,
    stop_reason: Option<String>,
}

struct RetiredOutput {
    event_tx: EventTx,
    journal: VecDeque<GenEvent>,
}

const OUTPUT_JOURNAL_CAPACITY: usize = uniserve_engine_api::EVENT_BUFFER_CAPACITY;
const OUTPUT_TERMINAL_RESERVE: usize = 2;

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
        let cpu_waker = executor.command_waker();
        let max_batch_ops = caps.max_batch_operations as usize;
        let transfer_capacity = (caps.pipeline_depth as usize)
            .saturating_mul(max_batch_ops)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);
        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);
        config.max_num_seqs = config.max_num_seqs.clamp(1, MAX_NUM_SEQS);
        if max_batch_ops > 0 {
            config.max_batch = config.max_batch.min(max_batch_ops.max(1));
        }
        // build from the worker's reported KV-cache groups (hybrid layouts);
        // Empty means one full-attention group.
        let bm = if caps.groups.is_empty() {
            BlockManager::new(caps.num_blocks as usize, caps.block_size as usize)
        } else {
            let specs: Vec<(uniserve_core::KvGroupKind, u32, u32)> = caps
                .groups
                .iter()
                .map(|g| (g.kind, g.block_offset, g.num_blocks))
                .collect();
            BlockManager::with_groups(caps.num_blocks as usize, caps.block_size as usize, &specs)
        };
        let usable_blocks = bm.request_page_capacity();
        let stats = Arc::new(SchedStats::default());
        stats
            .kv_cache
            .num_blocks
            .store(usable_blocks, Ordering::Relaxed);
        let caps_encoder_budget = caps.encoder_cache_budget as usize;
        let request_slots = RequestSlotPool::new(config.max_num_seqs);
        let latent_pages = LatentPagePool::new(caps.num_latent_pages, caps.latent_page_units);
        let kv_scratch_pages = KvScratchPagePool::new(
            caps.num_blocks,
            caps.scratch_capacity_tokens,
            caps.block_size,
        );
        let denoise_step_burst = denoise_step_burst_from_env();
        let flow_exclusive_batch = env::var(FLOW_EXCLUSIVE_BATCH_ENV)
            .is_ok_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE"));
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
                    "denoise_step_burst": denoise_step_burst,
                },
                "caps": {
                    "block_size": caps.block_size,
                    "num_blocks": caps.num_blocks,
                    "supported_work": &caps.supported_work,
                    "max_batch_operations": caps.max_batch_operations,
                    "pipeline_depth": caps.pipeline_depth,
                    "latent_page_units": caps.latent_page_units,
                    "num_latent_pages": caps.num_latent_pages,
                    "latent_width": caps.latent_width,
                    "latent_dtype": &caps.latent_dtype,
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
            logits_pipeline: crate::logits::default_pipeline(),
            custom_logits_processors: 0,
            enc_cache: EncoderCacheManager::new(caps_encoder_budget),
            reserved_encoder_entries: 0,
            request_slots,
            latent_pages,
            kv_scratch_pages,
            running: HashMap::new(),
            completed_outputs: HashMap::new(),
            retiring_sessions: HashMap::new(),
            order: Vec::new(),
            cpu_continuations: CpuContinuationPool::new(cpu_waker),
            cpu_task_timeout: Duration::from_secs(30),
            cpu_deadlines: HashMap::new(),
            reserved_blocks: 0,
            transfer_capacity,
            inflight_transfers: 0,
            step_id: 0,
            inflight_ops: HashMap::new(),
            pending_completions: HashMap::new(),
            pending_finishes: HashMap::new(),
            prompt_cohort: None,
            denoise_step_burst,
            flow_exclusive_batch,
            fatal: false,
            decisions: crate::policy::DecisionLog::default(),
            latency: crate::policy::LatencyHistory::new(),
            planner: GenerationPlanner::new(),
            batch_started: HashMap::new(),
            batch_partitions: HashMap::new(),
            pending_controls: VecDeque::new(),
            control_batches: HashMap::new(),
            authority_id: 1,
            next_op_id: 1,
            next_completion_seq: 1,
            next_product_generation: 1,
            next_collective_seq: 1,
            next_epoch: 1,
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
        let declaration = p.declaration();
        assert!(
            declaration.snapshotable
                && declaration.deterministic
                && declaration.max_output_tokens > 0
                && declaration.max_output_tokens < usize::MAX
                && declaration.max_outstanding_tasks == 1,
            "custom logits processors must declare deterministic snapshot state, bounded output, and one outstanding task per request"
        );
        self.logits_pipeline.push(Arc::from(p));
        self.custom_logits_processors = self.custom_logits_processors.saturating_add(1);
        self
    }
    /// Set the request-local deadline for one deterministic CPU continuation.
    pub fn with_cpu_task_timeout(mut self, timeout: Duration) -> Self {
        assert!(
            !timeout.is_zero(),
            "CPU continuation timeout must be positive"
        );
        self.cpu_task_timeout = timeout;
        self
    }
    /// Configure the per-step token budget.
    pub fn set_token_budget(&mut self, tokens: usize) {
        self.config.max_num_batched_tokens = tokens.max(1);
    }
    pub fn set_long_prefill_threshold(&mut self, n: usize) {
        self.config.long_prefill_threshold = n.max(1);
    }
    pub fn set_max_num_seqs(&mut self, n: usize) {
        self.config.max_num_seqs = n.clamp(1, self.request_slots.capacity());
    }
    /// Cap waiting and terminal-output-retained request state.
    pub fn set_max_num_waiting(&mut self, n: usize) {
        self.config.max_num_waiting = n.clamp(1, MAX_NUM_WAITING);
    }
    pub fn caps(&self) -> &WorkerCapabilities {
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
        let free_blocks = self.bm.free_request_pages();
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
            in_flight: self.executor.in_flight(),
            free_blocks: self.bm.free_request_pages(),
            total_blocks: self.usable_blocks,
            reserved_blocks: self.reserved_blocks,
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

    /// The operation-window bounds, current occupancy, observed peak, and
    /// aggregate lifecycle-phase delays over completed operations.
    pub fn resource_window_metrics(&self) -> ResourceWindowMetrics {
        use crate::trace::LifecyclePhase as P;
        let spans = [
            (P::Submitted, P::CompletionObserved),
            (P::CompletionObserved, P::SemanticallyCommitted),
            (P::SemanticallyCommitted, P::PubliclyCommitted),
            (P::Planned, P::PhysicallyReclaimed),
        ];
        let mut phase_delays: Vec<PhaseSpanDelay> = spans
            .iter()
            .map(|(from, to)| PhaseSpanDelay {
                from: *from,
                to: *to,
                count: 0,
                sum_us: 0,
                max_us: 0,
            })
            .collect();
        for trace in &self.completed_traces {
            for op in trace.operations() {
                for (index, (from, to)) in spans.iter().enumerate() {
                    if let Some(delay) = op.span_us(*from, *to) {
                        let entry = &mut phase_delays[index];
                        entry.count += 1;
                        entry.sum_us += delay;
                        entry.max_us = entry.max_us.max(delay);
                    }
                }
            }
        }
        ResourceWindowMetrics {
            max_operations: self
                .executor
                .pipeline_depth()
                .saturating_mul(self.config.max_batch),
            active_operations: self.inflight_ops.values().map(VecDeque::len).sum(),
            max_unresolved_window: self.caps.max_unresolved_window,
            peak_ops_in_batch: self.peak_ops_in_batch,
            phase_delays,
        }
    }

    /// A health snapshot the engine can expose: queue +
    /// resource pressure + policy/latency + backend caps + liveness.
    pub fn health_snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            running: self.running.len(),
            pending: self.pending.len(),
            in_flight: self.executor.in_flight(),
            free_blocks: self.bm.free_request_pages(),
            total_blocks: self.usable_blocks,
            reserved_blocks: self.reserved_blocks,
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
            supported_work: self.caps.supported_work.clone(),
            op_latency_us: self.latency.as_pairs(),
        }
    }

    /// The owner thread: block on the command channel when fully idle, else
    /// spin the schedule-ahead loop. Returns `true` if the engine died
    /// (executor/worker failure) rather than shutting down gracefully.
    pub fn run(mut self, rx: Receiver<Command>) -> bool {
        // Event-driven executors park on {result, command, worker-death}; the
        // command ingress wakes that wait. Polling executors use the crossbeam
        // selection path below.
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
                // is detected promptly. CPU continuations require a shorter
                // slice so ready results are incorporated promptly.
                let wait = if !self.cpu_tasks_pending() {
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

    /// Event-driven park over result, command, worker-death, CPU-continuation,
    /// and output-capacity wakes. The timeout is a liveness backstop.
    fn park_event_driven(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        let timeout = if self.cpu_tasks_pending() {
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
                let _ = st.event_tx.send(GenEvent::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.context.prompt_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            ids
        };
        let _ = queued;
        let running: Vec<RequestId> = self.running.keys().copied().collect();
        for id in running {
            self.finish(id, FinishReason::Aborted);
        }
    }

    /// Apply one command; returns true on shutdown.
    fn handle_command(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Submit(submission) => self.enqueue(*submission),
            Command::Cancel {
                request_id,
                output_token_count,
            } => self.mark_cancelled(request_id, false, output_token_count),
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => self.acknowledge_semantic(request_id, output_token_count),
            Command::StopAt {
                request_id,
                output_token_count,
            } => self.mark_stopped(request_id, output_token_count),
            Command::Abort(id) => self.mark_cancelled(id, true, None),
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

    fn mark_cancelled(&mut self, id: RequestId, abort: bool, output_token_count: Option<usize>) {
        if let Some(st) = self.running.get_mut(&id) {
            if let Some(output_token_count) = output_token_count {
                let Some(cutoff) = st.token_cutoffs.get(&output_token_count).cloned() else {
                    self.finish_after_inflight(id, FinishReason::Error, None);
                    return;
                };
                st.cancel_cutoff = Some(cutoff);
            }
            st.cancelled = true;
            st.aborted = abort;
            st.stop_matched = false;
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
            });
        }
    }

    fn acknowledge_semantic(&mut self, id: RequestId, output_token_count: usize) {
        let Some(current) = self.running.get(&id).map(|state| state.semantic_token_seq) else {
            return;
        };
        if current == output_token_count {
            return;
        }
        if output_token_count < current {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        }

        // Release every held commit whose decoded prefix the frontend has now
        // accepted, oldest first, so the worker applies them in exact chained
        // parent order. A held commit with no public token of its own shares the
        // count of the commit ahead of it and releases with it.
        let mut ready: Vec<PendingSemanticCommit> = Vec::new();
        if let Some(state) = self.running.get_mut(&id) {
            state.semantic_token_seq = output_token_count;
            if let Some((&acknowledged_cutoff, _)) =
                state.token_cutoffs.range(..=output_token_count).next_back()
            {
                state
                    .token_cutoffs
                    .retain(|token_count, _| *token_count >= acknowledged_cutoff);
            }
            while state.pending_commits.front().is_some_and(|pending| {
                pending
                    .token_count
                    .is_some_and(|tc| tc <= output_token_count)
            }) {
                ready.push(state.pending_commits.pop_front().expect("front present"));
            }
        }
        if ready.is_empty() {
            return;
        }
        for pending in ready {
            self.queue_commit(
                id,
                pending.expected_parent,
                pending.selected,
                pending.public_event_limit,
            );
        }
        self.finish_pending_if_idle(id);
    }

    fn mark_stopped(&mut self, id: RequestId, output_token_count: usize) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let Some(cutoff) = state.token_cutoffs.get(&output_token_count).cloned() else {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        };
        // Every provisional descendant beyond the matched stop is retracted: its
        // held commit is dropped un-applied and the ordered close cutoff
        // dominates it. Prefixes acknowledged before the match already committed.
        state.pending_commits.clear();
        state.cancel_cutoff = Some(cutoff);
        state.cancelled = true;
        state.aborted = false;
        state.stop_matched = true;
    }

    /// Accept only controls declared in the worker capability handshake.
    fn control_allowed(&self, op: &ControlOp) -> bool {
        self.caps.supported_controls.contains(&op.request_kind())
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

    fn trace_record(&mut self, record: serde_json::Value) {
        if let Some(sink) = self.trace_sink.as_mut() {
            sink.record(&record);
        }
    }

    fn trace_enabled(&self) -> bool {
        self.trace_sink.is_some()
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
        if let Some(capability) = self.missing_required_capability(&req) {
            let _ = event_tx.send(GenEvent::Rejected {
                message: format!(
                    "generation request requires worker capability `{capability}`, but the worker does not support it"
                ),
            });
            return;
        }
        if let Err(error) = req.validate_resources(&self.caps.generation_runtime_capabilities()) {
            let _ = event_tx.send(GenEvent::Rejected {
                message: format!("invalid generation resource declaration: {error}"),
            });
            return;
        }
        // Admission backpressure sheds load instead of letting the waiting
        // queue grow without bound under overload —
        // an unbounded burst would otherwise OOM the process and take down every
        // in-flight request. Reject the new submit with a typed event.
        let waiting = self.pending.len() + self.completed_outputs.len();
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
        // Multimodal requests reserve their configured bounded KV envelope at
        // admission so excess concurrency queues instead of exhausting KV.
        let reserve_worstcase = !context.images.is_empty() || req.behavior.gen_output;
        // A request with staged images usually encodes them before prefill.
        // Context-image requests prefill the text before each image position,
        // then encode the image into that marker gap.
        let phase0 = Phase::Prefill;
        // A lifecycle trace keyed by the canonical request key.
        let trace = crate::trace::RequestTrace::new(
            RequestKey::new(self.authority_id, req.request_id, self.next_epoch),
            uniserve_core::TraceId(req.request_id.0),
        );
        let st = ReqState {
            request_pool_idx: 0,
            epoch: self.next_epoch,
            version: 0,
            admission_digest: None,
            resolved_semantic: String::new(),
            resolved_producer_op_id: 0,
            committed_version: 0,
            committed_semantic: String::new(),
            committed_producer_op_id: 0,
            control_seq: 0,
            public_event_seq: 0,
            public_event_limit: 0,
            public_token_seq: 0,
            semantic_token_seq: 0,
            output_journal: VecDeque::new(),
            token_cutoffs: BTreeMap::new(),
            pending_commits: VecDeque::new(),
            cancel_cutoff: None,
            latest_device_version: None,
            cursor: GenerationCursor::new(phase0, worst, reserve_worstcase),
            context,
            event_tx,
            queued_at: now(),
            cancelled: false,
            aborted: false,
            stop_matched: false,
            cpu_pending: None,
            cpu_masks: None,
            cpu_generation: 0,
            trace,
            req,
        };
        self.next_epoch = self.next_epoch.saturating_add(1);
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
            u64::from(st.und.physical_kv_len),
            st.context.negative_prompt_ids.len() as u64,
            u64::from(cfg_branch_count(&st.req.image)),
            u64::from(self.caps.block_size),
        )
    }

    fn kv_branch_placements_for(
        &mut self,
        request_id: RequestId,
        request_key: RequestKey,
        op_id: OpId,
    ) -> Result<Vec<KvBranchPlacement>, &'static str> {
        let Some(state) = self.running.get(&request_id) else {
            return Err("generation request lost its runtime state");
        };
        let branch_count = usize::from(cfg_branch_count(&state.req.image).max(1));
        let block_size = u64::from(self.caps.block_size.max(1));
        let branch_tokens = self
            .num_vae(&state.req.image)
            .saturating_add(u64::from(self.caps.commit_marker_tokens));
        let conditioning_tokens = u64::from(state.und.physical_kv_len);
        let negative_tokens = state.context.negative_prompt_ids.len() as u64;
        let branch_pages = |prefix_tokens: u64| {
            usize::try_from(
                branch_tokens
                    .saturating_add(prefix_tokens)
                    .div_ceil(block_size),
            )
            .map_err(|_| "generation scratch page count exceeds usize")
        };
        let primary_pages = branch_pages(conditioning_tokens)?;
        let auxiliary_pages = branch_pages(conditioning_tokens.max(negative_tokens))?;
        let pages = self.kv_scratch_pages.pages_for(request_id).to_vec();
        let expected = primary_pages
            .saturating_add(auxiliary_pages.saturating_mul(branch_count.saturating_sub(1)));
        if pages.len() != expected {
            return Err("generation scratch allocation disagrees with its branch geometry");
        }
        let pages_to_zero = self
            .kv_scratch_pages
            .take_pages_to_zero(request_id)
            .into_iter()
            .collect::<HashSet<_>>();
        let mut placements = Vec::with_capacity(branch_count);
        let mut offset = 0usize;
        for branch in 0..branch_count {
            let width = if branch == 0 {
                primary_pages
            } else {
                auxiliary_pages
            };
            let block_table = pages[offset..offset + width].to_vec();
            let zero = block_table
                .iter()
                .copied()
                .filter(|page| pages_to_zero.contains(page))
                .collect();
            placements.push(KvBranchPlacement {
                request_key,
                op_id,
                branch_index: u32::try_from(branch + 1)
                    .map_err(|_| "generation branch index exceeds u32")?,
                group_id: 0,
                block_table,
                pages_to_zero: zero,
            });
            offset += width;
        }
        Ok(placements)
    }

    fn cap_max_vae_grid_tokens(&self) -> usize {
        if self.caps.max_vae_grid_tokens > 0 {
            self.caps.max_vae_grid_tokens as usize
        } else {
            self.caps.latent_capacity_units().min(usize::MAX as u64) as usize
        }
    }

    fn cap_commit_marker_tokens(&self) -> usize {
        self.caps.commit_marker_tokens.max(1) as usize
    }

    fn missing_required_capability(&self, request: &GenerationRequest) -> Option<&'static str> {
        let context_steps = request.context.iter().flat_map(|segment| match segment {
            uniserve_core::ContextSegment::Image { ingest, .. } => ingest.steps.clone(),
            uniserve_core::ContextSegment::UndTokens { .. } => Vec::new(),
        });
        let needs = request
            .behavior
            .capability_needs(&request.policy, context_steps);
        self.caps
            .generation_runtime_capabilities()
            .covers(&needs)
            .err()
    }

    fn worker_tracks_image_latent(&self) -> bool {
        self.caps.latent_page_units > 0
            && self.caps.num_latent_pages > 1
            && self
                .caps
                .resource_classes
                .contains(&ResourceClass::ImageLatent)
    }

    fn worker_image_latent_used(&self) -> u64 {
        (self.latent_pages.used_pages() as u64)
            .saturating_mul(u64::from(self.caps.latent_page_units))
    }

    fn worker_image_latent_units_for(&self, st: &ReqState) -> u64 {
        let downsample = (self.caps.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    fn can_schedule_denoise(&self, id: RequestId) -> bool {
        let requested_units = self
            .running
            .get(&id)
            .map(|st| self.worker_image_latent_units_for(st).max(1))
            .unwrap_or(0);
        let worker_capacity_ok = !self.worker_tracks_image_latent()
            || self.latent_pages.can_reserve(id, requested_units);
        let host_scratch_ok = self.running.get(&id).is_some_and(|st| {
            st.resources.host_scratch_tokens > 0
                || self
                    .kv_scratch_pages
                    .can_reserve(id, self.denoise_host_scratch_tokens(st))
        });
        worker_capacity_ok && host_scratch_ok
    }

    fn reserve_transition_resources(&mut self, transition: &mut PlannedTransition) -> bool {
        let id = transition.request_id;
        let resources = &transition.resources;
        if resources.latent_units > 0
            && self.worker_tracks_image_latent()
            && !self.latent_pages.can_reserve(id, resources.latent_units)
        {
            return false;
        }
        let uses_transfer = transition.bounds.max_transfer_bytes > 0;
        if uses_transfer && self.inflight_transfers >= self.transfer_capacity {
            return false;
        }
        let needs_host_scratch = self
            .running
            .get(&id)
            .is_some_and(|st| st.resources.host_scratch_tokens == 0)
            && resources.host_scratch_tokens > 0;
        if needs_host_scratch
            && !self
                .kv_scratch_pages
                .reserve(id, resources.host_scratch_tokens)
        {
            return false;
        }
        if resources.latent_units > 0
            && self.worker_tracks_image_latent()
            && !self.latent_pages.reserve(id, resources.latent_units)
        {
            if needs_host_scratch {
                self.kv_scratch_pages.release(id);
            }
            return false;
        }
        if uses_transfer {
            self.inflight_transfers += 1;
        }
        transition.reserved_us = uniserve_core::now_monotonic_us();
        if let Some(st) = self.running.get_mut(&id) {
            st.resources.host_scratch_tokens = st
                .resources
                .host_scratch_tokens
                .max(resources.host_scratch_tokens);
        }
        true
    }

    fn release_transition_resources(&mut self, id: RequestId, apply: &SchedulerApply) {
        for class in &apply.release_on_apply {
            match class {
                uniserve_worker_wire::ResourceClass::ImageLatent => {
                    self.latent_pages.release(id);
                }
                uniserve_worker_wire::ResourceClass::Scratch => {
                    self.kv_scratch_pages.release(id);
                    if let Some(st) = self.running.get_mut(&id) {
                        st.resources.host_scratch_tokens = 0;
                    }
                }
                _ => {}
            }
        }
    }

    /// KV tokens a decode op must have room for: the sampled token plus any
    /// speculative draft the worker verifies alongside it.
    fn decode_capacity_target(&self, pos: usize, spec_len: usize) -> usize {
        pos.saturating_add(1).saturating_add(spec_len)
    }

    fn cpu_tasks_pending(&self) -> bool {
        !self.cpu_deadlines.is_empty()
    }

    /// One loop iteration of the schedule-ahead loop. Returns true if any work
    /// was submitted or any result resolved.
    pub fn step(&mut self) -> bool {
        let progressed = self.step_nonblocking();
        if progressed || self.executor.in_flight() == 0 {
            return progressed;
        }
        match self.executor.next_result() {
            Ok(result) => {
                self.apply_result(result);
                self.step_nonblocking();
            }
            Err(error) => {
                self.on_executor_error(error);
                self.stats
                    .general
                    .in_flight
                    .store(self.executor.in_flight(), Ordering::Relaxed);
                self.publish_cache_stats();
            }
        }
        true
    }

    /// Nonblocking schedule-ahead tick used by the owner-thread reactor. It
    /// drains ready results, reaps cancellations, and fills available executor
    /// slots, but leaves any blocking result wait to `run`.
    fn step_nonblocking(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.step").entered();
        let mut progressed = self.flush_output_journals();
        progressed |= self.drain_cpu_continuations();
        progressed |= self.expire_cpu_continuations();
        // 1. Resolve one completed batch. Refilling immediately after one
        // completion preserves an occupied execution slot when multiple
        // responses become ready together at pipeline depth greater than one.
        // The owner loop returns here without parking while progress is being
        // made, so subsequent ready completions are handled on successive turns.
        progressed |= self.poll_one_result();

        // 2. reap cancellations before assembling.
        self.reap_cancellations();

        // 3. Admission owns request/resource residency and progresses even
        // while every execution slot is occupied. This lets the next batch see
        // the complete resident cohort instead of admitting only when a slot
        // happens to open.
        self.admit();
        progressed |= self.start_cpu_continuations();

        // 4. submit as many batches as pipeline capacity allows.
        while self.executor.can_submit() {
            self.admit();
            progressed |= self.start_cpu_continuations();
            let (new_reqs, ops) = self.assemble();
            if ops.is_empty() {
                break;
            }
            let controls = self.pending_controls.drain(..).collect();
            self.submit_batch(new_reqs, ops, controls);
            progressed = true;
        }
        while self.executor.can_submit() && !self.pending_controls.is_empty() {
            let controls = self.pending_controls.drain(..).collect();
            self.submit_batch(Vec::new(), Vec::new(), controls);
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
            .kv_cache
            .free_blocks
            .store(self.bm.free_request_pages(), Ordering::Relaxed);
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
    }

    /// Resolve at most one ready result so assembly can refill the freed slot
    /// before another completion is consumed.
    fn poll_one_result(&mut self) -> bool {
        let _span = tracing::trace_span!("scheduler.poll_one_result").entered();
        match self.executor.poll() {
            Ok(Some(result)) => {
                self.apply_result(result);
                true
            }
            Ok(None) => false,
            Err(error) => {
                self.on_executor_error(error);
                true
            }
        }
    }

    /// Whether any request currently has a prefill op in flight. Prompt work
    /// coalesces behind it: while one prefill batch runs, newly arrived
    /// prompts wait (decode fills the slot) and merge into the next prefill
    /// batch, so bursts cost one sweep instead of one sweep per arrival.
    fn any_prefill_inflight(&self) -> bool {
        self.inflight_ops
            .values()
            .flatten()
            .any(|op| op.operation.work.variant() == WorkVariant::TokenExtend)
    }

    fn any_denoise_inflight(&self) -> bool {
        self.inflight_ops
            .values()
            .flatten()
            .any(|op| op.operation.work.variant() == WorkVariant::GenFlow)
    }

    fn has_inflight(&self, id: RequestId) -> bool {
        self.inflight_ops
            .get(&id)
            .is_some_and(|queue| !queue.is_empty())
    }

    fn inflight_len(&self, id: RequestId) -> usize {
        self.inflight_ops.get(&id).map_or(0, VecDeque::len)
    }

    fn projected_cursor(&self, id: RequestId) -> Option<CursorProjection> {
        let st = self.running.get(&id)?;
        let applies = self
            .inflight_ops
            .get(&id)
            .into_iter()
            .flatten()
            .map(|op| &op.apply);
        Some(st.cursor.project(applies))
    }

    fn projected_version(&self, id: RequestId) -> Option<u64> {
        let state = self.running.get(&id)?;
        Some(if self.inflight_len(id) > 0 {
            1
        } else {
            state.version
        })
    }

    fn fixed_version(&self, id: RequestId) -> Option<VersionRef> {
        let state = self.running.get(&id)?;
        state.admission_digest.as_ref()?;
        Some(VersionRef {
            request_key: RequestKey::new(self.authority_id, id, state.epoch),
            producer_op_id: OpId(state.resolved_producer_op_id),
            point: Point::Fixed {
                point_index: state.version as u32,
                semantic_digest: state.resolved_semantic.clone(),
            },
        })
    }

    fn public_limit_for(&self, id: RequestId, apply: &SchedulerApply) -> u64 {
        self.running.get(&id).map_or(0, |state| {
            state.public_event_limit.max(
                state
                    .public_event_seq
                    .saturating_add(apply.output_event_bound as u64),
            )
        })
    }

    fn queue_commit(
        &mut self,
        id: RequestId,
        expected_parent: VersionRef,
        selected: VersionRef,
        public_event_limit: u64,
    ) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        state.control_seq = state.control_seq.saturating_add(1);
        state.committed_version = match &selected.point {
            Point::Fixed { point_index, .. } => u64::from(*point_index),
            Point::Device { .. } => state.committed_version,
        };
        state.committed_semantic = match &selected.point {
            Point::Fixed {
                semantic_digest, ..
            } => semantic_digest.clone(),
            Point::Device { .. } => state.committed_semantic.clone(),
        };
        state.committed_producer_op_id = selected.producer_op_id.0;
        state.public_event_limit = public_event_limit;
        state.trace.stamp_existing(
            selected.producer_op_id,
            crate::trace::LifecyclePhase::SemanticallyCommitted,
            uniserve_core::now_monotonic_us(),
        );
        self.pending_controls.push_back(Control::Commit {
            request_key: selected.request_key,
            control_seq: state.control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition: Disposition::Publish,
        });
    }

    fn acknowledge_controls(&mut self, controls: &[Control]) {
        for control in controls {
            if let Control::Close { request_key, .. } = control {
                let id = request_key.session_id;
                let Some(retiring) = self.retiring_sessions.get(&id).copied() else {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring session"
                    );
                    self.fatal = true;
                    continue;
                };
                if retiring.request_key != *request_key {
                    tracing::error!(
                        request_id = id.0,
                        epoch = request_key.epoch,
                        "close acknowledgement does not match a retiring session"
                    );
                    self.fatal = true;
                    continue;
                }
                match self.executor.control_wait(ControlOp::DropSession(id), None) {
                    Ok(_) => {
                        self.retiring_sessions.remove(&id);
                        self.bm.release(id);
                        self.latent_pages.release(id);
                        self.kv_scratch_pages.release(id);
                        if let Err(error) = self.request_slots.release(retiring.request_pool_idx) {
                            tracing::error!(
                                request_id = id.0,
                                request_pool_idx = retiring.request_pool_idx,
                                error,
                                "failed to release scheduler request slot"
                            );
                            self.fatal = true;
                        }
                    }
                    Err(error) => {
                        tracing::error!(
                            request_id = id.0,
                            %error,
                            "worker session retirement did not complete"
                        );
                        self.fatal = true;
                    }
                }
            }
        }
    }

    /// Whether a request may keep an additional operation in flight rooted on a
    /// predecessor's not-yet-observed selected point.
    ///
    /// Such a successor roots on a device version reference (`Point::Device`).
    /// The predecessor's tagged token product is the successor predicate.
    /// Terminal tokens and direct image triggers clear its continuation bit, so
    /// the worker resolves an already registered successor as a semantic no-op.
    /// The host can then apply the sampled token's terminal or branch transition
    /// without admitting an extra text token into that lineage.
    fn can_queue_decode_successor(&self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let Some(queue) = self.inflight_ops.get(&id).filter(|queue| !queue.is_empty()) else {
            return false;
        };
        if queue.len() >= self.executor.pipeline_depth().max(1)
            || queue.len() >= self.caps.max_unresolved_window as usize
            || state.cancelled
            || self.pending_finishes.contains_key(&id)
            || self.custom_logits_processors > 0
            || !matches!(state.lifecycle.phase, Phase::Prefill | Phase::DecodeUnd)
            || (state.lifecycle.phase == Phase::Prefill && state.starts_gen_after_context())
            || !Self::device_token_relay_eligible(state)
        {
            return false;
        }
        let Some(predecessor) = queue.back() else {
            return false;
        };
        if !self.executor.device_products_reachable(
            predecessor.operation.work.variant(),
            WorkVariant::TokenDecode,
        ) {
            return false;
        }
        if matches!(
            state.req.policy.trigger,
            uniserve_core::TriggerPolicyDescriptor::RoundCloseThenSuffix { .. }
        ) {
            return false;
        }
        // Only plain decode/extend predecessors may carry a device-relay
        // successor; any other variant is not a single-token advance.
        if queue.iter().any(|op| {
            !matches!(
                op.operation.work.variant(),
                WorkVariant::TokenExtend | WorkVariant::TokenDecode
            )
        }) {
            return false;
        }
        let Some(projected) = self.projected_cursor(id) else {
            return false;
        };
        projected.prompt_cursor as usize >= state.effective_prompt().len()
            && state.ingest.mm_cursor >= state.context.images.len()
            && state.und.tokens_emitted.saturating_add(queue.len()) < state.req.max_und_tokens
    }

    fn device_token_relay_eligible(state: &ReqState) -> bool {
        // A successor may consume the parent's device-selected point before host
        // observation whenever its own sampling state is device-representable
        // from registered coordinates and device products alone. Greedy and
        // stochastic selection (temperature, top-k, top-p, min-p, typical),
        // penalties folded from a device-resident committed count base plus
        // per-operation deltas, requested logprobs, the minimum-token floor and
        // force-finish flag (staged at the successor's exact projected point),
        // static allowed-token, logit-bias, and single-token bad-word masks,
        // positional forced tokens, and device finish predicates (EOS and
        // stop-token ids) all qualify. Stop strings also qualify: the request
        // samples device-continuously and registers bounded provisional
        // descendants, and a matched stop retracts every descendant beyond the
        // exact accepted prefix through the ordered close cutoff. Only
        // multi-token bad-word automata keep the request host-paced, because
        // their next mask depends on the not-yet-observed suffix.
        let sampling = &state.req.sampling;
        (!state.req.behavior.gen_output || state.req.policy.trigger.direct_token().is_some())
            && sampling.bad_words_ids.iter().all(|word| word.len() == 1)
    }

    /// Whether the latest host-resolved token may remain the exact device input
    /// to the next decode. CPU stop and EOS decisions are complete before this
    /// check; a continuing request therefore names the same sampled token.
    fn can_reuse_resolved_token_product(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            state.resources.worker_registered
                && self.custom_logits_processors == 0
                && state.und.tokens_emitted > 0
                && state
                    .latest_device_version
                    .as_ref()
                    .is_some_and(|resident| {
                        self.executor
                            .device_products_reachable(resident.producer, WorkVariant::TokenDecode)
                    })
                && state.is_replayable_text()
                && state.lifecycle.phase == Phase::DecodeUnd
                && !state.ingest.round_closing
                && (!state.req.behavior.gen_output
                    || state.req.policy.trigger.direct_token().is_some())
        })
    }

    fn can_schedule_next(&self, id: RequestId) -> bool {
        !self.pending_finishes.contains_key(&id)
            && self.output_window_ready(id)
            && self.cpu_continuation_ready(id)
            && self
                .running
                .get(&id)
                .is_some_and(|state| self.pending_commit_horizon_open(state))
            && (!self.has_inflight(id) || self.can_queue_decode_successor(id))
    }

    /// Whether the request may register another operation without exceeding its
    /// bounded provisional horizon. A request with no held commits is always
    /// open; a stop-string request that has decoded prefixes awaiting the
    /// frontend decision may run ahead by at most the unresolved-window depth,
    /// after which it waits for an acknowledgement so the horizon stays finite.
    fn pending_commit_horizon_open(&self, state: &ReqState) -> bool {
        let horizon = self.caps.max_unresolved_window.max(1) as usize;
        state.pending_commits.len() < horizon
    }

    /// A request needs the asynchronous CPU-continuation future only for custom
    /// logit processors, whose semantics are arbitrary host code. The built-in
    /// minimum-token floor, bad-word, and allowed-token masks are cheap,
    /// position- or set-derived host computations that run inline in
    /// [`Self::token_masks`], so they never suspend the request behind a future.
    fn cpu_continuation_required(&self, _state: &ReqState) -> bool {
        self.custom_logits_processors > 0
    }

    fn cpu_continuation_ready(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| {
            !self.cpu_continuation_required(state)
                || (state.cpu_pending.is_none() && state.cpu_masks.is_some())
        })
    }

    fn start_cpu_continuations(&mut self) -> bool {
        let ids = self.order.clone();
        let mut progressed = false;
        for id in ids {
            let required = self
                .running
                .get(&id)
                .is_some_and(|state| self.cpu_continuation_required(state));
            if !required {
                continue;
            }
            let pipeline = self.logits_pipeline.clone();
            let eos = self.ctrl.eos.clone();
            let Some(state) = self.running.get_mut(&id) else {
                continue;
            };
            if state.cancelled
                || state.cpu_pending.is_some()
                || state.cpu_masks.is_some()
                || self.pending_finishes.contains_key(&id)
            {
                continue;
            }
            state.cpu_generation = state.cpu_generation.saturating_add(1);
            let key = CpuTaskKey {
                request_id: id,
                epoch: state.epoch,
                point: state.version,
                generation: state.cpu_generation,
            };
            let task = CpuTask {
                key,
                n_generated: state.und.tokens_emitted,
                eos,
                generated: state.replay.generated_ids.clone(),
                sampling: state.req.sampling.clone(),
                pipeline,
            };
            match self.cpu_continuations.try_submit(task) {
                Ok(()) => {
                    state.cpu_pending = Some(key);
                    self.cpu_deadlines
                        .insert(key, Instant::now() + self.cpu_task_timeout);
                    progressed = true;
                }
                Err(_) => {
                    break;
                }
            }
        }
        progressed
    }

    fn drain_cpu_continuations(&mut self) -> bool {
        let ready = self.cpu_continuations.drain_ready();
        if ready.is_empty() {
            return false;
        }
        let mut failed = Vec::new();
        for result in ready {
            self.cpu_deadlines.remove(&result.key);
            let Some(state) = self.running.get_mut(&result.key.request_id) else {
                continue;
            };
            if state.cpu_pending != Some(result.key)
                || state.epoch != result.key.epoch
                || state.version != result.key.point
            {
                continue;
            }
            state.cpu_pending = None;
            match result.outcome {
                Ok(masks) => state.cpu_masks = Some(masks),
                Err(error) => {
                    tracing::error!(
                        request_id = result.key.request_id.0,
                        %error,
                        "CPU semantic continuation failed"
                    );
                    failed.push(result.key.request_id);
                }
            }
        }
        for id in failed {
            self.finish_after_inflight(id, FinishReason::Error, None);
        }
        true
    }

    fn expire_cpu_continuations(&mut self) -> bool {
        if self.cpu_deadlines.is_empty() {
            return false;
        }
        let now = Instant::now();
        let expired = self
            .cpu_deadlines
            .iter()
            .filter_map(|(key, deadline)| (*deadline <= now).then_some(*key))
            .collect::<Vec<_>>();
        let mut failed = Vec::new();
        for key in expired {
            self.cpu_deadlines.remove(&key);
            if let Some(state) = self.running.get_mut(&key.request_id)
                && state.cpu_pending == Some(key)
            {
                state.cpu_pending = None;
                state.cpu_generation = state.cpu_generation.saturating_add(1);
                failed.push(key.request_id);
            }
        }
        for id in &failed {
            tracing::error!(
                request_id = id.0,
                timeout_ms = self.cpu_task_timeout.as_millis(),
                "CPU semantic continuation exceeded its request-local deadline"
            );
            self.finish_after_inflight(*id, FinishReason::Error, None);
        }
        !failed.is_empty()
    }

    fn output_window_ready(&self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        if state.event_tx.is_closed() {
            return false;
        }
        let available = state
            .event_tx
            .capacity()
            .saturating_add(OUTPUT_JOURNAL_CAPACITY.saturating_sub(state.output_journal.len()));
        let reserved = self
            .inflight_ops
            .get(&id)
            .into_iter()
            .flatten()
            .map(|operation| operation.apply.output_event_bound)
            .sum::<usize>();
        available
            .saturating_sub(reserved)
            .saturating_sub(OUTPUT_TERMINAL_RESERVE)
            >= self.next_output_bound(id)
    }

    fn next_output_bound(&self, id: RequestId) -> usize {
        match self.peek_next_operation_variant(id) {
            Some(WorkVariant::TokenExtend | WorkVariant::TokenDecode) => 4,
            Some(WorkVariant::GenFlow) => usize::from(self.denoise_step_burst).saturating_add(2),
            Some(WorkVariant::Materialize) => 3,
            Some(_) | None => 2,
        }
    }

    fn register_inflight(&mut self, operation: Operation, apply: SchedulerApply, started: Instant) {
        let request_id = operation.request_key.session_id;
        self.inflight_ops
            .entry(request_id)
            .or_default()
            .push_back(InflightOp {
                operation,
                apply,
                started,
            });
    }

    fn stage_completion(&mut self, record: CompletionRecord, products: Arc<[ProductPayload]>) {
        let id = record.request_key.session_id;
        let op_id = record.op_id.0;
        let known = self.inflight_ops.get(&id).is_some_and(|queue| {
            queue.iter().any(|inflight| {
                inflight.operation.request_key == record.request_key
                    && inflight.operation.op_id.0 == op_id
            })
        });
        let duplicate = self
            .pending_completions
            .get(&id)
            .is_some_and(|pending| pending.contains_key(&op_id));
        if !known || duplicate {
            self.trace_record(json!({
                "event": "unknown_result_op_id",
                "at_s": now(),
                "request_id": id.0,
                "op_id": op_id,
            }));
            if self.running.contains_key(&id) {
                self.finish(id, FinishReason::Error);
            }
            return;
        }
        let arrival_seq = self.next_completion_seq;
        self.next_completion_seq = self.next_completion_seq.saturating_add(1);
        self.pending_completions.entry(id).or_default().insert(
            op_id,
            PendingCompletion {
                record,
                products,
                arrival_seq,
            },
        );
    }

    fn take_ready_completions(&mut self) -> Vec<PendingCompletion> {
        let mut ready = self
            .pending_completions
            .iter()
            .filter_map(|(id, pending)| {
                let inflight = self.inflight_ops.get(id)?.front()?;
                let op_id = inflight.operation.op_id.0;
                let completion = pending.get(&op_id)?;
                Some((
                    completion_priority(inflight.operation.work.variant()),
                    completion.arrival_seq,
                    *id,
                    op_id,
                ))
            })
            .collect::<Vec<_>>();
        ready.sort_unstable_by_key(|(priority, arrival_seq, ..)| (*priority, *arrival_seq));
        let mut completions = Vec::with_capacity(ready.len());
        for (_, _, id, op_id) in ready {
            let Some(pending) = self.pending_completions.get_mut(&id) else {
                continue;
            };
            if let Some(completion) = pending.remove(&op_id) {
                completions.push(completion);
            }
            if pending.is_empty() {
                self.pending_completions.remove(&id);
            }
        }
        completions
    }

    /// Resolve the front in-flight op for `id` by the worker's echoed `op_id`.
    fn pop_inflight(
        &mut self,
        request_key: RequestKey,
        op_id: u64,
    ) -> Option<(Operation, SchedulerApply, Instant)> {
        let id = request_key.session_id;
        let queue = self.inflight_ops.get_mut(&id)?;
        if op_id == 0
            || queue.front().is_none_or(|inflight| {
                inflight.operation.request_key != request_key || inflight.operation.op_id.0 != op_id
            })
        {
            return None;
        }
        let inflight = queue.pop_front().expect("front checked above");
        let empty = queue.is_empty();
        let parent_op_id = inflight.operation.parent.producer_op_id.0;
        if inflight.operation.bounds.max_transfer_bytes > 0 {
            self.inflight_transfers = self
                .inflight_transfers
                .checked_sub(1)
                .expect("completed transfer operation owns one reservation");
        }
        if empty {
            self.inflight_ops.remove(&id);
        }
        if parent_op_id > 0 {
            self.mark_operation_reclaimed(id, parent_op_id);
        }
        Some((inflight.operation, inflight.apply, inflight.started))
    }

    fn mark_operation_reclaimed(&mut self, id: RequestId, op_id: u64) {
        if let Some(st) = self.running.get_mut(&id) {
            st.trace.stamp_existing(
                uniserve_worker_wire::OpId(op_id),
                crate::trace::LifecyclePhase::ReleaseIssued,
                uniserve_core::now_monotonic_us(),
            );
        }
        // The operation's device products are now freed under event-safe
        // reclamation; record the terminal lifecycle phase.
        if let Some(st) = self.running.get_mut(&id) {
            st.trace.stamp_existing(
                uniserve_worker_wire::OpId(op_id),
                crate::trace::LifecyclePhase::PhysicallyReclaimed,
                uniserve_core::now_monotonic_us(),
            );
        }
    }

    fn release_products(&mut self, products: Vec<ProductRef>) {
        if products.is_empty() {
            return;
        }
        let mut handles = products
            .into_iter()
            .map(|product| u64::from(product.generation))
            .collect::<Vec<_>>();
        handles.sort_unstable();
        handles.dedup();
        self.gated_control(ControlOp::ReleaseProducts(handles));
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
    /// severity by class so a benign `InputError` does not spam warnings.
    fn on_executor_error(&mut self, e: anyhow::Error) {
        if e.downcast_ref::<WorkerLossError>().is_some() {
            tracing::warn!("worker state was lost; terminating affected live sessions: {e}");
            self.fail_all_running(&e.to_string());
            return;
        }
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
        } else if matches!(code, Some("InputError")) {
            // A malformed request is the client's fault, not a worker problem:
            // fail just that request at info level instead of warn-spam.
            tracing::info!(?code, "request rejected by worker (failing in-flight): {e}");
        } else {
            // Non-fatal worker errors keep the worker up. `retryable` is
            // surfaced so a recoverable class (OOM/transient) is visible to
            // operators; no automatic requeue is attempted here.
            tracing::warn!(
                ?code,
                retryable,
                "non-fatal worker error (failing in-flight, worker stays up): {e}"
            );
        }
        self.fail_all_inflight(&format!("{e}"));
    }

    fn apply_result(&mut self, report: CompletionReport) {
        let result_step_id = report.step_id;
        let returned_partitions = report
            .partitions
            .iter()
            .map(|partition| partition.partition_id)
            .collect::<Vec<_>>();
        let batch_complete = if let Some(pending) = self.batch_partitions.get_mut(&result_step_id) {
            for partition_id in &returned_partitions {
                if !pending.remove(partition_id) {
                    tracing::error!(
                        step_id = result_step_id,
                        partition_id,
                        "executor returned a duplicate or unknown partition"
                    );
                    self.fatal = true;
                }
            }
            pending.is_empty()
        } else {
            tracing::error!(
                step_id = result_step_id,
                "executor returned a result for an unknown batch"
            );
            self.fatal = true;
            false
        };
        let worker_exec_us = report
            .partitions
            .iter()
            .filter_map(|partition| partition.worker_exec_us)
            .reduce(u64::saturating_add);
        let forward_stats = report
            .partitions
            .iter()
            .filter_map(|partition| partition.forward_stats.clone())
            .collect::<Vec<_>>();
        let completion_count = report
            .partitions
            .iter()
            .map(|partition| partition.completions.len())
            .sum();
        if batch_complete {
            self.batch_partitions.remove(&result_step_id);
            if let Some(controls) = self.control_batches.remove(&result_step_id) {
                self.acknowledge_controls(&controls);
            }
        }
        let batch_roundtrip_us = self
            .batch_started
            .get(&result_step_id)
            .map(|start| start.elapsed().as_micros() as u64)
            .unwrap_or(0);
        if batch_complete {
            self.batch_started.remove(&result_step_id);
        }
        let worker_us = worker_exec_us.unwrap_or(0);
        if let Some(w) = worker_exec_us {
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
        let trace_enabled = self.trace_enabled();
        let forward_stats_trace = trace_enabled.then(|| {
            forward_stats
                .iter()
                .map(worker_forward_stats_trace)
                .collect::<Vec<_>>()
        });
        for stats in &forward_stats {
            self.record_worker_forward_stats(Some(stats));
        }
        for partition in report.partitions {
            let products = Arc::<[ProductPayload]>::from(partition.products);
            for record in partition.completions {
                self.stage_completion(record, Arc::clone(&products));
            }
        }
        let mut resolved_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));
        let mut progress_ops = trace_enabled.then(|| Vec::with_capacity(completion_count));
        loop {
            let completions = self.take_ready_completions();
            if completions.is_empty() {
                break;
            }
            let mut to_resolve = Vec::with_capacity(completions.len());
            for completion in completions {
                let PendingCompletion {
                    record,
                    products,
                    arrival_seq,
                } = completion;
                let id = record.request_key.session_id;
                let op_id = record.op_id.0;
                let Some((operation, apply, started)) =
                    self.pop_inflight(record.request_key, op_id)
                else {
                    self.trace_record(json!({
                        "event": "unknown_result_op_id",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                    }));
                    if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                };
                let operation_variant = operation.work.variant();
                let view = SequenceView::from_report(&record, products.as_ref());
                // fold this op's host-side round-trip latency into the history.
                let roundtrip_us = started.elapsed().as_micros() as u64;
                self.latency
                    .observe(operation_variant.as_wire_str(), roundtrip_us);
                // record the op-resolved lifecycle event (op_id echoed by the
                // worker, host round-trip + worker compute time).
                let sampled_token_ids_len = view.committed_tokens.len();
                let sampled_token_ids_last = view.committed_tokens.last().copied();
                if let Some(resolved_ops) = resolved_ops.as_mut() {
                    let image_hw = view
                        .image_png
                        .as_deref()
                        .and_then(|png| validate_png_artifact(png, None))
                        .map(|metadata| (metadata.height, metadata.width));
                    resolved_ops.push(json!({
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_wire_str(),
                        "transition_delta": apply.delta.as_str(),
                        "transition_replayability": apply.replayability_after_apply.as_str(),
                        "roundtrip_us": roundtrip_us,
                        "completion_copy_us": record.timing_counters.copy_us,
                        "completion_ready_to_observed_us": record.timing_counters.host_us,
                        "sampled_token": sampled_token_ids_last.is_some(),
                        "sampled_token_ids_len": sampled_token_ids_len,
                        "sampled_token_ids_last": sampled_token_ids_last,
                        "flow_done": view.flow_done,
                        "steps_completed": record.logical_lengths.latent_len,
                        "image_done": view.image_png.is_some(),
                        "image_hw": image_hw,
                        "kv_tokens": view.kv_visible_len,
                        "product_handle": view.encode_generation,
                    }));
                }
                if let Some(st) = self.running.get_mut(&id) {
                    let op_key = crate::trace::OperationKey::from(&operation);
                    let kind = operation_variant.as_wire_str();
                    let observed_us = uniserve_core::now_monotonic_us();
                    // Device phases are reconstructed from the completion
                    // record's asynchronously reported durations, anchored at
                    // submission and clamped to the host observation.
                    let submitted_us = st
                        .trace
                        .operations()
                        .iter()
                        .find(|op| op.key.op_id == op_key.op_id)
                        .and_then(|op| op.at(crate::trace::LifecyclePhase::Submitted));
                    if let Some(submitted_us) = submitted_us {
                        let timing = &record.timing_counters;
                        let device_started = submitted_us
                            .saturating_add(timing.queued_us)
                            .min(observed_us);
                        let producer_ready = device_started
                            .saturating_add(timing.device_us)
                            .min(observed_us);
                        let copy_ready = producer_ready
                            .saturating_add(timing.copy_us)
                            .min(observed_us);
                        st.trace.stamp(
                            op_key,
                            Some(kind),
                            crate::trace::LifecyclePhase::DeviceExecutionStarted,
                            device_started,
                        );
                        st.trace.stamp(
                            op_key,
                            Some(kind),
                            crate::trace::LifecyclePhase::ProducerReady,
                            producer_ready,
                        );
                        st.trace.stamp(
                            op_key,
                            Some(kind),
                            crate::trace::LifecyclePhase::CompletionCopyReady,
                            copy_ready,
                        );
                    }
                    st.trace.stamp(
                        op_key,
                        Some(kind),
                        crate::trace::LifecyclePhase::CompletionObserved,
                        observed_us,
                    );
                }
                let predicated_parent_point = (record.status == OpStatus::Predicated).then(|| {
                    self.running
                        .get(&id)
                        .map_or(0, |state| state.version.min(u64::from(u32::MAX)) as u32)
                });
                if let Err(error) = apply.validate_result(
                    &operation,
                    &record,
                    products.as_ref(),
                    predicated_parent_point,
                ) {
                    self.trace_record(json!({
                        "event": "transition_validation_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_wire_str(),
                        "error": transition_validation_error_str(&error),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }
                let semantic_blocked = self.running.get(&id).is_some_and(|state| state.cancelled)
                    || self.pending_finishes.contains_key(&id);
                if semantic_blocked {
                    self.release_transition_resources(id, &apply);
                    self.finish_pending_if_idle(id);
                    continue;
                }
                let expected_parent = self.fixed_version(id);
                let prefix_versions =
                    token_prefix_versions(Some(&operation), &record, expected_parent.as_ref());
                let cursor_result = self.running.get_mut(&id).map(|state| {
                    state
                        .cursor
                        .apply_transition(&operation, &apply, &record, products.as_ref())
                });
                if let Some(Err(error)) = cursor_result {
                    self.trace_record(json!({
                        "event": "cursor_transition_failed",
                        "at_s": now(),
                        "request_id": id.0,
                        "op_id": op_id,
                        "operation_type": operation_variant.as_wire_str(),
                        "error": cursor_apply_error_str(&error),
                    }));
                    if self.running.contains_key(&id) {
                        self.finish_after_inflight(id, FinishReason::Error, None);
                    }
                    continue;
                }
                // A state-advancing completion resolves a new point. Ordered commit
                // control emission below decides when that point becomes semantic.
                let advanced = record.status == OpStatus::Ok
                    && record.selected_point > 0
                    && operation.advances_state;
                let latest_device_version = if advanced {
                    let token = operation
                        .outputs
                        .iter()
                        .find(|output| {
                            output.kind == ProductKind::Token
                                && output.storage_class
                                    == uniserve_worker_wire::StorageClass::DeviceTensor
                        })
                        .cloned();
                    token.map(|token| ResidentDeviceVersion {
                        producer: operation.work.variant(),
                        token,
                        version: VersionRef {
                            request_key: operation.request_key,
                            producer_op_id: operation.op_id,
                            point: Point::Device {
                                point_index: record.selected_point,
                                selected_point: None,
                                producer_plan_digest: operation.plan_digest.clone(),
                            },
                        },
                    })
                } else {
                    None
                };
                let retain_device_version = !self.has_inflight(id);
                if let Some(state) = self.running.get_mut(&id)
                    && advanced
                {
                    state.version = u64::from(record.selected_point);
                    state.resolved_semantic = record.semantic_digest.clone();
                    state.resolved_producer_op_id = record.op_id.0;
                    state.latest_device_version = if retain_device_version {
                        latest_device_version
                    } else {
                        None
                    };
                }
                let selected_fixed = self.fixed_version(id);
                // A stop-string token op defers its semantic commit until the
                // frontend decoder rules on its exact prefix; every other
                // advancing op commits immediately in completion order. The
                // deferred commit is carried to the resolve pass below, where the
                // op's public token count is known and it joins the ordered
                // pending-commit queue.
                let mut deferred_commit: Option<(VersionRef, VersionRef, u64)> = None;
                if advanced
                    && let (Some(expected_parent), Some(selected)) =
                        (expected_parent, selected_fixed.clone())
                {
                    let public_event_limit = self.public_limit_for(id, &apply);
                    let decoder_decision_required = matches!(
                        operation_variant,
                        WorkVariant::TokenExtend
                            | WorkVariant::TokenDecode
                            | WorkVariant::TokenVerify
                    ) && self
                        .running
                        .get(&id)
                        .is_some_and(|state| !state.req.stop_strings.is_empty());
                    if decoder_decision_required {
                        deferred_commit = Some((expected_parent, selected, public_event_limit));
                    } else {
                        self.queue_commit(id, expected_parent, selected, public_event_limit);
                    }
                }
                self.release_transition_resources(id, &apply);
                if matches!(
                    operation_variant,
                    WorkVariant::GenFlow | WorkVariant::Materialize
                ) {
                    let consumed_latents = operation
                        .inputs
                        .iter()
                        .filter(|product| product.kind == ProductKind::Latent)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.release_products(consumed_latents);
                    }
                }
                let priority = completion_priority(operation_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.public_token_seq);
                if record.status == OpStatus::Predicated {
                    let unused_products = operation.outputs.to_vec();
                    self.release_products(unused_products);
                    self.finish_pending_if_idle(id);
                } else {
                    to_resolve.push((
                        priority,
                        arrival_seq,
                        id,
                        operation,
                        apply,
                        view,
                        selected_fixed,
                        public_tokens_before,
                        prefix_versions,
                        deferred_commit,
                    ));
                }
            }
            to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
            for (
                _priority,
                _seq_index,
                id,
                operation,
                apply,
                view,
                selected_fixed,
                public_tokens_before,
                prefix_versions,
                deferred_commit,
            ) in to_resolve
            {
                let token_operation = matches!(
                    operation.work.variant(),
                    WorkVariant::TokenExtend | WorkVariant::TokenDecode | WorkVariant::TokenVerify
                );
                if self.running.contains_key(&id) && !self.pending_finishes.contains_key(&id) {
                    self.resolve(id, operation, apply, view, prefix_versions.clone());
                }
                if token_operation {
                    let mut immediate_commit: Option<(VersionRef, VersionRef, u64)> = None;
                    if let Some(state) = self.running.get_mut(&id) {
                        let emitted_public = state.public_token_seq > public_tokens_before;
                        if emitted_public {
                            let emitted = state.public_token_seq - public_tokens_before;
                            for (offset, selected) in
                                prefix_versions.into_iter().take(emitted).enumerate()
                            {
                                state
                                    .token_cutoffs
                                    .insert(public_tokens_before + offset + 1, selected);
                            }
                            if emitted > 0
                                && !state.token_cutoffs.contains_key(&state.public_token_seq)
                                && let Some(selected) = selected_fixed
                            {
                                state.token_cutoffs.insert(state.public_token_seq, selected);
                            }
                        }
                        if let Some((expected_parent, selected, public_event_limit)) =
                            deferred_commit
                        {
                            // Chain the deferred commit onto the tip of the
                            // pending queue so the worker applies commits in
                            // exact parent order once the decoder acknowledges
                            // each prefix. An op that emits no public token
                            // joins the token count of the commit ahead of it,
                            // or commits immediately when the queue is empty.
                            let chained_parent = state
                                .pending_commits
                                .back()
                                .map(|pending| pending.selected.clone())
                                .unwrap_or(expected_parent);
                            let token_count = if emitted_public {
                                Some(state.public_token_seq)
                            } else {
                                state
                                    .pending_commits
                                    .back()
                                    .and_then(|pending| pending.token_count)
                            };
                            if token_count.is_none() && state.pending_commits.is_empty() {
                                immediate_commit =
                                    Some((chained_parent, selected, public_event_limit));
                            } else {
                                state.pending_commits.push_back(PendingSemanticCommit {
                                    token_count,
                                    expected_parent: chained_parent,
                                    selected,
                                    public_event_limit,
                                });
                            }
                        }
                    }
                    if let Some((expected_parent, selected, public_event_limit)) = immediate_commit
                    {
                        self.queue_commit(id, expected_parent, selected, public_event_limit);
                    }
                }
                self.finish_pending_if_idle(id);
                if let (Some(st), Some(progress_ops)) =
                    (self.running.get(&id), progress_ops.as_mut())
                {
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
        }
        if let (Some(resolved_ops), Some(progress_ops)) = (resolved_ops, progress_ops) {
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
        self.inflight_transfers = 0;
        self.pending_completions.clear();
        // the submitted batches whose results will now never return are
        // failed here, so drop their pending submit-timestamps too — otherwise
        // `batch_started` accumulates orphaned entries for every failed batch.
        self.batch_started.clear();
        self.batch_partitions.clear();
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

    fn fail_all_running(&mut self, message: &str) {
        self.inflight_ops.clear();
        self.inflight_transfers = 0;
        self.pending_completions.clear();
        self.batch_started.clear();
        self.batch_partitions.clear();
        let ids = self.running.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.emit(
                id,
                GenEvent::Error {
                    message: message.to_string(),
                },
            );
            self.finish(id, FinishReason::Error);
        }
    }

    fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host KV ownership then remains pinned through the close
        // acknowledgement and ordered worker session retirement.
        let cancelled: Vec<(RequestId, bool, bool)> = self
            .running
            .iter()
            .filter(|(id, s)| s.cancelled && !self.has_inflight(**id))
            .map(|(k, s)| (*k, s.aborted, s.stop_matched))
            .collect();
        for (id, aborted, stop_matched) in cancelled {
            let reason = if stop_matched {
                FinishReason::Stop
            } else if aborted {
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
    /// resident for its complete lifetime.
    fn admit(&mut self) {
        let bs = self.caps.block_size as usize;
        loop {
            if self.running.len() >= self.config.max_num_seqs || self.request_slots.is_empty() {
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
                if self.bm.free_request_pages() >= need && encoder_ok {
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

            // Exact checkpoints retain their physical KV identity. Until the
            // configured route provides relocatable checkpoint storage, a
            // resident request remains non-preemptible and admission queues.
            break;
        }
    }

    fn admit_running(&mut self, mut st: ReqState) {
        st.request_pool_idx = self
            .request_slots
            .acquire()
            .expect("admission checked request-slot capacity");
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
        st.trace.mark_admitted(uniserve_core::now_monotonic_us());
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
            "free_blocks": self.bm.free_request_pages(),
            "reserved_blocks": self.reserved_blocks,
            "reserved_encoder_entries": self.reserved_encoder_entries,
        }));
        let bs = self.caps.block_size as usize;
        if let Some(st) = self.running.get_mut(&id) {
            self.prefix_cache.lookup(st, &mut self.bm, &self.stats, bs);
        }
    }

    /// Assemble the per-step batch: walk the priority order, ask each request for
    /// at most one op, clip prefill chunks to the remaining token budget, and
    /// pair first-dispatch requests with their typed admission record.
    fn assemble(&mut self) -> (Vec<Admission>, Vec<PlannedTransition>) {
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
            // it cannot displace), fall through to the decode lane instead of
            // idling — otherwise a starved waiting prompt would stall ready
            // decodes forever.
            return self.assemble_pass(&ids, Some(AssemblyLane::Decode));
        }
        (new_reqs, ops)
    }

    fn request_has_prompt_work(&self, id: RequestId) -> bool {
        if self.running.get(&id).is_none_or(|state| state.cancelled) {
            return false;
        }
        self.inflight_ops
            .get(&id)
            .into_iter()
            .flatten()
            .any(|op| assembly_lane(op.operation.work.variant()) == AssemblyLane::Prefill)
            || self
                .peek_next_operation_variant(id)
                .is_some_and(|operation_variant| {
                    assembly_lane(operation_variant) == AssemblyLane::Prefill
                })
    }

    fn request_has_ready_decode(&self, id: RequestId) -> bool {
        self.running.get(&id).is_some_and(|state| !state.cancelled)
            && self
                .peek_next_operation_variant(id)
                .is_some_and(|operation_variant| {
                    assembly_lane(operation_variant) == AssemblyLane::Decode
                })
            && self.can_schedule_next(id)
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
            // The first scheduling turn after a cohort drains belongs to ready
            // decode work before another finite prompt cohort may open.
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
        let overlaps_decode = ids.iter().copied().any(|decode_id| {
            self.request_has_ready_decode(decode_id)
                && prompt_ids.iter().any(|prompt_id| *prompt_id != decode_id)
        });
        if overlaps_decode {
            self.prompt_cohort = Some(prompt_ids);
        }
    }

    fn assemble_pass(
        &mut self,
        ids: &[RequestId],
        lane: Option<AssemblyLane>,
    ) -> (Vec<Admission>, Vec<PlannedTransition>) {
        let mut admissions: Vec<Admission> = Vec::new();
        let mut ops: Vec<PlannedTransition> = Vec::new();
        let mut selected: HashSet<RequestId> = HashSet::new();
        // vLLM's per-step token budget with the clip rule: the budget, not the
        // chunk threshold, is the binding constraint.
        let mut budget: usize = self.config.max_num_batched_tokens;
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
            if !self.can_schedule_next(id) {
                continue;
            }
            let cancelled = self.running.get(&id).map(|s| s.cancelled).unwrap_or(true);
            if cancelled {
                continue;
            }
            let next_type = self.peek_next_operation_variant(id);
            // When a decode pass admits text prefill rows, the worker receives a
            // single mixed forward. There is no `supports_mixed_op_kinds` gate;
            // co-batched rows shift each other's numerics only through inherent
            // batched-kernel FP non-invariance, not structural corruption.
            let mut mixed_prefill = false;
            if let (Some(target), Some(operation_variant)) = (lane, next_type)
                && {
                    let candidate_lane = assembly_lane(operation_variant);
                    matches!(candidate_lane, AssemblyLane::Prefill | AssemblyLane::Decode)
                        && candidate_lane != target
                }
            {
                mixed_prefill = target == AssemblyLane::Decode
                    && operation_variant == WorkVariant::TokenExtend
                    && mixed_left > 0
                    && self.running.get(&id).is_some_and(|st| {
                        st.is_replayable_text() && !st.req.sampling.prompt_logprobs_requested()
                    });
                if !mixed_prefill {
                    continue;
                }
            }
            if next_type == Some(WorkVariant::GenFlow)
                && (denoise_occupies_decode_pipeline || !self.can_schedule_denoise(id))
            {
                continue;
            }
            // Diagnostic: a denoise step and text rows in one forward take the
            // packed route, which costs more than running each on its own. When
            // the flag is set a flow op only opens an empty batch, and the loop
            // below closes the batch as soon as one is placed.
            if self.flow_exclusive_batch
                && next_type == Some(WorkVariant::GenFlow)
                && !(ops.is_empty() && mixed_ops.is_empty())
            {
                continue;
            }
            // Build one operation when its exact resident resources fit.
            let op_budget = if mixed_prefill {
                budget.min(mixed_left)
            } else {
                budget
            };
            if let Some(mut op) = self.next_transition(id, op_budget) {
                if mixed_prefill {
                    mixed_left = mixed_left.saturating_sub(planned_op_token_cost(&op));
                }
                budget = budget.saturating_sub(planned_op_token_cost(&op));
                if !self.reserve_transition_resources(&mut op) {
                    self.return_unsent_blocks(id, &op);
                    tracing::debug!(
                        request_id = id.0,
                        "operation registration is paused by physical resource pressure"
                    );
                    continue;
                }
                let finish_token_ids = self
                    .running
                    .get(&id)
                    .map(|state| canonical_continuation_stop_token_ids(&state.req, &self.ctrl.eos))
                    .unwrap_or_default();
                if let Some(st) = self.running.get_mut(&id)
                    && !st.resources.worker_registered
                {
                    st.resources.worker_registered = true;
                    let request_key = RequestKey::new(self.authority_id, id, st.epoch);
                    let admission = Admission::new(
                        request_key,
                        st.request_pool_idx,
                        Some(UndAdmission {
                            sampling: st.req.sampling.clone(),
                            negative_token_ids: st.context.negative_prompt_ids.clone(),
                            finish_token_ids,
                            kv: KvAdmission {
                                prefix_len: st.ingest.prompt_cursor,
                                group_id: 0,
                            },
                        }),
                        st.req.behavior.gen_output.then(|| GenAdmission {
                            image: st.req.image.clone(),
                        }),
                    )
                    .expect("validated request produces a valid admission");
                    // The admission-root version is produced by the synthetic
                    // admission, identified by operation-id 0 — the sentinel the
                    // worker seeds as its initial committed version — so the first
                    // operation's fixed parent matches the worker's committed state.
                    st.resolved_semantic = admission.digest.clone();
                    st.resolved_producer_op_id = 0;
                    st.committed_semantic = admission.digest.clone();
                    st.committed_producer_op_id = 0;
                    st.latest_device_version = None;
                    st.admission_digest = Some(admission.digest.clone());
                    st.token_cutoffs.clear();
                    st.token_cutoffs.insert(
                        st.public_token_seq,
                        VersionRef {
                            request_key,
                            producer_op_id: OpId(0),
                            point: Point::Fixed {
                                point_index: 0,
                                semantic_digest: admission.digest.clone(),
                            },
                        },
                    );
                    admissions.push(admission);
                }
                selected.insert(id);
                let placed_flow = op.operation_variant == WorkVariant::GenFlow;
                if mixed_prefill {
                    mixed_ops.push(op);
                } else {
                    ops.push(op);
                }
                if self.flow_exclusive_batch && placed_flow {
                    budget = 0;
                }
            }
        }
        ops.extend(mixed_ops);
        (admissions, ops)
    }

    fn select_assembly_lane(&self, ids: &[RequestId]) -> Option<AssemblyLane> {
        let mut projected_decode_ready = false;
        let mut committed_decode_ready = false;
        let mut prefill_ready = false;
        for id in ids.iter().copied() {
            if self.running.get(&id).map(|s| s.cancelled).unwrap_or(true) {
                continue;
            }
            let Some(operation_type) = self.peek_next_operation_variant(id) else {
                continue;
            };
            match assembly_lane(operation_type) {
                AssemblyLane::Decode => {
                    if self.can_schedule_next(id) {
                        if self.has_inflight(id) {
                            projected_decode_ready = true;
                        } else {
                            committed_decode_ready = true;
                        }
                    }
                }
                AssemblyLane::Prefill => {
                    if self.can_schedule_next(id) {
                        prefill_ready = true;
                    }
                }
                AssemblyLane::Other => {}
            }
        }
        if prefill_ready && !self.any_prefill_inflight() {
            Some(AssemblyLane::Prefill)
        } else if committed_decode_ready || projected_decode_ready {
            Some(AssemblyLane::Decode)
        } else if prefill_ready {
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
        match self.peek_next_operation_variant(id) {
            Some(
                WorkVariant::EncodeVision | WorkVariant::EncodeLatent | WorkVariant::TokenExtend,
            ) => 0,
            Some(
                WorkVariant::TokenDecode
                | WorkVariant::TokenVerify
                | WorkVariant::Materialize
                | WorkVariant::TransferKvInstall,
            ) => 1,
            Some(WorkVariant::GenFlow) => 2,
            Some(
                WorkVariant::Draft
                | WorkVariant::GenTransition
                | WorkVariant::TransferProduct
                | WorkVariant::TransferKvPublish,
            )
            | None => 3,
        }
    }

    fn peek_next_operation_variant(&self, id: RequestId) -> Option<WorkVariant> {
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
                ImageIngestStep::VaeEncode => WorkVariant::EncodeLatent,
                ImageIngestStep::VitEncode => WorkVariant::EncodeVision,
            });
        }
        Some(match st.lifecycle.phase {
            Phase::Encode => match st.pending_image_step()? {
                ImageIngestStep::VaeEncode => WorkVariant::EncodeLatent,
                ImageIngestStep::VitEncode => WorkVariant::EncodeVision,
            },
            Phase::IngestState => WorkVariant::TokenExtend,
            Phase::Prefill
                if self.has_inflight(id)
                    && self.can_queue_decode_successor(id)
                    && self.projected_cursor(id).is_some_and(|cursor| {
                        cursor.prompt_cursor as usize >= st.effective_prompt().len()
                    }) =>
            {
                WorkVariant::TokenDecode
            }
            Phase::Prefill => WorkVariant::TokenExtend,
            Phase::DecodeUnd => WorkVariant::TokenDecode,
            Phase::CloseKv => WorkVariant::TokenExtend,
            Phase::PublishKv => WorkVariant::TransferKvPublish,
            Phase::TransitionGen => WorkVariant::GenTransition,
            Phase::DenoiseGen => WorkVariant::GenFlow,
            Phase::CommitGen => WorkVariant::Materialize,
            Phase::FeedbackEncode => {
                let feedback = st.req.policy.feedback.as_ref()?;
                match feedback.ingest.steps.get(st.feedback.ingest_step)? {
                    ImageIngestStep::VaeEncode => WorkVariant::EncodeLatent,
                    ImageIngestStep::VitEncode => WorkVariant::EncodeVision,
                }
            }
            Phase::FeedbackState => WorkVariant::TokenExtend,
        })
    }

    fn submit_batch(
        &mut self,
        admissions: Vec<Admission>,
        transitions: Vec<PlannedTransition>,
        mut controls: Vec<Control>,
    ) {
        let _span =
            tracing::trace_span!("scheduler.submit_batch", ops = transitions.len()).entered();
        self.step_id += 1;
        let step = self.step_id;
        let submit_at = Instant::now();
        let mut wire_ops = Vec::with_capacity(transitions.len());
        let mut input_products = Vec::new();
        let mut trace_ops = self
            .trace_enabled()
            .then(|| Vec::with_capacity(transitions.len()));
        let admitted_request_keys = admissions
            .iter()
            .map(|admission| admission.request_key)
            .collect::<HashSet<_>>();
        let mut kv_placements = HashMap::with_capacity(transitions.len());
        let mut kv_branch_placements = HashMap::with_capacity(transitions.len());
        let mut latent_placements = HashMap::with_capacity(transitions.len());
        for mut transition in transitions {
            let oid = self.next_op_id;
            self.next_op_id += 1;
            let request_id = transition.request_id;
            let Some((epoch, resolved_semantic, resolved_producer_op_id, latest_device_version)) =
                self.running.get(&request_id).and_then(|state| {
                    state.admission_digest.as_ref().map(|_| {
                        (
                            state.epoch,
                            state.resolved_semantic.clone(),
                            state.resolved_producer_op_id,
                            state.latest_device_version.clone(),
                        )
                    })
                })
            else {
                tracing::error!(
                    request_id = request_id.0,
                    "planned operation lost its session"
                );
                self.fatal = true;
                return;
            };
            if self.has_inflight(request_id) && !self.can_queue_decode_successor(request_id) {
                tracing::error!(
                    request_id = request_id.0,
                    "scheduler attempted to issue an unsafe projected successor"
                );
                self.fatal = true;
                return;
            }
            let Some(version) = self.projected_version(request_id) else {
                self.fatal = true;
                return;
            };
            let request_key = RequestKey::new(self.authority_id, request_id, epoch);
            // A device successor roots on the exact selected-point product of
            // either its in-flight predecessor or the latest resolved operation.
            // The first operation and CPU-gated transitions use the fixed
            // semantically committed parent.
            let projected_successor = self.has_inflight(request_id);
            let reusable_device_version =
                if !projected_successor && self.can_reuse_resolved_token_product(request_id) {
                    latest_device_version
                } else {
                    None
                };
            let (parent, predicate) = if projected_successor {
                let Some(predecessor) = self
                    .inflight_ops
                    .get(&request_id)
                    .and_then(|queue| queue.back())
                    .map(|op| &op.operation)
                else {
                    tracing::error!(
                        request_id = request_id.0,
                        "projected successor lost its predecessor operation"
                    );
                    self.fatal = true;
                    return;
                };
                (
                    VersionRef {
                        request_key,
                        producer_op_id: predecessor.op_id,
                        point: Point::Device {
                            point_index: u32::from(predecessor.bounds.max_points == 1),
                            selected_point: (predecessor.bounds.max_points > 1).then(|| {
                                predecessor
                                    .outputs
                                    .iter()
                                    .find(|output| output.kind == ProductKind::SelectedPoint)
                                    .cloned()
                                    .expect("multi-point predecessor has a selected-point product")
                            }),
                            producer_plan_digest: predecessor.plan_digest.clone(),
                        },
                    },
                    predecessor
                        .outputs
                        .iter()
                        .find(|output| output.kind == ProductKind::Token)
                        .cloned(),
                )
            } else if let Some(parent) = reusable_device_version {
                (parent.version, Some(parent.token))
            } else {
                (
                    VersionRef {
                        request_key,
                        producer_op_id: OpId(resolved_producer_op_id),
                        point: Point::Fixed {
                            point_index: version as u32,
                            semantic_digest: resolved_semantic,
                        },
                    },
                    None,
                )
            };
            // A device parent carries no embedded host point index. The
            // scheduler-owned resolved/projected cursor supplies it for result
            // validation without reading the device product.
            let device_parent = matches!(parent.point, Point::Device { .. });
            transition.device_parent_point = device_parent.then_some(version as u32);
            transition.predicate = predicate;
            transition.control_seq = self
                .running
                .get(&request_id)
                .map_or(0, |state| state.control_seq);
            let kv_lengths =
                transition_kv_lengths(&transition.delta, self.running.get(&request_id));
            let new_page_count = transition.new_blocks.len();
            transition.kv_capacity_pages = kv_lengths.map_or(0, |_| {
                self.bm
                    .blocks_for_group(request_id, 0)
                    .len()
                    .min(u32::MAX as usize) as u32
            });
            let mut operation_kv_placements = Vec::new();
            if let Some(lengths) = kv_lengths {
                for group_id in 0..self.bm.num_groups() {
                    let block_table = self
                        .bm
                        .blocks_for_group(request_id, group_id)
                        .iter()
                        .take(transition.kv_capacity_pages as usize)
                        .copied()
                        .collect::<Vec<_>>();
                    if block_table.len() != transition.kv_capacity_pages as usize {
                        tracing::error!(
                            request_id = request_id.0,
                            group_id,
                            expected_pages = transition.kv_capacity_pages,
                            actual_pages = block_table.len(),
                            "KV groups disagree on request capacity"
                        );
                        self.fatal = true;
                        return;
                    }
                    let pages_to_zero = if admitted_request_keys.contains(&request_key) {
                        let retained_pages = if group_id == 0 {
                            self.running
                                .get(&request_id)
                                .map(|state| {
                                    (state.ingest.prompt_cursor as usize)
                                        .div_ceil(self.caps.block_size as usize)
                                })
                                .unwrap_or_default()
                                .min(block_table.len())
                        } else {
                            0
                        };
                        block_table[retained_pages..].to_vec()
                    } else {
                        block_table[block_table.len().saturating_sub(new_page_count)..].to_vec()
                    };
                    operation_kv_placements.push(KvPlacement {
                        request_key,
                        op_id: OpId(oid),
                        group_id: group_id as u32,
                        block_table,
                        pages_to_zero,
                        prefix_length: lengths.prefix,
                        input_length: lengths.input,
                        visible_length: lengths.visible,
                        resulting_length: lengths.resulting,
                    });
                }
            }
            let output_event_bound = transition_output_bound(&transition);
            let planned_us = transition.planned_us;
            let reserved_us = transition.reserved_us;
            let registered = transition.register(
                request_key,
                OpId(oid),
                parent,
                &mut self.next_product_generation,
                output_event_bound,
            );
            let (operation, apply, payloads) = match registered {
                Ok(registered) => registered,
                Err(error) => {
                    tracing::error!(
                        request_id = request_id.0,
                        ?error,
                        "scheduler authority exhausted product identity space"
                    );
                    self.fatal = true;
                    self.fail_all_running("scheduler authority exhausted product identity space");
                    return;
                }
            };
            if !operation_kv_placements.is_empty() {
                kv_placements.insert(
                    (operation.request_key, operation.op_id),
                    operation_kv_placements,
                );
            }
            if operation.work.variant() == WorkVariant::GenFlow {
                let placements = match self.kv_branch_placements_for(
                    request_id,
                    operation.request_key,
                    operation.op_id,
                ) {
                    Ok(placements) => placements,
                    Err(error) => {
                        tracing::error!(
                            request_id = request_id.0,
                            error,
                            "invalid generation KV branch placement"
                        );
                        self.fatal = true;
                        return;
                    }
                };
                kv_branch_placements.insert((operation.request_key, operation.op_id), placements);
            }
            if matches!(
                operation.work.variant(),
                WorkVariant::GenTransition | WorkVariant::GenFlow | WorkVariant::Materialize
            ) {
                let Some(state) = self.running.get(&request_id) else {
                    self.fatal = true;
                    return;
                };
                let page_table = self.latent_pages.pages_for(request_id).to_vec();
                let latent_units = self.worker_image_latent_units_for(state).max(1);
                let (start_step, step_count) = match &apply.delta {
                    TransitionDelta::DenoiseGen {
                        start_step,
                        step_count,
                        ..
                    } => (u32::from(*start_step), u32::from(*step_count)),
                    TransitionDelta::TransitionGen { .. } => (0, 0),
                    TransitionDelta::CommitGen { .. } => (u32::from(state.image_gen.steps_done), 0),
                    _ => unreachable!("latent work has latent scheduler metadata"),
                };
                latent_placements.insert(
                    (operation.request_key, operation.op_id),
                    LatentPlacement {
                        request_key: operation.request_key,
                        op_id: operation.op_id,
                        page_table,
                        latent_units: latent_units.min(u64::from(u32::MAX)) as u32,
                        height: state.req.image.height,
                        width: state.req.image.width,
                        start_step,
                        step_count,
                    },
                );
            }
            let operation_variant = operation.work.variant().as_wire_str();
            if let Some(trace_ops) = trace_ops.as_mut() {
                let phase = self
                    .running
                    .get(&request_id)
                    .map(|state| phase_str(state.lifecycle.phase));
                trace_ops.push(json!({
                    "request_id": request_id.0,
                    "op_id": operation.op_id.0,
                    "operation_type": operation_variant,
                    "phase": phase,
                    "operation": operation_trace(&operation, &apply),
                    "transition": apply.delta.as_str(),
                    "resources": {
                        "release_on_apply": apply.release_on_apply,
                        "replayability_after_apply": apply.replayability_after_apply.as_str(),
                    },
                    "visibility": {
                        "und_tokens": format!("{:?}", apply.visibility.und_tokens),
                        "generated_image": apply.visibility.generated_image,
                    },
                }));
            }
            input_products.extend(payloads);
            let op_key = crate::trace::OperationKey::from(&operation);
            self.register_inflight(operation.clone(), apply, submit_at);
            wire_ops.push(operation);
            if let Some(st) = self.running.get_mut(&request_id) {
                st.latest_device_version = None;
                let now_us = uniserve_core::now_monotonic_us();
                // Backfill the two pre-registration phases from the builder now
                // that the operation carries its canonical identity.
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::trace::LifecyclePhase::Planned,
                    planned_us,
                );
                if reserved_us != 0 {
                    st.trace.stamp(
                        op_key,
                        Some(operation_variant),
                        crate::trace::LifecyclePhase::LogicalResourcesReserved,
                        reserved_us,
                    );
                }
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::trace::LifecyclePhase::WorkerRegistrationComplete,
                    now_us,
                );
                st.trace.stamp(
                    op_key,
                    Some(operation_variant),
                    crate::trace::LifecyclePhase::Submitted,
                    now_us,
                );
            }
        }
        self.peak_ops_in_batch = self.peak_ops_in_batch.max(wire_ops.len());
        self.stats
            .general
            .peak_ops
            .fetch_max(wire_ops.len(), Ordering::Relaxed);
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
            .kv_cache
            .free_blocks
            .store(self.bm.free_request_pages(), Ordering::Relaxed);
        let mixed = wire_ops.first().is_some_and(|first| {
            wire_ops
                .iter()
                .any(|operation| operation.work.variant() != first.work.variant())
        });
        self.batch_started.insert(step, submit_at);
        if let Some(trace_ops) = trace_ops {
            let operation_types: Vec<&'static str> = wire_ops
                .iter()
                .map(|operation| operation.work.variant().as_wire_str())
                .collect();
            let req_ids: Vec<u64> = wire_ops
                .iter()
                .map(|operation| operation.request_key.session_id.0)
                .collect();
            let admitted_session_ids: Vec<u64> = admissions
                .iter()
                .map(|admission| admission.request_key.session_id.0)
                .collect();
            self.trace_record(json!({
                "event": "batch_submitted",
                "at_s": now(),
                "step_id": step,
                "batch_size": wire_ops.len(),
                "mixed": mixed,
                "operation_types": operation_types,
                "request_ids": req_ids,
                "admitted_session_ids": admitted_session_ids,
                "ops": trace_ops,
                "scheduler": {
                    "policy": policy_str(self.config.policy),
                    "max_batch": self.config.max_batch,
                    "max_num_batched_tokens": self.config.max_num_batched_tokens,
                },
                "running": self.running.len(),
                "pending": self.pending.len(),
                "in_flight_before_submit": self.executor.in_flight(),
                "free_blocks": self.bm.free_request_pages(),
                "reserved_blocks": self.reserved_blocks,
                "worker_image_latent_active": self.worker_image_latent_used(),
                "worker_image_latent_capacity": self.caps.latent_capacity_units(),
            }));
        }
        if mixed {
            let operation_types: Vec<&'static str> = wire_ops
                .iter()
                .map(|operation| operation.work.variant().as_wire_str())
                .collect();
            let req_ids: Vec<u64> = wire_ops
                .iter()
                .map(|operation| operation.request_key.session_id.0)
                .collect();
            tracing::debug!(
                step_id = self.step_id,
                ?operation_types,
                ?req_ids,
                "submitting mixed forward batch"
            );
        }
        let releases = wire_ops
            .iter()
            .filter(|operation| operation.parent.producer_op_id.0 > 0)
            .map(|operation| Control::Release {
                request_key: operation.request_key,
                op_id: operation.parent.producer_op_id,
            })
            .collect::<Vec<_>>();
        controls.extend(releases);
        let partitions = self.partition_batch(
            wire_ops,
            &kv_placements,
            &kv_branch_placements,
            &latent_placements,
        );
        let partition_ids = partitions
            .iter()
            .map(|partition| partition.partition_id)
            .collect::<HashSet<_>>();
        self.batch_partitions.insert(step, partition_ids);
        let batch = Batch::new(step, admissions, partitions)
            .with_controls(controls.clone())
            .with_input_products(input_products);
        if let Err(e) = self.executor.submit(batch) {
            self.batch_started.remove(&step);
            self.batch_partitions.remove(&step);
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
            if !controls.is_empty() {
                self.control_batches.insert(step, controls);
            }
        }
    }

    fn partition_batch(
        &mut self,
        operations: Vec<Operation>,
        kv_placements: &HashMap<(RequestKey, OpId), Vec<KvPlacement>>,
        kv_branch_placements: &HashMap<(RequestKey, OpId), Vec<KvBranchPlacement>>,
        latent_placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    ) -> Vec<BatchPartition> {
        let mut routes: RouteDomainOperations = Vec::new();
        for operation in operations {
            let route = operation.route;
            let domain = operation.domain;
            let groups = if let Some((_, groups)) =
                routes.iter_mut().find(|(candidate, _)| *candidate == route)
            {
                groups
            } else {
                routes.push((route, Vec::new()));
                &mut routes.last_mut().expect("route was inserted").1
            };
            if let Some((_, members)) = groups
                .iter_mut()
                .find(|(candidate, _)| *candidate == domain)
            {
                members.push(operation);
            } else {
                groups.push((domain, vec![operation]));
            }
        }
        let mut partitions = Vec::new();
        let mut next_partition_id = 1u32;
        let mut next_submission_group = 1u32;
        for (route, groups) in routes {
            let mixed_capable = self.caps.tensorized_mixed;
            let mut mixed_candidates = Vec::new();
            let mut homogeneous = Vec::new();
            for (domain, operations) in groups {
                if !mixed_capable {
                    homogeneous.push((domain, operations));
                    continue;
                }
                let (candidates, independent): (Vec<_>, Vec<_>) = operations
                    .into_iter()
                    .partition(|operation| tensorized_mixed_runner_work(operation.work.variant()));
                if !candidates.is_empty() {
                    mixed_candidates.push((domain, candidates));
                }
                if !independent.is_empty() {
                    homogeneous.push((domain, independent));
                }
            }
            let candidate_variants = mixed_candidates
                .iter()
                .flat_map(|(_, operations)| {
                    operations.iter().map(|operation| operation.work.variant())
                })
                .collect::<HashSet<_>>();
            let tensorized_mixed = mixed_candidates.len() > 1
                && candidate_variants.contains(&WorkVariant::GenFlow)
                && candidate_variants.iter().any(|variant| {
                    matches!(
                        variant,
                        WorkVariant::TokenExtend
                            | WorkVariant::TokenDecode
                            | WorkVariant::TokenVerify
                            | WorkVariant::Draft
                    )
                });
            if !tensorized_mixed {
                homogeneous.append(&mut mixed_candidates);
            }
            if tensorized_mixed {
                let collective_seq = {
                    let value = self.next_collective_seq.max(1);
                    self.next_collective_seq = value.saturating_add(1);
                    value
                };
                let attention = partition_attention(
                    &mixed_candidates
                        .iter()
                        .flat_map(|(_, operations)| operations.iter().cloned())
                        .collect::<Vec<_>>(),
                );
                for (domain, operations) in mixed_candidates {
                    let request_pool_indices = self.request_pool_indices(&operations);
                    let kv_placements = self.kv_placements(&operations, kv_placements);
                    let kv_branch_placements =
                        self.kv_branch_placements(&operations, kv_branch_placements);
                    let latent_placements = self.latent_placements(&operations, latent_placements);
                    partitions.push(BatchPartition {
                        partition_id: next_partition_id,
                        submission_group: next_submission_group,
                        collective_seq,
                        domain,
                        route,
                        execution: ExecutionCapability::TensorizedMixed,
                        attention,
                        shape_class: 0,
                        operations,
                        request_pool_indices,
                        kv_placements,
                        kv_branch_placements,
                        latent_placements,
                    });
                    next_partition_id = next_partition_id.saturating_add(1);
                }
                next_submission_group = next_submission_group.saturating_add(1);
            }
            for (domain, operations) in homogeneous {
                let collective_seq = {
                    let value = self.next_collective_seq.max(1);
                    self.next_collective_seq = value.saturating_add(1);
                    value
                };
                let request_pool_indices = self.request_pool_indices(&operations);
                let kv_placements = self.kv_placements(&operations, kv_placements);
                let kv_branch_placements =
                    self.kv_branch_placements(&operations, kv_branch_placements);
                let latent_placements = self.latent_placements(&operations, latent_placements);
                partitions.push(BatchPartition {
                    partition_id: next_partition_id,
                    submission_group: next_submission_group,
                    collective_seq,
                    domain,
                    route,
                    execution: ExecutionCapability::DomainHomogeneous,
                    attention: partition_attention(&operations),
                    shape_class: 0,
                    operations,
                    request_pool_indices,
                    kv_placements,
                    kv_branch_placements,
                    latent_placements,
                });
                next_partition_id = next_partition_id.saturating_add(1);
                next_submission_group = next_submission_group.saturating_add(1);
            }
        }
        partitions
    }

    fn request_pool_indices(&self, operations: &[Operation]) -> Vec<u32> {
        operations
            .iter()
            .map(|operation| {
                self.running
                    .get(&operation.request_key.session_id)
                    .map(|state| state.request_pool_idx)
                    .expect("registered operation has a live request slot")
            })
            .collect()
    }

    fn kv_placements(
        &self,
        operations: &[Operation],
        placements: &HashMap<(RequestKey, OpId), Vec<KvPlacement>>,
    ) -> Vec<KvPlacement> {
        operations
            .iter()
            .filter(|operation| operation.kv_capacity_pages > 0)
            .flat_map(|operation| {
                placements
                    .get(&(operation.request_key, operation.op_id))
                    .into_iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect()
    }

    fn latent_placements(
        &self,
        operations: &[Operation],
        placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    ) -> Vec<LatentPlacement> {
        operations
            .iter()
            .filter_map(|operation| {
                placements
                    .get(&(operation.request_key, operation.op_id))
                    .cloned()
            })
            .collect()
    }

    fn kv_branch_placements(
        &self,
        operations: &[Operation],
        placements: &HashMap<(RequestKey, OpId), Vec<KvBranchPlacement>>,
    ) -> Vec<KvBranchPlacement> {
        operations
            .iter()
            .flat_map(|operation| {
                placements
                    .get(&(operation.request_key, operation.op_id))
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect()
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
        let all = self.bm.blocks_for_group(id, 0);
        let Some(st) = self.running.get_mut(&id) else {
            return Vec::new();
        };
        let sent = st.resources.blocks_sent.min(all.len());
        let new = all[sent..].to_vec();
        st.resources.blocks_sent = all.len();
        new
    }

    fn return_unsent_blocks(&mut self, id: RequestId, transition: &PlannedTransition) {
        if let Some(state) = self.running.get_mut(&id) {
            state.resources.blocks_sent = state
                .resources
                .blocks_sent
                .saturating_sub(transition.new_blocks.len());
        }
    }

    /// Build the next op for a running request, given the remaining per-step token
    /// `budget`. Returns `None` when the block budget cannot be satisfied.
    /// Data-plane causality gate: whether the request's next op may be dispatched
    /// given that its input tensors must be reachable on the worker that will run
    /// that will run it. Delegates to the executor, which is the `StageRouter`
    /// under a local staged topology and
    /// reports readiness from its `TensorMover`. Direct executors
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
        let committed_phase = self.running.get(&id)?.lifecycle.phase;
        let phase = if committed_phase == Phase::Prefill
            && self.has_inflight(id)
            && self.can_queue_decode_successor(id)
            && projection.prompt_cursor as usize >= self.running.get(&id)?.effective_prompt().len()
        {
            Phase::DecodeUnd
        } else {
            committed_phase
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
                let sampling_state = self.sampling_state(id, 0);
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::IngestText {
                        segment_index,
                        prompt_start: cursor as u32,
                        token_ids: chunk,
                        new_blocks,
                        sampling_state,
                    },
                )
            }
            Phase::DecodeUnd => {
                let projected_successor = self.has_inflight(id);
                // A successor registered before its predecessors resolve is
                // `inflight_len` unresolved points ahead of the committed cursor;
                // its minimum-token floor and force-finish flag are staged at
                // that exact projected point.
                let projected = if projected_successor {
                    self.inflight_len(id)
                } else {
                    0
                };
                let sampling_state = self.sampling_state(id, projected);
                let st = self.running.get(&id)?;
                // The prior committed token this decode continues from. It is
                // attached as a host input only when no exact selected-point
                // product is eligible for device continuation.
                let input_token = st.und.next_token;
                let projection = self.projected_cursor(id)?;
                let pos = projection.logical_pos;
                let relay_input = projected_successor || self.can_reuse_resolved_token_product(id);
                let capacity_target = self.decode_capacity_target(pos as usize, 0);
                if !self.bm.ensure_capacity(id, capacity_target) {
                    return None;
                }
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DecodeUnd {
                        position: pos,
                        new_blocks,
                        spec_token_ids: None,
                        sampling_state,
                        input_token,
                        relay_input,
                    },
                )
            }
            Phase::CloseKv => {
                let st = self.running.get(&id)?;
                let capacity_target =
                    self.decode_capacity_target(st.und.physical_kv_len as usize, 0);
                if !self.bm.ensure_capacity(id, capacity_target) {
                    return None;
                }
                let image_id = st.image_gen.image_id;
                let position = st.und.logical_pos;
                let physical_position = st.und.physical_kv_len;
                let token = st.und.next_token;
                let new_blocks = self.take_new_blocks(id);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::CloseKv {
                        image_id,
                        position,
                        physical_position,
                        token,
                        new_blocks,
                    },
                )
            }
            Phase::PublishKv => {
                let st = self.running.get(&id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::PublishKv {
                        image_id: st.image_gen.image_id,
                    },
                )
            }
            Phase::TransitionGen => {
                let st = self.running.get(&id)?;
                let conditioning = st.image_gen.conditioning.clone()?;
                let image_id = st.image_gen.image_id;
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::TransitionGen {
                        image_id,
                        latent_units,
                        conditioning,
                    },
                )
            }
            Phase::DenoiseGen => {
                // The flow runs exactly `image.steps` denoise quanta; once they
                // are all done, advance to the commit phase and plan its
                // transition rather than a spurious zero-length step. Termination
                // is host-driven off the committed step count, not a worker
                // completion flag.
                if self
                    .running
                    .get(&id)
                    .is_some_and(|st| st.image_gen.steps_done >= st.req.image.steps)
                {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.lifecycle.phase = Phase::CommitGen;
                    }
                    return self.next_transition(id, budget);
                }
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
                let remaining = st.req.image.steps.saturating_sub(timestep);
                let denoise_step_count = self.denoise_step_burst.max(1).min(remaining);
                let cfg = cfg_params(&st.req.image, cfg_branch_count(&st.req.image));
                let latent_units = self.worker_image_latent_units_for(st).max(1);
                let host_scratch_tokens = self.denoise_host_scratch_tokens(st);
                let image_id = st.image_gen.image_id;
                let conditioning = st.image_gen.conditioning.clone()?;
                let latent = st.image_gen.latent.clone()?;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::DenoiseGen {
                        image_id,
                        start_step: timestep,
                        step_count: denoise_step_count,
                        cfg,
                        latent_units,
                        host_scratch_tokens,
                        conditioning,
                        latent,
                    },
                )
            }
            Phase::CommitGen => {
                let st = self.running.get(&id)?;
                let image_id = st.image_gen.image_id;
                let latent = st.image_gen.latent.clone()?;
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::CommitGen { image_id, latent },
                )
            }
            Phase::FeedbackEncode => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let step_index = st.feedback.ingest_step;
                let step = feedback.ingest.steps.get(step_index).copied()?;
                let image_id = st.image_gen.image_id;
                let source = (feedback.source == uniserve_core::FeedbackSource::DeviceProduct)
                    .then(|| st.feedback.source_product.clone())
                    .flatten();
                let image_b64 = st.feedback.image_b64.clone().unwrap_or_default();
                let projection = self.projected_cursor(id)?;
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::EncodeFeedback {
                        image_id,
                        step_index,
                        step,
                        source,
                        image_b64,
                    },
                )
            }
            Phase::FeedbackState => {
                let st = self.running.get(&id)?;
                let feedback = st.req.policy.feedback.as_ref()?;
                let step_index = st.feedback.ingest_step;
                let is_final_step = step_index + 1 == feedback.ingest.steps.len();
                let image_id = st.image_gen.image_id;
                let logical_positions = feedback.ingest.logical_positions;
                let physical_kv_tokens = feedback.ingest.kv_effect(step_index)?;
                let feature = st.feedback.encoded_product.clone();
                let sample_continuation = is_final_step && feedback.sample_continuation;
                let Some(feature) = feature else {
                    self.finish(id, FinishReason::Error);
                    return None;
                };
                let projection = self.projected_cursor(id)?;
                let new_blocks = self.take_new_blocks(id);
                let sampling_state = self.sampling_state(id, 0);
                self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::FeedbackState {
                        image_id,
                        step_index,
                        is_final_step,
                        position: projection.logical_pos,
                        logical_positions,
                        physical_kv_tokens,
                        feature,
                        sample_continuation,
                        new_blocks,
                        sampling_state,
                    },
                )
            }
            Phase::IngestState => self.next_context_ingest_transition(id, budget, projection),
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
            if state.lifecycle.phase == Phase::IngestState {
                let feature = state.ingest.encoded_product.clone()?;
                let physical_kv_tokens = image.ingest.kv_effect(step_index)?;
                let physical_bound = match physical_kv_tokens {
                    uniserve_core::ImageKvEffect::Exact { tokens } => tokens,
                    uniserve_core::ImageKvEffect::Bounded { max_tokens } => max_tokens,
                    uniserve_core::ImageKvEffect::WorkerDefined => match step {
                        uniserve_core::ImageIngestStep::VaeEncode => {
                            self.cap_max_vae_grid_tokens().min(u32::MAX as usize) as u32
                        }
                        uniserve_core::ImageIngestStep::VitEncode => self.caps.max_vit_grid_tokens,
                    },
                };
                if !self.bm.ensure_capacity(
                    id,
                    projection.physical_kv_len.saturating_add(physical_bound) as usize,
                ) {
                    return None;
                }
                let new_blocks = self.take_new_blocks(id);
                return self.plan_intent(
                    id,
                    projection,
                    TransitionIntent::IngestImageState {
                        segment_index: image.segment_index,
                        step_index,
                        is_final_step,
                        position: projection.logical_pos,
                        logical_positions: image.ingest.logical_positions,
                        physical_kv_tokens,
                        feature,
                        new_blocks,
                    },
                );
            }
            let cache_read = state.req.cache.read;
            let cache_write = state.req.cache.write;
            let cache_key = encoder_cache_key(image.hash, step_index, step);
            let cached = if cache_read {
                self.enc_cache.lookup_product(cache_key)
            } else {
                None
            };
            if let Some(cached_product) = cached {
                let product = self.enc_cache.acquire(cache_key)?;
                if product != cached_product {
                    return None;
                }
                let Some(state) = self.running.get_mut(&id) else {
                    let _ = self.enc_cache.release(cache_key, &product);
                    return None;
                };
                state.ingest.acquired_encoder_pins.push(EncoderCachePin {
                    key: cache_key,
                    product: product.clone(),
                });
                state.ingest.encoded_product = Some(product);
                state.lifecycle.phase = Phase::IngestState;
                return self.next_context_ingest_transition(id, budget, projection);
            }
            let persistent_cache_key = cache_write.then_some(cache_key);
            return self.plan_intent(
                id,
                projection,
                TransitionIntent::EncodeImage {
                    segment_index: image.segment_index,
                    step_index,
                    step,
                    encoder_cache_key: persistent_cache_key,
                    image_b64: image.b64.clone(),
                    source_product: None,
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
        let sampling_state = self.sampling_state(id, 0);
        let new_blocks = self.take_new_blocks(id);
        self.plan_intent(
            id,
            projection,
            TransitionIntent::IngestText {
                segment_index,
                prompt_start: cursor as u32,
                token_ids: prompt[cursor..end].to_vec(),
                new_blocks,
                sampling_state,
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

    /// Compute the operation's allowed/suppress masks from the host-side
    /// logits-processor pipeline (minimum-token floor, bad-words, allowed
    /// tokens). `n_generated` is the count of generated tokens the sampled point
    /// follows: for a successor registered before its predecessors are observed
    /// it is the committed count plus the unresolved window depth, so the
    /// minimum-token floor is evaluated at the successor's own exact point
    /// without reading any device token.
    fn token_masks(
        &mut self,
        id: RequestId,
        n_generated: usize,
    ) -> (Option<Vec<u32>>, Option<Vec<u32>>) {
        let st = match self.running.get(&id) {
            Some(s) => s,
            None => return (None, None),
        };
        if self.cpu_continuation_required(st) {
            return st
                .cpu_masks
                .as_ref()
                .map(|masks| (masks.allowed.clone(), masks.suppress.clone()))
                .unwrap_or((Some(Vec::new()), None));
        }
        let ctx = crate::logits::ProcCtx {
            n_generated,
            eos: &self.ctrl.eos,
            generated: &st.replay.generated_ids,
            sampling: &st.req.sampling,
        };
        let (allowed, mut suppress) = crate::logits::run_pipeline(&self.logits_pipeline, &ctx);
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
        (allowed, suppress)
    }

    /// The branch-local static sampling state staged for one operation.
    ///
    /// `projected` is the number of unresolved predecessors the operation is
    /// registered behind: zero for a host-paced operation, the in-flight window
    /// depth for a successor registered before its predecessors are observed.
    /// The minimum-token floor and the force-finish flag are evaluated at the
    /// operation's own exact point (`tokens_emitted + projected`). Penalty
    /// counts are never staged here — they are a device-resident committed base
    /// plus per-operation deltas the worker folds on commit, so no host token
    /// history participates in a successor's penalty input.
    fn sampling_state(&mut self, id: RequestId, projected: usize) -> SamplingState {
        let n_generated = self.running.get(&id).map_or(0, |state| {
            state.und.tokens_emitted.saturating_add(projected)
        });
        let (allowed_token_ids, suppressed_token_ids) = self.token_masks(id, n_generated);
        let Some(state) = self.running.get(&id) else {
            return SamplingState::default();
        };
        let finish_token_ids = canonical_continuation_stop_token_ids(&state.req, &self.ctrl.eos);
        SamplingState {
            allowed_token_ids,
            suppressed_token_ids: suppressed_token_ids.unwrap_or_default(),
            finish_token_ids,
            force_finish: n_generated.saturating_add(1) >= state.req.max_und_tokens,
        }
    }

    fn invalidate_cpu_masks(&mut self, id: RequestId) {
        let required = self
            .running
            .get(&id)
            .is_some_and(|state| self.cpu_continuation_required(state));
        if required && let Some(state) = self.running.get_mut(&id) {
            state.cpu_masks = None;
        }
    }

    fn resolve_decode_text(
        &mut self,
        id: RequestId,
        view: SequenceView,
        prefix_versions: &[VersionRef],
    ) {
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
        let burst_result = view.committed_tokens.len() > 1;
        let tokens = if view.committed_tokens.is_empty() {
            vec![self.ctrl.eos[0]]
        } else {
            view.committed_tokens.clone()
        };
        for (idx, tok) in tokens.iter().copied().enumerate() {
            let is_last = idx + 1 == tokens.len();
            let logprob = is_last.then_some(view.sampled_logprob).flatten();
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
            // Once the image budget is spent the image-start trigger is
            // suppressed host-side: emit a non-trigger token so a bias toward the
            // trigger returns to ordinary text instead of an un-actionable signal.
            let tok = if direct_trigger {
                tok.wrapping_add(1)
            } else {
                tok
            };
            let top_logprobs = is_last.then(|| view.top_logprobs.clone());
            if self.emit_or_finish_und_token(
                id,
                tok,
                logprob,
                top_logprobs,
                is_last,
                prefix_versions.get(idx),
            ) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.und.next_token = tok;
                st.lifecycle.phase = Phase::DecodeUnd;
                st.und.round_tokens.push(tok);
            }
            self.invalidate_cpu_masks(id);
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
        operation: Operation,
        apply: SchedulerApply,
        mut view: SequenceView,
        prefix_versions: Vec<VersionRef>,
    ) {
        let operation_variant = operation.work.variant();
        if !view.prompt_logprobs.is_empty() {
            let positions = std::mem::take(&mut view.prompt_logprobs);
            self.resolve_prompt_logprobs(id, positions);
        }
        if operation_variant == WorkVariant::TokenDecode {
            return self.resolve_decode_text(id, view, &prefix_versions);
        }
        match operation_variant {
            WorkVariant::TokenExtend => {
                self.bm.activate(id);
                match &apply.delta {
                    crate::generation::TransitionDelta::CloseKv { .. } => return,
                    crate::generation::TransitionDelta::IngestImageState {
                        is_final_step, ..
                    } => {
                        if *is_final_step {
                            self.release_transient_products(id);
                        }
                        if *is_final_step
                            && self.running.get(&id).is_some_and(|st| {
                                st.ingest.mm_cursor >= st.context.images.len()
                                    && st.ingest.prompt_cursor >= st.context.prompt_ids.len() as u32
                            })
                        {
                            let bos = self.ctrl.bos;
                            if let Some(st) = self.running.get_mut(&id) {
                                st.und.next_token = bos;
                                st.und.round_tokens.clear();
                                st.lifecycle.phase = Phase::DecodeUnd;
                            }
                        }
                        return;
                    }
                    crate::generation::TransitionDelta::FeedbackState { is_final_step, .. } => {
                        if !is_final_step {
                            return;
                        }
                        self.release_transient_products(id);
                        let sample_continuation = self
                            .running
                            .get(&id)
                            .and_then(|st| st.req.policy.feedback.as_ref())
                            .is_some_and(|feedback| feedback.sample_continuation);
                        if let Some(st) = self.running.get_mut(&id) {
                            st.image_gen.images_done += 1;
                            st.und.text_since_image = 0;
                            st.und.round_tokens.clear();
                        }
                        if !sample_continuation {
                            let Some(next_token) = self.feedback_next_token(id) else {
                                return self.finish(id, FinishReason::Error);
                            };
                            if let Some(st) = self.running.get_mut(&id) {
                                st.und.next_token = next_token;
                                st.lifecycle.phase = Phase::DecodeUnd;
                            }
                            return;
                        }
                        let Some(tok) = view.committed_tokens.last().copied() else {
                            return self.finish(id, FinishReason::Error);
                        };
                        let (can_open_gen_branch, images_done, max_images) = {
                            let st = self.running.get_mut(&id).unwrap();
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
                            .is_some_and(|state| Self::direct_trigger_matches(state, tok));
                        if direct_trigger && can_open_gen_branch && images_done < max_images {
                            self.begin_image(id);
                            return;
                        }
                        let tok = if direct_trigger {
                            tok.wrapping_add(1)
                        } else {
                            tok
                        };
                        if self.emit_or_finish_und_token(
                            id,
                            tok,
                            view.sampled_logprob,
                            Some(view.top_logprobs.clone()),
                            true,
                            prefix_versions.last(),
                        ) {
                            return;
                        }
                        if let Some(st) = self.running.get_mut(&id) {
                            st.und.next_token = tok;
                            st.lifecycle.phase = Phase::DecodeUnd;
                            st.und.round_tokens.push(tok);
                        }
                        self.invalidate_cpu_masks(id);
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
                    _ => {}
                }
                // chunked prefill: a prefill op may only have consumed part of
                // the prompt; if so, advance the cursor and stay in Prefill.
                let (cursor, prompt_len) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.ingest.prompt_cursor as usize,
                        st.effective_prompt().len(),
                    )
                };
                if cursor < prompt_len {
                    return;
                }
                if self.running.get(&id).is_some_and(|st| {
                    st.ingest.mm_cursor < st.context.images.len()
                        || st.ingest.prompt_cursor < st.context.prompt_ids.len() as u32
                }) {
                    return;
                }
                let (starts_gen_after_context, can_open_gen_branch) = {
                    let st = self.running.get_mut(&id).unwrap();
                    st.und.tokens_emitted += 1;
                    (st.starts_gen_after_context(), st.can_open_gen_branch())
                };
                if self.running.get(&id).is_some_and(|state| {
                    state.req.sampling.prompt_logprobs_requested()
                        && state.ingest.prompt_logprobs_emitted
                            != state.context.prompt_ids.len().saturating_sub(1)
                }) {
                    tracing::error!(
                        request_id = id.0,
                        "prompt logprob scoring ended before every prompt position was resolved"
                    );
                    return self.finish(id, FinishReason::Error);
                }
                // the prompt is fully prefilled now — publish its full
                // blocks to the prefix cache for later requests to reuse.
                let bs = self.caps.block_size as usize;
                if let Some(st) = self.running.get_mut(&id) {
                    self.prefix_cache.cache_blocks(st, &mut self.bm, bs);
                }
                // A description-lowered prefix may already end at a branch trigger.
                // Treat that boundary exactly like a sampled trigger.
                if self.prefilled_gen_trigger(id) {
                    self.begin_image(id);
                    return;
                }
                // Immediate Gen-only profiles skip Und decode after context prep.
                if starts_gen_after_context {
                    self.begin_image(id);
                    return;
                }
                let tok = view
                    .committed_tokens
                    .last()
                    .copied()
                    .unwrap_or(self.ctrl.eos[0]);
                let logprob = view.sampled_logprob;
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
                if self.emit_or_finish_und_token(
                    id,
                    tok,
                    logprob,
                    Some(view.top_logprobs.clone()),
                    true,
                    prefix_versions.last(),
                ) {
                    return;
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.und.next_token = tok;
                    st.lifecycle.phase = Phase::DecodeUnd;
                }
                self.invalidate_cpu_masks(id);
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
            WorkVariant::GenFlow => {
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = match apply.delta {
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
                // The commit phase is entered host-side once the committed step
                // count reaches `image.steps` (see the `Phase::DenoiseGen`
                // planner); a worker completion flag does not drive termination.
            }
            WorkVariant::Materialize => {
                let image_id = self.running.get(&id).map_or(0, |st| st.image_gen.image_id);
                self.emit(id, GenEvent::ImageCommit { image_id });
                let image = view.image_png.clone();
                if let Some(image_b64) = image.clone() {
                    let Some(event) = image_done_event(image_id, image_b64) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let root = self.fixed_version(id);
                    self.emit_visible(id, event, root.as_ref(), PublicModality::Image);
                }
                self.bm.activate(id);
                let (continues_after_gen_commit, feedback_source) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.continues_after_gen_commit(),
                        st.req
                            .policy
                            .feedback
                            .as_ref()
                            .map(|feedback| feedback.source.clone()),
                    )
                };
                if continues_after_gen_commit {
                    let Some(feedback_source) = feedback_source else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let source_product = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            product.storage_class == uniserve_worker_wire::StorageClass::LatentArena
                                && product.kind == uniserve_worker_wire::ProductKind::Artifact
                        })
                        .cloned();
                    if feedback_source == uniserve_core::FeedbackSource::DeviceProduct
                        && source_product.is_none()
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    if feedback_source == uniserve_core::FeedbackSource::ArtifactProduct
                        && image.is_none()
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    if let Some(st) = self.running.get_mut(&id) {
                        st.feedback.image_b64 = image;
                        st.feedback.ingest_step = 0;
                        st.feedback.source_product = source_product;
                        st.feedback.encoded_product = None;
                        st.lifecycle.phase = Phase::FeedbackEncode;
                        if let Some(product) = st.feedback.source_product.clone() {
                            st.ingest.transient_encoder_products.push(product);
                        }
                    }
                } else {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.image_gen.images_done += 1;
                        st.und.text_since_image = 0;
                    }
                    self.finish(id, FinishReason::ImageDone);
                }
            }
            WorkVariant::EncodeVision | WorkVariant::EncodeLatent => match &apply.delta {
                crate::generation::TransitionDelta::EncodeImageStep {
                    encoder_cache_key, ..
                } => {
                    let Some(feature) = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            matches!(
                                product.kind,
                                uniserve_worker_wire::ProductKind::VisionFeature
                                    | uniserve_worker_wire::ProductKind::LatentFeature
                            )
                        })
                        .cloned()
                    else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let result_handle = u64::from(feature.generation);
                    if result_handle == 0
                        || view.encode_generation.map(u64::from) != Some(result_handle)
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    let mut free_products = Vec::new();
                    let selected_product = if let Some(cache_key) = encoder_cache_key {
                        if let Some(freed) = self.enc_cache.insert(*cache_key, feature.clone()) {
                            free_products.push(freed);
                        }
                        let Some(product) = self.enc_cache.acquire(*cache_key) else {
                            return self.finish(id, FinishReason::Error);
                        };
                        if let Some(st) = self.running.get_mut(&id) {
                            st.ingest.acquired_encoder_pins.push(EncoderCachePin {
                                key: *cache_key,
                                product: product.clone(),
                            });
                        }
                        product
                    } else if let Some(st) = self.running.get_mut(&id) {
                        st.ingest.transient_encoder_products.push(feature.clone());
                        feature.clone()
                    } else {
                        feature.clone()
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.ingest.encoded_product = Some(selected_product);
                        st.lifecycle.phase = Phase::IngestState;
                    }
                    if !free_products.is_empty() {
                        self.release_products(free_products);
                    }
                }
                crate::generation::TransitionDelta::EncodeFeedbackStep { .. } => {
                    let Some(feature) = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            matches!(
                                product.kind,
                                uniserve_worker_wire::ProductKind::VisionFeature
                                    | uniserve_worker_wire::ProductKind::LatentFeature
                            )
                        })
                        .cloned()
                    else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let handle = u64::from(feature.generation);
                    if handle == 0 || view.encode_generation.map(u64::from) != Some(handle) {
                        return self.finish(id, FinishReason::Error);
                    }
                    if let Some(st) = self.running.get_mut(&id) {
                        st.ingest.transient_encoder_products.push(feature.clone());
                        st.feedback.encoded_product = Some(feature);
                        st.lifecycle.phase = Phase::FeedbackState;
                    }
                }
                _ => self.finish(id, FinishReason::Error),
            },
            WorkVariant::TokenDecode
            | WorkVariant::TokenVerify
            | WorkVariant::Draft
            | WorkVariant::GenTransition
            | WorkVariant::TransferProduct
            | WorkVariant::TransferKvPublish
            | WorkVariant::TransferKvInstall => {}
        }
    }

    fn resolve_prompt_logprobs(&mut self, id: RequestId, positions: Vec<Vec<RankedToken>>) {
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

    fn release_transient_products(&mut self, id: RequestId) {
        let products = self
            .running
            .get_mut(&id)
            .map(|state| std::mem::take(&mut state.ingest.transient_encoder_products))
            .unwrap_or_default();
        self.release_products(products);
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
        let allocated_blocks = self.bm.blocks_for_group(id, 0).len();
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
            "free_blocks": self.bm.free_request_pages(),
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
            st.lifecycle.phase = Phase::CloseKv;
            st.image_gen.branch_pending = true;
        }
        self.promote_gen_branch_reservation(id);
    }

    fn flush_output_journals(&mut self) -> bool {
        let mut progressed = false;
        for state in self.running.values_mut() {
            if state.event_tx.is_closed() {
                state.cancelled = true;
                continue;
            }
            let before = state.output_journal.len();
            if flush_public_journal(&state.event_tx, &mut state.output_journal) {
                state.cancelled = true;
            }
            progressed |= state.output_journal.len() != before;
        }
        self.completed_outputs.retain(|_, output| {
            let before = output.journal.len();
            let closed = flush_public_journal(&output.event_tx, &mut output.journal);
            progressed |= output.journal.len() != before;
            !closed && !output.event_tx.is_closed() && !output.journal.is_empty()
        });
        progressed
    }

    fn emit(&mut self, id: RequestId, ev: GenEvent) {
        if let Some(st) = self.running.get_mut(&id) {
            if enqueue_public_event(&st.event_tx, &mut st.output_journal, ev) {
                st.cancelled = true;
            } else {
                st.public_event_seq = st.public_event_seq.saturating_add(1);
            }
        }
    }

    fn emit_visible(
        &mut self,
        id: RequestId,
        mut event: GenEvent,
        root: Option<&VersionRef>,
        modality: PublicModality,
    ) {
        let root = root.cloned().or_else(|| self.fixed_version(id));
        let Some(VersionRef {
            producer_op_id,
            point:
                Point::Fixed {
                    point_index,
                    semantic_digest,
                },
            ..
        }) = root
        else {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        };
        let event_seq = self
            .running
            .get(&id)
            .map_or(1, |state| state.public_event_seq.saturating_add(1));
        let commit = PublicCommit {
            event_seq,
            modality,
            committed_at: uniserve_core::now_monotonic_secs(),
            semantic_root: SemanticRoot {
                producer_op_id: producer_op_id.0,
                point_index,
                semantic_digest,
            },
        };
        match &mut event {
            GenEvent::TextToken { public_commit, .. }
            | GenEvent::ImageDone { public_commit, .. } => *public_commit = Some(commit),
            _ => {
                self.finish_after_inflight(id, FinishReason::Error, None);
                return;
            }
        }
        self.emit(id, event);
    }

    /// Emit a text token and retain its semantic history.
    fn emit_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        root: Option<&VersionRef>,
    ) -> bool {
        let action = if let Some(st) = self.running.get_mut(&id) {
            st.replay.generated_ids.push(tok);
            st.und.text_since_image = st.und.text_since_image.saturating_add(1);
            st.req.behavior.und_tokens
        } else {
            return false;
        };
        match action {
            uniserve_core::UndTokenAction::Emit => {
                let before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.public_event_seq);
                self.emit_visible(
                    id,
                    GenEvent::TextToken {
                        id: tok,
                        logprob,
                        public_commit: None,
                    },
                    root,
                    PublicModality::Text,
                );
                if let Some(state) = self.running.get_mut(&id)
                    && state.public_event_seq > before
                {
                    state.public_token_seq = state.public_token_seq.saturating_add(1);
                    if let Some(root) = root {
                        state.trace.stamp_existing(
                            root.producer_op_id,
                            crate::trace::LifecyclePhase::PubliclyCommitted,
                            uniserve_core::now_monotonic_us(),
                        );
                    }
                }
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
        top_logprobs: Option<Vec<RankedToken>>,
        root: Option<&VersionRef>,
    ) {
        if self.emit_text(id, tok, logprob, root)
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
        top_logprobs: Option<Vec<RankedToken>>,
        root: Option<&VersionRef>,
    ) {
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.req.policy.termination.emit_stop_token)
        {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
        }
    }

    fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<RankedToken>>,
        sampled: bool,
        root: Option<&VersionRef>,
    ) -> bool {
        let Some(state) = self.running.get(&id) else {
            return true;
        };
        // `tokens_emitted` already counts the token being resolved, so the floor
        // is cleared only once at least `min_tokens` tokens have been emitted.
        let generated = state.und.tokens_emitted;
        let under_floor = generated <= state.req.sampling.min_tokens;
        let stop_hit = state.req.policy.termination.stop_finishes
            && !under_floor
            && state.req.stop_token_ids.contains(&token_id);
        let eos_hit = state.req.policy.termination.eos_finishes
            && self.ctrl.eos.contains(&token_id)
            && !state.req.sampling.ignore_eos
            && !under_floor;
        let max_hit = generated >= state.req.max_und_tokens;

        if stop_hit {
            self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs, root);
            self.finish_after_inflight(id, FinishReason::Stop, Some(format!("token:{token_id}")));
            return true;
        }
        if eos_hit || max_hit {
            if !self.ctrl.eos.contains(&token_id) {
                if sampled {
                    self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
                } else {
                    self.emit_text(id, token_id, logprob, root);
                }
            }
            self.finish_after_inflight(
                id,
                if max_hit {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Eos
                },
                None,
            );
            return true;
        }
        if sampled {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
        } else {
            self.emit_text(id, token_id, logprob, root);
        }
        !self.running.contains_key(&id)
    }

    fn emit_st(&self, st: &mut ReqState, ev: GenEvent) {
        if enqueue_public_event(&st.event_tx, &mut st.output_journal, ev) {
            st.cancelled = true;
        } else {
            st.public_event_seq = st.public_event_seq.saturating_add(1);
        }
    }

    fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    fn finish_after_inflight(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<String>,
    ) {
        let semantic_pending = !matches!(reason, FinishReason::Error)
            && self
                .running
                .get(&id)
                .is_some_and(|state| !state.pending_commits.is_empty());
        if !self.has_inflight(id) && !semantic_pending {
            self.finish_with(id, reason, stop_reason);
            return;
        }
        if !self.pending_finishes.contains_key(&id) || matches!(reason, FinishReason::Error) {
            self.pending_finishes.insert(
                id,
                PendingFinish {
                    reason,
                    stop_reason,
                },
            );
        }
    }

    fn finish_pending_if_idle(&mut self, id: RequestId) {
        if self.has_inflight(id)
            || self
                .running
                .get(&id)
                .is_some_and(|state| !state.pending_commits.is_empty())
        {
            return;
        }
        if let Some(pending) = self.pending_finishes.remove(&id) {
            self.finish_with(id, pending.reason, pending.stop_reason);
        }
    }

    fn finish_with(&mut self, id: RequestId, reason: FinishReason, stop_reason: Option<String>) {
        self.pending_finishes.remove(&id);
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
        let mut awaits_close = false;
        let mut request_pool_idx = None;
        if let Some(mut st) = self.running.remove(&id) {
            request_pool_idx = Some(st.request_pool_idx);
            if let Some(key) = st.cpu_pending.take() {
                self.cpu_deadlines.remove(&key);
            }
            if st.admission_digest.is_some() {
                let request_key = RequestKey::new(self.authority_id, id, st.epoch);
                let cutoff = st.cancel_cutoff.clone().unwrap_or_else(|| VersionRef {
                    request_key,
                    producer_op_id: OpId(st.committed_producer_op_id),
                    point: Point::Fixed {
                        point_index: st.committed_version as u32,
                        semantic_digest: st.committed_semantic.clone(),
                    },
                });
                st.control_seq = st.control_seq.saturating_add(1);
                self.pending_controls.push_back(Control::Close {
                    request_key,
                    control_seq: st.control_seq,
                    cutoff,
                    reason: close_reason(&reason),
                });
                self.retiring_sessions.insert(
                    id,
                    RetiringSession {
                        request_key,
                        request_pool_idx: st.request_pool_idx,
                    },
                );
                awaits_close = true;
            }
            self.order.retain(|x| *x != id);
            self.reserved_encoder_entries = self
                .reserved_encoder_entries
                .saturating_sub(st.req.resources.encoder_cache_keys.len());
            if st.resources.reserve_worstcase {
                self.reserved_blocks = self
                    .reserved_blocks
                    .saturating_sub(st.resources.worstcase_blocks);
            }
            let mut free_encoder_products =
                std::mem::take(&mut st.ingest.transient_encoder_products);
            for pin in &st.ingest.acquired_encoder_pins {
                if let Some(product) = self.enc_cache.release(pin.key, &pin.product) {
                    free_encoder_products.push(product);
                }
            }
            self.release_products(free_encoder_products);
            // close + archive the lifecycle trace (reconstructable post-finish).
            st.trace.mark_finished(
                finish_reason_str(&reason),
                uniserve_core::now_monotonic_us(),
            );
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
            let terminal = GenEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: st.und.tokens_emitted,
                images: st.image_gen.images_done,
            };
            let closed = enqueue_public_event(&st.event_tx, &mut st.output_journal, terminal);
            if !closed {
                st.public_event_seq = st.public_event_seq.saturating_add(1);
                if !st.output_journal.is_empty() {
                    self.completed_outputs.insert(
                        id,
                        RetiredOutput {
                            event_tx: st.event_tx,
                            journal: st.output_journal,
                        },
                    );
                }
            }
        }
        if !awaits_close {
            self.bm.release(id);
            self.latent_pages.release(id);
            self.kv_scratch_pages.release(id);
            if let Some(index) = request_pool_idx
                && let Err(error) = self.request_slots.release(index)
            {
                tracing::error!(
                    request_id = id.0,
                    request_pool_idx = index,
                    error,
                    "failed to release scheduler request slot"
                );
                self.fatal = true;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct KvLengths {
    prefix: u32,
    input: u32,
    visible: u32,
    resulting: u32,
}

fn transition_kv_lengths(delta: &TransitionDelta, state: Option<&ReqState>) -> Option<KvLengths> {
    let (prefix, input) = match delta {
        TransitionDelta::IngestText {
            start,
            end,
            physical_start,
            ..
        } => (*physical_start, end.saturating_sub(*start)),
        TransitionDelta::IngestImageState {
            physical_start,
            physical_kv_tokens,
            ..
        }
        | TransitionDelta::FeedbackState {
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
        TransitionDelta::DecodeUnd {
            physical_position, ..
        }
        | TransitionDelta::CloseKv {
            physical_position, ..
        } => (*physical_position, 1),
        TransitionDelta::PublishKv { .. } => (state?.und.physical_kv_len, 0),
        TransitionDelta::TransitionGen { .. } | TransitionDelta::DenoiseGen { .. } => {
            (state?.und.physical_kv_len, 0)
        }
        TransitionDelta::EncodeImageStep { .. }
        | TransitionDelta::CommitGen { .. }
        | TransitionDelta::EncodeFeedbackStep { .. } => return None,
    };
    Some(KvLengths {
        prefix,
        input,
        visible: prefix,
        resulting: prefix.saturating_add(input),
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
fn planned_op_token_cost(transition: &PlannedTransition) -> usize {
    if transition.operation_variant != WorkVariant::GenFlow {
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

fn tensorized_mixed_runner_work(variant: WorkVariant) -> bool {
    matches!(
        variant,
        WorkVariant::TokenExtend
            | WorkVariant::TokenDecode
            | WorkVariant::TokenVerify
            | WorkVariant::Draft
            | WorkVariant::GenFlow
    )
}

fn partition_attention(operations: &[Operation]) -> AttentionRegime {
    let regimes = operations
        .iter()
        .map(|operation| match operation.work.variant() {
            WorkVariant::TokenExtend
            | WorkVariant::TokenDecode
            | WorkVariant::TokenVerify
            | WorkVariant::Draft => AttentionRegime::Causal,
            WorkVariant::GenFlow => AttentionRegime::Hybrid,
            WorkVariant::GenTransition
            | WorkVariant::EncodeVision
            | WorkVariant::EncodeLatent
            | WorkVariant::TransferProduct
            | WorkVariant::TransferKvPublish
            | WorkVariant::TransferKvInstall
            | WorkVariant::Materialize => AttentionRegime::None,
        })
        .collect::<HashSet<_>>();
    if regimes.len() == 1 {
        *regimes.iter().next().unwrap_or(&AttentionRegime::None)
    } else {
        AttentionRegime::Hybrid
    }
}

fn transition_output_bound(transition: &PlannedTransition) -> usize {
    match transition.operation_variant {
        WorkVariant::TokenVerify => transition
            .validation
            .expected_text_tokens
            .map_or(4, |range| {
                usize::try_from(range.max)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(2)
                    .saturating_add(2)
            }),
        WorkVariant::TokenExtend | WorkVariant::TokenDecode => 4,
        WorkVariant::GenFlow => transition.token_cost.saturating_add(2),
        WorkVariant::Materialize => 3,
        _ => 2,
    }
}

/// Flush as many ordered journal entries as the immediate consumer can accept.
/// Returns true when the consumer has closed.
fn flush_public_journal(event_tx: &EventTx, journal: &mut VecDeque<GenEvent>) -> bool {
    while let Some(event) = journal.pop_front() {
        match event_tx.send(event) {
            Ok(()) => {}
            Err(uniserve_engine_api::EventSendError::Full(event)) => {
                journal.push_front(*event);
                return false;
            }
            Err(uniserve_engine_api::EventSendError::Closed(_)) => {
                journal.clear();
                return true;
            }
        }
    }
    false
}

/// Append one event to the bounded public journal without changing its order.
/// Returns true when the consumer has closed.
fn enqueue_public_event(
    event_tx: &EventTx,
    journal: &mut VecDeque<GenEvent>,
    event: GenEvent,
) -> bool {
    if flush_public_journal(event_tx, journal) {
        return true;
    }
    if journal.is_empty() {
        match event_tx.send(event) {
            Ok(()) => return false,
            Err(uniserve_engine_api::EventSendError::Closed(_)) => return true,
            Err(uniserve_engine_api::EventSendError::Full(event)) => {
                journal.push_back(*event);
            }
        }
    } else {
        journal.push_back(event);
    }
    assert!(
        journal.len() <= OUTPUT_JOURNAL_CAPACITY,
        "scheduler exceeded the bounded public output journal"
    );
    false
}

fn operation_trace(operation: &Operation, apply: &SchedulerApply) -> serde_json::Value {
    let parent_kind = match operation.parent.point {
        Point::Fixed { .. } => "fixed",
        Point::Device { .. } => "device",
    };
    json!({
        "work": operation.work.variant().as_wire_str(),
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
    use super::*;
    use crate::generation::Replayability;
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

        let stops = canonical_continuation_stop_token_ids(&req, &[151_643, 151_645]);

        assert_eq!(stops, vec![4_242, 151_643, 151_645]);
    }

    fn cursor(phase: Phase, pos: u32) -> CursorProjection {
        CursorProjection {
            phase,
            prompt_cursor: pos,
            logical_pos: pos,
            physical_kv_len: pos,
            replayability: Replayability::Replayable,
        }
    }

    fn plan(
        req: &GenerationRequest,
        cursor: CursorProjection,
        intent: TransitionIntent,
    ) -> PlannedTransition {
        GenerationPlanner::new()
            .plan(req, cursor, intent)
            .expect("plan transition")
    }

    #[test]
    fn planned_cost_tracks_physical_sequence_work() {
        let req = request(1, 11);
        let extend = plan(
            &req,
            cursor(Phase::Prefill, 3),
            TransitionIntent::IngestText {
                segment_index: 0,
                prompt_start: 3,
                token_ids: vec![7; 8],
                new_blocks: Vec::new(),
                sampling_state: SamplingState::default(),
            },
        );
        assert_eq!(planned_op_token_cost(&extend), 8);

        let decode = plan(
            &req,
            cursor(Phase::DecodeUnd, 11),
            TransitionIntent::DecodeUnd {
                position: 11,
                new_blocks: Vec::new(),
                spec_token_ids: None,
                sampling_state: SamplingState::default(),
                input_token: 5,
                relay_input: false,
            },
        );
        assert_eq!(planned_op_token_cost(&decode), 1);

        let verify = plan(
            &req,
            cursor(Phase::DecodeUnd, 11),
            TransitionIntent::DecodeUnd {
                position: 11,
                new_blocks: Vec::new(),
                spec_token_ids: Some(vec![8, 9, 10]),
                sampling_state: SamplingState::default(),
                input_token: 5,
                relay_input: false,
            },
        );
        assert_eq!(planned_op_token_cost(&verify), 4);
    }
}
