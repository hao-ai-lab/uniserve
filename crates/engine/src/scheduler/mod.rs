//! Scheduler control loop: one owner thread drives the whole loop (single-owner,
//! no locks on engine state) — drain commands, advance each running request's
//! generation lifecycle, admit pending requests against the block budget,
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

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod admission;
mod batching;
pub(crate) mod bench_trace;
mod control;
pub(crate) mod cpu_continuation;
mod execution;
pub mod generation;
pub(crate) mod image_artifact;
pub mod logits;
pub mod policy;
pub(crate) mod prefix_cache;
mod publication;
pub mod queue;
pub mod stats_report;
pub mod trace;

pub use crate::executor::{ControlAck, ControlOp, Executor};
pub use logits::{LogitsProcessor, MaskContribution, ProcCtx, ProcessorDeclaration};
pub use policy::{DecisionLog, LatencyHistory, PolicyDecision, PolicyReason, PolicySnapshot};
pub use queue::{FcfsRequestQueue, PriorityRequestQueue, RequestQueue};
pub use stats_report::SchedStatsReporter;
pub use trace::{LifecyclePhase, OperationKey, OperationLifecycle, RequestTrace};

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use std::env;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::scheduler::cpu_continuation::{CpuContinuationPool, CpuMasks, CpuTask, CpuTaskKey};
use crate::scheduler::generation::{
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

/// Cumulative accounting for one physical execution domain.
#[derive(Default)]
pub struct DomainStats {
    pub active_credits: AtomicUsize,
    pub peak_credits: AtomicUsize,
    pub launched_operations: AtomicU64,
    pub completed_operations: AtomicU64,
    pub predicated_operations: AtomicU64,
    pub error_operations: AtomicU64,
    pub backpressure_events: AtomicU64,
    pub reclaimed_credits: AtomicU64,
    pub completed_partitions: AtomicU64,
    pub semantic_commits: AtomicU64,
    pub public_commits: AtomicU64,
    pub co_resident_partitions: AtomicU64,
    pub queue_us: AtomicU64,
    pub launch_us: AtomicU64,
    pub device_us: AtomicU64,
    pub completion_us: AtomicU64,
    pub semantic_commit_us: AtomicU64,
    pub public_commit_us: AtomicU64,
    pub co_resident_us: AtomicU64,
}

/// Domain-indexed scheduler accounting shared with the stats reporter.
#[derive(Default)]
pub struct ExecutionDomainStats {
    pub prefill: DomainStats,
    pub decode: DomainStats,
    pub flow: DomainStats,
}

impl ExecutionDomainStats {
    pub fn get(&self, domain: uniserve_worker_ipc::Domain) -> &DomainStats {
        match domain {
            uniserve_worker_ipc::Domain::Prefill => &self.prefill,
            uniserve_worker_ipc::Domain::Decode => &self.decode,
            uniserve_worker_ipc::Domain::Flow => &self.flow,
        }
    }
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
    pub domains: ExecutionDomainStats,
    pub worker: WorkerStats,
}

use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EventRx, EventSendError, EventTx, MediaEventTx, event_channel,
};
use crate::kv::{BlockPool, BlockTable, EncoderCacheManager, KvCacheCoordinator};
use crossbeam_channel::Receiver;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{BlockId, CfgParams, ImageIngestStep, encoder_cache_key};
use uniserve_core::{
    FinishReason, GenEvent, GenerationRequest, MediaEvent, MediaRequest, PositionLogprobs,
    PublicCommit, PublicModality, SemanticRoot, TokenLogprob,
};
use uniserve_core::{HashAlgo, RequestId};
use uniserve_worker_ipc::{
    Admission, AttentionRegime, Batch, BatchPartition, BlockTable as WireBlockTable, Bounds,
    CachePageAllocation, CloseReason, CompletionRecord, CompletionReport, Control, DType,
    DecodeKind, DecodePlacement, DimBound, Disposition, ExecutionCapability,
    ForwardRow as WireForwardRow, GenAdmission, GenMode, LatentPlacement, MediaAdmission,
    MediaProfileId, OpId, OpStatus, Operation, Point, PointRange, ProductKind, ProductPayload,
    ProductRef, RequestKey, ResourceClass, RouteId, SamplingState, ShapeBound, StorageClass,
    TimingCounters, UndAdmission, VersionRef, Work, WorkVariant, WorkerCapabilities,
    WorkerForwardStats,
};

use crate::executor::{WorkerExecError, WorkerLossError};
use crate::scheduler::image_artifact::validate_png_artifact;
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssemblyLane {
    Prefill,
    Decode,
    Other,
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
    pub(crate) trace: crate::scheduler::trace::RequestTrace,
}

struct FlowPrefixState {
    request_pool_idx: u32,
    block_tables: Vec<BlockTable>,
    new_pages: Vec<(u32, Vec<BlockId>)>,
    materialized: bool,
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
        self.replay.replayability == crate::scheduler::generation::Replayability::Replayable
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

struct KvSchedulerState {
    block_pool: BlockPool,
    coordinator: KvCacheCoordinator,
    usable_blocks: usize,
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
    Video { unit: u32 },
    Audio,
    Materialize,
}

#[derive(Default)]
struct MediaPlanner;

impl MediaPlanner {
    fn next(&self, cursor: MediaCursor) -> Option<MediaQuantum> {
        const DENOISE_STEPS: u32 = 4;
        // The fixed 124-frame profile is 17 * 7 + 5. Each decode unit owns
        // one finalized 17-frame temporal chunk plus the checkpoint-defined
        // five-frame tail/overlap geometry.
        const VIDEO_UNITS: u32 = 7;
        if !cursor.prepared {
            Some(MediaQuantum::Transition)
        } else if cursor.denoise_step < DENOISE_STEPS {
            Some(MediaQuantum::Flow {
                step: cursor.denoise_step,
            })
        } else if cursor.video_unit < VIDEO_UNITS {
            Some(MediaQuantum::Video {
                unit: cursor.video_unit,
            })
        } else if !cursor.audio_done {
            Some(MediaQuantum::Audio)
        } else if !cursor.materialized {
            Some(MediaQuantum::Materialize)
        } else {
            None
        }
    }

    fn advance(&self, mut cursor: MediaCursor, quantum: MediaQuantum) -> MediaCursor {
        match quantum {
            MediaQuantum::Transition => cursor.prepared = true,
            MediaQuantum::Flow { step } => cursor.denoise_step = step.saturating_add(1),
            MediaQuantum::Video { unit } => cursor.video_unit = unit.saturating_add(1),
            MediaQuantum::Audio => cursor.audio_done = true,
            MediaQuantum::Materialize => cursor.materialized = true,
        }
        cursor
    }

    fn work(&self, quantum: MediaQuantum) -> Work {
        match quantum {
            MediaQuantum::Transition => Work::Gen(GenMode::Transition),
            MediaQuantum::Flow { .. } => Work::Gen(GenMode::Flow),
            MediaQuantum::Video { .. } | MediaQuantum::Audio => Work::Gen(GenMode::Decode),
            MediaQuantum::Materialize => Work::Materialize,
        }
    }
}

struct MediaFlowState {
    request: MediaRequest,
    event_tx: MediaEventTx,
    request_pool_idx: u32,
    admission: Admission,
    admission_sent: bool,
    committed: MediaCursor,
    projected: MediaCursor,
    fixed_parent: VersionRef,
    projected_parent: VersionRef,
    cancelled: bool,
    failure: Option<String>,
}

enum ScheduledRequest {
    Generation(ReqState),
    Media(MediaFlowState),
}

#[derive(Default)]
struct ScheduledRequests {
    states: HashMap<RequestId, ScheduledRequest>,
}

impl ScheduledRequests {
    fn get(&self, id: &RequestId) -> Option<&ReqState> {
        match self.states.get(id) {
            Some(ScheduledRequest::Generation(state)) => Some(state),
            _ => None,
        }
    }

    fn get_mut(&mut self, id: &RequestId) -> Option<&mut ReqState> {
        match self.states.get_mut(id) {
            Some(ScheduledRequest::Generation(state)) => Some(state),
            _ => None,
        }
    }

    fn insert(&mut self, id: RequestId, state: ReqState) -> Option<ReqState> {
        match self.states.insert(id, ScheduledRequest::Generation(state)) {
            Some(ScheduledRequest::Generation(previous)) => Some(previous),
            Some(ScheduledRequest::Media(_)) => {
                unreachable!("request identity changed from media to generation")
            }
            None => None,
        }
    }

    fn remove(&mut self, id: &RequestId) -> Option<ReqState> {
        match self.states.remove(id) {
            Some(ScheduledRequest::Generation(state)) => Some(state),
            Some(ScheduledRequest::Media(_)) => {
                unreachable!("generation removal selected a media request")
            }
            None => None,
        }
    }

    fn contains_key(&self, id: &RequestId) -> bool {
        self.get(id).is_some()
    }

    fn len(&self) -> usize {
        self.states
            .values()
            .filter(|state| matches!(state, ScheduledRequest::Generation(_)))
            .count()
    }

    fn total_len(&self) -> usize {
        self.states.len()
    }

    fn keys(&self) -> impl Iterator<Item = &RequestId> {
        self.states.iter().filter_map(|(id, state)| {
            matches!(state, ScheduledRequest::Generation(_)).then_some(id)
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&RequestId, &ReqState)> {
        self.states.iter().filter_map(|(id, state)| match state {
            ScheduledRequest::Generation(state) => Some((id, state)),
            ScheduledRequest::Media(_) => None,
        })
    }

    fn values_mut(&mut self) -> impl Iterator<Item = &mut ReqState> {
        self.states.values_mut().filter_map(|state| match state {
            ScheduledRequest::Generation(state) => Some(state),
            ScheduledRequest::Media(_) => None,
        })
    }

    fn insert_media(&mut self, state: MediaFlowState) {
        let id = state.request.request_id;
        if self
            .states
            .insert(id, ScheduledRequest::Media(state))
            .is_some()
        {
            unreachable!("media admission reused a running request identity");
        }
    }

    fn media(&self, id: RequestId) -> Option<&MediaFlowState> {
        match self.states.get(&id) {
            Some(ScheduledRequest::Media(state)) => Some(state),
            _ => None,
        }
    }

    fn media_mut(&mut self, id: RequestId) -> Option<&mut MediaFlowState> {
        match self.states.get_mut(&id) {
            Some(ScheduledRequest::Media(state)) => Some(state),
            _ => None,
        }
    }

    fn media_ids(&self) -> Vec<RequestId> {
        self.states
            .iter()
            .filter_map(|(id, state)| matches!(state, ScheduledRequest::Media(_)).then_some(*id))
            .collect()
    }

    fn take_media(&mut self, id: RequestId) -> Option<MediaFlowState> {
        match self.states.remove(&id) {
            Some(ScheduledRequest::Media(state)) => Some(state),
            Some(ScheduledRequest::Generation(_)) => {
                unreachable!("media removal selected a generation request")
            }
            None => None,
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
    caps: WorkerCapabilities,
    kv: Option<KvSchedulerState>,
    ctrl: ControlTokens,
    config: SchedulerConfig,
    /// Pluggable host-side logits-processor pipeline.
    logits_pipeline: Vec<Arc<dyn crate::scheduler::logits::LogitsProcessor>>,
    custom_logits_processors: usize,
    /// Encoder-output cache (hashed, LRU, budgeted).
    enc_cache: EncoderCacheManager,
    /// Encoder-cache entries reserved by admitted image requests.
    reserved_encoder_entries: usize,
    request_slots: RequestSlotPool,
    latent_pages: LatentPagePool,
    running: ScheduledRequests,
    /// Terminal requests retain only their bounded public journal.
    completed_outputs: HashMap<RequestId, RetiredOutput>,
    /// Worker-visible sessions whose close transaction has been submitted but
    /// not yet acknowledged. Their worker page holdings and logical leases remain
    /// owned until the close report establishes the worker-side retirement
    /// ordering point.
    retiring_sessions: HashMap<RequestId, RetiringSession>,
    order: Vec<RequestId>, // stable cross-request iteration order
    pending_media: VecDeque<PendingMedia>,
    media_planner: MediaPlanner,
    retiring_media: HashMap<RequestId, RetiringMedia>,
    prefer_media: bool,
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
    /// Sequential denoise timesteps to execute per denoise op. The worker runs
    /// the exact same Euler steps and reports the cumulative step cursor.
    denoise_step_burst: u16,
    /// Diagnostic: keep denoise steps out of batches that carry text rows.
    flow_exclusive_batch: bool,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    /// Explainable policy decisions and per-op latency history.
    decisions: crate::scheduler::policy::DecisionLog,
    latency: crate::scheduler::policy::LatencyHistory,
    planner: GenerationPlanner,
    /// Submit timestamp per in-flight batch (for batch round-trip traces).
    batch_started: HashMap<u64, Instant>,
    /// Steps in the execution window that carry prompt work.
    prefill_steps: HashSet<u64>,
    /// Domain and physical-call identity for partitions still expected from each batch.
    batch_partitions: HashMap<u64, HashMap<u32, SubmittedPartitionAccounting>>,
    /// Worker duration per physical submission group across returned partitions.
    batch_group_worker_exec_us: HashMap<u64, HashMap<u32, u64>>,
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
    completed_traces: VecDeque<crate::scheduler::trace::RequestTrace>,
    trace_sink: Option<crate::scheduler::bench_trace::SchedulerTraceSink>,
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
    pub from: crate::scheduler::trace::LifecyclePhase,
    pub to: crate::scheduler::trace::LifecyclePhase,
    pub count: u64,
    pub sum_us: u64,
    pub max_us: u64,
}

/// Cumulative operation-window and lifecycle accounting for one domain.
#[derive(Debug, Clone)]
pub struct DomainWindowMetrics {
    pub domain: uniserve_worker_ipc::Domain,
    pub active_credits: usize,
    pub peak_credits: usize,
    pub launched_operations: u64,
    pub completed_operations: u64,
    pub predicated_operations: u64,
    pub error_operations: u64,
    pub backpressure_events: u64,
    pub reclaimed_credits: u64,
    pub completed_partitions: u64,
    pub semantic_commits: u64,
    pub public_commits: u64,
    pub co_resident_partitions: u64,
    pub queue_us: u64,
    pub launch_us: u64,
    pub device_us: u64,
    pub completion_us: u64,
    pub semantic_commit_us: u64,
    pub public_commit_us: u64,
    pub co_resident_us: u64,
}

/// The bounded operation-window and lifecycle observability surface.
#[derive(Debug, Clone)]
pub struct ResourceWindowMetrics {
    pub max_operations: usize,
    pub active_operations: usize,
    pub max_unresolved_window: u32,
    pub peak_ops_in_batch: usize,
    pub phase_delays: Vec<PhaseSpanDelay>,
    pub domains: Vec<DomainWindowMetrics>,
}

fn now() -> f64 {
    // route through the single shared epoch helper so every
    // component's wall-clock timestamps match. It never panics on the hot loop:
    // a wall clock set before the UNIX epoch (or stepped backward) clamps to 0
    // instead of unwrapping the `Result`.
    uniserve_core::now_unix_secs()
}

fn worker_float_dtype(value: &str) -> Option<uniserve_worker_ipc::DType> {
    match value {
        "float16" => Some(uniserve_worker_ipc::DType::F16),
        "bfloat16" => Some(uniserve_worker_ipc::DType::BF16),
        "float32" => Some(uniserve_worker_ipc::DType::F32),
        _ => None,
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
        | WorkVariant::GenDecode
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

fn domain_str(domain: uniserve_worker_ipc::Domain) -> &'static str {
    match domain {
        uniserve_worker_ipc::Domain::Prefill => "prefill",
        uniserve_worker_ipc::Domain::Decode => "decode",
        uniserve_worker_ipc::Domain::Flow => "flow",
    }
}

fn execution_capability_str(execution: ExecutionCapability) -> &'static str {
    match execution {
        ExecutionCapability::DomainHomogeneous => "domain_homogeneous",
        ExecutionCapability::TensorizedMixed => "tensorized_mixed",
    }
}

fn domain_window_metrics(
    domains: &ExecutionDomainStats,
    domain: uniserve_worker_ipc::Domain,
) -> DomainWindowMetrics {
    let stats = domains.get(domain);
    DomainWindowMetrics {
        domain,
        active_credits: stats.active_credits.load(Ordering::Relaxed),
        peak_credits: stats.peak_credits.load(Ordering::Relaxed),
        launched_operations: stats.launched_operations.load(Ordering::Relaxed),
        completed_operations: stats.completed_operations.load(Ordering::Relaxed),
        predicated_operations: stats.predicated_operations.load(Ordering::Relaxed),
        error_operations: stats.error_operations.load(Ordering::Relaxed),
        backpressure_events: stats.backpressure_events.load(Ordering::Relaxed),
        reclaimed_credits: stats.reclaimed_credits.load(Ordering::Relaxed),
        completed_partitions: stats.completed_partitions.load(Ordering::Relaxed),
        semantic_commits: stats.semantic_commits.load(Ordering::Relaxed),
        public_commits: stats.public_commits.load(Ordering::Relaxed),
        co_resident_partitions: stats.co_resident_partitions.load(Ordering::Relaxed),
        queue_us: stats.queue_us.load(Ordering::Relaxed),
        launch_us: stats.launch_us.load(Ordering::Relaxed),
        device_us: stats.device_us.load(Ordering::Relaxed),
        completion_us: stats.completion_us.load(Ordering::Relaxed),
        semantic_commit_us: stats.semantic_commit_us.load(Ordering::Relaxed),
        public_commit_us: stats.public_commit_us.load(Ordering::Relaxed),
        co_resident_us: stats.co_resident_us.load(Ordering::Relaxed),
    }
}

/// One submitted-but-unresolved operation tracked in a request's ordered queue.
/// Its apply record is paired by `(operation.request_key, operation.op_id)`.
enum InflightApply {
    Generation(SchedulerApply),
    Media(MediaCursor),
}

struct InflightOp {
    operation: Operation,
    apply: InflightApply,
    /// Submit timestamp, for the op's host round-trip latency history.
    started: Instant,
}

impl InflightOp {
    fn generation_apply(&self) -> &SchedulerApply {
        match &self.apply {
            InflightApply::Generation(apply) => apply,
            InflightApply::Media(_) => unreachable!("media operation entered generation planning"),
        }
    }
}

#[derive(Clone, Copy)]
struct SubmittedPartitionAccounting {
    domain: uniserve_worker_ipc::Domain,
    execution: ExecutionCapability,
    submission_group: u32,
    operation_count: usize,
}

#[derive(Clone)]
struct ProjectedBranch {
    phase: Phase,
    image_id: u32,
    conditioning_position: u32,
    steps_done: u16,
    conditioning: Option<ProductRef>,
    latent: Option<ProductRef>,
    feedback_source: Option<ProductRef>,
    feedback_feature: Option<ProductRef>,
    feedback_step: usize,
    chainable: bool,
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

const OUTPUT_JOURNAL_CAPACITY: usize = EVENT_BUFFER_CAPACITY;
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
        let max_batch_ops = caps
            .lanes
            .iter()
            .map(|lane| lane.max_batch_operations as usize)
            .min()
            .unwrap_or(caps.max_batch_operations as usize)
            .min(caps.max_batch_operations as usize);
        let max_batch_tokens = caps
            .lanes
            .iter()
            .map(|lane| lane.max_batch_tokens as usize)
            .min()
            .unwrap_or(caps.max_batch_tokens as usize)
            .min(caps.max_batch_tokens as usize);
        let transfer_capacity = (caps.pipeline_depth as usize)
            .saturating_mul(max_batch_ops)
            .clamp(1, MAX_INFLIGHT_TRANSFERS);
        config.max_num_waiting = config.max_num_waiting.clamp(1, MAX_NUM_WAITING);
        let flow_slot_reserve =
            usize::from(caps.uses_kv() && caps.supported_work.contains(&WorkVariant::GenFlow));
        let request_pool_capacity = caps.max_request_pool_size as usize;
        let main_request_capacity = request_pool_capacity
            .saturating_sub(flow_slot_reserve)
            .max(1);
        config.max_num_seqs = config
            .max_num_seqs
            .clamp(1, MAX_NUM_SEQS)
            .min(main_request_capacity);
        config.max_num_batched_tokens = config.max_num_batched_tokens.max(1).min(max_batch_tokens);
        if max_batch_ops > 0 {
            config.max_batch = config.max_batch.min(max_batch_ops.max(1));
        }
        let kv = caps.uses_kv().then(|| {
            let block_pool = if caps.groups.is_empty() {
                BlockPool::new(caps.num_blocks as usize, caps.block_size as usize)
            } else {
                let specs: Vec<(uniserve_core::KvGroupKind, u32, u32)> = caps
                    .groups
                    .iter()
                    .map(|g| (g.kind, g.block_offset, g.num_blocks))
                    .collect();
                BlockPool::with_groups(caps.num_blocks as usize, caps.block_size as usize, &specs)
            };
            let usable_blocks = block_pool.request_page_capacity();
            KvSchedulerState {
                block_pool,
                coordinator: KvCacheCoordinator::default(),
                usable_blocks,
            }
        });
        let stats = Arc::new(SchedStats::default());
        stats.kv_cache.num_blocks.store(
            kv.as_ref().map_or(0, |state| state.usable_blocks),
            Ordering::Relaxed,
        );
        let caps_encoder_budget = caps.encoder_cache_budget as usize;
        let request_slots = RequestSlotPool::new(request_pool_capacity);
        let latent_pages = LatentPagePool::new(caps.num_latent_pages, caps.latent_page_units);
        let denoise_step_burst = denoise_step_burst_from_env();
        let flow_exclusive_batch = env::var(FLOW_EXCLUSIVE_BATCH_ENV)
            .is_ok_and(|raw| matches!(raw.trim(), "1" | "true" | "TRUE"));
        let mut trace_sink = crate::scheduler::bench_trace::SchedulerTraceSink::from_env();
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
                    "max_batch_tokens": caps.max_batch_tokens,
                    "max_request_pool_size": caps.max_request_pool_size,
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
        let latent_dtype = worker_float_dtype(&caps.latent_dtype);
        Self {
            executor,
            caps,
            kv,
            ctrl,
            pending: make_queue(config.policy),
            config,
            logits_pipeline: crate::scheduler::logits::default_pipeline(),
            custom_logits_processors: 0,
            enc_cache: EncoderCacheManager::new(caps_encoder_budget),
            reserved_encoder_entries: 0,
            request_slots,
            latent_pages,
            running: ScheduledRequests::default(),
            completed_outputs: HashMap::new(),
            retiring_sessions: HashMap::new(),
            order: Vec::new(),
            pending_media: VecDeque::new(),
            media_planner: MediaPlanner,
            retiring_media: HashMap::new(),
            prefer_media: true,
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
            denoise_step_burst,
            flow_exclusive_batch,
            fatal: false,
            decisions: crate::scheduler::policy::DecisionLog::default(),
            latency: crate::scheduler::policy::LatencyHistory::new(),
            planner: GenerationPlanner::new(latent_dtype),
            batch_started: HashMap::new(),
            prefill_steps: HashSet::new(),
            batch_partitions: HashMap::new(),
            batch_group_worker_exec_us: HashMap::new(),
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
        if let Some(kv) = self.kv.as_mut() {
            kv.coordinator.set_prefix_enabled(on);
        }
    }
    pub fn set_hash_algo(&mut self, algo: HashAlgo) {
        if let Some(kv) = self.kv.as_mut() {
            kv.coordinator.set_hash_algo(algo);
        }
    }
    /// Register an extra logits processor — no other scheduler code changes.
    pub fn with_logits_processor(
        mut self,
        p: Box<dyn crate::scheduler::logits::LogitsProcessor>,
    ) -> Self {
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
        let flow_slot_reserve = usize::from(
            self.caps.uses_kv() && self.caps.supported_work.contains(&WorkVariant::GenFlow),
        );
        let capacity = self
            .request_slots
            .capacity()
            .saturating_sub(flow_slot_reserve)
            .max(1);
        self.config.max_num_seqs = n.clamp(1, capacity);
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

    fn kv_state(&self) -> &KvSchedulerState {
        self.kv
            .as_ref()
            .expect("generation scheduling requires worker KV resources")
    }

    fn free_kv_blocks(&self) -> usize {
        self.kv
            .as_ref()
            .map_or(0, |state| state.block_pool.free_request_pages())
    }

    fn usable_kv_blocks(&self) -> usize {
        self.kv.as_ref().map_or(0, |state| state.usable_blocks)
    }

    fn cached_kv_blocks(&self) -> usize {
        self.kv
            .as_ref()
            .map_or(0, |state| state.block_pool.cached_blocks())
    }

    fn media_state(&self, id: RequestId) -> Option<&MediaFlowState> {
        self.running.media(id)
    }

    fn media_state_mut(&mut self, id: RequestId) -> Option<&mut MediaFlowState> {
        self.running.media_mut(id)
    }

    fn media_ids(&self) -> Vec<RequestId> {
        self.running.media_ids()
    }

    fn take_media_state(&mut self, id: RequestId) -> Option<MediaFlowState> {
        self.running.take_media(id)
    }

    fn running_request_count(&self) -> usize {
        self.running.total_len()
    }

    fn pending_request_count(&self) -> usize {
        self.pending.len().saturating_add(self.pending_media.len())
    }

    /// Record one explainable policy decision; `free_blocks`
    /// is sampled from the block manager at the decision point.
    fn record_decision(
        &mut self,
        request: RequestId,
        reason: crate::scheduler::policy::PolicyReason,
        needed_blocks: usize,
    ) {
        let free_blocks = self.free_kv_blocks();
        self.decisions
            .record(crate::scheduler::policy::PolicyDecision {
                request,
                reason,
                free_blocks,
                needed_blocks,
            });
    }

    /// Structured scheduling facts the policy weighs. A
    /// read-only snapshot — a pluggable policy could consume this without
    /// reaching into scheduler internals.
    pub fn policy_snapshot(&self) -> crate::scheduler::policy::PolicySnapshot {
        let pq = self.stats.prefix.queries.load(Ordering::Relaxed);
        let ph = self.stats.prefix.hits.load(Ordering::Relaxed);
        let mq = self.stats.encoder.cache_queries.load(Ordering::Relaxed);
        let mh = self.stats.encoder.cache_hits.load(Ordering::Relaxed);
        crate::scheduler::policy::PolicySnapshot {
            waiting: self.pending_request_count(),
            running: self.running_request_count(),
            in_flight: self.executor.in_flight(),
            free_blocks: self.free_kv_blocks(),
            total_blocks: self.usable_kv_blocks(),
            reserved_blocks: self.reserved_blocks,
            cached_blocks: self.cached_kv_blocks(),
            prefix_hit_rate: if pq > 0 { ph as f32 / pq as f32 } else { 0.0 },
            mm_cache_hit_rate: if mq > 0 { mh as f32 / mq as f32 } else { 0.0 },
            op_latency_us: self.latency.as_pairs(),
        }
    }

    /// Drain the recent explainable policy decisions.
    pub fn take_policy_decisions(&mut self) -> Vec<crate::scheduler::policy::PolicyDecision> {
        self.decisions.drain()
    }

    /// Round-trip latency EWMA for one op kind (microseconds), if observed.
    pub fn op_latency_us(&self, kind: &str) -> Option<u64> {
        self.latency.get(kind)
    }

    /// The in-flight lifecycle trace of a running request.
    pub fn request_trace(&self, id: RequestId) -> Option<&crate::scheduler::trace::RequestTrace> {
        self.running.get(&id).map(|st| &st.trace)
    }

    /// Drain archived lifecycle traces of completed requests — the
    /// reconstructable record after a request has finished.
    pub fn take_completed_traces(&mut self) -> Vec<crate::scheduler::trace::RequestTrace> {
        self.completed_traces.drain(..).collect()
    }

    /// The operation-window bounds, current occupancy, observed peak, and
    /// aggregate lifecycle-phase delays over completed operations.
    pub fn resource_window_metrics(&self) -> ResourceWindowMetrics {
        use crate::scheduler::trace::LifecyclePhase as P;
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
            domains: [
                uniserve_worker_ipc::Domain::Prefill,
                uniserve_worker_ipc::Domain::Decode,
                uniserve_worker_ipc::Domain::Flow,
            ]
            .into_iter()
            .map(|domain| domain_window_metrics(&self.stats.domains, domain))
            .collect(),
        }
    }

    /// A health snapshot the engine can expose: queue +
    /// resource pressure + policy/latency + backend caps + liveness.
    pub fn health_snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            running: self.running_request_count(),
            pending: self.pending_request_count(),
            in_flight: self.executor.in_flight(),
            free_blocks: self.free_kv_blocks(),
            total_blocks: self.usable_kv_blocks(),
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

            if !progressed {
                self.park_for_progress();
                if self.fatal {
                    tracing::error!(
                        "engine fatal: executor/worker died during park; stopping the control loop"
                    );
                    self.abort_all_requests();
                    self.executor.shutdown();
                    return true;
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

    /// One park over result, command, worker-death, CPU-continuation, and
    /// output-capacity wakes. The timeout is solely a liveness deadline.
    fn park_for_progress(&mut self) {
        let _span = tracing::trace_span!("scheduler.park").entered();
        if let Err(e) = self.executor.park_for_event(IDLE_LIVENESS_POLL) {
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
}
#[derive(Clone, Copy)]
struct KvLengths {
    input: u32,
    visible: u32,
}

fn transition_kv_lengths(delta: &TransitionDelta) -> Option<KvLengths> {
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
        TransitionDelta::PublishKv {
            physical_kv_len, ..
        }
        | TransitionDelta::TransitionGen {
            physical_kv_len, ..
        }
        | TransitionDelta::DenoiseGen {
            physical_kv_len, ..
        } => (*physical_kv_len, 0),
        TransitionDelta::EncodeImageStep { .. }
        | TransitionDelta::CommitGen { .. }
        | TransitionDelta::EncodeFeedbackStep { .. } => return None,
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
    matches!(variant, WorkVariant::TokenDecode | WorkVariant::GenFlow)
}

fn flow_matches_mixed_bucket(
    operation: &Operation,
    forward_rows: &HashMap<(RequestKey, OpId), Vec<WireForwardRow>>,
    latent_placements: &HashMap<(RequestKey, OpId), LatentPlacement>,
    bucket: &uniserve_worker_ipc::MixedExecutionCapability,
) -> bool {
    if operation.work.variant() != WorkVariant::GenFlow {
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
    buckets: &[uniserve_worker_ipc::MixedExecutionCapability],
    forward_rows: &HashMap<(RequestKey, OpId), Vec<WireForwardRow>>,
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
        .map(|operation| match operation.work.variant() {
            WorkVariant::TokenExtend
            | WorkVariant::TokenDecode
            | WorkVariant::TokenVerify
            | WorkVariant::Draft => AttentionRegime::Causal,
            WorkVariant::GenFlow => AttentionRegime::Hybrid,
            WorkVariant::GenTransition
            | WorkVariant::GenDecode
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
        WorkVariant::GenDecode => 2,
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
            Err(EventSendError::Full(event)) => {
                journal.push_front(*event);
                return false;
            }
            Err(EventSendError::Closed(_)) => {
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
            Err(EventSendError::Closed(_)) => return true,
            Err(EventSendError::Full(event)) => {
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
    use crate::scheduler::generation::Replayability;
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
        GenerationPlanner::new(Some(uniserve_worker_ipc::DType::BF16))
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
