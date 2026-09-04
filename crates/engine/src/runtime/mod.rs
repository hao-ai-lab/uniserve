//! Request-family runtimes and the single-owner engine control loop.
//!
//! The loop combines request progress, scheduler policy, memory placement, and
//! asynchronous executor completions. All mutable engine state stays on its
//! owner thread.
//!
//! Scheduling admits from a [`RequestQueue`], chunks prefill against sequence and
//! token budgets, and reserves each request's maximum declared resources. A
//! resident request remains in place when capacity prevents relocation.
//!
//! Static worker state crosses the boundary once in [`NewRequest`]; subsequent
//! operations carry only step-specific deltas.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod admission;
mod batching;
pub(crate) mod bench_trace;
mod control;
mod engine_loop;
mod execution;
pub(crate) mod generation;
pub(crate) mod image_artifact;
mod inflight;
mod logits;
pub(crate) mod output;

pub(crate) use crate::executor::{
    Batch, BatchResult, Executor, ExecutorSubmitError, Op as LogicalOp, OpPlacement,
};

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use std::env;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::runtime::generation::{
    EncoderCachePin, GenerationCursor, GenerationPhase as Phase, NextOp, RuntimeApply,
    RuntimeContext, TransitionIntent, plan,
};

use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EventRx, EventSendError, EventTx, event_channel,
};
use crate::kv::{BlockPool, BlockTable, KvCacheCoordinator};
use crate::memory::{Allocation, Memory, MemoryLayout, Placement, worker_kv_state};
use crate::scheduler::{
    MAX_NUM_SEQS, MAX_NUM_WAITING, SchedStats, Scheduler, SchedulerConfig, SchedulingPolicy,
};
use crossbeam_channel::Receiver;
use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{
    ArtifactEvent, ArtifactHandle, DiffusionRequest, Event, FinishReason, GenerationRequest,
    MediaKind, PositionLogprobs, Request, TokenLogprob,
};
use uniserve_core::{BlockId, CfgParams, ImageIngestStep, encoder_cache_key};
use uniserve_core::{HashAlgo, RequestId, RuntimeFamily};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable as IpcBlockTable, Bounds, BufferId, BufferPlacement,
    CachePageAllocation, Checkpoint, CheckpointPoint, CloseReason, DType, DecodePlacement,
    DiffusionRequestParams, Disposition, LatentPlacement, MediaGeometry, ModelOutput, NewRequest,
    OpId, OpKind, OpPayload, OpStatus, Operation, PointRange, ProductKind, ProductPayload,
    ProductRef, RequestKey, RowGeometry, RunKind, SamplingState, ShapeBound, StorageClass,
    TimingCounters, UmmRequestParams, WorkerForwardStats, WorkerInfo,
};

use crate::executor::{WorkerExecError, WorkerLossError};
use crate::runtime::image_artifact::validate_png_artifact;
use inflight::{
    InflightApply, InflightOp, InflightWindow, PendingCompletion, PendingFinish,
    SubmittedRunAccounting,
};
use output::RequestOutput;
use serde_json::json;

/// Number of denoising steps planned for one scheduling burst by default.
pub(crate) const DEFAULT_DENOISE_STEP_BURST: u16 = 1;
const MAX_INFLIGHT_TRANSFERS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchKind {
    Prefill,
    Decode,
    Media,
}

const IDLE_LIVENESS_POLL: Duration = Duration::from_millis(500);
const PREFILL_WINDOW_CREDITS: usize = 2;
const DENOISE_STEP_BURST_ENV: &str = "UNISERVE_DENOISE_STEP_BURST";
const FLOW_EXCLUSIVE_BATCH_ENV: &str = "UNISERVE_FLOW_EXCLUSIVE_BATCH";

/// Builds a validated image-completion event from a PNG payload.
fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<Event> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(Event::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
        sha256: metadata.sha256,
        pixels_png_b64,
    })
}

/// Returns the product with the requested producer and kind.
fn find_product(
    products: &[ProductPayload],
    op_id: OpId,
    kind: ProductKind,
) -> Option<&ProductPayload> {
    products
        .iter()
        .find(|payload| payload.product.producer_op_id == op_id && payload.product.kind == kind)
}

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
    /// Decodes typed products and completion fields into a transition-neutral result view.
    fn from_report(record: &ModelOutput, products: &[ProductPayload]) -> Self {
        let logprobs = find_product(products, record.op_id, ProductKind::Logprob)
            .and_then(|payload| payload.value.bytes())
            .and_then(|bytes| LogprobBlob::decode(bytes).ok())
            .unwrap_or_default();
        let image_png = find_product(products, record.op_id, ProductKind::Artifact)
            .and_then(|payload| payload.value.bytes())
            .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok());
        Self {
            committed_tokens: record.committed_tokens().to_vec(),
            sampled_logprob: logprobs.sampled_logprob,
            top_logprobs: logprobs.top_logprobs,
            prompt_logprobs: logprobs.prompt_logprobs,
            image_png,
            encode_generation: record.product_generations.first().copied(),
            kv_visible_len: record.logical_lengths().kv_visible_len,
            flow_done: record.finish_flags().eos
                || record.finish_flags().length
                || record.finish_flags().stop,
        }
    }
}

/// Returns cache-version metadata for the committed token prefix.
fn token_prefix_versions(
    operation: Option<&Operation>,
    record: &ModelOutput,
    _parent: Option<&Checkpoint>,
) -> Vec<Checkpoint> {
    if operation.is_none() {
        return Vec::new();
    }
    let count = record.committed_tokens().len();
    if record.status != OpStatus::Ok || count == 0 || record.selected_point as usize != count {
        return Vec::new();
    }
    (1..=count)
        .map(|point_index| Checkpoint {
            op_id: record.op_id,
            point: CheckpointPoint::Fixed(point_index as u32),
        })
        .collect()
}

#[derive(Clone)]
/// Cooperative cancellation and shutdown tokens shared with engine clients.
pub struct ControlTokens {
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
}

impl Default for ControlTokens {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            bos: 151644,
            eos: vec![151645, 151643],
            end_of_image: 151653,
        }
    }
}

/// Engine-owned state for one admitted request.
pub(crate) struct ReqState {
    pub req: GenerationRequest,
    pub(crate) finish_token_ids: Vec<u32>,
    /// The cached sequence mappings. Each table owns its physical page
    /// references and therefore has exactly the request's lifetime.
    allocations: Option<RequestAllocations>,
    flow_prefix: Option<FlowPrefixState>,
    /// Stable scheduler-assigned row used by every worker-side request-indexed
    /// state owner for this request epoch. Zero denotes a pending request that
    /// has not entered the running set.
    /// EngineLoop-owned lifecycle generation and latest host-resolved worker version.
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
    pub(crate) token_cutoffs: BTreeMap<usize, Checkpoint>,
    /// Ordered semantic commits held for the frontend decoder's exact-prefix
    /// decision, oldest first. A stop-string request may register bounded
    /// provisional descendants before their predecessors are decoded, so more
    /// than one commit can await a decision at once; each carries the exact
    /// chained parent so the worker applies them in order once acknowledged.
    /// The frontend can retract every descendant beyond a matched stop by an
    /// exact cutoff, dropping the un-acknowledged tail without committing it.
    pub(crate) pending_commits: VecDeque<PendingSemanticCommit>,
    pub(crate) cancel_cutoff: Option<Checkpoint>,
    /// Exact worker-local selected-point product for the latest resolved state,
    /// together with the work variant that owns its physical pool. The scheduler
    /// retains this logical ownership until it submits a reachable consumer.
    pub(crate) latest_device_version: Option<ResidentDeviceVersion>,
    /// A false device predicate invalidated the unresolved successor chain.
    /// Already-submitted descendants must drain before scheduling resumes from
    /// the last host-resolved checkpoint.
    pub(crate) speculative_chain_invalidated: bool,
    pub(crate) context: RuntimeContext,
    output: RequestOutput,
    pub(crate) cursor: GenerationCursor,
    pub queued_at: f64,
    pub(crate) terminal_intent: TerminalIntent,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// Terminal action deferred until outstanding work is reconciled.
pub(crate) enum TerminalIntent {
    #[default]
    None,
    Cancel,
    Abort,
    StopMatched,
}

impl TerminalIntent {
    /// Returns whether the state is terminal.
    pub(crate) const fn is_terminal(self) -> bool {
        !matches!(self, Self::None)
    }
}

struct FlowPrefixState {
    allocations: RequestAllocations,
    new_pages: Vec<(u32, Vec<BlockId>)>,
    diffusion_finalized: bool,
}

impl FlowPrefixState {
    /// Returns the request-pool index.
    fn request_pool_idx(&self) -> u32 {
        self.allocations.request_slot()
    }

    /// Returns shared access to the request block tables.
    fn block_tables(&self) -> &[BlockTable] {
        self.allocations.block_tables()
    }
}

struct RetiringRequest {
    request_key: RequestKey,
    allocations: RequestAllocations,
    flow_prefix: Option<FlowPrefixState>,
}

struct RequestAllocations {
    request_slot: Allocation,
    kv: Allocation,
    latent: Option<Allocation>,
    buffers: HashMap<BufferId, Allocation>,
}

impl RequestAllocations {
    /// Returns the request-slot identifier.
    fn request_slot(&self) -> u32 {
        self.request_slot
            .request_slot()
            .expect("request slot allocation")
    }

    /// Returns shared access to the request block tables.
    fn block_tables(&self) -> &[BlockTable] {
        self.kv.kv_tables().expect("KV allocation")
    }

    /// Returns mutable access to the request block tables.
    fn block_tables_mut(&mut self) -> &mut Vec<BlockTable> {
        self.kv.kv_tables_mut().expect("KV allocation")
    }

    /// Takes ownership of the request buffer allocation.
    fn take_buffer(&mut self, id: BufferId) -> Option<Allocation> {
        self.buffers.remove(&id)
    }

    /// Releases the owned request allocation.
    fn free(self, memory: &mut Memory) {
        for allocation in self.buffers.into_values() {
            memory.free(allocation);
        }
        if let Some(latent) = self.latent {
            memory.free(latent);
        }
        memory.free(self.kv);
        memory.free(self.request_slot);
    }
}

#[derive(Clone)]
/// Semantic commit waiting for all required worker products.
pub(crate) struct PendingSemanticCommit {
    token_count: Option<usize>,
    expected_parent: Checkpoint,
    selected: Checkpoint,
    public_event_limit: u64,
}

#[derive(Clone)]
/// Latest worker-resident checkpoint available for further planning.
pub(crate) struct ResidentDeviceVersion {
    version: Checkpoint,
    token: ProductRef,
}

impl ReqState {
    /// Returns the request allocations.
    fn allocations(&self) -> &RequestAllocations {
        self.allocations.as_ref().expect("request is admitted")
    }

    /// Returns mutable access to the request allocations.
    fn allocations_mut(&mut self) -> &mut RequestAllocations {
        self.allocations.as_mut().expect("request is admitted")
    }

    /// Returns the request-pool index.
    fn request_pool_idx(&self) -> u32 {
        self.allocations().request_slot()
    }

    /// Returns shared access to the request block tables.
    fn block_tables(&self) -> &[BlockTable] {
        self.allocations().block_tables()
    }

    /// Returns mutable access to the request block tables.
    fn block_tables_mut(&mut self) -> &mut Vec<BlockTable> {
        self.allocations_mut().block_tables_mut()
    }

    /// Returns whether the request includes context images.
    fn has_context_images(&self) -> bool {
        !self.context.images.is_empty()
    }

    /// Returns whether execution continues after committing generation output.
    fn continues_after_gen_commit(&self) -> bool {
        self.req.behavior.continue_after_gen_commit
    }

    /// Returns whether the request can open a generation branch.
    fn can_open_gen_branch(&self) -> bool {
        self.req.behavior.gen_output
            && self.cursor.image_gen.images_done < self.req.image.max_images as usize
    }

    /// Returns whether generation starts after context ingestion.
    fn starts_gen_after_context(&self) -> bool {
        self.req.behavior.start_gen_after_context && self.can_open_gen_branch()
    }

    /// Returns whether the text request can be replayed.
    pub(crate) fn is_replayable_text(&self) -> bool {
        self.cursor.replay.replayability == crate::runtime::generation::Replayability::Replayable
    }

    /// Returns the prompt used for execution.
    pub(crate) fn effective_prompt(&self) -> &[u32] {
        &self.context.prompt_ids
    }

    /// Returns the pending image step, if one exists.
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
    decode_cursor: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MediaQuantum {
    Prepare,
    Denoise {
        step: u32,
    },
    Decode {
        cursor: u32,
        max_units: u32,
        finalizes: bool,
    },
}

/// Returns the next schedulable media quantum.
fn next_media_quantum(cursor: MediaCursor, geometry: MediaGeometry) -> Option<MediaQuantum> {
    if !cursor.prepared {
        Some(MediaQuantum::Prepare)
    } else if cursor.denoise_step < geometry.denoise_steps {
        Some(MediaQuantum::Denoise {
            step: cursor.denoise_step,
        })
    } else if cursor.decode_cursor < geometry.decode_units {
        Some(MediaQuantum::Decode {
            cursor: cursor.decode_cursor,
            max_units: 1,
            finalizes: cursor.decode_cursor.saturating_add(1) == geometry.decode_units,
        })
    } else {
        None
    }
}

/// Advances the request media cursor.
fn advance_media_cursor(mut cursor: MediaCursor, quantum: MediaQuantum) -> MediaCursor {
    match quantum {
        MediaQuantum::Prepare => cursor.prepared = true,
        MediaQuantum::Denoise { step } => cursor.denoise_step = step.saturating_add(1),
        MediaQuantum::Decode {
            cursor: decode_cursor,
            max_units,
            ..
        } => cursor.decode_cursor = decode_cursor.saturating_add(max_units),
    }
    cursor
}

/// Returns the request remaining media work.
fn media_work(quantum: MediaQuantum) -> RunKind {
    match quantum {
        MediaQuantum::Prepare => RunKind::DiffusionPrepare,
        MediaQuantum::Denoise { .. } => RunKind::DiffusionStep,
        MediaQuantum::Decode {
            finalizes: true, ..
        } => RunKind::DiffusionFinalize,
        MediaQuantum::Decode { .. } => RunKind::DiffusionDecode,
    }
}

struct MediaFlowState {
    request: DiffusionRequest,
    event_tx: EventTx,
    allocations: MediaAllocations,
    admission: NewRequest,
    admission_state: DiffusionRequestParamsState,
    committed: MediaCursor,
    projected: MediaCursor,
    fixed_parent: Checkpoint,
    projected_parent: Checkpoint,
    terminal_intent: MediaTerminalIntent,
    artifact: Option<ArtifactEvent>,
}

struct MediaAllocations {
    request_slot: Allocation,
    latent: Allocation,
}

impl MediaAllocations {
    /// Returns the request-slot identifier.
    fn request_slot(&self) -> u32 {
        self.request_slot
            .request_slot()
            .expect("media request slot allocation")
    }

    /// Returns the latent-page allocations.
    fn latent_pages(&self) -> &[u32] {
        match self.latent.placement() {
            Placement::Latent { pages, .. } => pages,
            _ => unreachable!("media latent allocation placement"),
        }
    }

    /// Releases the owned request allocation.
    fn free(self, memory: &mut Memory) {
        memory.free(self.latent);
        memory.free(self.request_slot);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiffusionRequestParamsState {
    Unsubmitted,
    InFlight,
    Registered,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MediaTerminalIntent {
    None,
    Finish(FinishReason),
    Failure(String),
}

enum DiffusionTerminal {
    Completed(ArtifactEvent),
    Failed(String),
    Finished(FinishReason),
}

impl MediaTerminalIntent {
    /// Returns whether the state is terminal.
    const fn is_terminal(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Records the first terminal reason while preserving an earlier terminal intent.
    fn finish(&mut self, reason: FinishReason) {
        if matches!(self, Self::None) {
            *self = Self::Finish(reason);
        }
    }
}

struct RetiringMedia {
    request_key: RequestKey,
    allocations: MediaAllocations,
}

struct PendingMedia {
    request: DiffusionRequest,
    event_tx: EventTx,
}

#[doc(hidden)]
/// Family-independent generation state shared by runtime wrappers.
pub struct RuntimeState {
    ctrl: ControlTokens,
    logits_pipeline: Vec<crate::runtime::logits::BuiltinLogitsProcessor>,
    waiting: HashMap<RequestId, ReqState>,
    waiting_media: HashMap<RequestId, PendingMedia>,
    running: HashMap<RequestId, ReqState>,
    running_media: HashMap<RequestId, MediaFlowState>,
    retiring_requests: HashMap<RequestId, RetiringRequest>,
    retiring_media: HashMap<RequestId, RetiringMedia>,
    inflight: InflightWindow,
    denoise_step_burst: u16,
    latent_dtype: Option<DType>,
    pending_commands: VecDeque<BatchCommand>,
    pending_buffer_frees: HashMap<BufferId, Allocation>,
    authority_id: u64,
    next_op_id: u64,
    next_product_generation: u64,
    next_epoch: u64,
}

/// Autoregressive request runtime.
pub struct ArRuntime(RuntimeState);
/// Terminal diffusion request runtime.
pub struct DiffusionRuntime(RuntimeState);
/// Unified multimodal request runtime.
pub struct UmmRuntime(RuntimeState);

/// Serving-profile facts consumed by a family runtime. Physical capacities are
/// resolved from [`WorkerInfo`] during engine construction; model procedure and
/// captured execution shapes remain worker-local.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeProfile {
    /// Numeric data type used by model activations.
    pub model_dtype: uniserve_core::ModelDtype,
    /// Model-family generation features and shape limits.
    pub generation_limits: uniserve_core::GenerationLimits,
    /// Data type used by diffusion latent products, when applicable.
    pub latent_dtype: Option<DType>,
    /// Maximum number of encoder products retained for reuse.
    pub encoder_cache_entries: usize,
}

/// Returns simulator capabilities for unified multimodal generation.
pub(crate) fn sim_umm_generation_limits() -> uniserve_core::GenerationLimits {
    uniserve_core::GenerationLimits {
        features: uniserve_core::GenerationFeatures::UNDERSTANDING
            | uniserve_core::GenerationFeatures::VISION_ENCODE
            | uniserve_core::GenerationFeatures::LATENT_ENCODE
            | uniserve_core::GenerationFeatures::IMAGE_GENERATION,
        max_latent_units: u64::MAX,
        latent_downsample: 16,
        max_vae_grid_tokens: u32::MAX,
        max_vit_grid_tokens: u32::MAX,
        max_latent_feature_bytes: 256 << 20,
        max_vision_feature_bytes: 256 << 20,
        commit_marker_tokens: 2,
        max_cfg_branches: 3,
        encoder_cache_entries: 256,
    }
}

impl RuntimeProfile {
    /// Builds an autoregressive text profile for the model data type.
    pub fn ar(model_dtype: uniserve_core::ModelDtype) -> Self {
        Self {
            latent_dtype: worker_float_dtype(Some(model_dtype)),
            model_dtype,
            generation_limits: uniserve_core::GenerationLimits {
                features: uniserve_core::GenerationFeatures::UNDERSTANDING,
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..Default::default()
            },
            encoder_cache_entries: 0,
        }
    }

    /// Builds a diffusion-only profile for the model data type.
    pub fn diffusion(model_dtype: uniserve_core::ModelDtype) -> Self {
        Self {
            latent_dtype: worker_float_dtype(Some(model_dtype)),
            model_dtype,
            generation_limits: uniserve_core::GenerationLimits {
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..Default::default()
            },
            encoder_cache_entries: 0,
        }
    }

    /// Builds a unified multimodal profile with explicit generation limits.
    pub fn umm(
        model_dtype: uniserve_core::ModelDtype,
        generation_limits: uniserve_core::GenerationLimits,
    ) -> Self {
        let encoder_cache_entries = generation_limits.encoder_cache_entries as usize;
        Self {
            latent_dtype: worker_float_dtype(Some(model_dtype)),
            model_dtype,
            generation_limits,
            encoder_cache_entries,
        }
    }

    /// Intersects configured profile limits with capabilities advertised by the worker.
    fn resolved(mut self, info: &WorkerInfo) -> Self {
        let supports = |kind| info.supported_ops.contains(&kind);
        let mut available = uniserve_core::GenerationFeatures::empty();
        if supports(OpKind::ArExtend) && supports(OpKind::ArDecode) {
            available.insert(uniserve_core::GenerationFeatures::UNDERSTANDING);
        }
        if supports(OpKind::EncoderExecute) {
            available.insert(
                uniserve_core::GenerationFeatures::VISION_ENCODE
                    | uniserve_core::GenerationFeatures::LATENT_ENCODE,
            );
        }
        if supports(OpKind::DiffusionStep) && supports(OpKind::DiffusionDecode) {
            available.insert(uniserve_core::GenerationFeatures::IMAGE_GENERATION);
        }
        self.generation_limits.features &= available;
        self.generation_limits.max_latent_units = self
            .generation_limits
            .max_latent_units
            .min(info.latent_capacity_units());
        let latent_bound = self
            .generation_limits
            .max_latent_units
            .min(u64::from(u32::MAX)) as u32;
        self.generation_limits.max_vae_grid_tokens =
            self.generation_limits.max_vae_grid_tokens.min(latent_bound);
        self.generation_limits.max_vit_grid_tokens = self
            .generation_limits
            .max_vit_grid_tokens
            .min(info.max_batch_tokens);
        self.generation_limits.max_latent_feature_bytes = self
            .generation_limits
            .max_latent_feature_bytes
            .min(info.buffer_pool_bytes);
        self.generation_limits.max_vision_feature_bytes = self
            .generation_limits
            .max_vision_feature_bytes
            .min(info.buffer_pool_bytes);
        self.encoder_cache_entries = if info.buffer_pool_bytes == 0 {
            0
        } else {
            self.encoder_cache_entries
        };
        self.generation_limits.encoder_cache_entries = self
            .generation_limits
            .encoder_cache_entries
            .min(self.encoder_cache_entries.min(u32::MAX as usize) as u32);
        self
    }
}

/// Runtime state selected by the admitted request family.
pub enum Runtime {
    /// Autoregressive text runtime.
    Ar(ArRuntime),
    /// Diffusion-only media runtime.
    Diffusion(DiffusionRuntime),
    /// Unified multimodal runtime.
    Umm(UmmRuntime),
}

impl Runtime {
    /// Creates an initialized instance.
    fn new(family: RuntimeFamily, state: RuntimeState) -> Self {
        match family {
            RuntimeFamily::Ar => Self::Ar(ArRuntime(state)),
            RuntimeFamily::Diffusion => Self::Diffusion(DiffusionRuntime(state)),
            RuntimeFamily::Umm => Self::Umm(UmmRuntime(state)),
        }
    }

    /// Returns the operation family.
    const fn family(&self) -> RuntimeFamily {
        match self {
            Self::Ar(_) => RuntimeFamily::Ar,
            Self::Diffusion(_) => RuntimeFamily::Diffusion,
            Self::Umm(_) => RuntimeFamily::Umm,
        }
    }

    /// Returns whether the pool accepts the operation.
    const fn accepts(&self, family: RuntimeFamily) -> bool {
        matches!(
            (self.family(), family),
            (RuntimeFamily::Ar, RuntimeFamily::Ar)
                | (RuntimeFamily::Diffusion, RuntimeFamily::Diffusion)
                | (RuntimeFamily::Umm, RuntimeFamily::Ar | RuntimeFamily::Umm)
        )
    }

    /// Returns shared access to the execution state.
    fn state(&self) -> &RuntimeState {
        self
    }

    /// Returns mutable access to the execution state.
    fn state_mut(&mut self) -> &mut RuntimeState {
        self
    }
}

impl std::ops::Deref for Runtime {
    type Target = RuntimeState;

    /// Returns shared access to the wrapped value.
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ar(runtime) => &runtime.0,
            Self::Diffusion(runtime) => &runtime.0,
            Self::Umm(runtime) => &runtime.0,
        }
    }
}

impl std::ops::DerefMut for Runtime {
    /// Returns mutable access to the wrapped value.
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ar(runtime) => &mut runtime.0,
            Self::Diffusion(runtime) => &mut runtime.0,
            Self::Umm(runtime) => &mut runtime.0,
        }
    }
}

/// Single-threaded owner of scheduling, memory, execution, and request state.
pub struct EngineLoop {
    executor: Box<dyn Executor>,
    pending_submission: Option<Batch>,
    info: WorkerInfo,
    profile: RuntimeProfile,
    /// Resident resource ownership and reservation accounting.
    memory: Memory,
    runtime: Runtime,
    scheduler: Scheduler,
    /// Submitted operations, ordered completions, and per-batch timing state.
    /// Diagnostic: keep denoise steps out of batches that carry text rows.
    flow_exclusive_batch: bool,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    trace_sink: Option<crate::runtime::bench_trace::RuntimeTraceSink>,
    /// Largest number of operations observed in one submitted batch.
    pub peak_ops_in_batch: usize,
    /// Shared scheduler counters and latency accumulators.
    pub stats: Arc<SchedStats>,
}

impl std::ops::Deref for EngineLoop {
    type Target = Runtime;

    /// Returns shared access to the wrapped value.
    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

impl std::ops::DerefMut for EngineLoop {
    /// Returns mutable access to the wrapped value.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runtime
    }
}

/// Returns the current scheduler time.
fn now() -> f64 {
    // route through the single shared epoch helper so every
    // component's wall-clock timestamps match. It never panics on the hot loop:
    // a wall clock set before the UNIX epoch (or stepped backward) clamps to 0
    // instead of unwrapping the `Result`.
    uniserve_core::now_unix_secs()
}

/// Returns the worker floating-point data type.
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

/// Adds the worker forward map.
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

/// Returns the terminal reason for a closed request.
fn close_reason(reason: &FinishReason) -> CloseReason {
    match reason {
        FinishReason::Cancelled | FinishReason::Aborted => CloseReason::Cancelled,
        FinishReason::Error => CloseReason::Error,
        _ => CloseReason::Completed,
    }
}

/// Returns tokens that stop a device-relay successor before host semantic resolution.
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

/// Builds ranked log-probability entries.
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

/// Returns the operation batch classification.
fn batch_kind(operation_variant: RunKind) -> BatchKind {
    match operation_variant {
        RunKind::ArExtend | RunKind::EncoderVision | RunKind::EncoderLatent => BatchKind::Prefill,
        RunKind::ArDecode | RunKind::ArVerify => BatchKind::Decode,
        RunKind::DiffusionStep
        | RunKind::DiffusionDecode
        | RunKind::DiffusionPrepare
        | RunKind::DiffusionFinalize
        | RunKind::TransferProduct
        | RunKind::TransferKvPublish
        | RunKind::TransferKvInstall => BatchKind::Media,
    }
}

/// Returns the scheduling priority for a completion.
fn completion_priority(operation_variant: RunKind) -> u8 {
    match operation_variant {
        RunKind::DiffusionStep | RunKind::DiffusionFinalize | RunKind::TransferKvInstall => 0,
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

/// Computes physical input and visible-prefix lengths for KV-affecting transitions.
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
        | TransitionIntent::PrepareGen {
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

/// Serializes worker forward-pass counters into the scheduler trace schema.
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

/// Computes ceiling division for unsigned values.
fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    let divisor = divisor.max(1);
    value.div_ceil(divisor)
}

/// Computes the physical transformer-token work charged to one scheduler step.
///
/// Denoising executes every latent token once per CFG branch at every
/// timestep, so its cost multiplies the compiled latent geometry rather than the
/// scalar per-step token cost.
fn planned_op_token_cost(transition: &NextOp) -> usize {
    if transition.operation_variant != RunKind::DiffusionStep {
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

/// Computes the event capacity required before scheduling one transition.
fn transition_output_bound(transition: &NextOp) -> usize {
    match transition.operation_variant {
        RunKind::ArVerify => transition
            .validation
            .expected_text_tokens
            .map_or(4, |range| {
                usize::try_from(range.max)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(2)
                    .saturating_add(2)
            }),
        RunKind::ArExtend | RunKind::ArDecode => 4,
        RunKind::DiffusionStep => transition.token_cost.saturating_add(2),
        RunKind::DiffusionDecode => 2,
        RunKind::DiffusionFinalize => 3,
        _ => 2,
    }
}

/// Records the operation in the optional benchmark trace.
fn operation_trace(operation: &Operation, apply: &RuntimeApply) -> serde_json::Value {
    let parent_kind = match operation.parent.point {
        CheckpointPoint::Fixed(_) => "fixed",
        CheckpointPoint::DeviceSelected => "device_selected",
    };
    json!({
        "kind": operation.kind.as_str(),
        "domain": format!("{:?}", operation.domain()),
        "parent_kind": parent_kind,
        "predicated": operation.predicate().is_some(),
        "inputs": operation.inputs().len(),
        "outputs": operation.outputs().len(),
        "max_tokens": operation.bounds().max_tokens,
        "max_kv_pages": operation.bounds().max_kv_pages,
        "output_event_bound": apply.output_event_bound,
    })
}

/// Reads the denoising burst size from the environment.
fn denoise_step_burst_from_env() -> u16 {
    env::var(DENOISE_STEP_BURST_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u16>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DENOISE_STEP_BURST)
}

/// Builds the IPC [`CfgParams`] for a denoise op.
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

/// Returns the number of classifier-free-guidance branches.
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
