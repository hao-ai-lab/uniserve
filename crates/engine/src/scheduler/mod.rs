//! Request algorithms and the single-owner engine control loop.
//!
//! The loop combines request progress, scheduler policy, memory params, and
//! asynchronous executor completions. All mutable engine state stays on its
//! owner thread.
//!
//! Scheduling admits from its waiting queue, chunks prefill against sequence and
//! token budgets, and reserves each request's maximum declared resources. A
//! resident request remains in place when capacity prevents relocation.
//!
//! Static worker state crosses the boundary once in [`NewRequest`]; subsequent
//! operations carry only step-specific deltas.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod config;
mod stats;
pub(crate) mod stats_report;

pub use config::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulerConfig, SchedulingPolicy,
};
use config::{MAX_NUM_SEQS, MAX_NUM_WAITING};
pub use stats::{
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats,
    SchedulerStats, TimingStats, WorkerStats,
};
pub use stats_report::SchedulerStatsReporter;
use uniserve_worker_ipc::{ForwardMode, PipelineStage, TransferMode};

mod admission;
mod allocation;
mod batching;
pub(crate) mod bench_trace;
mod control;
mod execution;
pub(crate) mod generation;
pub(crate) mod image_artifact;
mod inflight;
pub(crate) mod output;
mod run;

pub(crate) use crate::executor::{
    BatchResult, ExecutionBatch, Executor, ExecutorSubmitError, RequestPlacement,
};

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use std::env;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::scheduler::generation::{
    GenerationPhase as Phase, consumes_image_features, is_feedback_computation, is_prompt_extend,
};

use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EventRx, EventSendError, EventTx, event_channel,
};
use crate::kv::{BlockPool, BlockTable, KvCacheCoordinator};
use crate::memory::{Allocation, BufferPool, KVCacheManager, LatentPool, RequestPool};
use crossbeam_channel::Receiver;
use uniserve_core::{
    ArtifactEvent, DiffusionRequest, EngineCoreOutput, FinishReason, GenerationRequest, MediaKind,
    PositionLogprobs, Request, SharedMedia, TokenLogprob,
};
use uniserve_core::{BlockId, ImageIngestStep, encoder_cache_key};
use uniserve_core::{HashAlgo, RequestId, RuntimeFamily};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable as IpcBlockTable, Bounds, BufferAllocation, BufferId,
    CachePageAllocation, Computation, ComputationId, DType, DecodeRange, DimBound, ForwardBatch,
    ForwardStats, LatentParams, NewRequest, OpStatus, RequestKey, SamplingState, ScheduledRequest,
    ShapeBound, TensorRef, TimingCounters, UmmRequestParams, WorkerInfo,
};

use crate::executor::WorkerFailure;
use crate::scheduler::image_artifact::validate_png_artifact;
use inflight::{InflightInput, InflightOp, PendingBatch, PendingCompletion, PendingFinish};
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
fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<EngineCoreOutput> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(EngineCoreOutput::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
        sha256: metadata.sha256,
        pixels_png_b64,
    })
}

#[derive(Clone)]
/// Model token identifiers used for sequence and image-content boundaries.
pub struct SpecialTokenIds {
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
}

impl Default for SpecialTokenIds {
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
pub(crate) struct RequestState {
    pub req: GenerationRequest,
    pub(crate) finish_token_ids: Vec<u32>,
    /// The cached sequence mappings. Each table owns its physical page
    /// references and therefore has exactly the request's lifetime.
    allocations: Option<RequestAllocations>,
    flow_prefix: Option<FlowPrefixState>,
    /// Admission generation used to reject results from prior request lifetimes.
    pub(crate) request_epoch: u64,
    /// Most recent accepted state-producing operation.
    pub(crate) last_state_op_id: ComputationId,
    /// Last device token retained until a consumer is registered.
    pub(crate) latest_token: Option<TensorRef>,
    /// A false device predicate invalidated the unresolved successor chain.
    /// Already-submitted descendants must drain before scheduling resumes from
    /// the last host-observed execution result.
    pub(crate) speculative_chain_invalidated: bool,
    output: RequestOutput,
    /// Current accepted computation stage; scheduling does not advance it.
    phase: Phase,
    /// Prompt token range already accepted by the worker.
    num_computed_prompt_tokens: u32,
    num_ingested_images: usize,
    image_encoder_index: usize,
    input_image_features: Option<TensorRef>,
    round_closing: bool,
    /// Model position and accepted physical KV length differ for image inputs.
    logical_position: u32,
    kv_visible_len: u32,
    next_token: u32,
    num_generated_tokens: usize,
    image_id: u32,
    num_generated_images: usize,
    image_reservation_pending: bool,
    num_completed_denoise_steps: u16,
    /// Exact generations retained for conditioning, denoising, and feedback.
    image_conditioning: Option<uniserve_worker_ipc::BufferId>,
    image_latent: Option<TensorRef>,
    feedback_encoder_index: usize,
    feedback_source: Option<TensorRef>,
    feedback_features: Option<TensorRef>,
    /// Whether the current epoch is registered with its execution workers.
    worker_registered: bool,
    /// Number of positive-branch KV blocks already delivered to the worker.
    num_kv_blocks_sent: usize,
    /// Whether admission reserves the complete multimodal KV requirement.
    reserve_worstcase: bool,
    max_reserved_kv_blocks: usize,
    /// Per-group prefix hashes retained for publishing reusable input blocks.
    prefix_block_hashes: Vec<Vec<u64>>,
    prefix_cached: bool,
    /// Accepted text and control tokens used by penalties and trigger matching.
    generated_token_ids: Vec<u32>,
    /// Tokens since the last model round boundary, used by suffix-trigger matching.
    round_token_ids: Vec<u32>,
    text_tokens_since_image: usize,
    /// Host image bytes retained only while artifact-based feedback needs them.
    feedback_image_b64: Option<String>,
    /// False after a completed computation creates state unavailable from the input.
    replayable: bool,
    encoder_cache_pins: Vec<EncoderCachePin>,
    transient_encoder_products: Vec<TensorRef>,
    pub queued_at: f64,
    pub(crate) terminal_intent: TerminalIntent,
}

/// Request-held reference to a reusable encoder-cache entry.
struct EncoderCachePin {
    key: u64,
    product: TensorRef,
}

/// Terminal action deferred until outstanding work is reconciled.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) enum TerminalIntent {
    #[default]
    None,
    Finish(FinishReason),
    Failure(String),
}

impl TerminalIntent {
    /// Returns whether accepted descendants must drain before retirement.
    pub(crate) const fn is_terminal(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Records a terminal reason without replacing a previously resolved failure.
    fn finish(&mut self, reason: FinishReason) {
        if matches!(self, Self::None) {
            *self = Self::Finish(reason);
        }
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

/// Allocation ownership retained until the exact request epoch closes on every rank.
struct RetiringRequest {
    request_key: RequestKey,
    allocations: Vec<Allocation>,
    buffers: HashMap<BufferId, Allocation>,
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

    /// Consumes this active layout into the allocation handles needed for reclamation.
    fn into_allocations(self) -> impl Iterator<Item = Allocation> {
        self.buffers
            .into_values()
            .chain(self.latent)
            .chain([self.kv, self.request_slot])
    }

    /// Releases the owned request allocation.
    fn free(self, scheduler: &mut Scheduler) {
        for allocation in self.into_allocations() {
            scheduler.free_allocation(allocation);
        }
    }
}

impl RequestState {
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
        !self.req.multimodal_inputs.images.is_empty()
    }

    /// Returns whether execution continues after committing generation output.
    fn continues_after_gen_commit(&self) -> bool {
        self.req.continues_after_image()
    }

    /// Returns whether the request can open a generation branch.
    fn can_open_gen_branch(&self) -> bool {
        self.req.generates_images()
            && self.num_generated_images < self.req.image.max_images as usize
    }

    /// Returns whether generation starts after context ingestion.
    fn starts_gen_after_context(&self) -> bool {
        self.req.starts_with_image() && self.can_open_gen_branch()
    }

    /// Returns whether the text request can be replayed.
    pub(crate) fn is_replayable_text(&self) -> bool {
        self.replayable
    }

    /// Returns the pending image step, if one exists.
    fn pending_image_step(&self) -> Option<ImageIngestStep> {
        self.req
            .multimodal_inputs
            .images
            .get(self.num_ingested_images)?
            .encoders
            .get(self.image_encoder_index)
            .map(|input| input.encoder)
    }
}

struct MediaFlowState {
    request: DiffusionRequest,
    event_tx: EventTx,
    allocations: MediaAllocations,
    conditioning: Option<TensorRef>,
    latents: Vec<TensorRef>,
    video_segments: BTreeMap<u32, (u32, TensorRef)>,
    audio: Option<TensorRef>,
    admission: NewRequest,
    admission_state: WorkerRegistration,
    text_encoding_scheduled: bool,
    latent_preparation_scheduled: bool,
    num_scheduled_steps: u32,
    num_completed_steps: u32,
    num_scheduled_decode_chunks: u32,
    num_scheduled_video_chunks: u32,
    num_encoded_video_chunks: u32,
    audio_decoding_scheduled: bool,
    audio_encoding_scheduled: bool,
    audio_encoded: bool,
    muxing_scheduled: bool,
    muxed: bool,
    predecessor: ComputationId,
    terminal_intent: TerminalIntent,
    artifact: Option<ArtifactEvent>,
}

struct MediaAllocations {
    tensors: HashMap<(String, u32), MediaTensorAllocation>,
    request_slot: Allocation,
}

/// A request reserves each declared result once. Video ranges occupy disjoint
/// slices of its temporal result, independent of decoder Worker width.
struct MediaTensorAllocation {
    allocation: Allocation,
    dtype: DType,
    shape_bound: ShapeBound,
}

impl MediaTensorAllocation {
    fn bind(&self, product: &TensorRef, start_unit: u32) -> BufferAllocation {
        let Allocation::Buffer { offset, .. } = &self.allocation else {
            unreachable!("media tensor has buffer storage");
        };
        let unit_bytes = match product.shape_bound.dims.first() {
            Some(DimBound::Static(units)) if start_unit > 0 => {
                product.max_bytes() / u64::from(*units)
            }
            _ => 0,
        };
        BufferAllocation {
            buffer: product.buffer_id(),
            offset: offset + u64::from(start_unit) * unit_bytes,
            bytes: product.max_bytes(),
        }
    }
}

impl MediaAllocations {
    /// Returns the request-slot identifier.
    fn request_slot(&self) -> u32 {
        self.request_slot
            .request_slot()
            .expect("media request slot allocation")
    }

    /// Consumes media layout metadata once only storage lifetime remains.
    fn into_allocations(self) -> impl Iterator<Item = Allocation> {
        self.tensors
            .into_values()
            .map(|tensor| tensor.allocation)
            .chain([self.request_slot])
    }

    /// Releases the owned request allocation.
    fn free(self, scheduler: &mut Scheduler) {
        for allocation in self.into_allocations() {
            scheduler.free_allocation(allocation);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerRegistration {
    Unsubmitted,
    InFlight,
    Registered,
}

enum DiffusionTerminal {
    Completed(ArtifactEvent),
    Failed(String),
    Finished(FinishReason),
}

struct PendingMedia {
    request: DiffusionRequest,
    event_tx: EventTx,
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

/// Single-threaded owner of scheduling, memory, execution, and request state.
pub struct Scheduler {
    executor: Box<dyn Executor>,
    pending_submissions: VecDeque<ExecutionBatch>,
    worker_affinity: HashMap<(RequestKey, String), crate::WorkerId>,
    info: WorkerInfo,
    generation_limits: uniserve_core::GenerationLimits,
    /// Resident resource ownership and reservation accounting.
    cache: Option<KVCacheManager>,
    encoder_cache: crate::kv::EncoderCacheManager,
    reserved_encoder_entries: usize,
    request_pool: RequestPool,
    latent_pool: LatentPool,
    reserved_blocks: usize,
    buffer_pool: BufferPool,
    encoder_buffers: HashMap<uniserve_worker_ipc::BufferId, Allocation>,
    family: RuntimeFamily,
    ctrl: SpecialTokenIds,
    waiting: HashMap<RequestId, RequestState>,
    waiting_media: HashMap<RequestId, PendingMedia>,
    running: HashMap<RequestId, RequestState>,
    running_media: HashMap<RequestId, MediaFlowState>,
    retiring_requests: HashMap<RequestId, RetiringRequest>,
    transfer_capacity: usize,
    num_pending_transfers: usize,
    batch_id: u64,
    next_arrival_seq: u64,
    pending_operations: HashMap<RequestId, VecDeque<InflightOp>>,
    pending_completions: HashMap<RequestId, BTreeMap<ComputationId, PendingCompletion>>,
    pending_finishes: HashMap<RequestId, PendingFinish>,
    pending_batches: HashMap<u64, PendingBatch>,
    denoise_step_burst: u16,
    latent_dtype: Option<DType>,
    pending_commands: VecDeque<BatchCommand>,
    pending_buffer_frees: HashMap<BufferId, Allocation>,
    engine_id: u64,
    next_product_generation: u64,
    next_request_epoch: u64,
    config: SchedulerConfig,
    waiting_order: VecDeque<RequestId>,
    waiting_media_order: VecDeque<RequestId>,
    running_order: Vec<RequestId>,
    output: output::OutputSender,
    prefer_media: bool,
    /// Diagnostic: keep denoise steps out of batches that carry text rows.
    flow_exclusive_batch: bool,
    /// Engine-fatal latch: set when the executor/worker dies;
    /// the control loop exits and the host converts this into engine-dead.
    fatal: bool,
    trace_sink: Option<crate::scheduler::bench_trace::RuntimeTraceSink>,
    /// Largest number of operations observed in one submitted batch.
    pub peak_ops_in_batch: usize,
    /// Shared scheduler counters and latency accumulators.
    pub stats: Arc<SchedulerStats>,
}

impl Scheduler {
    /// Serializes the generation decisions consumed by scheduler trace readers.
    fn generation_trace(request: &GenerationRequest) -> serde_json::Value {
        serde_json::json!({
            "und_decode": request.decodes_text(),
            "und_tokens": request.emits_text(),
            "gen_output": request.generates_images(),
            "start_gen_after_context": request.starts_with_image(),
            "generated_image_feedback": request.feeds_back_images(),
            "continue_after_gen_commit": request.continues_after_image(),
            "finish_after_gen_commit": request.finishes_after_image(),
        })
    }

    /// Determines whether this service's configured model accepts a request family.
    const fn accepts_family(&self, family: RuntimeFamily) -> bool {
        matches!(
            (self.family, family),
            (RuntimeFamily::Ar, RuntimeFamily::Ar)
                | (RuntimeFamily::Diffusion, RuntimeFamily::Diffusion)
                | (RuntimeFamily::Umm, RuntimeFamily::Ar | RuntimeFamily::Umm)
        )
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
    if request.generates_images()
        && let Some(trigger) = request.image_generation.trigger.direct_token()
    {
        finish_token_ids.push(trigger);
    }
    finish_token_ids.sort_unstable();
    finish_token_ids.dedup();
    finish_token_ids
}

/// Returns the operation batch classification.
fn batch_kind(operation_variant: Computation) -> BatchKind {
    match operation_variant {
        Computation::Forward(ForwardMode::Prefill)
        | Computation::Pipeline(PipelineStage::TextEncoding)
        | Computation::Pipeline(PipelineStage::VisionEncoding)
        | Computation::Pipeline(PipelineStage::LatentEncoding) => BatchKind::Prefill,
        Computation::Forward(ForwardMode::Decode) | Computation::Forward(ForwardMode::Verify) => {
            BatchKind::Decode
        }
        Computation::Pipeline(PipelineStage::Denoising)
        | Computation::Pipeline(PipelineStage::VideoDecoding)
        | Computation::Pipeline(PipelineStage::LatentPreparation)
        | Computation::Pipeline(
            PipelineStage::VideoEncoding
            | PipelineStage::AudioEncoding
            | PipelineStage::AudioDecoding
            | PipelineStage::Muxing,
        )
        | Computation::Pipeline(PipelineStage::ImageDecoding)
        | Computation::Transfer(TransferMode::Tensor)
        | Computation::Transfer(TransferMode::KvPublish)
        | Computation::Transfer(TransferMode::KvInstall) => BatchKind::Media,
        Computation::Forward(ForwardMode::Mixed) => {
            unreachable!("scheduler selects individual forward computations")
        }
    }
}

/// Returns the scheduling priority for a completion.
fn completion_priority(operation_variant: Computation) -> u8 {
    match operation_variant {
        Computation::Pipeline(PipelineStage::Denoising)
        | Computation::Pipeline(PipelineStage::ImageDecoding)
        | Computation::Transfer(TransferMode::KvInstall) => 0,
        _ => 1,
    }
}

const OUTPUT_JOURNAL_CAPACITY: usize = EVENT_BUFFER_CAPACITY;
const OUTPUT_TERMINAL_RESERVE: usize = 2;

#[derive(Debug, Clone, Copy)]
struct KvLengths {
    input: u32,
    visible: u32,
}

/// Serializes worker forward-pass counters into the scheduler trace schema.
fn worker_forward_stats_trace(stats: &ForwardStats) -> serde_json::Value {
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

/// Computes the event capacity required before scheduling one transition.
fn operation_output_bound(code: Computation, bounds: &uniserve_worker_ipc::Bounds) -> usize {
    match code {
        Computation::Forward(ForwardMode::Verify) => (bounds.max_tokens as usize)
            .saturating_mul(2)
            .saturating_add(2),
        Computation::Forward(ForwardMode::Prefill) | Computation::Forward(ForwardMode::Decode) => 4,
        Computation::Pipeline(PipelineStage::Denoising) => {
            (bounds.max_tokens as usize).saturating_add(2)
        }
        Computation::Pipeline(PipelineStage::VideoDecoding) => 2,
        Computation::Pipeline(PipelineStage::ImageDecoding) => 3,
        _ => 2,
    }
}

/// Records the operation in the optional benchmark trace.
fn operation_trace(operation: &ScheduledRequest) -> serde_json::Value {
    let parent_kind = match operation.predecessor {
        Some(_) => "operation",
        None => "none",
    };
    json!({
        "kind": operation.code.as_str(),
            "parent_kind": parent_kind,
        "predicated": operation.predicate.as_ref().is_some(),
        "inputs": operation.tensor_inputs().count(),
        "outputs": operation.tensor_outputs().count(),
        "max_tokens": operation.bounds.max_tokens,
        "max_kv_pages": operation.bounds.max_kv_pages,
        "output_event_bound": operation_output_bound(operation.code, &operation.bounds),
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

/// Returns the number of classifier-free-guidance branches.
fn cfg_branch_count(image: &uniserve_core::ImageParams) -> u8 {
    image.cfg_branch_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{GenerationConstraint, ImageGenerationConfig, ImageParams, SamplingParams};

    fn request(id: u64, tokens: usize) -> GenerationRequest {
        let policy = ImageGenerationConfig::default();
        let constraint = GenerationConstraint::UndOnly;
        GenerationRequest {
            request_id: RequestId(id),
            prompt_token_ids: vec![0; tokens],
            multimodal_inputs: Default::default(),
            negative_prompt_token_ids: Vec::new(),
            constraint,
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            include_stop_token: false,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            image_generation: policy,
        }
    }

    #[test]
    fn direct_gen_trigger_stops_a_registered_text_successor() {
        let mut req = request(1, 8);
        req.constraint = GenerationConstraint::Default;
        req.image_generation.trigger = uniserve_core::ImageTrigger::Token { token_id: 4_242 };

        let stops = finish_token_ids(&req, &[151_643, 151_645]);

        assert_eq!(stops, vec![4_242, 151_643, 151_645]);
    }
}
