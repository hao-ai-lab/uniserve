//! Request algorithms and the single-owner engine control loop.
//!
//! The loop combines request progress, scheduler policy, storage allocations, and
//! asynchronous executor completions. All mutable engine state stays on its
//! owner thread.
//!
//! Scheduling admits from its waiting queue, chunks prefill against sequence and
//! token budgets, and reserves each request's maximum declared resources. A
//! resident request remains in place when capacity prevents relocation.
//!
//! Static worker state crosses the boundary once in [`NewRequest`]; subsequent
//! calls carry only step-specific deltas.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod config;
mod stats;
pub(crate) mod stats_report;

pub use config::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulerConfig, SchedulingPolicy,
};
use config::{MAX_NUM_SEQS, MAX_NUM_WAITING};
pub(crate) use execution::consuming_calls;
pub use stats::{
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats,
    SchedulerStats, TimingStats, WorkerStats,
};
pub use stats_report::SchedulerStatsReporter;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

mod admission;
mod allocation;
mod batching;
mod control;
mod denoising;
mod execution;
pub(crate) mod generation;
pub(crate) mod image_artifact;
mod inflight;
pub(crate) mod output;
mod placement;
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

use crate::handle::{Command, EVENT_BUFFER_CAPACITY, EventSendError, EventTx};
use crate::kv::{BlockPool, BlockTable, KvCacheCoordinator};
use crate::storage::{
    BufferPool, BufferSpan, KVCacheManager, KvAllocation, LatentPages, LatentPool, RequestPool,
    RequestSlot,
};
use crossbeam_channel::Receiver;
use uniserve_core::{
    ArtifactEvent, DiffusionRequest, EngineCoreOutput, FinishReason, GenerationRequest, MediaKind,
    PositionLogprobs, Request, SharedMedia, TokenLogprob,
};
use uniserve_core::{BlockId, ImageIngestStep, RejectionKind, encoder_cache_key};
use uniserve_core::{HashAlgo, RequestId, RuntimeFamily};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable as IpcBlockTable, Bounds, BufferAllocation, BufferId,
    CachePageAllocation, Call, CallId, CallKind, CallStatus, DEFAULT_COMPONENT, DType, DecodeRange,
    DimBound, ForwardBatch, ForwardStats, LatentParams, NewRequest, RequestKey, SamplingState,
    ShapeBound, TensorRef, TimingCounters, WorkerInfo,
};

use crate::executor::WorkerFailure;
use crate::scheduler::image_artifact::validate_png_artifact;
use allocation::{
    MediaAllocations, MediaStorage, MediaTensorAllocation, RequestAllocations, RetiringRequest,
    UnknownMediaWorker,
};
use denoising::{Denoising, LatentPlacement};
use generation::RequestState;
use inflight::{InflightCall, InflightInput, PendingCompletion, PendingFinish};
use output::RequestOutput;

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

/// Builds a validated image-completion event from a PNG payload.
fn image_done_event(image_id: u32, pixels_png_b64: String) -> Option<EngineCoreOutput> {
    let metadata = validate_png_artifact(&pixels_png_b64, None)?;
    Some(EngineCoreOutput::ImageDone {
        image_id,
        height: metadata.height,
        width: metadata.width,
        bytes: metadata.bytes,
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

struct MediaFlowState {
    request: DiffusionRequest,
    output: output::EventJournal,
    allocations: MediaAllocations,
    /// Logical products mapped to their reserved component output and unit slice.
    buffer_bindings: HashMap<BufferId, (String, u32, u32)>,
    conditioning: Option<TensorRef>,
    latents: Vec<TensorRef>,
    /// Decoded media units by the cursor of the round that produced them,
    /// until their encode round retires them.
    decoded_units: BTreeMap<u32, (u32, TensorRef)>,
    /// Encoded media units by the cursor of the round that produced them,
    /// until the muxer is handed them.
    encoded_units: BTreeMap<u32, (u32, TensorRef)>,
    /// Media unit count of each encode round that has completed and not yet
    /// been handed to the muxer, by the round's cursor.
    encoded_ready: BTreeMap<u32, u32>,
    /// Media units handed to the muxer so far; the next round it takes
    /// starts here.
    handed_units: u32,
    /// Encoded products carried by the muxing call in flight, retired with it.
    muxing_inputs: Vec<TensorRef>,
    audio: Option<TensorRef>,
    admission: NewRequest,
    admission_state: WorkerRegistration,
    text_encoding_scheduled: bool,
    latent_preparation_scheduled: bool,
    /// Solver progress of the request's latent trajectory.
    denoising: Denoising,
    scheduled_decode_units: u32,
    scheduled_encode_units: u32,
    encoded_video_units: u32,
    audio_decoding_scheduled: bool,
    audio_encoding_scheduled: bool,
    audio_encoded: bool,
    /// Whether a muxing call is in flight; the muxer takes one at a time.
    muxing_in_flight: bool,
    /// Whether the call that finalizes the artifact has been scheduled: the
    /// one that carries no media units, once every unit and the audio are in.
    final_muxing_scheduled: bool,
    muxed: bool,
    predecessor: CallId,
    terminal_intent: TerminalIntent,
    artifact: Option<ArtifactEvent>,
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
    /// Unix timestamp at which the request entered the waiting queue.
    queued_at: f64,
}

/// Permissive unified-multimodal limits for a scheduler built without
/// worker-advertised capacity.
///
/// A serving deployment supplies real limits through `EngineConfig`; this
/// default only bounds a scheduler constructed directly from an executor.
pub(crate) fn unbounded_umm_generation_limits() -> uniserve_core::GenerationLimits {
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

/// Single-threaded owner of scheduling, storage, execution, and request state.
pub struct Scheduler {
    executor: Box<dyn Executor>,
    inflight: inflight::Inflight,
    /// Resident resource ownership and reservation accounting.
    storage: allocation::Storage,
    placement: placement::Placement,
    info: WorkerInfo,
    generation_limits: uniserve_core::GenerationLimits,
    family: RuntimeFamily,
    ctrl: SpecialTokenIds,
    /// Token requests awaiting admission, in admission order: arrival order,
    /// or priority then arrival under the priority policy.
    waiting: VecDeque<RequestState>,
    /// Media requests awaiting admission, in arrival order.
    waiting_media: VecDeque<PendingMedia>,
    running: HashMap<RequestId, RequestState>,
    running_media: HashMap<RequestId, MediaFlowState>,
    retiring_requests: HashMap<RequestId, RetiringRequest>,
    transfer_capacity: usize,
    denoise_step_burst: u16,
    latent_dtype: Option<DType>,
    engine_id: u64,
    next_product_generation: u64,
    next_request_epoch: u64,
    config: SchedulerConfig,
    running_order: Vec<RequestId>,
    output: output::OutputSender,
    prefer_media: bool,
    /// Engine-fatal latch: set when the executor/worker dies or a scheduler
    /// invariant breaks; the control loop exits and the host converts this
    /// into engine-dead.
    fatal: bool,
    /// Shared scheduler counters and latency accumulators.
    pub stats: Arc<SchedulerStats>,
}

impl Scheduler {
    /// Latches engine-fatal for a scheduler invariant that no longer holds.
    ///
    /// Scheduling cannot continue from inconsistent state, so the control loop
    /// stops at its next check and fails every request, as it does when a
    /// worker dies. The caller abandons the operation that found the violation.
    fn invariant_broken(&mut self, invariant: &str) {
        tracing::error!(invariant, "scheduler invariant broken; stopping the engine");
        self.fatal = true;
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

/// Returns the call batch classification.
fn batch_kind(call_variant: CallKind) -> BatchKind {
    match call_variant {
        CallKind::Forward(ForwardMode::Prefill)
        | CallKind::Media(MediaCall::TextEncoding)
        | CallKind::Media(MediaCall::VisionEncoding)
        | CallKind::Media(MediaCall::LatentEncoding) => BatchKind::Prefill,
        CallKind::Forward(ForwardMode::Decode) | CallKind::Forward(ForwardMode::Verify) => {
            BatchKind::Decode
        }
        CallKind::Media(MediaCall::Denoising)
        | CallKind::Media(MediaCall::VideoDecoding)
        | CallKind::Media(MediaCall::LatentPreparation)
        | CallKind::Media(
            MediaCall::VideoEncoding
            | MediaCall::AudioEncoding
            | MediaCall::AudioDecoding
            | MediaCall::Muxing,
        )
        | CallKind::Media(MediaCall::ImageDecoding)
        | CallKind::Transfer(TransferMode::Tensor)
        | CallKind::Transfer(TransferMode::KvPublish)
        | CallKind::Transfer(TransferMode::KvInstall) => BatchKind::Media,
    }
}

/// Returns the scheduling priority for a completion.
fn completion_priority(call_variant: CallKind) -> u8 {
    match call_variant {
        CallKind::Media(MediaCall::Denoising)
        | CallKind::Media(MediaCall::ImageDecoding)
        | CallKind::Transfer(TransferMode::KvInstall) => 0,
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

/// Computes ceiling division for unsigned values.
fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    let divisor = divisor.max(1);
    value.div_ceil(divisor)
}

/// Computes the event capacity required before scheduling one transition.
fn call_output_bound(code: CallKind, bounds: &uniserve_worker_ipc::Bounds) -> usize {
    match code {
        CallKind::Forward(ForwardMode::Verify) => (bounds.max_tokens as usize)
            .saturating_mul(2)
            .saturating_add(2),
        CallKind::Forward(ForwardMode::Prefill) | CallKind::Forward(ForwardMode::Decode) => 4,
        CallKind::Media(MediaCall::Denoising) => (bounds.max_tokens as usize).saturating_add(2),
        CallKind::Media(MediaCall::VideoDecoding) => 2,
        CallKind::Media(MediaCall::ImageDecoding) => 3,
        _ => 2,
    }
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
