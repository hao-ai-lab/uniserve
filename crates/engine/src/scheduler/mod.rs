//! Request algorithms and the single-owner engine control loop.
//!
//! The loop combines request progress, scheduler policy, storage allocations, and
//! asynchronous executor completions. All mutable engine state stays on its
//! owner thread.
//!
//! Scheduling admits from its waiting queue and chunks prefill against sequence
//! and token budgets. A token request with context images or image generation
//! reserves its worst-case KV blocks at admission; a text-only request needs
//! only the blocks of its first prefill chunk. A request larger than the total
//! KV capacity is rejected. A resident request is never relocated or
//! preempted: when capacity is short, admission waits.
//!
//! Static worker state crosses the boundary once in [`NewRequest`]; subsequent
//! calls carry only step-specific deltas.
//!
//! Submodules:
//!
//! - `run`: construction and the owner-thread loop.
//! - `admission`: admission, resource reservation, and waiting-queue insertion.
//! - `batching`: generation-path batch selection and call planning.
//! - `generation`: per-request generation state, call builders, and result
//!   validation.
//! - `denoising`: solver progress of one latent trajectory.
//! - `graph`: the call graph of one video request.
//! - `execution`: submission, completion application, video media scheduling,
//!   and allocation reclamation.
//! - `inflight`: submitted batches and calls, and request-local completion
//!   ordering.
//! - `output`: public events, semantic result resolution, and termination.
//! - `allocation` and `placement`: storage ownership and worker placement.
//! - `control`: cancellation and frontend lifecycle commands.
//! - `config`, `stats`, and `stats_report`: limits and statistics.
//! - `image_artifact`: validation of worker-produced PNG artifacts.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod config;
mod stats;
pub(crate) mod stats_report;

pub use config::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, SchedulerConfig, SchedulingPolicy,
};
use config::{MAX_NUM_SEQS, MAX_NUM_WAITING};
pub(crate) use execution::generation_consuming_calls;
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
pub(crate) mod graph;
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
use crate::kv::{BlockPool, BlockTable, KvAllocation, KvCacheCoordinator};
use crate::storage::{
    BufferPool, BufferSpan, KVCacheManager, LatentPages, LatentPool, RequestPool, RequestSlot,
};
use crossbeam_channel::Receiver;
use uniserve_core::{
    ArtifactEvent, DiffusionRequest, EngineCoreOutput, FinishReason, GenerationRequest, MediaKind,
    PositionLogprobs, Request, SharedMedia, TokenLogprob,
};
use uniserve_core::{HashAlgo, RequestId, RuntimeFamily};
use uniserve_core::{ImageIngestStep, RejectionKind, UnitId, encoder_cache_key};
use uniserve_worker_ipc::{
    ArRequestParams, BatchCommand, BlockTable as IpcBlockTable, Bounds, BufferAllocation, BufferId,
    CacheUnitAllocation, Call, CallId, CallKind, CallStatus, DEFAULT_COMPONENT, DType, DecodeRange,
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
use graph::{VideoGraph, product};
use inflight::{InflightCall, InflightInput, PendingCompletion, PendingFinish};
use output::RequestOutput;

/// Number of denoising steps planned for one scheduling burst by default.
pub(crate) const DEFAULT_DENOISE_STEP_BURST: u16 = 1;
/// Upper bound on concurrently reserved transfer calls. Construction clamps
/// the executor's `queue_depth * max_batch_calls` to between one and this
/// value to form `Scheduler::transfer_capacity`.
const MAX_INFLIGHT_TRANSFERS: usize = 256;

/// Execution lane a call kind is batched in (see `batch_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BatchKind {
    /// Every forward prefill, and text, vision, and latent encoding.
    Prefill,
    /// Token decode and speculative verification.
    Decode,
    /// Latent preparation, denoising, image, video, and audio decoding, video
    /// and audio encoding, muxing, and transfers.
    Media,
}

/// Timeout of `Executor::poll` while the loop parks
/// (`Scheduler::park_for_progress`). A result ends the park, and an executor
/// with wake integration also returns early on commands and consumer reads;
/// the timeout is only a liveness deadline.
const IDLE_LIVENESS_POLL: Duration = Duration::from_millis(500);
/// Number of batches carrying prefill-lane calls that may await their
/// results before ready decode work takes precedence over further prefill
/// (`Scheduler::select_batch_kind`).
const PREFILL_WINDOW_CREDITS: usize = 2;
/// Environment variable overriding `DEFAULT_DENOISE_STEP_BURST`; a missing,
/// unparsable, or zero value selects the default.
const DENOISE_STEP_BURST_ENV: &str = "UNISERVE_DENOISE_STEP_BURST";

/// Builds a validated image-completion event from a base64 PNG payload.
///
/// Fully decodes the PNG through `validate_png_artifact`; returns `None` when
/// the payload is not valid base64 or not a fully decodable PNG with nonzero
/// dimensions.
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
///
/// Serving resolves these from the loaded model's configuration in the server
/// crate (`special_token_ids`); the `Default` values serve schedulers that
/// tests and examples construct directly.
pub struct SpecialTokenIds {
    /// Beginning-of-sequence token identifier.
    pub bos: u32,
    /// Token identifiers that terminate generation. The first entry is the
    /// primary EOS, which the output path substitutes for a completion that
    /// commits no token, so the list must not be empty.
    pub eos: Vec<u32>,
    /// Token identifier that terminates encoded image content.
    pub end_of_image: u32,
}

impl Default for SpecialTokenIds {
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

    /// Records a finish reason unless a terminal intent (finish or failure)
    /// is already recorded; the first recorded intent wins.
    fn finish(&mut self, reason: FinishReason) {
        if matches!(self, Self::None) {
            *self = Self::Finish(reason);
        }
    }
}

/// Negative-prompt KV prefix of a token request whose image generation uses
/// more than one guidance branch (`Scheduler::ensure_flow_prefix`).
///
/// The prefix holds its own request-pool slot and KV allocation, separate
/// from the request's main `RequestAllocations`, until
/// `Scheduler::free_flow_prefix` releases them after the last denoising step
/// or the request finishes.
struct FlowPrefixState {
    allocations: RequestAllocations,
    /// Units per KV group, as `(group_id, unit_ids)`, not yet declared to the
    /// worker; the next denoising call drains them as fresh-unit allocations.
    new_units: Vec<(u32, Vec<UnitId>)>,
    /// Set by the first successful denoising completion. Until then every
    /// denoising call carries the prefix's block tables and, when the negative
    /// prompt is not empty, prefills it.
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

/// Scheduler state of one admitted diffusion (media) request, advanced by the
/// media path in `execution` rather than by `RequestState`.
struct MediaFlowState {
    request: DiffusionRequest,
    /// The calls the request runs and the products connecting them.
    graph: VideoGraph,
    output: output::EventJournal,
    allocations: MediaAllocations,
    /// Each product the request reserved, by its declared name, with the
    /// component output it is reserved under.
    reservations: HashMap<&'static str, (String, u32)>,
    /// Logical products mapped to their reserved component output and unit slice.
    buffer_bindings: HashMap<BufferId, (String, u32, u32)>,
    /// The media reader's products, by name, once its call is scheduled.
    condition_media: HashMap<&'static str, TensorRef>,
    /// The vision encoder's token features, once its call is scheduled.
    vision_features: Option<TensorRef>,
    /// Visual condition units whose latent encoding is scheduled, and of
    /// those, the units whose encoding completed.
    scheduled_condition_units: u32,
    encoded_condition_units: u32,
    /// Each visual latent encoding round's rows, in unit order.
    condition_video_latents: Vec<TensorRef>,
    /// The audio latent encoding's rows, once its call is scheduled.
    condition_audio_latents: Option<TensorRef>,
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
    /// Static worker admission, sent as a `BatchCommand::Start` in the batch
    /// of the request's first scheduled call.
    admission: NewRequest,
    admission_state: WorkerRegistration,
    media_reading_scheduled: bool,
    vision_encoding_scheduled: bool,
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

/// Progress of a media request's worker admission (`NewRequest`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerRegistration {
    /// No scheduled call has carried the admission yet; no worker holds
    /// state for the request, so retirement needs no `Finish` command.
    Unsubmitted,
    /// The call carrying the admission is in flight; the request offers no
    /// further calls until it resolves.
    InFlight,
    /// A valid result of the admitting call has confirmed registration.
    Registered,
}

enum DiffusionTerminal {
    Completed(ArtifactEvent),
    Failed(String),
    Finished(FinishReason),
}

struct PendingMedia {
    request: DiffusionRequest,
    /// The request's call graph, built when it was queued.
    graph: VideoGraph,
    event_tx: EventTx,
    /// Unix timestamp at which the request entered the waiting queue.
    queued_at: f64,
}

/// Permissive unified-multimodal limits for a scheduler built without
/// model-derived limits.
///
/// `Scheduler::with_config_for_family` uses them for `RuntimeFamily::Umm`, and
/// `EngineConfig::sim` uses them as its default. Construction intersects them
/// with worker-reported capacities (`resolve_generation_limits` in `run`), so
/// the effective bounds come from the loaded workers. A serving deployment
/// supplies model limits through `EngineConfig`.
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
    /// Aggregate capacity view of the loaded workers.
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
    /// Finished requests whose storage stays allocated until their workers
    /// acknowledge the `Finish` command.
    retiring_requests: HashMap<RequestId, RetiringRequest>,
    /// Maximum number of transfer calls holding a reservation at once
    /// (`Inflight::num_pending_transfers`).
    transfer_capacity: usize,
    /// Maximum denoising steps one call covers.
    denoise_step_burst: u16,
    /// IPC dtype of image latents, mapped from the model dtype by
    /// `worker_float_dtype`.
    latent_dtype: Option<DType>,
    /// `engine_id` of every `RequestKey` this scheduler issues.
    engine_id: u64,
    /// Engine-wide counter for product generations, drawn by
    /// `generation::register_call` and `Scheduler::media_completion_product`;
    /// both fail once a generation would exceed `u32::MAX`.
    next_product_generation: u64,
    /// Epoch stamped on the next request state the scheduler creates, so
    /// results from an earlier lifetime of a request id never match it.
    next_request_epoch: u64,
    config: SchedulerConfig,
    /// Token and media requests in admission order. Media scheduling visits
    /// them in this order; batch assembly uses it to break ties within
    /// `assembly_priority`.
    running_order: Vec<RequestId>,
    output: output::OutputSender,
    /// Whether the next scheduling pass tries video media work before
    /// generation work (`Scheduler::schedule_batches`).
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

/// Returns wall-clock seconds since the Unix epoch.
///
/// Uses the shared `uniserve_core::now_unix_secs` so scheduler timestamps
/// match other components'. The clock is not monotonic; a time before the
/// epoch reads as zero instead of panicking.
fn now() -> f64 {
    uniserve_core::now_unix_secs()
}

/// Maps a model dtype to the IPC dtype of worker latents; `None` maps to
/// `None`.
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

/// Adds each per-key delta of a worker forward statistic into the shared
/// cumulative map in `SchedulerStats`. A poisoned lock is recovered rather
/// than propagated.
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
/// The sorted, deduplicated list holds the request's stop token ids, the EOS
/// ids unless `ignore_eos` is set, and the image-generation trigger when the
/// request generates images through a direct trigger token. It reaches the
/// worker in `ArRequestParams::finish_token_ids` and bounds verified-draft
/// prefixes in `validate_generation_result`.
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

/// Classifies a call kind into its execution lane.
///
/// Lane selection (`Scheduler::select_batch_kind`), batch assembly, and the
/// prefill flag set by `Inflight::register_pending_batch` read this mapping.
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
            MediaCall::MediaReading
            | MediaCall::VideoEncoding
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

/// Returns the order in which ready completions are applied; lower values go
/// first, and equal values keep arrival order.
///
/// Denoising, image decoding, and KV installation completions precede all
/// other kinds (`Inflight::take_ready_completions` and the resolution pass in
/// `execution`).
fn completion_priority(call_variant: CallKind) -> u8 {
    match call_variant {
        CallKind::Media(MediaCall::Denoising)
        | CallKind::Media(MediaCall::ImageDecoding)
        | CallKind::Transfer(TransferMode::KvInstall) => 0,
        _ => 1,
    }
}

/// Events a request's output journal may hold beyond what its bounded event
/// channel holds.
const OUTPUT_JOURNAL_CAPACITY: usize = EVENT_BUFFER_CAPACITY;
/// Output capacity, in events, that `Scheduler::output_window_ready` keeps
/// free for the request's terminal events.
const OUTPUT_TERMINAL_RESERVE: usize = 2;

/// KV extent of one planned generation call, in tokens.
#[derive(Debug, Clone, Copy)]
struct KvLengths {
    /// Tokens the call's KV-writing forward row appends: the prompt slice
    /// length, the feature write's token capacity, or one for other prefills,
    /// decode, and verify. Zero for KV publication, latent preparation, and
    /// denoising, which only read the request's KV.
    input: u32,
    /// Scheduled visible KV length before the call.
    visible: u32,
}

/// Computes ceiling division for unsigned values; a zero divisor is treated
/// as one.
fn ceil_div_u64(value: u64, divisor: u64) -> u64 {
    let divisor = divisor.max(1);
    value.div_ceil(divisor)
}

/// Returns the maximum number of output events one in-flight call can add to
/// its request's event journal.
///
/// `Scheduler::output_window_ready` charges this for every in-flight call; the
/// per-kind counts must stay in step with `Scheduler::next_output_bound`,
/// which charges the same events before the call is planned.
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
