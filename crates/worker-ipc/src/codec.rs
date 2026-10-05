//! FlatBuffers encoding and verified decoding for worker protocol messages.
//!
//! This module is the boundary where [`WorkerRequest`] and [`WorkerResponse`]
//! cross between owned Rust values and the FlatBuffers tables generated from
//! `schema/worker.fbs`. Senders encode frame payloads with [`encode_request`]
//! and [`encode_response`]: the `iceoryx` client and server, the `socket`
//! client, and the socket arm of `RankServer::respond` in `channel`. Received
//! payloads on either transport are decoded by `Frame::decode_request` and
//! `Frame::decode_response`, which call [`decode_request`] and
//! [`decode_response`]. The `worker-ipc-py` extension reaches the codec
//! through those transports.
//!
//! Table construction lives in the `encode` submodule, which reaches the
//! `*_to_fb` discriminant mappers defined here beside their `*_from_fb`
//! inverses through `use super::*`. Its builders are infallible, so the
//! encode functions fail only in validation.
//!
//! Decoding has three layers:
//!
//! 1. `flatbuffers::root` verifies the buffer structurally (offsets, bounds,
//!    UTF-8) before any field is read.
//! 2. The `*_from_table` and `*_from_fb` functions copy fields into owned
//!    values and report missing required fields, unknown discriminants, and
//!    illegal field combinations as [`CodecError::Invalid`].
//! 3. The domain `validate` methods (`Call`, `Batch`, `BatchOutput`,
//!    `WorkerInfo`, and others) check semantic invariants, reported as
//!    [`CodecError::Validation`]. Decoders run many of them as values are
//!    assembled, and aggregate validators such as `Batch::validate` run the
//!    nested ones again.
//!
//! [`encode_request`] validates a submitted batch and [`encode_response`]
//! validates info and result payloads before building the frame, so those
//! payloads are checked by both sender and receiver.

use std::collections::BTreeMap;

use flatbuffers::FlatBufferBuilder;

mod encode;
use uniserve_core::{KvCacheGroup, KvGroupKind, RequestId, SamplingParams, TokenLogprob, UnitId};

use crate::schema::uniserve::ipc as fbs;
use crate::{
    ArRequestParams, ArtifactHandle, Batch, BatchCommand, BatchOutput, BlockTable, Bounds,
    BufferAllocation, BufferId, CacheUnitAllocation, Call, CallCoordinates, CallId, CallKind,
    CallStatus, CanvasSampling, CanvasStep, DType, DecodeRange, DiffusionSamplingParams, DimBound,
    DrawLayout, ErrorCallIdentity, ErrorCode, FeatureKind, FinishFlags, ForwardBatch, ForwardMode,
    ForwardStats, KvCacheInfo, KvGroupTransfer, KvTransfer, LatentParams, Locator, MediaCall,
    MediaOutput, NewRequest, Readout, RequestKey, RequestKind, RequestOutput, ResponseKind, Rng,
    SamplingState, ShapeBound, TensorExport, TensorRef, TensorTransfer, TimingCounters,
    TransferHandle, TransferMode, TransferTransport, VideoAdmission, VisionInput, WorkerEndpoint,
    WorkerInfo, WorkerRequest, WorkerResponse, WorkerResponseError,
};
use uniserve_core::{
    AudioClip, ConditionMedia, ConditionRole, ConditionVision, ImageFit, MediaLocator, VideoClip,
    VideoCondition, VideoTask, VisionGrid,
};

/// Result type returned by FlatBuffers codec calls.
pub type CodecResult<T> = std::result::Result<T, CodecError>;

/// FlatBuffers construction, verification, and protocol validation failures.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Structural decoding failed: for example, the buffer fails FlatBuffers
    /// verification, or a field is missing, cannot be parsed, carries an
    /// unknown discriminant, or appears in an illegal combination.
    #[error("worker codec error: {0}")]
    Invalid(String),
    /// Decoded data violates a semantic worker-protocol invariant.
    #[error(transparent)]
    Validation(#[from] crate::ValidationError),
}

impl CodecError {
    /// Constructs a malformed-frame error with protocol context.
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

/// Extension methods for attaching protocol context to fallible extraction.
trait CodecContext<T> {
    /// Converts a missing value or source error into [`CodecError::Invalid`]
    /// carrying `message`; a source error's text is appended after it.
    fn context(self, message: &str) -> CodecResult<T>;

    /// Adds lazily constructed codec context to a missing value or source error.
    fn with_context<F, D>(self, message: F) -> CodecResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display;
}

impl<T> CodecContext<T> for Option<T> {
    /// Converts absence into a codec error with fixed context.
    fn context(self, message: &str) -> CodecResult<T> {
        self.ok_or_else(|| CodecError::invalid(message))
    }

    /// Converts absence into a codec error with lazily constructed context.
    fn with_context<F, D>(self, message: F) -> CodecResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.ok_or_else(|| CodecError::invalid(message().to_string()))
    }
}

impl<T, E> CodecContext<T> for std::result::Result<T, E>
where
    E: std::fmt::Display,
{
    /// Wraps a source error with fixed codec context.
    fn context(self, message: &str) -> CodecResult<T> {
        self.map_err(|error| CodecError::invalid(format!("{message}: {error}")))
    }

    /// Wraps a source error with lazily constructed codec context.
    fn with_context<F, D>(self, message: F) -> CodecResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.map_err(|error| CodecError::invalid(format!("{}: {error}", message())))
    }
}

// `codec_bail!` returns `Err(CodecError::Invalid)` from the innermost
// enclosing function or closure, so inside a `map` closure it produces that
// closure's `Err`, which the caller propagates. Both macros are textually
// scoped: the `encode` submodule, declared above them, cannot use them.
macro_rules! codec_bail {
    ($($arg:tt)*) => {
        return Err(CodecError::invalid(format!($($arg)*)))
    };
}

macro_rules! codec_ensure {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition {
            codec_bail!($($arg)*);
        }
    };
}

/// Validates a worker request and encodes it as a FlatBuffers payload.
///
/// Only a `Submit` request carries a payload to validate; `Batch::validate`
/// failures return [`CodecError::Validation`]. The returned bytes are the
/// frame payload without the transport `Header`.
pub fn encode_request(request: &WorkerRequest) -> CodecResult<Vec<u8>> {
    if let Some(batch) = request.batch() {
        batch.validate()?;
    }
    let mut builder = FlatBufferBuilder::new();
    let root = encode::request(&mut builder, request);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

/// Verifies and decodes a worker request payload.
///
/// Returns [`CodecError::Invalid`] for structural failures, for example when
/// the buffer fails FlatBuffers verification, a required field is missing or
/// unparsable, a discriminant is unknown, or fields appear in an illegal
/// combination (such as a payload on an `Info` or `Close` request), and
/// [`CodecError::Validation`] when the decoded batch violates a protocol
/// invariant.
pub fn decode_request(bytes: &[u8]) -> CodecResult<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_table(root)
}

/// Validates a worker response and encodes it as a FlatBuffers payload.
///
/// `Info` and `Result` payloads run `WorkerInfo::validate` and
/// `BatchOutput::validate`; `Ok` and `Error` responses are encoded without
/// validation.
pub fn encode_response(response: &WorkerResponse) -> CodecResult<Vec<u8>> {
    match response {
        WorkerResponse::Info { info, .. } => info.validate()?,
        WorkerResponse::Result { result, .. } => result.validate()?,
        WorkerResponse::Ok { .. } | WorkerResponse::Error { .. } => {}
    }
    let mut builder = FlatBufferBuilder::new();
    let root = encode::response(&mut builder, response);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

/// Verifies and decodes a worker response payload.
///
/// Structural failures return [`CodecError::Invalid`] as in
/// [`decode_request`], and an info or batch output that violates a protocol
/// invariant returns [`CodecError::Validation`]. Each response kind must carry
/// the fields it requires and none that it forbids (see
/// `response_from_table`).
pub fn decode_response(bytes: &[u8]) -> CodecResult<WorkerResponse> {
    let root = flatbuffers::root::<fbs::WorkerResponse>(bytes)
        .context("invalid WorkerResponse flatbuffer")?;
    response_from_table(root)
}

/// Decodes a verified FlatBuffers table into an owned request value.
fn request_from_table(request: fbs::WorkerRequest<'_>) -> CodecResult<WorkerRequest> {
    let kind = request_kind_from_fb(request.kind())?;
    let message_id = request.message_id();
    let batch = request.batch().map(batch_from_table).transpose()?;
    Ok(match kind {
        RequestKind::Info => {
            codec_ensure!(batch.is_none(), "info carries a payload");
            WorkerRequest::Info { message_id }
        }
        RequestKind::Submit => WorkerRequest::Submit {
            message_id,
            batch: Box::new(batch.context("submit request has no batch")?),
        },
        RequestKind::Close => {
            codec_ensure!(batch.is_none(), "close carries a payload");
            WorkerRequest::Close { message_id }
        }
    })
}

/// Decodes a response and enforces payload exclusivity for its discriminator.
fn response_from_table(response: fbs::WorkerResponse<'_>) -> CodecResult<WorkerResponse> {
    // Decode every optional branch before dispatch so each response kind can
    // enforce exclusivity between success payloads and structured error data.
    let kind = response_kind_from_fb(response.kind())?;
    let message_id = response.message_id();
    let info = response.info().map(info_from_table).transpose()?;
    let result = response.result().map(run_result_from_table).transpose()?;
    let payload_count = usize::from(info.is_some()) + usize::from(result.is_some());

    // Error metadata occupies independent optional fields on the wire.
    let message = response.message().map(str::to_owned);
    let code = response.code().map(str::to_owned);
    let fatal = response.fatal();
    let phase = response.phase().map(str::to_owned);
    let route = response.route().map(str::to_owned);
    let calls: Vec<ErrorCallIdentity> = response
        .calls()
        .map(|items| {
            items
                .iter()
                .map(error_call_from_table)
                .collect::<CodecResult<_>>()
        })
        .transpose()?
        .unwrap_or_default();
    let carries_error = message.is_some()
        || code.is_some()
        || fatal.is_some()
        || phase.is_some()
        || route.is_some()
        || !calls.is_empty();

    // The response discriminator defines the exact legal field combination.
    Ok(match kind {
        ResponseKind::Info => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid info response"
            );
            WorkerResponse::Info {
                message_id,
                info: info.context("info response has no info")?,
            }
        }
        ResponseKind::Result => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid result response"
            );
            WorkerResponse::Result {
                message_id,
                result: result.context("result response has no result")?,
            }
        }
        ResponseKind::Ok => {
            codec_ensure!(payload_count == 0 && !carries_error, "invalid ok response");
            WorkerResponse::Ok { message_id }
        }
        ResponseKind::Error => {
            codec_ensure!(
                payload_count == 0,
                "error response carries a success payload"
            );
            let message = message
                .filter(|value| !value.is_empty())
                .context("error response has no message")?;
            let code = code.filter(|value| !value.is_empty());
            WorkerResponse::Error {
                message_id,
                error: WorkerResponseError {
                    message,
                    code,
                    fatal: fatal.context("error response has no fatal flag")?,
                    phase,
                    route,
                    calls,
                },
            }
        }
    })
}

/// Decodes a submitted batch and validates it with `Batch::validate`.
///
/// Absent vectors decode as empty. Calls, admissions, commands, tensor
/// references, and exports are also validated individually as they are
/// decoded.
fn batch_from_table(run: fbs::Batch<'_>) -> CodecResult<Batch> {
    // Every collection keeps its wire order: `ForwardBatch::call_indices`
    // are positions in `calls`, and `commands` are an ordered sequence.
    let run = Batch {
        batch_id: run.batch_id(),
        collective_seq: run.collective_seq(),

        calls: run
            .calls()
            .map(|items| {
                items
                    .iter()
                    .map(call_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),

        // Scheduler-owned KV unit tables and the columnar forward rows. The
        // `ForwardBatch` columns are parallel arrays; `ForwardBatch::validate`
        // checks that their lengths agree.
        block_tables: run
            .block_tables()
            .map(|items| items.iter().map(block_table_from_table).collect())
            .unwrap_or_default(),
        new_cache_units: run
            .new_cache_units()
            .map(|items| items.iter().map(cache_unit_allocation_from_table).collect())
            .unwrap_or_default(),
        forward: ForwardBatch {
            call_indices: run
                .forward_call_indices()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            request_pool_indices: run
                .request_pool_indices()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            seq_lens: run
                .seq_lens()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            query_lens: run
                .query_lens()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            write_kv: run
                .write_kv()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
        },

        // Scheduler-owned latent, decode, and persistent-buffer allocations.
        latent_params: run
            .latent_params()
            .map(|items| {
                items
                    .iter()
                    .map(latent_params_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        decode_ranges: run
            .decode_ranges()
            .map(|items| {
                items
                    .iter()
                    .map(decode_range_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        buffer_allocations: run
            .buffer_allocations()
            .map(|items| {
                items
                    .iter()
                    .map(buffer_allocation_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),

        // Decode ordered controls and resolved host inputs.
        commands: run
            .commands()
            .map(|items| {
                items
                    .iter()
                    .map(command_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        kv_inputs: run
            .kv_inputs()
            .map(|items| {
                items
                    .iter()
                    .map(kv_transfer_from_table)
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        input_products: run
            .input_products()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_export_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };

    // Cross-collection references (call ids, buffers, products) can only be
    // checked once the whole batch is decoded.
    run.validate()?;
    Ok(run)
}

/// Decodes one request admission and validates it with `NewRequest::validate`,
/// which also runs the sampling and image parameter checks.
fn admission_from_table(admission: fbs::NewRequest<'_>) -> CodecResult<NewRequest> {
    let admission = NewRequest {
        request_key: request_key_from_table(admission.request_key(), "admission.request_key")?,
        request_pool_idx: admission.request_pool_idx(),
        prompt_token_ids: admission
            .prompt_token_ids()
            .map(|ids| ids.iter().collect())
            .unwrap_or_default(),
        input_images: admission.input_images(),
        ar: admission.ar().map(ar_params_from_table).transpose()?,
        image: admission.image().map(image_from_table).transpose()?,
        diffusion: admission
            .diffusion()
            .map(diffusion_params_from_table)
            .transpose()?,
        video: admission
            .video()
            .map(video_admission_from_table)
            .transpose()?,
    };
    admission.validate()?;
    Ok(admission)
}

/// Decodes a video request's task, tags and conditions;
/// `NewRequest::validate` checks them against the admission.
fn video_admission_from_table(table: fbs::VideoAdmission<'_>) -> CodecResult<VideoAdmission> {
    let task = match table.task() {
        fbs::VideoTask::T2va => VideoTask::T2va,
        fbs::VideoTask::Fl2va => VideoTask::Fl2va,
        fbs::VideoTask::Ref2va => VideoTask::Ref2va,
        other => codec_bail!("unknown video task {}", other.0),
    };
    Ok(VideoAdmission {
        task,
        text_tags: table
            .text_tags()
            .map(|tags| tags.iter().collect())
            .unwrap_or_default(),
        conditions: table
            .conditions()
            .map(|conditions| conditions.iter().map(video_condition_from_table).collect())
            .transpose()?
            .unwrap_or_default(),
    })
}

fn canvas_from_fb(canvas: &fbs::Canvas) -> uniserve_core::Canvas {
    uniserve_core::Canvas {
        width: canvas.width(),
        height: canvas.height(),
    }
}

fn audio_clip_from_fb(clip: &fbs::AudioClip) -> AudioClip {
    AudioClip {
        sample_rate: clip.sample_rate(),
        start_sample: clip.start_sample(),
        source_samples: clip.source_samples(),
        samples: clip.samples(),
    }
}

/// Decodes one condition, rejecting a media combination no condition has.
fn video_condition_from_table(table: fbs::VideoCondition<'_>) -> CodecResult<VideoCondition> {
    let role = match table.role() {
        fbs::ConditionRole::FirstFrame => ConditionRole::FirstFrame,
        fbs::ConditionRole::LastFrame => ConditionRole::LastFrame,
        fbs::ConditionRole::Reference => ConditionRole::Reference,
        other => codec_bail!("unknown condition role {}", other.0),
    };
    let audio = table.audio().map(audio_clip_from_fb);
    let media = match (table.image(), table.video(), audio) {
        (Some(fit), None, None) => ConditionMedia::Image(ImageFit {
            resized: canvas_from_fb(fit.resized()),
            left: fit.left(),
            top: fit.top(),
            size: canvas_from_fb(fit.size()),
        }),
        (None, Some(clip), soundtrack) => ConditionMedia::Video {
            clip: VideoClip {
                canvas: canvas_from_fb(clip.canvas()),
                start_frame: clip.start_frame(),
                frames: clip.frames(),
                vae_frames: clip.vae_frames(),
            },
            soundtrack,
        },
        (None, None, Some(clip)) => ConditionMedia::Audio(clip),
        _ => codec_bail!("a video condition carries an invalid media combination"),
    };
    let vision = table.vision().map(|vision| {
        let grid = vision.grid().copied().unwrap_or_default();
        ConditionVision {
            grid: VisionGrid {
                t: grid.t(),
                h: grid.h(),
                w: grid.w(),
            },
            tokens: vision.tokens(),
            frame_indices: vision
                .frame_indices()
                .map(|indices| indices.iter().collect())
                .unwrap_or_default(),
        }
    });
    Ok(VideoCondition {
        role,
        source: MediaLocator {
            name: table
                .source_name()
                .context("a video condition has no media locator")?
                .to_owned(),
            bytes: table.source_bytes(),
        },
        media,
        vision,
        latent_units: table
            .latent_units()
            .map(|units| units.iter().collect())
            .unwrap_or_default(),
        audio_rows: table.audio_rows(),
    })
}

/// Decodes autoregressive admission parameters, including sampling state.
fn ar_params_from_table(admission: fbs::ArRequestParams<'_>) -> CodecResult<ArRequestParams> {
    Ok(ArRequestParams {
        sampling: sampling_from_table(
            admission
                .sampling()
                .context("autoregressive request has no sampling parameters")?,
        )?,
        negative_token_ids: admission
            .negative_token_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        finish_token_ids: admission
            .finish_token_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        initial_position: admission.initial_position(),
        canvas: admission.canvas().map(|canvas| CanvasSampling {
            canvas_length: canvas.canvas_length(),
            max_steps: canvas.max_steps(),
            entropy_bound: canvas.entropy_bound(),
            t_min: canvas.t_min(),
            t_max: canvas.t_max(),
            confidence_threshold: canvas.confidence_threshold(),
            stability_threshold: canvas.stability_threshold(),
        }),
    })
}

/// Decodes a video denoiser's declaration; `WorkerInfo::validate` checks it.
fn video_denoiser_from_table(table: fbs::VideoDenoiserInfo<'_>) -> crate::VideoDenoiserInfo {
    crate::VideoDenoiserInfo {
        tasks: table
            .tasks()
            .map(|tasks| tasks.iter().map(str::to_owned).collect())
            .unwrap_or_default(),
        schedule_points: table.schedule_points(),
        video_shift: table.video_shift(),
        audio_shift: table.audio_shift(),
        canvases: table
            .canvases()
            .map(|canvases| {
                canvases
                    .iter()
                    .map(|canvas| uniserve_core::Canvas {
                        width: canvas.width(),
                        height: canvas.height(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        max_sequence_rows: table.max_sequence_rows(),
        condition_tiles: table.condition_tiles().map(|tiles| crate::ConditionTiles {
            rows: tiles.rows(),
            video: [tiles.frames(), tiles.height(), tiles.width()],
        }),
    }
}

/// Decodes diffusion admission parameters; `NewRequest::validate` checks the
/// frame, unit, and step counts.
fn diffusion_params_from_table(
    admission: fbs::DiffusionSamplingParams<'_>,
) -> CodecResult<DiffusionSamplingParams> {
    Ok(DiffusionSamplingParams {
        num_frames: admission.num_frames(),
        video_units: admission.video_units(),
        num_inference_steps: admission.num_inference_steps(),
        seed: admission.seed(),
        width: admission.width(),
        height: admission.height(),
    })
}

/// Decodes a KV unit table while preserving its page-major unit order.
fn block_table_from_table(table: fbs::BlockTable<'_>) -> BlockTable {
    BlockTable {
        request_pool_idx: table.request_pool_idx(),
        group_id: table.group_id(),
        start_page: table.start_page(),
        unit_ids: table
            .unit_ids()
            .map(|items| items.iter().map(UnitId).collect())
            .unwrap_or_default(),
        allocated_tokens: table.allocated_tokens(),
    }
}

/// Decodes newly assigned KV units for one request and cache group.
fn cache_unit_allocation_from_table(
    allocation: fbs::CacheUnitAllocation<'_>,
) -> CacheUnitAllocation {
    CacheUnitAllocation {
        request_pool_idx: allocation.request_pool_idx(),
        group_id: allocation.group_id(),
        unit_ids: allocation
            .unit_ids()
            .map(|items| items.iter().map(UnitId).collect())
            .unwrap_or_default(),
    }
}

/// Decodes the denoising-step range and optional latent pages for one call
/// that addresses a latent trajectory.
fn latent_params_from_table(params: fbs::LatentParams<'_>) -> CodecResult<LatentParams> {
    Ok(LatentParams {
        request_key: request_key_from_table(params.request_key(), "latent params.request_key")?,
        call_id: computation_id_from_fb(params.call_id())?,
        page_table: params
            .page_table()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        latent_units: params.latent_units(),
        height: params.height(),
        width: params.width(),
        start_step: params.start_step(),
        step_count: params.step_count(),
    })
}

/// Decodes the unit range for one video or audio decoding or encoding call.
fn decode_range_from_table(params: fbs::DecodeRange<'_>) -> CodecResult<DecodeRange> {
    Ok(DecodeRange {
        request_key: request_key_from_table(params.request_key(), "decode params.request_key")?,
        call_id: computation_id_from_fb(params.call_id())?,
        cursor: params.cursor(),
        max_units: params.max_units(),
    })
}

/// Decodes a persistent-buffer byte span and validates its buffer identity.
///
/// The span itself is checked later by `BufferAllocation::validate` through
/// `Batch::validate`.
fn buffer_allocation_from_table(
    params: fbs::BufferAllocation<'_>,
) -> CodecResult<BufferAllocation> {
    Ok(BufferAllocation {
        buffer: buffer_id_from_table(
            params
                .buffer()
                .context("buffer params has no buffer identity")?,
        )?,
        offset: params.offset(),
        bytes: params.bytes(),
    })
}

/// Reads the coordinates a call states.
///
/// Every call states them, so absence is a malformed frame rather than the
/// all-zero `CallCoordinates::default()`.
fn coordinates_from_table(table: Option<fbs::CallCoordinates<'_>>) -> CodecResult<CallCoordinates> {
    let value = table.context("call.coordinates")?;
    Ok(CallCoordinates {
        logical_position: value.logical_position(),
        kv_visible_len: value.kv_visible_len(),
        kv_computed_len: value.kv_computed_len(),
        flow_step: value.flow_step(),
    })
}

/// Decodes one call and validates the invariants it carries on its own with
/// `Call::validate`.
///
/// Relationships to other calls and to the batch's allocation tables are
/// checked afterwards by `Batch::validate`.
fn call_from_table(call: fbs::Call<'_>) -> CodecResult<Call> {
    let call = Call {
        request_key: request_key_from_table(call.request_key(), "call.request_key")?,
        call_id: computation_id_from_fb(call.call_id())?,
        coordinates: coordinates_from_table(call.coordinates())?,
        input_image: call.input_image().map(std::sync::Arc::from),
        kv_input: call.kv_input().map(buffer_id_from_table).transpose()?,
        kv_output: call.kv_output().map(buffer_id_from_table).transpose()?,
        input_token_ids: call
            .input_token_ids()
            .map(|ids| ids.iter().collect())
            .unwrap_or_default(),
        readout: call.readout().map(|readout| Readout {
            slot_tokens: readout
                .slot_tokens()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
            candidate_offsets: readout
                .candidate_offsets()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
            candidate_ids: readout
                .candidate_ids()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
        }),
        canvas: call.canvas().map(|canvas| CanvasStep {
            block: canvas.block(),
            step: canvas.step(),
        }),
        consumer_slots: call
            .consumer_slots()
            .map(|slots| slots.iter().collect())
            .unwrap_or_default(),
        sampling_state: call.sampling_state().map(|state| SamplingState {
            // Absent means no whitelist; a present empty list represents an
            // invalid all-masked distribution. The distinction must survive
            // decoding.
            allowed_token_ids: state.allowed_token_ids().map(|ids| ids.iter().collect()),
            suppressed_token_ids: state
                .suppressed_token_ids()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
            finish_token_ids: state
                .finish_token_ids()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
            transition_token_ids: state
                .transition_token_ids()
                .map(|ids| ids.iter().collect())
                .unwrap_or_default(),
            force_finish: state.force_finish(),
        }),
        component: call.component().to_owned(),
        code: computation_from_fb(call.code())?,
        bounds: Bounds {
            max_tokens: call.max_tokens(),
            max_kv_units: call.max_kv_units(),
            max_latent_bytes: call.max_latent_bytes(),
            max_completion_bytes: call.max_completion_bytes(),
            max_transfer_bytes: call.max_transfer_bytes(),
        },
        inputs: call
            .inputs()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_ref_from_table)
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        outputs: call
            .outputs()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_ref_from_table)
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        token_input: call.token_input().map(tensor_ref_from_table).transpose()?,
        token_output: call.token_output().map(tensor_ref_from_table).transpose()?,
        vision_inputs: call
            .vision_inputs()
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        Ok(VisionInput {
                            offset: item.offset(),
                            feature: tensor_ref_from_table(item.feature())?,
                        })
                    })
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        latent_feature_input: call
            .latent_feature_input()
            .map(tensor_ref_from_table)
            .transpose()?,
        encoder_output: call
            .encoder_output()
            .map(tensor_ref_from_table)
            .transpose()?,
        latent_input: call.latent_input().map(tensor_ref_from_table).transpose()?,
        latent_output: call
            .latent_output()
            .map(tensor_ref_from_table)
            .transpose()?,
        image_input: call.image_input().map(tensor_ref_from_table).transpose()?,
        image_output: call.image_output().map(tensor_ref_from_table).transpose()?,
        completion_output: call
            .completion_output()
            .map(tensor_ref_from_table)
            .transpose()?,
        transition_output: call
            .transition_output()
            .map(tensor_ref_from_table)
            .transpose()?,
        predicate: call.predicate().map(tensor_ref_from_table).transpose()?,
        rng: call.rng().map(rng_from_table).transpose()?,
    };
    call.validate()?;
    Ok(call)
}

/// Decodes one control-command union and validates it with
/// `BatchCommand::validate`.
fn command_from_table(envelope: fbs::BatchCommandEnvelope<'_>) -> CodecResult<BatchCommand> {
    // The FlatBuffers discriminator selects the only payload table permitted
    // to contribute command fields.
    let command = match envelope.command_type() {
        fbs::BatchCommand::StartCommand => {
            let start = envelope
                .command_as_start_command()
                .context("start command table is missing")?;
            BatchCommand::Start {
                request: Box::new(admission_from_table(
                    start.request().context("start control has no request")?,
                )?),
            }
        }

        fbs::BatchCommand::FinishCommand => {
            let finish = envelope
                .command_as_finish_command()
                .context("finish command table is missing")?;
            BatchCommand::Finish {
                request_key: request_key_from_table(
                    finish.request_key(),
                    "control.finish.request_key",
                )?,
                retained_buffers: finish
                    .retained_buffers()
                    .into_iter()
                    .flatten()
                    .map(buffer_id_from_table)
                    .collect::<CodecResult<Vec<_>>>()?,
            }
        }

        fbs::BatchCommand::FreeCommand => {
            let free = envelope
                .command_as_free_command()
                .context("free command table is missing")?;
            BatchCommand::Free {
                buffer: buffer_id_from_table(
                    free.buffer().context("free control has no buffer id")?,
                )?,
            }
        }
        _ => codec_bail!("batch command union is empty"),
    };

    command.validate()?;
    Ok(command)
}

/// Decodes a required request identity with a caller-specific error label.
fn request_key_from_table(
    request_key: Option<fbs::RequestKey<'_>>,
    label: &str,
) -> CodecResult<RequestKey> {
    let request_key = request_key.with_context(|| format!("{label} is missing"))?;
    Ok(RequestKey {
        engine_id: request_key.engine_id(),
        request_id: RequestId(request_key.request_id()),
        request_epoch: request_key.request_epoch(),
    })
}

/// Reconstructs shape bounds from extents and the single dynamic-axis marker.
///
/// `dynamic_axis` is `-1` (the schema default) when every axis is static;
/// otherwise it indexes the one `DimBound::Device` axis, whose extent is that
/// axis's maximum. An out-of-range marker is a malformed frame; zero extents
/// are left to `ShapeBound::validate`.
fn shape_bound_from_parts(
    extents: Option<flatbuffers::Vector<'_, u32>>,
    dynamic_axis: i32,
) -> CodecResult<ShapeBound> {
    let extents = extents
        .map(|items| items.iter().collect::<Vec<_>>())
        .unwrap_or_default();
    codec_ensure!(
        dynamic_axis >= -1 && dynamic_axis < extents.len() as i32,
        "typed value has an invalid dynamic axis"
    );
    Ok(ShapeBound {
        dims: extents
            .into_iter()
            .enumerate()
            .map(|(index, extent)| {
                if index as i32 == dynamic_axis {
                    DimBound::Device { max: extent }
                } else {
                    DimBound::Static(extent)
                }
            })
            .collect(),
    })
}

/// Decodes a result's raster axes: both indices set, or both -1 for a result
/// without a raster. Their range is left to `OutputInfo::validate`.
fn raster_axes_from_fb(height: i32, width: i32) -> CodecResult<Option<crate::RasterAxes>> {
    match (u32::try_from(height), u32::try_from(width)) {
        (Ok(height), Ok(width)) => Ok(Some(crate::RasterAxes { height, width })),
        _ => {
            codec_ensure!(
                height == -1 && width == -1,
                "a tensor result names one raster axis without the other"
            );
            Ok(None)
        }
    }
}

/// Decodes a role-independent tensor identity and its bounded capacity.
fn tensor_ref_from_table(reference: fbs::TensorRef<'_>) -> CodecResult<TensorRef> {
    let id = buffer_id_from_table(reference.id())?;
    let tensor = TensorRef {
        request_key: id.owner,
        producer_call_id: id.producer_call_id,
        output_index: id.output_index,
        generation: id.generation,
        dtype: dtype_from_fb(reference.dtype())?,
        shape_bound: shape_bound_from_parts(reference.extents(), reference.dynamic_axis())?,
    };
    tensor.validate()?;
    Ok(tensor)
}

/// Decodes deterministic random coordinates, rejecting an unknown draw layout.
fn rng_from_table(rng: fbs::Rng<'_>) -> CodecResult<Rng> {
    Ok(Rng {
        seed: rng.seed(),
        semantic_index_base: rng.semantic_index_base(),
        draw_layout: draw_layout_from_fb(rng.draw_layout())?,
    })
}

/// Decodes a batch output, preserving report order, then validates it with
/// `BatchOutput::validate`.
fn run_result_from_table(report: fbs::BatchOutput<'_>) -> CodecResult<BatchOutput> {
    // Completions and products are independent ordered streams;
    // `BatchOutput::validate` checks both once they are decoded.
    let report = BatchOutput {
        batch_id: report.batch_id(),
        completions: report
            .completions()
            .map(|items| {
                items
                    .iter()
                    .map(completion_record_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        products: report
            .products()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_export_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        worker_exec_us: report.worker_exec_us(),
        forward_stats: report.forward_stats().map(forward_stats_from_table),
    };
    report.validate()?;
    Ok(report)
}

/// Decodes one call completion and validates it with `RequestOutput::validate`.
fn completion_record_from_table(record: fbs::RequestOutput<'_>) -> CodecResult<RequestOutput> {
    let finish_flags = record
        .finish_flags()
        .context("completion result has no finish flags")?;
    let media_output = record
        .media_output()
        .map(|output| {
            let handle = match output.handle_type() {
                fbs::ArtifactHandle::PosixShmArtifact => {
                    let handle = output.handle_as_posix_shm_artifact().context(
                        "completion media output POSIX shared-storage handle is missing",
                    )?;
                    ArtifactHandle::PosixShm {
                        name: required_str(
                            handle.name(),
                            "completion.media_output.handle.value.name",
                        )?,
                    }
                }
                _ => codec_bail!("completion media output handle union is empty"),
            };
            Ok::<MediaOutput, CodecError>(MediaOutput {
                handle,
                bytes: output.bytes(),
            })
        })
        .transpose()?;

    let timing_counters = record
        .timing_counters()
        .context("completion record has no timing counters")?;

    let record = RequestOutput {
        sampled_logprob: record.sampled_logprob(),
        top_logprobs: record
            .top_logprobs()
            .map(|entries| entries.iter().map(token_logprob_from_fb).collect())
            .unwrap_or_default(),
        prompt_logprobs: record
            .prompt_logprobs()
            .map(|positions| {
                positions
                    .iter()
                    .map(|position| {
                        position
                            .entries()
                            .map(|entries| entries.iter().map(token_logprob_from_fb).collect())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default(),
        candidate_logprobs: record
            .candidate_logprobs()
            .map(|values| values.iter().collect())
            .unwrap_or_default(),
        request_key: request_key_from_table(record.request_key(), "completion.request_key")?,
        call_id: computation_id_from_fb(record.call_id())?,
        status: call_status_from_fb(record.status())?,
        product_generations: record
            .product_generations()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        error_code: record.error_code().map(error_code_from_fb).transpose()?,
        timing_counters: TimingCounters {
            queued_us: timing_counters.queued_us(),
            device_us: timing_counters.device_us(),
            copy_us: timing_counters.copy_us(),
            host_us: timing_counters.host_us(),
        },
        code: computation_from_fb(record.code())?,
        position: record.position(),
        kv_visible_len: record.kv_visible_len(),
        kv_computed_len: record.kv_computed_len(),
        num_completed_steps: record.num_completed_steps(),
        committed_tokens: record
            .committed_tokens()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        finish_flags: FinishFlags {
            eos: finish_flags.eos(),
            length: finish_flags.length(),
            stop: finish_flags.stop(),
        },
        media_output,
        kv_output: record.kv_output().map(kv_transfer_from_table).transpose()?,
    };

    // Domain validation applies cross-field status and generation invariants
    // after all completion fields have been decoded.
    record.validate()?;
    Ok(record)
}

/// Decodes a tensor export and its declared reference.
fn tensor_export_from_table(payload: fbs::TensorExport<'_>) -> CodecResult<TensorExport> {
    let value = transfer_handle_from_table(
        payload
            .value()
            .context("tensor export has no transfer descriptor")?,
    )?;
    let payload = TensorExport {
        product: tensor_ref_from_table(
            payload
                .product()
                .context("product payload has no product reference")?,
        )?,
        value,
    };
    payload.validate()?;
    Ok(payload)
}

/// Decodes the request and call identity attached to a worker error.
fn error_call_from_table(call: fbs::ErrorCallIdentity<'_>) -> CodecResult<ErrorCallIdentity> {
    Ok(ErrorCallIdentity {
        request_key: request_key_from_table(call.request_key(), "error call.request_key")?,
        call_id: computation_id_from_fb(call.call_id())?,
    })
}

/// Decodes worker capabilities and validates them with `WorkerInfo::validate`.
///
/// Absent `model_dtype` and `attention_backend` decode
/// as empty strings, and absent optional vectors as empty. A media call bound
/// more than once is a malformed frame.
fn info_from_table(info: fbs::WorkerInfo<'_>) -> CodecResult<WorkerInfo> {
    let info = WorkerInfo {
        model_name: required_str(info.model_name(), "info.model_name")?,
        endpoint: endpoint_from_table(info.endpoint().context("info has no endpoint")?)?,
        device: required_str(info.device(), "info.device")?,
        fabric_handles: info.fabric_handles(),
        transfer_backends: info
            .transfer_backends()
            .context("info has no transfer backends")?
            .iter()
            .map(str::to_owned)
            .collect(),
        world_size: info.world_size(),
        model_dtype: info.model_dtype().unwrap_or_default().to_owned(),
        attention_backend: info.attention_backend().unwrap_or_default().to_owned(),
        weight_formats: info
            .weight_formats()
            .map(|values| values.iter().map(str::to_owned).collect())
            .unwrap_or_default(),
        activation_formats: info
            .activation_formats()
            .map(|values| values.iter().map(str::to_owned).collect())
            .unwrap_or_default(),
        components: info
            .components()
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        Ok(crate::ComponentInfo {
                            name: required_str(item.name(), "component.name")?,
                            config: uniserve_core::ComponentConfig {
                                ranks: item
                                    .ranks()
                                    .map(|ranks| ranks.iter().map(|rank| rank as usize).collect())
                                    .unwrap_or_default(),
                                parallel_config: parallel_from_fb(
                                    item.parallel_config()
                                        .context("component requires parallel_config")?,
                                )?,
                                distribution: distribution_from_fb(item.distribution())?,
                                units_per_rank: item.units_per_rank() as usize,
                            },
                            outputs: item
                                .outputs()
                                .context("component requires tensor result declarations")?
                                .iter()
                                .map(|output| {
                                    Ok(crate::OutputInfo {
                                        name: required_str(output.name(), "tensor result.name")?,
                                        dtype: dtype_from_fb(output.dtype())?,
                                        shape_bound: shape_bound_from_parts(
                                            output.extents(),
                                            output.dynamic_axis(),
                                        )?,
                                        raster_axes: raster_axes_from_fb(
                                            output.height_axis(),
                                            output.width_axis(),
                                        )?,
                                    })
                                })
                                .collect::<CodecResult<Vec<_>>>()?,
                        })
                    })
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        supported_calls: info
            .supported_calls()
            .map(|items| {
                items
                    .iter()
                    .map(computation_from_fb)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        queue_depth: info.queue_depth(),
        max_batch_calls: info.max_batch_calls(),
        max_batch_tokens: info.max_batch_tokens(),
        max_prefill_calls: info.max_prefill_calls(),
        max_decode_calls: info.max_decode_calls(),
        request_slots: info.request_slots(),
        kv_cache: info.kv_cache().map(kv_cache_from_table).transpose()?,
        latent_page_units: info.latent_page_units(),
        latent_pages: info.latent_pages(),
        buffer_pool_bytes: info.buffer_pool_bytes(),
        encoder_cache_entries: info.encoder_cache_entries(),
        encoder_entry_bytes: info.encoder_entry_bytes(),
        max_unresolved_calls: info.max_unresolved_calls(),
        host_lane_capacity: info.host_lane_capacity(),
        media_components: {
            let mut components = std::collections::BTreeMap::new();
            if let Some(bindings) = info.media_components() {
                for binding in bindings {
                    let call = media_call_from_fb(binding.call())?;
                    let component = required_str(binding.component(), "media component")?;
                    if components.insert(call, component).is_some() {
                        codec_bail!("duplicate media component binding");
                    }
                }
            }
            components
        },
        num_inference_steps: info.num_inference_steps(),
        video_denoiser: info.video_denoiser().map(video_denoiser_from_table),
    };
    info.validate()?;
    Ok(info)
}

/// Decodes sampling parameters without range checks.
///
/// Only a `min_tokens` value that does not fit `usize` is rejected here;
/// `SamplingParams::validate`, run by `NewRequest::validate` from
/// `admission_from_table`, rejects non-finite values and values outside each
/// sampling transform's domain.
fn sampling_from_table(sampling: fbs::SamplingParams<'_>) -> CodecResult<SamplingParams> {
    let sampling = SamplingParams {
        // Scalar sampling controls map directly from the verified table.
        temperature: sampling.temperature(),
        top_k: sampling.top_k(),
        top_p: sampling.top_p(),
        ignore_eos: sampling.ignore_eos(),
        seed: sampling.seed(),
        min_p: sampling.min_p(),
        repetition_penalty: sampling.repetition_penalty(),
        frequency_penalty: sampling.frequency_penalty(),
        presence_penalty: sampling.presence_penalty(),

        // Token-specific controls become owned collections.
        logit_bias: sampling
            .logit_bias()
            .map(|items| {
                items
                    .iter()
                    .map(|item| (item.token_id(), item.bias()))
                    .collect()
            })
            .unwrap_or_default(),
        min_tokens: usize::try_from(sampling.min_tokens())
            .context("sampling.min_tokens does not fit usize")?,
        n_logprobs: sampling.n_logprobs(),
        bad_words_ids: sampling
            .bad_words_ids()
            .map(|lists| {
                lists
                    .iter()
                    .map(|list| {
                        list.items()
                            .map(|items| items.iter().collect())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default(),
        allowed_token_ids: sampling
            .allowed_token_ids()
            .map(|items| items.iter().collect()),
        return_logprobs: sampling.return_logprobs(),
        return_prompt_logprobs: sampling.return_prompt_logprobs(),
        n_prompt_logprobs: sampling.n_prompt_logprobs(),
        logprob_token_ids: sampling
            .logprob_token_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        typical_p: sampling.typical_p(),
        forced_token_ids: sampling
            .forced_token_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
    };
    Ok(sampling)
}

/// Decodes image-generation parameters, parsing the CFG renormalization type.
///
/// Non-finite and out-of-range controls are rejected by
/// `ImageParams::validate`, run by `NewRequest::validate` from
/// `admission_from_table`.
fn image_from_table(image: fbs::ImageParams<'_>) -> CodecResult<uniserve_core::ImageParams> {
    Ok(uniserve_core::ImageParams {
        steps: image.steps(),
        cfg_text_scale: image.cfg_text_scale(),
        cfg_img_scale: image.cfg_img_scale(),
        cfg_renorm_type: required_parse(image.cfg_renorm_type(), "image.cfg_renorm_type")?,
        cfg_renorm_min: image.cfg_renorm_min(),
        cfg_interval: (image.cfg_interval_lo(), image.cfg_interval_hi()),
        timestep_shift: image.timestep_shift(),
        height: image.height(),
        width: image.width(),
        seed: image.seed(),
        negative_prompt: image
            .negative_prompt()
            .map(str::to_string)
            .unwrap_or_default(),
        max_images: image.max_images(),
        image_prompts: image
            .image_prompts()
            .map(|items| items.iter().map(str::to_string).collect())
            .unwrap_or_default(),
        retain_images: image.retain_images(),
    })
}

/// Decodes per-mode, attention, graph, relay, and verification counters.
fn forward_stats_from_table(stats: fbs::ForwardStats<'_>) -> ForwardStats {
    ForwardStats {
        // Aggregate execution-mode counters.
        mode_counts: map_from_table(stats.mode_counts()),
        mode_tokens: map_from_table(stats.mode_tokens()),
        mode_us: map_from_table(stats.mode_us()),
        component_us: map_from_table(stats.component_us()),

        // Attention backend activity.
        attention_launches: stats.attention_launches(),
        attention_us: stats.attention_us(),
        attention_backend_counts: map_from_table(stats.attention_backend_counts()),

        // CUDA graph lifecycle and padding behavior.
        cuda_graph_captures: stats.cuda_graph_captures(),
        cuda_graph_replays: stats.cuda_graph_replays(),
        cuda_graph_misses: stats.cuda_graph_misses(),
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks(),
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens(),
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens(),
        cuda_graph_runtime_mode_counts: map_from_table(stats.cuda_graph_runtime_mode_counts()),

        // Decode relay cache effectiveness.
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits(),
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses(),
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits(),
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses(),

        // FlashInfer planning activity.
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls(),
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses(),
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows(),
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices(),
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls(),
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses(),

        // Speculative-verification outcomes.
        spec_verify_rows: stats.spec_verify_rows(),
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens(),
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens(),
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens(),
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens(),
        spec_verify_path_counts: map_from_table(stats.spec_verify_path_counts()),
    }
}

/// Decodes a string-to-counter map, ignoring entries without a key.
fn map_from_table(
    items: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<fbs::StringU64Pair<'_>>>>,
) -> BTreeMap<String, u64> {
    items
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.key().map(|key| (key.to_string(), item.value())))
                .collect()
        })
        .unwrap_or_default()
}

/// Decodes one KV cache group: its retention kind, page shape, and this
/// rank's layers and heads. `window` and `sink` are read only for sliding
/// windows.
fn kv_group_from_table(group: fbs::KvGroup<'_>) -> CodecResult<KvCacheGroup> {
    let kind = if group.kind() == fbs::KvGroupKind::Full {
        KvGroupKind::Full
    } else if group.kind() == fbs::KvGroupKind::SlidingWindow {
        KvGroupKind::SlidingWindow {
            window: group.window(),
            sink: group.sink(),
        }
    } else {
        codec_bail!("unknown KV group kind {}", group.kind().0)
    };
    Ok(KvCacheGroup {
        kind,
        page_tokens: group.page_tokens(),
        units_per_page: group.units_per_page(),
        layer_ids: group
            .layer_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        num_kv_heads: group.num_kv_heads(),
        total_kv_heads: group.total_kv_heads(),
        kv_head_offset: group.kv_head_offset(),
        head_dim: group.head_dim(),
    })
}

/// Decodes the loaded rank identity used by startup and physical products.
fn endpoint_from_table(endpoint: fbs::WorkerEndpoint<'_>) -> CodecResult<WorkerEndpoint> {
    let value = WorkerEndpoint {
        worker_id: required_str(endpoint.worker_id(), "endpoint.worker_id")?,
        rank: endpoint.rank(),
        node: required_str(endpoint.node(), "endpoint.node")?,
        address_space: required_str(endpoint.address_space(), "endpoint.address_space")?,
        incarnation: required_str(endpoint.incarnation(), "endpoint.incarnation")?,
    };
    value.validate()?;
    Ok(value)
}

/// Copies a required non-empty string or reports its protocol field name.
fn required_str(value: Option<&str>, label: &str) -> CodecResult<String> {
    value
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .with_context(|| format!("{label} is missing"))
}

/// Parses a required non-empty string into its domain type.
fn required_parse<T>(value: Option<&str>, label: &str) -> CodecResult<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    required_str(value, label)?
        .parse()
        .with_context(|| format!("{label} is invalid"))
}

/// Decodes a required call identity from its inline wire struct.
fn computation_id_from_fb(id: Option<&fbs::CallId>) -> CodecResult<CallId> {
    let id = id.context("computation identity is missing")?;
    Ok(CallId::new(id.batch_id(), id.request_index()))
}

/// Decodes and validates a persistent-buffer identity.
fn buffer_id_from_table(buffer: fbs::BufferId<'_>) -> CodecResult<BufferId> {
    let id = BufferId {
        owner: request_key_from_table(buffer.owner(), "buffer_id.owner")?,
        producer_call_id: computation_id_from_fb(buffer.producer_call_id())?,
        output_index: buffer.output_index(),
        generation: buffer.generation(),
    };
    id.validate()?;
    Ok(id)
}

/// Flattens shape bounds into extents and a single dynamic-axis marker.
///
/// The marker is `-1` when every axis is static. Only one device axis is
/// representable: with several, the last one wins. `ShapeBound::validate`
/// rejects such shapes.
fn shape_bound_to_parts(shape: &ShapeBound) -> (Vec<u32>, i32) {
    let mut dynamic_axis = -1;
    let extents = shape
        .dims
        .iter()
        .enumerate()
        .map(|(index, dim)| match dim {
            DimBound::Static(extent) => *extent,
            DimBound::Device { max } => {
                dynamic_axis = index as i32;
                *max
            }
        })
        .collect();
    (extents, dynamic_axis)
}

/// Copies a wire token-logprob entry (top or prompt logprob) into its owned
/// value.
fn token_logprob_from_fb(entry: &fbs::TokenLogprob) -> TokenLogprob {
    TokenLogprob {
        token_id: entry.token_id(),
        logprob: entry.logprob(),
        rank: entry.rank(),
    }
}

/// Decodes worker KV-cache capabilities from a verified table.
fn kv_cache_from_table(config: fbs::KVCacheInfo<'_>) -> CodecResult<KvCacheInfo> {
    Ok(KvCacheInfo {
        num_units: config.num_units(),
        unit_bytes: config.unit_bytes(),
        groups: config
            .groups()
            .map(|items| {
                items
                    .iter()
                    .map(kv_group_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        dtype: required_parse(config.dtype(), "info.kv_cache.dtype")?,
    })
}

/// Decodes how a component spreads independent units over its ranks; wire
/// `Local` maps to `None`, which runs the component as one model-parallel
/// group.
fn distribution_from_fb(
    value: fbs::ComponentDistribution,
) -> CodecResult<Option<uniserve_core::ComponentDistribution>> {
    match value {
        fbs::ComponentDistribution::Local => Ok(None),
        fbs::ComponentDistribution::TemporalUnits => {
            Ok(Some(uniserve_core::ComponentDistribution::TemporalUnits))
        }
        _ => codec_bail!("unknown component distribution"),
    }
}

/// Decodes a component's parallel degrees and sequence-parallel strategy.
///
/// The strategy is a FlatBuffers union: every variant except
/// `LocalSequence` requires its payload table. Wire `GatherSequence` maps to
/// `SequenceParallel::Allgather`. `WorkerInfo::validate` checks the degrees
/// against component membership.
fn parallel_from_fb(config: fbs::ParallelConfig<'_>) -> CodecResult<uniserve_core::ParallelConfig> {
    use uniserve_core::SequenceParallel;
    let sequence_parallel = match config.sequence_parallel_type() {
        fbs::SequenceParallel::LocalSequence => SequenceParallel::Local,
        fbs::SequenceParallel::UlyssesSequence => {
            let value = config
                .sequence_parallel_as_ulysses_sequence()
                .context("missing UlyssesSequence configuration")?;
            SequenceParallel::Ulysses {
                ulysses_degree: value.ulysses_degree() as usize,
            }
        }
        fbs::SequenceParallel::GatherSequence => {
            let value = config
                .sequence_parallel_as_gather_sequence()
                .context("missing GatherSequence configuration")?;
            SequenceParallel::Allgather {
                allgather_degree: value.allgather_degree() as usize,
            }
        }
        fbs::SequenceParallel::HybridSequence => {
            let value = config
                .sequence_parallel_as_hybrid_sequence()
                .context("missing HybridSequence configuration")?;
            SequenceParallel::Hybrid {
                ulysses_degree: value.ulysses_degree() as usize,
                allgather_degree: value.allgather_degree() as usize,
            }
        }
        _ => codec_bail!("unknown sequence parallel strategy"),
    };
    Ok(uniserve_core::ParallelConfig {
        tensor_parallel_size: config.tensor_parallel_size() as usize,
        pipeline_parallel_size: config.pipeline_parallel_size() as usize,
        sequence_parallel,
    })
}

fn forward_mode_to_fb(value: ForwardMode) -> fbs::ForwardMode {
    match value {
        ForwardMode::Prefill => fbs::ForwardMode::Prefill,
        ForwardMode::Decode => fbs::ForwardMode::Decode,
        ForwardMode::Verify => fbs::ForwardMode::Verify,
        ForwardMode::TokenDenoising => fbs::ForwardMode::TokenDenoising,
    }
}

fn forward_mode_from_fb(value: fbs::ForwardMode) -> CodecResult<ForwardMode> {
    Ok(match value {
        fbs::ForwardMode::Prefill => ForwardMode::Prefill,
        fbs::ForwardMode::Decode => ForwardMode::Decode,
        fbs::ForwardMode::Verify => ForwardMode::Verify,
        fbs::ForwardMode::TokenDenoising => ForwardMode::TokenDenoising,
        _ => codec_bail!("unknown forward_mode {}", value.0),
    })
}

fn media_call_to_fb(value: MediaCall) -> fbs::MediaCall {
    match value {
        MediaCall::MediaReading => fbs::MediaCall::MediaReading,
        MediaCall::VisionEncoding => fbs::MediaCall::VisionEncoding,
        MediaCall::LatentEncoding => fbs::MediaCall::LatentEncoding,
        MediaCall::TextEncoding => fbs::MediaCall::TextEncoding,
        MediaCall::LatentPreparation => fbs::MediaCall::LatentPreparation,
        MediaCall::Denoising => fbs::MediaCall::Denoising,
        MediaCall::ImageDecoding => fbs::MediaCall::ImageDecoding,
        MediaCall::VideoDecoding => fbs::MediaCall::VideoDecoding,
        MediaCall::AudioDecoding => fbs::MediaCall::AudioDecoding,
        MediaCall::VideoEncoding => fbs::MediaCall::VideoEncoding,
        MediaCall::AudioEncoding => fbs::MediaCall::AudioEncoding,
        MediaCall::Muxing => fbs::MediaCall::Muxing,
    }
}

fn media_call_from_fb(value: fbs::MediaCall) -> CodecResult<MediaCall> {
    Ok(match value {
        fbs::MediaCall::MediaReading => MediaCall::MediaReading,
        fbs::MediaCall::VisionEncoding => MediaCall::VisionEncoding,
        fbs::MediaCall::LatentEncoding => MediaCall::LatentEncoding,
        fbs::MediaCall::TextEncoding => MediaCall::TextEncoding,
        fbs::MediaCall::LatentPreparation => MediaCall::LatentPreparation,
        fbs::MediaCall::Denoising => MediaCall::Denoising,
        fbs::MediaCall::ImageDecoding => MediaCall::ImageDecoding,
        fbs::MediaCall::VideoDecoding => MediaCall::VideoDecoding,
        fbs::MediaCall::AudioDecoding => MediaCall::AudioDecoding,
        fbs::MediaCall::VideoEncoding => MediaCall::VideoEncoding,
        fbs::MediaCall::AudioEncoding => MediaCall::AudioEncoding,
        fbs::MediaCall::Muxing => MediaCall::Muxing,
        _ => codec_bail!("unknown media_call {}", value.0),
    })
}

fn transfer_mode_to_fb(value: TransferMode) -> fbs::TransferMode {
    match value {
        TransferMode::Tensor => fbs::TransferMode::Tensor,
        TransferMode::KvExport => fbs::TransferMode::KvExport,
        TransferMode::KvInstall => fbs::TransferMode::KvInstall,
    }
}

fn transfer_mode_from_fb(value: fbs::TransferMode) -> CodecResult<TransferMode> {
    Ok(match value {
        fbs::TransferMode::Tensor => TransferMode::Tensor,
        fbs::TransferMode::KvExport => TransferMode::KvExport,
        fbs::TransferMode::KvInstall => TransferMode::KvInstall,
        _ => codec_bail!("unknown transfer_mode {}", value.0),
    })
}

/// Encodes a call kind into the inline `CallKind` struct, setting exactly one
/// of its three tags and leaving the others at `None`.
fn computation_to_fb(value: CallKind) -> fbs::CallKind {
    let mut encoded = fbs::CallKind::default();
    match value {
        CallKind::Forward(mode) => encoded.set_forward_mode(forward_mode_to_fb(mode)),
        CallKind::Media(call) => encoded.set_media(media_call_to_fb(call)),
        CallKind::Transfer(mode) => encoded.set_transfer(transfer_mode_to_fb(mode)),
    }
    encoded
}

/// Decodes the inline `CallKind` struct.
///
/// The struct carries three independent tags whose `None` value means unset.
/// A frame with zero or several set tags, or with an unknown value in the set
/// tag, is malformed.
fn computation_from_fb(value: &fbs::CallKind) -> CodecResult<CallKind> {
    let present = u8::from(value.forward_mode() != fbs::ForwardMode::None)
        + u8::from(value.media() != fbs::MediaCall::None)
        + u8::from(value.transfer() != fbs::TransferMode::None);
    if present != 1 {
        codec_bail!("computation must select exactly one classification");
    }
    if value.forward_mode() != fbs::ForwardMode::None {
        Ok(CallKind::Forward(forward_mode_from_fb(
            value.forward_mode(),
        )?))
    } else if value.media() != fbs::MediaCall::None {
        Ok(CallKind::Media(media_call_from_fb(value.media())?))
    } else {
        Ok(CallKind::Transfer(transfer_mode_from_fb(value.transfer())?))
    }
}

/// Decodes transport-specific storage coordinates and common tensor metadata.
fn transfer_locator_from_table(value: fbs::Locator<'_>) -> CodecResult<Locator> {
    // The transport discriminant determines which coordinate fields are
    // required; fields belonging to other transports are not read.
    let transport = if value.transport() == fbs::TransferTransportKind::Local {
        TransferTransport::Local {
            endpoint: value
                .endpoint()
                .context("local transfer endpoint is missing")?
                .to_owned(),
            key: value.key(),
        }
    } else if value.transport() == fbs::TransferTransportKind::PosixShm {
        TransferTransport::PosixShm {
            endpoint: value
                .endpoint()
                .context("shared-storage transfer endpoint is missing")?
                .to_owned(),
            name: value
                .name()
                .context("shared-storage transfer name is missing")?
                .to_owned(),
        }
    } else if value.transport() == fbs::TransferTransportKind::CudaVmm {
        TransferTransport::CudaVmm {
            endpoint: value
                .endpoint()
                .context("CUDA VMM endpoint is missing")?
                .to_owned(),
            export_id: value
                .export_id()
                .context("CUDA VMM export identity is missing")?
                .to_owned(),
            storage_size_bytes: value.storage_size_bytes(),
            storage_offsets_bytes: value
                .storage_offsets_bytes()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            span_lengths: value
                .span_lengths()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            span_counts: value
                .span_counts()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            tensor_stride: value
                .tensor_stride()
                .map(|items| items.iter().collect())
                .unwrap_or_default(),
            ready_event_handle: value
                .ready_event_handle()
                .map(|bytes| bytes.bytes().to_vec())
                .unwrap_or_default(),
            allocation_handle: value
                .allocation_handle()
                .map(|bytes| bytes.bytes().to_vec())
                .unwrap_or_default(),
            acknowledgment_offset: value.acknowledgment_offset(),
        }
    } else if value.transport() == fbs::TransferTransportKind::Channel {
        TransferTransport::Channel {
            endpoint: value
                .endpoint()
                .context("channel locator has no endpoint")?
                .to_owned(),
            payload: value
                .payload()
                .map(|bytes| bytes.bytes().to_vec())
                .unwrap_or_default(),
        }
    } else {
        codec_bail!("unknown transfer transport {}", value.transport().0)
    };

    // Tensor metadata describes the logical view independently of transport.
    let locator = Locator {
        source: endpoint_from_table(value.source().context("locator has no source endpoint")?)?,
        transport,
        nbytes: value.nbytes(),
        dtype: value
            .dtype()
            .context("transfer dtype is missing")?
            .to_owned(),
        shape: value
            .shape()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        offset: value
            .offset()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        device: value
            .device()
            .context("transfer device is missing")?
            .to_owned(),
    };
    Ok(locator)
}

/// Decodes one physical tensor's logical shape and locators, then validates it
/// with `TensorTransfer::validate`, which also validates each `Locator`.
fn tensor_transfer_from_table(value: fbs::TensorTransfer<'_>) -> CodecResult<TensorTransfer> {
    let tensor = TensorTransfer {
        shape: value
            .shape()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        locations: value
            .locations()
            .map(|items| items.iter().map(transfer_locator_from_table).collect())
            .unwrap_or_else(|| Ok(Vec::new()))?,
    };
    tensor.validate()?;
    Ok(tensor)
}

/// Decodes a product-family transfer union and all referenced locators.
fn transfer_handle_from_table(value: fbs::TransferHandle<'_>) -> CodecResult<TransferHandle> {
    // The outer union selects product-family metadata; every physical tensor
    // is decoded through the same transport validator.
    Ok(match value.value_type() {
        fbs::TransferData::EncoderTransfer => {
            let transfer = value
                .value_as_encoder_transfer()
                .context("encoder transfer payload is missing")?;
            TransferHandle::Encoder {
                height: transfer.height(),
                width: transfer.width(),
                payload_kind: match transfer.payload_kind() {
                    fbs::FeatureKind::Vision => FeatureKind::Vision,
                    fbs::FeatureKind::Latent => FeatureKind::Latent,
                    other => codec_bail!("unknown encoder feature kind {}", other.0),
                },
                tensor: tensor_transfer_from_table(
                    transfer
                        .tensor()
                        .context("encoder transfer tensor is missing")?,
                )?,
            }
        }

        fbs::TransferData::DeviceProductTransfer => {
            let transfer = value
                .value_as_device_product_transfer()
                .context("device-product transfer payload is missing")?;
            TransferHandle::DeviceProduct {
                height: transfer.height(),
                width: transfer.width(),
                value_range: transfer.value_range().unwrap_or_default().to_owned(),
                tensor: tensor_transfer_from_table(
                    transfer
                        .tensor()
                        .context("device-product transfer tensor is missing")?,
                )?,
            }
        }

        fbs::TransferData::LatentTransfer => {
            let transfer = value
                .value_as_latent_transfer()
                .context("latent transfer payload is missing")?;
            TransferHandle::Latent {
                height: transfer.height(),
                width: transfer.width(),
                latent_units: transfer.latent_units(),
                step: transfer.step(),
                tensor: tensor_transfer_from_table(
                    transfer
                        .tensor()
                        .context("latent transfer tensor is missing")?,
                )?,
            }
        }
        other => codec_bail!("unknown transfer payload variant {}", other.0),
    })
}

/// Maps an element type to its stable FlatBuffers discriminant.
fn dtype_to_fb(dtype: DType) -> fbs::DType {
    match dtype {
        DType::U8 => fbs::DType::U8,
        DType::I32 => fbs::DType::I32,
        DType::I16 => fbs::DType::I16,
        DType::I64 => fbs::DType::I64,
        DType::F16 => fbs::DType::F16,
        DType::BF16 => fbs::DType::BF16,
        DType::F32 => fbs::DType::F32,
    }
}

/// Decodes a supported FlatBuffers element-type discriminant.
fn dtype_from_fb(dtype: fbs::DType) -> CodecResult<DType> {
    Ok(match dtype {
        fbs::DType::U8 => DType::U8,
        fbs::DType::I32 => DType::I32,
        fbs::DType::I16 => DType::I16,
        fbs::DType::I64 => DType::I64,
        fbs::DType::F16 => DType::F16,
        fbs::DType::BF16 => DType::BF16,
        fbs::DType::F32 => DType::F32,
        other => codec_bail!("unknown dtype {}", other.0),
    })
}

/// Maps a random-draw layout to its stable FlatBuffers discriminant.
fn draw_layout_to_fb(layout: DrawLayout) -> fbs::DrawLayout {
    match layout {
        DrawLayout::TargetSampling => fbs::DrawLayout::TargetSampling,
        DrawLayout::SpeculativeProposal => fbs::DrawLayout::SpeculativeProposal,
        DrawLayout::FlowNoise => fbs::DrawLayout::FlowNoise,
    }
}

/// Decodes a supported FlatBuffers random-draw layout.
fn draw_layout_from_fb(layout: fbs::DrawLayout) -> CodecResult<DrawLayout> {
    Ok(match layout {
        fbs::DrawLayout::TargetSampling => DrawLayout::TargetSampling,
        fbs::DrawLayout::SpeculativeProposal => DrawLayout::SpeculativeProposal,
        fbs::DrawLayout::FlowNoise => DrawLayout::FlowNoise,
        other => codec_bail!("unknown draw layout {}", other.0),
    })
}

/// Maps a call status to its stable FlatBuffers discriminant.
fn call_status_to_fb(status: CallStatus) -> fbs::CallStatus {
    match status {
        CallStatus::Ok => fbs::CallStatus::Ok,
        CallStatus::Predicated => fbs::CallStatus::Predicated,
        CallStatus::Error => fbs::CallStatus::Error,
    }
}

/// Decodes a supported FlatBuffers call status.
fn call_status_from_fb(status: fbs::CallStatus) -> CodecResult<CallStatus> {
    Ok(match status {
        fbs::CallStatus::Ok => CallStatus::Ok,
        fbs::CallStatus::Predicated => CallStatus::Predicated,
        fbs::CallStatus::Error => CallStatus::Error,
        other => codec_bail!("unknown completion status {}", other.0),
    })
}

/// Maps a worker error code to its stable FlatBuffers discriminant.
fn error_code_to_fb(code: ErrorCode) -> fbs::ErrorCode {
    match code {
        ErrorCode::InvalidCall => fbs::ErrorCode::InvalidCall,
        ErrorCode::ResourceExhausted => fbs::ErrorCode::ResourceExhausted,
        ErrorCode::ComputeError => fbs::ErrorCode::ComputeError,
        ErrorCode::Cancelled => fbs::ErrorCode::Cancelled,
        ErrorCode::Internal => fbs::ErrorCode::Internal,
    }
}

/// Decodes a supported FlatBuffers worker error code.
fn error_code_from_fb(code: fbs::ErrorCode) -> CodecResult<ErrorCode> {
    Ok(match code {
        fbs::ErrorCode::InvalidCall => ErrorCode::InvalidCall,
        fbs::ErrorCode::ResourceExhausted => ErrorCode::ResourceExhausted,
        fbs::ErrorCode::ComputeError => ErrorCode::ComputeError,
        fbs::ErrorCode::Cancelled => ErrorCode::Cancelled,
        fbs::ErrorCode::Internal => ErrorCode::Internal,
        other => codec_bail!("unknown error code {}", other.0),
    })
}

/// Maps a request kind to its stable FlatBuffers discriminant.
fn request_kind_to_fb(kind: RequestKind) -> fbs::ReqKind {
    match kind {
        RequestKind::Info => fbs::ReqKind::Info,
        RequestKind::Submit => fbs::ReqKind::Submit,
        RequestKind::Close => fbs::ReqKind::Close,
    }
}

/// Resolves a FlatBuffers request discriminant against the complete supported set.
///
/// Inverting `request_kind_to_fb` over `RequestKind::ALL` keeps the two
/// directions in agreement without a second hand-written table.
fn request_kind_from_fb(kind: fbs::ReqKind) -> CodecResult<RequestKind> {
    for candidate in RequestKind::ALL {
        if request_kind_to_fb(candidate) == kind {
            return Ok(candidate);
        }
    }
    codec_bail!("unknown request kind {}", kind.0)
}

/// Iterates over the stable wire names of all request kinds.
pub fn request_kind_names() -> impl Iterator<Item = &'static str> {
    RequestKind::ALL.into_iter().map(RequestKind::as_str)
}

/// Maps a response kind to its stable FlatBuffers discriminant.
fn response_kind_to_fb(kind: ResponseKind) -> fbs::RespKind {
    match kind {
        ResponseKind::Info => fbs::RespKind::Info,
        ResponseKind::Result => fbs::RespKind::Result,
        ResponseKind::Ok => fbs::RespKind::Ok,
        ResponseKind::Error => fbs::RespKind::Error,
    }
}

/// Decodes a supported FlatBuffers response discriminant.
fn response_kind_from_fb(kind: fbs::RespKind) -> CodecResult<ResponseKind> {
    if kind == fbs::RespKind::Info {
        Ok(ResponseKind::Info)
    } else if kind == fbs::RespKind::Result {
        Ok(ResponseKind::Result)
    } else if kind == fbs::RespKind::Ok {
        Ok(ResponseKind::Ok)
    } else if kind == fbs::RespKind::Error {
        Ok(ResponseKind::Error)
    } else {
        codec_bail!("unknown response kind {}", kind.0)
    }
}

/// Decodes a KV export descriptor.
///
/// Its buffer identities and tensor transfers are validated as they are
/// decoded; the descriptor as a whole is validated by its container:
/// `Batch::validate` for `kv_inputs` and `RequestOutput::validate` for a
/// completion's `kv_output`.
fn kv_transfer_from_table(transfer: fbs::KvTransfer<'_>) -> CodecResult<KvTransfer> {
    Ok(KvTransfer {
        groups: transfer
            .groups()
            .map(|items| {
                items
                    .iter()
                    .map(kv_group_transfer_from_table)
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
        source: buffer_id_from_table(transfer.source().context("KV transfer source is missing")?)?,
        destination: transfer
            .destination()
            .context("KV transfer destination is missing")?
            .to_owned(),
        base: transfer.base().map(buffer_id_from_table).transpose()?,
        base_extent: transfer.base_extent(),
        exported_extent: transfer.exported_extent(),
        compute_dtype: transfer
            .compute_dtype()
            .context("KV transfer compute dtype is missing")?
            .to_owned(),
    })
}

/// Decodes one cache group's share of a KV export, preserving the key,
/// value, scale tensor order.
fn kv_group_transfer_from_table(group: fbs::KvGroupTransfer<'_>) -> CodecResult<KvGroupTransfer> {
    Ok(KvGroupTransfer {
        start: group.start(),
        page_tokens: group.page_tokens(),
        tensors: group
            .tensors()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_transfer_from_table)
                    .collect::<CodecResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
    })
}
