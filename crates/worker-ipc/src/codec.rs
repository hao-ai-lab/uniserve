//! FlatBuffers encoding and verified decoding for worker protocol messages.

use std::collections::BTreeMap;

use flatbuffers::FlatBufferBuilder;
use uniserve_core::{BlockId, KvCacheGroup, KvGroupKind, RequestId, SamplingParams, TokenLogprob};

use crate::schema::uniserve::ipc as fbs;
use crate::{
    ArRequestParams, ArtifactHandle, Batch, BatchCommand, BatchOutput, BlockTable, Bounds,
    BufferAllocation, BufferId, CachePageAllocation, Call, CallCoordinates, CallId, CallKind,
    CallStatus, DType, DecodeRange, DiffusionSamplingParams, DimBound, DrawLayout,
    ErrorCallIdentity, ErrorCode, FeatureKind, FinishFlags, ForwardBatch, ForwardMode,
    ForwardStats, KvCacheInfo, KvTransfer, LatentParams, Locator, MediaOutput, NewRequest,
    PipelineStage, RegistrationAck, RequestKey, RequestKind, RequestOutput, ResponseKind, Rng,
    SamplingState, ShapeBound, TensorPublication, TensorRef, TensorTransfer, TimingCounters,
    TransferHandle, TransferMode, TransferTransport, UmmRequestParams, WorkerEndpoint, WorkerInfo,
    WorkerRequest, WorkerResponse, WorkerResponseError,
};

/// Result type returned by FlatBuffers codec calls.
pub type CodecResult<T> = std::result::Result<T, CodecError>;

/// FlatBuffers construction, verification, and protocol validation failures.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoded data is absent, malformed, or fails FlatBuffers verification.
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
    /// Replaces a missing value or source error with fixed codec context.
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

/// Encodes a validated worker request as a FlatBuffers frame.
pub fn encode_request(request: &WorkerRequest) -> CodecResult<Vec<u8>> {
    let object = request_to_fb(request)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

/// Verifies and decodes a worker request frame.
pub fn decode_request(bytes: &[u8]) -> CodecResult<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_table(root)
}

/// Encodes a validated worker response as a FlatBuffers frame.
pub fn encode_response(response: &WorkerResponse) -> CodecResult<Vec<u8>> {
    let object = response_to_fb(response)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

/// Verifies and decodes a worker response frame.
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
            batch: batch.context("submit request has no batch")?,
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

/// Decodes an owned run and validates all nested call and params contracts.
fn batch_from_table(run: fbs::Batch<'_>) -> CodecResult<Batch> {
    // Preserve wire order for calls, controls, and products because later
    // validation and execution interpret those collections positionally.
    let run = Batch {
        batch_id: run.batch_id(),
        collective_seq: run.collective_seq(),

        // Decode executable graph records in their submitted order.
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

        // Decode scheduler-owned KV params metadata.
        block_tables: run
            .block_tables()
            .map(|items| items.iter().map(block_table_from_table).collect())
            .unwrap_or_default(),
        new_cache_pages: run
            .new_cache_pages()
            .map(|items| items.iter().map(cache_page_allocation_from_table).collect())
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

        // Decode diffusion and persistent-buffer params metadata.
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
                    .map(tensor_publication_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };

    // Validate the assembled graph only after every cross-reference is owned.
    run.validate()?;
    Ok(run)
}

/// Decodes one request admission and validates its selected model family.
fn admission_from_table(admission: fbs::NewRequest<'_>) -> CodecResult<NewRequest> {
    let admission = NewRequest {
        request_key: request_key_from_table(admission.request_key(), "admission.request_key")?,
        request_pool_idx: admission.request_pool_idx(),
        prompt_token_ids: admission
            .prompt_token_ids()
            .map(|ids| ids.iter().collect())
            .unwrap_or_default(),
        ar: admission.ar().map(ar_params_from_table).transpose()?,
        umm: admission.umm().map(umm_params_from_table).transpose()?,
        diffusion: admission
            .diffusion()
            .map(diffusion_params_from_table)
            .transpose()?,
    };
    admission.validate()?;
    Ok(admission)
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
    })
}

/// Decodes unified-multimodal admission parameters and requires image settings.
fn umm_params_from_table(admission: fbs::UmmRequestParams<'_>) -> CodecResult<UmmRequestParams> {
    Ok(UmmRequestParams {
        image: image_from_table(
            admission
                .image()
                .context("unified-multimodal request has no image parameters")?,
        )?,
    })
}

/// Decodes diffusion admission parameters and their resolved media geometry.
fn diffusion_params_from_table(
    admission: fbs::DiffusionSamplingParams<'_>,
) -> CodecResult<DiffusionSamplingParams> {
    Ok(DiffusionSamplingParams {
        num_frames: admission.num_frames(),
        num_decode_chunks: admission.num_decode_chunks(),
        num_inference_steps: admission.num_inference_steps(),
        seed: admission.seed(),
    })
}

/// Decodes a logical KV block table while preserving page order.
fn block_table_from_table(table: fbs::BlockTable<'_>) -> BlockTable {
    BlockTable {
        request_pool_idx: table.request_pool_idx(),
        group_id: table.group_id(),
        page_ids: table
            .page_ids()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
        allocated_tokens: table.allocated_tokens(),
    }
}

/// Decodes newly assigned KV pages for one request and cache group.
fn cache_page_allocation_from_table(
    allocation: fbs::CachePageAllocation<'_>,
) -> CachePageAllocation {
    CachePageAllocation {
        request_pool_idx: allocation.request_pool_idx(),
        group_id: allocation.group_id(),
        page_ids: allocation
            .page_ids()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
    }
}

/// Decodes a latent-page params bound to a request call.
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

/// Decodes a diffusion decoder params bound to a request call.
fn decode_range_from_table(params: fbs::DecodeRange<'_>) -> CodecResult<DecodeRange> {
    Ok(DecodeRange {
        request_key: request_key_from_table(params.request_key(), "decode params.request_key")?,
        call_id: computation_id_from_fb(params.call_id())?,
        cursor: params.cursor(),
        max_units: params.max_units(),
    })
}

/// Decodes a persistent-buffer byte span and validates its buffer identity.
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

/// Reads the coordinates a call states. Every call states them, so absence is a
/// malformed frame rather than an origin default.
fn coordinates_from_table(table: Option<fbs::CallCoordinates<'_>>) -> CodecResult<CallCoordinates> {
    let value = table.context("call.coordinates")?;
    Ok(CallCoordinates {
        logical_position: value.logical_position(),
        kv_visible_len: value.kv_visible_len(),
        kv_computed_len: value.kv_computed_len(),
        flow_step: value.flow_step(),
    })
}

/// Writes the coordinates a call states.
fn coordinates_to_fb(value: CallCoordinates) -> fbs::CallCoordinatesT {
    fbs::CallCoordinatesT {
        logical_position: value.logical_position,
        kv_visible_len: value.kv_visible_len,
        kv_computed_len: value.kv_computed_len,
        flow_step: value.flow_step,
    }
}

/// Decodes one computation and its entry binding.
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
        consumer_slots: call
            .consumer_slots()
            .map(|slots| slots.iter().collect())
            .unwrap_or_default(),
        sampling_state: call.sampling_state().map(|state| SamplingState {
            // A missing whitelist and an empty whitelist have different semantics.
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
        entry: call.entry().to_owned(),
        code: computation_from_fb(call.code())?,
        bounds: Bounds {
            max_tokens: call.max_tokens(),
            max_kv_pages: call.max_kv_pages(),
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
        vision_input: call.vision_input().map(tensor_ref_from_table).transpose()?,
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

/// Decodes one control-command union and validates its lineage constraints.
fn command_from_table(envelope: fbs::BatchCommandEnvelope<'_>) -> CodecResult<BatchCommand> {
    // The FlatBuffers discriminator selects the only payload table permitted
    // to contribute command fields.
    let command = match envelope.command_type() {
        fbs::BatchCommand::StartCommand => {
            let start = envelope
                .command_as_start_command()
                .context("start command table is missing")?;
            BatchCommand::Start {
                request: admission_from_table(
                    start.request().context("start control has no request")?,
                )?,
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

/// Decodes deterministic random coordinates and validates their draw layout.
fn rng_from_table(rng: fbs::Rng<'_>) -> CodecResult<Rng> {
    Ok(Rng {
        seed: rng.seed(),
        semantic_index_base: rng.semantic_index_base(),
        draw_layout: draw_layout_from_fb(rng.draw_layout())?,
    })
}

/// Decodes a run result, preserving report order, then validates the aggregate.
fn run_result_from_table(report: fbs::BatchOutput<'_>) -> CodecResult<BatchOutput> {
    // Completions and products are independent ordered streams whose identities
    // are reconciled by `BatchOutput::validate` after both are materialized.
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
                    .map(tensor_publication_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        registration: RegistrationAck {
            visible: report
                .registration()
                .map(|ack| ack.visible())
                .context("completion report has no registration acknowledgement")?,
        },
        worker_exec_us: report.worker_exec_us(),
        forward_stats: report.forward_stats().map(forward_stats_from_table),
    };
    report.validate()?;
    Ok(report)
}

/// Decodes a model output and validates its status-dependent result data.
fn completion_record_from_table(record: fbs::RequestOutput<'_>) -> CodecResult<RequestOutput> {
    let finish_flags = record
        .finish_flags()
        .context("completion result has no finish flags")?;
    let media_output = record
        .media_output()
        .map(|output| {
            let handle = match output.handle_type() {
                fbs::ArtifactHandle::PosixShmArtifact => {
                    let handle = output
                        .handle_as_posix_shm_artifact()
                        .context("completion media output POSIX shared-memory handle is missing")?;
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
        request_key: request_key_from_table(record.request_key(), "completion.request_key")?,
        call_id: computation_id_from_fb(record.call_id())?,
        status: op_status_from_fb(record.status())?,
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

/// Decodes a tensor publication and its declared reference.
fn tensor_publication_from_table(
    payload: fbs::TensorPublication<'_>,
) -> CodecResult<TensorPublication> {
    let value = transfer_handle_from_table(
        payload
            .value()
            .context("tensor publication has no transfer descriptor")?,
    )?;
    let payload = TensorPublication {
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

/// Decodes worker capabilities and validates the advertised resource geometry.
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
        configuration_id: info.configuration_id().unwrap_or_default().to_owned(),
        components: info
            .components()
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        Ok(crate::EntryInfo {
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
                                .context("entry requires tensor result declarations")?
                                .iter()
                                .map(|output| {
                                    Ok(crate::OutputInfo {
                                        name: required_str(output.name(), "tensor result.name")?,
                                        dtype: dtype_from_fb(output.dtype())?,
                                        shape_bound: shape_bound_from_parts(
                                            output.extents(),
                                            output.dynamic_axis(),
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
        supported_ops: info
            .supported_ops()
            .map(|items| {
                items
                    .iter()
                    .map(computation_from_fb)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        queue_depth: info.queue_depth(),
        max_batch_ops: info.max_batch_ops(),
        max_batch_tokens: info.max_batch_tokens(),
        request_slots: info.request_slots(),
        kv_cache: info.kv_cache().map(kv_cache_from_table).transpose()?,
        latent_page_units: info.latent_page_units(),
        latent_pages: info.latent_pages(),
        buffer_pool_bytes: info.buffer_pool_bytes(),
        encoder_cache_entries: info.encoder_cache_entries(),
        encoder_entry_bytes: info.encoder_entry_bytes(),
        max_unresolved_ops: info.max_unresolved_ops(),
        host_lane_capacity: info.host_lane_capacity(),
        pipeline_components: {
            let mut components = std::collections::BTreeMap::new();
            if let Some(bindings) = info.pipeline_components() {
                for binding in bindings {
                    let stage = pipeline_stage_from_fb(binding.stage())?;
                    let component = required_str(binding.component(), "pipeline component")?;
                    if components.insert(stage, component).is_some() {
                        codec_bail!("duplicate pipeline component binding");
                    }
                }
            }
            components
        },
        num_inference_steps: info.num_inference_steps(),
    };
    info.validate()?;
    Ok(info)
}

/// Decodes sampling parameters and rejects non-finite or out-of-range values.
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
    validate_sampling(&sampling)?;
    Ok(sampling)
}

/// Decodes image-generation parameters and rejects non-finite controls.
fn image_from_table(image: fbs::ImageParams<'_>) -> CodecResult<uniserve_core::ImageParams> {
    for value in [
        image.cfg_text_scale(),
        image.cfg_img_scale(),
        image.cfg_renorm_min(),
        image.cfg_interval_lo(),
        image.cfg_interval_hi(),
        image.timestep_shift(),
    ] {
        codec_ensure!(value.is_finite(), "image parameters must be finite");
    }
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

/// Decodes full-attention or sliding-window KV group geometry.
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
        num_blocks: group.num_blocks(),
        kind,
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

/// Converts a validated request into its FlatBuffers object representation.
fn request_to_fb(request: &WorkerRequest) -> CodecResult<fbs::WorkerRequestT> {
    let batch = match request {
        WorkerRequest::Info { .. } | WorkerRequest::Close { .. } => None,
        WorkerRequest::Submit { batch, .. } => {
            batch.validate()?;
            Some(batch)
        }
    };
    Ok(fbs::WorkerRequestT {
        kind: request_kind_to_fb(request.kind()),
        message_id: request.message_id(),
        batch: batch.map(batch_to_fb).transpose()?.map(Box::new),
    })
}

/// Converts a response into the payload branch selected by its response kind.
fn response_to_fb(response: &WorkerResponse) -> CodecResult<fbs::WorkerResponseT> {
    // Split the closed Rust enum into mutually exclusive FlatBuffers fields.
    let (info, result, error) = match response {
        WorkerResponse::Info { info, .. } => (Some(info), None, None),
        WorkerResponse::Result { result, .. } => (None, Some(result), None),
        WorkerResponse::Ok { .. } => (None, None, None),
        WorkerResponse::Error { error, .. } => (None, None, Some(error)),
    };
    Ok(fbs::WorkerResponseT {
        kind: response_kind_to_fb(response.kind()),
        message_id: response.message_id(),
        info: info.map(info_to_fb).transpose()?.map(Box::new),
        result: result.map(run_result_to_fb).transpose()?.map(Box::new),
        message: error.map(|error| error.message.clone()),
        code: error.and_then(|error| error.code.clone()),
        fatal: error.map(|error| error.fatal),
        phase: error.and_then(|error| error.phase.clone()),
        route: error.and_then(|error| error.route.clone()),
        calls: Some(
            error
                .into_iter()
                .flat_map(|error| &error.calls)
                .map(error_call_to_fb)
                .collect(),
        ),
    })
}

fn batch_to_fb(run: &Batch) -> CodecResult<fbs::BatchT> {
    run.validate()?;

    Ok(fbs::BatchT {
        batch_id: run.batch_id,
        collective_seq: run.collective_seq,

        // Preserve executable graph order in the serialized vectors.
        calls: Some(
            run.calls
                .iter()
                .map(call_to_fb)
                .collect::<CodecResult<_>>()?,
        ),

        // Scheduler-owned params metadata is already validated as a unit.
        block_tables: Some(run.block_tables.iter().map(block_table_to_fb).collect()),
        new_cache_pages: Some(
            run.new_cache_pages
                .iter()
                .map(cache_page_allocation_to_fb)
                .collect(),
        ),
        forward_call_indices: Some(run.forward.call_indices.clone()),
        request_pool_indices: Some(run.forward.request_pool_indices.clone()),
        seq_lens: Some(run.forward.seq_lens.clone()),
        query_lens: Some(run.forward.query_lens.clone()),
        write_kv: Some(run.forward.write_kv.clone()),
        latent_params: Some(run.latent_params.iter().map(latent_params_to_fb).collect()),
        decode_ranges: Some(run.decode_ranges.iter().map(decode_range_to_fb).collect()),
        buffer_allocations: Some(
            run.buffer_allocations
                .iter()
                .map(buffer_allocation_to_fb)
                .collect(),
        ),

        // Controls and host inputs retain submission order.
        commands: Some(
            run.commands
                .iter()
                .map(|command| fbs::BatchCommandEnvelopeT {
                    command: command_to_fb(command),
                })
                .collect(),
        ),
        kv_inputs: Some(run.kv_inputs.iter().map(kv_transfer_to_fb).collect()),
        input_products: Some(
            run.input_products
                .iter()
                .map(tensor_publication_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
    })
}

/// Converts a validated admission and its selected parameter family.
fn admission_to_fb(admission: &NewRequest) -> CodecResult<fbs::NewRequestT> {
    admission.validate()?;
    Ok(fbs::NewRequestT {
        request_key: Some(Box::new(request_key_to_fb(admission.request_key))),
        request_pool_idx: admission.request_pool_idx,
        prompt_token_ids: Some(admission.prompt_token_ids.clone()),
        ar: admission
            .ar
            .as_ref()
            .map(ar_params_to_fb)
            .transpose()?
            .map(Box::new),
        umm: admission.umm.as_ref().map(umm_params_to_fb).map(Box::new),
        diffusion: admission
            .diffusion
            .as_ref()
            .map(diffusion_params_to_fb)
            .map(Box::new),
    })
}

/// Converts autoregressive admission parameters into their wire table.
fn ar_params_to_fb(admission: &ArRequestParams) -> CodecResult<fbs::ArRequestParamsT> {
    Ok(fbs::ArRequestParamsT {
        sampling: Some(Box::new(sampling_to_fb(&admission.sampling)?)),
        negative_token_ids: Some(admission.negative_token_ids.clone()),
        finish_token_ids: Some(admission.finish_token_ids.clone()),
        initial_position: admission.initial_position,
    })
}

/// Converts unified-multimodal admission parameters into their wire table.
fn umm_params_to_fb(admission: &UmmRequestParams) -> fbs::UmmRequestParamsT {
    fbs::UmmRequestParamsT {
        image: Some(Box::new(image_to_fb(&admission.image))),
    }
}

/// Converts diffusion admission parameters and resolved geometry.
fn diffusion_params_to_fb(admission: &DiffusionSamplingParams) -> fbs::DiffusionSamplingParamsT {
    fbs::DiffusionSamplingParamsT {
        num_frames: admission.num_frames,
        num_decode_chunks: admission.num_decode_chunks,
        num_inference_steps: admission.num_inference_steps,
        seed: admission.seed,
    }
}

/// Converts a logical KV block table while preserving page order.
fn block_table_to_fb(table: &BlockTable) -> fbs::BlockTableT {
    fbs::BlockTableT {
        request_pool_idx: table.request_pool_idx,
        group_id: table.group_id,
        page_ids: Some(table.page_ids.iter().map(|page| page.0).collect()),
        allocated_tokens: table.allocated_tokens,
    }
}

/// Converts newly assigned KV pages into their wire table.
fn cache_page_allocation_to_fb(allocation: &CachePageAllocation) -> fbs::CachePageAllocationT {
    fbs::CachePageAllocationT {
        request_pool_idx: allocation.request_pool_idx,
        group_id: allocation.group_id,
        page_ids: Some(allocation.page_ids.iter().map(|page| page.0).collect()),
    }
}

/// Converts a latent-page params into its wire table.
fn latent_params_to_fb(params: &LatentParams) -> fbs::LatentParamsT {
    fbs::LatentParamsT {
        request_key: Some(Box::new(request_key_to_fb(params.request_key))),
        call_id: Some(computation_id_to_fb(params.call_id)),
        page_table: Some(params.page_table.clone()),
        latent_units: params.latent_units,
        height: params.height,
        width: params.width,
        start_step: params.start_step,
        step_count: params.step_count,
    }
}

/// Converts a diffusion decoder params into its wire table.
fn decode_range_to_fb(params: &DecodeRange) -> fbs::DecodeRangeT {
    fbs::DecodeRangeT {
        request_key: Some(Box::new(request_key_to_fb(params.request_key))),
        call_id: Some(computation_id_to_fb(params.call_id)),
        cursor: params.cursor,
        max_units: params.max_units,
    }
}

/// Converts a persistent-buffer byte span into its wire table.
fn buffer_allocation_to_fb(params: &BufferAllocation) -> fbs::BufferAllocationT {
    fbs::BufferAllocationT {
        buffer: Some(Box::new(buffer_id_to_fb(params.buffer))),
        offset: params.offset,
        bytes: params.bytes,
    }
}

/// Encodes one validated computation directly into its wire table.
fn call_to_fb(call: &Call) -> CodecResult<fbs::CallT> {
    call.validate()?;
    Ok(fbs::CallT {
        request_key: Some(Box::new(request_key_to_fb(call.request_key))),
        call_id: Some(computation_id_to_fb(call.call_id)),
        coordinates: Some(Box::new(coordinates_to_fb(call.coordinates))),
        input_token_ids: Some(call.input_token_ids.clone()),
        consumer_slots: Some(call.consumer_slots.clone()),
        input_image: call.input_image.as_deref().map(str::to_owned),
        kv_input: call.kv_input.map(buffer_id_to_fb).map(Box::new),
        kv_output: call.kv_output.map(buffer_id_to_fb).map(Box::new),
        sampling_state: call.sampling_state.as_ref().map(|state| {
            Box::new(fbs::SamplingStateT {
                allowed_token_ids: state.allowed_token_ids.clone(),
                suppressed_token_ids: Some(state.suppressed_token_ids.clone()),
                finish_token_ids: Some(state.finish_token_ids.clone()),
                transition_token_ids: Some(state.transition_token_ids.clone()),
                force_finish: state.force_finish,
            })
        }),
        entry: call.entry.clone(),
        code: computation_to_fb(call.code),
        max_tokens: call.bounds.max_tokens,
        max_kv_pages: call.bounds.max_kv_pages,
        max_latent_bytes: call.bounds.max_latent_bytes,
        max_completion_bytes: call.bounds.max_completion_bytes,
        max_transfer_bytes: call.bounds.max_transfer_bytes,
        inputs: Some(
            call.inputs
                .iter()
                .map(tensor_ref_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        outputs: Some(
            call.outputs
                .iter()
                .map(tensor_ref_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        token_input: call
            .token_input
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        token_output: call
            .token_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        vision_input: call
            .vision_input
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        latent_feature_input: call
            .latent_feature_input
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        encoder_output: call
            .encoder_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        latent_input: call
            .latent_input
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        latent_output: call
            .latent_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        image_input: call
            .image_input
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        image_output: call
            .image_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        completion_output: call
            .completion_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        transition_output: call
            .transition_output
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        predicate: call
            .predicate
            .as_ref()
            .map(tensor_ref_to_fb)
            .transpose()?
            .map(Box::new),
        rng: call.rng.as_ref().map(rng_to_fb).map(Box::new),
    })
}

/// Converts a validated control command into its tagged wire union.
fn command_to_fb(command: &BatchCommand) -> fbs::BatchCommandT {
    // The closed Rust enum guarantees exactly one command payload table.
    match command {
        BatchCommand::Start { request } => {
            fbs::BatchCommandT::StartCommand(Box::new(fbs::StartCommandT {
                request: Some(Box::new(
                    admission_to_fb(request).expect("validated start request"),
                )),
            }))
        }
        BatchCommand::Finish {
            request_key,
            retained_buffers,
        } => fbs::BatchCommandT::FinishCommand(Box::new(fbs::FinishCommandT {
            request_key: Some(Box::new(request_key_to_fb(*request_key))),
            retained_buffers: Some(
                retained_buffers
                    .iter()
                    .copied()
                    .map(buffer_id_to_fb)
                    .collect(),
            ),
        })),
        BatchCommand::Free { buffer } => {
            fbs::BatchCommandT::FreeCommand(Box::new(fbs::FreeCommandT {
                buffer: Some(Box::new(buffer_id_to_fb(*buffer))),
            }))
        }
    }
}

/// Decodes a logical computation identity without depending on physical run numbering.
fn computation_id_from_fb(id: Option<&fbs::CallId>) -> CodecResult<CallId> {
    let id = id.context("computation identity is missing")?;
    Ok(CallId::new(id.batch_id(), id.request_index()))
}

fn computation_id_to_fb(id: CallId) -> fbs::CallIdT {
    fbs::CallIdT {
        batch_id: id.batch_id,
        request_index: id.request_index,
    }
}

/// Converts a request identity into its FlatBuffers object representation.
fn request_key_to_fb(request_key: RequestKey) -> fbs::RequestKeyT {
    fbs::RequestKeyT {
        engine_id: request_key.engine_id,
        request_id: request_key.request_id.0,
        request_epoch: request_key.request_epoch,
    }
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

/// Converts a persistent-buffer identity into its wire table.
fn buffer_id_to_fb(buffer: BufferId) -> fbs::BufferIdT {
    fbs::BufferIdT {
        owner: Some(Box::new(request_key_to_fb(buffer.owner))),
        producer_call_id: Some(computation_id_to_fb(buffer.producer_call_id)),
        output_index: buffer.output_index,
        generation: buffer.generation,
    }
}

/// Flattens shape bounds into extents and a single dynamic-axis marker.
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

/// Converts tensor identity and bounded geometry into its wire table.
fn tensor_ref_to_fb(tensor: &TensorRef) -> CodecResult<fbs::TensorRefT> {
    tensor.validate()?;
    let (extents, dynamic_axis) = shape_bound_to_parts(&tensor.shape_bound);
    Ok(fbs::TensorRefT {
        id: Box::new(buffer_id_to_fb(tensor.buffer_id())),
        dtype: dtype_to_fb(tensor.dtype),
        extents: Some(extents),
        dynamic_axis,
    })
}

/// Converts deterministic random coordinates into their wire table.
fn rng_to_fb(rng: &Rng) -> fbs::RngT {
    fbs::RngT {
        seed: rng.seed,
        semantic_index_base: rng.semantic_index_base,
        draw_layout: draw_layout_to_fb(rng.draw_layout),
    }
}

/// Converts a completion report into its FlatBuffers object representation.
fn run_result_to_fb(report: &BatchOutput) -> CodecResult<fbs::BatchOutputT> {
    report.validate()?;
    Ok(fbs::BatchOutputT {
        batch_id: report.batch_id,
        completions: Some(
            report
                .completions
                .iter()
                .map(completion_record_to_fb)
                .collect(),
        ),
        products: Some(
            report
                .products
                .iter()
                .map(tensor_publication_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        registration: Some(Box::new(fbs::RegistrationAckT {
            visible: report.registration.visible,
        })),
        worker_exec_us: report.worker_exec_us,
        forward_stats: report
            .forward_stats
            .as_ref()
            .map(forward_stats_to_fb)
            .map(Box::new),
    })
}

/// Converts the completion fields into their wire table.
fn token_logprob_from_fb(entry: &fbs::TokenLogprob) -> TokenLogprob {
    TokenLogprob {
        token_id: entry.token_id(),
        logprob: entry.logprob(),
        rank: entry.rank(),
    }
}

fn token_logprob_to_fb(entry: &TokenLogprob) -> fbs::TokenLogprobT {
    fbs::TokenLogprobT {
        token_id: entry.token_id,
        logprob: entry.logprob,
        rank: entry.rank,
    }
}

fn completion_record_to_fb(record: &RequestOutput) -> fbs::RequestOutputT {
    fbs::RequestOutputT {
        sampled_logprob: record.sampled_logprob,
        top_logprobs: Some(
            record
                .top_logprobs
                .iter()
                .map(token_logprob_to_fb)
                .collect(),
        ),
        prompt_logprobs: Some(
            record
                .prompt_logprobs
                .iter()
                .map(|entries| fbs::PositionLogprobsT {
                    entries: Some(entries.iter().map(token_logprob_to_fb).collect()),
                })
                .collect(),
        ),
        request_key: Some(Box::new(request_key_to_fb(record.request_key))),
        call_id: Some(computation_id_to_fb(record.call_id)),
        status: op_status_to_fb(record.status),
        product_generations: Some(record.product_generations.clone()),
        error_code: record.error_code.map(error_code_to_fb),
        timing_counters: Some(Box::new(fbs::TimingCountersT {
            queued_us: record.timing_counters.queued_us,
            device_us: record.timing_counters.device_us,
            copy_us: record.timing_counters.copy_us,
            host_us: record.timing_counters.host_us,
        })),
        code: computation_to_fb(record.code),
        position: record.position,
        kv_visible_len: record.kv_visible_len,
        kv_computed_len: record.kv_computed_len,
        num_completed_steps: record.num_completed_steps,
        committed_tokens: Some(record.committed_tokens.clone()),
        finish_flags: Some(Box::new(fbs::FinishFlagsT {
            eos: record.finish_flags.eos,
            length: record.finish_flags.length,
            stop: record.finish_flags.stop,
        })),
        kv_output: record
            .kv_output
            .as_ref()
            .map(kv_transfer_to_fb)
            .map(Box::new),
        media_output: record
            .media_output
            .as_ref()
            .map(media_output_to_fb)
            .map(Box::new),
    }
}

/// Converts media metadata and its transport-specific artifact handle.
fn media_output_to_fb(output: &MediaOutput) -> fbs::MediaOutputT {
    fbs::MediaOutputT {
        handle: match &output.handle {
            ArtifactHandle::PosixShm { name } => {
                fbs::ArtifactHandleT::PosixShmArtifact(Box::new(fbs::PosixShmArtifactT {
                    name: Some(name.clone()),
                }))
            }
        },
        bytes: output.bytes,
    }
}

/// Converts a tensor publication into its wire descriptor.
fn tensor_publication_to_fb(payload: &TensorPublication) -> CodecResult<fbs::TensorPublicationT> {
    Ok(fbs::TensorPublicationT {
        product: Some(Box::new(tensor_ref_to_fb(&payload.product)?)),
        value: Some(Box::new(transfer_handle_to_fb(&payload.value))),
    })
}

/// Converts an error's request and call identity into its wire table.
fn error_call_to_fb(call: &ErrorCallIdentity) -> fbs::ErrorCallIdentityT {
    fbs::ErrorCallIdentityT {
        request_key: Some(Box::new(request_key_to_fb(call.request_key))),
        call_id: Some(computation_id_to_fb(call.call_id)),
    }
}

/// Decodes worker KV-cache capabilities from a verified table.
fn kv_cache_from_table(config: fbs::KVCacheInfo<'_>) -> CodecResult<KvCacheInfo> {
    Ok(KvCacheInfo {
        block_size: config.block_size(),
        num_blocks: config.num_blocks(),
        num_layers: config.num_layers(),
        total_layers: config.total_layers(),
        layer_offset: config.layer_offset(),
        num_kv_heads: config.num_kv_heads(),
        total_kv_heads: config.total_kv_heads(),
        kv_head_offset: config.kv_head_offset(),
        head_dim: config.head_dim(),
        bytes_per_token: config.bytes_per_token(),
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

/// Converts KV-cache capacity and group geometry into their wire table.
fn kv_cache_to_fb(config: &KvCacheInfo) -> fbs::KVCacheInfoT {
    fbs::KVCacheInfoT {
        block_size: config.block_size,
        num_blocks: config.num_blocks,
        num_layers: config.num_layers,
        total_layers: config.total_layers,
        layer_offset: config.layer_offset,
        num_kv_heads: config.num_kv_heads,
        total_kv_heads: config.total_kv_heads,
        kv_head_offset: config.kv_head_offset,
        head_dim: config.head_dim,
        bytes_per_token: config.bytes_per_token,
        groups: Some(config.groups.iter().map(kv_group_to_fb).collect()),
        dtype: Some(config.dtype.as_str().to_owned()),
    }
}

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

fn parallel_to_fb(config: &uniserve_core::ParallelConfig) -> fbs::ParallelConfigT {
    use uniserve_core::SequenceParallel;
    let sequence_parallel = match config.sequence_parallel {
        SequenceParallel::Local => {
            fbs::SequenceParallelT::LocalSequence(Box::new(fbs::LocalSequenceT {}))
        }
        SequenceParallel::Ulysses { ulysses_degree } => {
            fbs::SequenceParallelT::UlyssesSequence(Box::new(fbs::UlyssesSequenceT {
                ulysses_degree: ulysses_degree as u32,
            }))
        }
        SequenceParallel::Ring { ring_degree } => {
            fbs::SequenceParallelT::RingSequence(Box::new(fbs::RingSequenceT {
                ring_degree: ring_degree as u32,
            }))
        }
        SequenceParallel::Hybrid {
            ulysses_degree,
            ring_degree,
        } => fbs::SequenceParallelT::HybridSequence(Box::new(fbs::HybridSequenceT {
            ulysses_degree: ulysses_degree as u32,
            ring_degree: ring_degree as u32,
        })),
        SequenceParallel::Allgather { allgather_degree } => {
            fbs::SequenceParallelT::GatherSequence(Box::new(fbs::GatherSequenceT {
                allgather_degree: allgather_degree as u32,
            }))
        }
        SequenceParallel::Attention2d {
            attn2d_row_size,
            attn2d_col_size,
            ulysses_degree,
        } => fbs::SequenceParallelT::Attention2dSequence(Box::new(fbs::Attention2dSequenceT {
            attn2d_row_size: attn2d_row_size as u32,
            attn2d_col_size: attn2d_col_size as u32,
            ulysses_degree: ulysses_degree as u32,
        })),
    };
    fbs::ParallelConfigT {
        tensor_parallel_size: config.tensor_parallel_size as u32,
        pipeline_parallel_size: config.pipeline_parallel_size as u32,
        sequence_parallel,
    }
}

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
        fbs::SequenceParallel::RingSequence => {
            let value = config
                .sequence_parallel_as_ring_sequence()
                .context("missing RingSequence configuration")?;
            SequenceParallel::Ring {
                ring_degree: value.ring_degree() as usize,
            }
        }
        fbs::SequenceParallel::HybridSequence => {
            let value = config
                .sequence_parallel_as_hybrid_sequence()
                .context("missing HybridSequence configuration")?;
            SequenceParallel::Hybrid {
                ulysses_degree: value.ulysses_degree() as usize,
                ring_degree: value.ring_degree() as usize,
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
        fbs::SequenceParallel::Attention2dSequence => {
            let value = config
                .sequence_parallel_as_attention_2d_sequence()
                .context("missing Attention2dSequence configuration")?;
            SequenceParallel::Attention2d {
                attn2d_row_size: value.attn2d_row_size() as usize,
                attn2d_col_size: value.attn2d_col_size() as usize,
                ulysses_degree: value.ulysses_degree() as usize,
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

/// Converts validated worker capabilities into their wire table.
fn info_to_fb(info: &WorkerInfo) -> CodecResult<fbs::WorkerInfoT> {
    info.validate()?;
    Ok(fbs::WorkerInfoT {
        model_name: Some(info.model_name.clone()),
        endpoint: Some(Box::new(endpoint_to_fb(&info.endpoint))),
        device: Some(info.device.clone()),
        transfer_backends: Some(info.transfer_backends.clone()),
        fabric_handles: info.fabric_handles,
        world_size: info.world_size,
        configuration_id: Some(info.configuration_id.clone()),
        components: Some(
            info.components
                .iter()
                .map(|component| {
                    Ok(fbs::EntryInfoT {
                        name: Some(component.name.clone()),
                        ranks: Some(
                            component
                                .config
                                .ranks
                                .iter()
                                .map(|&rank| rank as u64)
                                .collect(),
                        ),
                        parallel_config: Some(Box::new(parallel_to_fb(
                            &component.config.parallel_config,
                        ))),
                        distribution: match component.config.distribution {
                            None => fbs::ComponentDistribution::Local,
                            Some(uniserve_core::ComponentDistribution::TemporalUnits) => {
                                fbs::ComponentDistribution::TemporalUnits
                            }
                        },
                        units_per_rank: component.config.units_per_rank as u32,
                        outputs: Some(
                            component
                                .outputs
                                .iter()
                                .map(|output| {
                                    let (extents, dynamic_axis) =
                                        shape_bound_to_parts(&output.shape_bound);
                                    fbs::OutputInfoT {
                                        name: Some(output.name.clone()),
                                        dtype: dtype_to_fb(output.dtype),
                                        extents: Some(extents),
                                        dynamic_axis,
                                    }
                                })
                                .collect(),
                        ),
                    })
                })
                .collect::<CodecResult<Vec<_>>>()?,
        ),
        supported_ops: Some(
            info.supported_ops
                .iter()
                .copied()
                .map(computation_to_fb)
                .collect(),
        ),
        queue_depth: info.queue_depth,
        max_batch_ops: info.max_batch_ops,
        max_batch_tokens: info.max_batch_tokens,
        request_slots: info.request_slots,
        kv_cache: info.kv_cache.as_ref().map(kv_cache_to_fb).map(Box::new),
        latent_page_units: info.latent_page_units,
        latent_pages: info.latent_pages,
        buffer_pool_bytes: info.buffer_pool_bytes,
        encoder_cache_entries: info.encoder_cache_entries,
        encoder_entry_bytes: info.encoder_entry_bytes,
        max_unresolved_ops: info.max_unresolved_ops,
        host_lane_capacity: info.host_lane_capacity,
        pipeline_components: Some(
            info.pipeline_components
                .iter()
                .map(|(stage, component)| fbs::PipelineComponentT {
                    stage: pipeline_stage_to_fb(*stage),
                    component: Some(component.clone()),
                })
                .collect(),
        ),
        num_inference_steps: info.num_inference_steps,
    })
}

/// Converts sampling parameters into their FlatBuffers object representation.
fn sampling_to_fb(sampling: &SamplingParams) -> CodecResult<fbs::SamplingParamsT> {
    validate_sampling(sampling)?;

    Ok(fbs::SamplingParamsT {
        // Scalar controls copy directly into the object representation.
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        ignore_eos: sampling.ignore_eos,
        seed: sampling.seed,
        min_p: sampling.min_p,
        repetition_penalty: sampling.repetition_penalty,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,

        // Token-specific controls require owned FlatBuffers vectors.
        logit_bias: Some(
            sampling
                .logit_bias
                .iter()
                .map(|(token_id, bias)| fbs::TokenBiasT {
                    token_id: *token_id,
                    bias: *bias,
                })
                .collect(),
        ),
        min_tokens: sampling.min_tokens as u64,
        n_logprobs: sampling.n_logprobs,
        bad_words_ids: Some(
            sampling
                .bad_words_ids
                .iter()
                .map(|items| fbs::U32ListT {
                    items: Some(items.clone()),
                })
                .collect(),
        ),
        allowed_token_ids: sampling.allowed_token_ids.clone(),
        return_logprobs: sampling.return_logprobs,
        return_prompt_logprobs: sampling.return_prompt_logprobs,
        n_prompt_logprobs: sampling.n_prompt_logprobs,
        logprob_token_ids: Some(sampling.logprob_token_ids.clone()),
        typical_p: sampling.typical_p,
        forced_token_ids: Some(sampling.forced_token_ids.clone()),
    })
}

/// Validates floating-point sampling fields before they cross the wire boundary.
fn validate_sampling(sampling: &SamplingParams) -> CodecResult<()> {
    for value in [
        sampling.temperature,
        sampling.top_p,
        sampling.min_p,
        sampling.repetition_penalty,
        sampling.frequency_penalty,
        sampling.presence_penalty,
    ] {
        codec_ensure!(value.is_finite(), "sampling parameters must be finite");
    }
    for (_, value) in &sampling.logit_bias {
        codec_ensure!(value.is_finite(), "logit bias must be finite");
    }
    codec_ensure!(
        sampling.typical_p.is_finite() && sampling.typical_p > 0.0 && sampling.typical_p <= 1.0,
        "sampling.typical_p must be in (0, 1]"
    );
    Ok(())
}

/// Converts image-generation parameters into their wire table.
fn image_to_fb(image: &uniserve_core::ImageParams) -> fbs::ImageParamsT {
    fbs::ImageParamsT {
        steps: image.steps,
        cfg_text_scale: image.cfg_text_scale,
        cfg_img_scale: image.cfg_img_scale,
        cfg_renorm_type: Some(image.cfg_renorm_type.as_str().to_owned()),
        cfg_renorm_min: image.cfg_renorm_min,
        cfg_interval_lo: image.cfg_interval.0,
        cfg_interval_hi: image.cfg_interval.1,
        timestep_shift: image.timestep_shift,
        height: image.height,
        width: image.width,
        seed: image.seed,
        negative_prompt: Some(image.negative_prompt.clone()),
        max_images: image.max_images,
        image_prompts: Some(image.image_prompts.clone()),
        retain_images: image.retain_images,
    }
}

/// Converts forward-path counters into their wire table.
fn forward_stats_to_fb(stats: &ForwardStats) -> fbs::ForwardStatsT {
    fbs::ForwardStatsT {
        // Aggregate execution-mode counters.
        mode_counts: Some(map_to_fb(&stats.mode_counts)),
        mode_tokens: Some(map_to_fb(&stats.mode_tokens)),
        mode_us: Some(map_to_fb(&stats.mode_us)),

        // Attention backend activity.
        attention_launches: stats.attention_launches,
        attention_us: stats.attention_us,
        attention_backend_counts: Some(map_to_fb(&stats.attention_backend_counts)),

        // CUDA graph lifecycle and padding behavior.
        cuda_graph_captures: stats.cuda_graph_captures,
        cuda_graph_replays: stats.cuda_graph_replays,
        cuda_graph_misses: stats.cuda_graph_misses,
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: Some(map_to_fb(&stats.cuda_graph_runtime_mode_counts)),

        // Decode relay cache effectiveness.
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses,

        // FlashInfer planning activity.
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls,
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses,
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows,
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices,
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls,
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses,

        // Speculative-verification outcomes.
        spec_verify_rows: stats.spec_verify_rows,
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
        spec_verify_path_counts: Some(map_to_fb(&stats.spec_verify_path_counts)),

        // Per-component timing totals.
        component_us: Some(map_to_fb(&stats.component_us)),
    }
}

/// Converts a deterministic counter map into ordered key-value tables.
fn map_to_fb(map: &BTreeMap<String, u64>) -> Vec<fbs::StringU64PairT> {
    map.iter()
        .map(|(key, value)| fbs::StringU64PairT {
            key: Some(key.clone()),
            value: *value,
        })
        .collect()
}

/// Converts full-attention or sliding-window KV group geometry.
fn kv_group_to_fb(group: &KvCacheGroup) -> fbs::KvGroupT {
    match group.kind {
        KvGroupKind::Full => fbs::KvGroupT {
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::Full,
            window: 0,
            sink: 0,
        },
        KvGroupKind::SlidingWindow { window, sink } => fbs::KvGroupT {
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::SlidingWindow,
            window,
            sink,
        },
    }
}

/// Converts physical process coordinates into their wire table.
fn endpoint_to_fb(endpoint: &WorkerEndpoint) -> fbs::WorkerEndpointT {
    fbs::WorkerEndpointT {
        worker_id: Some(endpoint.worker_id.clone()),
        rank: endpoint.rank,
        node: Some(endpoint.node.clone()),
        address_space: Some(endpoint.address_space.clone()),
        incarnation: Some(endpoint.incarnation.clone()),
    }
}

fn forward_mode_to_fb(value: ForwardMode) -> fbs::ForwardMode {
    match value {
        ForwardMode::Prefill => fbs::ForwardMode::Prefill,
        ForwardMode::Decode => fbs::ForwardMode::Decode,
        ForwardMode::Verify => fbs::ForwardMode::Verify,
    }
}

fn forward_mode_from_fb(value: fbs::ForwardMode) -> CodecResult<ForwardMode> {
    Ok(match value {
        fbs::ForwardMode::Prefill => ForwardMode::Prefill,
        fbs::ForwardMode::Decode => ForwardMode::Decode,
        fbs::ForwardMode::Verify => ForwardMode::Verify,
        _ => codec_bail!("unknown forward_mode {}", value.0),
    })
}

fn pipeline_stage_to_fb(value: PipelineStage) -> fbs::PipelineStage {
    match value {
        PipelineStage::VisionEncoding => fbs::PipelineStage::VisionEncoding,
        PipelineStage::LatentEncoding => fbs::PipelineStage::LatentEncoding,
        PipelineStage::TextEncoding => fbs::PipelineStage::TextEncoding,
        PipelineStage::LatentPreparation => fbs::PipelineStage::LatentPreparation,
        PipelineStage::Denoising => fbs::PipelineStage::Denoising,
        PipelineStage::ImageDecoding => fbs::PipelineStage::ImageDecoding,
        PipelineStage::VideoDecoding => fbs::PipelineStage::VideoDecoding,
        PipelineStage::AudioDecoding => fbs::PipelineStage::AudioDecoding,
        PipelineStage::VideoEncoding => fbs::PipelineStage::VideoEncoding,
        PipelineStage::AudioEncoding => fbs::PipelineStage::AudioEncoding,
        PipelineStage::Muxing => fbs::PipelineStage::Muxing,
    }
}

fn pipeline_stage_from_fb(value: fbs::PipelineStage) -> CodecResult<PipelineStage> {
    Ok(match value {
        fbs::PipelineStage::VisionEncoding => PipelineStage::VisionEncoding,
        fbs::PipelineStage::LatentEncoding => PipelineStage::LatentEncoding,
        fbs::PipelineStage::TextEncoding => PipelineStage::TextEncoding,
        fbs::PipelineStage::LatentPreparation => PipelineStage::LatentPreparation,
        fbs::PipelineStage::Denoising => PipelineStage::Denoising,
        fbs::PipelineStage::ImageDecoding => PipelineStage::ImageDecoding,
        fbs::PipelineStage::VideoDecoding => PipelineStage::VideoDecoding,
        fbs::PipelineStage::AudioDecoding => PipelineStage::AudioDecoding,
        fbs::PipelineStage::VideoEncoding => PipelineStage::VideoEncoding,
        fbs::PipelineStage::AudioEncoding => PipelineStage::AudioEncoding,
        fbs::PipelineStage::Muxing => PipelineStage::Muxing,
        _ => codec_bail!("unknown pipeline_stage {}", value.0),
    })
}

fn transfer_mode_to_fb(value: TransferMode) -> fbs::TransferMode {
    match value {
        TransferMode::Tensor => fbs::TransferMode::Tensor,
        TransferMode::KvPublish => fbs::TransferMode::KvPublish,
        TransferMode::KvInstall => fbs::TransferMode::KvInstall,
    }
}

fn transfer_mode_from_fb(value: fbs::TransferMode) -> CodecResult<TransferMode> {
    Ok(match value {
        fbs::TransferMode::Tensor => TransferMode::Tensor,
        fbs::TransferMode::KvPublish => TransferMode::KvPublish,
        fbs::TransferMode::KvInstall => TransferMode::KvInstall,
        _ => codec_bail!("unknown transfer_mode {}", value.0),
    })
}

/// Encode exactly one classification in a three-byte inline wire struct.
fn computation_to_fb(value: CallKind) -> fbs::CallKindT {
    let mut encoded = fbs::CallKindT::default();
    match value {
        CallKind::Forward(mode) => encoded.forward_mode = forward_mode_to_fb(mode),
        CallKind::Pipeline(stage) => encoded.stage = pipeline_stage_to_fb(stage),
        CallKind::Transfer(mode) => encoded.transfer = transfer_mode_to_fb(mode),
    }
    encoded
}

/// Reject missing, conflicting, and unknown tags at the transport boundary.
fn computation_from_fb(value: &fbs::CallKind) -> CodecResult<CallKind> {
    let present = u8::from(value.forward_mode() != fbs::ForwardMode::None)
        + u8::from(value.stage() != fbs::PipelineStage::None)
        + u8::from(value.transfer() != fbs::TransferMode::None);
    if present != 1 {
        codec_bail!("computation must select exactly one classification");
    }
    if value.forward_mode() != fbs::ForwardMode::None {
        Ok(CallKind::Forward(forward_mode_from_fb(
            value.forward_mode(),
        )?))
    } else if value.stage() != fbs::PipelineStage::None {
        Ok(CallKind::Pipeline(pipeline_stage_from_fb(value.stage())?))
    } else {
        Ok(CallKind::Transfer(transfer_mode_from_fb(value.transfer())?))
    }
}

/// Decodes transport-specific storage coordinates and common tensor metadata.
fn transfer_locator_from_table(value: fbs::Locator<'_>) -> CodecResult<Locator> {
    // The transport discriminant determines which coordinate fields are
    // required; unrelated fields are deliberately ignored.
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
                .context("shared-memory transfer endpoint is missing")?
                .to_owned(),
            name: value
                .name()
                .context("shared-memory transfer name is missing")?
                .to_owned(),
        }
    } else if value.transport() == fbs::TransferTransportKind::CudaVmm {
        TransferTransport::CudaVmm {
            endpoint: value
                .endpoint()
                .context("CUDA VMM endpoint is missing")?
                .to_owned(),
            publication_id: value
                .publication_id()
                .context("CUDA VMM publication identity is missing")?
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

fn tensor_transfer_to_fb(value: &TensorTransfer) -> fbs::TensorTransferT {
    fbs::TensorTransferT {
        shape: Some(value.shape.clone()),
        locations: Some(value.locations.iter().map(transfer_locator_to_fb).collect()),
    }
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

/// Converts common tensor metadata and one transport coordinate set.
fn transfer_locator_to_fb(value: &Locator) -> fbs::LocatorT {
    // Populate transport-independent tensor metadata before selecting the
    // coordinate family stored in the shared FlatBuffers table.
    let mut output = fbs::LocatorT {
        source: Some(Box::new(endpoint_to_fb(&value.source))),
        nbytes: value.nbytes,
        dtype: Some(value.dtype.clone()),
        shape: Some(value.shape.clone()),
        device: Some(value.device.clone()),
        offset: Some(value.offset.clone()),
        ..Default::default()
    };

    match &value.transport {
        TransferTransport::Local { endpoint, key } => {
            output.transport = fbs::TransferTransportKind::Local;
            output.endpoint = Some(endpoint.clone());
            output.key = *key;
        }
        TransferTransport::PosixShm { endpoint, name } => {
            output.transport = fbs::TransferTransportKind::PosixShm;
            output.endpoint = Some(endpoint.clone());
            output.name = Some(name.clone());
        }
        TransferTransport::CudaVmm {
            endpoint,
            publication_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
            allocation_handle,
            acknowledgment_offset,
        } => {
            output.transport = fbs::TransferTransportKind::CudaVmm;
            output.endpoint = Some(endpoint.clone());
            output.publication_id = Some(publication_id.clone());
            output.storage_size_bytes = *storage_size_bytes;
            output.storage_offsets_bytes = Some(storage_offsets_bytes.clone());
            output.span_lengths = Some(span_lengths.clone());
            output.span_counts = Some(span_counts.clone());
            output.tensor_stride = Some(tensor_stride.clone());
            output.ready_event_handle = Some(ready_event_handle.clone());
            output.allocation_handle = Some(allocation_handle.clone());
            output.acknowledgment_offset = *acknowledgment_offset;
        }
        TransferTransport::Channel { endpoint, payload } => {
            output.transport = fbs::TransferTransportKind::Channel;
            output.endpoint = Some(endpoint.clone());
            output.payload = Some(payload.clone());
        }
    }

    output
}

/// Converts a transfer handle into its product-family wire union.
fn transfer_handle_to_fb(value: &TransferHandle) -> fbs::TransferHandleT {
    // The product family selects a self-contained transfer metadata table;
    // physical tensor order is preserved within each family.
    let value = match value {
        TransferHandle::Encoder {
            height,
            width,
            payload_kind,
            tensor,
        } => fbs::TransferDataT::EncoderTransfer(Box::new(fbs::EncoderTransferT {
            height: *height,
            width: *width,
            payload_kind: match payload_kind {
                FeatureKind::Vision => fbs::FeatureKind::Vision,
                FeatureKind::Latent => fbs::FeatureKind::Latent,
            },
            tensor: Some(Box::new(tensor_transfer_to_fb(tensor))),
        })),

        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            tensor,
        } => fbs::TransferDataT::DeviceProductTransfer(Box::new(fbs::DeviceProductTransferT {
            height: *height,
            width: *width,
            value_range: Some(value_range.clone()),
            tensor: Some(Box::new(tensor_transfer_to_fb(tensor))),
        })),

        TransferHandle::Latent {
            height,
            width,
            latent_units,
            step,
            tensor,
        } => fbs::TransferDataT::LatentTransfer(Box::new(fbs::LatentTransferT {
            height: *height,
            width: *width,
            latent_units: *latent_units,
            step: *step,
            tensor: Some(Box::new(tensor_transfer_to_fb(tensor))),
        })),
    };

    fbs::TransferHandleT { value }
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

/// Maps an call status to its stable FlatBuffers discriminant.
fn op_status_to_fb(status: CallStatus) -> fbs::CallStatus {
    match status {
        CallStatus::Ok => fbs::CallStatus::Ok,
        CallStatus::Predicated => fbs::CallStatus::Predicated,
        CallStatus::Error => fbs::CallStatus::Error,
    }
}

/// Decodes a supported FlatBuffers call status.
fn op_status_from_fb(status: fbs::CallStatus) -> CodecResult<CallStatus> {
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

fn kv_transfer_from_table(transfer: fbs::KvTransfer<'_>) -> CodecResult<KvTransfer> {
    Ok(KvTransfer {
        tensors: transfer
            .tensors()
            .map(|items| {
                items
                    .iter()
                    .map(tensor_transfer_from_table)
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
        published_extent: transfer.published_extent(),
        group_id: transfer.group_id(),
        page_size: transfer.page_size(),
        compute_dtype: transfer
            .compute_dtype()
            .context("KV transfer compute dtype is missing")?
            .to_owned(),
    })
}

fn kv_transfer_to_fb(transfer: &KvTransfer) -> fbs::KvTransferT {
    let KvTransfer {
        tensors,
        source,
        destination,
        base,
        base_extent,
        published_extent,
        group_id,
        compute_dtype,
        page_size,
    } = transfer;
    fbs::KvTransferT {
        tensors: Some(tensors.iter().map(tensor_transfer_to_fb).collect()),
        source: Some(Box::new(buffer_id_to_fb(*source))),
        destination: Some(destination.clone()),
        base: base.map(buffer_id_to_fb).map(Box::new),
        base_extent: *base_extent,
        published_extent: *published_extent,
        group_id: *group_id,
        compute_dtype: Some(compute_dtype.clone()),
        page_size: *page_size,
    }
}
