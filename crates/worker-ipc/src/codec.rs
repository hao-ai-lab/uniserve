//! FlatBuffers encoding and verified decoding for worker protocol messages.

use std::collections::BTreeMap;

use flatbuffers::FlatBufferBuilder;
use uniserve_core::{BlockId, KvCacheGroup, KvGroupKind, RequestId, SamplingParams};

use crate::schema::uniserve::ipc as fbs;
use crate::{
    ArRequestParams, ArtifactHandle, BatchCommand, BlockTable, Bounds, BufferAllocation, BufferId,
    CachePageAllocation, Checkpoint, CheckpointPoint, CloseReason, DType, DecodeRange,
    DiffusionRequestParams, DiffusionResult, DimBound, Disposition, DrawLayout, ErrorCode,
    ErrorOperationIdentity, FinishFlags, InlineValue, KvCacheConfig, LatentParams, Locator,
    LogicalLengths, MediaExecutionPlan, MediaGeometry, MediaOutput, MediaPlanRepeat,
    MediaPlanStage, MediaTrack, ModelOutput, NewRequest, OpCode, OpId, OpPayload, OpStatus,
    Operation, PointRange, ProductKind, ProductPayload, ProductRef, RegistrationAck, RequestKey,
    RequestKind, ResponseKind, ResultData, ResultPayload, Rng, RowGeometry, Run, RunResult,
    ShapeBound, StorageClass, TensorTransfer, TimingCounters, TokenSpan, TransferHandle,
    TransferTransport, UmmRequestParams, WorkerEndpoint, WorkerForwardStats, WorkerInfo,
    WorkerRequest, WorkerResponse, WorkerResponseError,
};

/// Result type returned by FlatBuffers codec operations.
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
    let call_id = request.call_id();
    let run = request.run().map(run_from_table).transpose()?;
    let run_id = request.poll_run_id();
    let payload_count = usize::from(run.is_some()) + usize::from(run_id.is_some());
    Ok(match kind {
        RequestKind::Info => {
            codec_ensure!(payload_count == 0, "info carries a payload");
            WorkerRequest::Info { call_id }
        }
        RequestKind::Submit => {
            codec_ensure!(payload_count == 1, "submit requires exactly one run");
            WorkerRequest::Submit {
                call_id,
                run: run.context("submit request has no run")?,
            }
        }
        RequestKind::Poll => {
            codec_ensure!(payload_count == 1, "poll requires one run id");
            WorkerRequest::Poll {
                call_id,
                run_id: run_id.context("poll request has no run id")?,
            }
        }
        RequestKind::Close => {
            codec_ensure!(payload_count == 0, "close carries a payload");
            WorkerRequest::Close { call_id }
        }
    })
}

/// Decodes a response and enforces payload exclusivity for its discriminator.
fn response_from_table(response: fbs::WorkerResponse<'_>) -> CodecResult<WorkerResponse> {
    // Decode every optional branch before dispatch so each response kind can
    // enforce exclusivity between success payloads and structured error data.
    let kind = response_kind_from_fb(response.kind())?;
    let call_id = response.call_id();
    let info = response.info().map(info_from_table).transpose()?;
    let result = response.result().map(run_result_from_table).transpose()?;
    let payload_count = usize::from(info.is_some()) + usize::from(result.is_some());

    // Error metadata occupies independent optional fields on the wire.
    let message = response.message().map(str::to_owned);
    let code = response.code().map(str::to_owned);
    let retryable = response.retryable();
    let fatal = response.fatal();
    let phase = response.phase().map(str::to_owned);
    let route = response.route().map(str::to_owned);
    let operations: Vec<ErrorOperationIdentity> = response
        .operations()
        .map(|items| {
            items
                .iter()
                .map(error_operation_from_table)
                .collect::<CodecResult<_>>()
        })
        .transpose()?
        .unwrap_or_default();
    let carries_error = message.is_some()
        || code.is_some()
        || retryable.is_some()
        || fatal.is_some()
        || phase.is_some()
        || route.is_some()
        || !operations.is_empty();

    // The response discriminator defines the exact legal field combination.
    Ok(match kind {
        ResponseKind::Info => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid info response"
            );
            WorkerResponse::Info {
                call_id,
                info: info.context("info response has no info")?,
            }
        }
        ResponseKind::Result => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid result response"
            );
            WorkerResponse::Result {
                call_id,
                result: result.context("result response has no result")?,
            }
        }
        ResponseKind::Ok => {
            codec_ensure!(payload_count == 0 && !carries_error, "invalid ok response");
            WorkerResponse::Ok { call_id }
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
                call_id,
                error: WorkerResponseError {
                    message,
                    code,
                    retryable: retryable.context("error response has no retryable flag")?,
                    fatal: fatal.context("error response has no fatal flag")?,
                    phase,
                    route,
                    operations,
                },
            }
        }
    })
}

/// Decodes an owned run and validates all nested operation and params contracts.
fn run_from_table(run: fbs::Run<'_>) -> CodecResult<Run> {
    // Preserve wire order for operations, controls, and products because later
    // validation and execution interpret those collections positionally.
    let run = Run {
        batch_id: run.batch_id(),
        run_id: run.run_id(),
        collective_seq: run.collective_seq(),

        // Decode executable graph records in their submitted order.
        operations: run
            .operations()
            .map(|items| {
                items
                    .iter()
                    .map(operation_from_table)
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
        forward_rows: run
            .forward_rows()
            .map(|items| items.iter().map(row_geometry_from_table).collect())
            .unwrap_or_default(),

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
        input_products: run
            .input_products()
            .map(|items| {
                items
                    .iter()
                    .map(product_payload_from_table)
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
    admission: fbs::DiffusionRequestParams<'_>,
) -> CodecResult<DiffusionRequestParams> {
    let geometry = admission
        .geometry()
        .context("diffusion request has no resolved geometry")?;
    Ok(DiffusionRequestParams {
        prompt_token_ids: admission
            .prompt_token_ids()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        seed: admission.seed(),
        geometry: MediaGeometry {
            frame_count: geometry.frame_count(),
            video_units: geometry.video_units(),
            prompt_tokens: geometry.prompt_tokens(),
            denoise_steps: geometry.denoise_steps(),
        },
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

/// Decodes the operation and sequence geometry for one forward row.
fn row_geometry_from_table(row: fbs::RowGeometry<'_>) -> RowGeometry {
    RowGeometry {
        operation_index: row.operation_index(),
        request_pool_index: row.request_pool_index(),
        seq_len: row.seq_len(),
        query_len: row.query_len(),
        write_kv: row.write_kv(),
    }
}

/// Decodes a latent-page params bound to a request operation.
fn latent_params_from_table(params: fbs::LatentParams<'_>) -> CodecResult<LatentParams> {
    Ok(LatentParams {
        request_key: request_key_from_table(params.request_key(), "latent params.request_key")?,
        op_id: OpId(params.op_id()),
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

/// Decodes a diffusion decoder params bound to a request operation.
fn decode_range_from_table(params: fbs::DecodeRange<'_>) -> CodecResult<DecodeRange> {
    Ok(DecodeRange {
        request_key: request_key_from_table(params.request_key(), "decode params.request_key")?,
        op_id: OpId(params.op_id()),
        track: match params.track() {
            fbs::MediaTrack::Video => MediaTrack::Video,
            fbs::MediaTrack::Audio => MediaTrack::Audio,
            value => codec_bail!("unknown media track {}", value.0),
        },
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

/// Decodes the single executable payload and its entry binding.
fn operation_from_table(operation: fbs::Operation<'_>) -> CodecResult<Operation> {
    let payload = operation.payload();
    let operation = Operation {
        request_key: request_key_from_table(operation.request_key(), "operation.request_key")?,
        op_id: OpId(operation.op_id()),
        parent: operation.parent().map(checkpoint_from_table).transpose()?,
        entry: operation.entry().to_owned(),
        payload: OpPayload {
            code: op_code_from_fb(payload.code())?,
            bounds: Bounds {
                max_points: payload.max_points(),
                max_tokens: payload.max_tokens(),
                max_kv_pages: payload.max_kv_pages(),
                max_latent_bytes: payload.max_latent_bytes(),
                max_completion_bytes: payload.max_completion_bytes(),
                max_transfer_bytes: payload.max_transfer_bytes(),
            },
            inputs: payload
                .inputs()
                .map(|items| {
                    items
                        .iter()
                        .map(product_ref_from_table)
                        .collect::<CodecResult<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default(),
            outputs: payload
                .outputs()
                .map(|items| {
                    items
                        .iter()
                        .map(product_ref_from_table)
                        .collect::<CodecResult<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default(),
            predicate: payload
                .predicate()
                .map(product_ref_from_table)
                .transpose()?,
            rng: payload.rng().map(rng_from_table).transpose()?,
            control_seq: payload.control_seq(),
        },
    };
    operation.validate()?;
    Ok(operation)
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

        fbs::BatchCommand::CommitCommand => {
            let commit = envelope
                .command_as_commit_command()
                .context("commit command table is missing")?;
            BatchCommand::Commit {
                request_key: request_key_from_table(
                    commit.request_key(),
                    "control.commit.request_key",
                )?,
                control_seq: commit.control_seq(),
                expected_parent: checkpoint_from_table(
                    commit
                        .expected_parent()
                        .context("commit control has no expected parent")?,
                )?,
                selected: checkpoint_from_table(
                    commit
                        .selected()
                        .context("commit control has no selected version")?,
                )?,
                public_event_limit: commit.public_event_limit(),
                disposition: disposition_from_fb(commit.disposition())?,
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
                control_seq: finish.control_seq(),
                cutoff: checkpoint_from_table(
                    finish.cutoff().context("finish control has no cutoff")?,
                )?,
                reason: close_reason_from_fb(finish.reason())?,
                retained_buffers: finish
                    .retained_buffers()
                    .into_iter()
                    .flatten()
                    .map(buffer_id_from_table)
                    .collect::<CodecResult<Vec<_>>>()?,
            }
        }

        fbs::BatchCommand::RetireCommand => {
            let retire = envelope
                .command_as_retire_command()
                .context("retire command table is missing")?;
            BatchCommand::Retire {
                request_key: request_key_from_table(
                    retire.request_key(),
                    "control.retire.request_key",
                )?,
                retained_buffers: retire
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
        authority_id: request_key.authority_id(),
        request_id: RequestId(request_key.request_id()),
        epoch: request_key.epoch(),
    })
}

/// Decodes the fixed or device-selected point of an operation checkpoint.
fn checkpoint_from_table(version: fbs::Checkpoint<'_>) -> CodecResult<Checkpoint> {
    let point = match version.point_type() {
        fbs::CheckpointPoint::CheckpointFixed => {
            let fixed = version
                .point_as_checkpoint_fixed()
                .context("fixed point table is missing")?;
            CheckpointPoint::Fixed(fixed.point())
        }
        fbs::CheckpointPoint::CheckpointDeviceSelected => {
            version
                .point_as_checkpoint_device_selected()
                .context("device-selected point table is missing")?;
            CheckpointPoint::DeviceSelected
        }
        _ => codec_bail!("checkpoint point union is empty"),
    };
    Ok(Checkpoint {
        op_id: OpId(version.op_id()),
        point,
    })
}

/// Decodes the shared identity fields of a scalar product reference.
fn scalar_id_from_table(
    id: fbs::ScalarId<'_>,
    label: &str,
) -> CodecResult<(RequestKey, OpId, u16, u32)> {
    Ok((
        request_key_from_table(id.request_key(), &format!("{label}.request_key"))?,
        OpId(id.producer_op_id()),
        id.output_index(),
        id.generation(),
    ))
}

/// Decodes a persistent-buffer descriptor and its declared byte bound.
fn buffer_descriptor_from_table(
    descriptor: fbs::BufferDescriptor<'_>,
    label: &str,
) -> CodecResult<(RequestKey, OpId, u16, u32, u64)> {
    let id = descriptor
        .id()
        .with_context(|| format!("{label}.id is missing"))?;
    let id = buffer_id_from_table(id)?;
    Ok((
        id.owner,
        id.producer_op_id,
        id.output_index,
        id.generation,
        descriptor.max_bytes(),
    ))
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

/// Decodes a typed value union and verifies its declared storage bounds.
fn product_ref_from_table(reference: fbs::ValueRef<'_>) -> CodecResult<ProductRef> {
    let scalar = |id: Option<fbs::ScalarId<'_>>, label: &str| {
        scalar_id_from_table(id.with_context(|| format!("{label}.id is missing"))?, label)
    };

    let buffer = |descriptor: Option<fbs::BufferDescriptor<'_>>, label: &str| {
        buffer_descriptor_from_table(
            descriptor.with_context(|| format!("{label}.buffer is missing"))?,
            label,
        )
    };

    let product = match reference.value_type() {
        // Scalar and host-relay values derive shape from compact family fields.
        fbs::ValueReference::TokenValue => {
            let value = reference
                .value_as_token_value()
                .context("token value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "token_value")?;
            ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Token,
                storage_class: match value.delivery() {
                    fbs::TokenDelivery::Inline => StorageClass::HostStaging,
                    fbs::TokenDelivery::Relay => StorageClass::RequestRelay,
                    _ => codec_bail!("unknown token delivery"),
                },
                dtype: DType::U32,
                shape_bound: ShapeBound {
                    dims: (value.max_tokens() > 1)
                        .then_some(DimBound::Static(value.max_tokens()))
                        .into_iter()
                        .collect(),
                },
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            }
        }
        fbs::ValueReference::LogprobValue => {
            let value = reference
                .value_as_logprob_value()
                .context("logprob value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "logprob_value")?;
            ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Logprob,
                storage_class: StorageClass::HostStaging,
                dtype: match value.format() {
                    fbs::LogprobFormat::PackedBytes => DType::U8,
                    fbs::LogprobFormat::F32Values => DType::F32,
                    _ => codec_bail!("unknown logprob format"),
                },
                shape_bound: ShapeBound {
                    dims: (value.max_count() > 1)
                        .then_some(DimBound::Device {
                            max: value.max_count(),
                        })
                        .into_iter()
                        .collect(),
                },
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            }
        }
        // Persistent tensor values require their descriptor byte bound to
        // agree with the reconstructed typed shape.
        fbs::ValueReference::FeatureBuffer => {
            let value = reference
                .value_as_feature_buffer()
                .context("feature buffer reference is missing")?;
            let (request_key, producer_op_id, output_index, generation, max_bytes) =
                buffer(value.buffer(), "feature_buffer")?;
            let shape_bound = shape_bound_from_parts(value.extents(), value.dynamic_axis())?;
            let product = ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: match value.feature() {
                    fbs::FeatureKind::Vision => ProductKind::VisionFeature,
                    fbs::FeatureKind::Latent => ProductKind::LatentFeature,
                    _ => codec_bail!("unknown feature kind"),
                },
                storage_class: StorageClass::LatentArena,
                dtype: dtype_from_fb(value.dtype())?,
                shape_bound,
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            };
            codec_ensure!(
                product.max_bytes() == max_bytes,
                "feature buffer byte bound disagrees with its typed shape"
            );
            product
        }
        fbs::ValueReference::TensorBuffer => {
            let value = reference
                .value_as_tensor_buffer()
                .context("tensor buffer reference is missing")?;
            let (request_key, producer_op_id, output_index, generation, max_bytes) =
                buffer(value.buffer(), "tensor_buffer")?;
            let product = ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Tensor,
                storage_class: StorageClass::DeviceTensor,
                dtype: dtype_from_fb(value.dtype())?,
                shape_bound: shape_bound_from_parts(value.extents(), value.dynamic_axis())?,
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            };
            codec_ensure!(
                product.max_bytes() == max_bytes,
                "tensor buffer byte bound disagrees with its typed shape"
            );
            product
        }
        fbs::ValueReference::KvBuffer => {
            let value = reference
                .value_as_kv_buffer()
                .context("KV buffer reference is missing")?;
            let (request_key, producer_op_id, output_index, generation, max_bytes) =
                buffer(value.buffer(), "kv_buffer")?;
            let product = ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Kv,
                storage_class: StorageClass::PagedKv,
                dtype: dtype_from_fb(value.dtype())?,
                shape_bound: shape_bound_from_parts(value.extents(), value.dynamic_axis())?,
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            };
            codec_ensure!(
                product.max_bytes() == max_bytes,
                "KV buffer byte bound disagrees with its typed shape"
            );
            product
        }
        fbs::ValueReference::LatentBuffer => {
            let value = reference
                .value_as_latent_buffer()
                .context("latent buffer reference is missing")?;
            let (request_key, producer_op_id, output_index, generation, max_bytes) =
                buffer(value.buffer(), "latent_buffer")?;
            let product = ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Latent,
                storage_class: StorageClass::LatentArena,
                dtype: dtype_from_fb(value.dtype())?,
                shape_bound: shape_bound_from_parts(value.extents(), value.dynamic_axis())?,
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            };
            codec_ensure!(
                product.max_bytes() == max_bytes,
                "latent buffer byte bound disagrees with its typed shape"
            );
            product
        }
        // Artifact use selects inline, feedback, or encoded-host storage.
        fbs::ValueReference::ArtifactValue => {
            let value = reference
                .value_as_artifact_value()
                .context("artifact value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "artifact_value")?;
            let storage_class = match value.use_() {
                fbs::ArtifactUse::Inline => StorageClass::HostStaging,
                fbs::ArtifactUse::Feedback => StorageClass::LatentArena,
                fbs::ArtifactUse::Encoded => StorageClass::PinnedOutput,
                _ => codec_bail!("unknown artifact use"),
            };
            let product = ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Artifact,
                storage_class,
                dtype: dtype_from_fb(value.dtype())?,
                shape_bound: shape_bound_from_parts(value.extents(), value.dynamic_axis())?,
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            };
            if value.use_() == fbs::ArtifactUse::Feedback {
                let (_, _, _, _, max_bytes) = buffer(value.buffer(), "artifact_value")?;
                codec_ensure!(
                    product.max_bytes() == max_bytes,
                    "artifact buffer byte bound disagrees with its typed shape"
                );
            }
            product
        }
        // Control products use fixed scalar representations and delivery rules.
        fbs::ValueReference::CompletionValue => {
            let value = reference
                .value_as_completion_value()
                .context("completion value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "completion_value")?;
            ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::Completion,
                storage_class: match value.delivery() {
                    fbs::CompletionDelivery::Relay => StorageClass::RequestRelay,
                    fbs::CompletionDelivery::DevicePredicate => StorageClass::DeviceTensor,
                    _ => codec_bail!("unknown completion delivery"),
                },
                dtype: DType::U8,
                shape_bound: ShapeBound::default(),
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            }
        }
        fbs::ValueReference::SamplingValue => {
            let value = reference
                .value_as_sampling_value()
                .context("sampling value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "sampling_value")?;
            let max_bytes = value.max_bytes();
            codec_ensure!(
                max_bytes > 0 && max_bytes <= u64::from(u32::MAX),
                "sampling value has an invalid byte bound"
            );
            ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::SamplingState,
                storage_class: StorageClass::HostStaging,
                dtype: DType::U8,
                shape_bound: ShapeBound {
                    dims: vec![DimBound::Device {
                        max: max_bytes as u32,
                    }],
                },
                point_range: PointRange::default(),
            }
        }
        fbs::ValueReference::SelectedPointValue => {
            let value = reference
                .value_as_selected_point_value()
                .context("selected-point value reference is missing")?;
            let (request_key, producer_op_id, output_index, generation) =
                scalar(value.id(), "selected_point_value")?;
            ProductRef {
                request_key,
                producer_op_id,
                output_index,
                generation,
                kind: ProductKind::SelectedPoint,
                storage_class: StorageClass::RequestRelay,
                dtype: DType::U32,
                shape_bound: ShapeBound::default(),
                point_range: PointRange {
                    base_point: value.base_point(),
                    max_points: value.max_points(),
                },
            }
        }
        _ => codec_bail!("value reference union is empty"),
    };

    // Variant decoding establishes shape; the shared validator enforces the
    // cross-variant identity and storage-class contract.
    product.validate()?;
    Ok(product)
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
fn run_result_from_table(report: fbs::RunResult<'_>) -> CodecResult<RunResult> {
    // Completions and products are independent ordered streams whose identities
    // are reconciled by `RunResult::validate` after both are materialized.
    let report = RunResult {
        batch_id: report.batch_id(),
        run_id: report.run_id(),
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
                    .map(product_payload_from_table)
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
        done: report.done(),
    };
    report.validate()?;
    Ok(report)
}

/// Decodes a model output union and validates its status-dependent result data.
fn completion_record_from_table(record: fbs::ModelOutput<'_>) -> CodecResult<ModelOutput> {
    // Every result-family table shares the same accounting shape; the macro
    // decodes that shape while preserving FlatBuffer presence errors.
    macro_rules! result_data {
        ($payload:expr) => {{
            let payload = $payload.context("completion result payload table is missing")?;
            let logical_lengths = payload
                .logical_lengths()
                .context("completion result has no logical lengths")?;
            let token_span = payload
                .token_span()
                .context("completion result has no token span")?;
            let finish_flags = payload
                .finish_flags()
                .context("completion result has no finish flags")?;
            let media_output = payload
                .media_output()
                .map(|output| {
                    let handle = match output.handle_type() {
                        fbs::ArtifactHandle::PosixShmArtifact => {
                            let handle = output.handle_as_posix_shm_artifact().context(
                                "completion media output POSIX shared-memory handle is missing",
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
            ResultData {
                logical_lengths: LogicalLengths {
                    token_len: logical_lengths.token_len(),
                    kv_visible_len: logical_lengths.kv_visible_len(),
                    kv_computed_len: logical_lengths.kv_computed_len(),
                    latent_len: logical_lengths.latent_len(),
                },
                token_span: TokenSpan {
                    base: token_span.base(),
                    len: token_span.len(),
                },
                committed_tokens: payload
                    .committed_tokens()
                    .map(|items| items.iter().collect())
                    .unwrap_or_default(),
                finish_flags: FinishFlags {
                    eos: finish_flags.eos(),
                    length: finish_flags.length(),
                    stop: finish_flags.stop(),
                },
                media_output,
            }
        }};
    }

    // Decode the tagged result union before assembling the common envelope.
    let payload = match record.payload_type() {
        fbs::ResultPayload::ArResult => {
            ResultPayload::Ar(result_data!(record.payload_as_ar_result()))
        }
        fbs::ResultPayload::EncoderResult => {
            ResultPayload::Encoder(result_data!(record.payload_as_encoder_result()))
        }
        fbs::ResultPayload::DiffusionResult => {
            let table = record
                .payload_as_diffusion_result()
                .context("completion diffusion result table is missing")?;
            ResultPayload::Diffusion(DiffusionResult {
                data: result_data!(Some(table)),
                next_cursor: table.next_cursor(),
                done: table.done(),
            })
        }
        fbs::ResultPayload::TransferResult => {
            ResultPayload::Transfer(result_data!(record.payload_as_transfer_result()))
        }
        _ => return Err(CodecError::invalid("completion result payload is missing")),
    };

    let timing_counters = record
        .timing_counters()
        .context("completion record has no timing counters")?;

    let record = ModelOutput {
        request_key: request_key_from_table(record.request_key(), "completion.request_key")?,
        op_id: OpId(record.op_id()),
        completion_slot_generation: record.completion_slot_generation(),
        status: op_status_from_fb(record.status())?,
        selected_point: record.selected_point(),
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
        payload,
    };

    // Domain validation applies cross-field status and generation invariants
    // after the wire union has been fully materialized.
    record.validate()?;
    Ok(record)
}

/// Decodes an inline-byte or external-transfer product payload.
fn product_payload_from_table(payload: fbs::ProductPayload<'_>) -> CodecResult<ProductPayload> {
    let value = match payload.value_type() {
        fbs::InlineValue::BytesValue => {
            let value = payload
                .value_as_bytes_value()
                .context("product payload byte value is missing")?;
            InlineValue::Bytes(
                value
                    .bytes()
                    .map(|bytes| bytes.bytes().to_vec())
                    .unwrap_or_default(),
            )
        }
        fbs::InlineValue::TransferHandle => {
            let value = payload
                .value_as_transfer_handle()
                .context("product payload transfer handle is missing")?;
            InlineValue::Transfer(transfer_handle_from_table(value)?)
        }
        _ => codec_bail!("product payload value union is empty"),
    };
    let payload = ProductPayload {
        product: product_ref_from_table(
            payload
                .product()
                .context("product payload has no product reference")?,
        )?,
        value,
    };
    payload.validate()?;
    Ok(payload)
}

/// Decodes the request and operation identity attached to a worker error.
fn error_operation_from_table(
    operation: fbs::ErrorOperationIdentity<'_>,
) -> CodecResult<ErrorOperationIdentity> {
    Ok(ErrorOperationIdentity {
        request_key: request_key_from_table(
            operation.request_key(),
            "error operation.request_key",
        )?,
        op_id: OpId(operation.op_id()),
    })
}

/// Decodes worker capabilities and validates the advertised resource geometry.
fn info_from_table(info: fbs::WorkerInfo<'_>) -> CodecResult<WorkerInfo> {
    let info = WorkerInfo {
        model_name: required_str(info.model_name(), "info.model_name")?,
        endpoint: endpoint_from_table(info.endpoint().context("info has no endpoint")?)?,
        device: required_str(info.device(), "info.device")?,
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
                            config: uniserve_core::EntryConfig {
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
                                    Ok(crate::TensorSpec {
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
                    .map(op_code_from_fb)
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
        max_unresolved_ops: info.max_unresolved_ops(),
        media_plan: info
            .media_plan()
            .map(|plan| -> CodecResult<MediaExecutionPlan> {
                Ok(MediaExecutionPlan {
                    stages: plan
                        .stages()
                        .context("media plan has no stages")?
                        .iter()
                        .map(|stage| {
                            Ok(MediaPlanStage {
                                name: required_str(stage.name(), "media stage.name")?,
                                operation: op_code_from_fb(stage.operation())?,
                                entry: required_str(stage.entry(), "media stage.entry")?,
                                dependencies: stage
                                    .dependencies()
                                    .map(|items| items.iter().map(str::to_owned).collect())
                                    .unwrap_or_default(),
                                input_from: stage
                                    .input_from()
                                    .filter(|value| !value.is_empty())
                                    .map(str::to_owned),
                                repeat: match stage.repeat() {
                                    fbs::MediaPlanRepeat::Once => MediaPlanRepeat::Once,
                                    fbs::MediaPlanRepeat::Fixed => MediaPlanRepeat::Fixed,
                                    fbs::MediaPlanRepeat::VideoUnits => MediaPlanRepeat::VideoUnits,
                                    _ => codec_bail!("media stage has an unknown repeat rule"),
                                },
                                count: stage.count(),
                            })
                        })
                        .collect::<CodecResult<Vec<_>>>()?,
                })
            })
            .transpose()?,
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
fn forward_stats_from_table(stats: fbs::WorkerForwardStats<'_>) -> WorkerForwardStats {
    WorkerForwardStats {
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
    let (run, run_id) = match request {
        WorkerRequest::Info { .. } | WorkerRequest::Close { .. } => (None, None),
        WorkerRequest::Submit { run, .. } => {
            run.validate()?;
            (Some(run), None)
        }
        WorkerRequest::Poll { run_id, .. } => (None, Some(*run_id)),
    };
    Ok(fbs::WorkerRequestT {
        kind: request_kind_to_fb(request.kind()),
        call_id: request.call_id(),
        run: run.map(run_to_fb).transpose()?.map(Box::new),
        poll_run_id: run_id,
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
        call_id: response.call_id(),
        info: info.map(info_to_fb).transpose()?.map(Box::new),
        result: result.map(run_result_to_fb).transpose()?.map(Box::new),
        message: error.map(|error| error.message.clone()),
        code: error.and_then(|error| error.code.clone()),
        retryable: error.map(|error| error.retryable),
        fatal: error.map(|error| error.fatal),
        phase: error.and_then(|error| error.phase.clone()),
        route: error.and_then(|error| error.route.clone()),
        operations: Some(
            error
                .into_iter()
                .flat_map(|error| &error.operations)
                .map(error_operation_to_fb)
                .collect(),
        ),
    })
}

fn run_to_fb(run: &Run) -> CodecResult<fbs::RunT> {
    run.validate()?;

    Ok(fbs::RunT {
        batch_id: run.batch_id,
        run_id: run.run_id,
        collective_seq: run.collective_seq,

        // Preserve executable graph order in the serialized vectors.
        operations: Some(
            run.operations
                .iter()
                .map(operation_to_fb)
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
        forward_rows: Some(run.forward_rows.iter().map(row_geometry_to_fb).collect()),
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
        input_products: Some(
            run.input_products
                .iter()
                .map(product_payload_to_fb)
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
fn diffusion_params_to_fb(admission: &DiffusionRequestParams) -> fbs::DiffusionRequestParamsT {
    fbs::DiffusionRequestParamsT {
        prompt_token_ids: Some(admission.prompt_token_ids.clone()),
        seed: admission.seed,
        geometry: Some(Box::new(fbs::MediaGeometryT {
            frame_count: admission.geometry.frame_count,
            video_units: admission.geometry.video_units,
            prompt_tokens: admission.geometry.prompt_tokens,
            denoise_steps: admission.geometry.denoise_steps,
        })),
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

/// Converts one forward row's operation and sequence geometry.
fn row_geometry_to_fb(row: &RowGeometry) -> fbs::RowGeometryT {
    fbs::RowGeometryT {
        operation_index: row.operation_index,
        request_pool_index: row.request_pool_index,
        seq_len: row.seq_len,
        query_len: row.query_len,
        write_kv: row.write_kv,
    }
}

/// Converts a latent-page params into its wire table.
fn latent_params_to_fb(params: &LatentParams) -> fbs::LatentParamsT {
    fbs::LatentParamsT {
        request_key: Some(Box::new(request_key_to_fb(params.request_key))),
        op_id: params.op_id.0,
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
        op_id: params.op_id.0,
        track: match params.track {
            MediaTrack::Video => fbs::MediaTrack::Video,
            MediaTrack::Audio => fbs::MediaTrack::Audio,
        },
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

/// Converts a validated operation into its family-specific payload union.
fn operation_to_fb(operation: &Operation) -> CodecResult<fbs::OperationT> {
    operation.validate()?;
    let payload = &operation.payload;
    Ok(fbs::OperationT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
        parent: operation
            .parent
            .as_ref()
            .map(|parent| Box::new(checkpoint_to_fb(parent))),
        entry: operation.entry.clone(),
        payload: Box::new(fbs::OpPayloadT {
            code: op_code_to_fb(payload.code),
            max_points: payload.bounds.max_points,
            max_tokens: payload.bounds.max_tokens,
            max_kv_pages: payload.bounds.max_kv_pages,
            max_latent_bytes: payload.bounds.max_latent_bytes,
            max_completion_bytes: payload.bounds.max_completion_bytes,
            max_transfer_bytes: payload.bounds.max_transfer_bytes,
            inputs: Some(
                payload
                    .inputs
                    .iter()
                    .map(product_ref_to_fb)
                    .collect::<CodecResult<_>>()?,
            ),
            outputs: Some(
                payload
                    .outputs
                    .iter()
                    .map(product_ref_to_fb)
                    .collect::<CodecResult<_>>()?,
            ),
            predicate: payload
                .predicate
                .as_ref()
                .map(product_ref_to_fb)
                .transpose()?
                .map(Box::new),
            rng: payload.rng.as_ref().map(rng_to_fb).map(Box::new),
            control_seq: payload.control_seq,
        }),
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
        BatchCommand::Commit {
            request_key,
            control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition,
        } => fbs::BatchCommandT::CommitCommand(Box::new(fbs::CommitCommandT {
            request_key: Some(Box::new(request_key_to_fb(*request_key))),
            control_seq: *control_seq,
            expected_parent: Some(Box::new(checkpoint_to_fb(expected_parent))),
            selected: Some(Box::new(checkpoint_to_fb(selected))),
            public_event_limit: *public_event_limit,
            disposition: disposition_to_fb(*disposition),
        })),
        BatchCommand::Finish {
            request_key,
            control_seq,
            cutoff,
            reason,
            retained_buffers,
        } => fbs::BatchCommandT::FinishCommand(Box::new(fbs::FinishCommandT {
            request_key: Some(Box::new(request_key_to_fb(*request_key))),
            control_seq: *control_seq,
            cutoff: Some(Box::new(checkpoint_to_fb(cutoff))),
            reason: close_reason_to_fb(*reason),
            retained_buffers: Some(
                retained_buffers
                    .iter()
                    .copied()
                    .map(buffer_id_to_fb)
                    .collect(),
            ),
        })),
        BatchCommand::Retire {
            request_key,
            retained_buffers,
        } => fbs::BatchCommandT::RetireCommand(Box::new(fbs::RetireCommandT {
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

/// Converts a request identity into its FlatBuffers object representation.
fn request_key_to_fb(request_key: RequestKey) -> fbs::RequestKeyT {
    fbs::RequestKeyT {
        authority_id: request_key.authority_id,
        request_id: request_key.request_id.0,
        epoch: request_key.epoch,
    }
}

/// Decodes and validates a persistent-buffer identity.
fn buffer_id_from_table(buffer: fbs::BufferId<'_>) -> CodecResult<BufferId> {
    let id = BufferId {
        owner: request_key_from_table(buffer.owner(), "buffer_id.owner")?,
        producer_op_id: OpId(buffer.producer_op_id()),
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
        producer_op_id: buffer.producer_op_id.0,
        output_index: buffer.output_index,
        generation: buffer.generation,
    }
}

/// Converts a checkpoint into its fixed or device-selected point union.
fn checkpoint_to_fb(version: &Checkpoint) -> fbs::CheckpointT {
    fbs::CheckpointT {
        op_id: version.op_id.0,
        point: match &version.point {
            CheckpointPoint::Fixed(point) => {
                fbs::CheckpointPointT::CheckpointFixed(Box::new(fbs::CheckpointFixedT {
                    point: *point,
                }))
            }
            CheckpointPoint::DeviceSelected => fbs::CheckpointPointT::CheckpointDeviceSelected(
                Box::new(fbs::CheckpointDeviceSelectedT {}),
            ),
        },
    }
}

/// Extracts a scalar product identity into its wire table.
fn scalar_id_to_fb(product: &ProductRef) -> fbs::ScalarIdT {
    fbs::ScalarIdT {
        request_key: Some(Box::new(request_key_to_fb(product.request_key))),
        producer_op_id: product.producer_op_id.0,
        output_index: product.output_index,
        generation: product.generation,
    }
}

/// Extracts a buffer identity and computed byte bound from a product reference.
fn buffer_descriptor_to_fb(product: &ProductRef) -> fbs::BufferDescriptorT {
    fbs::BufferDescriptorT {
        id: Some(Box::new(buffer_id_to_fb(product.buffer_id()))),
        max_bytes: product.max_bytes(),
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

/// Converts a validated product reference into its kind-specific wire union.
fn product_ref_to_fb(product: &ProductRef) -> CodecResult<fbs::ValueRefT> {
    product.validate()?;

    // Compute common identity, point, and shape metadata once before lowering
    // into the product-kind-specific FlatBuffer table.
    let scalar_id = || Some(Box::new(scalar_id_to_fb(product)));
    let point = &product.point_range;
    let (extents, dynamic_axis) = shape_bound_to_parts(&product.shape_bound);
    let value = match product.kind {
        // Scalar and host-relay products encode their delivery policy directly.
        ProductKind::Token => {
            codec_ensure!(
                product.dtype == DType::U32,
                "token value must use u32 elements"
            );
            let delivery = match product.storage_class {
                StorageClass::HostStaging => fbs::TokenDelivery::Inline,
                StorageClass::RequestRelay => fbs::TokenDelivery::Relay,
                _ => codec_bail!("token value has an invalid delivery"),
            };
            let max_tokens = u32::try_from(product.shape_bound.max_elements())
                .context("token value bound exceeds u32")?;
            fbs::ValueReferenceT::TokenValue(Box::new(fbs::TokenValueT {
                id: scalar_id(),
                delivery,
                max_tokens,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        ProductKind::Logprob => {
            codec_ensure!(
                matches!(product.dtype, DType::U8 | DType::F32),
                "logprob value has an unsupported format"
            );
            codec_ensure!(
                matches!(
                    product.storage_class,
                    StorageClass::HostStaging | StorageClass::PinnedOutput
                ),
                "logprob value must be host-visible"
            );
            let max_count = u32::try_from(product.shape_bound.max_elements())
                .context("logprob value bound exceeds u32")?;
            fbs::ValueReferenceT::LogprobValue(Box::new(fbs::LogprobValueT {
                id: scalar_id(),
                format: if product.dtype == DType::U8 {
                    fbs::LogprobFormat::PackedBytes
                } else {
                    fbs::LogprobFormat::F32Values
                },
                max_count,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        // Persistent tensors carry a buffer descriptor whose byte bound must
        // agree with the typed shape validated above.
        ProductKind::VisionFeature | ProductKind::LatentFeature => {
            codec_ensure!(
                product.storage_class == StorageClass::LatentArena,
                "feature value must use a persistent device buffer"
            );
            fbs::ValueReferenceT::FeatureBuffer(Box::new(fbs::FeatureBufferT {
                buffer: Some(Box::new(buffer_descriptor_to_fb(product))),
                feature: if product.kind == ProductKind::VisionFeature {
                    fbs::FeatureKind::Vision
                } else {
                    fbs::FeatureKind::Latent
                },
                dtype: dtype_to_fb(product.dtype),
                extents: Some(extents),
                dynamic_axis,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        ProductKind::Tensor => fbs::ValueReferenceT::TensorBuffer(Box::new(fbs::TensorBufferT {
            buffer: Some(Box::new(buffer_descriptor_to_fb(product))),
            dtype: dtype_to_fb(product.dtype),
            extents: Some(extents),
            dynamic_axis,
            base_point: point.base_point,
            max_points: point.max_points,
        })),
        ProductKind::Kv => {
            codec_ensure!(
                product.storage_class == StorageClass::PagedKv,
                "KV value must use paged KV storage"
            );
            fbs::ValueReferenceT::KvBuffer(Box::new(fbs::KvBufferT {
                buffer: Some(Box::new(buffer_descriptor_to_fb(product))),
                dtype: dtype_to_fb(product.dtype),
                extents: Some(extents),
                dynamic_axis,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        ProductKind::Latent => {
            codec_ensure!(
                product.storage_class == StorageClass::LatentArena,
                "latent value must use the latent arena"
            );
            fbs::ValueReferenceT::LatentBuffer(Box::new(fbs::LatentBufferT {
                buffer: Some(Box::new(buffer_descriptor_to_fb(product))),
                dtype: dtype_to_fb(product.dtype),
                extents: Some(extents),
                dynamic_axis,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        // Artifact storage determines whether the wire value is inline,
        // feedback-resident, or an encoded host output.
        ProductKind::Artifact => {
            let use_ = match product.storage_class {
                StorageClass::HostStaging => fbs::ArtifactUse::Inline,
                StorageClass::LatentArena => fbs::ArtifactUse::Feedback,
                StorageClass::PinnedOutput => fbs::ArtifactUse::Encoded,
                _ => codec_bail!("artifact value has an invalid use"),
            };
            fbs::ValueReferenceT::ArtifactValue(Box::new(fbs::ArtifactValueT {
                id: scalar_id(),
                buffer: (use_ == fbs::ArtifactUse::Feedback)
                    .then(|| Box::new(buffer_descriptor_to_fb(product))),
                use_,
                dtype: dtype_to_fb(product.dtype),
                extents: Some(extents),
                dynamic_axis,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        // Control products use fixed scalar representations.
        ProductKind::Completion => {
            codec_ensure!(product.dtype == DType::U8, "completion value must use u8");
            let delivery = match product.storage_class {
                StorageClass::RequestRelay => fbs::CompletionDelivery::Relay,
                StorageClass::DeviceTensor => fbs::CompletionDelivery::DevicePredicate,
                _ => codec_bail!("completion value has an invalid delivery"),
            };
            fbs::ValueReferenceT::CompletionValue(Box::new(fbs::CompletionValueT {
                id: scalar_id(),
                delivery,
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
        ProductKind::SamplingState => {
            codec_ensure!(
                product.dtype == DType::U8 && product.storage_class == StorageClass::HostStaging,
                "sampling value must be inline bytes"
            );
            fbs::ValueReferenceT::SamplingValue(Box::new(fbs::SamplingValueT {
                id: scalar_id(),
                max_bytes: product.max_bytes(),
            }))
        }
        ProductKind::SelectedPoint => {
            codec_ensure!(
                product.dtype == DType::U32 && product.storage_class == StorageClass::RequestRelay,
                "selected point must use the request relay"
            );
            fbs::ValueReferenceT::SelectedPointValue(Box::new(fbs::SelectedPointValueT {
                id: scalar_id(),
                base_point: point.base_point,
                max_points: point.max_points,
            }))
        }
    };

    Ok(fbs::ValueRefT { value })
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
fn run_result_to_fb(report: &RunResult) -> CodecResult<fbs::RunResultT> {
    report.validate()?;
    Ok(fbs::RunResultT {
        batch_id: report.batch_id,
        run_id: report.run_id,
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
                .map(product_payload_to_fb)
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
        done: report.done,
    })
}

/// Converts a model output into its family-specific result union.
fn completion_record_to_fb(record: &ModelOutput) -> fbs::ModelOutputT {
    // Common accounting fields are embedded in each family table so the wire
    // union remains self-contained after dispatch.
    macro_rules! result_fields {
        ($table:ident, $data:expr) => {
            fbs::$table {
                logical_lengths: Some(Box::new(fbs::LogicalLengthsT {
                    token_len: $data.logical_lengths.token_len,
                    kv_visible_len: $data.logical_lengths.kv_visible_len,
                    kv_computed_len: $data.logical_lengths.kv_computed_len,
                    latent_len: $data.logical_lengths.latent_len,
                })),
                token_span: Some(Box::new(fbs::TokenSpanT {
                    base: $data.token_span.base,
                    len: $data.token_span.len,
                })),
                committed_tokens: Some($data.committed_tokens.clone()),
                finish_flags: Some(Box::new(fbs::FinishFlagsT {
                    eos: $data.finish_flags.eos,
                    length: $data.finish_flags.length,
                    stop: $data.finish_flags.stop,
                })),
                media_output: $data
                    .media_output
                    .as_ref()
                    .map(media_output_to_fb)
                    .map(Box::new),
                ..Default::default()
            }
        };
    }

    let payload = match &record.payload {
        ResultPayload::Ar(data) => {
            fbs::ResultPayloadT::ArResult(Box::new(result_fields!(ArResultT, data)))
        }
        ResultPayload::Encoder(data) => {
            fbs::ResultPayloadT::EncoderResult(Box::new(result_fields!(EncoderResultT, data)))
        }
        ResultPayload::Diffusion(result) => {
            let mut table = result_fields!(DiffusionResultT, &result.data);
            table.next_cursor = result.next_cursor;
            table.done = result.done;
            fbs::ResultPayloadT::DiffusionResult(Box::new(table))
        }
        ResultPayload::Transfer(data) => {
            fbs::ResultPayloadT::TransferResult(Box::new(result_fields!(TransferResultT, data)))
        }
    };

    // Identity, terminal status, and timing wrap the selected result payload.
    fbs::ModelOutputT {
        request_key: Some(Box::new(request_key_to_fb(record.request_key))),
        op_id: record.op_id.0,
        completion_slot_generation: record.completion_slot_generation,
        status: op_status_to_fb(record.status),
        selected_point: record.selected_point,
        product_generations: Some(record.product_generations.clone()),
        error_code: record.error_code.map(error_code_to_fb),
        timing_counters: Some(Box::new(fbs::TimingCountersT {
            queued_us: record.timing_counters.queued_us,
            device_us: record.timing_counters.device_us,
            copy_us: record.timing_counters.copy_us,
            host_us: record.timing_counters.host_us,
        })),
        payload,
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

/// Converts a product value into its inline-byte or transfer-handle union.
fn product_payload_to_fb(payload: &ProductPayload) -> CodecResult<fbs::ProductPayloadT> {
    Ok(fbs::ProductPayloadT {
        product: Some(Box::new(product_ref_to_fb(&payload.product)?)),
        value: match &payload.value {
            InlineValue::Bytes(bytes) => {
                fbs::InlineValueT::BytesValue(Box::new(fbs::BytesValueT {
                    bytes: Some(bytes.clone()),
                }))
            }
            InlineValue::Transfer(handle) => {
                fbs::InlineValueT::TransferHandle(Box::new(transfer_handle_to_fb(handle)))
            }
        },
    })
}

/// Converts an error's request and operation identity into its wire table.
fn error_operation_to_fb(operation: &ErrorOperationIdentity) -> fbs::ErrorOperationIdentityT {
    fbs::ErrorOperationIdentityT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
    }
}

/// Decodes worker KV-cache capabilities from a verified table.
fn kv_cache_from_table(config: fbs::KvCacheConfig<'_>) -> CodecResult<KvCacheConfig> {
    Ok(KvCacheConfig {
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
fn kv_cache_to_fb(config: &KvCacheConfig) -> fbs::KvCacheConfigT {
    fbs::KvCacheConfigT {
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
                                    fbs::TensorSpecT {
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
                .map(op_code_to_fb)
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
        max_unresolved_ops: info.max_unresolved_ops,
        media_plan: info.media_plan.as_ref().map(|plan| {
            Box::new(fbs::MediaExecutionPlanT {
                stages: Some(
                    plan.stages
                        .iter()
                        .map(|stage| fbs::MediaPlanStageT {
                            name: Some(stage.name.clone()),
                            operation: op_code_to_fb(stage.operation),
                            entry: Some(stage.entry.clone()),
                            dependencies: Some(stage.dependencies.clone()),
                            input_from: Some(stage.input_from.clone().unwrap_or_default()),
                            repeat: match stage.repeat {
                                MediaPlanRepeat::Once => fbs::MediaPlanRepeat::Once,
                                MediaPlanRepeat::Fixed => fbs::MediaPlanRepeat::Fixed,
                                MediaPlanRepeat::VideoUnits => fbs::MediaPlanRepeat::VideoUnits,
                            },
                            count: stage.count,
                        })
                        .collect(),
                ),
            })
        }),
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
fn forward_stats_to_fb(stats: &WorkerForwardStats) -> fbs::WorkerForwardStatsT {
    fbs::WorkerForwardStatsT {
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

/// Maps a physical run kind to its stable FlatBuffers discriminant.
fn op_code_to_fb(variant: OpCode) -> fbs::OpCode {
    match variant {
        OpCode::ArExtend => fbs::OpCode::ArExtend,
        OpCode::ArDecode => fbs::OpCode::ArDecode,
        OpCode::ArVerify => fbs::OpCode::ArVerify,
        OpCode::EncoderVision => fbs::OpCode::EncoderVision,
        OpCode::EncoderText => fbs::OpCode::EncoderText,
        OpCode::EncoderLatent => fbs::OpCode::EncoderLatent,
        OpCode::TransferProduct => fbs::OpCode::TransferProduct,
        OpCode::TransferKvPublish => fbs::OpCode::TransferKvPublish,
        OpCode::TransferKvInstall => fbs::OpCode::TransferKvInstall,
        OpCode::DiffusionPrepare => fbs::OpCode::DiffusionPrepare,
        OpCode::DiffusionStep => fbs::OpCode::DiffusionStep,
        OpCode::DiffusionFinalize => fbs::OpCode::DiffusionFinalize,
        OpCode::DiffusionDecode => fbs::OpCode::DiffusionDecode,
        OpCode::MediaAppend => fbs::OpCode::MediaAppend,
    }
}

/// Resolves a FlatBuffers run discriminant against the complete supported set.
fn op_code_from_fb(variant: fbs::OpCode) -> CodecResult<OpCode> {
    for candidate in OpCode::ALL {
        if op_code_to_fb(candidate) == variant {
            return Ok(candidate);
        }
    }
    codec_bail!("unknown physical run kind {}", variant.0)
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
    } else if value.transport() == fbs::TransferTransportKind::CudaIpc {
        TransferTransport::CudaIpc {
            endpoint: value
                .endpoint()
                .context("CUDA IPC endpoint is missing")?
                .to_owned(),
            publication_id: value
                .publication_id()
                .context("CUDA IPC publication identity is missing")?
                .to_owned(),
            storage_handle: value
                .storage_handle()
                .map(|bytes| bytes.bytes().to_vec())
                .unwrap_or_default(),
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
                generation: transfer.generation(),
                height: transfer.height(),
                width: transfer.width(),
                payload_kind: match transfer.payload_kind() {
                    fbs::FeatureKind::Vision => ProductKind::VisionFeature,
                    fbs::FeatureKind::Latent => ProductKind::LatentFeature,
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
                generation: transfer.generation(),
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

        fbs::TransferData::KvTransfer => {
            let transfer = value
                .value_as_kv_transfer()
                .context("KV transfer payload is missing")?;
            TransferHandle::Kv {
                generation: transfer.generation(),
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
                source: checkpoint_from_table(
                    transfer.source().context("KV transfer source is missing")?,
                )?,
                destination: transfer
                    .destination()
                    .context("KV transfer destination is missing")?
                    .to_owned(),
                base: transfer.base().map(checkpoint_from_table).transpose()?,
                base_extent: transfer.base_extent(),
                published_extent: transfer.published_extent(),
                group_id: transfer.group_id(),
                page_size: transfer.page_size(),
                compute_dtype: transfer
                    .compute_dtype()
                    .context("KV transfer compute dtype is missing")?
                    .to_owned(),
            }
        }

        fbs::TransferData::LatentTransfer => {
            let transfer = value
                .value_as_latent_transfer()
                .context("latent transfer payload is missing")?;
            TransferHandle::Latent {
                generation: transfer.generation(),
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
        TransferTransport::CudaIpc {
            endpoint,
            publication_id,
            storage_handle,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
        } => {
            output.transport = fbs::TransferTransportKind::CudaIpc;
            output.endpoint = Some(endpoint.clone());
            output.publication_id = Some(publication_id.clone());
            output.storage_handle = Some(storage_handle.clone());
            output.storage_size_bytes = *storage_size_bytes;
            output.storage_offsets_bytes = Some(storage_offsets_bytes.clone());
            output.span_lengths = Some(span_lengths.clone());
            output.span_counts = Some(span_counts.clone());
            output.tensor_stride = Some(tensor_stride.clone());
            output.ready_event_handle = Some(ready_event_handle.clone());
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
            generation,
            height,
            width,
            payload_kind,
            tensor,
        } => fbs::TransferDataT::EncoderTransfer(Box::new(fbs::EncoderTransferT {
            generation: *generation,
            height: *height,
            width: *width,
            payload_kind: match payload_kind {
                ProductKind::VisionFeature => fbs::FeatureKind::Vision,
                ProductKind::LatentFeature => fbs::FeatureKind::Latent,
                _ => unreachable!("validated encoder transfer feature kind"),
            },
            tensor: Some(Box::new(tensor_transfer_to_fb(tensor))),
        })),

        TransferHandle::DeviceProduct {
            generation,
            height,
            width,
            value_range,
            tensor,
        } => fbs::TransferDataT::DeviceProductTransfer(Box::new(fbs::DeviceProductTransferT {
            generation: *generation,
            height: *height,
            width: *width,
            value_range: Some(value_range.clone()),
            tensor: Some(Box::new(tensor_transfer_to_fb(tensor))),
        })),

        TransferHandle::Kv {
            generation,
            tensors,
            source,
            destination,
            base,
            base_extent,
            published_extent,
            group_id,
            compute_dtype,
            page_size,
        } => fbs::TransferDataT::KvTransfer(Box::new(fbs::KvTransferT {
            generation: *generation,
            tensors: Some(tensors.iter().map(tensor_transfer_to_fb).collect()),
            source: Some(Box::new(checkpoint_to_fb(source))),
            destination: Some(destination.clone()),
            base: base.as_ref().map(checkpoint_to_fb).map(Box::new),
            base_extent: *base_extent,
            published_extent: *published_extent,
            group_id: *group_id,
            compute_dtype: Some(compute_dtype.clone()),
            page_size: *page_size,
        })),

        TransferHandle::Latent {
            generation,
            height,
            width,
            latent_units,
            step,
            tensor,
        } => fbs::TransferDataT::LatentTransfer(Box::new(fbs::LatentTransferT {
            generation: *generation,
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
        DType::U16 => fbs::DType::U16,
        DType::U32 => fbs::DType::U32,
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
        fbs::DType::U16 => DType::U16,
        fbs::DType::U32 => DType::U32,
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

/// Maps an operation status to its stable FlatBuffers discriminant.
fn op_status_to_fb(status: OpStatus) -> fbs::OpStatus {
    match status {
        OpStatus::Ok => fbs::OpStatus::Ok,
        OpStatus::Predicated => fbs::OpStatus::Predicated,
        OpStatus::Error => fbs::OpStatus::Error,
    }
}

/// Decodes a supported FlatBuffers operation status.
fn op_status_from_fb(status: fbs::OpStatus) -> CodecResult<OpStatus> {
    Ok(match status {
        fbs::OpStatus::Ok => OpStatus::Ok,
        fbs::OpStatus::Predicated => OpStatus::Predicated,
        fbs::OpStatus::Error => OpStatus::Error,
        other => codec_bail!("unknown completion status {}", other.0),
    })
}

/// Maps a worker error code to its stable FlatBuffers discriminant.
fn error_code_to_fb(code: ErrorCode) -> fbs::ErrorCode {
    match code {
        ErrorCode::InvalidOperation => fbs::ErrorCode::InvalidOperation,
        ErrorCode::ResourceExhausted => fbs::ErrorCode::ResourceExhausted,
        ErrorCode::ComputeError => fbs::ErrorCode::ComputeError,
        ErrorCode::Cancelled => fbs::ErrorCode::Cancelled,
        ErrorCode::Internal => fbs::ErrorCode::Internal,
    }
}

/// Decodes a supported FlatBuffers worker error code.
fn error_code_from_fb(code: fbs::ErrorCode) -> CodecResult<ErrorCode> {
    Ok(match code {
        fbs::ErrorCode::InvalidOperation => ErrorCode::InvalidOperation,
        fbs::ErrorCode::ResourceExhausted => ErrorCode::ResourceExhausted,
        fbs::ErrorCode::ComputeError => ErrorCode::ComputeError,
        fbs::ErrorCode::Cancelled => ErrorCode::Cancelled,
        fbs::ErrorCode::Internal => ErrorCode::Internal,
        other => codec_bail!("unknown error code {}", other.0),
    })
}

/// Maps a checkpoint disposition to its stable FlatBuffers discriminant.
fn disposition_to_fb(disposition: Disposition) -> fbs::Disposition {
    match disposition {
        Disposition::Publish => fbs::Disposition::Publish,
        Disposition::Retain => fbs::Disposition::Retain,
        Disposition::Discard => fbs::Disposition::Discard,
    }
}

/// Decodes a supported FlatBuffers checkpoint disposition.
fn disposition_from_fb(disposition: fbs::Disposition) -> CodecResult<Disposition> {
    Ok(match disposition {
        fbs::Disposition::Publish => Disposition::Publish,
        fbs::Disposition::Retain => Disposition::Retain,
        fbs::Disposition::Discard => Disposition::Discard,
        other => codec_bail!("unknown disposition {}", other.0),
    })
}

/// Maps a request close reason to its stable FlatBuffers discriminant.
fn close_reason_to_fb(reason: CloseReason) -> fbs::CloseReason {
    match reason {
        CloseReason::Completed => fbs::CloseReason::Completed,
        CloseReason::Cancelled => fbs::CloseReason::Cancelled,
        CloseReason::Error => fbs::CloseReason::Error,
        CloseReason::Preempted => fbs::CloseReason::Preempted,
    }
}

/// Decodes a supported FlatBuffers request close reason.
fn close_reason_from_fb(reason: fbs::CloseReason) -> CodecResult<CloseReason> {
    Ok(match reason {
        fbs::CloseReason::Completed => CloseReason::Completed,
        fbs::CloseReason::Cancelled => CloseReason::Cancelled,
        fbs::CloseReason::Error => CloseReason::Error,
        fbs::CloseReason::Preempted => CloseReason::Preempted,
        other => codec_bail!("unknown close reason {}", other.0),
    })
}

/// Maps a request kind to its stable FlatBuffers discriminant.
fn request_kind_to_fb(kind: RequestKind) -> fbs::ReqKind {
    match kind {
        RequestKind::Info => fbs::ReqKind::Info,
        RequestKind::Submit => fbs::ReqKind::Submit,
        RequestKind::Poll => fbs::ReqKind::Poll,
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
