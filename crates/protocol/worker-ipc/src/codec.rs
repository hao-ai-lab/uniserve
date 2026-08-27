//! Hand-written FlatBuffers codec for the worker execution protocol.

use std::collections::BTreeMap;

use flatbuffers::FlatBufferBuilder;
use uniserve_core::{BlockId, KvCacheGroupSpec, KvGroupKind, RankInfo, RequestId, SamplingParams};

use crate::schema::uniserve::wire as fbs;
use crate::{
    Admission, AttentionRegime, Batch, BatchPartition, BlockTable, Bounds, CacheCopy,
    CachePageAllocation, CloseReason, CompletionReport, Control, DType, DecodeKind,
    DecodePlacement, DimBound, Disposition, Domain, DrawLayout, ErrorCode, ErrorOperationIdentity,
    ExecutionCapability, FinishFlags, ForwardMode, GenAdmission, GraphBucketCapability,
    LaneCapabilities, LatentPlacement, LogicalLengths, MediaAdmission, MediaProfileId,
    MixedExecutionCapability, ModelOutput, OpId, OpStatus, Operation, PartitionCompletion, Point,
    PointRange, ProductKind, ProductPayload, ProductRef, RecoveryPlacement, RegistrationAck,
    RequestKey, RequestKind, ResourceClass, ResourcePressure, ResponseKind, Rng, RouteId,
    RowGeometry, SamplingOwnership, ShapeBound, SnapshotRef, StorageClass, TimingCounters,
    TokenSpan, UndAdmission, VersionRef, WorkerCapabilities, WorkerForwardStats, WorkerRequest,
    WorkerResponse, WorkerResponseError,
};

pub type CodecResult<T> = std::result::Result<T, CodecError>;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("worker codec error: {0}")]
    Invalid(String),
    #[error(transparent)]
    Protocol(#[from] crate::ProtocolError),
}

impl CodecError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

trait CodecContext<T> {
    fn context(self, message: &str) -> CodecResult<T>;
    fn with_context<F, D>(self, message: F) -> CodecResult<T>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display;
}

impl<T> CodecContext<T> for Option<T> {
    fn context(self, message: &str) -> CodecResult<T> {
        self.ok_or_else(|| CodecError::invalid(message))
    }

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
    fn context(self, message: &str) -> CodecResult<T> {
        self.map_err(|error| CodecError::invalid(format!("{message}: {error}")))
    }

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

pub fn encode_request(request: &WorkerRequest) -> CodecResult<Vec<u8>> {
    let object = request_to_fb(request)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

pub fn decode_request(bytes: &[u8]) -> CodecResult<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_table(root)
}

pub fn encode_response(response: &WorkerResponse) -> CodecResult<Vec<u8>> {
    let object = response_to_fb(response)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

pub fn decode_response(bytes: &[u8]) -> CodecResult<WorkerResponse> {
    let root = flatbuffers::root::<fbs::WorkerResponse>(bytes)
        .context("invalid WorkerResponse flatbuffer")?;
    response_from_table(root)
}

// ---------------------------------------------------------------------------
// Verified FlatBuffer table decoding reads fields directly into canonical wire
// values, with one allocation per owned field and one copy per byte vector.
// ---------------------------------------------------------------------------

fn request_from_table(request: fbs::WorkerRequest<'_>) -> CodecResult<WorkerRequest> {
    let kind = request_kind_from_fb(request.kind())?;
    let call_id = request.call_id();
    let batch = request.batch().map(batch_from_table).transpose()?;
    let step_id = request.step_id();
    let session_id = request.session_id().map(RequestId);
    let copies: Option<Vec<CacheCopy>> = request.copies().map(|items| {
        items
            .iter()
            .map(|copy| CacheCopy {
                group_id: copy.group_id(),
                source_page: BlockId(copy.src()),
                destination_page: BlockId(copy.dst()),
            })
            .collect()
    });
    let product_handles = request
        .product_handles()
        .map(|items| items.iter().collect());
    let snapshot = request.snapshot().map(snapshot_from_table).transpose()?;
    let recovery_placement = request
        .recovery_placement()
        .map(recovery_placement_from_table)
        .transpose()?;
    let payload_count = usize::from(batch.is_some())
        + usize::from(step_id.is_some())
        + usize::from(session_id.is_some())
        + usize::from(copies.is_some())
        + usize::from(product_handles.is_some())
        + usize::from(snapshot.is_some())
        + usize::from(recovery_placement.is_some());
    Ok(match kind {
        RequestKind::GetCapabilities => {
            codec_ensure!(payload_count == 0, "get_capabilities carries a payload");
            WorkerRequest::GetCapabilities { call_id }
        }
        RequestKind::Execute => {
            codec_ensure!(payload_count == 1, "execute requires exactly one batch");
            WorkerRequest::Execute {
                call_id,
                batch: batch.context("execute request has no batch")?,
            }
        }
        RequestKind::PollCompletions => {
            codec_ensure!(payload_count == 1, "poll_completions requires one step id");
            WorkerRequest::PollCompletions {
                call_id,
                step_id: step_id.context("poll_completions has no step id")?,
            }
        }
        RequestKind::DropSession => {
            codec_ensure!(payload_count == 1, "drop_session requires one session id");
            WorkerRequest::DropSession {
                session_id: session_id.context("drop_session has no session id")?,
            }
        }
        RequestKind::CopyKv => {
            codec_ensure!(payload_count == 1, "copy_kv requires one copy list");
            let copies = copies.context("copy_kv has no copy list")?;
            for copy in &copies {
                copy.validate()?;
            }
            WorkerRequest::CopyKv { copies }
        }
        RequestKind::ReleaseProducts => {
            codec_ensure!(
                payload_count == 1,
                "release_products requires one handle list"
            );
            WorkerRequest::ReleaseProducts {
                product_handles: product_handles.context("release_products has no handle list")?,
            }
        }
        RequestKind::GetPressure => {
            codec_ensure!(payload_count == 0, "get_pressure carries a payload");
            WorkerRequest::GetPressure { call_id }
        }
        RequestKind::SnapshotSession => {
            codec_ensure!(
                payload_count == 1,
                "snapshot_session requires one placement"
            );
            let recovery_placement =
                recovery_placement.context("snapshot_session has no recovery placement")?;
            recovery_placement.validate()?;
            WorkerRequest::SnapshotSession { recovery_placement }
        }
        RequestKind::RestoreSession => {
            codec_ensure!(
                payload_count == 2,
                "restore_session requires snapshot and placement"
            );
            let snapshot = snapshot.context("restore_session has no snapshot")?;
            let recovery_placement =
                recovery_placement.context("restore_session has no recovery placement")?;
            snapshot.validate()?;
            recovery_placement.validate()?;
            WorkerRequest::RestoreSession {
                snapshot,
                recovery_placement,
            }
        }
        RequestKind::Shutdown => {
            codec_ensure!(payload_count == 0, "shutdown carries a payload");
            WorkerRequest::Shutdown
        }
    })
}

fn response_from_table(response: fbs::WorkerResponse<'_>) -> CodecResult<WorkerResponse> {
    let kind = response_kind_from_fb(response.kind())?;
    let call_id = response.call_id();
    let capabilities = response
        .capabilities()
        .map(capabilities_from_table)
        .transpose()?;
    let completion_report = response
        .completion_report()
        .map(completion_report_from_table)
        .transpose()?;
    let pressure = response
        .pressure()
        .map(|items| items.iter().map(pressure_from_table).collect())
        .transpose()?;
    let snapshot = response.snapshot().map(snapshot_from_table).transpose()?;
    let payload_count = usize::from(capabilities.is_some())
        + usize::from(completion_report.is_some())
        + usize::from(pressure.is_some())
        + usize::from(snapshot.is_some());
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
    Ok(match kind {
        ResponseKind::Capabilities => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid capabilities response"
            );
            WorkerResponse::Capabilities {
                call_id,
                capabilities: capabilities.context("capabilities response has no capabilities")?,
            }
        }
        ResponseKind::Result => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid result response"
            );
            WorkerResponse::Result {
                call_id,
                completion_report: completion_report.context("result response has no report")?,
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
        ResponseKind::Pressure => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid pressure response"
            );
            WorkerResponse::Pressure {
                call_id,
                pressure: pressure.context("pressure response has no pressure values")?,
            }
        }
        ResponseKind::Snapshot => {
            codec_ensure!(
                payload_count == 1 && !carries_error,
                "invalid snapshot response"
            );
            WorkerResponse::Snapshot {
                call_id,
                snapshot: snapshot.context("snapshot response has no snapshot")?,
            }
        }
    })
}

fn batch_from_table(batch: fbs::Batch<'_>) -> CodecResult<Batch> {
    let batch = Batch {
        step_id: batch.step_id(),
        admissions: batch
            .admissions()
            .map(|items| {
                items
                    .iter()
                    .map(admission_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        partitions: batch
            .partitions()
            .map(|items| {
                items
                    .iter()
                    .map(partition_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        controls: batch
            .controls()
            .map(|items| {
                items
                    .iter()
                    .map(control_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        input_products: batch
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
    batch.validate()?;
    Ok(batch)
}

fn partition_from_table(partition: fbs::BatchPartition<'_>) -> CodecResult<BatchPartition> {
    let partition = BatchPartition {
        partition_id: partition.partition_id(),
        submission_group: partition.submission_group(),
        collective_seq: partition.collective_seq(),
        domain: domain_from_fb(partition.domain())?,
        route: RouteId(partition.route()),
        execution: execution_capability_from_fb(partition.execution())?,
        attention: attention_regime_from_fb(partition.attention())?,
        shape_class: partition.shape_class(),
        operations: partition
            .operations()
            .map(|items| {
                items
                    .iter()
                    .map(operation_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        block_tables: partition
            .block_tables()
            .map(|items| items.iter().map(block_table_from_table).collect())
            .unwrap_or_default(),
        new_cache_pages: partition
            .new_cache_pages()
            .map(|items| items.iter().map(cache_page_allocation_from_table).collect())
            .unwrap_or_default(),
        forward_rows: partition
            .forward_rows()
            .map(|items| items.iter().map(row_geometry_from_table).collect())
            .unwrap_or_default(),
        latent_placements: partition
            .latent_placements()
            .map(|items| {
                items
                    .iter()
                    .map(latent_placement_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        decode_placements: partition
            .decode_placements()
            .map(|items| {
                items
                    .iter()
                    .map(decode_placement_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    partition.validate()?;
    Ok(partition)
}

fn admission_from_table(admission: fbs::Admission<'_>) -> CodecResult<Admission> {
    let admission = Admission {
        request_key: request_key_from_table(admission.request_key(), "admission.request_key")?,
        request_pool_idx: admission.request_pool_idx(),
        digest: required_digest(admission.digest(), "admission.digest")?,
        und: admission.und().map(und_admission_from_table).transpose()?,
        gen_admission: admission
            .gen_admission()
            .map(gen_admission_from_table)
            .transpose()?,
        media: admission
            .media()
            .map(media_admission_from_table)
            .transpose()?,
    };
    admission.validate()?;
    Ok(admission)
}

fn und_admission_from_table(admission: fbs::UndAdmission<'_>) -> CodecResult<UndAdmission> {
    Ok(UndAdmission {
        sampling: sampling_from_table(
            admission
                .sampling()
                .context("und admission has no sampling spec")?,
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

fn gen_admission_from_table(admission: fbs::GenAdmission<'_>) -> CodecResult<GenAdmission> {
    Ok(GenAdmission {
        image: image_from_table(
            admission
                .image()
                .context("gen admission has no image spec")?,
        )?,
    })
}

fn media_admission_from_table(admission: fbs::MediaAdmission<'_>) -> CodecResult<MediaAdmission> {
    Ok(MediaAdmission {
        prompt: required_str(admission.prompt(), "media admission.prompt")?,
        seed: admission.seed(),
        profile: media_profile_from_fb(admission.profile())?,
        output_path: required_str(admission.output_path(), "media admission.output_path")?,
    })
}

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

fn row_geometry_from_table(row: fbs::RowGeometry<'_>) -> RowGeometry {
    RowGeometry {
        operation_index: row.operation_index(),
        request_pool_index: row.request_pool_index(),
        seq_len: row.seq_len(),
        query_len: row.query_len(),
    }
}

fn latent_placement_from_table(
    placement: fbs::LatentPlacement<'_>,
) -> CodecResult<LatentPlacement> {
    Ok(LatentPlacement {
        request_key: request_key_from_table(
            placement.request_key(),
            "latent placement.request_key",
        )?,
        op_id: OpId(placement.op_id()),
        page_table: placement
            .page_table()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        latent_units: placement.latent_units(),
        height: placement.height(),
        width: placement.width(),
        start_step: placement.start_step(),
        step_count: placement.step_count(),
    })
}

fn decode_placement_from_table(
    placement: fbs::DecodePlacement<'_>,
) -> CodecResult<DecodePlacement> {
    Ok(DecodePlacement {
        request_key: request_key_from_table(
            placement.request_key(),
            "decode placement.request_key",
        )?,
        op_id: OpId(placement.op_id()),
        kind: decode_kind_from_fb(placement.kind())?,
        start_unit: placement.start_unit(),
        unit_count: placement.unit_count(),
    })
}

fn operation_from_table(operation: fbs::Operation<'_>) -> CodecResult<Operation> {
    let operation = Operation {
        request_key: request_key_from_table(operation.request_key(), "operation.request_key")?,
        op_id: OpId(operation.op_id()),
        parent: version_ref_from_table(operation.parent().context("operation has no parent")?)?,
        work: work_from_fb(operation.work())?,
        route: RouteId(operation.route()),
        domain: domain_from_fb(operation.domain())?,
        advances_state: operation.advances_state(),
        bounds: bounds_from_table(operation.bounds().context("operation has no bounds")?),
        inputs: operation
            .inputs()
            .map(|items| {
                items
                    .iter()
                    .map(product_ref_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        outputs: operation
            .outputs()
            .map(|items| {
                items
                    .iter()
                    .map(product_ref_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        predicate: operation
            .predicate()
            .map(product_ref_from_table)
            .transpose()?,
        rng: operation.rng().map(rng_from_table).transpose()?,
        control_seq: operation.control_seq(),
        plan_digest: required_digest(operation.plan_digest(), "operation.plan_digest")?,
    };
    // No per-operation validate() here: the worker's Python `from_mapping` is the
    // authoritative ingress validator and recomputes the plan digest for every
    // operation (forged-digest rejection unchanged); running the SHA-256
    // recompute here too made the wire decode do the same work twice per op.
    Ok(operation)
}

fn control_from_table(envelope: fbs::ControlEnvelope<'_>) -> CodecResult<Control> {
    let control = match envelope.control_type() {
        fbs::Control::ControlCommit => {
            let commit = envelope
                .control_as_control_commit()
                .context("commit control table is missing")?;
            Control::Commit {
                request_key: request_key_from_table(
                    commit.request_key(),
                    "control.commit.request_key",
                )?,
                control_seq: commit.control_seq(),
                expected_parent: version_ref_from_table(
                    commit
                        .expected_parent()
                        .context("commit control has no expected parent")?,
                )?,
                selected: version_ref_from_table(
                    commit
                        .selected()
                        .context("commit control has no selected version")?,
                )?,
                public_event_limit: commit.public_event_limit(),
                disposition: disposition_from_fb(commit.disposition())?,
            }
        }
        fbs::Control::ControlClose => {
            let close = envelope
                .control_as_control_close()
                .context("close control table is missing")?;
            Control::Close {
                request_key: request_key_from_table(
                    close.request_key(),
                    "control.close.request_key",
                )?,
                control_seq: close.control_seq(),
                cutoff: version_ref_from_table(
                    close.cutoff().context("close control has no cutoff")?,
                )?,
                reason: close_reason_from_fb(close.reason())?,
            }
        }
        fbs::Control::ControlRelease => {
            let release = envelope
                .control_as_control_release()
                .context("release control table is missing")?;
            Control::Release {
                request_key: request_key_from_table(
                    release.request_key(),
                    "control.release.request_key",
                )?,
                op_id: OpId(release.op_id()),
            }
        }
        _ => codec_bail!("control union is empty"),
    };
    control.validate()?;
    Ok(control)
}

fn request_key_from_table(
    request_key: Option<fbs::RequestKey<'_>>,
    label: &str,
) -> CodecResult<RequestKey> {
    let request_key = request_key.with_context(|| format!("{label} is missing"))?;
    Ok(RequestKey {
        authority_id: request_key.authority_id(),
        session_id: RequestId(request_key.session_id()),
        epoch: request_key.epoch(),
    })
}

fn version_ref_from_table(version: fbs::VersionRef<'_>) -> CodecResult<VersionRef> {
    let point = match version.point_type() {
        fbs::Point::PointFixed => {
            let fixed = version
                .point_as_point_fixed()
                .context("fixed point table is missing")?;
            Point::Fixed {
                point_index: fixed.point_index(),
                semantic_digest: required_digest(
                    fixed.semantic_digest(),
                    "point.fixed.semantic_digest",
                )?,
            }
        }
        fbs::Point::PointDevice => {
            let device = version
                .point_as_point_device()
                .context("device point table is missing")?;
            Point::Device {
                point_index: device.point_index(),
                selected_point: device
                    .selected_point()
                    .map(product_ref_from_table)
                    .transpose()?,
                producer_plan_digest: required_digest(
                    device.producer_plan_digest(),
                    "point.device.producer_plan_digest",
                )?,
            }
        }
        _ => codec_bail!("version reference point union is empty"),
    };
    Ok(VersionRef {
        request_key: request_key_from_table(version.request_key(), "version_ref.request_key")?,
        producer_op_id: OpId(version.producer_op_id()),
        point,
    })
}

fn product_ref_from_table(product: fbs::ProductRef<'_>) -> CodecResult<ProductRef> {
    let point_range = product
        .point_range()
        .context("product reference has no point range")?;
    Ok(ProductRef {
        request_key: request_key_from_table(product.request_key(), "product_ref.request_key")?,
        producer_op_id: OpId(product.producer_op_id()),
        output_index: product.output_index(),
        generation: product.generation(),
        kind: product_kind_from_fb(product.kind())?,
        storage_class: storage_class_from_fb(product.storage_class())?,
        dtype: dtype_from_fb(product.dtype())?,
        shape_bound: shape_bound_from_table(
            product
                .shape_bound()
                .context("product reference has no shape bound")?,
        ),
        point_range: PointRange {
            base_point: point_range.base_point(),
            max_points: point_range.max_points(),
        },
    })
}

fn shape_bound_from_table(shape: fbs::ShapeBound<'_>) -> ShapeBound {
    ShapeBound {
        dims: shape
            .dims()
            .map(|dims| {
                dims.iter()
                    .map(|dim| {
                        if dim.device() {
                            DimBound::Device { max: dim.value() }
                        } else {
                            DimBound::Static(dim.value())
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn bounds_from_table(bounds: fbs::Bounds<'_>) -> Bounds {
    Bounds {
        max_points: bounds.max_points(),
        max_tokens: bounds.max_tokens(),
        max_kv_pages: bounds.max_kv_pages(),
        max_latent_bytes: bounds.max_latent_bytes(),
        max_completion_bytes: bounds.max_completion_bytes(),
        max_transfer_bytes: bounds.max_transfer_bytes(),
    }
}

fn rng_from_table(rng: fbs::Rng<'_>) -> CodecResult<Rng> {
    Ok(Rng {
        seed: rng.seed(),
        semantic_index_base: rng.semantic_index_base(),
        draw_layout: draw_layout_from_fb(rng.draw_layout())?,
    })
}

fn completion_report_from_table(
    report: fbs::CompletionReport<'_>,
) -> CodecResult<CompletionReport> {
    let report = CompletionReport {
        step_id: report.step_id(),
        partitions: report
            .partitions()
            .map(|items| {
                items
                    .iter()
                    .map(partition_completion_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    report.validate()?;
    Ok(report)
}

fn partition_completion_from_table(
    report: fbs::PartitionCompletion<'_>,
) -> CodecResult<PartitionCompletion> {
    let report = PartitionCompletion {
        partition_id: report.partition_id(),
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
    };
    report.validate()?;
    Ok(report)
}

fn completion_record_from_table(record: fbs::ModelOutput<'_>) -> CodecResult<ModelOutput> {
    let logical_lengths = record
        .logical_lengths()
        .context("completion record has no logical lengths")?;
    let token_span = record
        .token_span()
        .context("completion record has no token span")?;
    let finish_flags = record
        .finish_flags()
        .context("completion record has no finish flags")?;
    let timing_counters = record
        .timing_counters()
        .context("completion record has no timing counters")?;
    let record = ModelOutput {
        request_key: request_key_from_table(record.request_key(), "completion.request_key")?,
        op_id: OpId(record.op_id()),
        completion_slot_generation: record.completion_slot_generation(),
        status: op_status_from_fb(record.status())?,
        selected_point: record.selected_point(),
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
        committed_tokens: record
            .committed_tokens()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        finish_flags: FinishFlags {
            eos: finish_flags.eos(),
            length: finish_flags.length(),
            stop: finish_flags.stop(),
        },
        product_generations: record
            .product_generations()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        semantic_digest: required_digest(record.semantic_digest(), "completion.semantic_digest")?,
        error_code: record.error_code().map(error_code_from_fb).transpose()?,
        timing_counters: TimingCounters {
            queued_us: timing_counters.queued_us(),
            device_us: timing_counters.device_us(),
            copy_us: timing_counters.copy_us(),
            host_us: timing_counters.host_us(),
        },
    };
    record.validate()?;
    Ok(record)
}

fn product_payload_from_table(payload: fbs::ProductPayload<'_>) -> CodecResult<ProductPayload> {
    let payload = ProductPayload {
        product: product_ref_from_table(
            payload
                .product()
                .context("product payload has no product reference")?,
        )?,
        bytes: payload
            .bytes()
            .map(|bytes| bytes.bytes().to_vec())
            .unwrap_or_default(),
    };
    payload.validate()?;
    Ok(payload)
}

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

fn capabilities_from_table(caps: fbs::WorkerCapabilities<'_>) -> CodecResult<WorkerCapabilities> {
    let caps = WorkerCapabilities {
        block_size: caps.block_size(),
        num_blocks: caps.num_blocks(),
        num_layers: caps.num_layers(),
        num_kv_heads: caps.num_kv_heads(),
        head_dim: caps.head_dim(),
        supported_work: caps
            .supported_work()
            .map(|items| items.iter().map(work_from_fb).collect::<CodecResult<_>>())
            .transpose()?
            .unwrap_or_default(),
        latent_page_units: caps.latent_page_units(),
        num_latent_pages: caps.num_latent_pages(),
        latent_width: caps.latent_width(),
        latent_dtype: optional_parse(caps.latent_dtype(), "capabilities.latent_dtype")?,
        latent_downsample: caps.latent_downsample(),
        bytes_per_token: caps.bytes_per_token(),
        max_vae_grid_tokens: caps.max_vae_grid_tokens(),
        max_vit_grid_tokens: caps.max_vit_grid_tokens(),
        max_latent_feature_bytes: caps.max_latent_feature_bytes(),
        max_vision_feature_bytes: caps.max_vision_feature_bytes(),
        commit_marker_tokens: caps.commit_marker_tokens(),
        gen_rope_advance: caps.gen_rope_advance(),
        max_cfg_branches: caps.max_cfg_branches(),
        groups: caps
            .groups()
            .map(|items| {
                items
                    .iter()
                    .map(kv_group_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        kv_dtype: optional_parse(caps.kv_dtype(), "capabilities.kv_dtype")?,
        model_dtype: required_parse(caps.model_dtype(), "capabilities.model_dtype")?,
        attention_backend: required_parse(
            caps.attention_backend(),
            "capabilities.attention_backend",
        )?,
        rank: caps
            .rank()
            .map(rank_from_table)
            .context("capabilities have no rank")?,
        pipeline_depth: caps.pipeline_depth(),
        encoder_cache_budget: caps.encoder_cache_budget(),
        supported_controls: caps
            .supported_controls()
            .map(|items| {
                items
                    .iter()
                    .map(request_kind_from_fb)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        max_batch_operations: caps.max_batch_operations(),
        max_batch_tokens: caps.max_batch_tokens(),
        max_request_pool_size: caps.max_request_pool_size(),
        max_unresolved_window: caps.max_unresolved_window(),
        incremental_kv_publication: caps.incremental_kv_publication(),
        mixed_buckets: caps
            .mixed_buckets()
            .map(|items| items.iter().map(mixed_bucket_from_table).collect())
            .unwrap_or_default(),
        sampling_ownership: sampling_ownership_from_fb(caps.sampling_ownership())?,
        resource_classes: caps
            .resource_classes()
            .map(|items| {
                items
                    .iter()
                    .map(resource_class_from_fb)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        model_identity: optional_digest(caps.model_identity(), "capabilities.model_identity")?,
        weight_digest: optional_digest(caps.weight_digest(), "capabilities.weight_digest")?,
        protocol_layout_digest: required_digest(
            caps.protocol_layout_digest(),
            "capabilities.protocol_layout_digest",
        )?,
        lanes: caps
            .lanes()
            .map(|items| {
                items
                    .iter()
                    .map(lane_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    caps.validate()?;
    Ok(caps)
}

fn sampling_from_table(sampling: fbs::SamplingParams<'_>) -> CodecResult<SamplingParams> {
    let sampling = SamplingParams {
        temperature: sampling.temperature(),
        top_k: sampling.top_k(),
        top_p: sampling.top_p(),
        ignore_eos: sampling.ignore_eos(),
        seed: sampling.seed(),
        min_p: sampling.min_p(),
        repetition_penalty: sampling.repetition_penalty(),
        frequency_penalty: sampling.frequency_penalty(),
        presence_penalty: sampling.presence_penalty(),
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

fn forward_stats_from_table(stats: fbs::WorkerForwardStats<'_>) -> WorkerForwardStats {
    WorkerForwardStats {
        mode_counts: map_from_table(stats.mode_counts()),
        mode_tokens: map_from_table(stats.mode_tokens()),
        mode_us: map_from_table(stats.mode_us()),
        component_us: map_from_table(stats.component_us()),
        attention_launches: stats.attention_launches(),
        attention_us: stats.attention_us(),
        attention_backend_counts: map_from_table(stats.attention_backend_counts()),
        cuda_graph_captures: stats.cuda_graph_captures(),
        cuda_graph_replays: stats.cuda_graph_replays(),
        cuda_graph_misses: stats.cuda_graph_misses(),
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks(),
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens(),
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens(),
        cuda_graph_runtime_mode_counts: map_from_table(stats.cuda_graph_runtime_mode_counts()),
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits(),
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses(),
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits(),
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses(),
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls(),
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses(),
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows(),
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices(),
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls(),
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses(),
        spec_verify_rows: stats.spec_verify_rows(),
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens(),
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens(),
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens(),
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens(),
        spec_verify_path_counts: map_from_table(stats.spec_verify_path_counts()),
    }
}

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

fn kv_group_from_table(group: fbs::KvGroupSpec<'_>) -> CodecResult<KvCacheGroupSpec> {
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
    Ok(KvCacheGroupSpec {
        group_id: group.group_id(),
        block_offset: group.block_offset(),
        num_blocks: group.num_blocks(),
        kind,
    })
}

fn rank_from_table(rank: fbs::RankInfo<'_>) -> RankInfo {
    RankInfo {
        tp_rank: rank.tp_rank(),
        tp_size: rank.tp_size(),
    }
}

fn pressure_from_table(pressure: fbs::ResourcePressure<'_>) -> CodecResult<ResourcePressure> {
    Ok(ResourcePressure {
        class: resource_class_from_fb(pressure.class())?,
        total: pressure.total(),
        used: pressure.used(),
        evictable: pressure.evictable(),
        free: pressure.free(),
    })
}

fn required_str(value: Option<&str>, label: &str) -> CodecResult<String> {
    value
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .with_context(|| format!("{label} is missing"))
}

fn required_digest(value: Option<&str>, label: &str) -> CodecResult<uniserve_core::Digest> {
    let value = required_str(value, label)?;
    uniserve_core::Digest::try_from(value)
        .with_context(|| format!("{label} is not a canonical SHA-256 digest"))
}

fn optional_digest(value: Option<&str>, label: &str) -> CodecResult<Option<uniserve_core::Digest>> {
    match value.filter(|value| !value.is_empty()) {
        Some(value) => uniserve_core::Digest::try_from(value)
            .map(Some)
            .with_context(|| format!("{label} is not a canonical SHA-256 digest")),
        None => Ok(None),
    }
}

fn required_parse<T>(value: Option<&str>, label: &str) -> CodecResult<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    required_str(value, label)?
        .parse()
        .with_context(|| format!("{label} is invalid"))
}

fn optional_parse<T>(value: Option<&str>, label: &str) -> CodecResult<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().with_context(|| format!("{label} is invalid")))
        .transpose()
}

fn snapshot_from_table(snapshot: fbs::SnapshotRef<'_>) -> CodecResult<SnapshotRef> {
    let snapshot = SnapshotRef {
        version: version_ref_from_table(
            snapshot
                .version()
                .context("snapshot reference has no exact version")?,
        )?,
        digest: required_digest(snapshot.digest(), "snapshot digest")?,
        locator: required_str(snapshot.locator(), "snapshot locator")?,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn recovery_placement_from_table(
    placement: fbs::RecoveryPlacement<'_>,
) -> CodecResult<RecoveryPlacement> {
    let placement = RecoveryPlacement {
        request_key: request_key_from_table(
            placement.request_key(),
            "recovery placement.request_key",
        )?,
        request_pool_idx: placement.request_pool_idx(),
        block_tables: placement
            .block_tables()
            .map(|tables| tables.iter().map(block_table_from_table).collect())
            .unwrap_or_default(),
        latent_page_table: placement
            .latent_page_table()
            .map(|pages| pages.iter().collect())
            .unwrap_or_default(),
    };
    placement.validate()?;
    Ok(placement)
}

// ---------------------------------------------------------------------------
// Request / response framing
// ---------------------------------------------------------------------------

fn request_to_fb(request: &WorkerRequest) -> CodecResult<fbs::WorkerRequestT> {
    let (batch, step_id, session_id, copies, product_handles, snapshot, recovery_placement) =
        match request {
            WorkerRequest::GetCapabilities { .. }
            | WorkerRequest::GetPressure { .. }
            | WorkerRequest::Shutdown => (None, None, None, None, None, None, None),
            WorkerRequest::Execute { batch, .. } => {
                batch.validate()?;
                (Some(batch), None, None, None, None, None, None)
            }
            WorkerRequest::PollCompletions { step_id, .. } => {
                (None, Some(*step_id), None, None, None, None, None)
            }
            WorkerRequest::DropSession { session_id } => {
                (None, None, Some(*session_id), None, None, None, None)
            }
            WorkerRequest::CopyKv { copies } => {
                for copy in copies {
                    copy.validate()?;
                }
                (None, None, None, Some(copies), None, None, None)
            }
            WorkerRequest::ReleaseProducts { product_handles } => {
                (None, None, None, None, Some(product_handles), None, None)
            }
            WorkerRequest::SnapshotSession { recovery_placement } => {
                recovery_placement.validate()?;
                (None, None, None, None, None, None, Some(recovery_placement))
            }
            WorkerRequest::RestoreSession {
                snapshot,
                recovery_placement,
            } => {
                snapshot.validate()?;
                recovery_placement.validate()?;
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(snapshot),
                    Some(recovery_placement),
                )
            }
        };
    Ok(fbs::WorkerRequestT {
        kind: request_kind_to_fb(request.kind()),
        call_id: request.call_id(),
        batch: batch.map(batch_to_fb).transpose()?.map(Box::new),
        step_id,
        session_id: session_id.map(|id| id.0),
        copies: copies.map(|items| {
            items
                .iter()
                .map(|copy| fbs::BlockPairT {
                    group_id: copy.group_id,
                    src: copy.source_page.0,
                    dst: copy.destination_page.0,
                })
                .collect()
        }),
        product_handles: product_handles.cloned(),
        snapshot: snapshot.map(snapshot_to_fb).map(Box::new),
        recovery_placement: recovery_placement
            .map(recovery_placement_to_fb)
            .map(Box::new),
    })
}

fn response_to_fb(response: &WorkerResponse) -> CodecResult<fbs::WorkerResponseT> {
    let (capabilities, completion_report, pressure, error, snapshot) = match response {
        WorkerResponse::Capabilities { capabilities, .. } => {
            (Some(capabilities), None, None, None, None)
        }
        WorkerResponse::Result {
            completion_report, ..
        } => (None, Some(completion_report), None, None, None),
        WorkerResponse::Ok { .. } => (None, None, None, None, None),
        WorkerResponse::Error { error, .. } => (None, None, None, Some(error), None),
        WorkerResponse::Pressure { pressure, .. } => (None, None, Some(pressure), None, None),
        WorkerResponse::Snapshot { snapshot, .. } => (None, None, None, None, Some(snapshot)),
    };
    Ok(fbs::WorkerResponseT {
        kind: response_kind_to_fb(response.kind()),
        call_id: response.call_id(),
        capabilities: capabilities
            .map(capabilities_to_fb)
            .transpose()?
            .map(Box::new),
        completion_report: completion_report
            .map(completion_report_to_fb)
            .transpose()?
            .map(Box::new),
        pressure: pressure.map(|items| items.iter().map(pressure_to_fb).collect()),
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
        snapshot: snapshot.map(snapshot_to_fb).map(Box::new),
    })
}

// ---------------------------------------------------------------------------
// Batch, operations, admissions, controls
// ---------------------------------------------------------------------------

fn batch_to_fb(batch: &Batch) -> CodecResult<fbs::BatchT> {
    batch.validate()?;
    Ok(fbs::BatchT {
        step_id: batch.step_id,
        admissions: Some(
            batch
                .admissions
                .iter()
                .map(admission_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        partitions: Some(
            batch
                .partitions
                .iter()
                .map(partition_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        controls: Some(
            batch
                .controls
                .iter()
                .map(|control| fbs::ControlEnvelopeT {
                    control: control_to_fb(control),
                })
                .collect(),
        ),
        input_products: Some(
            batch
                .input_products
                .iter()
                .map(product_payload_to_fb)
                .collect(),
        ),
    })
}

fn partition_to_fb(partition: &BatchPartition) -> CodecResult<fbs::BatchPartitionT> {
    partition.validate()?;
    Ok(fbs::BatchPartitionT {
        partition_id: partition.partition_id,
        submission_group: partition.submission_group,
        collective_seq: partition.collective_seq,
        domain: domain_to_fb(partition.domain),
        route: partition.route.0,
        execution: execution_capability_to_fb(partition.execution),
        attention: attention_regime_to_fb(partition.attention),
        shape_class: partition.shape_class,
        operations: Some(
            partition
                .operations
                .iter()
                .map(operation_to_fb)
                .collect::<CodecResult<_>>()?,
        ),
        block_tables: Some(
            partition
                .block_tables
                .iter()
                .map(block_table_to_fb)
                .collect(),
        ),
        new_cache_pages: Some(
            partition
                .new_cache_pages
                .iter()
                .map(cache_page_allocation_to_fb)
                .collect(),
        ),
        forward_rows: Some(
            partition
                .forward_rows
                .iter()
                .map(row_geometry_to_fb)
                .collect(),
        ),
        latent_placements: Some(
            partition
                .latent_placements
                .iter()
                .map(latent_placement_to_fb)
                .collect(),
        ),
        decode_placements: Some(
            partition
                .decode_placements
                .iter()
                .map(decode_placement_to_fb)
                .collect(),
        ),
    })
}

fn admission_to_fb(admission: &Admission) -> CodecResult<fbs::AdmissionT> {
    admission.validate()?;
    Ok(fbs::AdmissionT {
        request_key: Some(Box::new(request_key_to_fb(admission.request_key))),
        request_pool_idx: admission.request_pool_idx,
        digest: Some(admission.digest.to_string()),
        und: admission
            .und
            .as_ref()
            .map(und_admission_to_fb)
            .transpose()?
            .map(Box::new),
        gen_admission: admission
            .gen_admission
            .as_ref()
            .map(gen_admission_to_fb)
            .map(Box::new),
        media: admission
            .media
            .as_ref()
            .map(media_admission_to_fb)
            .map(Box::new),
    })
}

fn und_admission_to_fb(admission: &UndAdmission) -> CodecResult<fbs::UndAdmissionT> {
    Ok(fbs::UndAdmissionT {
        sampling: Some(Box::new(sampling_to_fb(&admission.sampling)?)),
        negative_token_ids: Some(admission.negative_token_ids.clone()),
        finish_token_ids: Some(admission.finish_token_ids.clone()),
        initial_position: admission.initial_position,
    })
}

fn gen_admission_to_fb(admission: &GenAdmission) -> fbs::GenAdmissionT {
    fbs::GenAdmissionT {
        image: Some(Box::new(image_to_fb(&admission.image))),
    }
}

fn media_admission_to_fb(admission: &MediaAdmission) -> fbs::MediaAdmissionT {
    fbs::MediaAdmissionT {
        prompt: Some(admission.prompt.clone()),
        seed: admission.seed,
        profile: media_profile_to_fb(admission.profile),
        output_path: Some(admission.output_path.clone()),
    }
}

fn recovery_placement_to_fb(placement: &RecoveryPlacement) -> fbs::RecoveryPlacementT {
    fbs::RecoveryPlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        request_pool_idx: placement.request_pool_idx,
        block_tables: Some(
            placement
                .block_tables
                .iter()
                .map(block_table_to_fb)
                .collect(),
        ),
        latent_page_table: Some(placement.latent_page_table.clone()),
    }
}

fn block_table_to_fb(table: &BlockTable) -> fbs::BlockTableT {
    fbs::BlockTableT {
        request_pool_idx: table.request_pool_idx,
        group_id: table.group_id,
        page_ids: Some(table.page_ids.iter().map(|page| page.0).collect()),
        allocated_tokens: table.allocated_tokens,
    }
}

fn cache_page_allocation_to_fb(allocation: &CachePageAllocation) -> fbs::CachePageAllocationT {
    fbs::CachePageAllocationT {
        request_pool_idx: allocation.request_pool_idx,
        group_id: allocation.group_id,
        page_ids: Some(allocation.page_ids.iter().map(|page| page.0).collect()),
    }
}

fn row_geometry_to_fb(row: &RowGeometry) -> fbs::RowGeometryT {
    fbs::RowGeometryT {
        operation_index: row.operation_index,
        request_pool_index: row.request_pool_index,
        seq_len: row.seq_len,
        query_len: row.query_len,
    }
}

fn latent_placement_to_fb(placement: &LatentPlacement) -> fbs::LatentPlacementT {
    fbs::LatentPlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        op_id: placement.op_id.0,
        page_table: Some(placement.page_table.clone()),
        latent_units: placement.latent_units,
        height: placement.height,
        width: placement.width,
        start_step: placement.start_step,
        step_count: placement.step_count,
    }
}

fn decode_placement_to_fb(placement: &DecodePlacement) -> fbs::DecodePlacementT {
    fbs::DecodePlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        op_id: placement.op_id.0,
        kind: decode_kind_to_fb(placement.kind),
        start_unit: placement.start_unit,
        unit_count: placement.unit_count,
    }
}

fn operation_to_fb(operation: &Operation) -> CodecResult<fbs::OperationT> {
    operation.validate()?;
    Ok(fbs::OperationT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
        parent: Some(Box::new(version_ref_to_fb(&operation.parent))),
        work: work_to_fb(operation.work),
        route: operation.route.0,
        domain: domain_to_fb(operation.domain),
        advances_state: operation.advances_state,
        bounds: Some(Box::new(bounds_to_fb(&operation.bounds))),
        inputs: Some(operation.inputs.iter().map(product_ref_to_fb).collect()),
        outputs: Some(operation.outputs.iter().map(product_ref_to_fb).collect()),
        predicate: operation
            .predicate
            .as_ref()
            .map(product_ref_to_fb)
            .map(Box::new),
        rng: operation.rng.as_ref().map(rng_to_fb).map(Box::new),
        control_seq: operation.control_seq,
        plan_digest: Some(operation.plan_digest.to_string()),
    })
}

fn control_to_fb(control: &Control) -> fbs::ControlT {
    match control {
        Control::Commit {
            request_key,
            control_seq,
            expected_parent,
            selected,
            public_event_limit,
            disposition,
        } => fbs::ControlT::ControlCommit(Box::new(fbs::ControlCommitT {
            request_key: Some(Box::new(request_key_to_fb(*request_key))),
            control_seq: *control_seq,
            expected_parent: Some(Box::new(version_ref_to_fb(expected_parent))),
            selected: Some(Box::new(version_ref_to_fb(selected))),
            public_event_limit: *public_event_limit,
            disposition: disposition_to_fb(*disposition),
        })),
        Control::Close {
            request_key,
            control_seq,
            cutoff,
            reason,
        } => fbs::ControlT::ControlClose(Box::new(fbs::ControlCloseT {
            request_key: Some(Box::new(request_key_to_fb(*request_key))),
            control_seq: *control_seq,
            cutoff: Some(Box::new(version_ref_to_fb(cutoff))),
            reason: close_reason_to_fb(*reason),
        })),
        Control::Release { request_key, op_id } => {
            fbs::ControlT::ControlRelease(Box::new(fbs::ControlReleaseT {
                request_key: Some(Box::new(request_key_to_fb(*request_key))),
                op_id: op_id.0,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Identities and shared record parts
// ---------------------------------------------------------------------------

fn request_key_to_fb(request_key: RequestKey) -> fbs::RequestKeyT {
    fbs::RequestKeyT {
        authority_id: request_key.authority_id,
        session_id: request_key.session_id.0,
        epoch: request_key.epoch,
    }
}

fn version_ref_to_fb(version: &VersionRef) -> fbs::VersionRefT {
    fbs::VersionRefT {
        request_key: Some(Box::new(request_key_to_fb(version.request_key))),
        producer_op_id: version.producer_op_id.0,
        point: match &version.point {
            Point::Fixed {
                point_index,
                semantic_digest,
            } => fbs::PointT::PointFixed(Box::new(fbs::PointFixedT {
                point_index: *point_index,
                semantic_digest: Some(semantic_digest.to_string()),
            })),
            Point::Device {
                point_index,
                selected_point,
                producer_plan_digest,
            } => fbs::PointT::PointDevice(Box::new(fbs::PointDeviceT {
                point_index: *point_index,
                selected_point: selected_point
                    .as_ref()
                    .map(|value| Box::new(product_ref_to_fb(value))),
                producer_plan_digest: Some(producer_plan_digest.to_string()),
            })),
        },
    }
}

fn product_ref_to_fb(product: &ProductRef) -> fbs::ProductRefT {
    fbs::ProductRefT {
        request_key: Some(Box::new(request_key_to_fb(product.request_key))),
        producer_op_id: product.producer_op_id.0,
        output_index: product.output_index,
        generation: product.generation,
        kind: product_kind_to_fb(product.kind),
        storage_class: storage_class_to_fb(product.storage_class),
        dtype: dtype_to_fb(product.dtype),
        shape_bound: Some(Box::new(shape_bound_to_fb(&product.shape_bound))),
        point_range: Some(Box::new(fbs::PointRangeT {
            base_point: product.point_range.base_point,
            max_points: product.point_range.max_points,
        })),
    }
}

fn shape_bound_to_fb(shape: &ShapeBound) -> fbs::ShapeBoundT {
    fbs::ShapeBoundT {
        dims: Some(
            shape
                .dims
                .iter()
                .map(|dim| match dim {
                    DimBound::Static(extent) => fbs::DimBoundT {
                        device: false,
                        value: *extent,
                    },
                    DimBound::Device { max } => fbs::DimBoundT {
                        device: true,
                        value: *max,
                    },
                })
                .collect(),
        ),
    }
}

fn bounds_to_fb(bounds: &Bounds) -> fbs::BoundsT {
    fbs::BoundsT {
        max_points: bounds.max_points,
        max_tokens: bounds.max_tokens,
        max_kv_pages: bounds.max_kv_pages,
        max_latent_bytes: bounds.max_latent_bytes,
        max_completion_bytes: bounds.max_completion_bytes,
        max_transfer_bytes: bounds.max_transfer_bytes,
    }
}

fn rng_to_fb(rng: &Rng) -> fbs::RngT {
    fbs::RngT {
        seed: rng.seed,
        semantic_index_base: rng.semantic_index_base,
        draw_layout: draw_layout_to_fb(rng.draw_layout),
    }
}

// ---------------------------------------------------------------------------
// Completion report and records
// ---------------------------------------------------------------------------

fn completion_report_to_fb(report: &CompletionReport) -> CodecResult<fbs::CompletionReportT> {
    report.validate()?;
    Ok(fbs::CompletionReportT {
        step_id: report.step_id,
        partitions: Some(
            report
                .partitions
                .iter()
                .map(partition_completion_to_fb)
                .collect(),
        ),
    })
}

fn partition_completion_to_fb(report: &PartitionCompletion) -> fbs::PartitionCompletionT {
    fbs::PartitionCompletionT {
        partition_id: report.partition_id,
        completions: Some(
            report
                .completions
                .iter()
                .map(completion_record_to_fb)
                .collect(),
        ),
        products: Some(report.products.iter().map(product_payload_to_fb).collect()),
        registration: Some(Box::new(fbs::RegistrationAckT {
            visible: report.registration.visible,
        })),
        worker_exec_us: report.worker_exec_us,
        forward_stats: report
            .forward_stats
            .as_ref()
            .map(forward_stats_to_fb)
            .map(Box::new),
    }
}

fn completion_record_to_fb(record: &ModelOutput) -> fbs::ModelOutputT {
    fbs::ModelOutputT {
        request_key: Some(Box::new(request_key_to_fb(record.request_key))),
        op_id: record.op_id.0,
        completion_slot_generation: record.completion_slot_generation,
        status: op_status_to_fb(record.status),
        selected_point: record.selected_point,
        logical_lengths: Some(Box::new(fbs::LogicalLengthsT {
            token_len: record.logical_lengths.token_len,
            kv_visible_len: record.logical_lengths.kv_visible_len,
            kv_computed_len: record.logical_lengths.kv_computed_len,
            latent_len: record.logical_lengths.latent_len,
        })),
        token_span: Some(Box::new(fbs::TokenSpanT {
            base: record.token_span.base,
            len: record.token_span.len,
        })),
        committed_tokens: Some(record.committed_tokens.clone()),
        finish_flags: Some(Box::new(fbs::FinishFlagsT {
            eos: record.finish_flags.eos,
            length: record.finish_flags.length,
            stop: record.finish_flags.stop,
        })),
        product_generations: Some(record.product_generations.clone()),
        semantic_digest: Some(record.semantic_digest.to_string()),
        error_code: record.error_code.map(error_code_to_fb),
        timing_counters: Some(Box::new(fbs::TimingCountersT {
            queued_us: record.timing_counters.queued_us,
            device_us: record.timing_counters.device_us,
            copy_us: record.timing_counters.copy_us,
            host_us: record.timing_counters.host_us,
        })),
    }
}

fn product_payload_to_fb(payload: &ProductPayload) -> fbs::ProductPayloadT {
    fbs::ProductPayloadT {
        product: Some(Box::new(product_ref_to_fb(&payload.product))),
        bytes: Some(payload.bytes.clone()),
    }
}

fn error_operation_to_fb(operation: &ErrorOperationIdentity) -> fbs::ErrorOperationIdentityT {
    fbs::ErrorOperationIdentityT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
    }
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

fn graph_bucket_from_table(
    bucket: fbs::GraphBucketCapability<'_>,
) -> CodecResult<GraphBucketCapability> {
    Ok(GraphBucketCapability {
        phase: required_str(bucket.phase(), "graph_bucket.phase")?,
        batch_size: bucket.batch_size(),
        token_bucket: bucket.token_bucket(),
        attention_form: required_str(bucket.attention_form(), "graph_bucket.attention_form")?,
        height: bucket.height(),
        width: bucket.width(),
        cfg_branches: bucket.cfg_branches(),
        layout: bucket.layout().unwrap_or_default().to_string(),
    })
}

fn mixed_bucket_from_table(bucket: fbs::MixedExecutionCapability<'_>) -> MixedExecutionCapability {
    MixedExecutionCapability {
        decode_rows: bucket.decode_rows(),
        flow_rows: bucket.flow_rows(),
        height: bucket.height(),
        width: bucket.width(),
        cfg_branches: bucket.cfg_branches(),
    }
}

fn lane_from_table(lane: fbs::LaneCapabilities<'_>) -> CodecResult<LaneCapabilities> {
    Ok(LaneCapabilities {
        lane_id: required_str(lane.lane_id(), "lane.lane_id")?,
        domains: lane
            .domains()
            .map(|items| items.iter().map(domain_from_fb).collect::<CodecResult<_>>())
            .transpose()?
            .unwrap_or_default(),
        resolved_sm_count: lane.resolved_sm_count(),
        kv_capacity_tokens: lane.kv_capacity_tokens(),
        latent_capacity_units: lane.latent_capacity_units(),
        max_batch_operations: lane.max_batch_operations(),
        max_batch_tokens: lane.max_batch_tokens(),
        max_inflight: lane.max_inflight(),
        graph_buckets: lane
            .graph_buckets()
            .map(|items| {
                items
                    .iter()
                    .map(graph_bucket_from_table)
                    .collect::<CodecResult<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        eager_max_batch_operations: lane.eager_max_batch_operations(),
        eager_max_batch_tokens: lane.eager_max_batch_tokens(),
    })
}

fn graph_bucket_to_fb(bucket: &GraphBucketCapability) -> fbs::GraphBucketCapabilityT {
    fbs::GraphBucketCapabilityT {
        phase: Some(bucket.phase.clone()),
        batch_size: bucket.batch_size,
        token_bucket: bucket.token_bucket,
        attention_form: Some(bucket.attention_form.clone()),
        height: bucket.height,
        width: bucket.width,
        cfg_branches: bucket.cfg_branches,
        layout: Some(bucket.layout.clone()),
    }
}

fn mixed_bucket_to_fb(bucket: &MixedExecutionCapability) -> fbs::MixedExecutionCapabilityT {
    fbs::MixedExecutionCapabilityT {
        decode_rows: bucket.decode_rows,
        flow_rows: bucket.flow_rows,
        height: bucket.height,
        width: bucket.width,
        cfg_branches: bucket.cfg_branches,
    }
}

fn lane_to_fb(lane: &LaneCapabilities) -> fbs::LaneCapabilitiesT {
    fbs::LaneCapabilitiesT {
        lane_id: Some(lane.lane_id.clone()),
        domains: Some(lane.domains.iter().copied().map(domain_to_fb).collect()),
        resolved_sm_count: lane.resolved_sm_count,
        kv_capacity_tokens: lane.kv_capacity_tokens,
        latent_capacity_units: lane.latent_capacity_units,
        max_batch_operations: lane.max_batch_operations,
        max_batch_tokens: lane.max_batch_tokens,
        max_inflight: lane.max_inflight,
        graph_buckets: Some(lane.graph_buckets.iter().map(graph_bucket_to_fb).collect()),
        eager_max_batch_operations: lane.eager_max_batch_operations,
        eager_max_batch_tokens: lane.eager_max_batch_tokens,
    }
}

fn capabilities_to_fb(caps: &WorkerCapabilities) -> CodecResult<fbs::WorkerCapabilitiesT> {
    caps.validate()?;
    Ok(fbs::WorkerCapabilitiesT {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        num_kv_heads: caps.num_kv_heads,
        head_dim: caps.head_dim,
        supported_work: Some(
            caps.supported_work
                .iter()
                .copied()
                .map(work_to_fb)
                .collect(),
        ),
        latent_page_units: caps.latent_page_units,
        num_latent_pages: caps.num_latent_pages,
        latent_width: caps.latent_width,
        latent_dtype: Some(
            caps.latent_dtype
                .map(uniserve_core::ModelDtype::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        latent_downsample: caps.latent_downsample,
        bytes_per_token: caps.bytes_per_token,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
        max_latent_feature_bytes: caps.max_latent_feature_bytes,
        max_vision_feature_bytes: caps.max_vision_feature_bytes,
        commit_marker_tokens: caps.commit_marker_tokens,
        gen_rope_advance: caps.gen_rope_advance,
        max_cfg_branches: caps.max_cfg_branches,
        groups: Some(caps.groups.iter().map(kv_group_to_fb).collect()),
        kv_dtype: Some(
            caps.kv_dtype
                .map(uniserve_core::KvCacheDtype::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        model_dtype: Some(caps.model_dtype.as_str().to_owned()),
        attention_backend: Some(caps.attention_backend.as_wire_name()),
        rank: Some(Box::new(rank_to_fb(caps.rank))),
        pipeline_depth: caps.pipeline_depth,
        encoder_cache_budget: caps.encoder_cache_budget,
        supported_controls: Some(
            caps.supported_controls
                .iter()
                .copied()
                .map(request_kind_to_fb)
                .collect(),
        ),
        max_batch_operations: caps.max_batch_operations,
        max_batch_tokens: caps.max_batch_tokens,
        max_request_pool_size: caps.max_request_pool_size,
        max_unresolved_window: caps.max_unresolved_window,
        incremental_kv_publication: caps.incremental_kv_publication,
        mixed_buckets: Some(caps.mixed_buckets.iter().map(mixed_bucket_to_fb).collect()),
        sampling_ownership: sampling_ownership_to_fb(caps.sampling_ownership),
        resource_classes: Some(
            caps.resource_classes
                .iter()
                .copied()
                .map(resource_class_to_fb)
                .collect(),
        ),
        model_identity: Some(
            caps.model_identity
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
        ),
        weight_digest: Some(
            caps.weight_digest
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
        ),
        protocol_layout_digest: Some(caps.protocol_layout_digest.to_string()),
        lanes: Some(caps.lanes.iter().map(lane_to_fb).collect()),
    })
}

// ---------------------------------------------------------------------------
// Sampling, image, rank, pressure, and snapshot value encoders
// ---------------------------------------------------------------------------

fn sampling_to_fb(sampling: &SamplingParams) -> CodecResult<fbs::SamplingParamsT> {
    validate_sampling(sampling)?;
    Ok(fbs::SamplingParamsT {
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        ignore_eos: sampling.ignore_eos,
        seed: sampling.seed,
        min_p: sampling.min_p,
        repetition_penalty: sampling.repetition_penalty,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,
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

fn forward_stats_to_fb(stats: &WorkerForwardStats) -> fbs::WorkerForwardStatsT {
    fbs::WorkerForwardStatsT {
        mode_counts: Some(map_to_fb(&stats.mode_counts)),
        mode_tokens: Some(map_to_fb(&stats.mode_tokens)),
        mode_us: Some(map_to_fb(&stats.mode_us)),
        attention_launches: stats.attention_launches,
        attention_us: stats.attention_us,
        attention_backend_counts: Some(map_to_fb(&stats.attention_backend_counts)),
        cuda_graph_captures: stats.cuda_graph_captures,
        cuda_graph_replays: stats.cuda_graph_replays,
        cuda_graph_misses: stats.cuda_graph_misses,
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: Some(map_to_fb(&stats.cuda_graph_runtime_mode_counts)),
        text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
        text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
        text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
        text_decode_position_relay_misses: stats.text_decode_position_relay_misses,
        flashinfer_decode_plan_calls: stats.flashinfer_decode_plan_calls,
        flashinfer_decode_plan_reuses: stats.flashinfer_decode_plan_reuses,
        flashinfer_decode_plan_rows: stats.flashinfer_decode_plan_rows,
        flashinfer_decode_plan_indices: stats.flashinfer_decode_plan_indices,
        flashinfer_decode_graph_plan_calls: stats.flashinfer_decode_graph_plan_calls,
        flashinfer_decode_graph_plan_reuses: stats.flashinfer_decode_graph_plan_reuses,
        spec_verify_rows: stats.spec_verify_rows,
        spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
        spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
        spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
        spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
        spec_verify_path_counts: Some(map_to_fb(&stats.spec_verify_path_counts)),
        component_us: Some(map_to_fb(&stats.component_us)),
    }
}

fn map_to_fb(map: &BTreeMap<String, u64>) -> Vec<fbs::StringU64PairT> {
    map.iter()
        .map(|(key, value)| fbs::StringU64PairT {
            key: Some(key.clone()),
            value: *value,
        })
        .collect()
}

fn kv_group_to_fb(group: &KvCacheGroupSpec) -> fbs::KvGroupSpecT {
    match group.kind {
        KvGroupKind::Full => fbs::KvGroupSpecT {
            group_id: group.group_id,
            block_offset: group.block_offset,
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::Full,
            window: 0,
            sink: 0,
        },
        KvGroupKind::SlidingWindow { window, sink } => fbs::KvGroupSpecT {
            group_id: group.group_id,
            block_offset: group.block_offset,
            num_blocks: group.num_blocks,
            kind: fbs::KvGroupKind::SlidingWindow,
            window,
            sink,
        },
    }
}

fn rank_to_fb(rank: RankInfo) -> fbs::RankInfoT {
    fbs::RankInfoT {
        tp_rank: rank.tp_rank,
        tp_size: rank.tp_size,
    }
}

fn pressure_to_fb(pressure: &ResourcePressure) -> fbs::ResourcePressureT {
    fbs::ResourcePressureT {
        class: resource_class_to_fb(pressure.class),
        total: pressure.total,
        used: pressure.used,
        evictable: pressure.evictable,
        free: pressure.free,
    }
}

fn snapshot_to_fb(snapshot: &SnapshotRef) -> fbs::SnapshotRefT {
    fbs::SnapshotRefT {
        version: Some(Box::new(version_ref_to_fb(&snapshot.version))),
        digest: Some(snapshot.digest.to_string()),
        locator: Some(snapshot.locator.clone()),
    }
}

// ---------------------------------------------------------------------------
// Enum mappers
// ---------------------------------------------------------------------------

fn work_to_fb(variant: ForwardMode) -> fbs::ForwardMode {
    match variant {
        ForwardMode::TokenExtend => fbs::ForwardMode::TokenExtend,
        ForwardMode::TokenDecode => fbs::ForwardMode::TokenDecode,
        ForwardMode::TokenVerify => fbs::ForwardMode::TokenVerify,
        ForwardMode::Draft => fbs::ForwardMode::Draft,
        ForwardMode::EncodeVision => fbs::ForwardMode::EncodeVision,
        ForwardMode::EncodeLatent => fbs::ForwardMode::EncodeLatent,
        ForwardMode::TransferProduct => fbs::ForwardMode::TransferProduct,
        ForwardMode::TransferKvPublish => fbs::ForwardMode::TransferKvPublish,
        ForwardMode::TransferKvInstall => fbs::ForwardMode::TransferKvInstall,
        ForwardMode::GenTransition => fbs::ForwardMode::GenTransition,
        ForwardMode::GenFlow => fbs::ForwardMode::GenFlow,
        ForwardMode::Materialize => fbs::ForwardMode::Materialize,
        ForwardMode::GenDecode => fbs::ForwardMode::GenDecode,
    }
}

fn decode_kind_to_fb(kind: DecodeKind) -> fbs::DecodeKind {
    match kind {
        DecodeKind::Video => fbs::DecodeKind::Video,
        DecodeKind::Audio => fbs::DecodeKind::Audio,
    }
}

fn decode_kind_from_fb(kind: fbs::DecodeKind) -> CodecResult<DecodeKind> {
    if kind == fbs::DecodeKind::Video {
        Ok(DecodeKind::Video)
    } else if kind == fbs::DecodeKind::Audio {
        Ok(DecodeKind::Audio)
    } else {
        codec_bail!("unknown decode kind {}", kind.0)
    }
}

fn media_profile_to_fb(profile: MediaProfileId) -> fbs::MediaProfileId {
    match profile {
        MediaProfileId::MinimaxH3T2va => fbs::MediaProfileId::MinimaxH3T2va,
    }
}

fn media_profile_from_fb(profile: fbs::MediaProfileId) -> CodecResult<MediaProfileId> {
    if profile == fbs::MediaProfileId::MinimaxH3T2va {
        Ok(MediaProfileId::MinimaxH3T2va)
    } else {
        codec_bail!("unknown media profile {}", profile.0)
    }
}

fn work_from_fb(variant: fbs::ForwardMode) -> CodecResult<ForwardMode> {
    for candidate in ForwardMode::ALL {
        if work_to_fb(candidate) == variant {
            return Ok(candidate);
        }
    }
    codec_bail!("unknown work variant {}", variant.0)
}

fn domain_to_fb(domain: Domain) -> fbs::Domain {
    match domain {
        Domain::Prefill => fbs::Domain::Prefill,
        Domain::Decode => fbs::Domain::Decode,
        Domain::Flow => fbs::Domain::Flow,
    }
}

fn domain_from_fb(domain: fbs::Domain) -> CodecResult<Domain> {
    if domain == fbs::Domain::Prefill {
        Ok(Domain::Prefill)
    } else if domain == fbs::Domain::Decode {
        Ok(Domain::Decode)
    } else if domain == fbs::Domain::Flow {
        Ok(Domain::Flow)
    } else {
        codec_bail!("unknown domain {}", domain.0)
    }
}

fn execution_capability_to_fb(capability: ExecutionCapability) -> fbs::ExecutionCapability {
    match capability {
        ExecutionCapability::DomainHomogeneous => fbs::ExecutionCapability::DomainHomogeneous,
        ExecutionCapability::TensorizedMixed => fbs::ExecutionCapability::TensorizedMixed,
    }
}

fn execution_capability_from_fb(
    capability: fbs::ExecutionCapability,
) -> CodecResult<ExecutionCapability> {
    Ok(match capability {
        fbs::ExecutionCapability::DomainHomogeneous => ExecutionCapability::DomainHomogeneous,
        fbs::ExecutionCapability::TensorizedMixed => ExecutionCapability::TensorizedMixed,
        other => codec_bail!("unknown execution capability {}", other.0),
    })
}

fn attention_regime_to_fb(regime: AttentionRegime) -> fbs::AttentionRegime {
    match regime {
        AttentionRegime::None => fbs::AttentionRegime::None,
        AttentionRegime::Causal => fbs::AttentionRegime::Causal,
        AttentionRegime::Bidirectional => fbs::AttentionRegime::Bidirectional,
        AttentionRegime::Hybrid => fbs::AttentionRegime::Hybrid,
    }
}

fn attention_regime_from_fb(regime: fbs::AttentionRegime) -> CodecResult<AttentionRegime> {
    Ok(match regime {
        fbs::AttentionRegime::None => AttentionRegime::None,
        fbs::AttentionRegime::Causal => AttentionRegime::Causal,
        fbs::AttentionRegime::Bidirectional => AttentionRegime::Bidirectional,
        fbs::AttentionRegime::Hybrid => AttentionRegime::Hybrid,
        other => codec_bail!("unknown attention regime {}", other.0),
    })
}

fn sampling_ownership_to_fb(ownership: SamplingOwnership) -> fbs::SamplingOwnership {
    match ownership {
        SamplingOwnership::DesignatedRank => fbs::SamplingOwnership::DesignatedRank,
        SamplingOwnership::DeterministicSharded => fbs::SamplingOwnership::DeterministicSharded,
    }
}

fn sampling_ownership_from_fb(ownership: fbs::SamplingOwnership) -> CodecResult<SamplingOwnership> {
    Ok(match ownership {
        fbs::SamplingOwnership::DesignatedRank => SamplingOwnership::DesignatedRank,
        fbs::SamplingOwnership::DeterministicSharded => SamplingOwnership::DeterministicSharded,
        other => codec_bail!("unknown sampling ownership {}", other.0),
    })
}

fn product_kind_to_fb(kind: ProductKind) -> fbs::ProductKind {
    match kind {
        ProductKind::Token => fbs::ProductKind::Token,
        ProductKind::Logprob => fbs::ProductKind::Logprob,
        ProductKind::Draft => fbs::ProductKind::Draft,
        ProductKind::VisionFeature => fbs::ProductKind::VisionFeature,
        ProductKind::LatentFeature => fbs::ProductKind::LatentFeature,
        ProductKind::Kv => fbs::ProductKind::Kv,
        ProductKind::Latent => fbs::ProductKind::Latent,
        ProductKind::Artifact => fbs::ProductKind::Artifact,
        ProductKind::Completion => fbs::ProductKind::Completion,
        ProductKind::SamplingState => fbs::ProductKind::SamplingState,
        ProductKind::Finish => fbs::ProductKind::Finish,
        ProductKind::SelectedPoint => fbs::ProductKind::SelectedPoint,
        ProductKind::AcceptedSpan => fbs::ProductKind::AcceptedSpan,
        ProductKind::Continuation => fbs::ProductKind::Continuation,
    }
}

fn product_kind_from_fb(kind: fbs::ProductKind) -> CodecResult<ProductKind> {
    Ok(match kind {
        fbs::ProductKind::Token => ProductKind::Token,
        fbs::ProductKind::Logprob => ProductKind::Logprob,
        fbs::ProductKind::Draft => ProductKind::Draft,
        fbs::ProductKind::VisionFeature => ProductKind::VisionFeature,
        fbs::ProductKind::LatentFeature => ProductKind::LatentFeature,
        fbs::ProductKind::Kv => ProductKind::Kv,
        fbs::ProductKind::Latent => ProductKind::Latent,
        fbs::ProductKind::Artifact => ProductKind::Artifact,
        fbs::ProductKind::Completion => ProductKind::Completion,
        fbs::ProductKind::SamplingState => ProductKind::SamplingState,
        fbs::ProductKind::Finish => ProductKind::Finish,
        fbs::ProductKind::SelectedPoint => ProductKind::SelectedPoint,
        fbs::ProductKind::AcceptedSpan => ProductKind::AcceptedSpan,
        fbs::ProductKind::Continuation => ProductKind::Continuation,
        other => codec_bail!("unknown product kind {}", other.0),
    })
}

fn storage_class_to_fb(class: StorageClass) -> fbs::StorageClass {
    match class {
        StorageClass::DeviceTensor => fbs::StorageClass::DeviceTensor,
        StorageClass::PagedKv => fbs::StorageClass::PagedKv,
        StorageClass::LatentArena => fbs::StorageClass::LatentArena,
        StorageClass::HostStaging => fbs::StorageClass::HostStaging,
        StorageClass::PinnedOutput => fbs::StorageClass::PinnedOutput,
    }
}

fn storage_class_from_fb(class: fbs::StorageClass) -> CodecResult<StorageClass> {
    Ok(match class {
        fbs::StorageClass::DeviceTensor => StorageClass::DeviceTensor,
        fbs::StorageClass::PagedKv => StorageClass::PagedKv,
        fbs::StorageClass::LatentArena => StorageClass::LatentArena,
        fbs::StorageClass::HostStaging => StorageClass::HostStaging,
        fbs::StorageClass::PinnedOutput => StorageClass::PinnedOutput,
        other => codec_bail!("unknown storage class {}", other.0),
    })
}

fn dtype_to_fb(dtype: DType) -> fbs::DType {
    match dtype {
        DType::U8 => fbs::DType::U8,
        DType::U16 => fbs::DType::U16,
        DType::U32 => fbs::DType::U32,
        DType::I32 => fbs::DType::I32,
        DType::I64 => fbs::DType::I64,
        DType::F16 => fbs::DType::F16,
        DType::BF16 => fbs::DType::BF16,
        DType::F32 => fbs::DType::F32,
    }
}

fn dtype_from_fb(dtype: fbs::DType) -> CodecResult<DType> {
    Ok(match dtype {
        fbs::DType::U8 => DType::U8,
        fbs::DType::U16 => DType::U16,
        fbs::DType::U32 => DType::U32,
        fbs::DType::I32 => DType::I32,
        fbs::DType::I64 => DType::I64,
        fbs::DType::F16 => DType::F16,
        fbs::DType::BF16 => DType::BF16,
        fbs::DType::F32 => DType::F32,
        other => codec_bail!("unknown dtype {}", other.0),
    })
}

fn draw_layout_to_fb(layout: DrawLayout) -> fbs::DrawLayout {
    match layout {
        DrawLayout::TargetSampling => fbs::DrawLayout::TargetSampling,
        DrawLayout::SpeculativeProposal => fbs::DrawLayout::SpeculativeProposal,
        DrawLayout::FlowNoise => fbs::DrawLayout::FlowNoise,
    }
}

fn draw_layout_from_fb(layout: fbs::DrawLayout) -> CodecResult<DrawLayout> {
    Ok(match layout {
        fbs::DrawLayout::TargetSampling => DrawLayout::TargetSampling,
        fbs::DrawLayout::SpeculativeProposal => DrawLayout::SpeculativeProposal,
        fbs::DrawLayout::FlowNoise => DrawLayout::FlowNoise,
        other => codec_bail!("unknown draw layout {}", other.0),
    })
}

fn op_status_to_fb(status: OpStatus) -> fbs::OpStatus {
    match status {
        OpStatus::Ok => fbs::OpStatus::Ok,
        OpStatus::Predicated => fbs::OpStatus::Predicated,
        OpStatus::Error => fbs::OpStatus::Error,
    }
}

fn op_status_from_fb(status: fbs::OpStatus) -> CodecResult<OpStatus> {
    Ok(match status {
        fbs::OpStatus::Ok => OpStatus::Ok,
        fbs::OpStatus::Predicated => OpStatus::Predicated,
        fbs::OpStatus::Error => OpStatus::Error,
        other => codec_bail!("unknown completion status {}", other.0),
    })
}

fn error_code_to_fb(code: ErrorCode) -> fbs::ErrorCode {
    match code {
        ErrorCode::InvalidOperation => fbs::ErrorCode::InvalidOperation,
        ErrorCode::ResourceExhausted => fbs::ErrorCode::ResourceExhausted,
        ErrorCode::ComputeError => fbs::ErrorCode::ComputeError,
        ErrorCode::Cancelled => fbs::ErrorCode::Cancelled,
        ErrorCode::Internal => fbs::ErrorCode::Internal,
    }
}

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

fn disposition_to_fb(disposition: Disposition) -> fbs::Disposition {
    match disposition {
        Disposition::Publish => fbs::Disposition::Publish,
        Disposition::Retain => fbs::Disposition::Retain,
        Disposition::Discard => fbs::Disposition::Discard,
    }
}

fn disposition_from_fb(disposition: fbs::Disposition) -> CodecResult<Disposition> {
    Ok(match disposition {
        fbs::Disposition::Publish => Disposition::Publish,
        fbs::Disposition::Retain => Disposition::Retain,
        fbs::Disposition::Discard => Disposition::Discard,
        other => codec_bail!("unknown disposition {}", other.0),
    })
}

fn close_reason_to_fb(reason: CloseReason) -> fbs::CloseReason {
    match reason {
        CloseReason::Completed => fbs::CloseReason::Completed,
        CloseReason::Cancelled => fbs::CloseReason::Cancelled,
        CloseReason::Error => fbs::CloseReason::Error,
        CloseReason::Preempted => fbs::CloseReason::Preempted,
    }
}

fn close_reason_from_fb(reason: fbs::CloseReason) -> CodecResult<CloseReason> {
    Ok(match reason {
        fbs::CloseReason::Completed => CloseReason::Completed,
        fbs::CloseReason::Cancelled => CloseReason::Cancelled,
        fbs::CloseReason::Error => CloseReason::Error,
        fbs::CloseReason::Preempted => CloseReason::Preempted,
        other => codec_bail!("unknown close reason {}", other.0),
    })
}

fn request_kind_to_fb(kind: RequestKind) -> fbs::ReqKind {
    match kind {
        RequestKind::GetCapabilities => fbs::ReqKind::GetCapabilities,
        RequestKind::Execute => fbs::ReqKind::Execute,
        RequestKind::PollCompletions => fbs::ReqKind::PollCompletions,
        RequestKind::DropSession => fbs::ReqKind::DropSession,
        RequestKind::Shutdown => fbs::ReqKind::Shutdown,
        RequestKind::CopyKv => fbs::ReqKind::CopyKv,
        RequestKind::ReleaseProducts => fbs::ReqKind::ReleaseProducts,
        RequestKind::GetPressure => fbs::ReqKind::GetPressure,
        RequestKind::SnapshotSession => fbs::ReqKind::SnapshotSession,
        RequestKind::RestoreSession => fbs::ReqKind::RestoreSession,
    }
}

fn request_kind_from_fb(kind: fbs::ReqKind) -> CodecResult<RequestKind> {
    for candidate in RequestKind::ALL {
        if request_kind_to_fb(candidate) == kind {
            return Ok(candidate);
        }
    }
    codec_bail!("unknown request kind {}", kind.0)
}

pub fn request_kind_names() -> impl Iterator<Item = &'static str> {
    RequestKind::ALL.into_iter().map(RequestKind::as_wire_str)
}

fn response_kind_to_fb(kind: ResponseKind) -> fbs::RespKind {
    match kind {
        ResponseKind::Capabilities => fbs::RespKind::Capabilities,
        ResponseKind::Result => fbs::RespKind::Result,
        ResponseKind::Ok => fbs::RespKind::Ok,
        ResponseKind::Error => fbs::RespKind::Error,
        ResponseKind::Pressure => fbs::RespKind::Pressure,
        ResponseKind::Snapshot => fbs::RespKind::Snapshot,
    }
}

fn response_kind_from_fb(kind: fbs::RespKind) -> CodecResult<ResponseKind> {
    if kind == fbs::RespKind::Capabilities {
        Ok(ResponseKind::Capabilities)
    } else if kind == fbs::RespKind::Result {
        Ok(ResponseKind::Result)
    } else if kind == fbs::RespKind::Ok {
        Ok(ResponseKind::Ok)
    } else if kind == fbs::RespKind::Error {
        Ok(ResponseKind::Error)
    } else if kind == fbs::RespKind::Pressure {
        Ok(ResponseKind::Pressure)
    } else if kind == fbs::RespKind::Snapshot {
        Ok(ResponseKind::Snapshot)
    } else {
        codec_bail!("unknown response kind {}", kind.0)
    }
}

fn resource_class_to_fb(class: ResourceClass) -> fbs::ResourceClass {
    match class {
        ResourceClass::KvBlock => fbs::ResourceClass::KvBlock,
        ResourceClass::EncoderOutput => fbs::ResourceClass::EncoderOutput,
        ResourceClass::ImageLatent => fbs::ResourceClass::ImageLatent,
    }
}

fn resource_class_from_fb(class: fbs::ResourceClass) -> CodecResult<ResourceClass> {
    if class == fbs::ResourceClass::KvBlock {
        Ok(ResourceClass::KvBlock)
    } else if class == fbs::ResourceClass::EncoderOutput {
        Ok(ResourceClass::EncoderOutput)
    } else if class == fbs::ResourceClass::ImageLatent {
        Ok(ResourceClass::ImageLatent)
    } else {
        codec_bail!("unknown resource class {}", class.0)
    }
}
