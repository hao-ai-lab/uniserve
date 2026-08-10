//! Hand-written FlatBuffers codec for the worker execution protocol.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use flatbuffers::FlatBufferBuilder;
use uniserve_core::{BlockId, KvCacheGroupSpec, KvGroupKind, RankInfo, RequestId, SamplingParams};

use crate::schema::uniserve::wire as fbs;
use crate::{
    Admission, AttentionRegime, Batch, BatchPartition, Bounds, CacheCopy, CacheGroupPlacement,
    CloseReason, CompletionRecord, CompletionReport, Control, DType, DimBound, Disposition, Domain,
    DrawLayout, ErrorCode, ErrorOperationIdentity, ExecutionCapability, FinishFlags, GenAdmission,
    KvAdmission, KvBranchPlacement, KvPlacement, LatentPlacement, LogicalLengths, OpId, OpStatus,
    Operation, PartitionCompletion, Point, PointRange, ProductKind, ProductPayload, ProductRef,
    RecoveryPlacement, RegistrationAck, RequestKey, RequestKind, ResourceClass, ResourcePressure,
    ResponseKind, Rng, RouteId, SamplingOwnership, ShapeBound, SnapshotRef, StorageClass,
    TimingCounters, TokenSpan, UndAdmission, VersionRef, Work, WorkVariant, WorkerCapabilities,
    WorkerForwardStats, WorkerMetrics, WorkerRequest, WorkerResponse,
};

pub fn encode_request(request: &WorkerRequest) -> anyhow::Result<Vec<u8>> {
    let object = request_to_fb(request)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

pub fn decode_request(bytes: &[u8]) -> anyhow::Result<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_table(root)
}

pub fn encode_response(response: &WorkerResponse) -> anyhow::Result<Vec<u8>> {
    let object = response_to_fb(response)?;
    let mut builder = FlatBufferBuilder::new();
    let root = object.pack(&mut builder);
    builder.finish(root, None);
    Ok(builder.finished_data().to_vec())
}

pub fn decode_response(bytes: &[u8]) -> anyhow::Result<WorkerResponse> {
    let root = flatbuffers::root::<fbs::WorkerResponse>(bytes)
        .context("invalid WorkerResponse flatbuffer")?;
    response_from_table(root)
}

/// The original object-API (`unpack`) request decode, retained only so tests
/// can prove the accessor-based decode is behaviorally identical.
#[cfg(test)]
pub(crate) fn decode_request_unpack(bytes: &[u8]) -> anyhow::Result<WorkerRequest> {
    let root = fbs::root_as_worker_request(bytes).context("invalid WorkerRequest flatbuffer")?;
    request_from_fb(root.unpack())
}

/// The original object-API (`unpack`) response decode, retained only so tests
/// can prove the accessor-based decode is behaviorally identical.
#[cfg(test)]
pub(crate) fn decode_response_unpack(bytes: &[u8]) -> anyhow::Result<WorkerResponse> {
    let root = flatbuffers::root::<fbs::WorkerResponse>(bytes)
        .context("invalid WorkerResponse flatbuffer")?;
    response_from_fb(root.unpack())
}

// ---------------------------------------------------------------------------
// Accessor-based decode. Each `*_from_table` function mirrors its unpack-based
// `*_from_fb` counterpart field for field (same defaults, enum mappings, error
// messages, and evaluation order) but reads the verified flatbuffer tables
// directly: no intermediate object tree, one allocation per owned field, and
// byte vectors are copied with a single memcpy off the accessor slice.
// ---------------------------------------------------------------------------

fn request_from_table(request: fbs::WorkerRequest<'_>) -> anyhow::Result<WorkerRequest> {
    let request = WorkerRequest {
        kind: request_kind_from_fb(request.kind())?,
        call_id: request.call_id(),
        batch: request.batch().map(batch_from_table).transpose()?,
        step_id: request.step_id(),
        session_id: request.session_id().map(RequestId),
        copies: request.copies().map(|items| {
            items
                .iter()
                .map(|copy| CacheCopy {
                    group_id: copy.group_id(),
                    source_page: BlockId(copy.src()),
                    destination_page: BlockId(copy.dst()),
                })
                .collect()
        }),
        product_handles: request
            .product_handles()
            .map(|items| items.iter().collect()),
        snapshot: request.snapshot().map(snapshot_from_table).transpose()?,
        recovery_placement: request
            .recovery_placement()
            .map(recovery_placement_from_table)
            .transpose()?,
    };
    validate_request_shape(&request)?;
    Ok(request)
}

fn response_from_table(response: fbs::WorkerResponse<'_>) -> anyhow::Result<WorkerResponse> {
    let response = WorkerResponse {
        kind: response_kind_from_fb(response.kind())?,
        call_id: response.call_id(),
        capabilities: response
            .capabilities()
            .map(capabilities_from_table)
            .transpose()?,
        completion_report: response
            .completion_report()
            .map(completion_report_from_table)
            .transpose()?,
        metrics: response.metrics().map(metrics_from_table),
        pressure: response
            .pressure()
            .map(|items| items.iter().map(pressure_from_table).collect())
            .transpose()?,
        message: response.message().map(str::to_string),
        code: response.code().map(str::to_string),
        retryable: response.retryable(),
        fatal: response.fatal(),
        phase: response.phase().map(str::to_string),
        route: response.route().map(str::to_string),
        operations: response
            .operations()
            .map(|items| {
                items
                    .iter()
                    .map(error_operation_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        snapshot: response.snapshot().map(snapshot_from_table).transpose()?,
    };
    validate_response_shape(&response)?;
    Ok(response)
}

fn batch_from_table(batch: fbs::Batch<'_>) -> anyhow::Result<Batch> {
    let batch = Batch {
        step_id: batch.step_id(),
        admissions: batch
            .admissions()
            .map(|items| {
                items
                    .iter()
                    .map(admission_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        partitions: batch
            .partitions()
            .map(|items| {
                items
                    .iter()
                    .map(partition_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        controls: batch
            .controls()
            .map(|items| {
                items
                    .iter()
                    .map(control_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        input_products: batch
            .input_products()
            .map(|items| {
                items
                    .iter()
                    .map(product_payload_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    batch.validate()?;
    Ok(batch)
}

fn partition_from_table(partition: fbs::BatchPartition<'_>) -> anyhow::Result<BatchPartition> {
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
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        request_pool_indices: partition
            .request_pool_indices()
            .map(|items| items.iter().collect())
            .unwrap_or_default(),
        kv_placements: partition
            .kv_placements()
            .map(|items| {
                items
                    .iter()
                    .map(kv_placement_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        kv_branch_placements: partition
            .kv_branch_placements()
            .map(|items| {
                items
                    .iter()
                    .map(kv_branch_placement_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        latent_placements: partition
            .latent_placements()
            .map(|items| {
                items
                    .iter()
                    .map(latent_placement_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    partition.validate()?;
    Ok(partition)
}

fn admission_from_table(admission: fbs::Admission<'_>) -> anyhow::Result<Admission> {
    let admission = Admission {
        request_key: request_key_from_table(admission.request_key(), "admission.request_key")?,
        request_pool_idx: admission.request_pool_idx(),
        digest: required_str(admission.digest(), "admission.digest")?,
        und: admission.und().map(und_admission_from_table).transpose()?,
        gen_admission: admission
            .gen_admission()
            .map(gen_admission_from_table)
            .transpose()?,
    };
    admission.validate()?;
    Ok(admission)
}

fn und_admission_from_table(admission: fbs::UndAdmission<'_>) -> anyhow::Result<UndAdmission> {
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
        kv: kv_admission_from_table(admission.kv().context("und admission has no KV metadata")?),
    })
}

fn gen_admission_from_table(admission: fbs::GenAdmission<'_>) -> anyhow::Result<GenAdmission> {
    Ok(GenAdmission {
        image: image_from_table(
            admission
                .image()
                .context("gen admission has no image spec")?,
        )?,
    })
}

fn kv_admission_from_table(admission: fbs::KvAdmission<'_>) -> KvAdmission {
    KvAdmission {
        prefix_len: admission.prefix_len(),
        group_id: admission.group_id(),
    }
}

fn kv_placement_from_table(placement: fbs::KvPlacement<'_>) -> anyhow::Result<KvPlacement> {
    Ok(KvPlacement {
        request_key: request_key_from_table(placement.request_key(), "KV placement.request_key")?,
        op_id: OpId(placement.op_id()),
        group_id: placement.group_id(),
        block_table: placement
            .block_table()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
        pages_to_zero: placement
            .pages_to_zero()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
        prefix_length: placement.prefix_length(),
        input_length: placement.input_length(),
        visible_length: placement.visible_length(),
        resulting_length: placement.resulting_length(),
    })
}

fn kv_branch_placement_from_table(
    placement: fbs::KvBranchPlacement<'_>,
) -> anyhow::Result<KvBranchPlacement> {
    Ok(KvBranchPlacement {
        request_key: request_key_from_table(
            placement.request_key(),
            "KV branch placement.request_key",
        )?,
        op_id: OpId(placement.op_id()),
        branch_index: placement.branch_index(),
        group_id: placement.group_id(),
        block_table: placement
            .block_table()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
        pages_to_zero: placement
            .pages_to_zero()
            .map(|items| items.iter().map(BlockId).collect())
            .unwrap_or_default(),
    })
}

fn latent_placement_from_table(
    placement: fbs::LatentPlacement<'_>,
) -> anyhow::Result<LatentPlacement> {
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

fn operation_from_table(operation: fbs::Operation<'_>) -> anyhow::Result<Operation> {
    let operation = Operation {
        request_key: request_key_from_table(operation.request_key(), "operation.request_key")?,
        op_id: OpId(operation.op_id()),
        parent: version_ref_from_table(operation.parent().context("operation has no parent")?)?,
        work: Work::from_variant(work_from_fb(operation.work())?),
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
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        outputs: operation
            .outputs()
            .map(|items| {
                items
                    .iter()
                    .map(product_ref_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        kv_capacity_pages: operation.kv_capacity_pages(),
        predicate: operation
            .predicate()
            .map(product_ref_from_table)
            .transpose()?,
        rng: operation.rng().map(rng_from_table).transpose()?,
        control_seq: operation.control_seq(),
        plan_digest: required_str(operation.plan_digest(), "operation.plan_digest")?,
    };
    // No per-operation validate() here: the worker's Python `from_wire` is the
    // authoritative ingress validator and recomputes the plan digest for every
    // operation (forged-digest rejection unchanged); running the SHA-256
    // recompute here too made the wire decode do the same work twice per op.
    Ok(operation)
}

fn control_from_table(envelope: fbs::ControlEnvelope<'_>) -> anyhow::Result<Control> {
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
        _ => bail!("control union is empty"),
    };
    control.validate()?;
    Ok(control)
}

fn request_key_from_table(
    request_key: Option<fbs::RequestKey<'_>>,
    label: &str,
) -> anyhow::Result<RequestKey> {
    let request_key = request_key.with_context(|| format!("{label} is missing"))?;
    Ok(RequestKey {
        authority_id: request_key.authority_id(),
        session_id: RequestId(request_key.session_id()),
        epoch: request_key.epoch(),
    })
}

fn version_ref_from_table(version: fbs::VersionRef<'_>) -> anyhow::Result<VersionRef> {
    let point = match version.point_type() {
        fbs::Point::PointFixed => {
            let fixed = version
                .point_as_point_fixed()
                .context("fixed point table is missing")?;
            Point::Fixed {
                point_index: fixed.point_index(),
                semantic_digest: required_str(
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
                producer_plan_digest: required_str(
                    device.producer_plan_digest(),
                    "point.device.producer_plan_digest",
                )?,
            }
        }
        _ => bail!("version reference point union is empty"),
    };
    Ok(VersionRef {
        request_key: request_key_from_table(version.request_key(), "version_ref.request_key")?,
        producer_op_id: OpId(version.producer_op_id()),
        point,
    })
}

fn product_ref_from_table(product: fbs::ProductRef<'_>) -> anyhow::Result<ProductRef> {
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

fn rng_from_table(rng: fbs::Rng<'_>) -> anyhow::Result<Rng> {
    Ok(Rng {
        seed: rng.seed(),
        semantic_index_base: rng.semantic_index_base(),
        draw_layout: draw_layout_from_fb(rng.draw_layout())?,
    })
}

fn completion_report_from_table(
    report: fbs::CompletionReport<'_>,
) -> anyhow::Result<CompletionReport> {
    let report = CompletionReport {
        step_id: report.step_id(),
        partitions: report
            .partitions()
            .map(|items| {
                items
                    .iter()
                    .map(partition_completion_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    report.validate()?;
    Ok(report)
}

fn partition_completion_from_table(
    report: fbs::PartitionCompletion<'_>,
) -> anyhow::Result<PartitionCompletion> {
    let report = PartitionCompletion {
        partition_id: report.partition_id(),
        completions: report
            .completions()
            .map(|items| {
                items
                    .iter()
                    .map(completion_record_from_table)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        products: report
            .products()
            .map(|items| {
                items
                    .iter()
                    .map(product_payload_from_table)
                    .collect::<anyhow::Result<_>>()
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

fn completion_record_from_table(
    record: fbs::CompletionRecord<'_>,
) -> anyhow::Result<CompletionRecord> {
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
    let record = CompletionRecord {
        request_key: request_key_from_table(record.request_key(), "completion.request_key")?,
        op_id: OpId(record.op_id()),
        completion_slot_generation: record.completion_slot_generation(),
        status: op_status_from_fb(record.status())?,
        selected_point: record.selected_point(),
        logical_lengths: LogicalLengths {
            token_len: logical_lengths.token_len(),
            kv_visible_len: logical_lengths.kv_visible_len(),
            latent_len: logical_lengths.latent_len(),
            kv_reserved_len: logical_lengths.kv_reserved_len(),
            kv_initialized_len: logical_lengths.kv_initialized_len(),
            kv_committed_len: logical_lengths.kv_committed_len(),
            kv_published_len: logical_lengths.kv_published_len(),
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
        semantic_digest: required_str(record.semantic_digest(), "completion.semantic_digest")?,
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

fn product_payload_from_table(payload: fbs::ProductPayload<'_>) -> anyhow::Result<ProductPayload> {
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
) -> anyhow::Result<ErrorOperationIdentity> {
    Ok(ErrorOperationIdentity {
        request_key: request_key_from_table(
            operation.request_key(),
            "error operation.request_key",
        )?,
        op_id: OpId(operation.op_id()),
    })
}

fn capabilities_from_table(
    caps: fbs::WorkerCapabilities<'_>,
) -> anyhow::Result<WorkerCapabilities> {
    let caps = WorkerCapabilities {
        block_size: caps.block_size(),
        num_blocks: caps.num_blocks(),
        num_layers: caps.num_layers(),
        num_kv_heads: caps.num_kv_heads(),
        head_dim: caps.head_dim(),
        scratch_capacity_tokens: caps.scratch_capacity_tokens(),
        supported_work: caps
            .supported_work()
            .map(|items| {
                items
                    .iter()
                    .map(work_from_fb)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        latent_page_units: caps.latent_page_units(),
        num_latent_pages: caps.num_latent_pages(),
        latent_width: caps.latent_width(),
        latent_dtype: caps.latent_dtype().unwrap_or_default().to_string(),
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
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        kv_dtype: required_str(caps.kv_dtype(), "capabilities.kv_dtype")?,
        model_dtype: canonical_model_dtype(required_str(
            caps.model_dtype(),
            "capabilities.model_dtype",
        )?)?,
        attention_backend: required_str(
            caps.attention_backend(),
            "capabilities.attention_backend",
        )?,
        quantization: caps.quantization().map(str::to_string),
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
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        max_batch_operations: caps.max_batch_operations(),
        max_unresolved_window: caps.max_unresolved_window(),
        incremental_kv_publication: caps.incremental_kv_publication(),
        tensorized_mixed: caps.tensorized_mixed(),
        sampling_ownership: sampling_ownership_from_fb(caps.sampling_ownership())?,
        resource_classes: caps
            .resource_classes()
            .map(|items| {
                items
                    .iter()
                    .map(resource_class_from_fb)
                    .collect::<anyhow::Result<_>>()
            })
            .transpose()?
            .unwrap_or_default(),
        model_identity: caps
            .model_identity()
            .map(str::to_string)
            .context("capabilities.model_identity is missing")?,
        weight_digest: caps
            .weight_digest()
            .map(str::to_string)
            .context("capabilities.weight_digest is missing")?,
        protocol_layout_digest: caps
            .protocol_layout_digest()
            .map(str::to_string)
            .context("capabilities.protocol_layout_digest is missing")?,
    };
    caps.validate()?;
    Ok(caps)
}

fn sampling_from_table(sampling: fbs::SamplingParams<'_>) -> anyhow::Result<SamplingParams> {
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

fn image_from_table(image: fbs::ImageParams<'_>) -> anyhow::Result<uniserve_core::ImageParams> {
    for value in [
        image.cfg_text_scale(),
        image.cfg_img_scale(),
        image.cfg_renorm_min(),
        image.cfg_interval_lo(),
        image.cfg_interval_hi(),
        image.timestep_shift(),
    ] {
        anyhow::ensure!(value.is_finite(), "image parameters must be finite");
    }
    Ok(uniserve_core::ImageParams {
        steps: image.steps(),
        cfg_text_scale: image.cfg_text_scale(),
        cfg_img_scale: image.cfg_img_scale(),
        cfg_renorm_type: required_str(image.cfg_renorm_type(), "image.cfg_renorm_type")?,
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

fn metrics_from_table(metrics: fbs::WorkerMetrics<'_>) -> WorkerMetrics {
    WorkerMetrics {
        executes: metrics.executes(),
        operations_total: metrics.operations_total(),
        exec_us_total: metrics.exec_us_total(),
        last_exec_us: metrics.last_exec_us(),
        operation_counts: map_from_table(metrics.operation_counts()),
        operation_us: map_from_table(metrics.operation_us()),
        control_ok: map_from_table(metrics.control_ok()),
        control_err: map_from_table(metrics.control_err()),
        error_counts: map_from_table(metrics.error_counts()),
        cuda_graph_captures: metrics.cuda_graph_captures(),
        cuda_graph_replays: metrics.cuda_graph_replays(),
        cuda_graph_misses: metrics.cuda_graph_misses(),
        cuda_graph_fallbacks: metrics.cuda_graph_fallbacks(),
        cuda_graph_unpadded_tokens: metrics.cuda_graph_unpadded_tokens(),
        cuda_graph_padded_tokens: metrics.cuda_graph_padded_tokens(),
        cuda_graph_runtime_mode_counts: map_from_table(metrics.cuda_graph_runtime_mode_counts()),
        forward: metrics.forward().map(forward_stats_from_table),
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

fn kv_group_from_table(group: fbs::KvGroupSpec<'_>) -> anyhow::Result<KvCacheGroupSpec> {
    let kind = if group.kind() == fbs::KvGroupKind::Full {
        KvGroupKind::Full
    } else if group.kind() == fbs::KvGroupKind::SlidingWindow {
        KvGroupKind::SlidingWindow {
            window: group.window(),
            sink: group.sink(),
        }
    } else {
        bail!("unknown KV group kind {}", group.kind().0)
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
        pp_rank: rank.pp_rank(),
        pp_size: rank.pp_size(),
        dp_rank: rank.dp_rank(),
        dp_size: rank.dp_size(),
    }
}

fn pressure_from_table(pressure: fbs::ResourcePressure<'_>) -> anyhow::Result<ResourcePressure> {
    Ok(ResourcePressure {
        class: resource_class_from_fb(pressure.class())?,
        total: pressure.total(),
        used: pressure.used(),
        evictable: pressure.evictable(),
        free: pressure.free(),
    })
}

fn required_str(value: Option<&str>, label: &str) -> anyhow::Result<String> {
    value
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .with_context(|| format!("{label} is missing"))
}

fn snapshot_from_table(snapshot: fbs::SnapshotRef<'_>) -> anyhow::Result<SnapshotRef> {
    let snapshot = SnapshotRef {
        version: version_ref_from_table(
            snapshot
                .version()
                .context("snapshot reference has no exact version")?,
        )?,
        digest: required_str(snapshot.digest(), "snapshot digest")?,
        locator: required_str(snapshot.locator(), "snapshot locator")?,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn recovery_placement_from_table(
    placement: fbs::RecoveryPlacement<'_>,
) -> anyhow::Result<RecoveryPlacement> {
    let placement = RecoveryPlacement {
        request_key: request_key_from_table(
            placement.request_key(),
            "recovery placement.request_key",
        )?,
        request_pool_idx: placement.request_pool_idx(),
        cache_groups: placement
            .cache_groups()
            .map(|groups| {
                groups
                    .iter()
                    .map(|group| CacheGroupPlacement {
                        group_id: group.group_id(),
                        page_ids: group
                            .page_ids()
                            .map(|pages| pages.iter().map(BlockId).collect())
                            .unwrap_or_default(),
                        length: group.length(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    };
    placement.validate()?;
    Ok(placement)
}

// ---------------------------------------------------------------------------
// Request / response framing
// ---------------------------------------------------------------------------

fn request_to_fb(request: &WorkerRequest) -> anyhow::Result<fbs::WorkerRequestT> {
    Ok(fbs::WorkerRequestT {
        kind: request_kind_to_fb(request.kind),
        call_id: request.call_id,
        batch: request
            .batch
            .as_ref()
            .map(batch_to_fb)
            .transpose()?
            .map(Box::new),
        step_id: request.step_id,
        session_id: request.session_id.map(|id| id.0),
        copies: request.copies.as_ref().map(|items| {
            items
                .iter()
                .map(|copy| fbs::BlockPairT {
                    group_id: copy.group_id,
                    src: copy.source_page.0,
                    dst: copy.destination_page.0,
                })
                .collect()
        }),
        product_handles: request.product_handles.clone(),
        snapshot: request.snapshot.as_ref().map(snapshot_to_fb).map(Box::new),
        recovery_placement: request
            .recovery_placement
            .as_ref()
            .map(recovery_placement_to_fb)
            .map(Box::new),
    })
}

#[cfg(test)]
fn request_from_fb(request: fbs::WorkerRequestT) -> anyhow::Result<WorkerRequest> {
    let request = WorkerRequest {
        kind: request_kind_from_fb(request.kind)?,
        call_id: request.call_id,
        batch: request
            .batch
            .map(|batch| batch_from_fb(*batch))
            .transpose()?,
        step_id: request.step_id,
        session_id: request.session_id.map(RequestId),
        copies: request.copies.map(|items| {
            items
                .into_iter()
                .map(|copy| CacheCopy {
                    group_id: copy.group_id,
                    source_page: BlockId(copy.src),
                    destination_page: BlockId(copy.dst),
                })
                .collect()
        }),
        product_handles: request.product_handles,
        snapshot: request
            .snapshot
            .map(|value| snapshot_from_fb(*value))
            .transpose()?,
        recovery_placement: request
            .recovery_placement
            .map(|value| recovery_placement_from_fb(*value))
            .transpose()?,
    };
    validate_request_shape(&request)?;
    Ok(request)
}

fn validate_request_shape(request: &WorkerRequest) -> anyhow::Result<()> {
    let payload_count = usize::from(request.batch.is_some())
        + usize::from(request.step_id.is_some())
        + usize::from(request.session_id.is_some())
        + usize::from(request.copies.is_some())
        + usize::from(request.product_handles.is_some())
        + usize::from(request.snapshot.is_some())
        + usize::from(request.recovery_placement.is_some());
    match request.kind {
        RequestKind::Execute => {
            let batch = request
                .batch
                .as_ref()
                .context("execute request has no batch")?;
            anyhow::ensure!(
                payload_count == 1,
                "execute request carries unrelated control fields"
            );
            batch.validate()?;
        }
        RequestKind::PollCompletions => anyhow::ensure!(
            request.step_id.is_some() && payload_count == 1,
            "poll_completions requires exactly one step id"
        ),
        RequestKind::DropSession => anyhow::ensure!(
            request.session_id.is_some() && payload_count == 1,
            "drop_session requires exactly one session id"
        ),
        RequestKind::CopyKv => {
            anyhow::ensure!(
                request.copies.is_some() && payload_count == 1,
                "copy_kv requires exactly one cache-copy list"
            );
            for copy in request.copies.as_deref().unwrap_or_default() {
                copy.validate()?;
            }
        }
        RequestKind::ReleaseProducts => anyhow::ensure!(
            request.product_handles.is_some() && payload_count == 1,
            "release_products requires exactly one handle list"
        ),
        RequestKind::SnapshotSession => {
            anyhow::ensure!(
                request.recovery_placement.is_some() && payload_count == 1,
                "snapshot_session requires exactly one recovery placement"
            );
            request
                .recovery_placement
                .as_ref()
                .expect("checked recovery placement")
                .validate()?;
        }
        RequestKind::RestoreSession => {
            anyhow::ensure!(
                request.snapshot.is_some()
                    && request.recovery_placement.is_some()
                    && payload_count == 2,
                "restore_session requires one snapshot reference and recovery placement"
            );
            request
                .snapshot
                .as_ref()
                .expect("checked snapshot")
                .validate()?;
            request
                .recovery_placement
                .as_ref()
                .expect("checked recovery placement")
                .validate()?;
        }
        RequestKind::GetCapabilities
        | RequestKind::Shutdown
        | RequestKind::GetMetrics
        | RequestKind::GetPressure => anyhow::ensure!(
            payload_count == 0,
            "control request carries an unexpected payload"
        ),
    }
    Ok(())
}

fn response_to_fb(response: &WorkerResponse) -> anyhow::Result<fbs::WorkerResponseT> {
    Ok(fbs::WorkerResponseT {
        kind: response_kind_to_fb(response.kind),
        call_id: response.call_id,
        capabilities: response
            .capabilities
            .as_ref()
            .map(capabilities_to_fb)
            .transpose()?
            .map(Box::new),
        completion_report: response
            .completion_report
            .as_ref()
            .map(completion_report_to_fb)
            .transpose()?
            .map(Box::new),
        metrics: response.metrics.as_ref().map(metrics_to_fb).map(Box::new),
        pressure: response
            .pressure
            .as_ref()
            .map(|items| items.iter().map(pressure_to_fb).collect()),
        message: response.message.clone(),
        code: response.code.clone(),
        retryable: response.retryable,
        fatal: response.fatal,
        phase: response.phase.clone(),
        route: response.route.clone(),
        operations: Some(
            response
                .operations
                .iter()
                .map(error_operation_to_fb)
                .collect(),
        ),
        snapshot: response.snapshot.as_ref().map(snapshot_to_fb).map(Box::new),
    })
}

#[cfg(test)]
fn response_from_fb(response: fbs::WorkerResponseT) -> anyhow::Result<WorkerResponse> {
    let response = WorkerResponse {
        kind: response_kind_from_fb(response.kind)?,
        call_id: response.call_id,
        capabilities: response
            .capabilities
            .map(|caps| capabilities_from_fb(*caps))
            .transpose()?,
        completion_report: response
            .completion_report
            .map(|report| completion_report_from_fb(*report))
            .transpose()?,
        metrics: response.metrics.map(|metrics| metrics_from_fb(*metrics)),
        pressure: response
            .pressure
            .map(|items| items.into_iter().map(pressure_from_fb).collect())
            .transpose()?,
        message: response.message,
        code: response.code,
        retryable: response.retryable,
        fatal: response.fatal,
        phase: response.phase,
        route: response.route,
        operations: response
            .operations
            .unwrap_or_default()
            .into_iter()
            .map(error_operation_from_fb)
            .collect::<anyhow::Result<_>>()?,
        snapshot: response
            .snapshot
            .map(|value| snapshot_from_fb(*value))
            .transpose()?,
    };
    validate_response_shape(&response)?;
    Ok(response)
}

fn validate_response_shape(response: &WorkerResponse) -> anyhow::Result<()> {
    let payload_count = usize::from(response.capabilities.is_some())
        + usize::from(response.completion_report.is_some())
        + usize::from(response.metrics.is_some())
        + usize::from(response.pressure.is_some())
        + usize::from(response.snapshot.is_some());
    let carries_error = response.message.is_some()
        || response.code.is_some()
        || response.retryable.is_some()
        || response.fatal.is_some()
        || response.phase.is_some()
        || response.route.is_some()
        || !response.operations.is_empty();
    match response.kind {
        ResponseKind::Capabilities => anyhow::ensure!(
            response.capabilities.is_some() && payload_count == 1 && !carries_error,
            "capabilities response has the wrong payload"
        ),
        ResponseKind::Result => anyhow::ensure!(
            response.completion_report.is_some() && payload_count == 1 && !carries_error,
            "completion response has the wrong payload"
        ),
        ResponseKind::Metrics => anyhow::ensure!(
            response.metrics.is_some() && payload_count == 1 && !carries_error,
            "metrics response has the wrong payload"
        ),
        ResponseKind::Pressure => anyhow::ensure!(
            response.pressure.is_some() && payload_count == 1 && !carries_error,
            "pressure response has the wrong payload"
        ),
        ResponseKind::Snapshot => anyhow::ensure!(
            response.snapshot.is_some() && payload_count == 1 && !carries_error,
            "snapshot response has the wrong payload"
        ),
        ResponseKind::Ok => anyhow::ensure!(
            payload_count == 0 && !carries_error,
            "ok response carries an unexpected payload"
        ),
        ResponseKind::Error => anyhow::ensure!(
            payload_count == 0
                && response
                    .message
                    .as_deref()
                    .is_some_and(|message| !message.is_empty())
                && response
                    .code
                    .as_deref()
                    .is_some_and(|code| !code.is_empty())
                && response.retryable.is_some()
                && response.fatal.is_some(),
            "error response requires a code, message, retryability, and fatality"
        ),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch, operations, admissions, controls
// ---------------------------------------------------------------------------

fn batch_to_fb(batch: &Batch) -> anyhow::Result<fbs::BatchT> {
    batch.validate()?;
    Ok(fbs::BatchT {
        step_id: batch.step_id,
        admissions: Some(
            batch
                .admissions
                .iter()
                .map(admission_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
        partitions: Some(
            batch
                .partitions
                .iter()
                .map(partition_to_fb)
                .collect::<anyhow::Result<_>>()?,
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

fn partition_to_fb(partition: &BatchPartition) -> anyhow::Result<fbs::BatchPartitionT> {
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
                .collect::<anyhow::Result<_>>()?,
        ),
        request_pool_indices: Some(partition.request_pool_indices.clone()),
        kv_placements: Some(
            partition
                .kv_placements
                .iter()
                .map(kv_placement_to_fb)
                .collect(),
        ),
        kv_branch_placements: Some(
            partition
                .kv_branch_placements
                .iter()
                .map(kv_branch_placement_to_fb)
                .collect(),
        ),
        latent_placements: Some(
            partition
                .latent_placements
                .iter()
                .map(latent_placement_to_fb)
                .collect(),
        ),
    })
}

#[cfg(test)]
fn batch_from_fb(batch: fbs::BatchT) -> anyhow::Result<Batch> {
    let batch = Batch {
        step_id: batch.step_id,
        admissions: batch
            .admissions
            .unwrap_or_default()
            .into_iter()
            .map(admission_from_fb)
            .collect::<anyhow::Result<_>>()?,
        partitions: batch
            .partitions
            .unwrap_or_default()
            .into_iter()
            .map(partition_from_fb)
            .collect::<anyhow::Result<_>>()?,
        controls: batch
            .controls
            .unwrap_or_default()
            .into_iter()
            .map(|envelope| control_from_fb(envelope.control))
            .collect::<anyhow::Result<_>>()?,
        input_products: batch
            .input_products
            .unwrap_or_default()
            .into_iter()
            .map(product_payload_from_fb)
            .collect::<anyhow::Result<_>>()?,
    };
    batch.validate()?;
    Ok(batch)
}

#[cfg(test)]
fn partition_from_fb(partition: fbs::BatchPartitionT) -> anyhow::Result<BatchPartition> {
    let partition = BatchPartition {
        partition_id: partition.partition_id,
        submission_group: partition.submission_group,
        collective_seq: partition.collective_seq,
        domain: domain_from_fb(partition.domain)?,
        route: RouteId(partition.route),
        execution: execution_capability_from_fb(partition.execution)?,
        attention: attention_regime_from_fb(partition.attention)?,
        shape_class: partition.shape_class,
        operations: partition
            .operations
            .unwrap_or_default()
            .into_iter()
            .map(operation_from_fb)
            .collect::<anyhow::Result<_>>()?,
        request_pool_indices: partition.request_pool_indices.unwrap_or_default(),
        kv_placements: partition
            .kv_placements
            .unwrap_or_default()
            .into_iter()
            .map(kv_placement_from_fb)
            .collect::<anyhow::Result<_>>()?,
        kv_branch_placements: partition
            .kv_branch_placements
            .unwrap_or_default()
            .into_iter()
            .map(kv_branch_placement_from_fb)
            .collect::<anyhow::Result<_>>()?,
        latent_placements: partition
            .latent_placements
            .unwrap_or_default()
            .into_iter()
            .map(latent_placement_from_fb)
            .collect::<anyhow::Result<_>>()?,
    };
    partition.validate()?;
    Ok(partition)
}

fn admission_to_fb(admission: &Admission) -> anyhow::Result<fbs::AdmissionT> {
    admission.validate()?;
    Ok(fbs::AdmissionT {
        request_key: Some(Box::new(request_key_to_fb(admission.request_key))),
        request_pool_idx: admission.request_pool_idx,
        digest: Some(admission.digest.clone()),
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
    })
}

#[cfg(test)]
fn admission_from_fb(admission: fbs::AdmissionT) -> anyhow::Result<Admission> {
    let admission = Admission {
        request_key: request_key_from_fb(admission.request_key, "admission.request_key")?,
        request_pool_idx: admission.request_pool_idx,
        digest: required_string(admission.digest, "admission.digest")?,
        und: admission
            .und
            .map(|und| und_admission_from_fb(*und))
            .transpose()?,
        gen_admission: admission
            .gen_admission
            .map(|branch| gen_admission_from_fb(*branch))
            .transpose()?,
    };
    admission.validate()?;
    Ok(admission)
}

fn und_admission_to_fb(admission: &UndAdmission) -> anyhow::Result<fbs::UndAdmissionT> {
    Ok(fbs::UndAdmissionT {
        sampling: Some(Box::new(sampling_to_fb(&admission.sampling)?)),
        negative_token_ids: Some(admission.negative_token_ids.clone()),
        finish_token_ids: Some(admission.finish_token_ids.clone()),
        kv: Some(Box::new(kv_admission_to_fb(&admission.kv))),
    })
}

#[cfg(test)]
fn und_admission_from_fb(admission: fbs::UndAdmissionT) -> anyhow::Result<UndAdmission> {
    Ok(UndAdmission {
        sampling: sampling_from_fb(
            *admission
                .sampling
                .context("und admission has no sampling spec")?,
        )?,
        negative_token_ids: admission.negative_token_ids.unwrap_or_default(),
        finish_token_ids: admission.finish_token_ids.unwrap_or_default(),
        kv: kv_admission_from_fb(*admission.kv.context("und admission has no KV metadata")?),
    })
}

fn gen_admission_to_fb(admission: &GenAdmission) -> fbs::GenAdmissionT {
    fbs::GenAdmissionT {
        image: Some(Box::new(image_to_fb(&admission.image))),
    }
}

#[cfg(test)]
fn gen_admission_from_fb(admission: fbs::GenAdmissionT) -> anyhow::Result<GenAdmission> {
    Ok(GenAdmission {
        image: image_from_fb(*admission.image.context("gen admission has no image spec")?)?,
    })
}

fn kv_admission_to_fb(admission: &KvAdmission) -> fbs::KvAdmissionT {
    fbs::KvAdmissionT {
        prefix_len: admission.prefix_len,
        group_id: admission.group_id,
    }
}

fn recovery_placement_to_fb(placement: &RecoveryPlacement) -> fbs::RecoveryPlacementT {
    fbs::RecoveryPlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        request_pool_idx: placement.request_pool_idx,
        cache_groups: Some(
            placement
                .cache_groups
                .iter()
                .map(|group| fbs::CacheGroupPlacementT {
                    group_id: group.group_id,
                    page_ids: Some(group.page_ids.iter().map(|page| page.0).collect()),
                    length: group.length,
                })
                .collect(),
        ),
    }
}

#[cfg(test)]
fn recovery_placement_from_fb(
    placement: fbs::RecoveryPlacementT,
) -> anyhow::Result<RecoveryPlacement> {
    let placement = RecoveryPlacement {
        request_key: request_key_from_fb(placement.request_key, "recovery placement.request_key")?,
        request_pool_idx: placement.request_pool_idx,
        cache_groups: placement
            .cache_groups
            .unwrap_or_default()
            .into_iter()
            .map(|group| CacheGroupPlacement {
                group_id: group.group_id,
                page_ids: group
                    .page_ids
                    .unwrap_or_default()
                    .into_iter()
                    .map(BlockId)
                    .collect(),
                length: group.length,
            })
            .collect(),
    };
    placement.validate()?;
    Ok(placement)
}

#[cfg(test)]
fn kv_admission_from_fb(admission: fbs::KvAdmissionT) -> KvAdmission {
    KvAdmission {
        prefix_len: admission.prefix_len,
        group_id: admission.group_id,
    }
}

fn kv_placement_to_fb(placement: &KvPlacement) -> fbs::KvPlacementT {
    fbs::KvPlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        op_id: placement.op_id.0,
        group_id: placement.group_id,
        block_table: Some(placement.block_table.iter().map(|block| block.0).collect()),
        pages_to_zero: Some(
            placement
                .pages_to_zero
                .iter()
                .map(|block| block.0)
                .collect(),
        ),
        prefix_length: placement.prefix_length,
        input_length: placement.input_length,
        visible_length: placement.visible_length,
        resulting_length: placement.resulting_length,
    }
}

fn kv_branch_placement_to_fb(placement: &KvBranchPlacement) -> fbs::KvBranchPlacementT {
    fbs::KvBranchPlacementT {
        request_key: Some(Box::new(request_key_to_fb(placement.request_key))),
        op_id: placement.op_id.0,
        branch_index: placement.branch_index,
        group_id: placement.group_id,
        block_table: Some(placement.block_table.iter().map(|block| block.0).collect()),
        pages_to_zero: Some(
            placement
                .pages_to_zero
                .iter()
                .map(|block| block.0)
                .collect(),
        ),
    }
}

#[cfg(test)]
fn kv_branch_placement_from_fb(
    placement: fbs::KvBranchPlacementT,
) -> anyhow::Result<KvBranchPlacement> {
    Ok(KvBranchPlacement {
        request_key: request_key_from_fb(placement.request_key, "KV branch placement.request_key")?,
        op_id: OpId(placement.op_id),
        branch_index: placement.branch_index,
        group_id: placement.group_id,
        block_table: placement
            .block_table
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        pages_to_zero: placement
            .pages_to_zero
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
    })
}

#[cfg(test)]
fn kv_placement_from_fb(placement: fbs::KvPlacementT) -> anyhow::Result<KvPlacement> {
    Ok(KvPlacement {
        request_key: request_key_from_fb(placement.request_key, "KV placement.request_key")?,
        op_id: OpId(placement.op_id),
        group_id: placement.group_id,
        block_table: placement
            .block_table
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        pages_to_zero: placement
            .pages_to_zero
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        prefix_length: placement.prefix_length,
        input_length: placement.input_length,
        visible_length: placement.visible_length,
        resulting_length: placement.resulting_length,
    })
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

#[cfg(test)]
fn latent_placement_from_fb(placement: fbs::LatentPlacementT) -> anyhow::Result<LatentPlacement> {
    Ok(LatentPlacement {
        request_key: request_key_from_fb(placement.request_key, "latent placement.request_key")?,
        op_id: OpId(placement.op_id),
        page_table: placement.page_table.unwrap_or_default(),
        latent_units: placement.latent_units,
        height: placement.height,
        width: placement.width,
        start_step: placement.start_step,
        step_count: placement.step_count,
    })
}

fn operation_to_fb(operation: &Operation) -> anyhow::Result<fbs::OperationT> {
    operation.validate()?;
    Ok(fbs::OperationT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
        parent: Some(Box::new(version_ref_to_fb(&operation.parent))),
        work: work_to_fb(operation.work.variant()),
        route: operation.route.0,
        domain: domain_to_fb(operation.domain),
        advances_state: operation.advances_state,
        bounds: Some(Box::new(bounds_to_fb(&operation.bounds))),
        inputs: Some(operation.inputs.iter().map(product_ref_to_fb).collect()),
        outputs: Some(operation.outputs.iter().map(product_ref_to_fb).collect()),
        kv_capacity_pages: operation.kv_capacity_pages,
        predicate: operation
            .predicate
            .as_ref()
            .map(product_ref_to_fb)
            .map(Box::new),
        rng: operation.rng.as_ref().map(rng_to_fb).map(Box::new),
        control_seq: operation.control_seq,
        plan_digest: Some(operation.plan_digest.clone()),
    })
}

#[cfg(test)]
fn operation_from_fb(operation: fbs::OperationT) -> anyhow::Result<Operation> {
    let operation = Operation {
        request_key: request_key_from_fb(operation.request_key, "operation.request_key")?,
        op_id: OpId(operation.op_id),
        parent: version_ref_from_fb(*operation.parent.context("operation has no parent")?)?,
        work: Work::from_variant(work_from_fb(operation.work)?),
        route: RouteId(operation.route),
        domain: domain_from_fb(operation.domain)?,
        advances_state: operation.advances_state,
        bounds: bounds_from_fb(*operation.bounds.context("operation has no bounds")?),
        inputs: operation
            .inputs
            .unwrap_or_default()
            .into_iter()
            .map(product_ref_from_fb)
            .collect::<anyhow::Result<_>>()?,
        outputs: operation
            .outputs
            .unwrap_or_default()
            .into_iter()
            .map(product_ref_from_fb)
            .collect::<anyhow::Result<_>>()?,
        kv_capacity_pages: operation.kv_capacity_pages,
        predicate: operation
            .predicate
            .map(|predicate| product_ref_from_fb(*predicate))
            .transpose()?,
        rng: operation.rng.map(|rng| rng_from_fb(*rng)).transpose()?,
        control_seq: operation.control_seq,
        plan_digest: required_string(operation.plan_digest, "operation.plan_digest")?,
    };
    operation.validate()?;
    Ok(operation)
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

#[cfg(test)]
fn control_from_fb(control: fbs::ControlT) -> anyhow::Result<Control> {
    let control = match control {
        fbs::ControlT::ControlCommit(commit) => Control::Commit {
            request_key: request_key_from_fb(commit.request_key, "control.commit.request_key")?,
            control_seq: commit.control_seq,
            expected_parent: version_ref_from_fb(
                *commit
                    .expected_parent
                    .context("commit control has no expected parent")?,
            )?,
            selected: version_ref_from_fb(
                *commit
                    .selected
                    .context("commit control has no selected version")?,
            )?,
            public_event_limit: commit.public_event_limit,
            disposition: disposition_from_fb(commit.disposition)?,
        },
        fbs::ControlT::ControlClose(close) => Control::Close {
            request_key: request_key_from_fb(close.request_key, "control.close.request_key")?,
            control_seq: close.control_seq,
            cutoff: version_ref_from_fb(*close.cutoff.context("close control has no cutoff")?)?,
            reason: close_reason_from_fb(close.reason)?,
        },
        fbs::ControlT::ControlRelease(release) => Control::Release {
            request_key: request_key_from_fb(release.request_key, "control.release.request_key")?,
            op_id: OpId(release.op_id),
        },
        fbs::ControlT::NONE => bail!("control union is empty"),
    };
    control.validate()?;
    Ok(control)
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

#[cfg(test)]
fn request_key_from_fb(
    request_key: Option<Box<fbs::RequestKeyT>>,
    label: &str,
) -> anyhow::Result<RequestKey> {
    let request_key = request_key.with_context(|| format!("{label} is missing"))?;
    Ok(RequestKey {
        authority_id: request_key.authority_id,
        session_id: RequestId(request_key.session_id),
        epoch: request_key.epoch,
    })
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
                semantic_digest: Some(semantic_digest.clone()),
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
                producer_plan_digest: Some(producer_plan_digest.clone()),
            })),
        },
    }
}

#[cfg(test)]
fn version_ref_from_fb(version: fbs::VersionRefT) -> anyhow::Result<VersionRef> {
    let point = match version.point {
        fbs::PointT::PointFixed(fixed) => Point::Fixed {
            point_index: fixed.point_index,
            semantic_digest: required_string(fixed.semantic_digest, "point.fixed.semantic_digest")?,
        },
        fbs::PointT::PointDevice(device) => Point::Device {
            point_index: device.point_index,
            selected_point: device
                .selected_point
                .map(|value| product_ref_from_fb(*value))
                .transpose()?,
            producer_plan_digest: required_string(
                device.producer_plan_digest,
                "point.device.producer_plan_digest",
            )?,
        },
        fbs::PointT::NONE => bail!("version reference point union is empty"),
    };
    Ok(VersionRef {
        request_key: request_key_from_fb(version.request_key, "version_ref.request_key")?,
        producer_op_id: OpId(version.producer_op_id),
        point,
    })
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

#[cfg(test)]
fn product_ref_from_fb(product: fbs::ProductRefT) -> anyhow::Result<ProductRef> {
    let point_range = product
        .point_range
        .context("product reference has no point range")?;
    Ok(ProductRef {
        request_key: request_key_from_fb(product.request_key, "product_ref.request_key")?,
        producer_op_id: OpId(product.producer_op_id),
        output_index: product.output_index,
        generation: product.generation,
        kind: product_kind_from_fb(product.kind)?,
        storage_class: storage_class_from_fb(product.storage_class)?,
        dtype: dtype_from_fb(product.dtype)?,
        shape_bound: shape_bound_from_fb(
            *product
                .shape_bound
                .context("product reference has no shape bound")?,
        ),
        point_range: PointRange {
            base_point: point_range.base_point,
            max_points: point_range.max_points,
        },
    })
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

#[cfg(test)]
fn shape_bound_from_fb(shape: fbs::ShapeBoundT) -> ShapeBound {
    ShapeBound {
        dims: shape
            .dims
            .unwrap_or_default()
            .into_iter()
            .map(|dim| {
                if dim.device {
                    DimBound::Device { max: dim.value }
                } else {
                    DimBound::Static(dim.value)
                }
            })
            .collect(),
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

#[cfg(test)]
fn bounds_from_fb(bounds: fbs::BoundsT) -> Bounds {
    Bounds {
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

#[cfg(test)]
fn rng_from_fb(rng: fbs::RngT) -> anyhow::Result<Rng> {
    Ok(Rng {
        seed: rng.seed,
        semantic_index_base: rng.semantic_index_base,
        draw_layout: draw_layout_from_fb(rng.draw_layout)?,
    })
}

// ---------------------------------------------------------------------------
// Completion report and records
// ---------------------------------------------------------------------------

fn completion_report_to_fb(report: &CompletionReport) -> anyhow::Result<fbs::CompletionReportT> {
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

#[cfg(test)]
fn completion_report_from_fb(report: fbs::CompletionReportT) -> anyhow::Result<CompletionReport> {
    let report = CompletionReport {
        step_id: report.step_id,
        partitions: report
            .partitions
            .unwrap_or_default()
            .into_iter()
            .map(partition_completion_from_fb)
            .collect::<anyhow::Result<_>>()?,
    };
    report.validate()?;
    Ok(report)
}

#[cfg(test)]
fn partition_completion_from_fb(
    report: fbs::PartitionCompletionT,
) -> anyhow::Result<PartitionCompletion> {
    let report = PartitionCompletion {
        partition_id: report.partition_id,
        completions: report
            .completions
            .unwrap_or_default()
            .into_iter()
            .map(completion_record_from_fb)
            .collect::<anyhow::Result<_>>()?,
        products: report
            .products
            .unwrap_or_default()
            .into_iter()
            .map(product_payload_from_fb)
            .collect::<anyhow::Result<_>>()?,
        registration: RegistrationAck {
            visible: report
                .registration
                .map(|ack| ack.visible)
                .context("completion report has no registration acknowledgement")?,
        },
        worker_exec_us: report.worker_exec_us,
        forward_stats: report
            .forward_stats
            .map(|stats| forward_stats_from_fb(*stats)),
    };
    report.validate()?;
    Ok(report)
}

fn completion_record_to_fb(record: &CompletionRecord) -> fbs::CompletionRecordT {
    fbs::CompletionRecordT {
        request_key: Some(Box::new(request_key_to_fb(record.request_key))),
        op_id: record.op_id.0,
        completion_slot_generation: record.completion_slot_generation,
        status: op_status_to_fb(record.status),
        selected_point: record.selected_point,
        logical_lengths: Some(Box::new(fbs::LogicalLengthsT {
            token_len: record.logical_lengths.token_len,
            kv_visible_len: record.logical_lengths.kv_visible_len,
            latent_len: record.logical_lengths.latent_len,
            kv_reserved_len: record.logical_lengths.kv_reserved_len,
            kv_initialized_len: record.logical_lengths.kv_initialized_len,
            kv_committed_len: record.logical_lengths.kv_committed_len,
            kv_published_len: record.logical_lengths.kv_published_len,
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
        semantic_digest: Some(record.semantic_digest.clone()),
        error_code: record.error_code.map(error_code_to_fb),
        timing_counters: Some(Box::new(fbs::TimingCountersT {
            queued_us: record.timing_counters.queued_us,
            device_us: record.timing_counters.device_us,
            copy_us: record.timing_counters.copy_us,
            host_us: record.timing_counters.host_us,
        })),
    }
}

#[cfg(test)]
fn completion_record_from_fb(record: fbs::CompletionRecordT) -> anyhow::Result<CompletionRecord> {
    let logical_lengths = record
        .logical_lengths
        .context("completion record has no logical lengths")?;
    let token_span = record
        .token_span
        .context("completion record has no token span")?;
    let finish_flags = record
        .finish_flags
        .context("completion record has no finish flags")?;
    let timing_counters = record
        .timing_counters
        .context("completion record has no timing counters")?;
    let record = CompletionRecord {
        request_key: request_key_from_fb(record.request_key, "completion.request_key")?,
        op_id: OpId(record.op_id),
        completion_slot_generation: record.completion_slot_generation,
        status: op_status_from_fb(record.status)?,
        selected_point: record.selected_point,
        logical_lengths: LogicalLengths {
            token_len: logical_lengths.token_len,
            kv_visible_len: logical_lengths.kv_visible_len,
            latent_len: logical_lengths.latent_len,
            kv_reserved_len: logical_lengths.kv_reserved_len,
            kv_initialized_len: logical_lengths.kv_initialized_len,
            kv_committed_len: logical_lengths.kv_committed_len,
            kv_published_len: logical_lengths.kv_published_len,
        },
        token_span: TokenSpan {
            base: token_span.base,
            len: token_span.len,
        },
        committed_tokens: record.committed_tokens.unwrap_or_default(),
        finish_flags: FinishFlags {
            eos: finish_flags.eos,
            length: finish_flags.length,
            stop: finish_flags.stop,
        },
        product_generations: record.product_generations.unwrap_or_default(),
        semantic_digest: required_string(record.semantic_digest, "completion.semantic_digest")?,
        error_code: record.error_code.map(error_code_from_fb).transpose()?,
        timing_counters: TimingCounters {
            queued_us: timing_counters.queued_us,
            device_us: timing_counters.device_us,
            copy_us: timing_counters.copy_us,
            host_us: timing_counters.host_us,
        },
    };
    record.validate()?;
    Ok(record)
}

fn product_payload_to_fb(payload: &ProductPayload) -> fbs::ProductPayloadT {
    fbs::ProductPayloadT {
        product: Some(Box::new(product_ref_to_fb(&payload.product))),
        bytes: Some(payload.bytes.clone()),
    }
}

#[cfg(test)]
fn product_payload_from_fb(payload: fbs::ProductPayloadT) -> anyhow::Result<ProductPayload> {
    let payload = ProductPayload {
        product: product_ref_from_fb(
            *payload
                .product
                .context("product payload has no product reference")?,
        )?,
        bytes: payload.bytes.unwrap_or_default(),
    };
    payload.validate()?;
    Ok(payload)
}

fn error_operation_to_fb(operation: &ErrorOperationIdentity) -> fbs::ErrorOperationIdentityT {
    fbs::ErrorOperationIdentityT {
        request_key: Some(Box::new(request_key_to_fb(operation.request_key))),
        op_id: operation.op_id.0,
    }
}

#[cfg(test)]
fn error_operation_from_fb(
    operation: fbs::ErrorOperationIdentityT,
) -> anyhow::Result<ErrorOperationIdentity> {
    Ok(ErrorOperationIdentity {
        request_key: request_key_from_fb(operation.request_key, "error operation.request_key")?,
        op_id: OpId(operation.op_id),
    })
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

fn capabilities_to_fb(caps: &WorkerCapabilities) -> anyhow::Result<fbs::WorkerCapabilitiesT> {
    caps.validate()?;
    Ok(fbs::WorkerCapabilitiesT {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        num_kv_heads: caps.num_kv_heads,
        head_dim: caps.head_dim,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
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
        latent_dtype: Some(caps.latent_dtype.clone()),
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
        kv_dtype: Some(caps.kv_dtype.clone()),
        model_dtype: Some(caps.model_dtype.clone()),
        attention_backend: Some(caps.attention_backend.clone()),
        quantization: caps.quantization.clone(),
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
        max_unresolved_window: caps.max_unresolved_window,
        incremental_kv_publication: caps.incremental_kv_publication,
        tensorized_mixed: caps.tensorized_mixed,
        sampling_ownership: sampling_ownership_to_fb(caps.sampling_ownership),
        resource_classes: Some(
            caps.resource_classes
                .iter()
                .copied()
                .map(resource_class_to_fb)
                .collect(),
        ),
        model_identity: Some(caps.model_identity.clone()),
        weight_digest: Some(caps.weight_digest.clone()),
        protocol_layout_digest: Some(caps.protocol_layout_digest.clone()),
    })
}

#[cfg(test)]
fn capabilities_from_fb(caps: fbs::WorkerCapabilitiesT) -> anyhow::Result<WorkerCapabilities> {
    let caps = WorkerCapabilities {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        num_kv_heads: caps.num_kv_heads,
        head_dim: caps.head_dim,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
        supported_work: caps
            .supported_work
            .unwrap_or_default()
            .into_iter()
            .map(work_from_fb)
            .collect::<anyhow::Result<_>>()?,
        latent_page_units: caps.latent_page_units,
        num_latent_pages: caps.num_latent_pages,
        latent_width: caps.latent_width,
        latent_dtype: caps.latent_dtype.unwrap_or_default(),
        latent_downsample: caps.latent_downsample,
        bytes_per_token: caps.bytes_per_token,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
        max_latent_feature_bytes: caps.max_latent_feature_bytes,
        max_vision_feature_bytes: caps.max_vision_feature_bytes,
        commit_marker_tokens: caps.commit_marker_tokens,
        gen_rope_advance: caps.gen_rope_advance,
        max_cfg_branches: caps.max_cfg_branches,
        groups: caps
            .groups
            .unwrap_or_default()
            .into_iter()
            .map(kv_group_from_fb)
            .collect::<anyhow::Result<_>>()?,
        kv_dtype: required_string(caps.kv_dtype, "capabilities.kv_dtype")?,
        model_dtype: canonical_model_dtype(required_string(
            caps.model_dtype,
            "capabilities.model_dtype",
        )?)?,
        attention_backend: required_string(
            caps.attention_backend,
            "capabilities.attention_backend",
        )?,
        quantization: caps.quantization,
        rank: caps
            .rank
            .map(|rank| rank_from_fb(*rank))
            .context("capabilities have no rank")?,
        pipeline_depth: caps.pipeline_depth,
        encoder_cache_budget: caps.encoder_cache_budget,
        supported_controls: caps
            .supported_controls
            .unwrap_or_default()
            .into_iter()
            .map(request_kind_from_fb)
            .collect::<anyhow::Result<_>>()?,
        max_batch_operations: caps.max_batch_operations,
        max_unresolved_window: caps.max_unresolved_window,
        incremental_kv_publication: caps.incremental_kv_publication,
        tensorized_mixed: caps.tensorized_mixed,
        sampling_ownership: sampling_ownership_from_fb(caps.sampling_ownership)?,
        resource_classes: caps
            .resource_classes
            .unwrap_or_default()
            .into_iter()
            .map(resource_class_from_fb)
            .collect::<anyhow::Result<_>>()?,
        model_identity: caps
            .model_identity
            .context("capabilities.model_identity is missing")?,
        weight_digest: caps
            .weight_digest
            .context("capabilities.weight_digest is missing")?,
        protocol_layout_digest: caps
            .protocol_layout_digest
            .context("capabilities.protocol_layout_digest is missing")?,
    };
    caps.validate()?;
    Ok(caps)
}

fn canonical_model_dtype(value: String) -> anyhow::Result<String> {
    anyhow::ensure!(
        matches!(value.as_str(), "float16" | "bfloat16" | "float32"),
        "capabilities.model_dtype is not canonical: {value:?}"
    );
    Ok(value)
}

// ---------------------------------------------------------------------------
// Kept framing: sampling, image, metrics, ranks, pressure, snapshots
// ---------------------------------------------------------------------------

fn sampling_to_fb(sampling: &SamplingParams) -> anyhow::Result<fbs::SamplingParamsT> {
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

#[cfg(test)]
fn sampling_from_fb(sampling: fbs::SamplingParamsT) -> anyhow::Result<SamplingParams> {
    let sampling = SamplingParams {
        temperature: sampling.temperature,
        top_k: sampling.top_k,
        top_p: sampling.top_p,
        ignore_eos: sampling.ignore_eos,
        seed: sampling.seed,
        min_p: sampling.min_p,
        repetition_penalty: sampling.repetition_penalty,
        frequency_penalty: sampling.frequency_penalty,
        presence_penalty: sampling.presence_penalty,
        logit_bias: sampling
            .logit_bias
            .unwrap_or_default()
            .into_iter()
            .map(|item| (item.token_id, item.bias))
            .collect(),
        min_tokens: usize::try_from(sampling.min_tokens)
            .context("sampling.min_tokens does not fit usize")?,
        n_logprobs: sampling.n_logprobs,
        bad_words_ids: sampling
            .bad_words_ids
            .unwrap_or_default()
            .into_iter()
            .map(|items| items.items.unwrap_or_default())
            .collect(),
        allowed_token_ids: sampling.allowed_token_ids,
        return_logprobs: sampling.return_logprobs,
        return_prompt_logprobs: sampling.return_prompt_logprobs,
        n_prompt_logprobs: sampling.n_prompt_logprobs,
        logprob_token_ids: sampling.logprob_token_ids.unwrap_or_default(),
        typical_p: sampling.typical_p,
        forced_token_ids: sampling.forced_token_ids.unwrap_or_default(),
    };
    validate_sampling(&sampling)?;
    Ok(sampling)
}

fn validate_sampling(sampling: &SamplingParams) -> anyhow::Result<()> {
    for value in [
        sampling.temperature,
        sampling.top_p,
        sampling.min_p,
        sampling.repetition_penalty,
        sampling.frequency_penalty,
        sampling.presence_penalty,
    ] {
        anyhow::ensure!(value.is_finite(), "sampling parameters must be finite");
    }
    for (_, value) in &sampling.logit_bias {
        anyhow::ensure!(value.is_finite(), "logit bias must be finite");
    }
    anyhow::ensure!(
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
        cfg_renorm_type: Some(image.cfg_renorm_type.clone()),
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

#[cfg(test)]
fn image_from_fb(image: fbs::ImageParamsT) -> anyhow::Result<uniserve_core::ImageParams> {
    for value in [
        image.cfg_text_scale,
        image.cfg_img_scale,
        image.cfg_renorm_min,
        image.cfg_interval_lo,
        image.cfg_interval_hi,
        image.timestep_shift,
    ] {
        anyhow::ensure!(value.is_finite(), "image parameters must be finite");
    }
    Ok(uniserve_core::ImageParams {
        steps: image.steps,
        cfg_text_scale: image.cfg_text_scale,
        cfg_img_scale: image.cfg_img_scale,
        cfg_renorm_type: required_string(image.cfg_renorm_type, "image.cfg_renorm_type")?,
        cfg_renorm_min: image.cfg_renorm_min,
        cfg_interval: (image.cfg_interval_lo, image.cfg_interval_hi),
        timestep_shift: image.timestep_shift,
        height: image.height,
        width: image.width,
        seed: image.seed,
        negative_prompt: image.negative_prompt.unwrap_or_default(),
        max_images: image.max_images,
        image_prompts: image.image_prompts.unwrap_or_default(),
        retain_images: image.retain_images,
    })
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

#[cfg(test)]
fn forward_stats_from_fb(stats: fbs::WorkerForwardStatsT) -> WorkerForwardStats {
    WorkerForwardStats {
        mode_counts: map_from_fb(stats.mode_counts),
        mode_tokens: map_from_fb(stats.mode_tokens),
        mode_us: map_from_fb(stats.mode_us),
        component_us: map_from_fb(stats.component_us),
        attention_launches: stats.attention_launches,
        attention_us: stats.attention_us,
        attention_backend_counts: map_from_fb(stats.attention_backend_counts),
        cuda_graph_captures: stats.cuda_graph_captures,
        cuda_graph_replays: stats.cuda_graph_replays,
        cuda_graph_misses: stats.cuda_graph_misses,
        cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: map_from_fb(stats.cuda_graph_runtime_mode_counts),
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
        spec_verify_path_counts: map_from_fb(stats.spec_verify_path_counts),
    }
}

fn metrics_to_fb(metrics: &WorkerMetrics) -> fbs::WorkerMetricsT {
    fbs::WorkerMetricsT {
        executes: metrics.executes,
        operations_total: metrics.operations_total,
        exec_us_total: metrics.exec_us_total,
        last_exec_us: metrics.last_exec_us,
        operation_counts: Some(map_to_fb(&metrics.operation_counts)),
        operation_us: Some(map_to_fb(&metrics.operation_us)),
        control_ok: Some(map_to_fb(&metrics.control_ok)),
        control_err: Some(map_to_fb(&metrics.control_err)),
        error_counts: Some(map_to_fb(&metrics.error_counts)),
        cuda_graph_captures: metrics.cuda_graph_captures,
        cuda_graph_replays: metrics.cuda_graph_replays,
        cuda_graph_misses: metrics.cuda_graph_misses,
        cuda_graph_fallbacks: metrics.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: metrics.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: metrics.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: Some(map_to_fb(&metrics.cuda_graph_runtime_mode_counts)),
        forward: metrics
            .forward
            .as_ref()
            .map(forward_stats_to_fb)
            .map(Box::new),
    }
}

#[cfg(test)]
fn metrics_from_fb(metrics: fbs::WorkerMetricsT) -> WorkerMetrics {
    WorkerMetrics {
        executes: metrics.executes,
        operations_total: metrics.operations_total,
        exec_us_total: metrics.exec_us_total,
        last_exec_us: metrics.last_exec_us,
        operation_counts: map_from_fb(metrics.operation_counts),
        operation_us: map_from_fb(metrics.operation_us),
        control_ok: map_from_fb(metrics.control_ok),
        control_err: map_from_fb(metrics.control_err),
        error_counts: map_from_fb(metrics.error_counts),
        cuda_graph_captures: metrics.cuda_graph_captures,
        cuda_graph_replays: metrics.cuda_graph_replays,
        cuda_graph_misses: metrics.cuda_graph_misses,
        cuda_graph_fallbacks: metrics.cuda_graph_fallbacks,
        cuda_graph_unpadded_tokens: metrics.cuda_graph_unpadded_tokens,
        cuda_graph_padded_tokens: metrics.cuda_graph_padded_tokens,
        cuda_graph_runtime_mode_counts: map_from_fb(metrics.cuda_graph_runtime_mode_counts),
        forward: metrics.forward.map(|stats| forward_stats_from_fb(*stats)),
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

#[cfg(test)]
fn map_from_fb(items: Option<Vec<fbs::StringU64PairT>>) -> BTreeMap<String, u64> {
    items
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| item.key.map(|key| (key, item.value)))
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

#[cfg(test)]
fn kv_group_from_fb(group: fbs::KvGroupSpecT) -> anyhow::Result<KvCacheGroupSpec> {
    let kind = if group.kind == fbs::KvGroupKind::Full {
        KvGroupKind::Full
    } else if group.kind == fbs::KvGroupKind::SlidingWindow {
        KvGroupKind::SlidingWindow {
            window: group.window,
            sink: group.sink,
        }
    } else {
        bail!("unknown KV group kind {}", group.kind.0)
    };
    Ok(KvCacheGroupSpec {
        group_id: group.group_id,
        block_offset: group.block_offset,
        num_blocks: group.num_blocks,
        kind,
    })
}

fn rank_to_fb(rank: RankInfo) -> fbs::RankInfoT {
    fbs::RankInfoT {
        tp_rank: rank.tp_rank,
        tp_size: rank.tp_size,
        pp_rank: rank.pp_rank,
        pp_size: rank.pp_size,
        dp_rank: rank.dp_rank,
        dp_size: rank.dp_size,
    }
}

#[cfg(test)]
fn rank_from_fb(rank: fbs::RankInfoT) -> RankInfo {
    RankInfo {
        tp_rank: rank.tp_rank,
        tp_size: rank.tp_size,
        pp_rank: rank.pp_rank,
        pp_size: rank.pp_size,
        dp_rank: rank.dp_rank,
        dp_size: rank.dp_size,
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

#[cfg(test)]
fn pressure_from_fb(pressure: fbs::ResourcePressureT) -> anyhow::Result<ResourcePressure> {
    Ok(ResourcePressure {
        class: resource_class_from_fb(pressure.class)?,
        total: pressure.total,
        used: pressure.used,
        evictable: pressure.evictable,
        free: pressure.free,
    })
}

#[cfg(test)]
fn required_string(value: Option<String>, label: &str) -> anyhow::Result<String> {
    value
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{label} is missing"))
}

fn snapshot_to_fb(snapshot: &SnapshotRef) -> fbs::SnapshotRefT {
    fbs::SnapshotRefT {
        version: Some(Box::new(version_ref_to_fb(&snapshot.version))),
        digest: Some(snapshot.digest.clone()),
        locator: Some(snapshot.locator.clone()),
    }
}

#[cfg(test)]
fn snapshot_from_fb(snapshot: fbs::SnapshotRefT) -> anyhow::Result<SnapshotRef> {
    let snapshot = SnapshotRef {
        version: version_ref_from_fb(
            *snapshot
                .version
                .context("snapshot reference has no exact version")?,
        )?,
        digest: required_string(snapshot.digest, "snapshot digest")?,
        locator: required_string(snapshot.locator, "snapshot locator")?,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

// ---------------------------------------------------------------------------
// Enum mappers
// ---------------------------------------------------------------------------

fn work_to_fb(variant: WorkVariant) -> fbs::WorkVariant {
    match variant {
        WorkVariant::TokenExtend => fbs::WorkVariant::TokenExtend,
        WorkVariant::TokenDecode => fbs::WorkVariant::TokenDecode,
        WorkVariant::TokenVerify => fbs::WorkVariant::TokenVerify,
        WorkVariant::Draft => fbs::WorkVariant::Draft,
        WorkVariant::EncodeVision => fbs::WorkVariant::EncodeVision,
        WorkVariant::EncodeLatent => fbs::WorkVariant::EncodeLatent,
        WorkVariant::TransferProduct => fbs::WorkVariant::TransferProduct,
        WorkVariant::TransferKvPublish => fbs::WorkVariant::TransferKvPublish,
        WorkVariant::TransferKvInstall => fbs::WorkVariant::TransferKvInstall,
        WorkVariant::GenTransition => fbs::WorkVariant::GenTransition,
        WorkVariant::GenFlow => fbs::WorkVariant::GenFlow,
        WorkVariant::Materialize => fbs::WorkVariant::Materialize,
    }
}

fn work_from_fb(variant: fbs::WorkVariant) -> anyhow::Result<WorkVariant> {
    for candidate in WorkVariant::ALL {
        if work_to_fb(candidate) == variant {
            return Ok(candidate);
        }
    }
    bail!("unknown work variant {}", variant.0)
}

fn domain_to_fb(domain: Domain) -> fbs::Domain {
    match domain {
        Domain::Und => fbs::Domain::Und,
        Domain::Gen => fbs::Domain::Gen,
    }
}

fn domain_from_fb(domain: fbs::Domain) -> anyhow::Result<Domain> {
    if domain == fbs::Domain::Und {
        Ok(Domain::Und)
    } else if domain == fbs::Domain::Gen {
        Ok(Domain::Gen)
    } else {
        bail!("unknown domain {}", domain.0)
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
) -> anyhow::Result<ExecutionCapability> {
    Ok(match capability {
        fbs::ExecutionCapability::DomainHomogeneous => ExecutionCapability::DomainHomogeneous,
        fbs::ExecutionCapability::TensorizedMixed => ExecutionCapability::TensorizedMixed,
        other => bail!("unknown execution capability {}", other.0),
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

fn attention_regime_from_fb(regime: fbs::AttentionRegime) -> anyhow::Result<AttentionRegime> {
    Ok(match regime {
        fbs::AttentionRegime::None => AttentionRegime::None,
        fbs::AttentionRegime::Causal => AttentionRegime::Causal,
        fbs::AttentionRegime::Bidirectional => AttentionRegime::Bidirectional,
        fbs::AttentionRegime::Hybrid => AttentionRegime::Hybrid,
        other => bail!("unknown attention regime {}", other.0),
    })
}

fn sampling_ownership_to_fb(ownership: SamplingOwnership) -> fbs::SamplingOwnership {
    match ownership {
        SamplingOwnership::DesignatedRank => fbs::SamplingOwnership::DesignatedRank,
        SamplingOwnership::DeterministicSharded => fbs::SamplingOwnership::DeterministicSharded,
    }
}

fn sampling_ownership_from_fb(
    ownership: fbs::SamplingOwnership,
) -> anyhow::Result<SamplingOwnership> {
    Ok(match ownership {
        fbs::SamplingOwnership::DesignatedRank => SamplingOwnership::DesignatedRank,
        fbs::SamplingOwnership::DeterministicSharded => SamplingOwnership::DeterministicSharded,
        other => bail!("unknown sampling ownership {}", other.0),
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

fn product_kind_from_fb(kind: fbs::ProductKind) -> anyhow::Result<ProductKind> {
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
        other => bail!("unknown product kind {}", other.0),
    })
}

fn storage_class_to_fb(class: StorageClass) -> fbs::StorageClass {
    match class {
        StorageClass::DeviceTensor => fbs::StorageClass::DeviceTensor,
        StorageClass::PagedKv => fbs::StorageClass::PagedKv,
        StorageClass::LatentArena => fbs::StorageClass::LatentArena,
        StorageClass::HostStaging => fbs::StorageClass::HostStaging,
        StorageClass::CompletionArena => fbs::StorageClass::CompletionArena,
    }
}

fn storage_class_from_fb(class: fbs::StorageClass) -> anyhow::Result<StorageClass> {
    Ok(match class {
        fbs::StorageClass::DeviceTensor => StorageClass::DeviceTensor,
        fbs::StorageClass::PagedKv => StorageClass::PagedKv,
        fbs::StorageClass::LatentArena => StorageClass::LatentArena,
        fbs::StorageClass::HostStaging => StorageClass::HostStaging,
        fbs::StorageClass::CompletionArena => StorageClass::CompletionArena,
        other => bail!("unknown storage class {}", other.0),
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

fn dtype_from_fb(dtype: fbs::DType) -> anyhow::Result<DType> {
    Ok(match dtype {
        fbs::DType::U8 => DType::U8,
        fbs::DType::U16 => DType::U16,
        fbs::DType::U32 => DType::U32,
        fbs::DType::I32 => DType::I32,
        fbs::DType::I64 => DType::I64,
        fbs::DType::F16 => DType::F16,
        fbs::DType::BF16 => DType::BF16,
        fbs::DType::F32 => DType::F32,
        other => bail!("unknown dtype {}", other.0),
    })
}

fn draw_layout_to_fb(layout: DrawLayout) -> fbs::DrawLayout {
    match layout {
        DrawLayout::TargetSampling => fbs::DrawLayout::TargetSampling,
        DrawLayout::SpeculativeProposal => fbs::DrawLayout::SpeculativeProposal,
        DrawLayout::FlowNoise => fbs::DrawLayout::FlowNoise,
    }
}

fn draw_layout_from_fb(layout: fbs::DrawLayout) -> anyhow::Result<DrawLayout> {
    Ok(match layout {
        fbs::DrawLayout::TargetSampling => DrawLayout::TargetSampling,
        fbs::DrawLayout::SpeculativeProposal => DrawLayout::SpeculativeProposal,
        fbs::DrawLayout::FlowNoise => DrawLayout::FlowNoise,
        other => bail!("unknown draw layout {}", other.0),
    })
}

fn op_status_to_fb(status: OpStatus) -> fbs::OpStatus {
    match status {
        OpStatus::Ok => fbs::OpStatus::Ok,
        OpStatus::Predicated => fbs::OpStatus::Predicated,
        OpStatus::Error => fbs::OpStatus::Error,
    }
}

fn op_status_from_fb(status: fbs::OpStatus) -> anyhow::Result<OpStatus> {
    Ok(match status {
        fbs::OpStatus::Ok => OpStatus::Ok,
        fbs::OpStatus::Predicated => OpStatus::Predicated,
        fbs::OpStatus::Error => OpStatus::Error,
        other => bail!("unknown completion status {}", other.0),
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

fn error_code_from_fb(code: fbs::ErrorCode) -> anyhow::Result<ErrorCode> {
    Ok(match code {
        fbs::ErrorCode::InvalidOperation => ErrorCode::InvalidOperation,
        fbs::ErrorCode::ResourceExhausted => ErrorCode::ResourceExhausted,
        fbs::ErrorCode::ComputeError => ErrorCode::ComputeError,
        fbs::ErrorCode::Cancelled => ErrorCode::Cancelled,
        fbs::ErrorCode::Internal => ErrorCode::Internal,
        other => bail!("unknown error code {}", other.0),
    })
}

fn disposition_to_fb(disposition: Disposition) -> fbs::Disposition {
    match disposition {
        Disposition::Publish => fbs::Disposition::Publish,
        Disposition::Retain => fbs::Disposition::Retain,
        Disposition::Discard => fbs::Disposition::Discard,
    }
}

fn disposition_from_fb(disposition: fbs::Disposition) -> anyhow::Result<Disposition> {
    Ok(match disposition {
        fbs::Disposition::Publish => Disposition::Publish,
        fbs::Disposition::Retain => Disposition::Retain,
        fbs::Disposition::Discard => Disposition::Discard,
        other => bail!("unknown disposition {}", other.0),
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

fn close_reason_from_fb(reason: fbs::CloseReason) -> anyhow::Result<CloseReason> {
    Ok(match reason {
        fbs::CloseReason::Completed => CloseReason::Completed,
        fbs::CloseReason::Cancelled => CloseReason::Cancelled,
        fbs::CloseReason::Error => CloseReason::Error,
        fbs::CloseReason::Preempted => CloseReason::Preempted,
        other => bail!("unknown close reason {}", other.0),
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
        RequestKind::GetMetrics => fbs::ReqKind::GetMetrics,
        RequestKind::GetPressure => fbs::ReqKind::GetPressure,
        RequestKind::SnapshotSession => fbs::ReqKind::SnapshotSession,
        RequestKind::RestoreSession => fbs::ReqKind::RestoreSession,
    }
}

fn request_kind_from_fb(kind: fbs::ReqKind) -> anyhow::Result<RequestKind> {
    for candidate in RequestKind::ALL {
        if request_kind_to_fb(candidate) == kind {
            return Ok(candidate);
        }
    }
    bail!("unknown request kind {}", kind.0)
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
        ResponseKind::Metrics => fbs::RespKind::Metrics,
        ResponseKind::Pressure => fbs::RespKind::Pressure,
        ResponseKind::Snapshot => fbs::RespKind::Snapshot,
    }
}

fn response_kind_from_fb(kind: fbs::RespKind) -> anyhow::Result<ResponseKind> {
    if kind == fbs::RespKind::Capabilities {
        Ok(ResponseKind::Capabilities)
    } else if kind == fbs::RespKind::Result {
        Ok(ResponseKind::Result)
    } else if kind == fbs::RespKind::Ok {
        Ok(ResponseKind::Ok)
    } else if kind == fbs::RespKind::Error {
        Ok(ResponseKind::Error)
    } else if kind == fbs::RespKind::Metrics {
        Ok(ResponseKind::Metrics)
    } else if kind == fbs::RespKind::Pressure {
        Ok(ResponseKind::Pressure)
    } else if kind == fbs::RespKind::Snapshot {
        Ok(ResponseKind::Snapshot)
    } else {
        bail!("unknown response kind {}", kind.0)
    }
}

fn resource_class_to_fb(class: ResourceClass) -> fbs::ResourceClass {
    match class {
        ResourceClass::KvBlock => fbs::ResourceClass::KvBlock,
        ResourceClass::EncoderOutput => fbs::ResourceClass::EncoderOutput,
        ResourceClass::ImageLatent => fbs::ResourceClass::ImageLatent,
        ResourceClass::Scratch => fbs::ResourceClass::Scratch,
    }
}

fn resource_class_from_fb(class: fbs::ResourceClass) -> anyhow::Result<ResourceClass> {
    if class == fbs::ResourceClass::KvBlock {
        Ok(ResourceClass::KvBlock)
    } else if class == fbs::ResourceClass::EncoderOutput {
        Ok(ResourceClass::EncoderOutput)
    } else if class == fbs::ResourceClass::ImageLatent {
        Ok(ResourceClass::ImageLatent)
    } else if class == fbs::ResourceClass::Scratch {
        Ok(ResourceClass::Scratch)
    } else {
        bail!("unknown resource class {}", class.0)
    }
}
