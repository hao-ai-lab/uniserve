//! Strict FlatBuffers codec for protocol v3.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use flatbuffers::FlatBufferBuilder;
use uniserve_core::{BlockId, KvCacheGroupSpec, KvGroupKind, RankInfo, RequestId, SamplingParams};

use crate::schema::uniserve::wire as fbs;
use crate::{
    AdapterMode, Admission, Batch, EncodeDelta, EncodeInput, EncodeKind, EncodeOperation,
    EngineCaps, ErrorOperationIdentity, ExecutionConstraints, ExecutionResult, FlowAdmission,
    FlowDelta, FlowOperation, Guidance, ImageArtifact, KvAllocation, KvLeaseDelta,
    MaterializeDelta, MaterializeInput, MaterializeKind, MaterializeOperation, MaterializedProduct,
    Operation, OperationEnvelope, OperationResult, OperationType, PublishedKv, PublishedProduct,
    RequestKind, ResourceClass, ResourcePressure, ResponseKind, ResultDelta, SequenceAdmission,
    SequenceDelta, SequenceEffect, SequenceInput, SequenceMode, SequenceOperation,
    SessionProjection, SnapshotRef, TokenInput, TokenLogprob, TokenPolicy, TokenSource,
    TransferDelta, TransferKind, TransferOperation, WorkerForwardStats, WorkerMetrics,
    WorkerRequest, WorkerResponse,
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
    request_from_fb(root.unpack())
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
    response_from_fb(root.unpack())
}

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
        session_id: request.session_id.map(|id| id.0),
        copies: request.copies.as_ref().map(|items| {
            items
                .iter()
                .map(|(source, destination)| fbs::BlockPairT {
                    src: source.0,
                    dst: destination.0,
                })
                .collect()
        }),
        adapter_id: request.adapter_id,
        adapter_path: request.adapter_path.clone(),
        product_handles: request.product_handles.clone(),
        snapshot: request.snapshot.as_ref().map(snapshot_to_fb).map(Box::new),
    })
}

fn request_from_fb(request: fbs::WorkerRequestT) -> anyhow::Result<WorkerRequest> {
    let request = WorkerRequest {
        kind: request_kind_from_fb(request.kind)?,
        call_id: request.call_id,
        batch: request
            .batch
            .map(|batch| batch_from_fb(*batch))
            .transpose()?,
        session_id: request.session_id.map(RequestId),
        copies: request.copies.map(|items| {
            items
                .into_iter()
                .map(|pair| (BlockId(pair.src), BlockId(pair.dst)))
                .collect()
        }),
        adapter_id: request.adapter_id,
        adapter_path: request.adapter_path,
        product_handles: request.product_handles,
        snapshot: request
            .snapshot
            .map(|value| snapshot_from_fb(*value))
            .transpose()?,
    };
    validate_request_shape(&request)?;
    Ok(request)
}

fn validate_request_shape(request: &WorkerRequest) -> anyhow::Result<()> {
    let payload_count = usize::from(request.batch.is_some())
        + usize::from(request.session_id.is_some())
        + usize::from(request.copies.is_some())
        + usize::from(request.adapter_id.is_some())
        + usize::from(request.product_handles.is_some())
        + usize::from(request.snapshot.is_some());
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
        RequestKind::DropSession => anyhow::ensure!(
            request.session_id.is_some() && payload_count == 1,
            "drop_session requires exactly one session id"
        ),
        RequestKind::CopyKv => anyhow::ensure!(
            request.copies.is_some() && payload_count == 1,
            "copy_kv requires exactly one block-pair list"
        ),
        RequestKind::LoadAdapter => anyhow::ensure!(
            request.adapter_id.is_some()
                && request
                    .adapter_path
                    .as_deref()
                    .is_some_and(|path| !path.is_empty())
                && payload_count == 1,
            "load_adapter requires an id and path"
        ),
        RequestKind::UnloadAdapter => anyhow::ensure!(
            request.adapter_id.is_some() && request.adapter_path.is_none() && payload_count == 1,
            "unload_adapter requires exactly one id"
        ),
        RequestKind::ReleaseProducts => anyhow::ensure!(
            request.product_handles.is_some() && payload_count == 1,
            "release_products requires exactly one handle list"
        ),
        RequestKind::SnapshotSession => anyhow::ensure!(
            request.session_id.is_some() && payload_count == 1,
            "snapshot_session requires exactly one session id"
        ),
        RequestKind::RestoreSession => anyhow::ensure!(
            request.snapshot.is_some() && payload_count == 1,
            "restore_session requires exactly one snapshot reference"
        ),
        RequestKind::GetCapabilities
        | RequestKind::Shutdown
        | RequestKind::ResetPrefixCache
        | RequestKind::GetMetrics
        | RequestKind::GetPressure => anyhow::ensure!(
            payload_count == 0 && request.adapter_path.is_none(),
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
        result: response
            .result
            .as_ref()
            .map(execution_result_to_fb)
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
                .map(|operation| fbs::ErrorOperationIdentityT {
                    session_id: operation.session_id,
                    epoch: operation.epoch,
                    op_id: operation.op_id,
                })
                .collect(),
        ),
        snapshot: response.snapshot.as_ref().map(snapshot_to_fb).map(Box::new),
    })
}

fn response_from_fb(response: fbs::WorkerResponseT) -> anyhow::Result<WorkerResponse> {
    let response = WorkerResponse {
        kind: response_kind_from_fb(response.kind)?,
        call_id: response.call_id,
        capabilities: response
            .capabilities
            .map(|caps| capabilities_from_fb(*caps))
            .transpose()?,
        result: response
            .result
            .map(|result| execution_result_from_fb(*result))
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
            .map(|operation| ErrorOperationIdentity {
                session_id: operation.session_id,
                epoch: operation.epoch,
                op_id: operation.op_id,
            })
            .collect(),
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
        + usize::from(response.result.is_some())
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
            response.result.is_some() && payload_count == 1 && !carries_error,
            "execution response has the wrong payload"
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

fn batch_to_fb(batch: &Batch) -> anyhow::Result<fbs::BatchT> {
    batch.validate()?;
    Ok(fbs::BatchT {
        protocol_version: batch.protocol_version,
        step_id: batch.step_id,
        admissions: Some(
            batch
                .admissions
                .iter()
                .map(admission_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
        projections: Some(batch.projections.iter().map(projection_to_fb).collect()),
        operations: Some(
            batch
                .operations
                .iter()
                .map(operation_envelope_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
    })
}

fn batch_from_fb(batch: fbs::BatchT) -> anyhow::Result<Batch> {
    let batch = Batch {
        protocol_version: batch.protocol_version,
        step_id: batch.step_id,
        admissions: batch
            .admissions
            .unwrap_or_default()
            .into_iter()
            .map(admission_from_fb)
            .collect::<anyhow::Result<_>>()?,
        projections: batch
            .projections
            .unwrap_or_default()
            .into_iter()
            .map(projection_from_fb)
            .collect::<anyhow::Result<_>>()?,
        operations: batch
            .operations
            .unwrap_or_default()
            .into_iter()
            .map(operation_envelope_from_fb)
            .collect::<anyhow::Result<_>>()?,
    };
    batch.validate()?;
    Ok(batch)
}

fn projection_to_fb(projection: &SessionProjection) -> fbs::SessionProjectionT {
    fbs::SessionProjectionT {
        session_id: projection.session_id.0,
        epoch: projection.epoch,
        version: projection.version,
        last_op_id: projection.last_op_id,
        admission_digest: Some(projection.admission_digest.clone()),
        source_digest: Some(projection.source_digest.clone()),
        last_sampled_token: projection.last_sampled_token,
    }
}

fn projection_from_fb(projection: fbs::SessionProjectionT) -> anyhow::Result<SessionProjection> {
    Ok(SessionProjection {
        session_id: RequestId(projection.session_id),
        epoch: projection.epoch,
        version: projection.version,
        last_op_id: projection.last_op_id,
        admission_digest: required_string(
            projection.admission_digest,
            "session projection admission digest",
        )?,
        source_digest: required_string(
            projection.source_digest,
            "session projection source digest",
        )?,
        last_sampled_token: projection.last_sampled_token,
    })
}

fn admission_to_fb(admission: &Admission) -> anyhow::Result<fbs::AdmissionT> {
    admission.validate(crate::EXECUTION_PROTOCOL_VERSION)?;
    Ok(fbs::AdmissionT {
        session_id: admission.session_id.0,
        digest: Some(admission.digest.clone()),
        sequence: admission
            .sequence
            .as_ref()
            .map(sequence_admission_to_fb)
            .transpose()?
            .map(Box::new),
        flow: admission
            .flow
            .as_ref()
            .map(flow_admission_to_fb)
            .map(Box::new),
        adapter_id: admission.adapter_id,
    })
}

fn admission_from_fb(admission: fbs::AdmissionT) -> anyhow::Result<Admission> {
    let admission = Admission {
        session_id: RequestId(admission.session_id),
        digest: required_string(admission.digest, "admission.digest")?,
        sequence: admission
            .sequence
            .map(|sequence| sequence_admission_from_fb(*sequence))
            .transpose()?,
        flow: admission
            .flow
            .map(|flow| flow_admission_from_fb(*flow))
            .transpose()?,
        adapter_id: admission.adapter_id,
    };
    admission.validate(crate::EXECUTION_PROTOCOL_VERSION)?;
    Ok(admission)
}

fn sequence_admission_to_fb(
    admission: &SequenceAdmission,
) -> anyhow::Result<fbs::SequenceAdmissionT> {
    Ok(fbs::SequenceAdmissionT {
        sampling: Some(Box::new(sampling_to_fb(&admission.sampling)?)),
        negative_token_ids: Some(admission.negative_token_ids.clone()),
        kv: Some(Box::new(kv_allocation_to_fb(&admission.kv))),
    })
}

fn sequence_admission_from_fb(
    admission: fbs::SequenceAdmissionT,
) -> anyhow::Result<SequenceAdmission> {
    Ok(SequenceAdmission {
        sampling: sampling_from_fb(
            *admission
                .sampling
                .context("sequence admission has no sampling spec")?,
        )?,
        negative_token_ids: admission.negative_token_ids.unwrap_or_default(),
        kv: kv_allocation_from_fb(
            *admission
                .kv
                .context("sequence admission has no KV allocation")?,
        ),
    })
}

fn flow_admission_to_fb(admission: &FlowAdmission) -> fbs::FlowAdmissionT {
    fbs::FlowAdmissionT {
        image: Some(Box::new(image_to_fb(&admission.image))),
    }
}

fn flow_admission_from_fb(admission: fbs::FlowAdmissionT) -> anyhow::Result<FlowAdmission> {
    Ok(FlowAdmission {
        image: image_from_fb(
            *admission
                .image
                .context("flow admission has no image spec")?,
        )?,
    })
}

fn kv_allocation_to_fb(allocation: &KvAllocation) -> fbs::KvAllocationT {
    fbs::KvAllocationT {
        block_ids: Some(allocation.block_ids.iter().map(|block| block.0).collect()),
        prefix_len: allocation.prefix_len,
        group_id: allocation.group_id,
    }
}

fn kv_allocation_from_fb(allocation: fbs::KvAllocationT) -> KvAllocation {
    KvAllocation {
        block_ids: allocation
            .block_ids
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        prefix_len: allocation.prefix_len,
        group_id: allocation.group_id,
    }
}

fn operation_envelope_to_fb(
    envelope: &OperationEnvelope,
) -> anyhow::Result<fbs::OperationEnvelopeT> {
    envelope.validate(crate::EXECUTION_PROTOCOL_VERSION)?;
    Ok(fbs::OperationEnvelopeT {
        session_id: envelope.session_id.0,
        epoch: envelope.epoch,
        op_id: envelope.op_id,
        base_version: envelope.base_version,
        digest: Some(envelope.digest.clone()),
        admission_digest: Some(envelope.admission_digest.clone()),
        model_spec_digest: Some(envelope.model_spec_digest.clone()),
        weight_digest: Some(envelope.weight_digest.clone()),
        operation: operation_to_fb(&envelope.operation),
    })
}

fn operation_envelope_from_fb(
    envelope: fbs::OperationEnvelopeT,
) -> anyhow::Result<OperationEnvelope> {
    let envelope = OperationEnvelope {
        session_id: RequestId(envelope.session_id),
        epoch: envelope.epoch,
        op_id: envelope.op_id,
        base_version: envelope.base_version,
        digest: required_string(envelope.digest, "operation.digest")?,
        admission_digest: required_string(envelope.admission_digest, "operation.admission_digest")?,
        model_spec_digest: required_string(
            envelope.model_spec_digest,
            "operation.model_spec_digest",
        )?,
        weight_digest: required_string(envelope.weight_digest, "operation.weight_digest")?,
        operation: operation_from_fb(envelope.operation)?,
    };
    envelope.validate(crate::EXECUTION_PROTOCOL_VERSION)?;
    Ok(envelope)
}

fn operation_to_fb(operation: &Operation) -> fbs::OperationT {
    match operation {
        Operation::Sequence(value) => {
            fbs::OperationT::SequenceOperation(Box::new(sequence_operation_to_fb(value)))
        }
        Operation::Flow(value) => {
            fbs::OperationT::FlowOperation(Box::new(flow_operation_to_fb(value)))
        }
        Operation::Encode(value) => {
            fbs::OperationT::EncodeOperation(Box::new(encode_operation_to_fb(value)))
        }
        Operation::Materialize(value) => {
            fbs::OperationT::MaterializeOperation(Box::new(materialize_operation_to_fb(value)))
        }
        Operation::Transfer(value) => {
            fbs::OperationT::TransferOperation(Box::new(transfer_operation_to_fb(value)))
        }
    }
}

fn operation_from_fb(operation: fbs::OperationT) -> anyhow::Result<Operation> {
    Ok(match operation {
        fbs::OperationT::SequenceOperation(value) => {
            Operation::Sequence(sequence_operation_from_fb(*value)?)
        }
        fbs::OperationT::FlowOperation(value) => Operation::Flow(flow_operation_from_fb(*value)?),
        fbs::OperationT::EncodeOperation(value) => {
            Operation::Encode(encode_operation_from_fb(*value)?)
        }
        fbs::OperationT::MaterializeOperation(value) => {
            Operation::Materialize(materialize_operation_from_fb(*value)?)
        }
        fbs::OperationT::TransferOperation(value) => {
            Operation::Transfer(transfer_operation_from_fb(*value)?)
        }
        fbs::OperationT::NONE => bail!("operation union is empty"),
    })
}

fn sequence_operation_to_fb(operation: &SequenceOperation) -> fbs::SequenceOperationT {
    fbs::SequenceOperationT {
        mode: sequence_mode_to_fb(operation.mode),
        lease: Some(Box::new(lease_to_fb(&operation.lease))),
        position_start: operation.position.0,
        position_end: operation.position.1,
        policy: Some(Box::new(policy_to_fb(&operation.policy))),
        input: match &operation.input {
            SequenceInput::Tokens(input) => {
                fbs::SequenceInputT::TokenInput(Box::new(token_input_to_fb(input)))
            }
            SequenceInput::PublishedLogits(product) => {
                fbs::SequenceInputT::PublishedLogits(Box::new(fbs::PublishedLogitsT {
                    handle: product.handle,
                    locator: Some(product.locator.clone()),
                }))
            }
        },
    }
}

fn sequence_operation_from_fb(
    operation: fbs::SequenceOperationT,
) -> anyhow::Result<SequenceOperation> {
    Ok(SequenceOperation {
        mode: sequence_mode_from_fb(operation.mode)?,
        lease: operation
            .lease
            .map(|lease| lease_from_fb(*lease))
            .context("sequence operation has no KV lease")?,
        position: (operation.position_start, operation.position_end),
        policy: operation
            .policy
            .map(|policy| policy_from_fb(*policy))
            .context("sequence operation has no token policy")?,
        input: match operation.input {
            fbs::SequenceInputT::TokenInput(input) => {
                SequenceInput::Tokens(token_input_from_fb(*input)?)
            }
            fbs::SequenceInputT::PublishedLogits(product) => {
                SequenceInput::PublishedLogits(PublishedProduct {
                    handle: product.handle,
                    locator: product.locator.unwrap_or_default(),
                })
            }
            fbs::SequenceInputT::NONE => bail!("sequence input union is empty"),
        },
    })
}

fn token_input_to_fb(input: &TokenInput) -> fbs::TokenInputT {
    fbs::TokenInputT {
        token_ids: Some(input.token_ids.clone()),
        source: token_source_to_fb(input.source),
        draft_token_ids: Some(input.draft_token_ids.clone()),
        burst_tokens: input.burst_tokens,
        stop_token_ids: Some(input.stop_token_ids.clone()),
        stop_terminal: input.stop_terminal,
        return_all_logits: input.return_all_logits,
    }
}

fn token_input_from_fb(input: fbs::TokenInputT) -> anyhow::Result<TokenInput> {
    Ok(TokenInput {
        token_ids: input.token_ids.unwrap_or_default(),
        source: token_source_from_fb(input.source)?,
        draft_token_ids: input.draft_token_ids.unwrap_or_default(),
        burst_tokens: input.burst_tokens,
        stop_token_ids: input.stop_token_ids.unwrap_or_default(),
        stop_terminal: input.stop_terminal,
        return_all_logits: input.return_all_logits,
    })
}

fn flow_operation_to_fb(operation: &FlowOperation) -> fbs::FlowOperationT {
    fbs::FlowOperationT {
        latent_handle: operation.latent_handle,
        position: operation.position,
        start_step: operation.start_step,
        step_count: operation.step_count,
        conditioning_position: operation.conditioning_position,
        conditioning: operation
            .conditioning
            .as_ref()
            .map(published_kv_to_fb)
            .map(Box::new),
        guidance: Some(Box::new(guidance_to_fb(&operation.guidance))),
        image_prompt: Some(operation.image_prompt.clone()),
    }
}

fn flow_operation_from_fb(operation: fbs::FlowOperationT) -> anyhow::Result<FlowOperation> {
    Ok(FlowOperation {
        latent_handle: operation.latent_handle,
        position: operation.position,
        start_step: operation.start_step,
        step_count: operation.step_count,
        conditioning_position: operation.conditioning_position,
        conditioning: operation
            .conditioning
            .map(|conditioning| published_kv_from_fb(*conditioning))
            .transpose()?,
        guidance: guidance_from_fb(
            *operation
                .guidance
                .context("flow operation has no guidance")?,
        )?,
        image_prompt: operation.image_prompt.unwrap_or_default(),
    })
}

fn guidance_to_fb(guidance: &Guidance) -> fbs::GuidanceT {
    fbs::GuidanceT {
        branch_count: guidance.branch_count,
        text_scale: guidance.text_scale,
        image_scale: guidance.image_scale,
        renorm_type: Some(guidance.renorm_type.clone()),
        renorm_min: guidance.renorm_min,
        interval_lo: guidance.interval.0,
        interval_hi: guidance.interval.1,
    }
}

fn guidance_from_fb(guidance: fbs::GuidanceT) -> anyhow::Result<Guidance> {
    let guidance = Guidance {
        branch_count: guidance.branch_count,
        text_scale: guidance.text_scale,
        image_scale: guidance.image_scale,
        renorm_type: required_string(guidance.renorm_type, "guidance.renorm_type")?,
        renorm_min: guidance.renorm_min,
        interval: (guidance.interval_lo, guidance.interval_hi),
    };
    for value in [
        guidance.text_scale,
        guidance.image_scale,
        guidance.renorm_min,
        guidance.interval.0,
        guidance.interval.1,
    ] {
        anyhow::ensure!(value.is_finite(), "guidance contains a non-finite value");
    }
    Ok(guidance)
}

fn encode_operation_to_fb(operation: &EncodeOperation) -> fbs::EncodeOperationT {
    fbs::EncodeOperationT {
        kind: encode_kind_to_fb(operation.kind),
        lease: Some(Box::new(lease_to_fb(&operation.lease))),
        position_start: operation.position.0,
        position_end: operation.position.1,
        conditioning_position: operation.conditioning_position,
        input: match &operation.input {
            EncodeInput::InlineImage {
                base64,
                content_hash,
            } => fbs::EncodeInputT::InlineImage(Box::new(fbs::InlineImageT {
                base64: Some(base64.clone()),
                content_hash: *content_hash,
            })),
            EncodeInput::StagedProduct {
                handle,
                content_hash,
            } => fbs::EncodeInputT::StagedProduct(Box::new(fbs::StagedProductT {
                handle: *handle,
                content_hash: *content_hash,
            })),
            EncodeInput::CachedProduct { content_hash } => {
                fbs::EncodeInputT::CachedProduct(Box::new(fbs::CachedProductT {
                    content_hash: *content_hash,
                }))
            }
        },
    }
}

fn encode_operation_from_fb(operation: fbs::EncodeOperationT) -> anyhow::Result<EncodeOperation> {
    Ok(EncodeOperation {
        kind: encode_kind_from_fb(operation.kind)?,
        lease: operation
            .lease
            .map(|lease| lease_from_fb(*lease))
            .context("encode operation has no KV lease")?,
        position: (operation.position_start, operation.position_end),
        conditioning_position: operation.conditioning_position,
        input: match operation.input {
            fbs::EncodeInputT::InlineImage(input) => EncodeInput::InlineImage {
                base64: required_string(input.base64, "inline image")?,
                content_hash: input.content_hash,
            },
            fbs::EncodeInputT::StagedProduct(input) => EncodeInput::StagedProduct {
                handle: input.handle,
                content_hash: input.content_hash,
            },
            fbs::EncodeInputT::CachedProduct(input) => EncodeInput::CachedProduct {
                content_hash: input.content_hash,
            },
            fbs::EncodeInputT::NONE => bail!("encode input union is empty"),
        },
    })
}

fn materialize_operation_to_fb(operation: &MaterializeOperation) -> fbs::MaterializeOperationT {
    fbs::MaterializeOperationT {
        kind: materialize_kind_to_fb(operation.kind),
        lease: Some(Box::new(lease_to_fb(&operation.lease))),
        position: operation.position,
        conditioning_position: operation.conditioning_position,
        policy: Some(Box::new(policy_to_fb(&operation.policy))),
        input: match &operation.input {
            MaterializeInput::Latent { handle } => {
                fbs::MaterializeInputT::LatentProduct(Box::new(fbs::LatentProductT {
                    handle: *handle,
                }))
            }
            MaterializeInput::Published(product) => {
                fbs::MaterializeInputT::PublishedProduct(Box::new(published_to_fb(product)))
            }
        },
    }
}

fn materialize_operation_from_fb(
    operation: fbs::MaterializeOperationT,
) -> anyhow::Result<MaterializeOperation> {
    Ok(MaterializeOperation {
        kind: materialize_kind_from_fb(operation.kind)?,
        lease: operation
            .lease
            .map(|lease| lease_from_fb(*lease))
            .context("materialize operation has no KV lease")?,
        position: operation.position,
        conditioning_position: operation.conditioning_position,
        policy: operation
            .policy
            .map(|policy| policy_from_fb(*policy))
            .context("materialize operation has no token policy")?,
        input: match operation.input {
            fbs::MaterializeInputT::LatentProduct(input) => MaterializeInput::Latent {
                handle: input.handle,
            },
            fbs::MaterializeInputT::PublishedProduct(product) => {
                MaterializeInput::Published(published_from_fb(*product))
            }
            fbs::MaterializeInputT::NONE => bail!("materialize input union is empty"),
        },
    })
}

fn transfer_operation_to_fb(operation: &TransferOperation) -> fbs::TransferOperationT {
    fbs::TransferOperationT {
        kind: transfer_kind_to_fb(operation.kind),
        lease: Some(Box::new(lease_to_fb(&operation.lease))),
        position: operation.position,
        conditioning_position: operation.conditioning_position,
        policy: Some(Box::new(policy_to_fb(&operation.policy))),
        source: Some(Box::new(published_to_fb(&operation.source))),
    }
}

fn transfer_operation_from_fb(
    operation: fbs::TransferOperationT,
) -> anyhow::Result<TransferOperation> {
    Ok(TransferOperation {
        kind: transfer_kind_from_fb(operation.kind)?,
        lease: operation
            .lease
            .map(|lease| lease_from_fb(*lease))
            .context("transfer operation has no KV lease")?,
        position: operation.position,
        conditioning_position: operation.conditioning_position,
        policy: operation
            .policy
            .map(|policy| policy_from_fb(*policy))
            .context("transfer operation has no token policy")?,
        source: published_from_fb(
            *operation
                .source
                .context("transfer operation has no source")?,
        ),
    })
}

fn lease_to_fb(lease: &KvLeaseDelta) -> fbs::KvLeaseDeltaT {
    fbs::KvLeaseDeltaT {
        group_id: lease.group_id,
        new_blocks: Some(lease.new_blocks.iter().map(|block| block.0).collect()),
    }
}

fn lease_from_fb(lease: fbs::KvLeaseDeltaT) -> KvLeaseDelta {
    KvLeaseDelta {
        group_id: lease.group_id,
        new_blocks: lease
            .new_blocks
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
    }
}

fn policy_to_fb(policy: &TokenPolicy) -> fbs::TokenPolicyT {
    fbs::TokenPolicyT {
        allowed_tokens: Some(policy.allowed_tokens.clone()),
        suppress_tokens: Some(policy.suppress_tokens.clone()),
        recent_tokens: Some(policy.recent_tokens.clone()),
        publish_kv: policy.publish_kv,
        publish_kv_on_tokens: Some(policy.publish_kv_on_tokens.clone()),
    }
}

fn policy_from_fb(policy: fbs::TokenPolicyT) -> TokenPolicy {
    TokenPolicy {
        allowed_tokens: policy.allowed_tokens.unwrap_or_default(),
        suppress_tokens: policy.suppress_tokens.unwrap_or_default(),
        recent_tokens: policy.recent_tokens.unwrap_or_default(),
        publish_kv: policy.publish_kv,
        publish_kv_on_tokens: policy.publish_kv_on_tokens.unwrap_or_default(),
    }
}

fn published_to_fb(product: &PublishedProduct) -> fbs::PublishedProductT {
    fbs::PublishedProductT {
        handle: product.handle,
        locator: Some(product.locator.clone()),
    }
}

fn published_from_fb(product: fbs::PublishedProductT) -> PublishedProduct {
    PublishedProduct {
        handle: product.handle,
        locator: product.locator.unwrap_or_default(),
    }
}

fn published_kv_to_fb(snapshot: &PublishedKv) -> fbs::PublishedKvT {
    fbs::PublishedKvT {
        handle: snapshot.handle,
        locators: Some(snapshot.locators.clone()),
        source_version: snapshot.source_version,
        kv_tokens: snapshot.kv_tokens,
        block_ids: Some(snapshot.block_ids.iter().map(|block| block.0).collect()),
        group_id: snapshot.group_id,
        position: snapshot.position,
    }
}

fn published_kv_from_fb(snapshot: fbs::PublishedKvT) -> anyhow::Result<PublishedKv> {
    let snapshot = PublishedKv {
        handle: snapshot.handle,
        locators: snapshot.locators.unwrap_or_default(),
        source_version: snapshot.source_version,
        kv_tokens: snapshot.kv_tokens,
        block_ids: snapshot
            .block_ids
            .unwrap_or_default()
            .into_iter()
            .map(BlockId)
            .collect(),
        group_id: snapshot.group_id,
        position: snapshot.position,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn execution_result_to_fb(result: &ExecutionResult) -> anyhow::Result<fbs::ExecutionResultT> {
    Ok(fbs::ExecutionResultT {
        step_id: result.step_id,
        operations: Some(
            result
                .operations
                .iter()
                .map(operation_result_to_fb)
                .collect::<anyhow::Result<_>>()?,
        ),
        worker_exec_us: result.worker_exec_us,
        forward_stats: result
            .forward_stats
            .as_ref()
            .map(forward_stats_to_fb)
            .map(Box::new),
    })
}

fn execution_result_from_fb(result: fbs::ExecutionResultT) -> anyhow::Result<ExecutionResult> {
    Ok(ExecutionResult {
        step_id: result.step_id,
        operations: result
            .operations
            .unwrap_or_default()
            .into_iter()
            .map(operation_result_from_fb)
            .collect::<anyhow::Result<_>>()?,
        worker_exec_us: result.worker_exec_us,
        forward_stats: result
            .forward_stats
            .map(|stats| forward_stats_from_fb(*stats)),
    })
}

fn operation_result_to_fb(result: &OperationResult) -> anyhow::Result<fbs::OperationResultT> {
    Ok(fbs::OperationResultT {
        session_id: result.session_id.0,
        epoch: result.epoch,
        op_id: result.op_id,
        base_version: result.base_version,
        result_version: result.result_version,
        delta: result_delta_to_fb(&result.delta)?,
    })
}

fn operation_result_from_fb(result: fbs::OperationResultT) -> anyhow::Result<OperationResult> {
    Ok(OperationResult {
        session_id: RequestId(result.session_id),
        epoch: result.epoch,
        op_id: result.op_id,
        base_version: result.base_version,
        result_version: result.result_version,
        delta: result_delta_from_fb(result.delta)?,
    })
}

fn result_delta_to_fb(delta: &ResultDelta) -> anyhow::Result<fbs::ResultDeltaT> {
    Ok(match delta {
        ResultDelta::Sequence(delta) => {
            fbs::ResultDeltaT::SequenceDelta(Box::new(fbs::SequenceDeltaT {
                effect: Some(Box::new(sequence_effect_to_fb(&delta.effect)?)),
            }))
        }
        ResultDelta::Flow(delta) => fbs::ResultDeltaT::FlowDelta(Box::new(fbs::FlowDeltaT {
            steps_completed: delta.steps_completed,
            done: delta.done,
        })),
        ResultDelta::Encode(delta) => fbs::ResultDeltaT::EncodeDelta(Box::new(fbs::EncodeDeltaT {
            product_handle: delta.product_handle,
            kv_tokens: delta.kv_tokens,
            height: delta.image_size.map(|size| size.0),
            width: delta.image_size.map(|size| size.1),
        })),
        ResultDelta::Materialize(delta) => {
            fbs::ResultDeltaT::MaterializeDelta(Box::new(fbs::MaterializeDeltaT {
                product: materialized_product_to_fb(&delta.product),
                kv_tokens: delta.kv_tokens,
                sequence: delta
                    .sequence
                    .as_ref()
                    .map(sequence_effect_to_fb)
                    .transpose()?
                    .map(Box::new),
            }))
        }
        ResultDelta::Transfer(delta) => {
            fbs::ResultDeltaT::TransferDelta(Box::new(fbs::TransferDeltaT {
                product: delta.product.as_ref().map(published_to_fb).map(Box::new),
                kv_tokens: delta.kv_tokens,
                sequence: delta
                    .sequence
                    .as_ref()
                    .map(sequence_effect_to_fb)
                    .transpose()?
                    .map(Box::new),
            }))
        }
    })
}

fn result_delta_from_fb(delta: fbs::ResultDeltaT) -> anyhow::Result<ResultDelta> {
    Ok(match delta {
        fbs::ResultDeltaT::SequenceDelta(delta) => ResultDelta::Sequence(SequenceDelta {
            effect: sequence_effect_from_fb(
                *delta.effect.context("sequence delta has no effect")?,
            )?,
        }),
        fbs::ResultDeltaT::FlowDelta(delta) => ResultDelta::Flow(FlowDelta {
            steps_completed: delta.steps_completed,
            done: delta.done,
        }),
        fbs::ResultDeltaT::EncodeDelta(delta) => ResultDelta::Encode(EncodeDelta {
            product_handle: delta.product_handle,
            kv_tokens: delta.kv_tokens,
            image_size: match (delta.height, delta.width) {
                (Some(height), Some(width)) => Some((height, width)),
                (None, None) => None,
                _ => bail!("encode delta image dimensions are incomplete"),
            },
        }),
        fbs::ResultDeltaT::MaterializeDelta(delta) => ResultDelta::Materialize(MaterializeDelta {
            product: materialized_product_from_fb(delta.product)?,
            kv_tokens: delta.kv_tokens,
            sequence: delta
                .sequence
                .map(|effect| sequence_effect_from_fb(*effect))
                .transpose()?,
        }),
        fbs::ResultDeltaT::TransferDelta(delta) => ResultDelta::Transfer(TransferDelta {
            product: delta.product.map(|product| published_from_fb(*product)),
            kv_tokens: delta.kv_tokens,
            sequence: delta
                .sequence
                .map(|effect| sequence_effect_from_fb(*effect))
                .transpose()?,
        }),
        fbs::ResultDeltaT::NONE => bail!("result delta union is empty"),
    })
}

fn sequence_effect_to_fb(effect: &SequenceEffect) -> anyhow::Result<fbs::SequenceEffectT> {
    validate_sequence_effect(effect)?;
    Ok(fbs::SequenceEffectT {
        sampled_token_ids: Some(effect.sampled_token_ids.clone()),
        sampled_logprob: effect.sampled_logprob,
        top_logprobs: Some(
            effect
                .top_logprobs
                .iter()
                .map(token_logprob_to_fb)
                .collect(),
        ),
        prompt_logprobs: Some(
            effect
                .prompt_logprobs
                .iter()
                .map(|entries| fbs::PositionLogprobsT {
                    entries: Some(entries.iter().map(token_logprob_to_fb).collect()),
                })
                .collect(),
        ),
        accepted_draft_tokens: effect.accepted_draft_tokens,
        kv_tokens: effect.kv_tokens,
        published_logits: effect
            .published_logits
            .as_ref()
            .map(published_to_fb)
            .map(Box::new),
        published_kv: effect
            .published_kv
            .as_ref()
            .map(published_kv_to_fb)
            .map(Box::new),
    })
}

fn sequence_effect_from_fb(effect: fbs::SequenceEffectT) -> anyhow::Result<SequenceEffect> {
    let effect = SequenceEffect {
        sampled_token_ids: effect.sampled_token_ids.unwrap_or_default(),
        sampled_logprob: effect.sampled_logprob,
        top_logprobs: effect
            .top_logprobs
            .unwrap_or_default()
            .into_iter()
            .map(token_logprob_from_fb)
            .collect(),
        prompt_logprobs: effect
            .prompt_logprobs
            .unwrap_or_default()
            .into_iter()
            .map(|position| {
                position
                    .entries
                    .unwrap_or_default()
                    .into_iter()
                    .map(token_logprob_from_fb)
                    .collect()
            })
            .collect(),
        accepted_draft_tokens: effect.accepted_draft_tokens,
        kv_tokens: effect.kv_tokens,
        published_logits: effect
            .published_logits
            .map(|product| published_from_fb(*product)),
        published_kv: effect
            .published_kv
            .map(|snapshot| published_kv_from_fb(*snapshot))
            .transpose()?,
    };
    validate_sequence_effect(&effect)?;
    Ok(effect)
}

fn validate_sequence_effect(effect: &SequenceEffect) -> anyhow::Result<()> {
    if let Some(value) = effect.sampled_logprob {
        anyhow::ensure!(value.is_finite(), "sampled logprob must be finite");
    }
    for (position, entries) in std::iter::once(&effect.top_logprobs)
        .chain(effect.prompt_logprobs.iter())
        .enumerate()
    {
        for (index, entry) in entries.iter().enumerate() {
            anyhow::ensure!(
                entry.1.is_finite(),
                "logprob at {position}:{index} is not finite"
            );
            anyhow::ensure!(
                entry.2 > 0,
                "logprob rank at {position}:{index} must be positive"
            );
        }
    }
    Ok(())
}

fn token_logprob_to_fb(entry: &TokenLogprob) -> fbs::TokenLogprobT {
    fbs::TokenLogprobT {
        token_id: entry.0,
        logprob: entry.1,
        rank: entry.2,
    }
}

fn token_logprob_from_fb(entry: fbs::TokenLogprobT) -> TokenLogprob {
    TokenLogprob(entry.token_id, entry.logprob, entry.rank)
}

fn materialized_product_to_fb(product: &MaterializedProduct) -> fbs::MaterializedProductT {
    match product {
        MaterializedProduct::Image(image) => {
            fbs::MaterializedProductT::ImageArtifact(Box::new(fbs::ImageArtifactT {
                png_base64: Some(image.png_base64.clone()),
                height: image.height,
                width: image.width,
                handle: image.handle,
                locator: Some(image.locator.clone()),
            }))
        }
        MaterializedProduct::Published(product) => {
            fbs::MaterializedProductT::PublishedProduct(Box::new(published_to_fb(product)))
        }
        MaterializedProduct::Frame { count } => {
            fbs::MaterializedProductT::FrameRecord(Box::new(fbs::FrameRecordT { count: *count }))
        }
    }
}

fn materialized_product_from_fb(
    product: fbs::MaterializedProductT,
) -> anyhow::Result<MaterializedProduct> {
    Ok(match product {
        fbs::MaterializedProductT::ImageArtifact(image) => {
            MaterializedProduct::Image(ImageArtifact {
                png_base64: required_string(image.png_base64, "image artifact")?,
                height: image.height,
                width: image.width,
                handle: image.handle,
                locator: image.locator.unwrap_or_default(),
            })
        }
        fbs::MaterializedProductT::PublishedProduct(product) => {
            MaterializedProduct::Published(published_from_fb(*product))
        }
        fbs::MaterializedProductT::FrameRecord(frame) => {
            MaterializedProduct::Frame { count: frame.count }
        }
        fbs::MaterializedProductT::NONE => bail!("materialized product union is empty"),
    })
}

fn capabilities_to_fb(caps: &EngineCaps) -> anyhow::Result<fbs::EngineCapsT> {
    anyhow::ensure!(
        !caps.supported_operation_types.is_empty(),
        "worker capabilities declare no operations"
    );
    Ok(fbs::EngineCapsT {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
        supported_operation_types: Some(
            caps.supported_operation_types
                .iter()
                .copied()
                .map(operation_type_to_fb)
                .collect(),
        ),
        max_latent_size: caps.max_latent_size,
        latent_downsample: caps.latent_downsample,
        bytes_per_token: caps.bytes_per_token,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
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
        adapter_mode: adapter_mode_to_fb(caps.adapter_mode),
        execution_constraints: Some(Box::new(fbs::ExecutionConstraintsT {
            max_batch_operations: caps.execution_constraints.max_batch_operations,
        })),
        resource_classes: Some(
            caps.resource_classes
                .iter()
                .copied()
                .map(resource_class_to_fb)
                .collect(),
        ),
        model_spec_digest: Some(caps.model_spec_digest.clone()),
        weight_digest: Some(caps.weight_digest.clone()),
        restored_sessions: Some(
            caps.restored_sessions
                .iter()
                .map(|session| session.0)
                .collect(),
        ),
    })
}

fn capabilities_from_fb(caps: fbs::EngineCapsT) -> anyhow::Result<EngineCaps> {
    let caps = EngineCaps {
        block_size: caps.block_size,
        num_blocks: caps.num_blocks,
        num_layers: caps.num_layers,
        scratch_capacity_tokens: caps.scratch_capacity_tokens,
        supported_operation_types: caps
            .supported_operation_types
            .unwrap_or_default()
            .into_iter()
            .map(operation_type_from_fb)
            .collect::<anyhow::Result<_>>()?,
        max_latent_size: caps.max_latent_size,
        latent_downsample: caps.latent_downsample,
        bytes_per_token: caps.bytes_per_token,
        max_vae_grid_tokens: caps.max_vae_grid_tokens,
        max_vit_grid_tokens: caps.max_vit_grid_tokens,
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
        adapter_mode: adapter_mode_from_fb(caps.adapter_mode)?,
        execution_constraints: caps
            .execution_constraints
            .map(|constraints| ExecutionConstraints {
                max_batch_operations: constraints.max_batch_operations,
            })
            .context("capabilities have no execution constraints")?,
        resource_classes: caps
            .resource_classes
            .unwrap_or_default()
            .into_iter()
            .map(resource_class_from_fb)
            .collect::<anyhow::Result<_>>()?,
        model_spec_digest: caps
            .model_spec_digest
            .context("capabilities.model_spec_digest is missing")?,
        weight_digest: caps
            .weight_digest
            .context("capabilities.weight_digest is missing")?,
        restored_sessions: caps
            .restored_sessions
            .unwrap_or_default()
            .into_iter()
            .map(RequestId)
            .collect(),
    };
    anyhow::ensure!(
        !caps.supported_operation_types.is_empty(),
        "worker capabilities declare no operations"
    );
    anyhow::ensure!(
        (caps.model_spec_digest.is_empty() && caps.weight_digest.is_empty())
            || (caps.model_spec_digest.len() == 64 && caps.weight_digest.len() == 64),
        "worker capability model and weight identities are incomplete"
    );
    Ok(caps)
}

fn canonical_model_dtype(value: String) -> anyhow::Result<String> {
    anyhow::ensure!(
        matches!(value.as_str(), "float16" | "bfloat16" | "float32"),
        "capabilities.model_dtype is not canonical: {value:?}"
    );
    Ok(value)
}

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
    })
}

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

fn pressure_from_fb(pressure: fbs::ResourcePressureT) -> anyhow::Result<ResourcePressure> {
    Ok(ResourcePressure {
        class: resource_class_from_fb(pressure.class)?,
        total: pressure.total,
        used: pressure.used,
        evictable: pressure.evictable,
        free: pressure.free,
    })
}

fn required_string(value: Option<String>, label: &str) -> anyhow::Result<String> {
    value
        .filter(|value| !value.is_empty())
        .with_context(|| format!("{label} is missing"))
}

fn snapshot_to_fb(snapshot: &SnapshotRef) -> fbs::SnapshotRefT {
    fbs::SnapshotRefT {
        session_id: snapshot.session_id.0,
        epoch: snapshot.epoch,
        version: snapshot.version,
        digest: Some(snapshot.digest.clone()),
        locator: Some(snapshot.locator.clone()),
    }
}

fn snapshot_from_fb(snapshot: fbs::SnapshotRefT) -> anyhow::Result<SnapshotRef> {
    let snapshot = SnapshotRef {
        session_id: RequestId(snapshot.session_id),
        epoch: snapshot.epoch,
        version: snapshot.version,
        digest: required_string(snapshot.digest, "snapshot digest")?,
        locator: required_string(snapshot.locator, "snapshot locator")?,
    };
    anyhow::ensure!(
        snapshot.digest.len() == 64
            && snapshot
                .digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "snapshot digest must be a lowercase SHA-256 digest"
    );
    anyhow::ensure!(
        snapshot.locator == snapshot.digest,
        "snapshot locator must equal its content digest"
    );
    Ok(snapshot)
}

fn operation_type_to_fb(operation_type: OperationType) -> fbs::OperationType {
    match operation_type {
        OperationType::SequenceExtend => fbs::OperationType::SequenceExtend,
        OperationType::SequenceDecode => fbs::OperationType::SequenceDecode,
        OperationType::SequenceVerify => fbs::OperationType::SequenceVerify,
        OperationType::SequenceSample => fbs::OperationType::SequenceSample,
        OperationType::Flow => fbs::OperationType::Flow,
        OperationType::EncodeVision => fbs::OperationType::EncodeVision,
        OperationType::EncodeLatent => fbs::OperationType::EncodeLatent,
        OperationType::MaterializeImage => fbs::OperationType::MaterializeImage,
        OperationType::MaterializeFrame => fbs::OperationType::MaterializeFrame,
        OperationType::TransferProduct => fbs::OperationType::TransferProduct,
        OperationType::TransferKv => fbs::OperationType::TransferKv,
    }
}

fn operation_type_from_fb(operation_type: fbs::OperationType) -> anyhow::Result<OperationType> {
    if operation_type == fbs::OperationType::SequenceExtend {
        Ok(OperationType::SequenceExtend)
    } else if operation_type == fbs::OperationType::SequenceDecode {
        Ok(OperationType::SequenceDecode)
    } else if operation_type == fbs::OperationType::SequenceVerify {
        Ok(OperationType::SequenceVerify)
    } else if operation_type == fbs::OperationType::SequenceSample {
        Ok(OperationType::SequenceSample)
    } else if operation_type == fbs::OperationType::Flow {
        Ok(OperationType::Flow)
    } else if operation_type == fbs::OperationType::EncodeVision {
        Ok(OperationType::EncodeVision)
    } else if operation_type == fbs::OperationType::EncodeLatent {
        Ok(OperationType::EncodeLatent)
    } else if operation_type == fbs::OperationType::MaterializeImage {
        Ok(OperationType::MaterializeImage)
    } else if operation_type == fbs::OperationType::MaterializeFrame {
        Ok(OperationType::MaterializeFrame)
    } else if operation_type == fbs::OperationType::TransferProduct {
        Ok(OperationType::TransferProduct)
    } else if operation_type == fbs::OperationType::TransferKv {
        Ok(OperationType::TransferKv)
    } else {
        bail!("unknown operation type {}", operation_type.0)
    }
}

fn sequence_mode_to_fb(mode: SequenceMode) -> fbs::SequenceMode {
    match mode {
        SequenceMode::Extend => fbs::SequenceMode::Extend,
        SequenceMode::Decode => fbs::SequenceMode::Decode,
        SequenceMode::Verify => fbs::SequenceMode::Verify,
        SequenceMode::Sample => fbs::SequenceMode::Sample,
    }
}
fn sequence_mode_from_fb(mode: fbs::SequenceMode) -> anyhow::Result<SequenceMode> {
    if mode == fbs::SequenceMode::Extend {
        Ok(SequenceMode::Extend)
    } else if mode == fbs::SequenceMode::Decode {
        Ok(SequenceMode::Decode)
    } else if mode == fbs::SequenceMode::Verify {
        Ok(SequenceMode::Verify)
    } else if mode == fbs::SequenceMode::Sample {
        Ok(SequenceMode::Sample)
    } else {
        bail!("unknown sequence mode {}", mode.0)
    }
}
fn token_source_to_fb(source: TokenSource) -> fbs::TokenSource {
    match source {
        TokenSource::Wire => fbs::TokenSource::Wire,
        TokenSource::LastSampled => fbs::TokenSource::LastSampled,
    }
}
fn token_source_from_fb(source: fbs::TokenSource) -> anyhow::Result<TokenSource> {
    if source == fbs::TokenSource::Wire {
        Ok(TokenSource::Wire)
    } else if source == fbs::TokenSource::LastSampled {
        Ok(TokenSource::LastSampled)
    } else {
        bail!("unknown token source {}", source.0)
    }
}
fn encode_kind_to_fb(kind: EncodeKind) -> fbs::EncodeKind {
    match kind {
        EncodeKind::Vision => fbs::EncodeKind::Vision,
        EncodeKind::Latent => fbs::EncodeKind::Latent,
    }
}
fn encode_kind_from_fb(kind: fbs::EncodeKind) -> anyhow::Result<EncodeKind> {
    if kind == fbs::EncodeKind::Vision {
        Ok(EncodeKind::Vision)
    } else if kind == fbs::EncodeKind::Latent {
        Ok(EncodeKind::Latent)
    } else {
        bail!("unknown encode kind {}", kind.0)
    }
}
fn materialize_kind_to_fb(kind: MaterializeKind) -> fbs::MaterializeKind {
    match kind {
        MaterializeKind::Image => fbs::MaterializeKind::Image,
        MaterializeKind::Frame => fbs::MaterializeKind::Frame,
    }
}
fn materialize_kind_from_fb(kind: fbs::MaterializeKind) -> anyhow::Result<MaterializeKind> {
    if kind == fbs::MaterializeKind::Image {
        Ok(MaterializeKind::Image)
    } else if kind == fbs::MaterializeKind::Frame {
        Ok(MaterializeKind::Frame)
    } else {
        bail!("unknown materialize kind {}", kind.0)
    }
}
fn transfer_kind_to_fb(kind: TransferKind) -> fbs::TransferKind {
    match kind {
        TransferKind::Product => fbs::TransferKind::Product,
        TransferKind::Kv => fbs::TransferKind::Kv,
    }
}
fn transfer_kind_from_fb(kind: fbs::TransferKind) -> anyhow::Result<TransferKind> {
    if kind == fbs::TransferKind::Product {
        Ok(TransferKind::Product)
    } else if kind == fbs::TransferKind::Kv {
        Ok(TransferKind::Kv)
    } else {
        bail!("unknown transfer kind {}", kind.0)
    }
}

fn request_kind_to_fb(kind: RequestKind) -> fbs::ReqKind {
    match kind {
        RequestKind::GetCapabilities => fbs::ReqKind::GetCapabilities,
        RequestKind::Execute => fbs::ReqKind::Execute,
        RequestKind::DropSession => fbs::ReqKind::DropSession,
        RequestKind::Shutdown => fbs::ReqKind::Shutdown,
        RequestKind::CopyKv => fbs::ReqKind::CopyKv,
        RequestKind::LoadAdapter => fbs::ReqKind::LoadAdapter,
        RequestKind::UnloadAdapter => fbs::ReqKind::UnloadAdapter,
        RequestKind::ReleaseProducts => fbs::ReqKind::ReleaseProducts,
        RequestKind::ResetPrefixCache => fbs::ReqKind::ResetPrefixCache,
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
fn adapter_mode_to_fb(mode: AdapterMode) -> fbs::AdapterMode {
    match mode {
        AdapterMode::None => fbs::AdapterMode::None,
        AdapterMode::EngineWide => fbs::AdapterMode::EngineWide,
        AdapterMode::PerRequest => fbs::AdapterMode::PerRequest,
        AdapterMode::MultiAdapter => fbs::AdapterMode::MultiAdapter,
    }
}
fn adapter_mode_from_fb(mode: fbs::AdapterMode) -> anyhow::Result<AdapterMode> {
    if mode == fbs::AdapterMode::None {
        Ok(AdapterMode::None)
    } else if mode == fbs::AdapterMode::EngineWide {
        Ok(AdapterMode::EngineWide)
    } else if mode == fbs::AdapterMode::PerRequest {
        Ok(AdapterMode::PerRequest)
    } else if mode == fbs::AdapterMode::MultiAdapter {
        Ok(AdapterMode::MultiAdapter)
    } else {
        bail!("unknown adapter mode {}", mode.0)
    }
}
fn resource_class_to_fb(class: ResourceClass) -> fbs::ResourceClass {
    match class {
        ResourceClass::KvBlock => fbs::ResourceClass::KvBlock,
        ResourceClass::EncoderOutput => fbs::ResourceClass::EncoderOutput,
        ResourceClass::ImageLatent => fbs::ResourceClass::ImageLatent,
        ResourceClass::Scratch => fbs::ResourceClass::Scratch,
        ResourceClass::Adapter => fbs::ResourceClass::Adapter,
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
    } else if class == fbs::ResourceClass::Adapter {
        Ok(ResourceClass::Adapter)
    } else {
        bail!("unknown resource class {}", class.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlowOperation, Guidance};

    fn identity() -> String {
        "ab".repeat(32)
    }

    fn operation() -> OperationEnvelope {
        let mut operation = OperationEnvelope::unsealed(
            RequestId(7),
            Operation::Flow(FlowOperation {
                latent_handle: 7,
                position: 3,
                start_step: 2,
                step_count: 4,
                conditioning_position: 3,
                conditioning: None,
                guidance: Guidance {
                    branch_count: 3,
                    text_scale: 4.0,
                    image_scale: 1.0,
                    renorm_type: "global".into(),
                    renorm_min: 0.0,
                    interval: (0.0, 1.0),
                },
                image_prompt: "mountains".into(),
            }),
        );
        operation.admission_digest = identity();
        operation.model_spec_digest = identity();
        operation.weight_digest = identity();
        operation.seal(2, 9, 4);
        operation
    }

    #[test]
    fn request_round_trip_preserves_the_exact_union_variant() {
        let admission = Admission::new(
            RequestId(7),
            None,
            Some(FlowAdmission {
                image: uniserve_core::ImageParams::default(),
            }),
            None,
        )
        .unwrap();
        let mut operation = operation();
        operation.admission_digest = admission.digest.clone();
        operation.refresh_digest();
        let request = WorkerRequest::execute(Batch::new(11, vec![admission], vec![operation]));
        let decoded = decode_request(&encode_request(&request).unwrap()).unwrap();
        let decoded_operation = &decoded.batch.unwrap().operations[0].operation;
        assert!(matches!(decoded_operation, Operation::Flow(_)));
    }

    #[test]
    fn response_round_trip_preserves_typed_delta() {
        let operation = operation();
        let response = WorkerResponse::result(ExecutionResult {
            step_id: 5,
            operations: vec![OperationResult {
                session_id: operation.session_id,
                epoch: operation.epoch,
                op_id: operation.op_id,
                base_version: operation.base_version,
                result_version: operation.base_version + 1,
                delta: ResultDelta::Flow(FlowDelta {
                    steps_completed: 6,
                    done: false,
                }),
            }],
            worker_exec_us: Some(10),
            forward_stats: None,
        });
        let decoded = decode_response(&encode_response(&response).unwrap()).unwrap();
        assert!(matches!(
            decoded.result.unwrap().operations[0].delta,
            ResultDelta::Flow(_)
        ));
    }
}
