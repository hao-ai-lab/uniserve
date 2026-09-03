use std::collections::HashSet;

use uniserve_core::product_blob::LogprobBlob;
use uniserve_core::{
    BlockId, CfgParams, ContextSegment, GenerationRequest, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, RequestId, SamplingParams, SegmentPlacement, UndTokenAction,
};
use uniserve_worker_ipc::{
    Bounds, Checkpoint, CheckpointPoint, DType, DimBound, DrawLayout, InlineValue, ModelOutput,
    OpId, OpPayload, OpStatus, Operation, PointRange, ProductKind, ProductPayload, ProductRef,
    RequestKey, Rng, RunKind, SamplingState, ShapeBound, StorageClass, encode_sampling_state_bytes,
    encode_token_product_bytes,
};

use crate::memory::Allocation;
use crate::runtime::image_artifact::png_artifact_dims_b64;

/// A product reference minted by the planner for an operation output. The
/// owning `request_key` and `producer_op_id` are placeholder until
/// [`NextOp::register`] stamps the real identity. The shape
/// bound is empty (it carries identity, not a device geometry).
fn output_product(
    output_index: u16,
    kind: ProductKind,
    storage_class: StorageClass,
    dtype: DType,
) -> ProductRef {
    ProductRef {
        request_key: RequestKey::new(0, RequestId(0), 0),
        producer_op_id: OpId(0),
        output_index,
        generation: 0,
        kind,
        storage_class,
        dtype,
        shape_bound: ShapeBound::default(),
        point_range: PointRange::default(),
    }
}

fn bounded_product(
    output_index: u16,
    kind: ProductKind,
    storage_class: StorageClass,
    dtype: DType,
    shape_bound: ShapeBound,
) -> ProductRef {
    let mut product = output_product(output_index, kind, storage_class, dtype);
    product.shape_bound = shape_bound;
    product
}

fn dtype_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::U8 => 1,
        DType::U16 | DType::F16 | DType::BF16 => 2,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::I64 => 8,
    }
}

fn product_bound_bytes(product: &ProductRef) -> u64 {
    product
        .shape_bound
        .dims
        .iter()
        .fold(1_u64, |elements, dim| {
            elements.saturating_mul(u64::from(match dim {
                DimBound::Static(value) => *value,
                DimBound::Device { max } => *max,
            }))
        })
        .saturating_mul(dtype_bytes(product.dtype))
}

fn dynamic_element_bound(bytes: u64, dtype: DType) -> Result<ShapeBound, PlanningError> {
    let elements = bytes.div_ceil(dtype_bytes(dtype));
    let max = u32::try_from(elements).map_err(|_| PlanningError::ProductBoundTooLarge { bytes })?;
    if max == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    Ok(ShapeBound {
        dims: vec![DimBound::Device { max }],
    })
}

fn png_base64_bound(width: u32, height: u32) -> Result<u64, PlanningError> {
    let raw = u64::from(height)
        .checked_mul(u64::from(width).saturating_mul(3).saturating_add(1))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let png = raw
        .checked_mul(2)
        .and_then(|value| value.checked_add(1 << 20))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    Ok(png.div_ceil(3).saturating_mul(4))
}

/// A host-supplied input product reference for an operation's forward, owned by
/// the consuming operation's identity at a reserved input index so it never
/// collides with the operation's declared outputs. Its value is carried in the
/// batch's `input_products` under this same identity.
fn host_input_product(
    request_key: RequestKey,
    op_id: OpId,
    output_index: u16,
    kind: ProductKind,
    dtype: DType,
    elements: usize,
) -> Result<ProductRef, PlanningError> {
    let extent = u32::try_from(elements).map_err(|_| PlanningError::ProductBoundTooLarge {
        bytes: (elements as u64).saturating_mul(dtype_bytes(dtype)),
    })?;
    if extent == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    Ok(ProductRef {
        request_key,
        producer_op_id: op_id,
        output_index,
        generation: 0,
        kind,
        storage_class: StorageClass::HostStaging,
        dtype,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(extent)],
        },
        point_range: PointRange::default(),
    })
}

const RANKED_LOGPROB_BYTES: u64 = 12;

fn logprob_blob_bound(
    sampling: &SamplingParams,
    prompt_positions: u32,
) -> Result<Option<u64>, PlanningError> {
    let generated = sampling.generated_logprobs_requested();
    let prompt = sampling.prompt_logprobs_requested();
    if !generated && !prompt {
        return Ok(None);
    }
    let requested_ids = sampling
        .logprob_token_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len() as u64;
    let generated_entries = generated.then_some(
        1_u64
            .checked_add(u64::from(sampling.n_logprobs))
            .and_then(|value| value.checked_add(requested_ids))
            .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
    );
    let prompt_entries = prompt.then_some(
        1_u64
            .checked_add(u64::from(sampling.n_prompt_logprobs))
            .and_then(|value| value.checked_add(requested_ids))
            .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
    );
    let generated_bytes = generated_entries
        .unwrap_or(0)
        .checked_mul(RANKED_LOGPROB_BYTES)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let per_prompt_bytes = 4_u64
        .checked_add(
            prompt_entries
                .unwrap_or(0)
                .checked_mul(RANKED_LOGPROB_BYTES)
                .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?,
        )
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let prompt_bytes = u64::from(prompt_positions)
        .checked_mul(per_prompt_bytes)
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    let bytes = 1_u64
        .checked_add(if generated { 4 } else { 0 })
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(generated_bytes))
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(prompt_bytes))
        .ok_or(PlanningError::ProductBoundTooLarge { bytes: u64::MAX })?;
    Ok(Some(bytes))
}

/// The declared outputs of a token operation. The selected-token product also
/// carries the device continuation bit consumed by a registered descendant;
/// the worker masks that bit before the token reaches model input.
fn token_outputs(
    logprob_bound: Option<u64>,
    produces_transition_candidate: bool,
    max_points: u32,
) -> Result<Vec<ProductRef>, PlanningError> {
    let mut token = output_product(
        0,
        ProductKind::Token,
        StorageClass::RequestRelay,
        DType::U32,
    );
    token.point_range = PointRange {
        base_point: 0,
        max_points,
    };
    let point_range = token.point_range;
    let mut outputs = vec![token];
    if max_points > 1 {
        let mut selected_point = output_product(
            1,
            ProductKind::SelectedPoint,
            StorageClass::RequestRelay,
            DType::U32,
        );
        selected_point.point_range = point_range;
        outputs.push(selected_point);
    }
    if let Some(bytes) = logprob_bound {
        outputs.push(bounded_product(
            2,
            ProductKind::Logprob,
            StorageClass::HostStaging,
            DType::U8,
            dynamic_element_bound(bytes, DType::U8)?,
        ));
    }
    if produces_transition_candidate {
        outputs.push(output_product(
            3,
            ProductKind::Completion,
            StorageClass::RequestRelay,
            DType::U8,
        ));
    }
    Ok(outputs)
}

/// Products emitted by the state-advancing feedback extend. The completion
/// predicate identifies the final feedback state transition independently of
/// whether the model policy also requests a sampled continuation token.
fn feedback_state_outputs(
    logprob_bound: Option<u64>,
    sample_continuation: bool,
    produces_transition_candidate: bool,
) -> Result<Vec<ProductRef>, PlanningError> {
    let mut outputs = vec![output_product(
        0,
        ProductKind::Completion,
        StorageClass::RequestRelay,
        DType::U8,
    )];
    if sample_continuation {
        outputs.push(output_product(
            1,
            ProductKind::Token,
            StorageClass::RequestRelay,
            DType::U32,
        ));
        if let Some(bytes) = logprob_bound {
            outputs.push(bounded_product(
                2,
                ProductKind::Logprob,
                StorageClass::HostStaging,
                DType::U8,
                dynamic_element_bound(bytes, DType::U8)?,
            ));
        }
        if produces_transition_candidate {
            outputs.push(output_product(
                3,
                ProductKind::Completion,
                StorageClass::RequestRelay,
                DType::U8,
            ));
        }
    }
    Ok(outputs)
}

fn operation_sampling_delta(
    sampling: &SamplingParams,
    state: &SamplingState,
) -> Option<SamplingState> {
    let mut delta = state.clone();
    delta.finish_token_ids.clear();
    if delta.allowed_token_ids == sampling.allowed_token_ids {
        delta.allowed_token_ids = None;
    }
    (delta != SamplingState::default()).then_some(delta)
}

/// Runtime-private flattened context derived once at admission from ordered
/// canonical context segments.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeContext {
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) negative_prompt_ids: Vec<u32>,
    pub(crate) images: Vec<RuntimeImage>,
    token_segments: Vec<RuntimeTokenSegment>,
}

#[derive(Debug, Clone)]
struct RuntimeTokenSegment {
    segment_index: usize,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeImage {
    pub(crate) segment_index: usize,
    pub(crate) hash: u64,
    pub(crate) position: u32,
    pub(crate) b64: String,
    pub(crate) ingest: ImageIngestRecipe,
}

impl RuntimeContext {
    pub(crate) fn lower(request: &GenerationRequest) -> Result<Self, ContextLoweringError> {
        request
            .validate()
            .map_err(|error| ContextLoweringError::InvalidRequest(error.to_string()))?;
        let mut prompt_ids = Vec::new();
        let mut images = Vec::new();
        let mut token_segments = Vec::new();
        for (segment_index, segment) in request.context.iter().enumerate() {
            match segment {
                ContextSegment::UndTokens { token_ids, .. } => {
                    let start = prompt_ids.len();
                    prompt_ids.extend_from_slice(token_ids);
                    token_segments.push(RuntimeTokenSegment {
                        segment_index,
                        start,
                        end: prompt_ids.len(),
                    });
                }
                ContextSegment::Image { image, ingest } => {
                    let position = match image.placement {
                        SegmentPlacement::AtToken { position } => position,
                        SegmentPlacement::Append => prompt_ids.len() as u32,
                    };
                    if position as usize > prompt_ids.len() {
                        return Err(ContextLoweringError::ImagePositionBeyondContext {
                            position,
                            token_count: prompt_ids.len(),
                        });
                    }
                    images.push(RuntimeImage {
                        segment_index,
                        hash: image.hash,
                        position,
                        b64: image.b64.clone(),
                        ingest: ingest.clone(),
                    });
                }
            }
        }
        images.sort_by_key(|image| image.position);
        let negative_prompt_ids = request
            .negative_context
            .iter()
            .flat_map(|segment| match segment {
                ContextSegment::UndTokens { token_ids, .. } => token_ids.clone(),
                ContextSegment::Image { .. } => Vec::new(),
            })
            .collect();
        Ok(Self {
            prompt_ids,
            negative_prompt_ids,
            images,
            token_segments,
        })
    }

    pub(crate) fn token_segment_at(&self, offset: usize) -> Option<(usize, usize)> {
        self.token_segments
            .iter()
            .find(|segment| segment.start <= offset && offset < segment.end)
            .map(|segment| (segment.segment_index, segment.end))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ContextLoweringError {
    #[error("invalid generation request: {0}")]
    InvalidRequest(String),
    #[error("image position {position} exceeds context length {token_count}")]
    ImagePositionBeyondContext { position: u32, token_count: usize },
}

/// Lifecycle phase for a canonical generation request.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationPhase {
    /// Encode staged input images before continuing text prefill.
    Encode,
    IngestState,
    Prefill,
    DecodeUnd,
    CloseKv,
    PublishKv,
    PrepareGen,
    DenoiseGen,
    CommitGen,
    FeedbackEncode,
    FeedbackState,
}

/// Runtime-owned cursor for one running request. Each mutable concern has a
/// single typed owner; transition application is the only operation that
/// commits worker-derived lifecycle progress.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GenerationCursor {
    pub(crate) phase: GenerationPhase,
    pub(crate) ingest: ContextCursor,
    pub(crate) und: UndCursor,
    pub(crate) image_gen: GenCursor,
    pub(crate) feedback: FeedbackCursor,
    pub(crate) resources: ResourceCursor,
    pub(crate) replay: ReplayCursor,
    applied_op_ids: HashSet<u64>,
}

impl GenerationCursor {
    pub(crate) fn new(
        phase: GenerationPhase,
        worstcase_blocks: usize,
        reserve_worstcase: bool,
    ) -> Self {
        Self {
            phase,
            ingest: ContextCursor {
                prompt_cursor: 0,
                prompt_logprobs_processed: 0,
                prompt_logprobs_emitted: 0,
                mm_cursor: 0,
                acquired_encoder_pins: Vec::new(),
                transient_encoder_products: Vec::new(),
                pending_image_step: 0,
                encoded_product: None,
                round_closing: false,
            },
            und: UndCursor {
                logical_pos: 0,
                physical_kv_len: 0,
                next_token: 0,
                tokens_emitted: 0,
                text_since_image: 0,
                round_tokens: Vec::new(),
            },
            image_gen: GenCursor {
                image_id: 0,
                images_done: 0,
                branch_pending: false,
                cond_pos: 0,
                steps_done: 0,
                conditioning: None,
                latent: None,
            },
            feedback: FeedbackCursor {
                image_b64: None,
                ingest_step: 0,
                source_product: None,
                encoded_product: None,
            },
            resources: ResourceCursor {
                worker_registered: false,
                blocks_sent: 0,
                reserve_worstcase,
                worstcase_blocks,
            },
            replay: ReplayCursor {
                block_hashes: Vec::new(),
                prefix_cached_blocks: 0,
                blocks_cached: false,
                generated_ids: Vec::new(),
                replayability: Replayability::Replayable,
            },
            applied_op_ids: HashSet::new(),
        }
    }

    /// Apply one transition to either the committed cursor (with a worker
    /// result) or a lookahead clone (without one).
    pub(crate) fn apply(
        &mut self,
        operation: &Operation,
        apply: &RuntimeApply,
        outcome: Option<(&ModelOutput, &[ProductPayload])>,
    ) -> Result<(), CursorApplyError> {
        if let Some((record, _)) = outcome {
            let op_id = operation.op_id.0;
            if op_id == 0 {
                return Err(CursorApplyError::MissingOperationId);
            }
            if !self.applied_op_ids.insert(op_id) {
                return Err(CursorApplyError::DuplicateOperation { op_id });
            }
            if record.status == OpStatus::Predicated {
                return Ok(());
            }
        }
        match &apply.intent {
            TransitionIntent::IngestText {
                start,
                end,
                logical_start,
                physical_start,
                ..
            } => {
                let count = end.saturating_sub(*start);
                self.ingest.prompt_cursor = self.ingest.prompt_cursor.max(*end);
                self.und.logical_pos = self
                    .und
                    .logical_pos
                    .max(logical_start.saturating_add(count));
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .max(physical_start.saturating_add(count));
            }
            TransitionIntent::EncodeImageStep { .. } => {}
            TransitionIntent::IngestImageState {
                step_index,
                is_final_step,
                position,
                logical_positions,
                physical_start,
                physical_kv_tokens,
                ..
            } => {
                self.und.physical_kv_len = outcome.map_or_else(
                    || projected_kv_len(*physical_start, *physical_kv_tokens),
                    |(record, _)| Ok(record.logical_lengths().kv_visible_len),
                )?;
                if *is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(*logical_positions));
                    self.ingest.mm_cursor = self.ingest.mm_cursor.saturating_add(1);
                    self.ingest.pending_image_step = 0;
                    self.ingest.encoded_product = None;
                    self.phase = GenerationPhase::Prefill;
                } else {
                    self.ingest.pending_image_step = step_index.saturating_add(1);
                    self.ingest.encoded_product = None;
                    self.phase = GenerationPhase::Encode;
                }
            }
            TransitionIntent::DecodeUnd {
                logical_position,
                physical_position,
                ..
            } => {
                let actual_count = outcome.map_or(1, |(record, _)| {
                    record
                        .committed_tokens()
                        .len()
                        .max(1)
                        .min(u32::MAX as usize) as u32
                });
                self.und.logical_pos = self
                    .und
                    .logical_pos
                    .max(logical_position.saturating_add(actual_count));
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .max(physical_position.saturating_add(actual_count));
            }
            TransitionIntent::CloseKv {
                image_id,
                logical_position,
                physical_position,
                ..
            } => {
                self.und.physical_kv_len = outcome.map_or_else(
                    || Ok(physical_position.saturating_add(1)),
                    |(record, _)| Ok(record.logical_lengths().kv_visible_len),
                )?;
                self.image_gen.image_id = *image_id;
                self.image_gen.cond_pos = *logical_position;
                self.phase = GenerationPhase::PublishKv;
            }
            TransitionIntent::PublishKv { .. } => {
                self.image_gen.conditioning = outcome
                    .and_then(|(_, products)| {
                        products
                            .iter()
                            .find(|payload| {
                                payload.product.producer_op_id == operation.op_id
                                    && payload.product.kind == ProductKind::Kv
                            })
                            .map(|payload| payload.product.clone())
                    })
                    .or_else(|| {
                        operation
                            .outputs()
                            .iter()
                            .find(|product| product.kind == ProductKind::Kv)
                            .cloned()
                    });
                if self.image_gen.conditioning.is_none() {
                    return Err(CursorApplyError::MissingKvProduct);
                }
                self.phase = GenerationPhase::PrepareGen;
            }
            TransitionIntent::PrepareGen { .. } => {
                self.image_gen.latent = operation
                    .outputs()
                    .iter()
                    .find(|product| product.kind == ProductKind::Latent)
                    .cloned();
                if self.image_gen.latent.is_none() {
                    return Err(CursorApplyError::MissingLatentProduct);
                }
                self.image_gen.steps_done = 0;
                self.phase = GenerationPhase::DenoiseGen;
            }
            TransitionIntent::DenoiseGen {
                start_step,
                step_count,
                ..
            } => {
                let steps_completed =
                    outcome.map_or(start_step.saturating_add(*step_count), |(record, _)| {
                        record.logical_lengths().latent_len.min(u32::from(u16::MAX)) as u16
                    });
                self.image_gen.steps_done = self
                    .image_gen
                    .steps_done
                    .max(steps_completed.max(start_step.saturating_add(*step_count)));
                self.image_gen.latent = operation
                    .outputs()
                    .iter()
                    .find(|product| product.kind == ProductKind::Latent)
                    .cloned();
                if self.image_gen.latent.is_none() {
                    return Err(CursorApplyError::MissingLatentProduct);
                }
            }
            TransitionIntent::CommitGen { image_id, .. } => {
                self.image_gen.image_id = *image_id;
                self.feedback.source_product = operation
                    .outputs()
                    .iter()
                    .find(|output| {
                        output.kind == ProductKind::Artifact
                            && output.storage_class == StorageClass::LatentArena
                    })
                    .cloned();
                self.feedback.ingest_step = 0;
                self.feedback.encoded_product = None;
                self.phase = GenerationPhase::FeedbackEncode;
            }
            TransitionIntent::EncodeFeedbackStep {
                image_id,
                step_index,
                ..
            } => {
                self.image_gen.image_id = *image_id;
                self.feedback.ingest_step = *step_index;
                self.feedback.encoded_product = operation
                    .outputs()
                    .iter()
                    .find(|output| {
                        matches!(
                            output.kind,
                            ProductKind::VisionFeature | ProductKind::LatentFeature
                        )
                    })
                    .cloned();
                self.phase = GenerationPhase::FeedbackState;
            }
            TransitionIntent::FeedbackState {
                step_index,
                is_final_step,
                position,
                logical_positions,
                physical_start,
                physical_kv_tokens,
                ..
            } => {
                self.und.physical_kv_len = outcome.map_or_else(
                    || projected_kv_len(*physical_start, *physical_kv_tokens),
                    |(record, _)| Ok(record.logical_lengths().kv_visible_len),
                )?;
                if *is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(*logical_positions));
                    self.feedback.image_b64 = None;
                    self.feedback.ingest_step = 0;
                    self.feedback.source_product = None;
                    self.feedback.encoded_product = None;
                    self.phase = GenerationPhase::DecodeUnd;
                } else {
                    self.feedback.ingest_step = step_index.saturating_add(1);
                    self.feedback.encoded_product = None;
                    self.phase = GenerationPhase::FeedbackEncode;
                }
            }
        }
        self.replay.replayability =
            match (self.replay.replayability, apply.replayability_after_apply) {
                (Replayability::NotReplayable, _) | (_, Replayability::NotReplayable) => {
                    Replayability::NotReplayable
                }
                (Replayability::Replayable, Replayability::Replayable) => Replayability::Replayable,
            };
        Ok(())
    }

    pub(crate) fn project<'a>(
        &self,
        inflight: impl IntoIterator<Item = (&'a Operation, &'a RuntimeApply)>,
    ) -> Option<Self> {
        let mut projection = self.clone();
        for (operation, apply) in inflight {
            projection.apply(operation, apply, None).ok()?;
        }
        Some(projection)
    }
}

fn projected_kv_len(base: u32, effect: ImageKvEffect) -> Result<u32, CursorApplyError> {
    match effect {
        ImageKvEffect::Exact { tokens } => Ok(base.saturating_add(tokens)),
        ImageKvEffect::Bounded { max_tokens } => Ok(base.saturating_add(max_tokens)),
        ImageKvEffect::WorkerDefined => Err(CursorApplyError::UnprojectableKvEffect),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CursorApplyError {
    #[error("operation has no registered identity")]
    MissingOperationId,
    #[error("KV publication produced no KV product")]
    MissingKvProduct,
    #[error("generation operation produced no latent product")]
    MissingLatentProduct,
    #[error("worker-defined KV effect cannot be projected")]
    UnprojectableKvEffect,
    #[error("operation {op_id} was already applied")]
    DuplicateOperation { op_id: u64 },
}

/// Locate the completion product a given operation produced for `kind`.
fn find_product(
    products: &[ProductPayload],
    op_id: OpId,
    kind: ProductKind,
) -> Option<&ProductPayload> {
    products
        .iter()
        .find(|payload| payload.product.producer_op_id == op_id && payload.product.kind == kind)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContextCursor {
    pub(crate) prompt_cursor: u32,
    pub(crate) prompt_logprobs_processed: usize,
    pub(crate) prompt_logprobs_emitted: usize,
    pub(crate) mm_cursor: usize,
    pub(crate) acquired_encoder_pins: Vec<EncoderCachePin>,
    pub(crate) transient_encoder_products: Vec<ProductRef>,
    pub(crate) pending_image_step: usize,
    pub(crate) encoded_product: Option<ProductRef>,
    pub(crate) round_closing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncoderCachePin {
    pub(crate) key: u64,
    pub(crate) product: ProductRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UndCursor {
    pub(crate) logical_pos: u32,
    pub(crate) physical_kv_len: u32,
    pub(crate) next_token: u32,
    pub(crate) tokens_emitted: usize,
    pub(crate) text_since_image: usize,
    pub(crate) round_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenCursor {
    pub(crate) image_id: u32,
    pub(crate) images_done: usize,
    pub(crate) branch_pending: bool,
    pub(crate) cond_pos: u32,
    pub(crate) steps_done: u16,
    /// The published KV product a denoise op conditions on, captured from the
    /// producing operation's completion products.
    pub(crate) conditioning: Option<ProductRef>,
    /// The immutable generation-owned latent produced by the transition or
    /// latest flow quantum.
    pub(crate) latent: Option<ProductRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FeedbackCursor {
    pub(crate) image_b64: Option<String>,
    pub(crate) ingest_step: usize,
    pub(crate) source_product: Option<ProductRef>,
    pub(crate) encoded_product: Option<ProductRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResourceCursor {
    pub(crate) worker_registered: bool,
    pub(crate) blocks_sent: usize,
    pub(crate) reserve_worstcase: bool,
    pub(crate) worstcase_blocks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReplayCursor {
    pub(crate) block_hashes: Vec<Vec<u64>>,
    pub(crate) prefix_cached_blocks: usize,
    pub(crate) blocks_cached: bool,
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) replayability: Replayability,
}

/// Execution payload and scheduler-issued resources for one requested
/// lifecycle transition. The planner is the only code that turns these intents
/// into worker operations.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum TransitionIntent {
    IngestText {
        segment_index: usize,
        start: u32,
        end: u32,
        logical_start: u32,
        physical_start: u32,
        token_ids: Vec<u32>,
        new_blocks: Vec<BlockId>,
        sampling_state: SamplingState,
    },
    EncodeImageStep {
        segment_index: usize,
        step_index: usize,
        step: ImageIngestStep,
        encoder_cache_key: Option<u64>,
        image_b64: String,
        source_product: Option<ProductRef>,
    },
    IngestImageState {
        segment_index: usize,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        feature: ProductRef,
        new_blocks: Vec<BlockId>,
    },
    DecodeUnd {
        logical_position: u32,
        physical_position: u32,
        new_blocks: Vec<BlockId>,
        spec_token_ids: Option<Vec<u32>>,
        sampling_state: SamplingState,
        /// The previously committed token this decode continues from — the host
        /// input the worker's forward consumes. Ignored when `relay_input` is
        /// set, in which case no host token product is attached.
        input_token: u32,
        /// A device-relay successor roots on a predecessor's not-yet-committed
        /// selected point and consumes that predecessor's on-device sampled
        /// token. It carries no host input token so the worker uses the relay.
        relay_input: bool,
    },
    PublishKv {
        image_id: u32,
        physical_kv_len: u32,
    },
    PrepareGen {
        image_id: u32,
        physical_kv_len: u32,
        latent_units: u64,
        conditioning: ProductRef,
    },
    CloseKv {
        image_id: u32,
        logical_position: u32,
        physical_position: u32,
        token: u32,
        relay_input: bool,
        new_blocks: Vec<BlockId>,
    },
    DenoiseGen {
        image_id: u32,
        start_step: u16,
        step_count: u16,
        cfg: CfgParams,
        latent_units: u64,
        conditioning: ProductRef,
        latent: ProductRef,
        physical_kv_len: u32,
    },
    CommitGen {
        image_id: u32,
        step: u16,
        latent: ProductRef,
    },
    EncodeFeedbackStep {
        image_id: u32,
        step_index: usize,
        step: ImageIngestStep,
        source: Option<ProductRef>,
        image_b64: String,
    },
    FeedbackState {
        image_id: u32,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        feature: ProductRef,
        sample_continuation: bool,
        new_blocks: Vec<BlockId>,
        sampling_state: SamplingState,
    },
}

/// The reserved output index of a host-supplied input product, kept clear of an
/// operation's declared output indices so the worker keys it distinctly.
const HOST_INPUT_OUTPUT_INDEX: u16 = u16::MAX;
const SAMPLING_INPUT_OUTPUT_INDEX: u16 = u16::MAX - 1;

/// The immutable auxiliary feature product an encode operation produces.
fn encode_outputs(
    step: ImageIngestStep,
    handle: u32,
    resources: &uniserve_core::GenerationResourceBounds,
) -> Result<Vec<ProductRef>, PlanningError> {
    let (kind, bytes) = match step {
        ImageIngestStep::VaeEncode => (
            ProductKind::LatentFeature,
            resources.max_latent_feature_bytes,
        ),
        ImageIngestStep::VitEncode => (
            ProductKind::VisionFeature,
            resources.max_vision_feature_bytes,
        ),
    };
    let mut feature = bounded_product(
        0,
        kind,
        StorageClass::LatentArena,
        DType::BF16,
        dynamic_element_bound(bytes, DType::BF16)?,
    );
    feature.generation = handle;
    Ok(vec![feature])
}

/// One immutable image-latent generation. The output remains addressed by its
/// exact operation identity and logical generation until the scheduler releases
/// it after all registered readers have fenced.
fn latent_output(
    output_index: u16,
    resources: &uniserve_core::GenerationResourceBounds,
    dtype: DType,
) -> Result<ProductRef, PlanningError> {
    Ok(bounded_product(
        output_index,
        ProductKind::Latent,
        StorageClass::LatentArena,
        dtype,
        dynamic_element_bound(resources.max_image_latent_bytes, dtype)?,
    ))
}

/// Side-effect-free lowering from one scheduler decision to its next operation.
pub(crate) fn plan(
    latent_dtype: Option<DType>,
    kv_bytes_per_token: u64,
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    mut intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    match &mut intent {
        TransitionIntent::IngestImageState {
            physical_kv_tokens, ..
        }
        | TransitionIntent::FeedbackState {
            physical_kv_tokens, ..
        } => {
            *physical_kv_tokens = bounded_worker_kv(
                *physical_kv_tokens,
                request.resources.max_kv_tokens,
                cursor.und.physical_kv_len,
            );
        }
        TransitionIntent::DenoiseGen { step_count, .. } => {
            *step_count = (*step_count).max(1);
        }
        _ => {}
    }
    match intent {
        intent @ TransitionIntent::IngestText { .. }
        | intent @ TransitionIntent::IngestImageState { .. }
        | intent @ TransitionIntent::CloseKv { .. }
        | intent @ TransitionIntent::FeedbackState { .. } => {
            plan_ar_extend(request, cursor, intent)
        }
        intent @ TransitionIntent::DecodeUnd { .. } => plan_ar_decode(request, cursor, intent),
        intent @ TransitionIntent::EncodeImageStep { .. }
        | intent @ TransitionIntent::EncodeFeedbackStep { .. } => {
            plan_encode(request, cursor, intent)
        }
        intent @ TransitionIntent::PublishKv { .. } => {
            plan_kv_publish(kv_bytes_per_token, request, cursor, intent)
        }
        intent @ TransitionIntent::PrepareGen { .. } => {
            plan_diffusion_prepare(latent_dtype, request, cursor, intent)
        }
        intent @ TransitionIntent::DenoiseGen { .. } => {
            plan_diffusion_step(latent_dtype, request, cursor, intent)
        }
        intent @ TransitionIntent::CommitGen { .. } => {
            plan_diffusion_finalize(request, cursor, intent)
        }
    }
}

fn plan_ar_extend(
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    match &intent {
        TransitionIntent::IngestText {
            start,
            end,
            token_ids,
            sampling_state,
            ..
        } => {
            if *start != cursor.ingest.prompt_cursor {
                return Err(PlanningError::PromptCursorMismatch {
                    expected: cursor.ingest.prompt_cursor,
                    actual: *start,
                });
            }
            let token_count = token_ids.len().min(u32::MAX as usize) as u32;
            if *end != start.saturating_add(token_count) {
                return Err(PlanningError::PromptCursorMismatch {
                    expected: start.saturating_add(token_count),
                    actual: *end,
                });
            }
            let prompt_positions = request
                .sampling
                .prompt_logprobs_requested()
                .then_some(token_ids.len().saturating_sub(usize::from(*start == 0)))
                .unwrap_or(0)
                .min(u32::MAX as usize) as u32;
            let outputs = token_outputs(
                logprob_blob_bound(&request.sampling, prompt_positions)?,
                !sampling_state.transition_token_ids.is_empty(),
                1,
            )?;
            finish_plan(
                request,
                cursor,
                intent,
                RunKind::ArExtend,
                Vec::new(),
                outputs,
            )
        }
        TransitionIntent::IngestImageState {
            position, feature, ..
        } => {
            if *position != cursor.und.logical_pos {
                return Err(PlanningError::LogicalCursorMismatch {
                    expected: cursor.und.logical_pos,
                    actual: *position,
                });
            }
            let inputs = vec![feature.clone()];
            finish_plan(
                request,
                cursor,
                intent,
                RunKind::ArExtend,
                inputs,
                Vec::new(),
            )
        }
        TransitionIntent::CloseKv { .. } => finish_plan(
            request,
            cursor,
            intent,
            RunKind::ArExtend,
            Vec::new(),
            vec![output_product(
                0,
                ProductKind::Completion,
                StorageClass::RequestRelay,
                DType::U8,
            )],
        ),
        TransitionIntent::FeedbackState {
            position,
            feature,
            sample_continuation,
            sampling_state,
            ..
        } => {
            if request.policy.feedback.is_none() {
                return Err(PlanningError::FeedbackDisabled);
            }
            if *position != cursor.und.logical_pos {
                return Err(PlanningError::LogicalCursorMismatch {
                    expected: cursor.und.logical_pos,
                    actual: *position,
                });
            }
            let outputs = feedback_state_outputs(
                sample_continuation
                    .then(|| logprob_blob_bound(&request.sampling, 0))
                    .transpose()?
                    .flatten(),
                *sample_continuation,
                *sample_continuation && !sampling_state.transition_token_ids.is_empty(),
            )?;
            let inputs = vec![feature.clone()];
            finish_plan(request, cursor, intent, RunKind::ArExtend, inputs, outputs)
        }
        _ => unreachable!("token-extend planner received another forward mode"),
    }
}

fn plan_ar_decode(
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    let TransitionIntent::DecodeUnd {
        logical_position,
        spec_token_ids,
        sampling_state,
        ..
    } = &intent
    else {
        unreachable!("token-decode planner received another forward mode")
    };
    if *logical_position != cursor.und.logical_pos {
        return Err(PlanningError::LogicalCursorMismatch {
            expected: cursor.und.logical_pos,
            actual: *logical_position,
        });
    }
    let draft_count = spec_token_ids.as_ref().map_or(0, Vec::len);
    let work = if draft_count == 0 {
        RunKind::ArDecode
    } else {
        RunKind::ArVerify
    };
    let max_points = 1_usize.saturating_add(draft_count).min(u32::MAX as usize) as u32;
    let outputs = token_outputs(
        logprob_blob_bound(&request.sampling, 0)?,
        !sampling_state.transition_token_ids.is_empty(),
        max_points,
    )?;
    finish_plan(request, cursor, intent, work, Vec::new(), outputs)
}

fn plan_encode(
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    let (step, inputs, adds_completion) = match &intent {
        TransitionIntent::EncodeImageStep {
            step,
            image_b64,
            source_product,
            ..
        } => {
            if source_product.is_none() && image_b64.is_empty() {
                return Err(PlanningError::MissingImageInput);
            }
            (*step, source_product.clone().into_iter().collect(), false)
        }
        TransitionIntent::EncodeFeedbackStep { step, source, .. } => {
            if !request.behavior.generated_image_feedback || request.policy.feedback.is_none() {
                return Err(PlanningError::FeedbackDisabled);
            }
            (*step, source.clone().into_iter().collect(), true)
        }
        _ => unreachable!("encode planner received another forward mode"),
    };
    let work = match step {
        ImageIngestStep::VaeEncode => RunKind::EncoderLatent,
        ImageIngestStep::VitEncode => RunKind::EncoderVision,
    };
    let mut outputs = encode_outputs(step, 0, &request.resources)?;
    if adds_completion {
        outputs.push(output_product(
            1,
            ProductKind::Completion,
            StorageClass::RequestRelay,
            DType::U8,
        ));
    }
    finish_plan(request, cursor, intent, work, inputs, outputs)
}

fn plan_kv_publish(
    kv_bytes_per_token: u64,
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    let TransitionIntent::PublishKv {
        image_id,
        physical_kv_len,
    } = intent
    else {
        unreachable!("KV publication planner received another transition")
    };
    let bytes_per_token =
        u32::try_from(kv_bytes_per_token).map_err(|_| PlanningError::ProductBoundTooLarge {
            bytes: kv_bytes_per_token,
        })?;
    if physical_kv_len == 0 || bytes_per_token == 0 {
        return Err(PlanningError::MissingProductBound);
    }
    let kv = bounded_product(
        0,
        ProductKind::Kv,
        StorageClass::PagedKv,
        DType::U8,
        ShapeBound {
            dims: vec![
                DimBound::Static(physical_kv_len),
                DimBound::Static(bytes_per_token),
            ],
        },
    );
    finish_plan(
        request,
        cursor,
        TransitionIntent::PublishKv {
            image_id,
            physical_kv_len,
        },
        RunKind::TransferKvPublish,
        Vec::new(),
        vec![
            kv,
            output_product(
                1,
                ProductKind::Completion,
                StorageClass::RequestRelay,
                DType::U8,
            ),
        ],
    )
}

fn plan_diffusion_prepare(
    latent_dtype: Option<DType>,
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    if !request.behavior.gen_output {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let TransitionIntent::PrepareGen { conditioning, .. } = &intent else {
        unreachable!("media-prepare planner received another forward mode")
    };
    let inputs = vec![conditioning.clone()];
    finish_plan(
        request,
        cursor,
        intent,
        RunKind::DiffusionPrepare,
        inputs,
        vec![
            latent_output(
                0,
                &request.resources,
                latent_dtype.ok_or(PlanningError::MissingLatentDType)?,
            )?,
            output_product(
                1,
                ProductKind::Completion,
                StorageClass::RequestRelay,
                DType::U8,
            ),
        ],
    )
}

fn plan_diffusion_step(
    latent_dtype: Option<DType>,
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    if !request.behavior.gen_output {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let TransitionIntent::DenoiseGen {
        conditioning,
        latent,
        ..
    } = &intent
    else {
        unreachable!("media-denoise planner received another forward mode")
    };
    let inputs = vec![conditioning.clone(), latent.clone()];
    finish_plan(
        request,
        cursor,
        intent,
        RunKind::DiffusionStep,
        inputs,
        vec![
            latent_output(
                0,
                &request.resources,
                latent_dtype.ok_or(PlanningError::MissingLatentDType)?,
            )?,
            output_product(
                1,
                ProductKind::Completion,
                StorageClass::RequestRelay,
                DType::U8,
            ),
        ],
    )
}

fn plan_diffusion_finalize(
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
) -> Result<NextOp, PlanningError> {
    if !request.behavior.gen_output {
        return Err(PlanningError::GenerationBranchDisabled);
    }
    let TransitionIntent::CommitGen { latent, .. } = &intent else {
        unreachable!("diffusion_finalize planner received another forward mode")
    };
    let inputs = vec![latent.clone()];
    let feedback = request
        .behavior
        .generated_image_feedback
        .then_some(request.policy.feedback.as_ref())
        .flatten();
    let mut outputs = diffusion_finalize_outputs(request, feedback)?;
    outputs.push(output_product(
        2,
        ProductKind::Completion,
        StorageClass::RequestRelay,
        DType::U8,
    ));
    finish_plan(
        request,
        cursor,
        intent,
        RunKind::DiffusionFinalize,
        inputs,
        outputs,
    )
}

fn finish_plan(
    request: &GenerationRequest,
    cursor: &GenerationCursor,
    intent: TransitionIntent,
    kind: RunKind,
    inputs: Vec<ProductRef>,
    outputs: Vec<ProductRef>,
) -> Result<NextOp, PlanningError> {
    let operation_variant = kind;
    let produces_latent = matches!(
        operation_variant,
        RunKind::DiffusionPrepare | RunKind::DiffusionStep
    );
    let is_diffusion_finalize = operation_variant == RunKind::DiffusionFinalize;
    let produces_token = outputs
        .iter()
        .any(|output| output.kind == ProductKind::Token);
    let new_blocks = match &intent {
        TransitionIntent::IngestText { new_blocks, .. }
        | TransitionIntent::IngestImageState { new_blocks, .. }
        | TransitionIntent::DecodeUnd { new_blocks, .. }
        | TransitionIntent::CloseKv { new_blocks, .. }
        | TransitionIntent::FeedbackState { new_blocks, .. } => new_blocks.clone(),
        _ => Vec::new(),
    };
    let draft_token_ids = match &intent {
        TransitionIntent::DecodeUnd { spec_token_ids, .. } => {
            spec_token_ids.clone().unwrap_or_default()
        }
        _ => Vec::new(),
    };
    let draft_count =
        (!draft_token_ids.is_empty()).then(|| draft_token_ids.len().min(u32::MAX as usize) as u32);
    let token_cost = match &intent {
        TransitionIntent::IngestText { token_ids, .. } => token_ids.len(),
        TransitionIntent::IngestImageState {
            physical_kv_tokens, ..
        }
        | TransitionIntent::FeedbackState {
            physical_kv_tokens, ..
        } => image_kv_token_bound(
            *physical_kv_tokens,
            request.resources.max_kv_tokens,
            cursor.und.physical_kv_len,
        ),
        TransitionIntent::DecodeUnd { .. } => 1 + draft_token_ids.len(),
        TransitionIntent::PublishKv { .. } => 0,
        TransitionIntent::DenoiseGen { step_count, .. } => usize::from((*step_count).max(1)),
        TransitionIntent::EncodeImageStep { .. }
        | TransitionIntent::PrepareGen { .. }
        | TransitionIntent::CloseKv { .. }
        | TransitionIntent::CommitGen { .. }
        | TransitionIntent::EncodeFeedbackStep { .. } => 1,
    };
    let cfg_branches = match &intent {
        TransitionIntent::DenoiseGen { cfg, .. } => usize::from(cfg.branch_count.max(1)),
        _ => 1,
    };
    let input_tokens = match &intent {
        TransitionIntent::IngestText { token_ids, .. } => token_ids.clone(),
        TransitionIntent::DecodeUnd {
            spec_token_ids,
            input_token,
            relay_input,
            ..
        } => match spec_token_ids {
            Some(drafts) if !drafts.is_empty() => drafts.clone(),
            _ if *relay_input => Vec::new(),
            _ => vec![*input_token],
        },
        TransitionIntent::CloseKv {
            token, relay_input, ..
        } if !relay_input => vec![*token],
        _ => Vec::new(),
    };
    let input_image_bytes = match &intent {
        TransitionIntent::EncodeImageStep {
            source_product,
            image_b64,
            ..
        } if source_product.is_none() && !image_b64.is_empty() => {
            Some(image_b64.clone().into_bytes())
        }
        TransitionIntent::EncodeFeedbackStep {
            source, image_b64, ..
        } if source.is_none() && !image_b64.is_empty() => Some(image_b64.clone().into_bytes()),
        _ => None,
    };
    let source_sampling = match &intent {
        TransitionIntent::IngestText { sampling_state, .. }
        | TransitionIntent::DecodeUnd { sampling_state, .. } => Some(sampling_state),
        TransitionIntent::FeedbackState {
            sample_continuation: true,
            sampling_state,
            ..
        } => Some(sampling_state),
        _ => None,
    };
    let sampling_state =
        source_sampling.and_then(|state| operation_sampling_delta(&request.sampling, state));
    let allowed_text_tokens = source_sampling.and_then(|state| state.allowed_token_ids.clone());
    let expected_prompt_token_ids = match &intent {
        TransitionIntent::IngestText {
            start, token_ids, ..
        } if request.sampling.prompt_logprobs_requested() => {
            let mut tokens = token_ids.clone();
            if *start == 0 && !tokens.is_empty() {
                tokens.remove(0);
            }
            Some(tokens)
        }
        _ => None,
    };
    let encoder_pins = match &intent {
        TransitionIntent::EncodeImageStep {
            encoder_cache_key, ..
        } => encoder_cache_key.iter().copied().collect(),
        _ => Vec::new(),
    };
    let replayability_after_apply = match &intent {
        TransitionIntent::CloseKv { .. }
        | TransitionIntent::DenoiseGen { .. }
        | TransitionIntent::CommitGen { .. }
        | TransitionIntent::EncodeFeedbackStep { .. }
        | TransitionIntent::FeedbackState { .. } => Replayability::NotReplayable,
        _ => cursor.replay.replayability,
    };
    let latent_units = match &intent {
        TransitionIntent::PrepareGen { latent_units, .. }
        | TransitionIntent::DenoiseGen { latent_units, .. } => *latent_units,
        _ => 0,
    };
    let kv_target_tokens = transition_kv_target(&intent).or_else(|| {
        transition_may_write_worker_defined_kv(&intent).then_some(request.resources.max_kv_tokens)
    });
    let new_blocks_len = new_blocks.len();
    let max_latent_bytes = if produces_latent {
        request.resources.max_image_latent_bytes
    } else {
        outputs
            .iter()
            .filter(|output| output.storage_class == StorageClass::LatentArena)
            .map(product_bound_bytes)
            .max()
            .unwrap_or(0)
    };
    let max_completion_bytes = outputs
        .iter()
        .filter(|output| {
            matches!(
                output.storage_class,
                StorageClass::PinnedOutput | StorageClass::HostStaging
            )
        })
        .map(product_bound_bytes)
        .fold(0_u64, u64::saturating_add);
    let max_transfer_bytes = outputs
        .iter()
        .filter(|output| output.storage_class == StorageClass::PagedKv)
        .map(product_bound_bytes)
        .fold(0_u64, u64::saturating_add);
    let resources = TransitionResources {
        new_blocks: new_blocks_len,
        kv_target_tokens,
        latent_units: if produces_latent { latent_units } else { 0 },
        cfg_branches,
        encoder_pins,
        replayability_after_apply,
        free_latent_on_apply: is_diffusion_finalize,
    };
    let validation = TransitionValidation {
        expected_denoise_step: match &intent {
            TransitionIntent::DenoiseGen {
                start_step,
                step_count,
                ..
            } => Some(start_step.saturating_add(*step_count)),
            _ => None,
        },
        expects_encoder_handle: matches!(
            operation_variant,
            RunKind::EncoderLatent | RunKind::EncoderVision
        ),
        expects_latent_generation: produces_latent,
        expects_image_artifact: is_diffusion_finalize,
        expected_image_hw: is_diffusion_finalize
            .then_some((request.image.height, request.image.width)),
        requires_kv_publication: operation_variant == RunKind::TransferKvPublish,
        expected_image_kv: match &intent {
            TransitionIntent::IngestImageState {
                physical_start,
                physical_kv_tokens,
                ..
            }
            | TransitionIntent::FeedbackState {
                physical_start,
                physical_kv_tokens,
                ..
            } => Some(ImageKvExpectation {
                base: *physical_start,
                effect: *physical_kv_tokens,
            }),
            _ => None,
        },
        allows_sampled_tokens: produces_token,
        expects_sampled_token: produces_token,
        expected_text_tokens: match &intent {
            TransitionIntent::IngestText { .. } => Some(TextTokenCountRange { min: 1, max: 1 }),
            TransitionIntent::DecodeUnd { .. } => {
                let max = draft_count.map_or(1, |count| count.saturating_add(1));
                Some(TextTokenCountRange { min: 1, max })
            }
            TransitionIntent::CloseKv { .. } => Some(TextTokenCountRange { min: 0, max: 0 }),
            TransitionIntent::FeedbackState { .. } if produces_token => {
                Some(TextTokenCountRange { min: 1, max: 1 })
            }
            _ => None,
        },
        max_accepted_draft_tokens: draft_count,
        draft_token_ids: (!draft_token_ids.is_empty()).then_some(draft_token_ids),
        finish_token_ids: source_sampling
            .map(|state| state.finish_token_ids.clone())
            .unwrap_or_default(),
        allowed_text_tokens,
        generated_logprobs_requested: produces_token
            && request.sampling.generated_logprobs_requested()
            && matches!(
                operation_variant,
                RunKind::ArExtend | RunKind::ArDecode | RunKind::ArVerify
            ),
        expected_prompt_token_ids,
    };
    let bounds = Bounds {
        max_points: if operation_variant == RunKind::ArVerify {
            token_cost.min(u32::MAX as usize) as u32
        } else {
            1
        },
        max_tokens: token_cost.min(u32::MAX as usize) as u32,
        max_kv_pages: new_blocks_len.min(u32::MAX as usize) as u32,
        max_latent_bytes,
        max_completion_bytes,
        max_transfer_bytes,
    };
    let rng = match &intent {
        TransitionIntent::PrepareGen { image_id, .. } => Some(Rng {
            seed: request.image.seed.unwrap_or(0),
            semantic_index_base: u64::from(*image_id),
            draw_layout: DrawLayout::FlowNoise,
        }),
        _ if produces_token => transition_sampling_index(&intent).map(|semantic_index_base| Rng {
            seed: request.sampling.seed.unwrap_or(0),
            semantic_index_base,
            draw_layout: DrawLayout::TargetSampling,
        }),
        _ => None,
    };
    Ok(NextOp {
        kind,
        bounds,
        inputs,
        outputs,
        predicate: None,
        rng,
        control_seq: 0,
        operation_variant,
        request_id: request.request_id,
        planned_us: uniserve_core::now_monotonic_us(),
        reserved_us: 0,
        new_blocks,
        token_cost,
        input_tokens,
        input_image_bytes,
        sampling_state,
        buffer_allocations: Vec::new(),
        intent,
        resources,
        validation,
        visibility: OutputVisibilityPlan {
            und_tokens: request.behavior.und_tokens,
            generated_image: request.behavior.gen_output,
        },
    })
}

/// The immutable products of image materialization.
///
/// Public PNG bytes and a device-resident feedback source are independent
/// products. The latter is present only for the device-product feedback route;
/// it is consumed by a later non-state encode operation.
fn diffusion_finalize_outputs(
    request: &GenerationRequest,
    feedback: Option<&uniserve_core::GeneratedImageFeedbackRecipe>,
) -> Result<Vec<ProductRef>, PlanningError> {
    let public_bytes = png_base64_bound(request.image.width, request.image.height)?;
    let mut outputs = vec![bounded_product(
        0,
        ProductKind::Artifact,
        StorageClass::PinnedOutput,
        DType::U8,
        dynamic_element_bound(public_bytes, DType::U8)?,
    )];
    if let Some(feedback) = feedback
        && feedback.source == uniserve_core::FeedbackSource::DeviceProduct
    {
        outputs.push(bounded_product(
            1,
            ProductKind::Artifact,
            StorageClass::LatentArena,
            DType::BF16,
            dynamic_element_bound(
                u64::from(request.image.height)
                    .saturating_mul(u64::from(request.image.width))
                    .saturating_mul(3)
                    .saturating_mul(dtype_bytes(DType::BF16)),
                DType::BF16,
            )?,
        ));
    }
    Ok(outputs)
}

fn transition_kv_target(intent: &TransitionIntent) -> Option<usize> {
    let target = match intent {
        TransitionIntent::IngestText {
            end,
            start,
            physical_start,
            ..
        } => physical_start.saturating_add(end.saturating_sub(*start)),
        TransitionIntent::IngestImageState {
            physical_start,
            physical_kv_tokens,
            ..
        } => physical_start.saturating_add(match physical_kv_tokens {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionIntent::FeedbackState {
            physical_start,
            physical_kv_tokens,
            ..
        } => physical_start.saturating_add(match physical_kv_tokens {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionIntent::DecodeUnd {
            physical_position, ..
        } => physical_position.saturating_add(1),
        TransitionIntent::CloseKv {
            physical_position, ..
        } => physical_position.saturating_add(1),
        TransitionIntent::EncodeImageStep { .. }
        | TransitionIntent::PublishKv { .. }
        | TransitionIntent::PrepareGen { .. }
        | TransitionIntent::DenoiseGen { .. }
        | TransitionIntent::CommitGen { .. }
        | TransitionIntent::EncodeFeedbackStep { .. } => return None,
    };
    Some(target as usize)
}

fn transition_may_write_worker_defined_kv(intent: &TransitionIntent) -> bool {
    matches!(
        intent,
        TransitionIntent::IngestImageState { .. } | TransitionIntent::FeedbackState { .. }
    )
}

fn bounded_worker_kv(
    effect: ImageKvEffect,
    max_kv_tokens: usize,
    physical_kv_len: u32,
) -> ImageKvEffect {
    match effect {
        ImageKvEffect::WorkerDefined => ImageKvEffect::Bounded {
            max_tokens: max_kv_tokens
                .saturating_sub(physical_kv_len as usize)
                .min(u32::MAX as usize) as u32,
        },
        effect => effect,
    }
}

fn image_kv_token_bound(
    effect: ImageKvEffect,
    max_kv_tokens: usize,
    physical_kv_len: u32,
) -> usize {
    match bounded_worker_kv(effect, max_kv_tokens, physical_kv_len) {
        ImageKvEffect::Exact { tokens } => tokens as usize,
        ImageKvEffect::Bounded { max_tokens } => max_tokens as usize,
        ImageKvEffect::WorkerDefined => unreachable!("worker-defined KV effects are bounded"),
    }
}

fn transition_sampling_index(intent: &TransitionIntent) -> Option<u64> {
    match intent {
        TransitionIntent::IngestText { end, .. } => Some(u64::from(*end)),
        TransitionIntent::DecodeUnd {
            logical_position, ..
        } => Some(u64::from(logical_position.saturating_add(1))),
        TransitionIntent::CloseKv { .. } => None,
        TransitionIntent::FeedbackState {
            position,
            logical_positions,
            ..
        } => Some(u64::from(
            position.saturating_add((*logical_positions).max(1)),
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PlanningError {
    #[error("prompt cursor mismatch: expected {expected}, got {actual}")]
    PromptCursorMismatch { expected: u32, actual: u32 },
    #[error("logical cursor mismatch: expected {expected}, got {actual}")]
    LogicalCursorMismatch { expected: u32, actual: u32 },
    #[error("image generation branch is disabled")]
    GenerationBranchDisabled,
    #[error("generated-image feedback is disabled")]
    FeedbackDisabled,
    #[error("image encoding requires image bytes or a source product")]
    MissingImageInput,
    #[error("product bound must be nonzero")]
    MissingProductBound,
    #[error("worker did not report a latent dtype")]
    MissingLatentDType,
    #[error("product bound {bytes} bytes exceeds the IPC representation")]
    ProductBoundTooLarge { bytes: u64 },
    #[error("product generation counter exhausted")]
    ProductGenerationExhausted,
}

/// Ephemeral scheduler builder consumed when an operation is registered.
#[derive(Debug)]
pub(crate) struct NextOp {
    pub(crate) kind: RunKind,
    pub(crate) bounds: Bounds,
    pub(crate) inputs: Vec<ProductRef>,
    pub(crate) outputs: Vec<ProductRef>,
    pub(crate) predicate: Option<ProductRef>,
    pub(crate) rng: Option<Rng>,
    pub(crate) control_seq: u64,
    pub(crate) operation_variant: RunKind,
    pub(crate) request_id: RequestId,
    /// Monotonic microsecond stamps for the two pre-registration lifecycle
    /// phases. They are carried here because an operation gains its canonical
    /// `op_id` only at registration; the scheduler backfills them onto the
    /// operation lifecycle once the identity exists.
    pub(crate) planned_us: u64,
    pub(crate) reserved_us: u64,
    pub(crate) new_blocks: Vec<BlockId>,
    pub(crate) token_cost: usize,
    /// Host-known input token values the operation's forward consumes.
    pub(crate) input_tokens: Vec<u32>,
    /// Host-supplied input image bytes an encode operation's forward consumes.
    pub(crate) input_image_bytes: Option<Vec<u8>>,
    /// Branch-local processor state consumed by this operation's sampler.
    pub(crate) sampling_state: Option<SamplingState>,
    /// Memory-owned persistent output spans reserved before registration.
    pub(crate) buffer_allocations: Vec<Allocation>,
    pub(crate) intent: TransitionIntent,
    pub(crate) resources: TransitionResources,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
}

/// Runtime-owned completion policy and validation state paired with an
/// immutable registered operation.
#[derive(Debug)]
pub(crate) struct RuntimeApply {
    pub(crate) intent: TransitionIntent,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
    pub(crate) replayability_after_apply: Replayability,
    pub(crate) new_blocks: usize,
    pub(crate) free_latent_on_apply: bool,
    pub(crate) output_event_bound: usize,
}

impl NextOp {
    /// Consume this builder into the immutable operation, its scheduler apply
    /// record, and the exact host input payloads carried by the submission.
    pub(crate) fn register(
        self,
        request_key: RequestKey,
        op_id: OpId,
        parent: Checkpoint,
        next_product_generation: &mut u64,
        output_event_bound: usize,
    ) -> Result<(Operation, RuntimeApply, Vec<ProductPayload>), PlanningError> {
        let required_generations = self
            .outputs
            .iter()
            .filter(|product| product.generation == 0)
            .count()
            + usize::from(!self.input_tokens.is_empty() || self.input_image_bytes.is_some())
            + usize::from(self.sampling_state.is_some());
        let first_generation = (*next_product_generation).max(1);
        if required_generations > 0 {
            let last_generation = first_generation
                .checked_add(required_generations as u64 - 1)
                .ok_or(PlanningError::ProductGenerationExhausted)?;
            if last_generation > u64::from(u32::MAX) {
                return Err(PlanningError::ProductGenerationExhausted);
            }
        }
        let host_input = if !self.input_tokens.is_empty() {
            Some(host_input_product(
                request_key,
                op_id,
                HOST_INPUT_OUTPUT_INDEX,
                ProductKind::Token,
                DType::U32,
                self.input_tokens.len(),
            )?)
        } else if self.input_image_bytes.is_some() {
            Some(host_input_product(
                request_key,
                op_id,
                HOST_INPUT_OUTPUT_INDEX,
                ProductKind::Artifact,
                DType::U8,
                self.input_image_bytes.as_ref().map_or(0, Vec::len),
            )?)
        } else {
            None
        };
        let sampling_bytes = self
            .sampling_state
            .as_ref()
            .map(encode_sampling_state_bytes);
        let sampling_input = sampling_bytes
            .as_ref()
            .map(|bytes| {
                host_input_product(
                    request_key,
                    op_id,
                    SAMPLING_INPUT_OUTPUT_INDEX,
                    ProductKind::SamplingState,
                    DType::U8,
                    bytes.len(),
                )
            })
            .transpose()?;
        let mut acquire_generation = || {
            let generation =
                u32::try_from((*next_product_generation).max(1)).expect("generation preflight");
            *next_product_generation = u64::from(generation) + 1;
            generation
        };
        let outputs = self
            .outputs
            .into_iter()
            .map(|mut product| {
                product.request_key = request_key;
                product.producer_op_id = op_id;
                if product.generation == 0 {
                    product.generation = acquire_generation();
                }
                product
            })
            .collect();
        let mut inputs = self.inputs;
        let host_input = host_input.map(|mut product| {
            product.generation = acquire_generation();
            product
        });
        let sampling_input = sampling_input.map(|mut product| {
            product.generation = acquire_generation();
            product
        });
        inputs.extend(host_input.iter().cloned());
        inputs.extend(sampling_input.iter().cloned());
        let operation = Operation {
            request_key,
            op_id,
            parent,
            kind: self.kind,
            payload: OpPayload::new(
                self.kind,
                self.bounds,
                inputs,
                outputs,
                self.predicate,
                self.rng,
                self.control_seq,
            ),
        }
        .sealed();
        let mut input_products = Vec::with_capacity(2);
        if let Some(product) = host_input {
            let bytes = if !self.input_tokens.is_empty() {
                encode_token_product_bytes(&self.input_tokens)
            } else {
                self.input_image_bytes.unwrap_or_default()
            };
            input_products.push(ProductPayload {
                product,
                value: InlineValue::Bytes(bytes),
            });
        }
        if let (Some(product), Some(bytes)) = (sampling_input, sampling_bytes) {
            input_products.push(ProductPayload {
                product,
                value: InlineValue::Bytes(bytes),
            });
        }
        let apply = RuntimeApply {
            intent: self.intent,
            validation: self.validation,
            visibility: self.visibility,
            replayability_after_apply: self.resources.replayability_after_apply,
            new_blocks: self.resources.new_blocks,
            free_latent_on_apply: self.resources.free_latent_on_apply,
            output_event_bound,
        };
        Ok((operation, apply, input_products))
    }
}

impl RuntimeApply {
    pub(crate) fn validate_result(
        &self,
        operation: &Operation,
        record: &ModelOutput,
        products: &[ProductPayload],
        predicated_parent_point: Option<u32>,
    ) -> Result<(), TransitionValidationError> {
        if record.request_key != operation.request_key {
            return Err(TransitionValidationError::Identity {
                detail: "request_mismatch",
            });
        }
        if operation.op_id.0 == 0 || record.op_id != operation.op_id {
            return Err(TransitionValidationError::Identity {
                detail: "operation_id_mismatch",
            });
        }
        if record.status == OpStatus::Predicated {
            let declared_parent_point = match operation.parent.point {
                CheckpointPoint::Fixed(point) => point,
                CheckpointPoint::DeviceSelected => predicated_parent_point.unwrap_or(0),
            };
            let expected_point = predicated_parent_point.unwrap_or(declared_parent_point);
            if operation.predicate().is_none()
                || record.selected_point != expected_point
                || record.token_span().len != 0
                || !record.committed_tokens().is_empty()
                || !record.product_generations.is_empty()
                || products
                    .iter()
                    .any(|product| product.product.producer_op_id == operation.op_id)
            {
                return Err(TransitionValidationError::Status {
                    detail: "invalid_predicated_result",
                });
            }
            return Ok(());
        }
        let point_valid = if operation.advances_state() {
            (1..=operation.bounds().max_points.max(1)).contains(&record.selected_point)
        } else {
            record.selected_point == 0
        };
        if !point_valid {
            return Err(TransitionValidationError::Identity {
                detail: "selected_point_mismatch",
            });
        }
        self.validation.validate(operation.kind, record, products)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputVisibilityPlan {
    pub(crate) und_tokens: UndTokenAction,
    pub(crate) generated_image: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionResources {
    pub(crate) new_blocks: usize,
    pub(crate) kv_target_tokens: Option<usize>,
    pub(crate) latent_units: u64,
    /// CFG branch count for a denoise operation; the physical denoise token cost
    /// multiplies the latent geometry by this. One for every other work variant.
    pub(crate) cfg_branches: usize,
    pub(crate) encoder_pins: Vec<u64>,
    pub(crate) replayability_after_apply: Replayability,
    pub(crate) free_latent_on_apply: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Replayability {
    Replayable,
    NotReplayable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImageKvExpectation {
    pub(crate) base: u32,
    pub(crate) effect: ImageKvEffect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionValidation {
    pub(crate) expected_denoise_step: Option<u16>,
    pub(crate) expects_encoder_handle: bool,
    pub(crate) expects_latent_generation: bool,
    pub(crate) expects_image_artifact: bool,
    pub(crate) expected_image_hw: Option<(u32, u32)>,
    pub(crate) requires_kv_publication: bool,
    pub(crate) expected_image_kv: Option<ImageKvExpectation>,
    pub(crate) allows_sampled_tokens: bool,
    pub(crate) expects_sampled_token: bool,
    pub(crate) expected_text_tokens: Option<TextTokenCountRange>,
    pub(crate) max_accepted_draft_tokens: Option<u32>,
    pub(crate) draft_token_ids: Option<Vec<u32>>,
    pub(crate) finish_token_ids: Vec<u32>,
    pub(crate) allowed_text_tokens: Option<Vec<u32>>,
    pub(crate) generated_logprobs_requested: bool,
    pub(crate) expected_prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextTokenCountRange {
    pub(crate) min: u32,
    pub(crate) max: u32,
}

impl TransitionValidation {
    fn validate(
        &self,
        operation_variant: RunKind,
        record: &ModelOutput,
        products: &[ProductPayload],
    ) -> Result<(), TransitionValidationError> {
        if record.status != OpStatus::Ok {
            return Err(TransitionValidationError::Status {
                detail: "operation_failed",
            });
        }
        if let Some(expected_step) = self.expected_denoise_step {
            let steps_done = record.logical_lengths().latent_len.min(u32::from(u16::MAX)) as u16;
            if steps_done != expected_step {
                return Err(TransitionValidationError::Progress {
                    detail: "denoise_step_mismatch",
                });
            }
        }
        // An encode operation's output feature is identified by the generation
        // its completion reports; other work variants carry no encoder handle.
        let encoder_handle = record.product_generations.first().map(|g| u64::from(*g));
        if self.expects_encoder_handle && encoder_handle.filter(|handle| *handle != 0).is_none() {
            return Err(TransitionValidationError::Product {
                detail: "missing_encoder_handle",
            });
        }
        if self.expects_latent_generation
            && record
                .product_generations
                .first()
                .copied()
                .filter(|generation| *generation != 0)
                .is_none()
        {
            return Err(TransitionValidationError::Product {
                detail: "missing_latent_generation",
            });
        }
        // The artifact rides as a base64 PNG string; dimension validation reads
        // only the IHDR header from the base64 prefix. Decoding the full frame
        // here costs hundreds of milliseconds per generated image on the
        // response path; end-to-end decodability is enforced by the consumer.
        let image_png = find_product(products, record.op_id, ProductKind::Artifact)
            .and_then(|payload| payload.value.bytes())
            .and_then(|bytes| std::str::from_utf8(bytes).ok());
        if let Some(expected) = self.expected_image_hw {
            let actual = image_png.and_then(png_artifact_dims_b64).ok_or(
                TransitionValidationError::Product {
                    detail: "missing_image_dimensions",
                },
            )?;
            if actual != expected {
                return Err(TransitionValidationError::Product {
                    detail: "image_dimensions_mismatch",
                });
            }
        }
        if self.expects_image_artifact {
            let png = image_png.filter(|png| !png.is_empty()).ok_or(
                TransitionValidationError::Product {
                    detail: "missing_image_artifact",
                },
            )?;
            let dims = png_artifact_dims_b64(png).ok_or(TransitionValidationError::Product {
                detail: "invalid_image_artifact",
            })?;
            if let Some(expected) = self.expected_image_hw
                && dims != expected
            {
                return Err(TransitionValidationError::Product {
                    detail: "invalid_image_artifact",
                });
            }
        }
        if self.requires_kv_publication
            && find_product(products, record.op_id, ProductKind::Kv).is_none()
        {
            return Err(TransitionValidationError::Product {
                detail: "missing_kv_publication",
            });
        }
        if let Some(expected) = self.expected_image_kv
            && !self.requires_kv_publication
        {
            let actual = record.logical_lengths().kv_visible_len;
            match expected.effect {
                ImageKvEffect::Exact { tokens }
                    if actual != expected.base.saturating_add(tokens) =>
                {
                    return Err(TransitionValidationError::Progress {
                        detail: "image_kv_mismatch",
                    });
                }
                ImageKvEffect::Bounded { max_tokens }
                    if actual < expected.base
                        || actual > expected.base.saturating_add(max_tokens) =>
                {
                    return Err(TransitionValidationError::Progress {
                        detail: "image_kv_mismatch",
                    });
                }
                ImageKvEffect::WorkerDefined
                | ImageKvEffect::Exact { .. }
                | ImageKvEffect::Bounded { .. } => {}
            }
        }
        let sampled_tokens = record.committed_tokens();
        if !self.allows_sampled_tokens && !sampled_tokens.is_empty() {
            return Err(TransitionValidationError::Token {
                detail: "unexpected_sampled_token",
            });
        }
        if self.expects_sampled_token && sampled_tokens.is_empty() {
            return Err(TransitionValidationError::Token {
                detail: "missing_sampled_token",
            });
        }
        let accepted_draft_tokens = if operation_variant == RunKind::ArVerify {
            let drafts = self.draft_token_ids.as_deref().unwrap_or_default();
            let listed = record.committed_tokens();
            let terminal_prefix = !listed.is_empty()
                && listed.len() <= drafts.len()
                && listed == &drafts[..listed.len()]
                && self
                    .finish_token_ids
                    .contains(listed.last().expect("nonempty verified prefix"));
            let accepted = if terminal_prefix {
                listed.len()
            } else {
                let accepted = listed.len().saturating_sub(1);
                if listed.is_empty()
                    || accepted > drafts.len()
                    || listed[..accepted] != drafts[..accepted]
                {
                    return Err(TransitionValidationError::Token {
                        detail: "verified_draft_prefix_mismatch",
                    });
                }
                accepted
            };
            Some(accepted.min(u32::MAX as usize) as u32)
        } else {
            None
        };
        if let Some(max_accepted) = self.max_accepted_draft_tokens
            && accepted_draft_tokens.is_some_and(|accepted| accepted > max_accepted)
        {
            return Err(TransitionValidationError::Token {
                detail: "accepted_draft_count_exceeded",
            });
        }
        if let Some(expected) = self.expected_text_tokens {
            let listed = sampled_tokens.len().min(u32::MAX as usize) as u32;
            let actual = listed;
            if actual < expected.min || actual > expected.max {
                return Err(TransitionValidationError::Token {
                    detail: "text_token_count_mismatch",
                });
            }
        }
        if let Some(allowed) = self.allowed_text_tokens.as_deref()
            && sampled_tokens
                .iter()
                .any(|token_id| !allowed.contains(token_id))
        {
            return Err(TransitionValidationError::Token {
                detail: "sampled_token_outside_allowed_set",
            });
        }
        let logprobs = find_product(products, record.op_id, ProductKind::Logprob)
            .and_then(|payload| payload.value.bytes())
            .and_then(|bytes| LogprobBlob::decode(bytes).ok())
            .unwrap_or_default();
        let sampled_token = sampled_tokens.last().copied();
        let generated_candidates = logprobs.top_logprobs.as_slice();
        match (
            self.generated_logprobs_requested,
            sampled_token,
            generated_candidates.is_empty(),
        ) {
            (false, _, false) | (true, None, false) => {
                return Err(TransitionValidationError::Logprob {
                    detail: "unexpected_generated_logprobs",
                });
            }
            (true, Some(_), true) => {
                return Err(TransitionValidationError::Logprob {
                    detail: "missing_generated_logprobs",
                });
            }
            (true, Some(token_id), false) => {
                if generated_candidates[0].token_id != token_id {
                    return Err(TransitionValidationError::Logprob {
                        detail: "generated_logprob_token_mismatch",
                    });
                }
                let mut token_ids = std::collections::HashSet::new();
                if generated_candidates.iter().any(|candidate| {
                    candidate.rank == 0
                        || !candidate.logprob.is_finite()
                        || !token_ids.insert(candidate.token_id)
                }) {
                    return Err(TransitionValidationError::Logprob {
                        detail: "invalid_generated_logprob_candidates",
                    });
                }
            }
            (false, _, true) | (true, None, true) => {}
        }
        match (
            self.expected_prompt_token_ids.as_deref(),
            (!logprobs.prompt_logprobs.is_empty()).then_some(logprobs.prompt_logprobs.as_slice()),
        ) {
            (None, Some(positions)) if !positions.is_empty() => {
                return Err(TransitionValidationError::Logprob {
                    detail: "unexpected_prompt_logprobs",
                });
            }
            (Some(expected), actual) => {
                let actual = actual.unwrap_or_default();
                if actual.len() != expected.len() {
                    return Err(TransitionValidationError::Logprob {
                        detail: "prompt_logprob_count_mismatch",
                    });
                }
                for (&expected_token, candidates) in expected.iter().zip(actual) {
                    let Some(first) = candidates.first() else {
                        return Err(TransitionValidationError::Logprob {
                            detail: "empty_prompt_logprob_position",
                        });
                    };
                    if first.token_id != expected_token {
                        return Err(TransitionValidationError::Logprob {
                            detail: "prompt_logprob_token_mismatch",
                        });
                    }
                    let mut seen = std::collections::HashSet::new();
                    if candidates
                        .iter()
                        .any(|candidate| candidate.rank == 0 || !seen.insert(candidate.token_id))
                    {
                        return Err(TransitionValidationError::Logprob {
                            detail: "invalid_prompt_logprob_candidates",
                        });
                    }
                }
            }
            (None, None | Some(_)) => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum TransitionValidationError {
    #[error("worker status invalid: {detail}")]
    Status { detail: &'static str },
    #[error("worker identity invalid: {detail}")]
    Identity { detail: &'static str },
    #[error("worker progress invalid: {detail}")]
    Progress { detail: &'static str },
    #[error("worker product invalid: {detail}")]
    Product { detail: &'static str },
    #[error("worker token result invalid: {detail}")]
    Token { detail: &'static str },
    #[error("worker logprob result invalid: {detail}")]
    Logprob { detail: &'static str },
}

impl TransitionValidationError {
    pub(crate) fn detail(&self) -> &'static str {
        match self {
            Self::Status { detail }
            | Self::Identity { detail }
            | Self::Progress { detail }
            | Self::Product { detail }
            | Self::Token { detail }
            | Self::Logprob { detail } => detail,
        }
    }
}
