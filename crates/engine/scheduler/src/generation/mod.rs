use std::collections::HashSet;

use uniserve_core::product_blob::LogprobBlob;
use uniserve_core::{
    BlockId, CfgParams, ContextSegment, GenerationRequest, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, RequestId, SamplingParams, SegmentPlacement, UndTokenAction,
};
use uniserve_worker_wire::{
    Bounds, CompletionRecord, CreditVector, DType, DimBound, Domain, DrawLayout, EncodeMode,
    GenMode, OpId, OpStatus, Operation, Point, PointRange, ProductKind, ProductPayload, ProductRef,
    RequestKey, ResourceClass, Rng, RouteId, SamplingState, ShapeBound, StorageClass, TokenMode,
    TransferMode, VersionRef, Work, WorkVariant, encode_sampling_state_bytes,
    encode_token_product_bytes,
};

use crate::image_artifact::png_artifact_dims_b64;

/// The scheduler's single route identity; capability negotiation collapses to one
/// route in this control plane.
const ROUTE: RouteId = RouteId(0);

/// A product reference minted by the planner for an operation output. The
/// owning `request_key` and `producer_op_id` are placeholder until
/// [`PlannedTransition::register`] stamps the real identity. The shape
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
    let max = u32::try_from(elements)
        .map_err(|_| PlanningError::ProductBoundExceedsProtocol { bytes })?;
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
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
    let png = raw
        .checked_mul(2)
        .and_then(|value| value.checked_add(1 << 20))
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
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
    let extent =
        u32::try_from(elements).map_err(|_| PlanningError::ProductBoundExceedsProtocol {
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
            .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?,
    );
    let prompt_entries = prompt.then_some(
        1_u64
            .checked_add(u64::from(sampling.n_prompt_logprobs))
            .and_then(|value| value.checked_add(requested_ids))
            .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?,
    );
    let generated_bytes = generated_entries
        .unwrap_or(0)
        .checked_mul(RANKED_LOGPROB_BYTES)
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
    let per_prompt_bytes = 4_u64
        .checked_add(
            prompt_entries
                .unwrap_or(0)
                .checked_mul(RANKED_LOGPROB_BYTES)
                .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?,
        )
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
    let prompt_bytes = u64::from(prompt_positions)
        .checked_mul(per_prompt_bytes)
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
    let bytes = 1_u64
        .checked_add(if generated { 4 } else { 0 })
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(generated_bytes))
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(prompt_bytes))
        .ok_or(PlanningError::ProductBoundExceedsProtocol { bytes: u64::MAX })?;
    Ok(Some(bytes))
}

/// The declared outputs of a token operation. The selected-token product also
/// carries the device continuation bit consumed by a registered descendant;
/// the worker masks that bit before the token reaches model input.
fn token_outputs(
    logprob_bound: Option<u64>,
    produces_finish_candidate: bool,
    max_points: u32,
) -> Result<Vec<ProductRef>, PlanningError> {
    let mut token = output_product(
        0,
        ProductKind::Token,
        StorageClass::DeviceTensor,
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
            StorageClass::DeviceTensor,
            DType::U32,
        );
        selected_point.point_range = point_range;
        let mut accepted_span = bounded_product(
            2,
            ProductKind::AcceptedSpan,
            StorageClass::DeviceTensor,
            DType::U32,
            ShapeBound {
                dims: vec![DimBound::Static(max_points.saturating_add(1))],
            },
        );
        accepted_span.point_range = point_range;
        let mut continuation = bounded_product(
            3,
            ProductKind::Continuation,
            StorageClass::DeviceTensor,
            DType::I64,
            ShapeBound {
                dims: vec![DimBound::Static(4)],
            },
        );
        continuation.point_range = point_range;
        outputs.extend([selected_point, accepted_span, continuation]);
    }
    if produces_finish_candidate {
        outputs.push(output_product(
            4,
            ProductKind::Finish,
            StorageClass::DeviceTensor,
            DType::U8,
        ));
    }
    if let Some(bytes) = logprob_bound {
        outputs.push(bounded_product(
            5,
            ProductKind::Logprob,
            StorageClass::HostStaging,
            DType::U8,
            dynamic_element_bound(bytes, DType::U8)?,
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
    produces_finish_candidate: bool,
) -> Result<Vec<ProductRef>, PlanningError> {
    let mut outputs = vec![output_product(
        0,
        ProductKind::Completion,
        StorageClass::DeviceTensor,
        DType::U8,
    )];
    if sample_continuation {
        outputs.push(output_product(
            1,
            ProductKind::Token,
            StorageClass::DeviceTensor,
            DType::U32,
        ));
        if produces_finish_candidate {
            outputs.push(output_product(
                2,
                ProductKind::Finish,
                StorageClass::DeviceTensor,
                DType::U8,
            ));
        }
        if let Some(bytes) = logprob_bound {
            outputs.push(bounded_product(
                3,
                ProductKind::Logprob,
                StorageClass::HostStaging,
                DType::U8,
                dynamic_element_bound(bytes, DType::U8)?,
            ));
        }
    }
    Ok(outputs)
}

fn produces_finish_candidate(state: &SamplingState) -> bool {
    state.force_finish || !state.finish_token_ids.is_empty()
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

/// Scheduler-private flattened context derived once at admission from ordered
/// canonical context segments.
#[derive(Debug, Clone)]
pub(crate) struct SchedulerContext {
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) negative_prompt_ids: Vec<u32>,
    pub(crate) images: Vec<SchedulerImage>,
    token_segments: Vec<SchedulerTokenSegment>,
}

#[derive(Debug, Clone)]
struct SchedulerTokenSegment {
    segment_index: usize,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SchedulerImage {
    pub(crate) segment_index: usize,
    pub(crate) hash: u64,
    pub(crate) position: u32,
    pub(crate) b64: String,
    pub(crate) ingest: ImageIngestRecipe,
}

impl SchedulerContext {
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
                    token_segments.push(SchedulerTokenSegment {
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
                    images.push(SchedulerImage {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContextLoweringError {
    InvalidRequest(String),
    ImagePositionBeyondContext { position: u32, token_count: usize },
}

/// Lifecycle phase for a canonical generation request.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GenerationPhase {
    /// Encode staged input images before continuing text prefill.
    Encode,
    IngestState,
    Prefill,
    DecodeUnd,
    CloseKv,
    PublishKv,
    TransitionGen,
    DenoiseGen,
    CommitGen,
    FeedbackEncode,
    FeedbackState,
}

/// Scheduler-owned cursor for one running request. Each mutable concern has a
/// single typed owner; transition application is the only operation that
/// commits worker-derived lifecycle progress.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationCursor {
    pub(crate) lifecycle: LifecycleCursor,
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
            lifecycle: LifecycleCursor { phase },
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
                worker_image_latent_units: 0,
                host_scratch_tokens: 0,
                latent_credit_bytes: 0,
                scratch_credit_pages: 0,
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

    /// Apply one validated result exactly once. Planning and projection never
    /// mutate the committed cursor; this is the sole cursor-delta commit path.
    pub(crate) fn apply_transition(
        &mut self,
        operation: &Operation,
        apply: &SchedulerApply,
        record: &CompletionRecord,
        products: &[ProductPayload],
    ) -> Result<(), CursorApplyError> {
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
        match apply.delta {
            TransitionDelta::IngestText {
                start,
                end,
                logical_start,
                physical_start,
                ..
            } => {
                let count = end.saturating_sub(start);
                self.ingest.prompt_cursor = self.ingest.prompt_cursor.max(end);
                self.und.logical_pos = self
                    .und
                    .logical_pos
                    .max(logical_start.saturating_add(count));
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .max(physical_start.saturating_add(count));
            }
            TransitionDelta::EncodeImageStep { .. } => {}
            TransitionDelta::IngestImageState {
                step_index,
                is_final_step,
                position,
                logical_positions,
                ..
            } => {
                self.und.physical_kv_len = record.logical_lengths.kv_visible_len;
                if is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                    self.ingest.mm_cursor = self.ingest.mm_cursor.saturating_add(1);
                    self.ingest.pending_image_step = 0;
                    self.ingest.encoded_product = None;
                    self.lifecycle.phase = GenerationPhase::Prefill;
                } else {
                    self.ingest.pending_image_step = step_index.saturating_add(1);
                    self.ingest.encoded_product = None;
                    self.lifecycle.phase = GenerationPhase::Encode;
                }
            }
            TransitionDelta::DecodeUnd {
                logical_position,
                physical_position,
            } => {
                let actual_count = record.committed_tokens.len().max(1) as u32;
                self.und.logical_pos = self
                    .und
                    .logical_pos
                    .max(logical_position.saturating_add(actual_count));
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .max(physical_position.saturating_add(actual_count));
            }
            TransitionDelta::CloseKv { .. } => {
                self.und.physical_kv_len = record.logical_lengths.kv_visible_len;
                self.lifecycle.phase = GenerationPhase::PublishKv;
            }
            TransitionDelta::PublishKv { .. } => {
                self.image_gen.conditioning = products
                    .iter()
                    .find(|payload| {
                        payload.product.producer_op_id == operation.op_id
                            && payload.product.kind == ProductKind::Kv
                    })
                    .map(|payload| payload.product.clone());
                if self.image_gen.conditioning.is_none() {
                    return Err(CursorApplyError::MissingOperationId);
                }
                self.lifecycle.phase = GenerationPhase::TransitionGen;
            }
            TransitionDelta::TransitionGen { .. } => {
                self.image_gen.latent = operation
                    .outputs
                    .iter()
                    .find(|product| product.kind == ProductKind::Latent)
                    .cloned();
                if self.image_gen.latent.is_none() {
                    return Err(CursorApplyError::MissingLatentProduct);
                }
                self.image_gen.steps_done = 0;
                self.lifecycle.phase = GenerationPhase::DenoiseGen;
            }
            TransitionDelta::DenoiseGen {
                start_step,
                step_count,
                ..
            } => {
                let steps_completed =
                    record.logical_lengths.latent_len.min(u32::from(u16::MAX)) as u16;
                self.image_gen.steps_done = self
                    .image_gen
                    .steps_done
                    .max(steps_completed.max(start_step.saturating_add(step_count)));
                self.image_gen.latent = operation
                    .outputs
                    .iter()
                    .find(|product| product.kind == ProductKind::Latent)
                    .cloned();
                if self.image_gen.latent.is_none() {
                    return Err(CursorApplyError::MissingLatentProduct);
                }
            }
            TransitionDelta::CommitGen { .. } | TransitionDelta::EncodeFeedbackStep { .. } => {}
            TransitionDelta::FeedbackState {
                step_index,
                is_final_step,
                position,
                logical_positions,
                ..
            } => {
                self.und.physical_kv_len = record.logical_lengths.kv_visible_len;
                if is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                    self.feedback.image_b64 = None;
                    self.feedback.ingest_step = 0;
                    self.feedback.source_product = None;
                    self.feedback.encoded_product = None;
                    self.lifecycle.phase = GenerationPhase::DecodeUnd;
                } else {
                    self.feedback.ingest_step = step_index.saturating_add(1);
                    self.feedback.encoded_product = None;
                    self.lifecycle.phase = GenerationPhase::FeedbackEncode;
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

    pub fn context(&self) -> &ContextCursor {
        &self.ingest
    }

    pub fn und(&self) -> &UndCursor {
        &self.und
    }

    pub fn gen_cursor(&self) -> &GenCursor {
        &self.image_gen
    }

    pub fn resources(&self) -> &ResourceCursor {
        &self.resources
    }

    pub fn replay(&self) -> &ReplayCursor {
        &self.replay
    }

    pub(crate) fn project<'a>(
        &self,
        applies: impl IntoIterator<Item = &'a SchedulerApply>,
    ) -> CursorProjection {
        let mut projection = CursorProjection {
            phase: self.lifecycle.phase,
            prompt_cursor: self.ingest.prompt_cursor,
            logical_pos: self.und.logical_pos,
            physical_kv_len: self.und.physical_kv_len,
            replayability: self.replay.replayability,
        };
        for apply in applies {
            projection.apply(apply);
        }
        projection
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CursorApplyError {
    MissingOperationId,
    MissingLatentProduct,
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
pub struct LifecycleCursor {
    pub(crate) phase: GenerationPhase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCursor {
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
pub struct UndCursor {
    pub(crate) logical_pos: u32,
    pub(crate) physical_kv_len: u32,
    pub(crate) next_token: u32,
    pub(crate) tokens_emitted: usize,
    pub(crate) text_since_image: usize,
    pub(crate) round_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenCursor {
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
pub struct FeedbackCursor {
    pub(crate) image_b64: Option<String>,
    pub(crate) ingest_step: usize,
    pub(crate) source_product: Option<ProductRef>,
    pub(crate) encoded_product: Option<ProductRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCursor {
    pub(crate) worker_registered: bool,
    pub(crate) blocks_sent: usize,
    pub(crate) reserve_worstcase: bool,
    pub(crate) worstcase_blocks: usize,
    pub(crate) worker_image_latent_units: u64,
    pub(crate) host_scratch_tokens: u64,
    pub(crate) latent_credit_bytes: u64,
    pub(crate) scratch_credit_pages: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCursor {
    pub(crate) block_hashes: Vec<u64>,
    pub(crate) prefix_cached_blocks: usize,
    pub(crate) blocks_cached: bool,
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) replayability: Replayability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CursorProjection {
    pub(crate) phase: GenerationPhase,
    pub(crate) prompt_cursor: u32,
    pub(crate) logical_pos: u32,
    pub(crate) physical_kv_len: u32,
    pub(crate) replayability: Replayability,
}

impl CursorProjection {
    fn apply(&mut self, apply: &SchedulerApply) {
        match apply.delta {
            TransitionDelta::IngestText {
                start,
                end,
                logical_start,
                physical_start,
                ..
            } => {
                let count = end.saturating_sub(start);
                self.prompt_cursor = self.prompt_cursor.max(end);
                self.logical_pos = self.logical_pos.max(logical_start.saturating_add(count));
                self.physical_kv_len = self
                    .physical_kv_len
                    .max(physical_start.saturating_add(count));
            }
            TransitionDelta::EncodeImageStep { .. } => {}
            TransitionDelta::IngestImageState {
                position,
                logical_positions,
                physical_kv_tokens,
                is_final_step,
                ..
            } => {
                let physical = match physical_kv_tokens {
                    ImageKvEffect::Exact { tokens } => tokens,
                    ImageKvEffect::Bounded { max_tokens } => max_tokens,
                    ImageKvEffect::WorkerDefined => 0,
                };
                self.physical_kv_len = self.physical_kv_len.saturating_add(physical);
                if is_final_step {
                    self.logical_pos = self
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                }
            }
            TransitionDelta::DecodeUnd {
                logical_position,
                physical_position,
            } => {
                self.logical_pos = self.logical_pos.max(logical_position.saturating_add(1));
                self.physical_kv_len = self
                    .physical_kv_len
                    .max(physical_position.saturating_add(1));
            }
            TransitionDelta::CloseKv {
                physical_position, ..
            } => {
                self.physical_kv_len = self
                    .physical_kv_len
                    .max(physical_position.saturating_add(1));
                self.phase = GenerationPhase::PublishKv;
            }
            TransitionDelta::PublishKv { .. } => {
                self.phase = GenerationPhase::TransitionGen;
            }
            TransitionDelta::TransitionGen { .. } => {
                self.phase = GenerationPhase::DenoiseGen;
            }
            TransitionDelta::DenoiseGen { .. } => {}
            TransitionDelta::CommitGen { .. } | TransitionDelta::EncodeFeedbackStep { .. } => {}
            TransitionDelta::FeedbackState {
                position,
                logical_positions,
                physical_kv_tokens,
                is_final_step,
                ..
            } => {
                let physical = match physical_kv_tokens {
                    ImageKvEffect::Exact { tokens } => tokens,
                    ImageKvEffect::Bounded { max_tokens } => max_tokens,
                    ImageKvEffect::WorkerDefined => 0,
                };
                self.physical_kv_len = self.physical_kv_len.saturating_add(physical);
                if is_final_step {
                    self.logical_pos = self
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                }
            }
        }
        if apply.replayability_after_apply == Replayability::NotReplayable {
            self.replayability = Replayability::NotReplayable;
        }
    }
}

/// Execution payload and scheduler-issued resources for one requested
/// lifecycle transition. The planner is the only code that turns these intents
/// into worker operations.
#[derive(Debug, Clone)]
pub(crate) enum TransitionIntent {
    IngestText {
        segment_index: usize,
        prompt_start: u32,
        token_ids: Vec<u32>,
        new_blocks: Vec<BlockId>,
        sampling_state: SamplingState,
    },
    EncodeImage {
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
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        feature: ProductRef,
        new_blocks: Vec<BlockId>,
    },
    DecodeUnd {
        position: u32,
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
    },
    TransitionGen {
        image_id: u32,
        latent_units: u64,
        conditioning: ProductRef,
    },
    CloseKv {
        image_id: u32,
        position: u32,
        physical_position: u32,
        token: u32,
        new_blocks: Vec<BlockId>,
    },
    DenoiseGen {
        image_id: u32,
        start_step: u16,
        step_count: u16,
        cfg: CfgParams,
        latent_units: u64,
        host_scratch_tokens: u64,
        conditioning: ProductRef,
        latent: ProductRef,
    },
    CommitGen {
        image_id: u32,
        latent: ProductRef,
    },
    EncodeFeedback {
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
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        feature: ProductRef,
        sample_continuation: bool,
        new_blocks: Vec<BlockId>,
        sampling_state: SamplingState,
    },
}

/// The wire-operation ingredients a planned intent lowers to. The host-side
/// bookkeeping (`cfg_branches`, `allowed_text_tokens`, `expected_prompt_token_ids`)
/// rides alongside because guidance, token policy, and scored prompt tokens are
/// scheduler-owned rather than flat-wire fields.
struct Wire {
    work: Work,
    domain: Domain,
    inputs: Vec<ProductRef>,
    outputs: Vec<ProductRef>,
    new_blocks: Vec<BlockId>,
    draft_token_ids: Vec<u32>,
    token_cost: usize,
    cfg_branches: usize,
    allowed_text_tokens: Option<Vec<u32>>,
    expected_prompt_token_ids: Option<Vec<u32>>,
    /// Host-known input token values a token operation's forward consumes, when
    /// the host supplies them (prompt/forced tokens, the prior committed token,
    /// or a verified draft). Empty for operations with no host token input.
    input_tokens: Vec<u32>,
    /// Host-supplied input image bytes an encode operation's forward consumes.
    input_image_bytes: Option<Vec<u8>>,
    sampling_state: Option<SamplingState>,
}

/// The reserved output index of a host-supplied input product, kept clear of an
/// operation's declared output indices so the worker keys it distinctly.
const HOST_INPUT_OUTPUT_INDEX: u16 = u16::MAX;
const SAMPLING_INPUT_OUTPUT_INDEX: u16 = u16::MAX - 1;
const KV_PUBLICATION_DESCRIPTOR_BYTES: u64 = 1 << 20;

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
) -> Result<ProductRef, PlanningError> {
    Ok(bounded_product(
        output_index,
        ProductKind::Latent,
        StorageClass::LatentArena,
        DType::BF16,
        dynamic_element_bound(resources.max_image_latent_bytes, DType::BF16)?,
    ))
}

/// Side-effect-free planner for scheduler-local worker transitions.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GenerationPlanner;

impl GenerationPlanner {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn plan(
        &self,
        request: &GenerationRequest,
        cursor: CursorProjection,
        intent: TransitionIntent,
    ) -> Result<PlannedTransition, PlanningError> {
        let und_visibility = request.behavior.und_tokens;
        let image_visible = request.behavior.gen_output;
        let (
            wire,
            delta,
            encoder_pins,
            replayability_after_apply,
            latent_units,
            host_scratch_tokens,
        ) = match intent {
            TransitionIntent::IngestText {
                segment_index,
                prompt_start,
                token_ids,
                new_blocks,
                sampling_state,
            } => {
                if prompt_start != cursor.prompt_cursor {
                    return Err(PlanningError::PromptCursorMismatch {
                        expected: cursor.prompt_cursor,
                        actual: prompt_start,
                    });
                }
                let token_count = token_ids.len() as u32;
                let end = prompt_start.saturating_add(token_count);
                let scores_prompt = request.sampling.prompt_logprobs_requested();
                // The scored prompt positions the worker reports logprobs for
                // exclude the first position when the prefill starts at zero.
                let expected_prompt_token_ids = scores_prompt.then(|| {
                    let mut tokens = token_ids.clone();
                    if prompt_start == 0 && !tokens.is_empty() {
                        tokens.remove(0);
                    }
                    tokens
                });
                let finish_candidate = produces_finish_candidate(&sampling_state);
                let allowed_text_tokens = sampling_state.allowed_token_ids.clone();
                let sampling_delta = operation_sampling_delta(&request.sampling, &sampling_state);
                (
                    Wire {
                        work: Work::Token(TokenMode::Extend),
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: token_outputs(
                            logprob_blob_bound(
                                &request.sampling,
                                expected_prompt_token_ids
                                    .as_ref()
                                    .map_or(0, |tokens| tokens.len().min(u32::MAX as usize) as u32),
                            )?,
                            finish_candidate,
                            1,
                        )?,
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: token_count as usize,
                        cfg_branches: 1,
                        allowed_text_tokens,
                        expected_prompt_token_ids,
                        input_tokens: token_ids,
                        input_image_bytes: None,
                        sampling_state: sampling_delta,
                    },
                    TransitionDelta::IngestText {
                        segment_index,
                        start: prompt_start,
                        end,
                        logical_start: cursor.logical_pos,
                        physical_start: cursor.physical_kv_len,
                    },
                    Vec::new(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::EncodeImage {
                segment_index,
                step_index,
                step,
                encoder_cache_key,
                image_b64,
                source_product,
            } => {
                if source_product.is_none() && image_b64.is_empty() {
                    return Err(PlanningError::MissingImageInput);
                }
                let work = match step {
                    ImageIngestStep::VaeEncode => Work::Encode(EncodeMode::Latent),
                    ImageIngestStep::VitEncode => Work::Encode(EncodeMode::Vision),
                };
                let has_source_product = source_product.is_some();
                (
                    Wire {
                        work,
                        domain: Domain::Und,
                        inputs: source_product.into_iter().collect(),
                        outputs: encode_outputs(step, 0, &request.resources)?,
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: (!has_source_product && !image_b64.is_empty())
                            .then(|| image_b64.clone().into_bytes()),
                        sampling_state: None,
                    },
                    TransitionDelta::EncodeImageStep {
                        segment_index,
                        step_index,
                        encoder_cache_key,
                    },
                    encoder_cache_key.into_iter().collect(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::IngestImageState {
                segment_index,
                step_index,
                is_final_step,
                position,
                logical_positions,
                physical_kv_tokens,
                feature,
                new_blocks,
            } => {
                if position != cursor.logical_pos {
                    return Err(PlanningError::LogicalCursorMismatch {
                        expected: cursor.logical_pos,
                        actual: position,
                    });
                }
                let token_cost = image_kv_token_bound(
                    physical_kv_tokens,
                    request.resources.max_kv_tokens,
                    cursor.physical_kv_len,
                );
                (
                    Wire {
                        work: Work::Token(TokenMode::Extend),
                        domain: Domain::Und,
                        inputs: vec![feature],
                        outputs: Vec::new(),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: None,
                    },
                    TransitionDelta::IngestImageState {
                        segment_index,
                        step_index,
                        is_final_step,
                        position,
                        physical_start: cursor.physical_kv_len,
                        logical_positions,
                        physical_kv_tokens,
                    },
                    Vec::new(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::DecodeUnd {
                position,
                new_blocks,
                spec_token_ids,
                sampling_state,
                input_token,
                relay_input,
            } => {
                if position != cursor.logical_pos {
                    return Err(PlanningError::LogicalCursorMismatch {
                        expected: cursor.logical_pos,
                        actual: position,
                    });
                }
                let draft_token_ids = spec_token_ids.unwrap_or_default();
                let mode = if draft_token_ids.is_empty() {
                    TokenMode::Decode
                } else {
                    TokenMode::Verify
                };
                let token_cost = 1 + draft_token_ids.len();
                // A verify supplies the drafts it checks. A device-relay
                // successor attaches no host token so the worker consumes the
                // predecessor's on-device sampled token. A plain decode
                // continues from the prior committed host token.
                let input_tokens = if !draft_token_ids.is_empty() {
                    draft_token_ids.clone()
                } else if relay_input {
                    Vec::new()
                } else {
                    vec![input_token]
                };
                let finish_candidate = produces_finish_candidate(&sampling_state);
                let allowed_text_tokens = sampling_state.allowed_token_ids.clone();
                let sampling_delta = operation_sampling_delta(&request.sampling, &sampling_state);
                (
                    Wire {
                        work: Work::Token(mode),
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: token_outputs(
                            logprob_blob_bound(&request.sampling, 0)?,
                            finish_candidate,
                            token_cost.min(u32::MAX as usize) as u32,
                        )?,
                        new_blocks,
                        draft_token_ids,
                        token_cost,
                        cfg_branches: 1,
                        allowed_text_tokens,
                        expected_prompt_token_ids: None,
                        input_tokens,
                        input_image_bytes: None,
                        sampling_state: sampling_delta,
                    },
                    TransitionDelta::DecodeUnd {
                        logical_position: position,
                        physical_position: cursor.physical_kv_len,
                    },
                    Vec::new(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::PublishKv { image_id } => {
                let output = bounded_product(
                    0,
                    ProductKind::Kv,
                    StorageClass::PagedKv,
                    DType::U8,
                    dynamic_element_bound(KV_PUBLICATION_DESCRIPTOR_BYTES, DType::U8)?,
                );
                (
                    Wire {
                        work: Work::Transfer(TransferMode::KvPublish),
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: vec![output],
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: 0,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: None,
                    },
                    TransitionDelta::PublishKv { image_id },
                    Vec::new(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::TransitionGen {
                image_id,
                latent_units,
                conditioning,
            } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                (
                    Wire {
                        work: Work::Gen(GenMode::Transition),
                        domain: Domain::Gen,
                        inputs: vec![conditioning],
                        outputs: vec![
                            latent_output(0, &request.resources)?,
                            output_product(
                                1,
                                ProductKind::Completion,
                                StorageClass::DeviceTensor,
                                DType::U32,
                            ),
                        ],
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: None,
                    },
                    TransitionDelta::TransitionGen { image_id },
                    Vec::new(),
                    cursor.replayability,
                    latent_units,
                    0,
                )
            }
            TransitionIntent::CloseKv {
                image_id,
                position,
                physical_position,
                token,
                new_blocks,
            } => (
                Wire {
                    work: Work::Token(TokenMode::Extend),
                    domain: Domain::Und,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    new_blocks,
                    draft_token_ids: Vec::new(),
                    token_cost: 1,
                    cfg_branches: 1,
                    allowed_text_tokens: None,
                    expected_prompt_token_ids: None,
                    input_tokens: vec![token],
                    input_image_bytes: None,
                    sampling_state: None,
                },
                TransitionDelta::CloseKv {
                    image_id,
                    logical_position: position,
                    physical_position,
                },
                Vec::new(),
                Replayability::NotReplayable,
                0,
                0,
            ),
            TransitionIntent::DenoiseGen {
                image_id,
                start_step,
                step_count,
                cfg,
                latent_units,
                host_scratch_tokens,
                conditioning,
                latent,
            } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                let step_count = step_count.max(1);
                (
                    Wire {
                        work: Work::Gen(GenMode::Flow),
                        domain: Domain::Gen,
                        inputs: vec![conditioning, latent],
                        outputs: vec![latent_output(0, &request.resources)?],
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: usize::from(step_count),
                        cfg_branches: usize::from(cfg.branch_count.max(1)),
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: None,
                    },
                    TransitionDelta::DenoiseGen {
                        image_id,
                        start_step,
                        step_count,
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    latent_units,
                    host_scratch_tokens,
                )
            }
            TransitionIntent::CommitGen { image_id, latent } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                let feedback = request
                    .behavior
                    .generated_image_feedback
                    .then_some(request.policy.feedback.as_ref())
                    .flatten();
                (
                    Wire {
                        work: Work::Materialize,
                        domain: Domain::Gen,
                        inputs: vec![latent],
                        outputs: materialize_outputs(request, feedback)?,
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: None,
                    },
                    TransitionDelta::CommitGen { image_id },
                    Vec::new(),
                    Replayability::NotReplayable,
                    0,
                    0,
                )
            }
            TransitionIntent::EncodeFeedback {
                image_id,
                step_index,
                step,
                source,
                image_b64,
            } => {
                if !request.behavior.generated_image_feedback || request.policy.feedback.is_none() {
                    return Err(PlanningError::FeedbackDisabled);
                }
                let work = match step {
                    ImageIngestStep::VaeEncode => Work::Encode(EncodeMode::Latent),
                    ImageIngestStep::VitEncode => Work::Encode(EncodeMode::Vision),
                };
                (
                    Wire {
                        work,
                        domain: Domain::Und,
                        inputs: source.clone().into_iter().collect(),
                        outputs: encode_outputs(step, 0, &request.resources)?,
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: (source.is_none() && !image_b64.is_empty())
                            .then(|| image_b64.clone().into_bytes()),
                        sampling_state: None,
                    },
                    TransitionDelta::EncodeFeedbackStep {
                        image_id,
                        step_index,
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    0,
                    0,
                )
            }
            TransitionIntent::FeedbackState {
                image_id,
                step_index,
                is_final_step,
                position,
                logical_positions,
                physical_kv_tokens,
                feature,
                sample_continuation,
                new_blocks,
                sampling_state,
            } => {
                if request.policy.feedback.is_none() {
                    return Err(PlanningError::FeedbackDisabled);
                }
                if position != cursor.logical_pos {
                    return Err(PlanningError::LogicalCursorMismatch {
                        expected: cursor.logical_pos,
                        actual: position,
                    });
                }
                let finish_candidate =
                    sample_continuation && produces_finish_candidate(&sampling_state);
                let allowed_text_tokens = sample_continuation
                    .then(|| sampling_state.allowed_token_ids.clone())
                    .flatten();
                let sampling_delta = sample_continuation
                    .then(|| operation_sampling_delta(&request.sampling, &sampling_state))
                    .flatten();
                let token_cost = image_kv_token_bound(
                    physical_kv_tokens,
                    request.resources.max_kv_tokens,
                    cursor.physical_kv_len,
                );
                let outputs = feedback_state_outputs(
                    sample_continuation
                        .then(|| logprob_blob_bound(&request.sampling, 0))
                        .transpose()?
                        .flatten(),
                    sample_continuation,
                    finish_candidate,
                )?;
                (
                    Wire {
                        work: Work::Token(TokenMode::Extend),
                        domain: Domain::Und,
                        inputs: vec![feature],
                        outputs,
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost,
                        cfg_branches: 1,
                        allowed_text_tokens,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                        sampling_state: sampling_delta,
                    },
                    TransitionDelta::FeedbackState {
                        image_id,
                        step_index,
                        is_final_step,
                        position,
                        physical_start: cursor.physical_kv_len,
                        logical_positions,
                        physical_kv_tokens,
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    0,
                    0,
                )
            }
        };
        let operation_variant = wire.work.variant();
        let is_flow = operation_variant == WorkVariant::GenFlow;
        let produces_latent = matches!(
            operation_variant,
            WorkVariant::GenTransition | WorkVariant::GenFlow
        );
        let is_materialize = operation_variant == WorkVariant::Materialize;
        let produces_token = wire
            .outputs
            .iter()
            .any(|output| output.kind == ProductKind::Token);
        let draft_count = (!wire.draft_token_ids.is_empty())
            .then(|| wire.draft_token_ids.len().min(u32::MAX as usize) as u32);
        let kv_target_tokens = transition_kv_target(&delta).or_else(|| {
            transition_may_write_worker_defined_kv(&delta)
                .then_some(request.resources.max_kv_tokens)
        });
        let new_blocks_len = wire.new_blocks.len();
        let max_latent_bytes = if produces_latent {
            request.resources.max_image_latent_bytes
        } else {
            wire.outputs
                .iter()
                .filter(|output| output.storage_class == StorageClass::LatentArena)
                .map(product_bound_bytes)
                .max()
                .unwrap_or(0)
        };
        let max_completion_bytes = wire
            .outputs
            .iter()
            .filter(|output| {
                matches!(
                    output.storage_class,
                    StorageClass::CompletionArena | StorageClass::HostStaging
                )
            })
            .map(product_bound_bytes)
            .fold(0_u64, u64::saturating_add);
        let max_transfer_bytes = wire
            .outputs
            .iter()
            .filter(|output| output.storage_class == StorageClass::PagedKv)
            .map(product_bound_bytes)
            .fold(0_u64, u64::saturating_add);
        let resources = TransitionResources {
            new_blocks: new_blocks_len,
            kv_target_tokens,
            host_scratch_tokens: if is_flow { host_scratch_tokens } else { 0 },
            latent_units: if produces_latent { latent_units } else { 0 },
            cfg_branches: wire.cfg_branches,
            encoder_pins,
            replayability_after_apply,
            release_on_apply: if is_materialize {
                vec![ResourceClass::ImageLatent, ResourceClass::Scratch]
            } else {
                Vec::new()
            },
        };
        let validation = TransitionValidation {
            expected_denoise_step: match delta {
                TransitionDelta::DenoiseGen {
                    start_step,
                    step_count,
                    ..
                } => Some(start_step.saturating_add(step_count)),
                _ => None,
            },
            expects_encoder_handle: matches!(
                operation_variant,
                WorkVariant::EncodeLatent | WorkVariant::EncodeVision
            ),
            expects_latent_generation: produces_latent,
            expects_image_artifact: is_materialize,
            expected_image_hw: is_materialize
                .then_some((request.image.height, request.image.width)),
            requires_kv_publication: operation_variant == WorkVariant::TransferKvPublish,
            expected_image_kv: match delta {
                TransitionDelta::IngestImageState {
                    physical_start,
                    physical_kv_tokens,
                    ..
                }
                | TransitionDelta::FeedbackState {
                    physical_start,
                    physical_kv_tokens,
                    ..
                } => Some(ImageKvExpectation {
                    base: physical_start,
                    effect: bounded_worker_kv(
                        physical_kv_tokens,
                        request.resources.max_kv_tokens,
                        physical_start,
                    ),
                }),
                _ => None,
            },
            allows_sampled_tokens: produces_token,
            expects_sampled_token: produces_token,
            expected_text_tokens: match &delta {
                TransitionDelta::IngestText { .. } => Some(TextTokenCountRange { min: 1, max: 1 }),
                TransitionDelta::DecodeUnd { .. } => {
                    let max = draft_count.map_or(1, |count| count.saturating_add(1));
                    Some(TextTokenCountRange { min: 1, max })
                }
                TransitionDelta::CloseKv { .. } => Some(TextTokenCountRange { min: 0, max: 0 }),
                TransitionDelta::FeedbackState { .. } if produces_token => {
                    Some(TextTokenCountRange { min: 1, max: 1 })
                }
                _ => None,
            },
            max_accepted_draft_tokens: draft_count,
            draft_token_ids: (!wire.draft_token_ids.is_empty())
                .then(|| wire.draft_token_ids.clone()),
            finish_token_ids: wire
                .sampling_state
                .as_ref()
                .map(|state| state.finish_token_ids.clone())
                .unwrap_or_default(),
            allowed_text_tokens: wire.allowed_text_tokens,
            generated_logprobs_requested: produces_token
                && request.sampling.generated_logprobs_requested()
                && matches!(
                    operation_variant,
                    WorkVariant::TokenExtend | WorkVariant::TokenDecode | WorkVariant::TokenVerify
                ),
            expected_prompt_token_ids: wire.expected_prompt_token_ids,
        };
        let bounds = Bounds {
            max_points: if operation_variant == WorkVariant::TokenVerify {
                wire.token_cost.min(u32::MAX as usize) as u32
            } else {
                1
            },
            max_tokens: wire.token_cost.min(u32::MAX as usize) as u32,
            max_kv_pages: new_blocks_len.min(u32::MAX as usize) as u32,
            max_latent_bytes,
            max_completion_bytes,
            max_transfer_bytes,
        };
        let rng = match &delta {
            TransitionDelta::TransitionGen { image_id } => Some(Rng {
                seed: request.image.seed.unwrap_or(0),
                semantic_index_base: u64::from(*image_id),
                draw_layout: DrawLayout::FlowNoise,
            }),
            _ if produces_token => {
                transition_sampling_index(&delta).map(|semantic_index_base| Rng {
                    seed: request.sampling.seed.unwrap_or(0),
                    semantic_index_base,
                    draw_layout: DrawLayout::TargetSampling,
                })
            }
            _ => None,
        };
        Ok(PlannedTransition {
            work: wire.work,
            route: ROUTE,
            domain: wire.domain,
            bounds,
            inputs: wire.inputs,
            outputs: wire.outputs,
            predicate: None,
            rng,
            control_seq: 0,
            reserved_credits: CreditVector::ZERO,
            operation_variant,
            request_id: request.request_id,
            planned_us: uniserve_core::now_monotonic_us(),
            reserved_us: 0,
            new_blocks: wire.new_blocks,
            kv_capacity_pages: 0,
            token_cost: wire.token_cost,
            input_tokens: wire.input_tokens,
            input_image_bytes: wire.input_image_bytes,
            sampling_state: wire.sampling_state,
            device_parent_point: None,
            delta,
            resources,
            validation,
            visibility: OutputVisibilityPlan {
                und_tokens: und_visibility,
                generated_image: image_visible,
            },
        })
    }
}

/// The immutable products of image materialization.
///
/// Public PNG bytes and a device-resident feedback source are independent
/// products. The latter is present only for the device-product feedback route;
/// it is consumed by a later non-state encode operation.
fn materialize_outputs(
    request: &GenerationRequest,
    feedback: Option<&uniserve_core::GeneratedImageFeedbackRecipe>,
) -> Result<Vec<ProductRef>, PlanningError> {
    let public_bytes = png_base64_bound(request.image.width, request.image.height)?;
    let mut outputs = vec![bounded_product(
        0,
        ProductKind::Artifact,
        StorageClass::CompletionArena,
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

fn transition_kv_target(delta: &TransitionDelta) -> Option<usize> {
    let target = match delta {
        TransitionDelta::IngestText {
            end,
            start,
            physical_start,
            ..
        } => physical_start.saturating_add(end.saturating_sub(*start)),
        TransitionDelta::IngestImageState {
            physical_start,
            physical_kv_tokens,
            ..
        } => physical_start.saturating_add(match physical_kv_tokens {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionDelta::FeedbackState {
            physical_start,
            physical_kv_tokens,
            ..
        } => physical_start.saturating_add(match physical_kv_tokens {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionDelta::DecodeUnd {
            physical_position, ..
        } => physical_position.saturating_add(1),
        TransitionDelta::CloseKv {
            physical_position, ..
        } => physical_position.saturating_add(1),
        TransitionDelta::EncodeImageStep { .. }
        | TransitionDelta::PublishKv { .. }
        | TransitionDelta::TransitionGen { .. }
        | TransitionDelta::DenoiseGen { .. }
        | TransitionDelta::CommitGen { .. }
        | TransitionDelta::EncodeFeedbackStep { .. } => return None,
    };
    Some(target as usize)
}

fn transition_may_write_worker_defined_kv(delta: &TransitionDelta) -> bool {
    matches!(
        delta,
        TransitionDelta::IngestImageState { .. } | TransitionDelta::FeedbackState { .. }
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

fn transition_sampling_index(delta: &TransitionDelta) -> Option<u64> {
    match delta {
        TransitionDelta::IngestText { end, .. } => Some(u64::from(*end)),
        TransitionDelta::DecodeUnd {
            logical_position, ..
        } => Some(u64::from(logical_position.saturating_add(1))),
        TransitionDelta::CloseKv { .. } => None,
        TransitionDelta::FeedbackState {
            position,
            logical_positions,
            ..
        } => Some(u64::from(
            position.saturating_add((*logical_positions).max(1)),
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanningError {
    PromptCursorMismatch { expected: u32, actual: u32 },
    LogicalCursorMismatch { expected: u32, actual: u32 },
    GenerationBranchDisabled,
    FeedbackDisabled,
    MissingImageInput,
    MissingProductBound,
    ProductBoundExceedsProtocol { bytes: u64 },
    ProductGenerationExhausted,
}

/// Ephemeral scheduler builder consumed when an operation is registered.
#[derive(Debug)]
pub(crate) struct PlannedTransition {
    pub(crate) work: Work,
    pub(crate) route: RouteId,
    pub(crate) domain: Domain,
    pub(crate) bounds: Bounds,
    pub(crate) inputs: Vec<ProductRef>,
    pub(crate) outputs: Vec<ProductRef>,
    pub(crate) predicate: Option<ProductRef>,
    pub(crate) rng: Option<Rng>,
    pub(crate) control_seq: u64,
    /// Exact operation-scoped vector acquired before registration.
    pub(crate) reserved_credits: CreditVector,
    pub(crate) operation_variant: WorkVariant,
    pub(crate) request_id: RequestId,
    /// Monotonic microsecond stamps for the two pre-registration lifecycle
    /// phases. They are carried here because an operation gains its canonical
    /// `op_id` only at registration; the scheduler backfills them onto the
    /// operation lifecycle once the identity exists.
    pub(crate) planned_us: u64,
    pub(crate) reserved_us: u64,
    pub(crate) new_blocks: Vec<BlockId>,
    pub(crate) kv_capacity_pages: u32,
    pub(crate) token_cost: usize,
    /// Host-known input token values the operation's forward consumes.
    pub(crate) input_tokens: Vec<u32>,
    /// Host-supplied input image bytes an encode operation's forward consumes.
    pub(crate) input_image_bytes: Option<Vec<u8>>,
    /// Branch-local processor state consumed by this operation's sampler.
    pub(crate) sampling_state: Option<SamplingState>,
    /// For a `Point::Device`-rooted successor, the scheduler-owned point index
    /// selected by its parent. `None` for a fixed-parent op, whose parent point
    /// is carried directly on the parent `Point::Fixed`.
    pub(crate) device_parent_point: Option<u32>,
    pub(crate) delta: TransitionDelta,
    pub(crate) resources: TransitionResources,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
}

/// Scheduler-owned completion policy and validation state paired with an
/// immutable registered operation.
#[derive(Debug)]
pub(crate) struct SchedulerApply {
    pub(crate) delta: TransitionDelta,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
    pub(crate) replayability_after_apply: Replayability,
    pub(crate) release_on_apply: Vec<ResourceClass>,
    pub(crate) request_latent_credit: bool,
    pub(crate) reserved_credits: CreditVector,
    pub(crate) output_credit_bound: usize,
    pub(crate) device_parent_point: Option<u32>,
}

impl PlannedTransition {
    /// Consume this builder into the immutable operation, its scheduler apply
    /// record, and the exact host input payloads carried by the submission.
    pub(crate) fn register(
        self,
        request_key: RequestKey,
        op_id: OpId,
        parent: VersionRef,
        next_product_generation: &mut u64,
        output_credit_bound: usize,
    ) -> Result<(Operation, SchedulerApply, Vec<ProductPayload>), PlanningError> {
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
        let operation = Operation::registered(
            request_key,
            op_id,
            parent,
            self.work,
            self.route,
            self.domain,
            self.bounds,
            inputs,
            outputs,
            self.kv_capacity_pages,
            self.predicate,
            self.rng,
            self.control_seq,
        );
        let mut input_products = Vec::with_capacity(2);
        if let Some(product) = host_input {
            let bytes = if !self.input_tokens.is_empty() {
                encode_token_product_bytes(&self.input_tokens)
            } else {
                self.input_image_bytes.unwrap_or_default()
            };
            input_products.push(ProductPayload { product, bytes });
        }
        if let (Some(product), Some(bytes)) = (sampling_input, sampling_bytes) {
            input_products.push(ProductPayload { product, bytes });
        }
        let apply = SchedulerApply {
            delta: self.delta,
            validation: self.validation,
            visibility: self.visibility,
            replayability_after_apply: self.resources.replayability_after_apply,
            release_on_apply: self.resources.release_on_apply,
            request_latent_credit: self.resources.latent_units > 0,
            reserved_credits: self.reserved_credits,
            output_credit_bound,
            device_parent_point: self.device_parent_point,
        };
        Ok((operation, apply, input_products))
    }
}

impl SchedulerApply {
    pub(crate) fn validate_result(
        &self,
        operation: &Operation,
        record: &CompletionRecord,
        products: &[ProductPayload],
        predicated_parent_point: Option<u32>,
    ) -> Result<(), TransitionValidationError> {
        if record.request_key != operation.request_key {
            return Err(TransitionValidationError::SessionMismatch {
                expected: operation.request_key.session_id.0,
                actual: record.request_key.session_id.0,
            });
        }
        if operation.op_id.0 == 0 || record.op_id != operation.op_id {
            return Err(TransitionValidationError::OpIdMismatch {
                expected: operation.op_id.0,
                actual: record.op_id.0,
            });
        }
        let declared_parent_point = match operation.parent.point {
            Point::Fixed {
                point_index: parent_point,
                ..
            } => parent_point,
            // A device parent carries no host-known point index. The scheduler
            // records its selected point in the validation expectations.
            Point::Device { .. } => {
                let Some(device_parent_point) = self.device_parent_point else {
                    return Err(TransitionValidationError::VersionMismatch {
                        expected_base: 0,
                        actual_base: 0,
                        actual_result: u64::from(record.selected_point),
                    });
                };
                device_parent_point
            }
        };
        if record.status == OpStatus::Predicated {
            let expected_point = predicated_parent_point.unwrap_or(declared_parent_point);
            if operation.predicate.is_none()
                || record.selected_point != expected_point
                || record.token_span.len != 0
                || !record.committed_tokens.is_empty()
                || !record.product_generations.is_empty()
                || products
                    .iter()
                    .any(|product| product.product.producer_op_id == operation.op_id)
            {
                return Err(TransitionValidationError::OperationFailed);
            }
            return Ok(());
        }
        let point_valid = if operation.advances_state {
            (1..=operation.bounds.max_points.max(1)).contains(&record.selected_point)
        } else {
            record.selected_point == 0
        };
        if !point_valid {
            return Err(TransitionValidationError::VersionMismatch {
                expected_base: 0,
                actual_base: 0,
                actual_result: u64::from(record.selected_point),
            });
        }
        self.validation
            .validate(operation.work.variant(), record, products)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputVisibilityPlan {
    pub(crate) und_tokens: UndTokenAction,
    pub(crate) generated_image: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransitionDelta {
    IngestText {
        segment_index: usize,
        start: u32,
        end: u32,
        logical_start: u32,
        physical_start: u32,
    },
    EncodeImageStep {
        segment_index: usize,
        step_index: usize,
        encoder_cache_key: Option<u64>,
    },
    IngestImageState {
        segment_index: usize,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
    },
    DecodeUnd {
        logical_position: u32,
        physical_position: u32,
    },
    CloseKv {
        image_id: u32,
        logical_position: u32,
        physical_position: u32,
    },
    PublishKv {
        image_id: u32,
    },
    TransitionGen {
        image_id: u32,
    },
    DenoiseGen {
        image_id: u32,
        start_step: u16,
        step_count: u16,
    },
    CommitGen {
        image_id: u32,
    },
    EncodeFeedbackStep {
        image_id: u32,
        step_index: usize,
    },
    FeedbackState {
        image_id: u32,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
    },
}

impl TransitionDelta {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::IngestText { .. } => "ingest_text",
            Self::EncodeImageStep { .. } => "encode_image_step",
            Self::IngestImageState { .. } => "ingest_image_state",
            Self::DecodeUnd { .. } => "decode_und",
            Self::CloseKv { .. } => "close_kv",
            Self::PublishKv { .. } => "publish_kv",
            Self::TransitionGen { .. } => "transition_gen",
            Self::DenoiseGen { .. } => "denoise_gen",
            Self::CommitGen { .. } => "commit_gen",
            Self::EncodeFeedbackStep { .. } => "encode_feedback_step",
            Self::FeedbackState { .. } => "feedback_state",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionResources {
    pub(crate) new_blocks: usize,
    pub(crate) kv_target_tokens: Option<usize>,
    pub(crate) host_scratch_tokens: u64,
    pub(crate) latent_units: u64,
    /// CFG branch count for a denoise operation; the physical denoise token cost
    /// multiplies the latent geometry by this. One for every other work variant.
    pub(crate) cfg_branches: usize,
    pub(crate) encoder_pins: Vec<u64>,
    pub(crate) replayability_after_apply: Replayability,
    pub(crate) release_on_apply: Vec<ResourceClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Replayability {
    Replayable,
    NotReplayable,
}

impl Replayability {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Replayable => "replayable",
            Self::NotReplayable => "not_replayable",
        }
    }
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
        operation_variant: WorkVariant,
        record: &CompletionRecord,
        products: &[ProductPayload],
    ) -> Result<(), TransitionValidationError> {
        if record.status != OpStatus::Ok {
            return Err(TransitionValidationError::OperationFailed);
        }
        if let Some(expected_step) = self.expected_denoise_step {
            let steps_done = record.logical_lengths.latent_len.min(u32::from(u16::MAX)) as u16;
            if steps_done != expected_step {
                return Err(TransitionValidationError::DenoiseStepMismatch {
                    expected: expected_step,
                    actual: steps_done,
                });
            }
        }
        // An encode operation's output feature is identified by the generation
        // its completion reports; other work variants carry no encoder handle.
        let encoder_handle = record.product_generations.first().map(|g| u64::from(*g));
        if self.expects_encoder_handle && encoder_handle.filter(|handle| *handle != 0).is_none() {
            return Err(TransitionValidationError::MissingEncoderHandle);
        }
        if self.expects_latent_generation
            && record
                .product_generations
                .first()
                .copied()
                .filter(|generation| *generation != 0)
                .is_none()
        {
            return Err(TransitionValidationError::MissingLatentGeneration);
        }
        // The artifact rides as a base64 PNG string; dimension validation reads
        // only the IHDR header from the base64 prefix. Decoding the full frame
        // here costs hundreds of milliseconds per generated image on the
        // response path; end-to-end decodability is enforced by the consumer.
        let image_png = find_product(products, record.op_id, ProductKind::Artifact)
            .and_then(|payload| std::str::from_utf8(&payload.bytes).ok());
        if let Some(expected) = self.expected_image_hw {
            let actual = image_png
                .and_then(png_artifact_dims_b64)
                .ok_or(TransitionValidationError::MissingImageDimensions)?;
            if actual != expected {
                return Err(TransitionValidationError::ImageDimensionsMismatch {
                    expected,
                    actual,
                });
            }
        }
        if self.expects_image_artifact {
            let png = image_png
                .filter(|png| !png.is_empty())
                .ok_or(TransitionValidationError::MissingImageArtifact)?;
            let dims = png_artifact_dims_b64(png)
                .ok_or(TransitionValidationError::InvalidImageArtifact)?;
            if let Some(expected) = self.expected_image_hw
                && dims != expected
            {
                return Err(TransitionValidationError::InvalidImageArtifact);
            }
        }
        if self.requires_kv_publication
            && find_product(products, record.op_id, ProductKind::Kv).is_none()
        {
            return Err(TransitionValidationError::MissingKvPublication);
        }
        if let Some(expected) = self.expected_image_kv
            && !self.requires_kv_publication
        {
            let actual = record.logical_lengths.kv_visible_len;
            match expected.effect {
                ImageKvEffect::Exact { tokens }
                    if actual != expected.base.saturating_add(tokens) =>
                {
                    return Err(TransitionValidationError::ImageKvMismatch {
                        expected_max: expected.base.saturating_add(tokens),
                        actual,
                    });
                }
                ImageKvEffect::Bounded { max_tokens }
                    if actual < expected.base
                        || actual > expected.base.saturating_add(max_tokens) =>
                {
                    return Err(TransitionValidationError::ImageKvMismatch {
                        expected_max: expected.base.saturating_add(max_tokens),
                        actual,
                    });
                }
                ImageKvEffect::WorkerDefined
                | ImageKvEffect::Exact { .. }
                | ImageKvEffect::Bounded { .. } => {}
            }
        }
        let sampled_tokens = record.committed_tokens.as_slice();
        if !self.allows_sampled_tokens && !sampled_tokens.is_empty() {
            return Err(TransitionValidationError::UnexpectedSampledToken { operation_variant });
        }
        if self.expects_sampled_token && sampled_tokens.is_empty() {
            return Err(TransitionValidationError::MissingSampledToken { operation_variant });
        }
        let accepted_draft_tokens = if operation_variant == WorkVariant::TokenVerify {
            let drafts = self.draft_token_ids.as_deref().unwrap_or_default();
            let listed = record.committed_tokens.as_slice();
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
                    return Err(TransitionValidationError::VerifiedDraftPrefixMismatch);
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
            return Err(TransitionValidationError::AcceptedDraftCountExceeded {
                max: max_accepted,
                actual: accepted_draft_tokens.unwrap_or_default(),
            });
        }
        if let Some(expected) = self.expected_text_tokens {
            let listed = sampled_tokens.len().min(u32::MAX as usize) as u32;
            let actual = listed;
            if actual < expected.min || actual > expected.max {
                return Err(TransitionValidationError::TextTokenCountMismatch {
                    min: expected.min,
                    max: expected.max,
                    actual,
                });
            }
        }
        if let Some(allowed) = self.allowed_text_tokens.as_deref()
            && let Some(&token_id) = sampled_tokens
                .iter()
                .find(|token_id| !allowed.contains(token_id))
        {
            return Err(TransitionValidationError::SampledTokenOutsideAllowedSet { token_id });
        }
        let logprobs = find_product(products, record.op_id, ProductKind::Logprob)
            .and_then(|payload| LogprobBlob::decode(&payload.bytes).ok())
            .unwrap_or_default();
        let sampled_token = sampled_tokens.last().copied();
        let generated_candidates = logprobs.top_logprobs.as_slice();
        match (
            self.generated_logprobs_requested,
            sampled_token,
            generated_candidates.is_empty(),
        ) {
            (false, _, false) | (true, None, false) => {
                return Err(TransitionValidationError::UnexpectedGeneratedLogprobs {
                    operation_variant,
                });
            }
            (true, Some(token_id), true) => {
                return Err(TransitionValidationError::MissingGeneratedLogprobs { token_id });
            }
            (true, Some(token_id), false) => {
                if generated_candidates[0].token_id != token_id {
                    return Err(TransitionValidationError::GeneratedLogprobTokenMismatch {
                        expected: token_id,
                        actual: generated_candidates[0].token_id,
                    });
                }
                let mut token_ids = std::collections::HashSet::new();
                if generated_candidates.iter().any(|candidate| {
                    candidate.rank == 0
                        || !candidate.logprob.is_finite()
                        || !token_ids.insert(candidate.token_id)
                }) {
                    return Err(TransitionValidationError::InvalidGeneratedLogprobCandidates);
                }
            }
            (false, _, true) | (true, None, true) => {}
        }
        match (
            self.expected_prompt_token_ids.as_deref(),
            (!logprobs.prompt_logprobs.is_empty()).then_some(logprobs.prompt_logprobs.as_slice()),
        ) {
            (None, Some(positions)) if !positions.is_empty() => {
                return Err(TransitionValidationError::UnexpectedPromptLogprobs {
                    operation_variant,
                });
            }
            (Some(expected), actual) => {
                let actual = actual.unwrap_or_default();
                if actual.len() != expected.len() {
                    return Err(TransitionValidationError::PromptLogprobCountMismatch {
                        expected: expected.len(),
                        actual: actual.len(),
                    });
                }
                for (position, (&expected_token, candidates)) in
                    expected.iter().zip(actual).enumerate()
                {
                    let Some(first) = candidates.first() else {
                        return Err(TransitionValidationError::EmptyPromptLogprobPosition {
                            position,
                        });
                    };
                    if first.token_id != expected_token {
                        return Err(TransitionValidationError::PromptLogprobTokenMismatch {
                            position,
                            expected: expected_token,
                            actual: first.token_id,
                        });
                    }
                    let mut seen = std::collections::HashSet::new();
                    if candidates
                        .iter()
                        .any(|candidate| candidate.rank == 0 || !seen.insert(candidate.token_id))
                    {
                        return Err(TransitionValidationError::InvalidPromptLogprobCandidates {
                            position,
                        });
                    }
                }
            }
            (None, None | Some(_)) => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransitionValidationError {
    OperationFailed,
    SessionMismatch {
        expected: u64,
        actual: u64,
    },
    OpIdMismatch {
        expected: u64,
        actual: u64,
    },
    VersionMismatch {
        expected_base: u64,
        actual_base: u64,
        actual_result: u64,
    },
    DenoiseStepMismatch {
        expected: u16,
        actual: u16,
    },
    MissingEncoderHandle,
    MissingLatentGeneration,
    MissingImageArtifact,
    InvalidImageArtifact,
    MissingKvPublication,
    MissingImageDimensions,
    ImageDimensionsMismatch {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    ImageKvMismatch {
        expected_max: u32,
        actual: u32,
    },
    UnexpectedSampledToken {
        operation_variant: WorkVariant,
    },
    MissingSampledToken {
        operation_variant: WorkVariant,
    },
    AcceptedDraftCountExceeded {
        max: u32,
        actual: u32,
    },
    VerifiedDraftPrefixMismatch,
    TextTokenCountMismatch {
        min: u32,
        max: u32,
        actual: u32,
    },
    SampledTokenOutsideAllowedSet {
        token_id: u32,
    },
    UnexpectedPromptLogprobs {
        operation_variant: WorkVariant,
    },
    PromptLogprobCountMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyPromptLogprobPosition {
        position: usize,
    },
    PromptLogprobTokenMismatch {
        position: usize,
        expected: u32,
        actual: u32,
    },
    InvalidPromptLogprobCandidates {
        position: usize,
    },
    UnexpectedGeneratedLogprobs {
        operation_variant: WorkVariant,
    },
    MissingGeneratedLogprobs {
        token_id: u32,
    },
    GeneratedLogprobTokenMismatch {
        expected: u32,
        actual: u32,
    },
    InvalidGeneratedLogprobCandidates,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{
        GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
        GenerationResourceBounds, ImageParams, RequestId, SamplingParams, UndVisibility,
    };
    use uniserve_worker_wire::{FinishFlags, LogicalLengths, OpStatus, TimingCounters, TokenSpan};

    fn request(id: u64, tokens: Vec<u32>) -> GenerationRequest {
        let policy = GenerationPolicyDescriptor::default();
        let constraint = GenerationConstraint::UndOnly;
        GenerationRequest {
            request_id: RequestId(id),
            context: vec![ContextSegment::UndTokens {
                token_ids: tokens.clone(),
                visibility: UndVisibility::Internal,
            }],
            negative_context: Vec::new(),
            constraint,
            behavior: GenerationBehaviorDescriptor::resolve(constraint, &policy),
            sampling: SamplingParams::default(),
            image: ImageParams::default(),
            max_und_tokens: 32,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            policy,
            resources: GenerationResourceBounds {
                context_tokens: tokens.len(),
                max_kv_tokens: tokens.len() + 32,
                ..GenerationResourceBounds::default()
            },
        }
    }

    fn prefill_transition() -> PlannedTransition {
        GenerationPlanner::new()
            .plan(
                &request(9, vec![11, 12]),
                CursorProjection {
                    phase: GenerationPhase::Prefill,
                    prompt_cursor: 0,
                    logical_pos: 0,
                    physical_kv_len: 0,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![11, 12],
                    new_blocks: Vec::new(),
                    sampling_state: SamplingState::default(),
                },
            )
            .expect("plan sequence extension")
    }

    fn digest() -> String {
        "ab".repeat(32)
    }

    fn completion(request_key: RequestKey, op_id: u64, selected_point: u32) -> CompletionRecord {
        CompletionRecord {
            request_key,
            op_id: OpId(op_id),
            completion_slot_generation: 1,
            status: OpStatus::Ok,
            selected_point,
            logical_lengths: LogicalLengths {
                token_len: 1,
                kv_visible_len: 2,
                latent_len: 0,
                ..LogicalLengths::default()
            },
            token_span: TokenSpan::default(),
            committed_tokens: vec![13],
            finish_flags: FinishFlags::default(),
            product_generations: Vec::new(),
            semantic_digest: digest(),
            error_code: None,
            timing_counters: TimingCounters::default(),
        }
    }

    #[test]
    fn planner_emits_the_closed_token_variant() {
        let transition = prefill_transition();
        assert_eq!(transition.operation_variant, WorkVariant::TokenExtend);
        assert_eq!(transition.work, Work::Token(TokenMode::Extend));
        let TransitionDelta::IngestText { start, end, .. } = transition.delta else {
            panic!("expected an ingest-text transition");
        };
        assert_eq!((start, end), (0, 2));
        assert_eq!(transition.token_cost, 2);
        assert_eq!(
            transition.rng,
            Some(Rng {
                seed: 0,
                semantic_index_base: 2,
                draw_layout: DrawLayout::TargetSampling,
            })
        );
    }

    #[test]
    fn image_state_transition_bounds_the_declared_physical_span() {
        let mut request = request(10, vec![11, 12]);
        request.resources.max_kv_tokens = 128;
        request.resources.max_vision_feature_bytes = 128;
        let feature = encode_outputs(ImageIngestStep::VitEncode, 0, &request.resources)
            .expect("bounded vision feature")
            .remove(0);
        let transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::IngestState,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestImageState {
                    segment_index: 1,
                    step_index: 0,
                    is_final_step: true,
                    position: 2,
                    logical_positions: 1,
                    physical_kv_tokens: ImageKvEffect::Exact { tokens: 17 },
                    feature,
                    new_blocks: Vec::new(),
                },
            )
            .expect("plan exact image state transition");

        assert_eq!(transition.token_cost, 17);
        assert_eq!(transition.bounds.max_tokens, 17);
        assert_eq!(transition.resources.kv_target_tokens, Some(19));
    }

    #[test]
    fn planner_coordinates_transition_noise_by_semantic_image() {
        let mut request = request(19, vec![11, 12]);
        request.constraint = GenerationConstraint::Default;
        request.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 42 };
        request.behavior =
            GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
        request.image.seed = Some(29);
        request.resources.max_image_latent_bytes = 128;
        let mut conditioning = bounded_product(
            0,
            ProductKind::Kv,
            StorageClass::PagedKv,
            DType::U8,
            dynamic_element_bound(128, DType::U8).unwrap(),
        );
        conditioning.generation = 7;

        let transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::TransitionGen,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::NotReplayable,
                },
                TransitionIntent::TransitionGen {
                    image_id: 3,
                    latent_units: 64,
                    conditioning,
                },
            )
            .expect("plan generation transition");

        assert_eq!(
            transition.rng,
            Some(Rng {
                seed: 29,
                semantic_index_base: 3,
                draw_layout: DrawLayout::FlowNoise,
            })
        );
    }

    #[test]
    fn planner_declares_a_device_finish_product_for_a_finish_candidate() {
        let transition = GenerationPlanner::new()
            .plan(
                &request(15, vec![11, 12]),
                CursorProjection {
                    phase: GenerationPhase::Prefill,
                    prompt_cursor: 0,
                    logical_pos: 0,
                    physical_kv_len: 0,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![11, 12],
                    new_blocks: Vec::new(),
                    sampling_state: SamplingState {
                        finish_token_ids: vec![2, 7],
                        ..SamplingState::default()
                    },
                },
            )
            .expect("plan finish-aware extension");

        let finish = transition
            .outputs
            .iter()
            .find(|output| output.kind == ProductKind::Finish)
            .expect("device finish output");
        assert_eq!(finish.storage_class, StorageClass::DeviceTensor);
        assert_eq!(finish.dtype, DType::U8);
    }

    #[test]
    fn planner_registers_the_exact_logprob_blob_bound() {
        let mut request = request(14, vec![11, 12, 13]);
        request.sampling.return_logprobs = true;
        request.sampling.n_logprobs = 3;
        request.sampling.return_prompt_logprobs = true;
        request.sampling.n_prompt_logprobs = 4;
        request.sampling.logprob_token_ids = vec![17, 19, 17];
        let transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::Prefill,
                    prompt_cursor: 0,
                    logical_pos: 0,
                    physical_kv_len: 0,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![11, 12, 13],
                    new_blocks: Vec::new(),
                    sampling_state: SamplingState::default(),
                },
            )
            .expect("plan scored prefill");
        let logprob = transition
            .outputs
            .iter()
            .find(|output| output.kind == ProductKind::Logprob)
            .expect("bounded logprob output");
        // One sampled value, six generated candidates, and two prompt positions
        // with seven candidates each in the canonical LogprobBlob layout.
        assert_eq!(product_bound_bytes(logprob), 261);
        assert_eq!(transition.bounds.max_completion_bytes, 261);

        let request_key = RequestKey::new(1, RequestId(14), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 0,
                semantic_digest: digest(),
            },
        };
        let mut next_generation = 1;
        let (operation, _, _) = transition
            .register(request_key, OpId(21), parent, &mut next_generation, 0)
            .expect("register scored prefill");
        operation.validate().expect("bounded operation");
    }

    #[test]
    fn transition_validates_the_token_selected_under_its_branch_state() {
        let transition = GenerationPlanner::new()
            .plan(
                &request(13, vec![11, 12]),
                CursorProjection {
                    phase: GenerationPhase::Prefill,
                    prompt_cursor: 0,
                    logical_pos: 0,
                    physical_kv_len: 0,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![11, 12],
                    new_blocks: Vec::new(),
                    sampling_state: SamplingState {
                        allowed_token_ids: Some(vec![7]),
                        ..SamplingState::default()
                    },
                },
            )
            .expect("plan constrained extension");
        let request_key = RequestKey::new(1, RequestId(13), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 0,
                semantic_digest: digest(),
            },
        };
        let mut next_generation = 1;
        let (operation, apply, input_products) = transition
            .register(request_key, OpId(19), parent, &mut next_generation, 0)
            .expect("register constrained operation");
        let sampling_payload = input_products
            .into_iter()
            .find(|payload| payload.product.kind == ProductKind::SamplingState)
            .expect("branch-local sampling input");
        assert_eq!(
            uniserve_worker_wire::decode_sampling_state_bytes(&sampling_payload.bytes)
                .expect("decode sampling input")
                .allowed_token_ids,
            Some(vec![7])
        );
        assert!(operation.inputs.contains(&sampling_payload.product));
        let record = completion(request_key, 19, 1);

        assert!(matches!(
            apply.validate_result(&operation, &record, &[], None),
            Err(TransitionValidationError::SampledTokenOutsideAllowedSet { token_id: 13 })
        ));
    }

    #[test]
    fn predicated_completion_selects_its_actual_parent_and_preserves_cursor_state() {
        let mut transition = prefill_transition();
        let request_key = RequestKey::new(1, RequestId(15), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 0,
                semantic_digest: digest(),
            },
        };
        let mut predicate = transition
            .outputs
            .iter()
            .find(|output| output.kind == ProductKind::Token)
            .expect("token operation decision predicate")
            .clone();
        predicate.request_key = request_key;
        predicate.producer_op_id = OpId(19);
        predicate.generation = 1;
        transition.predicate = Some(predicate);
        let mut next_generation = 1;
        let (operation, apply, _) = transition
            .register(request_key, OpId(20), parent, &mut next_generation, 0)
            .expect("register predicated operation");
        let mut record = completion(request_key, 20, 0);
        record.status = OpStatus::Predicated;
        record.token_span.len = 0;
        record.committed_tokens.clear();
        record.product_generations.clear();
        record.finish_flags = FinishFlags::default();

        assert_eq!(
            apply.validate_result(&operation, &record, &[], Some(0)),
            Ok(())
        );
        let mut cursor = GenerationCursor::new(GenerationPhase::Prefill, 8, false);
        let mut expected = cursor.clone();
        expected.applied_op_ids.insert(20);
        cursor
            .apply_transition(&operation, &apply, &record, &[])
            .expect("apply predicated completion");
        assert_eq!(cursor, expected);
    }

    #[test]
    fn planner_closes_the_semantic_tail_before_bounded_kv_publication() {
        let request = request(10, vec![11, 12]);
        let cursor = CursorProjection {
            phase: GenerationPhase::CloseKv,
            prompt_cursor: 2,
            logical_pos: 3,
            physical_kv_len: 2,
            replayability: Replayability::Replayable,
        };
        let closure = GenerationPlanner::new()
            .plan(
                &request,
                cursor,
                TransitionIntent::CloseKv {
                    image_id: 1,
                    position: 3,
                    physical_position: 2,
                    token: 42,
                    new_blocks: vec![BlockId(9)],
                },
            )
            .expect("plan KV closure");
        assert_eq!(closure.work, Work::Token(TokenMode::Extend));
        assert_eq!(closure.input_tokens, vec![42]);
        assert!(closure.outputs.is_empty());
        assert_eq!(closure.resources.kv_target_tokens, Some(3));

        let publication = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::PublishKv,
                    physical_kv_len: 3,
                    replayability: Replayability::NotReplayable,
                    ..cursor
                },
                TransitionIntent::PublishKv { image_id: 1 },
            )
            .expect("plan KV publication");
        assert_eq!(publication.work, Work::Transfer(TransferMode::KvPublish));
        assert!(publication.inputs.is_empty());
        assert_eq!(publication.outputs[0].kind, ProductKind::Kv);
        assert_eq!(publication.outputs[0].storage_class, StorageClass::PagedKv);
        assert_eq!(
            publication.bounds.max_transfer_bytes,
            KV_PUBLICATION_DESCRIPTOR_BYTES
        );
    }

    #[test]
    fn materialize_declares_public_and_resident_immutable_products() {
        let mut request = request(12, vec![11, 12]);
        request.constraint = GenerationConstraint::Default;
        request.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 42 };
        request.policy.feedback = Some(uniserve_core::GeneratedImageFeedbackRecipe {
            source: uniserve_core::FeedbackSource::DeviceProduct,
            next_und_token: uniserve_core::FeedbackNextToken::EndOfImage,
            ingest: ImageIngestRecipe::vit_only(2, ImageKvEffect::WorkerDefined),
            sample_continuation: true,
        });
        request.behavior =
            GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
        request.resources.max_image_latent_bytes = 128;
        let latent = latent_output(0, &request.resources).expect("bounded latent");

        let transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::CommitGen,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::NotReplayable,
                },
                TransitionIntent::CommitGen {
                    image_id: 1,
                    latent,
                },
            )
            .expect("plan image materialization");
        assert_eq!(transition.outputs.len(), 2);
        assert_eq!(
            (
                transition.outputs[0].kind,
                transition.outputs[0].storage_class,
                transition.outputs[0].dtype,
            ),
            (
                ProductKind::Artifact,
                StorageClass::CompletionArena,
                DType::U8,
            )
        );
        assert_eq!(
            (
                transition.outputs[1].kind,
                transition.outputs[1].storage_class,
                transition.outputs[1].dtype,
            ),
            (
                ProductKind::Artifact,
                StorageClass::LatentArena,
                DType::BF16,
            )
        );
    }

    #[test]
    fn transition_accepts_only_the_registered_completion() {
        let transition = prefill_transition();
        let request_key = RequestKey::new(1, RequestId(9), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 5,
                semantic_digest: digest(),
            },
        };
        let mut next_product_generation = 1_u64;
        let (operation, apply, _) = transition
            .register(
                request_key,
                OpId(17),
                parent,
                &mut next_product_generation,
                0,
            )
            .expect("generation space");
        let mut generations = operation
            .outputs
            .iter()
            .chain(operation.inputs.iter())
            .map(|product| product.generation)
            .collect::<Vec<_>>();
        assert!(generations.iter().all(|generation| *generation > 0));
        let generation_count = generations.len();
        generations.sort_unstable();
        generations.dedup();
        assert_eq!(generations.len(), generation_count);

        let record = completion(request_key, 17, 1);
        assert_eq!(
            apply.validate_result(&operation, &record, &[], None),
            Ok(())
        );

        let stale = completion(request_key, 17, 2);
        assert!(matches!(
            apply.validate_result(&operation, &stale, &[], None),
            Err(TransitionValidationError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn product_generation_space_exhausts_without_reusing_an_identity() {
        let mut transition = prefill_transition();
        transition.outputs.truncate(1);
        transition.input_tokens.clear();
        transition.sampling_state = None;
        let request_key = RequestKey::new(1, RequestId(9), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 5,
                semantic_digest: digest(),
            },
        };
        let mut next_product_generation = u64::from(u32::MAX);
        let (operation, _, _) = transition
            .register(
                request_key,
                OpId(17),
                parent.clone(),
                &mut next_product_generation,
                0,
            )
            .expect("last generation");
        assert_eq!(operation.outputs[0].generation, u32::MAX);

        let mut exhausted = prefill_transition();
        exhausted.outputs.truncate(1);
        exhausted.input_tokens.clear();
        exhausted.sampling_state = None;
        assert_eq!(
            exhausted
                .register(
                    request_key,
                    OpId(18),
                    parent,
                    &mut next_product_generation,
                    0,
                )
                .map(|_| ()),
            Err(PlanningError::ProductGenerationExhausted)
        );
        assert_eq!(next_product_generation, u64::from(u32::MAX) + 1);
    }
}
