use std::collections::HashSet;

use uniserve_core::product_blob::{LogprobBlob, RankedToken};
use uniserve_core::{
    BlockId, CfgParams, ContextSegment, GenerationRequest, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, RequestId, SegmentPlacement, UndTokenAction,
};
use uniserve_worker_wire::{
    Bounds, CompletionRecord, DType, Domain, EncodeMode, GenMode, OpId, Operation, Point,
    PointRange, ProductKind, ProductPayload, ProductRef, RequestKey, ResourceClass, Rng, RouteId,
    ShapeBound, StorageClass, TokenMode, TransferMode, VersionRef, Work, WorkVariant,
    encode_token_product_bytes,
};

use crate::image_artifact::png_artifact_dims_b64;

/// The scheduler's single route identity; capability negotiation collapses to one
/// route in this control plane.
const ROUTE: RouteId = RouteId(0);

/// A product reference minted by the planner for an operation output. The
/// owning `request_key` and `producer_op_id` are placeholder until
/// [`PlannedTransition::assign_operation`] stamps the real identity. The shape
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

/// A host-supplied input product reference for an operation's forward, owned by
/// the consuming operation's identity at a reserved input index so it never
/// collides with the operation's declared outputs. Its value is carried in the
/// batch's `input_products` under this same identity.
fn host_input_product(
    request_key: RequestKey,
    op_id: OpId,
    kind: ProductKind,
    dtype: DType,
) -> ProductRef {
    ProductRef {
        request_key,
        producer_op_id: op_id,
        output_index: HOST_INPUT_OUTPUT_INDEX,
        generation: 0,
        kind,
        storage_class: StorageClass::HostStaging,
        dtype,
        shape_bound: ShapeBound::default(),
        point_range: PointRange::default(),
    }
}

/// The declared outputs of a token operation: a committed-token product, an
/// optional logprob product when logprobs were requested, and an optional KV
/// product when the operation publishes generation conditioning.
fn token_outputs(logprobs: bool, publishes_conditioning: bool) -> Vec<ProductRef> {
    let mut outputs = vec![output_product(
        0,
        ProductKind::Token,
        StorageClass::DeviceTensor,
        DType::U32,
    )];
    if logprobs {
        outputs.push(output_product(
            1,
            ProductKind::Logprob,
            StorageClass::HostStaging,
            DType::U8,
        ));
    }
    if publishes_conditioning {
        outputs.push(output_product(
            2,
            ProductKind::Kv,
            StorageClass::PagedKv,
            DType::BF16,
        ));
    }
    outputs
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
    Prefill,
    DecodeUnd,
    DenoiseGen,
    CommitGen,
    CommitWriteback,
    FeedbackIngest,
}

/// Scheduler-owned cursor for one running request. Each mutable concern has a
/// single typed owner; transition application is the only operation that
/// commits worker-derived lifecycle progress.
#[derive(Debug, Clone)]
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
                transient_encoder_handles: Vec::new(),
                pending_image_step: 0,
                staged_image: None,
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
                image_hw: (0, 0),
                conditioning: None,
            },
            feedback: FeedbackCursor {
                locator: None,
                sampled_token: None,
                sampled_logprob: None,
                top_logprobs: None,
                image_b64: None,
                ingest_step: 0,
                staged_image: None,
            },
            resources: ResourceCursor {
                worker_registered: false,
                blocks_sent: 0,
                reserve_worstcase,
                worstcase_blocks,
                worker_image_latent_units: 0,
                host_scratch_tokens: 0,
            },
            replay: ReplayCursor {
                preempted: false,
                block_hashes: Vec::new(),
                prefix_cached_blocks: 0,
                blocks_cached: false,
                generated_ids: Vec::new(),
                recompute_ids: None,
                replayability: Replayability::Replayable,
            },
            applied_op_ids: HashSet::new(),
        }
    }

    /// Apply one validated result exactly once. Planning and projection never
    /// mutate the committed cursor; this is the sole cursor-delta commit path.
    pub(crate) fn apply_transition(
        &mut self,
        transition: &PlannedTransition,
        record: &CompletionRecord,
        products: &[ProductPayload],
    ) -> Result<(), CursorApplyError> {
        let op_id = transition.operation.as_ref().map_or(0, |op| op.op_id.0);
        if op_id == 0 {
            return Err(CursorApplyError::MissingOperationId);
        }
        if !self.applied_op_ids.insert(op_id) {
            return Err(CursorApplyError::DuplicateOperation { op_id });
        }
        match transition.delta {
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
            TransitionDelta::IngestImageStep {
                step_index,
                is_final_step,
                position,
                logical_positions,
                ..
            } => {
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .saturating_add(record.logical_lengths.kv_visible_len);
                if is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                    self.ingest.mm_cursor = self.ingest.mm_cursor.saturating_add(1);
                    self.ingest.pending_image_step = 0;
                    self.ingest.staged_image = None;
                    self.lifecycle.phase = GenerationPhase::Prefill;
                } else {
                    self.ingest.pending_image_step = step_index.saturating_add(1);
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
            }
            TransitionDelta::CommitGen {
                logical_position,
                physical_position,
                logical_positions,
                physical_kv_tokens,
                ..
            } => {
                // A commit that writes its image KV back publishes a KV product;
                // that defers the KV advance to the writeback transfer, so it is
                // only applied here for a direct commit that publishes no KV.
                let has_locator = find_product(products, record.op_id, ProductKind::Kv).is_some();
                if !has_locator && physical_kv_tokens.is_some() {
                    let added = record.logical_lengths.kv_visible_len;
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(logical_position.saturating_add(logical_positions));
                    self.und.physical_kv_len = self
                        .und
                        .physical_kv_len
                        .max(physical_position.saturating_add(added));
                }
            }
            TransitionDelta::Feedback {
                logical_position,
                physical_position,
                logical_positions,
                ..
            } => {
                let added = record.logical_lengths.kv_visible_len;
                self.und.logical_pos = self
                    .und
                    .logical_pos
                    .max(logical_position.saturating_add(logical_positions));
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .max(physical_position.saturating_add(added));
                self.feedback.locator = None;
            }
            TransitionDelta::FeedbackIngestStep {
                step_index,
                is_final_step,
                position,
                logical_positions,
                ..
            } => {
                self.und.physical_kv_len = self
                    .und
                    .physical_kv_len
                    .saturating_add(record.logical_lengths.kv_visible_len);
                if is_final_step {
                    self.und.logical_pos = self
                        .und
                        .logical_pos
                        .max(position.saturating_add(logical_positions));
                    self.feedback.image_b64 = None;
                    self.feedback.ingest_step = 0;
                    self.feedback.staged_image = None;
                    self.lifecycle.phase = GenerationPhase::DecodeUnd;
                } else {
                    self.feedback.ingest_step = step_index.saturating_add(1);
                    self.lifecycle.phase = GenerationPhase::FeedbackIngest;
                }
            }
        }
        self.replay.replayability = match (
            self.replay.replayability,
            transition.resources.replayability_after_apply,
        ) {
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
        transitions: impl IntoIterator<Item = &'a PlannedTransition>,
    ) -> CursorProjection {
        let mut projection = CursorProjection {
            phase: self.lifecycle.phase,
            prompt_cursor: self.ingest.prompt_cursor,
            logical_pos: self.und.logical_pos,
            physical_kv_len: self.und.physical_kv_len,
            replayability: self.replay.replayability,
        };
        for transition in transitions {
            projection.apply(transition);
        }
        projection
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CursorApplyError {
    MissingOperationId,
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
    pub(crate) transient_encoder_handles: Vec<u64>,
    pub(crate) pending_image_step: usize,
    pub(crate) staged_image: Option<u64>,
    pub(crate) round_closing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncoderCachePin {
    pub(crate) key: u64,
    pub(crate) handle: u64,
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
    pub(crate) image_hw: (u32, u32),
    /// The published KV product a denoise op conditions on, captured from the
    /// producing operation's completion products.
    pub(crate) conditioning: Option<ProductRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackCursor {
    /// The committed KV product the writeback transfer installs, captured from
    /// the commit operation's completion products.
    pub(crate) locator: Option<ProductRef>,
    pub(crate) sampled_token: Option<u32>,
    pub(crate) sampled_logprob: Option<f32>,
    pub(crate) top_logprobs: Option<Vec<RankedToken>>,
    pub(crate) image_b64: Option<String>,
    pub(crate) ingest_step: usize,
    pub(crate) staged_image: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCursor {
    pub(crate) worker_registered: bool,
    pub(crate) blocks_sent: usize,
    pub(crate) reserve_worstcase: bool,
    pub(crate) worstcase_blocks: usize,
    pub(crate) worker_image_latent_units: u64,
    pub(crate) host_scratch_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCursor {
    pub(crate) preempted: bool,
    pub(crate) block_hashes: Vec<u64>,
    pub(crate) prefix_cached_blocks: usize,
    pub(crate) blocks_cached: bool,
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) recompute_ids: Option<Vec<u32>>,
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
    fn apply(&mut self, transition: &PlannedTransition) {
        match transition.delta {
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
            TransitionDelta::IngestImageStep {
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
            TransitionDelta::DenoiseGen { .. } => {}
            TransitionDelta::CommitGen {
                logical_position,
                physical_position,
                logical_positions,
                physical_kv_tokens,
                ..
            }
            | TransitionDelta::Feedback {
                logical_position,
                physical_position,
                logical_positions,
                physical_kv_tokens,
                ..
            } => {
                self.logical_pos = self
                    .logical_pos
                    .max(logical_position.saturating_add(logical_positions));
                if let Some(effect) = physical_kv_tokens {
                    let physical = match effect {
                        ImageKvEffect::Exact { tokens } => tokens,
                        ImageKvEffect::Bounded { max_tokens } => max_tokens,
                        ImageKvEffect::WorkerDefined => 0,
                    };
                    self.physical_kv_len = self
                        .physical_kv_len
                        .max(physical_position.saturating_add(physical));
                }
            }
            TransitionDelta::FeedbackIngestStep {
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
        if transition.resources.replayability_after_apply == Replayability::NotReplayable {
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
        allowed_tokens: Option<Vec<u32>>,
    },
    IngestImage {
        segment_index: usize,
        step_index: usize,
        step: ImageIngestStep,
        is_final_step: bool,
        position: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        encoder_cache_key: Option<u64>,
        cache_hit: bool,
        image_b64: String,
        staged_image: Option<u64>,
        new_blocks: Vec<BlockId>,
    },
    DecodeUnd {
        position: u32,
        new_blocks: Vec<BlockId>,
        spec_token_ids: Option<Vec<u32>>,
        allowed_tokens: Option<Vec<u32>>,
        /// The previously committed token this decode continues from — the host
        /// input the worker's forward consumes. Ignored when `relay_input` is
        /// set, in which case no host token product is attached.
        input_token: u32,
        /// A device-relay successor roots on a predecessor's not-yet-committed
        /// selected point and consumes that predecessor's on-device sampled
        /// token. It carries no host input token so the worker uses the relay.
        relay_input: bool,
    },
    DenoiseGen {
        image_id: u32,
        start_step: u16,
        step_count: u16,
        cfg: CfgParams,
        latent_units: u64,
        host_scratch_tokens: u64,
        conditioning: Option<ProductRef>,
    },
    CommitGen {
        image_id: u32,
        position: u32,
        new_blocks: Vec<BlockId>,
        allowed_tokens: Option<Vec<u32>>,
    },
    Feedback {
        image_id: u32,
        position: u32,
        locator: ProductRef,
        new_blocks: Vec<BlockId>,
        allowed_tokens: Option<Vec<u32>>,
    },
    FeedbackIngest {
        image_id: u32,
        step_index: usize,
        step: ImageIngestStep,
        is_final_step: bool,
        position: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        image_b64: String,
        staged_image: Option<u64>,
        new_blocks: Vec<BlockId>,
    },
}

/// The wire-operation ingredients a planned intent lowers to. The host-side
/// bookkeeping (`cfg_branches`, `allowed_text_tokens`, `expected_prompt_token_ids`)
/// rides alongside because the flat wire operation no longer carries guidance,
/// token policy, or scored prompt tokens.
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
}

/// The reserved output index of a host-supplied input product, kept clear of an
/// operation's declared output indices so the worker keys it distinctly.
const HOST_INPUT_OUTPUT_INDEX: u16 = u16::MAX;

/// The allowed-token mask a token operation validates its committed tokens
/// against, dropped when empty.
fn filter_allowed(allowed_tokens: Option<Vec<u32>>) -> Option<Vec<u32>> {
    allowed_tokens.filter(|tokens| !tokens.is_empty())
}

/// The immutable auxiliary feature product an encode operation produces.
fn encode_outputs(step: ImageIngestStep, handle: u32) -> Vec<ProductRef> {
    let kind = match step {
        ImageIngestStep::VaeEncode => ProductKind::LatentFeature,
        ImageIngestStep::VitEncode => ProductKind::VisionFeature,
    };
    let mut feature = output_product(0, kind, StorageClass::LatentArena, DType::BF16);
    feature.generation = handle;
    vec![feature]
}

/// The nonzero handle identifying an encode operation's feature product. A step
/// that reuses a staged feature echoes its handle so the successor validates
/// against the same content identity; a fresh step derives a stable nonzero
/// handle from the content-scoped fallback (encoder cache key or image id).
fn encode_handle(staged: Option<u64>, fallback: u64) -> u32 {
    (staged.unwrap_or(fallback) as u32) | 1
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
                allowed_tokens,
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
                (
                    Wire {
                        work: Work::Token(TokenMode::Extend),
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: token_outputs(
                            scores_prompt || request.sampling.generated_logprobs_requested(),
                            request.behavior.gen_output,
                        ),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: token_count as usize,
                        cfg_branches: 1,
                        allowed_text_tokens: filter_allowed(allowed_tokens),
                        expected_prompt_token_ids,
                        input_tokens: token_ids,
                        input_image_bytes: None,
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
            TransitionIntent::IngestImage {
                segment_index,
                step_index,
                step,
                is_final_step,
                position,
                logical_positions,
                physical_kv_tokens,
                encoder_cache_key,
                cache_hit,
                image_b64,
                staged_image,
                new_blocks,
            } => {
                if position != cursor.logical_pos {
                    return Err(PlanningError::LogicalCursorMismatch {
                        expected: cursor.logical_pos,
                        actual: position,
                    });
                }
                if !cache_hit && staged_image.is_none() && image_b64.is_empty() {
                    return Err(PlanningError::MissingImageInput);
                }
                let work = match step {
                    ImageIngestStep::VaeEncode => Work::Encode(EncodeMode::Latent),
                    ImageIngestStep::VitEncode => Work::Encode(EncodeMode::Vision),
                };
                let handle = encode_handle(
                    staged_image,
                    encoder_cache_key.unwrap_or_else(|| u64::from(position)),
                );
                (
                    Wire {
                        work,
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: encode_outputs(step, handle),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: (!image_b64.is_empty())
                            .then(|| image_b64.clone().into_bytes()),
                    },
                    TransitionDelta::IngestImageStep {
                        segment_index,
                        step_index,
                        is_final_step,
                        position,
                        physical_start: cursor.physical_kv_len,
                        logical_positions,
                        physical_kv_tokens,
                        encoder_cache_key,
                        cache_hit,
                        expected_encoder_handle: staged_image,
                    },
                    encoder_cache_key.into_iter().collect(),
                    cursor.replayability,
                    0,
                    0,
                )
            }
            TransitionIntent::DecodeUnd {
                position,
                new_blocks,
                spec_token_ids,
                allowed_tokens,
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
                let publishes_conditioning = !conditioning_trigger_tokens(request).is_empty();
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
                (
                    Wire {
                        work: Work::Token(mode),
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: token_outputs(
                            request.sampling.generated_logprobs_requested(),
                            publishes_conditioning,
                        ),
                        new_blocks,
                        draft_token_ids,
                        token_cost,
                        cfg_branches: 1,
                        allowed_text_tokens: filter_allowed(allowed_tokens),
                        expected_prompt_token_ids: None,
                        input_tokens,
                        input_image_bytes: None,
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
            TransitionIntent::DenoiseGen {
                image_id,
                start_step,
                step_count,
                cfg,
                latent_units,
                host_scratch_tokens,
                conditioning,
            } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                let step_count = step_count.max(1);
                (
                    Wire {
                        work: Work::Gen(GenMode::Flow),
                        domain: Domain::Gen,
                        inputs: conditioning.into_iter().collect(),
                        outputs: vec![output_product(
                            0,
                            ProductKind::Latent,
                            StorageClass::LatentArena,
                            DType::BF16,
                        )],
                        new_blocks: Vec::new(),
                        draft_token_ids: Vec::new(),
                        token_cost: usize::from(step_count),
                        cfg_branches: usize::from(cfg.branch_count.max(1)),
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
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
            TransitionIntent::CommitGen {
                image_id,
                position,
                new_blocks,
                allowed_tokens,
            } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                let feedback = request
                    .behavior
                    .generated_image_feedback
                    .then_some(request.policy.feedback.as_ref())
                    .flatten()
                    .filter(|recipe| {
                        matches!(recipe.writeback, uniserve_core::FeedbackWriteback::DirectKv)
                    });
                (
                    Wire {
                        work: Work::Materialize,
                        domain: Domain::Gen,
                        inputs: Vec::new(),
                        outputs: materialize_outputs(
                            request.sampling.generated_logprobs_requested(),
                            commit_writeback_required(request),
                        ),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: filter_allowed(allowed_tokens),
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                    },
                    TransitionDelta::CommitGen {
                        image_id,
                        logical_position: position,
                        physical_position: cursor.physical_kv_len,
                        logical_positions: feedback.map_or(0, |recipe| recipe.logical_positions),
                        physical_kv_tokens: feedback.map(|recipe| recipe.physical_kv_tokens),
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    0,
                    0,
                )
            }
            TransitionIntent::Feedback {
                image_id,
                position,
                locator,
                new_blocks,
                allowed_tokens,
            } => {
                if !request.behavior.generated_image_feedback || request.policy.feedback.is_none() {
                    return Err(PlanningError::FeedbackDisabled);
                }
                if !request.policy.feedback.as_ref().is_some_and(|feedback| {
                    matches!(
                        feedback.writeback,
                        uniserve_core::FeedbackWriteback::DirectKv
                    )
                }) {
                    return Err(PlanningError::FeedbackDisabled);
                }
                (
                    Wire {
                        work: Work::Transfer(TransferMode::KvInstall),
                        domain: Domain::Und,
                        inputs: vec![locator],
                        outputs: token_outputs(
                            request.sampling.generated_logprobs_requested(),
                            false,
                        ),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: filter_allowed(allowed_tokens),
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: None,
                    },
                    TransitionDelta::Feedback {
                        image_id,
                        logical_position: position,
                        physical_position: cursor.physical_kv_len,
                        logical_positions: request
                            .policy
                            .feedback
                            .as_ref()
                            .map_or(0, |recipe| recipe.logical_positions),
                        physical_kv_tokens: request
                            .policy
                            .feedback
                            .as_ref()
                            .map(|recipe| recipe.physical_kv_tokens),
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    0,
                    0,
                )
            }
            TransitionIntent::FeedbackIngest {
                image_id,
                step_index,
                step,
                is_final_step,
                position,
                logical_positions,
                physical_kv_tokens,
                image_b64,
                staged_image,
                new_blocks,
            } => {
                let Some(feedback) = request.policy.feedback.as_ref() else {
                    return Err(PlanningError::FeedbackDisabled);
                };
                if !matches!(
                    feedback.writeback,
                    uniserve_core::FeedbackWriteback::Reingest { .. }
                ) {
                    return Err(PlanningError::FeedbackIngestDisabled);
                }
                if position != cursor.logical_pos {
                    return Err(PlanningError::LogicalCursorMismatch {
                        expected: cursor.logical_pos,
                        actual: position,
                    });
                }
                if staged_image.is_none() && image_b64.is_empty() {
                    return Err(PlanningError::MissingImageInput);
                }
                let work = match step {
                    ImageIngestStep::VaeEncode => Work::Encode(EncodeMode::Latent),
                    ImageIngestStep::VitEncode => Work::Encode(EncodeMode::Vision),
                };
                let handle = encode_handle(staged_image, u64::from(image_id));
                (
                    Wire {
                        work,
                        domain: Domain::Und,
                        inputs: Vec::new(),
                        outputs: encode_outputs(step, handle),
                        new_blocks,
                        draft_token_ids: Vec::new(),
                        token_cost: 1,
                        cfg_branches: 1,
                        allowed_text_tokens: None,
                        expected_prompt_token_ids: None,
                        input_tokens: Vec::new(),
                        input_image_bytes: (!image_b64.is_empty())
                            .then(|| image_b64.clone().into_bytes()),
                    },
                    TransitionDelta::FeedbackIngestStep {
                        image_id,
                        step_index,
                        is_final_step,
                        position,
                        physical_start: cursor.physical_kv_len,
                        logical_positions,
                        physical_kv_tokens,
                        expected_encoder_handle: staged_image,
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
        let is_materialize = operation_variant == WorkVariant::Materialize;
        let draft_count = (!wire.draft_token_ids.is_empty())
            .then(|| wire.draft_token_ids.len().min(u32::MAX as usize) as u32);
        let kv_target_tokens = transition_kv_target(&delta).or_else(|| {
            transition_may_write_worker_defined_kv(&delta)
                .then_some(request.resources.max_kv_tokens)
        });
        let new_blocks_len = wire.new_blocks.len();
        let resources = TransitionResources {
            new_blocks: new_blocks_len,
            kv_target_tokens,
            host_scratch_tokens: if is_flow { host_scratch_tokens } else { 0 },
            latent_units: if is_flow { latent_units } else { 0 },
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
            expected_encoder_handle: match delta {
                TransitionDelta::IngestImageStep {
                    cache_hit: true,
                    expected_encoder_handle,
                    ..
                }
                | TransitionDelta::FeedbackIngestStep {
                    expected_encoder_handle,
                    ..
                } => expected_encoder_handle,
                _ => None,
            },
            expects_image_artifact: is_materialize,
            expected_image_hw: is_materialize
                .then_some((request.image.height, request.image.width)),
            requires_image_locator: is_materialize && commit_writeback_required(request),
            expected_image_kv: match delta {
                TransitionDelta::IngestImageStep {
                    physical_kv_tokens, ..
                }
                | TransitionDelta::FeedbackIngestStep {
                    physical_kv_tokens, ..
                } => Some(bounded_worker_kv(
                    physical_kv_tokens,
                    request.resources.max_kv_tokens,
                    cursor.physical_kv_len,
                )),
                TransitionDelta::CommitGen {
                    physical_kv_tokens, ..
                }
                | TransitionDelta::Feedback {
                    physical_kv_tokens, ..
                } => physical_kv_tokens.map(|effect| {
                    bounded_worker_kv(
                        effect,
                        request.resources.max_kv_tokens,
                        cursor.physical_kv_len,
                    )
                }),
                _ => None,
            },
            allows_sampled_tokens: matches!(
                operation_variant,
                WorkVariant::TokenExtend
                    | WorkVariant::TokenDecode
                    | WorkVariant::TokenVerify
                    | WorkVariant::Materialize
                    | WorkVariant::TransferKvInstall
            ),
            expects_sampled_token: matches!(
                operation_variant,
                WorkVariant::TokenExtend | WorkVariant::TokenDecode | WorkVariant::TokenVerify
            ),
            expected_text_tokens: match &delta {
                TransitionDelta::IngestText { .. } => Some(TextTokenCountRange { min: 1, max: 1 }),
                TransitionDelta::DecodeUnd { .. } => {
                    let max = draft_count.map_or(1, |count| count.saturating_add(1));
                    Some(TextTokenCountRange { min: 1, max })
                }
                _ => None,
            },
            max_accepted_draft_tokens: draft_count,
            allowed_text_tokens: wire.allowed_text_tokens,
            generated_logprobs_requested: request.sampling.generated_logprobs_requested()
                && matches!(
                    operation_variant,
                    WorkVariant::TokenExtend
                        | WorkVariant::TokenDecode
                        | WorkVariant::TokenVerify
                        | WorkVariant::Materialize
                        | WorkVariant::TransferKvInstall
                ),
            expected_prompt_token_ids: wire.expected_prompt_token_ids,
        };
        let bounds = Bounds {
            max_points: 1,
            max_tokens: wire.token_cost.min(u32::MAX as usize) as u32,
            max_kv_pages: new_blocks_len.min(u32::MAX as usize) as u32,
            max_latent_bytes: if is_flow { latent_units } else { 0 },
            max_completion_bytes: 0,
            max_transfer_bytes: 0,
        };
        Ok(PlannedTransition {
            work: wire.work,
            route: ROUTE,
            domain: wire.domain,
            bounds,
            inputs: wire.inputs,
            outputs: wire.outputs,
            rng: None,
            control_seq: 0,
            operation: None,
            operation_variant,
            request_id: request.request_id,
            draft_token_ids: wire.draft_token_ids,
            new_blocks: wire.new_blocks,
            token_cost: wire.token_cost,
            input_tokens: wire.input_tokens,
            input_image_bytes: wire.input_image_bytes,
            host_input: None,
            projected_parent_point: None,
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

/// The trigger tokens whose commit publishes generation conditioning. A
/// non-empty result means the request's token operations declare a KV
/// conditioning output.
fn conditioning_trigger_tokens(request: &GenerationRequest) -> Vec<u32> {
    if !request.behavior.gen_output {
        return Vec::new();
    }
    let trigger = &request.policy.trigger;
    let mut tokens = trigger
        .generated_suffix()
        .and_then(|suffix| suffix.last())
        .copied()
        .into_iter()
        .collect::<Vec<_>>();
    tokens.extend(trigger.round_close_token_ids().iter().copied());
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Whether a commit operation writes its committed image KV back into the
/// understanding lineage, producing a KV product the writeback transfer installs.
fn commit_writeback_required(request: &GenerationRequest) -> bool {
    request.behavior.generated_image_feedback
        && request.policy.feedback.as_ref().is_some_and(|feedback| {
            feedback.commit == uniserve_core::CommitRecipe::CommitGenThenWriteback
                && matches!(
                    feedback.writeback,
                    uniserve_core::FeedbackWriteback::DirectKv
                )
        })
}

/// The declared outputs of a commit (materialize) operation: an image artifact,
/// a committed continuation token, an optional logprob product, and an optional
/// KV product when the commit writes its image KV back.
fn materialize_outputs(logprobs: bool, writeback: bool) -> Vec<ProductRef> {
    let mut outputs = vec![
        output_product(
            0,
            ProductKind::Artifact,
            StorageClass::CompletionArena,
            DType::U8,
        ),
        output_product(
            1,
            ProductKind::Token,
            StorageClass::DeviceTensor,
            DType::U32,
        ),
    ];
    if logprobs {
        outputs.push(output_product(
            2,
            ProductKind::Logprob,
            StorageClass::HostStaging,
            DType::U8,
        ));
    }
    if writeback {
        outputs.push(output_product(
            3,
            ProductKind::Kv,
            StorageClass::PagedKv,
            DType::BF16,
        ));
    }
    outputs
}

fn transition_kv_target(delta: &TransitionDelta) -> Option<usize> {
    let target = match delta {
        TransitionDelta::IngestText {
            end,
            start,
            physical_start,
            ..
        } => physical_start.saturating_add(end.saturating_sub(*start)),
        TransitionDelta::IngestImageStep {
            physical_start,
            physical_kv_tokens,
            ..
        } => physical_start.saturating_add(match physical_kv_tokens {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionDelta::FeedbackIngestStep {
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
        TransitionDelta::CommitGen {
            physical_position,
            physical_kv_tokens: Some(effect),
            ..
        }
        | TransitionDelta::Feedback {
            physical_position,
            physical_kv_tokens: Some(effect),
            ..
        } => physical_position.saturating_add(match effect {
            ImageKvEffect::Exact { tokens } => *tokens,
            ImageKvEffect::Bounded { max_tokens } => *max_tokens,
            ImageKvEffect::WorkerDefined => return None,
        }),
        TransitionDelta::DenoiseGen { .. }
        | TransitionDelta::CommitGen { .. }
        | TransitionDelta::Feedback { .. } => return None,
    };
    Some(target as usize)
}

fn transition_may_write_worker_defined_kv(delta: &TransitionDelta) -> bool {
    matches!(
        delta,
        TransitionDelta::IngestImageStep { .. }
            | TransitionDelta::FeedbackIngestStep { .. }
            | TransitionDelta::CommitGen {
                physical_kv_tokens: Some(ImageKvEffect::WorkerDefined),
                ..
            }
            | TransitionDelta::Feedback {
                physical_kv_tokens: Some(ImageKvEffect::WorkerDefined),
                ..
            }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanningError {
    PromptCursorMismatch { expected: u32, actual: u32 },
    LogicalCursorMismatch { expected: u32, actual: u32 },
    GenerationBranchDisabled,
    FeedbackDisabled,
    FeedbackIngestDisabled,
    MissingImageInput,
}

/// Scheduler-local transition associated with one submitted worker op.
/// Scheduler-local transition associated with one submitted worker operation.
/// The wire-operation ingredients (`work`, `route`, `domain`, `bounds`,
/// `inputs`, `outputs`, `rng`, `control_seq`) are lowered by the planner; the
/// flat [`Operation`] is stamped with request and lineage identity at submit
/// time by [`PlannedTransition::assign_operation`]. The host-side
/// `draft_token_ids`, `new_blocks`, and `token_cost` carry accounting the wire
/// operation no longer transports.
#[derive(Debug, Clone)]
pub(crate) struct PlannedTransition {
    pub(crate) work: Work,
    pub(crate) route: RouteId,
    pub(crate) domain: Domain,
    pub(crate) bounds: Bounds,
    pub(crate) inputs: Vec<ProductRef>,
    pub(crate) outputs: Vec<ProductRef>,
    pub(crate) rng: Option<Rng>,
    pub(crate) control_seq: u64,
    pub(crate) operation: Option<Operation>,
    pub(crate) operation_variant: WorkVariant,
    pub(crate) request_id: RequestId,
    pub(crate) draft_token_ids: Vec<u32>,
    pub(crate) new_blocks: Vec<BlockId>,
    pub(crate) token_cost: usize,
    /// Host-known input token values the operation's forward consumes.
    pub(crate) input_tokens: Vec<u32>,
    /// Host-supplied input image bytes an encode operation's forward consumes.
    pub(crate) input_image_bytes: Option<Vec<u8>>,
    /// The reference of the host-supplied input product, stamped with the
    /// operation identity at [`PlannedTransition::assign_operation`]. Its value
    /// is transported in the batch's `input_products` under the same identity.
    pub(crate) host_input: Option<ProductRef>,
    /// For a `Point::Device`-rooted projected successor, the host-known point
    /// index of its (not-yet-observed) parent — the predecessor's committed and
    /// advanced version. `None` for a fixed-parent op, whose parent point is
    /// carried directly on the parent `Point::Fixed`. Used by
    /// [`PlannedTransition::validate_result`] to reconstruct the expected point.
    pub(crate) projected_parent_point: Option<u32>,
    pub(crate) delta: TransitionDelta,
    pub(crate) resources: TransitionResources,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
}

impl PlannedTransition {
    /// Stamp the flat operation with its request key, operation id, and parent
    /// version. The declared outputs inherit the owning identity.
    pub(crate) fn assign_operation(
        &mut self,
        request_key: RequestKey,
        op_id: OpId,
        parent: VersionRef,
    ) {
        let outputs = self
            .outputs
            .iter()
            .cloned()
            .map(|mut product| {
                product.request_key = request_key;
                product.producer_op_id = op_id;
                product
            })
            .collect();
        let mut inputs = self.inputs.clone();
        // Stamp the host-supplied input product with the operation identity and
        // list it among the operation's inputs; its value travels in the batch's
        // `input_products` under this same identity.
        if !self.input_tokens.is_empty() {
            let product = host_input_product(request_key, op_id, ProductKind::Token, DType::U32);
            self.host_input = Some(product.clone());
            inputs.push(product);
        } else if self.input_image_bytes.is_some() {
            let product = host_input_product(request_key, op_id, ProductKind::Artifact, DType::U8);
            self.host_input = Some(product.clone());
            inputs.push(product);
        }
        self.operation = Some(Operation::registered(
            request_key,
            op_id,
            parent,
            self.work,
            self.route,
            self.domain,
            self.bounds,
            inputs,
            outputs,
            // The KV blocks this step appends travel on the wire so the worker's
            // forward addresses the newly allocated pages at a block boundary.
            self.new_blocks.clone(),
            None,
            self.rng,
            self.control_seq,
        ));
    }

    /// The host-supplied input product value for this operation's forward, keyed
    /// by the input reference stamped in [`Self::assign_operation`]. `None` when
    /// the operation consumes no host-supplied input.
    pub(crate) fn input_product(&self) -> Option<ProductPayload> {
        let product = self.host_input.clone()?;
        let bytes = if !self.input_tokens.is_empty() {
            encode_token_product_bytes(&self.input_tokens)
        } else {
            self.input_image_bytes.clone().unwrap_or_default()
        };
        Some(ProductPayload { product, bytes })
    }

    pub(crate) fn validate_result(
        &self,
        record: &CompletionRecord,
        products: &[ProductPayload],
    ) -> Result<(), TransitionValidationError> {
        let Some(operation) = self.operation.as_ref() else {
            return Err(TransitionValidationError::OpIdMismatch {
                expected: 0,
                actual: record.op_id.0,
            });
        };
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
        let parent_point = match operation.parent.point {
            Point::Fixed {
                point_index: parent_point,
                ..
            } => parent_point,
            // A device parent carries no host-known point index. The projected
            // parent point threaded onto the transition at submit (the
            // predecessor's committed-and-advanced version) reconstructs it.
            Point::Device { .. } => {
                let Some(projected_parent_point) = self.projected_parent_point else {
                    return Err(TransitionValidationError::VersionMismatch {
                        expected_base: 0,
                        actual_base: 0,
                        actual_result: u64::from(record.selected_point),
                    });
                };
                projected_parent_point
            }
        };
        let expected_point = parent_point.saturating_add(u32::from(operation.advances_state));
        if record.selected_point != expected_point {
            return Err(TransitionValidationError::VersionMismatch {
                expected_base: u64::from(parent_point),
                actual_base: u64::from(parent_point),
                actual_result: u64::from(record.selected_point),
            });
        }
        self.validation
            .validate(self.operation_variant, record, products)
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
    IngestImageStep {
        segment_index: usize,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        encoder_cache_key: Option<u64>,
        cache_hit: bool,
        expected_encoder_handle: Option<u64>,
    },
    DecodeUnd {
        logical_position: u32,
        physical_position: u32,
    },
    DenoiseGen {
        image_id: u32,
        start_step: u16,
        step_count: u16,
    },
    CommitGen {
        image_id: u32,
        logical_position: u32,
        physical_position: u32,
        logical_positions: u32,
        physical_kv_tokens: Option<ImageKvEffect>,
    },
    Feedback {
        image_id: u32,
        logical_position: u32,
        physical_position: u32,
        logical_positions: u32,
        physical_kv_tokens: Option<ImageKvEffect>,
    },
    FeedbackIngestStep {
        image_id: u32,
        step_index: usize,
        is_final_step: bool,
        position: u32,
        physical_start: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        expected_encoder_handle: Option<u64>,
    },
}

impl TransitionDelta {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::IngestText { .. } => "ingest_text",
            Self::IngestImageStep { .. } => "ingest_image_step",
            Self::DecodeUnd { .. } => "decode_und",
            Self::DenoiseGen { .. } => "denoise_gen",
            Self::CommitGen { .. } => "commit_gen",
            Self::Feedback { .. } => "feedback",
            Self::FeedbackIngestStep { .. } => "feedback_ingest_step",
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionValidation {
    pub(crate) expected_denoise_step: Option<u16>,
    pub(crate) expects_encoder_handle: bool,
    pub(crate) expected_encoder_handle: Option<u64>,
    pub(crate) expects_image_artifact: bool,
    pub(crate) expected_image_hw: Option<(u32, u32)>,
    pub(crate) requires_image_locator: bool,
    pub(crate) expected_image_kv: Option<ImageKvEffect>,
    pub(crate) allows_sampled_tokens: bool,
    pub(crate) expects_sampled_token: bool,
    pub(crate) expected_text_tokens: Option<TextTokenCountRange>,
    pub(crate) max_accepted_draft_tokens: Option<u32>,
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
        if let Some(expected) = self.expected_encoder_handle
            && encoder_handle != Some(expected)
        {
            return Err(TransitionValidationError::EncoderHandleMismatch {
                expected,
                actual: encoder_handle,
            });
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
        if self.requires_image_locator
            && find_product(products, record.op_id, ProductKind::Kv).is_none()
        {
            return Err(TransitionValidationError::MissingImageLocator);
        }
        if let Some(expected) = self.expected_image_kv
            && !self.requires_image_locator
        {
            let actual = record.logical_lengths.kv_visible_len;
            match expected {
                ImageKvEffect::Exact { tokens } if actual != tokens => {
                    return Err(TransitionValidationError::ImageKvMismatch {
                        expected_max: tokens,
                        actual,
                    });
                }
                ImageKvEffect::Bounded { max_tokens } if actual > max_tokens => {
                    return Err(TransitionValidationError::ImageKvMismatch {
                        expected_max: max_tokens,
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
        // A verify operation's accepted draft prefix is the committed tokens less
        // the one bonus token; other variants report no acceptance count.
        let accepted_draft_tokens = (operation_variant == WorkVariant::TokenVerify).then(|| {
            record
                .committed_tokens
                .len()
                .saturating_sub(1)
                .min(u32::MAX as usize) as u32
        });
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
            if let Some(accepted) = accepted_draft_tokens.map(|count| count.saturating_add(1))
                && listed != accepted
            {
                return Err(TransitionValidationError::TextTokenCountInconsistent {
                    listed,
                    accepted,
                });
            }
            let actual = listed;
            if actual < expected.min || actual > expected.max {
                return Err(TransitionValidationError::TextTokenCountMismatch {
                    min: expected.min,
                    max: expected.max,
                    actual,
                });
            }
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
    EncoderHandleMismatch {
        expected: u64,
        actual: Option<u64>,
    },
    MissingImageArtifact,
    InvalidImageArtifact,
    MissingImageLocator,
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
    TextTokenCountInconsistent {
        listed: u32,
        accepted: u32,
    },
    TextTokenCountMismatch {
        min: u32,
        max: u32,
        actual: u32,
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
            lora_id: None,
            grammar: None,
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
                    allowed_tokens: None,
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
            completion_slot_generation: 0,
            status: OpStatus::Ok,
            selected_point,
            logical_lengths: LogicalLengths {
                token_len: 1,
                kv_visible_len: 2,
                latent_len: 0,
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
    }

    #[test]
    fn decode_conditioning_output_matches_the_resolved_branch_capability() {
        let policy = GenerationPolicyDescriptor {
            trigger: uniserve_core::TriggerPolicyDescriptor::Token { token_id: 42 },
            ..GenerationPolicyDescriptor::default()
        };
        let mut requests = [request(10, vec![11, 12]), request(11, vec![11, 12])];
        requests[0].constraint = GenerationConstraint::UndOnly;
        requests[1].constraint = GenerationConstraint::Default;
        for request in &mut requests {
            request.policy = policy.clone();
            request.behavior =
                GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
        }

        for request in &requests {
            let transition = GenerationPlanner::new()
                .plan(
                    request,
                    CursorProjection {
                        phase: GenerationPhase::DecodeUnd,
                        prompt_cursor: 2,
                        logical_pos: 2,
                        physical_kv_len: 2,
                        replayability: Replayability::Replayable,
                    },
                    TransitionIntent::DecodeUnd {
                        position: 2,
                        new_blocks: Vec::new(),
                        spec_token_ids: None,
                        allowed_tokens: None,
                        input_token: 7,
                        relay_input: false,
                    },
                )
                .expect("plan token decode");
            // The decode operation declares a KV conditioning output exactly when
            // the resolved behavior opens the generation branch.
            let publishes_conditioning = transition
                .outputs
                .iter()
                .any(|product| product.kind == ProductKind::Kv);
            assert_eq!(publishes_conditioning, request.behavior.gen_output);
        }
    }

    #[test]
    fn transition_accepts_only_the_registered_completion() {
        let mut transition = prefill_transition();
        let request_key = RequestKey::new(1, RequestId(9), 3);
        let parent = VersionRef {
            request_key,
            producer_op_id: OpId(1),
            point: Point::Fixed {
                point_index: 5,
                semantic_digest: digest(),
            },
        };
        transition.assign_operation(request_key, OpId(17), parent);

        let record = completion(request_key, 17, 6);
        assert_eq!(transition.validate_result(&record, &[]), Ok(()));

        let stale = completion(request_key, 17, 7);
        assert!(matches!(
            transition.validate_result(&stale, &[]),
            Err(TransitionValidationError::VersionMismatch { .. })
        ));
    }
}
