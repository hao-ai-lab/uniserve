use std::collections::HashSet;

use uniserve_core::{
    BlockId, CfgParams, ContextSegment, GenerationRequest, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, SegmentPlacement, UndTokenAction,
};
use uniserve_worker_wire::{
    EncodeInput, EncodeKind, EncodeOperation, FlowOperation, Guidance, KvLeaseDelta,
    MaterializeInput, MaterializeKind, MaterializeOperation, MaterializedProduct, Operation,
    OperationEnvelope, OperationKind, OperationResult, OperationType, PublishedKv,
    PublishedProduct, ResourceClass, ResultDelta, SequenceEffect, SequenceInput, SequenceMode,
    SequenceOperation, TokenInput, TokenPolicy, TokenSource, TransferKind, TransferOperation,
};

use crate::image_artifact::validate_png_artifact;

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
        result: &OperationResult,
    ) -> Result<(), CursorApplyError> {
        let op_id = transition.op.op_id;
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
                let ResultDelta::Encode(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                self.und.physical_kv_len = self.und.physical_kv_len.saturating_add(delta.kv_tokens);
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
                let ResultDelta::Sequence(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                let effect = &delta.effect;
                let actual_count = effect
                    .sampled_token_ids
                    .len()
                    .max(
                        effect
                            .accepted_draft_tokens
                            .map_or(0, |accepted| accepted as usize + 1),
                    )
                    .max(1) as u32;
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
                let ResultDelta::Flow(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                self.image_gen.steps_done = self.image_gen.steps_done.max(
                    delta
                        .steps_completed
                        .max(start_step.saturating_add(step_count)),
                );
            }
            TransitionDelta::CommitGen {
                logical_position,
                physical_position,
                logical_positions,
                physical_kv_tokens,
                ..
            } => {
                let ResultDelta::Materialize(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                let has_locator = match &delta.product {
                    MaterializedProduct::Image(image) => !image.locator.is_empty(),
                    MaterializedProduct::Published(product) => !product.locator.is_empty(),
                    MaterializedProduct::Frame { .. } => false,
                };
                if !has_locator && physical_kv_tokens.is_some() {
                    let added = delta.kv_tokens.unwrap_or(0);
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
                let ResultDelta::Transfer(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                let added = delta.kv_tokens.unwrap_or(0);
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
                let ResultDelta::Encode(delta) = &result.delta else {
                    return Err(CursorApplyError::ResultTypeMismatch);
                };
                self.und.physical_kv_len = self.und.physical_kv_len.saturating_add(delta.kv_tokens);
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
    ResultTypeMismatch,
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
    pub(crate) conditioning: Option<PublishedKv>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackCursor {
    pub(crate) locator: Option<String>,
    pub(crate) sampled_token: Option<u32>,
    pub(crate) sampled_logprob: Option<f32>,
    pub(crate) top_logprobs: Option<Vec<uniserve_worker_wire::TokenLogprob>>,
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
        recent_tokens: Option<Vec<u32>>,
        allowed_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
    },
    IngestImage {
        segment_index: usize,
        step_index: usize,
        step: ImageIngestStep,
        is_final_step: bool,
        position: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        worker_hash: u64,
        encoder_cache_key: Option<u64>,
        cache_hit: bool,
        image_b64: String,
        staged_image: Option<u64>,
        new_blocks: Vec<BlockId>,
    },
    DecodeUnd {
        position: u32,
        token_id: u32,
        token_source: TokenSource,
        new_blocks: Vec<BlockId>,
        spec_token_ids: Option<Vec<u32>>,
        recent_tokens: Option<Vec<u32>>,
        allowed_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
    },
    DenoiseGen {
        image_id: u32,
        position: u32,
        start_step: u16,
        step_count: u16,
        cfg: CfgParams,
        image_prompt: Option<String>,
        latent_units: u64,
        host_scratch_tokens: u64,
        conditioning: Option<PublishedKv>,
    },
    CommitGen {
        image_id: u32,
        position: u32,
        new_blocks: Vec<BlockId>,
        recent_tokens: Option<Vec<u32>>,
        allowed_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
    },
    Feedback {
        image_id: u32,
        position: u32,
        locator: String,
        new_blocks: Vec<BlockId>,
        recent_tokens: Option<Vec<u32>>,
        allowed_tokens: Option<Vec<u32>>,
        suppress_tokens: Option<Vec<u32>>,
    },
    FeedbackIngest {
        image_id: u32,
        step_index: usize,
        step: ImageIngestStep,
        worker_hash: u64,
        is_final_step: bool,
        position: u32,
        logical_positions: u32,
        physical_kv_tokens: ImageKvEffect,
        image_b64: String,
        staged_image: Option<u64>,
        new_blocks: Vec<BlockId>,
    },
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
        let (op, delta, encoder_pins, replayability_after_apply, latent_units, host_scratch_tokens) =
            match intent {
                TransitionIntent::IngestText {
                    segment_index,
                    prompt_start,
                    token_ids,
                    new_blocks,
                    recent_tokens,
                    allowed_tokens,
                    suppress_tokens,
                } => {
                    if prompt_start != cursor.prompt_cursor {
                        return Err(PlanningError::PromptCursorMismatch {
                            expected: cursor.prompt_cursor,
                            actual: prompt_start,
                        });
                    }
                    let token_count = token_ids.len() as u32;
                    let end = prompt_start.saturating_add(token_count);
                    (
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Sequence(SequenceOperation {
                                mode: SequenceMode::Extend,
                                lease: kv_lease(new_blocks),
                                position: (
                                    cursor.logical_pos,
                                    cursor.logical_pos.saturating_add(token_count),
                                ),
                                policy: token_policy(
                                    recent_tokens,
                                    allowed_tokens,
                                    suppress_tokens,
                                    request.behavior.gen_output,
                                    Vec::new(),
                                ),
                                input: SequenceInput::Tokens(TokenInput {
                                    token_ids,
                                    source: TokenSource::Wire,
                                    draft_token_ids: Vec::new(),
                                    return_all_logits: request.sampling.prompt_logprobs_requested(),
                                }),
                            }),
                        ),
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
                    worker_hash,
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
                    let kind = match step {
                        ImageIngestStep::VaeEncode => EncodeKind::Latent,
                        ImageIngestStep::VitEncode => EncodeKind::Vision,
                    };
                    let input = encode_input(cache_hit, staged_image, image_b64, worker_hash)?;
                    (
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Encode(EncodeOperation {
                                kind,
                                lease: kv_lease(new_blocks),
                                position: (
                                    position,
                                    position.saturating_add(logical_positions.max(1)),
                                ),
                                conditioning_position: position,
                                input,
                            }),
                        ),
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
                    token_id,
                    token_source,
                    new_blocks,
                    spec_token_ids,
                    recent_tokens,
                    allowed_tokens,
                    suppress_tokens,
                } => {
                    if position != cursor.logical_pos {
                        return Err(PlanningError::LogicalCursorMismatch {
                            expected: cursor.logical_pos,
                            actual: position,
                        });
                    }
                    let draft_token_ids = spec_token_ids.unwrap_or_default();
                    let mode = if draft_token_ids.is_empty() {
                        SequenceMode::Decode
                    } else {
                        SequenceMode::Verify
                    };
                    (
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Sequence(SequenceOperation {
                                mode,
                                lease: kv_lease(new_blocks),
                                position: (position, position.saturating_add(1)),
                                policy: token_policy(
                                    recent_tokens,
                                    allowed_tokens,
                                    suppress_tokens,
                                    false,
                                    conditioning_trigger_tokens(request),
                                ),
                                input: SequenceInput::Tokens(TokenInput {
                                    token_ids: vec![token_id],
                                    source: token_source,
                                    draft_token_ids,
                                    return_all_logits: false,
                                }),
                            }),
                        ),
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
                    position,
                    start_step,
                    step_count,
                    cfg,
                    image_prompt,
                    latent_units,
                    host_scratch_tokens,
                    conditioning,
                } => {
                    if !request.behavior.gen_output {
                        return Err(PlanningError::GenerationBranchDisabled);
                    }
                    let step_count = step_count.max(1);
                    (
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Flow(FlowOperation {
                                latent_handle: request.request_id.0,
                                position,
                                start_step,
                                step_count,
                                conditioning_position: position,
                                conditioning,
                                guidance: Guidance {
                                    branch_count: cfg.branch_count,
                                    text_scale: cfg.text_scale,
                                    image_scale: cfg.img_scale,
                                    renorm_type: cfg.renorm_type,
                                    renorm_min: cfg.renorm_min,
                                    interval: cfg.interval,
                                },
                                image_prompt: image_prompt.unwrap_or_default(),
                            }),
                        ),
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
                    recent_tokens,
                    allowed_tokens,
                    suppress_tokens,
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
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Materialize(MaterializeOperation {
                                kind: MaterializeKind::Image,
                                lease: kv_lease(new_blocks),
                                position,
                                conditioning_position: position,
                                policy: token_policy(
                                    recent_tokens,
                                    allowed_tokens,
                                    suppress_tokens,
                                    false,
                                    Vec::new(),
                                ),
                                input: MaterializeInput::Latent {
                                    handle: request.request_id.0,
                                },
                            }),
                        ),
                        TransitionDelta::CommitGen {
                            image_id,
                            logical_position: position,
                            physical_position: cursor.physical_kv_len,
                            logical_positions: feedback
                                .map_or(0, |recipe| recipe.logical_positions),
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
                    recent_tokens,
                    allowed_tokens,
                    suppress_tokens,
                } => {
                    if !request.behavior.generated_image_feedback
                        || request.policy.feedback.is_none()
                    {
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
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Transfer(TransferOperation {
                                kind: TransferKind::Kv,
                                lease: kv_lease(new_blocks),
                                position,
                                conditioning_position: position,
                                policy: token_policy(
                                    recent_tokens,
                                    allowed_tokens,
                                    suppress_tokens,
                                    false,
                                    Vec::new(),
                                ),
                                source: PublishedProduct {
                                    handle: request.request_id.0,
                                    locator,
                                },
                            }),
                        ),
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
                    worker_hash,
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
                    let kind = match step {
                        ImageIngestStep::VaeEncode => EncodeKind::Latent,
                        ImageIngestStep::VitEncode => EncodeKind::Vision,
                    };
                    let input = encode_input(false, staged_image, image_b64, worker_hash)?;
                    (
                        OperationEnvelope::unsealed(
                            request.request_id,
                            Operation::Encode(EncodeOperation {
                                kind,
                                lease: kv_lease(new_blocks),
                                position: (
                                    position,
                                    position.saturating_add(logical_positions.max(1)),
                                ),
                                conditioning_position: position,
                                input,
                            }),
                        ),
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
        let operation_type = op.operation_type();
        let draft_count = sequence_token_input(&op)
            .map(|input| input.draft_token_ids.len().min(u32::MAX as usize) as u32)
            .filter(|count| *count > 0);
        let allowed_text_tokens = operation_policy(&op)
            .map(|policy| policy.allowed_tokens.clone())
            .filter(|tokens| !tokens.is_empty());
        let expected_prompt_token_ids = sequence_token_input(&op)
            .filter(|input| input.return_all_logits)
            .map(|input| {
                let mut tokens = input.token_ids.clone();
                if matches!(&delta, TransitionDelta::IngestText { start: 0, .. })
                    && !tokens.is_empty()
                {
                    tokens.remove(0);
                }
                tokens
            });
        let kv_target_tokens = transition_kv_target(&delta).or_else(|| {
            transition_may_write_worker_defined_kv(&delta)
                .then_some(request.resources.max_kv_tokens)
        });
        let resources = TransitionResources {
            new_blocks: operation_new_block_count(&op),
            kv_target_tokens,
            host_scratch_tokens: if operation_type == OperationType::Flow {
                host_scratch_tokens
            } else {
                0
            },
            latent_units: if operation_type == OperationType::Flow {
                latent_units
            } else {
                0
            },
            encoder_pins,
            replayability_after_apply,
            release_on_apply: if operation_type == OperationType::MaterializeImage {
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
                operation_type,
                OperationType::EncodeLatent | OperationType::EncodeVision
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
            expects_image_artifact: operation_type == OperationType::MaterializeImage,
            expected_image_hw: (operation_type == OperationType::MaterializeImage)
                .then_some((request.image.height, request.image.width)),
            requires_image_locator: operation_type == OperationType::MaterializeImage
                && request.behavior.generated_image_feedback
                && request.policy.feedback.as_ref().is_some_and(|feedback| {
                    feedback.commit == uniserve_core::CommitRecipe::CommitGenThenWriteback
                        && matches!(
                            feedback.writeback,
                            uniserve_core::FeedbackWriteback::DirectKv
                        )
                }),
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
                operation_type,
                OperationType::SequenceExtend
                    | OperationType::SequenceDecode
                    | OperationType::SequenceVerify
                    | OperationType::MaterializeImage
                    | OperationType::TransferKv
            ),
            expects_sampled_token: matches!(
                operation_type,
                OperationType::SequenceExtend
                    | OperationType::SequenceDecode
                    | OperationType::SequenceVerify
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
            allowed_text_tokens,
            generated_logprobs_requested: request.sampling.generated_logprobs_requested()
                && matches!(
                    operation_type,
                    OperationType::SequenceExtend
                        | OperationType::SequenceDecode
                        | OperationType::SequenceVerify
                        | OperationType::MaterializeImage
                        | OperationType::TransferKv
                ),
            expected_prompt_token_ids,
        };
        Ok(PlannedTransition {
            op,
            operation_type,
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

fn kv_lease(new_blocks: Vec<BlockId>) -> KvLeaseDelta {
    KvLeaseDelta {
        group_id: 0,
        new_blocks,
    }
}

fn token_policy(
    recent_tokens: Option<Vec<u32>>,
    allowed_tokens: Option<Vec<u32>>,
    suppress_tokens: Option<Vec<u32>>,
    publish_kv: bool,
    publish_kv_on_tokens: Vec<u32>,
) -> TokenPolicy {
    TokenPolicy {
        allowed_tokens: allowed_tokens.unwrap_or_default(),
        suppress_tokens: suppress_tokens.unwrap_or_default(),
        recent_tokens: recent_tokens.unwrap_or_default(),
        publish_kv,
        publish_kv_on_tokens,
    }
}

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

fn encode_input(
    cached: bool,
    staged_handle: Option<u64>,
    base64: String,
    content_hash: u64,
) -> Result<EncodeInput, PlanningError> {
    if cached {
        return Ok(EncodeInput::CachedProduct { content_hash });
    }
    if let Some(handle) = staged_handle {
        return Ok(EncodeInput::StagedProduct {
            handle,
            content_hash,
        });
    }
    if base64.is_empty() {
        return Err(PlanningError::MissingImageInput);
    }
    Ok(EncodeInput::InlineImage {
        base64,
        content_hash,
    })
}

fn operation_lease(envelope: &OperationEnvelope) -> Option<&KvLeaseDelta> {
    match &envelope.operation {
        Operation::Sequence(operation) => Some(&operation.lease),
        Operation::Encode(operation) => Some(&operation.lease),
        Operation::Materialize(operation) => Some(&operation.lease),
        Operation::Transfer(operation) => Some(&operation.lease),
        Operation::Flow(_) => None,
    }
}

fn operation_new_block_count(envelope: &OperationEnvelope) -> usize {
    operation_lease(envelope).map_or(0, |lease| lease.new_blocks.len())
}

fn operation_policy(envelope: &OperationEnvelope) -> Option<&TokenPolicy> {
    match &envelope.operation {
        Operation::Sequence(operation) => Some(&operation.policy),
        Operation::Materialize(operation) => Some(&operation.policy),
        Operation::Transfer(operation) => Some(&operation.policy),
        Operation::Flow(_) | Operation::Encode(_) => None,
    }
}

fn sequence_token_input(envelope: &OperationEnvelope) -> Option<&TokenInput> {
    let Operation::Sequence(operation) = &envelope.operation else {
        return None;
    };
    let SequenceInput::Tokens(input) = &operation.input else {
        return None;
    };
    Some(input)
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
#[derive(Debug, Clone)]
pub(crate) struct PlannedTransition {
    pub(crate) op: OperationEnvelope,
    pub(crate) operation_type: OperationType,
    pub(crate) delta: TransitionDelta,
    pub(crate) resources: TransitionResources,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
}

impl PlannedTransition {
    pub(crate) fn assign_envelope(&mut self, epoch: u64, op_id: u64, base_version: u64) {
        self.op.seal(epoch, op_id, base_version);
    }

    pub(crate) fn validate_result(
        &self,
        result: &OperationResult,
    ) -> Result<(), TransitionValidationError> {
        if result.session_id != self.op.session_id {
            return Err(TransitionValidationError::SessionMismatch {
                expected: self.op.session_id.0,
                actual: result.session_id.0,
            });
        }
        if self.op.op_id == 0 || result.op_id != self.op.op_id {
            return Err(TransitionValidationError::OpIdMismatch {
                expected: self.op.op_id,
                actual: result.op_id,
            });
        }
        if result.epoch != self.op.epoch {
            return Err(TransitionValidationError::EpochMismatch {
                expected: self.op.epoch,
                actual: result.epoch,
            });
        }
        if result.base_version != self.op.base_version
            || result.result_version != self.op.base_version.saturating_add(1)
        {
            return Err(TransitionValidationError::VersionMismatch {
                expected_base: self.op.base_version,
                actual_base: result.base_version,
                actual_result: result.result_version,
            });
        }
        self.validation.validate(self.operation_type, result)
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
        operation_type: OperationType,
        result: &OperationResult,
    ) -> Result<(), TransitionValidationError> {
        if result.delta.kind() != operation_type.kind() {
            return Err(TransitionValidationError::ResultTypeMismatch {
                expected: operation_type.kind(),
                actual: result.delta.kind(),
            });
        }
        if let Some(expected_step) = self.expected_denoise_step {
            let ResultDelta::Flow(flow) = &result.delta else {
                return Err(TransitionValidationError::MissingDenoiseStep);
            };
            let steps_done = flow.steps_completed;
            if steps_done != expected_step {
                return Err(TransitionValidationError::DenoiseStepMismatch {
                    expected: expected_step,
                    actual: steps_done,
                });
            }
        }
        let encoder_handle = match &result.delta {
            ResultDelta::Encode(encode) => Some(encode.product_handle),
            _ => None,
        };
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
        let image = materialized_image(result);
        if let Some(expected) = self.expected_image_hw {
            let actual = image
                .map(|image| (image.height, image.width))
                .ok_or(TransitionValidationError::MissingImageDimensions)?;
            if actual != expected {
                return Err(TransitionValidationError::ImageDimensionsMismatch {
                    expected,
                    actual,
                });
            }
        }
        if self.expects_image_artifact {
            let image = image
                .filter(|image| !image.png_base64.is_empty())
                .ok_or(TransitionValidationError::MissingImageArtifact)?;
            let metadata = validate_png_artifact(&image.png_base64, self.expected_image_hw)
                .ok_or(TransitionValidationError::InvalidImageArtifact)?;
            let reported = Some((image.height, image.width));
            if reported != Some((metadata.height, metadata.width)) {
                return Err(TransitionValidationError::ImageArtifactDimensionsMismatch {
                    artifact: (metadata.height, metadata.width),
                    reported,
                });
            }
        }
        if self.requires_image_locator && result_locator(result).is_none_or(str::is_empty) {
            return Err(TransitionValidationError::MissingImageLocator);
        }
        if let Some(expected) = self.expected_image_kv
            && !self.requires_image_locator
        {
            let actual =
                result_kv_tokens(result).ok_or(TransitionValidationError::MissingImageKvTokens)?;
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
        let effect = result_sequence_effect(result);
        let sampled_tokens = effect
            .map(|effect| effect.sampled_token_ids.as_slice())
            .unwrap_or_default();
        if !self.allows_sampled_tokens && !sampled_tokens.is_empty() {
            return Err(TransitionValidationError::UnexpectedSampledToken { operation_type });
        }
        if self.expects_sampled_token && sampled_tokens.is_empty() {
            return Err(TransitionValidationError::MissingSampledToken { operation_type });
        }
        let accepted_draft_tokens = effect.and_then(|effect| effect.accepted_draft_tokens);
        if let Some(max_accepted) = self.max_accepted_draft_tokens
            && accepted_draft_tokens.is_some_and(|accepted| accepted > max_accepted)
        {
            return Err(TransitionValidationError::AcceptedDraftCountExceeded {
                max: max_accepted,
                actual: accepted_draft_tokens.unwrap_or_default(),
            });
        }
        if let Some(allowed) = &self.allowed_text_tokens
            && let Some(token_id) = sampled_tokens
                .iter()
                .copied()
                .find(|token_id| !allowed.contains(token_id))
        {
            return Err(TransitionValidationError::SampledTokenNotAllowed { token_id });
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
        let sampled_token = sampled_tokens.last().copied();
        let generated_candidates = effect
            .map(|effect| effect.top_logprobs.as_slice())
            .unwrap_or_default();
        match (
            self.generated_logprobs_requested,
            sampled_token,
            generated_candidates.is_empty(),
        ) {
            (false, _, false) | (true, None, false) => {
                return Err(TransitionValidationError::UnexpectedGeneratedLogprobs {
                    operation_type,
                });
            }
            (true, Some(token_id), true) => {
                return Err(TransitionValidationError::MissingGeneratedLogprobs { token_id });
            }
            (true, Some(token_id), false) => {
                if generated_candidates[0].0 != token_id {
                    return Err(TransitionValidationError::GeneratedLogprobTokenMismatch {
                        expected: token_id,
                        actual: generated_candidates[0].0,
                    });
                }
                let mut token_ids = std::collections::HashSet::new();
                if generated_candidates.iter().any(|candidate| {
                    candidate.2 == 0 || !candidate.1.is_finite() || !token_ids.insert(candidate.0)
                }) {
                    return Err(TransitionValidationError::InvalidGeneratedLogprobCandidates);
                }
            }
            (false, _, true) | (true, None, true) => {}
        }
        match (
            self.expected_prompt_token_ids.as_deref(),
            effect.map(|effect| effect.prompt_logprobs.as_slice()),
        ) {
            (None, Some(positions)) if !positions.is_empty() => {
                return Err(TransitionValidationError::UnexpectedPromptLogprobs { operation_type });
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
                    if first.0 != expected_token {
                        return Err(TransitionValidationError::PromptLogprobTokenMismatch {
                            position,
                            expected: expected_token,
                            actual: first.0,
                        });
                    }
                    let mut seen = std::collections::HashSet::new();
                    if candidates
                        .iter()
                        .any(|candidate| candidate.2 == 0 || !seen.insert(candidate.0))
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

fn result_sequence_effect(result: &OperationResult) -> Option<&SequenceEffect> {
    match &result.delta {
        ResultDelta::Sequence(delta) => Some(&delta.effect),
        ResultDelta::Materialize(delta) => delta.sequence.as_ref(),
        ResultDelta::Transfer(delta) => delta.sequence.as_ref(),
        ResultDelta::Flow(_) | ResultDelta::Encode(_) => None,
    }
}

fn materialized_image(result: &OperationResult) -> Option<&uniserve_worker_wire::ImageArtifact> {
    let ResultDelta::Materialize(delta) = &result.delta else {
        return None;
    };
    let MaterializedProduct::Image(image) = &delta.product else {
        return None;
    };
    Some(image)
}

fn result_locator(result: &OperationResult) -> Option<&str> {
    match &result.delta {
        ResultDelta::Materialize(delta) => match &delta.product {
            MaterializedProduct::Image(image) => Some(image.locator.as_str()),
            MaterializedProduct::Published(product) => Some(product.locator.as_str()),
            MaterializedProduct::Frame { .. } => None,
        },
        ResultDelta::Transfer(delta) => delta
            .product
            .as_ref()
            .map(|product| product.locator.as_str()),
        ResultDelta::Sequence(_) | ResultDelta::Flow(_) | ResultDelta::Encode(_) => None,
    }
}

fn result_kv_tokens(result: &OperationResult) -> Option<u32> {
    match &result.delta {
        ResultDelta::Sequence(delta) => delta.effect.kv_tokens,
        ResultDelta::Encode(delta) => Some(delta.kv_tokens),
        ResultDelta::Materialize(delta) => delta.kv_tokens,
        ResultDelta::Transfer(delta) => delta.kv_tokens,
        ResultDelta::Flow(_) => None,
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
    EpochMismatch {
        expected: u64,
        actual: u64,
    },
    VersionMismatch {
        expected_base: u64,
        actual_base: u64,
        actual_result: u64,
    },
    ResultTypeMismatch {
        expected: OperationKind,
        actual: OperationKind,
    },
    MissingDenoiseStep,
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
    ImageArtifactDimensionsMismatch {
        artifact: (u32, u32),
        reported: Option<(u32, u32)>,
    },
    MissingImageLocator,
    MissingImageDimensions,
    ImageDimensionsMismatch {
        expected: (u32, u32),
        actual: (u32, u32),
    },
    MissingImageKvTokens,
    ImageKvMismatch {
        expected_max: u32,
        actual: u32,
    },
    UnexpectedSampledToken {
        operation_type: OperationType,
    },
    MissingSampledToken {
        operation_type: OperationType,
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
    SampledTokenNotAllowed {
        token_id: u32,
    },
    UnexpectedPromptLogprobs {
        operation_type: OperationType,
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
        operation_type: OperationType,
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
    use uniserve_worker_wire::{SequenceDelta, SequenceEffect};

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
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan sequence extension")
    }

    #[test]
    fn planner_emits_the_closed_sequence_variant() {
        let transition = prefill_transition();
        assert_eq!(transition.operation_type, OperationType::SequenceExtend);
        let Operation::Sequence(sequence) = transition.op.operation else {
            panic!("expected sequence operation");
        };
        assert_eq!(sequence.mode, SequenceMode::Extend);
        assert_eq!(sequence.position, (0, 2));
        let SequenceInput::Tokens(tokens) = sequence.input else {
            panic!("expected token input");
        };
        assert_eq!(tokens.token_ids, vec![11, 12]);
    }

    #[test]
    fn decode_conditioning_policy_matches_the_resolved_branch_capability() {
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
                        token_id: 12,
                        token_source: TokenSource::Wire,
                        new_blocks: Vec::new(),
                        spec_token_ids: None,
                        recent_tokens: None,
                        allowed_tokens: None,
                        suppress_tokens: None,
                    },
                )
                .expect("plan sequence decode");
            let Operation::Sequence(sequence) = transition.op.operation else {
                panic!("expected sequence operation");
            };
            assert_eq!(
                sequence.policy.publish_kv_on_tokens,
                request
                    .behavior
                    .gen_output
                    .then_some(42)
                    .into_iter()
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn transition_accepts_only_the_sealed_typed_result() {
        let mut transition = prefill_transition();
        transition.assign_envelope(3, 17, 5);
        let result = OperationResult {
            session_id: RequestId(9),
            epoch: 3,
            op_id: 17,
            base_version: 5,
            result_version: 6,
            delta: ResultDelta::Sequence(SequenceDelta {
                effect: SequenceEffect {
                    kv_tokens: Some(2),
                    sampled_token_ids: vec![13],
                    ..SequenceEffect::default()
                },
            }),
        };
        assert_eq!(transition.validate_result(&result), Ok(()));

        let stale = OperationResult {
            result_version: 7,
            ..result
        };
        assert!(matches!(
            transition.validate_result(&stale),
            Err(TransitionValidationError::VersionMismatch { .. })
        ));
    }
}
