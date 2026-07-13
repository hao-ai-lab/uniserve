use std::collections::HashSet;

use uniserve_core::{
    BlockId, CfgParams, ContextSegment, GenerationRequest, ImageIngestRecipe, ImageIngestStep,
    ImageKvEffect, Modality, SegmentPlacement, UndTokenAction,
};
use uniserve_worker_wire::{ForwardOp, OpKind, ResourceClass, SeqResult, TokenSource};

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
                scratch_units: 0,
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
        result: &SeqResult,
    ) -> Result<(), CursorApplyError> {
        let op_id = transition
            .op_id
            .ok_or(CursorApplyError::MissingOperationId)?;
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
                    .saturating_add(result.num_tokens.unwrap_or(0));
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
                commits_nonterminal_stop_tail,
                ref stop_token_ids,
                ..
            } => {
                let mut actual_count = result
                    .sampled_token_ids
                    .as_ref()
                    .filter(|tokens| !tokens.is_empty())
                    .map_or(1, Vec::len)
                    .max(
                        result
                            .num_accepted_tokens
                            .map_or(0, |accepted| accepted as usize + 1),
                    ) as u32;
                if commits_nonterminal_stop_tail
                    && result.sampled_token_ids.as_ref().is_some_and(|tokens| {
                        tokens
                            .last()
                            .is_some_and(|token| stop_token_ids.contains(token))
                    })
                {
                    actual_count = actual_count.saturating_add(1);
                }
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
                self.image_gen.steps_done = self.image_gen.steps_done.max(
                    result
                        .num_steps_done
                        .unwrap_or_else(|| start_step.saturating_add(step_count)),
                );
            }
            TransitionDelta::CommitGen {
                logical_position,
                physical_position,
                logical_positions,
                physical_kv_tokens,
                ..
            } => {
                if result.locator.is_none() && physical_kv_tokens.is_some() {
                    let added = result.num_tokens.unwrap_or(0);
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
                let added = result.num_tokens.unwrap_or(0);
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
                    .saturating_add(result.num_tokens.unwrap_or(0));
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
    pub(crate) scratch_units: u64,
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
                token_count,
                ..
            } => {
                let count = u32::from(token_count.max(1));
                let end = logical_position.saturating_add(count);
                self.logical_pos = self.logical_pos.max(end);
                self.physical_kv_len = self
                    .physical_kv_len
                    .max(physical_position.saturating_add(count));
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
        token_count: u16,
        stop_token_ids: Option<Vec<u32>>,
        stop_terminal: bool,
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
        scratch_units: u64,
        host_scratch_tokens: u64,
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
        let (
            op,
            delta,
            encoder_pins,
            replayability_after_apply,
            latent_units,
            scratch_units,
            host_scratch_tokens,
        ) = match intent {
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
                    ForwardOp {
                        req_id: request.request_id,
                        kind: OpKind::PrefillUnd,
                        modality: Modality::Und,
                        new_block_ids: new_blocks,
                        pos_range: (
                            cursor.logical_pos,
                            cursor.logical_pos.saturating_add(token_count),
                        ),
                        token_ids: Some(token_ids),
                        recent_tokens,
                        allowed_tokens,
                        suppress_tokens,
                        return_all_logits: request.sampling.prompt_logprobs_requested(),
                        ..Default::default()
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
                let (kind, modality) = match step {
                    ImageIngestStep::VaeEncode => (OpKind::VaeEncode, Modality::Gen),
                    ImageIngestStep::VitEncode => (OpKind::VitEncode, Modality::Und),
                };
                (
                    ForwardOp {
                        req_id: request.request_id,
                        kind,
                        modality,
                        new_block_ids: new_blocks,
                        pos_range: (position, position.saturating_add(logical_positions.max(1))),
                        cond_pos: Some(position),
                        image_b64: (!cache_hit && !image_b64.is_empty()).then_some(image_b64),
                        image_in: staged_image,
                        mm_hash: Some(worker_hash),
                        ..Default::default()
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
                    },
                    encoder_cache_key.into_iter().collect(),
                    cursor.replayability,
                    0,
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
                token_count,
                stop_token_ids,
                stop_terminal,
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
                let token_count = token_count.max(1);
                let transition_stop_token_ids = stop_token_ids.clone().unwrap_or_default();
                (
                    ForwardOp {
                        req_id: request.request_id,
                        kind: OpKind::DecodeUnd,
                        modality: Modality::Und,
                        new_block_ids: new_blocks,
                        pos_range: (position, position.saturating_add(1)),
                        token_ids: Some(vec![token_id]),
                        token_source,
                        spec_token_ids,
                        decode_token_count: (token_count > 1).then_some(token_count),
                        decode_stop_token_ids: stop_token_ids,
                        decode_stop_terminal: stop_terminal,
                        recent_tokens,
                        allowed_tokens,
                        suppress_tokens,
                        ..Default::default()
                    },
                    TransitionDelta::DecodeUnd {
                        logical_position: position,
                        physical_position: cursor.physical_kv_len,
                        token_count,
                        stop_token_ids: transition_stop_token_ids,
                        commits_nonterminal_stop_tail: token_count > 1 && !stop_terminal,
                    },
                    Vec::new(),
                    cursor.replayability,
                    0,
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
                scratch_units,
                host_scratch_tokens,
            } => {
                if !request.behavior.gen_output {
                    return Err(PlanningError::GenerationBranchDisabled);
                }
                let step_count = step_count.max(1);
                (
                    ForwardOp {
                        req_id: request.request_id,
                        kind: OpKind::DenoiseGen,
                        modality: Modality::Gen,
                        pos_range: (position, position.saturating_add(1)),
                        timestep_idx: Some(start_step),
                        denoise_step_count: Some(step_count),
                        cond_pos: Some(position),
                        cfg: Some(cfg),
                        image_prompt,
                        ..Default::default()
                    },
                    TransitionDelta::DenoiseGen {
                        image_id,
                        start_step,
                        step_count,
                    },
                    Vec::new(),
                    Replayability::NotReplayable,
                    latent_units,
                    scratch_units,
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
                    ForwardOp {
                        req_id: request.request_id,
                        kind: OpKind::CommitGen,
                        modality: Modality::Gen,
                        new_block_ids: new_blocks,
                        pos_range: (position, position.saturating_add(1)),
                        cond_pos: Some(position),
                        recent_tokens,
                        allowed_tokens,
                        suppress_tokens,
                        ..Default::default()
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
                    ForwardOp {
                        req_id: request.request_id,
                        kind: OpKind::CommitWriteback,
                        modality: Modality::Und,
                        new_block_ids: new_blocks,
                        pos_range: (position, position.saturating_add(1)),
                        cond_pos: Some(position),
                        locator: Some(locator),
                        recent_tokens,
                        allowed_tokens,
                        suppress_tokens,
                        ..Default::default()
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
                let (kind, modality) = match step {
                    ImageIngestStep::VaeEncode => (OpKind::VaeEncode, Modality::Gen),
                    ImageIngestStep::VitEncode => (OpKind::VitEncode, Modality::Und),
                };
                (
                    ForwardOp {
                        req_id: request.request_id,
                        kind,
                        modality,
                        new_block_ids: new_blocks,
                        pos_range: (position, position.saturating_add(logical_positions.max(1))),
                        cond_pos: Some(position),
                        image_b64: Some(image_b64),
                        image_in: staged_image,
                        mm_hash: Some(worker_hash),
                        ..Default::default()
                    },
                    TransitionDelta::FeedbackIngestStep {
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
                    0,
                )
            }
        };
        let kind = op.kind;
        let kv_target_tokens = transition_kv_target(&delta).or_else(|| {
            transition_may_write_worker_defined_kv(&delta)
                .then_some(request.resources.max_kv_tokens)
        });
        let resources = TransitionResources {
            new_blocks: op.new_block_ids.len(),
            kv_target_tokens,
            scratch_units: if kind == OpKind::DenoiseGen {
                scratch_units
            } else {
                0
            },
            host_scratch_tokens: if kind == OpKind::DenoiseGen {
                host_scratch_tokens
            } else {
                0
            },
            latent_units: if kind == OpKind::DenoiseGen {
                latent_units
            } else {
                0
            },
            encoder_pins,
            replayability_after_apply,
            release_on_apply: match kind {
                OpKind::CommitGen => {
                    vec![ResourceClass::ImageLatent, ResourceClass::Scratch]
                }
                _ => Vec::new(),
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
            expects_encoder_handle: matches!(kind, OpKind::VaeEncode | OpKind::VitEncode),
            expected_encoder_handle: match delta {
                TransitionDelta::IngestImageStep {
                    cache_hit: true, ..
                } => op.image_in,
                _ => None,
            },
            expects_image_artifact: kind == OpKind::CommitGen,
            expected_image_hw: (kind == OpKind::CommitGen)
                .then_some((request.image.height, request.image.width)),
            requires_image_locator: kind == OpKind::CommitGen
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
                kind,
                OpKind::PrefillUnd
                    | OpKind::DecodeUnd
                    | OpKind::TargetVerifyUnd
                    | OpKind::CommitGen
                    | OpKind::CommitWriteback
            ),
            expects_sampled_token: matches!(
                kind,
                OpKind::PrefillUnd | OpKind::DecodeUnd | OpKind::TargetVerifyUnd
            ),
            expected_text_tokens: match &delta {
                TransitionDelta::IngestText { .. } => Some(TextTokenCountRange { min: 1, max: 1 }),
                TransitionDelta::DecodeUnd { token_count, .. } => {
                    let max = op
                        .spec_token_ids
                        .as_ref()
                        .map_or(u32::from((*token_count).max(1)), |tokens| {
                            tokens.len().saturating_add(1).min(u32::MAX as usize) as u32
                        });
                    Some(TextTokenCountRange { min: 1, max })
                }
                _ => None,
            },
            max_accepted_draft_tokens: op
                .spec_token_ids
                .as_ref()
                .map(|tokens| tokens.len().min(u32::MAX as usize) as u32),
            allowed_text_tokens: op.allowed_tokens.clone(),
            generated_logprobs_requested: request.sampling.generated_logprobs_requested()
                && matches!(
                    kind,
                    OpKind::PrefillUnd
                        | OpKind::DecodeUnd
                        | OpKind::TargetVerifyUnd
                        | OpKind::CommitGen
                        | OpKind::CommitWriteback
                ),
            expected_prompt_token_ids: op.return_all_logits.then(|| {
                let mut tokens = op.token_ids.clone().unwrap_or_default();
                if matches!(&delta, TransitionDelta::IngestText { start: 0, .. })
                    && !tokens.is_empty()
                {
                    tokens.remove(0);
                }
                tokens
            }),
        };
        Ok(PlannedTransition {
            op,
            op_id: None,
            kind,
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
            physical_position,
            token_count,
            ..
        } => physical_position.saturating_add(u32::from((*token_count).max(1))),
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
}

/// Scheduler-local transition associated with one submitted worker op.
#[derive(Debug, Clone)]
pub(crate) struct PlannedTransition {
    pub(crate) op: ForwardOp,
    pub(crate) op_id: Option<u64>,
    pub(crate) kind: OpKind,
    pub(crate) delta: TransitionDelta,
    pub(crate) resources: TransitionResources,
    pub(crate) validation: TransitionValidation,
    pub(crate) visibility: OutputVisibilityPlan,
}

impl PlannedTransition {
    pub(crate) fn assign_op_id(&mut self, op_id: u64) {
        self.op_id = Some(op_id);
        self.op.op_id = Some(op_id);
    }

    pub(crate) fn validate_result(
        &self,
        result: &SeqResult,
    ) -> Result<(), TransitionValidationError> {
        if let Some(expected) = self.op_id
            && result.op_id != Some(expected)
        {
            return Err(TransitionValidationError::OpIdMismatch {
                expected,
                actual: result.op_id,
            });
        }
        self.validation.validate(self.kind, result)
    }
}

impl std::ops::Deref for PlannedTransition {
    type Target = ForwardOp;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl std::ops::DerefMut for PlannedTransition {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.op
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
    },
    DecodeUnd {
        logical_position: u32,
        physical_position: u32,
        token_count: u16,
        stop_token_ids: Vec<u32>,
        commits_nonterminal_stop_tail: bool,
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
    pub(crate) scratch_units: u64,
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
    fn validate(&self, kind: OpKind, result: &SeqResult) -> Result<(), TransitionValidationError> {
        if result.op_kind != Some(kind) {
            return Err(TransitionValidationError::OpKindMismatch {
                expected: kind,
                actual: result.op_kind,
            });
        }
        if let Some(expected_step) = self.expected_denoise_step {
            let steps_done = result
                .num_steps_done
                .ok_or(TransitionValidationError::MissingDenoiseStep)?;
            if steps_done != expected_step {
                return Err(TransitionValidationError::DenoiseStepMismatch {
                    expected: expected_step,
                    actual: steps_done,
                });
            }
        }
        if self.expects_encoder_handle
            && result
                .encoder_handle
                .filter(|handle| *handle != 0)
                .is_none()
        {
            return Err(TransitionValidationError::MissingEncoderHandle);
        }
        if let Some(expected) = self.expected_encoder_handle
            && result.encoder_handle != Some(expected)
        {
            return Err(TransitionValidationError::EncoderHandleMismatch {
                expected,
                actual: result.encoder_handle,
            });
        }
        if let Some(expected) = self.expected_image_hw {
            let actual = result
                .image_hw
                .ok_or(TransitionValidationError::MissingImageDimensions)?;
            if actual != expected {
                return Err(TransitionValidationError::ImageDimensionsMismatch {
                    expected,
                    actual,
                });
            }
        }
        if self.expects_image_artifact {
            let image = result
                .image_png_b64
                .as_deref()
                .filter(|value| !value.is_empty())
                .ok_or(TransitionValidationError::MissingImageArtifact)?;
            let metadata = validate_png_artifact(image, self.expected_image_hw)
                .ok_or(TransitionValidationError::InvalidImageArtifact)?;
            if result.image_hw != Some((metadata.height, metadata.width)) {
                return Err(TransitionValidationError::ImageArtifactDimensionsMismatch {
                    artifact: (metadata.height, metadata.width),
                    reported: result.image_hw,
                });
            }
        }
        if self.requires_image_locator && result.locator.as_deref().is_none_or(str::is_empty) {
            return Err(TransitionValidationError::MissingImageLocator);
        }
        if let Some(expected) = self.expected_image_kv
            && !self.requires_image_locator
        {
            let actual = result
                .num_tokens
                .ok_or(TransitionValidationError::MissingImageKvTokens)?;
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
        if !self.allows_sampled_tokens
            && (result.sampled_token_id.is_some()
                || result
                    .sampled_token_ids
                    .as_ref()
                    .is_some_and(|tokens| !tokens.is_empty()))
        {
            return Err(TransitionValidationError::UnexpectedSampledToken { kind });
        }
        if self.expects_sampled_token
            && result.sampled_token_id.is_none()
            && result
                .sampled_token_ids
                .as_ref()
                .is_none_or(|tokens| tokens.is_empty())
        {
            return Err(TransitionValidationError::MissingSampledToken { kind });
        }
        if let Some(max_accepted) = self.max_accepted_draft_tokens
            && result
                .num_accepted_tokens
                .is_some_and(|accepted| accepted > max_accepted)
        {
            return Err(TransitionValidationError::AcceptedDraftCountExceeded {
                max: max_accepted,
                actual: result.num_accepted_tokens.unwrap_or_default(),
            });
        }
        if let Some(allowed) = &self.allowed_text_tokens {
            let returned = result
                .sampled_token_ids
                .as_ref()
                .filter(|tokens| !tokens.is_empty())
                .cloned()
                .or_else(|| result.sampled_token_id.map(|token| vec![token]))
                .unwrap_or_default();
            if let Some(token_id) = returned
                .into_iter()
                .find(|token_id| !allowed.contains(token_id))
            {
                return Err(TransitionValidationError::SampledTokenNotAllowed { token_id });
            }
        }
        if let Some(expected) = self.expected_text_tokens {
            let listed = result
                .sampled_token_ids
                .as_ref()
                .filter(|tokens| !tokens.is_empty())
                .map(|tokens| tokens.len() as u32);
            let scalar = u32::from(result.sampled_token_id.is_some());
            let accepted = result
                .num_accepted_tokens
                .map(|count| count.saturating_add(scalar));
            if let (Some(listed), Some(accepted)) = (listed, accepted)
                && listed != accepted
            {
                return Err(TransitionValidationError::TextTokenCountInconsistent {
                    listed,
                    accepted,
                });
            }
            let actual = listed.or(accepted).unwrap_or(scalar);
            if actual < expected.min || actual > expected.max {
                return Err(TransitionValidationError::TextTokenCountMismatch {
                    min: expected.min,
                    max: expected.max,
                    actual,
                });
            }
        }
        let sampled_token = result
            .sampled_token_ids
            .as_ref()
            .and_then(|tokens| tokens.last().copied())
            .or(result.sampled_token_id);
        let generated_candidates = result.top_logprobs.as_deref().unwrap_or_default();
        match (
            self.generated_logprobs_requested,
            sampled_token,
            generated_candidates.is_empty(),
        ) {
            (false, _, false) | (true, None, false) => {
                return Err(TransitionValidationError::UnexpectedGeneratedLogprobs { kind });
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
            result.prompt_logprobs.as_deref(),
        ) {
            (None, Some(positions)) if !positions.is_empty() => {
                return Err(TransitionValidationError::UnexpectedPromptLogprobs { kind });
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransitionValidationError {
    OpIdMismatch {
        expected: u64,
        actual: Option<u64>,
    },
    OpKindMismatch {
        expected: OpKind,
        actual: Option<OpKind>,
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
        kind: OpKind,
    },
    MissingSampledToken {
        kind: OpKind,
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
        kind: OpKind,
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
        kind: OpKind,
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
    use base64::Engine as _;

    use super::*;
    use uniserve_core::{
        GenerationBehaviorDescriptor, GenerationConstraint, GenerationPolicyDescriptor,
        GenerationResourceBounds, ImageParams, RequestId, SamplingParams, UndVisibility,
    };

    fn test_request(id: u64, tokens: Vec<u32>) -> GenerationRequest {
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

    fn png_b64(width: u32, height: u32) -> String {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("PNG header");
            writer
                .write_image_data(&vec![0; (width * height) as usize])
                .expect("PNG pixels");
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn cursor_exposes_typed_views_without_duplicate_state() {
        let mut cursor = GenerationCursor::new(GenerationPhase::Prefill, 8, true);
        cursor.ingest.prompt_cursor = 3;
        cursor.und.logical_pos = 5;
        cursor.und.physical_kv_len = 7;
        cursor.image_gen.image_id = 2;

        assert_eq!(cursor.context().prompt_cursor, 3);
        assert_eq!(cursor.und().logical_pos, 5);
        assert_eq!(cursor.und().physical_kv_len, 7);
        assert_eq!(cursor.gen_cursor().image_id, 2);
        assert_eq!(cursor.resources().worstcase_blocks, 8);
        assert_eq!(cursor.replay().generated_ids, Vec::<u32>::new());
    }

    #[test]
    fn planned_transition_validates_op_id_and_encoder_handle() {
        let request = test_request(9, vec![1, 2, 3, 4]);
        let mut transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::Encode,
                    prompt_cursor: 4,
                    logical_pos: 4,
                    physical_kv_len: 4,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestImage {
                    segment_index: 1,
                    step_index: 0,
                    step: ImageIngestStep::VitEncode,
                    is_final_step: true,
                    position: 4,
                    logical_positions: 1,
                    physical_kv_tokens: ImageKvEffect::WorkerDefined,
                    worker_hash: 123,
                    encoder_cache_key: Some(123),
                    cache_hit: false,
                    image_b64: "aGVsbG8=".to_string(),
                    staged_image: None,
                    new_blocks: Vec::new(),
                },
            )
            .expect("plan image ingest");
        transition.assign_op_id(77);

        assert_eq!(transition.op.kind, OpKind::VitEncode,);
        assert!(matches!(
            transition.delta,
            TransitionDelta::IngestImageStep {
                position: 4,
                step_index: 0,
                is_final_step: true,
                ..
            }
        ));
        assert_eq!(transition.resources.encoder_pins, vec![123]);
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_kind: Some(OpKind::VitEncode),
                op_id: Some(77),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingEncoderHandle)
        );
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_kind: Some(OpKind::VitEncode),
                op_id: Some(78),
                encoder_handle: Some(1),
                num_tokens: Some(4),
                ..Default::default()
            }),
            Err(TransitionValidationError::OpIdMismatch {
                expected: 77,
                actual: Some(78)
            })
        );
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_kind: Some(OpKind::VitEncode),
                op_id: Some(77),
                encoder_handle: Some(1),
                locator: Some("encoder-locator".to_string()),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingImageKvTokens)
        );
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_kind: Some(OpKind::VitEncode),
                op_id: Some(77),
                encoder_handle: Some(1),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingImageKvTokens)
        );
        assert!(
            transition
                .validate_result(&SeqResult {
                    req_id: RequestId(9),
                    op_kind: Some(OpKind::VitEncode),
                    op_id: Some(77),
                    encoder_handle: Some(1),
                    num_tokens: Some(4),
                    ..Default::default()
                })
                .is_ok()
        );
    }

    #[test]
    fn intermediate_image_ingest_requires_a_bounded_kv_result() {
        let request = test_request(10, vec![1, 2, 3]);
        let mut transition = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::Encode,
                    prompt_cursor: 3,
                    logical_pos: 3,
                    physical_kv_len: 3,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::IngestImage {
                    segment_index: 1,
                    step_index: 0,
                    step: ImageIngestStep::VaeEncode,
                    is_final_step: false,
                    position: 3,
                    logical_positions: 1,
                    physical_kv_tokens: ImageKvEffect::WorkerDefined,
                    worker_hash: 456,
                    encoder_cache_key: Some(456),
                    cache_hit: false,
                    image_b64: "aGVsbG8=".to_string(),
                    staged_image: None,
                    new_blocks: Vec::new(),
                },
            )
            .expect("plan intermediate image ingest");
        transition.assign_op_id(81);

        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(10),
                op_kind: Some(OpKind::VaeEncode),
                op_id: Some(81),
                encoder_handle: Some(2),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingImageKvTokens)
        );
        assert!(
            transition
                .validate_result(&SeqResult {
                    req_id: RequestId(10),
                    op_kind: Some(OpKind::VaeEncode),
                    op_id: Some(81),
                    encoder_handle: Some(2),
                    num_tokens: Some(1),
                    ..Default::default()
                })
                .is_ok()
        );
    }

    #[test]
    fn cursor_projection_applies_planned_lifecycle_without_mutating_cursor() {
        let cursor = GenerationCursor::new(GenerationPhase::Prefill, 8, false);
        let request = test_request(1, vec![1, 2, 3]);
        let planner = GenerationPlanner::new();
        let prefill = planner
            .plan(
                &request,
                cursor.project(std::iter::empty()),
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![1, 2, 3],
                    new_blocks: vec![uniserve_core::BlockId(1), uniserve_core::BlockId(2)],
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan prefill");
        let after_prefill = cursor.project([&prefill]);
        let decode = planner
            .plan(
                &request,
                after_prefill,
                TransitionIntent::DecodeUnd {
                    position: 3,
                    token_id: 3,
                    token_source: TokenSource::Wire,
                    new_blocks: Vec::new(),
                    spec_token_ids: None,
                    token_count: 2,
                    stop_token_ids: None,
                    stop_terminal: true,
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan decode");

        let projected = cursor.project([&prefill, &decode]);

        assert_eq!(cursor.ingest.prompt_cursor, 0);
        assert_eq!(cursor.und.logical_pos, 0);
        assert_eq!(projected.prompt_cursor, 3);
        assert_eq!(projected.logical_pos, 5);
    }

    #[test]
    fn validated_transition_applies_exactly_once() {
        let request = test_request(1, vec![1, 2]);
        let mut cursor = GenerationCursor::new(GenerationPhase::Prefill, 4, false);
        let mut transition = GenerationPlanner::new()
            .plan(
                &request,
                cursor.project(std::iter::empty()),
                TransitionIntent::IngestText {
                    segment_index: 0,
                    prompt_start: 0,
                    token_ids: vec![1, 2],
                    new_blocks: Vec::new(),
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan prefill");
        transition.assign_op_id(5);
        let result = SeqResult {
            req_id: RequestId(1),
            op_kind: Some(OpKind::PrefillUnd),
            sampled_token_id: Some(3),
            op_id: Some(5),
            ..Default::default()
        };
        transition.validate_result(&result).expect("valid result");
        cursor
            .apply_transition(&transition, &result)
            .expect("first application");
        assert_eq!(cursor.ingest.prompt_cursor, 2);
        assert_eq!(
            cursor.apply_transition(&transition, &result),
            Err(CursorApplyError::DuplicateOperation { op_id: 5 })
        );
    }

    #[test]
    fn decode_and_denoise_results_validate_operation_shape() {
        let request = test_request(1, vec![1, 2]);
        let mut decode = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::DecodeUnd,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::DecodeUnd {
                    position: 2,
                    token_id: 2,
                    token_source: TokenSource::Wire,
                    new_blocks: Vec::new(),
                    spec_token_ids: None,
                    token_count: 1,
                    stop_token_ids: None,
                    stop_terminal: true,
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan decode");
        decode.assign_op_id(8);
        assert_eq!(
            decode.validate_result(&SeqResult {
                req_id: RequestId(1),
                op_kind: Some(OpKind::PrefillUnd),
                op_id: Some(8),
                sampled_token_id: Some(3),
                ..Default::default()
            }),
            Err(TransitionValidationError::OpKindMismatch {
                expected: OpKind::DecodeUnd,
                actual: Some(OpKind::PrefillUnd),
            })
        );
        assert_eq!(
            decode.validate_result(&SeqResult {
                req_id: RequestId(1),
                op_kind: Some(OpKind::DecodeUnd),
                op_id: Some(8),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingSampledToken {
                kind: OpKind::DecodeUnd,
            })
        );
        assert_eq!(
            decode.validate_result(&SeqResult {
                req_id: RequestId(1),
                op_kind: Some(OpKind::DecodeUnd),
                op_id: Some(8),
                sampled_token_id: Some(4),
                sampled_token_ids: Some(vec![3, 4]),
                ..Default::default()
            }),
            Err(TransitionValidationError::TextTokenCountMismatch {
                min: 1,
                max: 1,
                actual: 2,
            })
        );

        let mut generation = request;
        generation.constraint = GenerationConstraint::Default;
        generation.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 9 };
        generation.behavior =
            GenerationBehaviorDescriptor::resolve(generation.constraint, &generation.policy);
        let mut denoise = GenerationPlanner::new()
            .plan(
                &generation,
                CursorProjection {
                    phase: GenerationPhase::DenoiseGen,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::DenoiseGen {
                    image_id: 1,
                    position: 2,
                    start_step: 4,
                    step_count: 3,
                    cfg: uniserve_core::CfgParams {
                        branch_count: 3,
                        text_scale: 4.0,
                        img_scale: 1.5,
                        renorm_type: "none".to_string(),
                        renorm_min: 0.0,
                        interval: (0.0, 1.0),
                    },
                    image_prompt: None,
                    latent_units: 64,
                    scratch_units: 3,
                    host_scratch_tokens: 12,
                },
            )
            .expect("plan denoise");
        denoise.assign_op_id(9);
        assert_eq!(denoise.resources.latent_units, 64);
        assert_eq!(denoise.resources.scratch_units, 3);
        assert_eq!(denoise.resources.host_scratch_tokens, 12);
        assert_eq!(
            denoise.resources.replayability_after_apply,
            Replayability::NotReplayable
        );
        assert_eq!(
            denoise.validate_result(&SeqResult {
                req_id: RequestId(1),
                op_kind: Some(OpKind::DenoiseGen),
                op_id: Some(9),
                num_steps_done: Some(6),
                ..Default::default()
            }),
            Err(TransitionValidationError::DenoiseStepMismatch {
                expected: 7,
                actual: 6,
            })
        );
    }

    #[test]
    fn generated_logprob_results_must_match_the_sampled_token() {
        let mut request = test_request(1, vec![1, 2]);
        request.sampling.return_logprobs = true;
        request.sampling.n_logprobs = 2;
        let mut decode = GenerationPlanner::new()
            .plan(
                &request,
                CursorProjection {
                    phase: GenerationPhase::DecodeUnd,
                    prompt_cursor: 2,
                    logical_pos: 2,
                    physical_kv_len: 2,
                    replayability: Replayability::Replayable,
                },
                TransitionIntent::DecodeUnd {
                    position: 2,
                    token_id: 2,
                    token_source: TokenSource::Wire,
                    new_blocks: Vec::new(),
                    spec_token_ids: None,
                    token_count: 1,
                    stop_token_ids: None,
                    stop_terminal: true,
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan decode");
        decode.assign_op_id(21);

        let result = |top_logprobs| SeqResult {
            req_id: RequestId(1),
            op_kind: Some(OpKind::DecodeUnd),
            op_id: Some(21),
            sampled_token_id: Some(7),
            top_logprobs,
            ..Default::default()
        };
        assert_eq!(
            decode.validate_result(&result(None)),
            Err(TransitionValidationError::MissingGeneratedLogprobs { token_id: 7 })
        );
        assert_eq!(
            decode.validate_result(&result(Some(vec![uniserve_worker_wire::TokenLogprob(
                8, -0.1, 1,
            )]))),
            Err(TransitionValidationError::GeneratedLogprobTokenMismatch {
                expected: 7,
                actual: 8,
            })
        );
        assert_eq!(
            decode.validate_result(&result(Some(vec![
                uniserve_worker_wire::TokenLogprob(7, -0.2, 2),
                uniserve_worker_wire::TokenLogprob(7, -0.1, 1),
            ]))),
            Err(TransitionValidationError::InvalidGeneratedLogprobCandidates)
        );
        decode
            .validate_result(&result(Some(vec![
                uniserve_worker_wire::TokenLogprob(7, -0.2, 1),
                uniserve_worker_wire::TokenLogprob(8, -0.1, 1),
            ])))
            .expect("valid generated candidates with tied vocab ranks");
    }

    #[test]
    fn prompt_logprob_plan_scores_exact_prefill_positions() {
        let mut request = test_request(1, vec![10, 11, 12]);
        request.sampling.return_prompt_logprobs = true;
        request.sampling.n_prompt_logprobs = 1;
        let mut prefill = GenerationPlanner::new()
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
                    token_ids: vec![10, 11, 12],
                    new_blocks: Vec::new(),
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan prefill");
        assert!(prefill.op.return_all_logits);
        prefill.assign_op_id(22);
        let valid = SeqResult {
            req_id: RequestId(1),
            op_kind: Some(OpKind::PrefillUnd),
            op_id: Some(22),
            sampled_token_id: Some(13),
            prompt_logprobs: Some(vec![
                vec![uniserve_worker_wire::TokenLogprob(11, -0.2, 1)],
                vec![uniserve_worker_wire::TokenLogprob(12, -0.3, 1)],
            ]),
            ..Default::default()
        };
        prefill
            .validate_result(&valid)
            .expect("exact prompt positions");

        let mut wrong = valid;
        wrong.prompt_logprobs.as_mut().expect("positions")[1][0].0 = 99;
        assert_eq!(
            prefill.validate_result(&wrong),
            Err(TransitionValidationError::PromptLogprobTokenMismatch {
                position: 1,
                expected: 12,
                actual: 99,
            })
        );
    }

    #[test]
    fn non_replayable_cursor_stays_non_replayable_through_text_decode() {
        let request = test_request(5, vec![1, 2]);
        let mut cursor = GenerationCursor::new(GenerationPhase::DecodeUnd, 4, false);
        cursor.replay.replayability = Replayability::NotReplayable;
        cursor.und.logical_pos = 2;
        cursor.und.physical_kv_len = 2;
        let mut decode = GenerationPlanner::new()
            .plan(
                &request,
                cursor.project(std::iter::empty()),
                TransitionIntent::DecodeUnd {
                    position: 2,
                    token_id: 2,
                    token_source: TokenSource::Wire,
                    new_blocks: Vec::new(),
                    spec_token_ids: None,
                    token_count: 1,
                    stop_token_ids: None,
                    stop_terminal: true,
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan decode");
        assert_eq!(
            decode.resources.replayability_after_apply,
            Replayability::NotReplayable
        );
        decode.assign_op_id(12);
        let result = SeqResult {
            req_id: RequestId(5),
            op_kind: Some(OpKind::DecodeUnd),
            op_id: Some(12),
            sampled_token_id: Some(3),
            ..Default::default()
        };
        decode.validate_result(&result).expect("validate decode");
        cursor
            .apply_transition(&decode, &result)
            .expect("apply decode");
        assert_eq!(cursor.replay.replayability, Replayability::NotReplayable);
    }

    #[test]
    fn generated_image_commit_validates_dimensions_and_feedback_writeback() {
        let mut request = test_request(3, vec![1, 2]);
        request.constraint = GenerationConstraint::Default;
        request.image.height = 480;
        request.image.width = 640;
        request.policy.trigger = uniserve_core::TriggerPolicyDescriptor::Token { token_id: 9 };
        request.policy.feedback = Some(uniserve_core::GeneratedImageFeedbackRecipe {
            commit: uniserve_core::CommitRecipe::CommitGenThenWriteback,
            writeback: uniserve_core::FeedbackWriteback::DirectKv,
            next_und_token: uniserve_core::FeedbackNextToken::EndOfImage,
            logical_positions: 1,
            physical_kv_tokens: ImageKvEffect::Exact { tokens: 1 },
        });
        request.behavior =
            GenerationBehaviorDescriptor::resolve(request.constraint, &request.policy);
        let mut commit = GenerationPlanner::new()
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
                    position: 2,
                    new_blocks: Vec::new(),
                    recent_tokens: None,
                    allowed_tokens: None,
                    suppress_tokens: None,
                },
            )
            .expect("plan commit");
        commit.assign_op_id(10);
        let valid_png = png_b64(640, 480);
        assert_eq!(
            commit.validate_result(&SeqResult {
                req_id: RequestId(3),
                op_kind: Some(OpKind::CommitGen),
                op_id: Some(10),
                image_hw: Some((640, 480)),
                image_png_b64: Some(valid_png.clone()),
                locator: Some("image".to_string()),
                ..Default::default()
            }),
            Err(TransitionValidationError::ImageDimensionsMismatch {
                expected: (480, 640),
                actual: (640, 480),
            })
        );
        assert_eq!(
            commit.validate_result(&SeqResult {
                req_id: RequestId(3),
                op_kind: Some(OpKind::CommitGen),
                op_id: Some(10),
                image_hw: Some((480, 640)),
                image_png_b64: Some(valid_png.clone()),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingImageLocator)
        );
        assert_eq!(
            commit.validate_result(&SeqResult {
                req_id: RequestId(3),
                op_kind: Some(OpKind::CommitGen),
                op_id: Some(10),
                image_hw: Some((480, 640)),
                locator: Some("image".to_string()),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingImageArtifact)
        );
        assert_eq!(
            commit.validate_result(&SeqResult {
                req_id: RequestId(3),
                op_kind: Some(OpKind::CommitGen),
                op_id: Some(10),
                image_hw: Some((480, 640)),
                image_png_b64: Some("cG5n".to_string()),
                locator: Some("image".to_string()),
                ..Default::default()
            }),
            Err(TransitionValidationError::InvalidImageArtifact)
        );
        assert_eq!(
            commit.validate_result(&SeqResult {
                req_id: RequestId(3),
                op_kind: Some(OpKind::CommitGen),
                op_id: Some(10),
                image_hw: Some((480, 640)),
                image_png_b64: Some(valid_png),
                locator: Some("image".to_string()),
                ..Default::default()
            }),
            Ok(())
        );
    }
}
