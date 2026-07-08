use uniserve_worker_wire::{ForwardOp, OpKind, SeqResult};

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
}

/// Scheduler-owned lifecycle cursor for one running request.
#[derive(Debug, Clone)]
pub struct GenerationCursor {
    pub(crate) phase: GenerationPhase,
    pub(crate) pos: u32,
    pub(crate) next_token: u32,
    pub(crate) n_generated: usize,
    pub(crate) worstcase_blocks: usize,
    pub(crate) reserve_worstcase: bool,
    pub(crate) image_id: u32,
    pub(crate) images_done: usize,
    pub(crate) text_since_image: usize,
    pub(crate) default_image_pending: bool,
    pub(crate) cond_pos: u32,
    pub(crate) steps_done: u16,
    pub(crate) prompt_cursor: u32,
    pub(crate) preempted: bool,
    pub(crate) block_hashes: Vec<u64>,
    pub(crate) prefix_cached_blocks: usize,
    pub(crate) blocks_cached: bool,
    pub(crate) generated_ids: Vec<u32>,
    pub(crate) recompute_ids: Option<Vec<u32>>,
    pub(crate) mm_cursor: usize,
    pub(crate) mm_acquired: Vec<u64>,
    pub(crate) kvlen: u32,
    pub(crate) context_encode_step: u8,
    pub(crate) gen_h: u32,
    pub(crate) gen_w: u32,
    pub(crate) round_tokens: Vec<u32>,
    pub(crate) context_round_closing: bool,
    pub(crate) worker_image_latent_units: u64,
}

impl GenerationCursor {
    pub(crate) fn new(
        phase: GenerationPhase,
        worstcase_blocks: usize,
        reserve_worstcase: bool,
    ) -> Self {
        Self {
            phase,
            pos: 0,
            next_token: 0,
            n_generated: 0,
            worstcase_blocks,
            reserve_worstcase,
            image_id: 0,
            images_done: 0,
            text_since_image: 0,
            default_image_pending: false,
            cond_pos: 0,
            steps_done: 0,
            prompt_cursor: 0,
            preempted: false,
            block_hashes: Vec::new(),
            prefix_cached_blocks: 0,
            blocks_cached: false,
            generated_ids: Vec::new(),
            recompute_ids: None,
            mm_cursor: 0,
            mm_acquired: Vec::new(),
            kvlen: 0,
            context_encode_step: 0,
            gen_h: 0,
            gen_w: 0,
            round_tokens: Vec::new(),
            context_round_closing: false,
            worker_image_latent_units: 0,
        }
    }

    pub fn context(&self) -> ContextCursor {
        ContextCursor {
            prompt_cursor: self.prompt_cursor,
            mm_cursor: self.mm_cursor,
            pending_image_step: self.context_encode_step,
            round_closing: self.context_round_closing,
        }
    }

    pub fn und(&self) -> UndCursor {
        UndCursor {
            logical_pos: self.pos,
            physical_kv_len: self.kvlen,
            next_token: self.next_token,
            tokens_emitted: self.n_generated,
            round_tokens: self.round_tokens.clone(),
        }
    }

    pub fn gen_cursor(&self) -> GenCursor {
        GenCursor {
            image_id: self.image_id,
            images_done: self.images_done,
            cond_pos: self.cond_pos,
            steps_done: self.steps_done,
            image_hw: (self.gen_h, self.gen_w),
        }
    }

    pub fn resources(&self) -> ResourceCursor {
        ResourceCursor {
            reserve_worstcase: self.reserve_worstcase,
            worstcase_blocks: self.worstcase_blocks,
            worker_image_latent_units: self.worker_image_latent_units,
        }
    }

    pub fn replay(&self) -> ReplayCursor {
        ReplayCursor {
            preempted: self.preempted,
            prefix_cached_blocks: self.prefix_cached_blocks,
            blocks_cached: self.blocks_cached,
            generated_ids: self.generated_ids.clone(),
            recompute_ids: self.recompute_ids.clone(),
        }
    }

    pub(crate) fn project<'a>(
        &self,
        transitions: impl IntoIterator<Item = &'a PlannedTransition>,
    ) -> CursorProjection {
        let mut projection = CursorProjection {
            phase: self.phase,
            prompt_cursor: self.prompt_cursor,
            logical_pos: self.pos,
            physical_kv_len: self.kvlen,
        };
        for transition in transitions {
            projection.apply(transition);
        }
        projection
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCursor {
    pub prompt_cursor: u32,
    pub mm_cursor: usize,
    pub pending_image_step: u8,
    pub round_closing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndCursor {
    pub logical_pos: u32,
    pub physical_kv_len: u32,
    pub next_token: u32,
    pub tokens_emitted: usize,
    pub round_tokens: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenCursor {
    pub image_id: u32,
    pub images_done: usize,
    pub cond_pos: u32,
    pub steps_done: u16,
    pub image_hw: (u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceCursor {
    pub reserve_worstcase: bool,
    pub worstcase_blocks: usize,
    pub worker_image_latent_units: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCursor {
    pub preempted: bool,
    pub prefix_cached_blocks: usize,
    pub blocks_cached: bool,
    pub generated_ids: Vec<u32>,
    pub recompute_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CursorProjection {
    pub(crate) phase: GenerationPhase,
    pub(crate) prompt_cursor: u32,
    pub(crate) logical_pos: u32,
    pub(crate) physical_kv_len: u32,
}

impl CursorProjection {
    fn apply(&mut self, transition: &PlannedTransition) {
        match transition.delta {
            TransitionDelta::IngestText { end, .. } => {
                self.prompt_cursor = self.prompt_cursor.max(end);
                self.logical_pos = self.logical_pos.max(end);
                self.physical_kv_len = self.physical_kv_len.max(end);
            }
            TransitionDelta::IngestImageStep { position } => {
                self.logical_pos = self.logical_pos.max(position.saturating_add(1));
            }
            TransitionDelta::DecodeUnd {
                position,
                token_count,
            } => {
                let end = position.saturating_add(u32::from(token_count.max(1)));
                self.logical_pos = self.logical_pos.max(end);
                self.physical_kv_len = self.physical_kv_len.max(end);
            }
            TransitionDelta::DenoiseGen { .. }
            | TransitionDelta::CommitGen { .. }
            | TransitionDelta::Feedback { .. }
            | TransitionDelta::Other => {}
        }
    }
}

/// Side-effect-free planner facade for scheduler-local worker transitions.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GenerationPlanner;

impl GenerationPlanner {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) fn plan(&self, op: &ForwardOp) -> PlannedTransition {
        PlannedTransition::from_forward_op(op)
    }
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
}

impl PlannedTransition {
    pub(crate) fn from_forward_op(op: &ForwardOp) -> Self {
        let delta = TransitionDelta::from_forward_op(op);
        let resources = TransitionResources::from_forward_op(op);
        let validation = TransitionValidation::from_forward_op(op);
        Self {
            op: op.clone(),
            op_id: op.op_id,
            kind: op.kind,
            delta,
            resources,
            validation,
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransitionDelta {
    IngestText { start: u32, end: u32 },
    IngestImageStep { position: u32 },
    DecodeUnd { position: u32, token_count: u16 },
    DenoiseGen { start_step: u16, step_count: u16 },
    CommitGen { image_id: Option<u32> },
    Feedback { position: u32 },
    Other,
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
            Self::Other => "other",
        }
    }

    fn from_forward_op(op: &ForwardOp) -> Self {
        match op.kind {
            OpKind::PrefillUnd => Self::IngestText {
                start: op.pos_range.0,
                end: op.pos_range.1,
            },
            OpKind::VaeEncode | OpKind::VitEncode => Self::IngestImageStep {
                position: op.pos_range.0,
            },
            OpKind::DecodeUnd | OpKind::TargetVerifyUnd => Self::DecodeUnd {
                position: op.pos_range.0,
                token_count: op.decode_token_count.unwrap_or(1).max(1),
            },
            OpKind::DenoiseGen => Self::DenoiseGen {
                start_step: op.timestep_idx.unwrap_or(0),
                step_count: op.denoise_step_count.unwrap_or(1).max(1),
            },
            OpKind::CommitGen => Self::CommitGen {
                image_id: op.cond_pos,
            },
            OpKind::CommitWriteback => Self::Feedback {
                position: op.pos_range.0,
            },
            OpKind::Sample | OpKind::EncodeFrame => Self::Other,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransitionResources {
    pub(crate) new_blocks: usize,
    pub(crate) kv_target_tokens: Option<usize>,
    pub(crate) scratch_units: u64,
    pub(crate) latent_units: u64,
    pub(crate) encoder_pins: Vec<u64>,
    pub(crate) replayability_after_apply: Replayability,
}

impl TransitionResources {
    fn from_forward_op(op: &ForwardOp) -> Self {
        Self {
            new_blocks: op.new_block_ids.len(),
            kv_target_tokens: Some(op.pos_range.1 as usize),
            scratch_units: u64::from(op.cfg.as_ref().map_or(0, |cfg| cfg.branch_count)),
            latent_units: u64::from(op.kind == OpKind::DenoiseGen),
            encoder_pins: op.mm_hash.iter().copied().collect(),
            replayability_after_apply: match op.kind {
                OpKind::DenoiseGen | OpKind::CommitGen | OpKind::CommitWriteback => {
                    Replayability::NotReplayable
                }
                _ => Replayability::Replayable,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Replayability {
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
    pub(crate) expected_denoise_step_count: Option<u16>,
    pub(crate) expects_encoder_handle: bool,
    pub(crate) allows_sampled_tokens: bool,
}

impl TransitionValidation {
    fn from_forward_op(op: &ForwardOp) -> Self {
        Self {
            expected_denoise_step_count: op.denoise_step_count,
            expects_encoder_handle: matches!(op.kind, OpKind::VaeEncode | OpKind::VitEncode),
            allows_sampled_tokens: matches!(
                op.kind,
                OpKind::PrefillUnd
                    | OpKind::DecodeUnd
                    | OpKind::TargetVerifyUnd
                    | OpKind::CommitGen
                    | OpKind::CommitWriteback
            ),
        }
    }

    fn validate(&self, kind: OpKind, result: &SeqResult) -> Result<(), TransitionValidationError> {
        if let Some(step_count) = self.expected_denoise_step_count {
            let steps_done = result
                .num_steps_done
                .ok_or(TransitionValidationError::MissingDenoiseStep)?;
            if steps_done < step_count && !result.denoise_done {
                return Err(TransitionValidationError::DenoiseStepRegression {
                    expected_at_least: step_count,
                    actual: steps_done,
                });
            }
        }
        if self.expects_encoder_handle && result.encoder_handle.is_none() {
            return Err(TransitionValidationError::MissingEncoderHandle);
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
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TransitionValidationError {
    OpIdMismatch { expected: u64, actual: Option<u64> },
    MissingDenoiseStep,
    DenoiseStepRegression { expected_at_least: u16, actual: u16 },
    MissingEncoderHandle,
    UnexpectedSampledToken { kind: OpKind },
}

#[cfg(test)]
mod tests {
    use super::*;
    use uniserve_core::{Modality, RequestId};

    #[test]
    fn cursor_exposes_typed_views_without_duplicate_state() {
        let mut cursor = GenerationCursor::new(GenerationPhase::Prefill, 8, true);
        cursor.prompt_cursor = 3;
        cursor.pos = 5;
        cursor.kvlen = 7;
        cursor.image_id = 2;

        assert_eq!(cursor.context().prompt_cursor, 3);
        assert_eq!(cursor.und().logical_pos, 5);
        assert_eq!(cursor.und().physical_kv_len, 7);
        assert_eq!(cursor.gen_cursor().image_id, 2);
        assert_eq!(cursor.resources().worstcase_blocks, 8);
        assert_eq!(cursor.replay().generated_ids, Vec::<u32>::new());
    }

    #[test]
    fn planned_transition_validates_op_id_and_encoder_handle() {
        let op = ForwardOp {
            req_id: RequestId(9),
            kind: OpKind::VitEncode,
            modality: Modality::Und,
            pos_range: (4, 5),
            mm_hash: Some(123),
            op_id: Some(77),
            ..Default::default()
        };
        let transition = PlannedTransition::from_forward_op(&op);

        assert_eq!(transition.op.kind, OpKind::VitEncode,);
        assert_eq!(
            transition.delta,
            TransitionDelta::IngestImageStep { position: 4 }
        );
        assert_eq!(transition.resources.encoder_pins, vec![123]);
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_id: Some(77),
                ..Default::default()
            }),
            Err(TransitionValidationError::MissingEncoderHandle)
        );
        assert_eq!(
            transition.validate_result(&SeqResult {
                req_id: RequestId(9),
                op_id: Some(78),
                encoder_handle: Some(1),
                ..Default::default()
            }),
            Err(TransitionValidationError::OpIdMismatch {
                expected: 77,
                actual: Some(78)
            })
        );
        assert!(
            transition
                .validate_result(&SeqResult {
                    req_id: RequestId(9),
                    op_id: Some(77),
                    encoder_handle: Some(1),
                    ..Default::default()
                })
                .is_ok()
        );
    }

    #[test]
    fn cursor_projection_applies_planned_lifecycle_without_mutating_cursor() {
        let cursor = GenerationCursor::new(GenerationPhase::Prefill, 8, false);
        let prefill = PlannedTransition::from_forward_op(&ForwardOp {
            req_id: RequestId(1),
            kind: OpKind::PrefillUnd,
            modality: Modality::Und,
            pos_range: (0, 3),
            new_block_ids: vec![uniserve_core::BlockId(1), uniserve_core::BlockId(2)],
            ..Default::default()
        });
        let decode = PlannedTransition::from_forward_op(&ForwardOp {
            req_id: RequestId(1),
            kind: OpKind::DecodeUnd,
            modality: Modality::Und,
            pos_range: (3, 4),
            decode_token_count: Some(2),
            ..Default::default()
        });

        let projected = cursor.project([&prefill, &decode]);

        assert_eq!(cursor.prompt_cursor, 0);
        assert_eq!(cursor.pos, 0);
        assert_eq!(projected.prompt_cursor, 3);
        assert_eq!(projected.logical_pos, 5);
    }
}
