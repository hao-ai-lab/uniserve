//! Client-visible event production and per-request output accounting.

use super::*;

fn flush_public_journal(event_tx: &EventTx, journal: &mut VecDeque<GenerationEvent>) -> bool {
    while let Some(event) = journal.pop_front() {
        match event_tx.send(event) {
            Ok(()) => {}
            Err(EventSendError::Full(event)) => {
                journal.push_front(*event);
                return false;
            }
            Err(EventSendError::Closed(_)) => {
                journal.clear();
                return true;
            }
        }
    }
    false
}

pub(super) struct RequestOutput {
    pub(super) event_tx: EventTx,
    pub(super) event_seq: u64,
    pub(super) tokens_sent: usize,
    pub(super) tokens_acked: usize,
    journal: VecDeque<GenerationEvent>,
}

impl RequestOutput {
    pub(super) fn new(event_tx: EventTx) -> Self {
        Self {
            event_tx,
            event_seq: 0,
            tokens_sent: 0,
            tokens_acked: 0,
            journal: VecDeque::new(),
        }
    }

    pub(super) fn is_closed(&self) -> bool {
        self.event_tx.is_closed()
    }

    pub(super) fn available_capacity(&self) -> usize {
        self.event_tx
            .capacity()
            .saturating_add(OUTPUT_JOURNAL_CAPACITY.saturating_sub(self.journal.len()))
    }

    fn flush(&mut self) -> bool {
        flush_public_journal(&self.event_tx, &mut self.journal)
    }

    pub(super) fn enqueue(&mut self, event: GenerationEvent) -> bool {
        if self.flush() {
            return true;
        }
        if self.journal.is_empty() {
            match self.event_tx.send(event) {
                Ok(()) => {
                    self.event_seq = self.event_seq.saturating_add(1);
                    return false;
                }
                Err(EventSendError::Closed(_)) => return true,
                Err(EventSendError::Full(event)) => self.journal.push_back(*event),
            }
        } else {
            self.journal.push_back(event);
        }
        assert!(
            self.journal.len() <= OUTPUT_JOURNAL_CAPACITY,
            "scheduler exceeded the bounded public output journal"
        );
        self.event_seq = self.event_seq.saturating_add(1);
        false
    }
}

struct RetiredOutput {
    event_tx: EventTx,
    journal: VecDeque<GenerationEvent>,
}

#[derive(Default)]
pub(super) struct OutputSender {
    retired: HashMap<RequestId, RetiredOutput>,
}

impl OutputSender {
    pub(super) fn retained_len(&self) -> usize {
        self.retired.len()
    }

    pub(super) fn retire(&mut self, id: RequestId, output: RequestOutput) {
        if !output.journal.is_empty() {
            self.retired.insert(
                id,
                RetiredOutput {
                    event_tx: output.event_tx,
                    journal: output.journal,
                },
            );
        }
    }

    pub(super) fn flush_retired(&mut self) -> bool {
        let mut progressed = false;
        self.retired.retain(|_, output| {
            let before = output.journal.len();
            let closed = flush_public_journal(&output.event_tx, &mut output.journal);
            progressed |= output.journal.len() != before;
            !closed && !output.event_tx.is_closed() && !output.journal.is_empty()
        });
        progressed
    }
}

impl Scheduler {
    pub(super) fn resolve_decode_text(
        &mut self,
        id: RequestId,
        view: SequenceView,
        prefix_versions: &[VersionRef],
    ) {
        self.activate_request_tables(id);
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.cursor.ingest.round_closing)
        {
            let close_token = self
                .running
                .get(&id)
                .map(|state| state.cursor.und.next_token)
                .unwrap_or_default();
            if let Some(state) = self.running.get_mut(&id) {
                state.cursor.ingest.round_closing = false;
            }
            return self.close_context_round(id, close_token);
        }
        let burst_result = view.committed_tokens.len() > 1;
        let tokens = if view.committed_tokens.is_empty() {
            vec![self.ctrl.eos[0]]
        } else {
            view.committed_tokens.clone()
        };
        for (idx, tok) in tokens.iter().copied().enumerate() {
            let is_last = idx + 1 == tokens.len();
            let logprob = is_last.then_some(view.sampled_logprob).flatten();
            let is_round_close = self
                .running
                .get(&id)
                .is_some_and(|st| st.req.policy.trigger.round_close_token_ids().contains(&tok));
            if is_round_close {
                if let Some(st) = self.running.get_mut(&id) {
                    st.cursor.und.tokens_emitted += 1;
                }
                if burst_result {
                    return self.close_context_round(id, tok);
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.cursor.und.next_token = tok;
                    st.cursor.ingest.round_closing = true;
                }
                return;
            }
            let (can_open_gen_branch, images_done, max_images) = {
                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.cursor.und.tokens_emitted += 1;
                (
                    st.can_open_gen_branch(),
                    st.cursor.image_gen.images_done,
                    st.req.image.max_images as usize,
                )
            };
            let direct_trigger = self
                .running
                .get(&id)
                .is_some_and(|st| Self::direct_trigger_matches(st, tok));
            if direct_trigger && can_open_gen_branch && images_done < max_images {
                self.begin_image(id);
                return;
            }
            // Once the image budget is spent the image-start trigger is
            // suppressed host-side: emit a non-trigger token so a bias toward the
            // trigger returns to ordinary text instead of an un-actionable signal.
            let tok = if direct_trigger {
                tok.wrapping_add(1)
            } else {
                tok
            };
            let top_logprobs = is_last.then(|| view.top_logprobs.clone());
            if self.emit_or_finish_und_token(
                id,
                tok,
                logprob,
                top_logprobs,
                is_last,
                prefix_versions.get(idx),
            ) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.cursor.und.next_token = tok;
                st.cursor.phase = Phase::DecodeUnd;
                st.cursor.und.round_tokens.push(tok);
            }
            if can_open_gen_branch
                && images_done < max_images
                && self
                    .running
                    .get(&id)
                    .is_some_and(Self::generated_trigger_matches)
            {
                self.begin_image(id);
                return;
            }
        }
    }

    pub(super) fn resolve(
        &mut self,
        id: RequestId,
        operation: Operation,
        apply: SchedulerApply,
        mut view: SequenceView,
        prefix_versions: Vec<VersionRef>,
    ) {
        let operation_variant = operation.work;
        if !view.prompt_logprobs.is_empty() {
            let positions = std::mem::take(&mut view.prompt_logprobs);
            self.resolve_prompt_logprobs(id, positions);
        }
        if operation_variant == ForwardMode::TokenDecode {
            return self.resolve_decode_text(id, view, &prefix_versions);
        }
        match operation_variant {
            ForwardMode::TokenExtend => {
                self.activate_request_tables(id);
                match &apply.intent {
                    crate::scheduler::generation::TransitionIntent::CloseKv { .. } => return,
                    crate::scheduler::generation::TransitionIntent::IngestImageState {
                        is_final_step,
                        ..
                    } => {
                        if *is_final_step {
                            self.release_transient_products(id);
                        }
                        if *is_final_step
                            && self.running.get(&id).is_some_and(|st| {
                                st.cursor.ingest.mm_cursor >= st.context.images.len()
                                    && st.cursor.ingest.prompt_cursor
                                        >= st.context.prompt_ids.len() as u32
                            })
                        {
                            let bos = self.ctrl.bos;
                            if let Some(st) = self.running.get_mut(&id) {
                                st.cursor.und.next_token = bos;
                                st.cursor.und.round_tokens.clear();
                                st.cursor.phase = Phase::DecodeUnd;
                            }
                        }
                        return;
                    }
                    crate::scheduler::generation::TransitionIntent::FeedbackState {
                        is_final_step,
                        ..
                    } => {
                        if !is_final_step {
                            return;
                        }
                        self.release_transient_products(id);
                        let sample_continuation = self
                            .running
                            .get(&id)
                            .and_then(|st| st.req.policy.feedback.as_ref())
                            .is_some_and(|feedback| feedback.sample_continuation);
                        if let Some(st) = self.running.get_mut(&id) {
                            st.cursor.image_gen.images_done += 1;
                            st.cursor.und.text_since_image = 0;
                            st.cursor.und.round_tokens.clear();
                        }
                        if !sample_continuation {
                            let Some(next_token) = self.feedback_next_token(id) else {
                                return self.finish(id, FinishReason::Error);
                            };
                            if let Some(st) = self.running.get_mut(&id) {
                                st.cursor.und.next_token = next_token;
                                st.cursor.phase = Phase::DecodeUnd;
                            }
                            return;
                        }
                        let Some(tok) = view.committed_tokens.last().copied() else {
                            return self.finish(id, FinishReason::Error);
                        };
                        let (can_open_gen_branch, images_done, max_images) = {
                            let st = self.running.get_mut(&id).unwrap();
                            st.cursor.und.tokens_emitted += 1;
                            (
                                st.can_open_gen_branch(),
                                st.cursor.image_gen.images_done,
                                st.req.image.max_images as usize,
                            )
                        };
                        let direct_trigger = self
                            .running
                            .get(&id)
                            .is_some_and(|state| Self::direct_trigger_matches(state, tok));
                        if direct_trigger && can_open_gen_branch && images_done < max_images {
                            self.begin_image(id);
                            return;
                        }
                        let tok = if direct_trigger {
                            tok.wrapping_add(1)
                        } else {
                            tok
                        };
                        if self.emit_or_finish_und_token(
                            id,
                            tok,
                            view.sampled_logprob,
                            Some(view.top_logprobs.clone()),
                            true,
                            prefix_versions.last(),
                        ) {
                            return;
                        }
                        if let Some(st) = self.running.get_mut(&id) {
                            st.cursor.und.next_token = tok;
                            st.cursor.phase = Phase::DecodeUnd;
                            st.cursor.und.round_tokens.push(tok);
                        }
                        if can_open_gen_branch
                            && images_done < max_images
                            && self
                                .running
                                .get(&id)
                                .is_some_and(Self::generated_trigger_matches)
                        {
                            self.begin_image(id);
                        }
                        return;
                    }
                    _ => {}
                }
                // chunked prefill: a prefill op may only have consumed part of
                // the prompt; if so, advance the cursor and stay in Prefill.
                let (cursor, prompt_len) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.cursor.ingest.prompt_cursor as usize,
                        st.effective_prompt().len(),
                    )
                };
                if cursor < prompt_len {
                    return;
                }
                if self.running.get(&id).is_some_and(|st| {
                    st.cursor.ingest.mm_cursor < st.context.images.len()
                        || st.cursor.ingest.prompt_cursor < st.context.prompt_ids.len() as u32
                }) {
                    return;
                }
                let (starts_gen_after_context, can_open_gen_branch) = {
                    let st = self.running.get_mut(&id).unwrap();
                    st.cursor.und.tokens_emitted += 1;
                    (st.starts_gen_after_context(), st.can_open_gen_branch())
                };
                if self.running.get(&id).is_some_and(|state| {
                    state.req.sampling.prompt_logprobs_requested()
                        && state.cursor.ingest.prompt_logprobs_emitted
                            != state.context.prompt_ids.len().saturating_sub(1)
                }) {
                    tracing::error!(
                        request_id = id.0,
                        "prompt logprob scoring ended before every prompt position was resolved"
                    );
                    return self.finish(id, FinishReason::Error);
                }
                // the prompt is fully prefilled now — publish its full
                // blocks to the prefix cache for later requests to reuse.
                if let Some(st) = self.running.get_mut(&id) {
                    let kv = self.kv_budget.cache();
                    cache_prompt_blocks(&kv.coordinator, st, &kv.block_pool);
                }
                // A description-lowered prefix may already end at a branch trigger.
                // Treat that boundary exactly like a sampled trigger.
                if self.prefilled_gen_trigger(id) {
                    self.begin_image(id);
                    return;
                }
                // Immediate Gen-only profiles skip Und decode after context prep.
                if starts_gen_after_context {
                    self.begin_image(id);
                    return;
                }
                let tok = view
                    .committed_tokens
                    .last()
                    .copied()
                    .unwrap_or(self.ctrl.eos[0]);
                let logprob = view.sampled_logprob;
                let (images_done, max_images) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.cursor.image_gen.images_done,
                        st.req.image.max_images as usize,
                    )
                };
                // the model requested an image inline; honor it while the
                // request is still under its image budget.
                let direct_trigger = self
                    .running
                    .get(&id)
                    .is_some_and(|st| Self::direct_trigger_matches(st, tok));
                if direct_trigger && can_open_gen_branch && images_done < max_images {
                    self.begin_image(id);
                    return;
                }
                if self.emit_or_finish_und_token(
                    id,
                    tok,
                    logprob,
                    Some(view.top_logprobs.clone()),
                    true,
                    prefix_versions.last(),
                ) {
                    return;
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.cursor.und.next_token = tok;
                    st.cursor.phase = Phase::DecodeUnd;
                }
                if can_open_gen_branch
                    && images_done < max_images
                    && self
                        .running
                        .get(&id)
                        .is_some_and(Self::generated_trigger_matches)
                {
                    self.begin_image(id);
                }
            }
            ForwardMode::GenFlow => {
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = match apply.intent {
                        crate::scheduler::generation::TransitionIntent::DenoiseGen {
                            start_step,
                            ..
                        } => start_step,
                        _ => st.cursor.image_gen.steps_done,
                    };
                    (
                        st.cursor.image_gen.image_id,
                        st.req.image.height,
                        st.req.image.width,
                        st.req.image.steps,
                        prev,
                    )
                };
                let sd = self
                    .running
                    .get(&id)
                    .map(|s| s.cursor.image_gen.steps_done)
                    .unwrap_or(0);
                if prev_sd == 0 && sd >= 1 {
                    self.emit(
                        id,
                        GenerationEvent::ImageBegin {
                            image_id,
                            height: h,
                            width: w,
                            steps,
                        },
                    );
                }
                for step in prev_sd.saturating_add(1)..=sd {
                    self.emit(id, GenerationEvent::ImageStep { image_id, step });
                }
                // The commit phase is entered host-side once the committed step
                // count reaches `image.steps` (see the `Phase::DenoiseGen`
                // planner); a worker completion flag does not drive termination.
            }
            ForwardMode::GenDecode => {}
            ForwardMode::Materialize => {
                let image_id = self
                    .running
                    .get(&id)
                    .map_or(0, |st| st.cursor.image_gen.image_id);
                self.emit(id, GenerationEvent::ImageCommit { image_id });
                let image = view.image_png.clone();
                if let Some(image_b64) = image.clone() {
                    let Some(event) = image_done_event(image_id, image_b64) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let root = self.fixed_version(id);
                    self.emit_visible(id, event, root.as_ref(), PublicModality::Image);
                }
                self.activate_request_tables(id);
                let (continues_after_gen_commit, feedback_source) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.continues_after_gen_commit(),
                        st.req
                            .policy
                            .feedback
                            .as_ref()
                            .map(|feedback| feedback.source.clone()),
                    )
                };
                if continues_after_gen_commit {
                    let Some(feedback_source) = feedback_source else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let source_product = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            product.storage_class == uniserve_worker_ipc::StorageClass::LatentArena
                                && product.kind == uniserve_worker_ipc::ProductKind::Artifact
                        })
                        .cloned();
                    if feedback_source == uniserve_core::FeedbackSource::DeviceProduct
                        && source_product.is_none()
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    if feedback_source == uniserve_core::FeedbackSource::ArtifactProduct
                        && image.is_none()
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    if let Some(st) = self.running.get_mut(&id) {
                        st.cursor.feedback.image_b64 = image;
                        st.cursor.feedback.ingest_step = 0;
                        st.cursor.feedback.source_product = source_product;
                        st.cursor.feedback.encoded_product = None;
                        st.cursor.phase = Phase::FeedbackEncode;
                        if let Some(product) = st.cursor.feedback.source_product.clone() {
                            st.cursor.ingest.transient_encoder_products.push(product);
                        }
                    }
                } else {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.cursor.image_gen.images_done += 1;
                        st.cursor.und.text_since_image = 0;
                    }
                    self.finish(id, FinishReason::ImageDone);
                }
            }
            ForwardMode::EncodeVision | ForwardMode::EncodeLatent => match &apply.intent {
                crate::scheduler::generation::TransitionIntent::EncodeImageStep {
                    encoder_cache_key,
                    ..
                } => {
                    let Some(feature) = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            matches!(
                                product.kind,
                                uniserve_worker_ipc::ProductKind::VisionFeature
                                    | uniserve_worker_ipc::ProductKind::LatentFeature
                            )
                        })
                        .cloned()
                    else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let result_handle = u64::from(feature.generation);
                    if result_handle == 0
                        || view.encode_generation.map(u64::from) != Some(result_handle)
                    {
                        return self.finish(id, FinishReason::Error);
                    }
                    let mut free_products = Vec::new();
                    let selected_product = if let Some(cache_key) = encoder_cache_key {
                        if let Some(freed) = self
                            .kv_budget
                            .encoder_cache
                            .insert(*cache_key, feature.clone())
                        {
                            free_products.push(freed);
                        }
                        let Some(product) = self.kv_budget.encoder_cache.acquire(*cache_key) else {
                            return self.finish(id, FinishReason::Error);
                        };
                        if let Some(st) = self.running.get_mut(&id) {
                            st.cursor
                                .ingest
                                .acquired_encoder_pins
                                .push(EncoderCachePin {
                                    key: *cache_key,
                                    product: product.clone(),
                                });
                        }
                        product
                    } else if let Some(st) = self.running.get_mut(&id) {
                        st.cursor
                            .ingest
                            .transient_encoder_products
                            .push(feature.clone());
                        feature.clone()
                    } else {
                        feature.clone()
                    };
                    if let Some(st) = self.running.get_mut(&id) {
                        st.cursor.ingest.encoded_product = Some(selected_product);
                        st.cursor.phase = Phase::IngestState;
                    }
                    if !free_products.is_empty() {
                        self.release_products(free_products);
                    }
                }
                crate::scheduler::generation::TransitionIntent::EncodeFeedbackStep { .. } => {
                    let Some(feature) = operation
                        .outputs
                        .iter()
                        .find(|product| {
                            matches!(
                                product.kind,
                                uniserve_worker_ipc::ProductKind::VisionFeature
                                    | uniserve_worker_ipc::ProductKind::LatentFeature
                            )
                        })
                        .cloned()
                    else {
                        return self.finish(id, FinishReason::Error);
                    };
                    let handle = u64::from(feature.generation);
                    if handle == 0 || view.encode_generation.map(u64::from) != Some(handle) {
                        return self.finish(id, FinishReason::Error);
                    }
                    if let Some(st) = self.running.get_mut(&id) {
                        st.cursor
                            .ingest
                            .transient_encoder_products
                            .push(feature.clone());
                        st.cursor.feedback.encoded_product = Some(feature);
                        st.cursor.phase = Phase::FeedbackState;
                    }
                }
                _ => self.finish(id, FinishReason::Error),
            },
            ForwardMode::TokenDecode
            | ForwardMode::TokenVerify
            | ForwardMode::Draft
            | ForwardMode::GenTransition
            | ForwardMode::TransferProduct
            | ForwardMode::TransferKvPublish
            | ForwardMode::TransferKvInstall => {}
        }
    }

    pub(super) fn resolve_prompt_logprobs(
        &mut self,
        id: RequestId,
        positions: Vec<Vec<RankedToken>>,
    ) {
        let total = positions.len();
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let processed_before = state.cursor.ingest.prompt_logprobs_processed;
        let emitted_before = state.cursor.ingest.prompt_logprobs_emitted;
        let expected_total = state.context.prompt_ids.len().saturating_sub(1);
        state.cursor.ingest.prompt_logprobs_processed = processed_before.saturating_add(total);
        let skip = emitted_before.saturating_sub(processed_before);
        let remaining = expected_total.saturating_sub(emitted_before);
        let selected: Vec<_> = positions.into_iter().skip(skip).take(remaining).collect();
        state.cursor.ingest.prompt_logprobs_emitted = emitted_before.saturating_add(selected.len());
        if selected.is_empty() {
            return;
        }
        self.emit(
            id,
            GenerationEvent::PromptLogprobs {
                positions: selected
                    .into_iter()
                    .map(|entries| PositionLogprobs {
                        entries: ranked_logprobs(entries),
                    })
                    .collect(),
            },
        );
    }

    pub(super) fn release_transient_products(&mut self, id: RequestId) {
        let products = self
            .running
            .get_mut(&id)
            .map(|state| std::mem::take(&mut state.cursor.ingest.transient_encoder_products))
            .unwrap_or_default();
        self.release_products(products);
    }

    pub(super) fn record_gen_trigger_for_replay(&mut self, id: RequestId) {
        let Some(start) = self
            .running
            .get(&id)
            .and_then(|st| st.req.policy.trigger.direct_token())
        else {
            return;
        };
        if let Some(st) = self.running.get_mut(&id)
            && st.continues_after_gen_commit()
            && st.cursor.und.tokens_emitted > st.cursor.replay.generated_ids.len()
            && st.cursor.replay.generated_ids.last().copied() != Some(start)
        {
            st.cursor.replay.generated_ids.push(start);
        }
    }

    pub(super) fn promote_gen_branch_reservation(&mut self, id: RequestId) -> bool {
        let Some((required_blocks, reserves_envelope)) = self.running.get(&id).map(|st| {
            (
                st.cursor.resources.worstcase_blocks,
                st.cursor.resources.reserve_worstcase,
            )
        }) else {
            return false;
        };
        let allocated_blocks = self
            .running
            .get(&id)
            .and_then(|state| state.block_tables.first())
            .map_or(0, BlockTable::len);
        if !reserves_envelope || allocated_blocks < required_blocks {
            self.trace_record(json!({
                "event": "gen_branch_reservation_rejected",
                "at_s": now(),
                "request_id": id.0,
                "required_blocks": required_blocks,
                "allocated_blocks": allocated_blocks,
            }));
            self.finish(id, FinishReason::Error);
            return false;
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.cursor.image_gen.branch_pending = false;
        }
        self.trace_record(json!({
            "event": "gen_branch_capacity_ready",
            "at_s": now(),
            "request_id": id.0,
            "required_blocks": required_blocks,
            "allocated_blocks": allocated_blocks,
            "free_blocks": self.kv_budget.free_blocks(),
            "reserved_blocks": self.kv_budget.reserved_blocks,
        }));
        true
    }

    pub(super) fn begin_image(&mut self, id: RequestId) {
        self.record_gen_trigger_for_replay(id);
        if let Some(st) = self.running.get_mut(&id) {
            st.cursor.image_gen.cond_pos = st.cursor.und.logical_pos;
            st.cursor.image_gen.steps_done = 0;
            st.cursor.image_gen.image_id += 1;
            st.cursor.und.text_since_image = 0;
            st.cursor.phase = Phase::CloseKv;
            st.cursor.image_gen.branch_pending = true;
        }
        self.promote_gen_branch_reservation(id);
    }

    pub(super) fn flush_output_journals(&mut self) -> bool {
        let mut progressed = false;
        for state in self.running.values_mut() {
            if state.output.is_closed() {
                state.terminal_intent = TerminalIntent::Cancel;
                continue;
            }
            let before = state.output.journal.len();
            if state.output.flush() {
                state.terminal_intent = TerminalIntent::Cancel;
            }
            progressed |= state.output.journal.len() != before;
        }
        progressed |= self.output.flush_retired();
        progressed
    }

    pub(super) fn emit(&mut self, id: RequestId, ev: GenerationEvent) {
        if let Some(st) = self.running.get_mut(&id) {
            if st.output.enqueue(ev) {
                st.terminal_intent = TerminalIntent::Cancel;
            }
        }
    }

    pub(super) fn emit_visible(
        &mut self,
        id: RequestId,
        mut event: GenerationEvent,
        root: Option<&VersionRef>,
        modality: PublicModality,
    ) -> bool {
        let root = root.cloned().or_else(|| self.fixed_version(id));
        let Some(VersionRef {
            producer_op_id,
            point: Point::Fixed { point_index },
            ..
        }) = root
        else {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return false;
        };
        let event_seq = self
            .running
            .get(&id)
            .map_or(1, |state| state.output.event_seq.saturating_add(1));
        let commit = PublicCommit {
            event_seq,
            modality,
            committed_at: uniserve_core::now_monotonic_secs(),
            semantic_root: SemanticRoot {
                producer_op_id,
                point_index,
            },
        };
        match &mut event {
            GenerationEvent::TextToken { public_commit, .. }
            | GenerationEvent::ImageDone { public_commit, .. } => *public_commit = Some(commit),
            _ => {
                self.finish_after_inflight(id, FinishReason::Error, None);
                return false;
            }
        }
        let before = self
            .running
            .get(&id)
            .map_or(0, |state| state.output.event_seq);
        self.emit(id, event);
        let published = self
            .running
            .get(&id)
            .is_some_and(|state| state.output.event_seq > before);
        published
    }

    /// Emit a text token and retain its semantic history.
    pub(super) fn emit_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        root: Option<&VersionRef>,
    ) -> bool {
        let action = if let Some(st) = self.running.get_mut(&id) {
            st.cursor.replay.generated_ids.push(tok);
            st.cursor.und.text_since_image = st.cursor.und.text_since_image.saturating_add(1);
            st.req.behavior.und_tokens
        } else {
            return false;
        };
        match action {
            uniserve_core::UndTokenAction::Emit => {
                let published = self.emit_visible(
                    id,
                    GenerationEvent::TextToken {
                        id: tok,
                        logprob,
                        public_commit: None,
                    },
                    root,
                    PublicModality::Text,
                );
                let first_token = published
                    && self
                        .running
                        .get(&id)
                        .is_some_and(|state| state.output.tokens_sent == 0);
                if published && let Some(state) = self.running.get_mut(&id) {
                    state.output.tokens_sent = state.output.tokens_sent.saturating_add(1);
                }
                if first_token && self.trace_enabled() {
                    self.trace_record(json!({
                        "event": "first_public_token",
                        "at_s": now(),
                        "request_id": id.0,
                    }));
                }
                true
            }
            uniserve_core::UndTokenAction::KeepInternal => false,
            uniserve_core::UndTokenAction::Reject => {
                self.finish(id, FinishReason::Error);
                false
            }
        }
    }

    pub(super) fn emit_sampled_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<RankedToken>>,
        root: Option<&VersionRef>,
    ) {
        if self.emit_text(id, tok, logprob, root)
            && let Some(top_logprobs) = top_logprobs.filter(|entries| !entries.is_empty())
        {
            self.emit(
                id,
                GenerationEvent::TokenLogprobs {
                    id: tok,
                    candidates: ranked_logprobs(top_logprobs),
                },
            );
        }
    }

    pub(super) fn emit_terminal_stop_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<RankedToken>>,
        root: Option<&VersionRef>,
    ) {
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.req.policy.termination.emit_stop_token)
        {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
        }
    }

    pub(super) fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<RankedToken>>,
        sampled: bool,
        root: Option<&VersionRef>,
    ) -> bool {
        let Some(state) = self.running.get(&id) else {
            return true;
        };
        // `tokens_emitted` already counts the token being resolved, so the floor
        // is cleared only once at least `min_tokens` tokens have been emitted.
        let generated = state.cursor.und.tokens_emitted;
        let under_floor = generated <= state.req.sampling.min_tokens;
        let stop_hit = state.req.policy.termination.stop_finishes
            && !under_floor
            && state.req.stop_token_ids.contains(&token_id);
        let eos_hit = state.req.policy.termination.eos_finishes
            && self.ctrl.eos.contains(&token_id)
            && !state.req.sampling.ignore_eos
            && !under_floor;
        let max_hit = generated >= state.req.max_und_tokens;

        if stop_hit {
            self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs, root);
            self.finish_after_inflight(
                id,
                FinishReason::Stop,
                Some(uniserve_core::StopReason::Token(token_id)),
            );
            return true;
        }
        if eos_hit || max_hit {
            if !self.ctrl.eos.contains(&token_id) {
                if sampled {
                    self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
                } else {
                    self.emit_text(id, token_id, logprob, root);
                }
            }
            self.finish_after_inflight(
                id,
                if max_hit {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Eos
                },
                None,
            );
            return true;
        }
        if sampled {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs, root);
        } else {
            self.emit_text(id, token_id, logprob, root);
        }
        !self.running.contains_key(&id)
    }

    pub(super) fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    pub(super) fn finish_after_inflight(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<uniserve_core::StopReason>,
    ) {
        let semantic_pending = !matches!(reason, FinishReason::Error)
            && self
                .running
                .get(&id)
                .is_some_and(|state| !state.pending_commits.is_empty());
        if !self.inflight.contains(id) && !semantic_pending {
            self.finish_with(id, reason, stop_reason);
            return;
        }
        if !self.inflight.finishes.contains_key(&id) || matches!(reason, FinishReason::Error) {
            self.inflight.finishes.insert(
                id,
                PendingFinish {
                    reason,
                    stop_reason,
                },
            );
        }
    }

    pub(super) fn finish_pending_if_idle(&mut self, id: RequestId) {
        if self.inflight.contains(id)
            || self
                .running
                .get(&id)
                .is_some_and(|state| !state.pending_commits.is_empty())
        {
            return;
        }
        if let Some(pending) = self.inflight.finishes.remove(&id) {
            self.finish_with(id, pending.reason, pending.stop_reason);
        }
    }

    pub(super) fn finish_with(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<uniserve_core::StopReason>,
    ) {
        self.inflight.finishes.remove(&id);
        if reason == FinishReason::Error
            && let Some(state) = self.running.get(&id)
        {
            tracing::error!(
                request_id = id.0,
                phase = ?state.cursor.phase,
                generated_tokens = state.cursor.und.tokens_emitted,
                images_done = state.cursor.image_gen.images_done,
                image_id = state.cursor.image_gen.image_id,
                denoise_steps_done = state.cursor.image_gen.steps_done,
                logical_position = state.cursor.und.logical_pos,
                physical_kv_len = state.cursor.und.physical_kv_len,
                "scheduler request terminated with an internal error"
            );
        }
        let mut awaits_close = false;
        let mut request_pool_idx = None;
        let mut flow_pool_idx = None;
        if let Some(mut st) = self.running.remove(&id) {
            request_pool_idx = Some(st.request_pool_idx);
            flow_pool_idx = st
                .flow_prefix
                .as_ref()
                .map(|prefix| prefix.request_pool_idx);
            if st.cursor.resources.worker_registered {
                let request_key = RequestKey::new(self.authority_id, id, st.epoch);
                let cutoff = st.cancel_cutoff.clone().unwrap_or_else(|| VersionRef {
                    request_key,
                    producer_op_id: OpId(st.committed_producer_op_id),
                    point: Point::Fixed {
                        point_index: st.committed_version as u32,
                    },
                });
                st.control_seq = st.control_seq.saturating_add(1);
                self.pending_controls.push_back(Control::Close {
                    request_key,
                    control_seq: st.control_seq,
                    cutoff,
                    reason: close_reason(&reason),
                });
                self.retiring_sessions.insert(
                    id,
                    RetiringSession {
                        request_key,
                        request_pool_idx: st.request_pool_idx,
                        _block_tables: std::mem::take(&mut st.block_tables),
                        flow_prefix: st.flow_prefix.take(),
                    },
                );
                awaits_close = true;
            }
            self.order.retain(|request| *request != id);
            self.kv_budget.reserved_encoder_entries = self
                .kv_budget
                .reserved_encoder_entries
                .saturating_sub(st.req.resources.encoder_cache_keys.len());
            if st.cursor.resources.reserve_worstcase {
                self.kv_budget.reserved_blocks = self
                    .kv_budget
                    .reserved_blocks
                    .saturating_sub(st.cursor.resources.worstcase_blocks);
            }
            let mut free_encoder_products =
                std::mem::take(&mut st.cursor.ingest.transient_encoder_products);
            for pin in &st.cursor.ingest.acquired_encoder_pins {
                if let Some(product) = self.kv_budget.encoder_cache.release(pin.key, &pin.product) {
                    free_encoder_products.push(product);
                }
            }
            self.release_products(free_encoder_products);
            self.trace_request_finished(super::control::FinishedTrace {
                id,
                reason: &reason,
                stop_reason: stop_reason.as_ref(),
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: st.cursor.und.tokens_emitted,
                images: st.cursor.image_gen.images_done,
                queue: "running",
            });
            let terminal = GenerationEvent::Finished {
                reason,
                stop_reason,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: st.cursor.und.tokens_emitted,
                images: st.cursor.image_gen.images_done,
            };
            let closed = st.output.enqueue(terminal);
            if !closed {
                self.output.retire(id, st.output);
            }
        }
        if !awaits_close {
            self.kv_budget.latent_pages.release(id);
            if let Some(index) = flow_pool_idx {
                let _ = self.kv_budget.request_slots.release(index);
            }
            if let Some(index) = request_pool_idx
                && let Err(error) = self.kv_budget.request_slots.release(index)
            {
                tracing::error!(
                    request_id = id.0,
                    request_pool_idx = index,
                    error,
                    "failed to release scheduler request slot"
                );
                self.fatal = true;
            }
        }
    }
}

fn cache_prompt_blocks(coordinator: &KvCacheCoordinator, state: &mut ReqState, pool: &BlockPool) {
    if state.cursor.replay.blocks_cached {
        return;
    }
    let prompt = state.effective_prompt().to_vec();
    if coordinator.cache_prefix(
        pool,
        &state.block_tables,
        &prompt,
        &state.cursor.replay.block_hashes,
        state.req.cache.write,
    ) {
        state.cursor.replay.blocks_cached = true;
    }
}
