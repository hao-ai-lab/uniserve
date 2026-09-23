//! Client-visible event publication and per-request output accounting.
//!
//! Events accumulate in request-local journals when the bounded consumer channel
//! is full, preserving order without blocking the engine owner thread.

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

/// Flushes journaled public events into the output channel.
fn flush_public_journal(event_tx: &EventTx, journal: &mut VecDeque<EngineCoreOutput>) -> bool {
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

/// EngineCoreOutput journal and usage counters for one request.
pub(super) struct RequestOutput {
    pub(super) events: EventJournal,
    pub(super) tokens_sent: usize,
    pub(super) tokens_acked: usize,
    /// Prompt score positions already consumed and delivered to the caller.
    pub(super) prompt_logprobs_processed: usize,
    pub(super) prompt_logprobs_emitted: usize,
    /// Ends of token batches awaiting a stop-string decoder decision.
    pub(super) decoder_boundaries: VecDeque<usize>,
}

impl RequestOutput {
    /// Creates an output journal with the requested capacity.
    pub(super) fn new(event_tx: EventTx) -> Self {
        Self {
            events: EventJournal::new(event_tx),
            tokens_sent: 0,
            tokens_acked: 0,
            prompt_logprobs_processed: 0,
            prompt_logprobs_emitted: 0,
            decoder_boundaries: VecDeque::new(),
        }
    }
}

/// Ordered publication shared by token and media requests. Progress can be
/// coalesced while terminal events retain their place until the caller reads.
pub(super) struct EventJournal {
    pub(super) event_tx: EventTx,
    journal: VecDeque<EngineCoreOutput>,
}

impl EventJournal {
    pub(super) fn new(event_tx: EventTx) -> Self {
        Self {
            event_tx,
            journal: VecDeque::new(),
        }
    }

    /// Returns whether the journal is closed.
    pub(super) fn is_closed(&self) -> bool {
        self.event_tx.is_closed()
    }

    /// Returns the number of events the journal can accept.
    pub(super) fn available_capacity(&self) -> usize {
        self.event_tx
            .capacity()
            .saturating_add(OUTPUT_JOURNAL_CAPACITY.saturating_sub(self.journal.len()))
    }

    /// Flushes pending journal entries.
    fn flush(&mut self) -> bool {
        flush_public_journal(&self.event_tx, &mut self.journal)
    }

    /// Delivers an event or journals it in order when the public channel is full.
    pub(super) fn enqueue(&mut self, event: EngineCoreOutput) -> bool {
        if self.flush() {
            return false;
        }
        if matches!(event, EngineCoreOutput::MediaProgress { .. })
            && let Some(last @ EngineCoreOutput::MediaProgress { .. }) = self.journal.back_mut()
        {
            *last = event;
            return true;
        }
        if self.journal.is_empty() {
            match self.event_tx.send(event) {
                Ok(()) => {
                    return true;
                }
                Err(EventSendError::Closed(_)) => return false,
                Err(EventSendError::Full(event)) => self.journal.push_back(*event),
            }
        } else {
            self.journal.push_back(event);
        }
        assert!(
            self.journal.len() <= OUTPUT_JOURNAL_CAPACITY,
            "scheduler exceeded the bounded public output journal"
        );
        true
    }
}

#[derive(Default)]
/// Non-blocking event publisher with request-local ordered buffering.
pub(crate) struct OutputSender {
    retired: HashMap<RequestId, EventJournal>,
}

impl OutputSender {
    /// Returns the number of retired requests still awaiting output delivery.
    pub(super) fn retained_len(&self) -> usize {
        self.retired.len()
    }

    /// Retains pending events after computational state has retired.
    pub(super) fn retire(&mut self, id: RequestId, output: EventJournal) {
        if !output.journal.is_empty() {
            self.retired.insert(id, output);
        }
    }

    /// Flushes journal entries that are safe to retire.
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
    /// Publishes committed autoregressive tokens and advances text-generation state.
    pub(super) fn resolve_decode_text(
        &mut self,
        id: RequestId,
        mut record: uniserve_worker_ipc::RequestOutput,
    ) {
        self.activate_request_tables(id);
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.round_closing)
        {
            let close_token = self
                .running
                .get(&id)
                .map(|state| state.next_token)
                .unwrap_or_default();
            if let Some(state) = self.running.get_mut(&id) {
                state.round_closing = false;
            }
            return self.close_context_round(id, close_token);
        }
        let burst_result = record.committed_tokens.len() > 1;
        let tokens = if record.committed_tokens.is_empty() {
            vec![self.ctrl.eos[0]]
        } else {
            std::mem::take(&mut record.committed_tokens)
        };
        for (idx, tok) in tokens.iter().copied().enumerate() {
            let is_last = idx + 1 == tokens.len();
            let logprob = is_last.then_some(record.sampled_logprob).flatten();
            let is_round_close = self.running.get(&id).is_some_and(|st| {
                st.req
                    .image_generation
                    .trigger
                    .round_close_token_ids()
                    .contains(&tok)
            });
            if is_round_close {
                if let Some(st) = self.running.get_mut(&id) {
                    st.num_generated_tokens += 1;
                }
                if burst_result {
                    return self.close_context_round(id, tok);
                }
                if let Some(st) = self.running.get_mut(&id) {
                    st.next_token = tok;
                    st.round_closing = true;
                }
                return;
            }
            let (can_open_gen_branch, images_done, max_images) = {
                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.num_generated_tokens += 1;
                (
                    st.can_open_gen_branch(),
                    st.num_generated_images,
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
            let top_logprobs = is_last.then(|| std::mem::take(&mut record.top_logprobs));
            if self.emit_or_finish_und_token(id, tok, logprob, top_logprobs, is_last) {
                return;
            }
            if let Some(st) = self.running.get_mut(&id) {
                st.next_token = tok;
                st.phase = Phase::DecodeUnd;
                st.round_token_ids.push(tok);
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

    /// Applies one completed call to its request and emits observable output.
    ///
    /// Each phase applies its accepted progress so a worker completion cannot advance
    /// a request through an unrelated generation stage.
    pub(super) fn resolve(
        &mut self,
        id: RequestId,
        call: Call,
        mut record: uniserve_worker_ipc::RequestOutput,
        media: Option<&SharedMedia>,
    ) {
        let call_variant = call.code;

        // Prompt scores share the completion but precede its phase transition.
        if !record.prompt_logprobs.is_empty() {
            let positions = std::mem::take(&mut record.prompt_logprobs);
            self.resolve_prompt_logprobs(id, positions);
        }

        if call_variant == CallKind::Forward(ForwardMode::Decode) {
            return self.resolve_decode_text(id, record);
        }

        match call_variant {
            CallKind::Forward(ForwardMode::Prefill) => {
                self.activate_request_tables(id);

                // State-ingest and feedback calls use autoregressive extension
                // transport while retaining their own completion transitions.
                match (
                    consumes_image_features(&call),
                    is_feedback_computation(&call),
                ) {
                    (false, _) if !is_prompt_extend(&call) => return,
                    (true, false) => {
                        let is_final_step = self
                            .running
                            .get(&id)
                            .is_some_and(|state| state.image_encoder_index == 0);
                        if is_final_step {
                            self.free_transient_products(id);
                        }

                        if is_final_step
                            && self.running.get(&id).is_some_and(|st| {
                                st.num_ingested_images >= st.req.multimodal_inputs.images.len()
                                    && st.num_computed_prompt_tokens
                                        >= st.req.prompt_token_ids.len() as u32
                            })
                        {
                            let bos = self.ctrl.bos;
                            if let Some(st) = self.running.get_mut(&id) {
                                st.next_token = bos;
                                st.round_token_ids.clear();
                                st.phase = Phase::DecodeUnd;
                            }
                        }
                        return;
                    }
                    (true, true) => {
                        let is_final_step = self
                            .running
                            .get(&id)
                            .is_some_and(|state| state.feedback_encoder_index == 0);
                        if !is_final_step {
                            return;
                        }

                        self.free_transient_products(id);
                        let sample_continuation = self
                            .running
                            .get(&id)
                            .is_some_and(|st| st.req.image_generation.sample_feedback_continuation);

                        if let Some(st) = self.running.get_mut(&id) {
                            st.num_generated_images += 1;
                            st.text_tokens_since_image = 0;
                            st.round_token_ids.clear();
                        }

                        if !sample_continuation {
                            let Some(next_token) = self.feedback_next_token(id) else {
                                return self.finish(id, FinishReason::Error);
                            };
                            if let Some(st) = self.running.get_mut(&id) {
                                st.next_token = next_token;
                                st.phase = Phase::DecodeUnd;
                            }
                            return;
                        }

                        let Some(tok) = record.committed_tokens.last().copied() else {
                            return self.finish(id, FinishReason::Error);
                        };

                        let (can_open_gen_branch, images_done, max_images) = {
                            let st = self.running.get_mut(&id).unwrap();
                            st.num_generated_tokens += 1;
                            (
                                st.can_open_gen_branch(),
                                st.num_generated_images,
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

                        if self.emit_or_finish_und_token(
                            id,
                            tok,
                            record.sampled_logprob,
                            Some(std::mem::take(&mut record.top_logprobs)),
                            true,
                        ) {
                            return;
                        }

                        if let Some(st) = self.running.get_mut(&id) {
                            st.next_token = tok;
                            st.phase = Phase::DecodeUnd;
                            st.round_token_ids.push(tok);
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

                // Partial prefill remains in ingest until every text and multimodal
                // position has been consumed.
                let (cursor, prompt_len) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.num_computed_prompt_tokens as usize,
                        st.req.prompt_token_ids.len(),
                    )
                };
                if cursor < prompt_len {
                    return;
                }

                if self.running.get(&id).is_some_and(|st| {
                    st.num_ingested_images < st.req.multimodal_inputs.images.len()
                        || st.num_computed_prompt_tokens < st.req.prompt_token_ids.len() as u32
                }) {
                    return;
                }

                let (starts_gen_after_context, can_open_gen_branch) = {
                    let st = self.running.get_mut(&id).unwrap();
                    st.num_generated_tokens += 1;
                    (st.starts_gen_after_context(), st.can_open_gen_branch())
                };

                if self.running.get(&id).is_some_and(|state| {
                    state.req.sampling.prompt_logprobs_requested()
                        && state.output.prompt_logprobs_emitted
                            != state.req.prompt_token_ids.len().saturating_sub(1)
                }) {
                    tracing::error!(
                        request_id = id.0,
                        "prompt logprob scoring ended before every prompt position was resolved"
                    );
                    return self.finish(id, FinishReason::Error);
                }

                // Only a complete prompt publishes reusable KV blocks.
                let state = self
                    .running
                    .get(&id)
                    .expect("prefill request remains active");
                let key = RequestKey::new(self.engine_id, id, state.request_epoch);
                let worker = self
                    .placement
                    .affinity
                    .get(&(key, DEFAULT_COMPONENT.to_owned()))
                    .expect("prefill has a bound model component");
                let source = Arc::new(
                    self.executor
                        .info()
                        .workers
                        .iter()
                        .find(|(id, _)| id == worker)
                        .expect("prefill Worker remains loaded")
                        .1
                        .endpoint
                        .clone(),
                );
                if let Some(st) = self.running.get_mut(&id) {
                    let kv = self
                        .storage
                        .cache
                        .as_ref()
                        .expect("generation has a KV cache");
                    cache_prompt_blocks(&kv.coordinator, st, &kv.block_pool, &source);
                }

                // A description-lowered prefix may already end at a branch trigger.
                if self.prefilled_gen_trigger(id) {
                    self.begin_image(id);
                    return;
                }

                // Immediate Gen-only profiles skip Und decode after context prep.
                if starts_gen_after_context {
                    self.begin_image(id);
                    return;
                }

                let tok = record
                    .committed_tokens
                    .last()
                    .copied()
                    .unwrap_or(self.ctrl.eos[0]);
                let logprob = record.sampled_logprob;

                let (images_done, max_images) = {
                    let st = self.running.get(&id).unwrap();
                    (st.num_generated_images, st.req.image.max_images as usize)
                };

                // An inline image trigger consumes the remaining branch budget.
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
                    Some(std::mem::take(&mut record.top_logprobs)),
                    true,
                ) {
                    return;
                }

                if let Some(st) = self.running.get_mut(&id) {
                    st.next_token = tok;
                    st.phase = Phase::DecodeUnd;
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
            CallKind::Media(MediaCall::Denoising) => {
                // Publish every newly committed step exactly once, including steps
                // coalesced into a single worker completion.
                let (image_id, h, w, steps, prev_sd) = {
                    let st = self.running.get_mut(&id).unwrap();
                    let prev = record
                        .num_completed_steps
                        .saturating_sub(call.bounds.max_tokens)
                        .min(u32::from(u16::MAX)) as u16;
                    (
                        st.image_id,
                        st.req.image.height,
                        st.req.image.width,
                        st.req.image.steps,
                        prev,
                    )
                };

                let sd = self
                    .running
                    .get(&id)
                    .map_or(0, |s| s.denoising.completed())
                    .min(u32::from(u16::MAX)) as u16;

                if prev_sd == 0 && sd >= 1 {
                    self.emit(
                        id,
                        EngineCoreOutput::ImageBegin {
                            image_id,
                            height: h,
                            width: w,
                            steps,
                        },
                    );
                }

                for step in prev_sd.saturating_add(1)..=sd {
                    self.emit(id, EngineCoreOutput::ImageStep { image_id, step });
                }

                // The host planner enters commit after the configured step count;
                // worker completion flags do not determine diffusion termination.
            }
            CallKind::Media(
                MediaCall::VideoDecoding
                | MediaCall::AudioDecoding
                | MediaCall::AudioEncoding
                | MediaCall::Muxing,
            ) => {}
            CallKind::Media(MediaCall::ImageDecoding) => {
                // Commit becomes visible before optional feedback state is prepared.
                let image_id = self.running.get(&id).map_or(0, |st| st.image_id);
                self.emit(id, EngineCoreOutput::ImageCommit { image_id });

                let image = media
                    .and_then(|value| std::str::from_utf8(value.as_bytes()).ok())
                    .map(str::to_owned);
                if let Some(image_b64) = image.clone() {
                    let Some(event) = image_done_event(image_id, image_b64) else {
                        return self.finish(id, FinishReason::Error);
                    };
                    self.emit(id, event);
                }

                self.activate_request_tables(id);
                let (continues_after_gen_commit, feedback_source) = {
                    let st = self.running.get(&id).unwrap();
                    (
                        st.continues_after_gen_commit(),
                        st.req.image_generation.feedback_source.clone(),
                    )
                };

                if continues_after_gen_commit {
                    let Some(feedback_source) = feedback_source else {
                        return self.finish(id, FinishReason::Error);
                    };

                    let source_product = call.image_output.clone();

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

                    // Feedback retains the configured source representation until
                    // the encode and state-ingest stages consume it.
                    if let Some(st) = self.running.get_mut(&id) {
                        st.feedback_image_b64 = image;
                        st.feedback_encoder_index = 0;
                        st.feedback_source = source_product;
                        st.feedback_features = None;
                        st.phase = Phase::FeedbackEncode;
                        if let Some(product) = st.feedback_source.clone() {
                            st.transient_encoder_products.push(product);
                        }
                    }
                } else {
                    if let Some(st) = self.running.get_mut(&id) {
                        st.num_generated_images += 1;
                        st.text_tokens_since_image = 0;
                    }
                    self.finish(id, FinishReason::ImageDone);
                }
            }
            CallKind::Media(MediaCall::VisionEncoding)
            | CallKind::Media(MediaCall::LatentEncoding) => {
                match is_feedback_computation(&call) {
                    false => {
                        let encoder_cache_key = self.running.get(&id).and_then(|state| {
                            let image = state
                                .req
                                .multimodal_inputs
                                .images
                                .get(state.num_ingested_images)?;
                            let encoder = image.encoders.get(state.image_encoder_index)?;
                            state.req.cache.write.then(|| {
                                encoder_cache_key(
                                    image.hash,
                                    state.image_encoder_index,
                                    encoder.encoder,
                                )
                            })
                        });
                        let Some(feature) = call.encoder_output.clone() else {
                            return self.finish(id, FinishReason::Error);
                        };

                        let mut free_products = Vec::new();

                        // Cache ownership transfers the backing allocation out of the
                        // request; uncached products remain request-local transients.
                        let selected_product = if let Some(cache_key) = encoder_cache_key {
                            if let Some(freed) = self
                                .storage
                                .encoder_cache
                                .insert(cache_key, feature.clone())
                            {
                                free_products.push(freed);
                            }

                            let stored = self.storage.encoder_cache.peek_product(cache_key)
                                == Some(feature.clone());

                            if stored {
                                let allocation = self.running.get_mut(&id).and_then(|state| {
                                    state.allocations_mut().take_buffer(feature.buffer_id())
                                });
                                let Some(allocation) = allocation else {
                                    return self.finish(id, FinishReason::Error);
                                };
                                if let Err(allocation) = self
                                    .storage
                                    .retain_encoder_buffer(feature.buffer_id(), allocation)
                                {
                                    self.storage
                                        .pending_buffer_frees
                                        .insert(feature.buffer_id(), allocation);
                                    self.inflight
                                        .pending_commands
                                        .push_back(BatchCommand::Free {
                                            buffer: feature.buffer_id(),
                                        });
                                    return self.finish(id, FinishReason::Error);
                                }
                            }

                            let Some(product) = self.storage.encoder_cache.acquire(cache_key)
                            else {
                                return self.finish(id, FinishReason::Error);
                            };
                            if let Some(st) = self.running.get_mut(&id) {
                                st.encoder_cache_pins.push(EncoderCachePin {
                                    key: cache_key,
                                    product: product.clone(),
                                });
                            }
                            product
                        } else if let Some(st) = self.running.get_mut(&id) {
                            st.transient_encoder_products.push(feature.clone());
                            feature.clone()
                        } else {
                            feature.clone()
                        };

                        if let Some(st) = self.running.get_mut(&id) {
                            st.input_image_features = Some(selected_product);
                            st.phase = Phase::IngestState;
                        }

                        if !free_products.is_empty() {
                            self.free_buffers(
                                free_products.into_iter().map(|product| product.buffer_id()),
                            );
                        }
                    }
                    true => {
                        let Some(feature) = call.encoder_output.clone() else {
                            return self.finish(id, FinishReason::Error);
                        };

                        if let Some(st) = self.running.get_mut(&id) {
                            st.transient_encoder_products.push(feature.clone());
                            st.feedback_features = Some(feature);
                            st.phase = Phase::FeedbackState;
                        }
                    }
                }
            }
            CallKind::Media(MediaCall::TextEncoding)
            | CallKind::Forward(ForwardMode::Decode)
            | CallKind::Forward(ForwardMode::Verify)
            | CallKind::Media(MediaCall::LatentPreparation)
            | CallKind::Media(MediaCall::VideoEncoding)
            | CallKind::Transfer(TransferMode::Tensor)
            | CallKind::Transfer(TransferMode::KvPublish)
            | CallKind::Transfer(TransferMode::KvInstall) => {}
        }
    }

    /// Deduplicates overlapping prompt scores and emits only newly resolved positions.
    pub(super) fn resolve_prompt_logprobs(
        &mut self,
        id: RequestId,
        positions: Vec<Vec<TokenLogprob>>,
    ) {
        let total = positions.len();
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let processed_before = state.output.prompt_logprobs_processed;
        let emitted_before = state.output.prompt_logprobs_emitted;
        let expected_total = state.req.prompt_token_ids.len().saturating_sub(1);
        state.output.prompt_logprobs_processed = processed_before.saturating_add(total);
        let skip = emitted_before.saturating_sub(processed_before);
        let remaining = expected_total.saturating_sub(emitted_before);
        let selected: Vec<_> = positions.into_iter().skip(skip).take(remaining).collect();
        state.output.prompt_logprobs_emitted = emitted_before.saturating_add(selected.len());
        if selected.is_empty() {
            return;
        }
        self.emit(
            id,
            EngineCoreOutput::PromptLogprobs {
                positions: selected
                    .into_iter()
                    .map(|entries| PositionLogprobs { entries })
                    .collect(),
            },
        );
    }

    /// Releases the transient products.
    pub(super) fn free_transient_products(&mut self, id: RequestId) {
        let products = self
            .running
            .get_mut(&id)
            .map(|state| std::mem::take(&mut state.transient_encoder_products))
            .unwrap_or_default();
        self.free_buffers(products.into_iter().map(|product| product.buffer_id()));
    }

    /// Records the gen trigger for replay.
    pub(super) fn record_gen_trigger_for_replay(&mut self, id: RequestId) {
        let Some(start) = self
            .running
            .get(&id)
            .and_then(|st| st.req.image_generation.trigger.direct_token())
        else {
            return;
        };
        if let Some(st) = self.running.get_mut(&id)
            && st.continues_after_gen_commit()
            && st.num_generated_tokens > st.generated_token_ids.len()
            && st.generated_token_ids.last().copied() != Some(start)
        {
            st.generated_token_ids.push(start);
        }
    }

    /// Converts a worst-case KV reservation into concrete generation-branch capacity.
    pub(super) fn promote_gen_branch_reservation(&mut self, id: RequestId) -> bool {
        let Some((required_blocks, reserves_envelope)) = self
            .running
            .get(&id)
            .map(|st| (st.max_reserved_kv_blocks, st.reserve_worstcase))
        else {
            return false;
        };
        let allocated_blocks = self
            .running
            .get(&id)
            .and_then(|state| state.block_tables().first())
            .map_or(0, BlockTable::len);
        if !reserves_envelope || allocated_blocks < required_blocks {
            self.finish(id, FinishReason::Error);
            return false;
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.image_reservation_pending = false;
        }
        true
    }

    /// Begins the image.
    pub(super) fn begin_image(&mut self, id: RequestId) {
        self.record_gen_trigger_for_replay(id);
        let has_unresolved_descendants = self.inflight.has_pending_calls(id);
        if let Some(st) = self.running.get_mut(&id) {
            st.speculative_chain_invalidated |= has_unresolved_descendants;

            st.denoising.close();
            st.image_id += 1;
            st.text_tokens_since_image = 0;
            st.phase = Phase::CloseKv;
            st.image_reservation_pending = true;
        }
        self.promote_gen_branch_reservation(id);
    }

    /// Flushes the output journals.
    pub(super) fn flush_output_journals(&mut self) -> bool {
        let mut progressed = false;
        for (events, intent) in self
            .running
            .values_mut()
            .map(|state| (&mut state.output.events, &mut state.terminal_intent))
            .chain(
                self.running_media
                    .values_mut()
                    .map(|state| (&mut state.output, &mut state.terminal_intent)),
            )
        {
            let before = events.journal.len();
            if events.is_closed() || events.flush() {
                *intent = TerminalIntent::Finish(FinishReason::Cancelled);
            }
            progressed |= events.journal.len() != before;
        }
        progressed |= self.output.flush_retired();
        progressed
    }

    /// Returns whether the request accepted an event into its ordered output.
    pub(super) fn emit(&mut self, id: RequestId, event: EngineCoreOutput) -> bool {
        let Some(state) = self.running.get_mut(&id) else {
            return false;
        };
        let accepted = state.output.events.enqueue(event);
        if !accepted {
            state.terminal_intent = TerminalIntent::Finish(FinishReason::Cancelled);
        }
        accepted
    }

    /// Emits a text token and advance output accounting.
    pub(super) fn emit_text(&mut self, id: RequestId, tok: u32, logprob: Option<f32>) -> bool {
        let Some(st) = self.running.get_mut(&id) else {
            return false;
        };
        st.generated_token_ids.push(tok);
        st.text_tokens_since_image = st.text_tokens_since_image.saturating_add(1);
        if !st.req.emits_text() {
            return false;
        }
        let published = self.emit(id, EngineCoreOutput::TextToken { id: tok, logprob });
        if published && let Some(state) = self.running.get_mut(&id) {
            state.output.tokens_sent = state.output.tokens_sent.saturating_add(1);
        }
        true
    }

    /// Emits one visible token and its requested candidate log probabilities.
    pub(super) fn emit_sampled_text(
        &mut self,
        id: RequestId,
        tok: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
    ) {
        if self.emit_text(id, tok, logprob)
            && let Some(top_logprobs) = top_logprobs.filter(|entries| !entries.is_empty())
        {
            self.emit(
                id,
                EngineCoreOutput::TokenLogprobs {
                    id: tok,
                    candidates: top_logprobs,
                },
            );
        }
    }

    /// Emits the terminal stop token.
    pub(super) fn emit_terminal_stop_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
    ) {
        if self
            .running
            .get(&id)
            .is_some_and(|state| state.req.include_stop_token)
        {
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        }
    }

    /// Resolves stop conditions for one understanding token and emits it when appropriate.
    pub(super) fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
        sampled: bool,
    ) -> bool {
        let Some(state) = self.running.get(&id) else {
            return true;
        };
        // `tokens_emitted` already counts the token being resolved, so the floor
        // is cleared only once at least `min_tokens` tokens have been emitted.
        let generated = state.num_generated_tokens;
        let under_floor = generated <= state.req.sampling.min_tokens;
        let stop_hit = !under_floor && state.req.stop_token_ids.contains(&token_id);
        let eos_hit =
            self.ctrl.eos.contains(&token_id) && !state.req.sampling.ignore_eos && !under_floor;
        let max_hit = generated >= state.req.max_und_tokens;

        if stop_hit {
            self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs);
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
                    self.emit_sampled_text(id, token_id, logprob, top_logprobs);
                } else {
                    self.emit_text(id, token_id, logprob);
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
            self.emit_sampled_text(id, token_id, logprob, top_logprobs);
        } else {
            self.emit_text(id, token_id, logprob);
        }
        !self.running.contains_key(&id)
    }

    /// Finishes one request without a concrete stop-token or stop-string cause.
    pub(super) fn finish(&mut self, id: RequestId, reason: FinishReason) {
        self.finish_with(id, reason, None);
    }

    /// Defers terminal cleanup until in-flight work and decoder decisions are drained.
    pub(super) fn finish_after_inflight(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<uniserve_core::StopReason>,
    ) {
        let decoder_pending = !matches!(reason, FinishReason::Error)
            && self
                .running
                .get(&id)
                .is_some_and(|state| !state.output.decoder_boundaries.is_empty());
        if !self.inflight.has_pending_calls(id) && !decoder_pending {
            self.finish_with(id, reason, stop_reason);
            return;
        }
        if !self.inflight.pending_finishes.contains_key(&id)
            || matches!(reason, FinishReason::Error)
        {
            self.inflight.pending_finishes.insert(
                id,
                PendingFinish {
                    reason,
                    stop_reason,
                },
            );
        }
    }

    /// Applies a deferred finish once no in-flight work or decoder decision remains.
    pub(super) fn finish_pending_if_idle(&mut self, id: RequestId) {
        if self.inflight.has_pending_calls(id)
            || self
                .running
                .get(&id)
                .is_some_and(|state| !state.output.decoder_boundaries.is_empty())
        {
            return;
        }
        if let Some(pending) = self.inflight.pending_finishes.remove(&id) {
            self.finish_with(id, pending.reason, pending.stop_reason);
        }
    }

    /// Publishes terminal accounting and releases every resource owned by one request.
    pub(super) fn finish_with(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<uniserve_core::StopReason>,
    ) {
        self.inflight.pending_finishes.remove(&id);

        // Report accepted request progress before error teardown removes it.
        if reason == FinishReason::Error
            && let Some(state) = self.running.get(&id)
        {
            tracing::error!(
                request_id = id.0,
                phase = ?state.phase,
                generated_tokens = state.num_generated_tokens,
                images_done = state.num_generated_images,
                image_id = state.image_id,
                denoise_steps_done = state.denoising.completed(),
                logical_position = state.logical_position,
                physical_kv_len = state.kv_visible_len,
                "scheduler request terminated with an internal error"
            );
        }

        // Registered requests retain their allocations until every worker pool
        // acknowledges the ordered close command.
        let mut awaits_close = false;
        let mut allocations = None;
        let mut flow_prefix = None;
        if let Some(mut st) = self.running.remove(&id) {
            let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
            if st.worker_registered {
                let retained_buffers = self.storage.retained_buffers(request_key);
                let command = BatchCommand::Finish {
                    request_key,
                    retained_buffers,
                };
                self.inflight.pending_commands.push_back(command);
                let mut allocation = st.allocations.take().expect("admitted allocations");
                let buffers = std::mem::take(&mut allocation.buffers);
                let allocations = st
                    .flow_prefix
                    .take()
                    .into_iter()
                    .map(|prefix| prefix.allocations)
                    .chain([allocation])
                    .collect();
                self.retiring_requests.insert(
                    id,
                    RetiringRequest {
                        request_key,
                        allocations,
                        media_allocations: None,
                        buffers,
                    },
                );
                awaits_close = true;
            } else {
                allocations = st.allocations.take();
                flow_prefix = st.flow_prefix.take();
            }

            // Remove admission reservations before exposing terminal output so
            // the next scheduler step observes the released capacity.
            self.running_order.retain(|request| *request != id);
            self.storage.reserved_encoder_entries = self
                .storage
                .reserved_encoder_entries
                .saturating_sub(st.req.num_encoder_cache_entries());
            if st.reserve_worstcase {
                self.storage.reserved_blocks = self
                    .storage
                    .reserved_blocks
                    .saturating_sub(st.max_reserved_kv_blocks);
            }

            // Releasing the request's encoder pins may make cache products
            // physically reclaimable by their owning worker.
            let mut free_encoder_products = std::mem::take(&mut st.transient_encoder_products);
            for pin in &st.encoder_cache_pins {
                if let Some(product) = self.storage.encoder_cache.release(pin.key, &pin.product) {
                    free_encoder_products.push(product);
                }
            }
            self.free_buffers(
                free_encoder_products
                    .into_iter()
                    .map(|product| product.buffer_id()),
            );

            // A full event channel transfers ownership to the retired-output
            // queue, which drains the terminal event under normal backpressure.
            let terminal = EngineCoreOutput::Finished {
                reason,
                stop_reason,
                prompt_tokens: st.req.prompt_token_ids.len(),
                completion_tokens: st.num_generated_tokens,
                images: st.num_generated_images,
            };
            let accepted = st.output.events.enqueue(terminal);
            if accepted {
                self.output.retire(id, st.output.events);
            }
        }

        // Unregistered requests have no outstanding worker reference, so their
        // allocations can return to the scheduler immediately.
        if !awaits_close {
            if let Some(prefix) = flow_prefix {
                prefix.allocations.free(&mut self.storage);
            }
            if let Some(allocations) = allocations {
                allocations.free(&mut self.storage);
            }
        }
    }
}

/// Caches the prompt blocks.
fn cache_prompt_blocks(
    coordinator: &KvCacheCoordinator,
    state: &mut RequestState,
    pool: &BlockPool,
    source: &Arc<uniserve_worker_ipc::WorkerEndpoint>,
) {
    if state.prefix_cached {
        return;
    }
    let prompt = state.req.prompt_token_ids.clone();
    if coordinator.cache_prefix(
        pool,
        state.block_tables(),
        &prompt,
        &state.prefix_block_hashes,
        state.req.cache.write,
        source,
    ) {
        state.prefix_cached = true;
    }
}
