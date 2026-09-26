//! Client-visible event publication and per-request output accounting.
//!
//! Events accumulate in request-local journals when the bounded consumer channel
//! is full, preserving order without blocking the engine owner thread.
//!
//! This module also owns the semantic resolution of completed calls
//! (`Scheduler::resolve`): it turns accepted worker results into public events
//! and the request's next generation phase, and it owns the termination of
//! running token requests (`Scheduler::finish_with`), which releases scheduler
//! resources and publishes the terminal `Finished` event.
//! `RequestState::process_generation_result` has already applied a call's
//! accepted progress (cursors, encoder indices, denoising steps, phase) by the
//! time `resolve` runs.

use super::*;
use uniserve_worker_ipc::{ForwardMode, MediaCall, TransferMode};

/// How one understanding token ends or continues its request
/// (`Scheduler::und_token_outcome`).
enum TokenOutcome {
    /// The token is ordinary output.
    Continue,
    /// The token is one of the request's stop tokens, published only with
    /// `include_stop_token`.
    StopToken,
    /// The token ends the request by EOS or the token limit; it is published
    /// unless it is an EOS token.
    Finish(FinishReason),
}

/// Flushes journaled public events into the output channel in order.
///
/// Returns `true` only when the receiver has closed, in which case the journal
/// is discarded. Returns `false` when the journal drained completely or when
/// the channel filled again, leaving the remainder journaled.
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
    /// `TextToken` events accepted into the ordered output.
    pub(super) tokens_sent: usize,
    /// Text-token count the frontend decoder has acknowledged.
    ///
    /// `Scheduler::acknowledge_output` treats an acknowledgement that exceeds
    /// `tokens_sent` or moves backwards as an error finish; `mark_stopped` and
    /// cancellation apply the same upper bound.
    pub(super) tokens_acked: usize,
    /// Prompt score positions received from worker completions.
    pub(super) prompt_logprobs_processed: usize,
    /// Prompt score positions published as `PromptLogprobs` events.
    pub(super) prompt_logprobs_emitted: usize,
    /// Ends of the tokens each resolved token-producing call published,
    /// awaiting a stop-string decoder decision, as `tokens_sent` values in
    /// publication order. Populated only for requests with stop strings;
    /// `usize::MAX` marks the call being resolved. `can_schedule_next`
    /// refuses another call while the queue holds `max_unresolved_calls`
    /// entries; acknowledgements drain it, and a non-error finish waits until
    /// it is empty.
    pub(super) decoder_boundaries: VecDeque<usize>,
}

impl RequestOutput {
    /// Creates empty output accounting that publishes through `event_tx`.
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
///
/// The journal holds at most `OUTPUT_JOURNAL_CAPACITY` events beyond what the
/// bounded channel holds, and `enqueue` panics if that bound is exceeded. For
/// token requests the scheduler stays within it by scheduling a call only when
/// `Scheduler::output_window_ready` finds room for the worst-case output of
/// every in-flight call and the next one, plus a terminal reserve.
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

    /// Returns whether the request's event receiver has closed.
    pub(super) fn is_closed(&self) -> bool {
        self.event_tx.is_closed()
    }

    /// Returns the number of events the journal can accept: free channel slots
    /// plus the journal's remaining headroom.
    pub(super) fn available_capacity(&self) -> usize {
        self.event_tx
            .capacity()
            .saturating_add(OUTPUT_JOURNAL_CAPACITY.saturating_sub(self.journal.len()))
    }

    /// Flushes pending journal entries; returns `true` when the receiver closed.
    fn flush(&mut self) -> bool {
        flush_public_journal(&self.event_tx, &mut self.journal)
    }

    /// Delivers an event or journals it in order when the public channel is full.
    ///
    /// Returns `true` when the event was delivered, journaled, or coalesced, and
    /// `false` when the receiver has closed and the event was dropped.
    pub(super) fn enqueue(&mut self, event: EngineCoreOutput) -> bool {
        if self.flush() {
            return false;
        }

        // A journaled progress event that has not been delivered is superseded
        // by the newer one; coalescing only replaces the journal tail, so no
        // other event loses its place.
        if matches!(event, EngineCoreOutput::MediaProgress { .. })
            && let Some(last @ EngineCoreOutput::MediaProgress { .. }) = self.journal.back_mut()
        {
            *last = event;
            return true;
        }

        // Send directly only when nothing is journaled; otherwise the event
        // would overtake earlier ones.
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
///
/// Holds the journals of finished token and media requests whose remaining
/// events, including the terminal one, have not yet fit into their channels,
/// so request state can retire while its output drains under backpressure.
pub(crate) struct OutputSender {
    retired: HashMap<RequestId, EventJournal>,
}

impl OutputSender {
    /// Returns the number of retired requests still awaiting output delivery.
    ///
    /// Admission counts these toward `max_num_waiting`, so undelivered
    /// terminal output applies backpressure to new submissions.
    pub(super) fn retained_len(&self) -> usize {
        self.retired.len()
    }

    /// Retains pending events after computational state has retired.
    ///
    /// A journal with nothing pending is dropped immediately.
    pub(super) fn retire(&mut self, id: RequestId, output: EventJournal) {
        if !output.journal.is_empty() {
            self.retired.insert(id, output);
        }
    }

    /// Flushes retired journals, dropping each once it drains or its receiver
    /// closes. Returns whether any journal changed length.
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

    /// Hands every retained journal to its receiver as the control loop stops.
    ///
    /// A stopped scheduler can no longer flush journals into their full
    /// channels, so each journal's pending events, including the terminal
    /// event, move to the receiver, which yields them after the channel's
    /// buffered events (`EventTx::close_with`).
    pub(super) fn hand_off_retired(&mut self) {
        for (_, output) in self.retired.drain() {
            output.event_tx.close_with(output.journal);
        }
    }
}

impl Scheduler {
    /// Publishes committed autoregressive tokens and advances text-generation state.
    ///
    /// Tokens are applied in order and resolution stops at the first token that
    /// finishes the request, opens an image branch, or is a round-close token.
    /// The sampled log probability and top candidates belong to the last
    /// committed token only.
    pub(super) fn resolve_decode_text(
        &mut self,
        id: RequestId,
        mut record: uniserve_worker_ipc::RequestOutput,
    ) {
        self.activate_request_tables(id);

        // A round-close token committed alone by the previous decode became
        // this decode's input; its completion now decides the round with that
        // token and ignores the tokens this call committed.
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
        // A completion without committed tokens resolves as the first EOS token.
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
            // A round-close token inside a multi-token commit decides the round
            // immediately. A single committed close token is deferred: it
            // becomes the next decode's input and `round_closing` makes that
            // decode's completion decide the round.
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
            // A direct trigger token opens the image branch without being
            // published as text.
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

            // A trigger sequence matched against `generated_token_ids` opens the
            // branch after its last token has been recorded there.
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
    /// `RequestState::process_generation_result` has already applied the call's
    /// accepted progress, so the cursors, encoder indices, denoising step count,
    /// and phase read here reflect this completion. Resolution adds the
    /// remaining effects for the call's kind, such as stop conditions,
    /// image-branch triggers, prefix- and encoder-cache publication, feedback
    /// state, and public events. `media`
    /// carries the base64 PNG artifact of an image-decoding completion.
    ///
    /// The caller resolves only a running request, and every path that
    /// finishes it returns; a request that is no longer running has nothing
    /// left to update, so resolution stops there.
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
                    // A prefill with neither image features nor a token output
                    // is the `CloseKv` write. It publishes nothing, and
                    // `process_generation_result` has already moved the request
                    // to `PublishKv`.
                    (false, _) if !is_prompt_extend(&call) => return,
                    // Input-image state ingest. The encoder index wraps to zero
                    // once the image's last encoder output has been ingested.
                    (true, false) => {
                        let is_final_step = self
                            .running
                            .get(&id)
                            .is_some_and(|state| state.image_encoder_index == 0);
                        if is_final_step {
                            self.free_transient_products(id);
                        }

                        // Ingesting the last image of a fully consumed prompt
                        // starts a readout's canvases, a canvas-generating
                        // request's first block, or understanding decode from
                        // BOS.
                        if is_final_step
                            && self.running.get(&id).is_some_and(|st| {
                                st.num_ingested_images >= st.req.multimodal_inputs.images.len()
                                    && st.num_computed_prompt_tokens
                                        >= st.req.prompt_token_ids.len() as u32
                            })
                        {
                            let bos = self.ctrl.bos;
                            if let Some(st) = self.running.get_mut(&id) {
                                if st.req.is_readout() {
                                    st.phase = Phase::Readout;
                                } else if st.req.is_canvas_generation() {
                                    st.phase = Phase::Canvas;
                                } else {
                                    st.next_token = bos;
                                    st.round_token_ids.clear();
                                    st.phase = Phase::DecodeUnd;
                                }
                            }
                        }
                        return;
                    }
                    // Generated-image feedback ingest. Only the last feedback
                    // encoder's completion counts the image and chooses how
                    // text decode resumes.
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

                        // Without a sampled continuation, the configured
                        // `feedback_next_token` is the next decode input, and
                        // its absence is an error; with one, this call's
                        // sampled token is resolved like any decoded token.
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

                        let Some(st) = self.running.get_mut(&id) else {
                            return;
                        };
                        st.num_generated_tokens += 1;
                        let (can_open_gen_branch, images_done, max_images) = (
                            st.can_open_gen_branch(),
                            st.num_generated_images,
                            st.req.image.max_images as usize,
                        );

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

                // Prompt extension. Partial prefill remains in ingest until every
                // text and multimodal position has been consumed.
                let Some(st) = self.running.get(&id) else {
                    return;
                };
                let (cursor, prompt_len) = (
                    st.num_computed_prompt_tokens as usize,
                    st.req.prompt_token_ids.len(),
                );
                if cursor < prompt_len {
                    return;
                }

                if self.running.get(&id).is_some_and(|st| {
                    st.num_ingested_images < st.req.multimodal_inputs.images.len()
                        || st.num_computed_prompt_tokens < st.req.prompt_token_ids.len() as u32
                }) {
                    return;
                }

                // A readout's complete prompt samples nothing; its canvases
                // follow. So does a canvas-generating request's, and each of
                // its block commits, which extend the context the same way;
                // the next block follows.
                if let Some(st) = self.running.get_mut(&id)
                    && (st.req.is_readout() || st.req.is_canvas_generation())
                {
                    st.phase = if st.req.is_readout() {
                        Phase::Readout
                    } else {
                        Phase::Canvas
                    };
                    return;
                }

                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                st.num_generated_tokens += 1;
                let (starts_gen_after_context, can_open_gen_branch) =
                    (st.starts_gen_after_context(), st.can_open_gen_branch());

                // Prompt scoring covers every prompt position after the first;
                // a complete prompt with unscored positions is an error.
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

                let Some(st) = self.running.get(&id) else {
                    return;
                };
                let (images_done, max_images) =
                    (st.num_generated_images, st.req.image.max_images as usize);

                // A direct trigger token opens the image branch without being
                // published as text.
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
            // A canvas step that stopped its block left the block's tokens
            // for their commit; they are published first, in order, and the
            // request finishes on the first that ends it.
            CallKind::Forward(ForwardMode::TokenDenoising) if call.canvas.is_some() => {
                let tokens = match self.running.get(&id) {
                    Some(st) if st.phase == Phase::CommitCanvas => st.canvas_commit.clone(),
                    _ => return,
                };
                self.emit_or_finish_block(id, &tokens);
            }
            CallKind::Forward(ForwardMode::TokenDenoising) => {
                // `process_generation_result` accepted the pass's rows and
                // their log-probabilities. Once every row has reported, the
                // request answers with all of them, in report order.
                let Some(st) = self.running.get_mut(&id) else {
                    return;
                };
                if st.readout_rows < st.req.readout.len() {
                    return;
                }
                let candidate_logprobs = std::mem::take(&mut st.readout_logprobs);
                if candidate_logprobs.len() != st.req.readout_candidates() {
                    tracing::error!(
                        request_id = id.0,
                        "a completed readout lacks candidate log-probabilities"
                    );
                    return self.finish(id, FinishReason::Error);
                }
                self.emit(id, EngineCoreOutput::Readout { candidate_logprobs });
                self.finish(id, FinishReason::Completed);
            }
            CallKind::Media(MediaCall::Denoising) => {
                // Publish every newly committed step exactly once, including steps
                // coalesced into a single worker completion. `bounds.max_tokens`
                // is the call's step count, so subtracting it from the reported
                // total gives the steps completed before this call. Step
                // numbers saturate at `u16::MAX` to fit the event fields.
                let Some(st) = self.running.get(&id) else {
                    return;
                };
                let (image_id, h, w, steps, prev_sd) = {
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
                let Some(st) = self.running.get(&id) else {
                    return;
                };
                let (continues_after_gen_commit, feedback_source) = (
                    st.continues_after_gen_commit(),
                    st.req.image_generation.feedback_source.clone(),
                );

                if continues_after_gen_commit {
                    let Some(feedback_source) = feedback_source else {
                        return self.finish(id, FinishReason::Error);
                    };

                    // The configured feedback source must exist: the decoder's
                    // device image product or the published PNG artifact.
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
                    // A request that does not continue after its image commit
                    // ends with this image.
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
                    // Input-image encoding: the product feeds the next state
                    // ingest and, when the request writes the encoder cache,
                    // becomes a shared cache entry.
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
                            // Insertion may return an evicted entry or, when a
                            // concurrent miss already stored this key, this
                            // call's redundant product; either is freed once
                            // the request's product is selected.
                            if let Some(freed) = self
                                .storage
                                .encoder_cache
                                .insert(cache_key, feature.clone())
                            {
                                free_products.push(freed);
                            }

                            let stored = self.storage.encoder_cache.peek_product(cache_key)
                                == Some(feature.clone());

                            // Only a product the cache now holds moves its
                            // buffer from the request to encoder retention.
                            if stored {
                                let allocation = self.running.get_mut(&id).and_then(|state| {
                                    state.allocations_mut()?.take_buffer(feature.buffer_id())
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

                            // The request pins the resident entry, which may be an
                            // earlier equivalent product rather than this call's.
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
                    // Feedback encoding: the product is request-local and
                    // released once feedback ingest completes.
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
            // These calls publish nothing and change no state here. `Decode`
            // returned above and appears only for exhaustiveness.
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

    /// Publishes one completion's prompt scores as a `PromptLogprobs` event.
    ///
    /// Every received position counts toward `prompt_logprobs_processed`.
    /// Positions beyond the prompt's scorable range (every prompt position
    /// after the first) are dropped, and nothing is published when no position
    /// remains.
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

    /// Frees the request's transient products: uncached input-image encoder
    /// outputs, feedback encoder outputs, and the device feedback source.
    pub(super) fn free_transient_products(&mut self, id: RequestId) {
        let products = self
            .running
            .get_mut(&id)
            .map(|state| std::mem::take(&mut state.transient_encoder_products))
            .unwrap_or_default();
        self.free_buffers(products.into_iter().map(|product| product.buffer_id()));
    }

    /// Appends the direct trigger token to `generated_token_ids` for a request
    /// that continues after its image commit, when `num_generated_tokens`
    /// counts more tokens than that list records and the list does not
    /// already end with the trigger.
    ///
    /// A direct trigger opens the image branch without passing through
    /// `emit_text`. This keeps the trigger in the accepted history that
    /// bad-word masks (`build_token_masks`) and generated-trigger matching
    /// read.
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
    ///
    /// Succeeds, clearing `image_reservation_pending`, when the request reserves
    /// its worst-case envelope and every block table already covers
    /// `max_reserved_kv_tokens` tokens. Otherwise finishes the request with
    /// `FinishReason::Error` and returns `false`; a request that is not running
    /// also returns `false`.
    pub(super) fn promote_gen_branch_reservation(&mut self, id: RequestId) -> bool {
        let Some((required_tokens, reserves_envelope)) = self
            .running
            .get(&id)
            .map(|st| (st.max_reserved_kv_tokens, st.reserve_worstcase))
        else {
            return false;
        };
        let covered = self
            .running
            .get(&id)
            .and_then(RequestState::block_tables)
            .is_some_and(|tables| {
                tables
                    .iter()
                    .all(|table| table.capacity_tokens() >= required_tokens)
            });
        if !reserves_envelope || !covered {
            self.finish(id, FinishReason::Error);
            return false;
        }
        if let Some(st) = self.running.get_mut(&id) {
            st.image_reservation_pending = false;
        }
        true
    }

    /// Opens the next image-generation branch.
    ///
    /// Moves the request to `Phase::CloseKv` under a new `image_id` and checks
    /// its KV reservation, which can finish the request with an error. When
    /// calls of the request are still in flight, its speculative chain is
    /// marked invalidated: those results are discarded on arrival, and nothing
    /// new is scheduled for the request until they drain.
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

    /// Flushes the journals of running token and media requests and of retired
    /// requests. Returns whether any journal changed length.
    ///
    /// A running request whose receiver has closed is marked for cancellation
    /// through its terminal intent; it retires once its in-flight calls drain.
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
    ///
    /// Returns `false` for a request that is not running. A closed receiver
    /// also returns `false` and marks the request for cancellation.
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

    /// Records an accepted text token and publishes it when the request emits text.
    ///
    /// A running request always records the token in `generated_token_ids`.
    /// Returns `true` when the token is visible text, even if a closed receiver
    /// dropped it, and `false` when the request is not running or does not emit
    /// text. `tokens_sent` advances only when the event was accepted.
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

    /// Records accepted text tokens and publishes them as one event when the
    /// request emits text.
    ///
    /// A running request always records the tokens in `generated_token_ids`.
    /// `tokens_sent` advances only when the event was accepted.
    pub(super) fn emit_text_tokens(&mut self, id: RequestId, tokens: Vec<u32>) {
        let Some(st) = self.running.get_mut(&id) else {
            return;
        };
        st.generated_token_ids.extend_from_slice(&tokens);
        st.text_tokens_since_image = st.text_tokens_since_image.saturating_add(tokens.len());
        if !st.req.emits_text() || tokens.is_empty() {
            return;
        }
        let count = tokens.len();
        let published = self.emit(id, EngineCoreOutput::TextTokens { ids: tokens });
        if published && let Some(state) = self.running.get_mut(&id) {
            state.output.tokens_sent = state.output.tokens_sent.saturating_add(count);
        }
    }

    /// Resolves the stop conditions of a stopped canvas block's tokens in
    /// order and publishes the visible ones as one event.
    ///
    /// Every token counts toward `num_generated_tokens` and is resolved as
    /// `emit_or_finish_und_token` resolves a sampled token: the first token
    /// that ends the request (a stop token, EOS, or the token limit) closes
    /// the block, which is published up to it, and the request finishes after
    /// its in-flight work. Tokens after it are never generated output.
    pub(super) fn emit_or_finish_block(&mut self, id: RequestId, tokens: &[u32]) {
        let mut visible = Vec::with_capacity(tokens.len());
        let mut finish = None;
        for &token_id in tokens {
            let Some(state) = self.running.get_mut(&id) else {
                return;
            };
            state.num_generated_tokens += 1;
            match self.und_token_outcome(id, token_id) {
                TokenOutcome::Continue => visible.push(token_id),
                TokenOutcome::StopToken => {
                    if self
                        .running
                        .get(&id)
                        .is_some_and(|state| state.req.include_stop_token)
                    {
                        visible.push(token_id);
                    }
                    finish = Some((
                        FinishReason::Stop,
                        Some(uniserve_core::StopReason::Token(token_id)),
                    ));
                    break;
                }
                TokenOutcome::Finish(reason) => {
                    if !self.ctrl.eos.contains(&token_id) {
                        visible.push(token_id);
                    }
                    finish = Some((reason, None));
                    break;
                }
            }
        }
        self.emit_text_tokens(id, visible);
        if let Some((reason, stop_reason)) = finish {
            self.finish_after_inflight(id, reason, stop_reason);
        }
    }

    /// Decides how one understanding token, already counted in
    /// `num_generated_tokens`, ends or continues its request.
    ///
    /// Stop tokens and EOS are honored only once at least `min_tokens` tokens
    /// precede the token; the token limit applies regardless of that floor,
    /// and a stop-token match takes precedence over EOS and the limit. A
    /// request that is not running continues, and its caller finds it gone.
    fn und_token_outcome(&self, id: RequestId, token_id: u32) -> TokenOutcome {
        let Some(state) = self.running.get(&id) else {
            return TokenOutcome::Continue;
        };
        let generated = state.num_generated_tokens;
        let under_floor = generated <= state.req.sampling.min_tokens;
        if !under_floor && state.req.stop_token_ids.contains(&token_id) {
            TokenOutcome::StopToken
        } else if generated >= state.req.max_und_tokens {
            TokenOutcome::Finish(FinishReason::MaxTokens)
        } else if self.ctrl.eos.contains(&token_id)
            && !state.req.sampling.ignore_eos
            && !under_floor
        {
            TokenOutcome::Finish(FinishReason::Eos)
        } else {
            TokenOutcome::Continue
        }
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

    /// Publishes a matched stop token only when the request includes stop tokens
    /// in its output.
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
    ///
    /// The caller has already counted the token in `num_generated_tokens`.
    /// A stop-token match takes precedence and publishes the token only with
    /// `include_stop_token`. A request that finishes on EOS or the token limit
    /// publishes the token unless it is an EOS token; an EOS token that does
    /// not finish the request (under the floor or with `ignore_eos`) is
    /// published as text. Returns `true` when the caller must stop resolving:
    /// the request finished, has a deferred finish, or is no longer running.
    pub(super) fn emit_or_finish_und_token(
        &mut self,
        id: RequestId,
        token_id: u32,
        logprob: Option<f32>,
        top_logprobs: Option<Vec<TokenLogprob>>,
        sampled: bool,
    ) -> bool {
        if !self.running.contains_key(&id) {
            return true;
        }
        match self.und_token_outcome(id, token_id) {
            TokenOutcome::StopToken => {
                self.emit_terminal_stop_token(id, token_id, logprob, top_logprobs);
                self.finish_after_inflight(
                    id,
                    FinishReason::Stop,
                    Some(uniserve_core::StopReason::Token(token_id)),
                );
                return true;
            }
            TokenOutcome::Finish(reason) => {
                if !self.ctrl.eos.contains(&token_id) {
                    if sampled {
                        self.emit_sampled_text(id, token_id, logprob, top_logprobs);
                    } else {
                        self.emit_text(id, token_id, logprob);
                    }
                }
                self.finish_after_inflight(id, reason, None);
                return true;
            }
            TokenOutcome::Continue => {}
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
    ///
    /// Finishes immediately when nothing is outstanding; an error finish does
    /// not wait for stop-string decoder decisions. Otherwise records a
    /// `PendingFinish` that `finish_pending_if_idle` applies later. The first
    /// recorded reason is kept, except that an error replaces it.
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
    ///
    /// Applies to token requests in `running`; media requests terminate
    /// through `finish_media`. For a request that is not running, this only
    /// clears a pending finish. Allocations of a request registered with its
    /// workers move to `retiring_requests` until the queued `Finish` command
    /// is acknowledged; an unregistered request's allocations are freed here.
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
                let mut allocations: Vec<RequestAllocations> = st
                    .flow_prefix
                    .take()
                    .into_iter()
                    .map(|prefix| prefix.allocations)
                    .collect();
                let mut buffers = HashMap::new();
                match st.allocations.take() {
                    Some(mut allocation) => {
                        buffers = std::mem::take(&mut allocation.buffers);
                        allocations.push(allocation);
                    }
                    None => self.invariant_broken("a registered request holds its allocations"),
                }
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
                let reserved = self.storage.cache().map_or(0, |cache| {
                    cache
                        .coordinator
                        .units_for_tokens(st.max_reserved_kv_tokens)
                });
                self.storage.reserved_units = self.storage.reserved_units.saturating_sub(reserved);
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
            // queue, which drains the terminal event under normal backpressure
            // or hands it to the receiver when the control loop stops. A
            // closed receiver drops the terminal event with the journal.
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

impl Scheduler {
    /// Applies a completed forward's accepted KV extent to the request's
    /// cache tables.
    ///
    /// First publishes the prompt pages the request has now computed, keyed
    /// by the loaded prefill worker that holds them, so every page of a
    /// sliding-window group enters the prefix cache while its table still
    /// holds it. Then retires the sliding-window pages no later reader needs:
    /// calls apply in submission order, so every earlier reader has
    /// completed, and later calls read no history before the accepted extent
    /// minus the window. A request without its prefill worker or KV cache
    /// breaks a scheduler invariant.
    pub(super) fn advance_request_kv(&mut self, id: RequestId) {
        let Some(state) = self.running.get(&id) else {
            return;
        };
        let key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let source = self
            .placement
            .affinity
            .get(&(key, DEFAULT_COMPONENT.to_owned()))
            .and_then(|worker| {
                self.executor
                    .info()
                    .workers
                    .iter()
                    .find(|(id, _)| id == worker)
            })
            .map(|(_, worker)| Arc::new(worker.endpoint.clone()));
        let (Some(source), Some(state), Some(kv)) = (
            source,
            self.running.get_mut(&id),
            self.storage.cache.as_ref(),
        ) else {
            self.invariant_broken(
                "a prefilled request keeps its loaded prefill worker and KV cache",
            );
            return;
        };

        // The request's prompt, hashes and publication progress are borrowed
        // beside its tables for the duration of the update.
        let prompt = std::mem::take(&mut state.req.prompt_token_ids);
        let hashes = std::mem::take(&mut state.prefix_page_hashes);
        let mut published = std::mem::take(&mut state.prefix_published);
        let computed = state.num_computed_prompt_tokens as usize;
        let visible = state.kv_visible_len as usize;
        let cache_write = state.req.cache.write;
        let advanced = state.kv_mut().is_some_and(|allocation| {
            kv.coordinator.publish_prompt(
                &kv.block_pool,
                allocation,
                &prompt,
                &hashes,
                computed,
                &mut published,
                cache_write,
                &source,
            ) && kv.coordinator.release_window(allocation, visible)
        });
        state.req.prompt_token_ids = prompt;
        state.prefix_page_hashes = hashes;
        state.prefix_published = published;
        if !advanced {
            self.invariant_broken("an admitted request's KV tables match the cache groups");
        }
    }
}
