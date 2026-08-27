use super::*;

pub(super) struct FinishedTrace<'a> {
    pub id: RequestId,
    pub reason: &'a FinishReason,
    pub stop_reason: Option<&'a uniserve_core::StopReason>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub images: usize,
    pub queue: &'static str,
}

impl Scheduler {
    /// Abort every queued/gated/running request with a terminal event.
    pub(super) fn abort_all_requests(&mut self) {
        while let Some(submission) = self.pending_media.pop_front() {
            let _ = submission.event_tx.send(MediaEvent::Aborted);
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(id, MediaEvent::Aborted, CloseReason::Cancelled, None);
        }
        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(st) = self.pending.pop_request() {
                ids.push(st.req.request_id);
                let _ = st.event_tx.send(GenerationEvent::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.context.prompt_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            ids
        };
        let _ = queued;
        let running: Vec<RequestId> = self.running.keys().copied().collect();
        for id in running {
            self.finish(id, FinishReason::Aborted);
        }
    }

    /// Apply one command; returns true on shutdown.
    pub(super) fn handle_command(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Submit { request, event_tx } => self.enqueue(*request, event_tx),
            Command::SubmitMedia { request, event_tx } => {
                self.enqueue_media(PendingMedia { request, event_tx })
            }
            Command::Cancel {
                request_id,
                output_token_count,
            } => self.mark_cancelled(request_id, false, output_token_count),
            Command::Acknowledge {
                request_id,
                output_token_count,
            } => self.acknowledge_semantic(request_id, output_token_count),
            Command::StopAt {
                request_id,
                output_token_count,
            } => self.mark_stopped(request_id, output_token_count),
            Command::Abort(id) => self.mark_cancelled(id, true, None),
            Command::Shutdown => return true,
        }
        false
    }

    /// Test harness direct enqueue (the production path is `run` over the channel).
    #[doc(hidden)]
    pub fn submit_for_test(&mut self, request: GenerationRequest) -> EventRx {
        let (event_tx, event_rx) = event_channel();
        self.enqueue(request, event_tx);
        event_rx
    }

    pub(super) fn mark_cancelled(
        &mut self,
        id: RequestId,
        abort: bool,
        output_token_count: Option<usize>,
    ) {
        if let Some(index) = self
            .pending_media
            .iter()
            .position(|submission| submission.request.request_id == id)
            && let Some(submission) = self.pending_media.remove(index)
        {
            let _ = submission.event_tx.send(MediaEvent::Aborted);
            return;
        }
        if self.media_state(id).is_some() {
            self.media_state_mut(id)
                .expect("media state exists")
                .terminal_intent
                .cancel();
            let drained = !self.has_inflight(id);
            if drained {
                self.finish_media(id, MediaEvent::Aborted, CloseReason::Cancelled, None);
            }
            return;
        }
        if let Some(st) = self.running.get_mut(&id) {
            if let Some(output_token_count) = output_token_count {
                let Some(cutoff) = st.token_cutoffs.get(&output_token_count).cloned() else {
                    self.finish_after_inflight(id, FinishReason::Error, None);
                    return;
                };
                st.cancel_cutoff = Some(cutoff);
            }
            st.terminal_intent = if abort {
                TerminalIntent::Abort
            } else {
                TerminalIntent::Cancel
            };
        }
        // also drop from the waiting queue if not yet admitted (reporting the reason)
        if let Some(st) = self.pending.remove_request(id) {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.trace_request_finished(FinishedTrace {
                id,
                reason: &reason,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
                queue: "pending",
            });
            let _ = st.event_tx.send(GenerationEvent::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
    }

    pub(super) fn acknowledge_semantic(&mut self, id: RequestId, output_token_count: usize) {
        let Some(current) = self.running.get(&id).map(|state| state.tokens_acked) else {
            return;
        };
        if current == output_token_count {
            return;
        }
        if output_token_count < current {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        }

        // Release every held commit whose decoded prefix the frontend has now
        // accepted, oldest first, so the worker applies them in exact chained
        // parent order. A held commit with no public token of its own shares the
        // count of the commit ahead of it and releases with it.
        let mut ready: Vec<PendingSemanticCommit> = Vec::new();
        if let Some(state) = self.running.get_mut(&id) {
            state.tokens_acked = output_token_count;
            if let Some((&acknowledged_cutoff, _)) =
                state.token_cutoffs.range(..=output_token_count).next_back()
            {
                state
                    .token_cutoffs
                    .retain(|token_count, _| *token_count >= acknowledged_cutoff);
            }
            while state.pending_commits.front().is_some_and(|pending| {
                pending
                    .token_count
                    .is_some_and(|tc| tc <= output_token_count)
            }) {
                ready.push(state.pending_commits.pop_front().expect("front present"));
            }
        }
        if ready.is_empty() {
            return;
        }
        for pending in ready {
            self.queue_commit(
                id,
                pending.expected_parent,
                pending.selected,
                pending.public_event_limit,
            );
        }
        self.finish_pending_if_idle(id);
    }

    pub(super) fn mark_stopped(&mut self, id: RequestId, output_token_count: usize) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        let Some(cutoff) = state.token_cutoffs.get(&output_token_count).cloned() else {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        };
        // Every provisional descendant beyond the matched stop is retracted: its
        // held commit is dropped un-applied and the ordered close cutoff
        // dominates it. Prefixes acknowledged before the match already committed.
        state.pending_commits.clear();
        state.cancel_cutoff = Some(cutoff);
        state.terminal_intent = TerminalIntent::StopMatched;
    }

    /// Accept only controls declared in the worker capability handshake.
    pub(super) fn control_allowed(&self, op: &ControlOp) -> bool {
        self.caps.supported_controls.contains(&op.request_kind())
    }

    /// Dispatch a control only if the worker declares support; otherwise drop it
    /// (logged) instead of fire-and-forgetting into an UnsupportedControl error.
    pub(super) fn gated_control(&mut self, op: ControlOp) {
        if self.control_allowed(&op) {
            let _ = self.executor.control(op);
        } else {
            tracing::debug!(
                control = op.method(),
                "skipping control absent from worker supported_controls"
            );
        }
    }

    pub(super) fn trace_record(&mut self, record: serde_json::Value) {
        if let Some(sink) = self.trace_sink.as_mut() {
            sink.record(&record);
        }
    }

    pub(super) fn trace_enabled(&self) -> bool {
        self.trace_sink.is_some()
    }

    pub(super) fn trace_request_queued(&mut self, st: &ReqState, queue: &'static str) {
        self.trace_record(json!({
            "event": "request_queued",
            "at_s": st.queued_at,
            "request_id": st.req.request_id.0,
            "trace_id": st.req.request_id.0,
            "queue": queue,
            "generation": &st.req.behavior,
            "initial_phase": st.cursor.phase,
            "prompt_tokens": st.context.prompt_ids.len(),
            "max_tokens": st.req.max_und_tokens,
            "priority": st.req.priority,
            "reserve_worstcase": st.cursor.resources.reserve_worstcase,
            "worstcase_blocks": st.cursor.resources.worstcase_blocks,
            "image": {
                "steps": st.req.image.steps,
                "max_images": st.req.image.max_images,
                "height": st.req.image.height,
                "width": st.req.image.width,
                "retain_images": st.req.image.retain_images,
            },
            "pending": self.pending.len(),
            "running": self.running.len(),
        }));
    }

    pub(super) fn trace_request_finished(&mut self, trace: FinishedTrace<'_>) {
        let FinishedTrace {
            id,
            reason,
            stop_reason,
            prompt_tokens,
            completion_tokens,
            images,
            queue,
        } = trace;
        self.trace_record(json!({
            "event": "request_finished",
            "at_s": now(),
            "request_id": id.0,
            "reason": reason,
            "stop_reason": stop_reason,
            "queue": queue,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "images": images,
            "pending": self.pending.len(),
            "running": self.running.len(),
            "in_flight": self.executor.in_flight(),
        }));
    }
}
