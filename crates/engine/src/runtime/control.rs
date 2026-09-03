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

impl EngineLoop {
    /// Abort every queued/gated/running request with a terminal event.
    pub(super) fn abort_all_requests(&mut self) {
        self.pending_submission = None;
        while let Some(id) = self.scheduler.pop_media() {
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let _ = submission.event_tx.send(Event::Finished {
                reason: FinishReason::Aborted,
                stop_reason: None,
                prompt_tokens: submission.request.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(
                id,
                DiffusionTerminal::Finished(FinishReason::Aborted),
                CloseReason::Cancelled,
                None,
            );
        }
        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(id) = self.scheduler.pop() {
                let st = self
                    .waiting
                    .remove(&id)
                    .expect("scheduler waiting order names runtime state");
                ids.push(st.req.request_id);
                let _ = st.output.event_tx.send(Event::Finished {
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
            Command::Submit { request, event_tx } => {
                if !self.runtime.accepts(request.family()) {
                    let _ = event_tx.send(Event::Rejected {
                        message: format!(
                            "request family {:?} does not match the {:?} runtime",
                            request.family(),
                            self.runtime.family()
                        ),
                    });
                } else {
                    match request {
                        Request::Ar(request) | Request::Umm(request) => {
                            self.enqueue(request, event_tx)
                        }
                        Request::Diffusion(request) => {
                            self.enqueue_media(PendingMedia { request, event_tx })
                        }
                    }
                }
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
        if self.scheduler.remove_media(id)
            && let Some(submission) = self.waiting_media.remove(&id)
        {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            let _ = submission.event_tx.send(Event::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: submission.request.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
            return;
        }
        if self.media_state(id).is_some() {
            self.media_state_mut(id)
                .expect("media state exists")
                .terminal_intent
                .finish(if abort {
                    FinishReason::Aborted
                } else {
                    FinishReason::Cancelled
                });
            let drained = !self.inflight.contains(id);
            if drained {
                self.finish_media(
                    id,
                    DiffusionTerminal::Finished(if abort {
                        FinishReason::Aborted
                    } else {
                        FinishReason::Cancelled
                    }),
                    CloseReason::Cancelled,
                    None,
                );
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
        if self.scheduler.remove(id)
            && let Some(st) = self.waiting.remove(&id)
        {
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
            let _ = st.output.event_tx.send(Event::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.context.prompt_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
    }

    pub(super) fn acknowledge_semantic(&mut self, id: RequestId, output_token_count: usize) {
        let Some(current) = self.running.get(&id).map(|state| state.output.tokens_acked) else {
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
            state.output.tokens_acked = output_token_count;
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
            "pending": self.scheduler.waiting_len(),
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
            "pending": self.scheduler.waiting_len(),
            "running": self.running.len(),
            "in_flight": self.inflight.batch_started.len(),
        }));
    }
}
