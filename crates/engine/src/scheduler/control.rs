//! Cancellation, terminal cleanup, and lifecycle trace handling.

use super::*;

/// Terminal request facts recorded in a scheduler trace event.
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
    /// Removes the queued media identity while preserving FIFO order for its peers.
    fn remove_waiting_media(&mut self, request_id: RequestId) -> bool {
        let Some(index) = self
            .waiting_media_order
            .iter()
            .position(|id| *id == request_id)
        else {
            return false;
        };
        self.waiting_media_order.remove(index);
        true
    }

    /// Aborts every queued/gated/running request with a terminal event.
    pub(super) fn abort_all_requests(&mut self) {
        self.pending_submissions.clear();
        while let Some(id) = self.waiting_media_order.pop_front() {
            let submission = self
                .waiting_media
                .remove(&id)
                .expect("scheduler media order names runtime state");
            let _ = submission.event_tx.send(EngineCoreOutput::Finished {
                reason: FinishReason::Aborted,
                stop_reason: None,
                prompt_tokens: submission.request.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
        let media = self.media_ids();
        for id in media {
            self.finish_media(id, DiffusionTerminal::Finished(FinishReason::Aborted));
        }
        let queued: Vec<RequestId> = {
            let mut ids = Vec::new();
            while let Some(id) = self.waiting_order.pop_front() {
                let st = self
                    .waiting
                    .remove(&id)
                    .expect("scheduler waiting order names runtime state");
                ids.push(st.req.request_id);
                let _ = st.output.event_tx.send(EngineCoreOutput::Finished {
                    reason: FinishReason::Aborted,
                    stop_reason: None,
                    prompt_tokens: st.req.prompt_token_ids.len(),
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

    /// Applies one command; returns true on shutdown.
    pub(super) fn handle_command(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Submit { request, event_tx } => {
                if !self.accepts_family(request.family()) {
                    let _ = event_tx.send(EngineCoreOutput::Rejected {
                        message: format!(
                            "request family {:?} does not match the {:?} runtime",
                            request.family(),
                            self.family
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
            } => self.acknowledge_output(request_id, output_token_count),
            Command::StopAt {
                request_id,
                output_token_count,
            } => self.mark_stopped(request_id, output_token_count),
            Command::Abort(id) => self.mark_cancelled(id, true, None),
            Command::Shutdown => return true,
        }
        false
    }

    /// Selects commands with available destinations without bypassing an older
    /// command for the same request. Other requests remain independently eligible.
    pub(super) fn take_commands(
        &mut self,
        include: impl Fn(&BatchCommand) -> bool,
    ) -> Vec<BatchCommand> {
        let mut blocked = HashSet::new();
        let mut selected = Vec::new();
        for command in std::mem::take(&mut self.pending_commands) {
            let request = command.request_key();
            if blocked.contains(&request)
                || !include(&command)
                || !self.executor.command_has_capacity(&command)
            {
                blocked.insert(request);
                self.pending_commands.push_back(command);
            } else {
                selected.push(command);
            }
        }
        selected
    }

    /// Enqueues a request directly for deterministic engine-loop tests.
    #[doc(hidden)]
    pub fn submit_for_test(&mut self, request: GenerationRequest) -> EventRx {
        let (event_tx, event_rx) = event_channel();
        self.enqueue(request, event_tx);
        event_rx
    }

    /// Records cancellation or abortion across queued, media, and running request states.
    pub(super) fn mark_cancelled(
        &mut self,
        id: RequestId,
        abort: bool,
        output_token_count: Option<usize>,
    ) {
        if self.remove_waiting_media(id)
            && let Some(submission) = self.waiting_media.remove(&id)
        {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            let _ = submission.event_tx.send(EngineCoreOutput::Finished {
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
            let drained = !self.has_pending_calls(id);
            if drained {
                self.finish_media(
                    id,
                    DiffusionTerminal::Finished(if abort {
                        FinishReason::Aborted
                    } else {
                        FinishReason::Cancelled
                    }),
                );
            }
            return;
        }
        if let Some(st) = self.running.get_mut(&id) {
            if let Some(output_token_count) = output_token_count {
                if output_token_count > st.output.tokens_sent {
                    self.finish_after_inflight(id, FinishReason::Error, None);
                    return;
                }
                st.output.tokens_acked = output_token_count;
            }
            st.output.decoder_boundaries.clear();
            st.terminal_intent = if abort {
                TerminalIntent::Finish(FinishReason::Aborted)
            } else {
                TerminalIntent::Finish(FinishReason::Cancelled)
            };
        }
        // also drop from the waiting queue if not yet admitted (reporting the reason)
        if let Some(st) = self.waiting.remove(&id) {
            self.waiting_order.retain(|queued| *queued != id);
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            self.trace_request_finished(FinishedTrace {
                id,
                reason: &reason,
                stop_reason: None,
                prompt_tokens: st.req.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
                queue: "pending",
            });
            let _ = st.output.event_tx.send(EngineCoreOutput::Finished {
                reason,
                stop_reason: None,
                prompt_tokens: st.req.prompt_token_ids.len(),
                completion_tokens: 0,
                images: 0,
            });
        }
    }

    /// Releases output capacity for prefixes accepted by the frontend decoder.
    pub(super) fn acknowledge_output(&mut self, id: RequestId, output_token_count: usize) {
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

        if let Some(state) = self.running.get_mut(&id) {
            if output_token_count > state.output.tokens_sent {
                self.finish_after_inflight(id, FinishReason::Error, None);
                return;
            }
            state.output.tokens_acked = output_token_count;
            while state
                .output
                .decoder_boundaries
                .front()
                .is_some_and(|count| *count <= output_token_count)
            {
                state.output.decoder_boundaries.pop_front();
            }
        }
        self.finish_pending_if_idle(id);
    }

    /// Marks the lifecycle tracker as stopped.
    pub(super) fn mark_stopped(&mut self, id: RequestId, output_token_count: usize) {
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        if output_token_count > state.output.tokens_sent {
            self.finish_after_inflight(id, FinishReason::Error, None);
            return;
        }
        state.output.tokens_acked = output_token_count;
        state.output.decoder_boundaries.clear();
        state.terminal_intent = TerminalIntent::Finish(FinishReason::Stop);
    }

    /// Returns mutable access to the active trace record.
    pub(super) fn trace_record(&mut self, record: serde_json::Value) {
        if let Some(sink) = self.trace_sink.as_mut() {
            sink.record(&record);
        }
    }

    /// Returns whether call tracing is enabled.
    pub(super) fn trace_enabled(&self) -> bool {
        self.trace_sink.is_some()
    }

    /// Records a complete scheduler-trace snapshot when a request enters a queue.
    pub(super) fn trace_request_queued(&mut self, st: &RequestState, queue: &'static str) {
        self.trace_record(json!({
            "event": "request_queued",
            "at_s": st.queued_at,
            "request_id": st.req.request_id.0,
            "trace_id": st.req.request_id.0,
            "queue": queue,
            "generation": Self::generation_trace(&st.req),
            "initial_phase": st.phase,
            "prompt_tokens": st.req.prompt_token_ids.len(),
            "max_tokens": st.req.max_und_tokens,
            "priority": st.req.priority,
            "reserve_worstcase": st.reserve_worstcase,
            "worstcase_blocks": st.max_reserved_kv_blocks,
            "image": {
                "steps": st.req.image.steps,
                "max_images": st.req.image.max_images,
                "height": st.req.image.height,
                "width": st.req.image.width,
                "retain_images": st.req.image.retain_images,
            },
            "pending": self.waiting_order.len(),
            "running": self.running.len(),
        }));
    }

    /// Records terminal request accounting in the scheduler trace.
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
            "pending": self.waiting_order.len(),
            "running": self.running.len(),
            "in_flight": self.pending_batches.len(),
        }));
    }
}
