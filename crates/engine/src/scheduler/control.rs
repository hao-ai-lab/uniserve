//! Cancellation, terminal cleanup, and lifecycle handling.

use super::*;

impl Scheduler {
    /// Removes a queued media request while preserving FIFO order for its peers.
    fn remove_waiting_media(&mut self, request_id: RequestId) -> Option<PendingMedia> {
        let index = self
            .waiting_media
            .iter()
            .position(|submission| submission.request.request_id == request_id)?;
        self.waiting_media.remove(index)
    }

    /// Removes a queued token request while preserving admission order for its peers.
    fn remove_waiting(&mut self, request_id: RequestId) -> Option<RequestState> {
        let index = self
            .waiting
            .iter()
            .position(|queued| queued.req.request_id == request_id)?;
        self.waiting.remove(index)
    }

    /// Aborts every queued/gated/running request with a terminal event.
    pub(super) fn abort_all_requests(&mut self) {
        self.inflight.pending_submissions.clear();
        while let Some(submission) = self.waiting_media.pop_front() {
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
            while let Some(st) = self.waiting.pop_front() {
                ids.push(st.req.request_id);
                let _ = st.output.events.event_tx.send(EngineCoreOutput::Finished {
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
                        kind: RejectionKind::Invalid,
                        message: format!(
                            "request family {:?} does not match the {:?} runtime",
                            request.family(),
                            self.family
                        ),
                    });
                } else {
                    match *request {
                        Request::Ar(request) | Request::Umm(request) => {
                            self.enqueue(request, event_tx)
                        }
                        Request::Diffusion(request) => self.enqueue_media(PendingMedia {
                            request,
                            event_tx,
                            queued_at: now(),
                        }),
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
        for command in std::mem::take(&mut self.inflight.pending_commands) {
            let request = command.request_key();
            if blocked.contains(&request)
                || !include(&command)
                || !self.executor.command_has_capacity(&command)
            {
                blocked.insert(request);
                self.inflight.pending_commands.push_back(command);
            } else {
                selected.push(command);
            }
        }
        selected
    }

    /// Records cancellation or abortion across queued, media, and running request states.
    pub(super) fn mark_cancelled(
        &mut self,
        id: RequestId,
        abort: bool,
        output_token_count: Option<usize>,
    ) {
        if let Some(submission) = self.remove_waiting_media(id) {
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
        if let Some(state) = self.media_state_mut(id) {
            state.terminal_intent.finish(if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            });
            let drained = !self.inflight.has_pending_calls(id);
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
        if let Some(st) = self.remove_waiting(id) {
            let reason = if abort {
                FinishReason::Aborted
            } else {
                FinishReason::Cancelled
            };
            let _ = st.output.events.event_tx.send(EngineCoreOutput::Finished {
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
}
