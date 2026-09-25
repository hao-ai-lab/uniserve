//! Cancellation, terminal cleanup, and lifecycle handling.
//!
//! The scheduler thread applies each frontend `Command` (see `crate::handle`)
//! here. Cancelling a queued request finishes it at once. A running request
//! records a terminal intent and closes only after every submitted call has
//! resolved: a media request with nothing in flight finishes immediately, and
//! `reap_cancellations` finishes the rest once their calls drain. The
//! acknowledgement commands keep the scheduler's view of the frontend decoder
//! (`RequestOutput::tokens_acked` and `decoder_boundaries`) in step, and
//! `abort_all_requests` ends every request when the control loop stops.

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
    ///
    /// `Scheduler::run` calls this on shutdown and after engine-fatal, and
    /// `Scheduler::step` on shutdown, each right before closing the executor.
    /// Shutdown covers both a `Shutdown` command and a disconnected command
    /// channel. Running requests finish immediately rather than waiting for
    /// their in-flight calls, and batches assembled but not yet submitted are
    /// dropped. Events still journaled behind a full channel, including these
    /// terminal events, are handed to their receivers, since no later flush
    /// will deliver them.
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

        // Every running request has now retired its journal, so the retained
        // journals hold all output the control loop can no longer deliver.
        self.output.hand_off_retired();
    }

    /// Applies one command; returns true on shutdown.
    ///
    /// A submission whose request family the runtime does not serve is
    /// rejected here; every other submission is validated by `enqueue` or
    /// `enqueue_media`, which report their own rejections.
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
    ///
    /// A command is deferred when `include` rejects it or
    /// `Executor::command_has_capacity` reports a full destination; deferred
    /// commands keep their relative order in `pending_commands`.
    pub(super) fn take_commands(
        &mut self,
        include: impl Fn(&BatchCommand) -> bool,
    ) -> Vec<BatchCommand> {
        let mut blocked = HashSet::new();
        let mut selected = Vec::new();
        for command in std::mem::take(&mut self.inflight.pending_commands) {
            let request = command.request_key();
            // Once one command of a request is deferred, every later command
            // of that request is deferred with it.
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
    ///
    /// A queued request finishes here with `Aborted` or `Cancelled`. A running
    /// media request records the reason, keeping an earlier terminal intent,
    /// and finishes here only when none of its calls are in flight. A running
    /// token request records the reason as its terminal intent and is finished
    /// by `reap_cancellations` once its calls drain; an `output_token_count`
    /// beyond the tokens already sent schedules an error finish instead.
    /// Unknown requests are ignored.
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
        // A request not yet admitted leaves the waiting queue with its reason.
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
    ///
    /// Acknowledging the current count is a no-op. A count below the current
    /// acknowledgement or above the tokens already sent finishes the request
    /// with an error once its in-flight calls drain. Otherwise every decoder
    /// boundary the new prefix covers is resolved, which may apply a finish
    /// deferred on those decisions. Requests that are not running are ignored.
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

    /// Stops a running token request at the prefix where the frontend decoder
    /// matched a stop string.
    ///
    /// Records that prefix as acknowledged, drops pending decoder decisions,
    /// and sets a `Stop` terminal intent that `reap_cancellations` applies once
    /// the request's calls drain. A count above the tokens already sent
    /// instead finishes the request with an error once its in-flight calls
    /// drain. Requests that are not running are ignored.
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
