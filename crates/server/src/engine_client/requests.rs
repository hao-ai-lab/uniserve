//! Registry of frontend requests spanning preprocessing, engine receipt, and output.
//!
//! `RequestRegistry` maps each external `ServeRequestId` to one record with
//! two parts that have independent lifetimes:
//!
//! - The engine side (`engine_request_id`, `engine_pending`) is reserved by
//!   `register`, claimed once by `claim_submission`, and released by
//!   `release_engine`, which `EngineClient::submit_generation` and
//!   `submit_media` run when submission fails after the claim, or when the
//!   engine event stream ends or its receiver is dropped.
//! - The statistics side (`stats`) exists only for requests registered with a
//!   `ModelEventIdentity`, which `ServingRuntime` supplies for every request.
//!   The serving layer updates it through `mark_submitting`, `accept`, and
//!   `observe`, and its `LifecycleGuard` retires it through `complete` into a
//!   bounded history of terminal snapshots.
//!
//! A record leaves the active map only when it holds neither statistics nor a
//! claimed engine side, so an external identifier cannot be registered again
//! while the engine receiver or the decoded response still holds it. Cancel
//! and abort commands are recorded by `mark_control`; an abort overrides a
//! pending cancellation, later lifecycle updates never clear a pending control
//! state, and `complete` turns it into the matching terminal state.
//!
//! `complete` is also where a request's lifecycle reaches the `uniserve:`
//! request metric families: every request the engine accepted is recorded
//! there once, from its final statistics, under the engine's model name as
//! engine 0.

use crate::serving::{
    LifecycleTerminal, ModelEventIdentity, RequestLifecycleState, RequestOutput,
    RequestStatsSnapshot, ServeRequestId,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use tokio::sync::Notify;
use uniserve_observability::{EngineLabels, METRICS};

struct RequestRecord {
    // The engine identifier reserved by `register`; `release_engine` clears
    // it. `engine_pending` is true from `claim_submission` until
    // `release_engine`.
    engine_request_id: Option<uniserve_core::RequestId>,
    engine_pending: bool,
    // `None` for a request registered without a `ModelEventIdentity`; such a
    // record tracks only the engine side. With an identity, the statistics
    // stay here until `complete` moves them to the completed history,
    // independently of the engine receiver's lifetime.
    stats: Option<RequestStatsSnapshot>,
    // Set by `accept`. Only a request the engine accepted has consumed engine
    // work, so only such a request is recorded in the request metrics.
    accepted: bool,
    // Stable name of the finish status carried by the output's `Finished`
    // event, recorded by `observe` for the request-success metric.
    finished_reason: Option<&'static str>,
    // Set by `observe` on the output's `Usage` event. Until then the token
    // counts in `stats` are partial (output events need not carry their
    // token identifiers), so only a request with final usage contributes to
    // the token metrics.
    usage_reported: bool,
}

#[derive(Default)]
struct RequestRegistryState {
    active: HashMap<ServeRequestId, RequestRecord>,
    // Terminal snapshots, bounded by `RequestRegistry::completed_retention`;
    // `completed_order` lists their identifiers oldest first for eviction.
    completed: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed_order: VecDeque<ServeRequestId>,
}

pub(crate) struct RequestRegistry {
    state: Mutex<RequestRegistryState>,
    // Signalled after an engine side is released, a request completes, or
    // `mark_control` changes or removes a record. `notify_waiters` stores no
    // permit, so every waiter creates its `Notified` future before checking
    // the state it waits on.
    changed: Notify,
    // Maximum number of terminal snapshots kept for `stats` lookups.
    completed_retention: usize,
    // Labels of the request metric series completed requests are recorded
    // under; they match the series `StatsLogger` reads.
    metric_labels: EngineLabels,
}

impl RequestRegistry {
    /// Creates an empty registry that retains the most recent 1024 terminal
    /// snapshots and records completed requests under `model_name`, the
    /// engine's model name, as engine 0.
    pub(crate) fn new(model_name: String) -> Self {
        Self {
            state: Mutex::new(RequestRegistryState::default()),
            changed: Notify::new(),
            completed_retention: 1024,
            metric_labels: EngineLabels {
                model_name,
                engine: 0,
            },
        }
    }

    /// Locks the shared state and recovers it after poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, RequestRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reserves one external identity before preprocessing or engine submission.
    ///
    /// Returns `false` without changing anything when an active record already
    /// holds `request_id`. Otherwise it discards any retained terminal snapshot
    /// of an earlier request with the same identifier and inserts a record
    /// whose engine side is reserved but unclaimed. With `identity`, the record
    /// starts in `Compiling` with zero compile time until `mark_submitting`.
    pub(crate) fn register(
        &self,
        request_id: ServeRequestId,
        engine_request_id: uniserve_core::RequestId,
        identity: Option<ModelEventIdentity>,
    ) -> bool {
        let mut state = self.lock();
        if state.active.contains_key(&request_id) {
            return false;
        }
        state.completed.remove(&request_id);
        state.completed_order.retain(|id| id != &request_id);
        let stats = identity.map(|identity| {
            let mut stats = RequestStatsSnapshot::submitting(
                request_id.clone(),
                identity.served_name,
                identity.description,
                0,
            );
            stats.state = RequestLifecycleState::Compiling;
            stats
        });
        state.active.insert(
            request_id,
            RequestRecord {
                engine_request_id: Some(engine_request_id),
                engine_pending: false,
                stats,
                accepted: false,
                finished_reason: None,
                usage_reported: false,
            },
        );
        true
    }

    /// Attaches exactly one engine receiver to the reserved request incarnation.
    ///
    /// # Errors
    ///
    /// - `Error::UnknownRequestId` when no active record holds `request_id`,
    ///   the record reserved a different engine identifier, or its engine side
    ///   was already released.
    /// - `Error::DuplicateRequestId` when the engine side is already claimed.
    pub(crate) fn claim_submission(
        &self,
        request_id: &str,
        engine_request_id: uniserve_core::RequestId,
    ) -> crate::engine_client::Result<()> {
        let mut state = self.lock();
        let record = state
            .active
            .get_mut(request_id)
            .filter(|record| record.engine_request_id == Some(engine_request_id))
            .ok_or_else(|| crate::engine_client::Error::UnknownRequestId {
                request_id: request_id.to_string(),
            })?;
        if record.engine_pending {
            return Err(crate::engine_client::Error::DuplicateRequestId {
                request_id: request_id.to_string(),
            });
        }
        record.engine_pending = true;
        Ok(())
    }

    /// Releases the engine receiver without retiring a still-live decoded response.
    ///
    /// Does nothing unless the active record still holds `engine_request_id`,
    /// so a repeated release is harmless. Clearing the engine identifier stops
    /// later control commands from reaching the engine and makes the
    /// registration unclaimable. A record without statistics is removed.
    pub(crate) fn release_engine(
        &self,
        request_id: &str,
        engine_request_id: uniserve_core::RequestId,
    ) {
        let mut state = self.lock();
        let Some(record) = state
            .active
            .get_mut(request_id)
            .filter(|record| record.engine_request_id == Some(engine_request_id))
        else {
            return;
        };
        record.engine_pending = false;
        record.engine_request_id = None;
        if record.stats.is_none() {
            state.active.remove(request_id);
        }
        drop(state);
        self.changed.notify_waiters();
    }

    /// Records the measured compile time and moves the request to `Submitting`.
    ///
    /// A pending cancellation or abort keeps its state. Does nothing for an
    /// unknown request or one without statistics.
    pub(crate) fn mark_submitting(&self, request_id: &str, compile_us: u64) {
        if let Some(stats) = self
            .lock()
            .active
            .get_mut(request_id)
            .and_then(|record| record.stats.as_mut())
        {
            if !matches!(
                stats.state,
                RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
            ) {
                stats.state = RequestLifecycleState::Submitting;
            }
            stats.timings.compile_us = compile_us;
        }
    }

    /// Accepts an engine submission into lifecycle tracking.
    ///
    /// Returns `false` without changing anything when the request is unknown,
    /// has no statistics, or has a pending cancellation or abort. The check and
    /// the transition to `Accepted` happen under one registry lock, so no
    /// control can be recorded between them.
    pub(crate) fn accept(&self, request_id: &str) -> bool {
        let mut registry = self.lock();
        let Some(record) = registry.active.get_mut(request_id) else {
            return false;
        };
        let Some(stats) = record.stats.as_mut() else {
            return false;
        };
        if matches!(
            stats.state,
            RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
        ) {
            return false;
        }
        stats.state = RequestLifecycleState::Accepted;
        record.accepted = true;
        true
    }

    /// Records control and resolves the matching engine identity under one lock.
    ///
    /// Returns the engine `RequestId` only while the engine side is claimed,
    /// so the caller forwards the control to the engine exactly when a
    /// submission exists; `None` also covers an unknown `request_id`. A record
    /// with statistics moves to `requested`, except that `Aborting` is never
    /// downgraded and a pending `Cancelling` is replaced only by `Aborting`.
    /// A record without statistics records no control state; it is removed
    /// when its engine side is unclaimed.
    pub(crate) fn mark_control(
        &self,
        request_id: &str,
        requested: RequestLifecycleState,
    ) -> Option<uniserve_core::RequestId> {
        let mut registry = self.lock();
        let record = registry.active.get_mut(request_id)?;
        let engine_request_id = record.engine_request_id.filter(|_| record.engine_pending);
        let Some(stats) = record.stats.as_mut() else {
            // A request without statistics that is controlled before submission
            // owns no output guard, so the record is removed here; a later
            // `claim_submission` then fails with `UnknownRequestId`.
            if !record.engine_pending {
                registry.active.remove(request_id);
                drop(registry);
                self.changed.notify_waiters();
            }
            return engine_request_id;
        };
        let previous = stats.state;
        stats.state = match (previous, requested) {
            (RequestLifecycleState::Aborting, _) => RequestLifecycleState::Aborting,
            (RequestLifecycleState::Cancelling, RequestLifecycleState::Aborting) => {
                RequestLifecycleState::Aborting
            }
            (RequestLifecycleState::Cancelling, _) => RequestLifecycleState::Cancelling,
            (_, requested) => requested,
        };
        let changed = stats.state != previous;
        drop(registry);
        if changed {
            self.changed.notify_waiters();
        }
        engine_request_id
    }

    /// Returns the terminal outcome of a pending cancellation or abort.
    ///
    /// Returns `None` when no control is pending, including for an unknown
    /// request, a request without statistics, or one already completed.
    pub(crate) fn control_terminal(&self, request_id: &str) -> Option<LifecycleTerminal> {
        match self
            .lock()
            .active
            .get(request_id)
            .and_then(|record| record.stats.as_ref())
            .map(|stats| stats.state)
        {
            Some(RequestLifecycleState::Cancelling) => Some(LifecycleTerminal::Cancelled),
            Some(RequestLifecycleState::Aborting) => Some(LifecycleTerminal::Aborted),
            _ => None,
        }
    }

    /// Waits until a cancellation or abort is pending and returns its outcome.
    ///
    /// Stays pending while `control_terminal` reports nothing, which is
    /// permanent for an unknown request, one without statistics, or one already
    /// completed, so callers race it against the work it interrupts.
    pub(crate) async fn wait_for_control(&self, request_id: &str) -> LifecycleTerminal {
        loop {
            let changed = self.changed.notified();
            if let Some(terminal) = self.control_terminal(request_id) {
                return terminal;
            }
            changed.await;
        }
    }

    /// Returns the terminal outcome recorded when a lifecycle guard is dropped
    /// before a terminal event: `Aborted` when an abort is pending and
    /// `Cancelled` otherwise.
    pub(crate) fn drop_terminal(&self, request_id: &str) -> LifecycleTerminal {
        match self
            .lock()
            .active
            .get(request_id)
            .and_then(|record| record.stats.as_ref())
            .map(|stats| stats.state)
        {
            Some(RequestLifecycleState::Aborting) => LifecycleTerminal::Aborted,
            _ => LifecycleTerminal::Cancelled,
        }
    }

    /// Incorporates one serving event into request statistics and surfaces pending control.
    ///
    /// `elapsed_us` is measured from the request's lifecycle start. Returns the
    /// pending control outcome, without applying `event`, when a cancellation
    /// or abort is pending; returns `None` after applying the event otherwise,
    /// and also for an unknown request or one without statistics.
    pub(crate) fn observe(
        &self,
        request_id: &str,
        event: &RequestOutput,
        elapsed_us: u64,
    ) -> Option<LifecycleTerminal> {
        let mut state = self.lock();
        let record = state.active.get_mut(request_id)?;
        let stats = record.stats.as_mut()?;

        // External cancellation and abort state dominates every later stream
        // observation and asks the caller to terminate the producer.
        match stats.state {
            RequestLifecycleState::Cancelling => return Some(LifecycleTerminal::Cancelled),
            RequestLifecycleState::Aborting => return Some(LifecycleTerminal::Aborted),
            _ => {}
        }

        // Each event updates only the lifecycle dimensions it authoritatively
        // carries; the final usage event reconciles all cumulative counters.
        match event {
            RequestOutput::Accepted {
                compile_duration_us,
                prompt_token_count,
                ..
            } => {
                stats.state = RequestLifecycleState::Accepted;
                stats.prompt_tokens = (*prompt_token_count).min(u32::MAX as usize) as u32;
                stats.timings.compile_us = *compile_duration_us;
            }
            // `queued_at` and `scheduled_at` are engine wall-clock UNIX
            // seconds; a negative difference is clamped to zero.
            RequestOutput::Scheduled {
                queued_at,
                scheduled_at,
                cache,
                resources,
                ..
            } => {
                stats.state = RequestLifecycleState::Scheduled;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings.queue_us =
                    (*queued_at).zip(*scheduled_at).map(|(queued, scheduled)| {
                        ((scheduled - queued).max(0.0) * 1_000_000.0) as u64
                    });
            }
            RequestOutput::TextDelta { token_ids, .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.visible_output_tokens = stats
                    .visible_output_tokens
                    .saturating_add(token_ids.len().min(u32::MAX as usize) as u32);
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(elapsed_us);
            }
            // Internal tokens affect usage but never establish visible-output
            // latency. The event carries no token identifiers, so this counts
            // events until a `Usage` event overwrites the total.
            RequestOutput::InternalTextDelta { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.internal_tokens = stats.internal_tokens.saturating_add(1);
            }
            RequestOutput::Artifact(_) => {
                stats.state = RequestLifecycleState::Streaming;
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(elapsed_us);
            }
            RequestOutput::MediaProgress { .. }
            | RequestOutput::ReasoningDelta { .. }
            | RequestOutput::OutputBlockStart { .. }
            | RequestOutput::OutputBlockEnd { .. }
            | RequestOutput::ToolCallStart { .. }
            | RequestOutput::ToolCallArgumentsDelta { .. }
            | RequestOutput::ToolCallEnd { .. } => {
                stats.state = RequestLifecycleState::Streaming;
            }
            // Image begin and commit are public media progress boundaries.
            RequestOutput::ImageBegin {
                elapsed_us: event_elapsed,
                ..
            }
            | RequestOutput::ImageCommit {
                elapsed_us: event_elapsed,
                ..
            } => {
                stats.state = RequestLifecycleState::Streaming;
                stats
                    .timings
                    .first_visible_output_us
                    .get_or_insert(*event_elapsed);
            }
            RequestOutput::ImageStep { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_steps = stats.image_steps.saturating_add(1);
            }
            RequestOutput::ImageDone { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.image_count = stats.image_count.saturating_add(1);
            }
            // Terminal usage is the canonical aggregate produced by the assembler.
            RequestOutput::Usage {
                prompt_tokens,
                visible_output_tokens,
                internal_tokens,
                image_count,
                image_steps,
                cache,
                resources,
                timings,
            } => {
                stats.prompt_tokens = *prompt_tokens;
                stats.visible_output_tokens = *visible_output_tokens;
                stats.internal_tokens = *internal_tokens;
                stats.image_count = *image_count;
                stats.image_steps = *image_steps;
                stats.cache = cache.clone();
                stats.resources = resources.clone();
                stats.timings = timings.clone();
                record.usage_reported = true;
            }
            // Completion transitions are applied by `complete` after
            // observation, which also records the finish status.
            RequestOutput::Finished { reason, .. } => {
                record.finished_reason = Some(super::metrics::finished_reason_name(reason));
            }
            RequestOutput::Rejected { .. }
            | RequestOutput::Cancelled { .. }
            | RequestOutput::Aborted { .. }
            | RequestOutput::Failed { .. } => {}
        }
        None
    }

    /// Moves an active request into retained terminal history with final timing.
    ///
    /// Returns the recorded terminal state: a pending `Cancelling` or
    /// `Aborting` becomes `Cancelled` or `Aborted` regardless of `terminal`.
    /// `total_us` keeps the larger of any `Usage`-reported total and
    /// `elapsed_us`. Returns `None` without recording anything for an unknown
    /// request or one without statistics, including a second call. The record
    /// stays active while its engine side is claimed.
    ///
    /// A request the engine accepted is also recorded in the request metrics:
    /// its outcome (`metrics::record_request_outcome`) under its finish status
    /// when it finished, under `abort` when it was cancelled or aborted, and
    /// under `error` when it failed, and its token usage
    /// (`metrics::record_request_usage`) when its output reported final
    /// usage. A rejected request is not recorded.
    pub(crate) fn complete(
        &self,
        request_id: &str,
        terminal: RequestLifecycleState,
        elapsed_us: u64,
    ) -> Option<RequestLifecycleState> {
        let mut state = self.lock();
        let record = state.active.get_mut(request_id)?;
        let mut stats = record.stats.take()?;
        let accepted = record.accepted;
        let finished_reason = record.finished_reason;
        let usage_reported = record.usage_reported;
        if !record.engine_pending {
            state.active.remove(request_id);
        }
        let actual_terminal = match stats.state {
            RequestLifecycleState::Cancelling => RequestLifecycleState::Cancelled,
            RequestLifecycleState::Aborting => RequestLifecycleState::Aborted,
            _ => terminal,
        };
        stats.state = actual_terminal;
        stats.timings.total_us = stats.timings.total_us.max(elapsed_us);

        let finished_reason = match actual_terminal {
            RequestLifecycleState::Finished => finished_reason,
            RequestLifecycleState::Cancelled | RequestLifecycleState::Aborted => Some("abort"),
            RequestLifecycleState::Failed => Some("error"),
            _ => None,
        };
        if accepted && let Some(finished_reason) = finished_reason {
            super::metrics::record_request_outcome(
                &METRICS.request,
                &self.metric_labels,
                finished_reason,
                &stats,
            );
            if usage_reported {
                super::metrics::record_request_usage(&METRICS.request, &self.metric_labels, &stats);
            }
        }

        Self::insert_completed(&mut state, stats, self.completed_retention);
        drop(state);
        self.changed.notify_waiters();
        Some(actual_terminal)
    }

    /// Appends a terminal snapshot as the newest history entry, replacing any
    /// entry with the same identifier, and evicts the oldest entries beyond
    /// `retention`.
    fn insert_completed(
        state: &mut RequestRegistryState,
        stats: RequestStatsSnapshot,
        retention: usize,
    ) {
        let request_id = stats.request_id.clone();
        state.completed_order.retain(|id| id != &request_id);
        state.completed.insert(request_id.clone(), stats);
        state.completed_order.push_back(request_id);
        while state.completed_order.len() > retention {
            if let Some(expired) = state.completed_order.pop_front() {
                state.completed.remove(&expired);
            }
        }
    }

    /// Returns the active statistics, or else the retained terminal snapshot.
    ///
    /// Returns `None` for a request without statistics, an unknown one, or one
    /// whose snapshot was evicted from the bounded history.
    pub(crate) fn stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        let state = self.lock();
        state
            .active
            .get(request_id)
            .and_then(|record| record.stats.as_ref())
            .or_else(|| state.completed.get(request_id))
            .cloned()
    }

    /// Returns the number of requests whose statistics are still active.
    ///
    /// Records without statistics, including completed requests whose engine
    /// side is still claimed, are not counted.
    pub(crate) fn active_count(&self) -> usize {
        self.lock()
            .active
            .values()
            .filter(|record| record.stats.is_some())
            .count()
    }

    /// Waits for a request to leave the active registry and returns its final statistics.
    ///
    /// The record leaves only when it holds neither statistics nor a claimed
    /// engine side. The result follows `stats`, so it is `None` for a request
    /// without statistics or one already evicted from the history.
    pub(crate) async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        loop {
            let changed = self.changed.notified();
            if !self.lock().active.contains_key(request_id) {
                return self.stats(request_id);
            }
            changed.await;
        }
    }

    /// Waits until the active registry is empty.
    ///
    /// Unlike `active_count`, this also waits for records without statistics
    /// and for completed requests whose engine side is still claimed.
    pub(crate) async fn drain(&self) {
        loop {
            let changed = self.changed.notified();
            if self.lock().active.is_empty() {
                return;
            }
            changed.await;
        }
    }
}
