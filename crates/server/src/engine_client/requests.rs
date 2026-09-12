//! One frontend request record spanning preprocessing, engine receipt, and output.

use crate::serving::{
    LifecycleTerminal, ModelEventIdentity, RequestLifecycleState, RequestOutput,
    RequestStatsSnapshot, ServeRequestId,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use tokio::sync::Notify;

struct RequestRecord {
    engine_request_id: Option<uniserve_core::RequestId>,
    engine_pending: bool,
    // Raw engine consumers (including retained video jobs) own their output
    // accounting. Decoded responses retain these statistics until their final
    // public event, independently of the engine receiver's lifetime.
    stats: Option<RequestStatsSnapshot>,
}

#[derive(Default)]
struct RequestRegistryState {
    active: HashMap<ServeRequestId, RequestRecord>,
    completed: HashMap<ServeRequestId, RequestStatsSnapshot>,
    completed_order: VecDeque<ServeRequestId>,
}

pub(crate) struct RequestRegistry {
    state: Mutex<RequestRegistryState>,
    changed: Notify,
    completed_retention: usize,
}

impl Default for RequestRegistry {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            state: Mutex::new(RequestRegistryState::default()),
            changed: Notify::new(),
            completed_retention: 1024,
        }
    }
}

impl RequestRegistry {
    /// Locks the shared state and recovers it after poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, RequestRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reserves one external identity before preprocessing or engine submission.
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
            },
        );
        true
    }

    /// Attaches exactly one engine receiver to the reserved request incarnation.
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

    /// Marks the request as submitting.
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
    pub(crate) fn accept(&self, request_id: &str) -> bool {
        let mut registry = self.lock();
        let Some(stats) = registry
            .active
            .get_mut(request_id)
            .and_then(|record| record.stats.as_mut())
        else {
            return false;
        };
        if matches!(
            stats.state,
            RequestLifecycleState::Cancelling | RequestLifecycleState::Aborting
        ) {
            return false;
        }
        stats.state = RequestLifecycleState::Accepted;
        true
    }

    /// Records control and resolves the matching engine identity under one lock.
    pub(crate) fn mark_control(
        &self,
        request_id: &str,
        requested: RequestLifecycleState,
    ) -> Option<uniserve_core::RequestId> {
        let mut registry = self.lock();
        let record = registry.active.get_mut(request_id)?;
        let engine_request_id = record.engine_request_id.filter(|_| record.engine_pending);
        let Some(stats) = record.stats.as_mut() else {
            // A raw request cancelled before submission owns no output guard.
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

    /// Returns the accepted control command terminal event.
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

    /// Waits the for control.
    pub(crate) async fn wait_for_control(&self, request_id: &str) -> LifecycleTerminal {
        loop {
            let changed = self.changed.notified();
            if let Some(terminal) = self.control_terminal(request_id) {
                return terminal;
            }
            changed.await;
        }
    }

    /// Returns the terminal event emitted when the guard is dropped.
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
    pub(crate) fn observe(
        &self,
        request_id: &str,
        event: &RequestOutput,
        elapsed_us: u64,
    ) -> Option<LifecycleTerminal> {
        let mut state = self.lock();
        let stats = state.active.get_mut(request_id)?.stats.as_mut()?;

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
            // Internal tokens affect usage but never establish visible-output latency.
            RequestOutput::InternalTextDelta { .. } => {
                stats.state = RequestLifecycleState::Streaming;
                stats.internal_tokens = stats.internal_tokens.saturating_add(1);
            }
            RequestOutput::ReasoningDelta { .. }
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
            }
            // Completion transitions are applied by `complete` after observation.
            RequestOutput::Finished { .. }
            | RequestOutput::Rejected { .. }
            | RequestOutput::Cancelled { .. }
            | RequestOutput::Aborted { .. }
            | RequestOutput::Failed { .. } => {}
        }
        None
    }

    /// Moves an active request into retained terminal history with final timing.
    pub(crate) fn complete(
        &self,
        request_id: &str,
        terminal: RequestLifecycleState,
        elapsed_us: u64,
    ) -> Option<RequestLifecycleState> {
        let mut state = self.lock();
        let record = state.active.get_mut(request_id)?;
        let mut stats = record.stats.take()?;
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
        Self::insert_completed(&mut state, stats, self.completed_retention);
        drop(state);
        self.changed.notify_waiters();
        Some(actual_terminal)
    }

    /// Inserts the completed.
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

    /// Returns a snapshot of the current statistics.
    pub(crate) fn stats(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        let state = self.lock();
        state
            .active
            .get(request_id)
            .and_then(|record| record.stats.as_ref())
            .or_else(|| state.completed.get(request_id))
            .cloned()
    }

    /// Returns the number of active requests.
    pub(crate) fn active_count(&self) -> usize {
        self.lock()
            .active
            .values()
            .filter(|record| record.stats.is_some())
            .count()
    }

    /// Waits for a request to leave the active registry and returns its final statistics.
    pub(crate) async fn drain_request(&self, request_id: &str) -> Option<RequestStatsSnapshot> {
        loop {
            let changed = self.changed.notified();
            if !self.lock().active.contains_key(request_id) {
                return self.stats(request_id);
            }
            changed.await;
        }
    }

    /// Drains completed entries from the tracker.
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
