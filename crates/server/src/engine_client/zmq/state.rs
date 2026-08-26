//! Frontend-side request registry with per-engine routing state.

use std::collections::{BTreeMap, HashMap};

use tokio::sync::mpsc;
use tracing::trace;

use crate::engine_client::client::state::{EventReceiver, EventSender};
use crate::engine_client::error::{Error, Result};
use crate::engine_client::zmq::EngineId;
use crate::engine_client::zmq::transport::ConnectedEngine;
use uniserve_core::codec::RoutedGenerationEvent;
use uniserve_core::codec::stats::SchedulerStats;

#[derive(Debug)]
struct TrackedRequest {
    sender: EventSender,
    engine_id: EngineId,
}

/// The latest real scheduler-side load snapshot observed from one engine.
///
/// These counters come from `scheduler_stats` on the normal engine event path
/// and are the preferred routing signal once available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EngineLoadSnapshot {
    /// Requests still counted on the scheduler's waiting side.
    waiting: usize,
    /// Requests currently counted on the scheduler's running side.
    running: usize,
}

#[derive(Debug, Default)]
struct EngineRoutingState {
    /// Requests admitted by this frontend that have not finished yet.
    ///
    /// This is used both as the bootstrap fallback before real scheduler stats
    /// exist and as a lower bound afterwards so asynchronous scheduler
    /// snapshots cannot erase frontend admission history.
    inflight: usize,
    /// The latest real scheduler snapshot received from this engine, if any.
    last_scheduler_stats: Option<EngineLoadSnapshot>,
}

impl EngineRoutingState {
    /// Compute the routing score used to pick the least-loaded engine.
    ///
    /// Scheduler stats can raise the load estimate above the frontend-local
    /// view, but they should not lower it below requests this frontend has
    /// already admitted. Waiting requests still get the same extra penalty
    /// as the original `waiting * 4 + running` score.
    fn routing_score(&self) -> usize {
        const WAITING_WEIGHT: usize = 4;

        let Some(stats) = self.last_scheduler_stats else {
            return self.inflight;
        };

        let scheduler_total = stats.running + stats.waiting;
        self.inflight.max(scheduler_total) + stats.waiting * (WAITING_WEIGHT - 1)
    }

    /// Replace the local routing view with a fresh real scheduler snapshot.
    fn apply_scheduler_counts(&mut self, next: EngineLoadSnapshot) {
        self.last_scheduler_stats = Some(next);
    }
}

/// Internal registry for tracking active requests and their event stream
/// senders.
///
/// This is used to route incoming outputs to the correct request stream, and to
/// ensure proper cleanup of senders when requests finish or the client shuts
/// down.
#[derive(Debug)]
pub(crate) struct RequestRegistry {
    closed: bool,
    requests: HashMap<String, TrackedRequest>,
    routing_per_engine: BTreeMap<EngineId, EngineRoutingState>,
}

impl RequestRegistry {
    pub(crate) fn new(engines: &[ConnectedEngine]) -> Self {
        Self {
            closed: false,
            requests: HashMap::default(),
            routing_per_engine: engines
                .iter()
                .map(|engine| (engine.engine_id.clone(), EngineRoutingState::default()))
                .collect(),
        }
    }

    /// Register a newly added request. Create the per-request output channel
    /// bound to its `request_id` and return the selected engine id.
    ///
    /// When `data_parallel_rank` is provided, the request is routed directly to
    /// the engine at that rank index, bypassing load balancing. Otherwise
    /// the engine with the fewest in-flight requests is chosen.
    pub(crate) fn register(
        &mut self,
        request_id: String,
        data_parallel_rank: Option<u32>,
    ) -> Result<(EngineId, EventReceiver)> {
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }

        let engine_id = self.choose_engine_for_request(data_parallel_rank)?;
        let (tx, rx) = mpsc::channel(uniserve_engine::EVENT_BUFFER_CAPACITY);
        self.requests.insert(
            request_id,
            TrackedRequest {
                sender: tx,
                engine_id: engine_id.clone(),
            },
        );

        let state =
            self.routing_per_engine
                .get_mut(&engine_id)
                .ok_or_else(|| Error::ClientClosed {
                    message: format!("request registry is missing engine {engine_id:?}"),
                })?;
        state.inflight += 1;

        Ok((engine_id, rx))
    }

    fn choose_engine_for_request(&mut self, data_parallel_rank: Option<u32>) -> Result<EngineId> {
        if let Some(rank) = data_parallel_rank {
            // Route to the engine at the specified rank index.
            let engine_id = EngineId::from_engine_index(rank);
            return self
                .routing_per_engine
                .contains_key(&engine_id)
                .then_some(engine_id)
                .ok_or_else(|| Error::InvalidDataParallelRank {
                    rank,
                    num_engines: self.routing_per_engine.len() as u32,
                });
        }

        self.routing_per_engine
            .iter()
            .min_by_key(|(_, state)| state.routing_score())
            .map(|(engine_id, _)| engine_id.clone())
            .ok_or_else(|| Error::ClientClosed {
                message: "request registry has no connected engines".to_string(),
            })
    }

    /// Filter the given request IDs to the subset that are still tracked as
    /// active and can be aborted, grouped by engine.
    pub(crate) fn abortable_request_ids(
        &self,
        request_ids: &[String],
    ) -> BTreeMap<EngineId, Vec<String>> {
        let mut by_engine = BTreeMap::new();
        for request_id in request_ids {
            let Some(tracked) = self.requests.get(request_id.as_str()) else {
                continue;
            };
            by_engine
                .entry(tracked.engine_id.clone())
                .or_insert_with(Vec::new)
                .push(request_id.clone());
        }
        by_engine
    }

    pub(crate) fn engine_for_request(&self, request_id: &str) -> Option<EngineId> {
        self.requests
            .get(request_id)
            .map(|tracked| tracked.engine_id.clone())
    }

    /// Obtain the stream sender for one output. If it indicates the request is
    /// finished, it will be removed from the registry.
    pub(crate) fn sender_for_event(
        &mut self,
        routed: &RoutedGenerationEvent,
    ) -> Option<EventSender> {
        if routed.is_terminal() {
            self.remove(routed.external_request_id.as_str())
                .map(|tracked| tracked.0)
        } else {
            self.requests
                .get(routed.external_request_id.as_str())
                .map(|tracked| tracked.sender.clone())
        }
    }

    /// Obtain stream senders for a whole engine event batch under one
    /// registry lock. Finished outputs are removed before returning.
    pub(crate) fn senders_for_events<'a>(
        &mut self,
        events: impl IntoIterator<Item = &'a RoutedGenerationEvent>,
    ) -> Vec<Option<EventSender>> {
        events
            .into_iter()
            .map(|event| self.sender_for_event(event))
            .collect()
    }

    /// Apply one scheduler stats update for the given engine to the local
    /// routing state. Returns `false` if the engine is unknown to the
    /// client.
    pub(crate) fn apply_scheduler_stats(
        &mut self,
        engine_index: u32,
        stats: &SchedulerStats,
    ) -> bool {
        self.apply_scheduler_counts(
            engine_index,
            EngineLoadSnapshot {
                waiting: stats.num_waiting_reqs as usize,
                running: stats.num_running_reqs as usize,
            },
        )
    }

    /// Mark the registry as closed, detach and return all tracked senders.
    pub(crate) fn close(&mut self) -> Vec<EventSender> {
        if self.closed {
            return Vec::new();
        }

        self.closed = true;
        std::mem::take(&mut self.requests)
            .into_values()
            .map(|tracked| tracked.sender)
            .collect()
    }

    /// Remove one request from the local registry. Returns the tracked entry if
    /// it exists.
    #[must_use]
    pub(crate) fn remove(&mut self, request_id: &str) -> Option<(EventSender, EngineId)> {
        let tracked = self.requests.remove(request_id)?;
        if let Some(state) = self.routing_per_engine.get_mut(&tracked.engine_id) {
            // `inflight` is balanced 1:1 with entries in `self.requests`, so it
            // should never be zero here. Saturate defensively anyway so an
            // unexpected double-remove cannot wrap the count to `usize::MAX`
            // (which would permanently poison this engine's routing score) or
            // panic in debug builds.
            state.inflight = state.inflight.saturating_sub(1);
        }
        Some((tracked.sender, tracked.engine_id))
    }

    fn apply_scheduler_counts(&mut self, engine_index: u32, next: EngineLoadSnapshot) -> bool {
        let engine_id = EngineId::from_engine_index(engine_index);
        let Some(state) = self.routing_per_engine.get_mut(&engine_id) else {
            return false;
        };

        let previous = state.last_scheduler_stats;
        if previous != Some(next) {
            trace!(
                ?engine_id,
                previous_waiting = previous.map(|stats| stats.waiting),
                previous_running = previous.map(|stats| stats.running),
                waiting = next.waiting,
                running = next.running,
                "updated scheduler routing counts",
            );
        }

        state.apply_scheduler_counts(next);
        true
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed
    }
}
