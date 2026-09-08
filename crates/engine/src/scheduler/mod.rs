//! Request ordering, lifecycle queues, fairness, and batch-limit policy.

mod queue;
mod stats;
pub(crate) mod stats_report;

pub(crate) use queue::RequestQueue;
pub use stats::{
    DomainStats, EncoderStats, ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats,
    SchedStats, TimingStats, WorkerStats,
};
pub use stats_report::SchedStatsReporter;

use std::collections::VecDeque;

use crate::runtime::output::OutputSender;
use uniserve_core::RequestId;

/// Default maximum operations admitted to one batch.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one scheduled batch.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of sequences in one scheduled batch.
pub const DEFAULT_MAX_NUM_SEQS: usize = 128;
/// Default token threshold above which a prefill is chunked.
pub const DEFAULT_LONG_PREFILL_THRESHOLD: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
/// Default token budget reserved for mixed prefill work.
pub const DEFAULT_MIXED_PREFILL_TOKENS: usize = 0;
/// Default upper bound for queued requests.
pub(crate) const DEFAULT_MAX_NUM_WAITING: usize = 4096;
/// Hard upper bound for configured waiting requests.
pub(crate) const MAX_NUM_WAITING: usize = 65_536;
/// Hard upper bound for configured active sequences.
pub(crate) const MAX_NUM_SEQS: usize = 65_536;

/// Ordering attributes required by a scheduler waiting queue.
pub(crate) trait QueueItem {
    /// Returns the request identifier.
    fn request_id(&self) -> RequestId;
    /// Returns the scheduling priority.
    fn priority(&self) -> i32;
    /// Returns the time at which the request entered the queue.
    fn queued_at(&self) -> f64;
}

#[derive(Clone, Copy)]
/// Queued request identity and ordering metadata.
pub(crate) struct WaitingRequest {
    request_id: RequestId,
    priority: i32,
    queued_at: f64,
}

impl QueueItem for WaitingRequest {
    /// Returns the request identifier.
    fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the scheduling priority.
    fn priority(&self) -> i32 {
        self.priority
    }

    /// Returns the time at which the request entered the queue.
    fn queued_at(&self) -> f64 {
        self.queued_at
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
/// Policy used to order requests waiting for admission.
pub enum SchedulingPolicy {
    /// Orders requests by arrival time.
    #[default]
    Fcfs,
    /// Orders requests by priority, then arrival time.
    Priority,
}

#[derive(Clone, Copy, Debug)]
/// Sequence, token, queue, and mixed-lane limits for scheduling.
pub struct SchedulerConfig {
    /// Waiting-queue ordering policy.
    pub policy: SchedulingPolicy,
    /// Maximum operations submitted in one batch.
    pub max_batch: usize,
    /// Maximum text tokens represented in one batch.
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently resident request sequences.
    pub max_num_seqs: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub long_prefill_threshold: usize,
    /// Maximum requests retained in the waiting queue.
    pub max_num_waiting: usize,
    /// Prefill token budget allowed to share a decode batch.
    pub mixed_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
    /// Returns the default value.
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            long_prefill_threshold: DEFAULT_LONG_PREFILL_THRESHOLD,
            max_num_waiting: DEFAULT_MAX_NUM_WAITING,
            mixed_prefill_tokens: DEFAULT_MIXED_PREFILL_TOKENS,
        }
    }
}

/// Global scheduler policy state. Family progress and allocations are owned by
/// Runtime and Memory; this type contains only shared ordering and limits.
pub struct Scheduler {
    pub(crate) config: SchedulerConfig,
    waiting: RequestQueue<WaitingRequest>,
    waiting_media: VecDeque<RequestId>,
    pub(crate) running_order: Vec<RequestId>,
    pub(crate) output: OutputSender,
    pub(crate) prefer_media: bool,
}

impl Scheduler {
    /// Constructs scheduler ordering state from its limits and policy.
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            waiting: RequestQueue::new(config.policy),
            waiting_media: VecDeque::new(),
            running_order: Vec::new(),
            output: OutputSender::default(),
            prefer_media: true,
            config,
        }
    }

    /// Returns the effective scheduler configuration.
    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Returns the waiting-queue ordering policy.
    pub const fn policy(&self) -> SchedulingPolicy {
        self.config.policy
    }

    /// Enqueues a text request using its priority and arrival time.
    pub(crate) fn enqueue(&mut self, request_id: RequestId, priority: i32, queued_at: f64) {
        self.waiting.add_request(WaitingRequest {
            request_id,
            priority,
            queued_at,
        });
    }

    /// Returns the next queued request without removing it.
    pub(crate) fn peek(&self) -> Option<RequestId> {
        self.waiting.peek_request().map(QueueItem::request_id)
    }

    /// Removes and returns the next queued text request.
    pub(crate) fn pop(&mut self) -> Option<RequestId> {
        self.waiting.pop_request().map(|entry| entry.request_id)
    }

    /// Removes a text request from the waiting queue.
    pub(crate) fn remove(&mut self, request_id: RequestId) -> bool {
        self.waiting.remove_request(request_id).is_some()
    }

    /// Returns the number of waiting text requests.
    pub(crate) fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Enqueues a media request in FIFO order.
    pub(crate) fn enqueue_media(&mut self, request_id: RequestId) {
        self.waiting_media.push_back(request_id);
    }

    /// Removes and returns the next queued media request.
    pub(crate) fn pop_media(&mut self) -> Option<RequestId> {
        self.waiting_media.pop_front()
    }

    /// Returns a media request to the front of the queue.
    pub(crate) fn push_media_front(&mut self, request_id: RequestId) {
        self.waiting_media.push_front(request_id);
    }

    /// Removes a media request from the waiting queue.
    pub(crate) fn remove_media(&mut self, request_id: RequestId) -> bool {
        let Some(index) = self
            .waiting_media
            .iter()
            .position(|candidate| *candidate == request_id)
        else {
            return false;
        };
        self.waiting_media.remove(index);
        true
    }

    /// Returns the number of waiting media requests.
    pub(crate) fn waiting_media_len(&self) -> usize {
        self.waiting_media.len()
    }
}
