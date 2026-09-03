//! Global request ordering, fairness, and batch-limit policy.

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

pub const DEFAULT_MAX_BATCH: usize = 128;
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
pub const DEFAULT_MAX_NUM_SEQS: usize = 128;
pub const DEFAULT_LONG_PREFILL_THRESHOLD: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
pub const DEFAULT_MIXED_PREFILL_TOKENS: usize = 0;
pub(crate) const DEFAULT_MAX_NUM_WAITING: usize = 4096;
pub(crate) const MAX_NUM_WAITING: usize = 65_536;
pub(crate) const MAX_NUM_SEQS: usize = 65_536;

pub(crate) trait QueueItem {
    fn request_id(&self) -> RequestId;
    fn priority(&self) -> i32;
    fn queued_at(&self) -> f64;
}

#[derive(Clone, Copy)]
pub(crate) struct WaitingRequest {
    request_id: RequestId,
    priority: i32,
    queued_at: f64,
}

impl QueueItem for WaitingRequest {
    fn request_id(&self) -> RequestId {
        self.request_id
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn queued_at(&self) -> f64 {
        self.queued_at
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingPolicy {
    #[default]
    Fcfs,
    Priority,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    pub policy: SchedulingPolicy,
    pub max_batch: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub long_prefill_threshold: usize,
    pub max_num_waiting: usize,
    pub mixed_prefill_tokens: usize,
}

impl Default for SchedulerConfig {
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

/// Global scheduler policy state. Family progress and placements are owned by
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

    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub const fn policy(&self) -> SchedulingPolicy {
        self.config.policy
    }

    pub(crate) fn enqueue(&mut self, request_id: RequestId, priority: i32, queued_at: f64) {
        self.waiting.add_request(WaitingRequest {
            request_id,
            priority,
            queued_at,
        });
    }

    pub(crate) fn peek(&self) -> Option<RequestId> {
        self.waiting.peek_request().map(QueueItem::request_id)
    }

    pub(crate) fn pop(&mut self) -> Option<RequestId> {
        self.waiting.pop_request().map(|entry| entry.request_id)
    }

    pub(crate) fn remove(&mut self, request_id: RequestId) -> bool {
        self.waiting.remove_request(request_id).is_some()
    }

    pub(crate) fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn enqueue_media(&mut self, request_id: RequestId) {
        self.waiting_media.push_back(request_id);
    }

    pub(crate) fn pop_media(&mut self) -> Option<RequestId> {
        self.waiting_media.pop_front()
    }

    pub(crate) fn push_media_front(&mut self, request_id: RequestId) {
        self.waiting_media.push_front(request_id);
    }

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

    pub(crate) fn waiting_media_len(&self) -> usize {
        self.waiting_media.len()
    }
}
