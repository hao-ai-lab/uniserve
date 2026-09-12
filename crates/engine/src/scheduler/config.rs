//! Scheduling policy and configured queue and batch limits.

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
