//! Scheduling policy and configured queue and batch limits.
//!
//! These are requested limits: `Scheduler::with_model_limits` clamps the token
//! and sequence limits, and the batch limit when the executor advertises a
//! nonzero call limit, to the executor's capacity, and the sequence and
//! waiting limits to the hard bounds below.

/// Default maximum calls selected by one generation scheduling pass.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one generation scheduling pass.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of concurrently running requests.
pub const DEFAULT_MAX_NUM_SEQS: usize = 128;
/// Default token threshold above which a prefill is chunked.
pub const DEFAULT_LONG_PREFILL_THRESHOLD: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
/// Default upper bound for queued requests.
pub(crate) const DEFAULT_MAX_NUM_WAITING: usize = 4096;
/// Hard upper bound for configured waiting requests.
pub(crate) const MAX_NUM_WAITING: usize = 65_536;
/// Hard upper bound for configured active sequences.
pub(crate) const MAX_NUM_SEQS: usize = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
/// Policy used to order requests waiting for admission.
///
/// It orders the token-request queue only; media requests wait in arrival
/// order under either policy.
pub enum SchedulingPolicy {
    /// Orders requests by arrival time.
    #[default]
    Fcfs,
    /// Orders requests by ascending priority value, then arrival time.
    Priority,
}

#[derive(Clone, Copy, Debug)]
/// Sequence, token, and queue limits for scheduling.
pub struct SchedulerConfig {
    /// Waiting-queue ordering policy.
    pub policy: SchedulingPolicy,
    /// Maximum calls selected by one generation scheduling pass, across the
    /// per-call-kind batches the pass emits.
    pub max_batch: usize,
    /// Token budget of one generation scheduling pass. Each selected call is
    /// charged its token work, which for a denoising call covers its latent
    /// units and guidance branches.
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests, token and media together.
    pub max_num_seqs: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub long_prefill_threshold: usize,
    /// Admission limit on waiting requests. Submission counts waiting token
    /// and media requests together with finished requests whose output
    /// events are still retained.
    pub max_num_waiting: usize,
    /// Block-diffusion sampling of every canvas the deployment generates, as
    /// the workers are launched with. A block-diffusion runtime has no
    /// autoregressive decode, so each request it does not read out runs
    /// under this sampling; without it the runtime serves readouts only.
    pub canvas: Option<uniserve_core::CanvasSampling>,
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
            canvas: None,
        }
    }
}
