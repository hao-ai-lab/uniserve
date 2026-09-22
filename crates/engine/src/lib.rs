//! In-process scheduling, memory management, worker execution, and request control.
//!
//! The crate owns the engine thread and exports request handles, configuration,
//! execution backends, and statistics. The `testing` feature additionally
//! exports a GPU-free simulated executor for control-plane tests.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
mod executor;
mod handle;
mod kv;
/// Scheduler-owned allocation and worker-params primitives.
pub mod memory;
mod scheduler;
#[cfg(feature = "testing")]
mod sim;
mod worker;

pub use crate::core::{EngineConfig, EngineCore, WorkerConfig, WorkerRank};
pub use crate::executor::{
    BatchResult, CallResult, CommandOutcome, ComponentConfig, ComponentDistribution,
    ExecutionBatch, Executor, ExecutorError, ExecutorInfo, ExecutorSubmitError, ParallelConfig,
    RequestPlacement, SequenceParallel, TransferBackend, TransferConfig, TransferConfigError,
    TransferEdge, WorkerExecError, WorkerFailure, WorkerId, WorkerResult,
};
pub use crate::handle::{
    Command, EVENT_BUFFER_CAPACITY, EngineHandle, EventRx, EventSendError, EventTx,
    StreamCancelCause, SubmitError,
};
pub use crate::scheduler::SpecialTokenIds;
pub use crate::scheduler::{
    DEFAULT_LONG_PREFILL_THRESHOLD, DEFAULT_MAX_BATCH, DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS, DEFAULT_MIXED_PREFILL_TOKENS, DomainStats, EncoderStats,
    ExecutionDomainStats, GeneralStats, KvCacheStats, PrefixStats, Scheduler, SchedulerConfig,
    SchedulerStats, SchedulerStatsReporter, SchedulingPolicy, TimingStats, WorkerStats,
};
#[cfg(feature = "testing")]
pub use crate::sim::{BatchEvent, SimEngine, SimExecutor};
pub use crate::worker::{
    BatchSubmitError, FlashInferBackend, FlashInferBackendParseError, LaneConfig, WorkerExecutor,
    WorkerGroup, WorkerProcessArgs,
};
pub use uniserve_worker_ipc::{AttentionBackend, DEFAULT_COMPONENT};
