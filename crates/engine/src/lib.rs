//! In-process scheduling, storage management, worker execution, and request control.
//!
//! The crate sits between the HTTP frontend (`uniserve-server`) and the
//! Python worker processes. `EngineCore` (`core`) launches the worker
//! processes, or takes a supplied executor, and runs the `Scheduler` on its
//! owner thread. Frontends submit and control requests through a cloneable
//! `EngineHandle` (`handle`) and receive each request's events on its
//! `EventRx`. On the owner thread, the `Scheduler` (`scheduler`) admits
//! requests, assigns KV pages (`kv`) and other worker storage (`storage`),
//! and submits calls through the `Executor` trait (`executor`);
//! `WorkerExecutor` (`worker`) splits each batch into per-worker submissions
//! to the worker processes.
//!
//! The `testing` feature additionally exports `SimExecutor`, `SimEngine`, and
//! `BatchEvent` (`sim`), a GPU-free executor for scheduler and frontend tests.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
mod core;
mod executor;
mod handle;
mod kv;
mod scheduler;
#[cfg(feature = "testing")]
mod sim;
/// Scheduler-owned allocation of worker storage; only its `OutOfStorage`
/// error is public.
pub mod storage;
mod worker;

pub use crate::core::{EngineConfig, EngineCore, WorkerConfig, WorkerRank, WorkerRole};
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
pub use crate::sim::{
    BatchEvent, SimEngine, SimExecutor, sim_candidate_logprob, sim_canvas_stop_step, sim_text_token,
};
pub use crate::worker::{
    BatchSubmitError, CallReaders, ExpertExchange, ExpertParallelPlacement, FlashInferBackend,
    FlashInferBackendParseError, LaneConfig, WorkerExecutor, WorkerGroup, WorkerProcessArgs,
};
pub use uniserve_worker_ipc::{
    AttentionBackend, ConditionTiles, DEFAULT_COMPONENT, VideoDenoiserInfo,
};
